"""Table 1 must agree with the Tests tab, and with the data it prints.

Driven by a 10x20 frame that carries missing values in every shape a real
upload does: a numeric column that is mostly missing, a grouping column with
its own gaps, a category that appears once, a column that exists in only one
arm, a constant column, a fully empty column, and free text.

Every discrepancy asserted here was live before these tests existed.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp

from conftest import make_session

GROUP = "arm"


def build(n: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def holes(s: pd.Series, k: int) -> pd.Series:
        s = s.copy()
        if k:
            s.iloc[rng.choice(n, min(k, n), replace=False)] = np.nan
        return s

    frac = lambda p: max(1, int(round(n * p)))  # noqa: E731

    d = {}
    d["arm"] = holes(pd.Series(rng.choice(["A", "B"], n), dtype=object), frac(0.05))
    d["age"] = holes(pd.Series(rng.normal(60, 10, n)), frac(0.10))
    d["crp"] = holes(pd.Series(rng.lognormal(1.0, 1.1, n)), frac(0.15))
    d["visits"] = holes(pd.Series(rng.poisson(3, n).astype(float)), frac(0.05))
    d["bmi"] = pd.Series(rng.normal(27, 4, n))
    d["const_num"] = holes(pd.Series(np.full(n, 5.0)), frac(0.10))
    arm_vals = pd.Series(d["arm"]).fillna("A").values
    d["sbp"] = holes(
        pd.Series(rng.normal(120, 15, n) + np.where(arm_vals == "B", 12.0, 0.0)),
        frac(0.10),
    )
    d["ldl"] = holes(
        pd.Series(rng.normal(110, np.where(arm_vals == "B", 30.0, 5.0))), frac(0.05)
    )
    d["troponin"] = holes(pd.Series(rng.normal(0.05, 0.02, n)), frac(0.70))
    d["sex"] = holes(pd.Series(rng.choice(["M", "F"], n), dtype=object), frac(0.10))
    d["cp"] = holes(
        pd.Series(rng.choice(["typical", "atypical", "none"], n), dtype=object),
        frac(0.20),
    )
    d["stage"] = holes(
        pd.Series(
            rng.choice(["I", "II", "III", "IV"], n, p=[0.5, 0.3, 0.15, 0.05]),
            dtype=object,
        ),
        frac(0.10),
    )
    d["diabetes"] = holes(
        pd.Series(rng.choice(["Yes", "No"], n), dtype=object), frac(0.05)
    )
    d["smoker"] = holes(pd.Series(rng.integers(0, 2, n).astype(float)), frac(0.10))
    d["site"] = holes(pd.Series(["central"] * n, dtype=object), frac(0.10))
    d["notes"] = holes(pd.Series([f"note_{i}" for i in range(n)], dtype=object), frac(0.10))
    one_arm = pd.Series(rng.choice(["x", "y"], n), dtype=object)
    one_arm[pd.Series(d["arm"]).values == "B"] = np.nan
    d["armonly"] = one_arm
    d["empty"] = pd.Series([np.nan] * n, dtype=object)
    rare = pd.Series(rng.choice(["common", "common", "common", "mid"], n), dtype=object)
    rare.iloc[0] = "singleton"
    d["rare"] = holes(rare, frac(0.05))
    d["weight"] = holes(pd.Series(rng.normal(75, 12, n)), frac(0.05))
    if n > 3:
        d["weight"].iloc[1] = 999.0

    df = pd.DataFrame(d)
    assert df.shape[1] == 20
    return df


@pytest.fixture()
def small() -> str:
    """Exactly 10 x 20 — the size that used to defeat numeric detection."""
    df = build(10)
    assert df.shape == (10, 20)
    return make_session(df, "t1c_small")


@pytest.fixture()
def wide() -> str:
    """Same schema at 200 rows, so the p-values have something to say."""
    return make_session(build(200), "t1c_wide")


def _table1(client, sid, variables, **extra):
    r = client.post(
        "/api/stats/table1",
        json={
            "session_id": sid,
            "variables": variables,
            "group_column": GROUP,
            **extra,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _rows(payload) -> dict:
    return {row["variable"]: row for row in payload["rows"]}


def _p(row):
    """Table 1 formats the p for display; parse it back, or None."""
    raw = row.get("p_value")
    if raw is None:
        return None
    s = str(raw)
    if s.startswith("<"):
        return 0.0005
    try:
        return float(s)
    except ValueError:
        return None


# ── numeric vs categorical detection ──────────────────────────────────────────


def test_continuous_columns_survive_a_ten_row_dataset(client, small):
    """`nunique() > 10` is unreachable at 10 rows.

    Every continuous measurement was therefore listed as a categorical
    variable with one category per patient at 11.1% each, and handed a
    chi-square.
    """
    rows = _rows(_table1(client, small, ["age", "bmi", "crp", "sbp", "ldl", "weight"]))
    for var in ("age", "bmi", "crp", "sbp", "ldl", "weight"):
        assert rows[var]["type"] == "numeric", f"{var} fell back to categorical"


def test_integer_coded_columns_stay_categorical(client, small):
    """The fix must not sweep 0/1 flags and small counts into numeric."""
    rows = _rows(_table1(client, small, ["smoker", "visits", "const_num"]))
    for var in ("smoker", "visits", "const_num"):
        assert rows[var]["type"] == "categorical", f"{var} became numeric"


# ── Table 1 vs the Tests tab ──────────────────────────────────────────────────


def test_ttest_matches_the_tests_tab(client, wide):
    """Table 1 forced Welch and labelled the row plain "t-test".

    The Tests tab lets Levene choose, so the same two columns produced two
    different p-values in two tabs with nothing on screen to explain the gap.
    """
    for var in ("age", "bmi", "sbp"):
        row = _rows(_table1(client, wide, [var]))[var]
        assert row["type"] == "numeric"
        assert row["test"].startswith("t-test"), row["test"]
        rt = client.post(
            "/api/stats/ttest",
            json={"session_id": wide, "column": var, "group_column": GROUP},
        )
        assert rt.status_code == 200, rt.text
        assert _p(row) == pytest.approx(rt.json()["p"], abs=5e-4), (
            f"{var}: Table 1 {row['p_value']} vs Tests {rt.json()['p']}"
        )


def test_welch_is_named_when_welch_is_used(client):
    """Unequal variances: Levene fires, and the row must say which t-test ran.

    A plain "t-test" label next to a Welch p is what let the two tabs differ
    without the table admitting anything had changed.
    """
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            GROUP: ["A"] * 60 + ["B"] * 60,
            "spread": np.r_[rng.normal(10, 1, 60), rng.normal(11, 9, 60)],
        }
    )
    sid = make_session(df, "t1c_welch")
    # Judge normality per group: pooling an sd-1 and an sd-9 arm makes the
    # combined column look heavy-tailed, which would send Table 1 to
    # Mann-Whitney and never reach the Student-vs-Welch choice under test.
    row = _rows(
        _table1(client, sid, ["spread"], normality_mode="within_group")
    )["spread"]
    assert row["test"] == "t-test (Welch)"
    rt = client.post(
        "/api/stats/ttest",
        json={"session_id": sid, "column": "spread", "group_column": GROUP},
    )
    assert rt.json()["variance_assumption"] == "welch"
    assert _p(row) == pytest.approx(rt.json()["p"], abs=5e-4)


def test_chisquare_matches_the_tests_tab(client, wide):
    for var in ("sex", "cp", "stage", "diabetes", "smoker"):
        row = _rows(_table1(client, wide, [var]))[var]
        rc = client.post(
            "/api/stats/chisquare",
            json={"session_id": wide, "row_column": var, "col_column": GROUP},
        )
        assert rc.status_code == 200, rc.text
        chi = rc.json()
        if not str(row["test"]).startswith("Chi"):
            continue  # small-cell fallback; a different test by design
        assert _p(row) == pytest.approx(chi["p"], abs=5e-4), (
            f"{var}: Table 1 {row['p_value']} vs Tests {chi['p']}"
        )


def test_printed_counts_add_up_to_the_rows_the_test_used(client, wide):
    for var in ("sex", "cp", "stage", "diabetes"):
        row = _rows(_table1(client, wide, [var]))[var]
        chi = client.post(
            "/api/stats/chisquare",
            json={"session_id": wide, "row_column": var, "col_column": GROUP},
        ).json()
        per_group = 0
        for sub in row["sub_rows"]:
            for cell in sub["group_stats"].values():
                per_group += int(cell.split(" ")[0])
        assert per_group == chi["n"], f"{var}: printed {per_group}, tested {chi['n']}"


# ── tables with nothing to test ───────────────────────────────────────────────


def test_a_constant_column_gets_no_p_value(client, wide):
    """chi2_contingency answers a 1-row table with dof 0 and p exactly 1.0.

    Printed as-is that reads as a tested, non-significant result.
    """
    row = _rows(_table1(client, wide, ["const_num"]))["const_num"]
    assert row["p_value"] is None
    assert row["test"] is None


def test_a_column_present_in_one_arm_only_gets_no_p_value(client, wide):
    payload = _table1(client, wide, ["armonly"])
    row = _rows(payload)["armonly"]
    assert row["p_value"] is None
    assert any("armonly" in str(w) for w in payload["warnings"])


def test_an_empty_column_does_not_raise(client, wide):
    row = _rows(_table1(client, wide, ["empty"]))["empty"]
    assert row["p_value"] is None
    assert row["sub_rows"] == []


def test_free_text_is_refused_rather_than_tested(client, wide):
    """199 distinct notes produced a chi-square on a 171x2 table."""
    payload = _table1(client, wide, ["notes"])
    row = _rows(payload)["notes"]
    assert row["p_value"] is None
    assert any("too many" in str(w) for w in payload["warnings"])


def test_a_group_with_one_observation_gets_no_t_test(client, small):
    """scipy returns nan; the row used to print an em dash under a Welch label."""
    payload = _table1(client, small, ["troponin"])
    row = _rows(payload)["troponin"]
    assert row["p_value"] is None
    assert row["test"] is None
    assert any("fewer than 2 values" in str(w) for w in payload["warnings"])


def test_chisquare_endpoint_refuses_a_degenerate_table(client, wide):
    r = client.post(
        "/api/stats/chisquare",
        json={"session_id": wide, "row_column": "site", "col_column": GROUP},
    )
    assert r.status_code == 400
    assert "nothing to test" in r.json()["detail"]


# ── missingness must never become a category ──────────────────────────────────


def test_missing_does_not_enter_the_categorical_smd(client, wide):
    """The SMD stringified before dropping NaN, so "nan" became a level."""
    row = _rows(_table1(client, wide, ["cp"]))["cp"]
    df = build(200)
    pairs = df[["cp", GROUP]].dropna()
    levels = sorted(pairs["cp"].unique())
    assert len(levels) == 3 and "nan" not in levels
    assert row["smd"] is not None
    # Recompute the 3-level SMD by hand on complete pairs only.
    g1 = pairs[pairs[GROUP] == "A"]["cp"]
    g2 = pairs[pairs[GROUP] == "B"]["cp"]
    p1 = np.array([(g1 == c).mean() for c in levels[:-1]])
    p2 = np.array([(g2 == c).mean() for c in levels[:-1]])
    s_pool = (np.diag(p1 * (1 - p1)) + np.diag(p2 * (1 - p2))) / 2
    diff = p1 - p2
    expected = float(np.sqrt(diff @ np.linalg.inv(s_pool) @ diff))
    assert row["smd"] == pytest.approx(expected, abs=1e-3)


def test_zero_percent_is_formatted_like_every_other_percentage(client, wide):
    row = _rows(_table1(client, wide, ["armonly"]))["armonly"]
    cells = [c for sub in row["sub_rows"] for c in sub["group_stats"].values()]
    assert any(c.startswith("0 ") for c in cells)
    assert not any(c.endswith("(0%)") for c in cells), cells


# ── the numbers themselves ────────────────────────────────────────────────────


def test_numeric_summaries_match_a_hand_calculation(client, wide):
    df = build(200)
    row = _rows(_table1(client, wide, ["bmi"]))["bmi"]
    s = df["bmi"].dropna()
    overall = row["stat_rows"][0]["overall"]
    mean, sd = (float(x) for x in overall.split(" ± "))
    assert mean == pytest.approx(s.mean(), abs=0.01)
    assert sd == pytest.approx(s.std(ddof=1), abs=0.01)
    assert row["overall_n"] == int(s.notna().sum())


def test_mann_whitney_p_matches_scipy(client, wide):
    df = build(200)
    row = _rows(_table1(client, wide, ["crp"]))["crp"]
    assert row["test"] == "Mann-Whitney"
    arrs = [df[df[GROUP] == g]["crp"].dropna().to_numpy() for g in ("A", "B")]
    expected = sp.mannwhitneyu(*arrs, alternative="two-sided").pvalue
    assert _p(row) == pytest.approx(expected, abs=5e-4)
