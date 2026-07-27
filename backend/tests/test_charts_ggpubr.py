"""ggpubr-equivalent chart endpoints: significance brackets, error plot, ECDF.

The bracket endpoint is the one with teeth. A figure that prints stars is
making a claim, and the two ways it usually lies are showing unadjusted
p-values from a dozen comparisons and not saying which test produced them.
Both are pinned here.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def three_groups() -> str:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "value": np.concatenate([
                rng.normal(10, 2, 40),
                rng.normal(13, 2, 40),
                rng.normal(10.4, 2, 40),
            ]),
            "arm": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
        }
    )
    return make_session(df, "ggpubr_three")


def _cmp(client, sid, **kw):
    body = {"session_id": sid, "y": "value", "group": "arm"}
    body.update(kw)
    return client.post("/api/charts/compare_means", json=body)


def test_every_pair_is_compared_and_labelled(client, three_groups):
    r = _cmp(client, three_groups)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["levels"] == ["A", "B", "C"]
    assert len(d["comparisons"]) == 3  # 3 choose 2
    for c in d["comparisons"]:
        assert c["stars"] in {"****", "***", "**", "*", "ns"}
        assert c["x1"] != c["x2"]


def test_brackets_are_stacked_shortest_span_first(client, three_groups):
    """Otherwise adjacent-pair brackets get drawn over the A-vs-C span."""
    d = _cmp(client, three_groups).json()
    spans = [c["span"] for c in d["comparisons"]]
    assert spans == sorted(spans)
    assert [c["level"] for c in d["comparisons"]] == [0, 1, 2]


def test_adjusted_p_is_what_gets_shown_and_it_is_declared(client, three_groups):
    d = _cmp(client, three_groups, p_adjust="bonferroni").json()
    assert d["p_adjust"] == "bonferroni"
    assert d["p_shown_is_adjusted"] is True
    for c in d["comparisons"]:
        assert c["p_shown"] == pytest.approx(c["p_adj"])
        assert c["p_adj"] >= c["p"] - 1e-12, "adjustment must not shrink a p-value"


def test_unadjusted_is_possible_but_flagged(client, three_groups):
    d = _cmp(client, three_groups, p_adjust="none").json()
    assert d["p_shown_is_adjusted"] is False
    for c in d["comparisons"]:
        assert c["p_shown"] == pytest.approx(c["p"])


def test_auto_picks_the_rank_test_when_a_group_is_not_normal(client):
    rng = np.random.default_rng(5)
    skewed = np.exp(rng.normal(0, 1.4, 60))  # lognormal, fails Shapiro
    df = pd.DataFrame(
        {"value": np.concatenate([skewed, rng.normal(5, 1, 60)]),
         "arm": ["A"] * 60 + ["B"] * 60}
    )
    sid = make_session(df, "ggpubr_skew")
    d = _cmp(client, sid).json()
    assert d["test"] == "Mann-Whitney U"
    assert "failed Shapiro" in d["test_selected_by"]


def test_auto_picks_welch_when_every_group_looks_normal(client, three_groups):
    d = _cmp(client, three_groups).json()
    assert d["test"] == "Welch t-test"
    assert "passed Shapiro" in d["test_selected_by"]


def test_explicit_method_overrides_the_screen(client, three_groups):
    d = _cmp(client, three_groups, method="wilcoxon").json()
    assert d["test"] == "Mann-Whitney U"
    assert d["test_selected_by"] == "requested"


def test_omnibus_is_reported_for_three_groups_and_names_its_limit(client, three_groups):
    d = _cmp(client, three_groups).json()
    assert d["omnibus"]["test"] == "One-way ANOVA"
    assert "not Welch-corrected" in d["omnibus"]["note"]


def test_no_omnibus_for_a_two_group_comparison(client):
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {"value": np.concatenate([rng.normal(0, 1, 30), rng.normal(1, 1, 30)]),
         "arm": ["A"] * 30 + ["B"] * 30}
    )
    sid = make_session(df, "ggpubr_two")
    assert _cmp(client, sid).json()["omnibus"] == {}


def test_reference_group_keeps_only_its_own_comparisons(client, three_groups):
    d = _cmp(client, three_groups, ref_group="A").json()
    assert len(d["comparisons"]) == 2
    assert all("A" in (c["group1"], c["group2"]) for c in d["comparisons"])


def test_unknown_reference_group_is_a_400_that_lists_the_real_ones(client, three_groups):
    r = _cmp(client, three_groups, ref_group="Z")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "A" in detail and "B" in detail


def test_p_label_mode_prints_numbers_instead_of_stars(client, three_groups):
    d = _cmp(client, three_groups, label="p").json()
    assert all(c["label"].startswith("p") for c in d["comparisons"])


def test_star_thresholds_follow_the_published_convention(client):
    from routers.charts import _stars

    assert _stars(0.00005) == "****"
    assert _stars(0.0005) == "***"
    assert _stars(0.005) == "**"
    assert _stars(0.03) == "*"
    assert _stars(0.05) == "*"      # boundary is inclusive
    assert _stars(0.051) == "ns"
    assert _stars(float("nan")) == "ns"


def test_single_usable_level_is_a_400(client):
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "arm": ["A", "A", "A"]})
    sid = make_session(df, "ggpubr_one")
    r = _cmp(client, sid)
    assert r.status_code == 400
    assert "at least two levels" in r.json()["detail"]


def test_bad_method_and_adjust_are_refused(client, three_groups):
    assert _cmp(client, three_groups, method="anova").status_code == 400
    assert _cmp(client, three_groups, p_adjust="sidak").status_code == 400


# ── error plot ──────────────────────────────────────────────────────────────

def _err(client, sid, **kw):
    body = {"session_id": sid, "y": "value", "group": "arm"}
    body.update(kw)
    return client.post("/api/charts/errorplot", json=body)


def test_ci_is_wider_than_se_and_se_narrower_than_sd(client, three_groups):
    width = {}
    for spread in ("sd", "se", "ci"):
        d = _err(client, three_groups, spread=spread).json()
        row = d["rows"][0]
        width[spread] = row["upper"] - row["lower"]
    assert width["se"] < width["ci"] < width["sd"] or width["se"] < width["sd"]
    assert width["se"] < width["ci"], "a 95% CI must exceed ±1 SE"


def test_ci_uses_t_not_z(client):
    """With n = 5 the t multiplier is 2.776; z would give 1.96."""
    df = pd.DataFrame({"value": [10.0, 11.0, 12.0, 13.0, 14.0], "arm": ["A"] * 5})
    sid = make_session(df, "ggpubr_small")
    row = _err(client, sid).json()["rows"][0]
    sd = float(np.std([10, 11, 12, 13, 14], ddof=1))
    se = sd / np.sqrt(5)
    assert (row["upper"] - row["centre"]) == pytest.approx(2.7764 * se, rel=1e-3)


def test_spread_label_names_what_the_whisker_is(client, three_groups):
    assert _err(client, three_groups, spread="sd").json()["spread_label"] == "mean ± SD"
    assert "95% CI" in _err(client, three_groups, spread="ci").json()["spread_label"]
    assert _err(
        client, three_groups, centre="median", spread="iqr"
    ).json()["spread_label"] == "median with IQR"


def test_median_with_an_sd_whisker_is_refused(client, three_groups):
    r = _err(client, three_groups, centre="median", spread="sd")
    assert r.status_code == 400
    assert "median" in r.json()["detail"]


def test_mean_with_an_iqr_whisker_is_refused(client, three_groups):
    r = _err(client, three_groups, centre="mean", spread="iqr")
    assert r.status_code == 400


def test_errorplot_without_a_group_returns_one_row(client, three_groups):
    d = client.post(
        "/api/charts/errorplot",
        json={"session_id": three_groups, "y": "value"},
    ).json()
    assert len(d["rows"]) == 1
    assert d["rows"][0]["group"] == "All"
    assert d["rows"][0]["n"] == 120


# ── ECDF ────────────────────────────────────────────────────────────────────

def test_ecdf_curve_is_monotone_and_ends_at_one(client, three_groups):
    d = client.post(
        "/api/charts/ecdf",
        json={"session_id": three_groups, "x": "value", "group": "arm"},
    ).json()
    assert len(d["curves"]) == 3
    for c in d["curves"]:
        assert c["y"] == sorted(c["y"])
        assert c["x"] == sorted(c["x"])
        assert c["y"][-1] == pytest.approx(1.0)
        assert c["y"][0] == pytest.approx(1 / c["n"])


def test_ecdf_reports_ks_for_exactly_two_groups(client):
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {"value": np.concatenate([rng.normal(0, 1, 80), rng.normal(2, 1, 80)]),
         "arm": ["A"] * 80 + ["B"] * 80}
    )
    sid = make_session(df, "ggpubr_ks")
    d = client.post(
        "/api/charts/ecdf", json={"session_id": sid, "x": "value", "group": "arm"}
    ).json()
    from scipy import stats as sp

    a = df.loc[df.arm == "A", "value"].to_numpy()
    b = df.loc[df.arm == "B", "value"].to_numpy()
    ref = sp.ks_2samp(a, b)
    assert d["ks"]["statistic"] == pytest.approx(float(ref.statistic))
    assert d["ks"]["p"] == pytest.approx(float(ref.pvalue))


def test_ecdf_omits_ks_when_there_are_three_groups(client, three_groups):
    d = client.post(
        "/api/charts/ecdf",
        json={"session_id": three_groups, "x": "value", "group": "arm"},
    ).json()
    assert d["ks"] == {}


def test_ecdf_needs_two_points(client):
    df = pd.DataFrame({"value": [1.0]})
    sid = make_session(df, "ggpubr_tiny")
    r = client.post("/api/charts/ecdf", json={"session_id": sid, "x": "value"})
    assert r.status_code == 400
