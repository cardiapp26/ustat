"""Actuarial (Cutler-Ederer) life table: POST /api/models/survival/life_table."""
import io
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.stats import norm

from main import app
from services import life_table as lt
from services import store

client = TestClient(app)
URL = "/api/models/survival/life_table"


# ── Helpers ──────────────────────────────────────────────────────────────────

# A five-interval textbook-style table: l, w, d known per interval of width 1.
D = [10, 8, 6, 5, 4]
W = [5, 7, 6, 4, 45]  # everyone left at the end is withdrawn in the last interval
N = 100
BREAKS = [0, 1, 2, 3, 4, 5]


def textbook_frame() -> pd.DataFrame:
    rows = []
    for i, (d, w) in enumerate(zip(D, W)):
        rows += [{"time": i + 0.5, "event": 1}] * d
        rows += [{"time": i + 0.5, "event": 0}] * w
    df = pd.DataFrame(rows)
    assert len(df) == N
    return df


def reference_table(d, w, widths, n, alpha=0.05):
    """Plain python loops, written independently of services.life_table."""
    z = norm.ppf(1 - alpha / 2)
    out = []
    l = n
    S = 1.0
    green = 0.0
    for di, wi, h in zip(d, w, widths):
        eff = l - wi / 2
        q = di / eff
        p = 1 - q
        S *= p
        green += di / (eff * (eff - di))
        se = S * math.sqrt(green)
        out.append({
            "l": l, "eff": eff, "q": q, "p": p, "S": S, "se": se,
            "hazard": 2 * q / (h * (1 + p)), "se_q": math.sqrt(q * p / eff),
            "lo": max(0.0, S - z * se), "hi": min(1.0, S + z * se),
        })
        l -= di + wi
    return out


def post(df, sid, **body):
    store.save(sid, df)
    return client.post(URL, json={"session_id": sid, "time_col": "time", "event_col": "event", **body})


# ── Numbers ──────────────────────────────────────────────────────────────────


def test_textbook_table_matches_independent_loops():
    r = post(textbook_frame(), "lt_textbook", breaks=BREAKS)
    assert r.status_code == 200, r.text
    body = r.json()
    g = body["groups"][0]
    assert g["group"] is None and g["n"] == N and g["events"] == sum(D) and g["censored"] == N - sum(D)
    assert body["breaks"] == BREAKS and body["interval_width"] == 1

    ref = reference_table(D, W, [1] * 5, N)
    rows = g["intervals"]
    assert len(rows) == 5
    for row, e in zip(rows, ref):
        assert row["n_entering"] == e["l"]
        assert row["effective_at_risk"] == pytest.approx(e["eff"])
        assert row["q"] == pytest.approx(e["q"])
        assert row["p"] == pytest.approx(e["p"])
        assert row["cumulative_survival"] == pytest.approx(e["S"])
        assert row["se_cumulative"] == pytest.approx(e["se"])
        assert row["hazard"] == pytest.approx(e["hazard"])
        assert row["se_q"] == pytest.approx(e["se_q"])
        assert row["ci_low"] == pytest.approx(e["lo"])
        assert row["ci_high"] == pytest.approx(e["hi"])
    assert [r_["withdrawn"] for r_ in rows] == W
    assert [r_["events"] for r_ in rows] == D
    # Hand-checked first interval: l' = 97.5, q = 10 / 97.5.
    assert rows[0]["effective_at_risk"] == 97.5
    assert rows[0]["q"] == pytest.approx(0.1025641, abs=1e-7)
    # S stays above 0.5 throughout, so the median is not reached.
    assert g["median_survival"] is None


def test_hazard_identity():
    # Cutler-Ederer hazard: 2q / (h (1 + p)) == d / (h (l' - d / 2)).
    r = post(textbook_frame(), "lt_haz", breaks=[0, 2, 3, 5, 6, 7.5])
    rows = r.json()["groups"][0]["intervals"]
    assert rows[-1]["hazard"] is None  # nobody left beyond 5
    for row in rows[:-2]:
        h = row["interval_end"] - row["interval_start"]
        assert row["hazard"] == pytest.approx(2 * row["q"] / (h * (1 + row["p"])))
        assert row["hazard"] == pytest.approx(row["events"] / (h * (row["effective_at_risk"] - row["events"] / 2)))


def test_median_interpolated_inside_interval():
    # 20 subjects, one-year intervals: 6 deaths in y1, 6 in y2, rest censored at 2.5.
    rows = [{"time": 0.5, "event": 1}] * 6 + [{"time": 1.5, "event": 1}] * 6 + [{"time": 2.5, "event": 0}] * 8
    r = post(pd.DataFrame(rows), "lt_median", breaks=[0, 1, 2, 3])
    g = r.json()["groups"][0]
    s1 = 1 - 6 / 20
    s2 = s1 * (1 - 6 / 14)
    assert g["intervals"][0]["cumulative_survival"] == pytest.approx(s1)
    assert g["intervals"][1]["cumulative_survival"] == pytest.approx(s2)
    assert s1 > 0.5 >= s2
    assert g["median_survival"] == pytest.approx(1 + (s1 - 0.5) / (s1 - s2))


def test_all_censored_survival_one():
    df = pd.DataFrame({"time": [0.5, 1.5, 1.7, 2.2, 2.9, 0.1], "event": [0] * 6})
    g = post(df, "lt_allcens", breaks=[0, 1, 2, 3]).json()["groups"][0]
    assert g["events"] == 0 and g["censored"] == 6
    for row in g["intervals"]:
        assert row["q"] == 0 and row["p"] == 1
        assert row["cumulative_survival"] == 1
        assert row["se_cumulative"] == 0
        assert row["hazard"] == 0
    assert g["median_survival"] is None


def test_everyone_fails_in_first_interval_then_nulls():
    df = pd.DataFrame({"time": [0.2, 0.4, 0.6, 0.9, 0.95], "event": [1] * 5})
    g = post(df, "lt_allevent", breaks=[0, 1, 2, 3]).json()["groups"][0]
    first, *rest = g["intervals"]
    assert first["q"] == 1 and first["p"] == 0
    assert first["cumulative_survival"] == 0 and first["se_cumulative"] == 0
    assert first["ci_low"] == 0 and first["ci_high"] == 0
    assert first["hazard"] == pytest.approx(2.0)  # 2q / (h (1 + p)) with q=1, p=0, h=1
    assert g["median_survival"] == pytest.approx(0.5)  # S falls 1 -> 0 linearly over [0, 1)
    for row in rest:
        assert row["n_entering"] == 0 and row["effective_at_risk"] == 0
        for key in ("q", "p", "cumulative_survival", "se_cumulative", "ci_low", "ci_high", "hazard", "se_q"):
            assert row[key] is None


def test_exact_break_belongs_to_interval_starting_there():
    df = pd.DataFrame({"time": [1.0, 2.0, 2.0, 0.0], "event": [1, 1, 0, 0]})
    rows = post(df, "lt_exact", breaks=[0, 1, 2, 3]).json()["groups"][0]["intervals"]
    assert [r["events"] for r in rows] == [0, 1, 1]
    assert [r["withdrawn"] for r in rows] == [1, 0, 1]
    assert [r["n_entering"] for r in rows] == [4, 3, 2]


def test_last_break_and_beyond_with_explicit_breaks():
    df = pd.DataFrame({
        "time": [0.5, 1.5, 2.0, 2.0, 3.5, 9.0],
        "event": [1, 1, 1, 0, 1, 1],
    })
    body = post(df, "lt_beyond", breaks=[0, 1, 2]).json()
    rows = body["groups"][0]["intervals"]
    # t=2.0 sits on the last break: event stays an event, censoring is a withdrawal.
    # t=3.5 and t=9 are beyond: administratively censored (withdrawn) in the last interval.
    assert [r["events"] for r in rows] == [1, 2]
    assert [r["withdrawn"] for r in rows] == [0, 3]
    assert any("beyond the last break" in w for w in body["warnings"])
    assert any("exactly at the last break" in w for w in body["warnings"])


def test_default_breaks_cover_follow_up_without_warning_about_beyond():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"time": rng.uniform(0, 47, 200), "event": rng.integers(0, 2, 200)})
    body = post(df, "lt_default").json()
    b = body["breaks"]
    assert b[0] == 0 and b[-1] > df["time"].max()
    assert body["interval_width"] == 5  # 47 / 10 = 4.7 rounds up to 5
    assert len(b) - 1 == 10
    assert not any("beyond the last break" in w for w in body["warnings"])
    assert any("extended automatically" in w for w in body["warnings"])
    rows = body["groups"][0]["intervals"]
    assert sum(r["events"] + r["withdrawn"] for r in rows) == 200


def test_width_and_n_intervals():
    df = textbook_frame()
    body = post(df, "lt_width", interval_width=1, n_intervals=3).json()
    assert body["breaks"] == [0, 1, 2, 3]
    assert any("beyond the last break" in w for w in body["warnings"])
    body = post(df, "lt_width2", interval_width=2).json()
    assert body["breaks"] == [0, 2, 4, 6]  # covers max time 4.5 strictly
    body = post(df, "lt_width3", n_intervals=5).json()
    assert len(body["breaks"]) == 6


def test_grouped_output_matches_per_group_reference():
    a = textbook_frame().assign(grp="A")
    b = pd.DataFrame({"time": [0.5] * 4 + [1.5] * 6 + [2.5] * 10, "event": [1] * 4 + [0] * 6 + [0] * 10, "grp": "B"})
    r = post(pd.concat([b, a], ignore_index=True), "lt_grouped", breaks=[0, 1, 2, 3, 4, 5], group_col="grp")
    assert r.status_code == 200, r.text
    body = r.json()
    assert [g["group"] for g in body["groups"]] == ["A", "B"]
    ga, gb = body["groups"]
    assert ga["n"] == 100 and gb["n"] == 20 and gb["events"] == 4
    ref = reference_table(D, W, [1] * 5, N)
    assert ga["intervals"][4]["cumulative_survival"] == pytest.approx(ref[4]["S"])
    # Group B: l = 20, w = 0, d = 4 in interval 1, then 16 enter interval 2 with 6 withdrawn.
    assert gb["intervals"][0]["q"] == pytest.approx(4 / 20)
    assert gb["intervals"][1]["effective_at_risk"] == pytest.approx(16 - 3)
    assert gb["intervals"][2]["effective_at_risk"] == pytest.approx(10 - 5)
    header = body["export_rows"][0]
    assert header[0] == "grp" and len(body["export_rows"]) == 1 + 2 * 5
    assert body["export_rows"][1][0] == "A" and body["export_rows"][-1][0] == "B"


def test_response_shape_and_text_fields():
    body = post(textbook_frame(), "lt_shape", breaks=BREAKS).json()
    for key in ("groups", "breaks", "interval_width", "warnings", "result_text", "methods_text",
                "r_code", "export_rows", "n_total", "n_analyzed", "n_excluded"):
        assert key in body
    row = body["groups"][0]["intervals"][0]
    assert set(row) == {
        "interval_start", "interval_end", "n_entering", "withdrawn", "events", "effective_at_risk",
        "q", "p", "cumulative_survival", "se_cumulative", "ci_low", "ci_high", "hazard", "se_q",
    }
    assert "Cutler-Ederer" in body["result_text"] and "Greenwood" in body["methods_text"]
    assert len(body["export_rows"]) == 6 and len(body["export_rows"][0]) == len(body["export_rows"][1])
    for field in ("result_text", "methods_text", "r_code"):
        assert "\u2014" not in body[field]


def test_missing_rows_are_excluded_with_warning():
    df = textbook_frame()
    df.loc[0, "time"] = np.nan
    df.loc[1, "event"] = np.nan
    body = post(df, "lt_missing", breaks=BREAKS).json()
    assert body["n_excluded"] == 2 and body["n_analyzed"] == N - 2
    assert any("excluded" in w for w in body["warnings"])


# ── Bad input ────────────────────────────────────────────────────────────────


def test_unknown_session_404():
    r = client.post(URL, json={"session_id": "lt_nope", "time_col": "time", "event_col": "event"})
    assert r.status_code == 404


def test_negative_time_422():
    df = pd.DataFrame({"time": [1.0, -2.0, 3.0], "event": [1, 0, 1]})
    assert post(df, "lt_neg").status_code == 422


def test_non_binary_event_422():
    df = pd.DataFrame({"time": [1.0, 2.0, 3.0], "event": [0, 1, 2]})
    assert post(df, "lt_nonbin").status_code == 422


def test_all_rows_unusable_400():
    df = pd.DataFrame({"time": ["a", "b"], "event": ["Yes", "No"]})
    assert post(df, "lt_unusable").status_code == 400


@pytest.mark.parametrize("breaks", [[0, 2, 1], [0, 1, 1, 2], [1, 2, 3], [0], [0, float("nan")]])
def test_bad_breaks_422(breaks):
    if any(isinstance(b, float) and math.isnan(b) for b in breaks):
        pytest.skip("NaN is not valid JSON")
    assert post(textbook_frame(), "lt_badbreaks", breaks=breaks).status_code == 422


@pytest.mark.parametrize("extra", [
    {"interval_width": 0}, {"interval_width": -1}, {"n_intervals": 0}, {"n_intervals": 5000},
    {"breaks": [0, 1, 2], "interval_width": 1}, {"alpha": 0}, {"alpha": 1.5},
    {"interval_width": 0.001},
])
def test_bad_layout_params_422(extra):
    assert post(textbook_frame(), "lt_badparam", **extra).status_code == 422


def test_unknown_or_duplicate_columns_422():
    store.save("lt_cols", textbook_frame())
    base = {"session_id": "lt_cols", "time_col": "time", "event_col": "event"}
    assert client.post(URL, json={**base, "time_col": "nope"}).status_code == 422
    assert client.post(URL, json={**base, "group_col": "nope"}).status_code == 422
    assert client.post(URL, json={**base, "group_col": "event"}).status_code == 422


# ── Pure functions ───────────────────────────────────────────────────────────


def test_nice_width():
    assert lt.nice_width(4.7) == 5
    assert lt.nice_width(0.7) == 1
    assert lt.nice_width(1.0) == 1
    assert lt.nice_width(1.1) == 2
    assert lt.nice_width(2.3) == 2.5
    assert lt.nice_width(0.023) == 0.025
    assert lt.default_width(0) == 1


def test_compute_life_table_pure():
    t = lt.compute_life_table([0.5, 1.5], [1, 0], [0, 1, 2], alpha=0.05)
    r0, r1 = t["intervals"]
    assert r0["n_entering"] == 2 and r0["q"] == pytest.approx(0.5)
    assert r1["n_entering"] == 1 and r1["withdrawn"] == 1 and r1["effective_at_risk"] == pytest.approx(0.5)
    assert r1["cumulative_survival"] == pytest.approx(0.5)  # no further events
    assert t["median_survival"] == pytest.approx(1.0)  # S reaches exactly 0.5 at the end of interval 1


# ── Cross-checks against R ───────────────────────────────────────────────────

RSCRIPT = shutil.which("Rscript")


def _rscript(code: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([RSCRIPT, "-e", code], capture_output=True, text=True, timeout=timeout)


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not installed")
def test_r_code_reproduces_python_table():
    """Run the generated base-R code on the same data and compare every column."""
    rng = np.random.default_rng(21)
    n = 150
    df = pd.DataFrame({
        "time": np.round(rng.exponential(6, n), 2),
        "event": rng.integers(0, 2, n),
        "grp": rng.choice(["a", "b"], n),
    })
    body = post(df, "lt_r", breaks=[0, 2, 4, 8, 12, 20], group_col="grp").json()
    with tempfile.TemporaryDirectory() as tmp:
        csv_in = Path(tmp) / "dat.csv"
        csv_out = Path(tmp) / "out.csv"
        df.to_csv(csv_in, index=False)
        code = f'dat <- read.csv("{csv_in}")\n' + body["r_code"] + f'\nwrite.csv(res, "{csv_out}", row.names = FALSE)\n'
        res = _rscript(code)
        assert res.returncode == 0, res.stderr
        out = pd.read_csv(csv_out)
    assert len(out) == sum(len(g["intervals"]) for g in body["groups"])
    cols = {
        "n_entering": "n_entering", "withdrawn": "withdrawn", "events": "events",
        "effective_at_risk": "effective_at_risk", "q": "q", "p": "p",
        "cumulative_survival": "cumulative_survival", "se_cumulative": "se_cumulative",
        "ci_low": "ci_low", "ci_high": "ci_high", "hazard": "hazard", "se_q": "se_q",
    }
    i = 0
    for g in body["groups"]:
        sub = out[out["group"] == g["group"]].reset_index(drop=True)
        assert len(sub) == len(g["intervals"])
        for j, row in enumerate(g["intervals"]):
            for key, col in cols.items():
                py, rv = row[key], sub.loc[j, col]
                if py is None:
                    assert pd.isna(rv), (g["group"], j, key)
                else:
                    assert float(rv) == pytest.approx(py, rel=1e-9, abs=1e-12), (g["group"], j, key)
            i += 1
    assert i == len(out)


def _has_kmsurv() -> bool:
    return RSCRIPT is not None and _rscript("library(KMsurv)", timeout=60).returncode == 0


@pytest.mark.skipif(not _has_kmsurv(), reason="R package KMsurv not installed")
def test_matches_kmsurv_lifetab():
    body = post(textbook_frame(), "lt_kmsurv", breaks=BREAKS).json()
    rows = body["groups"][0]["intervals"]
    code = (
        "library(KMsurv)\n"
        f"tab <- lifetab(c({', '.join(str(b) for b in BREAKS[:-1])}, Inf), {N}, "
        f"c({', '.join(map(str, W))}), c({', '.join(map(str, D))}))\n"
        "cat(sprintf('%.10f', tab$surv), sep = ',')\ncat('\\n')\n"
        "cat(sprintf('%.10f', tab$se.surv), sep = ',')\n"
    )
    res = _rscript(code)
    assert res.returncode == 0, res.stderr
    surv_line, se_line = res.stdout.strip().splitlines()[-2:]
    surv = [float(x) for x in surv_line.split(",")]
    se = [float(x) for x in se_line.split(",")]
    # lifetab reports survival at the START of each interval (1 first); ours is at the end.
    for j in range(1, len(rows)):
        assert surv[j] == pytest.approx(rows[j - 1]["cumulative_survival"], rel=1e-8)
        assert se[j] == pytest.approx(rows[j - 1]["se_cumulative"], rel=1e-8)
