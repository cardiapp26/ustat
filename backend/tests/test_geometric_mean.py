"""Geometric mean, geometric SD (GSD) and the CI of the GM."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from conftest import make_session
from routers.stats.descriptive import _fmt_one_stat, _geometric_stats

GM_KEYS = (
    "geometric_mean",
    "geometric_sd",
    "geometric_mean_ci_lower",
    "geometric_mean_ci_upper",
)


def test_hand_computed_one_ten_hundred():
    g = _geometric_stats([1, 10, 100])
    # ln x = 0, ln 10, 2 ln 10 -> mean ln 10, sd ln 10 (n - 1 denominator)
    assert g["geometric_mean"] == pytest.approx(10.0)
    assert g["geometric_sd"] == pytest.approx(math.exp(2.302585092994046))
    assert g["geometric_sd"] == pytest.approx(10.0)
    assert g["geometric_mean_note"] is None


def test_gmean_equals_exp_mean_log():
    x = np.array([2.5, 3.1, 8.0, 0.4, 19.0, 5.5])
    g = _geometric_stats(x)
    assert g["geometric_mean"] == pytest.approx(float(np.exp(np.log(x).mean())))
    assert g["geometric_mean"] == pytest.approx(float(scipy_stats.gmean(x)))


def test_ci_matches_manual_t_formula():
    x = np.array([2.5, 3.1, 8.0, 0.4, 19.0, 5.5, 7.2])
    n = len(x)
    ln = np.log(x)
    t = scipy_stats.t.ppf(0.975, n - 1)
    half = t * ln.std(ddof=1) / math.sqrt(n)
    g = _geometric_stats(x)
    assert g["geometric_mean_ci_lower"] == pytest.approx(math.exp(ln.mean() - half))
    assert g["geometric_mean_ci_upper"] == pytest.approx(math.exp(ln.mean() + half))
    assert g["geometric_mean_ci_lower"] < g["geometric_mean"] < g["geometric_mean_ci_upper"]


def test_constant_values_give_gsd_one():
    g = _geometric_stats([4.0, 4.0, 4.0])
    assert g["geometric_mean"] == pytest.approx(4.0)
    assert g["geometric_sd"] == pytest.approx(1.0)
    assert g["geometric_mean_ci_lower"] == pytest.approx(4.0)
    assert g["geometric_mean_ci_upper"] == pytest.approx(4.0)


@pytest.mark.parametrize("bad", [[1, 2, 0, 4], [1, -2, 3], [0, 0, 0]])
def test_null_and_note_for_non_positive(bad):
    g = _geometric_stats(bad)
    assert all(g[k] is None for k in GM_KEYS)
    assert "zero or negative" in g["geometric_mean_note"]


def test_null_and_note_for_single_value():
    g = _geometric_stats([5.0])
    assert all(g[k] is None for k in GM_KEYS)
    assert g["geometric_mean_note"]


def test_descriptive_endpoint_positive_and_nonpositive(client):
    pos = [1.0, 10.0, 100.0, 10.0, 1.0, 100.0]
    df = pd.DataFrame(
        {
            "titre": pos,
            "delta": [-1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "zeros": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    sid = make_session(df, "geomean_descriptive")
    r = client.get(f"/api/stats/{sid}/descriptive")
    assert r.status_code == 200, r.text
    body = r.json()

    ln = np.log(pos)
    t = scipy_stats.t.ppf(0.975, len(pos) - 1)
    half = t * ln.std(ddof=1) / math.sqrt(len(pos))
    titre = body["titre"]
    assert titre["geometric_mean"] == pytest.approx(10.0)
    assert titre["geometric_sd"] == pytest.approx(math.exp(ln.std(ddof=1)))
    assert titre["geometric_mean_ci_lower"] == pytest.approx(math.exp(ln.mean() - half))
    assert titre["geometric_mean_ci_upper"] == pytest.approx(math.exp(ln.mean() + half))
    assert titre["geometric_mean_note"] is None
    # existing keys untouched
    assert titre["harmonic_mean"] is not None and titre["mean"] == pytest.approx(np.mean(pos))

    for col in ("delta", "zeros"):
        assert all(body[col][k] is None for k in GM_KEYS)
        assert "zero or negative" in body[col]["geometric_mean_note"]


def test_column_summary_has_geometric_mean(client):
    df = pd.DataFrame({"v": [1.0, 10.0, 100.0, 1000.0]})
    sid = make_session(df, "geomean_column_summary")
    r = client.get(
        f"/api/stats/{sid}/column_summary", params={"column": "v", "kind": "numeric"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["geometric_mean"] == pytest.approx(10 ** 1.5)
    assert body["geometric_sd"] is not None
    assert body["geometric_mean_note"] is None


def test_fmt_one_stat_geometric_options():
    a = pd.Series([1.0, 10.0, 100.0])
    assert _fmt_one_stat(a, "geometric_mean") == "10.00 (10.00)"
    g = _geometric_stats(a.to_numpy())
    expected = (
        f"10.00 [{g['geometric_mean_ci_lower']:.2f}–{g['geometric_mean_ci_upper']:.2f}]"
    )
    assert _fmt_one_stat(a, "gm_ci") == expected
    # non-positive data renders the standard missing placeholder
    bad = pd.Series([0.0, 1.0, 2.0])
    assert _fmt_one_stat(bad, "geometric_mean") == _fmt_one_stat(pd.Series([], dtype=float), "mean_sd")


def test_table1_geometric_mean_rows_and_warning(client):
    df = pd.DataFrame(
        {
            "arm": ["A"] * 4 + ["B"] * 4,
            "titre": [1.5, 10.0, 100.0, 10.0, 2.0, 20.0, 200.0, 20.0],
            "bad": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    sid = make_session(df, "geomean_table1")
    payload = {
        "session_id": sid,
        "group_column": "arm",
        "variables": ["titre", "bad"],
        "variable_kinds": {"titre": "numeric", "bad": "numeric"},
        "selected_stats": ["geometric_mean", "gm_ci"],
    }
    r = client.post("/api/stats/table1", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    rows = {row["variable"]: row for row in body["rows"]}

    titre = rows["titre"]["stat_rows"]
    assert [s["label"] for s in titre] == [
        "Geometric mean (GSD)",
        "Geometric mean [95% CI]",
    ]
    gm_a = float(scipy_stats.gmean([1.5, 10.0, 100.0, 10.0]))
    # one-decimal column: decimals follow the column, like mean and SD
    assert titre[0]["group_stats"]["A"].startswith(f"{gm_a:.1f} (")
    assert titre[1]["group_stats"]["A"].startswith(f"{gm_a:.1f} [")

    placeholder = _fmt_one_stat(pd.Series([], dtype=float), "mean_sd")
    bad = rows["bad"]["stat_rows"]
    assert bad[0]["overall"] == placeholder
    assert bad[1]["overall"] == placeholder
    assert any("'bad'" in w and "geometric mean" in w for w in body["warnings"])
    assert not any("'titre'" in w and "geometric mean" in w for w in body["warnings"])
