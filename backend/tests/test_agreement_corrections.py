"""Corrections to the method-agreement endpoints.

* Bland-Altman: Student t (not 1.96) for the CI of the bias, and Bland &
  Altman (1986, 1999) confidence intervals for both limits of agreement.
* Passing-Bablok: the real 1983 algorithm (all pairwise slopes, shifted
  median, C_gamma bounds, Cusum linearity test) instead of Theil-Sen.

No R reference is used because the `mcr` package is not installed here; the
Passing-Bablok checks compare with an independent loop implementation written
straight from the paper, plus exact and symmetry properties.
"""
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conftest import make_session


# ── Bland-Altman ─────────────────────────────────────────────────────────────

def _ba_frame(n=30, seed=11):
    rng = np.random.default_rng(seed)
    truth = rng.normal(100, 15, n)
    return pd.DataFrame({"m1": truth + rng.normal(0, 5, n),
                         "m2": truth + rng.normal(2, 5, n)})


def _ba(client, df, sid, **kw):
    s = make_session(df, sid)
    r = client.post("/api/agreement/bland_altman", json={"session_id": s, "method1": "m1", "method2": "m2", **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_ba_mean_diff_ci_uses_student_t(client):
    df = _ba_frame()
    d = _ba(client, df, "ba_t")
    diffs = (df["m1"] - df["m2"]).to_numpy()
    n = len(diffs)
    m, sd = diffs.mean(), diffs.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    assert d["ci_mean_diff"]["low"] == pytest.approx(m - t * sd / np.sqrt(n), abs=1e-4)
    assert d["ci_mean_diff"]["high"] == pytest.approx(m + t * sd / np.sqrt(n), abs=1e-4)
    # strictly wider than the old normal-approximation interval
    assert d["ci_mean_diff"]["high"] - d["ci_mean_diff"]["low"] > 2 * 1.96 * sd / np.sqrt(n)


def test_ba_loa_stays_196_sd(client):
    df = _ba_frame()
    d = _ba(client, df, "ba_loa")
    diffs = (df["m1"] - df["m2"]).to_numpy()
    m, sd = diffs.mean(), diffs.std(ddof=1)
    assert d["limits_of_agreement"]["lower"] == pytest.approx(m - 1.96 * sd, abs=1e-4)
    assert d["limits_of_agreement"]["upper"] == pytest.approx(m + 1.96 * sd, abs=1e-4)


def test_ba_loa_confidence_intervals(client):
    df = _ba_frame()
    d = _ba(client, df, "ba_loaci")
    diffs = (df["m1"] - df["m2"]).to_numpy()
    n = len(diffs)
    m, sd = diffs.mean(), diffs.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    se = sd * np.sqrt(1 / n + 1.96 ** 2 / (2 * (n - 1)))
    lo, up = m - 1.96 * sd, m + 1.96 * sd
    for key in (d, d["limits_of_agreement"]):
        assert key["ci_loa_lower"]["low"] == pytest.approx(lo - t * se, abs=1e-4)
        assert key["ci_loa_lower"]["high"] == pytest.approx(lo + t * se, abs=1e-4)
        assert key["ci_loa_upper"]["low"] == pytest.approx(up - t * se, abs=1e-4)
        assert key["ci_loa_upper"]["high"] == pytest.approx(up + t * se, abs=1e-4)
    # each limit lies inside its own interval
    assert d["ci_loa_lower"]["low"] < lo < d["ci_loa_lower"]["high"]
    assert d["ci_loa_upper"]["low"] < up < d["ci_loa_upper"]["high"]


def test_ba_alpha_changes_ci_level(client):
    df = _ba_frame()
    d05 = _ba(client, df, "ba_a05")
    d10 = _ba(client, df, "ba_a10", alpha=0.10)
    w05 = d05["ci_mean_diff"]["high"] - d05["ci_mean_diff"]["low"]
    w10 = d10["ci_mean_diff"]["high"] - d10["ci_mean_diff"]["low"]
    assert w10 < w05
    # the limits themselves do not move with alpha
    assert d10["limits_of_agreement"]["lower"] == d05["limits_of_agreement"]["lower"]


def test_ba_text_and_export_rows_report_loa_cis(client):
    d = _ba(client, _ba_frame(), "ba_text")
    labels = [row[0] for row in d["export_rows"]]
    assert any("Lower LOA" in s and "CI" in s for s in labels)
    assert any("Upper LOA" in s and "CI" in s for s in labels)
    assert "lower limit" in d["result_text"] and "upper limit" in d["result_text"]


def test_ba_invalid_alpha_400(client):
    s = make_session(_ba_frame(), "ba_badalpha")
    r = client.post("/api/agreement/bland_altman",
                    json={"session_id": s, "method1": "m1", "method2": "m2", "alpha": 1.5})
    assert r.status_code == 400


# ── Passing-Bablok ───────────────────────────────────────────────────────────

def _pb(client, df, sid, **kw):
    s = make_session(df, sid)
    r = client.post("/api/agreement/passing_bablok",
                    json={"session_id": s, "method1": "x", "method2": "y", **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _reference_pb(x, y, alpha=0.05):
    """Loop implementation straight from Passing & Bablok (1983)."""
    n = len(x)
    slopes = []
    for i, j in itertools.combinations(range(n), 2):
        dx, dy = x[j] - x[i], y[j] - y[i]
        if dx == 0 and dy == 0:
            continue
        s = np.copysign(np.inf, dy) if dx == 0 else dy / dx
        if s == -1:
            continue
        slopes.append(s)
    s = sorted(slopes)
    big_n = len(s)
    k = sum(1 for v in s if v < -1)
    if big_n % 2:
        b = s[(big_n + 1) // 2 + k - 1]
    else:
        b = (s[big_n // 2 + k - 1] + s[big_n // 2 + k]) / 2
    a = float(np.median(np.asarray(y) - b * np.asarray(x)))
    c = stats.norm.ppf(1 - alpha / 2) * np.sqrt(n * (n - 1) * (2 * n + 5) / 18)
    m1 = int(round((big_n - c) / 2))
    m2 = big_n - m1 + 1
    lo, hi = s[m1 + k - 1], s[m2 + k - 1]
    return {"slope": b, "intercept": a, "lo": lo, "hi": hi, "N": big_n, "K": k,
            "a_lo": float(np.median(np.asarray(y) - hi * np.asarray(x))),
            "a_hi": float(np.median(np.asarray(y) - lo * np.asarray(x)))}


def _noisy(n=40, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(100, 20, n)
    return pd.DataFrame({"x": x, "y": 0.95 * x + 3 + rng.normal(0, 5, n)})


def test_pb_exact_line_gives_slope_2_intercept_1(client):
    x = np.arange(1.0, 21.0)
    d = _pb(client, pd.DataFrame({"x": x, "y": 2 * x + 1}), "pb_exact")
    assert d["slope"] == 2.0 and d["intercept"] == 1.0
    assert d["ci_slope"] == {"low": 2.0, "high": 2.0}
    assert d["ci_intercept"] == {"low": 1.0, "high": 1.0}
    assert d["cusum_p"] == 1.0


def test_pb_matches_reference_implementation(client):
    df = _noisy()
    d = _pb(client, df, "pb_ref")
    ref = _reference_pb(df["x"].tolist(), df["y"].tolist())
    assert d["slope"] == pytest.approx(ref["slope"], abs=1e-4)
    assert d["intercept"] == pytest.approx(ref["intercept"], abs=1e-4)
    assert d["ci_slope"]["low"] == pytest.approx(ref["lo"], abs=1e-4)
    assert d["ci_slope"]["high"] == pytest.approx(ref["hi"], abs=1e-4)
    assert d["ci_intercept"]["low"] == pytest.approx(ref["a_lo"], abs=1e-4)
    assert d["ci_intercept"]["high"] == pytest.approx(ref["a_hi"], abs=1e-4)
    assert d["n_slopes"] == ref["N"] and d["K"] == ref["K"]


def test_pb_hand_checked_tiny_dataset(client):
    # x = 1..10, y = x plus a small +/-0.5 zigzag: 45 pairwise slopes, none equal to -1,
    # K = 0, so the fit is the plain median slope and must sit very close to 1.
    x = np.arange(1.0, 11.0)
    y = x + np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 1]) / 2
    d = _pb(client, pd.DataFrame({"x": x, "y": y}), "pb_tiny")
    ref = _reference_pb(x.tolist(), y.tolist())
    assert d["n_slopes"] == ref["N"] == 45
    assert d["slope"] == pytest.approx(ref["slope"], abs=1e-4)
    assert d["ci_slope"]["low"] <= d["slope"] <= d["ci_slope"]["high"]
    assert d["ci_slope"]["low"] < 1 < d["ci_slope"]["high"]


def test_pb_swapping_methods_gives_reciprocal_slope(client):
    df = _noisy(n=60, seed=9)
    d1 = _pb(client, df, "pb_swap1")
    swapped = df.rename(columns={"x": "y", "y": "x"})
    d2 = _pb(client, swapped, "pb_swap2")
    assert d1["slope"] * d2["slope"] == pytest.approx(1.0, abs=2e-4)


def test_pb_response_keys_preserved_and_extended(client):
    d = _pb(client, _noisy(), "pb_keys")
    for key in ("slope", "intercept", "ci_slope", "ci_intercept", "cusum_p", "assumptions",
                "summary", "export_rows", "interpretation", "result_text", "r_code"):
        assert key in d
    assert d["method"] == "Passing-Bablok (1983)"
    assert d["n_slopes"] > 0 and isinstance(d["K"], int)
    assert d["assumptions"][0]["name"].startswith("Linearity")


def test_pb_cusum_flags_curvature(client):
    rng = np.random.default_rng(5)
    x = np.linspace(1, 50, 80)
    y = 0.02 * (x - 25) ** 2 + x + rng.normal(0, 0.3, 80)
    d = _pb(client, pd.DataFrame({"x": x, "y": y}), "pb_curve")
    assert d["cusum_p"] < 0.05
    assert d["assumptions"][0]["met"] is False


def test_pb_cusum_accepts_linear_data(client):
    d = _pb(client, _noisy(n=80, seed=21), "pb_lin")
    assert d["cusum_p"] > 0.05
    assert d["assumptions"][0]["met"] is True


def test_pb_ties_in_x_are_handled(client):
    rng = np.random.default_rng(2)
    x = np.repeat(np.arange(1.0, 9.0), 3)
    y = 1.5 * x + 2 + rng.normal(0, 0.4, len(x))
    d = _pb(client, pd.DataFrame({"x": x, "y": y}), "pb_ties")
    # vertical pairs (dx = 0, dy != 0) are +/-inf slopes in the paper but do not break the fit
    assert d["slope"] == pytest.approx(1.5, abs=0.15)
    assert d["ci_slope"]["low"] is not None


def test_pb_too_few_pairs_400(client):
    s = make_session(pd.DataFrame({"x": np.arange(5.0), "y": np.arange(5.0)}), "pb_small")
    r = client.post("/api/agreement/passing_bablok", json={"session_id": s, "method1": "x", "method2": "y"})
    assert r.status_code == 400
