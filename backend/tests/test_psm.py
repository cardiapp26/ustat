"""Characterization tests for Propensity Score Matching (PSM).

These tests exist primarily to lock current numerical behavior before
large-scale refactoring of models.py. They are intentionally conservative:
- Small, fully deterministic synthetic datasets
- Fixed random seeds
- Focus on the pure statistical helper functions + key endpoint invariants

When refactoring PSM logic (especially matching, SMD calculation, Rosenbaum bounds),
these tests should continue to pass with only minimal tolerance adjustments.
"""

import numpy as np
import pandas as pd
from conftest import make_session

from services.psm import (
    _compute_smd,
    _variance_ratio,
    _rosenbaum_bounds,
    _fit_propensity_scores,
)


# ──────────────────────────────────────────────────────────────────────────────
# Pure helper function tests (highest value for refactoring safety)
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_smd_continuous():
    """Basic SMD for continuous variable (Austin 2011 pooled-SD convention)."""
    treated = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    control = pd.Series([5.0, 6.0, 7.0, 8.0, 9.0])

    smd = _compute_smd(treated, control)
    # Mean diff = 5.0, pooled SD should be ~3.162 (sqrt( (2.5^2 + 2.5^2)/2 ) * sqrt(2)?)
    # Exact value with ddof=1 pooled SD denominator:
    # sd_t = 1.5811, sd_c = 1.5811 → pooled = 1.5811
    # smd = 5.0 / 1.5811 ≈ 3.162
    assert 3.15 < smd < 3.18


def test_compute_smd_binary():
    """SMD for binary variable using the correct denominator."""
    treated = pd.Series([1, 1, 1, 1, 0])
    control = pd.Series([0, 0, 0, 1, 0])

    smd = _compute_smd(treated, control)
    # p1=0.8, p0=0.2 → denom = sqrt( (0.8*0.2 + 0.2*0.8)/2 ) = sqrt(0.16) = 0.4
    # abs(0.8-0.2)/0.4 = 1.5
    assert abs(smd - 1.5) < 0.001


def test_variance_ratio():
    """Rubin's variance ratio should be >1 when treated has higher variance."""
    treated = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0])
    control = pd.Series([4.0, 5.0, 6.0, 7.0, 8.0])

    vr = _variance_ratio(treated, control)
    assert vr is not None
    assert vr > 1.0


def test_rosenbaum_bounds_classic_example():
    """
    Classic Rosenbaum bounds example with known discordant pairs.

    b=20 (treated better), c=5 → strongly significant at Γ=1.
    At high Γ the upper bound p-value should exceed 0.05.
    """
    # 25 discordant pairs: 20 favorable to treatment, 5 unfavorable
    pairs = [(1, 0)] * 20 + [(0, 1)] * 5

    res = _rosenbaum_bounds(pairs, gamma_max=4.0, n_gamma=40, alpha=0.05)

    assert res["applicable"] is True
    assert res["b"] == 20
    assert res["c"] == 5
    assert res["discordant_pairs"] == 25
    assert res["p_unbiased"] < 0.01   # very strong signal at Γ=1 (small fp drift tolerated after extraction)

    # At Γ=1 the upper bound p should still be tiny (small fp drift tolerated)
    p_at_1 = [r["p_upper"] for r in res["curve"] if r["gamma"] < 1.01][0]
    assert p_at_1 < 0.01

    # There must be some critical gamma where it crosses alpha=0.05
    assert res["critical_gamma"] is not None
    assert 1.0 < res["critical_gamma"] < 4.0


def test_rosenbaum_no_discordant_pairs():
    """When there are no discordant pairs, Rosenbaum bounds are inapplicable."""
    pairs = [(1, 1), (0, 0), (1, 1)]
    res = _rosenbaum_bounds(pairs)
    assert res["applicable"] is False
    assert "No discordant" in res["reason"]


# ──────────────────────────────────────────────────────────────────────────────
# Propensity score fitting (internal)
# ──────────────────────────────────────────────────────────────────────────────

def test_fit_propensity_scores_logistic_deterministic():
    """Logistic propensity model should be reproducible with fixed seed."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 3))
    # Simple linear truth
    logit = -0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2]
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)

    ps1 = _fit_propensity_scores(X, y, "logistic", random_state=123)
    ps2 = _fit_propensity_scores(X, y, "logistic", random_state=123)

    np.testing.assert_allclose(ps1, ps2, rtol=1e-10)
    assert np.all((ps1 >= 0) & (ps1 <= 1))


# ──────────────────────────────────────────────────────────────────────────────
# Full endpoint characterization tests (synthetic but realistic)
# ──────────────────────────────────────────────────────────────────────────────

def test_psm_basic_1to1_greedy_reduces_smd(client):
    """
    End-to-end sanity check: after 1:1 greedy PSM with reasonable caliper,
    average SMD across covariates should drop substantially.
    """
    rng = np.random.default_rng(123)
    n = 400

    # Confounded data: treatment assignment depends on X1, X2
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)  # noise covariate

    logit_t = -0.8 + 1.5 * x1 + 1.2 * x2
    p_t = 1 / (1 + np.exp(-logit_t))
    treat = (rng.uniform(0, 1, n) < p_t).astype(int)

    # Outcome also depends on treatment + confounders (for later use)
    y = (0.8 * treat + 0.6 * x1 + 0.4 * x2 + rng.normal(0, 1, n) > 0).astype(int)

    df = pd.DataFrame({
        "treat": treat,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "outcome": y,
    })

    sid = make_session(df, "psm_basic_1")

    payload = {
        "session_id": sid,
        "treatment_col": "treat",
        "covariates": ["x1", "x2", "x3"],
        "caliper": 0.2,
        "caliper_scale": "logit",
        "ratio": 1,
        "matching_method": "greedy",
        "score_method": "logistic",
        "random_state": 42,
        "outcome_type": "binary",
        "outcome_col": "outcome",
    }

    r = client.post("/api/models/psm", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()

    # Basic invariants
    assert data["n_matched_pairs"] > 30          # should find decent matches
    assert data["avg_smd_after"] < data["avg_smd_before"] * 0.6  # meaningful reduction
    assert data["balance_achieved"] in (True, False)  # just present
    assert data["matched_session_id"] == sid + "_psm"

    # The matched data should actually exist in the store
    from services import store as store_module
    matched_df = store_module.get(sid + "_psm")
    assert matched_df is not None
    assert len(matched_df) == data["n_matched_pairs"] * 2


def test_psm_rosenbaum_is_returned_when_requested(client):
    """When compute_rosenbaum=True on 1:1 binary outcome, the field must be present and well-formed."""
    rng = np.random.default_rng(7)
    n = 300

    x = rng.normal(0, 1, n)
    logit_t = -1.0 + 2.0 * x
    p_t = 1 / (1 + np.exp(-logit_t))
    treat = (rng.uniform(0, 1, n) < p_t).astype(int)

    # Strong treatment effect on binary outcome
    y = (treat * 1.8 + rng.normal(0, 1.5, n) > 0).astype(int)

    df = pd.DataFrame({"t": treat, "x": x, "y": y})
    sid = make_session(df, "psm_rosen")

    payload = {
        "session_id": sid,
        "treatment_col": "t",
        "covariates": ["x"],
        "caliper": 0.25,
        "ratio": 1,
        "matching_method": "greedy",
        "score_method": "logistic",
        "outcome_type": "binary",
        "outcome_col": "y",
        "compute_rosenbaum": True,
        "rosenbaum_gamma_max": 3.0,
    }

    r = client.post("/api/models/psm", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "rosenbaum" in data
    rb = data["rosenbaum"]
    assert rb["applicable"] is True
    assert "critical_gamma" in rb
    assert isinstance(rb["curve"], list)
    assert len(rb["curve"]) > 5


# ──────────────────────────────────────────────────────────────────────────────
# Bad option strings must read as 400, not 500
# ──────────────────────────────────────────────────────────────────────────────

def _score_method_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.normal(0, 1, n)
    p_t = 1 / (1 + np.exp(-(-0.5 + 1.2 * x)))
    treat = (rng.uniform(0, 1, n) < p_t).astype(int)
    y = (0.7 * treat + 0.5 * x + rng.normal(0, 1, n) > 0).astype(int)
    return pd.DataFrame({"t": treat, "x": x, "y": y})


def test_psm_unknown_score_method_400_not_500(client):
    """score_method is a free-form string; an unknown value is caller input.

    services.psm signals it with ValueError; unwrapped that became a 500 with
    no detail, so the UI could not say which option was wrong.
    """
    sid = make_session(_score_method_df(31), "psm_bad_score_method")
    r = client.post("/api/models/psm", json={
        "session_id": sid,
        "treatment_col": "t",
        "covariates": ["x"],
        "outcome_col": "y",
        "score_method": "random_forest",
    })
    assert r.status_code == 400, r.text
    assert "random_forest" in r.json()["detail"]


def test_iptw_unknown_score_method_400_not_500(client):
    """Same defect on the IPTW endpoint, which shares the propensity fitter."""
    sid = make_session(_score_method_df(32), "iptw_bad_score_method")
    r = client.post("/api/models/iptw", json={
        "session_id": sid,
        "treatment_col": "t",
        "covariates": ["x"],
        "outcome_col": "y",
        "score_method": "random_forest",
    })
    assert r.status_code == 400, r.text
    assert "random_forest" in r.json()["detail"]
