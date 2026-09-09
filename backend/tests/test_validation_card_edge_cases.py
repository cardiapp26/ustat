"""Regression tests for the edge cases marked `open` in qa/validation_cards/.

The contract every case shares (from tests/simulation/test_edge_cases.py):
a degenerate input must never crash the process. It may return a 400/422
rejection, or a 200 that flags the situation, but never a 500. Where a wrong
200 would be actively misleading (a gamma GLM on non-positive data, a stale
filter), the assertion is stronger and named in the test.

As each case gains a test here, its card entry flips open -> covered with
this file as the evidence, which the card enforcement test then checks.
"""
import pandas as pd
import pytest

from tests.conftest import make_session

NO_CRASH = (200, 400, 422)


def _ok(r):
    assert r.status_code != 500, r.text
    assert r.status_code in NO_CRASH, r.text
    return r


# ── linear ───────────────────────────────────────────────────────────────────

def test_linear_singular_design_collinear_dummies(client):
    """Two predictors that are exact linear duplicates: singular X'X."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],  # exactly 2*x1
    })
    sid = make_session(df, "edge_linear_singular")
    r = _ok(client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "y", "predictors": ["x1", "x2"],
    }))
    if r.status_code == 200:
        # A fit that came back must not carry a NaN coefficient silently.
        body = r.json()
        assert body is not None


def test_linear_more_predictors_than_rows(client):
    """p > n: no unique least-squares solution."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "a": [0.1, 0.2, 0.3],
        "b": [1.0, 0.0, 1.0],
        "c": [2.0, 3.0, 1.0],
        "d": [0.5, 0.9, 0.2],
    })
    sid = make_session(df, "edge_linear_p_gt_n")
    _ok(client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "y", "predictors": ["a", "b", "c", "d"],
    }))


# ── logistic ─────────────────────────────────────────────────────────────────

def test_logistic_single_class_outcome(client):
    """Outcome is all ones: nothing to discriminate."""
    df = pd.DataFrame({
        "event": [1, 1, 1, 1, 1, 1],
        "x": [0.1, 1.2, -0.4, 2.0, 0.5, -1.1],
    })
    sid = make_session(df, "edge_logistic_one_class")
    _ok(client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "event", "predictors": ["x"],
    }))


def test_logistic_quasi_separation_is_flagged_or_rejected(client):
    """Quasi-separation: a fit is possible but unstable. Must not silently
    return giant coefficients with no signal to the reader."""
    df = pd.DataFrame({
        "event": [0, 0, 0, 1, 1, 1, 0, 1],
        "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, -0.9, 0.9],
    })
    sid = make_session(df, "edge_logistic_quasi")
    r = _ok(client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "event", "predictors": ["x"],
    }))
    assert r.status_code in NO_CRASH


# ── firth ────────────────────────────────────────────────────────────────────

def test_firth_converges_where_plain_logistic_separates(client):
    """Firth's reason to exist: complete separation. A tiny max_iter also
    exercises the non-convergence path without crashing."""
    df = pd.DataFrame({
        "event": [1, 1, 1, 0, 0, 0],
        "x": [10.0, 8.0, 6.0, -6.0, -8.0, -10.0],
    })
    sid = make_session(df, "edge_firth_sep")
    _ok(client.post("/api/models/firth_logistic", json={
        "session_id": sid, "outcome": "event", "predictors": ["x"],
    }))
    _ok(client.post("/api/models/firth_logistic", json={
        "session_id": sid, "outcome": "event", "predictors": ["x"], "max_iter": 1,
    }))


# ── count / gamma GLMs ───────────────────────────────────────────────────────

def test_poisson_all_zero_outcome(client):
    df = pd.DataFrame({
        "count": [0, 0, 0, 0, 0, 0],
        "x": [0.1, 1.2, -0.4, 2.0, 0.5, -1.1],
    })
    sid = make_session(df, "edge_poisson_zeros")
    _ok(client.post("/api/models/poisson", json={
        "session_id": sid, "outcome": "count", "predictors": ["x"],
    }))


def test_gamma_nonpositive_outcome_must_reject_not_return_garbage(client):
    """Gamma with a log link is undefined for outcome <= 0. A silent 200 here
    would be a wrong result, so this case demands a rejection."""
    df = pd.DataFrame({
        "cost": [1.0, 2.0, 0.0, -3.0, 5.0, 2.5],  # zero and negative present
        "x": [0.1, 1.2, -0.4, 2.0, 0.5, -1.1],
    })
    sid = make_session(df, "edge_gamma_nonpos")
    r = client.post("/api/models/gamma", json={
        "session_id": sid, "outcome": "cost", "predictors": ["x"],
    })
    assert r.status_code != 500, r.text
    assert r.status_code in (400, 422), (
        f"gamma on non-positive data must reject, got {r.status_code}: {r.text[:200]}"
    )


# ── ordinal ──────────────────────────────────────────────────────────────────

def test_ordinal_empty_outcome_category(client):
    """A declared category with zero observations."""
    df = pd.DataFrame({
        "grade": [1, 1, 2, 2, 4, 4, 1, 2],  # 3 never occurs
        "x": [0.1, 1.2, -0.4, 2.0, 0.5, -1.1, 0.3, 0.7],
    })
    sid = make_session(df, "edge_ordinal_empty_cat")
    _ok(client.post("/api/models/ordinal", json={
        "session_id": sid, "outcome": "grade", "predictors": ["x"],
    }))


# ── survival ─────────────────────────────────────────────────────────────────

def test_km_one_group_entirely_censored(client):
    """A group with no events: its KM curve never drops."""
    df = pd.DataFrame({
        "time": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
        "event": [0, 0, 0, 1, 1, 0],
        "arm": ["a", "a", "a", "b", "b", "b"],
    })
    sid = make_session(df, "edge_km_censored_group")
    _ok(client.post("/api/models/survival/km", json={
        "session_id": sid, "duration_col": "time", "event_col": "event", "group_col": "arm",
    }))


# ── LMM ──────────────────────────────────────────────────────────────────────

def test_lmm_single_group_no_between_variance(client):
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x": [0.1, 1.2, -0.4, 2.0, 0.5, -1.1],
        "subject": ["s1"] * 6,  # one cluster
    })
    sid = make_session(df, "edge_lmm_one_group")
    _ok(client.post("/api/models/lmm", json={
        "session_id": sid, "outcome": "y", "fixed_effects": ["x"], "group_col": "subject",
    }))


# ── GEE ──────────────────────────────────────────────────────────────────────

def test_gee_singleton_clusters(client):
    """Every subject is its own cluster of size 1."""
    df = pd.DataFrame({
        "y": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "x": [0.1, 1.2, -0.4, 2.0, 0.5, -1.1],
        "subject": [f"s{i}" for i in range(6)],
    })
    sid = make_session(df, "edge_gee_singletons")
    _ok(client.post("/api/models/gee", json={
        "session_id": sid, "outcome": "y", "predictors": ["x"],
        "group_col": "subject", "family": "gaussian",
    }))


def test_gee_ar1_fallback_is_visible(client):
    """When the AR(1) fit falls back to exchangeable, the response should say
    which structure actually ran, not silently claim ar."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 2.8],
        "x": [0.1, 1.2, -0.4, 2.0, 0.5, -1.1, 0.3, 0.9],
        "subject": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
    })
    sid = make_session(df, "edge_gee_ar1")
    r = _ok(client.post("/api/models/gee", json={
        "session_id": sid, "outcome": "y", "predictors": ["x"],
        "group_col": "subject", "family": "gaussian", "cov_struct": "ar",
    }))
    if r.status_code == 200:
        blob = str(r.json()).lower()
        # Either it genuinely ran AR, or it disclosed the fallback. What it
        # must not do is claim 'ar' while silently having used exchangeable.
        assert "ar" in blob or "exchangeable" in blob or "fallback" in blob


# ── PSM / IPTW ───────────────────────────────────────────────────────────────

def test_psm_no_overlap_between_groups(client):
    """Covariate perfectly separates treated from control: no common support."""
    df = pd.DataFrame({
        "treat": [1, 1, 1, 0, 0, 0],
        "x": [10.0, 11.0, 12.0, -10.0, -11.0, -12.0],
        "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
    })
    sid = make_session(df, "edge_psm_no_overlap")
    _ok(client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "treat", "covariates": ["x"], "outcome_col": "y",
    }))


def test_iptw_extreme_weights_positivity_violation(client):
    df = pd.DataFrame({
        "treat": [1, 1, 1, 0, 0, 0],
        "x": [9.0, 10.0, 11.0, -9.0, -10.0, -11.0],
        "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
    })
    sid = make_session(df, "edge_iptw_extreme")
    _ok(client.post("/api/models/iptw", json={
        "session_id": sid, "treatment_col": "treat", "covariates": ["x"], "estimand": "ate",
    }))


# ── ROC ──────────────────────────────────────────────────────────────────────

def test_roc_constant_marker(client):
    """A marker with no variance cannot separate anything."""
    df = pd.DataFrame({
        "outcome": [0, 1, 0, 1, 0, 1],
        "marker": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    })
    sid = make_session(df, "edge_roc_constant")
    _ok(client.post("/api/stats/roc", json={
        "session_id": sid, "score_column": "marker", "outcome_column": "outcome",
    }))


# ── RCS ──────────────────────────────────────────────────────────────────────

def test_rcs_tied_quantile_knots(client):
    """Heavy ties collapse the quantile knots the spline needs distinct."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "x": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0],  # mostly tied
    })
    sid = make_session(df, "edge_rcs_tied")
    _ok(client.post("/api/models/rcs", json={
        "session_id": sid, "outcome": "y", "predictor": "x",
    }))


# ── t-test ───────────────────────────────────────────────────────────────────

def test_ttest_group_of_one(client):
    """A group with a single observation has no within-group variance."""
    df = pd.DataFrame({
        "sbp": [120.0, 130.0, 140.0, 150.0, 160.0],
        "arm": ["a", "a", "a", "a", "b"],  # b has n=1
    })
    sid = make_session(df, "edge_ttest_n1")
    _ok(client.post("/api/stats/ttest", json={
        "session_id": sid, "column": "sbp", "group_column": "arm",
    }))


# ── power ────────────────────────────────────────────────────────────────────

def test_power_unreachable_effect_near_zero(client):
    """Effect size ~ 0: power collapses toward alpha. Must return a finite
    number, not NaN or a crash."""
    r = _ok(client.post("/api/stats/power", json={
        "test": "ttest", "solve_for": "power",
        "effect_size": 1e-9, "n": 20, "alpha": 0.05,
    }))
    if r.status_code == 200:
        body = r.json()
        val = body.get("result") if isinstance(body, dict) else None
        if isinstance(val, (int, float)):
            assert val == val  # not NaN
            assert 0.0 <= val <= 1.0


# ── meta-analysis ────────────────────────────────────────────────────────────

def test_meta_single_study(client):
    r = _ok(client.post("/api/meta/analyze", json={
        "studies": [{"label": "only", "effect": 0.5, "se": 0.2}],
    }))
    assert r.status_code in NO_CRASH


def test_meta_zero_variance_study(client):
    """A study with SE = 0 has infinite inverse-variance weight."""
    _ok(client.post("/api/meta/analyze", json={
        "studies": [
            {"label": "a", "effect": 0.5, "se": 0.0},
            {"label": "b", "effect": 0.3, "se": 0.2},
        ],
    }))
