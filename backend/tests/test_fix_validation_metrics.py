"""Regression tests for the model-validation fixes in services/model_validation.py.

Each test pins one defect that produced a wrong number without an error:
optimism correction that paired outcomes with shuffled predictions, a
calibration curve evaluated on a different spline basis from the fitted one,
censored subjects counted as event-free, and a linear predictor misread as a
probability.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix

from services.model_validation import (
    add_validation_to_result,
    competing_risks_calibration,
    compute_calibration_slope_intercept,
    compute_cox_calibration_slope,
    flexible_calibration_curve,
    ipcw_horizon_weights,
    optimism_corrected_metrics,
)


def _logistic_data(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    lp = -0.3 + 1.0 * X[:, 0] - 0.7 * X[:, 1]
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    return X, y


# ── optimism correction ───────────────────────────────────────────────────────

def test_optimism_is_small_for_a_large_well_specified_model():
    """n = 1000 with 2 predictors: the corrected AUC stays near the apparent one.

    The old code returned a corrected AUC of about 0.5 here.
    """
    X, y = _logistic_data(1000, 2, seed=1)
    opt = optimism_corrected_metrics(y, X, n_boot=60)
    assert opt["apparent_auc"] > 0.72
    assert abs(opt["optimism_auc"]) < 0.02
    assert opt["optimism_corrected_auc"] > 0.70
    assert 0.9 < opt["optimism_corrected_calibration_slope"] < 1.05
    assert opt["n_boot_used"] == 60


def test_optimism_is_large_for_an_overfitted_model():
    """n = 80 with 20 noise predictors: apparent AUC is high, corrected is not."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 20))
    y = rng.binomial(1, 0.5, 80)
    opt = optimism_corrected_metrics(y, X, n_boot=60)
    assert opt["apparent_auc"] > 0.75
    assert opt["optimism_auc"] > 0.15
    assert opt["optimism_corrected_auc"] < 0.65
    assert opt["optimism_corrected_calibration_slope"] < 0.6


def test_optimism_rejects_predicted_probabilities():
    X, y = _logistic_data(200, 2, seed=3)
    probs = np.full(len(y), 0.5)
    with pytest.raises(ValueError, match="design matrix"):
        optimism_corrected_metrics(y, probs)


def test_add_validation_without_design_matrix_reports_unavailable():
    X, y = _logistic_data(200, 2, seed=4)
    probs = 1 / (1 + np.exp(-(X[:, 0] - 0.7 * X[:, 1])))
    original: dict = {"test": "x"}
    out = add_validation_to_result(original, y, probs, n_boot=30)
    assert original == {"test": "x"}
    assert out["optimism_correction"]["available"] is False
    with_x = add_validation_to_result(original, y, probs, n_boot=30, X=X)
    assert "optimism_corrected_auc" in with_x["optimism_correction"]


# ── calibration slope scale ───────────────────────────────────────────────────

def test_linear_predictor_inside_unit_interval_is_not_read_as_probability():
    """An LP confined to [0, 1] must give slope 1 when the model is correct."""
    rng = np.random.default_rng(5)
    lp = rng.uniform(0.0, 1.0, 4000)
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    explicit = compute_calibration_slope_intercept(y, linear_predictor=lp)
    reference = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params
    assert explicit["calibration_slope"] == pytest.approx(reference[1], abs=1e-3)
    assert explicit["calibration_intercept"] == pytest.approx(reference[0], abs=1e-3)
    # The legacy guess treats the same numbers as probabilities.
    guessed = compute_calibration_slope_intercept(y, probs_or_lp=lp)
    assert abs(guessed["calibration_slope"] - explicit["calibration_slope"]) > 0.3
    stated = compute_calibration_slope_intercept(y, probs_or_lp=lp, scale="lp")
    assert stated == explicit


def test_probabilities_outside_unit_interval_are_rejected():
    y = np.array([0, 1] * 20)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_calibration_slope_intercept(y, probs=np.linspace(-1, 2, 40))


def test_cox_calibration_slope_ignores_series_index():
    """A Series with a shifted index must be used by position, not by label."""
    rng = np.random.default_rng(6)
    n = 300
    lp = rng.normal(size=n)
    t = rng.exponential(1 / np.exp(lp))
    df = pd.DataFrame({"t": t, "e": np.ones(n, dtype=int)})
    by_array = compute_cox_calibration_slope(df, "t", "e", lp)
    by_series = compute_cox_calibration_slope(df, "t", "e", pd.Series(lp, index=np.arange(n) + 1000))
    assert by_array["calibration_slope"] is not None
    assert by_series == by_array


# ── flexible calibration curve ────────────────────────────────────────────────

def test_flexible_curve_is_evaluated_on_the_fitted_basis():
    rng = np.random.default_rng(7)
    n = 1500
    lp_true = rng.normal(0, 1.6, n)
    probs = 1 / (1 + np.exp(-lp_true))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.4 + 0.6 * lp_true))))

    out = flexible_calibration_curve(y, probs)
    assert out["available"] is True

    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    frame = pd.DataFrame({"y": y.astype(float), "lp": np.log(clipped / (1 - clipped))})
    X = dmatrix("cr(lp, df=4)", frame, return_type="dataframe")
    model = sm.GLM(frame["y"], X, family=sm.families.Binomial()).fit()
    grid_p = np.array([pt["predicted"] for pt in out["curve"]])
    Xg = build_design_matrices([X.design_info], {"lp": np.log(grid_p / (1 - grid_p))}, return_type="dataframe")[0]
    expected = np.asarray(model.predict(Xg))
    observed = np.array([pt["observed"] for pt in out["curve"]])
    np.testing.assert_allclose(observed, expected, atol=2e-4)


# ── censoring before the horizon ──────────────────────────────────────────────

def _censored_cohort(seed: int, n: int = 6000) -> tuple[np.ndarray, np.ndarray, float]:
    """Exponential event times (rate 0.10) with heavy independent censoring."""
    rng = np.random.default_rng(seed)
    t_event = rng.exponential(10.0, n)
    t_cens = rng.exponential(8.0, n)
    duration = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)
    return duration, event, 1 - float(np.exp(-0.5))  # true risk at t = 5


def test_ipcw_weights_recover_the_true_risk():
    duration, event, true_risk = _censored_cohort(8)
    w = ipcw_horizon_weights(duration, event, 5.0)
    obs = ((duration <= 5.0) & (event == 1)).astype(float)
    assert np.all(w[(event == 0) & (duration < 5.0)] == 0)
    assert np.average(obs, weights=w) == pytest.approx(true_risk, abs=0.02)
    # Counting the censored as event-free, as the old code did, is far off.
    assert obs.mean() < true_risk - 0.08


def test_flexible_curve_with_horizon_is_not_biased_by_censoring():
    duration, event, true_risk = _censored_cohort(9)
    rng = np.random.default_rng(10)
    probs = np.clip(true_risk + rng.normal(0, 0.02, len(duration)), 0.01, 0.99)
    out = flexible_calibration_curve(event, probs, duration=duration, time_horizon=5.0)
    assert out["available"] is True
    assert out["censoring_adjustment"] == "ipcw"
    assert out["n_censored_before_horizon"] > 0
    assert "weight 0" in out["note"]
    mid = out["curve"][len(out["curve"]) // 2]
    assert mid["observed"] == pytest.approx(true_risk, abs=0.04)


def test_competing_risks_calibration_uses_ipcw():
    rng = np.random.default_rng(11)
    n = 6000
    t1, t2, tc = rng.exponential(10.0, n), rng.exponential(20.0, n), rng.exponential(8.0, n)
    duration = np.minimum.reduce([t1, t2, tc])
    status = np.where(duration == tc, 0, np.where(duration == t1, 1, 2))
    # True cause-1 cumulative incidence at t = 5 with total hazard 0.15.
    true_cif = (0.10 / 0.15) * (1 - np.exp(-0.15 * 5))
    df = pd.DataFrame({"t": duration, "s": status, "cif": true_cif + rng.normal(0, 0.005, n)})
    out = competing_risks_calibration(df, "t", "s", "cif", time_horizon=5.0, n_bins=4)
    assert out["available"] is True
    assert out["censoring_adjustment"] == "ipcw"
    for row in out["bins"]:
        assert row["observed_cumulative_incidence"] == pytest.approx(true_cif, abs=0.04)
        assert row["n_censored_before_horizon"] > 0
