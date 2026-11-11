"""Regression tests for the survival-ML benchmark and Cox-diagnostics fixes.

Covers: survival probabilities that ran backwards (high risk, high survival),
a partial hazard passed on as a linear predictor, an undisclosed ridge penalty
on the "classical" Cox model, string predictors entered as integer codes with
missing as a level, and the Cox diagnostics endpoint dropping every row when a
predictor was categorical.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from lifelines import CoxPHFitter, KaplanMeierFitter

from services.model_validation import optimism_corrected_metrics
from services.survival_ml import (
    _prepare_survival_design,
    _survival_probs_from_risk,
    run_survival_ml_benchmark,
)


def _cohort(n: int = 400, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 10, n)
    bmi = rng.normal(27, 4, n)
    stage = rng.choice(["I", "II", "III"], n)
    lp = 0.05 * (age - 60) - 0.04 * (bmi - 27) + 0.9 * (stage == "II") + 0.2 * (stage == "III")
    t = rng.exponential(8 / np.exp(lp))
    cens = rng.uniform(2, 15, n)
    return pd.DataFrame({
        "age": age, "bmi": bmi, "stage": stage,
        "duration": np.minimum(t, cens), "event": (t <= cens).astype(int),
    })


# ── survival probabilities from a risk score ──────────────────────────────────

def test_higher_risk_gives_lower_survival():
    df = _cohort()
    risk = 0.05 * (df["age"] - 60).to_numpy()
    times = np.percentile(df["duration"], [25, 50, 75])
    surv = _survival_probs_from_risk(risk, df["duration"].to_numpy(), df["event"].to_numpy(), times)
    assert surv.shape == (len(df), 3)
    order = np.argsort(risk)
    assert np.all(np.diff(surv[order], axis=0) <= 1e-12)
    assert np.all(np.diff(surv, axis=1) <= 1e-12)
    assert np.corrcoef(risk, surv[:, 1])[0, 1] < -0.9


def test_survival_probs_are_anchored_to_the_observed_baseline():
    """Mean predicted S(t) tracks Kaplan-Meier, unlike the old fixed ramp."""
    df = _cohort()
    risk = 0.05 * (df["age"] - 60).to_numpy()
    times = np.percentile(df["duration"], [25, 50, 75])
    surv = _survival_probs_from_risk(risk, df["duration"].to_numpy(), df["event"].to_numpy(), times)
    km = KaplanMeierFitter().fit(df["duration"], df["event"]).survival_function_at_times(times).to_numpy()
    np.testing.assert_allclose(surv.mean(axis=0), km, atol=0.03)


# ── design matrix ─────────────────────────────────────────────────────────────

def test_string_predictor_is_dummy_coded_and_missing_rows_dropped():
    df = _cohort(60)
    df.loc[[0, 1, 2, 3, 4], "stage"] = None
    X, durations, events, names, kept = _prepare_survival_design(df, "duration", "event", ["age", "stage"])
    assert names == ["age", "stage_II", "stage_III"]
    assert len(X) == 55 and len(durations) == 55 and len(events) == 55
    assert not set(kept) & {0, 1, 2, 3, 4}
    assert set(np.unique(X[["stage_II", "stage_III"]].to_numpy())) == {0.0, 1.0}


def test_event_must_be_zero_one():
    df = _cohort(60)
    df["event"] = df["event"] + 1
    with pytest.raises(ValueError, match="0 .*1"):
        _prepare_survival_design(df, "duration", "event", ["age"])


def test_non_positive_durations_are_dropped():
    df = _cohort(60)
    df.loc[[5, 6], "duration"] = [0.0, -1.0]
    X, durations, _, _, kept = _prepare_survival_design(df, "duration", "event", ["age"])
    assert len(X) == 58 and np.all(durations > 0)
    assert 5 not in kept and 6 not in kept


# ── benchmark ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bench() -> dict:
    return run_survival_ml_benchmark(
        _cohort(), "duration", "event", ["age", "bmi", "stage"],
        n_estimators=40, cv_folds=3, include_partial_dependence=False,
    )


def test_cox_baseline_is_unpenalised_linear_predictor(bench):
    df = _cohort()
    X, durations, events, _, _ = _prepare_survival_design(df, "duration", "event", ["age", "bmi", "stage"])
    ref = CoxPHFitter().fit(X.assign(duration=durations, event=events), "duration", "event")
    expected = ref.predict_log_partial_hazard(X).to_numpy()
    got = np.array(bench["validation_ready_risk_scores"]["cox_risk_scores"])
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert bench["classical_cox"]["penalizer"] == 0.0
    # In-sample Cox recalibration of an unpenalised Cox LP is exactly 1.
    assert bench["classical_cox"]["calibration_slope"] == pytest.approx(1.0, abs=1e-3)
    assert bench["ml_gradient_boosting_survival"]["calibration_slope"] is None
    assert bench["auto_comparison"]["models"][0]["calibration_slope"] == pytest.approx(1.0, abs=1e-3)


def test_benchmark_survival_probs_fall_with_risk(bench):
    probs = bench["validation_ready_survival_probs"]
    scores = bench["validation_ready_risk_scores"]
    for key_s, key_p in (("cox_risk_scores", "cox_survival_probs"), ("ml_risk_scores", "ml_survival_probs")):
        risk = np.array(scores[key_s])
        surv = np.array(probs[key_p])
        assert np.corrcoef(risk, surv[:, -1])[0, 1] < -0.5


def test_benchmark_ibs_beats_an_uninformative_model(bench):
    """The reversed curves scored an IBS above 0.25; correct ones are below."""
    for model in ("cox", "ml"):
        ibs = bench["full_phase9_validation"][model]["integrated_brier_score"]["ibs"]
        assert ibs is not None and ibs < 0.25


def test_benchmark_calibration_bins_run_in_the_right_direction(bench):
    assert bench["calibration_assessment"]["censoring_adjustment"] == "ipcw"
    for curve in bench["calibration_assessment"]["cox"]:
        bins = curve["bins"]
        assert bins[-1]["mean_predicted_event"] > bins[0]["mean_predicted_event"]
        assert bins[-1]["observed_event_rate"] > bins[0]["observed_event_rate"]
        assert curve["ici"] < 0.1


# ── endpoints ─────────────────────────────────────────────────────────────────

def test_cox_diagnostics_accepts_a_categorical_predictor(client):
    sid = make_session(_cohort(), "fix_validation_cox_diag")
    r = client.post("/api/model_diagnostics/cox_diagnostics", json={
        "session_id": sid, "duration_col": "duration", "event_col": "event",
        "predictors": ["age", "stage"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 400
    assert [row["variable"] for row in body["ph_test"]] == ["age", "stage_II", "stage_III"]
    assert all(row["p"] is not None for row in body["ph_test"])
    epv = next(a for a in body["assumptions"] if a["name"] == "Events per variable")
    assert "3 model terms" in epv["detail"]


def test_model_validation_probability_mode_does_not_invent_optimism(client):
    rng = np.random.default_rng(31)
    n = 300
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-x))
    sid = make_session(pd.DataFrame({"y": rng.binomial(1, p), "p": p}), "fix_validation_prob_mode")
    r = client.post("/api/model_diagnostics/model_validation", json={
        "session_id": sid, "outcome": "y", "prob_column": "p", "n_boot": 50,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["optimism_correction"]["available"] is False
    assert "predictors" in body["optimism_correction"]["reason"]
    assert body["bootstrap_performance"]["auc"]["mean"] > 0.6
    assert body["calibration"]["calibration_slope"] == pytest.approx(1.0, abs=0.35)


def test_optimism_function_matches_predictors_mode_direction():
    """Sanity link between the service function and a noise model."""
    rng = np.random.default_rng(32)
    X = rng.normal(size=(120, 12))
    y = rng.binomial(1, 0.5, 120)
    opt = optimism_corrected_metrics(y, X, n_boot=40)
    assert opt["optimism_auc"] > 0.05
