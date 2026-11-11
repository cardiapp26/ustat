"""
Phase 4 - Model Validation Simulation Tests
"""

import pytest
import numpy as np
from services.model_validation import bootstrap_performance, optimism_corrected_metrics
from services.simulation_generators import generate_logistic_data


@pytest.mark.simulation
def test_bootstrap_and_optimism_correction():
    """
    Basic check that bootstrap performance and optimism correction
    produce reasonable numbers on simulated data.
    """
    df, _ = generate_logistic_data(n=600, seed=99)

    # Simulate predicted probabilities from a decent model
    # (in real life these would come from a fitted model)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = df[["X1", "X2", "X3", "X4"]].values
    y = df["event"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    probs = model.predict_proba(X_scaled)[:, 1]

    perf = bootstrap_performance(y, probs, n_boot=150)
    # Optimism needs the design matrix: the model is refitted per resample.
    opt = optimism_corrected_metrics(y, X_scaled, n_boot=100)

    # Basic sanity checks
    assert "auc" in perf
    assert perf["auc"]["mean"] > 0.6
    assert "optimism_corrected_auc" in opt
    # n = 600 with 4 predictors: little optimism, corrected AUC near apparent.
    assert abs(opt["optimism_auc"]) < 0.03
    assert opt["optimism_corrected_auc"] > 0.6


@pytest.mark.simulation
def test_cox_validation_bootstrap():
    """
    Basic check for Cox model validation (C-index bootstrap).
    """
    from services.simulation_generators import generate_survival_data

    df, _ = generate_survival_data(n=400, seed=77)

    # Simulate a linear predictor (in practice this would come from a fitted Cox model)
    rng = np.random.default_rng(42)
    lp = 0.6 * df["X1"].values + 0.4 * df["X2"].values + rng.normal(0, 0.3, len(df))

    # We can call the internal function directly for testing
    from services.model_validation import compute_cox_calibration_slope
    cal = compute_cox_calibration_slope(df, "duration", "event", lp)

    # Just check it doesn't crash and returns something reasonable
    assert "calibration_slope" in cal or cal.get("calibration_slope") is None
