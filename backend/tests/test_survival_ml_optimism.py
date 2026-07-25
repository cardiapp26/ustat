"""The survival ML benchmark must not present optimistic numbers as honest ones.

`ml_risk = gbr.predict(X)` scores the gradient-boosting model on the same rows
it was fitted on, so its C-index is an apparent (resubstitution) value. On a
small dataset it can approach 1.0 while the cross-validated estimate sits at
chance. Reporting the two side by side without saying which is which invites
a false "ML beats Cox" conclusion, so the response has to label the apparent
values and warn when cross-validation contradicts them.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

SEED = 20260725
PREFIX = "/api/survival_advanced"


@pytest.fixture(scope="module")
def overfit_df():
    """Few events relative to the noise → boosting memorises the training rows."""
    rng = np.random.default_rng(SEED)
    n = 300
    age = rng.normal(62, 10, n)
    ldl = rng.normal(130, 30, n)
    sex = rng.integers(0, 2, n)
    lp = 0.04 * (age - 62) + 0.008 * (ldl - 130) + 0.3 * sex
    t = rng.exponential(1 / (0.05 * np.exp(lp)))
    cens = rng.uniform(1, 10)
    return pd.DataFrame({
        "age": age,
        "ldl": ldl,
        "sex": sex,
        "duration": np.minimum(t, cens),
        "event": (t <= cens).astype(int),
    })


@pytest.fixture(scope="module")
def sid_overfit(overfit_df):
    return make_session(overfit_df, "survival_ml_optimism_session")


@pytest.fixture(scope="module")
def benchmark(sid_overfit):
    # Module-scoped so the benchmark runs once for all four assertions; the
    # shared `client` fixture is function-scoped, so build a client here.
    from fastapi.testclient import TestClient
    from main import app

    r = TestClient(app).post(f"{PREFIX}/ml_survival_benchmark", json={
        "session_id": sid_overfit,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["age", "ldl", "sex"],
        "n_estimators": 100,
        "cv_folds": 3,
        "include_partial_dependence": False,
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_headline_c_indices_are_labelled_apparent(benchmark):
    assert benchmark["classical_cox"]["c_index_type"] == "apparent"
    assert benchmark["ml_gradient_boosting_survival"]["c_index_type"] == "apparent"


def test_result_text_distinguishes_apparent_from_cross_validated(benchmark):
    text = benchmark["result_text"].lower()
    assert "apparent" in text
    assert "cross-validated" in text


def test_assumptions_state_the_resubstitution_caveat(benchmark):
    joined = " ".join(benchmark["assumptions"]).lower()
    assert "apparent" in joined
    assert "repeated_cv" in joined


def test_overfitting_is_warned_about_when_cv_contradicts_the_apparent_value(benchmark):
    ml_c = benchmark["ml_gradient_boosting_survival"]["c_index"]
    cv_mean = benchmark["repeated_cv"]["summary"]["mean"]
    # Guard the premise: this fixture is chosen to overfit. If that stops being
    # true the assertion below would pass vacuously, so fail loudly instead.
    assert ml_c - cv_mean > 0.10, (
        f"fixture no longer overfits (apparent {ml_c} vs CV {cv_mean}); "
        "pick a harder dataset or this test proves nothing"
    )
    warnings = " ".join(benchmark["warnings"]).lower()
    assert "overfitting" in warnings
    assert "cross-validated" in warnings
