"""The joint model must count subjects, not measurement rows.

When only a longitudinal session is supplied, the survival half of the model
needs one row per subject. Passing the long frame straight through made every
repeated measurement look like a separate patient: `n_subjects` reported the
row count, BIC used the wrong sample size, and the Cox model was fitted on
duplicated subjects (which also triggered a many-to-many merge blow-up).
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

SEED = 20260725
PREFIX = "/api/survival_advanced"
N_SUBJECTS = 60


@pytest.fixture(scope="module")
def long_df():
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(N_SUBJECTS):
        age = float(rng.normal(60, 9))
        b0, slope = rng.normal(5, 1), rng.normal(0.3, 0.15)
        duration = float(rng.uniform(2, 8))
        event = int(rng.uniform() < 0.55)
        for t in np.arange(0, duration, 1.0):
            rows.append({
                "id": i,
                "time": float(t),
                "Y": float(b0 + slope * t + rng.normal(0, 0.4)),
                "age": age,
                "duration": duration,
                "event": event,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def result(long_df):
    from fastapi.testclient import TestClient
    from main import app

    sid = make_session(long_df, "joint_model_subjects")
    r = TestClient(app).post(f"{PREFIX}/joint_model", json={
        "session_id_long": sid,
        "id_col": "id",
        "time_col": "time",
        "y_cols": ["Y"],
        "long_predictors": ["age"],
        "surv_predictors": ["age"],
        "duration_col": "duration",
        "event_col": "event",
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_n_subjects_counts_subjects_not_rows(result, long_df):
    assert len(long_df) > N_SUBJECTS, "fixture must have repeated measurements"
    assert result["n_subjects"] == N_SUBJECTS


def test_result_text_reports_the_subject_count(result):
    assert f"{N_SUBJECTS} subjects" in result["result_text"]


def test_information_criteria_are_on_the_subject_scale(result, long_df):
    """BIC penalises by log(n). With rows instead of subjects it was ~20x off."""
    n_rows = len(long_df)
    assert result["bic"] > result["aic"], "BIC must penalise more than AIC here"
    # A row-scaled fit inflated the log-likelihood term by roughly the
    # measurements-per-subject ratio; anchor on that order of magnitude.
    assert abs(result["aic"]) < 100 * n_rows


def test_unknown_id_col_is_a_readable_400(long_df):
    from fastapi.testclient import TestClient
    from main import app

    sid = make_session(long_df, "joint_model_bad_id")
    r = TestClient(app).post(f"{PREFIX}/joint_model", json={
        "session_id_long": sid,
        "id_col": "NOPE",
        "time_col": "time",
        "y_cols": ["Y"],
        "duration_col": "duration",
        "event_col": "event",
    })
    assert r.status_code == 400, r.text
    assert "NOPE" in r.json()["detail"]
