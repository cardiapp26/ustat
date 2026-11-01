"""Coverage tests for routers/models_survival.py.

Exercises the five POST endpoints mounted under /api/models:
  * /survival/km
  * /survival/cox
  * /survival/cox_tv   (long-format start/stop)
  * /rcs               (continuous predictor + knots)
  * /survival/cox_rcs  (1-2 spline terms + optional interaction)

Tests assert response shape and sane numeric ranges, not exact floats.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session

SEED = 20260531


# ── Synthetic data ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def surv_df():
    """Standard survival frame: duration/event + continuous & categorical preds."""
    rng = np.random.default_rng(SEED)
    n = 240
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    sex = rng.integers(0, 2, n)
    dm = rng.integers(0, 2, n)
    # Event probability mildly depends on AGE/DM so Cox has signal.
    lp = -2.0 + 0.03 * (age - 60) + 0.4 * dm
    p = 1 / (1 + np.exp(-lp))
    event = (rng.uniform(0, 1, n) < p).astype(int)
    duration = rng.exponential(500, n).clip(1, 1825)
    sex_str = np.where(sex == 1, "M", "F")
    return pd.DataFrame({
        "AGE": age,
        "LDL": ldl,
        "SEX": sex,
        "SEX_STR": sex_str,
        "DM": dm,
        "event": event,
        "duration": duration,
    })


@pytest.fixture(scope="module")
def surv_sid(surv_df):
    return make_session(surv_df, "tsurv_main")


@pytest.fixture(scope="module")
def tv_df():
    """Long-format (subject, interval) frame for cox_tv: start/stop per row."""
    rng = np.random.default_rng(SEED + 1)
    n_subj = 80
    rows = []
    for sid in range(n_subj):
        n_int = rng.integers(1, 4)  # 1-3 intervals per subject
        t = 0.0
        age = float(rng.normal(60, 10))
        for k in range(n_int):
            stop = t + float(rng.uniform(10, 200))
            ldl = float(rng.normal(120, 30))
            last = (k == n_int - 1)
            evt = int(last and rng.uniform() < 0.4)
            rows.append({
                "pid": sid,
                "start": t,
                "stop": stop,
                "ev": evt,
                "AGE": age,
                "LDL": ldl,
            })
            t = stop
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tv_sid(tv_df):
    return make_session(tv_df, "tsurv_tv")


# ── /survival/km ─────────────────────────────────────────────────────────────

def test_km_basic(client, surv_sid):
    r = client.post("/api/models/survival/km", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "Kaplan-Meier"
    assert len(body["groups"]) == 1
    grp = body["groups"][0]
    assert grp["n"] > 0
    assert grp["events"] >= 0
    assert isinstance(grp["curve"], list) and len(grp["curve"]) > 0
    assert body["logrank"] is None


def test_km_grouped_logrank(client, surv_sid):
    r = client.post("/api/models/survival/km", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
        "group_col": "DM",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["groups"]) == 2
    assert body["logrank"] is not None
    assert body["logrank"]["test"] == "Log-rank"
    p = body["logrank"]["p"]
    assert p is None or 0.0 <= p <= 1.0


def test_km_stratified(client, surv_sid):
    r = client.post("/api/models/survival/km", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
        "group_col": "DM",
        "stratify_col": "SEX",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "strata" in body
    assert body["stratify_col"] == "SEX"
    assert len(body["strata"]) >= 1
    for stratum in body["strata"]:
        assert "groups" in stratum and len(stratum["groups"]) >= 1


def test_km_nonbinary_event_422(client, surv_df):
    df = surv_df.copy()
    df["bad_event"] = 2  # not 0/1
    sid = make_session(df, "tsurv_km_bad")
    r = client.post("/api/models/survival/km", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "bad_event",
    })
    assert r.status_code == 422, r.text


# ── /survival/cox ────────────────────────────────────────────────────────────

def test_cox_basic(client, surv_sid):
    r = client.post("/api/models/survival/cox", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["AGE", "LDL"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "Cox Proportional Hazards"
    assert body["n"] > 0
    names = {c["variable"] for c in body["coefficients"]}
    assert {"AGE", "LDL"} <= names
    for c in body["coefficients"]:
        assert c["hr"] is None or c["hr"] > 0
        assert c["p"] is None or 0.0 <= c["p"] <= 1.0
    assert body["concordance"] is None or 0.0 <= body["concordance"] <= 1.0
    assert "ph_test" in body


def test_cox_categorical_predictor(client, surv_sid):
    """A string SEX column should be dummy-encoded, not crash."""
    r = client.post("/api/models/survival/cox", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["AGE", "SEX_STR"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    names = {c["variable"] for c in body["coefficients"]}
    # AGE numeric stays; SEX_STR becomes a dummy like SEX_STR_M.
    assert "AGE" in names
    assert any(n.startswith("SEX_STR") for n in names)


def test_cox_interaction(client, surv_sid):
    r = client.post("/api/models/survival/cox", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["AGE", "LDL"],
        "interactions": [["AGE", "LDL"]],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "AGE:LDL" in body["interactions_used"]


def test_cox_nonbinary_event_422(client, surv_df):
    df = surv_df.copy()
    df["bad_event"] = 3
    sid = make_session(df, "tsurv_cox_bad")
    r = client.post("/api/models/survival/cox", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "bad_event",
        "predictors": ["AGE"],
    })
    assert r.status_code == 422, r.text


# ── /survival/cox_tv ─────────────────────────────────────────────────────────

def test_cox_tv_basic(client, tv_sid):
    r = client.post("/api/models/survival/cox_tv", json={
        "session_id": tv_sid,
        "id_col": "pid",
        "start_col": "start",
        "stop_col": "stop",
        "event_col": "ev",
        "predictors": ["AGE", "LDL"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "time-varying" in body["model"]
    assert body["n_subjects"] > 0
    assert body["n_intervals"] >= body["n_subjects"]
    names = {c["variable"] for c in body["coefficients"]}
    assert {"AGE", "LDL"} <= names
    assert isinstance(body["result_text"], str) and len(body["result_text"]) > 0


def test_cox_tv_missing_column_422(client, tv_sid):
    r = client.post("/api/models/survival/cox_tv", json={
        "session_id": tv_sid,
        "id_col": "pid",
        "start_col": "start",
        "stop_col": "stop",
        "event_col": "ev",
        "predictors": ["NOPE"],
    })
    assert r.status_code == 422, r.text


def test_cox_tv_stop_le_start_422(client):
    """Rows with stop <= start must be rejected."""
    rng = np.random.default_rng(SEED + 2)
    n = 40
    df = pd.DataFrame({
        "pid": np.arange(n),
        "start": np.full(n, 10.0),
        "stop": np.full(n, 5.0),  # stop < start
        "ev": rng.integers(0, 2, n),
        "AGE": rng.normal(60, 10, n),
    })
    sid = make_session(df, "tsurv_tv_bad")
    r = client.post("/api/models/survival/cox_tv", json={
        "session_id": sid,
        "id_col": "pid",
        "start_col": "start",
        "stop_col": "stop",
        "event_col": "ev",
        "predictors": ["AGE"],
    })
    assert r.status_code == 422, r.text


# ── /rcs ─────────────────────────────────────────────────────────────────────

def test_rcs_logistic(client, surv_sid):
    r = client.post("/api/models/rcs", json={
        "session_id": surv_sid,
        "predictor": "AGE",
        "outcome": "event",
        "model_type": "logistic",
        "n_knots": 4,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_type"] == "logistic"
    assert body["effect_type"] == "OR"
    assert len(body["knots"]) == 4
    assert len(body["x_values"]) == len(body["or_values"]) == 200
    assert body["n"] > 0


def test_rcs_cox(client, surv_sid):
    r = client.post("/api/models/rcs", json={
        "session_id": surv_sid,
        "predictor": "AGE",
        "model_type": "cox",
        "duration_col": "duration",
        "event_col": "event",
        "n_knots": 3,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["effect_type"] == "HR"
    assert len(body["knots"]) == 3
    assert body["n_events"] is not None and body["n_events"] >= 0
    assert len(body["x_values"]) == 200


def test_rcs_cox_with_covariate(client, surv_sid):
    """Adjusted Cox-RCS should emit a crude (unadjusted) overlay block."""
    r = client.post("/api/models/rcs", json={
        "session_id": surv_sid,
        "predictor": "AGE",
        "model_type": "cox",
        "duration_col": "duration",
        "event_col": "event",
        "covariates": ["LDL"],
        "n_knots": 4,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["covariates_used"] == ["LDL"]
    assert body["crude"] is not None
    assert len(body["crude"]["x_values"]) == 200


def test_rcs_bad_knots_422(client, surv_sid):
    r = client.post("/api/models/rcs", json={
        "session_id": surv_sid,
        "predictor": "AGE",
        "outcome": "event",
        "model_type": "logistic",
        "n_knots": 8,  # Harrell table covers 3 to 7
    })
    assert r.status_code == 422, r.text


def test_rcs_cox_missing_duration_422(client, surv_sid):
    r = client.post("/api/models/rcs", json={
        "session_id": surv_sid,
        "predictor": "AGE",
        "model_type": "cox",
        "n_knots": 4,
        # duration_col / event_col omitted
    })
    assert r.status_code == 422, r.text


# ── /survival/cox_rcs ────────────────────────────────────────────────────────

def test_cox_rcs_single_term(client, surv_sid):
    r = client.post("/api/models/survival/cox_rcs", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
        "spline_terms": [{"column": "AGE", "n_knots": 4}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] > 0
    assert body["n_events"] >= 5
    assert 0.0 <= body["concordance"] <= 1.0
    assert len(body["spline_terms"]) == 1
    assert len(body["curves_1d"]) == 1
    assert "AGE" in body["nonlinearity"]
    assert body["surface_2d"] is None


def test_cox_rcs_two_terms_interaction(client, surv_sid):
    r = client.post("/api/models/survival/cox_rcs", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
        "spline_terms": [
            {"column": "AGE", "n_knots": 3},
            {"column": "LDL", "n_knots": 3},
        ],
        "include_interaction": True,
        "grid_size": 20,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["spline_terms"]) == 2
    assert len(body["curves_1d"]) == 2
    assert body["interaction"] is not None
    assert body["surface_2d"] is not None
    surf = body["surface_2d"]
    assert len(surf["x"]) == 20 and len(surf["y"]) == 20
    assert len(surf["hr"]) == 20


def test_cox_rcs_interaction_requires_two_terms_422(client, surv_sid):
    r = client.post("/api/models/survival/cox_rcs", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
        "spline_terms": [{"column": "AGE", "n_knots": 4}],
        "include_interaction": True,  # invalid with only 1 term
    })
    assert r.status_code == 422, r.text


def test_cox_rcs_too_many_terms_422(client, surv_sid):
    r = client.post("/api/models/survival/cox_rcs", json={
        "session_id": surv_sid,
        "duration_col": "duration",
        "event_col": "event",
        "spline_terms": [
            {"column": "AGE", "n_knots": 4},
            {"column": "LDL", "n_knots": 4},
            {"column": "AGE", "n_knots": 4},
        ],
    })
    assert r.status_code == 422, r.text
