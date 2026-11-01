"""Absolute risk F(t0 | x) from Cox-RCS (Austin 2025) against R survival.

Reference: survival::coxph on the Harrell 4-knot basis of age in
qa/models_audit/dataset.csv, then summary(survfit(fit, newdata), times = 12)
with the default conf.type = "log". The data have no tied event times, so
lifelines' Efron fit and R's agree to optimiser noise.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from lifelines import CoxPHFitter

from services.rcs_absolute_risk import absolute_risk
from services.rcs_basis import harrell_knots, rcs_basis

_AUDIT_CSV = Path(__file__).resolve().parents[2] / "qa" / "models_audit" / "dataset.csv"
AGES = np.array([40.0, 50.0, 62.2, 75.0, 85.0])

R_AGE_ONLY = {
    "risk": [0.4703612630, 0.4388000108, 0.5178278896, 0.7054278124, 0.8081515764],
    "low": [0.2404179521, 0.3418247121, 0.4314781047, 0.6025536874, 0.4828324857],
    "high": [0.6306953376, 0.5214870056, 0.5910624623, 0.7816742262, 0.9288319227],
}
R_AGE_BMI = {
    "risk": [0.4829224935, 0.4436948621, 0.5136565217, 0.7046839205, 0.8184476839],
    "low": [0.2483768515, 0.3458900791, 0.4270820245, 0.6016800733, 0.4929963742],
    "high": [0.6442776566, 0.5268755348, 0.5871486164, 0.7810514088, 0.9349881504],
}


@pytest.fixture(scope="module")
def audit_df():
    return pd.read_csv(_AUDIT_CSV)


def _fit_and_predict(df: pd.DataFrame, with_bmi: bool):
    x = df["age"].to_numpy(float)
    knots = harrell_knots(x, 4)
    design = np.column_stack([x, rcs_basis(x, knots)])
    new = np.column_stack([AGES, rcs_basis(AGES, knots)])
    if with_bmi:
        design = np.column_stack([design, df["bmi"].to_numpy(float)])
        new = np.column_stack([new, np.full(len(AGES), df["bmi"].mean())])
    fit_df = pd.DataFrame(design, columns=[f"c{i}" for i in range(design.shape[1])])
    fit_df["T"] = df["time"].to_numpy(float)
    fit_df["E"] = df["status"].to_numpy(float)
    cph = CoxPHFitter().fit(fit_df, "T", "E")
    return absolute_risk(
        design, fit_df["T"].to_numpy(), fit_df["E"].to_numpy(),
        cph.params_.to_numpy(), cph.variance_matrix_.to_numpy(), new, 12.0,
    )


@pytest.mark.parametrize("with_bmi,ref", [(False, R_AGE_ONLY), (True, R_AGE_BMI)])
def test_absolute_risk_matches_survfit(audit_df, with_bmi, ref):
    ar = _fit_and_predict(audit_df, with_bmi)
    np.testing.assert_allclose(ar.risk, ref["risk"], atol=2e-4)
    np.testing.assert_allclose(ar.ci_low, ref["low"], atol=2e-4)
    np.testing.assert_allclose(ar.ci_high, ref["high"], atol=2e-4)


def test_risk_is_zero_before_first_event(audit_df):
    x = audit_df["age"].to_numpy(float)
    design = x[:, None]
    ar = absolute_risk(design, audit_df["time"].to_numpy(float), audit_df["status"].to_numpy(float),
                       np.array([0.02]), np.array([[1e-4]]), np.array([[50.0]]), 1e-9)
    assert ar.risk[0] == 0.0 and ar.ci_low[0] == 0.0 and ar.ci_high[0] == 0.0


def _post(client, df, name, **extra):
    sid = make_session(df, name)
    return client.post("/api/models/rcs", json={
        "session_id": sid, "predictor": "age", "n_knots": 4, "model_type": "cox",
        "duration_col": "time", "event_col": "status", **extra,
    })


def test_endpoint_returns_curve_at_median_follow_up_by_default(client, audit_df):
    r = _post(client, audit_df, "ar_default")
    assert r.status_code == 200, r.text
    ar = r.json()["absolute_risk"]
    assert ar["horizon_is_default"] is True
    assert ar["horizon"] == pytest.approx(float(np.median(audit_df["time"])))
    assert len(ar["risk"]) == len(ar["x_values"]) == 200
    risk, lo, hi = (np.array(ar[k]) for k in ("risk", "ci_low", "ci_high"))
    assert np.all((lo <= risk) & (risk <= hi)) and np.all((risk > 0) & (risk < 1))
    assert ar["n_at_risk"] == int((audit_df["time"] >= ar["horizon"]).sum())


def test_endpoint_uses_requested_horizon(client, audit_df):
    r = _post(client, audit_df, "ar_h12", risk_horizon=12)
    ar = r.json()["absolute_risk"]
    assert ar["horizon"] == 12 and ar["horizon_is_default"] is False
    # Risk at the grid point closest to age 62.2 tracks the survfit value.
    xs = np.array(ar["x_values"])
    i = int(np.argmin(np.abs(xs - 62.2)))
    assert ar["risk"][i] == pytest.approx(0.5178, abs=0.01)


def test_endpoint_rejects_horizon_beyond_follow_up(client, audit_df):
    r = _post(client, audit_df, "ar_h_bad", risk_horizon=float(audit_df["time"].max()) + 1)
    assert r.status_code == 422
    assert "risk_horizon" in r.text


def test_non_cox_models_have_no_absolute_risk(client, audit_df):
    sid = make_session(audit_df, "ar_logit")
    r = client.post("/api/models/rcs", json={"session_id": sid, "predictor": "age",
                                            "model_type": "logistic", "outcome": "event_binary"})
    assert r.status_code == 200
    assert r.json()["absolute_risk"] is None
