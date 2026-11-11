"""Regression tests: recurrent-event follow-up, NB dispersion, PWP label, joint model."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from fastapi.testclient import TestClient

from services.joint_model import fit_latent_class_joint_model, fit_time_varying_joint_model

PREFIX = "/api/survival_advanced"


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


def _recurrent_frame(seed: int = 7, n: int = 150) -> pd.DataFrame:
    """Gamma-frailty recurrent events: true NB dispersion alpha = 0.5."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        trt = int(rng.integers(0, 2))
        end = float(rng.uniform(20, 40))
        rate = 0.15 * np.exp(-0.5 * trt) * rng.gamma(2.0, 0.5)
        t = 0.0
        while True:
            nxt = t + float(rng.exponential(1 / rate))
            if nxt >= end:
                rows.append({"id": i, "start": t, "stop": end, "event": 0, "trt": trt})
                break
            rows.append({"id": i, "start": t, "stop": nxt, "event": 1, "trt": trt})
            t = nxt
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def recurrent(client):
    df = _recurrent_frame()
    sid = make_session(df, "advsurv_recurrent")
    out = {}
    for scale in ("total", "gap"):
        r = client.post(f"{PREFIX}/recurrent_lwyy", json={
            "session_id": sid, "id_col": "id", "start_col": "start", "stop_col": "stop",
            "event_col": "event", "predictors": ["trt"], "model_type": "both",
            "time_scale": scale, "include_negative_binomial": True,
        })
        assert r.status_code == 200, r.text
        out[scale] = r.json()
    return df, out


def test_follow_up_is_the_same_on_the_gap_scale(recurrent):
    df, out = recurrent
    per = df.groupby("id")
    expected = float((per["stop"].max() - per["start"].min()).sum())
    assert out["total"]["total_followup"] == pytest.approx(expected, abs=1e-3)
    assert out["gap"]["total_followup"] == pytest.approx(expected, abs=1e-3)


def test_negative_binomial_estimates_alpha_and_ignores_time_scale(recurrent):
    _, out = recurrent
    nb_total = out["total"]["negative_binomial"]
    nb_gap = out["gap"]["negative_binomial"]
    assert nb_total["available"], nb_total
    assert 0.2 < nb_total["alpha"] < 0.9, "true dispersion is 0.5, not the GLM default of 1"
    assert [c["variable"] for c in nb_total["coefficients"]] == ["const", "trt"]
    assert nb_gap["coefficients"] == nb_total["coefficients"]


def test_event_order_stratified_model_is_labelled_pwp(recurrent):
    _, out = recurrent
    assert "Prentice-Williams-Peterson" in out["total"]["wlw"]["model"]
    assert "gap time" in out["gap"]["wlw"]["model"]
    assert "fit_pwp" in out["total"]["r_code"]


def _joint_frames(seed: int = 21, n: int = 200, sex_effect: float = 3.0):
    rng = np.random.default_rng(seed)
    long_rows, surv_rows = [], []
    for i in range(n):
        sex = int(rng.integers(0, 2))
        b0, b1 = rng.normal(0, 0.8), rng.normal(0, 0.1)
        duration = float(rng.uniform(3, 9))
        level = 5 + sex_effect * sex + b0
        event = int(rng.uniform() < 1 / (1 + np.exp(-(level - 6.5) * 0.6)))
        surv_rows.append({"id": i, "duration": duration, "event": event, "sex": sex})
        for t in np.arange(0, duration, 1.0):
            y = level + (0.2 + b1) * t + rng.normal(0, 0.3)
            long_rows.append({"id": i, "time": float(t), "Y": float(y), "sex": sex})
    return pd.DataFrame(long_rows), pd.DataFrame(surv_rows)


def test_trajectory_value_includes_longitudinal_fixed_effects():
    """The association value must track the observed marker, sex effect included."""
    from services import joint_model as jm

    long_df, surv_df = _joint_frames()
    captured = {}
    original = jm._extract_lmm_predictions

    def spy(trajectory, df, time_col, id_col, association):
        preds = original(trajectory, df, time_col, id_col, association)
        captured["preds"] = preds
        return preds

    jm._extract_lmm_predictions = spy
    try:
        fit_time_varying_joint_model(long_df, surv_df, long_predictors=["sex"], association=["value"])
    finally:
        jm._extract_lmm_predictions = original
    preds = captured["preds"].merge(surv_df[["id", "sex"]], on="id")
    gap = preds.groupby("sex")["value"].mean().diff().iloc[-1]
    observed_gap = long_df.groupby("sex")["Y"].mean().diff().iloc[-1]
    assert gap == pytest.approx(observed_gap, abs=0.4)
    assert gap > 2.0, "without the sex fixed effect the two groups had the same value"


def test_spline_trajectory_is_subject_specific_with_slope_and_area():
    long_df, surv_df = _joint_frames(n=120)
    res = fit_time_varying_joint_model(
        long_df, surv_df, long_predictors=["sex"], association=["value", "slope", "area"],
        time_spline=True,
    )
    by_var = {c["variable"]: c for c in res["cox_coefficients"]}
    assert set(by_var) == {"Y_value", "Y_slope", "Y_area"}
    assert all(c["se"] is not None and np.isfinite(c["coef"]) for c in by_var.values())


def test_linear_slope_and_area_match_closed_form():
    from services import joint_model as jm
    import statsmodels.formula.api as smf

    long_df, _ = _joint_frames(n=60)
    mdf = smf.mixedlm("Y ~ time + sex", long_df, groups=long_df["id"], re_formula="~time").fit(reml=False)
    traj = jm._Trajectory(mdf, long_df, "id", "time", "time", ["sex"])
    ids = np.array([0, 1, 2])
    times = np.array([2.0, 4.0, 6.0])
    v0 = traj.value(ids, np.zeros(3))
    slope = traj.slope(ids, times)
    assert traj.value(ids, times) == pytest.approx(v0 + slope * times)
    assert traj.area(ids, times) == pytest.approx(v0 * times + 0.5 * slope * times ** 2)
    expected_slope = [mdf.params["time"] + mdf.random_effects[g]["time"] for g in ids]
    assert slope == pytest.approx(expected_slope, abs=1e-6)


def test_missing_marker_values_do_not_crash_and_penalizer_is_reported():
    long_df, surv_df = _joint_frames(n=80)
    long_df.loc[[3, 17, 40], "Y"] = np.nan
    res = fit_time_varying_joint_model(long_df, surv_df, long_predictors=["sex"])
    assert res["cox_penalizer"] == 0.0
    assert "without penalisation" in res["note"]
    res_lc = fit_latent_class_joint_model(long_df, surv_df, latent_classes=2)
    assert "cox_penalizer" in res_lc


def test_latent_classes_enter_cox_as_indicators():
    long_df, surv_df = _joint_frames(n=150)
    res = fit_latent_class_joint_model(long_df, surv_df, latent_classes=3)
    names = [c["variable"] for c in res["cox_coefficients"]]
    assert names == ["latent_class_1", "latent_class_2"]


def test_unknown_association_is_a_422(client):
    long_df, _ = _joint_frames(n=40)
    long_df = long_df.merge(_joint_frames(n=40)[1][["id", "duration", "event"]], on="id")
    sid = make_session(long_df, "advsurv_joint_bad_assoc")
    r = client.post(f"{PREFIX}/joint_model", json={
        "session_id_long": sid, "id_col": "id", "time_col": "time", "y_cols": ["Y"],
        "duration_col": "duration", "event_col": "event", "association": ["bogus"],
    })
    assert r.status_code == 422, r.text
    assert "bogus" in r.json()["detail"]
