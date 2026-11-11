"""Regression tests: RMST tail, landmark Cox coding, landmark boundary, CIF."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from fastapi.testclient import TestClient

from services.survival_advanced_service import (
    _aalen_johansen_cif,
    _landmark_cox_design,
    _rmst_one_group_raw,
)

PREFIX = "/api/survival_advanced"


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


def test_rmst_extends_flat_to_tau_like_survfit():
    """R: print(survfit(Surv(c(1,2,3,4), c(1,0,1,0)) ~ 1), rmean=5) -> 3.25 (se 0.839)."""
    res = _rmst_one_group_raw(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 0, 1, 0]), 5.0)
    assert res["rmst"] == pytest.approx(3.25)
    assert res["se"] == pytest.approx(0.8385, abs=1e-3)


def test_rmst_all_censored_before_tau_is_tau():
    res = _rmst_one_group_raw(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]), 10.0)
    assert res["rmst"] == pytest.approx(10.0)


def test_rmst_unchanged_when_tau_within_follow_up():
    res = _rmst_one_group_raw(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 0, 1, 0]), 3.5)
    assert res["rmst"] == pytest.approx(1 + 0.75 * 2 + 0.375 * 0.5)


def test_rmst_endpoint_warns_when_group_follow_up_ends_before_tau(client):
    rng = np.random.default_rng(3)
    short = pd.DataFrame({"t": rng.uniform(0.5, 4, 30), "e": rng.integers(0, 2, 30), "g": "short"})
    long = pd.DataFrame({"t": rng.uniform(0.5, 10, 30), "e": rng.integers(0, 2, 30), "g": "long"})
    sid = make_session(pd.concat([short, long], ignore_index=True), "advsurv_rmst_tail")
    r = client.post(f"{PREFIX}/rmst", json={
        "session_id": sid, "duration_col": "t", "event_col": "e", "group_col": "g", "tau": 8.0,
    })
    assert r.status_code == 200, r.text
    assert any("short" in w and "before tau" in w for w in r.json()["warnings"])


def test_landmark_design_dummy_codes_nominal_predictors():
    frame = pd.DataFrame({
        "t": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "e": [1, 0, 1, 1, 0, 1],
        "stage": ["I", "II", "III", "I", "II", "III"], "age": [50, 60, 70, 55, 65, 75],
    })
    design = _landmark_cox_design(frame, "t", "e", ["stage", "age", "stage"])
    assert list(design.columns) == ["t", "e", "age", "stage_II", "stage_III"]
    assert design["stage_II"].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]


def _landmark_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 300
    stage = rng.choice(["I", "II", "III"], n)
    rate = np.select([stage == "II"], [0.25], default=0.08)
    t = np.round(rng.exponential(1 / rate), 0) + 1
    e = (t < 20).astype(int)
    return pd.DataFrame({"t": np.minimum(t, 20), "e": e, "stage": stage})


def test_landmark_cox_reports_one_row_per_stage_level(client):
    sid = make_session(_landmark_frame(), "advsurv_landmark_stage")
    r = client.post(f"{PREFIX}/landmark", json={
        "session_id": sid, "duration_col": "t", "event_col": "e",
        "landmark_time": 2, "predictors": ["stage"],
    })
    assert r.status_code == 200, r.text
    rows = {row["variable"]: row for row in r.json()["cox_results"]}
    assert set(rows) == {"stage_II", "stage_III"}
    assert rows["stage_II"]["HR"] > 1.8
    assert 0.6 < rows["stage_III"]["HR"] < 1.6


def test_landmark_excludes_events_at_the_landmark(client):
    df = _landmark_frame()
    sid = make_session(df, "advsurv_landmark_boundary")
    r = client.post(f"{PREFIX}/landmark", json={
        "session_id": sid, "duration_col": "t", "event_col": "e", "landmark_time": 3,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert (df["t"] == 3).sum() > 0, "fixture needs subjects exactly at the landmark"
    assert d["n_landmark"] == int((df["t"] > 3).sum())
    assert d["plot"]["data"][0]["y"][0] == pytest.approx(1.0)
    assert "at or before the landmark" in d["result_text"]


def test_aalen_johansen_handles_ties_exactly():
    """Hand calculation: t=1 (n=4): one event 1, one event 2; t=2 (n=2): one event 1."""
    times, cif = _aalen_johansen_cif(np.array([1.0, 1.0, 2.0, 3.0]), np.array([1, 2, 1, 0]), 1)
    assert times == [0.0, 1.0, 2.0, 3.0]
    assert cif == pytest.approx([0.0, 0.25, 0.25 + 0.5 * 0.5, 0.5])


def test_fine_gray_cif_is_deterministic_with_ties(client):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"t": rng.integers(1, 8, 120).astype(float), "e": rng.integers(0, 3, 120)})
    sid = make_session(df, "advsurv_cif_ties")
    body = {"session_id": sid, "duration_col": "t", "event_col": "e", "event_of_interest": 1}
    runs = [client.post(f"{PREFIX}/fine_gray", json=body) for _ in range(3)]
    assert all(r.status_code == 200 for r in runs), runs[0].text
    curves = [r.json()["plot"]["data"][0] for r in runs]
    assert curves[0] == curves[1] == curves[2]
    assert all(float(x).is_integer() for x in curves[0]["x"])
