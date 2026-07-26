"""Rubin's-rules pooling: generic pooler + Fine-Gray multiple-imputation path."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store
from services.missing_data import pool_rubin_terms


def test_pool_rubin_terms_within_and_between():
    # Same estimate every imputation, se=0.1 → no between variance; pooled se=0.1.
    per = [{"x": (1.0, 0.1)} for _ in range(5)]
    out = pool_rubin_terms(per)["x"]
    assert abs(out["coef"] - 1.0) < 1e-9
    assert abs(out["se"] - 0.1) < 1e-6
    assert out["fmi"] == 0.0

    # Between-imputation spread inflates the pooled SE above the within-se.
    per2 = [{"x": (c, 0.1)} for c in [0.8, 0.9, 1.0, 1.1, 1.2]]
    out2 = pool_rubin_terms(per2)["x"]
    assert abs(out2["coef"] - 1.0) < 1e-9
    assert out2["se"] > 0.1           # T = Ubar + (1+1/m)·B
    assert out2["fmi"] > 0.0


def _seed_competing():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(0, 1, n)
    t = rng.exponential(np.exp(-0.4 * x) * 5).clip(0.1, 20)
    ev = rng.choice([0, 1, 2], n, p=[0.3, 0.45, 0.25])
    df = pd.DataFrame({"time": t, "status": ev, "x": x, "grp": rng.integers(0, 2, n)})
    df.loc[rng.choice(n, 40, replace=False), "x"] = np.nan
    store.save("fg_mi", df)
    return "fg_mi"


def test_fine_gray_mice_is_pooled():
    sid = _seed_competing()
    client = TestClient(app)
    base = {"session_id": sid, "duration_col": "time", "event_col": "status",
            "event_of_interest": 1, "predictors": ["x", "grp"]}
    r = client.post("/api/survival_advanced/fine_gray", json={**base, "imputation": "mice"})
    assert r.status_code == 200, r.text
    rr = r.json()["regression_result"]
    assert rr["mi_pooled"] is True
    assert rr["n_imputations"] >= 2
    by = {c["variable"]: c for c in rr["coefficients"]}
    assert "x" in by and by["x"]["shr"] > 0 and by["x"]["shr_low"] is not None
    # x carried the missingness → higher fraction of missing information than grp.
    assert by["x"]["fmi"] > by["grp"]["fmi"]


def test_landmark_cox_mice_is_pooled():
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(0, 1, n)
    t = rng.exponential(np.exp(-0.5 * x) * 6).clip(0.1, 30)
    ev = (t < rng.exponential(15, n)).astype(int)
    df = pd.DataFrame({"time": t, "event": ev, "x": x, "grp": rng.integers(0, 2, n)})
    df.loc[rng.choice(n, 60, replace=False), "x"] = np.nan
    store.save("lm_mi", df)
    client = TestClient(app)
    base = {"session_id": "lm_mi", "duration_col": "time", "event_col": "event",
            "landmark_time": 2.0, "predictors": ["x", "grp"]}
    r = client.post("/api/survival_advanced/landmark", json={**base, "imputation": "mice"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["cox_mi_note"] and "pooled" in j["cox_mi_note"].lower()
    by = {c["variable"]: c for c in j["cox_results"]}
    assert by["x"]["HR"] > 1 and by["x"]["fmi"] > by["grp"]["fmi"]


def test_rmst_mice_pools_group_covariate():
    rng = np.random.default_rng(8)
    n = 300
    grp = rng.integers(0, 2, n)
    t = rng.exponential(np.where(grp == 1, 8, 5)).clip(0.1, 25)
    ev = (t < rng.exponential(12, n)).astype(int)
    df = pd.DataFrame({"time": t, "event": ev, "arm": grp.astype(float)})
    df.loc[rng.choice(n, 45, replace=False), "arm"] = np.nan  # missing GROUP only
    store.save("rmst_mi", df)
    client = TestClient(app)
    base = {"session_id": "rmst_mi", "duration_col": "time", "event_col": "event",
            "tau": 10.0, "group_col": "arm"}
    j = client.post("/api/survival_advanced/rmst", json={**base, "imputation": "mice"}).json()
    assert j["rmst_mi_note"] and "pooled" in j["rmst_mi_note"].lower()
    # Two groups pooled with a fraction of missing information from the imputed arm.
    assert len(j["rmst_by_group"]) == 2
    assert all("fmi" in v and v["fmi"] > 0 for v in j["rmst_by_group"].values())
    assert j["contrasts"] and j["contrasts"][0]["fmi"] > 0
    contrast = j["contrasts"][0]
    assert contrast["p_raw"] is not None
    assert contrast["p"] == round(contrast["p_raw"], 6)
