"""Smoke + contract tests for the linear / GLM endpoints in routers/models.py.

These endpoints (linear, delta_sensitivity, polynomial, lmm, gamma, negbinom,
linear_diag, melt) had no dedicated coverage; the services-layer extraction
nearly shipped a missing-`req`-annotation bug because only gee/ordinal/stepwise
were tested. This file locks the rest.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture(scope="module")
def sid():
    rng = np.random.default_rng(11)
    n = 200
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    dm = rng.integers(0, 2, n)
    y_cont = 2.0 + 0.05 * age + 0.02 * ldl + 1.5 * dm + rng.normal(0, 2, n)
    y_pos = np.exp(0.01 * age + 0.005 * ldl + rng.normal(0, 0.2, n)) + 0.1  # strictly > 0 (gamma)
    y_count = rng.poisson(np.exp(0.2 + 0.02 * age + 0.3 * dm)).astype(int)   # counts (negbinom)
    grp = np.repeat(np.arange(n // 5), 5)[:n]                                # 40 clusters (lmm)
    # Give the outcome a genuine per-group random intercept so MixedLM has a
    # non-degenerate variance component to estimate (otherwise it returns NaN).
    grp_effect = rng.normal(0, 3, n // 5)[grp]
    y_cont = y_cont + grp_effect
    # A predictor with missing values for the delta-sensitivity (MNAR) path
    ldl_miss = ldl.copy()
    miss_idx = rng.choice(n, size=30, replace=False)
    ldl_miss[miss_idx] = np.nan
    # Wide repeated-measures columns for /melt
    v1 = rng.normal(50, 8, n)
    v2 = v1 + rng.normal(2, 3, n)
    v3 = v1 + rng.normal(4, 3, n)
    df = pd.DataFrame({
        "AGE": age, "LDL": ldl, "DM": dm,
        "y_cont": y_cont, "y_pos": y_pos, "y_count": y_count,
        "grp": grp, "LDL_miss": ldl_miss,
        "PID": np.arange(n), "v1": v1, "v2": v2, "v3": v3,
    })
    return make_session(df, "tlin_main")


def test_linear(client, sid):
    r = client.post("/api/models/linear",
                    json={"session_id": sid, "outcome": "y_cont", "predictors": ["AGE", "LDL", "DM"]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert any(c.get("variable") == "AGE" for c in b["coefficients"])
    assert 0 <= b["r_squared"] <= 1


def test_linear_robust_se(client, sid):
    r = client.post("/api/models/linear",
                    json={"session_id": sid, "outcome": "y_cont", "predictors": ["AGE", "LDL"],
                          "robust_se": True})
    assert r.status_code == 200, r.text


def test_polynomial(client, sid):
    r = client.post("/api/models/polynomial",
                    json={"session_id": sid, "outcome": "y_cont", "predictor": "AGE", "degree": 3})
    assert r.status_code == 200, r.text
    assert "coefficients" in r.json()


def test_lmm(client, sid):
    r = client.post("/api/models/lmm",
                    json={"session_id": sid, "outcome": "y_cont",
                          "fixed_effects": ["AGE", "LDL"], "group_col": "grp"})
    assert r.status_code == 200, r.text


def test_gamma(client, sid):
    r = client.post("/api/models/gamma",
                    json={"session_id": sid, "outcome": "y_pos", "predictors": ["AGE", "LDL"]})
    assert r.status_code == 200, r.text
    assert "coefficients" in r.json()


def test_negbinom(client, sid):
    r = client.post("/api/models/negbinom",
                    json={"session_id": sid, "outcome": "y_count", "predictors": ["AGE", "DM"]})
    assert r.status_code == 200, r.text
    assert "coefficients" in r.json()


def test_linear_diag(client, sid):
    r = client.post("/api/models/linear_diag",
                    json={"session_id": sid, "outcome": "y_cont", "predictors": ["AGE", "LDL"]})
    assert r.status_code == 200, r.text


# ── Rank-deficient design matrix ─────────────────────────────────────────────
# Before `require_full_rank_design`, a design with more parameters than
# complete cases (or with collinear predictors) still reached .fit() and came
# back with NaN/Inf coefficients, which then failed JSON serialization and
# surfaced as main.py's generic "small subgroup or degenerate stratum"
# handler — a diagnosis that has nothing to do with the actual cause.

def test_more_predictors_than_observations_gives_the_real_reason(client):
    rng = np.random.default_rng(20260922)
    n, p = 100, 120
    df = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    df["y"] = rng.normal(size=n)
    sid = make_session(df, "tlin_p_gt_n")
    r = client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "y", "predictors": [f"x{i}" for i in range(p)],
    })
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "more predictors than observations" in detail
    assert "121" in detail and "100" in detail
    # The old, wrong diagnosis must not resurface under a different cause.
    assert "subgroup" not in detail and "stratum" not in detail


def test_a_predictor_that_is_a_linear_combination_of_others_gives_the_real_reason(client, sid):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"AGE": np.linspace(20, 90, 50)})
    df["AGE2"] = df["AGE"] * 2.0  # exact linear combination of AGE — n > p, still rank-deficient
    df["y"] = 1.0 + 0.1 * df["AGE"] + rng.normal(0, 1, 50)
    ldsid = make_session(df, "tlin_collinear")
    r = client.post("/api/models/linear", json={
        "session_id": ldsid, "outcome": "y", "predictors": ["AGE", "AGE2"],
    })
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "linear combination" in detail
    assert "subgroup" not in detail and "stratum" not in detail


def test_a_well_specified_model_with_many_predictors_still_fits(client):
    # Regression guard: the rank check must not false-positive on a design
    # that is merely wide, as opposed to actually rank-deficient.
    rng = np.random.default_rng(6)
    n, p = 100, 40
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(0, 1, n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["y"] = y
    sid = make_session(df, "tlin_wide_full_rank")
    r = client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "y", "predictors": [f"x{i}" for i in range(p)],
    })
    assert r.status_code == 200, r.text
    assert len(r.json()["coefficients"]) == p + 1  # + the intercept


def test_melt(client):
    # Separate session — /melt overwrites the stored frame with the long format.
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({
        "PID": np.arange(n),
        "t0": rng.normal(50, 5, n),
        "t1": rng.normal(52, 5, n),
        "t2": rng.normal(54, 5, n),
    })
    msid = make_session(df, "tlin_melt")
    r = client.post("/api/models/melt",
                    json={"session_id": msid, "id_col": "PID",
                          "value_cols": ["t0", "t1", "t2"], "time_var_name": "visit",
                          "value_var_name": "ef"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["rows"] == n * 3
    assert b["time_var"] == "visit"
