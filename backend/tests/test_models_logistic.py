"""Tests for routers/models_logistic.py.

Covers the four POST endpoints mounted under /api/models:
  * /logistic        — standard logistic regression
  * /firth_logistic  — Firth penalized logistic
  * /poisson         — Poisson count regression
  * /logistic_table  — univariate + multivariate OR table

Binary outcome = `event`; predictors AGE, LDL, DM. Poisson uses a
Poisson-distributed `count` column.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session

SEED = 20240531


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(SEED)
    n = 300
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    dm = rng.integers(0, 2, n)
    logit_p = -5 + 0.05 * age + 0.01 * ldl + 0.6 * dm
    p = 1 / (1 + np.exp(-logit_p))
    event = (rng.uniform(0, 1, n) < p).astype(int)
    # Poisson count outcome, mean depends mildly on predictors so there is signal.
    lam = np.exp(0.2 + 0.01 * (age - 60) + 0.3 * dm)
    count = rng.poisson(lam).astype(int)
    # A constant outcome column for the single-value validation path.
    all_ones = np.ones(n, dtype=int)
    # A non-binary numeric outcome (2/3 values) for the binary-check path.
    bad_outcome = rng.integers(2, 4, n)
    return pd.DataFrame({
        "AGE": age,
        "LDL": ldl,
        "DM": dm,
        "event": event,
        "count": count,
        "all_ones": all_ones,
        "bad_outcome": bad_outcome,
    })


@pytest.fixture(scope="module")
def sid(synth):
    return make_session(synth, "tlog_main")


# ── /logistic ────────────────────────────────────────────────────────────────

def test_logistic_happy_path(client, sid):
    r = client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE", "LDL", "DM"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] == "event"
    assert body["n"] > 0
    assert isinstance(body["coefficients"], list)
    # const + 3 predictors
    assert len(body["coefficients"]) == 4
    var_names = {c["variable"] for c in body["coefficients"]}
    assert "const" in var_names
    for c in body["coefficients"]:
        assert c["odds_ratio"] > 0
        assert 0.0 <= c["p"] <= 1.0
    # Model-level stats present and in sane ranges.
    assert 0.0 <= body["nagelkerke_r2"] <= 1.0
    if body["auc"] is not None:
        assert 0.0 <= body["auc"] <= 1.0
    assert body["omnibus"]["df"] == 3
    assert isinstance(body["result_text"], str) and body["result_text"]


def test_logistic_robust_se(client, sid):
    r = client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE", "LDL"], "robust_se": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "[Robust SE]" in body["model"]
    assert len(body["coefficients"]) == 3


def test_logistic_scaling(client, sid):
    r = client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE", "LDL", "DM"],
        "scale_factors": {"AGE": 10},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    var_names = {c["variable"] for c in body["coefficients"]}
    # Scaled predictor is renamed.
    assert any("per 10 units" in v for v in var_names)


def test_logistic_non_binary_outcome_422(client, sid):
    r = client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "bad_outcome",
        "predictors": ["AGE", "LDL"],
    })
    assert r.status_code == 422, r.text
    assert "binary" in r.json()["detail"].lower()


def test_logistic_single_value_outcome_422(client, sid):
    r = client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "all_ones",
        "predictors": ["AGE", "LDL"],
    })
    assert r.status_code == 422, r.text
    assert "one unique value" in r.json()["detail"].lower()


# ── /firth_logistic ──────────────────────────────────────────────────────────

def test_firth_logistic_happy_path(client, sid):
    r = client.post("/api/models/firth_logistic", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE", "LDL", "DM"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "Firth Penalized Logistic Regression"
    assert body["n"] > 0
    assert body["n_events"] >= 0
    assert isinstance(body["converged"], bool)
    assert body["iterations"] >= 1
    assert len(body["coefficients"]) == 4
    for c in body["coefficients"]:
        assert c["odds_ratio"] > 0
        assert 0.0 <= c["p"] <= 1.0
    assert 0.0 <= body["nagelkerke_r2"] <= 1.0
    if body["auc"] is not None:
        assert 0.0 <= body["auc"] <= 1.0
    _assert_firth_wald_disclosed(body)


def _assert_firth_wald_disclosed(body):
    """Firth intervals are Wald while logistf defaults to profile likelihood.
    The method has to travel with the numbers: in the field the panel renders
    beside the table, and in the paragraph that gets pasted into a paper."""
    assert body["ci_method"] == "Wald"
    assert "profile" in body["method_note"]
    assert "Wald-based, not penalised profile likelihood" in body["result_text"]


def test_firth_logistic_non_binary_422(client, sid):
    r = client.post("/api/models/firth_logistic", json={
        "session_id": sid, "outcome": "bad_outcome",
        "predictors": ["AGE"],
    })
    assert r.status_code == 422, r.text
    assert "binary" in r.json()["detail"].lower()


# ── /poisson ─────────────────────────────────────────────────────────────────

def test_poisson_happy_path(client, sid):
    r = client.post("/api/models/poisson", json={
        "session_id": sid, "outcome": "count",
        "predictors": ["AGE", "DM"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] == "count"
    assert body["n"] > 0
    assert len(body["coefficients"]) == 3  # const + 2
    for c in body["coefficients"]:
        assert c["irr"] > 0
        assert 0.0 <= c["p"] <= 1.0
        assert c["irr_ci_low"] <= c["irr_ci_high"]
    assert isinstance(body["aic"], float)
    assert isinstance(body["result_text"], str) and body["result_text"]


def test_poisson_robust_se(client, sid):
    r = client.post("/api/models/poisson", json={
        "session_id": sid, "outcome": "count",
        "predictors": ["AGE"], "robust_se": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "[Robust SE]" in body["model"]


def test_poisson_negative_counts_422(client, sid, synth):
    neg = synth.copy()
    neg["neg_count"] = synth["count"] - 5  # introduces negatives
    neg_sid = make_session(neg, "tlog_neg")
    r = client.post("/api/models/poisson", json={
        "session_id": neg_sid, "outcome": "neg_count",
        "predictors": ["AGE"],
    })
    assert r.status_code == 422, r.text
    assert "non-negative" in r.json()["detail"].lower()


def test_poisson_fractional_counts_422(client, sid, synth):
    frac = synth.copy()
    frac["frac_count"] = synth["count"].astype(float) + 0.5
    frac_sid = make_session(frac, "tlog_frac")
    r = client.post("/api/models/poisson", json={
        "session_id": frac_sid, "outcome": "frac_count",
        "predictors": ["AGE"],
    })
    assert r.status_code == 422, r.text
    assert "integer" in r.json()["detail"].lower()


# ── /logistic_table ──────────────────────────────────────────────────────────

def test_logistic_table_all(client, sid):
    r = client.post("/api/models/logistic_table", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE", "LDL", "DM"], "selection": "all",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "Logistic OR Table"
    assert body["use_firth"] is False
    assert "ci_method" not in body
    assert "profile likelihood" not in body["result_text"]
    assert body["n_total"] == 300
    assert isinstance(body["table"], list) and len(body["table"]) >= 3
    for row in body["table"]:
        assert "variable" in row
        if row["uni_or"] is not None:
            assert row["uni_or"] > 0
    assert body["model_stats"] is not None
    assert 0.0 <= body["model_stats"]["nagelkerke_r2"] <= 1.0


def test_logistic_table_p05_selection(client, sid):
    r = client.post("/api/models/logistic_table", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE", "LDL", "DM"], "selection": "p05",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["selection_method"] == "Univariate p < 0.05"
    assert body["n_multi"] <= body["n_total"]


def test_logistic_table_firth(client, sid):
    r = client.post("/api/models/logistic_table", json={
        "session_id": sid, "outcome": "event",
        "predictors": ["AGE", "LDL", "DM"],
        "selection": "all", "use_firth": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["use_firth"] is True
    assert "Firth" in body["model"]
    assert isinstance(body["table"], list) and len(body["table"]) >= 3
    _assert_firth_wald_disclosed(body)


def test_logistic_table_non_binary_422(client, sid):
    r = client.post("/api/models/logistic_table", json={
        "session_id": sid, "outcome": "bad_outcome",
        "predictors": ["AGE", "LDL"], "selection": "all",
    })
    assert r.status_code == 422, r.text
    assert "binary" in r.json()["detail"].lower()
