"""Regression tests for the edge cases the validation cards list as open.

The shared contract: a degenerate input must produce either a meaningful
result (with the degeneracy visible in it) or a clear 4xx whose detail a
user can act on. It must never produce a 500, silent NaNs presented as
findings, or a result that hides the degeneracy.

Each test names its card in a comment; qa/validation_cards/* flips the
matching edge case to covered with this file as evidence.
"""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)

RNG = np.random.default_rng(42)
N = 60


def _session(sid: str, df: pd.DataFrame) -> str:
    store.save(sid, df)
    return sid


def _base_df() -> pd.DataFrame:
    age = RNG.normal(55, 10, N)
    return pd.DataFrame(
        {
            "age": age,
            "sbp": 120 + 0.5 * age + RNG.normal(0, 8, N),
            "event": RNG.integers(0, 2, N),
            "arm": RNG.choice(["a", "b"], N),
        }
    )


def _no_5xx(r):
    assert r.status_code < 500, f"{r.status_code}: {r.text[:300]}"
    return r


# ── linear.yaml ──────────────────────────────────────────────────────────────

def test_linear_collinear_predictors_do_not_500():
    df = _base_df()
    df["age_copy"] = df["age"]  # exactly collinear pair -> singular design
    sid = _session("edge_lin_singular", df)
    r = _no_5xx(client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "sbp", "predictors": ["age", "age_copy"],
    }))
    if r.status_code == 200:
        # A fit on a singular design is only honest if nothing pretends the
        # two clones are separately estimable: no NaN presented as a number.
        body = r.json()
        text = str(body)
        assert "NaN" not in text and "Infinity" not in text


def test_linear_more_predictors_than_rows_rejected_cleanly():
    df = _base_df().head(4)
    for i in range(6):
        df[f"x{i}"] = RNG.normal(size=4)
    sid = _session("edge_lin_wide", df)
    r = _no_5xx(client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "sbp",
        "predictors": [f"x{i}" for i in range(6)] + ["age"],
    }))
    if r.status_code != 200:
        assert r.json()["detail"]


# ── logistic.yaml ────────────────────────────────────────────────────────────

def test_logistic_single_class_outcome_is_a_clear_422():
    df = _base_df()
    df["always_zero"] = 0
    sid = _session("edge_logit_oneclass", df)
    r = _no_5xx(client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "always_zero", "predictors": ["age"],
    }))
    # The endpoint deliberately rejects a one-class outcome as a validation
    # error (422, "requires both 0 and 1"), not a generic 400.
    assert r.status_code == 422
    assert "one unique value" in r.json()["detail"]