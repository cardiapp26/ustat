"""The causal-sensitivity suite is exposed on three routes; they must not drift.

`routers/models/sensitivity.py` used to hold a line-for-line copy of
`survival_advanced_service.fit_causal_sensitivity`. It now delegates, and these
tests pin that: the two full-suite routes must return identical payloads, while
the `models` route keeps its stricter request validation.

`model_diagnostics/causal_sensitivity` is deliberately a smaller, data-free
calculator — it is checked here only for the subset it shares.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

SEED = 41
FULL_SUITE_KEYS = {
    "test", "e_value", "e_value_smd", "quantitative_bias_analysis",
    "multi_confounder_sensitivity", "manski_bounds", "rosenbaum_bounds",
    "negative_control_analysis", "warnings", "assumptions", "result_text",
    "export_rows", "r_code",
}


@pytest.fixture(scope="module")
def sid():
    rng = np.random.default_rng(SEED)
    n = 300
    trt = rng.integers(0, 2, n)
    age = rng.normal(60, 10, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 0.8 * trt + 0.02 * (age - 60)))))
    frame = pd.DataFrame({
        "trt": trt,
        "age": np.round(age, 2),
        "y": y,
        "pair": np.arange(n) // 2,
        "negctl": rng.binomial(1, 0.3, n),
    })
    return make_session(frame, "causal_sensitivity_session")


def _payloads(sid):
    return [
        {"observed_estimate": 2.5, "measure": "rr"},
        {"observed_estimate": 2.5, "ci_low": 1.4, "ci_high": 4.1, "measure": "or",
         "rare_outcome": True, "baseline_risk": 0.1},
        {"observed_estimate": 1.8, "measure": "hr", "smd": 0.35, "session_id": sid,
         "treatment_col": "trt", "outcome_col": "y", "match_id_col": "pair",
         "negative_control_outcome_col": "negctl"},
    ]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_both_full_suite_routes_return_identical_payloads(client, sid, idx):
    payload = _payloads(sid)[idx]
    a = client.post("/api/models/causal_sensitivity", json=payload)
    b = client.post("/api/survival_advanced/causal_sensitivity", json=payload)
    assert a.status_code == 200, a.text
    assert b.status_code == 200, b.text
    assert a.json() == b.json(), "the two routes have drifted apart again"


def test_full_suite_shape_is_complete(client, sid):
    r = client.post("/api/models/causal_sensitivity", json=_payloads(sid)[2])
    assert r.status_code == 200, r.text
    assert set(r.json()) == FULL_SUITE_KEYS


def test_models_route_keeps_the_stricter_validation(client):
    # observed_estimate is required here (the survival_advanced twin defaults it).
    r = client.post("/api/models/causal_sensitivity", json={"measure": "rr"})
    assert r.status_code == 422, r.text

    # measure is a Literal on this route.
    r = client.post("/api/models/causal_sensitivity",
                    json={"observed_estimate": 2.0, "measure": "nonsense"})
    assert r.status_code == 422, r.text

    # observed_estimate must be > 0.
    r = client.post("/api/models/causal_sensitivity", json={"observed_estimate": 0})
    assert r.status_code == 422, r.text


def test_inverted_confidence_interval_is_a_readable_400(client):
    r = client.post("/api/models/causal_sensitivity",
                    json={"observed_estimate": 2.0, "ci_low": 3.0, "ci_high": 2.0})
    assert r.status_code == 400, r.text
    assert "ci_low" in r.json()["detail"]


def test_diagnostics_route_is_a_documented_subset(client):
    """The third registration is a smaller calculator, not a fourth copy."""
    payload = {"observed_estimate": 2.5, "measure": "rr"}
    small = client.post("/api/model_diagnostics/causal_sensitivity", json=payload)
    full = client.post("/api/models/causal_sensitivity", json=payload)
    assert small.status_code == 200, small.text
    assert full.status_code == 200, full.text

    small_keys, full_keys = set(small.json()), set(full.json())
    assert small_keys < full_keys, "expected a strict subset of the full suite"
    # The shared core must still agree numerically.
    assert small.json()["e_value"] == full.json()["e_value"]
