"""A two-level text exposure (control/treat) must be usable in the causal endpoints.

It used to be coerced to NaN, every row was dropped, and the request failed
with "Not enough complete observations".
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from fastapi import HTTPException

from routers.causal import IV2SLSRequest, MediationRequest, iv_2sls, mediation

N = 300


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    z = rng.normal(0, 1, N)
    treat = (0.9 * z + rng.normal(0, 1, N) > 0).astype(int)
    med = 50 + 15 * treat + rng.normal(0, 8, N)
    y = 60 + 4 * treat + 0.3 * med + rng.normal(0, 6, N)
    return pd.DataFrame({
        "z": z, "treat_num": treat, "arm": np.where(treat == 1, "treat", "control"),
        "med": med, "y": y, "grp3": rng.choice(["a", "b", "c"], N),
    })


def test_iv_accepts_a_two_level_text_exposure(frame):
    sid = make_session(frame, "causal_iv_text")
    text = iv_2sls(IV2SLSRequest(session_id=sid, outcome="y", endogenous="arm", instruments=["z"]))
    num = iv_2sls(IV2SLSRequest(session_id=sid, outcome="y", endogenous="treat_num", instruments=["z"]))
    assert text["n"] == num["n"] == N
    assert text["exposure_coding"] == {"column": "arm", "coded_0": "control", "coded_1": "treat"}
    assert num["exposure_coding"] is None


def test_mediation_accepts_a_two_level_text_exposure(frame):
    sid = make_session(frame, "causal_med_text")
    res = mediation(MediationRequest(session_id=sid, outcome="y", treatment="arm", mediator="med", bootstrap=50))
    assert res["exposure_coding"]["coded_1"] == "treat"
    assert res["n"] == N


def test_three_level_text_exposure_is_rejected_with_a_clear_message(frame):
    sid = make_session(frame, "causal_iv_3lvl")
    with pytest.raises(HTTPException) as err:
        iv_2sls(IV2SLSRequest(session_id=sid, outcome="y", endogenous="grp3", instruments=["z"]))
    assert err.value.status_code == 422
    assert "exactly two levels" in err.value.detail


def test_result_text_states_how_the_text_exposure_was_coded(frame):
    sid = make_session(frame, "causal_iv_text_sentence")
    text = iv_2sls(IV2SLSRequest(session_id=sid, outcome="y", endogenous="arm", instruments=["z"]))
    assert text["result_text"].endswith("arm was coded 1 = 'treat', 0 = 'control'.")
    num = iv_2sls(IV2SLSRequest(session_id=sid, outcome="y", endogenous="treat_num", instruments=["z"]))
    assert "was coded" not in num["result_text"]
