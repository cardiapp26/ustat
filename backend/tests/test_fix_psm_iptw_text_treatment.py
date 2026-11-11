"""PSM and IPTW accept a two-level text treatment once the treated level is named."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from fastapi import HTTPException

from routers.models.psm_iptw import IPTWRequest, PSMRequest, _run_iptw, _run_psm

N = 300
COVS = ["age", "bmi"]


@pytest.fixture(scope="module")
def sid() -> str:
    rng = np.random.default_rng(21)
    age = rng.normal(60, 10, N)
    bmi = rng.normal(27, 4, N)
    treat = rng.binomial(1, 1 / (1 + np.exp(-(0.04 * (age - 60) + 0.05 * (bmi - 27)))))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 - 0.7 * treat + 0.03 * (age - 60)))))
    frame = pd.DataFrame({
        "age": age, "bmi": bmi, "treat_num": treat,
        "arm": np.where(treat == 1, "treat", "control"), "y": y,
    })
    return make_session(frame, "psm_iptw_text_treatment")


def test_psm_text_treatment_matches_the_numeric_coding(sid):
    text = _run_psm(PSMRequest(session_id=sid, treatment_col="arm", treated_level="treat", covariates=COVS, outcome_col="y"))
    num = _run_psm(PSMRequest(session_id=sid, treatment_col="treat_num", covariates=COVS, outcome_col="y"))
    assert text["treatment_coding"] == {"column": "arm", "treated": "treat", "control": "control"}
    assert num["treatment_coding"] is None
    assert text["n_treated"] == num["n_treated"]
    assert text["n_matched_pairs"] == num["n_matched_pairs"]


def test_iptw_text_treatment_matches_the_numeric_coding(sid):
    text = _run_iptw(IPTWRequest(session_id=sid, treatment_col="arm", treated_level="treat", covariates=COVS, outcome_col="y"))
    num = _run_iptw(IPTWRequest(session_id=sid, treatment_col="treat_num", covariates=COVS, outcome_col="y"))
    assert text["treatment_coding"]["treated"] == "treat"
    assert text["n_treated"] == num["n_treated"]


def test_naming_the_other_level_flips_the_groups(sid):
    flipped = _run_iptw(IPTWRequest(session_id=sid, treatment_col="arm", treated_level="control", covariates=COVS))
    straight = _run_iptw(IPTWRequest(session_id=sid, treatment_col="arm", treated_level="treat", covariates=COVS))
    assert flipped["n_treated"] == straight["n_control"]


@pytest.mark.parametrize("level, fragment", [(None, "treated_level"), ("placebo", "not a level")])
def test_text_treatment_without_a_valid_level_is_rejected(sid, level, fragment):
    with pytest.raises(HTTPException) as err:
        _run_iptw(IPTWRequest(session_id=sid, treatment_col="arm", treated_level=level, covariates=COVS))
    assert err.value.status_code == 422
    assert fragment in err.value.detail
