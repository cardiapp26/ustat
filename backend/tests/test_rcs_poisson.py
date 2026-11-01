"""Poisson and quasi-Poisson RCS against R glm() on the Harrell basis.

Reference values: glm(visits ~ rcspline.eval(age, nk = 4, inclx = TRUE),
family = poisson / quasipoisson) on qa/models_audit/dataset.csv (n = 300).
Wald tests use vcov(fit) (scaled by the Pearson dispersion for
quasipoisson, referred to F with df.residual); the interaction tests are
anova(reduced, full, test = "Chisq" / "F").
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_session

from services.rcs_basis import harrell_knots, rcs_basis
from services.rcs_glm import fit_glm, nested_test

_AUDIT_CSV = Path(__file__).resolve().parents[2] / "qa" / "models_audit" / "dataset.csv"


@pytest.fixture(scope="module")
def audit_df():
    return pd.read_csv(_AUDIT_CSV)


def _post(client, df, name, **payload):
    sid = make_session(df, name)
    body = {"session_id": sid, "predictor": "age", "outcome": "visits", "n_knots": 4, **payload}
    return client.post("/api/models/rcs", json=body)


def _ok(client, df, name, **payload):
    r = _post(client, df, name, **payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_poisson_tests_aic_and_dispersion_match_r(client, audit_df):
    d = _ok(client, audit_df, "rcs_pois", model_type="poisson")
    assert d["effect_type"] == "IRR"
    assert d["test_statistic"] == "chi2"
    assert d["overall_wald"] == pytest.approx(43.93105287, rel=1e-4)
    assert d["overall_p"] == pytest.approx(1.560944275e-09, rel=1e-3)
    assert d["nonlinearity_wald"] == pytest.approx(5.986664854, rel=1e-4)
    assert d["nonlinearity_p"] == pytest.approx(0.05012013643, rel=1e-3)
    assert d["aic"] == pytest.approx(1526.289826, abs=1e-2)
    assert d["dispersion"] == pytest.approx(2.183857394, rel=1e-6)
    assert d["total_count"] == int(audit_df["visits"].sum())
    aic = {row["n_knots"]: row["aic"] for row in d["knot_comparison"]}
    assert aic[3] == pytest.approx(1527.747065, abs=1e-2)
    assert aic[4] == pytest.approx(1526.289826, abs=1e-2)
    assert aic[5] == pytest.approx(1526.691717, abs=1e-2)


def test_quasipoisson_uses_f_tests_and_wider_band(client, audit_df):
    d = _ok(client, audit_df, "rcs_qpois", model_type="quasipoisson")
    assert d["effect_type"] == "IRR"
    assert d["test_statistic"] == "F"
    assert d["overall_f"] == pytest.approx(6.705419346, rel=1e-4)
    assert d["overall_p"] == pytest.approx(2.159852622e-04, rel=1e-3)
    assert d["nonlinearity_f"] == pytest.approx(1.370662517, rel=1e-4)
    assert d["nonlinearity_p"] == pytest.approx(0.2555455975, rel=1e-3)
    assert d["dispersion"] == pytest.approx(2.183857394, rel=1e-6)
    # No likelihood: R reports AIC as NA, and there is no knot AIC table.
    assert d["aic"] is None and d["log_likelihood"] is None
    assert d["knot_comparison"] == []


# R: exp(contrast) at the grid point nearest 70 (69.98366834...) vs age 62.2.
@pytest.mark.parametrize("model_type,low,high", [
    ("poisson", 0.9506253633, 1.145871671),
    ("quasipoisson", 0.9091357441, 1.198165049),
])
def test_irr_curve_matches_r_contrast(client, audit_df, model_type, low, high):
    d = _ok(client, audit_df, f"rcs_irr_{model_type}", model_type=model_type, ref_value=62.2)
    i = int(np.argmin(np.abs(np.asarray(d["x_values"]) - 70.0)))
    assert d["x_values"][i] == pytest.approx(69.98366834170854, abs=1e-4)
    assert d["or_values"][i] == pytest.approx(1.043692806, rel=1e-5)
    assert d["ci_low"][i] == pytest.approx(low, rel=1e-5)
    assert d["ci_high"][i] == pytest.approx(high, rel=1e-5)


def test_exposure_offset_matches_r(client, audit_df):
    d = _ok(client, audit_df, "rcs_pois_off", model_type="poisson", exposure_col="followup_years")
    assert d["exposure_col"] == "followup_years"
    assert d["overall_wald"] == pytest.approx(57.67619463, rel=1e-4)
    assert d["nonlinearity_wald"] == pytest.approx(9.09977463, rel=1e-4)
    assert d["aic"] == pytest.approx(1789.03538, abs=1e-2)
    aic = {row["n_knots"]: row["aic"] for row in d["knot_comparison"]}
    assert aic[3] == pytest.approx(1796.072359, abs=1e-2)
    assert aic[5] == pytest.approx(1789.362924, abs=1e-2)


@pytest.mark.parametrize("model_type,stat_key,stat,p,test", [
    ("poisson", "lr_stat", 3.621011038, 0.3054031986, "chi2"),
    ("quasipoisson", "f", 0.5490258835, 0.6491722251, "F"),
])
def test_interaction_test_matches_r_anova(client, audit_df, model_type, stat_key, stat, p, test):
    d = _ok(
        client, audit_df, f"rcs_ix_{model_type}", model_type=model_type,
        covariates=["sex"], interaction_covariates=["sex"],
    )
    ix = d["interaction"]
    assert ix["test"] == test and ix["df"] == 3
    assert ix[stat_key] == pytest.approx(stat, rel=1e-4)
    assert ix["p"] == pytest.approx(p, rel=1e-3)


def test_count_outcome_validation(client, audit_df):
    r = _post(client, audit_df, "rcs_pois_bad_y", model_type="poisson", outcome="bmi")
    assert r.status_code == 422
    assert "count outcome" in r.json()["detail"]
    neg = audit_df.assign(visits=audit_df["visits"] - 1)
    assert _post(client, neg, "rcs_pois_neg", model_type="quasipoisson").status_code == 422


def test_exposure_validation(client, audit_df):
    zero = audit_df.assign(followup_years=audit_df["followup_years"].where(audit_df.index != 0, 0.0))
    r = _post(client, zero, "rcs_pois_zero_exp", model_type="poisson", exposure_col="followup_years")
    assert r.status_code == 422 and "> 0" in r.json()["detail"]
    r = _post(client, audit_df, "rcs_log_exp", model_type="logistic", outcome="event_binary", exposure_col="followup_years")
    assert r.status_code == 422 and "Poisson" in r.json()["detail"]
    r = _post(client, audit_df, "rcs_exp_eq_pred", model_type="poisson", exposure_col="age")
    assert r.status_code == 422


def test_unknown_model_type_lists_choices(client, audit_df):
    r = _post(client, audit_df, "rcs_bad_type", model_type="negbin")
    assert r.status_code == 422
    assert "quasipoisson" in r.json()["detail"]


def test_nested_test_requires_likelihood_or_dispersion():
    rng = np.random.default_rng(0)
    x = rng.uniform(20, 80, 200)
    y = rng.poisson(np.exp(0.02 * x)).astype(float)
    X = np.column_stack([np.ones_like(x), x, rcs_basis(x, harrell_knots(x, 4))])
    full = fit_glm("quasipoisson", y, X)
    reduced = fit_glm("quasipoisson", y, X[:, :2])
    res = nested_test("quasipoisson", full, reduced)
    assert res["test"] == "F" and res["df"] == 2
    with pytest.raises(ValueError):
        nested_test("poisson", full, reduced)
