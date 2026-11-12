"""Regression tests for fixes in the regression-model services.

One test per defect: E-value rules, stepwise model type and criterion,
predictor-to-column lookup, binary outcome coding, nominal SMD, propensity
model, and the LMM identifier heuristic.
"""
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from fastapi import HTTPException

from services.column_identifiers import is_id_like
from services.causal_sensitivity import e_value
from services.psm import (
    _compute_smd,
    _fit_propensity_scores,
    _ks_p,
    _variance_ratio,
    smd_before_after,
    smd_by_level,
)
from services.regression import (
    _uni_p_for_pred,
    add_pairwise_interactions,
    dummy_column_map,
    encode_binary_outcome,
    stepwise_backward,
    stepwise_forward,
)


# ── E-value ──────────────────────────────────────────────────────────────────

def _ev(rr: float) -> float:
    return rr + math.sqrt(rr * (rr - 1.0))


@pytest.mark.parametrize("est,lo,hi,expected", [
    (3.0, 2.0, 4.5, _ev(2.0)),            # harmful: lower limit
    (0.5, 0.3, 0.8, _ev(1 / 0.8)),        # protective: upper limit, inverted
    (1.5, 0.9, 2.5, 1.0),                 # CI includes the null
])
def test_e_value_ci_uses_limit_closest_to_null(est, lo, hi, expected):
    res = e_value(estimate=est, ci_low=lo, ci_high=hi, measure="rr")
    assert res["e_value_ci"] == pytest.approx(expected, abs=1e-3)


def test_e_value_or_rare_outcome_is_treated_as_rr():
    res = e_value(estimate=3.0, measure="or", rare_outcome=True)
    assert res["e_value_point_estimate"] == pytest.approx(_ev(3.0), abs=1e-3)
    assert res["rr_conversion"] == "rare_outcome"


def test_e_value_or_common_outcome_uses_square_root():
    res = e_value(estimate=3.0, measure="or", rare_outcome=False)
    assert res["e_value_point_estimate"] == pytest.approx(_ev(math.sqrt(3.0)), abs=1e-3)
    assert res["rr_conversion"] == "sqrt_or"


def test_e_value_or_with_baseline_risk_uses_it():
    res = e_value(estimate=3.0, measure="or", baseline_risk=0.3)
    rr = 3.0 / (0.7 + 0.3 * 3.0)
    assert res["e_value_point_estimate"] == pytest.approx(_ev(rr), abs=1e-3)
    assert res["baseline_risk_used"] == 0.3


def test_e_value_hr_rare_and_common():
    rare = e_value(estimate=2.0, measure="hr", rare_outcome=True)
    assert rare["e_value_point_estimate"] == pytest.approx(_ev(2.0), abs=1e-3)
    common = e_value(estimate=2.0, measure="hr", rare_outcome=False)
    rr = (1 - 0.5 ** math.sqrt(2.0)) / (1 - 0.5 ** math.sqrt(0.5))
    assert common["e_value_point_estimate"] == pytest.approx(_ev(rr), abs=1e-3)


# ── Stepwise ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def linear_frame():
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "noise": rng.normal(0, 1, n),
        "sex": rng.choice(["F", "M"], n),
        "stage": rng.choice(["I", "II", "III"], n),
    })
    y = 100 + 0.8 * df["age"] + 6.0 * (df["stage"] == "III") + rng.normal(0, 5, n)
    return df, y


def test_stepwise_forward_linear_selects_real_predictors(linear_frame):
    df, y = linear_frame
    selected = stepwise_forward(y, df, list(df.columns), model_type="linear")
    assert "age" in selected and "stage" in selected
    assert "noise" not in selected


def test_stepwise_backward_linear_drops_noise(linear_frame):
    df, y = linear_frame
    selected = stepwise_backward(y, df, list(df.columns), model_type="linear")
    assert set(selected) == {"age", "stage"}


def test_stepwise_aic_criterion_lowers_aic(linear_frame):
    df, y = linear_frame
    selected = stepwise_forward(y, df, list(df.columns), model_type="linear", criterion="aic")
    assert "age" in selected and "stage" in selected

    def aic(terms):
        X = sm.add_constant(pd.get_dummies(df[terms], drop_first=True).astype(float))
        return sm.OLS(y, X).fit().aic

    assert aic(selected) <= aic(list(df.columns)) + 1e-9


def test_stepwise_forward_keeps_base_terms(linear_frame):
    df, y = linear_frame
    selected = stepwise_forward(y, df, ["age"], model_type="linear", base=["noise"])
    assert selected[0] == "noise" and "age" in selected


def test_stepwise_rejects_unknown_arguments(linear_frame):
    df, y = linear_frame
    with pytest.raises(ValueError):
        stepwise_forward(y, df, ["age"], model_type="cox")
    with pytest.raises(ValueError):
        stepwise_backward(y, df, ["age"], criterion="bic")


# ── Predictor-to-column lookup ───────────────────────────────────────────────

@pytest.fixture()
def prefix_frame():
    rng = np.random.default_rng(3)
    n = 60
    return pd.DataFrame({
        "sex": rng.choice(["F", "M"], n),
        "sex_partner": rng.normal(0, 1, n),
        "age": rng.normal(50, 8, n),
    })


def test_interaction_does_not_pick_up_prefix_sibling(prefix_frame):
    preds = ["sex", "sex_partner", "age"]
    enc = pd.get_dummies(prefix_frame[preds], drop_first=True).astype(float)
    _, added = add_pairwise_interactions(
        enc, [["sex", "age"]], preds, dummy_column_map(prefix_frame, preds))
    assert added == ["sex_M:age"]


def test_univariable_p_lookup_uses_the_predictors_own_columns(prefix_frame):
    column_map = dummy_column_map(prefix_frame, ["sex", "sex_partner"])
    uni = {"sex_M": {"p": 0.6}, "sex_partner": {"p": 0.001}}
    assert _uni_p_for_pred("sex", uni, column_map) == 0.6
    assert _uni_p_for_pred("sex_partner", uni, column_map) == 0.001


# ── Binary outcome coding ────────────────────────────────────────────────────

def test_text_outcome_reports_the_modelled_level():
    y = pd.Series(["Case", "Control", "Case", None], index=[5, 9, 11, 20])
    coded, info = encode_binary_outcome(y)
    assert info == {"event_level": "Control", "reference_level": "Case", "coded_from_text": True}
    assert list(coded.index) == [5, 9, 11, 20]
    assert coded.iloc[:3].tolist() == [0.0, 1.0, 0.0] and np.isnan(coded.iloc[3])


@pytest.mark.parametrize("values,dtype", [
    (["a", "b", "c", "a"], object),
    (["a", "b", "c", "a"], "category"),
    (["a", "b", "c", "a"], "string"),
    ([0, 1, 2, 1], int),
])
def test_non_binary_outcome_is_a_422(values, dtype):
    with pytest.raises(HTTPException) as exc:
        encode_binary_outcome(pd.Series(values, dtype=dtype))
    assert exc.value.status_code == 422


@pytest.mark.parametrize("dtype", ["category", "string"])
def test_two_level_category_and_string_outcomes_are_coded(dtype):
    coded, info = encode_binary_outcome(pd.Series(["no", "yes", "yes", "no"], dtype=dtype))
    assert coded.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert info["event_level"] == "yes"


# ── PSM balance and propensity model ─────────────────────────────────────────

@pytest.fixture()
def race_groups():
    treated = pd.Series(["A"] * 30 + ["B"] * 50 + ["C"] * 20)
    control = pd.Series(["A"] * 34 + ["B"] * 34 + ["C"] * 32)
    return treated, control


def test_nominal_smd_is_the_largest_level_smd(race_groups):
    treated, control = race_groups
    levels = smd_by_level(treated, control)
    p1, p0 = 0.50, 0.34
    expected_b = abs(p1 - p0) / math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    assert levels["B"] == pytest.approx(expected_b)
    assert _compute_smd(treated, control) == pytest.approx(max(levels.values()))


def test_nominal_smd_does_not_depend_on_level_labels(race_groups):
    treated, control = race_groups
    relabel = {"A": "z", "B": "m", "C": "a"}
    assert _compute_smd(treated.map(relabel), control.map(relabel)) == pytest.approx(
        _compute_smd(treated, control))


def test_nominal_covariate_has_no_variance_ratio_or_ks(race_groups):
    treated, control = race_groups
    assert _variance_ratio(treated, control) is None
    assert _ks_p(treated, control) is None


def test_smd_before_after_reports_levels(race_groups):
    treated, control = race_groups
    before, after, per_level = smd_before_after(treated, control, treated, treated)
    assert after == 0.0 and before > 0.3
    assert set(per_level) == {"A", "B", "C"}
    assert per_level["B"]["after"] == 0.0
    _, _, none_for_numeric = smd_before_after(
        pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 3.0, 4.0]),
        pd.Series([1.0, 2.0]), pd.Series([2.0, 3.0]))
    assert none_for_numeric is None


def test_propensity_model_is_maximum_likelihood():
    rng = np.random.default_rng(11)
    n = 60
    X = rng.normal(size=(n, 3))
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.4 + 1.5 * X[:, 0] - 1.0 * X[:, 1])))).astype(int)
    ps = _fit_propensity_scores(X, y, "logistic", random_state=1)
    ml = sm.Logit(y, sm.add_constant(X)).fit(disp=False).predict(sm.add_constant(X))
    np.testing.assert_allclose(ps, ml, atol=2e-3)


# ── LMM identifier heuristic ─────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["steroid", "thyroid", "nodes", "num_vessels", "number_of_stents", "caseload"])
def test_ordinary_covariates_are_not_identifiers(name):
    rng = np.random.default_rng(5)
    assert not is_id_like(name, pd.Series(rng.integers(0, 4, 200)))


def test_continuous_covariate_is_not_an_identifier():
    rng = np.random.default_rng(5)
    assert not is_id_like("ldl", pd.Series(rng.normal(120, 30, 200)))


@pytest.mark.parametrize("name", ["patient_id", "subjectID", "record_no", "id"])
def test_repeated_subject_identifier_is_detected(name):
    assert is_id_like(name, pd.Series(np.repeat(np.arange(40), 5)))


def test_unique_integer_code_is_detected_whatever_its_name():
    assert is_id_like("x", pd.Series(np.arange(1000, 1200)))


def test_upload_and_session_identifiers_use_full_column_values(client):
    frame = pd.DataFrame({
        "pid": np.arange(3000),
        "visit_number": np.r_[np.zeros(2000), np.arange(1000)].astype(float) / 2,
        "score": np.r_[np.arange(2000), np.repeat(100, 1000)],
        "steroid": np.tile([0, 1], 1500),
        "age": np.linspace(20.1, 80.1, 3000),
    })
    uploaded = client.post(
        "/api/upload/", files={"file": ("ids.csv", frame.to_csv(index=False), "text/csv")},
    )
    assert uploaded.status_code == 200, uploaded.text
    session = uploaded.json()
    expected = {"pid": True, "visit_number": True, "score": False, "steroid": False, "age": False}
    for payload in (session, client.get(f"/api/sessions/{session['session_id']}").json()):
        assert {c["name"]: c["id_like"] for c in payload["columns"]} == expected
