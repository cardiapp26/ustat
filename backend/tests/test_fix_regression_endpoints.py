"""Regression tests for fixes in the regression-model endpoints."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from conftest import make_session

SEED = 20261110


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(SEED)
    n = 400
    age = rng.normal(60, 10, n)
    bmi = rng.normal(27, 4, n)
    noise = rng.normal(0, 1, n)
    sex = rng.choice(["F", "M"], n)
    race = rng.choice(["A", "B", "C"], n, p=[0.4, 0.35, 0.25])
    lp = -0.3 + 0.06 * (age - 60) + 0.08 * (bmi - 27) + 0.5 * (sex == "M")
    event = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    sbp = 120 + 0.7 * (age - 60) + 1.2 * (bmi - 27) + rng.normal(0, 8, n)
    latent = 0.08 * (age - 60) + rng.logistic(0, 1, n)
    grade = np.digitize(latent, [-1.0, 0.5, 2.0])
    treat_lp = -0.2 + 0.04 * (age - 60) + 0.9 * (race == "B") - 0.6 * (race == "C")
    treat = (rng.random(n) < 1 / (1 + np.exp(-treat_lp))).astype(int)
    df = pd.DataFrame({
        "age": age, "bmi": bmi, "noise": noise, "sex": sex, "race": race,
        "sex_partner": rng.normal(0, 1, n),
        "event": event, "event_txt": np.where(event == 1, "Case", "Control"),
        "sbp": sbp, "grade": grade, "treat": treat,
    })
    df["three"] = rng.choice(["x", "y", "z"], n)
    df["bmi_gap"] = df["bmi"].where(rng.random(n) > 0.15)
    return df


@pytest.fixture(scope="module")
def sid(frame):
    return make_session(frame, "fix_regression_endpoints")


def _post(client, path, payload, expect=200):
    r = client.post(path, json=payload)
    assert r.status_code == expect, r.text
    return r.json()


@pytest.mark.parametrize("outcome, warns", [("event", True), ("sbp", False)])
def test_linear_binary_outcome_warns_without_preventing_the_fit(client, sid, outcome, warns):
    result = _post(client, "/api/models/linear", {
        "session_id": sid, "outcome": outcome, "predictors": ["age"],
    })
    assert result["n"] == 400
    assert any("Consider logistic regression" in message for message in result.get("warnings", [])) == warns


# ── Logistic ─────────────────────────────────────────────────────────────────

def test_hosmer_lemeshow_survives_dropped_rows(client, sid, frame):
    res = _post(client, "/api/models/logistic", {
        "session_id": sid, "outcome": "event", "predictors": ["age", "bmi_gap"]})
    assert res["n"] < len(frame)
    assert res["hosmer_lemeshow"] is not None
    assert "Hosmer-Lemeshow" in res["result_text"]


def test_text_outcome_names_the_event_level(client, sid):
    num = _post(client, "/api/models/logistic", {"session_id": sid, "outcome": "event", "predictors": ["age"]})
    txt = _post(client, "/api/models/logistic", {"session_id": sid, "outcome": "event_txt", "predictors": ["age"]})
    assert num["outcome_event_level"] == "1"
    assert txt["outcome_event_level"] == "Control" and txt["outcome_reference_level"] == "Case"
    assert "'Control'" in txt["result_text"]
    or_num = next(c for c in num["coefficients"] if c["variable"] == "age")["odds_ratio"]
    or_txt = next(c for c in txt["coefficients"] if c["variable"] == "age")["odds_ratio"]
    assert or_txt == pytest.approx(1 / or_num, rel=1e-6)


@pytest.mark.parametrize("path", ["/api/models/logistic", "/api/models/firth_logistic", "/api/models/logistic_table"])
def test_three_level_outcome_is_a_422(client, sid, path):
    _post(client, path, {"session_id": sid, "outcome": "three", "predictors": ["age"]}, expect=422)


def test_firth_states_its_pseudo_r2_type(client, sid):
    firth = _post(client, "/api/models/firth_logistic", {"session_id": sid, "outcome": "event", "predictors": ["age"]})
    plain = _post(client, "/api/models/logistic", {"session_id": sid, "outcome": "event", "predictors": ["age"]})
    assert firth["pseudo_r2_type"] == "Cox-Snell" and plain["pseudo_r2_type"] == "McFadden"
    assert firth["mcfadden_r2"] == pytest.approx(plain["mcfadden_r2"], abs=5e-3)


def test_logistic_robust_label_is_hc0(client, sid, frame):
    res = _post(client, "/api/models/logistic", {
        "session_id": sid, "outcome": "event", "predictors": ["age", "bmi"], "robust_se": True})
    assert res["robust_se_type"] == "HC0"
    X = sm.add_constant(frame[["age", "bmi"]])
    hc0 = sm.Logit(frame["event"], X).fit(disp=False, cov_type="HC0")
    se = next(c for c in res["coefficients"] if c["variable"] == "age")["se"]
    assert se == pytest.approx(float(hc0.bse["age"]), rel=1e-8)


def test_logistic_warns_about_options_it_does_not_apply(client, sid):
    res = _post(client, "/api/models/logistic", {
        "session_id": sid, "outcome": "event", "predictors": ["age"], "selection": "p05", "use_firth": True})
    text = " ".join(res["warnings"])
    assert "selection='p05'" in text and "use_firth" in text


def test_logistic_interaction_ignores_prefix_sibling(client, sid):
    res = _post(client, "/api/models/logistic", {
        "session_id": sid, "outcome": "event", "predictors": ["sex", "sex_partner", "age"],
        "interactions": [["sex", "age"]]})
    terms = [c["variable"] for c in res["coefficients"] if ":" in c["variable"]]
    assert terms == ["sex_M:age"]


def test_mice_logistic_rows_are_complete_and_text_matches(client, sid):
    res = _post(client, "/api/models/logistic", {
        "session_id": sid, "outcome": "event", "predictors": ["age", "bmi_gap"], "imputation": "mice"})
    assert res["pooled_from_imputations"] is True
    assert res["fit_statistics_source"] == "first imputed dataset"
    row = next(c for c in res["coefficients"] if c["variable"] == "age")
    for key in ("B", "se", "wald", "p", "odds_ratio", "or_ci_low", "or_ci_high", "rubin_df"):
        assert row[key] is not None, key
    assert row["or_ci_low"] < row["odds_ratio"] < row["or_ci_high"]
    assert row["odds_ratio"] == pytest.approx(np.exp(row["B"]))
    if row["p"] < 0.05:
        assert f"age (OR = {row['odds_ratio']:.2f}" in res["result_text"]
    assert "first imputed dataset" in res["result_text"]


# ── Linear ───────────────────────────────────────────────────────────────────

def test_mice_linear_rows_have_confidence_intervals(client, sid):
    res = _post(client, "/api/models/linear", {
        "session_id": sid, "outcome": "sbp", "predictors": ["age", "bmi_gap"], "imputation": "mice"})
    row = next(c for c in res["coefficients"] if c["variable"] == "bmi_gap")
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert res["fit_statistics_source"] == "first imputed dataset"
    assert any("first imputed dataset" in w for w in res["warnings"])


def test_linear_stepwise_uses_ols(client, sid):
    # Before the fix forward selection returned [] here: every candidate was
    # fitted with Logit on a continuous outcome and each fit raised.
    payload = {"session_id": sid, "model_type": "linear", "outcome": "sbp", "candidates": ["age", "bmi", "sex"]}
    for direction in ("forward", "backward", "both"):
        res = _post(client, "/api/models/stepwise", {**payload, "direction": direction})
        assert {"age", "bmi"} <= set(res["selected"]), direction
        assert res["final_aic"] is not None


def test_stepwise_aic_criterion_runs_and_bad_one_is_rejected(client, sid):
    payload = {"session_id": sid, "model_type": "linear", "outcome": "sbp", "candidates": ["age", "bmi", "noise"]}
    res = _post(client, "/api/models/stepwise", {**payload, "criterion": "aic"})
    assert res["criterion"] == "aic" and {"age", "bmi"} <= set(res["selected"])
    _post(client, "/api/models/stepwise", {**payload, "criterion": "bic"}, expect=422)
    _post(client, "/api/models/stepwise", {**payload, "model_type": "cox"}, expect=422)


def test_stepwise_text_outcome_with_dropped_rows(client, sid):
    res = _post(client, "/api/models/stepwise", {
        "session_id": sid, "model_type": "logistic", "outcome": "event_txt",
        "candidates": ["age", "bmi_gap"], "direction": "forward"})
    assert "age" in res["selected"]
    assert res["outcome_event_level"] == "Control"
    assert res["n_excluded"] > 0


def test_compare_models_aic_matches_linear_screen(client, sid):
    lin = _post(client, "/api/models/linear", {"session_id": sid, "outcome": "sbp", "predictors": ["age", "bmi"]})
    cmp_ = _post(client, "/api/model_compare/compare_models", {
        "session_id": sid, "outcome": "sbp", "model_type": "linear",
        "model_specs": [{"name": "m1", "predictors": ["age"]}, {"name": "m2", "predictors": ["age", "bmi"]}]})
    m2 = next(m for m in cmp_["models"] if m["name"] == "m2")
    assert m2["aic"] == pytest.approx(lin["aic"], abs=1e-3)
    assert m2["bic"] == pytest.approx(lin["bic"], abs=1e-3)


def test_multi_outcome_robust_p_uses_t_like_linear(client, sid, frame):
    from routers.models.multi_outcome_regression import _fit_outcome

    small = frame.iloc[:40]
    lin_sid = make_session(small, "fix_regression_small")
    lin = _post(client, "/api/models/linear", {
        "session_id": lin_sid, "outcome": "sbp", "predictors": ["noise", "bmi"], "robust_se": True})
    assert lin["robust_se_type"] == "HC3"
    p_lin = next(c for c in lin["coefficients"] if c["variable"] == "noise")["p"]
    fit = _fit_outcome(small, "sbp", ["noise", "bmi"], "listwise", True, False)
    assert fit["coefficients"]["noise"]["p"] == pytest.approx(p_lin, abs=2e-4)
    X = sm.add_constant(small[["noise", "bmi"]])
    p_z = float(sm.OLS(small["sbp"], X).fit(cov_type="HC3").pvalues["noise"])
    assert abs(p_z - p_lin) > 1e-4


# ── GLM ──────────────────────────────────────────────────────────────────────

def test_ordinal_thresholds_are_cutpoints(client, sid, frame):
    res = _post(client, "/api/models/ordinal", {"session_id": sid, "outcome": "grade", "predictors": ["age"]})
    cuts = [t["coef"] for t in res["thresholds"]]
    assert cuts == sorted(cuts) and len(cuts) == 3
    model = OrderedModel(frame["grade"], frame[["age"]], distr="logit")
    fit = model.fit(method="bfgs", disp=False, maxiter=200)
    expected = model.transform_threshold_params(fit.params.values[1:])[1:-1]
    np.testing.assert_allclose(cuts, expected, rtol=1e-3, atol=1e-3)
    assert res["thresholds"][1]["raw_param"] != pytest.approx(cuts[1])


def test_poisson_robust_label_is_hc0(client, sid):
    res = _post(client, "/api/models/poisson", {
        "session_id": sid, "outcome": "grade", "predictors": ["age"], "robust_se": True})
    assert res["robust_se_type"] == "HC0"


# ── PSM / IPTW ───────────────────────────────────────────────────────────────

def test_psm_reports_level_smd_for_nominal_covariate(client, sid):
    res = _post(client, "/api/models/psm", {
        "session_id": sid, "treatment_col": "treat", "covariates": ["age", "race"]})
    levels = res["smd_by_level"]["race"]
    assert set(levels) == {"A", "B", "C"}
    assert res["smd_before"]["race"] == pytest.approx(max(v["before"] for v in levels.values()), abs=1e-4)
    assert res["smd_after"]["race"] == pytest.approx(max(v["after"] for v in levels.values()), abs=1e-4)
    assert "age" not in res["smd_by_level"]
    if res["smd_after"]["race"] >= 0.10:
        assert res["balance_achieved"] is False


def test_iptw_rejects_continuous_outcome(client, sid):
    payload = {"session_id": sid, "treatment_col": "treat", "covariates": ["age", "bmi"], "outcome_type": "binary"}
    bad = client.post("/api/models/iptw", json={**payload, "outcome_col": "sbp"})
    assert bad.status_code == 422 and "0/1" in bad.json()["detail"]
    ok = _post(client, "/api/models/iptw", {**payload, "outcome_col": "event"})
    assert ok["outcome_result"]["coefficients"][0]["odds_ratio"] > 0
