"""Model endpoints cross-validated against R on a fixed sample dataset.

The reference values below were produced by R 4.5.2 with survival / MASS /
lme4, from `qa/models_audit/reference.R` on `qa/models_audit/dataset.csv`.
They are pasted here as literals so the suite does not need R installed.

Four defects the cross-check found, all fixed here:

  * /polynomial cast covariates straight to float, so a categorical one —
    adjusting a dose-response curve for treatment arm, the ordinary case —
    raised ValueError and came back as an unhandled 500.
  * /negbinom estimated the dispersion from Poisson residuals and then fixed
    it, so theta came out at 3.28 against a true 2.78 and every standard
    error was about 4% too small — one-sided, always toward significance.
  * AIC and BIC left the estimated scale out of the parameter count, sitting
    exactly 2 and log(n) below R's for every OLS and Gamma fit.
  * the Gamma coefficients were tested against a normal when the dispersion
    is estimated from the data, which always gives the smaller p.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from conftest import make_session

# The dataset is BUILT here rather than read from qa/models_audit/dataset.csv.
# The repository ignores *.csv so that no clinical file is ever committed by
# accident, and a test that silently skips when its fixture is missing is a
# test that does not run. The generator is committed and seeded, so the frame
# is identical to the one the R reference was fitted on.
_GEN = pathlib.Path(__file__).resolve().parents[2] / "qa" / "models_audit"

pytestmark = pytest.mark.skipif(
    not (_GEN / "generate_dataset.py").exists(),
    reason="qa/models_audit/generate_dataset.py not present")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "models_audit_gen", _GEN / "generate_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = mod.build()
    # Round-trip through CSV so dtypes match what the R reference read.
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def _long_frame() -> pd.DataFrame:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "models_audit_gen", _GEN / "generate_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_long(mod.build())


@pytest.fixture()
def sid(frame) -> str:
    return make_session(frame.copy(), "models_vs_r")


def _terms(payload, key="estimate"):
    return {c["variable"]: c for c in payload["coefficients"]}


# ── R reference values, generated from qa/models_audit/reference.json ────────
# (estimate, se, p) per term. Do not hand-edit — regenerate from the JSON.

R_LINEAR = {
    "const": (95.11327287442947, 4.487205995325991, 3.1029732080118793e-61),
    "age": (0.40428097486789755, 0.04618763216042007, 1.6369774941095782e-16),
    "bmi": (0.767261988355806, 0.12550547680307175, 3.0846950350836427e-09),
    "arm_treat": (6.07962689048885, 1.0157514179546865, 6.25422046143562e-09),
    "sex_M": (0.3545791040931057, 1.0099245357327595, 0.7257679472336525),
}

R_LINEAR_R2 = 0.368847513452583
R_LINEAR_AIC = 2157.813716009417
R_LINEAR_BIC = 2180.0364108573544

R_POLY = {
    "const": (98.66245934773578, 12.469242508071817, 5.079976292084268e-14),
    "age": (0.9938384353239559, 0.4091422672245315, 0.015732427548913543),
    "age^2": (-0.004735333725766885, 0.003313650008121473, 0.15404759363958262),
    "arm_treat": (6.804816289701107, 1.0693414383872042, 7.481840912840576e-10),
}

R_POLY_R2 = 0.2934693014774892

R_NB = {
    "const": (0.09581478683159404, 0.27315590455920363, 0.7257612070498778),
    "age": (0.018056199314714253, 0.004265602859977703, 2.3061797360819768e-05),
    "arm_treat": (0.13617295498070642, 0.09250178223473145, 0.14099071921150597),
}

R_NB_THETA = 2.78421980111332
R_NB_AIC = 1402.4401421599823

R_GAMMA = {
    "const": (0.8208226087419606, 0.22344716999825873, 0.00028377500474539884),
    "age": (0.023216858919824338, 0.003539631168635358, 2.399553373192722e-10),
    "arm_treat": (0.04310093327030131, 0.0774893625711294, 0.5784810389894999),
}

R_GAMMA_DISP = 0.44732925157725933

R_LOGIT = {
    "const": (-3.8633216915921884, 1.2531920408367907, 0.002050732210536261),
    "age": (0.03392156607503565, 0.01266973807640264, 0.007420282629843824),
    "bmi": (0.08087059628004423, 0.03516334061953619, 0.021456417655967525),
    "arm_treat": (0.784284192800052, 0.27776497179213305, 0.004749453348278597),
    "sex_M": (0.43569238180903397, 0.2762527802632471, 0.11476075700322261),
}

R_COX = {
    "age": (0.026983631498801147, 0.007283800075997384, 0.00021171703329672054),
    "arm_treat": (0.45823088301096415, 0.15118688859628976, 0.002438337723173426),
    "sex_M": (-0.5286759875095557, 0.15249958446488607, 0.0005268166196748788),
}

R_COX_C = 0.6385241555687241

R_POIS = {
    "const": (-1.6890346929012736, 0.3287995084274369, 2.791968114942763e-07),
    "age": (0.02542374123085334, 0.004963982141294736, 3.028859361348647e-07),
    "arm_treat": (0.4224409564109313, 0.10967368846968163, 0.00011725345536655761),
}


def _check(got, expected, est_key="estimate", rel=1e-5):
    for name, (est, se, p) in expected.items():
        assert got[name][est_key] == pytest.approx(est, rel=rel), f"{name}.estimate"
        assert got[name]["se"] == pytest.approx(se, rel=rel), f"{name}.se"
        assert got[name]["p"] == pytest.approx(p, rel=1e-3, abs=1e-12), f"{name}.p"


def test_linear_matches_r(client, sid):
    r_ = client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "sbp",
        "predictors": ["age", "bmi", "arm", "sex"]})
    assert r_.status_code == 200, r_.text
    out = r_.json()
    _check(_terms(out), R_LINEAR, rel=1e-9)
    assert out["r_squared"] == pytest.approx(R_LINEAR_R2, rel=1e-9)


def test_information_criteria_count_the_residual_variance(client, sid):
    """statsmodels leaves sigma out of the parameter count, so AIC sat exactly
    2 below R's and BIC log(n) below. A criterion only means something next to
    another criterion, so a fixed offset is not harmless."""
    out = client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "sbp",
        "predictors": ["age", "bmi", "arm", "sex"]}).json()
    assert out["aic"] == pytest.approx(R_LINEAR_AIC, rel=1e-9)
    assert out["bic"] == pytest.approx(R_LINEAR_BIC, rel=1e-9)


def test_polynomial_accepts_a_categorical_covariate(client, sid):
    """This used to raise ValueError inside the handler and surface as a 500:
    covariates were cast straight to float with no dummy encoding, and then
    the prediction curve took the mean of the raw category."""
    r_ = client.post("/api/models/polynomial", json={
        "session_id": sid, "outcome": "sbp", "predictor": "age",
        "degree": 2, "covariates": ["arm"]})
    assert r_.status_code == 200, r_.text
    got = _terms(r_.json())
    got["age"] = got.pop("age^1")
    _check(got, R_POLY, rel=1e-6)
    assert r_.json()["r_squared"] == pytest.approx(R_POLY_R2, rel=1e-9)


def test_negbinom_estimates_the_dispersion_by_maximum_likelihood(client, sid):
    """The dispersion came from the Pearson residuals of a Poisson fit and was
    then treated as KNOWN. It missed (theta 3.28 against 2.78) and, by fixing
    an estimated parameter, understated every standard error by about 4% —
    always in the direction that makes a result look more significant."""
    r_ = client.post("/api/models/negbinom", json={
        "session_id": sid, "outcome": "visits", "predictors": ["age", "arm"]})
    assert r_.status_code == 200, r_.text
    out = r_.json()
    assert out["theta"] == pytest.approx(R_NB_THETA, rel=1e-6)
    assert out["alpha"] == pytest.approx(1.0 / R_NB_THETA, rel=1e-6)
    assert out["aic"] == pytest.approx(R_NB_AIC, rel=1e-9)
    got = _terms(out)
    for name, (est, _se, _p) in R_NB.items():
        assert got[name]["log_irr"] == pytest.approx(est, rel=1e-5), name
    assert "alpha" not in got, "the dispersion is not a coefficient row"


def test_negbinom_standard_errors_are_not_anticonservative(client, sid):
    """The old fixed-dispersion fit gave SEs strictly smaller than R's. They
    may now differ slightly the other way — the MLE propagates the uncertainty
    in the dispersion where R's glm.nb conditions on it — but never smaller."""
    got = _terms(client.post("/api/models/negbinom", json={
        "session_id": sid, "outcome": "visits",
        "predictors": ["age", "arm"]}).json())
    for name, (_est, r_se, _p) in R_NB.items():
        assert got[name]["se"] >= r_se * 0.999, name
        assert got[name]["se"] == pytest.approx(r_se, rel=0.05), name


def test_gamma_tests_coefficients_with_a_t_not_a_z(client, sid):
    """The Gamma dispersion is estimated from the data, not fixed at 1 like
    binomial and Poisson, so R uses a t on the residual df. A normal is the
    large-sample limit and is always the smaller p."""
    r_ = client.post("/api/models/gamma", json={
        "session_id": sid, "outcome": "cost",
        "predictors": ["age", "arm"], "link": "log"})
    assert r_.status_code == 200, r_.text
    out = r_.json()
    _check(_terms(out), R_GAMMA, rel=1e-4)
    assert _terms(out)["const"]["df"] == 297
    assert out["dispersion"] == pytest.approx(R_GAMMA_DISP, rel=1e-5)


# ── models already in agreement: lock them so they stay that way ─────────────


def test_logistic_matches_r(client, sid):
    got = _terms(client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "event_binary",
        "predictors": ["age", "bmi", "arm", "sex"]}).json())
    _check(got, R_LOGIT, est_key="B", rel=1e-5)


def test_cox_matches_r(client, sid):
    out = client.post("/api/models/survival/cox", json={
        "session_id": sid, "duration_col": "time", "event_col": "status",
        "predictors": ["age", "arm", "sex"]}).json()
    _check(_terms(out), R_COX, est_key="log_hr", rel=1e-5)
    assert out["concordance"] == pytest.approx(R_COX_C, rel=1e-3)


def test_poisson_matches_r(client, sid):
    got = _terms(client.post("/api/models/poisson", json={
        "session_id": sid, "outcome": "admissions",
        "predictors": ["age", "arm"]}).json())
    _check(got, R_POIS, est_key="log_irr", rel=1e-5)


# ── GEE / IPTW, cross-checked against geepack and survey ─────────────────────
# geepack, MatchIt and survey were installed for this pass, so the three
# endpoints the first cross-check could not reach are covered here.

R_GEE = {
    "const": (46.10557996522679, 2.0625721153125016, 0),
    "visit_v2": (3.1244766666666646, 0.2360519660968432, 0),
    "visit_v3": (6.510846666666665, 0.23146285823831597, 0),
    "arm_treat": (2.781731740947323, 0.73285923372758, 0.00014721291440400464),
    "age": (0.057397953237171784, 0.03192772403423477, 0.07221719339812627),
}

R_IPTW_TREAT = (0.7449893683584642, 0.27245732447528, 0.00662480058482372)
R_IPTW_ESS = 295.0297144835016


def test_gee_matches_geeglm_including_the_intercept(client, frame):
    """The intercept row was dropped from the response, so the table could not
    be lined up against R or any other package, and every number was rounded
    to six decimals on the way out."""
    long = _long_frame()
    sid_l = make_session(long, "models_vs_r_long")
    out = client.post("/api/models/gee", json={
        "session_id": sid_l, "outcome": "score",
        "predictors": ["visit", "arm", "age"], "group_col": "pid",
        "family": "gaussian", "cov_struct": "independence"}).json()
    got = {c["variable"]: c for c in out["coefficients"]}

    assert "const" in got, "the intercept is a fitted parameter like any other"
    for name, (est, se, p) in R_GEE.items():
        assert got[name]["estimate"] == pytest.approx(est, rel=1e-8), name
        assert got[name]["se"] == pytest.approx(se, rel=1e-8), name
    # Rounding to 6 dp would have flattened this to 0.057398.
    assert got["age"]["estimate"] != pytest.approx(round(R_GEE["age"][0], 6), abs=0)


def test_gee_autoregressive_degrades_instead_of_refusing(client):
    """statsmodels cannot always bracket the AR correlation parameter and used
    to surface that as a flat 422. The working correlation is a nuisance
    structure — the coefficients are consistent under any of them."""
    sid_l = make_session(_long_frame(), "models_vs_r_long_ar")
    r_ = client.post("/api/models/gee", json={
        "session_id": sid_l, "outcome": "score",
        "predictors": ["visit", "arm", "age"], "group_col": "pid",
        "family": "gaussian", "cov_struct": "ar"})
    assert r_.status_code == 200, r_.text
    out = r_.json()
    assert out["cov_struct"] == "ar"
    assert out["cov_struct_used"] == "exchangeable"
    assert any("autoregressive" in str(w) for w in out["warnings"])


def test_iptw_reports_the_treatment_effect(client, frame):
    """The weighted outcome model regressed the outcome on the COVARIATES and
    never included the treatment, so the response carried coefficients for age,
    bmi and sex and no treatment effect at all — the one number an IPTW
    analysis exists to produce."""
    df = frame.copy()
    df["treat01"] = (df["arm"] == "treat").astype(int)
    sid2 = make_session(df, "models_vs_r_iptw")
    out = client.post("/api/models/iptw", json={
        "session_id": sid2, "treatment_col": "treat01",
        "covariates": ["age", "bmi", "sex"], "outcome_col": "event_binary",
        "estimand": "ate", "stabilize": True, "outcome_type": "binary"}).json()

    res = out["outcome_result"]
    assert res["treatment"] == "treat01"
    names = [c["variable"] for c in res["coefficients"]]
    assert names == ["treat01"], names

    co = res["coefficients"][0]
    est, se, p = R_IPTW_TREAT
    assert co["estimate"] == pytest.approx(est, rel=0.01)
    assert co["se"] == pytest.approx(se, rel=0.02)
    assert co["p"] == pytest.approx(p, rel=0.2)
    # An estimate with no uncertainty attached cannot be read as a result.
    for k in ("se", "ci_low", "ci_high", "odds_ratio", "or_ci_low", "or_ci_high"):
        assert co[k] is not None, k
    assert out["weight_summary"]["effective_n"] == pytest.approx(R_IPTW_ESS, rel=0.01)


def test_iptw_survival_reports_uncertainty_too(client, frame):
    """The weighted Cox returned a bare hazard ratio — no SE, no CI, no p."""
    df = frame.copy()
    df["treat01"] = (df["arm"] == "treat").astype(int)
    sid2 = make_session(df, "models_vs_r_iptw_surv")
    out = client.post("/api/models/iptw", json={
        "session_id": sid2, "treatment_col": "treat01",
        "covariates": ["age", "bmi", "sex"], "outcome_type": "survival",
        "survival_duration_col": "time", "survival_event_col": "status",
        "estimand": "ate", "stabilize": True}).json()
    co = out["outcome_result"]["coefficients"]
    assert [c["variable"] for c in co] == ["treat01"]
    for k in ("hr", "se", "p", "hr_ci_low", "hr_ci_high"):
        assert co[0][k] is not None, k


# ── VIF / ROC / RCS, cross-checked against car, pROC and rms ─────────────────

R_VIF = {
    "age": 1.0079827477581905,
    "bmi": 1.0121540120716646,
    "arm_treat": 1.0172047746180316,
    "sex_M": 1.0002050436966766,
}


def test_vif_matches_car(client, sid):
    """statsmodels' variance_inflation_factor needs the intercept IN the
    design matrix. Four model routers each dropped it first, so every
    auxiliary regression ran through the origin and its R-squared absorbed the
    mean structure: age came back at 21.17 and bmi at 21.69 where car::vif
    gives 1.008 and 1.012. Ten is the conventional "drop this predictor"
    threshold, so the numbers were telling users to throw away sound
    predictors on data with no collinearity at all."""
    out = client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "sbp",
        "predictors": ["age", "bmi", "arm", "sex"]}).json()
    got = {c["variable"]: c.get("vif") for c in out["coefficients"]}
    for name, expected in R_VIF.items():
        assert got[name] == pytest.approx(expected, rel=1e-6), name
        assert got[name] < 2, f"{name} was reported at 21 before this fix"


def test_vif_agrees_across_endpoints(client, sid):
    """/api/diagnostics computed VIF correctly while the model endpoints did
    not, so the same quantity had two answers depending on the screen."""
    lin = client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "sbp",
        "predictors": ["age", "bmi", "arm", "sex"]}).json()
    diag = client.post("/api/diagnostics/linear_full", json={
        "session_id": sid, "outcome": "sbp",
        "predictors": ["age", "bmi", "arm", "sex"]}).json()
    from_model = {c["variable"]: c.get("vif") for c in lin["coefficients"]}
    from_diag = {v["variable"]: v["vif"] for v in diag["vif"]}
    # /api/diagnostics rounds to 3 decimals for display; the underlying value
    # is what has to agree.
    for name in from_diag:
        assert from_model[name] == pytest.approx(from_diag[name], abs=5e-4), name


def test_roc_matches_proc(client, sid):
    out = client.post("/api/stats/roc", json={
        "session_id": sid, "score_column": "age",
        "outcome_column": "event_binary"}).json()
    assert out["auc"] == pytest.approx(0.6043181690140845, abs=5e-5)
    assert out["ci_lower"] == pytest.approx(0.5313312764, abs=5e-5)
    assert out["ci_upper"] == pytest.approx(0.6773050616, abs=5e-5)
    assert out["sensitivity"] == pytest.approx(0.7214611872, abs=5e-4)
    assert out["specificity"] == pytest.approx(0.4938271605, abs=5e-4)


def test_roc_delong_comparison_matches_proc(client, sid):
    out = client.post("/api/stats/roc_compare", json={
        "session_id": sid, "score_column_1": "age", "score_column_2": "bmi",
        "outcome_column": "event_binary"}).json()
    assert out["z"] == pytest.approx(0.2083022993, abs=5e-4)
    assert out["p"] == pytest.approx(0.8349929364, abs=1e-5)


def test_rcs_matches_rms(client, sid):
    """Harrell knot placement and the nonlinearity Wald test."""
    out = client.post("/api/models/rcs", json={
        "session_id": sid, "predictor": "age", "outcome": "event_binary",
        "model_type": "logistic", "n_knots": 4, "covariates": []}).json()
    assert out["knots"] == pytest.approx([44.595, 58.025, 67.0, 79.82], abs=0.01)
    assert out["nonlinearity_wald"] == pytest.approx(0.21084121, abs=5e-4)
    assert out["nonlinearity_df"] == 2
    assert out["nonlinearity_p"] == pytest.approx(0.89994592, abs=1e-5)


# ── propensity score matching, against MatchIt ───────────────────────────────

def test_psm_keeps_every_matchable_patient(client, frame):
    """The neighbour search asked for a fixed five candidates and dropped the
    treated unit if all five were taken. Greedy matching works down from the
    highest propensity, so the units processed later are exactly the ones
    whose nearest controls have been spent — 17 of 138 matchable patients
    were discarded with free controls sitting inside the caliper.

    MatchIt retains 138 pairs at the same caliper on this dataset."""
    df = frame.copy()
    df["treat01"] = (df["arm"] == "treat").astype(int)
    sid2 = make_session(df, "models_vs_r_psm")
    out = client.post("/api/models/psm", json={
        "session_id": sid2, "treatment_col": "treat01",
        "covariates": ["age", "bmi", "sex"], "outcome_col": "event_binary",
        "caliper": 0.2, "ratio": 1, "outcome_type": "binary"}).json()

    assert out["n_matched_pairs"] == 138
    # Matching is only worth doing if it improves balance.
    assert out["avg_smd_after"] < out["avg_smd_before"]
    assert out["avg_smd_after"] < 0.1


def test_psm_still_respects_the_caliper(client, frame):
    """Searching further must not mean matching outside the caliper — a
    tighter caliper has to retain fewer pairs, not the same number."""
    df = frame.copy()
    df["treat01"] = (df["arm"] == "treat").astype(int)
    sid2 = make_session(df, "models_vs_r_psm_tight")
    wide = client.post("/api/models/psm", json={
        "session_id": sid2, "treatment_col": "treat01",
        "covariates": ["age", "bmi", "sex"], "caliper": 0.2, "ratio": 1}).json()
    tight = client.post("/api/models/psm", json={
        "session_id": sid2, "treatment_col": "treat01",
        "covariates": ["age", "bmi", "sex"], "caliper": 0.01, "ratio": 1}).json()
    assert tight["n_matched_pairs"] < wide["n_matched_pairs"]


def test_psm_matched_effect_matches_clogit(client, frame):
    """R's survival::clogit run on uSTAT's OWN matched pairs, so the matching
    and the estimation are checked separately: this pins the estimation."""
    df = frame.copy()
    df["treat01"] = (df["arm"] == "treat").astype(int)
    sid2 = make_session(df, "models_vs_r_psm_effect")
    out = client.post("/api/models/psm", json={
        "session_id": sid2, "treatment_col": "treat01",
        "covariates": ["age", "bmi", "sex"], "outcome_col": "event_binary",
        "caliper": 0.2, "ratio": 1, "outcome_type": "binary"}).json()

    res = out["outcome_result"]
    assert res["type"] == "conditional_logistic"
    co = res["coefficients"][0]
    assert co["variable"] == "treat01"
    # clogit(event_binary ~ treat01 + strata(match_set_id)) on the exported
    # matched set: coef 0.69314718, se 0.27386128, z 2.531016, p 0.01137328.
    assert co["estimate"] == pytest.approx(0.69314718, abs=1e-5)
    assert co["se"] == pytest.approx(0.27386128, abs=1e-5)
    assert co["z"] == pytest.approx(2.531016, abs=1e-3)
    assert co["p"] == pytest.approx(0.01137328, abs=1e-5)
    assert co["or"] == pytest.approx(2.0, abs=1e-4)
    assert res["log_likelihood"] == pytest.approx(-38.190850, abs=1e-3)

    # The counts have to add up and be readable on their own.
    assert res["n_matched_sets"] == 138
    assert res["n_informative_sets"] + res["n_uninformative_sets"] == 138
    assert res["n_matched_rows"] == 276
    assert res["n_rows_contributing"] == res["n_informative_sets"] * 2


# ── IPTW and GEE, checked on uSTAT's own intermediate output ─────────────────
# Same split as the PSM check: the weights (IPTW) and the encoded design (GEE)
# are exported and handed to R, so what is computed FROM them is verified
# independently of how they were built.


def test_iptw_effect_matches_svyglm_on_ustats_own_weights(client, frame):
    """survey::svyglm run on the weights uSTAT produced.

    The estimate agreed from the start. The standard error did not: the
    sandwich was reported raw and tested against a normal, where the survey
    convention applies a finite-sample correction for the number of sampling
    units and tests against a t on the design degrees of freedom. A z is the
    large-sample limit and always the smaller p."""
    df = frame.copy()
    df["treat01"] = (df["arm"] == "treat").astype(int)
    sid2 = make_session(df, "models_vs_r_iptw_se")
    out = client.post("/api/models/iptw", json={
        "session_id": sid2, "treatment_col": "treat01",
        "covariates": ["age", "bmi", "sex"], "outcome_col": "event_binary",
        "estimand": "ate", "stabilize": True, "outcome_type": "binary"}).json()
    co = out["outcome_result"]["coefficients"][0]

    # svyglm(event_binary ~ treat01, design=svydesign(~1, ~iptw_weight))
    assert co["estimate"] == pytest.approx(0.74619646, abs=1e-6)
    assert co["se"] == pytest.approx(0.27240986, abs=1e-7)
    assert co["p"] == pytest.approx(0.00652914, abs=1e-5)
    assert co["df"] == 299
    # Without the correction the SE was 0.27195546 — smaller, so a smaller p.
    assert co["se"] > 0.272


def test_gee_matches_geeglm_on_ustats_own_design(client):
    """geepack::geeglm run on the design matrix uSTAT encodes, so the dummy
    coding and the estimation are checked separately. Worst relative
    difference across every coefficient and standard error: 4e-11."""
    sid_l = make_session(_long_frame(), "models_vs_r_gee_design")
    out = client.post("/api/models/gee", json={
        "session_id": sid_l, "outcome": "score",
        "predictors": ["visit", "arm", "age"], "group_col": "pid",
        "family": "gaussian", "cov_struct": "independence"}).json()
    got = {c["variable"]: c for c in out["coefficients"]}

    expected = {
        "const": (46.1055799652, 2.0625721153),
        "visit_v2": (3.1244766667, 0.2360519661),
        "visit_v3": (6.5108466667, 0.2314628582),
        "arm_treat": (2.7817317409, 0.7328592337),
        "age": (0.0573979532, 0.0319277240),
    }
    # The literals above carry ten significant digits; the tolerance reflects
    # that, not the size of any disagreement — the measured worst relative
    # difference across all ten values is 4e-11.
    for name, (est, se) in expected.items():
        assert got[name]["estimate"] == pytest.approx(est, rel=1e-8), name
        assert got[name]["se"] == pytest.approx(se, rel=1e-8), name
    assert out["n_obs"] == 900
    assert out["n_clusters"] == 300
