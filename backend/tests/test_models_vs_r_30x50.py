"""Regression and survival endpoints on 30 rows, against R.

Every literal here was produced by R 4.5.2 on the frame that
`qa/models_audit_30x50/generate_dataset.py` builds, and the R script that
produced it is committed next to the generator. The frame is rebuilt from the
generator rather than read from a CSV, because the repository ignores *.csv to
keep clinical files out and a test that skips when its fixture is missing is a
test that does not run.

Thirty rows is the point. Every value pinned below is one that either did not
exist or was wrong before this audit, and each was wrong in a way that only a
small sample makes visible.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_GEN = pathlib.Path(__file__).resolve().parents[2] / "qa" / "models_audit_30x50"

pytestmark = pytest.mark.skipif(
    not (_GEN / "generate_dataset.py").exists(),
    reason="qa/models_audit_30x50/generate_dataset.py is not present",
)


def _generator():
    spec = importlib.util.spec_from_file_location(
        "gen3050", _GEN / "generate_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def frames():
    g = _generator()
    df = g.build()
    return {
        "wide": df,
        "long": g.build_long(df),
        "recurrent": g.build_recurrent(df),
        "multistate": g.build_multistate(df),
    }


@pytest.fixture()
def sid(client, frames):
    from services import store
    store.save("r3050", frames["wide"])
    store.save("r3050_ms", frames["multistate"])
    return "r3050"


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ── robust standard errors keep the t ────────────────────────────────────────

def test_robust_se_is_hc3_on_the_residual_df_t(client, sid):
    """R: coeftest(lm(sbp ~ age+bmi+arm), vcovHC(type="HC3"), df = 21).

    statsmodels switches to a normal the moment cov_type is set, so asking for
    robust errors also swapped the reference distribution. HC3 is a
    small-sample correction; pairing it with the large-sample distribution
    takes the correction back out.
    """
    r = client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "sbp",
        "predictors": ["age", "bmi", "arm"], "robust_se": True})
    assert r.status_code == 200
    got = {c["variable"]: c for c in r.json()["coefficients"]}
    expected = {          # se,          p (t on 21 df)
        "age":       (0.110212950764988,  0.025192516408906518),
        "bmi":       (0.35752932456966885, 0.01681367957198056),
        "arm_treat": (2.658875056255584,  0.04242511636566832),
    }
    for name, (se, p) in expected.items():
        assert _rel(got[name]["se"], se) < 1e-9, name
        assert _rel(got[name]["p"], p) < 1e-6, name


# ── a predictor with no variance ─────────────────────────────────────────────

@pytest.mark.parametrize("path,body,est_key", [
    ("/api/models/linear",
     {"outcome": "sbp", "predictors": ["age", "const_num"]}, "estimate"),
    ("/api/models/logistic",
     {"outcome": "event_binary", "predictors": ["age", "const_num"]}, "B"),
    ("/api/models/poisson",
     {"outcome": "admissions", "predictors": ["age", "const_num"]}, "log_irr"),
])
def test_constant_predictor_is_dropped_not_reported(client, sid, path, body, est_key):
    """A column holding one repeated value became the intercept.

    statsmodels' add_constant defaults to has_constant="skip", so the design
    already "had" a constant and none was added. The zero-variance predictor
    was then reported as a predictor — p = 1.2e-12 in the linear model, 0.044
    in the logistic one — and no intercept row appeared at all.
    """
    r = client.post(path, json={"session_id": sid, **body})
    assert r.status_code == 200
    j = r.json()
    names = [c["variable"] for c in j["coefficients"]]
    assert "const_num" not in names
    assert "const" in names, "the intercept must be reported in its own right"
    assert any("every row" in str(w) for w in (j.get("warnings") or [])), \
        "dropping a predictor silently is how it went unnoticed"


def test_constant_predictor_leaves_the_rest_matching_r(client, sid):
    """R: lm(sbp ~ age + const_num) -> (Intercept) 121.97787224, age 0.34227251,
    const_num aliased to NA."""
    r = client.post("/api/models/linear", json={
        "session_id": sid, "outcome": "sbp", "predictors": ["age", "const_num"]})
    got = {c["variable"]: c["estimate"] for c in r.json()["coefficients"]}
    assert _rel(got["const"], 121.97787224016143) < 1e-9
    assert _rel(got["age"], 0.3422725119806366) < 1e-9


# ── Fine-Gray ────────────────────────────────────────────────────────────────

def test_fine_gray_matches_crr(client, sid):
    """R: cmprsk::crr(time, cmp_status, model.matrix(~age+arm)[,-1], failcode=1).

    The augmented rows carried no start time, so a competing-event subject sat
    in every earlier risk set once per later event time and had no row at all
    for the period before their own event. The subdistribution risk set was
    inflated in proportion to how much competing risk the data held, and the
    coefficients shrank toward the null by the same amount: on 1000 rows with
    543 competing events the treatment coefficient read 0.306 against crr's
    0.610.
    """
    r = client.post("/api/survival_advanced/fine_gray", json={
        "session_id": sid, "duration_col": "time", "event_col": "cmp_status",
        "event_of_interest": 1, "predictors": ["age", "arm"]})
    assert r.status_code == 200
    got = {c["variable"]: c["estimate"]
           for c in r.json()["regression_result"]["coefficients"]}
    assert _rel(got["age"], 0.003864714747) < 1e-4
    assert _rel(got["arm_treat"], 0.9434693735) < 1e-4


# ── shared frailty ───────────────────────────────────────────────────────────

def test_frailty_reports_the_unpenalised_cluster_robust_fit(client, sid):
    """R: coxph(Surv(time,status) ~ age + arm + cluster(site)).

    The reported model used to be refitted with an L2 penalty that grew with
    the estimated heterogeneity, while the cluster never entered the
    likelihood at any point. That shrank the effect toward the null AND
    shrank its standard error, so the estimate was attenuated and its
    precision overstated at once: 1.073 (SE 0.373) where R's frailty fit
    gives 1.232 (SE 0.459).
    """
    r = client.post("/api/survival_advanced/frailty", json={
        "session_id": sid, "duration_col": "time", "event_col": "status",
        "cluster_col": "site", "predictors": ["age", "arm"]})
    assert r.status_code == 200
    got = {c["variable"]: c for c in r.json()["coefficients"]}
    assert _rel(got["age"]["estimate"], 0.01452091) < 1e-4
    assert _rel(got["age"]["se"], 0.01850769) < 1e-4
    assert _rel(got["arm_treat"]["estimate"], 1.23560916) < 1e-4
    assert _rel(got["arm_treat"]["se"], 0.51508875) < 1e-4


def test_frailty_dummy_codes_a_three_level_category(client, sid):
    """R: coxph(Surv(time,status) ~ age + stage + cluster(site)) gives stageII
    and stageIII. Integer codes gave one coefficient for a linear trend across
    alphabetically ordered levels."""
    r = client.post("/api/survival_advanced/frailty", json={
        "session_id": sid, "duration_col": "time", "event_col": "status",
        "cluster_col": "site", "predictors": ["age", "stage"]})
    assert r.status_code == 200
    got = {c["variable"]: c for c in r.json()["coefficients"]}
    assert {"stage_II", "stage_III"} <= set(got)
    assert _rel(got["stage_II"]["estimate"], 0.36922493) < 1e-4
    assert _rel(got["stage_III"]["estimate"], -0.04674958) < 1e-3


# ── multistate ───────────────────────────────────────────────────────────────

def test_multistate_dummy_codes_a_three_level_category(client, frames):
    from services import store
    ms = frames["multistate"].merge(
        frames["wide"][["pid", "stage"]].rename(columns={"pid": "id"}),
        on="id", how="left")
    store.save("r3050_ms2", ms)
    r = client.post("/api/survival_advanced/multistate", json={
        "session_id": "r3050_ms2", "id_col": "id", "from_state_col": "from_state",
        "to_state_col": "to_state", "entry_col": "entry", "exit_col": "exit",
        "event_col": "event", "predictors": ["age", "stage"]})
    assert r.status_code == 200
    names = [c["variable"]
             for c in r.json()["results"]["1->2"]["coefficients"]]
    assert "stage" not in names
    assert {"stage_II", "stage_III"} <= set(names)


def test_dynamic_prediction_accepts_a_categorical_predictor(client, frames):
    """It averaged the at-risk subjects' covariates straight off the raw
    frame, so a text column was a TypeError and any categorical predictor
    returned a 500."""
    from services import store
    store.save("r3050_ms3", frames["multistate"])
    r = client.post("/api/survival_advanced/dynamic_prediction", json={
        "session_id": "r3050_ms3", "id_col": "id",
        "from_state_col": "from_state", "to_state_col": "to_state",
        "entry_col": "entry", "exit_col": "exit", "event_col": "event",
        "predictors": ["age", "arm"], "landmark_time": 3.0,
        "current_state": 1, "horizon": 8.0, "n_points": 5})
    assert r.status_code == 200


# ── refusals that used to be crashes ─────────────────────────────────────────

@pytest.mark.parametrize("endpoint,extra", [
    ("psm", {"caliper": 0.2, "caliper_scale": "logit"}),
    ("iptw", {"estimand": "ate", "stabilize": True}),
])
def test_text_treatment_column_is_422_not_500(client, sid, endpoint, extra):
    """Both endpoints already carried this check and neither could reach it:
    the cast to float/int ran first and raised, so a trial storing its
    allocation as "control"/"treat" got a 500 with a traceback."""
    r = client.post(f"/api/models/{endpoint}", json={
        "session_id": sid, "treatment_col": "arm",
        "covariates": ["age", "bmi", "sex"], "outcome_col": "event_binary",
        "outcome_type": "binary", **extra})
    assert r.status_code == 422
    assert "binary" in r.json()["detail"]
    assert "control" in r.json()["detail"], "say which levels were found"


def test_psm_iptw_errors_carry_no_traceback(client, sid):
    r = client.post("/api/models/psm", json={
        "session_id": sid, "treatment_col": "arm",
        "covariates": ["age"], "outcome_col": "event_binary"})
    assert "Traceback" not in str(r.json().get("detail", ""))
    assert "site-packages" not in str(r.json().get("detail", ""))


def test_cox_rcs_accepts_a_categorical_covariate(client, sid):
    """Every covariate was coerced with pd.to_numeric(errors="coerce"), which
    turned a categorical into NaN, dropped every row, and reported "not enough
    complete rows" — pointing the user at their data for the coercion's
    doing. The covariates are dummy-coded further down, so the coercion was
    contradicted by the code below it."""
    r = client.post("/api/models/survival/cox_rcs", json={
        "session_id": sid, "duration_col": "time", "event_col": "status",
        "spline_terms": [{"column": "age", "n_knots": 3}],
        "covariates": ["arm"]})
    assert r.status_code == 200
    assert r.json()["n"] == 30


def test_landmark_allows_the_group_to_be_an_adjustment_covariate(client, sid):
    """Plot the landmark curves by arm and adjust for arm: `needed` held the
    column twice, so df[needed] returned duplicate columns and every later
    work[col] was a DataFrame."""
    r = client.post("/api/survival_advanced/landmark", json={
        "session_id": sid, "duration_col": "time", "event_col": "status",
        "landmark_time": 5.0, "group_col": "arm",
        "predictors": ["age", "arm"]})
    assert r.status_code == 200


# ── things that were already right, kept honest ──────────────────────────────

def test_cox_matches_r_at_n_30(client, sid):
    """R: coxph(Surv(time,status) ~ age + arm + sex)."""
    r = client.post("/api/models/survival/cox", json={
        "session_id": sid, "duration_col": "time", "event_col": "status",
        "predictors": ["age", "arm", "sex"]})
    got = {c["variable"]: c for c in r.json()["coefficients"]}
    for name, coef, se in (("age", 0.01385724542, 0.01998672381),
                           ("arm_treat", 1.368661354, 0.5690646946),
                           ("sex_M", 0.2254069222, 0.5676425976)):
        assert _rel(got[name]["log_hr"], coef) < 1e-4, name
        assert _rel(got[name]["se"], se) < 1e-4, name


def test_km_matches_survfit_including_the_ci_transform(client, sid):
    """R: survfit(Surv(time,status) ~ arm, conf.type="log-log") + survdiff.

    The confidence limits are log-log (cloglog), the transform SAS and Stata
    default to and the one that keeps the limits inside [0, 1]. R's survfit
    defaults to conf.type="log" instead, so a naive comparison reads a
    convention as a disagreement — at t = 5 the control arm is [0.500, 0.931]
    here against R's default [0.621, 1.000], and both are correct for their
    own transform.
    """
    r = client.post("/api/models/survival/km", json={
        "session_id": sid, "duration_col": "time", "event_col": "status",
        "group_col": "arm", "survival_times": [5.0, 10.0]})
    assert r.status_code == 200
    j = r.json()

    assert _rel(j["logrank"]["chi2"], 8.071260000939327) < 1e-9
    assert j["logrank"]["df"] == 1
    assert _rel(j["logrank"]["p"], 0.0044972833562596385) < 1e-9

    groups = {g["group"]: g for g in j["groups"]}
    assert groups["control"]["median_survival"] == 15.368
    assert groups["treat"]["median_survival"] == 2.761

    expected = {                    # survival, ci_low, ci_high  (log-log)
        ("control", 5.0):  (0.8,                0.4998239816557543, 0.9307172940541176),
        ("control", 10.0): (0.6666666666666667, 0.37530323006811855, 0.8455622342817989),
        ("treat", 5.0):    (0.26666666666666666, 0.08258082492540529, 0.49633566553142),
        ("treat", 10.0):   (0.13333333333333336, 0.02187320945194147, 0.3457322109524664),
    }
    for name, g in groups.items():
        for point in g["survival_at"]:
            surv, lo, hi = expected[(name, float(point["time"]))]
            assert _rel(point["survival"], surv) < 1e-9, (name, point["time"])
            assert _rel(point["ci_low"], lo) < 1e-9, (name, point["time"])
            assert _rel(point["ci_high"], hi) < 1e-9, (name, point["time"])


# ── the Correlation panel: Pairwise, Matrix, ICC, Cohen's κ ─────────────────

@pytest.fixture()
def rater_sid(client, frames):
    from services import store
    g = _generator()
    store.save("r3050_ag", g.build_raters(frames["wide"]))
    return "r3050_ag"


def test_icc_interval_belongs_to_the_same_icc_as_the_estimate(client, rater_sid):
    """R: irr::icc(model="twoway", type="agreement", unit="single").

    The estimate is ICC(A,1) — absolute agreement, so a systematic offset
    between raters counts against it. The interval used to be the consistency
    one, which ignores that offset: [0.814, 0.955] where the agreement
    interval is [0.800, 0.952]. Rater B is 2.1 points high by construction
    here, so the two forms have to differ.
    """
    r = client.post("/api/stats/icc", json={
        "session_id": rater_sid, "rater1_col": "rater_a", "rater2_col": "rater_b"})
    assert r.status_code == 200
    j = r.json()
    assert _rel(j["icc"], 0.9009200782048743) < 1e-9
    assert _rel(j["ci_low"], 0.7997117086715196) < 1e-8
    assert _rel(j["ci_high"], 0.9519402629195801) < 1e-8


def test_cohens_kappa_matches_irr_and_reports_expected_agreement(client, rater_sid):
    """R: irr::kappa2. `pe` used to return `po`, so the response said observed
    and expected agreement were both 0.90 on data where chance agreement is
    0.33 — a reader would conclude the raters agreed no better than chance."""
    r = client.post("/api/stats/cohens_kappa", json={
        "session_id": rater_sid, "rater1_col": "cat_a", "rater2_col": "cat_b"})
    assert r.status_code == 200
    j = r.json()
    assert _rel(j["kappa"], 0.8497495826377296) < 1e-9
    assert _rel(j["po"], 0.9) < 1e-12
    assert _rel(j["pe"], 0.33444444444444443) < 1e-9
    assert j["po"] != j["pe"]
    assert _rel(j["z"], 6.58652994716785) < 1e-8
    assert j["p"] is not None and j["p"] < 1e-9
    assert j["ci_high"] <= 1.0, "kappa cannot exceed 1"


def test_fleiss_kappa_matches_irr_unrounded(client, rater_sid):
    """R: irr::kappam.fleiss. The response rounded to four decimals on the way
    out, and used a different published null variance than the reference
    implementation (0.0813 against 0.0792)."""
    r = client.post("/api/stats/fleiss_kappa", json={
        "session_id": rater_sid, "rater_cols": ["cat_a", "cat_b", "cat_c"]})
    assert r.status_code == 200
    j = r.json()
    assert _rel(j["kappa"], 0.7370129870129871) < 1e-12
    assert _rel(j["se"], 0.0792075046) < 1e-8
    assert _rel(j["z"], 9.304838) < 1e-6
    assert j["kappa"] != round(j["kappa"], 4), "do not round on the way out"


def test_correlation_pair_computes_the_method_it_was_asked_for(client, sid):
    """Anything other than "pearson" fell through to Spearman in silence.
    R: cor.test(age, bmi) — pearson 0.3051861136, spearman 0.3717948718,
    kendall 0.2421652422."""
    expected = {
        "pearson": 0.30518611362239095,
        "spearman": 0.3717948717948718,
        "kendall": 0.24216524216524216,
    }
    for method, r_value in expected.items():
        r = client.post("/api/stats/correlation_pair", json={
            "session_id": sid, "var1": "age", "var2": "bmi", "method": method})
        assert r.status_code == 200, method
        j = r.json()
        assert j["method"] == method
        assert _rel(j["r"], r_value) < 1e-9, method


def test_correlation_pair_rejects_an_unknown_method(client, sid):
    r = client.post("/api/stats/correlation_pair", json={
        "session_id": sid, "var1": "age", "var2": "bmi", "method": "banana"})
    assert r.status_code == 422


def test_pearson_ci_uses_the_exact_normal_quantile(client, sid):
    """R: cor.test(age, bmi)$conf.int. The hardcoded 1.96 moved the Fisher-z
    limits in the sixth decimal."""
    j = client.post("/api/stats/correlation_pair", json={
        "session_id": sid, "var1": "age", "var2": "bmi", "method": "pearson"}).json()
    assert _rel(j["ci_low"], -0.08464449646) < 1e-8
    assert _rel(j["ci_high"], 0.61399232820) < 1e-8


def test_gee_matches_geeglm_at_n_30(client, frames):
    """R: geepack::geeglm(score ~ visit+arm+age, id=pid, corstr="exchangeable")."""
    from services import store
    store.save("r3050_long", frames["long"])
    r = client.post("/api/models/gee", json={
        "session_id": "r3050_long", "outcome": "score",
        "predictors": ["visit", "arm", "age"], "group_col": "pid",
        "family": "gaussian", "cov_struct": "exchangeable"})
    assert r.status_code == 200
    got = {c["variable"]: c for c in r.json()["coefficients"]}
    assert _rel(got["const"]["estimate"], 51.72989584) < 1e-6
    assert _rel(got["arm_treat"]["estimate"], 2.89774153) < 1e-6
