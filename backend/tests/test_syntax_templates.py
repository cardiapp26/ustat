"""The syntax view: services/syntax_templates.py + /api/project/syntax.

The Python side of every template must be valid Python (ast.parse); the R
side gets structural checks (and a parse when Rscript is on the machine).
Unknown panels and unknown subtypes return None end to end -- an honest gap,
never a guessed formula.
"""
import ast
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from main import app
from services.syntax import ENDPOINTS
from services.syntax_templates import _endpoint_key, translate

client = TestClient(app)


def _parses_r(code: str) -> bool:
    if not shutil.which("Rscript"):
        return True  # structural checks stand on machines without R
    proc = subprocess.run(
        ["Rscript", "-e", "invisible(parse(text=commandArgs(TRUE)[1]))", code],
        capture_output=True, text=True, timeout=60,
    )
    return proc.returncode == 0


def test_every_covered_combo_generates_valid_code():
    combos = [
        ("hypothesis", {"test": "ttest_1sample", "col": "age", "mu": 50}),
        ("hypothesis", {"test": "ttest_2sample", "col": "age", "groupCol": "arm"}),
        ("hypothesis", {"test": "anova", "col": "age", "groupCol": "arm"}),
        ("hypothesis", {"test": "mannwhitney", "col": "age", "groupCol": "arm"}),
        ("hypothesis", {"test": "kruskal", "col": "age", "groupCol": "arm"}),
        ("hypothesis", {"test": "ancova", "col": "sbp", "groupCol": "arm", "covariates": ["age", "bmi"]}),
        ("hypothesis", {"test": "two_way", "col": "sbp", "groupCol": "arm", "factor2": "sex"}),
        ("hypothesis", {"test": "chisquare", "col": "smoker", "col2": "arm"}),
        ("hypothesis", {"test": "fisher", "col": "smoker", "col2": "arm"}),
        ("models", {"model": "linear", "outcome": "sbp", "predictors": ["age", "bmi"]}),
        ("models", {"model": "logistic", "outcome": "dm", "predictors": ["age", "bmi"]}),
        ("models", {"model": "poisson", "outcome": "events", "predictors": ["age"]}),
        ("models", {"model": "cox", "durationCol": "t", "eventCol": "died", "predictors": ["age", "arm"]}),
    ]
    for panel, params in combos:
        out = translate(panel, params)
        assert out is not None, (panel, params)
        ast.parse(out["python"])
        assert _parses_r(out["r"]), (panel, params, out["r"])
        # The honesty disclaimer must ride every translation.
        assert "provenance line" in out["python"]
        assert "provenance line" in out["r"]


def test_column_names_land_in_the_code():
    out = translate("models", {"model": "logistic", "outcome": "dm", "predictors": ["age", "bmi"]})
    assert "dm ~ age + bmi" in out["python"]
    assert "dm ~ age + bmi" in out["r"]
    assert "exp(" in out["r"]  # odds ratios, not raw coefficients


def test_unknown_panel_and_subtype_return_none():
    assert translate("roc", {"y": "DM"}) is None
    assert translate("hypothesis", {"test": "mancova", "col": "x"}) is None
    assert translate("models", {"model": "firth"}) is None


def test_endpoint_round_trip_and_honest_nulls():
    r = client.post(
        "/api/project/syntax",
        json={"panel": "hypothesis", "params": {"test": "anova", "col": "age", "groupCol": "arm"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert "anova_lm" in body["python"]
    assert "aov(" in body["r"]

    r2 = client.post("/api/project/syntax", json={"panel": "meta", "params": {}})
    assert r2.json() == {"title": None, "python": None, "r": None}


# ── Recorded requests (endpoint + body) ──────────────────────────────────────

def _post(url: str, **body) -> dict:
    return {"method": "POST", "url": url, "body": {"session_id": "{sid}", **body}}


# One realistic request per endpoint template. The first three keep their
# positions: the tests below index them.
REQUESTS = [
    _post("/api/survival_advanced/fine_gray", duration_col="time", event_col="status",
          event_of_interest=1, group_col="arm", predictors=["age"]),
    _post("/api/models/ordinal", outcome="health", predictors=["x"],
          level_order=["Poor", "Fair", "Good"]),
    _post("/api/models/firth_logistic", outcome="event", predictors=["age", "dm"]),
    # hypothesis tests
    _post("/api/stats/ttest", column="sbp", group_column="arm", method="auto"),
    _post("/api/stats/ttest", column="sbp", mu=120),
    _post("/api/repeated/paired_ttest", col1="sbp_pre", col2="sbp_post"),
    _post("/api/stats/anova", column="sbp", group_col="stage"),
    _post("/api/stats/mannwhitney", column="crp", group_column="arm"),
    _post("/api/repeated/wilcoxon_signed_rank", column1="pain_pre", column2="pain_post"),
    _post("/api/stats/kruskal", column="crp", group_column="stage", posthoc_correction="bonferroni"),
    _post("/api/repeated/friedman", columns=["pain_w0", "pain_w4", "pain_w12"]),
    _post("/api/stats/jonckheere_terpstra", column="crp", group_column="nyha"),
    _post("/api/stats/normality", variables=["sbp", "crp"], group_column="arm"),
    _post("/api/stats/chisquare", row_column="smoker", col_column="arm"),
    _post("/api/stats/fisher", row_col="smoker", col_col="arm"),
    _post("/api/categorical/mcnemar", col1="resp_w0", col2="resp_w12"),
    _post("/api/categorical/cochran_q", columns=["resp_w0", "resp_w4", "resp_w12"]),
    _post("/api/categorical/cochran_armitage", ordinal_col="dose", event_col="ae",
          level_order=["Low", "Mid", "High"]),
    # association and agreement
    _post("/api/stats/correlation_pair", var1="age", var2="sbp", method="spearman"),
    _post("/api/stats/correlation_matrix", variables=["age", "sbp", "bmi"], method="kendall"),
    {"method": "GET", "url": "/api/stats/0b9f2c4e-1234-4abc-9def-0123456789ab/correlation?method=spearman"},
    _post("/api/stats/icc", rater1_col="reader_a", rater2_col="reader_b"),
    _post("/api/reliability/cronbach", items=["q1", "q2", "q3", "q4"]),
    # models
    _post("/api/models/linear", outcome="sbp", predictors=["age", "sex"], robust_se=True,
          interactions=[["age", "sex"]]),
    _post("/api/models/logistic", outcome="dm", predictors=["age", "bmi"],
          scale_factors={"age": 10}),
    _post("/api/models/poisson", outcome="visits", predictors=["age"], robust_se=True),
    _post("/api/models/negbinom", outcome="visits", predictors=["age", "sex"]),
    _post("/api/models/gamma", outcome="cost", predictors=["age"], link="log"),
    _post("/api/models/survival/cox", duration_col="t", event_col="died",
          predictors=["age", "arm"], imputation="mice"),
    _post("/api/models/survival/km", duration_col="t", event_col="died", group_col="arm",
          survival_times=[365, 1825], pairwise=True, pairwise_correction="bh"),
    _post("/api/survival_advanced/rmst", duration_col="t", event_col="died", group_col="arm", tau=1825),
    # diagnostic accuracy, causal, evidence synthesis
    _post("/api/stats/roc", score_column="troponin", outcome_column="mi", direction="higher"),
    _post("/api/stats/roc_compare", score_column_1="troponin", score_column_2="ckmb",
          outcome_column="mi"),
    _post("/api/models/psm", treatment_col="statin", covariates=["age", "sex", "ldl"],
          outcome_col="mace", caliper=0.2, ratio=1),
    _post("/api/models/iptw", treatment_col="statin", covariates=["age", "sex"],
          estimand="ate", stabilize=True, outcome_type="survival",
          survival_duration_col="t", survival_event_col="died", weight_truncation="percentile"),
    {"method": "POST", "url": "/api/meta/analyze", "body": {
        "measure": "OR", "tau2_method": "DL", "studies": [
            {"label": "Trial A", "e1": 12, "n1": 100, "e2": 20, "n2": 98},
            {"label": "Trial B", "e1": 0, "n1": 40, "e2": 3, "n2": 41},
            {"label": "Trial C", "effect": 0.8, "ci_low": 0.6, "ci_high": 1.07},
            {"label": "Trial D", "effect": 0.7, "se": 0.2},
        ]}},
]


def _by_url(fragment: str, **match) -> dict:
    """The REQUESTS entry for an endpoint (and body values, when several)."""
    for req in REQUESTS:
        body = req.get("body") or {}
        if req["url"].endswith(fragment) and all(body.get(k) == v for k, v in match.items()):
            return translate("x", {}, req)
    raise KeyError(fragment)


@pytest.mark.parametrize("req", REQUESTS, ids=lambda r: r["url"].rsplit("/", 2)[-1])
def test_every_endpoint_template_generates_valid_code(req):
    out = translate("anything", {}, req)
    assert out is not None, req["url"]
    ast.parse(out["python"])
    assert _parses_r(out["r"]), out["r"]
    assert "provenance line" in out["python"] and "provenance line" in out["r"]
    assert "\u2014" not in out["python"] + out["r"] + out["title"]


def test_every_endpoint_template_has_a_request_here():
    # A template added without a request above would never be parsed here.
    assert {_endpoint_key(r["url"]) for r in REQUESTS} == set(ENDPOINTS)


@pytest.mark.parametrize("path", sorted(ENDPOINTS))
def test_every_endpoint_template_survives_a_bare_body(path):
    # Optional fields absent (no group, no predictors, no studies): the code
    # must still parse, never a KeyError or a dangling formula.
    out = translate("x", {}, {"method": "POST", "url": path, "body": {"session_id": "s"}})
    assert out is not None, path
    ast.parse(out["python"])
    assert _parses_r(out["r"]), out["r"]


def test_awkward_column_names_are_quoted_in_both_languages():
    out = translate("x", {}, _post("/api/models/linear", outcome="systolic bp",
                                   predictors=["age (y)", 'say "hi"']))
    assert "Q('systolic bp')" in out["python"]
    assert "`systolic bp` ~ `age (y)`" in out["r"]
    ast.parse(out["python"])
    assert _parses_r(out["r"]), out["r"]


def test_ttest_syntax_follows_the_variance_rule_that_was_requested():
    auto = _by_url("/ttest", method="auto")
    assert "stats.levene(a, b).pvalue >= 0.05" in auto["python"]
    assert 'var.equal = lev[1, "Pr(>F)"] >= 0.05' in auto["r"]
    welch = translate("x", {}, _post("/api/stats/ttest", column="y", group_col="g", method="welch"))
    assert "equal_var=False" in welch["python"] and "var.equal = FALSE" in welch["r"]
    legacy = translate("x", {}, _post("/api/stats/ttest", column="y", group_col="g", equal_var=True))
    assert "var.equal = TRUE" in legacy["r"]
    one = _by_url("/ttest", mu=120)
    assert "popmean=120" in one["python"] and "mu = 120" in one["r"]


def test_rank_tests_pin_r_to_scipys_p_value_method():
    assert "exact = FALSE, correct = TRUE" in _by_url("/mannwhitney")["r"]
    assert "paired = TRUE, correct = FALSE" in _by_url("/wilcoxon_signed_rank")["r"]
    assert "exact = FALSE" in _by_url("/jonckheere_terpstra")["r"]
    assert 'method = "spearman", exact = FALSE' in _by_url("/correlation_pair")["r"]
    assert 'method = "kendall", exact = FALSE' in _by_url("/correlation_matrix")["r"]


def test_crosstab_syntax_reproduces_ustats_small_sample_switches():
    chi = _by_url("/chisquare")
    assert "fisher_exact" in chi["python"] and "PermutationMethod(n_resamples=5000)" in chi["python"]
    assert "simulate.p.value = TRUE, B = 5000" in chi["r"]
    mc = _by_url("/mcnemar")
    assert "< 25" in mc["python"] and "binom.test" in mc["r"]


def test_cochran_armitage_uses_the_stated_order_and_an_exact_r_match():
    r = _by_url("/cochran_armitage")["r"]
    assert 'lv <- c("Low", "Mid", "High")' in r
    assert "prop.trend.test" in r and "seq_along(lv) - 1" in r
    assert "sqrt((N - 1) / N)" in _by_url("/cochran_armitage")["python"]


def test_model_syntax_matches_ustats_intervals_and_robust_errors():
    logit = _by_url("/logistic")
    assert "I(age / 10)" in logit["python"] and "I(age / 10)" in logit["r"]
    assert "confint.default(fit)" in logit["r"]
    pois = _by_url("/poisson")
    assert 'cov_type="HC0"' in pois["python"] and 'type = "HC0"' in pois["r"]
    lin = _by_url("/linear")
    assert 'cov_type="HC3", use_t=True' in lin["python"] and 'type = "HC3"' in lin["r"]
    assert "age:sex" in lin["r"]
    gam = _by_url("/gamma")
    assert "use_t=True" in gam["python"] and "df = df.residual(fit)" in gam["r"]
    assert "observed information" in _by_url("/negbinom")["r"]


def test_survival_syntax_pins_the_lifelines_defaults():
    km = _by_url("/km")
    assert 'conf.type = "log-log"' in km["r"] and 'p.adjust.method = "BH"' in km["r"]
    assert "c(365, 1825)" in km["r"]
    cox = _by_url("/cox")
    assert 'transform = "rank"' in cox["r"] and "pooled 5 imputations" in cox["r"]
    rmst = _by_url("/rmst")
    assert "arm = as.integer(pair$arm == lv[p[1]])" in rmst["r"] and "tau = 1825" in rmst["r"]


def test_roc_psm_iptw_and_meta_syntax():
    roc = _by_url("/roc")
    assert 'direction = "<"' in roc["r"] and 'ci.auc(r, method = "delong")' in roc["r"]
    assert 'roc.test(r1, r2, method = "delong", paired = TRUE)' in _by_url("/roc_compare")["r"]
    psm = _by_url("/psm")["r"]
    assert 'link = "linear.logit"' in psm and "caliper = 0.2" in psm and "clogit(" in psm
    iptw = _by_url("/iptw")["r"]
    assert 'estimand = "ATE", stabilize = TRUE' in iptw and "robust = TRUE" in iptw
    assert "quantile(d$w, c(0.01, 0.99))" in iptw
    meta = _by_url("/meta/analyze")
    assert 'add = 0.5, to = "only0"' in meta["r"] and 'method = "DL"' in meta["r"]
    assert 'method_re="dl"' in meta["python"] and "zero_correction=0.5" in meta["python"]


def test_request_wins_over_panel_params():
    out = translate("models", {"model": "linear", "outcome": "y", "predictors": ["x"]}, REQUESTS[2])
    assert "logistf" in out["r"]


def test_fine_gray_syntax_separates_grays_test_from_what_ustat_reports():
    r = translate("x", {}, REQUESTS[0])["r"]
    assert "cif$Tests" in r
    assert "survdiff(Surv(time, status == 1) ~ arm" in r
    assert "NOT cumulative incidence" in translate("x", {}, REQUESTS[0])["python"]


def test_ordinal_syntax_carries_the_category_order():
    out = translate("x", {}, REQUESTS[1])
    assert 'levels = c("Poor", "Fair", "Good")' in out["r"]
    assert "['Poor', 'Fair', 'Good']" in out["python"]


def test_firth_syntax_asks_logistf_for_the_intervals_ustat_reports():
    assert "pl = FALSE" in translate("x", {}, REQUESTS[2])["r"]


def test_session_segment_in_a_get_url_is_normalised():
    assert _endpoint_key("/api/stats/0b9f2c4e-1234-4abc-9def-0123456789ab/descriptive?column=a") == "/api/stats/{sid}/descriptive"
    assert _endpoint_key("/api/stats/{sid}/descriptive") == "/api/stats/{sid}/descriptive"


def test_endpoint_accepts_a_recorded_request():
    r = client.post("/api/project/syntax", json={"panel": "unknown", "params": {}, "request": REQUESTS[1]})
    assert r.status_code == 200, r.text
    assert "polr" in r.json()["r"]
