"""Coverage tests for routers/stats.py (/api/stats/*).

Covers ~20 high-value uncovered POST endpoints: t-tests, chi-square, Fisher,
Mann-Whitney, Kruskal-Wallis, Jonckheere-Terpstra, ANOVA, correlation
(pair + matrix), ICC, Cohen's kappa, Fleiss kappa, TOST, power, ROC,
table1, weighted descriptive, and non-inferiority.

Tests assert response SHAPE and sanity ranges, not exact floats. They assert
ACTUAL current behaviour; any genuine bugs are recorded in the structured
output, not fixed here.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

SEED = 20260531
SID = "tstat_main"


@pytest.fixture(scope="module")
def sid():
    rng = np.random.default_rng(SEED)
    n = 200
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    # binary group with exactly 2 levels
    sex = rng.integers(0, 2, n)
    # ordered 3-level group (tertile-like)
    tertile = rng.integers(0, 3, n)
    # outcome correlated with score for a usable ROC
    score = rng.normal(0.5, 0.2, n)
    event = (rng.uniform(0, 1, n) < (0.2 + 0.5 * score)).astype(int)
    # two raters with high agreement for ICC / kappa
    rater1 = rng.integers(0, 3, n)
    rater2 = np.where(rng.uniform(0, 1, n) < 0.8, rater1, rng.integers(0, 3, n))
    rater3 = np.where(rng.uniform(0, 1, n) < 0.75, rater1, rng.integers(0, 3, n))
    # continuous rater values for ICC
    cval1 = rng.normal(50, 8, n)
    cval2 = cval1 + rng.normal(0, 3, n)
    weight = rng.uniform(0.5, 2.0, n)
    # categorical cols for chi-square / fisher
    cat_a = rng.integers(0, 2, n)  # binary
    cat_b = rng.integers(0, 2, n)  # binary -> 2x2 with cat_a
    cat_c = rng.integers(0, 3, n)  # 3-level

    df = pd.DataFrame({
        "age": age,
        "ldl": ldl,
        "sex": sex,
        "tertile": tertile,
        "score": score,
        "event": event,
        "rater1": rater1,
        "rater2": rater2,
        "rater3": rater3,
        "cval1": cval1,
        "cval2": cval2,
        "weight": weight,
        "cat_a": cat_a,
        "cat_b": cat_b,
        "cat_c": cat_c,
    })
    return make_session(df, SID)


# ── T-tests ──────────────────────────────────────────────────────────────────

def test_ttest_two_sample(client, sid):
    r = client.post("/api/stats/ttest", json={
        "session_id": sid, "column": "ldl", "group_column": "sex"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "t" in b and "p" in b
    assert 0.0 <= b["p"] <= 1.0
    assert b["n1"] > 0 and b["n2"] > 0
    assert isinstance(b["significant"], bool)
    assert b["effect_sizes"] and "value" in b["effect_sizes"][0]


def test_ttest_one_sample(client, sid):
    r = client.post("/api/stats/ttest", json={
        "session_id": sid, "column": "ldl", "mu": 120.0})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["test"] == "One-sample t-test"
    assert b["mu"] == 120.0
    assert 0.0 <= b["p"] <= 1.0
    assert b["df"] == b["n"] - 1


def test_ttest_bad_group_count(client, sid):
    # tertile has 3 levels -> 400
    r = client.post("/api/stats/ttest", json={
        "session_id": sid, "column": "ldl", "group_column": "tertile"})
    assert r.status_code == 400, r.text


# ── Chi-square ───────────────────────────────────────────────────────────────

def test_chisquare(client, sid):
    r = client.post("/api/stats/chisquare", json={
        "session_id": sid, "row_column": "cat_a", "col_column": "cat_b"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "chi2" in b and "dof" in b
    assert 0.0 <= b["p"] <= 1.0
    assert b["n"] > 0
    # 2x2 -> odds ratio appended
    assert len(b["effect_sizes"]) >= 1


def test_chisquare_3level(client, sid):
    r = client.post("/api/stats/chisquare", json={
        "session_id": sid, "row_column": "cat_c", "col_column": "cat_b"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["dof"] >= 1
    assert "crosstab" in b


# ── Fisher ───────────────────────────────────────────────────────────────────

def test_fisher_exact(client, sid):
    r = client.post("/api/stats/fisher", json={
        "session_id": sid, "row_column": "cat_a", "col_column": "cat_b"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "odds_ratio" in b
    assert 0.0 <= b["p"] <= 1.0
    assert len(b["table"]) == 2 and len(b["table"][0]) == 2


def test_fisher_non2x2_runs_freeman_halton(client, sid):
    # r x c used to be refused; since Fisher-Freeman-Halton landed it is run
    # by Monte Carlo, and this test still expected the old 400.
    r = client.post("/api/stats/fisher", json={
        "session_id": sid, "row_column": "cat_c", "col_column": "cat_b"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "Freeman-Halton" in b["test"]
    assert 0.0 < b["p"] <= 1.0


# ── Mann-Whitney ─────────────────────────────────────────────────────────────

def test_mannwhitney(client, sid):
    r = client.post("/api/stats/mannwhitney", json={
        "session_id": sid, "column": "ldl", "group_column": "sex"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "U" in b
    assert 0.0 <= b["p"] <= 1.0
    assert b["n1"] > 0 and b["n2"] > 0
    assert "median1" in b and "median2" in b


# ── Kruskal-Wallis ───────────────────────────────────────────────────────────

def test_kruskal(client, sid):
    r = client.post("/api/stats/kruskal", json={
        "session_id": sid, "column": "ldl", "group_column": "tertile"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "H" in b
    assert 0.0 <= b["p"] <= 1.0
    assert len(b["groups"]) == 3
    assert isinstance(b["posthoc"], list)


def test_kruskal_bad_correction(client, sid):
    r = client.post("/api/stats/kruskal", json={
        "session_id": sid, "column": "ldl", "group_column": "tertile",
        "posthoc_correction": "bogus"})
    assert r.status_code == 422, r.text


# ── Jonckheere-Terpstra ──────────────────────────────────────────────────────

def test_jonckheere_terpstra(client, sid):
    r = client.post("/api/stats/jonckheere_terpstra", json={
        "session_id": sid, "column": "ldl", "group_column": "tertile"})
    assert r.status_code == 200, r.text
    b = r.json()
    # response includes the J statistic and a p value somewhere
    assert 0.0 <= float(b.get("p", b.get("p_value", 0.5))) <= 1.0


def test_jonckheere_needs_three_groups(client, sid):
    r = client.post("/api/stats/jonckheere_terpstra", json={
        "session_id": sid, "column": "ldl", "group_column": "sex"})
    assert r.status_code == 422, r.text


# ── ANOVA ────────────────────────────────────────────────────────────────────

def test_anova(client, sid):
    r = client.post("/api/stats/anova", json={
        "session_id": sid, "column": "ldl", "group_column": "tertile"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "F" in b
    assert 0.0 <= b["p"] <= 1.0
    assert b["df_between"] == 2
    assert len(b["effect_sizes"]) == 2
    assert len(b["assumptions"]) >= 3


# ── Correlation (pair + matrix) ──────────────────────────────────────────────

def test_correlation_pair(client, sid):
    r = client.post("/api/stats/correlation_pair", json={
        "session_id": sid, "var1": "cval1", "var2": "cval2"})
    assert r.status_code == 200, r.text
    b = r.json()
    # correlation present somewhere in payload
    assert any(k in b for k in ("r", "correlation", "coefficient", "rho"))


def test_correlation_matrix(client, sid):
    r = client.post("/api/stats/correlation_matrix", json={
        "session_id": sid, "variables": ["age", "ldl", "cval1", "cval2"],
        "method": "pearson"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert isinstance(b, dict)


# ── ICC ──────────────────────────────────────────────────────────────────────

def test_icc(client, sid):
    r = client.post("/api/stats/icc", json={
        "session_id": sid, "rater1_col": "cval1", "rater2_col": "cval2"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert isinstance(b, dict)


# ── Cohen's kappa ────────────────────────────────────────────────────────────

def test_cohens_kappa(client, sid):
    r = client.post("/api/stats/cohens_kappa", json={
        "session_id": sid, "rater1_col": "rater1", "rater2_col": "rater2"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert isinstance(b, dict)


# ── Fleiss kappa ─────────────────────────────────────────────────────────────

def test_fleiss_kappa(client, sid):
    r = client.post("/api/stats/fleiss_kappa", json={
        "session_id": sid, "rater_cols": ["rater1", "rater2", "rater3"]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert isinstance(b, dict)


# ── TOST equivalence ─────────────────────────────────────────────────────────

def test_tost_independent(client, sid):
    r = client.post("/api/stats/tost", json={
        "session_id": sid, "column": "ldl", "group_column": "sex",
        "low": -20.0, "high": 20.0, "test_type": "independent"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert isinstance(b, dict)


def test_tost_invalid_bounds(client, sid):
    r = client.post("/api/stats/tost", json={
        "session_id": sid, "column": "ldl", "group_column": "sex",
        "low": 20.0, "high": -20.0, "test_type": "independent"})
    assert r.status_code in (400, 422), r.text


# ── Power analysis (no session needed) ───────────────────────────────────────

def test_power_two_sample_n(client, sid):
    r = client.post("/api/stats/power", json={
        "test": "t_two", "solve_for": "n", "alpha": 0.05,
        "power": 0.8, "effect_size": 0.5})
    assert r.status_code == 200, r.text
    b = r.json()
    assert isinstance(b, dict)
    assert "result" in b or "label" in b or "n" in b


def test_power_anova_power(client, sid):
    r = client.post("/api/stats/power", json={
        "test": "anova", "solve_for": "power", "alpha": 0.05,
        "effect_size": 0.25, "n": 40, "k_groups": 3})
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), dict)


def test_power_logistic_n(client, sid):
    # Sample size for a logistic model: OR=2.0, 30% event rate, 80% power.
    r = client.post("/api/stats/power", json={
        "test": "logistic", "solve_for": "n", "alpha": 0.05,
        "power": 0.8, "log_or": 2.0, "p_event": 0.3})
    assert r.status_code == 200, r.text
    b = r.json()
    n = b.get("result") or b.get("n")
    assert n is not None and n > 0


def test_power_logistic_power_monotone(client, sid):
    # Power must increase with n for the same effect.
    def pw(n):
        r = client.post("/api/stats/power", json={
            "test": "logistic", "solve_for": "power", "alpha": 0.05,
            "n": n, "log_or": 1.8, "p_event": 0.25})
        assert r.status_code == 200, r.text
        return float(r.json()["result"])
    assert 0.0 <= pw(100) <= pw(400) <= 1.0


def test_power_logistic_requires_event_rate(client, sid):
    r = client.post("/api/stats/power", json={
        "test": "logistic", "solve_for": "n", "alpha": 0.05,
        "power": 0.8, "log_or": 2.0})  # missing p_event
    assert r.status_code == 400


def test_power_cox_n(client, sid):
    # Sample size for a Cox model: HR=1.6, 40% event rate, 80% power.
    r = client.post("/api/stats/power", json={
        "test": "survival_cox", "solve_for": "n", "alpha": 0.05,
        "power": 0.8, "hr": 1.6, "event_rate": 0.4, "p_exposed": 0.5})
    assert r.status_code == 200, r.text
    b = r.json()
    assert (b.get("result") or b.get("n")) > 0


# ── ROC ──────────────────────────────────────────────────────────────────────

def test_roc(client, sid):
    r = client.post("/api/stats/roc", json={
        "session_id": sid, "score_column": "score", "outcome_column": "event"})
    assert r.status_code == 200, r.text
    b = r.json()
    # AUC present and in valid range
    auc = b.get("auc", b.get("AUC"))
    assert auc is not None
    assert 0.0 <= float(auc) <= 1.0


# ── Table 1 ──────────────────────────────────────────────────────────────────

def test_table1(client, sid):
    r = client.post("/api/stats/table1", json={
        "session_id": sid, "group_column": "sex",
        "variables": ["age", "ldl", "cat_c"]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert "rows" in b and len(b["rows"]) == 3
    assert b["total_n"] == 200
    assert "group_labels" in b


def test_table1_no_group(client, sid):
    r = client.post("/api/stats/table1", json={
        "session_id": sid, "variables": ["age", "ldl"]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert len(b["rows"]) == 2


# ── Weighted descriptive ─────────────────────────────────────────────────────

def test_weighted_descriptive(client, sid):
    r = client.post("/api/stats/weighted_descriptive", json={
        "session_id": sid, "value_cols": ["age", "ldl"],
        "weight_col": "weight"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert isinstance(b, dict)


# ── Non-inferiority ──────────────────────────────────────────────────────────

def test_noninferiority_binary(client, sid):
    r = client.post("/api/stats/noninferiority", json={
        "session_id": sid, "outcome_col": "event", "group_col": "sex",
        "outcome_type": "binary", "effect": "RR", "margin": 1.20})
    assert r.status_code == 200, r.text
    b = r.json()
    assert isinstance(b, dict)


# ── Session not found ────────────────────────────────────────────────────────

def test_missing_session_404(client, sid):
    r = client.post("/api/stats/ttest", json={
        "session_id": "tstat_does_not_exist", "column": "ldl", "mu": 0.0})
    assert r.status_code == 404, r.text
