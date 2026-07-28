"""The reported df must belong to the test that produced t and p.

`ttest_ind(equal_var=False)` was used whenever Levene failed, but the response
hardcoded `df = n1 + n2 - 2` — the pooled value. A Welch t reported against the
pooled df understates the tail, so a manuscript would carry a df that does not
correspond to its own p-value.

The same helper also claimed "using Welch correction" for one-way ANOVA, whose
omnibus F is scipy's classic equal-variance `f_oneway`; only the post-hoc
switches to Games-Howell.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from scipy import stats as sp

SEED = 20260725


def _levene_violating_pair():
    """Two groups with clearly unequal variances, so the Welch path is taken."""
    rng = np.random.default_rng(SEED)
    g0 = np.round(rng.normal(52.59, 9.0, 164), 4)
    g1 = np.round(rng.normal(56.63, 3.0, 139), 4)
    return g0, g1


@pytest.fixture(scope="module")
def welch_pair():
    return _levene_violating_pair()


@pytest.fixture(scope="module")
def welch_result(welch_pair):
    from fastapi.testclient import TestClient
    from main import app

    g0, g1 = welch_pair
    df = pd.DataFrame({
        "age": np.concatenate([g0, g1]),
        "heart_disease": [0] * len(g0) + [1] * len(g1),
    })
    sid = make_session(df, "welch_df_session")
    r = TestClient(app).post("/api/stats/ttest", json={
        "session_id": sid, "column": "age", "group_column": "heart_disease",
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_welch_path_was_actually_taken(welch_result):
    assert "Welch" in welch_result["test"]
    assert welch_result["df_method"] == "welch_satterthwaite"


def test_reported_df_matches_welch_satterthwaite(welch_result, welch_pair):
    g0, g1 = welch_pair
    n1, n2 = len(g0), len(g1)
    a, b = g0.var(ddof=1) / n1, g1.var(ddof=1) / n2
    expected = (a + b) ** 2 / ((a ** 2) / (n1 - 1) + (b ** 2) / (n2 - 1))
    assert welch_result["df"] == pytest.approx(expected, rel=1e-12)
    # The whole point: it must not be the pooled value.
    assert welch_result["df"] != pytest.approx(n1 + n2 - 2)


def test_t_and_p_agree_with_scipy_welch(welch_result, welch_pair):
    g0, g1 = welch_pair
    t_w, p_w = sp.ttest_ind(g0, g1, equal_var=False)
    assert welch_result["t"] == pytest.approx(t_w, rel=1e-12)
    assert welch_result["p"] == pytest.approx(p_w, rel=1e-12)


def test_p_is_reproducible_from_the_reported_t_and_df(welch_result):
    """The triple must be internally consistent, which is what broke before."""
    recomputed = float(2 * sp.t.sf(abs(welch_result["t"]), welch_result["df"]))
    assert recomputed == pytest.approx(welch_result["p"], rel=1e-9)


def test_result_text_rounds_the_fractional_df(welch_result):
    assert f"t({welch_result['df']:.2f})" in welch_result["result_text"]


def test_pooled_path_reports_pooled_df():
    from fastapi.testclient import TestClient
    from main import app

    rng = np.random.default_rng(SEED + 1)
    g0 = np.round(rng.normal(10, 2, 80), 4)
    g1 = np.round(rng.normal(10.5, 2, 80), 4)   # equal variances → no Welch
    df = pd.DataFrame({"y": np.concatenate([g0, g1]), "g": [0] * 80 + [1] * 80})
    sid = make_session(df, "pooled_df_session")
    r = TestClient(app).post("/api/stats/ttest", json={
        "session_id": sid, "column": "y", "group_column": "g",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["df_method"] == "pooled"
    assert body["df"] == pytest.approx(len(g0) + len(g1) - 2)


def test_anova_switches_the_omnibus_when_levene_fails():
    """This used to assert the opposite — that the omnibus stayed the classic
    equal-variance F and said so. Disclosing the limitation was honest, but a
    robust Games-Howell post-hoc hanging off a non-robust omnibus is still the
    wrong test, so Levene now decides the omnibus as well."""
    from fastapi.testclient import TestClient
    from main import app
    from statsmodels.stats.oneway import anova_oneway

    rng = np.random.default_rng(SEED + 2)
    groups = [rng.normal(10, 1, 60), rng.normal(11, 5, 60), rng.normal(12, 12, 60)]
    frame = pd.DataFrame({
        "y": np.concatenate(groups),
        "g": ["A"] * 60 + ["B"] * 60 + ["C"] * 60,
    })
    sid = make_session(frame, "anova_levene_session")
    r = TestClient(app).post("/api/stats/anova", json={
        "session_id": sid, "column": "y", "group_column": "g",
    })
    assert r.status_code == 200, r.text
    out = r.json()
    levene = next(a for a in out["assumptions"] if "Levene" in a["name"])
    assert levene["met"] is False
    assert out["variance_assumption"] == "welch"
    assert "Welch" in out["test"]

    expected = anova_oneway(groups, use_var="unequal")
    assert out["F"] == pytest.approx(float(expected.statistic), rel=1e-9)
    assert out["p"] == pytest.approx(float(expected.pvalue), rel=1e-9)
    # Welch's denominator df is fractional and smaller than the pooled n - k.
    assert out["df_denominator"] < out["df_within"]


def test_anova_keeps_the_classic_f_when_variances_are_equal():
    from fastapi.testclient import TestClient
    from main import app
    from scipy import stats as sp

    rng = np.random.default_rng(SEED + 3)
    groups = [rng.normal(10, 2, 50), rng.normal(11, 2, 50), rng.normal(12, 2, 50)]
    frame = pd.DataFrame({
        "y": np.concatenate(groups),
        "g": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
    })
    sid = make_session(frame, "anova_equalvar_session")
    r = TestClient(app).post("/api/stats/anova", json={
        "session_id": sid, "column": "y", "group_column": "g",
    })
    out = r.json()
    assert out["variance_assumption"] == "equal"
    assert out["test"] == "One-way ANOVA"
    assert out["F"] == pytest.approx(float(sp.f_oneway(*groups).statistic), rel=1e-9)


def test_ttest_still_states_welch_was_applied(welch_result):
    levene = next(a for a in welch_result["assumptions"] if "Levene" in a["name"])
    assert levene["met"] is False
    assert "Welch correction applied" in levene["detail"]


# ── Variance-assumption selection and the R snippet ──────────────────────────


def _post(payload):
    from fastapi.testclient import TestClient
    from main import app
    r = TestClient(app).post("/api/stats/ttest", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def sid_unequal(welch_pair):
    g0, g1 = welch_pair
    frame = pd.DataFrame({
        "age": np.concatenate([g0, g1]),
        "heart_disease": [0] * len(g0) + [1] * len(g1),
    })
    return make_session(frame, "ttest_variance_choice")


def test_equal_var_false_forces_welch(sid_unequal, welch_pair):
    g0, g1 = welch_pair
    body = _post({"session_id": sid_unequal, "column": "age",
                  "group_column": "heart_disease", "equal_var": False})
    assert body["variance_assumption"] == "welch"
    assert body["variance_assumption_selected_by"] == "request (equal_var)"
    assert body["t"] == pytest.approx(sp.ttest_ind(g0, g1, equal_var=False)[0], rel=1e-12)


def test_equal_var_true_forces_student_even_when_levene_fails(sid_unequal, welch_pair):
    """The parameter used to be accepted and then silently ignored."""
    g0, g1 = welch_pair
    body = _post({"session_id": sid_unequal, "column": "age",
                  "group_column": "heart_disease", "equal_var": True})
    levene = next(a for a in body["assumptions"] if "Levene" in a["name"])
    assert levene["met"] is False, "fixture must violate equal variances"
    assert body["variance_assumption"] == "student"
    assert body["df"] == pytest.approx(len(g0) + len(g1) - 2)
    assert body["t"] == pytest.approx(sp.ttest_ind(g0, g1, equal_var=True)[0], rel=1e-12)


@pytest.mark.parametrize("method,expected", [("student", "student"), ("welch", "welch")])
def test_method_parameter_overrides_levene(sid_unequal, method, expected):
    body = _post({"session_id": sid_unequal, "column": "age",
                  "group_column": "heart_disease", "method": method})
    assert body["variance_assumption"] == expected
    assert body["variance_assumption_selected_by"] == "request (method)"


def test_method_beats_the_legacy_equal_var_alias(sid_unequal):
    body = _post({"session_id": sid_unequal, "column": "age", "group_column": "heart_disease",
                  "method": "welch", "equal_var": True})
    assert body["variance_assumption"] == "welch"


def test_default_is_still_levene_driven(sid_unequal):
    body = _post({"session_id": sid_unequal, "column": "age", "group_column": "heart_disease"})
    assert body["variance_assumption_selected_by"] == "auto (Levene)"


@pytest.mark.parametrize("payload,var_equal", [
    ({"method": "welch"}, "FALSE"),
    ({"method": "student"}, "TRUE"),
    ({"equal_var": False}, "FALSE"),
    ({"equal_var": True}, "TRUE"),
])
def test_r_snippet_matches_the_test_that_ran(sid_unequal, payload, var_equal):
    """var.equal = TRUE is Student's; emitting it for a Welch result is wrong."""
    body = _post({"session_id": sid_unequal, "column": "age",
                  "group_column": "heart_disease", **payload})
    assert f"var.equal = {var_equal}" in body["r_code"]
    assert (body["variance_assumption"] == "welch") == (var_equal == "FALSE")


def test_invalid_method_is_rejected(sid_unequal):
    from fastapi.testclient import TestClient
    from main import app
    r = TestClient(app).post("/api/stats/ttest", json={
        "session_id": sid_unequal, "column": "age",
        "group_column": "heart_disease", "method": "nonsense"})
    assert r.status_code == 422, r.text
