"""Tests for the Bayesian statistics endpoint (POST /api/bayesian)."""
import numpy as np
import pytest
import pandas as pd
from conftest import make_session


def _bayes(client, sid, **kwargs):
    body = {"session_id": sid}
    body.update(kwargs)
    return client.post("/api/bayesian", json=body)


def _assert_common_shape(d):
    assert "analysis" in d
    assert "statistic_value" in d
    assert "bf10" in d and "bf01" in d
    assert d["bf10"] > 0 and d["bf01"] >= 0
    assert "interpretation" in d
    assert "r_code" in d
    assert "plot_coords" in d


# ── ttest_one ─────────────────────────────────────────────────────────────────

def test_ttest_one_happy_path(client):
    np.random.seed(1)
    x = np.random.normal(loc=5.0, scale=1.0, size=40)
    df = pd.DataFrame({"score": x})
    sid = make_session(df, "bayes_ttest_one")
    r = _bayes(client, sid, analysis_type="ttest_one", outcome="score", mu=0.0)
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["n"] == 40
    assert d["df"] == 39
    # Clearly one-sided: mean far from mu=0 -> strong evidence for H1
    assert d["bf10"] > 10


def test_ttest_one_no_effect_favors_null(client):
    np.random.seed(2)
    x = np.random.normal(loc=0.0, scale=1.0, size=200)
    df = pd.DataFrame({"score": x})
    sid = make_session(df, "bayes_ttest_one_null")
    r = _bayes(client, sid, analysis_type="ttest_one", outcome="score", mu=0.0)
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    # Data centered at mu -> evidence should favor null (bf10 < 1)
    assert d["bf10"] < 1


def test_ttest_one_too_few_rows(client):
    df = pd.DataFrame({"score": [1.0, 2.0]})
    sid = make_session(df, "bayes_ttest_one_small")
    r = _bayes(client, sid, analysis_type="ttest_one", outcome="score", mu=0.0)
    assert r.status_code == 400, r.text


# ── ttest_paired ──────────────────────────────────────────────────────────────

def test_ttest_paired_happy_path(client):
    np.random.seed(3)
    n = 40
    before = np.random.normal(loc=10.0, scale=2.0, size=n)
    after = before + 3.0 + np.random.normal(loc=0.0, scale=0.5, size=n)
    df = pd.DataFrame({"before": before, "after": after})
    sid = make_session(df, "bayes_ttest_paired")
    r = _bayes(client, sid, analysis_type="ttest_paired", outcome="before", predictor="after")
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["n"] == n
    assert d["bf10"] > 10  # clear paired difference


def test_ttest_paired_missing_predictor(client):
    df = pd.DataFrame({"before": [1.0, 2.0, 3.0, 4.0], "after": [1.5, 2.5, 3.5, 4.5]})
    sid = make_session(df, "bayes_ttest_paired_missing")
    r = _bayes(client, sid, analysis_type="ttest_paired", outcome="before")
    assert r.status_code == 400, r.text


def test_ttest_paired_too_few_rows(client):
    df = pd.DataFrame({"before": [1.0, 2.0], "after": [1.5, 2.5]})
    sid = make_session(df, "bayes_ttest_paired_small")
    r = _bayes(client, sid, analysis_type="ttest_paired", outcome="before", predictor="after")
    assert r.status_code == 400, r.text


# ── ttest_ind ─────────────────────────────────────────────────────────────────

def test_ttest_ind_happy_path(client):
    np.random.seed(4)
    n = 30
    g1 = np.random.normal(loc=5.0, scale=1.0, size=n)
    g2 = np.random.normal(loc=10.0, scale=1.0, size=n)
    df = pd.DataFrame({
        "value": np.concatenate([g1, g2]),
        "group": ["A"] * n + ["B"] * n,
    })
    sid = make_session(df, "bayes_ttest_ind")
    r = _bayes(client, sid, analysis_type="ttest_ind", outcome="value", predictor="group")
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["n"] == 2 * n
    assert d["bf10"] > 10  # clear group difference


def test_ttest_ind_missing_predictor(client):
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "group": ["A", "A", "B", "B"]})
    sid = make_session(df, "bayes_ttest_ind_missing")
    r = _bayes(client, sid, analysis_type="ttest_ind", outcome="value")
    assert r.status_code == 400, r.text


def test_ttest_ind_wrong_group_count(client):
    df = pd.DataFrame({
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "group": ["A", "B", "C", "A", "B", "C"],
    })
    sid = make_session(df, "bayes_ttest_ind_3groups")
    r = _bayes(client, sid, analysis_type="ttest_ind", outcome="value", predictor="group")
    assert r.status_code == 400, r.text


# ── correlation ───────────────────────────────────────────────────────────────

def test_correlation_happy_path(client):
    np.random.seed(5)
    n = 60
    x = np.random.normal(size=n)
    y = 0.9 * x + np.random.normal(scale=0.2, size=n)
    df = pd.DataFrame({"x": x, "y": y})
    sid = make_session(df, "bayes_corr")
    r = _bayes(client, sid, analysis_type="correlation", outcome="x", predictor="y")
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["n"] == n
    assert d["statistic_value"] > 0.7
    assert d["bf10"] > 10  # strong correlation -> strong evidence


def test_correlation_missing_predictor(client):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 3.0, 4.0, 5.0]})
    sid = make_session(df, "bayes_corr_missing")
    r = _bayes(client, sid, analysis_type="correlation", outcome="x")
    assert r.status_code == 400, r.text


def test_correlation_too_few_rows(client):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0]})
    sid = make_session(df, "bayes_corr_small")
    r = _bayes(client, sid, analysis_type="correlation", outcome="x", predictor="y")
    assert r.status_code == 400, r.text


# ── regression ────────────────────────────────────────────────────────────────

def test_regression_happy_path(client):
    np.random.seed(6)
    n = 50
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    y = 3.0 * x1 - 2.0 * x2 + np.random.normal(scale=0.3, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    sid = make_session(df, "bayes_regression")
    r = _bayes(client, sid, analysis_type="regression", outcome="y", predictors=["x1", "x2"])
    assert r.status_code == 200, r.text
    d = r.json()
    _assert_common_shape(d)
    assert d["n"] == n
    assert d["statistic_value"] > 0.8  # high R^2 given strong linear relation
    assert d["bf10"] > 10


def test_regression_missing_predictors(client):
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "x1": [1, 2, 3, 4, 5, 6, 7]})
    sid = make_session(df, "bayes_regression_missing")
    r = _bayes(client, sid, analysis_type="regression", outcome="y")
    assert r.status_code == 400, r.text


def test_regression_too_few_rows(client):
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x1": [1.0, 2.0, 3.0]})
    sid = make_session(df, "bayes_regression_small")
    r = _bayes(client, sid, analysis_type="regression", outcome="y", predictors=["x1"])
    assert r.status_code == 400, r.text


# ── Cross-cutting invalid input ──────────────────────────────────────────────

def test_unknown_analysis_type(client):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    sid = make_session(df, "bayes_unknown_type")
    r = _bayes(client, sid, analysis_type="not_a_real_type", outcome="x")
    assert r.status_code == 422, r.text


def test_missing_column(client):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    sid = make_session(df, "bayes_missing_col")
    r = _bayes(client, sid, analysis_type="ttest_one", outcome="does_not_exist")
    assert r.status_code == 400, r.text


def test_session_not_found(client):
    r = _bayes(client, "nonexistent_session_id", analysis_type="ttest_one", outcome="x")
    assert r.status_code == 404, r.text


def test_tiny_bf01_is_not_rounded_to_zero(client):
    # A strong effect gives BF10 in the millions; BF01 = 1/BF10 was rounded
    # to four decimals and came back as exactly 0.0.
    rng = np.random.default_rng(11)
    n = 150
    df = pd.DataFrame({
        "value": np.r_[rng.normal(0, 1, n), rng.normal(0.75, 1, n)],
        "group": ["A"] * n + ["B"] * n,
    })
    sid = make_session(df, "bayes_tiny_bf01")
    r = _bayes(client, sid, analysis_type="ttest_ind", outcome="value", predictor="group")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["bf10"] > 1e4
    assert 0 < d["bf01"] < 1e-4
    assert d["bf01"] * d["bf10"] == pytest.approx(1.0, rel=1e-3)
