"""Regression tests for the Cox / RCS defects found in the model audit.

Each test names the wrong result it guards against.
"""

import asyncio

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from conftest import make_session
from lifelines import CoxPHFitter

from routers.models.cox import (
    CoxRCSRequest,
    CoxRequest,
    KMRequest,
    RCSRequest,
    SplineTerm,
    _encode_cox_predictor,
    _km_step_quantile,
    cox_rcs,
    cox_regression,
    kaplan_meier,
    rcs_regression,
)
from services.cox_encoding import encode_design
from services.rcs_basis import harrell_knots, rcs_basis

N = 400


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(50, 10, N)
    age = rng.normal(60, 8, N)
    bmi = rng.normal(27, 4, N)
    sex = rng.choice(["F", "M"], N)
    grp = rng.choice(["a", "b", "c"], N)
    lp = -1 + 0.03 * (x - 50) + 0.05 * (age - 60) - 0.08 * (bmi - 27) + 0.7 * (sex == "M")
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    t = rng.exponential(10 * np.exp(-lp)) + 0.01
    e = (t < 15).astype(int)
    return pd.DataFrame({
        "x": x, "age": age, "bmi": bmi, "sex": sex, "grp": grp,
        "y": y, "t": np.minimum(t, 15), "e": e,
    })


def _run(coro):
    return asyncio.run(coro)


def test_rcs_glm_covariate_summary_matches_the_fitted_coefficients(frame):
    """The summary was shifted one column: age showed bmi's coefficient."""
    sid = make_session(frame, "audit_cov_summary")
    res = _run(rcs_regression(RCSRequest(
        session_id=sid, predictor="x", outcome="y",
        covariates=["age", "bmi", "sex"], model_type="logistic",
    )))
    x = frame["x"].to_numpy()
    design = np.column_stack([
        np.ones(N), x, rcs_basis(x, harrell_knots(x, 4)),
        frame["age"], frame["bmi"], (frame["sex"] == "M").astype(float),
    ])
    truth = sm.Logit(frame["y"].to_numpy(), design).fit(disp=0).params[-3:]
    got = [row["coef"] for row in res["covariates_summary"]]
    assert [row["name"] for row in res["covariates_summary"]] == ["age", "bmi", "sex_M"]
    np.testing.assert_allclose(got, truth, atol=1e-4)


def test_rcs_drops_rows_with_a_missing_categorical_covariate(frame):
    """Missing sex was analysed as the reference level: n stayed 400."""
    df = frame.copy()
    df.loc[:99, "sex"] = np.nan
    sid = make_session(df, "audit_rcs_nan")
    res = _run(rcs_regression(RCSRequest(
        session_id=sid, predictor="x", outcome="y", covariates=["sex"], model_type="logistic",
    )))
    assert res["n"] == N - 100
    assert res["n_excluded"] == 100


def test_rcs_knots_keep_their_precision_on_a_small_scale(frame):
    """Knots near 0.01 were all reported as 0.01 after rounding to 2 decimals."""
    df = frame.assign(xs=frame["x"] / 5000)
    sid = make_session(df, "audit_rcs_scale")
    res = _run(rcs_regression(RCSRequest(session_id=sid, predictor="xs", outcome="y", model_type="logistic")))
    np.testing.assert_allclose(res["knots"], harrell_knots(df["xs"].to_numpy(), 4), rtol=1e-3)
    # The 5-knot logistic fit failed on this scale and left the AIC table.
    assert [row["n_knots"] for row in res["knot_comparison"]] == [3, 4, 5]


def test_rcs_declared_categorical_covariate_is_dummy_coded(frame):
    """Groups coded 1/2/3 with a categorical kind entered as one linear term."""
    from services import store

    df = frame.assign(stage=np.tile([1, 2, 3, 2], N // 4))
    sid = make_session(df, "audit_rcs_kind")
    store.set_kind_overrides(sid, {"stage": "categorical"})
    res = _run(rcs_regression(RCSRequest(
        session_id=sid, predictor="x", outcome="y", covariates=["stage"], model_type="logistic",
    )))
    assert res["covariates_used"] == ["stage_2", "stage_3"]


def test_encode_design_keeps_missing_categories_missing_and_maps_members():
    df = pd.DataFrame({
        "stage": ["i", "ii", None, "iii"],
        "stage_score": [1.0, 2.0, 3.0, 4.0],
    })
    design = encode_design(df, ["stage", "stage_score"])
    assert design.frame.loc[2, ["stage_ii", "stage_iii"]].isna().all()
    # "stage" must not pick up the unrelated "stage_score" column.
    assert design.columns_of("stage") == ["stage_ii", "stage_iii"]


def test_encode_cox_predictor_does_not_code_missing_as_reference():
    design, _ = _encode_cox_predictor(pd.Series(["a", "b", "c", None, np.nan, "b"]), "g")
    assert design.iloc[[3, 4]].isna().all().all()
    assert design.dropna().shape[0] == 4


def test_km_step_quantile_follows_the_step_curve():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0])
    surv = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    assert _km_step_quantile(times, surv, 0.5) == 3.0
    # Exactly on the level over [2, 3): R reports the midpoint.
    assert _km_step_quantile(times, surv, 0.6) == 2.5
    # Never reached: None, not the last time of the curve.
    assert _km_step_quantile(times, np.array([1.0, 0.95, 0.9, 0.9, 0.9, 0.9]), 0.5) is None


def test_km_logrank_ignores_rows_without_a_group(frame):
    """NaN groups counted as a fourth group, so p came from df = 3."""
    from lifelines.statistics import multivariate_logrank_test

    df = frame.copy()
    df.loc[:79, "grp"] = np.nan
    sid = make_session(df, "audit_km_nan")
    res = kaplan_meier(KMRequest(session_id=sid, duration_col="t", event_col="e", group_col="grp"))
    kept = df.dropna(subset=["grp"])
    truth = multivariate_logrank_test(kept["t"], kept["grp"], kept["e"])
    assert res["logrank"]["df"] == 2
    assert res["logrank"]["p"] == pytest.approx(truth.p_value, rel=1e-9)
    assert res["n_excluded"] == 80


def test_cox_mice_pools_log_hazard_ratios_with_rubins_rules(frame):
    """The pooled table held log(log HR); protective effects vanished."""
    df = frame.assign(trt=(frame["sex"] == "M").astype(int)).drop(columns=["sex", "grp"])
    lp = -0.8 * df["trt"] + 0.03 * (df["age"] - 60)
    rng = np.random.default_rng(7)
    df["t"] = rng.exponential(10 * np.exp(-lp)) + 0.01
    df["e"] = (rng.random(N) < 0.8).astype(int)
    complete = CoxPHFitter().fit(df[["t", "e", "age", "trt"]], "t", "e")
    df.loc[rng.choice(N, 60, replace=False), "age"] = np.nan
    sid = make_session(df, "audit_cox_mice")
    res = _run(cox_regression(CoxRequest(
        session_id=sid, duration_col="t", event_col="e", predictors=["age", "trt"], imputation="mice",
    )))
    rows = {row["variable"]: row for row in res["coefficients"]}
    assert res["pooled_from_imputations"] is True
    assert set(rows) == {"age", "trt"}
    for name in ("age", "trt"):
        assert rows[name]["log_hr"] == pytest.approx(complete.params_[name], abs=3 * complete.standard_errors_[name])
        assert rows[name]["hr_ci_low"] < rows[name]["hr"] < rows[name]["hr_ci_high"]
        assert rows[name]["se"] >= 0.9 * complete.standard_errors_[name]
    assert rows["trt"]["hr"] < 1


def test_cox_global_ph_test_is_the_joint_statistic(frame):
    """The global row was the sum of the per-term chi-squares."""
    df = frame.assign(bmi2=frame["bmi"] + 0.9 * (frame["age"] - 60))
    sid = make_session(df, "audit_cox_ph")
    res = _run(cox_regression(CoxRequest(
        session_id=sid, duration_col="t", event_col="e", predictors=["age", "bmi2"],
    )))
    ph = res["ph_test"]
    per_term_sum = sum(t["test_stat"] for t in ph["per_term"])
    assert ph["global"]["df"] == 2
    assert ph["global"]["chi2"] != pytest.approx(per_term_sum, rel=1e-3)

    # With a single term the joint statistic is that term's statistic.
    one = _run(cox_regression(CoxRequest(session_id=sid, duration_col="t", event_col="e", predictors=["age"])))
    assert one["ph_test"]["global"]["chi2"] == pytest.approx(one["ph_test"]["per_term"][0]["test_stat"], rel=1e-6)


def test_cox_counts_rows_lost_to_a_missing_category(frame):
    df = frame.copy()
    df.loc[:49, "grp"] = np.nan
    sid = make_session(df, "audit_cox_cat_nan")
    res = _run(cox_regression(CoxRequest(
        session_id=sid, duration_col="t", event_col="e", predictors=["age", "grp"], imputation="median",
    )))
    assert res["n_analyzed"] == N - 50
    assert res["n_excluded"] == 50


def test_cox_rcs_covariates_drop_missing_categories(frame):
    df = frame.copy()
    df.loc[:99, "sex"] = np.nan
    sid = make_session(df, "audit_coxrcs_nan")
    res = _run(cox_rcs(CoxRCSRequest(
        session_id=sid, duration_col="t", event_col="e",
        spline_terms=[SplineTerm(column="x")], covariates=["sex"],
    )))
    assert res["n"] == N - 100
