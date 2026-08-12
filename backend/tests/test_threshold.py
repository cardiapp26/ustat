"""Two-piecewise (threshold) regression.

The numbers below were taken from R: at a fixed breakpoint the hinge model is
ordinary regression, so `lm`/`glm` on the same basis must reproduce the slopes,
their standard errors and the log-likelihood exactly. The breakpoint itself is
checked against `segmented::segmented()`, which reaches it by Muggeo's
iterative method rather than by profiling a grid — agreement there is agreement
between two different algorithms, not a restatement of one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from routers.threshold import MIN_SIDE, _encode_covariates
from services import store


@pytest.fixture()
def kinked(client) -> str:
    """A real breakpoint at x = 40: slope +0.30 below, -0.10 above."""
    # Draw order matters: these are the exact rows the R reference was run on,
    # so a variable generated out of sequence would shift every number below.
    rng = np.random.default_rng(20260812)
    n = 400
    x = rng.uniform(0, 100, n)
    age = rng.normal(60, 12, n)
    sex = rng.integers(0, 2, n)
    site = rng.choice(["A", "B", "C"], n)
    y = (10 + 0.30 * x - 0.40 * np.clip(x - 40.0, 0, None)
         + 0.05 * age + 1.5 * sex + rng.normal(0, 2.5, n))
    eta = -2.0 + 0.06 * x - 0.09 * np.clip(x - 40.0, 0, None) + 0.02 * (age - 60) + 0.4 * sex
    event = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    df = pd.DataFrame({"x": x, "y": y, "event": event, "age": age, "sex": sex, "site": site})
    for col, idx in [("y", [5, 88]), ("age", [12]), ("x", [301]), ("sex", [200, 201])]:
        df.loc[idx, col] = np.nan
    store.save("thr", df)
    return "thr"


def _run(client, sid, **extra):
    body = {"session_id": sid, "outcome": "y", "exposure": "x",
            "outcome_kind": "continuous", "grid_n": 400, **extra}
    r = client.post("/api/threshold/analyze", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ── agreement with R ───────────────────────────────────────────────────────────


def test_it_finds_the_breakpoint_that_is_actually_there(client, kinked):
    # R: segmented(lm(y ~ x), seg.Z = ~x) → psi = 40.435 on these rows.
    out = _run(client, kinked)
    assert out["breakpoint"] == pytest.approx(40.4, abs=0.6)
    assert out["breakpoint_ci"]["low"] < 40.0 < out["breakpoint_ci"]["high"]


def test_the_two_slopes_match_r_at_the_reported_breakpoint(client, kinked):
    # R, with xa = pmax(x - k, 0) at uSTAT's k:
    #   coef(x) = 0.26168124, se = 0.013723095
    #   coef(x) + coef(xa) = -0.089448771, se = 0.0079502933
    out = _run(client, kinked)
    assert out["effect_below"]["beta"] == pytest.approx(0.26168124, rel=1e-6)
    assert out["effect_below"]["se"] == pytest.approx(0.013723095, rel=1e-6)
    assert out["effect_above"]["beta"] == pytest.approx(-0.089448771, rel=1e-6)
    assert out["effect_above"]["se"] == pytest.approx(0.0079502933, rel=1e-6)


def test_the_slope_above_uses_the_covariance_not_the_two_standard_errors(client, kinked):
    # SE(b1 + b2) needs Cov(b1, b2). Adding the two SEs in quadrature ignores
    # it and, since the pair is strongly negatively correlated here, reports an
    # interval roughly twice as wide as the truth.
    out = _run(client, kinked)
    naive = float(np.hypot(out["effect_below"]["se"], out["effect_difference"]["se"]))
    assert out["effect_above"]["se"] < 0.6 * naive


def test_a_linear_outcome_is_tested_against_t_not_z(client, kinked):
    # R reports p = 6.3295011e-54 for the change in slope; the normal
    # approximation gives 1.37e-73 — twenty orders of magnitude out, because a
    # linear model's coefficients are t-distributed on the residual df.
    out = _run(client, kinked)
    assert out["effect_difference"]["p"] == pytest.approx(6.3295011e-54, rel=1e-3)


def test_the_log_likelihoods_match_r(client, kinked):
    # R: logLik(lm(y ~ x)) = -1067.6161, logLik(lm(y ~ x + xa)) = -947.03475
    out = _run(client, kinked)
    assert out["loglik_single"] == pytest.approx(-1067.6161, rel=1e-6)
    assert out["loglik_segmented"] == pytest.approx(-947.03475, rel=1e-6)
    assert out["lr_stat"] == pytest.approx(2 * (-947.03475 + 1067.6161), rel=1e-5)


def test_a_binary_outcome_keeps_the_normal_reference(client, kinked):
    # A GLM is normal-based, so z stays: R's glm gives p = 2.9419639e-07 for
    # the change in slope and uSTAT must not switch that one to t.
    out = _run(client, kinked, outcome="event", outcome_kind="binary",
               covariates=["age", "sex"])
    assert out["effect_difference"]["p"] == pytest.approx(2.9419639e-07, rel=1e-3)
    assert out["effect_below"]["ratio"] == pytest.approx(np.exp(out["effect_below"]["beta"]))


def test_adjustment_dummy_codes_a_text_covariate(client, kinked):
    # Left as a number, "site" would say ward C is three times ward A.
    df = store.get("thr")
    enc = _encode_covariates(df, ["age", "sex", "site"], ["site"])
    assert "age" in enc.columns and "sex" in enc.columns
    assert [c for c in enc.columns if c.startswith("site")] == ["site_B", "site_C"]


# ── the parts that keep it honest ──────────────────────────────────────────────


def test_both_models_are_fitted_on_the_same_rows(client, kinked):
    # Likelihoods from different subsets are not comparable and the test
    # between them is meaningless, so the complete-case rule spans every
    # variable in the model, including covariates the single-line fit uses.
    out = _run(client, kinked, covariates=["age", "sex"])
    df = store.get("thr")
    expected = int(df[["y", "x", "age", "sex"]].notna().all(axis=1).sum())
    assert out["n_used"] == expected
    assert out["n_dropped"] == len(df) - expected


def test_a_flat_profile_is_called_out(client):
    """A straight line with noise on it — no kink anywhere.

    This is the case the caveat exists for. Searching 400 breakpoints and
    keeping the best one squeezes the likelihood-ratio test to p = 0.048 on
    data generated from a single straight line: "significant" at the
    conventional cut, and completely spurious. The p-value is therefore not
    what this test asserts on. What has to hold is that the panel says the
    profile is flat, which is the honest signal that the inflection point it
    just printed is not identified.
    """
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 100, 300)
    store.save("thr_flat", pd.DataFrame({"x": x, "y": 2.0 * x + rng.normal(0, 30, 300)}))
    out = _run(client, "thr_flat")
    assert any("flat" in w.lower() for w in out["warnings"])
    assert out["lr_p"] > 0.01  # nowhere near the 1e-54 of a real breakpoint


def test_the_caveat_about_the_p_value_travels_with_the_result(client, kinked):
    out = _run(client, kinked)
    assert "optimistic" in out["caveat"]


def test_every_candidate_keeps_enough_observations_on_each_side(client, kinked):
    out = _run(client, kinked)
    df = store.get("thr").dropna(subset=["x", "y"])
    below = int((df["x"] <= out["breakpoint"]).sum())
    assert below >= MIN_SIDE and len(df) - below >= MIN_SIDE


def test_a_dataset_too_small_to_split_is_refused(client):
    rng = np.random.default_rng(1)
    store.save("thr_tiny", pd.DataFrame({"x": rng.normal(size=12), "y": rng.normal(size=12)}))
    r = client.post("/api/threshold/analyze",
                    json={"session_id": "thr_tiny", "outcome": "y", "exposure": "x"})
    assert r.status_code == 400
    assert "too few" in r.json()["detail"].lower()


def test_a_binary_outcome_that_is_not_zero_one_is_refused(client, kinked):
    r = client.post("/api/threshold/analyze", json={
        "session_id": kinked, "outcome": "age", "exposure": "x", "outcome_kind": "binary"})
    assert r.status_code == 400
    assert "0 / 1" in r.json()["detail"]


def test_survival_without_a_time_column_is_refused(client, kinked):
    r = client.post("/api/threshold/analyze", json={
        "session_id": kinked, "outcome": "event", "exposure": "x", "outcome_kind": "survival"})
    assert r.status_code == 400
    assert "time" in r.json()["detail"].lower()


def test_an_unknown_column_is_refused(client, kinked):
    r = client.post("/api/threshold/analyze", json={
        "session_id": kinked, "outcome": "y", "exposure": "nope"})
    assert r.status_code == 400


def test_the_sentence_carries_the_numbers_it_claims(client, kinked):
    out = _run(client, kinked)
    text = out["result_text"]
    assert "inflection point" in text
    assert f"{out['n_used']}" in text
    assert "log-likelihood ratio test" in text


def test_the_curve_is_drawn_across_the_observed_range(client, kinked):
    out = _run(client, kinked)
    df = store.get("thr")
    assert out["curve"]["x"][0] == pytest.approx(float(df["x"].min()), rel=1e-9)
    assert out["curve"]["x"][-1] == pytest.approx(float(df["x"].max()), rel=1e-9)
    assert len(out["profile"]) > 50
