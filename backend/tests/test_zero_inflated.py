"""Zero-inflated Poisson / negative binomial regression with a Vuong test,
for POST /api/models/zip and /api/models/zinb."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson

from conftest import make_session

SEED = 9031
N = 2500
TRUE_LOG_IRR_X = 0.4
TRUE_INFL_Z = 0.9
TRUE_ALPHA = 0.7


def _frame(kind="zip", structural=True, n=N, seed=SEED):
    """kind: 'zip' (Poisson counts), 'zinb' (NB counts). structural=False
    switches the structural zeros off, giving plain Poisson / NB data."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.integers(0, 2, n)
    t = rng.uniform(0.5, 3.0, n)
    mu = t * np.exp(0.3 + TRUE_LOG_IRR_X * x)
    if kind == "zinb":
        counts = rng.poisson(rng.gamma(1.0 / TRUE_ALPHA, mu * TRUE_ALPHA))
    else:
        counts = rng.poisson(mu)
    pi = 1.0 / (1.0 + np.exp(-(-0.6 + TRUE_INFL_Z * z))) if structural else np.zeros(n)
    y = np.where(rng.random(n) < pi, 0, counts)
    return pd.DataFrame({"y": y.astype(int), "x": x, "z": z, "t": t})


@pytest.fixture(scope="module")
def zip_frame():
    return _frame("zip")


@pytest.fixture(scope="module")
def zip_sid(zip_frame):
    return make_session(zip_frame, "zi_zip_main")


@pytest.fixture(scope="module")
def zinb_frame():
    return _frame("zinb")


@pytest.fixture(scope="module")
def zinb_sid(zinb_frame):
    return make_session(zinb_frame, "zi_zinb_main")


def _post(client, endpoint, sid, **kw):
    body = {"session_id": sid, "outcome": "y", "predictors": ["x"]}
    body.update(kw)
    return client.post(f"/api/models/{endpoint}", json=body)


def _by_var(rows):
    return {c["variable"]: c for c in rows}


# ── ZIP ──────────────────────────────────────────────────────────────────────


def test_zip_recovers_truth_and_matches_statsmodels(client, zip_sid, zip_frame):
    r = _post(client, "zip", zip_sid, inflation_predictors=["z"], exposure_col="t")
    assert r.status_code == 200, r.text
    d = r.json()
    count = _by_var(d["count_coefficients"])
    infl = _by_var(d["inflation_coefficients"])
    assert abs(count["x"]["log_irr"] - TRUE_LOG_IRR_X) < 0.06
    assert abs(count["x"]["irr"] - np.exp(TRUE_LOG_IRR_X)) < 0.12
    assert count["x"]["irr_ci_low"] < np.exp(TRUE_LOG_IRR_X) < count["x"]["irr_ci_high"]
    assert abs(infl["z"]["logit"] - TRUE_INFL_Z) < 0.3
    assert infl["z"]["or_ci_low"] < np.exp(TRUE_INFL_Z) < infl["z"]["or_ci_high"]
    # direct statsmodels fit on the same design
    X = sm.add_constant(zip_frame[["x"]]).to_numpy()
    Z = sm.add_constant(zip_frame[["z"]]).to_numpy()
    ref = ZeroInflatedPoisson(zip_frame["y"].to_numpy(), X, exog_infl=Z,
                              exposure=zip_frame["t"].to_numpy()).fit(
        method="bfgs", maxiter=1000, disp=0)
    assert count["x"]["log_irr"] == pytest.approx(float(ref.params[3]), abs=2e-3)
    assert infl["z"]["logit"] == pytest.approx(float(ref.params[1]), abs=2e-3)
    assert d["loglik"] == pytest.approx(float(ref.llf), abs=1e-2)
    assert d["aic"] == pytest.approx(float(ref.aic), abs=2e-2)
    assert d["converged"] is True
    assert d["rate_model"] is True and d["exposure_col"] == "t"


def test_zip_vuong_prefers_zip_when_zeros_are_inflated(client, zip_sid):
    d = _post(client, "zip", zip_sid, inflation_predictors=["z"], exposure_col="t").json()
    v = d["vuong"]
    assert v["standard_model"] == "Poisson"
    assert v["statistic"]["aic"] > 1.96 and v["p"]["aic"] < 0.05
    assert v["statistic"]["bic"] > 1.96
    assert v["preferred"] == "zero_inflated"
    assert "Zero-inflated" in v["preferred_label"]
    assert "Vuong" in d["result_text"] and "preferred" in d["result_text"]
    # the zero-inflated model reproduces the zeros; Poisson badly misses them
    assert abs(d["expected_zeros"] - d["n_zeros"]) < 0.05 * d["n_zeros"]
    assert d["expected_zeros_standard"] < d["n_zeros"] * 0.85


def test_zip_vuong_formula_matches_manual_computation(client, zip_sid, zip_frame):
    d = _post(client, "zip", zip_sid, exposure_col="t").json()
    X = sm.add_constant(zip_frame[["x"]]).to_numpy()
    y = zip_frame["y"].to_numpy()
    t = zip_frame["t"].to_numpy()
    zi = ZeroInflatedPoisson(y, X, exposure=t).fit(method="bfgs", maxiter=1000, disp=0)
    po = sm.Poisson(y, X, exposure=t).fit(disp=0)
    m = zi.model.loglikeobs(zi.params) - po.model.loglikeobs(po.params)
    n = len(m)
    k = len(zi.params) - len(po.params)
    raw = np.sqrt(n) * m.mean() / m.std(ddof=1)
    aic = np.sqrt(n) * (m.mean() - k / n) / m.std(ddof=1)
    bic = np.sqrt(n) * (m.mean() - k * np.log(n) / (2 * n)) / m.std(ddof=1)
    v = d["vuong"]["statistic"]
    assert v["raw"] == pytest.approx(raw, abs=0.05)
    assert v["aic"] == pytest.approx(aic, abs=0.05)
    assert v["bic"] == pytest.approx(bic, abs=0.05)
    assert d["vuong"]["k_zi"] - d["vuong"]["k_standard"] == k


def test_zip_default_is_intercept_only_inflation(client, zip_sid):
    d = _post(client, "zip", zip_sid, exposure_col="t").json()
    assert [c["variable"] for c in d["inflation_coefficients"]] == ["const"]
    assert d["inflation_predictors"] == []
    for key in ("model", "outcome", "n", "n_excluded", "aic", "bic", "loglik", "n_zeros",
                "expected_zeros", "converged", "warnings", "count_coefficients",
                "inflation_coefficients", "vuong", "result_text"):
        assert key in d
    assert d["alpha"] is None
    c = d["count_coefficients"][0]
    for key in ("variable", "log_irr", "irr", "se", "z", "p", "irr_ci_low", "irr_ci_high"):
        assert key in c
    i = d["inflation_coefficients"][0]
    for key in ("logit", "or", "se", "z", "p", "or_ci_low", "or_ci_high"):
        assert key in i


def test_zip_without_inflation_does_not_favour_zip_on_poisson_data(client):
    df = _frame("zip", structural=False)
    s = make_session(df, "zi_plain_poisson")
    d = _post(client, "zip", s, inflation_predictors=["z"], exposure_col="t").json()
    v = d["vuong"]
    assert v["preferred"] != "zero_inflated"
    assert v["statistic"]["aic"] < 1.96
    assert v["statistic"]["bic"] < 1.96


def test_zip_without_exposure_runs(client, zip_sid):
    d = _post(client, "zip", zip_sid).json()
    assert d["rate_model"] is False and d["exposure_col"] is None
    assert "exposure" not in d["result_text"].lower()


# ── ZINB ─────────────────────────────────────────────────────────────────────


def test_zinb_recovers_truth_and_matches_statsmodels(client, zinb_sid, zinb_frame):
    r = _post(client, "zinb", zinb_sid, inflation_predictors=["z"], exposure_col="t")
    assert r.status_code == 200, r.text
    d = r.json()
    count = _by_var(d["count_coefficients"])
    infl = _by_var(d["inflation_coefficients"])
    assert abs(count["x"]["log_irr"] - TRUE_LOG_IRR_X) < 0.08
    assert abs(infl["z"]["logit"] - TRUE_INFL_Z) < 0.4
    assert 0.45 < d["alpha"] < 1.0
    assert d["theta"] == pytest.approx(1.0 / d["alpha"], rel=1e-9)
    assert d["alpha_se"] is not None and d["alpha_se"] > 0
    X = sm.add_constant(zinb_frame[["x"]]).to_numpy()
    Z = sm.add_constant(zinb_frame[["z"]]).to_numpy()
    ref = ZeroInflatedNegativeBinomialP(
        zinb_frame["y"].to_numpy(), X, exog_infl=Z, exposure=zinb_frame["t"].to_numpy(), p=2
    ).fit(method="bfgs", maxiter=2000, disp=0)
    assert d["loglik"] == pytest.approx(float(ref.llf), abs=1e-2)
    assert d["alpha"] == pytest.approx(float(ref.params[-1]), abs=5e-3)
    assert d["converged"] is True
    assert d["vuong"]["standard_model"] == "Negative binomial"
    assert d["vuong"]["preferred"] == "zero_inflated"


def test_zinb_predicted_zeros_match_statsmodels_prob_zero(client, zinb_sid, zinb_frame):
    d = _post(client, "zinb", zinb_sid, inflation_predictors=["z"], exposure_col="t").json()
    X = sm.add_constant(zinb_frame[["x"]]).to_numpy()
    Z = sm.add_constant(zinb_frame[["z"]]).to_numpy()
    model = ZeroInflatedNegativeBinomialP(
        zinb_frame["y"].to_numpy(), X, exog_infl=Z, exposure=zinb_frame["t"].to_numpy(), p=2)
    res = model.fit(method="bfgs", maxiter=2000, disp=0)
    p0 = res.predict(which="prob-zero", exposure=zinb_frame["t"].to_numpy())
    assert d["expected_zeros"] == pytest.approx(float(np.sum(p0)), rel=2e-3)


def test_zinb_on_plain_poisson_data_does_not_favour_zero_inflation(client):
    df = _frame("zip", structural=False)
    s = make_session(df, "zi_zinb_poisson_data")
    r = _post(client, "zinb", s, exposure_col="t")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["vuong"]["preferred"] != "zero_inflated"
    assert d["alpha"] is not None and d["alpha"] > 0


# ── Validation (422) ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("endpoint", ["zip", "zinb"])
def test_negative_counts_rejected(client, zip_frame, endpoint):
    df = zip_frame.copy()
    df.loc[2, "y"] = -1
    s = make_session(df, f"zi_neg_{endpoint}")
    r = _post(client, endpoint, s)
    assert r.status_code == 422
    assert "non-negative" in r.json()["detail"]


@pytest.mark.parametrize("endpoint", ["zip", "zinb"])
def test_fractional_counts_rejected(client, zip_frame, endpoint):
    df = zip_frame.copy().astype({"y": float})
    df.loc[2, "y"] = 1.5
    s = make_session(df, f"zi_frac_{endpoint}")
    r = _post(client, endpoint, s)
    assert r.status_code == 422
    assert "integer" in r.json()["detail"]


@pytest.mark.parametrize("endpoint", ["zip", "zinb"])
def test_outcome_without_zeros_rejected(client, zip_frame, endpoint):
    df = zip_frame.copy()
    df["y"] = df["y"] + 1
    s = make_session(df, f"zi_nozero_{endpoint}")
    r = _post(client, endpoint, s)
    assert r.status_code == 422
    assert "at least one zero" in r.json()["detail"]


@pytest.mark.parametrize("endpoint", ["zip", "zinb"])
def test_all_zero_outcome_rejected(client, zip_frame, endpoint):
    df = zip_frame.copy()
    df["y"] = 0
    s = make_session(df, f"zi_allzero_{endpoint}")
    r = _post(client, endpoint, s)
    assert r.status_code == 422
    assert "non-zero" in r.json()["detail"]


def test_unknown_columns_rejected(client, zip_sid):
    assert _post(client, "zip", zip_sid, predictors=["nope"]).status_code == 422
    assert _post(client, "zip", zip_sid, inflation_predictors=["nope"]).status_code == 422
    assert _post(client, "zip", zip_sid, outcome="nope").status_code == 422
    assert _post(client, "zip", zip_sid, predictors=[]).status_code == 422


def test_outcome_cannot_be_a_predictor(client, zip_sid):
    assert _post(client, "zip", zip_sid, predictors=["x", "y"]).status_code == 422


def test_exposure_validation(client, zip_sid, zip_frame):
    r = _post(client, "zip", zip_sid, exposure_col="nope")
    assert r.status_code == 422 and "not found" in r.json()["detail"]
    r = _post(client, "zip", zip_sid, predictors=["x", "t"], exposure_col="t")
    assert r.status_code == 422 and "predictor" in r.json()["detail"]
    r = _post(client, "zip", zip_sid, inflation_predictors=["t"], exposure_col="t")
    assert r.status_code == 422 and "inflation predictor" in r.json()["detail"]
    df = zip_frame.copy()
    df.loc[3, "t"] = 0.0
    s = make_session(df, "zi_exp_zero")
    r = _post(client, "zip", s, exposure_col="t")
    assert r.status_code == 422 and "strictly positive" in r.json()["detail"]


def test_unknown_session_is_404(client):
    r = client.post("/api/models/zip", json={"session_id": "no-such-session", "outcome": "y", "predictors": ["x"]})
    assert r.status_code == 404


def test_missing_rows_are_listwise_excluded(client, zip_frame):
    df = zip_frame.copy()
    df.loc[[0, 1, 2], "x"] = np.nan
    s = make_session(df, "zi_na")
    d = _post(client, "zip", s, exposure_col="t").json()
    assert d["n"] == len(zip_frame) - 3
    assert d["n_excluded"] == 3
