"""Poisson regression with an exposure (follow-up time) offset, and the
Pearson overdispersion check, for POST /api/models/poisson."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from conftest import make_session

SEED = 7120


def _rate_frame(n=900, overdispersed=False):
    rng = np.random.default_rng(SEED)
    group = rng.integers(0, 2, n)
    age = rng.normal(50, 8, n)
    # follow-up depends on group, so ignoring it biases the unadjusted IRR
    time = np.where(group == 1, rng.uniform(0.5, 2.0, n), rng.uniform(2.0, 6.0, n))
    rate = np.exp(-1.5 + 0.5 * group + 0.01 * (age - 50))
    mu = rate * time
    if overdispersed:
        # gamma-Poisson mixture: variance = mu + mu^2 / theta, theta small
        counts = rng.poisson(rng.gamma(shape=0.8, scale=mu / 0.8))
    else:
        counts = rng.poisson(mu)
    return pd.DataFrame({"events": counts.astype(int), "group": group, "age": age, "py": time})


@pytest.fixture(scope="module")
def frame():
    return _rate_frame()


@pytest.fixture(scope="module")
def sid(frame):
    return make_session(frame, "pois_exposure_main")


def _post(client, sid, **kw):
    body = {"session_id": sid, "outcome": "events", "predictors": ["group", "age"]}
    body.update(kw)
    return client.post("/api/models/poisson", json=body)


# statsmodels 0.14 returns NaN coefficients for a Series exposure, so the
# reference fits below pass the exposure as a plain array.


def _by_var(d):
    return {c["variable"]: c for c in d["coefficients"]}


def test_exposure_matches_statsmodels_offset_fit(client, sid, frame):
    r = _post(client, sid, exposure_col="py")
    assert r.status_code == 200, r.text
    d = r.json()
    X = sm.add_constant(frame[["group", "age"]])
    ref = sm.GLM(frame["events"], X, family=sm.families.Poisson(), exposure=frame["py"].to_numpy()).fit()
    got = _by_var(d)
    assert got["group"]["irr"] == pytest.approx(float(np.exp(ref.params["group"])), rel=1e-8)
    assert got["group"]["se"] == pytest.approx(float(ref.bse["group"]), rel=1e-8)
    assert got["age"]["irr"] == pytest.approx(float(np.exp(ref.params["age"])), rel=1e-8)
    assert d["aic"] == pytest.approx(float(ref.aic), rel=1e-8)
    # the simulated rate ratio for group is exp(0.5) = 1.65
    assert abs(np.log(got["group"]["irr"]) - 0.5) < 0.2


def test_exposure_response_fields(client, sid):
    d = _post(client, sid, exposure_col="py").json()
    assert d["exposure_col"] == "py"
    assert d["rate_model"] is True
    assert "py" not in _by_var(d)
    assert "exposure" in d["result_text"].lower() and "py" in d["result_text"]
    assert "rate ratio" in d["result_text"].lower()


def test_rate_model_differs_from_unadjusted_fit(client, sid):
    rate = _by_var(_post(client, sid, exposure_col="py").json())["group"]["irr"]
    plain = _post(client, sid)
    assert plain.status_code == 200
    pd_ = plain.json()
    assert pd_["exposure_col"] is None
    assert pd_["rate_model"] is False
    count_irr = _by_var(pd_)["group"]["irr"]
    # follow-up is shorter in group 1, so the plain count ratio understates the rate ratio
    assert count_irr < rate - 0.2


def test_without_exposure_keeps_existing_keys(client, sid):
    d = _post(client, sid).json()
    for key in ("model", "outcome", "n", "n_excluded", "imputation", "aic", "bic",
                "warnings", "coefficients", "result_text"):
        assert key in d
    c = d["coefficients"][0]
    for key in ("variable", "log_irr", "irr", "se", "z", "p", "ci_low", "ci_high",
                "irr_ci_low", "irr_ci_high", "vif"):
        assert key in c


def test_dispersion_equidispersed_data(client, sid, frame):
    d = _post(client, sid, exposure_col="py").json()
    X = sm.add_constant(frame[["group", "age"]])
    ref = sm.GLM(frame["events"], X, family=sm.families.Poisson(), exposure=frame["py"].to_numpy()).fit()
    assert d["dispersion"] == pytest.approx(float(ref.pearson_chi2 / ref.df_resid), rel=1e-8)
    assert d["dispersion"] < 1.5
    assert d["overdispersed"] is False
    assert d["dispersion_note"] is None


def test_dispersion_flags_overdispersed_data(client):
    df = _rate_frame(overdispersed=True)
    sid2 = make_session(df, "pois_exposure_over")
    d = _post(client, sid2, exposure_col="py").json()
    assert d["dispersion"] > 1.5
    assert d["overdispersed"] is True
    note = d["dispersion_note"].lower()
    assert "negative binomial" in note and "robust" in note
    assert "overdispersion" in d["result_text"].lower()


def test_robust_se_with_exposure(client, sid):
    d = _post(client, sid, exposure_col="py", robust_se=True).json()
    assert "[Robust SE]" in d["model"]
    assert d["rate_model"] is True


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_exposure_must_be_strictly_positive(client, frame, bad):
    df = frame.copy()
    df.loc[3, "py"] = bad
    s = make_session(df, f"pois_exposure_bad_{bad}")
    r = _post(client, s, exposure_col="py")
    assert r.status_code == 422
    assert "strictly positive" in r.json()["detail"]


def test_exposure_must_be_numeric(client, frame):
    df = frame.copy()
    df["py"] = df["py"].astype(object)
    df.loc[5, "py"] = "unknown"
    s = make_session(df, "pois_exposure_text")
    r = _post(client, s, exposure_col="py")
    assert r.status_code == 422
    assert "numeric" in r.json()["detail"]


def test_exposure_cannot_be_a_predictor(client, sid):
    r = _post(client, sid, predictors=["group", "py"], exposure_col="py")
    assert r.status_code == 422
    assert "predictor" in r.json()["detail"]


def test_exposure_unknown_column(client, sid):
    r = _post(client, sid, exposure_col="nope")
    assert r.status_code == 422
    assert "not found" in r.json()["detail"]


def test_exposure_cannot_be_the_outcome(client, sid):
    r = _post(client, sid, exposure_col="events")
    assert r.status_code == 422


def test_missing_exposure_rows_are_listwise_excluded(client, frame):
    df = frame.copy()
    df.loc[[0, 1, 2, 3], "py"] = np.nan
    s = make_session(df, "pois_exposure_na")
    d = _post(client, s, exposure_col="py").json()
    assert d["n"] == len(frame) - 4
    assert d["n_excluded"] == 4
