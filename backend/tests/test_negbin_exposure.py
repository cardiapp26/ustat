"""Negative binomial regression with an exposure (follow-up time) offset,
for POST /api/models/negbinom."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from conftest import make_session

SEED = 4417
THETA = 1.5  # true NB size; alpha = 1 / THETA


def _rate_frame(n=1200):
    rng = np.random.default_rng(SEED)
    group = rng.integers(0, 2, n)
    age = rng.normal(50, 8, n)
    # follow-up depends on group, so ignoring it biases the unadjusted IRR
    time = np.where(group == 1, rng.uniform(0.5, 2.0, n), rng.uniform(2.0, 6.0, n))
    mu = np.exp(-1.5 + 0.5 * group + 0.01 * (age - 50)) * time
    counts = rng.poisson(rng.gamma(shape=THETA, scale=mu / THETA))
    return pd.DataFrame({"events": counts.astype(int), "group": group, "age": age, "py": time})


@pytest.fixture(scope="module")
def frame():
    return _rate_frame()


@pytest.fixture(scope="module")
def sid(frame):
    return make_session(frame, "nb_exposure_main")


def _post(client, sid, **kw):
    body = {"session_id": sid, "outcome": "events", "predictors": ["group", "age"]}
    body.update(kw)
    return client.post("/api/models/negbinom", json=body)


def _by_var(d):
    return {c["variable"]: c for c in d["coefficients"]}


def _reference(frame, exposure):
    X = sm.add_constant(frame[["group", "age"]])
    joint = sm.NegativeBinomial(frame["events"], X, exposure=exposure).fit(disp=0, maxiter=200)
    alpha = float(joint.params["alpha"])
    glm = sm.GLM(frame["events"], X, family=sm.families.NegativeBinomial(alpha=alpha),
                 exposure=exposure).fit()
    return joint, glm, alpha


def test_exposure_matches_statsmodels_joint_and_glm_fit(client, sid, frame):
    r = _post(client, sid, exposure_col="py")
    assert r.status_code == 200, r.text
    d = r.json()
    joint, glm, alpha = _reference(frame, frame["py"].to_numpy())
    got = _by_var(d)
    assert got["group"]["irr"] == pytest.approx(float(np.exp(glm.params["group"])), rel=1e-6)
    assert got["group"]["se"] == pytest.approx(float(glm.bse["group"]), rel=1e-6)
    assert got["age"]["irr"] == pytest.approx(float(np.exp(glm.params["age"])), rel=1e-6)
    assert d["alpha"] == pytest.approx(alpha, rel=1e-6)
    assert d["theta"] == pytest.approx(1.0 / alpha, rel=1e-6)
    assert d["aic"] == pytest.approx(float(joint.aic), rel=1e-6)
    assert d["bic"] == pytest.approx(float(joint.bic), rel=1e-6)
    # the simulated rate ratio for group is exp(0.5)
    assert abs(np.log(got["group"]["irr"]) - 0.5) < 0.25
    # alpha is recovered (1 / 1.5 = 0.67)
    assert 0.45 < d["alpha"] < 0.95


def test_exposure_response_fields_and_text(client, sid):
    d = _post(client, sid, exposure_col="py").json()
    assert d["exposure_col"] == "py"
    assert d["rate_model"] is True
    assert "py" not in _by_var(d)
    text = d["result_text"].lower()
    assert "exposure" in text and "py" in text and "rate ratio" in text
    assert "alpha" in d and "alpha_se" in d and "theta" in d and d["converged"] is True


def test_rate_model_differs_from_count_model(client, sid):
    rate = _by_var(_post(client, sid, exposure_col="py").json())["group"]["irr"]
    plain = _post(client, sid)
    assert plain.status_code == 200
    pd_ = plain.json()
    assert pd_["exposure_col"] is None
    assert pd_["rate_model"] is False
    assert "exposure" not in pd_["result_text"].lower()
    # follow-up is shorter in group 1, so the plain count ratio understates the rate ratio
    assert _by_var(pd_)["group"]["irr"] < rate - 0.2


def test_without_exposure_keeps_existing_keys(client, sid, frame):
    d = _post(client, sid).json()
    for key in ("model", "outcome", "n", "n_excluded", "aic", "bic", "alpha", "alpha_se",
                "theta", "converged", "dispersion_note", "warnings", "coefficients"):
        assert key in d
    for key in ("variable", "log_irr", "irr", "se", "z", "p", "ci_low", "ci_high",
                "irr_ci_low", "irr_ci_high", "vif"):
        assert key in d["coefficients"][0]
    # no exposure: identical to a plain statsmodels NB fit
    X = sm.add_constant(frame[["group", "age"]])
    ref = sm.NegativeBinomial(frame["events"], X).fit(disp=0, maxiter=200)
    assert d["alpha"] == pytest.approx(float(ref.params["alpha"]), rel=1e-6)
    assert d["aic"] == pytest.approx(float(ref.aic), rel=1e-6)


def test_robust_se_with_exposure(client, sid):
    d = _post(client, sid, exposure_col="py", robust_se=True).json()
    assert "[Robust SE]" in d["model"]
    assert d["rate_model"] is True


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_exposure_must_be_strictly_positive(client, frame, bad):
    df = frame.copy()
    df.loc[3, "py"] = bad
    s = make_session(df, f"nb_exposure_bad_{bad}")
    r = _post(client, s, exposure_col="py")
    assert r.status_code == 422
    assert "strictly positive" in r.json()["detail"]


def test_exposure_must_be_numeric(client, frame):
    df = frame.copy()
    df["py"] = df["py"].astype(object)
    df.loc[5, "py"] = "unknown"
    s = make_session(df, "nb_exposure_text")
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
    assert _post(client, sid, exposure_col="events").status_code == 422


def test_missing_exposure_rows_are_listwise_excluded(client, frame):
    df = frame.copy()
    df.loc[[0, 1, 2, 3], "py"] = np.nan
    s = make_session(df, "nb_exposure_na")
    d = _post(client, s, exposure_col="py").json()
    assert d["n"] == len(frame) - 4
    assert d["n_excluded"] == 4
