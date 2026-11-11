"""Poisson and negative binomial BIC must sit on the log-likelihood scale.

statsmodels' GLM `bic` is deviance-based, so the panel showed AIC = 391 next
to BIC = -179 for the same model.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_session

from routers.models.glm import NegBinomRequest, PoissonRequest, negative_binomial_regression, poisson_regression

N = 200


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, N)
    y = rng.poisson(rng.gamma(1.5, np.exp(0.2 + 0.4 * x) / 1.5))
    return pd.DataFrame({"x": x, "y": y})


def test_poisson_bic_is_loglik_based():
    sid = make_session(_frame(), "bic_poisson")
    res = poisson_regression(PoissonRequest(session_id=sid, outcome="y", predictors=["x"]))
    n_params = len(res["coefficients"])
    assert res["bic"] == pytest.approx(res["aic"] - 2 * n_params + np.log(N) * n_params)
    assert res["bic"] > res["aic"] > 0


def test_negbinom_bic_exceeds_aic():
    sid = make_session(_frame(), "bic_negbinom")
    res = negative_binomial_regression(NegBinomRequest(session_id=sid, outcome="y", predictors=["x"]))
    assert res["bic"] > res["aic"] > 0
