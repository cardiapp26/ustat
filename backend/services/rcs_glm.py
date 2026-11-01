"""Fitting the non-survival RCS models in one place.

The RCS endpoint fits the same model up to four times (main model, model
without interactions, crude model, one per knot count in the AIC table).
`fit_glm` is that one fit for every outcome type the endpoint supports:

- logistic: binary outcome, odds ratios (statsmodels Logit).
- linear: continuous outcome, mean differences (OLS).
- poisson: counts, incidence rate ratios (GLM, Poisson family, log link).
  An optional exposure enters as offset log(exposure), so the curve is a
  ratio of rates per unit of exposure.
- quasipoisson: the same mean model with the dispersion estimated from the
  Pearson statistic, sum(r_pearson^2) / df_resid, as R's
  glm(family = quasipoisson). Coefficients equal the Poisson ones; standard
  errors are scaled by sqrt(dispersion). There is no likelihood, so AIC is
  undefined (R reports NA) and tests use F with the residual df.

`nested_test` compares a model with its sub-model the way R's anova.glm
does: likelihood-ratio chi-square, or F on the deviance change divided by
the larger model's dispersion when that dispersion is estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import chi2
from scipy.stats import f as f_dist

GLM_MODEL_TYPES = ("logistic", "linear", "poisson", "quasipoisson")
COUNT_MODEL_TYPES = ("poisson", "quasipoisson")
# Models whose dispersion is estimated: their Wald tests use F(df, df_resid).
F_TEST_MODEL_TYPES = ("linear", "quasipoisson")


@dataclass(frozen=True)
class GLMFit:
    params: np.ndarray
    cov_params: np.ndarray
    # None for quasipoisson: a quasi-likelihood has no log-likelihood.
    log_lik: Optional[float]
    n_coef: int
    df_resid: float
    # True when the dispersion is estimated, so Wald tests use F.
    f_tests: bool
    deviance: Optional[float] = None
    # Pearson chi-square / df_resid for count models, else None.
    dispersion: Optional[float] = None

    @property
    def test_df_resid(self) -> Optional[float]:
        """Residual df for F-based Wald tests, or None for chi-square."""
        return self.df_resid if self.f_tests else None


def fit_glm(
    model_type: str,
    y: np.ndarray,
    X: np.ndarray,
    offset: Optional[np.ndarray] = None,
) -> GLMFit:
    """Fit `model_type` on design `X` (intercept column included)."""
    import statsmodels.api as sm_api

    if model_type == "logistic":
        return _wrap(sm_api.Logit(y, X).fit(disp=0, maxiter=200), X, f_tests=False)
    if model_type == "linear":
        return _wrap(sm_api.OLS(y, X).fit(), X, f_tests=True)
    if model_type in COUNT_MODEL_TYPES:
        model = sm_api.GLM(y, X, family=sm_api.families.Poisson(), offset=offset)
        quasi = model_type == "quasipoisson"
        # scale="X2" is the Pearson estimate R's quasipoisson uses.
        res = model.fit(scale="X2") if quasi else model.fit()
        dispersion = float(res.pearson_chi2 / res.df_resid) if res.df_resid > 0 else None
        return _wrap(
            res, X, f_tests=quasi, deviance=float(res.deviance),
            dispersion=dispersion, has_likelihood=not quasi,
        )
    raise ValueError(f"Unknown GLM model_type: {model_type}")


def _wrap(
    res,
    X: np.ndarray,
    f_tests: bool,
    deviance: Optional[float] = None,
    dispersion: Optional[float] = None,
    has_likelihood: bool = True,
) -> GLMFit:
    llf = float(res.llf) if has_likelihood else float("nan")
    return GLMFit(
        params=np.asarray(res.params, dtype=float),
        cov_params=np.asarray(res.cov_params(), dtype=float),
        log_lik=llf if np.isfinite(llf) else None,
        n_coef=int(X.shape[1]),
        df_resid=float(res.df_resid),
        f_tests=f_tests,
        deviance=deviance,
        dispersion=dispersion,
    )


def nested_test(model_type: str, full: GLMFit, reduced: GLMFit) -> dict:
    """Test the columns in `full` but not in `reduced` (R anova.glm style)."""
    df = full.n_coef - reduced.n_coef
    if model_type == "quasipoisson":
        if full.deviance is None or reduced.deviance is None or not full.dispersion:
            raise ValueError("quasi-Poisson nested test needs deviances and a dispersion")
        f_stat = (reduced.deviance - full.deviance) / df / full.dispersion
        return {
            "test": "F",
            "f": round(f_stat, 4),
            "df": df,
            "df_resid": full.df_resid,
            "p": round(float(f_dist.sf(f_stat, df, full.df_resid)), 6),
        }
    if full.log_lik is None or reduced.log_lik is None:
        raise ValueError("likelihood-ratio test needs both log-likelihoods")
    lr = 2.0 * (full.log_lik - reduced.log_lik)
    return {
        "test": "chi2",
        "lr_stat": round(lr, 4),
        "df": df,
        "p": round(float(chi2.sf(lr, df=df)), 6),
        "log_lik_full": round(full.log_lik, 4),
        "log_lik_reduced": round(reduced.log_lik, 4),
    }
