"""Inference helpers for restricted cubic spline models.

Two pieces that Harrell's RMS treats as the standard report for a spline
term, kept out of the endpoint so both RCS routes share them:

- Joint Wald tests on a block of coefficients. The overall-association test
  covers every column of the term (linear + cubic, plus any interaction
  columns involving it, as rms::anova does), df = k - 1 without
  interactions. The non-linearity test covers only the cubic columns,
  df = k - 2.
- An AIC comparison across knot counts (RMS §2.4.6: the principal choice is
  between 3, 4 and 5 knots, and AIC gives a data-based choice).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2
from scipy.stats import f as f_dist

from services.rcs_basis import AIC_CANDIDATE_KNOTS, harrell_knots, rcs_basis
from services.rcs_glm import fit_glm


def joint_wald(
    params: np.ndarray,
    cov_params: np.ndarray,
    idx: Sequence[int],
    resid_df: Optional[float] = None,
) -> Optional[dict]:
    """Joint Wald test that every coefficient in `idx` is zero.

    With `resid_df` (ordinary least squares) the statistic is referred to
    F(df, resid_df) as W / df, which is what rms::anova reports for ols();
    otherwise to chi-square(df), as for lrm() and cph().

    Returns None for an empty block or a singular covariance block; the
    caller reports that as an unavailable test rather than a p of 1.
    """
    index = np.asarray(list(idx), dtype=int)
    if index.size == 0:
        return None
    beta = np.asarray(params, dtype=float)[index]
    cov = np.asarray(cov_params, dtype=float)[np.ix_(index, index)]
    try:
        wald = float(beta @ np.linalg.solve(cov, beta))
    except np.linalg.LinAlgError:
        logger.warning("RCS joint Wald test skipped: singular covariance block")
        return None
    if not np.isfinite(wald):
        return None
    df = int(index.size)
    if resid_df is not None and resid_df > 0:
        f_stat = wald / df
        return {
            "test": "F",
            "wald": round(wald, 4),
            "f": round(f_stat, 4),
            "df": df,
            "df_resid": float(resid_df),
            "p": float(f_dist.sf(f_stat, df, resid_df)),
        }
    return {"test": "chi2", "wald": round(wald, 4), "df": df, "p": float(chi2.sf(wald, df=df))}


def aic_from_loglik(log_lik: float, n_coef: int, model_type: str) -> float:
    """AIC as R's AIC() counts it.

    The linear model also estimates the residual variance, so it adds one
    parameter; statsmodels' OLS.aic leaves it out and sits 2 below R.
    """
    k = n_coef + (1 if model_type == "linear" else 0)
    return float(-2.0 * log_lik + 2.0 * k)


@dataclass(frozen=True)
class RCSOutcome:
    """Outcome data for one RCS fit; Cox uses duration/event, others y.

    `offset` is log(exposure) for the count models, else None.
    """

    model_type: str
    y: Optional[np.ndarray] = None
    duration: Optional[np.ndarray] = None
    event: Optional[np.ndarray] = None
    offset: Optional[np.ndarray] = None


def _fit_loglik(outcome: RCSOutcome, design: np.ndarray) -> tuple[float, int]:
    """Fit the model on `design` (no intercept column) and return (llf, n_coef)."""
    if outcome.model_type == "cox":
        from lifelines import CoxPHFitter

        cols = [f"_c{i}" for i in range(design.shape[1])]
        fit_df = pd.DataFrame(design, columns=cols)
        fit_df["_dur_"] = outcome.duration
        fit_df["_evt_"] = outcome.event
        cph = CoxPHFitter()
        cph.fit(fit_df, duration_col="_dur_", event_col="_evt_")
        return float(cph.log_likelihood_), design.shape[1]

    X = np.column_stack([np.ones(design.shape[0]), design])
    fit = fit_glm(outcome.model_type, outcome.y, X, outcome.offset)
    if fit.log_lik is None:
        raise ValueError("model has no log-likelihood")
    return fit.log_lik, fit.n_coef


async def knot_aic_comparison(
    x: np.ndarray,
    cov_mat: Optional[np.ndarray],
    outcome: RCSOutcome,
    chosen_k: int,
) -> list[dict]:
    """AIC for Harrell-default knots at 3, 4, 5 (and the chosen k if larger).

    Fits the main-effects model (spline + covariates) for each knot count.
    Counts the predictor cannot support (fewer than k + 2 unique values) or
    fits that fail are left out. `delta_aic` is relative to the lowest AIC.
    Quasi-Poisson has no likelihood and so no AIC: the table is empty.
    """
    if outcome.model_type == "quasipoisson":
        return []
    counts = sorted(set(AIC_CANDIDATE_KNOTS) | {chosen_k})
    n_unique = len(np.unique(x))
    rows: list[dict] = []
    for k in counts:
        if n_unique < k + 2:
            continue
        knots = harrell_knots(x, k)
        if np.any(np.diff(knots) <= 0):
            continue
        parts = [x, rcs_basis(x, knots)]
        if cov_mat is not None:
            parts.append(cov_mat)
        design = np.column_stack(parts)
        try:
            llf, n_coef = await asyncio.to_thread(_fit_loglik, outcome, design)
        except Exception:
            logger.exception("RCS knot comparison fit failed for k={}", k)
            continue
        if not np.isfinite(llf):
            continue
        rows.append({
            "n_knots": k,
            "aic": round(aic_from_loglik(llf, n_coef, outcome.model_type), 3),
            "log_likelihood": round(llf, 4),
        })
    if not rows:
        return rows
    best = min(r["aic"] for r in rows)
    return [
        {**r, "delta_aic": round(r["aic"] - best, 3), "best": r["aic"] == best, "chosen": r["n_knots"] == chosen_k}
        for r in rows
    ]
