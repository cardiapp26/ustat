"""Absolute event risk at a time horizon from a fitted Cox model.

Austin (2025, Stat Methods Med Res 34(2):277-285) recommends plotting the
cumulative incidence F(t0 | x) = 1 - S(t0 | x) against the spline variable
next to the relative hazard ratio, because the hazard ratio depends on the
chosen reference value while the absolute risk does not.

S(t0 | x) uses the Breslow cumulative hazard. The confidence band follows
survival::survfit.coxph (conf.type = "log"): the variance of H(t0 | x) is the
Tsiatis estimate (hazard increments plus the coefficient-uncertainty term),
and the limits are exp(-H -/+ z se(H)) on the survival scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AbsoluteRisk:
    risk: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


@dataclass(frozen=True)
class _BaselineSums:
    """Risk-set sums up to the horizon, with covariates centred at `center`."""

    center: np.ndarray
    hazard: float          # sum d_i / S0(t_i)
    hazard_sq: float       # sum d_i / S0(t_i)^2
    weighted_mean: np.ndarray  # sum d_i * zbar(t_i) / S0(t_i)


def _baseline_sums(
    design: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    beta: np.ndarray,
    horizon: float,
) -> _BaselineSums:
    center = design.mean(axis=0)
    zc = design - center
    weights = np.exp(zc @ beta)

    order = np.argsort(-duration, kind="stable")
    t_sorted = duration[order]
    w_sorted = weights[order]
    wz_sorted = w_sorted[:, None] * zc[order]
    # Cumulative sums from the longest time down give the risk set at t:
    # everyone with duration >= t.
    s0_cum = np.cumsum(w_sorted)
    s1_cum = np.cumsum(wz_sorted, axis=0)

    event_times = np.unique(duration[(event == 1) & (duration <= horizon)])
    hazard = 0.0
    hazard_sq = 0.0
    weighted_mean = np.zeros(design.shape[1])
    for t in event_times:
        # Last index in the descending order whose duration is >= t.
        last = np.searchsorted(-t_sorted, -t, side="right") - 1
        s0 = s0_cum[last]
        zbar = s1_cum[last] / s0
        d = float(np.sum((duration == t) & (event == 1)))
        hazard += d / s0
        hazard_sq += d / s0 ** 2
        weighted_mean = weighted_mean + d * zbar / s0
    return _BaselineSums(center, hazard, hazard_sq, weighted_mean)


def absolute_risk(
    design: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    beta: np.ndarray,
    cov_beta: np.ndarray,
    x_new: np.ndarray,
    horizon: float,
    z: float = 1.96,
) -> AbsoluteRisk:
    """F(horizon | x) with a survfit-style 95% band for each row of `x_new`.

    Args:
        design: (n, p) covariate matrix the model was fitted on (no intercept).
        duration, event: survival outcome, event coded 0/1.
        beta, cov_beta: fitted coefficients and their covariance.
        x_new: (m, p) covariate rows to predict for.
        horizon: time t0, in the units of `duration`.
    """
    beta = np.asarray(beta, dtype=float)
    cov_beta = np.asarray(cov_beta, dtype=float)
    sums = _baseline_sums(design, duration, event, beta, horizon)

    xc = np.asarray(x_new, dtype=float) - sums.center
    rel = np.exp(xc @ beta)
    cum_hazard = rel * sums.hazard
    # Derivative of H(t0 | x) with respect to beta: rel * sum d (x - zbar) / S0.
    grad = rel[:, None] * (xc * sums.hazard - sums.weighted_mean)
    var = rel ** 2 * sums.hazard_sq + np.einsum("ij,jk,ik->i", grad, cov_beta, grad)
    se = np.sqrt(np.maximum(var, 0.0))

    surv = np.exp(-cum_hazard)
    surv_low = np.exp(-cum_hazard - z * se)
    surv_high = np.exp(-cum_hazard + z * se)
    return AbsoluteRisk(
        risk=1.0 - surv,
        ci_low=1.0 - np.minimum(surv_high, 1.0),
        ci_high=1.0 - surv_low,
    )
