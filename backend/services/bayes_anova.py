"""JZS Bayes factor for a one-way ANOVA (one fixed factor).

Rouder, Morey, Speckman & Province (2012), "Default Bayes factors for ANOVA
designs", J. Math. Psych. 56:356-374; the model BayesFactor::anovaBF fits
for a single fixed factor:

    y = mu + X* theta + e,   e ~ N(0, sigma^2 I)
    theta ~ N(0, g sigma^2 I_{J-1}),   g ~ InvGamma(1/2, r^2 / 2)
    p(mu, sigma^2) proportional to 1 / sigma^2

where X* = X Q codes the J groups in an orthonormal sum-to-zero basis Q, so
the J - 1 effects are exchangeable and no group is singled out. Given g the
Bayes factor against the intercept-only model is available in closed form,

    BF(g) = |I + g Xc'Xc|^(-1/2) (1 - yc'Xc (Xc'Xc + I/g)^(-1) Xc'yc / yc'yc)^(-(N-1)/2)

(Xc, yc centred), and BF10 is its average over the prior on g, a
one-dimensional integral done here on the log-g scale around its mode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, stats

# BayesFactor's named prior scales for fixed effects.
RSCALE_NAMED = {"medium": 0.5, "wide": math.sqrt(2) / 2, "ultrawide": 1.0}


@dataclass(frozen=True)
class OneWayBF:
    log_bf10: float
    rel_error: float
    k_groups: int
    n: int


def _sum_to_zero_basis(k: int) -> np.ndarray:
    """k x (k-1) orthonormal basis of the contrasts that sum to zero."""
    centring = np.eye(k) - 1.0 / k
    u, _, _ = np.linalg.svd(centring)
    return u[:, : k - 1]


def jzs_oneway_bf(y: np.ndarray, groups: np.ndarray, rscale: float = 0.5) -> OneWayBF:
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    levels = list(dict.fromkeys(groups.tolist()))
    k, n = len(levels), len(y)
    if k < 2:
        raise ValueError("At least two groups are needed.")
    if n <= k:
        raise ValueError("More observations than groups are needed.")

    design = np.column_stack([(groups == lv).astype(float) for lv in levels])
    xs = design @ _sum_to_zero_basis(k)
    xc = xs - xs.mean(axis=0)
    yc = y - y.mean()
    yy = float(yc @ yc)
    if yy <= 0:
        raise ValueError("The outcome does not vary.")
    xtx = xc.T @ xc
    xty = xc.T @ yc
    eye = np.eye(k - 1)
    a_prior, b_prior = 0.5, rscale * rscale / 2.0

    def log_integrand(log_g: float) -> float:
        g = math.exp(log_g)
        a = xtx + eye / g
        _, logdet_a = np.linalg.slogdet(a)
        resid = yy - float(xty @ np.linalg.solve(a, xty))
        log_bf_g = (
            -0.5 * ((k - 1) * log_g + logdet_a)
            - 0.5 * (n - 1) * (math.log(resid) - math.log(yy))
        )
        # Jacobian of g = exp(log g) included.
        return log_bf_g + stats.invgamma.logpdf(g, a=a_prior, scale=b_prior) + log_g

    mode = optimize.minimize_scalar(
        lambda t: -log_integrand(t), bounds=(-30.0, 30.0), method="bounded",
        options={"xatol": 1e-10},
    ).x
    peak = log_integrand(mode)
    value, abserr = integrate.quad(
        lambda t: math.exp(log_integrand(t) - peak), -40.0, 40.0,
        points=[mode], limit=500, epsabs=0.0, epsrel=1e-10,
    )
    return OneWayBF(
        log_bf10=peak + math.log(value),
        rel_error=abserr / value if value > 0 else float("inf"),
        k_groups=k,
        n=n,
    )
