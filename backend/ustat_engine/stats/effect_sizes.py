"""Effect sizes, their magnitude labels, and the descriptives printed beside them.

Moved out of `services/stat_utils.py` verbatim, for the same reason
`sorted_groups` moved into `frame/levels.py`: a browser-side t-test has to
report the same Hedges' g, the same confidence interval and the same
"medium"/"large" label as the server did, and a second implementation of the
Hedges correction or of the magnitude thresholds would disagree in the third
decimal without either side raising a word.

`services.stat_utils` re-exports every name here, so nothing that already
imported them had to change.

`_es_magnitude` came along even though the t-test only needs two of its
branches: it is the shared vocabulary for every effect size in the app, and
splitting it -- half here, half there -- would be exactly the drift this move
exists to prevent.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp


def _es_magnitude(name: str, val: float) -> str:
    """Cohen's magnitude label for common effect sizes."""
    v = abs(val)
    if name in ("cohen_d", "hedges_g"):
        if v < 0.2:
            return "negligible"
        if v < 0.5:
            return "small"
        if v < 0.8:
            return "medium"
        return "large"
    if name == "cohen_f":
        if v < 0.10:
            return "negligible"
        if v < 0.25:
            return "small"
        if v < 0.40:
            return "medium"
        return "large"
    if name in ("r", "pearson_r", "point_biserial_r"):
        if v < 0.10:
            return "negligible"
        if v < 0.30:
            return "small"
        if v < 0.50:
            return "medium"
        return "large"
    if name in ("eta_squared", "eta2"):
        if v < 0.01:
            return "negligible"
        if v < 0.06:
            return "small"
        if v < 0.14:
            return "medium"
        return "large"
    if name in ("cramers_v", "cramer_v"):
        if v < 0.10:
            return "negligible"
        if v < 0.30:
            return "small"
        if v < 0.50:
            return "medium"
        return "large"
    if name == "odds_ratio":
        if v < 1.5:
            return "negligible"
        if v < 2.5:
            return "small"
        if v < 4.0:
            return "medium"
        return "large"
    if name == "rank_biserial_r":
        if v < 0.10:
            return "negligible"
        if v < 0.30:
            return "small"
        if v < 0.50:
            return "medium"
        return "large"
    return ""


def welch_satterthwaite_df(g1: np.ndarray, g2: np.ndarray) -> float:
    """Welch–Satterthwaite degrees of freedom for two independent samples.

    Fractional by construction, and always ≤ the pooled n1 + n2 − 2. A Welch
    test reported against the pooled df understates the tail and overstates
    significance, so the two must never be mixed.
    """
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1 = float(np.var(g1, ddof=1))
    v2 = float(np.var(g2, ddof=1))
    a, b = v1 / n1, v2 / n2
    denom = (a ** 2) / (n1 - 1) + (b ** 2) / (n2 - 1)
    if denom <= 0:
        return float(n1 + n2 - 2)
    return float((a + b) ** 2 / denom)


def cohen_d(g1: np.ndarray, g2: np.ndarray) -> dict:
    """Cohen's d with 95% CI (Hedges-corrected = Hedges' g for small samples)."""
    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    # Not `sp` — that name is scipy.stats at module level, and shadowing it
    # here broke the t quantile below.
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return {
            "name": "cohen_d",
            "value": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "magnitude": "negligible",
        }
    d = (m1 - m2) / s_pooled
    # Hedges' correction for small samples
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    g = d * j
    # Large-sample standard error of a two-sample d (Hedges & Olkin):
    #     sqrt((n1 + n2) / (n1 * n2) + g^2 / (2 * (n1 + n2)))
    #
    # The expression here used to be
    #     sqrt((n1+n2)/(n1*n2)) * sqrt(1 + g^2 * n1 * n2 / (2 * (n1+n2)))
    # which multiplies out to sqrt((n1+n2)/(n1*n2) + g^2 / 2) — the second
    # term missing its (n1 + n2) divisor. At n = 100 per arm that is 5.5x too
    # wide, so a g of 1.17 carried a 95% CI of [-0.48, 2.81] next to a p of
    # 1.6e-14: an interval spanning zero beside overwhelming significance.
    se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    # t rather than 1.96 — with small groups the normal quantile is too short.
    crit = float(sp.t.ppf(0.975, max(n1 + n2 - 2, 1)))
    ci_lo = g - crit * se
    ci_hi = g + crit * se
    return {
        "name": "hedges_g",
        "value": round(g, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "magnitude": _es_magnitude("hedges_g", g),
    }


def cohen_d_one_sample(x: np.ndarray, mu: float) -> dict:
    """Cohen's d for one-sample: (mean - mu) / sd."""
    n = len(x)
    d = (x.mean() - mu) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0
    se = np.sqrt(1 / n + d**2 / (2 * n))
    crit = float(sp.t.ppf(0.975, max(n - 1, 1)))
    ci_lo = d - crit * se
    ci_hi = d + crit * se
    return {
        "name": "cohen_d",
        "value": round(d, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "magnitude": _es_magnitude("cohen_d", d),
    }


def group_summary(x: np.ndarray, label: str = "Sample") -> dict:
    """Standard descriptive statistics for a numeric array."""
    return {
        "label": label,
        "n": int(len(x)),
        "mean": round(float(x.mean()), 4),
        "sd": round(float(x.std(ddof=1)), 4),
        "median": round(float(np.median(x)), 4),
        "q1": round(float(np.percentile(x, 25)), 4),
        "q3": round(float(np.percentile(x, 75)), 4),
        "min": round(float(x.min()), 4),
        "max": round(float(x.max()), 4),
    }


__all__ = [
    "_es_magnitude",
    "cohen_d",
    "cohen_d_one_sample",
    "group_summary",
    "welch_satterthwaite_df",
]
