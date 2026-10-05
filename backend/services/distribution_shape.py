"""Shared distribution shape estimators and existing SPSS standard errors."""

from typing import Optional

import numpy as np
from scipy import stats as scipy_stats


def _finite(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def distribution_shape(x: np.ndarray) -> dict:
    """Skewness and excess kurtosis as SPSS reports them, with z-scores.

    ``bias=False`` gives G1/G2 — the sample estimators SPSS, Excel and
    ``e1071::skewness(type = 2)`` print, and the ones the published standard
    errors below are derived for. scipy's default (bias=True) is the g1/g2
    population form and would make the z-scores mildly wrong at small n, which
    is exactly where they get used.
    """
    n = int(len(x))
    out: dict = {
        "n": n,
        "skewness": None, "skew_se": None, "skew_z": None,
        "kurtosis": None, "kurt_se": None, "kurt_z": None,
    }
    if n < 3 or float(np.std(x)) == 0.0:
        return out

    skew = _finite(scipy_stats.skew(x, bias=False))
    se_skew = float(np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))))
    out["skewness"] = skew
    out["skew_se"] = se_skew
    out["skew_z"] = _finite(skew / se_skew) if skew is not None and se_skew > 0 else None

    if n >= 4:
        kurt = _finite(scipy_stats.kurtosis(x, fisher=True, bias=False))
        out["kurtosis"] = kurt
        if n > 3:
            se_kurt = float(
                2 * se_skew * np.sqrt((n * n - 1) / ((n - 3) * (n + 5)))
            )
            out["kurt_se"] = se_kurt
            out["kurt_z"] = (
                _finite(kurt / se_kurt) if kurt is not None and se_kurt > 0 else None
            )
    return out

