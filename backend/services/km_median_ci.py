"""95% confidence interval for the Kaplan-Meier median survival time."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from lifelines.utils import median_survival_times


def _finite_or_none(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def km_median_ci(kmf) -> Tuple[Optional[float], Optional[float]]:
    """(low, high) CI of the median from a fitted KaplanMeierFitter.

    Uses the pointwise survival-curve CI (lifelines default log-log
    transform), i.e. the Brookmeyer-Crowley construction. A bound is None
    when it is not reached or undefined.
    """
    try:
        ci = median_survival_times(kmf.confidence_interval_)
        low = _finite_or_none(ci.iloc[0, 0])
        high = _finite_or_none(ci.iloc[0, 1])
    except Exception:
        return None, None
    return low, high
