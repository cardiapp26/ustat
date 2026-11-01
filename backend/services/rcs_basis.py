"""Restricted Cubic Spline basis (Harrell).

Shared by the univariate /api/models/rcs endpoint and the multivariable
/api/models/survival/cox_rcs endpoint.

The basis follows Harrell's "Regression Modeling Strategies" §2.4.5:
  - n_knots knots produces (n_knots - 1) basis columns:
    one linear term (the raw x) plus (n_knots - 2) cubic-spline columns.
  - Default knot positions are Harrell's recommended quantiles (RMS
    Table 2.3, the same values Hmisc::rcspline.eval uses), but custom
    positions can be supplied.
  - Below 100 observations the outer two default knots move to the 5th
    smallest and 5th largest value (RMS §2.4.6; rcspline.eval does the same).
"""

from typing import Optional

import numpy as np


# Harrell RMS Table 2.3. Hmisc builds these as seq(outer, 1 - outer, length=k)
# with outer = .10 (k=3), .05 (k=4..6) and .025 (k=7); 7 knots are stored
# unrounded so they match R exactly.
KNOT_PERCENTILES: dict[int, list[float]] = {
    3: [10, 50, 90],
    4: [5, 35, 65, 95],
    5: [5, 27.5, 50, 72.5, 95],
    6: [5, 23, 41, 59, 77, 95],
    7: [float(v) for v in np.linspace(2.5, 97.5, 7)],
}

SUPPORTED_KNOT_COUNTS: tuple[int, ...] = tuple(sorted(KNOT_PERCENTILES))

# Knot counts compared by AIC. Harrell: more than 5 knots are seldom needed
# and the principal choice is between 3, 4 and 5 (RMS §2.4.6).
AIC_CANDIDATE_KNOTS: tuple[int, ...] = (3, 4, 5)

SMALL_SAMPLE_N = 100


def knot_count_error(n_knots: int) -> str:
    """User-facing message for an unsupported knot count."""
    allowed = ", ".join(str(k) for k in SUPPORTED_KNOT_COUNTS)
    return f"n_knots must be one of {allowed}. Got: {n_knots}"


def harrell_knots(x: np.ndarray, n_knots: int) -> np.ndarray:
    """Default Harrell knots for `n_knots` in SUPPORTED_KNOT_COUNTS.

    With fewer than 100 observations the outer knots are replaced by the 5th
    smallest and 5th largest value, as in Hmisc::rcspline.eval. The swap is
    skipped when it would break strict ordering (heavily tied data), so the
    basis never sees coincident knots from this rule.

    Raises:
        ValueError: when n_knots is outside the supported set.
    """
    if n_knots not in KNOT_PERCENTILES:
        raise ValueError(knot_count_error(n_knots))
    knots = np.percentile(x, KNOT_PERCENTILES[n_knots])
    n = len(x)
    if 5 <= n < SMALL_SAMPLE_N:
        xs = np.sort(x)
        adjusted = knots.copy()
        adjusted[0] = xs[4]
        adjusted[-1] = xs[n - 5]
        if np.all(np.diff(adjusted) > 0):
            knots = adjusted
    return knots


def validate_custom_knots(
    knot_positions: list[float],
    n_knots: int,
    x: np.ndarray,
    column_name: str = "predictor",
) -> np.ndarray:
    """Validate user-supplied knot positions.

    Returns:
        Sorted ndarray of knot positions.

    Raises:
        ValueError: when count does not match `n_knots`, when positions are not
        strictly ascending, or when positions fall outside the data range.
    """
    if len(knot_positions) != n_knots:
        raise ValueError(
            f"knot_positions for '{column_name}' must have exactly {n_knots} "
            f"entries to match n_knots={n_knots}. Got {len(knot_positions)}."
        )
    arr = np.asarray(knot_positions, dtype=float)
    if np.any(np.diff(arr) <= 0):
        raise ValueError(
            f"knot_positions for '{column_name}' must be strictly ascending. "
            f"Got: {knot_positions}"
        )
    x_lo, x_hi = float(np.min(x)), float(np.max(x))
    if arr[0] < x_lo or arr[-1] > x_hi:
        raise ValueError(
            f"knot_positions for '{column_name}' must lie within the data "
            f"range [{x_lo:.4g}, {x_hi:.4g}]. Got: {knot_positions}"
        )
    return arr


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell restricted cubic spline basis.

    Args:
        x: 1D array of length n.
        knots: 1D array of knot positions (length n_knots ≥ 3).

    Returns:
        ndarray of shape (n, n_knots - 2). The columns are the (n_knots − 2)
        cubic-spline basis vectors. The linear `x` column is NOT included —
        callers stack it themselves (matches Harrell's parameterisation).
    """
    k = len(knots)
    if k < 3:
        raise ValueError("rcs_basis requires at least 3 knots.")
    cols = []
    kk = knots[-1]      # last knot
    k1 = knots[-2]      # second-to-last knot
    denom = (kk - knots[0]) ** 2
    for j in range(k - 2):
        t1 = np.maximum(x - knots[j], 0) ** 3
        t2 = np.maximum(x - k1, 0) ** 3
        t3 = np.maximum(x - kk, 0) ** 3
        col = t1 - ((kk - knots[j]) / (kk - k1)) * t2 + ((k1 - knots[j]) / (kk - k1)) * t3
        cols.append(col / denom)
    return np.column_stack(cols)


def resolve_knots(
    x: np.ndarray,
    n_knots: int,
    knot_positions: Optional[list[float]] = None,
    column_name: str = "predictor",
) -> np.ndarray:
    """Resolve final knot positions: validate custom or compute Harrell defaults."""
    if knot_positions is None:
        return harrell_knots(x, n_knots)
    return validate_custom_knots(knot_positions, n_knots, x, column_name)
