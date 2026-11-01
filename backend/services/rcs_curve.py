"""Curve helpers for restricted cubic spline results.

- `relative_curve`: effect of x relative to a reference value with a
  delta-method 95% band. The reference row is built at the exact reference
  value, so the curve equals 1 (or 0) there and not at the nearest grid point.
- `find_nadir`: the x that minimises the linear predictor (lowest risk for
  Cox/logistic, lowest mean for linear). Xu et al. (2025, rcssci) and Austin
  (2025) use it as a biologically meaningful reference for U- and J-shaped
  curves. At the edge of the evaluated range the curve has no interior
  minimum, and the result says so.
- `distribution`: histogram of the observed predictor over the plotted
  range, drawn under the curve so sparse tails with wide bands are visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

Z95 = 1.96


@dataclass(frozen=True)
class RelativeCurve:
    estimate: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def relative_curve(
    grid_rows: np.ndarray,
    ref_row: np.ndarray,
    params: np.ndarray,
    cov_params: np.ndarray,
    log_link: bool,
) -> RelativeCurve:
    """Effect at each grid row relative to `ref_row` (ratio when `log_link`)."""
    params = np.asarray(params, dtype=float)
    diffs = np.asarray(grid_rows, dtype=float) - np.asarray(ref_row, dtype=float).reshape(1, -1)
    rel = diffs @ params
    var = np.einsum("ij,jk,ik->i", diffs, np.asarray(cov_params, dtype=float), diffs)
    se = np.sqrt(np.maximum(var, 0.0))
    low, high = rel - Z95 * se, rel + Z95 * se
    if log_link:
        return RelativeCurve(np.exp(rel), np.exp(low), np.exp(high))
    return RelativeCurve(rel, low, high)


@dataclass(frozen=True)
class Nadir:
    x: float
    at_boundary: bool


def find_nadir(lp: Callable[[np.ndarray], np.ndarray], grid: np.ndarray) -> Nadir:
    """Minimum of the linear predictor over [grid[0], grid[-1]].

    Takes the best grid point, then refines between its neighbours with a
    bounded scalar search. A minimum on the first or last grid point is
    reported with `at_boundary=True`: the curve keeps falling beyond the
    evaluated range, so there is no interior nadir to use as a reference.
    """
    values = lp(grid)
    i = int(np.argmin(values))
    if i in (0, len(grid) - 1):
        return Nadir(float(grid[i]), True)
    lo, hi = float(grid[i - 1]), float(grid[i + 1])
    res = minimize_scalar(lambda v: float(lp(np.array([v]))[0]), bounds=(lo, hi), method="bounded",
                          options={"xatol": (hi - lo) * 1e-6})
    x_best = float(res.x) if res.success and res.fun <= values[i] else float(grid[i])
    return Nadir(x_best, False)


def distribution(x: np.ndarray, lo: float, hi: float, max_bins: int = 40) -> dict:
    """Histogram of `x` on [lo, hi]; observations outside are counted apart."""
    inside = x[(x >= lo) & (x <= hi)]
    bins = int(np.clip(np.ceil(np.sqrt(len(inside))), 10, max_bins))
    counts, edges = np.histogram(inside, bins=bins, range=(lo, hi))
    return {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "n_outside": int(len(x) - len(inside)),
    }
