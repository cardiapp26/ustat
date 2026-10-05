"""Hodges-Lehmann location estimates with distribution-free confidence intervals.

Pure functions (no FastAPI / pandas), written to reproduce R's
``wilcox.test(..., conf.int = TRUE)`` as closely as practical:

* two independent samples (``hl_two_sample``): estimate = median of all
  n1 * n2 pairwise differences x_i - y_j (sample 1 minus sample 2).
* one sample / paired differences (``hl_one_sample``, ``hl_paired``):
  estimate = pseudomedian, the median of the Walsh averages
  (x_i + x_j) / 2 for i <= j.

Confidence interval, as in R:

* exact (both samples smaller than 50 / n < 50, no ties, and for the one-sample
  case no values equal to mu): order statistics of the sorted differences /
  Walsh averages at the lower alpha/2 quantile of the exact Mann-Whitney U /
  signed-rank null distribution (Bauer 1972; Hollander & Wolfe).
* otherwise: the normal approximation with tie correction and a continuity
  correction. The interval limits are the shifts d at which the standardised
  rank statistic W(d) crosses the normal quantiles. W(d) is a step function that
  only changes at a pairwise difference / Walsh average, so the crossing is
  located by bisection over the sorted distinct candidates instead of R's
  uniroot (tolerance 1e-4); the limits therefore agree with R to ~1e-4.

Memory cap: the candidate array holds n1 * n2 (or n (n + 1) / 2) float64 values.
Above ``MAX_PAIRS`` (4 million, about 32 MB) no estimate is produced: the
functions return a result whose numeric fields are ``None`` and whose ``note``
says so. Nothing is subsampled, so a returned interval is always the full
distribution-free one.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import stats as scipy_stats

MAX_PAIRS = 4_000_000
_EPS = 10 * np.finfo(float).eps


# ---------------------------------------------------------------------------
# exact null distributions
# ---------------------------------------------------------------------------

def _wilcox_counts(m: int, n: int) -> list[int]:
    """Number of arrangements giving each Mann-Whitney U = 0..m*n.

    The generating function is the Gaussian binomial [m+n choose m]_q =
    prod_{i=1..m} (1 - q^(n+i)) / (1 - q^i), evaluated with exact integers.
    """
    size = m * n + 1
    c = [0] * size
    c[0] = 1
    for i in range(1, m + 1):
        shift = n + i
        for k in range(size - 1, shift - 1, -1):  # multiply by (1 - q^shift)
            c[k] -= c[k - shift]
        for k in range(i, size):  # divide by (1 - q^i)
            c[k] += c[k - i]
    return c


def _signrank_counts(n: int) -> list[int]:
    """Number of sign patterns giving each signed-rank V+ = 0..n(n+1)/2."""
    size = n * (n + 1) // 2 + 1
    c = [0] * size
    c[0] = 1
    for r in range(1, n + 1):
        for k in range(size - 1, r - 1, -1):
            c[k] += c[k - r]
    return c


def _lower_quantile(counts: list[int], p: float) -> int:
    """Smallest q with P(T <= q) >= p, with R's qwilcox/qsignrank tolerance."""
    total = sum(counts)
    target = p - _EPS
    acc = 0
    for q, cnt in enumerate(counts):
        acc += cnt
        if acc / total >= target:
            return q
    return len(counts) - 1


def _cdf(counts: list[int], q: int) -> float:
    """P(T <= q); 0 for q < 0."""
    if q < 0:
        return 0.0
    total = sum(counts)
    return sum(counts[: q + 1]) / total


def qwilcox_lower(p: float, m: int, n: int) -> int:
    """R's ``qwilcox(p, m, n)`` for p <= 0.5 (rank-sum statistic U)."""
    return _lower_quantile(_wilcox_counts(m, n), p)


def qsignrank_lower(p: float, n: int) -> int:
    """R's ``qsignrank(p, n)`` for p <= 0.5."""
    return _lower_quantile(_signrank_counts(n), p)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _empty(method: str, conf_level: float, note: str) -> dict:
    return {
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "confidence_level": float(conf_level),
        "achieved_confidence_level": None,
        "method": method,
        "note": note,
    }


def _clean(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name}: no finite values")
    return arr


def _check_conf(conf_level: float) -> None:
    if not (0.0 < conf_level < 1.0):
        raise ValueError("conf_level must lie strictly between 0 and 1")


def _step_root(
    W: Callable[[float], float],
    cand: np.ndarray,
    zq: float,
) -> float:
    """Shift d at which the non-increasing step function W(d) crosses zq.

    ``cand`` are the sorted distinct jump points (pairwise differences or Walsh
    averages); cand[0] / cand[-1] are R's mumin / mumax. Mirrors R's ``root()``:
    returns the bracket end when W does not cross inside it.
    """
    if W(float(cand[0])) - zq <= 0:
        return float(cand[0]) + 0.0
    if W(float(cand[-1])) - zq >= 0:
        return float(cand[-1]) + 0.0
    lo, hi = 0, len(cand) - 1  # smallest k in [0, m-2] with W(mid_k) < zq, else m-1
    while lo < hi:
        mid = (lo + hi) // 2
        rep = 0.5 * (float(cand[mid]) + float(cand[mid + 1]))
        if W(rep) < zq:
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo]) + 0.0


def _ci_note(kind: str, method: str, conf_level: float, achieved: Optional[float]) -> str:
    pct = f"{conf_level * 100:g}%"
    base = (
        f"Hodges-Lehmann {kind} with a distribution-free {pct} CI "
        f"({method}); reproduces R wilcox.test(..., conf.int = TRUE)."
    )
    if achieved is not None and conf_level - achieved > (1 - conf_level) / 2:
        base += (
            f" The requested {pct} level is not attainable with this sample size;"
            f" the achieved confidence level is {achieved * 100:.1f}%."
        )
    return base


# ---------------------------------------------------------------------------
# two independent samples
# ---------------------------------------------------------------------------

def hl_two_sample(
    x,
    y,
    conf_level: float = 0.95,
    max_pairs: int = MAX_PAIRS,
) -> dict:
    """Location shift x - y (median of pairwise differences) with CI.

    Equivalent to ``wilcox.test(x, y, conf.int = TRUE, conf.level = ...)`` in R
    (two-sided, mu = 0, continuity correction on).
    """
    _check_conf(conf_level)
    xa, ya = _clean(x, "x"), _clean(y, "y")
    n1, n2 = xa.size, ya.size
    if n1 * n2 > max_pairs:
        return _empty(
            "not computed",
            conf_level,
            (
                f"Hodges-Lehmann shift not computed: {n1} x {n2} = {n1 * n2:,} pairwise "
                f"differences exceed the {max_pairs:,}-pair memory cap."
            ),
        )

    alpha = 1.0 - conf_level
    diffs = np.sort(np.subtract.outer(xa, ya).ravel())
    estimate = float(np.median(diffs)) + 0.0
    pooled = np.concatenate([xa, ya])
    has_ties = np.unique(pooled).size != pooled.size

    if n1 < 50 and n2 < 50 and not has_ties:
        counts = _wilcox_counts(n1, n2)
        qu = _lower_quantile(counts, alpha / 2)
        if qu == 0:
            qu = 1
        ql = n1 * n2 - qu
        achieved_alpha = 2.0 * _cdf(counts, qu - 1)
        achieved = 1.0 - achieved_alpha
        method = "exact (Mann-Whitney U distribution)"
        return {
            "estimate": estimate,
            "ci_low": float(diffs[qu - 1]) + 0.0,
            "ci_high": float(diffs[ql]) + 0.0,
            "confidence_level": float(conf_level),
            "achieved_confidence_level": float(achieved),
            "method": method,
            "note": _ci_note("location shift (sample 1 minus sample 2)", method, conf_level, achieved),
        }

    def W(d: float) -> float:
        ranks = scipy_stats.rankdata(np.concatenate([xa - d, ya]))
        dz = float(ranks[:n1].sum()) - n1 * (n1 + 1) / 2.0 - n1 * n2 / 2.0
        _, tie_counts = np.unique(ranks, return_counts=True)
        tie_term = float(np.sum(tie_counts.astype(float) ** 3 - tie_counts)) / (
            (n1 + n2) * (n1 + n2 - 1)
        )
        sigma = np.sqrt((n1 * n2 / 12.0) * ((n1 + n2 + 1) - tie_term))
        if sigma == 0:
            return float("nan")
        corr = 0.5 * np.sign(dz)
        return (dz - corr) / sigma

    cand = np.unique(diffs)
    if cand.size < 2 or not np.isfinite(W(float(cand[0]))):
        method = "normal approximation (degenerate: all observations equal)"
        return _empty(method, conf_level, "Hodges-Lehmann CI undefined: zero variance of the rank statistic.")
    z = float(scipy_stats.norm.isf(alpha / 2))
    ci_low = _step_root(W, cand, z)
    ci_high = _step_root(W, cand, -z)
    method = "normal approximation with tie and continuity correction"
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": float(conf_level),
        "achieved_confidence_level": float(conf_level),
        "method": method,
        "note": _ci_note("location shift (sample 1 minus sample 2)", method, conf_level, None),
    }


# ---------------------------------------------------------------------------
# one sample / paired differences
# ---------------------------------------------------------------------------

def _walsh_averages(x: np.ndarray) -> np.ndarray:
    """All (x_i + x_j) / 2 for i <= j, unsorted."""
    return np.concatenate([(x[i] + x[i:]) / 2.0 for i in range(x.size)])


def hl_one_sample(
    values,
    mu: float = 0.0,
    conf_level: float = 0.95,
    max_pairs: int = MAX_PAIRS,
) -> dict:
    """Pseudomedian of ``values`` with a signed-rank CI.

    Equivalent to ``wilcox.test(values, mu = mu, conf.int = TRUE, ...)``: values
    equal to ``mu`` are dropped (Wilcoxon's rule), and the estimate and interval
    are on the scale of the original values (not shifted by mu).
    """
    _check_conf(conf_level)
    arr = _clean(values, "values")
    nonzero = arr[arr - mu != 0]
    n = int(nonzero.size)
    n_zero = int(arr.size - n)
    if n == 0:
        return _empty("not computed", conf_level, "Hodges-Lehmann estimate undefined: every value equals mu.")
    if n * (n + 1) // 2 > max_pairs:
        return _empty(
            "not computed",
            conf_level,
            (
                f"Hodges-Lehmann pseudomedian not computed: {n * (n + 1) // 2:,} Walsh "
                f"averages exceed the {max_pairs:,}-average memory cap."
            ),
        )

    alpha = 1.0 - conf_level
    walsh = np.sort(_walsh_averages(nonzero))
    estimate = float(np.median(walsh)) + 0.0
    ties = np.unique(np.abs(nonzero - mu)).size != n

    if n < 50 and not ties and n_zero == 0:
        counts = _signrank_counts(n)
        qu = _lower_quantile(counts, alpha / 2)
        if qu == 0:
            qu = 1
        ql = n * (n + 1) // 2 - qu
        achieved = 1.0 - 2.0 * _cdf(counts, qu - 1)
        method = "exact (signed-rank distribution)"
        return {
            "estimate": estimate,
            "ci_low": float(walsh[qu - 1]) + 0.0,
            "ci_high": float(walsh[ql]) + 0.0,
            "confidence_level": float(conf_level),
            "achieved_confidence_level": float(achieved),
            "method": method,
            "note": _ci_note("pseudomedian (median of Walsh averages)", method, conf_level, achieved),
        }

    def W(d: float) -> float:
        xd = nonzero - d
        xd = xd[xd != 0]
        nx = xd.size
        if nx == 0:
            return float("nan")
        dr = scipy_stats.rankdata(np.abs(xd))
        zd = float(dr[xd > 0].sum()) - nx * (nx + 1) / 4.0
        _, tie_counts = np.unique(dr, return_counts=True)
        var = nx * (nx + 1) * (2 * nx + 1) / 24.0 - float(
            np.sum(tie_counts.astype(float) ** 3 - tie_counts)
        ) / 48.0
        if var <= 0:
            return float("nan")
        corr = 0.5 * np.sign(zd)
        return (zd - corr) / np.sqrt(var)

    cand = np.unique(walsh)
    if cand.size < 2 or not np.isfinite(W(float(cand[0]))):
        method = "normal approximation (degenerate: all values equal)"
        return _empty(method, conf_level, "Hodges-Lehmann CI undefined: zero variance of the signed-rank statistic.")
    z = float(scipy_stats.norm.isf(alpha / 2))
    ci_low = _step_root(W, cand, z)
    ci_high = _step_root(W, cand, -z)
    method = "normal approximation with tie and continuity correction"
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": float(conf_level),
        "achieved_confidence_level": float(conf_level),
        "method": method,
        "note": _ci_note("pseudomedian (median of Walsh averages)", method, conf_level, None),
    }


def hl_paired(
    first,
    second,
    conf_level: float = 0.95,
    max_pairs: int = MAX_PAIRS,
) -> dict:
    """Pseudomedian of first - second, ``wilcox.test(first, second, paired = TRUE, conf.int = TRUE)``."""
    a = np.asarray(first, dtype=float).ravel()
    b = np.asarray(second, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("paired samples must have equal length")
    return hl_one_sample(a - b, 0.0, conf_level, max_pairs)
