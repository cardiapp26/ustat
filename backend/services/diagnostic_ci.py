"""Confidence intervals for diagnostic accuracy measures (pure functions).

* Wilson score interval for a binomial proportion (sensitivity, specificity,
  PPV, NPV, accuracy).
* Simel, Samsa and Matchar (1991) log-method interval for likelihood ratios.
* Fagan post-test probability from a pre-test probability and an LR.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from scipy.stats import norm

Interval = Tuple[float, float]


def _z(alpha: float) -> float:
    return float(norm.ppf(1.0 - alpha / 2.0))


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Optional[Interval]:
    """Wilson score interval for ``successes / n``; None when ``n <= 0``."""
    if n is None or n <= 0 or successes < 0 or successes > n:
        return None
    z = _z(alpha)
    z2 = z * z
    p = successes / n
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    low = max(0.0, centre - half)
    high = min(1.0, centre + half)
    return (low, high)


def simel_lr_ci(
    tp: int, fn: int, fp: int, tn: int, which: str = "pos", alpha: float = 0.05
) -> Optional[Interval]:
    """Simel et al. (1991) log-method CI for LR+ (``which='pos'``) or LR- (``'neg'``).

    se(ln LR+) = sqrt(1/tp - 1/(tp+fn) + 1/fp - 1/(fp+tn))
    se(ln LR-) = sqrt(1/fn - 1/(tp+fn) + 1/tn - 1/(fp+tn))

    A 0.5 continuity correction is added to every cell when any cell is 0.
    Returns None when the interval is undefined (an empty diseased or
    non-diseased group).
    """
    if which not in {"pos", "neg"}:
        raise ValueError("which must be 'pos' or 'neg'")
    if min(tp, fn, fp, tn) < 0:
        return None
    if (tp + fn) <= 0 or (fp + tn) <= 0:
        return None
    a, b, c, d = float(tp), float(fn), float(fp), float(tn)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    diseased = a + b
    healthy = c + d
    sens = a / diseased
    spec = d / healthy
    z = _z(alpha)
    if which == "pos":
        lr = sens / (1.0 - spec)
        se = math.sqrt(1.0 / a - 1.0 / diseased + 1.0 / c - 1.0 / healthy)
    else:
        lr = (1.0 - sens) / spec
        se = math.sqrt(1.0 / b - 1.0 / diseased + 1.0 / d - 1.0 / healthy)
    if not (lr > 0.0) or not math.isfinite(lr) or not math.isfinite(se):
        return None
    ln_lr = math.log(lr)
    return (math.exp(ln_lr - z * se), math.exp(ln_lr + z * se))


def fagan_post_test(
    pretest: float, sens: float, spec: float
) -> Optional[dict]:
    """Post-test probabilities (odds * LR) for a positive and a negative test.

    Returns None when ``pretest`` is outside (0, 1).
    """
    if pretest is None or not (0.0 < pretest < 1.0):
        return None
    odds = pretest / (1.0 - pretest)

    def _prob(lr: float) -> float:
        if math.isinf(lr):
            return 1.0
        post_odds = odds * lr
        return post_odds / (1.0 + post_odds)

    lr_pos = sens / (1.0 - spec) if (1.0 - spec) > 0 else (
        math.inf if sens > 0 else 0.0
    )
    lr_neg = (1.0 - sens) / spec if spec > 0 else (
        math.inf if (1.0 - sens) > 0 else 0.0
    )
    return {
        "pretest": float(pretest),
        "post_positive": _prob(lr_pos),
        "post_negative": _prob(lr_neg),
    }
