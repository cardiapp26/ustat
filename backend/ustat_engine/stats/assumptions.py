"""Normality and equal-variance checks, as the app words them.

Moved verbatim from `services/stat_utils.py`; `services.stat_utils` re-exports
both names so every existing caller is untouched.

These are not decoration. `check_equal_variances` is what decides Student vs
Welch when the caller asked for "auto", so a browser run whose Levene differed
from the server's would not merely print a different sentence -- it would run a
different test and report a different p. The tiering inside `check_normality`
(Shapiro under 50, Lilliefors to 2000, a skewness bypass above that) has the
same property for the assumption line that is printed next to the result.

statsmodels is imported inside the function, exactly as it was: the small-sample
Shapiro path never needs it, and in the browser an unconditional import would
charge every t-test for a package most of them do not reach.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp


def check_normality(x: np.ndarray, label: str = "Sample") -> dict:
    """Test normality using the appropriate test for sample size.

    Tier 1: n < 50  → Shapiro-Wilk (most powerful for small samples)
    Tier 2: 50 ≤ n ≤ 2000 → Kolmogorov-Smirnov with Lilliefors correction
    Tier 3: n > 2000 → CLT skewness bypass (|skew| ≤ 1.5) → Lilliefors
    """
    n = len(x)
    if n < 3:
        return {
            "name": f"Normality ({label})",
            "met": True,
            "detail": "Too few obs to test",
        }
    if np.std(x) == 0:
        return {
            "name": f"Normality ({label})",
            "met": True,
            "detail": "Constant values (no variation)",
        }

    if n < 50:
        # Small sample — Shapiro-Wilk is most powerful
        stat, p = sp.shapiro(x)
        if np.isnan(p):
            return {
                "name": f"Normality ({label})",
                "met": True,
                "detail": "Test inconclusive",
            }
        test_name = "Shapiro-Wilk"
    elif n <= 2000:
        # Medium sample — Kolmogorov-Smirnov with Lilliefors correction
        from statsmodels.stats.diagnostic import lilliefors as _lf

        stat, p = _lf(x, dist="norm")
        test_name = "Kolmogorov-Smirnov (Lilliefors)"
    else:
        # Large sample — CLT bypass if skewness is acceptable
        skew = float(sp.skew(x))
        if abs(skew) <= 1.5:
            return {
                "name": f"Normality ({label})",
                "met": True,
                "detail": f"CLT bypass (n={n}, |skewness|={abs(skew):.2f} ≤ 1.5)",
            }
        from statsmodels.stats.diagnostic import lilliefors as _lf

        stat, p = _lf(x, dist="norm")
        test_name = "Kolmogorov-Smirnov (Lilliefors)"

    return {
        "name": f"Normality ({label})",
        "met": bool(p >= 0.05),
        "detail": f"{test_name}: p = {p:.4f}",
    }


def check_equal_variances(
    groups: list[np.ndarray],
    names: list[str],
    on_violation: str = "",
) -> dict:
    """Levene's test for homogeneity of variances.

    `on_violation` states what the CALLER actually does when the assumption
    fails. It used to be hardcoded to "using Welch correction", which was true
    for the t-test but false for one-way ANOVA — that omnibus F stays the
    classic equal-variance statistic and only the post-hoc switches. Reporting
    a correction that was never applied is a false methods claim, so each
    caller now supplies its own wording.
    """
    if len(groups) < 2:
        return {"name": "Equal variances", "met": True, "detail": "Single group"}
    stat, p = sp.levene(*groups)
    violated = p < 0.05
    suffix = f" — violated, {on_violation}" if (violated and on_violation) else (
        " — violated" if violated else ""
    )
    return {
        "name": "Equal variances (Levene)",
        "met": bool(not violated),
        "detail": f"F = {stat:.3f}, p = {p:.4f}{suffix}",
    }


__all__ = ["check_equal_variances", "check_normality"]
