"""The p-value spelling policy, in one place both runtimes read.

Moved verbatim from `services/number_format.py`, which now re-exports these
three names and keeps the rest of its formatters (they are server-side
presentation and nothing in the engine calls them).

Only `format_p` and its two supports moved, because only they are reached from
a result the engine returns: `results_ttest_ind` prints `p = <0.001` through
this function. A second copy in the browser that rounded to `0.000` at the
boundary would contradict the `p` field sitting beside it in the same payload.
"""
from __future__ import annotations

import math

DASH = "—"  # em dash for missing


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def format_p(p, *, prefix: bool = False) -> str:
    """Canonical p-value string: '<0.001' or exact 3-decimal (0.035 → '0.035').
    `prefix=True` → 'p<0.001' / 'p=0.035'."""
    if not _finite(p):
        return DASH
    n = float(p)
    if n < 0.001:
        return "p<0.001" if prefix else "<0.001"
    body = f"{n:.3f}"
    return f"p={body}" if prefix else body


__all__ = ["DASH", "_finite", "format_p"]
