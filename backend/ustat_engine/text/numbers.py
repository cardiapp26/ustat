"""The p-value spelling policy, in one place both runtimes read.

Moved verbatim from `services/number_format.py`, which now re-exports these
three names and keeps the rest of its formatters (they are server-side
presentation and nothing in the engine calls them).

Engine results use this same policy for tables and prose.
"""
from __future__ import annotations

import math

DASH = "—"  # em dash for missing


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def format_p(p, *, prefix: bool = False, decimals: int = 3) -> str:
    """Canonical p-value string: '<0.001' or exact 3-decimal (0.035 → '0.035').
    `prefix=True` → 'p<0.001' / 'p=0.035'."""
    if not _finite(p):
        return DASH
    n = float(p)
    if n < 0.001:
        return "p<0.001" if prefix else "<0.001"
    body = f"{n:.{decimals}f}"
    return f"p={body}" if prefix else body


def format_p_clause(p, *, decimals: int = 3) -> str:
    """Write a p-value as part of a sentence."""
    body = format_p(p, decimals=decimals)
    if body.startswith("<"):
        return f"p < {body[1:]}"
    return f"p = {body}"


__all__ = ["DASH", "_finite", "format_p", "format_p_clause"]
