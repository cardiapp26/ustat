"""Fragments every generated sentence is built out of.

Moved verbatim from `services/text_generators.py`, which re-exports them. They
are shared across analyses -- an effect size renders the same way in a t-test
paragraph and an ANOVA one -- so they sit beside the generators rather than
inside any single family's module.
"""
from __future__ import annotations

from .numbers import format_p


def _p_str(p: float) -> str:
    return format_p(p)


def _es_str(es: dict) -> str:
    """Format effect size dict as inline text."""
    name = es.get("name", "").replace("_", " ")
    val = es.get("value", 0)
    mag = es.get("magnitude", "")
    ci_lo = es.get("ci_low")
    ci_hi = es.get("ci_high")
    s = f"{name} = {val:.3f}"
    if ci_lo is not None and ci_hi is not None:
        s += f" (95% CI: {ci_lo:.3f}–{ci_hi:.3f})"
    if mag:
        s += f" [{mag}]"
    return s


def _df_str(df) -> str:
    """Whole df prints bare; a Welch–Satterthwaite df keeps 2 decimals."""
    if not isinstance(df, (int, float)):
        return str(df)
    f = float(df)
    if f != f:  # NaN
        return ""
    return str(int(f)) if float(f).is_integer() else f"{f:.2f}"


__all__ = ["_df_str", "_es_str", "_p_str"]
