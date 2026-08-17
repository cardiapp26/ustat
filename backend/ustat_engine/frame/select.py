"""Select Cases: which rows an analysis is allowed to see.

This is the same code the server has always run -- it was `validate_conditions`
and `_apply_conditions` in `services/store.py` -- moved here unchanged so the
browser runs it too.

Moving it matters for a reason beyond tidiness. The engine's sources are hashed
into the sha256 fingerprint that the browser and the server compare before any
local run is allowed. With the filter inside the engine, the filter semantics
are inside that hash: a browser whose idea of "eq on a numeric column that
might be text" differed from the server's would fail the fingerprint check and
could not run at all. Left in `services/`, it would be the one part of "which
patients are in this analysis" that the two runtimes could disagree about
silently, and disagreeing about the denominator is worse than disagreeing about
the test.

`keep_mask` is the single source of the semantics; `apply_conditions` is
literally `df[keep_mask(df, conditions)]`. Callers that need positions rather
than rows (the grid marks excluded rows instead of hiding them) take the mask.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..errors import EngineError

VALID_FILTER_OPERATORS = {
    "eq",
    "ne",
    "gt",
    "lt",
    "gte",
    "lte",
    "missing",
    "not_missing",
    "contains",
}


def validate_conditions(df: pd.DataFrame, conditions: List[dict]) -> None:
    for i, cond in enumerate(conditions or [], start=1):
        col = cond.get("column", "")
        if col not in df.columns:
            raise EngineError(f"Condition {i}: column '{col}' not found", status_hint=404)
        op = cond.get("operator", "eq")
        if op not in VALID_FILTER_OPERATORS:
            allowed = ", ".join(sorted(VALID_FILTER_OPERATORS))
            raise EngineError(
                f"Condition {i}: unsupported operator '{op}'. Use one of: {allowed}.",
                status_hint=422,
            )


def keep_mask(df: pd.DataFrame, conditions: List[dict]) -> np.ndarray:
    """Boolean array over the UNFILTERED frame's row POSITIONS.

    Positional rather than index-labelled on purpose: the grid addresses rows by
    position (it marks excluded rows rather than hiding them, because a cell
    edit names a position in the full sheet), and a mask carrying the frame's
    index would have to be re-aligned at every such call site.
    """
    if not conditions:
        return np.ones(len(df), dtype=bool)

    mask = pd.Series([True] * len(df), index=df.index)
    for i, cond in enumerate(conditions):
        col = cond.get("column", "")
        if col not in df.columns:
            continue
        op = cond.get("operator", "eq")
        val = cond.get("value", "")
        join = cond.get("join", "AND")

        if op == "missing":
            cond_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        elif op == "not_missing":
            cond_mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
        elif op == "contains":
            cond_mask = df[col].astype(str).str.contains(str(val), case=False, na=False)
        else:
            # Try numeric comparison first, fall back to string
            try:
                num_val = float(val)
                s = pd.to_numeric(df[col], errors="coerce")
                if op == "eq":
                    cond_mask = s == num_val
                elif op == "ne":
                    cond_mask = s != num_val
                elif op == "gt":
                    cond_mask = s > num_val
                elif op == "lt":
                    cond_mask = s < num_val
                elif op == "gte":
                    cond_mask = s >= num_val
                elif op == "lte":
                    cond_mask = s <= num_val
                else:
                    cond_mask = pd.Series([True] * len(df), index=df.index)
            except (ValueError, TypeError):
                s = df[col].astype(str)
                if op == "eq":
                    cond_mask = s == str(val)
                elif op == "ne":
                    cond_mask = s != str(val)
                else:
                    cond_mask = pd.Series([True] * len(df), index=df.index)

        if i == 0 or join == "AND":
            mask = mask & cond_mask
        else:
            mask = mask | cond_mask

    return mask.to_numpy(dtype=bool)


def apply_conditions(df: pd.DataFrame, conditions: List[dict]) -> pd.DataFrame:
    """The rows `conditions` keeps, in order.

    The empty case returns `df` itself rather than a copy of every row: an
    unfiltered session asks for this on every analysis, and callers have always
    got the store's own frame back when no filter was set.
    """
    if not conditions:
        return df
    return df[keep_mask(df, conditions)]


__all__ = [
    "VALID_FILTER_OPERATORS",
    "apply_conditions",
    "keep_mask",
    "validate_conditions",
]
