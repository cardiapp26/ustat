"""Helpers for obvious numeric sentinels in clinical columns.

The upload pipeline normalises common locale decimals, but downstream code can
still see classic missing-code sentinels such as BMI=999.  These helpers keep
the policy central and conservative: values are flagged only when they are both
outside a caller-provided plausibility range and extreme relative to the body of
the column.
"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np
import pandas as pd


PLAUSIBLE_MAX_BY_NAME = {
    "age": 120.0,
    "bmi": 100.0,
    "body_mass_index": 100.0,
}


def plausibility_max_for_column(name: str | None) -> Optional[float]:
    if not name:
        return None
    key = str(name).strip().lower()
    if key in PLAUSIBLE_MAX_BY_NAME:
        return PLAUSIBLE_MAX_BY_NAME[key]
    if "bmi" in key:
        return PLAUSIBLE_MAX_BY_NAME["bmi"]
    if key in {"fu_days", "followup_days", "follow_up_days"}:
        return None
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Numeric coercion with support for simple comma decimals."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip()
    comma_decimal = text.str.match(r"^[+-]?\d+,\d+$", na=False)
    text = text.mask(comma_decimal, text.str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")


def flag_sentinels(series: pd.Series, max_plausible: Optional[float]) -> pd.Series:
    """Return a boolean mask for obvious high-side sentinel values.

    A binary or sparsely-coded column (say two 1s in 382 rows of 0) has a
    body whose 99th percentile is itself 0: fewer than 1% of rows are
    nonzero, so even the top percentile lands on the majority value. The old
    robust_high = q99 * 1.5 then evaluated to 0, and every genuine 1 in the
    column cleared that "threshold" and was flagged as an implausible
    sentinel — reported as missing data it never was. Nothing about the
    IQR-based rule is wrong for a continuous column with real spread; it
    breaks specifically when the body has none, because there is no scale
    left to call anything "extreme relative to" the body.

    q99 == 0 (with iqr == 0, since q99 >= q3) is exactly that no-spread case,
    and it is the only one where the fallback multiplies by zero. There, no
    caller-supplied ceiling means no basis to flag anything at all.
    """
    mask = pd.Series(False, index=series.index)
    numeric = coerce_numeric(series)
    observed = numeric.dropna()
    if observed.empty:
        return mask

    body = observed
    if max_plausible is not None:
        body = observed[observed <= max_plausible]
    if body.empty:
        body = observed

    q1 = float(body.quantile(0.25))
    q3 = float(body.quantile(0.75))
    iqr = q3 - q1
    q99 = float(body.quantile(0.99))
    if iqr > 0:
        robust_high = q99 + 5.0 * iqr
    elif q99 > 0:
        robust_high = q99 * 1.5
    else:
        robust_high = None  # body has no spread to measure "extreme" against

    if max_plausible is not None:
        threshold = max_plausible if robust_high is None else max(max_plausible, robust_high)
    else:
        threshold = robust_high

    if threshold is None:
        return mask

    return (numeric > threshold).fillna(False)


def sentinel_values(series: pd.Series, max_plausible: Optional[float]) -> Set[float]:
    numeric = coerce_numeric(series)
    vals = numeric[flag_sentinels(series, max_plausible)].dropna().unique()
    return {float(v) for v in vals}


def mask_sentinels(series: pd.Series, max_plausible: Optional[float]) -> pd.Series:
    numeric = coerce_numeric(series)
    return numeric.mask(flag_sentinels(series, max_plausible), np.nan)


def values_are_numeric(series: pd.Series) -> bool:
    """Does every value present in this column parse as a number?

    Not the same question as `is_numeric_dtype`. A column added after upload
    arrives as object and everything typed into it is stored as text, so a
    column holding nothing but numbers answers False to the dtype question and
    True to this one — which is the question the grid's range badge and the
    cell writer both actually need.

    An empty column answers True: there is nothing in it to contradict a
    number, so the first value typed decides what it is. One genuine word
    answers False, and keeps it false.
    """
    present = series.dropna()
    if present.dtype == object:
        # A cell holding "" or "   " is a gap on screen, not a value.
        present = present[present.astype(str).str.strip() != ""]
    if len(present) == 0:
        return True
    return bool(pd.to_numeric(present, errors="coerce").notna().all())
