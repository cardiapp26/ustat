"""User-declared missing-value codes: 99 = not recorded, 8 = don't know.

Clinical files mark missing values with numbers. Left alone, a 999 in an age
column is an age of 999 in every mean, model and test. SPSS solves this with
user-missing values, and uSTAT keeps the same two properties:

* the codes stay in the data. The grid shows them, CSV/Excel exports keep
  them and a .sav export declares them user-missing again, so nothing about
  the file is lost;
* every analysis sees them as missing. ``apply_missing_codes`` runs inside
  ``store.get_filtered``, the one path analyses read data through, before the
  case filter, so ``age > 50`` does not select a 999 either.

A column's codes come from its metadata: ``missing_codes`` (set in the Data
Dictionary, any values, matched by ``level_key`` so 99.0 is the code "99"),
and ``missing_ranges`` / ``missing_user_values`` (imported from SPSS / Stata).

``suggest_missing_codes`` only proposes; a 99 can be a real heart rate, so
nothing is marked missing without the user saying so.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services.number_format import level_key

MISSING_CODES_KEY = "missing_codes"


def _range_mask(series: pd.Series, ranges: Any) -> Optional[pd.Series]:
    """Cells inside an SPSS-style list of {lo, hi} ranges or discrete values."""
    if not isinstance(ranges, list) or not ranges:
        return None
    numeric = pd.to_numeric(series, errors="coerce")
    keys = series.map(level_key)
    mask = pd.Series(False, index=series.index)
    for item in ranges:
        if isinstance(item, dict):
            lo, hi = item.get("lo"), item.get("hi", item.get("lo"))
            lo_n = pd.to_numeric(pd.Series([lo]), errors="coerce").iloc[0]
            hi_n = pd.to_numeric(pd.Series([hi]), errors="coerce").iloc[0]
            if pd.notna(lo_n) and pd.notna(hi_n):
                mask |= (numeric >= lo_n) & (numeric <= hi_n)
            elif lo is not None:
                mask |= keys == level_key(lo)
        elif item is not None:
            mask |= keys == level_key(item)
    return mask


def column_missing_mask(series: pd.Series, meta: Optional[dict]) -> Optional[pd.Series]:
    """True where a cell holds one of the column's declared missing codes, or
    None when the column declares none."""
    meta = meta or {}
    codes = meta.get(MISSING_CODES_KEY)
    masks = []
    if isinstance(codes, list) and codes:
        wanted = {level_key(c) for c in codes if c is not None and level_key(c) != ""}
        if wanted:
            masks.append(series.map(level_key).isin(wanted))
    for key in ("missing_ranges", "missing_user_values"):
        m = _range_mask(series, meta.get(key))
        if m is not None:
            masks.append(m)
    if not masks:
        return None
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out & series.notna()


def _restore_numeric(series: pd.Series) -> pd.Series:
    """A text code ("UNK") keeps a numeric column as text; once masked, the
    rest may be numbers again."""
    if series.dtype != object:
        return series
    parsed = pd.to_numeric(series, errors="coerce")
    return parsed if parsed.notna().sum() == series.notna().sum() else series


def apply_missing_codes(df: pd.DataFrame, metadata: Optional[Dict[str, dict]]) -> pd.DataFrame:
    """``df`` with every declared missing code set to NaN. Returns ``df``
    itself (no copy) when no column has a code present."""
    if df is None or not metadata:
        return df
    masked: Dict[str, pd.Series] = {}
    for col, meta in metadata.items():
        if col not in df.columns or not isinstance(meta, dict):
            continue
        mask = column_missing_mask(df[col], meta)
        if mask is None or not mask.any():
            continue
        masked[col] = _restore_numeric(df[col].mask(mask))
    if not masked:
        return df
    out = df.copy()
    for col, series in masked.items():
        out[col] = series
    return out


def missing_code_counts(df: pd.DataFrame, metadata: Optional[Dict[str, dict]]) -> Dict[str, int]:
    """How many cells each column's declared codes turn into missing."""
    counts: Dict[str, int] = {}
    for col, meta in (metadata or {}).items():
        if col not in df.columns or not isinstance(meta, dict):
            continue
        mask = column_missing_mask(df[col], meta)
        if mask is not None and mask.any():
            counts[col] = int(mask.sum())
    return counts


# Conventional sentinel codes in clinical and survey data. A value qualifies
# only when it is also detached from the rest of the column (see below).
_SENTINELS = (99, 999, 9999, 99999, -9, -99, -999, -9999, 88, 888, 8888, 77, 777, 98, 998, 97, 997)
# Survey-style small codes (8 = don't know, 9 = no answer) on short scales.
_SMALL_SENTINELS = (8, 9)
_SMALL_SCALE_MAX = 6
_DETACHED_FACTOR = 1.5


def suggest_missing_codes(series: pd.Series) -> List[dict]:
    """Values that look like missing codes: a conventional sentinel that sits
    far outside everything else in the column. 999 among ages 18 to 95 is
    proposed; 99 among ages up to 98 is not, because it could be real."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return []
    counts = numeric.value_counts()
    out: List[dict] = []
    integer_valued = bool(np.all(np.mod(numeric.to_numpy(), 1) == 0))
    for code in _SENTINELS + _SMALL_SENTINELS:
        if code not in counts.index:
            continue
        rest = numeric[numeric != code]
        rest = rest[~rest.isin(_SENTINELS + _SMALL_SENTINELS)]
        if rest.empty:
            continue
        if code in _SMALL_SENTINELS:
            small_scale = (integer_valued and rest.nunique() <= _SMALL_SCALE_MAX + 1
                           and rest.min() >= 0 and rest.max() <= _SMALL_SCALE_MAX)
            if not (small_scale and code > rest.max()):
                continue
            reason = (f"{code} lies beyond a {int(rest.min())} to {int(rest.max())} scale, "
                      "a common code for don't know / no answer")
        elif code > 0:
            if not (rest.max() > 0 and code > rest.max() * _DETACHED_FACTOR):
                continue
            reason = f"{code} is far above every other value (largest {rest.max():g})"
        else:
            if not (code < rest.min() and (rest.min() >= 0 or code < rest.min() * _DETACHED_FACTOR)):
                continue
            reason = f"{code} is far below every other value (smallest {rest.min():g})"
        out.append({"value": level_key(code), "count": int(counts[code]), "reason": reason})
    return out


__all__ = [
    "MISSING_CODES_KEY",
    "apply_missing_codes",
    "column_missing_mask",
    "missing_code_counts",
    "suggest_missing_codes",
]
