"""Normalising a two-level categorical before anything is tested on it.

Moved verbatim from `services/category_health.py`; that module re-exports every
name here, so no existing caller changed.

This decides *which rows exist*. `clean_two_level` folds `M`/`male`/`Male` into
one level, treats `n/a` as missing, and reports what it dropped -- so it sets n1
and n2 before the t-test ever sees them. A browser copy that folded one spelling
differently would not produce a rounding difference; it would test a different
number of patients and report a confidently wrong result.

Only the two-level cleaner moved. `rare_level_warnings` stayed behind: it is a
regression-modelling concern, nothing in the engine calls it, and moving code
the engine does not run only widens the surface the fingerprint has to cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


@dataclass
class CategoryCleanResult:
    series: pd.Series
    levels: list
    warnings: List[dict]
    n_dropped: int = 0


_MISSING_TOKENS = {
    "", ".", "-", "--", "?", "na", "n/a", "nan", "null",
    "missing", "unknown", "unk",
}

# "none" used to be in the set above. In clinical data it is far more often a
# real level — chest pain "none", comorbidity "none" — than a stand-in for a
# blank, and treating it as missing deleted the level from the table, the test
# and every model that goes through here, without a word to the user.

# Blanks and punctuation are unambiguous and dropping them silently is what a
# user expects. A word like "unknown" or "n/a" may well be a category the user
# meant to keep, so those get named in a warning instead.
_SILENT_MISSING_TOKENS = {"", ".", "-", "--", "?"}

_SEX_MAP = {
    "f": "Female",
    "female": "Female",
    "woman": "Female",
    "women": "Female",
    "m": "Male",
    "male": "Male",
    "man": "Male",
    "men": "Male",
}

_BINARY_MAP = {
    "0": "0",
    "1": "1",
    "no": "0",
    "n": "0",
    "false": "0",
    "negative": "0",
    "neg": "0",
    "absent": "0",
    "yes": "1",
    "y": "1",
    "true": "1",
    "positive": "1",
    "pos": "1",
    "present": "1",
}


def _stable_levels(s: pd.Series) -> list:
    return sorted(s.dropna().unique().tolist(), key=lambda x: (str(type(x)), str(x)))


def clean_two_level(series: pd.Series, keep: str | Iterable | None = "auto") -> CategoryCleanResult:
    """Normalize obvious binary/case variants while preserving true 3+ level data.

    The helper intentionally only collapses well-known binary spellings:
    ``M/Male`` + ``F/Female`` and common yes/no or 0/1 labels. Stray values
    next to a recognized two-level variable are treated as missing with a
    warning; otherwise extra levels are left intact so callers can keep their
    existing "must have exactly 2 groups" validation.
    """
    raw = series.copy()
    warnings: List[dict] = []

    text = raw.astype("string").str.strip()
    lowered = text.str.casefold()
    token_missing = lowered.isin(_MISSING_TOKENS) & raw.notna()
    missing = raw.isna() | token_missing

    cleaned = text.mask(missing, pd.NA)

    spoken = lowered[token_missing & ~lowered.isin(_SILENT_MISSING_TOKENS)]
    if not spoken.empty:
        counts = spoken.value_counts()
        warnings.append({
            "variable": str(series.name) if series.name is not None else None,
            "dropped_levels": [
                {"level": str(level), "n": int(n)} for level, n in counts.items()
            ],
            "note": (
                f"'{series.name}': {int(counts.sum())} row(s) hold a value that "
                f"reads as missing ({', '.join(repr(str(l)) for l in counts.index)}) "
                "and were excluded from the counts and the test. If any of these "
                "is a real category, recode it before analysing."
            ),
        })

    observed = set(lowered[~missing].dropna().tolist())
    sex_labels = {_SEX_MAP[v] for v in observed if v in _SEX_MAP}
    if (sex_labels == {"Female", "Male"}) or (observed and observed.issubset(set(_SEX_MAP))):
        mapper = _SEX_MAP
    elif observed and observed.issubset(set(_BINARY_MAP)):
        mapper = _BINARY_MAP
    else:
        mapper = {}

    if mapper:
        mapped = lowered.map(mapper).astype("string")
        known = mapped.notna()
        cleaned = mapped.where(known, pd.NA)
        unknown = lowered[~missing & ~known]
        if not unknown.empty:
            counts = unknown.value_counts()
            warnings.append({
                "variable": str(series.name) if series.name is not None else None,
                "dropped_levels": [
                    {"level": str(level), "n": int(n)} for level, n in counts.items()
                ],
                "note": (
                    "Unrecognized values were treated as missing after normalizing "
                    "the two-level variable."
                ),
            })
    else:
        cleaned = cleaned.astype("object")

    if keep not in (None, "auto"):
        keep_set = {str(v) for v in keep}
        keep_mask = cleaned.astype("string").isin(keep_set) | cleaned.isna()
        dropped = cleaned[~keep_mask].astype(str).value_counts()
        if not dropped.empty:
            warnings.append({
                "variable": str(series.name) if series.name is not None else None,
                "dropped_levels": [
                    {"level": str(level), "n": int(n)} for level, n in dropped.items()
                ],
                "note": "Values outside the requested two levels were treated as missing.",
            })
        cleaned = cleaned.where(keep_mask, pd.NA)

    n_dropped = int(cleaned.isna().sum() - raw.isna().sum())
    return CategoryCleanResult(
        series=cleaned,
        levels=_stable_levels(cleaned),
        warnings=warnings,
        n_dropped=max(0, n_dropped),
    )


__all__ = ["CategoryCleanResult", "clean_two_level"]
