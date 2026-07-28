"""Detect dirty / very rare categorical levels before they poison a model.

Typed clinical data often arrives with category typos: a ``sex`` column with
``M``, ``F``, ``""``, ``x`` and ``Female`` is technically 5 levels, and a
naive ``pd.get_dummies`` will silently dummy them all — every rare typo
becomes its own predictor with n<5, which destabilises the fit and hands the
user nonsense coefficients.

This module returns warnings (not errors) the caller can attach to the
response so the user notices what their data looks like.
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


def rare_level_warnings(
    df: pd.DataFrame,
    predictors: List[str],
    *,
    min_rows: int = 5,
) -> List[dict]:
    """Return a per-predictor list of rare-category warnings.

    A warning is raised for a categorical column with ≥3 levels where at
    least one level has fewer than ``min_rows`` observations. The warning
    lists the offending levels and the dominant levels, so the user can
    decide whether the rare ones are typos to recode or genuine.

    Numeric columns are skipped.

    Returns a list of dicts:

        [{"variable": "sex", "rare_levels": [{"level": "x", "n": 1}],
          "kept_levels": [{"level": "M", "n": 55}, {"level": "F", "n": 44}],
          "note": "..."}, ...]
    """
    out: List[dict] = []
    for col in predictors:
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        counts = s.dropna().astype(str).str.strip().value_counts()
        # Need ≥3 effective levels for the dummy-bloat pattern to bite.
        if len(counts) < 3:
            continue
        rare = counts[counts < min_rows]
        if rare.empty:
            continue
        kept = counts[counts >= min_rows]
        out.append({
            "variable": str(col),
            "rare_levels": [{"level": str(lvl), "n": int(n)} for lvl, n in rare.items()],
            "kept_levels": [{"level": str(lvl), "n": int(n)} for lvl, n in kept.items()],
            "note": (
                f"'{col}' has {len(rare)} category(ies) with <{min_rows} rows "
                f"({', '.join(repr(str(l)) for l in rare.index)}). They will "
                f"each become a separate dummy predictor and may destabilise the "
                f"fit. Consider recoding them as missing or merging into the "
                f"dominant levels."
            ),
        })
    return out


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
