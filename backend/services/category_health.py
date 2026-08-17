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

from typing import List

import pandas as pd


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


# ── Moved into the engine ──────────────────────────────────────────────────
# `clean_two_level` decides which rows survive into a two-group test: it folds
# 'M'/'male'/'Male' into one level and drops 'n/a'. That sets n1 and n2, so a
# browser-side copy that folded one spelling differently would test a different
# set of patients and report a confident, wrong result. It therefore lives
# inside the sha256 the two runtimes compare, and is re-exported here so every
# existing caller keeps its import.
#
# `rare_level_warnings` above stayed: nothing in the engine calls it, and
# moving code the engine never runs only widens what the fingerprint covers.
from ustat_engine.frame.category_health import (  # noqa: F401,E402
    CategoryCleanResult,
    clean_two_level,
)
