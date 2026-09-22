"""Which order a categorical column's levels run in, low to high.

Order-aware methods (ordinal logistic regression, Jonckheere-Terpstra,
Cochran-Armitage) take the order as part of the hypothesis: sort Poor / Fair /
Good alphabetically and the thresholds, or the trend, come out reversed with
nothing on screen to say so. This module is the one place that decides the
order, so every such method reads the same answer from the same sources:

1. an order the request states explicitly;
2. the order the user set in the Data Dictionary (metadata key
   ``level_order``, a list, so the .ustat file's sorted-key JSON keeps it);
3. numeric codes, sorted as numbers;
4. a recognised grading vocabulary (mild < moderate < severe, ...).

Anything else has no known order and ``resolve_level_order`` returns None;
the caller decides whether to refuse or to fall back and say so.

Levels are matched through ``level_key``, the same canonical spelling the
Data Dictionary keys its inputs by, so a float64 code 1.0 and the dictionary's
"1" are the same level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from fastapi import HTTPException

from services import store
from services.number_format import level_key

SOURCE_REQUEST = "request"
SOURCE_DICTIONARY = "data dictionary"
SOURCE_NUMERIC = "numeric value"
SOURCE_RECOGNISED = "recognised ordinal labels"

# Grading vocabularies seen in clinical data, lowest first. A column matches a
# vocabulary only when every one of its levels is in it.
ORDINAL_VOCABULARIES: List[List[str]] = [
    ["none", "mild", "moderate", "severe"],
    ["none", "low", "medium", "high"],
    ["low", "medium", "high"],
    ["low", "mid", "high"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["absent", "mild", "moderate", "marked"],
    ["i", "ii", "iii", "iv"],
    ["grade 1", "grade 2", "grade 3", "grade 4"],
    ["mild", "moderate", "severe"],
    ["small", "medium", "large"],
    ["negative", "equivocal", "positive"],
    ["very poor", "poor", "fair", "good", "very good", "excellent"],
    # Turkish, the primary user base's own labels.
    ["yok", "hafif", "orta", "ağır"],
    ["yok", "hafif", "orta", "şiddetli"],
    ["düşük", "orta", "yüksek"],
    ["çok kötü", "kötü", "orta", "iyi", "çok iyi"],
]


def _norm(label) -> str:
    # casefold() maps "I" to "i" but leaves the Turkish dotless "ı" alone, so
    # "AĞIR" and "ağır" would otherwise normalise differently.
    return str(label).strip().casefold().replace("ı", "i")


def recognised_ranks(levels: Sequence) -> Optional[List[int]]:
    """Rank each level by a known grading vocabulary, or None if unrecognised."""
    lowered = [_norm(lv) for lv in levels]
    if len(set(lowered)) != len(lowered):
        return None
    for vocab in ORDINAL_VOCABULARIES:
        normed = [_norm(w) for w in vocab]
        if all(w in normed for w in lowered):
            return [normed.index(w) for w in lowered]
    return None


@dataclass(frozen=True)
class LevelOrder:
    """Levels low to high: ``levels`` are values as they occur in the data,
    ``keys`` their ``level_key`` spellings, ``source`` where the order came
    from (one of the SOURCE_* constants)."""

    levels: tuple
    keys: tuple
    source: str


def dictionary_order(session_id: str, column: str) -> Optional[List[str]]:
    """The low-to-high order set for ``column`` in the Data Dictionary."""
    meta = (store.get_metadata(session_id) or {}).get(column) or {}
    order = meta.get("level_order")
    if not isinstance(order, list) or not order:
        return None
    return [level_key(v) for v in order]


def _present_levels(values: Iterable) -> dict:
    """level_key -> first raw value with that key, in order of appearance."""
    by_key: dict = {}
    for v in pd.Series(list(values), dtype=object).dropna():
        k = level_key(v)
        if k != "":
            by_key.setdefault(k, v)
    return by_key


def _follow(by_key: dict, order: Sequence[str], column: str, where: str) -> tuple:
    keys = [level_key(o) for o in order]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise HTTPException(
            status_code=422,
            detail=f"The {where} for '{column}' lists {dupes} more than once.",
        )
    unplaced = [k for k in by_key if k not in keys]
    if unplaced:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The {where} for '{column}' does not place {unplaced}, which the "
                f"data contains. It must list every level low to high; update it "
                f"and run again."
            ),
        )
    # Levels the order names but the (filtered) data lacks are simply absent.
    return tuple(k for k in keys if k in by_key)


def resolve_level_order(
    values: Iterable,
    column: str,
    *,
    session_id: Optional[str] = None,
    explicit: Optional[Sequence] = None,
) -> Optional[LevelOrder]:
    """Low-to-high order of the levels in ``values``, or None if unknown.

    Raises 422 when an explicit or dictionary order exists but does not cover
    every level present: a stale order is a conflict to surface, not a reason
    to fall through to a guess.
    """
    by_key = _present_levels(values)

    def _make(keys: tuple, source: str) -> LevelOrder:
        return LevelOrder(tuple(by_key[k] for k in keys), keys, source)

    if explicit is not None:
        return _make(_follow(by_key, explicit, column, "requested level order"), SOURCE_REQUEST)

    if session_id is not None:
        stored = dictionary_order(session_id, column)
        if stored is not None:
            return _make(_follow(by_key, stored, column, "Data Dictionary category order"), SOURCE_DICTIONARY)

    keys = list(by_key)
    numeric = pd.to_numeric(pd.Series(keys, dtype=object), errors="coerce")
    if len(keys) and numeric.notna().all():
        ranked = [k for _, k in sorted(zip(numeric.tolist(), keys))]
        return _make(tuple(ranked), SOURCE_NUMERIC)

    ranks = recognised_ranks([by_key[k] for k in keys])
    if ranks is not None:
        return _make(tuple(k for _, k in sorted(zip(ranks, keys))), SOURCE_RECOGNISED)
    return None


__all__ = [
    "LevelOrder",
    "ORDINAL_VOCABULARIES",
    "SOURCE_DICTIONARY",
    "SOURCE_NUMERIC",
    "SOURCE_RECOGNISED",
    "SOURCE_REQUEST",
    "dictionary_order",
    "level_key",
    "recognised_ranks",
    "resolve_level_order",
]
