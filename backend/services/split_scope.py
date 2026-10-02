"""Split File: run an analysis on one level of a grouping variable.

SPSS's SPLIT FILE repeats every procedure per group. Here the browser keeps
the split (variable and the level on view) and states it on each request in
the `X-Ustat-Split` header; this module reads it into a context variable for
the duration of the request, and `store.get_filtered` keeps only that
level's rows. Nothing is stored server-side, so there is nothing to persist,
undo or clean up, and an endpoint that changes data (which reads the store
directly, not through get_filtered) is unaffected by the header.

Order, as in SPSS: missing codes, then Select Cases, then the split, then
Weight Cases.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote

import pandas as pd

from services.number_format import level_key

HEADER = "x-ustat-split"


@dataclass(frozen=True)
class SplitScope:
    column: str
    level: str


current_split: ContextVar[Optional[SplitScope]] = ContextVar("current_split", default=None)


class SplitScopeError(ValueError):
    """The header names a split that cannot be applied."""


def parse_header(raw: Optional[str]) -> Optional[SplitScope]:
    """`X-Ustat-Split` is percent-encoded JSON {"column": ..., "level": ...}:
    header values are Latin-1, column names and levels are not."""
    if not raw:
        return None
    try:
        data = json.loads(unquote(raw))
    except (ValueError, TypeError) as exc:
        raise SplitScopeError("X-Ustat-Split is not valid JSON.") from exc
    column, level = data.get("column"), data.get("level")
    if not isinstance(column, str) or not column or level is None:
        raise SplitScopeError("X-Ustat-Split needs a column and a level.")
    return SplitScope(column=column, level=str(level))


def apply_split(df: pd.DataFrame, scope: SplitScope) -> pd.DataFrame:
    if scope.column not in df.columns:
        raise SplitScopeError(f"Split File column '{scope.column}' not found.")
    keys = df[scope.column].map(lambda v: None if pd.isna(v) else level_key(v))
    return df[(keys == scope.level).to_numpy()]
