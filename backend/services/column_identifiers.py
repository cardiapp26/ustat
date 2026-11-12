"""Identifier heuristics shared by session metadata and model validation."""

import re

import pandas as pd

_ID_NAME_TOKENS = frozenset({"id", "no", "num", "number", "patient", "subject", "case", "record", "mrn"})
_ID_MIN_LEVELS = 10
_ID_MIN_UNIQUE_RATIO = 0.95


def is_id_like(name: str, series: pd.Series) -> bool:
    """Identify high-cardinality ID names and nearly unique integer/text codes."""
    if len(series) < 5:
        return False
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    tokens = re.findall(r"[a-z0-9]+", spaced.lower())
    named_id = bool(tokens and tokens[-1] in _ID_NAME_TOKENS)
    code_like = pd.api.types.is_integer_dtype(series) or not pd.api.types.is_numeric_dtype(series)
    if not (named_id or code_like):
        return False
    n = int(series.count())
    if n < 5:
        return False
    try:
        n_levels = int(series.nunique())
    except TypeError:
        return False
    return (named_id and n_levels >= _ID_MIN_LEVELS) or (code_like and n_levels / n > _ID_MIN_UNIQUE_RATIO)
