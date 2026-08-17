"""How a categorical column's levels are ordered and spelled.

Both of these used to live in `services/` (`stat_utils.sorted_groups`,
`number_format.level_key`), which the engine may not import -- and both decide
something a browser-side run has to get byte-identical to the server: which
group comes first in a table, and what string a code is keyed by. A second
implementation of either would produce answers that are numerically right and
still disagree with the server's, row for row.

So they live here, and `services.stat_utils` / `services.number_format` re-export
them. Nothing else changes: every existing caller keeps its import.
"""
from __future__ import annotations

import math

import pandas as pd


def sorted_groups(series: "pd.Series") -> list:
    """Stable, value-code order for grouped output (Table 1 columns, ANOVA /
    t-test / Kruskal group rows, crosstab levels, KM curves, etc.).

    Sort by the underlying value code numerically when every distinct value is
    numeric-coercible, else lexicographically by string. Without this, groups
    follow their order of appearance in the data, so results come out scrambled
    relative to the value labels (e.g. 3, 1, 2 instead of 1, 2, 3).
    """
    vals = list(pd.Series(series).dropna().unique())
    try:
        return sorted(vals, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def level_key(value) -> str:
    """Canonical string for a category code, matching the frontend.

    Value labels are keyed by whatever string the UI showed when they were
    typed, and the grid renders a JSON number — so a code of zero is the key
    "0". Python's ``str`` on the same code read from a float64 column gives
    "0.0", and a column labelled through the grid then displayed its raw codes
    everywhere the levels came from the server. Worse, the Data Dictionary
    editor keyed its inputs off that second spelling, so labels typed there
    landed under keys the rest of the app never looked up.

    A whole number loses its ".0"; everything else is left alone. Non-numeric
    values pass through with surrounding whitespace trimmed, and a missing
    value has no code, so it returns the empty string for the caller to
    replace with whatever it calls missing.
    """
    if value is None:
        return ""
    try:
        if isinstance(value, bool):
            # bool is an int subclass; "True"/"False" is what the grid shows.
            return str(value)
        n = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if not math.isfinite(n):
        return ""
    if n.is_integer():
        return str(int(n))
    return str(value).strip()


__all__ = ["level_key", "sorted_groups"]
