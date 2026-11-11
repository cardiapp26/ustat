"""Design-matrix encoding shared by the Cox and RCS endpoints.

One place decides which columns are numeric and which are dummy-coded, and
records which design columns each source column produced. Two mistakes this
replaces:

- `pd.get_dummies(..., dummy_na=False)` gives a row with a missing category
  zeros in every indicator, which is the coding of the reference level. The
  later `dropna()` then kept the row, so missing values were analysed as the
  reference category. Here those rows carry NaN in the indicators and are
  dropped with the other incomplete rows.
- Looking up a predictor's dummy columns by the prefix "name_" also matched
  unrelated columns ("stage" matched "stage_score"). `members` is the explicit
  map instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A text column counts as numeric when at least this share of its values
# parse as numbers and it has more than two distinct numeric values.
NUMERIC_SHARE = 0.8


@dataclass(frozen=True)
class EncodedDesign:
    """Encoded columns plus the design columns each source column produced."""

    frame: pd.DataFrame
    members: dict[str, list[str]]

    def columns_of(self, name: str) -> list[str]:
        return list(self.members.get(name, []))


def _as_numeric(col: pd.Series) -> pd.Series | None:
    """The column as floats when it is numeric or mostly numeric text."""
    if pd.api.types.is_bool_dtype(col):
        return col.astype(float)
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce").astype(float)
    coerced = pd.to_numeric(col, errors="coerce")
    if coerced.notna().mean() >= NUMERIC_SHARE and coerced.dropna().nunique() > 2:
        return coerced.astype(float)
    return None


def _dummies(col: pd.Series, name: str) -> pd.DataFrame:
    """Reference-coded indicators; rows with a missing category stay NaN."""
    out = pd.get_dummies(col, prefix=name, drop_first=True, dummy_na=False).astype(float)
    out.loc[col.isna().to_numpy(), :] = np.nan
    return out


def _level_label(value: object) -> object:
    """Integer-valued codes as "2", not "2.0", so dummy names read grp_2."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


def encode_design(
    df: pd.DataFrame,
    cols: list[str],
    categorical: frozenset[str] = frozenset(),
) -> EncodedDesign:
    """Encode `cols` of `df`: numeric columns first, then dummy-coded ones.

    `categorical` names columns declared categorical (kind override or value
    labels). With more than two levels they are dummy-coded even when stored
    as numbers, so groups coded 1/2/3 do not enter as one linear term.
    """
    numeric_parts: list[pd.DataFrame] = []
    dummy_parts: list[pd.DataFrame] = []
    members: dict[str, list[str]] = {}
    for name in dict.fromkeys(cols):
        col = df[name]
        declared = name in categorical and col.dropna().nunique() > 2
        if declared and pd.api.types.is_numeric_dtype(col):
            col = col.astype(object).map(_level_label, na_action="ignore")
        numeric = None if declared else _as_numeric(col)
        if numeric is not None:
            numeric_parts.append(numeric.to_frame(name))
            members[name] = [name]
        else:
            part = _dummies(col, name)
            dummy_parts.append(part)
            members[name] = list(part.columns)
    parts = numeric_parts + dummy_parts
    frame = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return EncodedDesign(frame=frame, members=members)
