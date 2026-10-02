"""Weight Cases: SPSS frequency weights, applied once for every analysis.

A frequency weight says how many identical cases a row stands for (aggregated
data: one row per covariate pattern with a count). Replicating each row that
many times is then exactly the data the weights describe, so applying the
weight inside `store.get_filtered` makes every analysis, every N and every
test weighted in the same way, as SPSS's WEIGHT BY does.

That equivalence holds only for whole-number frequencies. Sampling or survey
weights (fractional, with strata and clusters) are a different thing: they
need design-based variance, and replicating rows would give wrong standard
errors with the right-looking N. Those are refused here, not approximated.

Cases with a missing, zero or negative weight are left out, as in SPSS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Upper bound on the replicated frame, to keep one session's memory bounded.
MAX_WEIGHTED_ROWS = 2_000_000


class CaseWeightError(ValueError):
    """The weight column cannot be used as frequency weights."""


@dataclass(frozen=True)
class WeightSummary:
    column: str
    n_rows: int
    n_excluded: int
    sum_weights: int


def _weights(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise CaseWeightError(f"Weight column '{column}' not found.")
    w = pd.to_numeric(df[column], errors="coerce")
    text_values = df[column].notna() & w.isna()
    if text_values.any():
        raise CaseWeightError(
            f"'{column}' holds non-numeric values; frequency weights must be numbers."
        )
    positive = w[w > 0]
    if (positive != np.round(positive)).any():
        raise CaseWeightError(
            f"'{column}' holds fractional weights. Weight Cases uses frequency weights "
            "(whole-number counts); sampling or survey weights need design-based "
            "methods and are not applied by replication."
        )
    return w


def summarize(df: pd.DataFrame, column: str) -> WeightSummary:
    w = _weights(df, column)
    used = w > 0
    total = int(w[used].sum())
    if total == 0:
        raise CaseWeightError(f"'{column}' has no positive weights.")
    if total > MAX_WEIGHTED_ROWS:
        raise CaseWeightError(
            f"The weights in '{column}' add up to {total:,} cases, above the "
            f"{MAX_WEIGHTED_ROWS:,} uSTAT can replicate."
        )
    return WeightSummary(column=column, n_rows=int(used.sum()),
                         n_excluded=int((~used).sum()), sum_weights=total)


def apply_weights(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Each row repeated weight times; missing, zero and negative weights dropped.

    The index labels are kept (repeated), so a label still names the stored
    row it came from.
    """
    summarize(df, column)  # validates against the data as it is now
    w = pd.to_numeric(df[column], errors="coerce")
    keep = (w > 0).to_numpy()
    counts = w[keep].astype(np.int64).to_numpy()
    kept = df[keep]
    return kept.loc[kept.index.repeat(counts)]
