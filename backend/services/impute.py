"""
Missing-data imputation helper shared across stats.py and models.py.

Strategies
----------
listwise  : drop any row missing ANY value in the selected columns (SPSS/R default)
median    : fill numeric columns with the column median, then drop remaining NaN
mean      : fill numeric columns with the column mean, then drop remaining NaN
mice      : ONE completed dataset from sklearn's IterativeImputer.

WHAT ``mice`` HERE IS NOT. It is a single stochastic completion, not multiple
imputation: m = 1, no between-imputation variance, no Rubin pooling. Standard
errors computed on its output are too small, because they treat imputed cells
as if they had been observed. Proper multiple imputation with Rubin's rules
lives in ``services.missing_data`` (``mice_multiple`` + the ``pool_*``
functions) and is what an inferential analysis must use.

Because the two are one word apart in the UI and worlds apart in what they
license, every caller can ask what actually ran:
``apply_imputation_reported`` returns an :class:`ImputationReport` alongside
the frame, carrying the strategy requested, the strategy applied, why they
differ when they do, and how many cells each column had filled. The plain
``apply_imputation`` keeps its old one-value signature for the ~40 call sites
that do not surface it yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from services.dirty_value_guard import flag_sentinels, plausibility_max_for_column, sentinel_values

#: What the single-completion IterativeImputer path is called in a report, so a
#: result never says "mice" where a reader would hear "multiply imputed".
MICE_SINGLE = "mice_single"

#: Human-readable labels for the report's ``applied`` field.
STRATEGY_LABELS: Dict[str, str] = {
    "listwise": "Complete-case (listwise deletion)",
    "median": "Single median fill",
    "mean": "Single mean fill",
    MICE_SINGLE: "MICE, one completed dataset (no Rubin pooling)",
}


@dataclass(frozen=True)
class ImputationReport:
    """What actually happened, as opposed to what was asked for.

    ``requested`` is the caller's string; ``applied`` is the code path that ran.
    They differ exactly when ``fallback_reason`` is set, and a report whose two
    fields disagree is the thing a result must show the user rather than
    quietly absorb.
    """

    requested: str
    applied: str
    n_rows_in: int
    n_rows_out: int
    n_filled: Dict[str, int] = field(default_factory=dict)
    n_imputations: int = 1
    pooled: bool = False
    fallback_reason: Optional[str] = None
    iterations: Optional[int] = None
    converged: Optional[bool] = None

    @property
    def n_rows_dropped(self) -> int:
        return self.n_rows_in - self.n_rows_out

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe form for a router response."""
        return {
            "requested": self.requested,
            "applied": self.applied,
            "label": STRATEGY_LABELS.get(self.applied, self.applied),
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "n_rows_dropped": self.n_rows_dropped,
            "n_filled": dict(self.n_filled),
            "n_filled_total": int(sum(self.n_filled.values())),
            "n_imputations": self.n_imputations,
            "pooled": self.pooled,
            "fallback_reason": self.fallback_reason,
            "iterations": self.iterations,
            "converged": self.converged,
        }


def _missing_counts(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    return {c: int(df[c].isna().sum()) for c in cols}


def apply_imputation_reported(
    df: pd.DataFrame, cols: List[str], strategy: str = "listwise"
) -> Tuple[pd.DataFrame, ImputationReport]:
    """``apply_imputation``, plus a record of what ran.

    The frame returned is byte-for-byte what ``apply_imputation`` returns for
    the same arguments; the report is the part that was previously lost.
    """
    requested = strategy if isinstance(strategy, str) and strategy else "listwise"
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df, ImputationReport(
            requested=requested, applied="none", n_rows_in=len(df), n_rows_out=len(df)
        )

    n_rows_in = len(df)

    # ------------------------------------------------------------------
    # Listwise (complete-case): always the safe default
    # ------------------------------------------------------------------
    if strategy in ("listwise", "none", "", None):
        out = df.dropna(subset=valid_cols)
        return out, ImputationReport(
            requested=requested, applied="listwise",
            n_rows_in=n_rows_in, n_rows_out=len(out),
            n_filled={c: 0 for c in valid_cols},
        )

    df = df.copy()
    before = _missing_counts(df, valid_cols)
    num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(df[c])]
    applied = requested
    fallback_reason: Optional[str] = None
    iterations: Optional[int] = None
    converged: Optional[bool] = None

    # ------------------------------------------------------------------
    # Median imputation
    # ------------------------------------------------------------------
    if strategy == "median":
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

    # ------------------------------------------------------------------
    # Mean imputation (not recommended for skewed clinical data)
    # ------------------------------------------------------------------
    elif strategy == "mean":
        for col in num_cols:
            df[col] = df[col].fillna(df[col].mean())

    # ------------------------------------------------------------------
    # MICE: a single completed dataset, and named as such
    # ------------------------------------------------------------------
    elif strategy == "mice":
        applied = MICE_SINGLE
        if num_cols:
            try:
                # sklearn >= 0.21 requires the experimental flag in older builds
                try:
                    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                except ImportError:
                    pass
                from sklearn.impute import IterativeImputer

                max_iter = 10
                imp = IterativeImputer(random_state=42, max_iter=max_iter, verbose=0)
                df[num_cols] = imp.fit_transform(df[num_cols])
                iterations = int(getattr(imp, "n_iter_", max_iter) or max_iter)
                # IterativeImputer stops early once the round-to-round change
                # falls under tol; running the full budget means it never did.
                converged = iterations < max_iter
            except Exception as exc:
                # A fallback the caller cannot see is a fallback that gets
                # reported as MICE in a paper. Record it and say so out loud.
                applied = "median"
                fallback_reason = (
                    f"IterativeImputer failed ({type(exc).__name__}: {exc}); "
                    "filled with the column median instead."
                )
                logger.warning("impute: MICE fell back to median; {}", exc)
                for col in num_cols:
                    df[col] = df[col].fillna(df[col].median())

    # Always drop rows still missing after imputation
    # (covers non-numeric columns and edge cases)
    out = df.dropna(subset=valid_cols)
    after = _missing_counts(df, valid_cols)
    report = ImputationReport(
        requested=requested,
        applied=applied,
        n_rows_in=n_rows_in,
        n_rows_out=len(out),
        n_filled={c: before[c] - after[c] for c in valid_cols},
        n_imputations=1,
        pooled=False,
        fallback_reason=fallback_reason,
        iterations=iterations,
        converged=converged,
    )
    return out, report


def apply_imputation(df: pd.DataFrame, cols: List[str], strategy: str = "listwise") -> pd.DataFrame:
    """Return a cleaned DataFrame after applying `strategy` to `cols`.

    Thin wrapper over :func:`apply_imputation_reported` for callers that do not
    surface the report yet. New code should prefer the reporting form.
    """
    frame, _ = apply_imputation_reported(df, cols, strategy)
    return frame


def apply_passive_imputation(df: pd.DataFrame, formulas: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Recompute derived variables after imputation.

    Example: {"bmi": "weight / (height ** 2)"}. Expressions are evaluated with
    pandas.eval against dataframe columns only.
    """
    if not formulas:
        return df
    out = df.copy()
    for target, expr in formulas.items():
        try:
            out[target] = out.eval(expr)
        except Exception:
            continue
    return out


def add_survival_auxiliary_variables(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    *,
    prefix: str = "__surv_aux",
) -> pd.DataFrame:
    """
    Add survival-specific auxiliary variables for imputation models:
    log time and Nelson-Aalen cumulative hazard at each subject's follow-up.
    """
    out = df.copy()
    if duration_col not in out.columns or event_col not in out.columns:
        return out
    duration = pd.to_numeric(out[duration_col], errors="coerce")
    event = pd.to_numeric(out[event_col], errors="coerce").fillna(0).astype(int)
    out[f"{prefix}_log_time"] = np.log(np.clip(duration.astype(float), 1e-8, None))
    try:
        from lifelines import NelsonAalenFitter

        mask = duration.notna()
        naf = NelsonAalenFitter()
        naf.fit(duration[mask].astype(float), event_observed=event[mask].astype(int))
        cumulative = naf.cumulative_hazard_at_times(duration.fillna(duration.median()).astype(float)).to_numpy()
        out[f"{prefix}_nelson_aalen"] = np.asarray(cumulative, dtype=float)
    except Exception:
        order = duration.rank(method="average", pct=True)
        out[f"{prefix}_nelson_aalen"] = -np.log(np.clip(1.0 - order.fillna(order.median()), 1e-6, 1.0))
    return out


def missing_info(df: pd.DataFrame, cols: List[str]) -> dict:
    """Return a structured missing-value summary for the given columns."""
    valid_cols = [c for c in cols if c in df.columns]
    total = len(df)

    per_col: dict = {}
    for col in valid_cols:
        max_plausible = plausibility_max_for_column(col)
        raw_missing = df[col].isna()
        implausible = flag_sentinels(df[col], max_plausible)
        n = int((raw_missing | implausible).sum())
        per_col[col] = {
            "count": n,
            "raw_count": int(raw_missing.sum()),
            "n_implausible": int(implausible.sum()),
            "implausible_values": sorted(sentinel_values(df[col], max_plausible)),
            "pct": round(n / total * 100, 1) if total > 0 else 0.0,
        }

    if valid_cols:
        masks = []
        for col in valid_cols:
            max_plausible = plausibility_max_for_column(col)
            masks.append(df[col].isna() | flag_sentinels(df[col], max_plausible))
        rows_affected = int(pd.concat(masks, axis=1).any(axis=1).sum())
    else:
        rows_affected = 0
    return {
        "total_rows": total,
        "rows_affected": rows_affected,
        "pct_affected": round(rows_affected / total * 100, 1) if total > 0 else 0.0,
        "per_column": per_col,
    }
