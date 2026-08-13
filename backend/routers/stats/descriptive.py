from __future__ import annotations

from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import json as _json
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException, Query
from pydantic import AliasChoices, BaseModel, Field
from loguru import logger

from services import store
from services.category_health import clean_two_level
from services.dirty_value_guard import (
    coerce_numeric,
    flag_sentinels,
    plausibility_max_for_column,
    values_are_numeric,
)
from services.stat_utils import (
    sorted_groups,
    _categorical_p_with_rule,
    check_equal_variances,
    looks_continuous,
)
from services.impute import apply_imputation, missing_info
from services.number_format import level_key

router = APIRouter()


def _get_df(session_id: str, *, allow_missing: bool = False) -> pd.DataFrame | None:
    df = store.get_filtered(session_id)
    if df is None:
        if allow_missing:
            return None
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None in dicts/lists."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _stored_kind(
    session_id: str, df: pd.DataFrame, col: str, requested: Optional[str] = None
) -> Optional[str]:
    if requested:
        return requested
    try:
        override = (store.get_kind_overrides(session_id) or {}).get(col)
        if override:
            return override
    except Exception:
        pass
    return None


def _plausibility_warnings(col: str, series: pd.Series) -> list[dict]:
    numeric = coerce_numeric(series)
    key = str(col).strip().lower()
    mask = pd.Series(False, index=series.index)
    rule = None
    if key == "age":
        mask = numeric.notna() & ((numeric < 0) | (numeric > 120))
        rule = "expected 0 <= age <= 120"
    elif key in {"bmi", "body_mass_index"} or "bmi" in key:
        mask = numeric.notna() & ((numeric <= 10) | (numeric >= 100))
        rule = "expected 10 < bmi < 100"
    elif key in {"fu_days", "followup_days", "follow_up_days"}:
        mask = numeric.notna() & (numeric <= 0)
        rule = "expected fu_days > 0"
    if not mask.any():
        return []
    vals = sorted({float(v) for v in numeric[mask].dropna().unique()})
    return [
        {
            "variable": col,
            "n_implausible": int(mask.sum()),
            "implausible_values": vals,
            "rule": rule,
            "note": "Values were retained for display but should be reviewed.",
        }
    ]


# ── 1. Missing Data Summary ─────────────────────────────────────────────────────


@router.get("/{session_id}/missing")
def get_missing(session_id: str, columns: str = Query("")):
    """
    Return per-column missing counts and total rows affected for the given
    comma-separated list of column names.
    """
    df = _get_df(session_id, allow_missing=True)
    if df is None:
        return {"columns": [], "total_rows": 0}
    cols = [
        c.strip() for c in columns.split(",") if c.strip() and c.strip() in df.columns
    ]
    if not cols:
        cols = df.columns.tolist()
    return missing_info(df, cols)


# ── 2. Descriptive Statistics ──────────────────────────────────────────────────


def _normality_test(s_clean: pd.Series) -> tuple[float, str]:
    """Return (p_value, test_name)."""
    n = len(s_clean)
    if n < 3:
        return 1.0, "—"
    if n < 50:
        _, p = scipy_stats.shapiro(s_clean)
        return float(p), "Shapiro-Wilk"
    if n <= 2000:
        from statsmodels.stats.diagnostic import lilliefors as _lilliefors

        _, p = _lilliefors(s_clean.values, dist="norm")
        return float(p), "Kolmogorov-Smirnov (Lilliefors)"

    skewness = float(scipy_stats.skew(s_clean))
    if abs(skewness) <= 1.5:
        return 0.999, "Skewness (CLT bypass)"
    from statsmodels.stats.diagnostic import lilliefors as _lilliefors

    _, p = _lilliefors(s_clean.values, dist="norm")
    return float(p), "Kolmogorov-Smirnov (Lilliefors)"


@router.get("/{session_id}/descriptive")
def descriptive(session_id: str, column: Optional[str] = None):
    df = _get_df(session_id)
    overrides = store.get_kind_overrides(session_id) or {}
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols.extend(
        [
            c
            for c, kind in overrides.items()
            if kind == "numeric" and c in df.columns and c not in num_cols
        ]
    )
    if column:
        if column not in df.columns or column not in num_cols:
            raise HTTPException(status_code=400, detail="Column not numeric")
        num_cols = [column]

    # Resolve session-persisted decimal overrides once so each column
    # carries its display hint to the frontend (Summary tile, exports).
    decimals_override = _resolve_decimals_override(session_id, None)

    results = {}
    for col in num_cols:
        numeric = coerce_numeric(df[col]).replace([np.inf, -np.inf], np.nan)
        s = numeric.dropna()
        if len(s) < 3:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        n = len(s)
        p_norm, norm_test = _normality_test(s)

        results[col] = {
            "n": int(n),
            "missing": int(df[col].isna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "se": float(s.sem()),
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "skewness": float(scipy_stats.skew(s)),
            "kurtosis": float(scipy_stats.kurtosis(s)),
            "normality_p": float(p_norm),
            "normality_test": norm_test,
            "normal": bool(p_norm >= 0.05),
            "warnings": _plausibility_warnings(col, df[col]),
            # Suggested decimal places for displaying sample-valued stats
            # (mean, median, quartiles, min/max). Honours user overrides
            # and auto-detects integer-valued columns.
            "display_decimals": _col_decimals(df, col, decimals_override, fallback=2),
        }
    return _sanitize(results)


# ── 3. Frequency Table ─────────────────────────────────────────────────────────


@router.get("/{session_id}/frequency")
def frequency(session_id: str, column: Optional[str] = None):
    df = _get_df(session_id)
    cols = df.columns.tolist()
    if column:
        if column not in df.columns:
            raise HTTPException(status_code=400, detail="Column not found")
        cols = [column]

    results = {}
    for col in cols:
        s = df[col]
        total = len(s)
        vc = s.value_counts(dropna=False)
        categories = []
        for k, v in vc.items():
            categories.append(
                {
                    "value": str(k) if pd.notna(k) else "Missing",
                    "count": int(v),
                    "pct": round(v / total * 100, 1),
                }
            )
        results[col] = {
            "n": int(s.count()),
            "missing": int(s.isna().sum()),
            "categories": categories,
        }
    return results


# ── 4. Sparklines ──────────────────────────────────────────────────────────────


@router.get("/{session_id}/column_badges")
def get_column_badges(session_id: str):
    """Per-column facts for the grid header: missing count and value range.

    Computed over the whole dataframe. The grid's own `preview` is capped at
    2000 rows by the upload endpoint, so anything counted there describes the
    top of the file rather than the column, which is exactly the kind of
    number a header badge is read as being about the column.

    Range is returned only for numeric columns; min and max of a category
    label is an artefact of alphabetical order, not a fact about the data.
    """
    df = _get_df(session_id, allow_missing=True)
    if df is None:
        return {"n_rows": 0, "columns": {}}

    n_rows = int(len(df))
    out: dict = {}
    for col in df.columns:
        max_plausible = plausibility_max_for_column(col)
        raw_missing = df[col].isna()
        implausible = flag_sentinels(df[col], max_plausible)
        n_missing = int((raw_missing | implausible).sum())
        entry: dict = {
            "n_missing": n_missing,
            "pct_missing": round(n_missing / n_rows * 100, 1) if n_rows else 0.0,
        }
        # Sentinels (999, -99 …) are excluded from the range for the same
        # reason they count as missing: a max of 999 for a heart rate is the
        # placeholder, not the largest observation.
        s = df[col][~(raw_missing | implausible)]
        if len(s) > 0:
            # Gate on whether the VALUES are numbers, not on the dtype pandas
            # happens to be holding them in. A column added after upload comes
            # back as object, and everything typed into it is stored as text,
            # so `is_numeric_dtype` was False and a column full of numbers got
            # no range at all — while the columns beside it, numeric since
            # upload, showed theirs.
            #
            # Every non-missing value has to parse: one genuine word makes this
            # a text column, and a min/max over "apple" and "3" is an artefact
            # of sort order rather than a fact about the data. Same predicate
            # the cell writer uses, so what gets a range and what gets stored
            # as a number cannot drift apart.
            if values_are_numeric(s):
                numeric = pd.to_numeric(s, errors="coerce").replace(
                    [np.inf, -np.inf], np.nan).dropna()
                if len(numeric) > 0:
                    entry["min"] = float(numeric.min())
                    entry["max"] = float(numeric.max())
                    entry["n_valid"] = int(len(numeric))
        out[col] = entry

    return {"n_rows": n_rows, "columns": out}


@router.get("/{session_id}/sparklines")
def get_sparklines(session_id: str):
    df = _get_df(session_id, allow_missing=True)
    if df is None:
        return {}
    result = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            result[col] = {"type": "empty", "data": []}
            continue
        if pd.api.types.is_numeric_dtype(s):
            n_bins = min(14, max(4, int(len(s) ** 0.38)))
            counts, _ = np.histogram(s, bins=n_bins)
            result[col] = {"type": "numeric", "data": counts.tolist()}
        else:
            vc = s.value_counts(normalize=True)
            n_cats = min(6, len(vc))
            result[col] = {
                "type": "categorical",
                "data": [float(v) for v in vc.head(n_cats).values],
                "labels": vc.head(n_cats).index.astype(str).tolist(),
            }
    return result


# ── 5. Refresh ─────────────────────────────────────────────────────────────────


@router.get("/{session_id}/refresh")
def refresh_session(session_id: str):
    """Return updated session metadata after in-place operations.

    Honours user kind overrides and re-attaches per-column metadata
    (value_labels, analysis_excluded, display_name) so flows that refresh after
    a mutation (rename, fill, duplicate…) don't silently drop them.
    """
    df = _get_df(session_id)
    from routers.upload import _detect_kind

    overrides = store.get_kind_overrides(session_id) or {}
    meta = store.get_metadata(session_id) or {}
    columns = []
    for col in df.columns:
        kind = overrides.get(col) or _detect_kind(df[col])
        c = {"name": col, "dtype": str(df[col].dtype), "kind": kind}
        m = meta.get(col, {}) or {}
        if m.get("value_labels"):
            c["value_labels"] = m["value_labels"]
        if m.get("analysis_excluded") is not None:
            c["analysis_excluded"] = bool(m["analysis_excluded"])
        if m.get("display_name"):
            c["display_name"] = m["display_name"]
        columns.append(c)
    preview_df = df.head(2000).replace([np.inf, -np.inf], np.nan)
    preview = _json.loads(
        preview_df.to_json(
            orient="records", default_handler=str, date_format="iso", date_unit="s"
        )
    )
    return {"rows": len(df), "columns": columns, "preview": preview}


# ── 6. Raw Data Columns ────────────────────────────────────────────────────────


@router.get("/{session_id}/raw")
def get_raw_columns(session_id: str, columns: str = ""):
    df = _get_df(session_id)
    cols = (
        [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if columns
        else list(df.columns)
    )
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])][:12]
    result = {}
    for col in cols:
        vals = df[col].where(df[col].notna(), other=None).tolist()[:3000]
        result[col] = vals
    return result


# ── 7. Column Summary (QQ + Outliers) ──────────────────────────────────────────


@router.get("/{session_id}/column_summary")
def column_summary(session_id: str, column: str, kind: Optional[str] = None):
    df = _get_df(session_id)
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
    s = df[column]

    resolved_kind = _stored_kind(session_id, df, column, kind)
    if resolved_kind == "numeric":
        is_num = True
    elif resolved_kind in ("categorical", "text", "boolean"):
        is_num = False
    else:
        is_num = looks_continuous(s)

    if is_num:
        s_clean = coerce_numeric(s).replace([np.inf, -np.inf], np.nan).dropna()
        n_clean = len(s_clean)
        if n_clean < 3:
            raise HTTPException(
                status_code=400, detail="Need at least 3 numeric values."
            )
        n_bins = min(40, max(10, int(np.sqrt(n_clean))))
        counts, edges = np.histogram(s_clean, bins=n_bins)
        histogram = [
            {
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]

        (theo, sample), _ = scipy_stats.probplot(s_clean)
        step = max(1, len(theo) // 300)
        qq = [
            {"x": float(theo[i]), "y": float(sample[i])}
            for i in range(0, len(theo), step)
        ]

        p_norm, norm_test_name = _normality_test(s_clean)
        mean_val = float(s_clean.mean())
        std_val = float(s_clean.std())
        q1, q3 = float(s_clean.quantile(0.25)), float(s_clean.quantile(0.75))
        iqr_val = q3 - q1

        fence_low = q1 - 1.5 * iqr_val
        fence_high = q3 + 1.5 * iqr_val

        non_out = s_clean[(s_clean >= fence_low) & (s_clean <= fence_high)]
        whisker_low = float(non_out.min()) if len(non_out) else float(s_clean.min())
        whisker_high = float(non_out.max()) if len(non_out) else float(s_clean.max())

        out_mask = (s_clean < fence_low) | (s_clean > fence_high)
        outliers = [
            {"row": int(idx) + 1, "value": float(val)}
            for idx, val in zip(s_clean.index[out_mask], s_clean[out_mask])
        ]

        z_extremes = []
        normality_deviants = []
        if std_val > 0 and n_clean >= 3:
            z_series = (s_clean - mean_val) / std_val
            s_sorted_idx = s_clean.sort_values().index
            s_sorted_vals = s_clean.loc[s_sorted_idx].values

            all_points_info = []
            for i, idx in enumerate(s_sorted_idx):
                val = float(s_sorted_vals[i])
                rank = i + 1
                theo_q = float(scipy_stats.norm.ppf((rank - 0.375) / (n_clean + 0.25)))
                expected_val = mean_val + std_val * theo_q
                residual = val - expected_val
                z = float(z_series[idx])

                info = {
                    "row": int(idx) + 1,
                    "value": round(val, 4),
                    "z": round(z, 3),
                    "residual": round(residual, 4),
                    "abs_residual": abs(residual),
                    "qq_x": round(theo_q, 4),
                }
                all_points_info.append(info)
                if abs(z) > 2.0:
                    z_extremes.append(info)

            all_points_info.sort(key=lambda d: d["abs_residual"], reverse=True)
            normality_deviants = all_points_info[:10]
            z_extremes.sort(key=lambda d: abs(d["z"]), reverse=True)

        return {
            "type": "numeric",
            "n": int(s_clean.count()),
            "missing": int(s.isna().sum()),
            "mean": mean_val,
            "std": std_val,
            "median": float(s_clean.median()),
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr_val),
            "min": float(s_clean.min()),
            "max": float(s_clean.max()),
            "skewness": float(s_clean.skew()),
            "kurtosis": float(s_clean.kurtosis()),
            "whisker_low": whisker_low,
            "whisker_high": whisker_high,
            "outliers": outliers,
            "z_extremes": z_extremes,
            "normality_deviants": normality_deviants,
            "histogram": histogram,
            "raw_values": s_clean.sample(min(2000, n_clean), random_state=42).tolist(),
            "qq": qq,
            "normality_p": float(p_norm),
            "normality_test": norm_test_name,
            "normal": bool(p_norm >= 0.05),
            "normality_label": "Normally distributed"
            if p_norm >= 0.05
            else "Non-normal distribution",
            "warnings": _plausibility_warnings(column, s),
        }

    else:
        total = len(s)
        vc = s.value_counts(dropna=False)
        categories = [
            {
                # level_key so a float64 code reads "0", the same string the
                # grid shows and the same key its value labels are stored
                # under. str() gave "0.0", which matched no label and put raw
                # codes in the distribution chart of a fully labelled column.
                "value": level_key(k) if pd.notna(k) else "Missing",
                "count": int(v),
                "pct": round(v / total * 100, 1),
            }
            for k, v in vc.items()
        ]
        return {
            "type": "categorical",
            "n": int(s.count()),
            "missing": int(s.isna().sum()),
            "n_categories": int(s.nunique()),
            "categories": categories,
        }


# ── 8. Table 1 (clinical characteristics) ──────────────────────────────────────


class Table1Request(BaseModel):
    session_id: str
    group_column: Optional[str] = None
    variables: list[str]
    variable_kinds: Optional[dict] = None
    selected_stats: Optional[list[str]] = None
    # None means "decide from the request": with a grouping column the
    # assumption behind the t-test and ANOVA is normality WITHIN each group,
    # so that is what gets tested. Pass "overall" explicitly to test the
    # pooled sample instead.
    normality_mode: Optional[str] = None
    # Optional per-column decimal overrides keyed by column name. Values
    # supplied here win over (a) the session-persisted decimals map and
    # (b) the auto integer-detection logic in _col_decimals().
    column_decimals: Optional[Dict[str, int]] = None


def _fmt_p(p: float) -> str:
    from services.number_format import format_p

    return format_p(p)


_STAT_LABELS: dict[str, str] = {
    "mean_sd": "Mean ± SD",
    "median_iqr": "Median [IQR]",
    "se": "SE of Mean",
    "ci95": "95% CI",
    "variance": "Variance",
    "min_max": "Min – Max",
    "n": "N (non-missing)",
    "missing": "Missing n (%)",
    "p10": "10th Pctl",
    "p25": "25th Pctl",
    "p75": "75th Pctl",
    "p90": "90th Pctl",
    "p95": "95th Pctl",
}


def _f(v: float, d: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{d}f}"


_NATURAL_DECIMALS_CAP = 6  # never auto-detect beyond this many places


def _natural_decimals(
    series: pd.Series, cap: int = _NATURAL_DECIMALS_CAP
) -> Optional[int]:
    """Smallest number of decimal places that losslessly represents every
    value in the column.

    A column of {15.4, 16.2, 17.0} round-trips at 1 decimal; {15.42, 16.0}
    needs 2. Returns None when the column is empty or non-finite so the
    caller can fall back. Capped at ``cap`` so a float-noise column
    (0.1 + 0.2 == 0.30000000000000004) doesn't return 17.

    Used so Table 1 / Summary inherit the *source variable's* precision —
    a 1-decimal lab value reads "15.4", not "15.42".
    """
    clean = series.dropna()
    if len(clean) == 0:
        return None
    try:
        arr = clean.astype(float).to_numpy()
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
    for d in range(0, cap + 1):
        # rint avoids banker's-rounding surprises; tolerance absorbs the
        # float64 representation error of e.g. 15.4.
        if np.allclose(arr, np.round(arr, d), rtol=0.0, atol=0.5 * 10 ** (-d - 4)):
            return d
    return cap


def _col_decimals(
    df: pd.DataFrame,
    col: str,
    override: Optional[Dict[str, int]] = None,
    fallback: int = 2,
) -> int:
    """Resolve per-column decimal places for descriptive display.

    Resolution order:
      1. Explicit ``override`` mapping (request body / session store).
      2. Integer-dtype column → 0 (e.g. counts, days, ages).
      3. Float column whose values are all whole numbers → 0 (e.g. days
         column re-read from SPSS as float64 but holding integers only).
      4. Float column's *natural* precision — the fewest decimals that
         represent every value losslessly, capped at ``fallback`` so a
         1-decimal lab value (15.4) reads "15.4" not "15.42" while a
         high-precision column still tops out at the 2-decimal default.
      5. Otherwise the supplied fallback (default 2).

    Industry convention (AMA, ICMJE) is that Table 1 statistics inherit the
    precision of the source variable; a follow-up-days column should report
    integer medians, not 1697.50, and a 1-decimal column should not gain a
    spurious second decimal.
    """
    if override and col in override:
        try:
            return max(0, int(override[col]))
        except (TypeError, ValueError):
            pass
    if col not in df.columns:
        return fallback
    s = df[col]
    if pd.api.types.is_integer_dtype(s):
        return 0
    if pd.api.types.is_float_dtype(s):
        clean = s.dropna()
        if len(clean) > 0:
            try:
                if (clean.mod(1) == 0).all():
                    return 0
            except (TypeError, ValueError):
                pass
            # Inherit the column's natural precision, but never exceed the
            # fallback — a messy float column still displays at 2 places.
            natural = _natural_decimals(clean)
            if natural is not None:
                return min(natural, fallback)
    return fallback


def _resolve_decimals_override(
    session_id: Optional[str],
    request_override: Optional[Dict[str, int]],
) -> Dict[str, int]:
    """Merge the persisted per-session decimals map with a request-supplied
    one. Request values take precedence so callers can preview overrides
    without committing them to the session."""
    base: Dict[str, int] = {}
    if session_id:
        try:
            base = dict(store.get_decimals(session_id) or {})
        except Exception:  # pragma: no cover — defensive
            base = {}
    if request_override:
        for k, v in request_override.items():
            try:
                base[k] = max(0, int(v))
            except (TypeError, ValueError):
                continue
    return base


def _fmt_one_stat(
    a: pd.Series,
    stat: str,
    *,
    df: Optional[pd.DataFrame] = None,
    col: Optional[str] = None,
    override: Optional[Dict[str, int]] = None,
) -> str:
    """Format a single Table 1 statistic.

    When ``df`` + ``col`` are supplied, the column's natural decimal places
    are honoured (integer columns render as integers, float overrides win).
    SE / variance keep an extra digit of precision per AMA convention.
    """
    if len(a) == 0:
        return "—"
    # Column-aware decimal places for sample-valued statistics (mean,
    # median, quartiles, min/max). Falls back to the legacy 2-decimal
    # default when no column context is supplied.
    if df is not None and col is not None:
        d = _col_decimals(df, col, override, fallback=2)
    else:
        d = 2

    def fc(v: float, dd: Optional[int] = None) -> str:
        return _f(v, d if dd is None else dd)

    if stat == "mean_sd":
        return f"{fc(a.mean())} ± {fc(a.std())}"
    if stat == "median_iqr":
        q1, q3 = a.quantile(0.25), a.quantile(0.75)
        return f"{fc(a.median())} [{fc(q1)}–{fc(q3)}]"
    if stat == "se":
        # SE keeps an extra digit of precision since it shrinks with √n.
        return fc(a.sem(), max(d, 3))
    if stat == "ci95":
        if len(a) < 2:
            return "—"
        se = a.sem()
        m = a.mean()
        t_crit = scipy_stats.t.ppf(0.975, df=len(a) - 1)
        ci = t_crit * se
        return f"{fc(m)} [{fc(m - ci)}–{fc(m + ci)}]"
    if stat == "variance":
        return fc(a.var(), max(d, 3))
    if stat == "min_max":
        return f"{fc(a.min())} – {fc(a.max())}"
    if stat == "n":
        return str(int(len(a)))
    if stat == "missing":
        return str(int(a.isna().sum()) if hasattr(a, "isna") else 0)
    pct_map = {"p10": 0.10, "p25": 0.25, "p75": 0.75, "p90": 0.90, "p95": 0.95}
    if stat in pct_map:
        return fc(a.quantile(pct_map[stat]))
    return "—"


def _missing_stat_row(
    s_col: pd.Series,
    group_series: dict[str, pd.Series],
) -> Optional[dict]:
    """Build a display-only missingness row.

    This deliberately lives outside categorical ``sub_rows`` so missing
    observations cannot be mistaken for a level by tests, SMD calculations,
    exports, or other API consumers.
    """

    def _value(series: pd.Series) -> str:
        n_missing = int(series.isna().sum())
        pct = round(n_missing / len(series) * 100, 1) if len(series) else 0.0
        return f"{n_missing} ({pct}%)"

    if not s_col.isna().any():
        return None
    return {
        "label": _STAT_LABELS["missing"],
        "overall": _value(s_col),
        "group_stats": {gl: _value(gs) for gl, gs in group_series.items()},
    }


def _build_stat_rows(
    s_col: pd.Series,
    group_series: dict[str, pd.Series],
    stats: list[str],
    normal: bool,
    *,
    df: Optional[pd.DataFrame] = None,
    col: Optional[str] = None,
    override: Optional[Dict[str, int]] = None,
) -> list[dict]:
    rows_out = []
    s_all = s_col.dropna().astype(float)

    for stat in stats:
        resolved = stat
        if stat == "auto":
            resolved = "mean_sd" if normal else "median_iqr"

        label = _STAT_LABELS.get(resolved, resolved)
        if resolved == "missing":
            missing_row = _missing_stat_row(s_col, group_series)
            if missing_row is None:
                missing_row = {
                    "label": label,
                    "overall": "0 (0.0%)",
                    "group_stats": {
                        gl: "0 (0.0%)" for gl in group_series
                    },
                }
            overall_val = missing_row["overall"]
            grp_vals = missing_row["group_stats"]
        else:
            overall_val = _fmt_one_stat(
                s_all,
                resolved,
                df=df,
                col=col,
                override=override,
            )
            grp_vals = {
                gl: _fmt_one_stat(
                    gs.dropna().astype(float),
                    resolved,
                    df=df,
                    col=col,
                    override=override,
                )
                for gl, gs in group_series.items()
            }

        rows_out.append(
            {"label": label, "overall": overall_val, "group_stats": grp_vals}
        )
    return rows_out


@router.post("/table1")
def table1(req: Table1Request):
    df = _get_df(req.session_id).copy()
    rows = []
    warnings: list = []
    sel_stats: list[str] = req.selected_stats if req.selected_stats else ["auto"]
    # Per-column decimal overrides: merge the session-persisted map with
    # any request-supplied overrides (request wins). Auto-detection still
    # applies for columns absent from both.
    decimals_override = _resolve_decimals_override(req.session_id, req.column_decimals)

    groups = None
    group_labels = []
    group_ns: dict = {}
    if req.group_column and req.group_column in df.columns:
        cleaned_group = clean_two_level(df[req.group_column])
        df[req.group_column] = cleaned_group.series
        warnings.extend(cleaned_group.warnings)
        groups = sorted_groups(df[req.group_column])
        group_labels = [str(g) for g in groups]
        group_ns = {str(g): int((df[req.group_column] == g).sum()) for g in groups}

    # What the parametric tests below actually assume is normality within each
    # group — the pooled sample of two groups that differ is a mixture, and can
    # fail Shapiro because the groups differ rather than because either is
    # skewed (or pass it while both groups are skewed the opposite way). R and
    # Python users testing this by hand run Shapiro per group, and uSTAT
    # disagreeing with them was a wrong default, not a difference of opinion.
    # "overall" is still honoured when asked for explicitly.
    requested_mode = (req.normality_mode or "").strip().lower() or None
    normality_mode = requested_mode or ("within_group" if groups is not None else "overall")

    for var in req.variables:
        if var not in df.columns:
            continue
        s = df[var]

        provided_kind = _stored_kind(
            req.session_id,
            df,
            var,
            (req.variable_kinds or {}).get(var),
        )
        if provided_kind == "numeric":
            is_num = True
        elif provided_kind in ("categorical", "text", "boolean"):
            is_num = False
        else:
            is_num = looks_continuous(s)

        if is_num:
            s = coerce_numeric(s).replace([np.inf, -np.inf], np.nan)
            df[var] = s
            warnings.extend(_plausibility_warnings(var, s))
            s_all = s.dropna().astype(float)
            p_norm, norm_test_name = _normality_test(s_all)
            normal_overall = p_norm >= 0.05

            group_series: dict[str, pd.Series] = {}
            group_arrs: list[pd.Series] = []
            if groups is not None:
                for g, gl in zip(groups, group_labels):
                    gs = df[df[req.group_column] == g][var]
                    group_series[gl] = gs
                    group_arrs.append(gs.dropna().astype(float))

            per_group_norm: dict[str, dict] = {}
            if (
                normality_mode == "within_group"
                and groups is not None
                and len(group_arrs) >= 2
            ):
                for gl, arr in zip(group_labels, group_arrs):
                    if len(arr) >= 3:
                        pg, pg_name = _normality_test(arr)
                        per_group_norm[gl] = {
                            "p": round(float(pg), 4),
                            "test": pg_name,
                            "normal": bool(pg >= 0.05),
                            "n": int(len(arr)),
                        }
                    else:
                        per_group_norm[gl] = {
                            "p": None,
                            "test": "n<3",
                            "normal": False,
                            "n": int(len(arr)),
                        }
                normal = len(per_group_norm) > 0 and all(
                    v["normal"] for v in per_group_norm.values()
                )
            else:
                normal = normal_overall

            stat_rows = _build_stat_rows(
                s,
                group_series,
                sel_stats,
                normal,
                df=df,
                col=var,
                override=decimals_override,
            )
            missing_row = (
                None
                if "missing" in sel_stats
                else _missing_stat_row(s, group_series)
            )

            p_value_str: Optional[str] = None
            test_name_str: Optional[str] = None
            significant = False
            p_raw: Optional[float] = None
            # A group with fewer than two values supports no comparison at
            # all: scipy returns nan for the t-test and for Mann-Whitney
            # alike, and that nan used to be printed as an em dash beside a
            # confident test name. The guard used to fire only on the
            # parametric branch, so the non-parametric one still printed it.
            thin = [
                gl for gl, arr in zip(group_labels, group_arrs) if len(arr) < 2
            ]
            if groups is not None and thin:
                warnings.append(
                    f"'{var}': no p-value — group(s) "
                    + ", ".join(repr(g) for g in thin)
                    + " have fewer than 2 values with this variable recorded."
                )
            elif groups is not None and len(group_arrs) >= 2:
                try:
                    if len(groups) == 2:
                        if normal:
                            # Pick Student vs Welch the same way /api/stats/ttest
                            # does. Table 1 used to force equal_var=False and
                            # label the row plain "t-test", so opening the same
                            # pair in the Tests tab — which lets Levene decide —
                            # could print a different p for the same two columns
                            # with nothing to explain the gap.
                            lev = check_equal_variances(
                                list(group_arrs), list(group_labels)
                            )
                            use_welch = not lev["met"]
                            _, p_t = scipy_stats.ttest_ind(
                                *group_arrs, equal_var=not use_welch
                            )
                            test_name_str = "t-test (Welch)" if use_welch else "t-test"
                        else:
                            _, p_t = scipy_stats.mannwhitneyu(
                                *group_arrs, alternative="two-sided"
                            )
                            test_name_str = "Mann-Whitney"
                    else:
                        if normal:
                            _, p_t = scipy_stats.f_oneway(*group_arrs)
                            test_name_str = "ANOVA"
                        else:
                            _, p_t = scipy_stats.kruskal(*group_arrs)
                            test_name_str = "Kruskal-Wallis"
                    p_value_str = _fmt_p(float(p_t))
                    p_raw = float(p_t)
                    significant = bool(float(p_t) < 0.05)
                except Exception:
                    logger.exception("Table 1 statistical test failed")
                    p_value_str = "N/A"

            smd_val: Optional[float] = None
            if groups is not None and len(group_arrs) >= 2:
                try:

                    def _smd_num_pair(g1, g2) -> Optional[float]:
                        if len(g1) == 0 or len(g2) == 0:
                            return None
                        ps = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
                        if not np.isfinite(ps) or ps <= 0:
                            return None
                        return float(abs(g1.mean() - g2.mean()) / ps)

                    from itertools import combinations as _comb

                    pair_smds = []
                    for i, j in _comb(range(len(group_arrs)), 2):
                        s_smd = _smd_num_pair(group_arrs[i], group_arrs[j])
                        if s_smd is not None:
                            pair_smds.append(s_smd)
                    if pair_smds:
                        smd_val = round(max(pair_smds), 4)
                except Exception:
                    logger.exception("SMD numerical calculation failed")

            row: dict = {
                "variable": var,
                "type": "numeric",
                # The formatted p is for the table; the raw one is for anyone
                # comparing against another tool or pooling for meta-analysis.
                # Only the rounded string used to be returned.
                "p_raw": p_raw,
                "overall_n": int(len(s_all)),
                "normal": normal,
                "normality_test": norm_test_name,
                "normality_p": round(p_norm, 4),
                # The effective mode, not the requested one: a caller that
                # sends nothing has to be able to read back what was tested.
                "normality_mode": normality_mode,
                "per_group_normality": per_group_norm,
                "stat_rows": stat_rows,
                "missing_row": missing_row,
                "p_value": p_value_str,
                "test": test_name_str,
                "significant": significant,
                "smd": smd_val,
                "stat_label": stat_rows[0]["label"] if stat_rows else "",
                "overall": stat_rows[0]["overall"] if stat_rows else "",
                "group_stats": stat_rows[0]["group_stats"] if stat_rows else {},
            }

        else:
            cleaned_var = clean_two_level(s)
            s = cleaned_var.series
            warnings.extend(cleaned_var.warnings)
            if var in df.columns:
                df[var] = s
            vc_all = s.value_counts(dropna=True)
            total_all = s.count()
            category_group_series = (
                {
                    gl: df[df[req.group_column] == g][var]
                    for g, gl in zip(groups, group_labels)
                }
                if groups is not None
                else {}
            )
            missing_row = _missing_stat_row(s, category_group_series)
            # Two strings per level: `cat` matches the stringified column,
            # `shown` is what the row is labelled with. They differ for a
            # float64 code — "0.0" matches the data, "0" is what the grid
            # displays and what the value labels are keyed by.
            cats = [str(v) for v in vc_all.index.tolist()]
            shown_for = {str(v): level_key(v) for v in vc_all.index.tolist()}
            sub_rows = []
            for cat in cats:
                n_all = int((s.astype(str) == cat).sum())
                pct_all = round(n_all / total_all * 100, 1) if total_all else 0.0
                sub: dict = {
                    "category": shown_for[cat],
                    "overall": f"{n_all} ({pct_all}%)",
                    "group_stats": {},
                }
                if groups is not None:
                    for g, gl in zip(groups, group_labels):
                        g_s = df[df[req.group_column] == g][var]
                        n_g = int((g_s.astype(str) == cat).sum())
                        t_g = g_s.count()
                        pct_g = round(n_g / t_g * 100, 1) if t_g else 0.0
                        sub["group_stats"][gl] = f"{n_g} ({pct_g}%)"
                sub_rows.append(sub)

            p_val: Optional[str] = None
            test_name: Optional[str] = None
            p_chi_raw: Optional[float] = None
            if groups is not None:
                try:
                    # Drop incomplete pairs BEFORE stringifying. astype(str)
                    # turns NaN into the literal "nan", which pd.crosstab then
                    # counts as a real category: the test gained a row, a
                    # degree of freedom and the missing rows themselves, so it
                    # measured association with missingness. The displayed
                    # categories come from value_counts(dropna=True), so the
                    # extra row never appeared — the printed table and the
                    # printed p described different tables.
                    pairs = df[[var, req.group_column]].dropna()
                    ct = pd.crosstab(
                        pairs[var].astype(str), pairs[req.group_column]
                    )
                    p_chi_raw, reason = _categorical_p_with_rule(ct.values)
                    if p_chi_raw is None:
                        # A one-row or one-column table has no association to
                        # test. Saying so beats printing the p = 1.000 that
                        # chi2_contingency returns at dof 0, which reads as a
                        # tested, non-significant result.
                        p_val = None
                        test_name = None
                        warnings.append(f"'{var}': no p-value — {reason.lower()}.")
                    else:
                        test_name = reason
                        p_val = _fmt_p(float(p_chi_raw))
                        n_test = int(ct.values.sum())
                        if n_test < len(df):
                            warnings.append(
                                f"'{var}': the test uses {n_test} of {len(df)} "
                                "rows; the rest are missing the variable or the "
                                "grouping column."
                            )
                except Exception:
                    logger.exception("Categorical test failed in Table 1")
                    p_val = "N/A"

            cat_smd: Optional[float] = None
            if groups is not None and len(groups) >= 2:
                try:

                    def _smd_cat_pair(
                        g1_s: pd.Series, g2_s: pd.Series
                    ) -> Optional[float]:
                        all_cats = sorted(set(g1_s.dropna()) | set(g2_s.dropna()))
                        if len(all_cats) < 2:
                            return None
                        if len(all_cats) == 2:
                            target = all_cats[0]
                            p1 = (g1_s == target).mean()
                            p2 = (g2_s == target).mean()
                            pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
                            if pooled <= 0:
                                return None
                            return float(abs(p1 - p2) / pooled)

                        p1_vec = np.array([(g1_s == c).mean() for c in all_cats[:-1]])
                        p2_vec = np.array([(g2_s == c).mean() for c in all_cats[:-1]])

                        # Multinomial covariance: diag(p) - p p'. The
                        # off-diagonal -p_i p_j terms were missing, so the
                        # categories were treated as if they varied
                        # independently when in fact they are constrained to
                        # sum to 1. That biases the Mahalanobis distance the
                        # SMD is built from.
                        def _multinomial_cov(pv: np.ndarray) -> np.ndarray:
                            return np.diag(pv) - np.outer(pv, pv)

                        s1 = _multinomial_cov(p1_vec)
                        s2 = _multinomial_cov(p2_vec)
                        s_pool = (s1 + s2) / 2
                        diff = p1_vec - p2_vec
                        det = np.linalg.det(s_pool)
                        if det <= 1e-12:
                            return None
                        return float(np.sqrt(diff @ np.linalg.inv(s_pool) @ diff))

                    from itertools import combinations as _comb

                    # dropna before astype(str): otherwise NaN becomes the
                    # string "nan", the .dropna() inside _smd_cat_pair no
                    # longer sees it, and missingness enters the SMD as a
                    # category of its own.
                    g_series = [
                        df[df[req.group_column] == g][var].dropna().astype(str)
                        for g in groups
                    ]
                    pair_smds = []
                    for i, j in _comb(range(len(g_series)), 2):
                        s_smd = _smd_cat_pair(g_series[i], g_series[j])
                        if s_smd is not None and np.isfinite(s_smd):
                            pair_smds.append(s_smd)
                    if pair_smds:
                        cat_smd = round(max(pair_smds), 4)
                except Exception:
                    logger.exception("SMD categorical calculation failed")

            row = {
                "variable": var,
                "type": "categorical",
                "p_raw": float(p_chi_raw) if p_chi_raw is not None else None,
                "stat_label": "n (%)",
                "overall": f"n={total_all}",
                "overall_n": int(total_all),
                "p_value": p_val,
                "test": test_name,
                "significant": bool(p_chi_raw is not None and p_chi_raw < 0.05),
                "sub_rows": sub_rows,
                "missing_row": missing_row,
                "group_stats": {},
                "stat_rows": [],
                "smd": cat_smd,
            }
        rows.append(row)

    return _sanitize(
        {
            "group_column": req.group_column,
            "group_labels": group_labels,
            "group_ns": group_ns,
            "total_n": len(df),
            "warnings": warnings,
            "rows": rows,
        }
    )


# ── 9. Weighted Descriptive Statistics ────────────────────────────────────────


class WeightedDescriptiveRequest(BaseModel):
    session_id: str
    value_cols: List[str] = Field(
        validation_alias=AliasChoices("value_cols", "value_columns")
    )
    weight_col: str = Field(
        validation_alias=AliasChoices("weight_col", "weight_column")
    )
    group_col: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("group_col", "group_column"),
    )
    imputation: Optional[str] = "listwise"


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return float(np.interp(q, cw, v))


@router.post("/weighted_descriptive")
def weighted_descriptive(req: WeightedDescriptiveRequest):
    from statsmodels.stats.weightstats import DescrStatsW

    df_full = _get_df(req.session_id)
    for c in [req.weight_col, *req.value_cols] + (
        [req.group_col] if req.group_col else []
    ):
        if c not in df_full.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.value_cols:
        raise HTTPException(status_code=422, detail="Select at least one value column.")

    cols = [req.weight_col, *req.value_cols] + (
        [req.group_col] if req.group_col else []
    )
    strategy = req.imputation or "listwise"
    if strategy in ("listwise", "none", "", None):
        df = df_full[cols].copy().reset_index(drop=True)
    else:
        df = apply_imputation(df_full[cols], cols, strategy).reset_index(drop=True)
    w_all = coerce_numeric(df[req.weight_col])

    results: List[dict] = []
    for col in req.value_cols:
        x = coerce_numeric(df[col])
        mask = x.notna() & w_all.notna() & (w_all > 0)
        xv = x[mask].values.astype(float)
        wv = w_all[mask].values.astype(float)
        if len(xv) < 3:
            results.append(
                {"column": col, "error": "fewer than 3 valid weighted observations"}
            )
            continue
        d = DescrStatsW(xv, weights=wv, ddof=1)
        lo, hi = d.tconfint_mean(alpha=0.05)
        kish = float((wv.sum() ** 2) / np.sum(wv**2))
        uniq = np.unique(xv)
        row = {
            "column": col,
            "n": int(len(xv)),
            "sum_weights": round(float(wv.sum()), 4),
            "ess_kish": round(kish, 2),
            "w_mean": round(float(d.mean), 6),
            "w_sd": round(float(d.std), 6),
            "w_se": round(float(d.std_mean), 6),
            "ci_low": round(float(lo), 6),
            "ci_high": round(float(hi), 6),
            "w_median": round(_weighted_quantile(xv, wv, 0.5), 6),
            "w_q1": round(_weighted_quantile(xv, wv, 0.25), 6),
            "w_q3": round(_weighted_quantile(xv, wv, 0.75), 6),
        }
        if set(uniq.tolist()) <= {0.0, 1.0} and len(uniq) == 2:
            p = float(np.sum(wv * xv) / np.sum(wv))
            se_p = float(np.sqrt(p * (1 - p) / kish))
            row["w_proportion"] = round(p, 6)
            row["w_proportion_ci_low"] = round(
                max(0.0, p - 1.959963984540054 * se_p), 6
            )
            row["w_proportion_ci_high"] = round(
                min(1.0, p + 1.959963984540054 * se_p), 6
            )
        results.append(row)

    comparison = None
    if req.group_col:
        group_base = (
            df[df[req.group_col].notna()] if req.group_col in df.columns else df
        )
        groups = sorted_groups(group_base[req.group_col])
        if len(groups) == 2:
            col = req.value_cols[0]
            x = coerce_numeric(df[col])
            parts = []
            for g in groups:
                m = (df[req.group_col] == g) & x.notna() & w_all.notna() & (w_all > 0)
                parts.append(
                    (str(g), x[m].values.astype(float), w_all[m].values.astype(float))
                )
            if all(len(p[1]) >= 3 for p in parts):
                from statsmodels.stats.weightstats import (
                    CompareMeans,
                    DescrStatsW as _D,
                )

                d1 = _D(parts[0][1], weights=parts[0][2], ddof=1)
                d2 = _D(parts[1][1], weights=parts[1][2], ddof=1)
                cm = CompareMeans(d1, d2)
                tstat, pval, dfree = cm.ttest_ind(usevar="unequal")
                diff = float(d1.mean - d2.mean)
                lo, hi = cm.tconfint_diff(alpha=0.05, usevar="unequal")
                comparison = {
                    "variable": col,
                    "group_a": parts[0][0],
                    "group_b": parts[1][0],
                    "w_mean_a": round(float(d1.mean), 4),
                    "w_mean_b": round(float(d2.mean), 4),
                    "diff": round(diff, 4),
                    "ci_low": round(float(lo), 4),
                    "ci_high": round(float(hi), 4),
                    "t": round(float(tstat), 4),
                    "df": round(float(dfree), 2),
                    "p": round(float(pval), 6),
                }

    n_total = int((w_all.notna() & (w_all > 0)).sum())
    result_text = (
        f"Weighted descriptive statistics on n = {n_total} rows using '{req.weight_col}' as the "
        f"sampling weight (design-based, weights only). "
        + (
            f"Weighted {comparison['variable']}: {comparison['group_a']} = {comparison['w_mean_a']} vs "
            f"{comparison['group_b']} = {comparison['w_mean_b']}, Δ = {comparison['diff']} "
            f"(95% CI {comparison['ci_low']}–{comparison['ci_high']}), weighted t-test p = "
            f"{'<0.001' if comparison['p'] < 0.001 else round(comparison['p'], 3)}."
            if comparison
            else ""
        )
    )

    export_rows = [
        [
            "Variable",
            "n",
            "ESS",
            "Weighted mean",
            "Weighted SD",
            "95% CI low",
            "95% CI high",
            "Weighted median",
        ]
    ]
    for r in results:
        if "error" in r:
            continue
        export_rows.append(
            [
                r["column"],
                r["n"],
                r["ess_kish"],
                r["w_mean"],
                r["w_sd"],
                r["ci_low"],
                r["ci_high"],
                r["w_median"],
            ]
        )

    try:
        store.log_action(
            req.session_id,
            "weighted_descriptive",
            {
                "weight_col": req.weight_col,
                "n_value_cols": len(req.value_cols),
                "group_col": req.group_col,
            },
        )
    except Exception:
        logger.exception("Logging weighted descriptive action failed")

    return _sanitize(
        {
            "test": "Weighted descriptive statistics",
            "weight_col": req.weight_col,
            "n": n_total,
            "results": results,
            "comparison": comparison,
            "assumptions": [
                {
                    "name": "Weights-only design",
                    "met": True,
                    "detail": "Design-based estimation with sampling weights. Strata / cluster (full complex survey) not modelled — SEs assume independent weighted observations.",
                },
                {
                    "name": "Effective sample size",
                    "met": True,
                    "detail": "Kish's ESS = (Σw)² / Σw² reported per variable; large weight variation shrinks ESS and widens CIs.",
                },
            ],
            "result_text": result_text,
            "export_rows": export_rows,
            "r_code": (
                "library(survey)\n"
                f"des <- svydesign(ids = ~1, weights = ~{req.weight_col}, data = data)\n"
                f"svymean(~{' + '.join(req.value_cols)}, des)\n"
                + (
                    f"svyttest({req.value_cols[0]} ~ {req.group_col}, des)\n"
                    if req.group_col
                    else ""
                )
            ),
        }
    )
