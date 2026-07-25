import json as _json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from services import store, plot_render
from services.dirty_value_guard import (
    coerce_numeric,
    mask_sentinels,
    plausibility_max_for_column,
)
from services.stat_utils import sorted_groups, _categorical_p_with_rule

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


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
    return [
        {
            "variable": col,
            "n_implausible": int(mask.sum()),
            "implausible_values": sorted(
                {float(v) for v in numeric[mask].dropna().unique()}
            ),
            "rule": rule,
            "note": "Values were retained for display but should be reviewed.",
        }
    ]


class ChartRequest(BaseModel):
    session_id: str
    x: str
    y: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None
    bins: int = 20


@router.post("/histogram")
def histogram(req: ChartRequest):
    df = _get_df(req.session_id)
    if req.x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.x}' not found")
    s = coerce_numeric(df[req.x]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        raise HTTPException(
            status_code=400, detail="Need at least 2 numeric values for a histogram."
        )
    counts, edges = np.histogram(s, bins=req.bins)
    kde_x = np.linspace(s.min(), s.max(), 200)
    kde_points = []
    if len(s) >= 3 and float(s.std()) > 0:
        kde = scipy_stats.gaussian_kde(s)
        kde_points = [
            {"x": float(kx), "y": float(ky)} for kx, ky in zip(kde_x, kde(kde_x))
        ]
    return {
        "type": "histogram",
        "x": req.x,
        "bins": [
            {"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(len(counts))
        ],
        "kde": kde_points,
        "stats": {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
        },
        "warnings": _plausibility_warnings(req.x, df[req.x]),
    }


@router.post("/scatter")
def scatter(req: ChartRequest):
    df = _get_df(req.session_id)

    # Build deduplicated column list
    needed = [req.x, req.y]
    if req.color and req.color not in needed:
        needed.append(req.color)
    if req.shape and req.shape not in needed:
        needed.append(req.shape)

    for col in needed:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    # Clean: replace inf→nan on numeric cols only, then drop missing
    sub = df[needed].copy()
    for col in needed:
        if sub[col].dtype.kind in ("f", "i", "u"):
            sub[col] = sub[col].replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna()

    if len(sub) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough non-missing data points to draw scatter (need ≥ 2)",
        )

    # Regression only when both axes are numeric
    x_numeric = df[req.x].dtype.kind in ("f", "i", "u")
    y_numeric = df[req.y].dtype.kind in ("f", "i", "u")

    reg: dict = {}
    if x_numeric and y_numeric:
        x_arr = sub[req.x].astype(float).tolist()
        y_arr = sub[req.y].astype(float).tolist()
        try:
            slope, intercept, r, p, se = scipy_stats.linregress(x_arr, y_arr)
            if np.isnan(r) or np.isinf(r):
                raise ValueError("degenerate")
            line_x = [float(sub[req.x].min()), float(sub[req.x].max())]
            line_y = [float(slope * lx + intercept) for lx in line_x]
            reg = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r": float(r),
                "r2": float(r**2),
                "p": float(p),
                "se": float(se),
                "line_x": line_x,
                "line_y": line_y,
            }
        except Exception:
            reg = {
                "slope": None,
                "intercept": None,
                "r": None,
                "r2": None,
                "p": None,
                "se": None,
                "line_x": [],
                "line_y": [],
                "note": "Regression unavailable (constant or degenerate data)",
            }
    else:
        reg = {
            "slope": None,
            "intercept": None,
            "r": None,
            "r2": None,
            "p": None,
            "se": None,
            "line_x": [],
            "line_y": [],
            "note": "Regression requires two numeric axes",
        }

    # Serialize points safely (NaN → null via json round-trip)
    points = _json.loads(
        sub.to_json(
            orient="records", default_handler=str, date_format="iso", date_unit="s"
        )
    )

    return {
        "type": "scatter",
        "x": req.x,
        "y": req.y,
        "points": points,
        "regression": reg,
        "color": req.color,
    }


@router.post("/boxplot")
def boxplot(req: ChartRequest):
    df = _get_df(req.session_id)
    if req.color:
        result = []
        for grp, sub in df.groupby(req.color):
            mask = sub[req.x].notna()
            vals = sub.loc[mask, req.x].tolist()
            indices = sub.loc[mask].index.tolist()
            result.append({"group": str(grp), "values": vals, "row_indices": indices})
    else:
        mask = df[req.x].notna()
        vals = df.loc[mask, req.x].tolist()
        indices = df.loc[mask].index.tolist()
        result = [{"group": "All", "values": vals, "row_indices": indices}]
    return {"type": "boxplot", "x": req.x, "groups": result}


class PairedBoxRequest(BaseModel):
    session_id: str
    y: str
    group: str
    pair_id: str


@router.post("/paired_box")
def paired_box(req: PairedBoxRequest):
    """Matched-pair box plot: one box per group plus a connector line joining
    each pair's two values (e.g. PSM-matched cohorts, before/after cases).
    Groups are looked up via `pair_id` (typically PSM's `match_set_id` or a
    per-case ID column) rather than row position, so unsorted or filtered
    data still pairs correctly."""
    df = _get_df(req.session_id)
    for col in (req.y, req.group, req.pair_id):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    y = coerce_numeric(df[req.y]).replace([np.inf, -np.inf], np.nan)
    work = pd.DataFrame({
        "y": y,
        "group": df[req.group].astype(str),
        "pair_id": df[req.pair_id].astype(str),
    }, index=df.index)
    work = work.dropna(subset=["y"])

    levels = sorted_groups(df.loc[work.index, req.group])
    levels = [str(lvl) for lvl in levels]
    if len(levels) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Group column '{req.group}' must have exactly 2 levels (found {len(levels)}).",
        )

    groups_out = []
    side: dict[str, pd.Series] = {}
    for lvl in levels:
        sub = work[work["group"] == lvl]
        groups_out.append({
            "group": lvl, "values": sub["y"].tolist(), "row_indices": sub.index.tolist(),
            # Parallel to values/row_indices — lets the frontend seed each
            # point's jitter from the same pair_id used in `pairs`, so a
            # connector line lands exactly on its two marker positions.
            "pair_ids": sub["pair_id"].tolist(),
        })
        # First occurrence per pair_id — duplicates within a side can't be
        # unambiguously paired, so only the first is used as the pair partner.
        side[lvl] = sub.dropna(subset=["pair_id"]).drop_duplicates(subset="pair_id", keep="first").set_index("pair_id")["y"]

    g0, g1 = levels
    common_ids = sorted(set(side[g0].index) & set(side[g1].index))
    pairs = [
        {"pair_id": pid, "y0": float(side[g0][pid]), "y1": float(side[g1][pid])}
        for pid in common_ids
    ]

    return {
        "type": "paired_box",
        "y": req.y,
        "group": req.group,
        "pair_id": req.pair_id,
        "groups": groups_out,
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_unpaired": len(work) - 2 * len(pairs),
    }


class SplomRequest(BaseModel):
    session_id: str
    variables: List[str]
    color: Optional[str] = None


@router.post("/splom")
def splom(req: SplomRequest):
    df = _get_df(req.session_id)

    if len(req.variables) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 variables")

    needed = list(req.variables)
    if req.color and req.color not in needed:
        needed.append(req.color)

    for col in needed:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    sub = df[needed].replace([np.inf, -np.inf], np.nan).dropna()

    if len(sub) < 3:
        raise HTTPException(
            status_code=400,
            detail="Not enough data after removing missing values (need ≥ 3 rows)",
        )

    # Build column arrays
    data_cols = {col: sub[col].tolist() for col in req.variables}
    color_values = sub[req.color].tolist() if req.color else None

    # Pairwise Pearson r matrix
    corr: dict = {}
    for a in req.variables:
        for b in req.variables:
            if a == b:
                corr[f"{a}||{b}"] = 1.0
            else:
                key = f"{a}||{b}"
                try:
                    r, _ = scipy_stats.pearsonr(
                        sub[a].astype(float), sub[b].astype(float)
                    )
                    corr[key] = (
                        round(float(r), 4) if not (np.isnan(r) or np.isinf(r)) else None
                    )
                except Exception:
                    corr[key] = None

    return {
        "variables": req.variables,
        "n": len(sub),
        "data": data_cols,
        "color": req.color,
        "color_values": color_values,
        "corr": corr,
    }


@router.post("/bar")
def bar(req: ChartRequest):
    df = _get_df(req.session_id)
    if req.y:
        grp = df.groupby(req.x)[req.y].mean().reset_index()
        return {
            "type": "bar",
            "x": req.x,
            "y": req.y,
            "data": [
                {"label": str(row[req.x]), "value": float(row[req.y])}
                for _, row in grp.iterrows()
            ],
        }
    else:
        counts = df[req.x].value_counts()
        return {
            "type": "bar",
            "x": req.x,
            "y": "count",
            "data": [{"label": str(k), "value": int(v)} for k, v in counts.items()],
        }


# ── Forest plot ─────────────────────────────────────────────────────────────────


class ForestRow(BaseModel):
    label: str
    est: float
    ci_low: float
    ci_high: float
    weight: Optional[float] = None  # for meta-analysis weighting
    group: Optional[str] = None  # optional group label (sub-heading)
    n: Optional[int] = None  # optional sample-size annotation


class ForestRequest(BaseModel):
    rows: List[ForestRow]
    effect_label: str = "OR"  # OR / HR / RR / β / Mean difference
    x_axis: str = "log"  # "log" for OR/HR/RR, "linear" for β/diff
    null_line: float = 1.0  # reference value (1.0 for log-scale, 0 for linear)
    title: Optional[str] = None
    sort_by: Optional[str] = None  # "effect" | "p" | None (preserve order)
    # Meta-analysis (optional):
    do_meta: bool = False
    meta_method: str = "DL"  # DerSimonian-Laird random-effects


@router.post("/forest")
def forest_plot(req: ForestRequest):
    """Forest plot data + optional DerSimonian-Laird meta-analysis pool.

    Accepts a flat row array of {label, est, ci_low, ci_high, weight?, group?}
    and returns Plotly-ready traces + (when do_meta=True) a pooled diamond
    with I² heterogeneity and τ². Same backend serves two UI hooks:
    univariate-OR screening (from logistic_table) and study-level
    meta-analysis (free-form upload).
    """
    rows = [r.dict() for r in req.rows]
    if not rows:
        raise HTTPException(status_code=422, detail="rows array is empty.")
    if req.sort_by == "effect":
        rows.sort(key=lambda r: r["est"])
    # SE inferred from CI assuming symmetric on the log/linear scale.
    log_scale = req.x_axis == "log"
    for r in rows:
        if log_scale:
            if r["est"] <= 0 or r["ci_low"] <= 0 or r["ci_high"] <= 0:
                raise HTTPException(
                    status_code=422,
                    detail="Log-scale forest plots require positive est, ci_low, and ci_high values.",
                )
            r["log_est"] = float(np.log(max(r["est"], 1e-12)))
            r["log_low"] = float(np.log(max(r["ci_low"], 1e-12)))
            r["log_high"] = float(np.log(max(r["ci_high"], 1e-12)))
            r["se"] = (r["log_high"] - r["log_low"]) / (2 * 1.96)
        else:
            r["se"] = (r["ci_high"] - r["ci_low"]) / (2 * 1.96)

    meta = None
    if req.do_meta and len(rows) >= 2:
        ests = np.array(
            [r["log_est"] if log_scale else r["est"] for r in rows], dtype=float
        )
        ses = np.array([r["se"] for r in rows], dtype=float)
        wts_fe = 1.0 / (ses**2)
        wts_fe = wts_fe / wts_fe.sum()  # normalise
        # Fixed-effect mean
        mu_fe = float(np.sum(wts_fe * ests))
        # Cochran Q and τ² (DerSimonian-Laird)
        q = float(np.sum((1.0 / (ses**2)) * (ests - mu_fe) ** 2))
        dfree = len(rows) - 1
        c = float(
            np.sum(1.0 / (ses**2))
            - np.sum((1.0 / (ses**2)) ** 2) / np.sum(1.0 / (ses**2))
        )
        tau2 = max(0.0, (q - dfree) / c if c > 0 else 0.0)
        # Random-effects re-weighting
        wts_re = 1.0 / (ses**2 + tau2)
        mu_re = float(np.sum(wts_re * ests) / np.sum(wts_re))
        var_re = float(1.0 / np.sum(wts_re))
        se_re = float(np.sqrt(var_re))
        ci_low_re = float(mu_re - 1.96 * se_re)
        ci_high_re = float(mu_re + 1.96 * se_re)
        i2 = max(0.0, (q - dfree) / q * 100.0) if q > 0 else 0.0
        from scipy.stats import chi2 as _chi2

        q_p = float(1 - _chi2.cdf(q, dfree)) if dfree > 0 else 1.0
        if log_scale:
            pooled_est = float(np.exp(mu_re))
            pooled_low = float(np.exp(ci_low_re))
            pooled_high = float(np.exp(ci_high_re))
        else:
            pooled_est = float(mu_re)
            pooled_low = float(ci_low_re)
            pooled_high = float(ci_high_re)
        meta = {
            "method": req.meta_method,
            "pooled_est": round(pooled_est, 4),
            "pooled_ci_low": round(pooled_low, 4),
            "pooled_ci_high": round(pooled_high, 4),
            "tau2": round(tau2, 6),
            "Q": round(q, 4),
            "Q_df": dfree,
            "Q_p": round(q_p, 4),
            "I_squared_pct": round(i2, 2),
            "k_studies": len(rows),
            "result_text": (
                f"DerSimonian-Laird random-effects meta-analysis (k = {len(rows)} studies). "
                f"Pooled {req.effect_label} = {pooled_est:.3f} (95% CI {pooled_low:.3f}–{pooled_high:.3f}). "
                f"Heterogeneity Q({dfree}) = {q:.2f}, p = {q_p:.4f}, I² = {i2:.1f}%, τ² = {tau2:.4f}."
            ),
        }

    return {
        "type": "forest",
        "effect_label": req.effect_label,
        "x_axis": req.x_axis,
        "null_line": req.null_line,
        "title": req.title,
        "rows": rows,
        "meta": meta,
    }


class SubgroupBarRequest(BaseModel):
    session_id: str
    y_col: str
    subgroup_col: str
    xaxis_col: str
    color_col: Optional[str] = None
    y_mode: str = "mean"  # "mean" or "percentage"
    target_value: Optional[str] = None
    error_type: str = "ci"  # "ci", "se", "sd", "none"


@router.post("/subgroup_bar")
def subgroup_bar(req: SubgroupBarRequest):
    df = _get_df(req.session_id)

    # Check if selected columns exist
    for col in [req.y_col, req.subgroup_col, req.xaxis_col]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    if req.color_col and req.color_col not in df.columns:
        raise HTTPException(
            status_code=400, detail=f"Column '{req.color_col}' not found"
        )

    # Get subset of columns
    cols_to_use = [req.y_col, req.subgroup_col, req.xaxis_col]
    if req.color_col:
        cols_to_use.append(req.color_col)

    sub = df[cols_to_use].copy()

    # Drop missing in grouping columns, but handle Y missing per cell safely
    sub = sub.dropna(
        subset=[req.subgroup_col, req.xaxis_col]
        + ([req.color_col] if req.color_col else [])
    )

    if len(sub) == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid data points found after dropping missing values in grouping variables.",
        )

    # Get unique groups, ordered by value code (numeric when coercible, else
    # string) so multi-digit codes (1, 2, 10) don't sort as 1, 10, 2.
    subgroups = sorted_groups(sub[req.subgroup_col])
    x_vals = sorted_groups(sub[req.xaxis_col])
    color_groups = sorted_groups(sub[req.color_col]) if req.color_col else x_vals
    warnings = []
    if req.y_mode == "mean":
        max_plausible = plausibility_max_for_column(req.y_col)
        raw_y = sub[req.y_col]
        masked_y = mask_sentinels(raw_y, max_plausible)
        if masked_y.isna().sum() > coerce_numeric(raw_y).isna().sum():
            warnings.append(
                {
                    "variable": req.y_col,
                    "note": "Implausible high sentinel values were treated as missing for mean bars.",
                }
            )
        sub[req.y_col] = masked_y

    # ── Percentage "success" level — resolved ONCE over the whole subset, not
    # per cell. Picking it per cell (the old behaviour) let different bars
    # measure different levels, so the chart was not comparable.
    pct_target = None
    if req.y_mode == "percentage":
        pct_target = req.target_value
        if pct_target is None:
            levels = sorted(str(v) for v in sub[req.y_col].dropna().unique())
            if "1" in levels:
                pct_target = "1"
            elif "1.0" in levels:
                pct_target = "1.0"
            elif levels:
                pct_target = levels[-1]  # deterministic fallback
            else:
                pct_target = "1"

    Z = 1.959963984540054  # 95% normal quantile

    def _wilson_pct(successes: int, n: int) -> tuple:
        """Wilson score interval (×100). Returns (point%, low%, high%)."""
        if n == 0:
            return 0.0, 0.0, 0.0
        p = successes / n
        denom = 1.0 + Z * Z / n
        center = (p + Z * Z / (2 * n)) / denom
        half = (Z / denom) * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
        return (
            p * 100.0,
            max(0.0, center - half) * 100.0,
            min(1.0, center + half) * 100.0,
        )

    traces = []
    for cg in color_groups:
        tr = {
            "name": str(cg),
            "x_subgroup": [],
            "x_xaxis": [],
            "y": [],
            "error": [],
            "error_low": [],
            "error_high": [],
            "ns": [],
        }
        for sg in subgroups:
            iter_x_vals = x_vals if req.color_col else [cg]
            for xv in iter_x_vals:
                mask = (sub[req.subgroup_col] == sg) & (sub[req.xaxis_col] == xv)
                if req.color_col:
                    mask = mask & (sub[req.color_col] == cg)
                cell = sub.loc[mask, req.y_col].dropna()
                n = int(len(cell))

                val, e_low, e_high = 0.0, 0.0, 0.0
                if n > 0 and req.y_mode == "percentage":
                    successes = int((cell.astype(str) == str(pct_target)).sum())
                    p = successes / n
                    val, lo, hi = _wilson_pct(successes, n)
                    se = np.sqrt(p * (1 - p) / n) * 100.0
                    sd = np.sqrt(p * (1 - p)) * 100.0
                    if req.error_type == "ci":
                        e_low, e_high = (
                            max(0.0, val - lo),
                            max(0.0, hi - val),
                        )  # asymmetric (Wilson)
                    elif req.error_type == "se":
                        e_low = e_high = se
                    elif req.error_type == "sd":
                        e_low = e_high = sd
                elif n > 0:
                    nums = pd.to_numeric(cell, errors="coerce").dropna()
                    m = int(len(nums))
                    if m > 0:
                        val = float(nums.mean())
                        sd = float(nums.std(ddof=1)) if m > 1 else 0.0
                        se = sd / np.sqrt(m)
                        if req.error_type == "ci":
                            tcrit = (
                                float(scipy_stats.t.ppf(0.975, m - 1)) if m > 1 else 0.0
                            )
                            e_low = e_high = tcrit * se  # t-distribution CI half-width
                        elif req.error_type == "se":
                            e_low = e_high = se
                        elif req.error_type == "sd":
                            e_low = e_high = sd

                tr["x_subgroup"].append(str(sg))
                tr["x_xaxis"].append(str(xv))
                tr["y"].append(val)
                tr["error"].append(e_high)  # legacy symmetric field (= upper offset)
                tr["error_low"].append(e_low)
                tr["error_high"].append(e_high)
                tr["ns"].append(n)
        traces.append(tr)

    _err_label = {
        "ci": "95% CI",
        "se": "± 1 SE",
        "sd": "± 1 SD",
        "none": "no error bars",
    }.get(req.error_type, req.error_type)
    return {
        "type": "subgroup_bar",
        "y_col": req.y_col,
        "subgroup_col": req.subgroup_col,
        "xaxis_col": req.xaxis_col,
        "color_col": req.color_col,
        "y_mode": req.y_mode,
        "target_value": pct_target if req.y_mode == "percentage" else req.target_value,
        "error_type": req.error_type,
        "traces": traces,
        "warnings": warnings,
        "method_note": (
            "Means use a t-distribution CI (t_{n−1}); percentages use the Wilson "
            "score interval (bounded to 0–100%, accurate for small n and extreme "
            f"proportions). Error bars show {_err_label}."
        ),
    }


class ScoreFigureSpec(BaseModel):
    score_col: str
    label: Optional[str] = None
    components: List[str]
    component_labels: Optional[Dict[str, str]] = None


class ScoreCompositeRequest(BaseModel):
    session_id: str
    group_col: str
    scores: List[ScoreFigureSpec]
    group_order: Optional[List[str]] = None
    bins: int = 8
    title: Optional[str] = None
    positive_values: List[str] = ["1", "true", "yes", "y", "present", "positive"]


def _format_p_value(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "p = NA"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _score_group_pvalue(
    score: pd.Series, groups: pd.Series, group_levels: list[str]
) -> Optional[float]:
    samples = [
        pd.to_numeric(score[groups == g], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        for g in group_levels
    ]
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2:
        return None
    try:
        if len(samples) == 2:
            return float(
                scipy_stats.mannwhitneyu(
                    samples[0], samples[1], alternative="two-sided"
                ).pvalue
            )
        return float(scipy_stats.kruskal(*samples).pvalue)
    except Exception:
        return None


def _component_positive(series: pd.Series, positive_values: list[str]) -> pd.Series:
    non_missing = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_positive = numeric.notna() & (numeric > 0)
    normalized = series.astype(str).str.strip().str.lower()
    string_positive = normalized.isin({str(v).strip().lower() for v in positive_values})
    return non_missing & (numeric_positive | string_positive)


def _component_pvalue(
    values: pd.Series,
    groups: pd.Series,
    group_levels: list[str],
    positive_values: list[str],
) -> tuple[Optional[float], Optional[str]]:
    valid = values.notna() & groups.notna()
    if not valid.any():
        return None, None
    pos = _component_positive(values[valid], positive_values)
    valid_groups = groups[valid]
    table = []
    for g in group_levels:
        mask = valid_groups == g
        positives = int(pos[mask].sum())
        total = int(mask.sum())
        table.append([positives, max(total - positives, 0)])
    if len(table) < 2 or any(sum(row) == 0 for row in table):
        return None, None
    try:
        p_value, test_name = _categorical_p_with_rule(np.array(table))
        return float(p_value), test_name
    except Exception:
        return None, None


def _score_xbins(values: pd.Series, bins: int) -> dict:
    clean = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if clean.empty:
        return {"nbinsx": max(2, int(bins))}
    arr = clean.to_numpy(dtype=float)
    is_integer = np.all(np.isclose(arr, np.round(arr)))
    if is_integer and len(np.unique(arr)) <= 50:
        return {
            "xbins": {
                "start": float(np.floor(arr.min()) - 0.5),
                "end": float(np.ceil(arr.max()) + 0.5),
                "size": 1,
            }
        }
    return {"nbinsx": max(2, int(bins))}


@router.post("/score_composite")
def score_composite(req: ScoreCompositeRequest):
    """Build a manuscript-style score distribution + component prevalence figure.

    The endpoint is intentionally generic: any two score columns can be compared
    across a grouping column, with each score paired to its own binary component
    columns. It returns both computed summaries and a Plotly-ready 5-panel
    figure matching the common clinical manuscript layout:
    score histograms, score overlap boxplots, and grouped component prevalence
    bars.
    """
    df = _get_df(req.session_id)
    if req.group_col not in df.columns:
        raise HTTPException(
            status_code=400, detail=f"Column '{req.group_col}' not found"
        )
    if len(req.scores) != 2:
        raise HTTPException(
            status_code=400,
            detail="Select exactly two score columns for a 5-panel score-composite figure.",
        )

    needed = [req.group_col]
    for spec in req.scores:
        if spec.score_col not in df.columns:
            raise HTTPException(
                status_code=400, detail=f"Column '{spec.score_col}' not found"
            )
        if not spec.components:
            raise HTTPException(
                status_code=400,
                detail=f"Select at least one component for '{spec.score_col}'.",
            )
        needed.append(spec.score_col)
        for comp in spec.components:
            if comp not in df.columns:
                raise HTTPException(
                    status_code=400, detail=f"Column '{comp}' not found"
                )
            needed.append(comp)

    sub = df[list(dict.fromkeys(needed))].copy()
    sub["_score_group"] = sub[req.group_col].astype(str)
    if req.group_order:
        group_levels = [
            str(g)
            for g in req.group_order
            if str(g) in set(sub["_score_group"].dropna())
        ]
    else:
        group_levels = [str(g) for g in pd.unique(sub["_score_group"].dropna())]
    if len(group_levels) < 2:
        raise HTTPException(
            status_code=400, detail="Need at least two non-missing groups."
        )

    colors = ["#4f86c6", "#dd7b6e", "#6fbf73", "#9b6ec8", "#d19a2e"]
    axis_ids = [
        ("x", "y"),
        ("x2", "y2"),
        ("x3", "y3"),
        ("x4", "y4"),
        ("x5", "y5"),
    ]
    trace_data: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    shapes: list[dict[str, Any]] = []
    score_summaries: list[dict[str, Any]] = []
    component_ticks: list[dict[str, list[Any]]] = []

    panel_titles = [
        f"A  {req.scores[0].label or req.scores[0].score_col} Score Distribution",
        f"B  {req.scores[1].label or req.scores[1].score_col} Score Distribution",
        "C  Score Overlap (Box Plots)",
        f"D  {req.scores[0].label or req.scores[0].score_col} Component Prevalence",
        f"E  {req.scores[1].label or req.scores[1].score_col} Component Prevalence",
    ]
    title_positions = [
        (0.0, 1.06),
        (0.37, 1.06),
        (0.74, 1.06),
        (0.0, 0.47),
        (0.74, 0.47),
    ]
    for text, (xpos, ypos) in zip(panel_titles, title_positions):
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": xpos,
                "y": ypos,
                "text": f"<b>{text}</b>",
                "showarrow": False,
                "xanchor": "left",
                "font": {"size": 14, "color": "#111827"},
            }
        )

    for score_idx, spec in enumerate(req.scores):
        label = spec.label or spec.score_col
        score_values = pd.to_numeric(sub[spec.score_col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        p_value = _score_group_pvalue(score_values, sub["_score_group"], group_levels)
        n_by_group = {
            g: int(score_values[sub["_score_group"] == g].dropna().shape[0])
            for g in group_levels
        }
        score_summaries.append(
            {
                "score_col": spec.score_col,
                "label": label,
                "p_value": p_value,
                "p_text": _format_p_value(p_value),
                "n_by_group": n_by_group,
                "components": [],
            }
        )

        xref, yref = axis_ids[score_idx]
        bin_kwargs = _score_xbins(score_values, req.bins)
        for group_idx, group in enumerate(group_levels):
            vals = (
                score_values[sub["_score_group"] == group]
                .dropna()
                .astype(float)
                .tolist()
            )
            trace = {
                "type": "histogram",
                "x": vals,
                "name": f"{group} (n={len(vals)})",
                "marker": {
                    "color": colors[group_idx % len(colors)],
                    "line": {"color": "white", "width": 0.5},
                },
                "opacity": 0.78,
                "showlegend": score_idx == 0,
                "legendgroup": group,
                "xaxis": xref,
                "yaxis": yref,
            }
            trace.update(bin_kwargs)
            trace_data.append(trace)

            box_x = [f"{group}<br>{label}"] * len(vals)
            trace_data.append(
                {
                    "type": "box",
                    "x": box_x,
                    "y": vals,
                    "name": group,
                    "marker": {"color": colors[group_idx % len(colors)]},
                    "line": {"color": "#111827", "width": 1},
                    "boxpoints": "outliers" if len(vals) < 500 else False,
                    "showlegend": False,
                    "legendgroup": group,
                    "xaxis": "x3",
                    "yaxis": "y3",
                }
            )

        annotations.append(
            {
                "xref": f"{xref} domain",
                "yref": f"{yref} domain",
                "x": 0.86,
                "y": 0.9,
                "text": _format_p_value(p_value),
                "showarrow": False,
                "font": {"size": 12, "color": "#4b5563"},
            }
        )

    box_max_candidates = [
        float(
            pd.to_numeric(sub[spec.score_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .max()
        )
        for spec in req.scores
    ]
    box_max_candidates = [v for v in box_max_candidates if np.isfinite(v)]
    box_y_max = max(box_max_candidates or [1.0])
    for score_idx, spec in enumerate(req.scores):
        label = spec.label or spec.score_col
        x0 = f"{group_levels[0]}<br>{label}"
        x1 = f"{group_levels[-1]}<br>{label}"
        y = box_y_max * (0.94 if score_idx == 0 else 0.88)
        shapes.append(
            {
                "type": "line",
                "xref": "x3",
                "yref": "y3",
                "x0": x0,
                "x1": x1,
                "y0": y,
                "y1": y,
                "line": {"color": "#9ca3af", "width": 1},
            }
        )
        annotations.append(
            {
                "xref": "x3",
                "yref": "y3",
                "x": x0 if score_idx == 0 else x1,
                "y": y + max(box_y_max * 0.04, 0.5),
                "text": score_summaries[score_idx]["p_text"],
                "showarrow": False,
                "font": {"size": 12, "color": "#4b5563"},
            }
        )

    for score_idx, spec in enumerate(req.scores):
        xref, yref = axis_ids[3 + score_idx]
        labels = [
            spec.component_labels.get(c, c) if spec.component_labels else c
            for c in spec.components
        ]
        component_ticks.append(
            {"tickvals": list(range(len(labels))), "ticktext": labels}
        )
        component_rows = []
        bar_width = 0.8 / max(len(group_levels), 1)
        for group_idx, group in enumerate(group_levels):
            y_values = []
            text_values = []
            for comp in spec.components:
                mask = sub["_score_group"] == group
                raw = sub.loc[mask, comp]
                valid = raw.notna()
                n = int(valid.sum())
                positives = (
                    int(_component_positive(raw[valid], req.positive_values).sum())
                    if n
                    else 0
                )
                prevalence = (positives / n * 100.0) if n else 0.0
                y_values.append(prevalence)
                text_values.append(f"{positives}/{n}")
            x_positions = [
                comp_idx - 0.4 + (group_idx + 0.5) * bar_width
                for comp_idx in range(len(labels))
            ]
            trace_data.append(
                {
                    "type": "bar",
                    "x": x_positions,
                    "y": y_values,
                    "width": [bar_width * 0.92] * len(labels),
                    "name": group,
                    "text": text_values,
                    "textposition": "none",
                    "customdata": labels,
                    "hovertemplate": "%{customdata}<br>%{fullData.name}: %{y:.1f}% (%{text})<extra></extra>",
                    "marker": {"color": colors[group_idx % len(colors)]},
                    "showlegend": score_idx == 1,
                    "legendgroup": group,
                    "xaxis": xref,
                    "yaxis": yref,
                }
            )

        for comp_idx, (comp, label) in enumerate(zip(spec.components, labels)):
            p_value, test_name = _component_pvalue(
                sub[comp], sub["_score_group"], group_levels, req.positive_values
            )
            component_rows.append(
                {
                    "component": comp,
                    "label": label,
                    "p_value": p_value,
                    "p_text": _format_p_value(p_value),
                    "test": test_name,
                }
            )
            annotations.append(
                {
                    "xref": xref,
                    "yref": yref,
                    "x": comp_idx,
                    "y": 103,
                    "text": "ns"
                    if p_value is not None and p_value >= 0.05
                    else _format_p_value(p_value),
                    "showarrow": False,
                    "font": {"size": 11, "color": "#9ca3af"},
                }
            )
        score_summaries[score_idx]["components"] = component_rows

    layout = {
        "title": {
            "text": req.title
            or "Score Distributions and Component Prevalence by Group",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": "#111827"},
        },
        "height": 760,
        "barmode": "overlay",
        "boxmode": "group",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "Arial, sans-serif", "size": 12, "color": "#111827"},
        "margin": {"l": 70, "r": 30, "t": 90, "b": 90},
        "legend": {
            "orientation": "v",
            "x": 0.98,
            "y": 0.98,
            "xanchor": "right",
            "yanchor": "top",
        },
        "annotations": annotations,
        "shapes": shapes,
        "xaxis": {
            "domain": [0.0, 0.29],
            "anchor": "y",
            "title": {"text": req.scores[0].label or req.scores[0].score_col},
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        "yaxis": {
            "domain": [0.57, 1.0],
            "anchor": "x",
            "title": {"text": "Number of Patients"},
            "gridcolor": "#e5e7eb",
            "rangemode": "tozero",
        },
        "xaxis2": {
            "domain": [0.36, 0.65],
            "anchor": "y2",
            "title": {"text": req.scores[1].label or req.scores[1].score_col},
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        "yaxis2": {
            "domain": [0.57, 1.0],
            "anchor": "x2",
            "title": {"text": "Number of Patients"},
            "gridcolor": "#e5e7eb",
            "rangemode": "tozero",
        },
        "xaxis3": {
            "domain": [0.72, 1.0],
            "anchor": "y3",
            "tickangle": 0,
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        "yaxis3": {
            "domain": [0.57, 1.0],
            "anchor": "x3",
            "title": {"text": "Score Value"},
            "gridcolor": "#e5e7eb",
            "rangemode": "tozero",
        },
        "xaxis4": {
            "domain": [0.0, 0.68],
            "anchor": "y4",
            "tickangle": 0,
            "gridcolor": "#e5e7eb",
            "zeroline": False,
            **component_ticks[0],
        },
        "yaxis4": {
            "domain": [0.0, 0.42],
            "anchor": "x4",
            "title": {"text": "Prevalence (%)"},
            "range": [0, 108],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        "xaxis5": {
            "domain": [0.76, 1.0],
            "anchor": "y5",
            "tickangle": 0,
            "gridcolor": "#e5e7eb",
            "zeroline": False,
            **component_ticks[1],
        },
        "yaxis5": {
            "domain": [0.0, 0.42],
            "anchor": "x5",
            "title": {"text": "Prevalence (%)"},
            "range": [0, 108],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
    }

    return {
        "type": "score_composite",
        "group_col": req.group_col,
        "groups": group_levels,
        "scores": score_summaries,
        "figure": {"data": trace_data, "layout": layout},
        "method_note": (
            "Score comparisons use Mann-Whitney U for two groups or Kruskal-Wallis for more groups. "
            "Component prevalence uses chi-square when all expected cells are ≥ 5; otherwise Fisher exact "
            "for 2×2 tables or Fisher-Freeman-Halton Monte Carlo for larger r×c tables."
        ),
    }


# ── Kaplan-Meier composite (NEJM-style multi-endpoint cumulative incidence) ──


class KMEndpointSpec(BaseModel):
    duration_col: str
    event_col: str
    label: Optional[str] = None


class KMCompositeRequest(BaseModel):
    session_id: str
    group_col: str  # treatment arm / comparison column
    endpoints: List[KMEndpointSpec]  # 1-4 endpoints, one panel each
    risk_times: Optional[List[float]] = None  # x-axis ticks for No.-at-risk
    group_order: Optional[List[str]] = None
    # 1 - S(t) climbing from 0 (event accrual) vs S(t) falling from 1.
    as_cumulative_incidence: bool = True
    inset: bool = True  # magnified zoom sub-panel per endpoint
    inset_max_pct: Optional[float] = None  # inset y-max in %; None = auto
    as_percent: bool = True  # y in % (0-100) vs proportion (0-1)
    imputation: str = "listwise"
    title: Optional[str] = None


_KM_COMPOSITE_COLORS = ["#9ca3af", "#1f6f8b", "#dd7b6e", "#6fbf73", "#9b6ec8"]
_PANEL_LETTERS = ["A", "B", "C", "D"]


def _km_composite_grid(n_panels: int) -> tuple[int, int]:
    """Column/row count for the panel grid (max 2 columns, NEJM-style)."""
    ncols = 1 if n_panels == 1 else 2
    nrows = (n_panels + ncols - 1) // ncols
    return ncols, nrows


@router.post("/km_composite")
def km_composite(req: KMCompositeRequest):
    """NEJM-style composite Kaplan-Meier figure.

    Runs one KM analysis per endpoint against a shared grouping (arm) column
    and lays the panels out in a 2-column grid. Each panel plots cumulative
    incidence (1 - S(t)) as step curves, optionally with a magnified zoom
    inset and a per-panel No.-at-risk table, matching the common trial
    primary-endpoint figure. Curves and log-rank p-values are computed with
    the same lifelines helpers as the main survival panel.
    """
    from routers.models.cox import (
        _km_fit_groups,
        _km_logrank,
        _drop_invalid_survival_rows,
    )
    from services.impute import apply_imputation

    df_full = _get_df(req.session_id)
    if req.group_col not in df_full.columns:
        raise HTTPException(
            status_code=400, detail=f"Column '{req.group_col}' not found"
        )
    if not (1 <= len(req.endpoints) <= 4):
        raise HTTPException(status_code=400, detail="Select between 1 and 4 endpoints.")
    for spec in req.endpoints:
        for col in (spec.duration_col, spec.event_col):
            if col not in df_full.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    scale = 100.0 if req.as_percent else 1.0
    ncols, nrows = _km_composite_grid(len(req.endpoints))

    # Grid geometry (paper fractions). Rows leave a gap beneath each panel for
    # the No.-at-risk rows; columns leave a left gutter for the y-axis title.
    left_pad, right_pad, col_gap = 0.07, 0.02, 0.09
    top_pad, bottom_pad, row_gap = 0.05, 0.16, 0.19
    # Vertical offsets for the No.-at-risk block below each panel. `risk_gap`
    # clears the axis tick labels; `risk_row_h` is the per-arm row pitch.
    risk_gap, risk_row_h = 0.055, 0.028
    usable_w = 1.0 - left_pad - right_pad - (ncols - 1) * col_gap
    cell_w = usable_w / ncols
    usable_h = 1.0 - top_pad - bottom_pad - (nrows - 1) * row_gap
    cell_h = usable_h / nrows

    colors = _KM_COMPOSITE_COLORS
    trace_data: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    layout_axes: dict[str, Any] = {}
    endpoint_summaries: list[dict[str, Any]] = []

    # Establish a stable arm order once (from the first endpoint's clean data),
    # so colors/legend/at-risk rows line up across every panel.
    group_levels: Optional[list[str]] = None

    for idx, spec in enumerate(req.endpoints):
        label = spec.label or spec.event_col
        cols = [spec.duration_col, spec.event_col, req.group_col]
        sub = df_full[cols].copy()
        sub[spec.duration_col] = pd.to_numeric(sub[spec.duration_col], errors="coerce")
        sub[spec.event_col] = pd.to_numeric(sub[spec.event_col], errors="coerce")
        sub = apply_imputation(sub, [spec.duration_col, spec.event_col], req.imputation)
        sub, _w, _ni = _drop_invalid_survival_rows(
            sub, spec.duration_col, spec.event_col
        )
        sub = sub.dropna(subset=[req.group_col])
        if sub.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No valid rows for endpoint '{label}' after cleaning duration/event.",
            )
        ev_vals = sorted(
            pd.to_numeric(sub[spec.event_col], errors="coerce").dropna().unique()
        )
        if set(ev_vals) - {0, 1, 0.0, 1.0}:
            raise HTTPException(
                status_code=422,
                detail=f"Event column '{spec.event_col}' must be binary 0/1. Found: {ev_vals[:8]}",
            )

        groups = _km_fit_groups(
            sub,
            spec.duration_col,
            spec.event_col,
            req.group_col,
            survival_times=None,
            risk_times=req.risk_times,
            include_censors=False,
        )
        present = [g["group"] for g in groups]
        if group_levels is None:
            if req.group_order:
                ordered = [str(g) for g in req.group_order if str(g) in set(present)]
                group_levels = ordered or present
            else:
                group_levels = present
            if len(group_levels) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Need at least two non-missing groups in the arm column.",
                )
        by_group = {g["group"]: g for g in groups}

        logrank = _km_logrank(sub, spec.duration_col, spec.event_col, req.group_col)
        p_value = logrank.get("p") if logrank else None

        main_ref = idx + 1
        inset_ref = idx + 5
        main_x = "x" if main_ref == 1 else f"x{main_ref}"
        main_y = "y" if main_ref == 1 else f"y{main_ref}"
        inset_x = f"x{inset_ref}"
        inset_y = f"y{inset_ref}"

        r, c = divmod(idx, ncols)
        x0 = left_pad + c * (cell_w + col_gap)
        x1 = x0 + cell_w
        y1 = 1.0 - top_pad - r * (cell_h + row_gap)
        y0 = y1 - cell_h

        # Curve max time (shared x range for at-risk positioning).
        tmax = 0.0
        curves: dict[str, dict[str, list[float]]] = {}
        for grp in group_levels:
            g = by_group.get(grp)
            pts = g["curve"] if g else []
            xs = [p["time"] for p in pts if p["time"] is not None]
            ys = [
                (1.0 - p["survival"]) * scale
                if req.as_cumulative_incidence
                else p["survival"] * scale
                for p in pts
                if p["survival"] is not None
            ]
            curves[grp] = {"x": xs, "y": ys}
            if xs:
                tmax = max(tmax, max(xs))
        tmax = tmax or 1.0

        final_by_group: dict[str, float] = {}
        n_by_group: dict[str, int] = {}
        inset_peak = 0.0
        for gi, grp in enumerate(group_levels):
            xs = curves[grp]["x"]
            ys = curves[grp]["y"]
            g = by_group.get(grp)
            n_by_group[grp] = int(g["n"]) if g else 0
            final_by_group[grp] = round(ys[-1], 1) if ys else 0.0
            inset_peak = max(inset_peak, ys[-1] if ys else 0.0)
            color = colors[gi % len(colors)]
            trace_data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": xs,
                    "y": ys,
                    "line": {"color": color, "width": 2, "shape": "hv"},
                    "name": str(grp),
                    "legendgroup": str(grp),
                    "showlegend": idx == 0,
                    "xaxis": main_x,
                    "yaxis": main_y,
                    "hovertemplate": f"{grp}<br>%{{x}}: %{{y:.1f}}<extra></extra>",
                }
            )
            if req.inset:
                trace_data.append(
                    {
                        "type": "scatter",
                        "mode": "lines",
                        "x": xs,
                        "y": ys,
                        "line": {"color": color, "width": 1.6, "shape": "hv"},
                        "name": str(grp),
                        "legendgroup": str(grp),
                        "showlegend": False,
                        "xaxis": inset_x,
                        "yaxis": inset_y,
                        "hoverinfo": "skip",
                    }
                )

        # Full-range y for the main panel.
        y_full = 100.0 if req.as_percent else 1.0
        # Inset y-max: explicit, else a little above the peak accrual.
        if req.inset:
            if req.inset_max_pct is not None:
                inset_top = req.inset_max_pct / (100.0 / scale)
            else:
                inset_top = max(inset_peak * 1.25, scale * 0.02)

        # Main axes.
        main_x_key = "xaxis" if main_ref == 1 else f"xaxis{main_ref}"
        main_y_key = "yaxis" if main_ref == 1 else f"yaxis{main_ref}"
        is_bottom_row = r == nrows - 1
        risk_times = req.risk_times or []
        # The built-in axis title would land on the No.-at-risk numbers, so when
        # a risk table is drawn we suppress it and add our own annotation below
        # the table instead.
        show_axis_title = is_bottom_row and not risk_times
        layout_axes[main_x_key] = {
            "domain": [x0, x1],
            "anchor": main_y,
            "range": [0, tmax],
            "title": {"text": "Months since Randomization"}
            if show_axis_title
            else None,
            "gridcolor": "#eef1f4",
            "zeroline": False,
        }
        layout_axes[main_y_key] = {
            "domain": [y0, y1],
            "anchor": main_x,
            "range": [0, y_full],
            "title": {"text": "Percentage of Patients"} if c == 0 else None,
            "gridcolor": "#eef1f4",
            "zeroline": False,
        }
        if req.inset:
            ix0 = x0 + 0.34 * cell_w
            ix1 = x1 - 0.02 * cell_w
            iy0 = y0 + 0.40 * cell_h
            iy1 = y1 - 0.02 * cell_h
            layout_axes[f"xaxis{inset_ref}"] = {
                "domain": [ix0, ix1],
                "anchor": inset_y,
                "range": [0, tmax],
                "showgrid": False,
                "zeroline": False,
                "tickfont": {"size": 8},
                "ticklen": 2,
            }
            layout_axes[f"yaxis{inset_ref}"] = {
                "domain": [iy0, iy1],
                "anchor": inset_x,
                "range": [0, inset_top],
                "showgrid": False,
                "zeroline": False,
                "tickfont": {"size": 8},
                "ticklen": 2,
            }
            # Final cumulative-incidence value label at the end of each inset curve.
            for gi, grp in enumerate(group_levels):
                if curves[grp]["y"]:
                    annotations.append(
                        {
                            "xref": inset_x,
                            "yref": inset_y,
                            "x": tmax,
                            "y": curves[grp]["y"][-1],
                            "text": f"{final_by_group[grp]:.1f}",
                            "showarrow": False,
                            "xanchor": "left",
                            "xshift": 2,
                            "font": {"size": 9, "color": colors[gi % len(colors)]},
                        }
                    )

        # Panel letter + endpoint title.
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": x0,
                "y": min(y1 + 0.035, 1.0),
                "text": f"<b>{_PANEL_LETTERS[idx]}</b>  {label}",
                "showarrow": False,
                "xanchor": "left",
                "font": {"size": 12, "color": "#111827"},
            }
        )
        # Log-rank p on the main panel.
        annotations.append(
            {
                "xref": main_x,
                "yref": main_y,
                "x": tmax * 0.5,
                "y": y_full * 0.9,
                "text": _format_p_value(p_value),
                "showarrow": False,
                "font": {"size": 11, "color": "#4b5563"},
            }
        )

        # No.-at-risk rows beneath the panel.
        if risk_times:
            annotations.append(
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": x0,
                    "y": y0 - risk_gap,
                    "text": "<b>No. at Risk</b>",
                    "showarrow": False,
                    "xanchor": "left",
                    "font": {"size": 9, "color": "#374151"},
                }
            )
            for gi, grp in enumerate(group_levels):
                g = by_group.get(grp)
                at_risk = g.get("at_risk") if g else None
                row_y = y0 - risk_gap - (gi + 1) * risk_row_h
                # Arm name in the left gutter so it never collides with the
                # t=0 count that sits at the panel's left edge.
                annotations.append(
                    {
                        "xref": "paper",
                        "yref": "paper",
                        "x": max(x0 - 0.06, 0.002),
                        "y": row_y,
                        "text": str(grp),
                        "showarrow": False,
                        "xanchor": "left",
                        "font": {"size": 9, "color": colors[gi % len(colors)]},
                    }
                )
                if at_risk:
                    for ti, t in enumerate(risk_times):
                        px = x0 + (float(t) / tmax) * cell_w if tmax else x0
                        annotations.append(
                            {
                                "xref": "paper",
                                "yref": "paper",
                                "x": min(max(px, x0), x1),
                                "y": row_y,
                                "text": str(at_risk[ti]) if ti < len(at_risk) else "",
                                "showarrow": False,
                                "xanchor": "center",
                                "font": {"size": 9, "color": "#374151"},
                            }
                        )
            if is_bottom_row:
                # X-axis title below the at-risk table (suppressed on the axis).
                annotations.append(
                    {
                        "xref": "paper",
                        "yref": "paper",
                        "x": (x0 + x1) / 2.0,
                        "y": y0 - risk_gap - (len(group_levels) + 1) * risk_row_h,
                        "text": "Months since Randomization",
                        "showarrow": False,
                        "xanchor": "center",
                        "font": {"size": 12, "color": "#111827"},
                    }
                )

        endpoint_summaries.append(
            {
                "label": label,
                "duration_col": spec.duration_col,
                "event_col": spec.event_col,
                "p_value": p_value,
                "p_text": _format_p_value(p_value),
                "final_by_group": final_by_group,
                "n_by_group": n_by_group,
            }
        )

    # Drop None-valued axis titles (Plotly rejects title:{text:None} noisily).
    for axis in layout_axes.values():
        if axis.get("title") is None:
            axis.pop("title", None)

    layout = {
        "title": {
            "text": req.title
            or "Composite Primary End Point and Individual Components",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 15, "color": "#111827"},
        },
        "height": 760 if nrows > 1 else 440,
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "Arial, sans-serif", "size": 12, "color": "#111827"},
        "margin": {"l": 60, "r": 30, "t": 70, "b": 40},
        # Curves are keyed by the arm-name labels in the No.-at-risk rows and
        # the colored end-of-inset value labels, matching the trial-figure
        # convention, so no separate legend box is drawn.
        "showlegend": False,
        "annotations": annotations,
        **layout_axes,
    }

    return {
        "type": "km_composite",
        "group_col": req.group_col,
        "groups": group_levels or [],
        "endpoints": endpoint_summaries,
        "as_cumulative_incidence": req.as_cumulative_incidence,
        "figure": {"data": trace_data, "layout": layout},
        "method_note": (
            "Cumulative incidence (1 - Kaplan-Meier survival) by group; "
            "p-values from the log-rank test. Number at risk shown beneath each panel."
        ),
    }


# ── Server-side static rendering (headless / API / reports) ──────────────────
# The other chart endpoints return Plotly trace data for the browser to draw.
# This one renders a full figure spec to a static image on the server (kaleido),
# so non-browser callers can obtain a PNG/SVG/PDF directly. Styling stays in the
# caller's figure — the server never builds traces, so there is no drift from
# what the frontend shows. plotly/kaleido are optional; absent → 503.


class RenderRequest(BaseModel):
    figure: Dict[str, Any]  # Plotly figure: {"data": [...], "layout": {...}}
    format: str = "png"  # png | svg | jpeg | pdf | webp
    width: Optional[int] = None  # px; None → figure's own layout size
    height: Optional[int] = None
    scale: float = 2.0  # device-pixel multiplier (print/retina)


@router.post("/render")
def render_chart(req: RenderRequest):
    """Render a Plotly figure spec to a static image and return the raw bytes."""
    try:
        image = plot_render.render_figure(
            req.figure,
            fmt=req.format,
            width=req.width,
            height=req.height,
            scale=req.scale,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except plot_render.RenderUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    media_type = plot_render.MIME_TYPES.get(
        req.format.lower(), "application/octet-stream"
    )
    return Response(content=image, media_type=media_type)
