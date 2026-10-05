"""Reliability analysis: Cronbach's alpha with item diagnostics."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from services import store
from services.stat_utils import sanitize_nonfinite

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _alpha(values: pd.DataFrame) -> float | None:
    total_var = values.sum(axis=1).var(ddof=1)
    if total_var <= 0 or not np.isfinite(total_var):
        return None
    k = values.shape[1]
    return float((k / (k - 1)) * (1 - values.var(ddof=1).sum() / total_var))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CRONBACH'S ALPHA — Full reliability report
# ═══════════════════════════════════════════════════════════════════════════════

class CronbachRequest(BaseModel):
    session_id: str
    items: List[str]


@router.post("/cronbach")
def cronbach(req: CronbachRequest):
    df = _get_df(req.session_id)

    missing = [c for c in req.items if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    if len(req.items) < 2:
        raise HTTPException(400, "Need at least 2 items for reliability analysis.")
    if len(set(req.items)) != len(req.items):
        raise HTTPException(422, "Select distinct scale items.")

    # Coerce to numeric and drop incomplete cases
    df_items = df[req.items].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df_items)
    if n < 3:
        raise HTTPException(400, "Need at least 3 complete cases.")

    k = len(req.items)

    # ── Cronbach's alpha ─────────────────────────────────────────────────
    alpha = _alpha(df_items)
    if alpha is None:
        raise HTTPException(400, "Scale total has zero variance; reliability is undefined.")
    item_sds = df_items.std(ddof=1)
    standardized_alpha = _alpha((df_items - df_items.mean()) / item_sds) if (item_sds > 0).all() else None

    # Deterministic odd/even split in selected item order.
    half1_items, half2_items = req.items[::2], req.items[1::2]
    half1 = df_items[half1_items].sum(axis=1)
    half2 = df_items[half2_items].sum(axis=1)
    half_r = float(half1.corr(half2)) if half1.std(ddof=1) > 0 and half2.std(ddof=1) > 0 else None
    spearman_brown = (float(2 * half_r / (1 + half_r))
                      if half_r is not None and half_r > -1 else None)
    half_total_var = (half1 + half2).var(ddof=1)
    guttman_split_half = (float(2 * (1 - (half1.var(ddof=1) + half2.var(ddof=1)) / half_total_var))
                          if half_total_var > 0 else None)

    is_binary = bool(df_items.isin([0, 1]).all().all())
    kr20 = alpha if is_binary else None  # KR-20 equals raw alpha for 0/1 items.
    total_scores = df_items.sum(axis=1)
    total_var_population = float(total_scores.var(ddof=0))
    kr21 = (float(k / (k - 1) * (1 - total_scores.mean() * (k - total_scores.mean()) /
                              (k * total_var_population)))
            if is_binary and total_var_population > 0 else None)

    # ── McDonald's Omega (ω) ─────────────────────────────────────────────
    try:
        from sklearn.decomposition import FactorAnalysis
        # Standardize the data so we get standardized loadings
        df_scaled = (df_items - df_items.mean()) / df_items.std(ddof=1)
        # Drop columns with zero variance if any
        df_scaled = df_scaled.loc[:, df_scaled.var(ddof=1) > 0]
        if df_scaled.shape[1] >= 2:
            fa = FactorAnalysis(n_components=1, random_state=42, max_iter=500)
            fa.fit(df_scaled.values)
            loadings = np.clip(fa.components_[0], -0.999, 0.999)
            num = np.sum(loadings)**2
            den = num + np.sum(1.0 - loadings**2)
            omega = float(num / den) if den > 0 else None
        else:
            omega = None
    except Exception:
        omega = None

    # ── Item-total correlations ──────────────────────────────────────────
    total = df_items.sum(axis=1)
    item_total_r = {col: float(df_items[col].corr(total - df_items[col])) for col in req.items}

    # ── Alpha-if-item-deleted ────────────────────────────────────────────
    alpha_if_deleted = {}
    for col in req.items:
        remaining = [c for c in req.items if c != col]
        sub = df_items[remaining]
        k2 = len(remaining)
        alpha_if_deleted[col] = _alpha(sub) if k2 > 1 else None

    # ── Item statistics ──────────────────────────────────────────────────
    item_stats = []
    for col in req.items:
        item_stats.append({
            "item": col,
            "mean": round(float(df_items[col].mean()), 4),
            "sd": round(float(df_items[col].std(ddof=1)), 4),
            "item_total_r": round(item_total_r[col], 4),
            "alpha_if_deleted": round(alpha_if_deleted[col], 4) if alpha_if_deleted[col] is not None else None,
        })

    # ── Scale summary ────────────────────────────────────────────────────
    scale_total = df_items.sum(axis=1)
    scale_summary = {
        "mean": round(float(scale_total.mean()), 4),
        "sd": round(float(scale_total.std(ddof=1)), 4),
        "min": round(float(scale_total.min()), 4),
        "max": round(float(scale_total.max()), 4),
        "skewness": round(float(scale_total.skew()), 4),
    }

    # ── Interpretation ───────────────────────────────────────────────────
    if alpha > 0.9:
        interpretation = "Excellent"
    elif alpha > 0.8:
        interpretation = "Good"
    elif alpha > 0.7:
        interpretation = "Acceptable"
    elif alpha > 0.6:
        interpretation = "Questionable"
    elif alpha > 0.5:
        interpretation = "Poor"
    else:
        interpretation = "Unacceptable"

    # ── Result text ──────────────────────────────────────────────────────
    finite_item_correlations = [value for value in item_total_r.values() if np.isfinite(value)]
    item_correlation_text = (
        f"Item-total correlations ranged from {min(finite_item_correlations):.3f} to {max(finite_item_correlations):.3f}."
        if finite_item_correlations else "Item-total correlations could not be estimated."
    )
    result_text = (
        f"A reliability analysis was conducted on a {k}-item scale (n = {n}). "
        f"Cronbach's alpha was {alpha:.3f}, indicating {interpretation.lower()} internal consistency. "
        f"The scale mean was {scale_summary['mean']:.2f} (SD = {scale_summary['sd']:.2f}). "
        + item_correlation_text
    )

    # ── Export rows ──────────────────────────────────────────────────────
    export_rows = [
        ["Item", "Mean", "SD", "Item-Total r", "Alpha if Deleted"],
    ]
    for s in item_stats:
        export_rows.append([
            s["item"], s["mean"], s["sd"], s["item_total_r"],
            s["alpha_if_deleted"] if s["alpha_if_deleted"] is not None else "N/A",
        ])
    export_rows.append([
        "Scale Total", scale_summary["mean"], scale_summary["sd"], "", round(alpha, 4),
    ])

    # ── R code ───────────────────────────────────────────────────────────
    items_str = ", ".join(f'"{it}"' for it in req.items)
    r_code = (f'library(psych)\nitems <- na.omit(data[, c({items_str})])\n'
              'a <- alpha(items, check.keys = FALSE)\na$total$raw_alpha\na$total$std.alpha\n'
              '# Independent comparison; psych::omega differs from sklearn FactorAnalysis.\n'
              'omega(items, nfactors = 1, fm = "ml", plot = FALSE)')

    return sanitize_nonfinite({
        "test": "Cronbach's Alpha Reliability Analysis",
        "alpha": round(alpha, 4),
        "standardized_alpha": round(standardized_alpha, 4) if standardized_alpha is not None else None,
        "split_half": {
            "method": "odd_even", "half1_items": half1_items, "half2_items": half2_items,
            "r": round(half_r, 4) if half_r is not None else None,
            "spearman_brown": round(spearman_brown, 4) if spearman_brown is not None else None,
            "guttman": round(guttman_split_half, 4) if guttman_split_half is not None else None,
            "note": "Unequal half lengths; Spearman-Brown is approximate." if k % 2 else None,
        },
        "kr20": round(kr20, 4) if kr20 is not None else None,
        "kr21": round(kr21, 4) if kr21 is not None else None,
        "kr_note": "KR-21 assumes approximately equal item difficulty." if is_binary else "KR coefficients require 0/1 items.",
        "omega": round(omega, 4) if omega is not None else None,
        "n": n,
        "k": k,
        "significant": alpha > 0.7,  # conventional threshold for acceptable
        "effect_sizes": [
            {"name": "Cronbach's alpha", "value": round(alpha, 4), "magnitude": interpretation},
            *( [{"name": "McDonald's omega", "value": round(omega, 4), "magnitude": interpretation}] if omega is not None else [] )
        ],
        "assumptions": [],
        "item_stats": item_stats,
        "scale_summary": scale_summary,
        "interpretation": interpretation,
        "result_text": result_text + (f" McDonald's omega was {omega:.3f}." if omega is not None else ""),
        "export_rows": export_rows,
        "r_code": r_code,
        "omega_method": "One-factor maximum-likelihood model via sklearn FactorAnalysis on standardized items.",
        "omega_note": "psych::omega uses a different factor model; its value is not an exact replication of this estimate.",
    })
