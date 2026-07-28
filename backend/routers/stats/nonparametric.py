from __future__ import annotations

import asyncio
from typing import Optional, List, Any
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store
from services.category_health import clean_two_level
from services.impute import apply_imputation
from services.text_generators import (
    methods_mannwhitney,
    methods_kruskal,
    results_mannwhitney,
    results_kruskal,
    r_mannwhitney,
    r_kruskal,
)
from services.stat_utils import (
    rank_biserial_r,
    group_summary,
    epsilon_squared,
    dunn_test,
    sorted_groups,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
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


def _two_level_work(
    df: pd.DataFrame, value_col: str, group_col: str
) -> tuple[pd.DataFrame, list]:
    work = df[[value_col, group_col]].copy()
    cleaned = clean_two_level(work[group_col])
    work[group_col] = cleaned.series
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    return work.dropna(), cleaned.warnings


# ── 1. Mann-Whitney U ──────────────────────────────────────────────────────────


class MannWhitneyRequest(BaseModel):
    session_id: str
    column: str
    group_column: str


@router.post("/mannwhitney")
def mannwhitney(req: MannWhitneyRequest):
    df = _get_df(req.session_id)
    work, warnings = _two_level_work(df, req.column, req.group_column)
    groups = sorted_groups(work[req.group_column])
    if len(groups) != 2:
        raise HTTPException(
            status_code=400, detail="Group column must have exactly 2 groups"
        )
    g1 = work[work[req.group_column] == groups[0]][req.column].values.astype(float)
    g2 = work[work[req.group_column] == groups[1]][req.column].values.astype(float)
    stat, p = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    sig = bool(p < 0.05)
    es = rank_biserial_r(float(stat), len(g1), len(g2))
    p_str = "<0.001" if p < 0.001 else f"{p:.4f}"
    ret = {
        "test": "Mann-Whitney U test",
        "group1": str(groups[0]),
        "n1": int(len(g1)),
        "median1": float(np.median(g1)),
        "iqr1": float(np.percentile(g1, 75) - np.percentile(g1, 25)),
        "group2": str(groups[1]),
        "n2": int(len(g2)),
        "median2": float(np.median(g2)),
        "iqr2": float(np.percentile(g2, 75) - np.percentile(g2, 25)),
        "U": float(stat),
        "p": float(p),
        "significant": sig,
        "effect_sizes": [es],
        "summary": {
            str(groups[0]): group_summary(g1, str(groups[0])),
            str(groups[1]): group_summary(g2, str(groups[1])),
        },
        "interpretation": f"{'Significant' if sig else 'No significant'} difference (U = {stat:.1f}, p = {p_str}, r = {es['value']:.3f} [{es['magnitude']}])",
        "methods_text": methods_mannwhitney(req.column, req.group_column),
        "r_code": r_mannwhitney(req.column, req.group_column),
    }
    if warnings:
        ret["warnings"] = warnings
    ret["result_text"] = results_mannwhitney(ret)
    return _sanitize(ret)


# ── 2. Kruskal-Wallis ──────────────────────────────────────────────────────────


class KruskalRequest(BaseModel):
    session_id: str
    column: str
    group_column: str
    posthoc_correction: Optional[str] = "holm"


@router.post("/kruskal")
def kruskal(req: KruskalRequest):
    df = _get_df(req.session_id)
    grp_dict = {
        str(name): g[req.column].dropna().astype(float).values
        for name, g in df.groupby(req.group_column)
    }
    group_data = list(grp_dict.values())
    if len(group_data) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 groups")
    stat, p = scipy_stats.kruskal(*group_data)
    sig = bool(p < 0.05)
    n_total = sum(len(g) for g in group_data)
    es = epsilon_squared(float(stat), n_total)
    p_str = "<0.001" if p < 0.001 else f"{p:.4f}"

    pc = (req.posthoc_correction or "holm").lower()
    if pc not in {"holm", "bonferroni", "fdr", "none"}:
        raise HTTPException(
            status_code=422,
            detail=f"posthoc_correction must be holm | bonferroni | fdr | none, got '{req.posthoc_correction}'",
        )
    posthoc = dunn_test(grp_dict, correction=pc) if sig and len(grp_dict) > 2 else []

    group_stats = (
        df.groupby(req.group_column)[req.column]
        .agg(
            n="count",
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    ret = {
        "test": "Kruskal-Wallis test",
        "H": float(stat),
        "p": float(p),
        "significant": sig,
        "effect_sizes": [es],
        "posthoc": posthoc,
        "posthoc_method": f"Dunn's test ({pc.title() if pc != 'fdr' else 'FDR'} correction)"
        if posthoc
        else None,
        "groups": [
            {
                k: (float(v) if hasattr(v, "__float__") else str(v))
                for k, v in row.items()
            }
            for row in group_stats.to_dict(orient="records")
        ],
        "interpretation": f"{'Significant' if sig else 'No significant'} difference across groups (H = {stat:.2f}, p = {p_str}, ε² = {es['value']:.3f} [{es['magnitude']}])",
        "methods_text": methods_kruskal(req.column, req.group_column),
        "r_code": r_kruskal(req.column, req.group_column),
    }
    ret["result_text"] = results_kruskal(ret)
    return _sanitize(ret)


# Ordinal vocabularies seen in clinical grading scales, lowest first. Matched
# case-insensitively and only when every level of the column is covered.
_ORDINAL_WORDS = [
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
]


def _ordinal_rank_map(levels: list) -> Optional[list]:
    """Rank each level by a known ordinal vocabulary, or None if unrecognised."""
    lowered = [str(lv).strip().casefold() for lv in levels]
    for vocab in _ORDINAL_WORDS:
        if all(w in vocab for w in lowered) and len(set(lowered)) == len(lowered):
            return [vocab.index(w) for w in lowered]
    return None


# ── 3. Jonckheere-Terpstra ─────────────────────────────────────────────────────


class JonckheereRequest(BaseModel):
    session_id: str
    column: str
    group_column: str
    scores: Optional[List[float]] = None
    alpha: float = 0.05


@router.post("/jonckheere_terpstra")
def jonckheere_terpstra(req: JonckheereRequest):
    df = _get_df(req.session_id)
    for c in (req.column, req.group_column):
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")
    sub = df[[req.column, req.group_column]].dropna()
    if len(sub) < 5:
        raise HTTPException(422, "Need at least 5 non-null rows.")

    raw_levels = list(pd.unique(sub[req.group_column]))
    numeric_levels = all(
        isinstance(x, (int, float, np.integer, np.floating))
        or (
            isinstance(x, str)
            and x.replace(".", "", 1).replace("-", "", 1).isdigit()
        )
        for x in raw_levels
    )
    levels = sorted(
        raw_levels,
        key=lambda x: (
            (0, float(x))
            if isinstance(x, (int, float, np.integer, np.floating))
            or (
                isinstance(x, str)
                and x.replace(".", "", 1).replace("-", "", 1).isdigit()
            )
            else (1, str(x))
        ),
    )
    order_warnings: list = []
    order_source = "numeric value" if numeric_levels else "alphabetical (assumed)"
    if not numeric_levels and req.scores is None:
        # This test measures a trend ACROSS the ordering, so the ordering is
        # the hypothesis. Labels like Low / Medium / High sort to High, Low,
        # Medium, which reverses the trend and changes the p-value — and none
        # of that was visible in the response. Recognise the usual clinical
        # ordinal words, and say out loud which order was used either way.
        ranked = _ordinal_rank_map(raw_levels)
        if ranked is not None:
            levels = [lev for _, lev in sorted(zip(ranked, raw_levels))]
            order_source = "recognised ordinal labels"
            order_warnings.append(
                f"'{req.group_column}' was ordered as "
                + " < ".join(str(lv) for lv in levels)
                + " from its labels. Send `scores` to set the order explicitly."
            )
        else:
            order_warnings.append(
                f"'{req.group_column}' has non-numeric levels, so they were "
                "placed in alphabetical order: "
                + " < ".join(str(lv) for lv in levels)
                + ". Jonckheere-Terpstra tests a trend ACROSS that order — if "
                "this is not the clinical order, send `scores` to set it, "
                "because the trend direction and the p-value depend on it."
            )
    if req.scores is not None:
        if len(req.scores) != len(levels):
            raise HTTPException(
                422,
                f"Custom scores must match the number of levels ({len(levels)}); got {len(req.scores)}.",
            )
        levels = [lev for _, lev in sorted(zip(req.scores, levels), key=lambda t: t[0])]
    K = len(levels)
    if K < 3:
        raise HTTPException(
            422,
            f"Jonckheere-Terpstra requires ≥ 3 ordered groups; got {K}. "
            "For 2 groups use Mann-Whitney; for unordered groups use Kruskal-Wallis.",
        )

    groups: list[np.ndarray] = []
    for lev in levels:
        vals = sub.loc[sub[req.group_column] == lev, req.column].astype(float).values
        if len(vals) == 0:
            raise HTTPException(422, f"Group '{lev}' has zero observations.")
        groups.append(vals)
    n_k = np.array([len(g) for g in groups], dtype=float)
    N = float(n_k.sum())

    J = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            xi = groups[i][:, None]
            xj = groups[j][None, :]
            J += float(np.sum(xj > xi) + 0.5 * np.sum(xj == xi))

    sum_n2 = float(np.sum(n_k**2))
    sum_n2_2n_p3 = float(np.sum(n_k**2 * (2 * n_k + 3)))
    E_J = (N**2 - sum_n2) / 4.0
    Var_J = (N**2 * (2 * N + 3) - sum_n2_2n_p3) / 72.0
    if Var_J <= 0:
        raise HTTPException(
            422, "Jonckheere-Terpstra variance is zero — group sizes degenerate."
        )
    z = (J - E_J) / np.sqrt(Var_J)
    p_two = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z)))
    sig = bool(p_two < req.alpha)
    p_str = "<0.001" if p_two < 0.001 else f"{p_two:.4f}"
    direction = "increasing" if z > 0 else "decreasing" if z < 0 else "flat"

    level_rows = []
    for lev, g in zip(levels, groups):
        level_rows.append(
            {
                "level": str(lev),
                "n": int(len(g)),
                "median": round(float(np.median(g)), 4),
                "q1": round(float(np.percentile(g, 25)), 4),
                "q3": round(float(np.percentile(g, 75)), 4),
                "mean": round(float(np.mean(g)), 4),
            }
        )

    return _sanitize(
        {
            "test": "Jonckheere-Terpstra trend test",
            "J": round(J, 4),
            "E_J": round(E_J, 4),
            "Var_J": round(Var_J, 6),
            "z": round(z, 4),
            "statistic": round(z, 4),
            "p": p_two,
            "significant": sig,
            "effect_sizes": [],
            "warnings": order_warnings,
            # The order the trend was measured across, stated rather than
            # implied: it decides the direction and the p-value.
            "level_order": [str(lv) for lv in levels],
            "level_order_source": order_source,
            "assumptions": [
                "Ordered (ordinal) exposure with \u22653 levels",
                "Continuous (or at least ordinal) outcome",
                "Independence between observations",
            ],
            "summary": {
                "n": int(N),
                "n_levels": K,
                "direction": direction,
                "levels": level_rows,
            },
            "interpretation": (
                f"{'Significant' if sig else 'No significant'} monotone trend in "
                f"{req.column} across {K} ordered levels of {req.group_column} "
                f"(J = {J:.2f}, Z = {z:.3f}, p = {p_str}; direction: {direction}) "
                f"measured across the order "
                + " < ".join(str(lv) for lv in levels) + "."
            ),
            "result_text": (
                f"The Jonckheere-Terpstra non-parametric trend test was used to "
                f"assess whether {req.column} changed monotonically across {K} "
                f"ordered levels of {req.group_column} (n = {int(N)}). The trend "
                f"was {'statistically significant' if sig else 'not statistically significant'} "
                f"(J = {J:.2f}, standardised Z = {z:.3f}, two-sided p = {p_str}), "
                f"with a {direction} trend in medians."
            ),
            "export_rows": [
                ["Statistic", "Value"],
                ["J statistic", round(J, 4)],
                ["Expected E(J)", round(E_J, 4)],
                ["Standardised Z", round(z, 4)],
                ["Two-sided p-value", round(p_two, 6)],
            ],
        }
    )


# ── ROC Helpers ────────────────────────────────────────────────────────────────


def _roc_metrics_at_cutoff(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("inf")
    return {
        "cutoff": round(float(threshold), 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "accuracy": round(acc, 4),
        "lr_pos": round(lr_pos, 4) if not np.isinf(lr_pos) else None,
        "lr_neg": round(lr_neg, 4) if not np.isinf(lr_neg) else None,
        "youden_j": round(sens + spec - 1, 4),
    }


def _delong_placement_values(y: np.ndarray, scores: np.ndarray):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n1, n0 = len(pos_idx), len(neg_idx)
    s_pos = scores[pos_idx]
    s_neg = scores[neg_idx]
    V_pos = (
        np.sum(s_neg[:, None] < s_pos[None, :], axis=0).astype(float)
        + 0.5 * np.sum(s_neg[:, None] == s_pos[None, :], axis=0).astype(float)
    ) / n0
    V_neg = (
        np.sum(s_pos[:, None] > s_neg[None, :], axis=0).astype(float)
        + 0.5 * np.sum(s_pos[:, None] == s_neg[None, :], axis=0).astype(float)
    ) / n1
    return V_pos, V_neg


def _delong_compare(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    V_pos1, V_neg1 = _delong_placement_values(y, s1)
    V_pos2, V_neg2 = _delong_placement_values(y, s2)
    n_pos, n_neg = len(V_pos1), len(V_neg1)
    auc1 = float(np.mean(V_pos1))
    auc2 = float(np.mean(V_pos2))

    s11 = np.var(V_pos1, ddof=1) / n_pos + np.var(V_neg1, ddof=1) / n_neg
    s22 = np.var(V_pos2, ddof=1) / n_pos + np.var(V_neg2, ddof=1) / n_neg
    s12 = (
        np.cov(V_pos1, V_pos2, ddof=1)[0, 1] / n_pos
        + np.cov(V_neg1, V_neg2, ddof=1)[0, 1] / n_neg
    )

    var_diff = max(s11 + s22 - 2 * s12, 1e-12)
    diff = auc1 - auc2
    se_diff = np.sqrt(var_diff)
    z = diff / se_diff
    p = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
    z95 = 1.95996
    ci_diff_low = float(diff - z95 * se_diff)
    ci_diff_high = float(diff + z95 * se_diff)

    se1 = np.sqrt(max(s11, 1e-12))
    se2 = np.sqrt(max(s22, 1e-12))
    ci1_low = max(0.0, float(auc1 - z95 * se1))
    ci1_high = min(1.0, float(auc1 + z95 * se1))
    ci2_low = max(0.0, float(auc2 - z95 * se2))
    ci2_high = min(1.0, float(auc2 + z95 * se2))

    return {
        "auc_1": round(auc1, 4),
        "auc_2": round(auc2, 4),
        "ci_1_low": round(ci1_low, 4),
        "ci_1_high": round(ci1_high, 4),
        "ci_2_low": round(ci2_low, 4),
        "ci_2_high": round(ci2_high, 4),
        "difference": round(diff, 4),
        "ci_diff_low": round(ci_diff_low, 4),
        "ci_diff_high": round(ci_diff_high, 4),
        "se_diff": round(float(se_diff), 6),
        "z": round(float(z), 4),
        "p": round(p, 6),
        "significant": bool(p < 0.05),
    }


def _validate_roc_inputs(
    df: pd.DataFrame, score_col: str, outcome_col: str, imputation: str = "listwise"
):
    for col in [score_col, outcome_col]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    df = apply_imputation(df, [score_col, outcome_col], imputation)
    if len(df) < 10:
        raise HTTPException(
            status_code=400,
            detail="Not enough data (need ≥ 10 rows after removing missing)",
        )
    try:
        y = df[outcome_col].astype(float).astype(int)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail=f"Outcome '{outcome_col}' could not be converted to 0/1",
        )
    uniq = sorted(y.unique().tolist())
    if len(uniq) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Outcome must have exactly 2 unique values. Found: {uniq[:6]}",
        )
    if set(uniq) != {0, 1}:
        raise HTTPException(
            status_code=400, detail=f"Outcome values must be 0 and 1. Found: {uniq}"
        )
    try:
        scores = df[score_col].astype(float)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400, detail=f"Score column '{score_col}' must be numeric"
        )
    if scores.nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Score column '{score_col}' has no variation (constant)",
        )
    return scores.values, y.values, df


# ── 4. ROC Curve ───────────────────────────────────────────────────────────────


class ROCRequest(BaseModel):
    session_id: str
    score_column: str
    outcome_column: str
    direction: Optional[str] = "auto"
    manual_cutoff: Optional[float] = None
    imputation: Optional[str] = "listwise"
    stratify_by: Optional[str] = None
    stratify_values: Optional[List[Any]] = None


@router.post("/roc")
async def roc_analysis(req: ROCRequest):
    """ROC analysis for a numeric score predicting a binary 0/1 outcome. Use ``direction='lower'`` when higher scores indicate lower event risk; the backend reports ``1 - AUC`` with swapped confidence-interval bounds. Use ``direction='higher'`` when higher scores indicate higher event risk, and ``direction='auto'`` to flip automatically when the naive AUC is < 0.5."""
    df_full = _get_df(req.session_id)
    return await asyncio.to_thread(_run_roc, req, df_full)


def _run_roc(req: ROCRequest, df_full: pd.DataFrame):
    from sklearn.metrics import roc_curve, roc_auc_score

    scores_arr, y_arr, df = _validate_roc_inputs(
        df_full,
        req.score_column,
        req.outcome_column,
        imputation=req.imputation or "listwise",
    )

    if req.stratify_by:
        if req.stratify_by not in df.columns:
            raise HTTPException(
                400, f"Stratification column '{req.stratify_by}' not found."
            )

        strata_results = {}
        strata_values = req.stratify_values or sorted_groups(df[req.stratify_by])

        for val in strata_values:
            mask = df[req.stratify_by] == val
            if mask.sum() < 20:
                continue

            s_scores = scores_arr[mask.values]
            s_y = y_arr[mask.values]

            try:
                fpr_s, tpr_s, th_s = roc_curve(s_y, s_scores)
                auc_s = float(roc_auc_score(s_y, s_scores))

                j_scores = tpr_s + (1 - fpr_s) - 1
                best_idx = int(np.argmax(j_scores))
                best_cut = float(th_s[best_idx])

                strata_results[str(val)] = {
                    "n": int(mask.sum()),
                    "auc": round(auc_s, 4),
                    "optimal_cutoff": round(best_cut, 4),
                    "sensitivity_at_opt": round(float(tpr_s[best_idx]), 4),
                    "specificity_at_opt": round(float(1 - fpr_s[best_idx]), 4),
                }
            except Exception:
                continue

        if strata_results:
            return {
                "test": "ROC Analysis (Stratified)",
                "stratified_by": req.stratify_by,
                "strata": strata_results,
                "note": "Separate ROC analysis performed within each stratum.",
            }

    try:
        fpr, tpr, thresholds = roc_curve(y_arr, scores_arr)
        auc = float(roc_auc_score(y_arr, scores_arr))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ROC computation failed: {exc}")

    direction_req = (req.direction or "auto").lower()
    if direction_req not in {"auto", "higher", "lower"}:
        raise HTTPException(
            status_code=422,
            detail=f"direction must be 'auto' | 'higher' | 'lower', got '{req.direction}'",
        )
    flipped = False
    if direction_req == "lower" or (direction_req == "auto" and auc < 0.5):
        scores_arr = -scores_arr
        fpr, tpr, thresholds = roc_curve(y_arr, scores_arr)
        auc = float(roc_auc_score(y_arr, scores_arr))
        flipped = True
    direction_used = "lower" if flipped else "higher"

    try:
        V_pos, V_neg = _delong_placement_values(
            y_arr.astype(int), scores_arr.astype(float)
        )
        n_pos_d, n_neg_d = len(V_pos), len(V_neg)
        var_auc = float(
            np.var(V_pos, ddof=1) / max(n_pos_d, 1)
            + np.var(V_neg, ddof=1) / max(n_neg_d, 1)
        )
        var_auc = max(var_auc, 1e-12)
        se_auc = float(np.sqrt(var_auc))
        z95 = 1.95996
        ci_low = max(0.0, auc - z95 * se_auc)
        ci_high = min(1.0, auc + z95 * se_auc)
        z_auc = (auc - 0.5) / se_auc
        p_auc = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(z_auc))))
    except Exception:
        se_auc = None
        ci_low = None
        ci_high = None
        z_auc = None
        p_auc = None

    def _to_user(t: float) -> float:
        return -t if flipped else t

    def _from_user(t: float) -> float:
        return -t if flipped else t

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    optimal = _roc_metrics_at_cutoff(scores_arr, y_arr, best_thresh)
    if flipped:
        optimal["cutoff"] = round(_to_user(best_thresh), 6)

    manual = None
    if req.manual_cutoff is not None:
        thr_internal = _from_user(float(req.manual_cutoff))
        manual = _roc_metrics_at_cutoff(scores_arr, y_arr, thr_internal)
        if flipped:
            manual["cutoff"] = round(float(req.manual_cutoff), 6)

    n_pts = len(fpr)
    step = max(1, n_pts // 300)
    indices = list(range(0, n_pts, step))
    if (n_pts - 1) not in indices:
        indices.append(n_pts - 1)

    curve = []
    for i in indices:
        thr = float(thresholds[i])
        m = _roc_metrics_at_cutoff(scores_arr, y_arr, thr)
        curve.append(
            {
                "fpr": round(float(fpr[i]), 6),
                "tpr": round(float(tpr[i]), 6),
                "threshold": round(_to_user(thr), 6),
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "ppv": m["ppv"],
                "npv": m["npv"],
                "lr_pos": m["lr_pos"],
                "lr_neg": m["lr_neg"],
                "youden_j": m["youden_j"],
            }
        )

    return _sanitize(
        {
            "test": "ROC Analysis",
            "n": len(df),
            "n_positive": int(y_arr.sum()),
            "n_negative": int((y_arr == 0).sum()),
            "auc": round(auc, 4),
            "auc_se": round(se_auc, 6) if se_auc is not None else None,
            "ci_lower": round(ci_low, 4) if ci_low is not None else None,
            "ci_upper": round(ci_high, 4) if ci_high is not None else None,
            "auc_z": round(z_auc, 4) if z_auc is not None else None,
            "auc_p": round(p_auc, 6) if p_auc is not None else None,
            "auc_test": "H0: AUC = 0.5 (DeLong two-sided z-test)",
            "direction_requested": direction_req,
            "direction_used": direction_used,
            "direction_flipped": flipped,
            "optimal_cutoff": optimal["cutoff"],
            "sensitivity": optimal["sensitivity"],
            "specificity": optimal["specificity"],
            "tp": optimal["tp"],
            "tn": optimal["tn"],
            "fp": optimal["fp"],
            "fn": optimal["fn"],
            "optimal": optimal,
            "manual": manual,
            "curve": curve,
            "interpretation": (
                f"AUC = {auc:.3f} — "
                f"{'Excellent' if auc >= 0.9 else 'Good' if auc >= 0.8 else 'Fair' if auc >= 0.7 else 'Poor'} "
                "discriminative ability"
            ),
            "result_text": (
                f"ROC analysis was performed for {req.score_column} predicting {req.outcome_column} (n = {len(df)}). "
                f"The area under the curve was {auc:.2f}"
                + (
                    f" (95% CI {ci_low:.2f}–{ci_high:.2f}, p = "
                    f"{'<0.001' if (p_auc is not None and p_auc < 0.001) else f'{p_auc:.3f}' if p_auc is not None else 'n/a'})"
                    if ci_low is not None and ci_high is not None
                    else ""
                )
                + ", indicating "
                f"{'excellent' if auc >= 0.9 else 'good' if auc >= 0.8 else 'fair' if auc >= 0.7 else 'poor'} discrimination "
                + (
                    "(lower values predict the event — score sign auto-flipped from the request default). "
                    if flipped
                    else "(higher values predict the event). "
                )
                + f"At the optimal cutoff ({optimal['cutoff']:.2f}, Youden's J), sensitivity was {optimal['sensitivity'] * 100:.1f}% "
                f"and specificity was {optimal['specificity'] * 100:.1f}%."
            ),
        }
    )


# ── 5. ROC Compare ─────────────────────────────────────────────────────────────


class ROCCompareRequest(BaseModel):
    session_id: str
    score_column_1: str
    score_column_2: str
    outcome_column: str
    direction_1: Optional[str] = "auto"
    direction_2: Optional[str] = "auto"


@router.post("/roc_compare")
async def roc_compare(req: ROCCompareRequest):
    df_full = _get_df(req.session_id)
    return await asyncio.to_thread(_run_roc_compare, req, df_full)


def _run_roc_compare(req: ROCCompareRequest, df_full: pd.DataFrame):
    from sklearn.metrics import roc_curve, roc_auc_score

    s1_arr, y_arr, _ = _validate_roc_inputs(
        df_full, req.score_column_1, req.outcome_column
    )
    s2_arr, y_arr2, _ = _validate_roc_inputs(
        df_full, req.score_column_2, req.outcome_column
    )

    if not np.array_equal(y_arr, y_arr2):
        df_clean = df_full.dropna(
            subset=[req.score_column_1, req.score_column_2, req.outcome_column]
        )
        if len(df_clean) < 10:
            raise HTTPException(
                status_code=400,
                detail="Not enough complete rows for comparison (need ≥ 10)",
            )
        y_arr = df_clean[req.outcome_column].astype(float).astype(int).values
        s1_arr = df_clean[req.score_column_1].astype(float).values
        s2_arr = df_clean[req.score_column_2].astype(float).values

    def _resolve_direction(scores: np.ndarray, y: np.ndarray, req_dir: str):
        d = (req_dir or "auto").lower()
        if d not in {"auto", "higher", "lower"}:
            raise HTTPException(
                status_code=422,
                detail=f"direction must be 'auto' | 'higher' | 'lower', got '{req_dir}'",
            )
        naive_auc = float(roc_auc_score(y, scores))
        flipped = (d == "lower") or (d == "auto" and naive_auc < 0.5)
        return (-scores if flipped else scores), flipped, d

    s1_arr, flipped_1, dir_req_1 = _resolve_direction(
        s1_arr, y_arr, req.direction_1 or "auto"
    )
    s2_arr, flipped_2, dir_req_2 = _resolve_direction(
        s2_arr, y_arr, req.direction_2 or "auto"
    )

    result = _delong_compare(y_arr, s1_arr, s2_arr)
    result["direction_1_requested"] = dir_req_1
    result["direction_2_requested"] = dir_req_2
    result["direction_1_used"] = "lower" if flipped_1 else "higher"
    result["direction_2_used"] = "lower" if flipped_2 else "higher"
    result["direction_1_flipped"] = bool(flipped_1)
    result["direction_2_flipped"] = bool(flipped_2)
    result["score_1"] = req.score_column_1
    result["score_2"] = req.score_column_2
    result["n"] = int(len(y_arr))

    def _roc_curve_pts(scores, y):
        fpr, tpr, _ = roc_curve(y, scores)
        n_pts = len(fpr)
        step = max(1, n_pts // 300)
        idx = list(range(0, n_pts, step))
        if (n_pts - 1) not in idx:
            idx.append(n_pts - 1)
        return [
            {"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6)}
            for i in idx
        ]

    result["curve_1"] = _roc_curve_pts(s1_arr, y_arr)
    result["curve_2"] = _roc_curve_pts(s2_arr, y_arr)

    auc1, auc2 = result["auc_1"], result["auc_2"]
    diff = result["difference"]
    p = result["p"]
    p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
    ci_lo = result["ci_diff_low"]
    ci_hi = result["ci_diff_high"]
    winner = req.score_column_1 if diff > 0 else req.score_column_2
    loser = req.score_column_2 if diff > 0 else req.score_column_1
    higher_auc = max(auc1, auc2)
    lower_auc = min(auc1, auc2)

    ci_report_lo = min(ci_lo, ci_hi)
    ci_report_hi = max(ci_lo, ci_hi)

    if result["significant"]:
        result["interpretation"] = (
            f"{winner} significantly improved discrimination over {loser} "
            f"(AUC {higher_auc:.3f} vs. {lower_auc:.3f}; "
            f"ΔAUC = {abs(diff):.3f}, 95% CI: {ci_report_lo:.3f}–{ci_report_hi:.3f}, "
            f"DeLong p = {p_str})."
        )
    else:
        result["interpretation"] = (
            f"No significant difference between {req.score_column_1} and {req.score_column_2} "
            f"(AUC {auc1:.3f} vs. {auc2:.3f}; "
            f"ΔAUC = {abs(diff):.3f}, 95% CI: {ci_report_lo:.3f}–{ci_report_hi:.3f}, "
            f"DeLong p = {p_str})."
        )

    result["result_text"] = result["interpretation"]
    return _sanitize(result)


# ── 6. ROC Multi Compare ───────────────────────────────────────────────────────


class ROCMultiCompareRequest(BaseModel):
    session_id: str
    score_columns: List[str]
    outcome_column: str
    directions: Optional[List[str]] = None
    p_adjust: Optional[str] = "holm"


@router.post("/roc_multi_compare")
async def roc_multi_compare(req: ROCMultiCompareRequest):
    df_full = _get_df(req.session_id)
    return await asyncio.to_thread(_run_roc_multi_compare, req, df_full)


def _run_roc_multi_compare(req: ROCMultiCompareRequest, df_full: pd.DataFrame):
    from sklearn.metrics import roc_auc_score

    if len(req.score_columns) < 2:
        raise HTTPException(
            status_code=422, detail="Need at least 2 score columns to compare."
        )
    if len(req.score_columns) != len(set(req.score_columns)):
        raise HTTPException(
            status_code=422, detail="Duplicate entries in score_columns."
        )

    for c in req.score_columns + [req.outcome_column]:
        if c not in df_full.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")

    df = df_full.dropna(subset=list(req.score_columns) + [req.outcome_column]).copy()
    if len(df) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough complete rows after dropping NaN (need ≥ 10, got {len(df)}).",
        )

    y_arr = df[req.outcome_column].astype(float).astype(int).values
    unique = set(np.unique(y_arr).tolist())
    if unique - {0, 1} or unique == {0} or unique == {1}:
        raise HTTPException(
            status_code=422,
            detail=f"Outcome must be binary 0/1 with both classes present (got {sorted(unique)}).",
        )

    dirs_in = list(req.directions or [])
    while len(dirs_in) < len(req.score_columns):
        dirs_in.append("auto")
    K = len(req.score_columns)
    scores: List[np.ndarray] = []
    scores_meta: List[dict] = []
    for col, d_in in zip(req.score_columns, dirs_in):
        d = (d_in or "auto").lower()
        if d not in {"auto", "higher", "lower"}:
            raise HTTPException(
                status_code=422,
                detail=f"direction for '{col}' must be 'auto'|'higher'|'lower', got '{d_in}'.",
            )
        raw = df[col].astype(float).values
        naive_auc = float(roc_auc_score(y_arr, raw))
        flipped = (d == "lower") or (d == "auto" and naive_auc < 0.5)
        scores.append(-raw if flipped else raw)
        scores_meta.append(
            {
                "name": col,
                "direction_requested": d,
                "direction_used": "lower" if flipped else "higher",
                "direction_flipped": bool(flipped),
            }
        )

    place: List[tuple] = [_delong_placement_values(y_arr, s) for s in scores]
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())

    z95 = 1.95996
    per_score: List[dict] = []
    aucs = np.zeros(K, dtype=float)
    ses = np.zeros(K, dtype=float)
    for i, (V_pos, V_neg) in enumerate(place):
        auc = float(np.mean(V_pos))
        var = np.var(V_pos, ddof=1) / n_pos + np.var(V_neg, ddof=1) / n_neg
        se = float(np.sqrt(max(var, 1e-12)))
        aucs[i] = auc
        ses[i] = se
        ci_lo = max(0.0, auc - z95 * se)
        ci_hi = min(1.0, auc + z95 * se)
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_arr, scores[i])
        curve_step = max(1, len(fpr) // 300)
        curve_idx = list(range(0, len(fpr), curve_step))
        if (len(fpr) - 1) not in curve_idx:
            curve_idx.append(len(fpr) - 1)
        per_score.append(
            {
                **scores_meta[i],
                "auc": round(auc, 4),
                "se": round(se, 6),
                "ci_low": round(ci_lo, 4),
                "ci_high": round(ci_hi, 4),
                "curve": [
                    {"fpr": round(float(fpr[k]), 6), "tpr": round(float(tpr[k]), 6)}
                    for k in curve_idx
                ],
            }
        )

    pairs: List[dict] = []
    raw_ps: List[float] = []
    for i in range(K):
        Vpi, Vni = place[i]
        for j in range(i + 1, K):
            Vpj, Vnj = place[j]
            cov = (
                np.cov(Vpi, Vpj, ddof=1)[0, 1] / n_pos
                + np.cov(Vni, Vnj, ddof=1)[0, 1] / n_neg
            )
            var_diff = max(ses[i] ** 2 + ses[j] ** 2 - 2 * float(cov), 1e-12)
            se_diff = float(np.sqrt(var_diff))
            diff = float(aucs[i] - aucs[j])
            z = diff / se_diff if se_diff > 0 else 0.0
            p = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
            ci_lo = float(diff - z95 * se_diff)
            ci_hi = float(diff + z95 * se_diff)
            pairs.append(
                {
                    "a": req.score_columns[i],
                    "b": req.score_columns[j],
                    "auc_a": round(float(aucs[i]), 4),
                    "auc_b": round(float(aucs[j]), 4),
                    "delta_auc": round(diff, 4),
                    "se_diff": round(se_diff, 6),
                    "ci_low": round(ci_lo, 4),
                    "ci_high": round(ci_hi, 4),
                    "z": round(float(z), 4),
                    "p_raw": round(p, 6),
                }
            )
            raw_ps.append(p)

    method = (req.p_adjust or "holm").lower()
    m = len(raw_ps)
    if m == 0 or method == "none":
        p_adj_list = list(raw_ps)
    elif method == "bonferroni":
        p_adj_list = [min(1.0, p * m) for p in raw_ps]
    elif method == "holm":
        order = sorted(range(m), key=lambda k: raw_ps[k])
        p_adj_arr = [0.0] * m
        running = 0.0
        for rank, idx in enumerate(order):
            adj = (m - rank) * raw_ps[idx]
            running = max(running, adj)
            p_adj_arr[idx] = min(1.0, running)
        p_adj_list = p_adj_arr
    else:
        raise HTTPException(
            status_code=422,
            detail=f"p_adjust must be 'holm'|'bonferroni'|'none', got '{req.p_adjust}'.",
        )

    for pair, p_adj in zip(pairs, p_adj_list):
        pair["p_adj"] = round(float(p_adj), 6)
        pair["significant"] = bool(p_adj < 0.05)

    return _sanitize(
        {
            "test": "ROC Multi-Curve DeLong",
            "n": int(len(y_arr)),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "outcome": req.outcome_column,
            "scores": per_score,
            "pairs": pairs,
            "n_pairs": m,
            "p_adjust": method,
            "method_note": (
                "Per-column AUC reported with DeLong (1988) 95% confidence interval. "
                "Pairwise ΔAUC inference uses the same DeLong covariance machinery "
                "(placement values + Mann-Whitney U variance), so every pair is tested "
                "on the same paired sample (NaN-complete-case across every score and the "
                f"outcome). Multiple-comparison adjustment: {method}."
            ),
        }
    )


# ── 7. ROC Combined ────────────────────────────────────────────────────────────


class ROCCombinedRequest(BaseModel):
    session_id: str
    predictor_columns: List[str]
    outcome_column: str
    model_name: Optional[str] = "Combined Model"


@router.post("/roc_combined")
def roc_combined(req: ROCCombinedRequest):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    df_full = _get_df(req.session_id)

    if req.outcome_column not in df_full.columns:
        raise HTTPException(
            status_code=400, detail=f"Outcome column '{req.outcome_column}' not found"
        )
    missing_cols = [c for c in req.predictor_columns if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400, detail=f"Predictor column(s) not found: {missing_cols}"
        )
    if len(req.predictor_columns) < 1:
        raise HTTPException(
            status_code=400, detail="At least one predictor column is required"
        )

    cols = req.predictor_columns + [req.outcome_column]
    df = df_full.dropna(subset=cols)
    if len(df) < 20:
        raise HTTPException(
            status_code=400,
            detail="Not enough complete rows after removing missing (need ≥ 20)",
        )

    parts = []
    for col in req.predictor_columns:
        col_s = df[col]
        if pd.api.types.is_numeric_dtype(col_s):
            parts.append(col_s.rename(col).to_frame())
        else:
            parts.append(pd.get_dummies(col_s, prefix=col, drop_first=True))
    X = pd.concat(parts, axis=1).astype(float).values

    try:
        y = df[req.outcome_column].astype(float).astype(int).values
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400, detail="Outcome could not be converted to 0/1 integers"
        )
    uniq = sorted(set(y.tolist()))
    if set(uniq) != {0, 1}:
        raise HTTPException(
            status_code=400, detail=f"Outcome must be exactly 0 and 1. Found: {uniq}"
        )

    try:
        from sklearn.model_selection import cross_val_predict

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        n_cv = min(10, max(3, len(y) // 10))
        prob = cross_val_predict(model, X_sc, y, cv=n_cv, method="predict_proba")[:, 1]
        model.fit(X_sc, y)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model fitting failed: {exc}")

    try:
        fpr, tpr, thresholds = roc_curve(y, prob)
        auc = float(roc_auc_score(y, prob))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ROC computation failed: {exc}")
    flipped = False
    if auc < 0.5:
        prob = 1.0 - prob
        fpr, tpr, thresholds = roc_curve(y, prob)
        auc = float(roc_auc_score(y, prob))
        flipped = True

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    optimal = _roc_metrics_at_cutoff(prob, y, best_thresh)

    n_pts = len(fpr)
    step = max(1, n_pts // 300)
    indices = list(range(0, n_pts, step))
    if (n_pts - 1) not in indices:
        indices.append(n_pts - 1)
    curve = [
        {"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6)}
        for i in indices
    ]
    flip_note = (
        "The combined score direction was auto-flipped because lower fitted scores predicted the event. "
        if flipped
        else ""
    )

    return _sanitize(
        {
            "test": "ROC Analysis (Combined Model)",
            "model_name": req.model_name,
            "predictors": req.predictor_columns,
            "n": int(len(df)),
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "auc": round(auc, 4),
            "direction_used": "lower" if flipped else "higher",
            "direction_flipped": bool(flipped),
            "optimal": optimal,
            "curve": curve,
            "result_text": (
                f"A combined model ({req.model_name}) using {len(req.predictor_columns)} predictors "
                f"({', '.join(req.predictor_columns)}) was evaluated (n = {len(df)}). "
                f"The AUC was {auc:.3f}, indicating "
                f"{'excellent' if auc >= 0.9 else 'good' if auc >= 0.8 else 'fair' if auc >= 0.7 else 'poor'} discrimination. "
                f"{flip_note}"
                f"At the optimal cutoff ({optimal['cutoff']:.3f}), sensitivity was {optimal['sensitivity'] * 100:.1f}% "
                f"and specificity was {optimal['specificity'] * 100:.1f}%."
            ),
        }
    )
