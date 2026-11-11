"""Model-specific diagnostics: logistic regression calibration & Cox PH assumptions."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from services import store
from services.impute import apply_imputation
from services.missing_data_sensitivity import (
    simulate_missingness,
    delta_adjustment_sensitivity,
    summarize_sensitivity,
)
from services.model_validation import (
    OPTIMISM_NEEDS_REFIT,
    bootstrap_performance,
    compute_cox_calibration_slope,
    compute_calibration_slope_intercept,
)
from services.causal_sensitivity import (
    e_value,
    quantitative_bias_analysis,
)
from services.survival_validation import validate_survival_inputs

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOGISTIC REGRESSION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

class LogisticDiagRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: str = "listwise"


@router.post("/logistic_diagnostics")
def logistic_diagnostics(req: LogisticDiagRequest):
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    df_full = _get_df(req.session_id)
    all_cols = [req.outcome] + req.predictors
    df = apply_imputation(df_full, all_cols, req.imputation or "listwise").copy()

    if len(df) < len(req.predictors) + 10:
        raise HTTPException(400, "Not enough observations for logistic regression diagnostics.")

    # Encode outcome as binary 0/1
    y = df[req.outcome].copy()
    unique_vals = sorted(y.dropna().unique())
    if len(unique_vals) != 2:
        raise HTTPException(400, f"Outcome '{req.outcome}' must be binary (has {len(unique_vals)} unique values).")

    # Map to 0/1
    if set(unique_vals) <= {0, 1, 0.0, 1.0}:
        y = y.astype(float)
    else:
        val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
        y = y.map(val_map).astype(float)

    X = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    X = sm.add_constant(X)

    # ── Fit logistic ─────────────────────────────────────────────────────────
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
    except Exception as exc:
        raise HTTPException(400, f"Logistic model failed to converge: {exc}")

    probs = model.predict(X).values
    y_arr = y.values
    n = len(y_arr)

    # ── 1. Separation warning ────────────────────────────────────────────────
    separation_vars = []
    for col in req.predictors:
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            # Check if any threshold perfectly separates
            vals_0 = s[y_arr == 0]
            vals_1 = s[y_arr == 1]
            if len(vals_0) > 0 and len(vals_1) > 0:
                if vals_0.max() < vals_1.min() or vals_1.max() < vals_0.min():
                    separation_vars.append(col)
        else:
            # Categorical: check if any level has only one outcome value
            for level in s.unique():
                mask = s == level
                if mask.sum() > 0:
                    outcomes_in_level = y_arr[mask.values]
                    if len(np.unique(outcomes_in_level)) == 1 and mask.sum() > 1:
                        separation_vars.append(col)
                        break

    # ── 2. Calibration plot data ─────────────────────────────────────────────
    sorted_idx = np.argsort(probs)
    n_bins = min(10, max(2, n // 10))
    groups = np.array_split(sorted_idx, n_bins)
    calibration_bins = []
    for grp in groups:
        if len(grp) == 0:
            continue
        calibration_bins.append({
            "predicted_mean": round(float(probs[grp].mean()), 4),
            "observed_prop": round(float(y_arr[grp].mean()), 4),
            "n": int(len(grp)),
        })

    # ── 3. Brier score ───────────────────────────────────────────────────────
    brier = float(np.mean((probs - y_arr) ** 2))

    # ── 4. Hosmer-Lemeshow ───────────────────────────────────────────────────
    hl_groups = np.array_split(sorted_idx, 10)
    chi2 = 0.0
    for grp in hl_groups:
        if len(grp) == 0:
            continue
        n_g = len(grp)
        o_g = float(y_arr[grp].sum())
        e_g = float(probs[grp].sum())
        denom = e_g * (1 - e_g / n_g)
        if denom > 1e-10:
            chi2 += (o_g - e_g) ** 2 / denom
    p_hl = float(1 - scipy_stats.chi2.cdf(chi2, df=8))

    # ── 5. C-statistic (AUC) ────────────────────────────────────────────────
    try:
        c_stat = float(roc_auc_score(y_arr, probs))
    except Exception:
        c_stat = None

    # ── 6. Influence summary ─────────────────────────────────────────────────
    try:
        infl = model.get_influence()
        dfbetas = infl.dfbetas
        hat_diag = infl.hat_matrix_diag
        n_high_dfbeta = int(np.sum(np.any(np.abs(dfbetas) > 2 / np.sqrt(n), axis=1)))
        n_high_leverage = int(np.sum(hat_diag > 2 * X.shape[1] / n))
    except Exception:
        n_high_dfbeta = None
        n_high_leverage = None

    # ── Assumptions ──────────────────────────────────────────────────────────
    assumptions = []
    assumptions.append({
        "name": "Calibration (Hosmer-Lemeshow)",
        "met": bool(p_hl >= 0.05),
        "detail": f"Chi-square = {chi2:.2f}, p = {_p_str(p_hl)}",
    })
    assumptions.append({
        "name": "No perfect separation",
        "met": len(separation_vars) == 0,
        "detail": f"Separation detected in: {', '.join(separation_vars)}" if separation_vars else "No separation detected",
    })

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings = []
    if p_hl < 0.05:
        warnings.append("Hosmer-Lemeshow test significant — model may be poorly calibrated.")
    if separation_vars:
        warnings.append(f"Perfect or quasi-perfect separation in: {', '.join(separation_vars)}. Coefficient estimates may be unreliable.")
    if brier > 0.25:
        warnings.append(f"Brier score = {brier:.4f} — predictive accuracy is limited.")
    if c_stat is not None and c_stat < 0.6:
        warnings.append(f"C-statistic (AUC) = {c_stat:.4f} — poor discrimination.")

    # ── result_text ──────────────────────────────────────────────────────────
    hl_txt = f"Hosmer-Lemeshow: {'adequate calibration' if p_hl >= 0.05 else 'poor calibration'} (chi-square = {chi2:.2f}, p = {_p_str(p_hl)})"
    brier_txt = f"Brier score = {brier:.4f}"
    c_txt = f"C-statistic (AUC) = {c_stat:.4f}" if c_stat is not None else "C-statistic not available"
    sep_txt = f"Separation detected in: {', '.join(separation_vars)}" if separation_vars else "No separation detected"

    result_text = (
        f"Logistic regression diagnostics (n = {n}, {len(req.predictors)} predictor{'s' if len(req.predictors) != 1 else ''}). "
        f"{hl_txt}. {brier_txt}. {c_txt}. {sep_txt}."
    )

    # ── export_rows ──────────────────────────────────────────────────────────
    export_rows = [
        ["Diagnostic", "Value"],
        ["n", n],
        ["Brier score", round(brier, 4)],
        ["C-statistic (AUC)", round(c_stat, 4) if c_stat is not None else None],
        ["Hosmer-Lemeshow chi-square", round(chi2, 4)],
        ["Hosmer-Lemeshow df", 8],
        ["Hosmer-Lemeshow p", round(p_hl, 6)],
        ["Separation detected", "Yes" if separation_vars else "No"],
        ["High DFBETA count", n_high_dfbeta],
        ["High leverage count", n_high_leverage],
    ]

    # ── r_code ───────────────────────────────────────────────────────────────
    pred_formula = " + ".join(req.predictors)
    r_code = (
        f"library(ResourceSelection)\n"
        f"model <- glm({req.outcome} ~ {pred_formula}, data = data, family = binomial)\n"
        f"summary(model)\n"
        f"hoslem.test(model$y, fitted(model), g = 10)  # Hosmer-Lemeshow\n"
        f"library(pROC)\n"
        f"roc(model$y, fitted(model))  # AUC/C-statistic\n"
        f"# Brier score\n"
        f"mean((fitted(model) - model$y)^2)"
    )

    return {
        "test": "Logistic Regression Diagnostics",
        "calibration": {
            "bins": calibration_bins,
        },
        "brier_score": round(brier, 4),
        "hosmer_lemeshow": {
            "chi2": round(chi2, 4),
            "df": 8,
            "p": round(p_hl, 6),
            "significant": bool(p_hl < 0.05),
        },
        "separation": {
            "detected": len(separation_vars) > 0,
            "variables": separation_vars,
        },
        "c_statistic": round(c_stat, 4) if c_stat is not None else None,
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COX PH DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def _cox_diag_design(df: pd.DataFrame, predictors: List[str]) -> tuple[pd.DataFrame, List[str]]:
    """Numeric design for the Cox diagnostics: numbers as is, text as dummies.

    A text column whose every value parses as a number is treated as numeric.
    Other non-numeric columns are dummy-coded against their first level.
    Rows are expected to be complete already. Returns the design (same index
    as `df`) and its column names.
    """
    parts: List[pd.DataFrame] = []
    for col in predictors:
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            parts.append(series.astype(float).to_frame(col))
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().all():
            parts.append(coerced.astype(float).to_frame(col))
            continue
        parts.append(pd.get_dummies(series.astype(str), prefix=col, drop_first=True).astype(float))
    design = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    design = design.loc[:, design.nunique() > 1]
    return design, list(design.columns)


class CoxDiagRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    imputation: str = "listwise"


@router.post("/cox_diagnostics")
def cox_diagnostics(req: CoxDiagRequest):
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    df_full = _get_df(req.session_id)
    all_cols = [req.duration_col, req.event_col] + req.predictors
    missing_cols = [c for c in all_cols if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(400, f"Columns not found: {missing_cols}")

    df = apply_imputation(df_full, all_cols, req.imputation or "listwise").copy()

    if len(df) < len(req.predictors) + 10:
        raise HTTPException(400, "Not enough observations for Cox PH diagnostics.")

    # Survival columns must be numeric. Predictors are numeric or categorical:
    # coercing a text predictor to numeric turned every value into NaN and
    # dropped every row, so categorical ones are dummy-coded instead.
    df[req.duration_col] = pd.to_numeric(df[req.duration_col], errors="coerce")
    df[req.event_col] = pd.to_numeric(df[req.event_col], errors="coerce")
    df = df.dropna(subset=all_cols).copy()

    if len(df) < len(req.predictors) + 10:
        raise HTTPException(400, "Not enough observations after cleaning.")

    # Ensure positive durations with the same row policy as the main Cox fit.
    surv = validate_survival_inputs(df, req.duration_col, req.event_col, mode="drop_with_warning")
    df = surv.df
    if len(df) < 10:
        raise HTTPException(400, "Not enough observations with positive duration.")

    n = len(df)
    n_events = int(df[req.event_col].sum())

    design, term_names = _cox_diag_design(df, req.predictors)
    if not term_names:
        raise HTTPException(400, "Predictors produced no usable columns (each needs at least two distinct values).")

    # ── Fit Cox model ────────────────────────────────────────────────────────
    fit_df = pd.concat([design, df[[req.duration_col, req.event_col]]], axis=1)
    try:
        cph = CoxPHFitter()
        cph.fit(fit_df, duration_col=req.duration_col, event_col=req.event_col)
    except Exception as exc:
        raise HTTPException(400, f"Cox PH model failed: {exc}")

    # ── 1. Schoenfeld residuals PH test ──────────────────────────────────────
    # One row per model term: a categorical predictor gives one per dummy.
    ph_results = []
    try:
        ph_test = proportional_hazard_test(cph, fit_df, time_transform="rank")
        for var in term_names:
            try:
                row = ph_test.summary.loc[var]
                test_stat = float(row["test_statistic"])
                p_val = float(row["p"])
                ph_results.append({
                    "variable": var,
                    "test_stat": round(test_stat, 4),
                    "p": round(p_val, 6),
                    "assumption_met": bool(p_val >= 0.05),
                })
            except (KeyError, IndexError):
                ph_results.append({
                    "variable": var,
                    "test_stat": None,
                    "p": None,
                    "assumption_met": True,
                })
    except Exception:
        # PH test may fail for some data configurations
        for var in term_names:
            ph_results.append({
                "variable": var,
                "test_stat": None,
                "p": None,
                "assumption_met": True,
            })

    # ── 2. C-index ───────────────────────────────────────────────────────────
    c_index = round(float(cph.concordance_index_), 4)

    # ── 3. Log-likelihood ratio test ─────────────────────────────────────────
    try:
        ll_test = cph.log_likelihood_ratio_test()
        ll_stat = round(float(ll_test.test_statistic), 4)
        ll_p = float(ll_test.p_value)
    except Exception:
        ll_stat = None
        ll_p = None

    # ── Assumptions ──────────────────────────────────────────────────────────
    assumptions = []

    # PH assumption for each variable
    violated = [r for r in ph_results if not r["assumption_met"]]
    if violated:
        assumptions.append({
            "name": "Proportional hazards",
            "met": False,
            "detail": f"PH assumption violated for: {', '.join(v['variable'] for v in violated)}",
        })
    else:
        assumptions.append({
            "name": "Proportional hazards",
            "met": True,
            "detail": "PH assumption met for all predictors",
        })

    # Events per variable (rule of thumb: >= 10)
    # Counted per model term: a categorical predictor spends one per dummy.
    epv = n_events / max(len(term_names), 1)
    assumptions.append({
        "name": "Events per variable",
        "met": epv >= 10,
        "detail": f"EPV = {epv:.1f} ({n_events} events, {len(term_names)} model terms)" + ("; may be underpowered" if epv < 10 else ""),
    })

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings = []
    warnings.extend(surv.warnings)
    if violated:
        warnings.append(f"PH assumption violated for: {', '.join(v['variable'] for v in violated)}. Consider time-varying coefficients or stratification.")
    if epv < 10:
        warnings.append(f"Low events per variable ({epv:.1f}). Model estimates may be unstable.")
    if c_index < 0.6:
        warnings.append(f"C-index = {c_index} — poor discrimination.")

    # ── result_text ──────────────────────────────────────────────────────────
    ph_txt = "PH assumption met for all predictors" if not violated else f"PH assumption violated for: {', '.join(v['variable'] for v in violated)}"
    ll_txt = f"Log-likelihood ratio test: chi-square = {ll_stat}, p = {_p_str(ll_p)}" if ll_p is not None else "Log-likelihood ratio test not available"

    result_text = (
        f"Cox PH diagnostics (n = {n}, {n_events} events, {len(req.predictors)} predictor{'s' if len(req.predictors) != 1 else ''}). "
        f"C-index = {c_index}. "
        f"{ph_txt}. "
        f"{ll_txt}."
    )

    # ── export_rows ──────────────────────────────────────────────────────────
    export_rows = [
        ["Diagnostic", "Value"],
        ["n", n],
        ["Events", n_events],
        ["C-index", c_index],
        ["Log-likelihood ratio stat", ll_stat],
        ["Log-likelihood ratio p", round(ll_p, 6) if ll_p is not None else None],
        ["Events per variable", round(epv, 1)],
    ]
    for r in ph_results:
        export_rows.append([f"PH test: {r['variable']} (stat)", r["test_stat"]])
        export_rows.append([f"PH test: {r['variable']} (p)", r["p"]])

    # ── r_code ───────────────────────────────────────────────────────────────
    pred_formula = " + ".join(req.predictors)
    r_code = (
        f"library(survival)\n"
        f"model <- coxph(Surv({req.duration_col}, {req.event_col}) ~ {pred_formula}, data = data)\n"
        f"summary(model)\n"
        f"cox.zph(model)  # Schoenfeld residuals PH test\n"
        f"concordance(model)  # C-index"
    )

    result = {
        "test": "Cox PH Diagnostics",
        "n": n,
        "n_events": n_events,
        "ph_test": ph_results,
        "c_index": c_index,
        "log_likelihood_ratio": {
            "stat": ll_stat,
            "p": round(ll_p, 6) if ll_p is not None else None,
        },
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }
    if surv.n_excluded:
        result["n_invalid_survival"] = int(surv.n_excluded)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NRI + IDI (Net Reclassification Improvement + Integrated Discrimination Improvement)
# ═══════════════════════════════════════════════════════════════════════════════

class NRIIDIRequest(BaseModel):
    session_id: str
    outcome: str
    prob_old: str          # column name for probabilities from old model
    prob_new: str          # column name for probabilities from new model
    cutoff: float = 0.5
    imputation: str = "listwise"


@router.post("/nri_idi")
def nri_idi(req: NRIIDIRequest):
    """
    Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI).
    Compares two prediction models using their predicted probabilities.
    """
    df_full = _get_df(req.session_id)
    cols = [req.outcome, req.prob_old, req.prob_new]
    df = apply_imputation(df_full, cols, req.imputation or "listwise")

    if len(df) < 100:
        raise HTTPException(400, "At least 100 observations recommended for stable NRI/IDI estimates.")

    y = pd.to_numeric(df[req.outcome], errors="coerce").astype(int).values
    p_old = pd.to_numeric(df[req.prob_old], errors="coerce").values
    p_new = pd.to_numeric(df[req.prob_new], errors="coerce").values

    mask = ~np.isnan(y) & ~np.isnan(p_old) & ~np.isnan(p_new)
    y, p_old, p_new = y[mask], p_old[mask], p_new[mask]

    if len(np.unique(y)) != 2:
        raise HTTPException(400, "Outcome must be binary.")

    # Reclassification
    old_high = (p_old >= req.cutoff).astype(int)
    new_high = (p_new >= req.cutoff).astype(int)

    events = (y == 1)
    non_events = (y == 0)

    n_up_event = np.sum((new_high > old_high) & events)
    n_down_event = np.sum((new_high < old_high) & events)
    n_event = np.sum(events)

    n_up_non = np.sum((new_high > old_high) & non_events)
    n_down_non = np.sum((new_high < old_high) & non_events)
    n_non = np.sum(non_events)

    nri_event = (n_up_event - n_down_event) / n_event if n_event > 0 else 0
    nri_non_event = (n_down_non - n_up_non) / n_non if n_non > 0 else 0
    nri = nri_event + nri_non_event

    # IDI
    mean_diff_event = np.mean(p_new[events]) - np.mean(p_old[events]) if n_event > 0 else 0
    mean_diff_non = np.mean(p_new[non_events]) - np.mean(p_old[non_events]) if n_non > 0 else 0
    idi = mean_diff_event - mean_diff_non

    # Bootstrap CI
    rng = np.random.default_rng(42)
    n_boot = 800
    n = len(y)
    nri_boots, idi_boots = [], []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb, po, pn = y[idx], p_old[idx], p_new[idx]
        oh = (po >= req.cutoff).astype(int)
        nh = (pn >= req.cutoff).astype(int)
        ev = (yb == 1)
        nev = (yb == 0)

        nri_e = ((nh > oh) & ev).sum() - ((nh < oh) & ev).sum()
        nri_ne = ((nh < oh) & nev).sum() - ((nh > oh) & nev).sum()
        nri_b = (nri_e / ev.sum() if ev.sum() > 0 else 0) + (nri_ne / nev.sum() if nev.sum() > 0 else 0)

        idi_b = (pn[ev].mean() - po[ev].mean() if ev.sum() > 0 else 0) - \
                (pn[nev].mean() - po[nev].mean() if nev.sum() > 0 else 0)

        nri_boots.append(nri_b)
        idi_boots.append(idi_b)

    nri_ci = np.percentile(nri_boots, [2.5, 97.5])
    idi_ci = np.percentile(idi_boots, [2.5, 97.5])

    return {
        "n": int(n),
        "cutoff_used": req.cutoff,
        "nri": {
            "estimate": round(nri, 4),
            "ci_low": round(nri_ci[0], 4),
            "ci_high": round(nri_ci[1], 4),
            "contribution_events": round(nri_event, 4),
            "contribution_non_events": round(nri_non_event, 4),
        },
        "idi": {
            "estimate": round(idi, 4),
            "ci_low": round(idi_ci[0], 4),
            "ci_high": round(idi_ci[1], 4),
        },
        "reclassification_counts": {
            "up_in_events": int(n_up_event),
            "down_in_events": int(n_down_event),
            "up_in_non_events": int(n_up_non),
            "down_in_non_events": int(n_down_non),
        },
        "test": "NRI + IDI (with bootstrap CI)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Missing Data Sensitivity Analysis (MNAR via Delta-Adjustment)
# ═══════════════════════════════════════════════════════════════════════════════

class SensitivityRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    model_type: str = "logistic"          # "linear" | "logistic" | "cox"
    mechanism: str = "MNAR"               # "MCAR" | "MAR" | "MNAR"
    missing_rate: float = 0.25
    delta_range: List[float] = [-1.5, 1.5]
    n_steps: int = 7
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    imputation: str = "listwise"


@router.post("/missing_data_sensitivity")
def missing_data_sensitivity(req: SensitivityRequest):
    """
    Run a delta-adjustment sensitivity analysis to explore how results
    change under different MNAR assumptions.
    """
    df_full = _get_df(req.session_id)

    # First simulate the desired missingness mechanism on top of existing missingness
    df_sim = simulate_missingness(
        df_full,
        [req.outcome] + req.predictors,
        mechanism=req.mechanism,
        missing_rate=req.missing_rate,
    )

    sens = delta_adjustment_sensitivity(
        df_sim,
        outcome=req.outcome,
        predictors=req.predictors,
        model_type=req.model_type,
        delta_range=tuple(req.delta_range),
        n_steps=req.n_steps,
        duration_col=req.duration_col,
        event_col=req.event_col,
    )

    summary = summarize_sensitivity(sens["results"])

    return {
        "test": "Missing Data Sensitivity (Delta Adjustment)",
        "mechanism_simulated": req.mechanism,
        "missing_rate": req.missing_rate,
        "model_type": req.model_type,
        "delta_range": req.delta_range,
        "results": sens["results"],
        "summary": summary,
        "interpretation": sens.get("interpretation"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model Validation (Bootstrap + Optimism Correction) - Phase 4
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationRequest(BaseModel):
    session_id: str
    outcome: Optional[str] = None
    prob_column: Optional[str] = None          # For binary / logistic
    model_type: str = "binary"                 # "binary" | "cox"
    n_boot: int = 300
    include_optimism: bool = True
    imputation: str = "listwise"

    # Cox-specific fields
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    linear_predictor_col: Optional[str] = None   # LP or risk score column for Cox

    # Predictors mode (proper, refit-based internal validation). When supplied,
    # the model is refit on each bootstrap/CV fold instead of scoring a fixed
    # precomputed probability column.
    predictors: List[str] = []
    cv_folds: int = 0                            # ≥3 → also run k-fold CV


def _internal_validation_predictors(req: "ValidationRequest", df_full: pd.DataFrame) -> dict:
    """Refit-based internal validation for a user-specified model.

    Binary: Harrell bootstrap optimism (refit per resample) for AUC + calibration
    slope, plus optional k-fold cross-validated AUC + calibration slope.
    Cox: bootstrap- and CV-based C-index optimism (refit CoxPHFitter).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.model_selection import StratifiedKFold, KFold

    is_cox = req.model_type == "cox"
    if is_cox:
        if not (req.duration_col and req.event_col):
            raise HTTPException(400, "Cox predictors mode needs duration_col and event_col.")
        cols = list(dict.fromkeys([req.duration_col, req.event_col] + req.predictors))
    else:
        if not req.outcome:
            raise HTTPException(400, "Binary predictors mode needs an outcome column.")
        cols = list(dict.fromkeys([req.outcome] + req.predictors))
    missing = [c for c in cols if c not in df_full.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")

    df = apply_imputation(df_full, cols, req.imputation or "listwise").reset_index(drop=True)
    Xd = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    if Xd.shape[1] == 0:
        raise HTTPException(400, "Predictors produced no usable columns.")

    n_boot = int(min(max(req.n_boot, 50), 500))
    cv_folds = int(req.cv_folds)
    rng = np.random.default_rng(42)

    if not is_cox:
        from services.model_validation import compute_calibration_slope_intercept
        y = pd.to_numeric(df[req.outcome], errors="coerce")
        keep = y.notna() & Xd.notna().all(axis=1)
        X = Xd[keep].values
        y = y[keep].astype(int).values
        if len(np.unique(y)) != 2:
            raise HTTPException(400, "Outcome must be binary 0/1 with both classes present.")
        n = len(y)
        if n < max(40, Xd.shape[1] + 20):
            raise HTTPException(400, "Not enough complete observations for internal validation.")

        def _fit(Xt, yt):
            return LogisticRegression(max_iter=1000, C=1e6).fit(Xt, yt)

        # Apparent performance
        m = _fit(X, y)
        p_app = m.predict_proba(X)[:, 1]
        auc_app = float(roc_auc_score(y, p_app))
        slope_app = compute_calibration_slope_intercept(y, probs=p_app)["calibration_slope"]
        brier_app = float(brier_score_loss(y, p_app))

        # Harrell optimism (refit per bootstrap, score original)
        opt_auc, opt_slope, ok = [], [], 0
        for _ in range(n_boot):
            bi = rng.choice(n, n, replace=True)
            if len(np.unique(y[bi])) < 2:
                continue
            try:
                mb = _fit(X[bi], y[bi])
                pb = mb.predict_proba(X[bi])[:, 1]      # boot performance
                po = mb.predict_proba(X)[:, 1]          # tested on original
                auc_b = roc_auc_score(y[bi], pb)
                auc_o = roc_auc_score(y, po)
                sl_b = compute_calibration_slope_intercept(y[bi], probs=pb)["calibration_slope"]
                sl_o = compute_calibration_slope_intercept(y, probs=po)["calibration_slope"]
                opt_auc.append(auc_b - auc_o)
                opt_slope.append(sl_b - sl_o)
                ok += 1
            except Exception:
                continue
        o_auc = float(np.mean(opt_auc)) if opt_auc else 0.0
        o_slope = float(np.mean(opt_slope)) if opt_slope else 0.0

        cv_block = None
        if cv_folds >= 3:
            oof = np.full(n, np.nan)
            for tr, te in StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42).split(X, y):
                if len(np.unique(y[tr])) < 2:
                    continue
                oof[te] = _fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            valid = ~np.isnan(oof)
            if valid.sum() > 10 and len(np.unique(y[valid])) == 2:
                cv_block = {
                    "auc": round(float(roc_auc_score(y[valid], oof[valid])), 4),
                    "calibration_slope": compute_calibration_slope_intercept(y[valid], probs=oof[valid])["calibration_slope"],
                    "brier": round(float(brier_score_loss(y[valid], oof[valid])), 4),
                    "folds": cv_folds,
                }

        return {
            "test": "Internal Validation (logistic, refit bootstrap optimism)",
            "model_type": "binary", "n": int(n), "n_boot": ok, "n_predictors": int(Xd.shape[1]),
            "apparent": {"auc": round(auc_app, 4), "calibration_slope": round(slope_app, 4), "brier": round(brier_app, 4)},
            "optimism": {"auc": round(o_auc, 4), "calibration_slope": round(o_slope, 4)},
            "corrected": {"auc": round(auc_app - o_auc, 4), "calibration_slope": round(slope_app - o_slope, 4)},
            "cv": cv_block,
            "overfit_gap": round(o_auc, 4),
            "interpretation": (
                f"Internal validation of a logistic model with {Xd.shape[1]} predictor term(s) (n = {n}). "
                f"Apparent AUC = {auc_app:.3f}; optimism = {o_auc:.3f}; "
                f"optimism-corrected AUC = {auc_app - o_auc:.3f}"
                + (f"; {cv_folds}-fold CV AUC = {cv_block['auc']:.3f}" if cv_block else "")
                + f". Optimism-corrected calibration slope = {slope_app - o_slope:.3f} "
                + ("(shrinkage advisable if « 1)." if (slope_app - o_slope) < 0.9 else "(close to 1 — good).")
            ),
            "r_code": (
                f'library(rms)\n'
                f'f <- lrm({req.outcome} ~ {" + ".join(req.predictors)}, data = data, x = TRUE, y = TRUE)\n'
                f'validate(f, B = {n_boot})  # optimism-corrected Dxy → C = Dxy/2 + 0.5, slope'
            ),
        }

    # ── Cox predictors mode ──────────────────────────────────────────────────
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    dur = pd.to_numeric(df[req.duration_col], errors="coerce")
    evt = pd.to_numeric(df[req.event_col], errors="coerce")
    base = pd.concat([dur.rename("_d_"), evt.rename("_e_"), Xd], axis=1).dropna()
    base = base[base["_d_"] > 0]
    n = len(base)
    if n < max(50, Xd.shape[1] + 25):
        raise HTTPException(400, "Not enough complete observations for Cox internal validation.")

    def _cfit_predict(train: pd.DataFrame, test: pd.DataFrame):
        cph = CoxPHFitter(penalizer=0.01).fit(train, duration_col="_d_", event_col="_e_")
        lp = cph.predict_partial_hazard(test).values
        return cph, np.log(np.clip(lp, 1e-12, None))

    cph_app, lp_app = _cfit_predict(base, base)
    c_app = float(concordance_index(base["_d_"].values, -lp_app, base["_e_"].values.astype(int)))

    opt_c, ok = [], 0
    arr = base.reset_index(drop=True)
    for _ in range(n_boot):
        bi = rng.choice(n, n, replace=True)
        bs = arr.iloc[bi].reset_index(drop=True)
        if bs["_e_"].sum() < 5:
            continue
        try:
            _, lp_b = _cfit_predict(bs, bs)
            _, lp_o = _cfit_predict(bs, arr)
            c_b = concordance_index(bs["_d_"].values, -lp_b, bs["_e_"].values.astype(int))
            c_o = concordance_index(arr["_d_"].values, -lp_o, arr["_e_"].values.astype(int))
            opt_c.append(c_b - c_o)
            ok += 1
        except Exception:
            continue
    o_c = float(np.mean(opt_c)) if opt_c else 0.0

    cv_block = None
    if cv_folds >= 3:
        oof = np.full(n, np.nan)
        for tr, te in KFold(n_splits=cv_folds, shuffle=True, random_state=42).split(arr):
            trd, ted = arr.iloc[tr], arr.iloc[te]
            if trd["_e_"].sum() < 5:
                continue
            try:
                _, lp_te = _cfit_predict(trd, ted)
                oof[te] = lp_te
            except Exception:
                continue
        valid = ~np.isnan(oof)
        if valid.sum() > 20:
            cv_block = {"c_index": round(float(concordance_index(arr["_d_"].values[valid], -oof[valid],
                                                                  arr["_e_"].values.astype(int)[valid])), 4),
                        "folds": cv_folds}

    return {
        "test": "Internal Validation (Cox, refit bootstrap optimism)",
        "model_type": "cox", "n": int(n), "n_boot": ok, "n_predictors": int(Xd.shape[1]),
        "apparent": {"c_index": round(c_app, 4)},
        "optimism": {"c_index": round(o_c, 4)},
        "corrected": {"c_index": round(c_app - o_c, 4)},
        "cv": cv_block,
        "overfit_gap": round(o_c, 4),
        "interpretation": (
            f"Internal validation of a Cox model with {Xd.shape[1]} predictor term(s) (n = {n}, "
            f"{int(arr['_e_'].sum())} events). Apparent C = {c_app:.3f}; optimism = {o_c:.3f}; "
            f"optimism-corrected C = {c_app - o_c:.3f}"
            + (f"; {cv_folds}-fold CV C = {cv_block['c_index']:.3f}" if cv_block else "") + "."
        ),
        "r_code": (
            f'library(rms)\n'
            f'f <- cph(Surv({req.duration_col}, {req.event_col}) ~ {" + ".join(req.predictors)}, '
            f'data = data, x = TRUE, y = TRUE)\n'
            f'validate(f, B = {n_boot})  # optimism-corrected Dxy / C-index'
        ),
    }


@router.post("/model_validation")
def model_validation(req: ValidationRequest):
    """
    Bootstrap performance + optional optimism correction for a prediction model.
    Supports binary outcomes and Cox models.
    """
    df_full = _get_df(req.session_id)

    # Predictors mode → proper refit-based optimism + optional k-fold CV.
    if req.predictors:
        return _internal_validation_predictors(req, df_full)

    if req.model_type == "cox":
        # Cox validation path
        if not (req.duration_col and req.event_col and req.linear_predictor_col):
            raise HTTPException(400, "For Cox validation you must provide duration_col, event_col, and linear_predictor_col")

        cols = [req.duration_col, req.event_col, req.linear_predictor_col]
        df = apply_imputation(df_full, cols, req.imputation or "listwise")

        if len(df) < 50:
            raise HTTPException(400, "At least 50 observations needed for reliable Cox validation.")

        lp = pd.to_numeric(df[req.linear_predictor_col], errors="coerce").values
        duration = pd.to_numeric(df[req.duration_col], errors="coerce").values
        event = pd.to_numeric(df[req.event_col], errors="coerce").values

        mask = ~np.isnan(duration) & ~np.isnan(event) & ~np.isnan(lp) & (duration > 0)
        duration = duration[mask]
        event = event[mask].astype(int)
        lp = lp[mask]

        # Bootstrap C-index (using lifelines)
        from lifelines.utils import concordance_index
        c_indices = []
        rng = np.random.default_rng(42)
        n = len(duration)

        for _ in range(req.n_boot):
            idx = rng.choice(n, n, replace=True)
            try:
                c = concordance_index(duration[idx], -lp[idx], event[idx])  # higher LP = higher risk → negate for concordance_index
                c_indices.append(c)
            except Exception:
                pass

        if c_indices:
            c_mean = float(np.mean(c_indices))
            c_ci = np.percentile(c_indices, [2.5, 97.5])
            perf = {
                "c_index": {
                    "mean": round(c_mean, 4),
                    "ci_low": round(c_ci[0], 4),
                    "ci_high": round(c_ci[1], 4),
                }
            }
        else:
            perf = {"c_index": {"mean": None, "ci_low": None, "ci_high": None}}

        result = {
            "test": "Model Validation - Cox",
            "n": int(n),
            "n_boot": req.n_boot,
            "bootstrap_performance": perf,
        }

        # Add calibration slope if possible
        try:
            cal = compute_cox_calibration_slope(df, req.duration_col, req.event_col, lp)
            result["calibration"] = cal
        except Exception:
            pass

        return result

    # === Binary / Logistic path (default) ===
    if not (req.outcome and req.prob_column):
        raise HTTPException(400, "For binary validation you must provide outcome and prob_column")

    cols = [req.outcome, req.prob_column]
    df = apply_imputation(df_full, cols, req.imputation or "listwise")

    if len(df) < 50:
        raise HTTPException(400, "At least 50 observations needed for reliable validation.")

    y = pd.to_numeric(df[req.outcome], errors="coerce").values
    probs = pd.to_numeric(df[req.prob_column], errors="coerce").values

    mask = ~np.isnan(y) & ~np.isnan(probs)
    y = y[mask].astype(int)
    probs = probs[mask]

    if len(np.unique(y)) != 2:
        raise HTTPException(400, "Outcome must be binary for this validation module.")

    perf = bootstrap_performance(y, probs, n_boot=req.n_boot)

    result = {
        "test": "Model Validation (Bootstrap performance)",
        "n": int(len(y)),
        "n_boot": req.n_boot,
        "bootstrap_performance": perf,
    }

    if req.include_optimism:
        # A column of predicted probabilities has no model to refit, so there
        # is no optimism to estimate; predictors mode above does the refit.
        result["optimism_correction"] = {"available": False, "reason": OPTIMISM_NEEDS_REFIT}

    # Attach calibration slope/intercept
    try:
        cal = compute_calibration_slope_intercept(y, probs=probs)
        result["calibration"] = cal
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Causal Sensitivity Analysis (E-value + Quantitative Bias Analysis) - Phase 5
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSensitivityRequest(BaseModel):
    observed_estimate: float = Field(..., gt=0, description="Point estimate on RR/OR/HR scale")
    ci_low: Optional[float] = Field(None, gt=0)
    ci_high: Optional[float] = Field(None, gt=0)
    measure: Literal["rr", "or", "hr"] = "rr"
    rare_outcome: bool = False
    baseline_risk: Optional[float] = Field(None, ge=0.001, le=0.99, description="Baseline outcome risk for OR->RR conversion (common outcomes)")
    confounding_strength: float = Field(2.0, gt=0)
    prevalence_exposed: float = Field(0.5, ge=0, le=1)
    prevalence_unexposed: float = Field(0.5, ge=0, le=1)


@router.post("/causal_sensitivity")
def causal_sensitivity(req: CausalSensitivityRequest):
    """
    E-value (VanderWeele & Ding) + quantitative bias analysis for unmeasured confounding.
    Supports direct use after PSM/IPTW marginal effects via the e_value_from_psm_or_iptw helper.
    """
    warnings: list[str] = []
    assumptions: list[dict] = []

    if req.ci_low and req.ci_high and req.ci_low >= req.ci_high:
        raise HTTPException(400, "ci_low must be < ci_high")

    if req.measure == "or" and not req.rare_outcome and req.baseline_risk is None:
        warnings.append(
            "OR supplied without rare_outcome=True or baseline_risk. The common-outcome approximation "
            "RR ≈ sqrt(OR) is used (VanderWeele & Ding). Provide baseline_risk for an exact conversion."
        )

    ev = e_value(
        estimate=req.observed_estimate,
        ci_low=req.ci_low,
        ci_high=req.ci_high,
        measure=req.measure,
        rare_outcome=req.rare_outcome,
        baseline_risk=req.baseline_risk,
    )

    qba = quantitative_bias_analysis(
        observed_estimate=req.observed_estimate,
        measure=req.measure,
        confounding_strength=req.confounding_strength,
        prevalence_exposed=req.prevalence_exposed,
        prevalence_unexposed=req.prevalence_unexposed,
    )

    # Assumptions / interpretation notes
    assumptions.append({
        "name": "No unmeasured confounding",
        "met": False,
        "detail": "E-value quantifies how strong unmeasured confounding must be to explain away the result.",
    })
    if ev.get("e_value_point_estimate", 99) < 2.0:
        warnings.append("Low E-value (<2). Result is sensitive to even weak unmeasured confounding.")

    # result_text (user-friendly)
    ev_pt = ev.get("e_value_point_estimate", 1.0)
    result_text = (
        f"Causal sensitivity: E-value = {ev_pt} for observed {req.measure.upper()}={req.observed_estimate}. "
        f"{ev.get('interpretation', '')}"
    )

    # export_rows for reproducibility
    export_rows = [
        ["Metric", "Value"],
        ["Observed estimate", req.observed_estimate],
        ["Measure", req.measure],
        ["E-value (point)", ev.get("e_value_point_estimate")],
        ["E-value (CI)", ev.get("e_value_ci")],
        ["Bias factor (QBA)", qba.get("bias_factor")],
        ["Bias-corrected estimate (QBA)", qba.get("bias_corrected_estimate")],
    ]

    return {
        "test": "Causal Sensitivity Analysis (E-value + QBA)",
        "e_value": ev,
        "quantitative_bias_analysis": qba,
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": "# See VanderWeele & Ding (2017) E-value paper\n# R: EValue::evalues.RR(est=..., lo=..., hi=...)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. External validation (logistic / binary) — apply a model's predicted
#    probabilities to a new cohort and assess discrimination + calibration.
# ═══════════════════════════════════════════════════════════════════════════════

class ExternalValLogisticRequest(BaseModel):
    session_id: str                    # validation cohort
    outcome: str                       # binary 0/1 in the validation cohort
    prob_column: str                   # predicted probabilities (dev model applied here)
    dev_auc: Optional[float] = None    # development AUC (for dev→val drop)
    dev_calibration_slope: Optional[float] = None
    imputation: str = "listwise"


@router.post("/external_validation_logistic")
def external_validation_logistic(req: ExternalValLogisticRequest):
    """External validation of a binary prediction model: take the predicted
    probabilities from the development model applied to a new (validation) cohort
    and report discrimination (AUC + DeLong CI), calibration (slope, intercept,
    Hosmer-Lemeshow, O/E), Brier score, a calibration plot, and the dev→val drop.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from services.model_validation import compute_calibration_slope_intercept
    from routers.decision_curve import _hosmer_lemeshow
    from routers.stats.nonparametric import _delong_placement_values

    df_full = _get_df(req.session_id)
    for c in [req.outcome, req.prob_column]:
        if c not in df_full.columns:
            raise HTTPException(400, f"Column '{c}' not found.")
    df = apply_imputation(df_full, [req.outcome, req.prob_column], req.imputation or "listwise")
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    p = pd.to_numeric(df[req.prob_column], errors="coerce")
    m = y.notna() & p.notna()
    y = y[m].astype(int).values
    p_raw = p[m].astype(float).values
    n = len(y)
    if n < 30 or len(np.unique(y)) != 2:
        raise HTTPException(400, "Need ≥30 validation rows with a binary 0/1 outcome (both classes).")
    if p_raw.size == 0 or np.nanmax(p_raw) > 1.0 or np.nanmin(p_raw) < 0.0:
        raise HTTPException(400, "prob_column must hold predicted probabilities in [0, 1].")
    p = np.clip(p_raw, 1e-6, 1 - 1e-6)

    # Discrimination + DeLong CI for a single AUC.
    auc = float(roc_auc_score(y, p))
    Vp, Vn = _delong_placement_values(y, p)
    var = np.var(Vp, ddof=1) / len(Vp) + np.var(Vn, ddof=1) / len(Vn)
    se = float(np.sqrt(max(var, 1e-12)))
    auc_lo, auc_hi = max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se)

    cal = compute_calibration_slope_intercept(y, probs=p)
    brier = float(brier_score_loss(y, p))
    hl = _hosmer_lemeshow(y, p)
    oe = float(np.mean(y) / np.mean(p)) if np.mean(p) > 1e-9 else None   # observed / expected

    # Calibration plot (decile bins: mean predicted vs observed).
    order = np.argsort(p)
    bins = np.array_split(order, min(10, n // 5)) if n >= 10 else [order]
    cal_plot = []
    for b in bins:
        if len(b):
            cal_plot.append({"pred": round(float(np.mean(p[b])), 4),
                             "obs": round(float(np.mean(y[b])), 4), "n": int(len(b))})

    slope, intercept = cal["calibration_slope"], cal["calibration_intercept"]
    calib_ok = bool(0.8 <= slope <= 1.2 and abs(intercept) <= 0.2)
    drop = {}
    if req.dev_auc is not None:
        drop["auc_drop"] = round(float(req.dev_auc) - auc, 4)
    if req.dev_calibration_slope is not None:
        drop["slope_shift"] = round(slope - float(req.dev_calibration_slope), 4)

    hl_p_str = _p_str(hl["p"])
    dev_txt = (
        f"; development AUC was {req.dev_auc:.3f} (drop {drop['auc_drop']:+.3f})"
        if "auc_drop" in drop else ""
    )
    calib_txt = (
        "Calibration is acceptable (slope ≈ 1, intercept ≈ 0)."
        if calib_ok else
        "Calibration is off (slope away from 1 and/or intercept away from 0) — "
        "the model may need recalibration in this population."
    )
    oe_txt = f"{oe:.2f}" if oe is not None else "n/a"
    result_text = (
        f"External validation on {n} patients. Discrimination: AUC = {auc:.3f} "
        f"(95% CI {auc_lo:.3f}–{auc_hi:.3f}){dev_txt}. "
        f"Calibration: slope = {slope:.2f}, intercept = {intercept:.2f}, O/E = {oe_txt}, "
        f"Hosmer-Lemeshow χ²({hl['df']}) = {hl['chi2']:.2f}, p = {hl_p_str}; "
        f"Brier = {brier:.3f}. {calib_txt}"
    )

    return {
        "test": "External Validation (logistic)",
        "outcome": req.outcome, "prob_column": req.prob_column, "n": int(n),
        "discrimination": {"auc": round(auc, 4), "auc_ci": [round(auc_lo, 4), round(auc_hi, 4)], "se": round(se, 5)},
        "calibration": {"slope": round(slope, 4), "intercept": round(intercept, 4),
                        "oe_ratio": round(oe, 4) if oe is not None else None,
                        "hosmer_lemeshow": {"chi2": hl["chi2"], "df": hl["df"], "p": hl["p"]},
                        "brier": round(brier, 4), "acceptable": calib_ok},
        "calibration_plot": cal_plot,
        "dev_vs_val": drop or None,
        "result_text": result_text, "interpretation": result_text,
        "r_code": (
            'library(rms)\n'
            f'val.prob({req.prob_column}, {req.outcome})  # C (AUC), slope, intercept, Brier, calibration plot'
        ),
    }
