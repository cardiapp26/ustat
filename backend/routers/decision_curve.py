"""Decision curve analysis and calibration for prediction models."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp
from sklearn.metrics import brier_score_loss, roc_auc_score
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from services import store
from services.category_health import clean_two_level, rare_level_warnings
from services.impute import apply_imputation
from services.decision_curve import (
    add_bootstrap_correction_to_dca,
    decision_curve_analysis_binary,
    decision_curve_analysis_survival,
)
from services.external_validation import (
    evaluate_external_validation,
    transportability_analysis,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _clean_predictor_categories(df: pd.DataFrame, predictors: List[str]) -> tuple[pd.DataFrame, list]:
    work = df.copy()
    warnings = []
    for col in predictors:
        if col not in work.columns or pd.api.types.is_numeric_dtype(work[col]):
            continue
        cleaned = clean_two_level(work[col])
        work[col] = cleaned.series
        warnings.extend(cleaned.warnings)
    work = work.dropna(subset=[c for c in predictors if c in work.columns])
    warnings.extend(rare_level_warnings(work, predictors))
    return work, warnings


def _hosmer_lemeshow(y: np.ndarray, probs: np.ndarray, n_groups: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test.

    Bins observations by deciles of predicted probability (default g=10).
    χ² = Σ (O_g − E_g)² / [E_g (1 − E_g/n_g)]
    df = g − 2
    p > 0.05 ⇒ model fits the data (failure to reject good-fit null).

    Returns dict with chi2, df, p, n_groups_used, and the per-group table
    (predicted mean, observed events, expected events, n).
    """
    from scipy.stats import chi2 as _chi2

    y = np.asarray(y, dtype=float)
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    if n < n_groups * 2:
        # Not enough data; cap groups so each bin has ≥ 2 observations.
        n_groups = max(2, min(n_groups, n // 2))

    # Sort by predicted probability, then bin into equal-sized groups.
    order = np.argsort(probs)
    y_s = y[order]
    p_s = probs[order]
    # Use np.array_split for near-equal bin sizes even when n is not divisible.
    idx_split = np.array_split(np.arange(n), n_groups)

    chi2_val = 0.0
    groups = []
    used_groups = 0
    for idx_arr in idx_split:
        if len(idx_arr) == 0:
            continue
        used_groups += 1
        obs = float(y_s[idx_arr].sum())
        exp = float(p_s[idx_arr].sum())
        ng = int(len(idx_arr))
        # Denominator zero-guard: if expected events are 0 OR equal to n_g
        # (everyone perfectly predicted), the bin's contribution to χ² is 0.
        denom = exp * (1 - exp / ng) if ng > 0 else 0.0
        if denom > 0:
            chi2_val += (obs - exp) ** 2 / denom
        groups.append({
            "predicted_mean": round(float(p_s[idx_arr].mean()), 4),
            "observed_events": int(obs),
            "expected_events": round(exp, 4),
            "n": ng,
        })

    df_val = max(used_groups - 2, 1)
    p_val = float(1 - _chi2.cdf(chi2_val, df_val))
    return {
        "chi2": round(chi2_val, 4),
        "df": df_val,
        "p": p_val,
        "n_groups": used_groups,
        "interpretation": "Good fit (p > 0.05)" if p_val > 0.05 else "Poor fit (p ≤ 0.05) — model misspecified",
        "groups": groups,
    }


def _fit_logistic(df: pd.DataFrame, outcome: str, predictors: List[str]):
    """Fit logistic regression and return model + predicted probabilities."""
    X = pd.get_dummies(df[predictors], drop_first=True).astype(float)
    X = sm.add_constant(X)
    y = df[outcome].astype(float)

    if y.nunique() < 2:
        raise HTTPException(400, "Outcome must be binary (0/1) with at least one event and one non-event.")
    if not set(y.unique()).issubset({0, 1, 0.0, 1.0}):
        raise HTTPException(400, "Outcome must be coded as 0/1 for calibration analysis.")

    try:
        model = sm.Logit(y, X).fit(disp=False, maxiter=100)
    except Exception as exc:
        raise HTTPException(400, f"Logistic regression failed: {exc}")

    probs = model.predict(X)
    return model, X, y.values, probs.values


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CALIBRATION PLOT
# ═══════════════════════════════════════════════════════════════════════════════

class CalibrationRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    n_bins: int = 10
    imputation: str = "listwise"


@router.post("/calibration")
def calibration(req: CalibrationRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation)
    df, cat_warnings = _clean_predictor_categories(df, req.predictors)
    n_excluded = n_total - len(df)

    if len(df) < 20:
        raise HTTPException(400, "Need at least 20 complete observations for calibration analysis.")

    model, X, y, probs = _fit_logistic(df, req.outcome, req.predictors)

    # ── Bin predictions ────────────────────────────────────────────────────
    n_bins = max(2, min(req.n_bins, len(df) // 5))  # at least 5 per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    predicted_means = []
    observed_props = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)

        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue

        pred_mean = float(probs[mask].mean())
        obs_events = int(y[mask].sum())
        obs_prop = obs_events / n_in_bin

        # Wilson CI for observed proportion
        if n_in_bin > 0:
            ci_low, ci_high = _wilson_ci(obs_events, n_in_bin)
        else:
            ci_low, ci_high = 0.0, 1.0

        bins.append({
            "predicted_mean": round(pred_mean, 4),
            "observed_prop": round(obs_prop, 4),
            "n": n_in_bin,
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
        })
        predicted_means.append(pred_mean)
        observed_props.append(obs_prop)

    # ── Calibration slope & intercept ──────────────────────────────────────
    if len(predicted_means) >= 2:
        cal_X = sm.add_constant(np.array(predicted_means))
        cal_model = sm.OLS(np.array(observed_props), cal_X).fit()
        cal_intercept = round(float(cal_model.params[0]), 4)
        cal_slope = round(float(cal_model.params[1]), 4)
    else:
        cal_intercept = 0.0
        cal_slope = 1.0

    # ── E/O ratio ──────────────────────────────────────────────────────────
    expected_events = float(probs.sum())
    observed_events = float(y.sum())
    eo_ratio = round(expected_events / observed_events, 4) if observed_events > 0 else float("inf")

    # ── Discrimination metrics ─────────────────────────────────────────────
    brier = round(float(brier_score_loss(y, probs)), 4)
    try:
        c_stat = round(float(roc_auc_score(y, probs)), 4)
    except ValueError:
        c_stat = None

    # ── Result text ────────────────────────────────────────────────────────
    slope_interp = "well-calibrated" if 0.8 <= cal_slope <= 1.2 else ("overfitting" if cal_slope < 0.8 else "underfitting")
    # Hosmer-Lemeshow goodness-of-fit test (deciles).
    hl = _hosmer_lemeshow(y, probs, n_groups=10)

    result_text = (
        f"Calibration analysis of {req.outcome} predicted by {', '.join(req.predictors)} "
        f"(n = {len(df)}, {n_excluded} excluded). "
        f"Calibration slope = {cal_slope}, intercept = {cal_intercept} ({slope_interp}). "
        f"E/O ratio = {eo_ratio}. Brier score = {brier}. "
        f"Hosmer-Lemeshow χ²({hl['df']}) = {hl['chi2']}, p = {_p_str(hl['p'])}."
    )
    if c_stat is not None:
        result_text += f" C-statistic (AUC) = {c_stat}."


    return {
        "test": "Calibration Analysis",
        "bins": bins,
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
        "eo_ratio": eo_ratio,
        "brier_score": brier,
        "hosmer_lemeshow": hl,
        "c_statistic": c_stat,
        "warnings": cat_warnings,
        "n": len(df),
        "n_excluded": n_excluded,
        "plot_data": {
            "predicted": predicted_means,
            "observed": observed_props,
            "identity_line": [0, 1],
        },
        "result_text": result_text,
        "export_rows": [
            ["Statistic", "Value"],
            ["Calibration slope", cal_slope],
            ["Calibration intercept", cal_intercept],
            ["E/O ratio", eo_ratio],
            ["Brier score", brier],
            ["C-statistic (AUC)", c_stat],
            ["n", len(df)],
            ["n excluded", n_excluded],
            *[
                [f"Bin {i+1}: predicted={b['predicted_mean']}, observed={b['observed_prop']}", f"n={b['n']}"]
                for i, b in enumerate(bins)
            ],
        ],
        "r_code": "library(rms)\nval.prob(predicted, observed)",
    }


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = sp.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DECISION CURVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 1b. HOSMER-LEMESHOW (standalone)
# ═══════════════════════════════════════════════════════════════════════════════

class HLRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    n_groups: int = 10
    imputation: str = "listwise"


@router.post("/hosmer_lemeshow")
def hosmer_lemeshow_endpoint(req: HLRequest):
    """Standalone Hosmer-Lemeshow goodness-of-fit for a logistic model.

    Fits the logistic model from the predictors, bins predicted probabilities
    into n_groups deciles, returns χ²/df/p plus the per-group observed vs
    expected table. p > 0.05 ⇒ model fits.
    """
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation)
    df, cat_warnings = _clean_predictor_categories(df, req.predictors)
    n_excluded = n_total - len(df)

    if len(df) < req.n_groups * 2:
        raise HTTPException(400, f"Need at least {req.n_groups * 2} complete observations for an H-L test with {req.n_groups} groups.")

    _, _, y, probs = _fit_logistic(df, req.outcome, req.predictors)
    hl = _hosmer_lemeshow(y, probs, n_groups=req.n_groups)
    return {
        "test": "Hosmer-Lemeshow Goodness-of-Fit",
        **hl,
        "warnings": cat_warnings,
        "n": int(len(df)),
        "n_excluded": int(n_excluded),
        "outcome": req.outcome,
        "predictors": req.predictors,
        "result_text": (
            f"Hosmer-Lemeshow χ²({hl['df']}) = {hl['chi2']}, p = {_p_str(hl['p'])}. "
            f"{hl['interpretation']}."
        ),
    }


class DCARequest(BaseModel):
    session_id: str
    # Traditional mode: fit logistic from predictors
    outcome: Optional[str] = None
    predictors: Optional[List[str]] = None
    # Phase 13 flexible mode: pre-computed predictions (recommended after survival ML / external validation)
    probability_col: Optional[str] = None          # column in session containing probabilities (0-1)
    risk_col: Optional[str] = None                 # column containing risk scores (higher = worse)
    # Survival DCA mode (uses duration + event + risk to approximate event prob by horizon)
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    time_horizon: Optional[float] = None

    threshold_range: List[float] = Field(default_factory=lambda: [0.01, 0.99])
    n_thresholds: int = 100
    bootstrap_corrected: bool = False
    n_boot: int = 200
    imputation: str = "listwise"


@router.post("/dca")
def dca(req: DCARequest):
    """
    Phase 13: Full router + frontend integration ready.

    Supports three input styles:
    1. Traditional: session + outcome + predictors (fits logistic internally)
    2. Pre-computed binary: session + probability_col or risk_col + outcome
    3. Survival: session + duration_col + event_col + risk_col (+ optional time_horizon)

    Always returns the rich Phase 13 payload (summary, standardized NB, assumptions,
    warnings, result_text) while preserving legacy curve shape for existing clients.
    """
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    # Determine columns we need for imputation
    needed_cols: List[str] = []
    if req.outcome:
        needed_cols.append(req.outcome)
    if req.predictors:
        needed_cols.extend(req.predictors)
    if req.probability_col:
        needed_cols.append(req.probability_col)
    if req.risk_col:
        needed_cols.append(req.risk_col)
    if req.duration_col:
        needed_cols.append(req.duration_col)
    if req.event_col:
        needed_cols.append(req.event_col)

    needed_cols = list(dict.fromkeys(needed_cols))  # dedup preserve order
    df = apply_imputation(df_full, needed_cols, req.imputation) if needed_cols else df_full
    cat_warnings = []
    if req.predictors:
        df, cat_warnings = _clean_predictor_categories(df, req.predictors)
    n_excluded = n_total - len(df)

    if len(df) < 20:
        raise HTTPException(400, "Need at least 20 complete observations for DCA.")

    # ── Resolve y and p (or risk) ─────────────────────────────────────────────
    dca_y = None
    dca_p = None
    dca_thresholds = np.linspace(req.threshold_range[0], req.threshold_range[1], req.n_thresholds)

    if req.duration_col and req.event_col and req.risk_col:
        # Survival mode (Phase 13 highlight)
        duration = df[req.duration_col].values
        event = df[req.event_col].values
        risk = df[req.risk_col].values
        service_res = decision_curve_analysis_survival(
            duration=duration,
            event=event,
            risk=risk,
            time_horizon=req.time_horizon,
            n_thresholds=req.n_thresholds,
            threshold_range=tuple(req.threshold_range),
        )
        service_res["mode"] = "survival"
        service_res["duration_col"] = req.duration_col
        service_res["event_col"] = req.event_col
        service_res["risk_col"] = req.risk_col
        event_times = df.loc[df[req.event_col] == 1, req.duration_col]
        horizon = req.time_horizon or float(np.percentile(event_times if len(event_times) else df[req.duration_col], 75))
        dca_y = ((df[req.duration_col].values <= horizon) & (df[req.event_col].values == 1)).astype(float)
        risk = df[req.risk_col].values
        dca_p = 1 / (1 + np.exp(-0.8 * (risk - np.median(risk))))

    elif (req.probability_col or req.risk_col) and req.outcome:
        # Pre-computed probability / risk mode (best for feeding from survival ML benchmark)
        p_or_risk = df[req.probability_col or req.risk_col].values
        y = df[req.outcome].values
        # If user gave risk scores (not probabilities), the service will still work
        # because decision_curve_analysis_binary accepts any monotonic score.
        service_res = decision_curve_analysis_binary(
            y=y,
            p=p_or_risk,
            n_thresholds=req.n_thresholds,
            threshold_range=tuple(req.threshold_range),
        )
        service_res["mode"] = "precomputed"
        service_res["probability_col"] = req.probability_col
        service_res["risk_col"] = req.risk_col
        service_res["outcome"] = req.outcome
        dca_y = np.asarray(y, dtype=float)
        dca_p = np.asarray(p_or_risk, dtype=float)

    elif req.outcome and req.predictors:
        # Traditional mode (backward compat)
        model, X, y, probs = _fit_logistic(df, req.outcome, req.predictors)
        service_res = decision_curve_analysis_binary(
            y=y,
            p=probs,
            n_thresholds=req.n_thresholds,
            threshold_range=tuple(req.threshold_range),
        )
        service_res["mode"] = "logistic_fitted"
        service_res["predictors"] = req.predictors
        service_res["outcome"] = req.outcome
        dca_y = np.asarray(y, dtype=float)
        dca_p = np.asarray(probs, dtype=float)
    else:
        raise HTTPException(400, "Provide either (outcome + predictors), (outcome + probability_col/risk_col), or (duration_col + event_col + risk_col)")

    if "error" in service_res:
        raise HTTPException(400, service_res["error"])

    if req.bootstrap_corrected and dca_y is not None and dca_p is not None:
        service_res = add_bootstrap_correction_to_dca(
            service_res,
            dca_y,
            dca_p,
            thresholds=dca_thresholds,
            n_boot=req.n_boot,
        )

    # Common enrichment
    service_res["n"] = len(df)
    service_res["n_excluded"] = n_excluded
    service_res["session_id"] = req.session_id
    if cat_warnings:
        service_res["warnings"] = list(service_res.get("warnings") or []) + cat_warnings

    # Legacy curve shape for old clients
    if "curves" in service_res and isinstance(service_res["curves"], dict) and "model_net_benefit" in service_res["curves"]:
        curves = service_res["curves"]
        service_res["curves"] = {
            "model": {"thresholds": curves["thresholds"], "net_benefit": curves["model_net_benefit"]},
            "treat_all": {"thresholds": curves["thresholds"], "net_benefit": curves.get("treat_all_net_benefit", [])},
            "treat_none": {"thresholds": curves["thresholds"], "net_benefit": curves.get("treat_none_net_benefit", [])},
        }

    service_res["r_code"] = "library(dcurves)  # or rms::val.prob / dca package"

    return service_res


class IntegratedExtValDCARequest(BaseModel):
    session_id: str                         # validation / target cohort
    duration_col: str
    event_col: str
    prediction_col: str                     # LP, risk score, event probability, state probability, CIF
    dev_session_id: Optional[str] = None
    covariates: Optional[List[str]] = None
    survival_prob_cols: Optional[List[str]] = None
    time_points: Optional[List[float]] = None
    time_horizon: Optional[float] = None
    threshold_range: List[float] = Field(default_factory=lambda: [0.01, 0.50])
    n_thresholds: int = 100
    bootstrap_corrected_dca: bool = True
    n_boot: int = 200
    flexible_calibration: bool = True
    prediction_source: str = "precomputed"  # survival_ml | joint_model | multistate | fine_gray | precomputed
    competing_risk_status_col: Optional[str] = None
    competing_risk_event_code: int = 1
    predicted_cif_col: Optional[str] = None
    imputation: str = "listwise"


@router.post("/integrated_extval_dca")
def integrated_extval_dca(req: IntegratedExtValDCARequest):
    """
    One-call integration: model predictions -> External Validation -> DCA.

    Designed for Survival ML, joint-model dynamic predictions, multistate state
    probabilities, Fine-Gray CIFs, or any precomputed prediction column.
    """
    val_full = _get_df(req.session_id)
    needed = [req.duration_col, req.event_col, req.prediction_col]
    if req.survival_prob_cols:
        needed.extend(req.survival_prob_cols)
    if req.competing_risk_status_col:
        needed.append(req.competing_risk_status_col)
    if req.predicted_cif_col:
        needed.append(req.predicted_cif_col)
    if req.covariates:
        needed.extend(req.covariates)
    needed = list(dict.fromkeys(needed))
    missing = [c for c in needed if c not in val_full.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    val_df = apply_imputation(val_full, needed, req.imputation)
    if len(val_df) < 20:
        raise HTTPException(400, "Need at least 20 complete validation observations.")

    survival_probs = None
    if req.survival_prob_cols:
        survival_probs = val_df[req.survival_prob_cols].to_numpy(dtype=float)

    transport = None
    weights = None
    if req.dev_session_id and req.covariates:
        dev_full = _get_df(req.dev_session_id)
        dev_needed = [c for c in req.covariates if c in dev_full.columns]
        dev_df = apply_imputation(dev_full, dev_needed + ([req.event_col] if req.event_col in dev_full.columns else []), req.imputation)
        transport = transportability_analysis(
            dev_df,
            val_df,
            req.covariates,
            duration_col=req.duration_col,
            event_col=req.event_col,
            predicted_lp_col=req.prediction_col,
        )
        weights = transport.get("weights") if transport else None

    extval = evaluate_external_validation(
        val_df,
        duration_col=req.duration_col,
        event_col=req.event_col,
        predicted_lp_col=req.prediction_col,
        survival_probs=survival_probs,
        time_points=req.time_points,
        sample_weight=weights,
        flexible_calibration=req.flexible_calibration,
        calibration_time_horizon=req.time_horizon,
        competing_risk_status_col=req.competing_risk_status_col,
        competing_risk_event_code=req.competing_risk_event_code,
        predicted_cif_col=req.predicted_cif_col,
    )

    dca_res = decision_curve_analysis_survival(
        duration=val_df[req.duration_col].to_numpy(dtype=float),
        event=val_df[req.event_col].to_numpy(dtype=int),
        risk=val_df[req.prediction_col].to_numpy(dtype=float),
        time_horizon=req.time_horizon,
        n_thresholds=req.n_thresholds,
        threshold_range=tuple(req.threshold_range),
    )
    if req.bootstrap_corrected_dca and "error" not in dca_res:
        event_times = val_df.loc[val_df[req.event_col] == 1, req.duration_col]
        horizon = req.time_horizon or float(np.percentile(event_times if len(event_times) else val_df[req.duration_col], 75))
        y = ((val_df[req.duration_col].to_numpy(dtype=float) <= horizon) & (val_df[req.event_col].to_numpy(dtype=int) == 1)).astype(float)
        risk = val_df[req.prediction_col].to_numpy(dtype=float)
        p = 1 / (1 + np.exp(-0.8 * (risk - np.median(risk))))
        thresholds = np.linspace(req.threshold_range[0], req.threshold_range[1], req.n_thresholds)
        dca_res = add_bootstrap_correction_to_dca(dca_res, y, p, thresholds=thresholds, n_boot=req.n_boot)

    return {
        "test": "Integrated External Validation + Decision Curve Analysis",
        "prediction_source": req.prediction_source,
        "n_validation": int(len(val_df)),
        "external_validation": extval,
        "decision_curve": dca_res,
        "transportability": {k: v for k, v in (transport or {}).items() if k != "weights"} if transport else None,
        "pipeline": {
            "prediction_col": req.prediction_col,
            "survival_prob_cols": req.survival_prob_cols or [],
            "prediction_to_extval": True,
            "prediction_to_dca": True,
            "iptw_weighted_validation": bool(weights is not None),
            "bootstrap_corrected_dca": bool(req.bootstrap_corrected_dca),
        },
        "result_text": (
            f"Integrated validation/DCA pipeline for {req.prediction_source} predictions "
            f"on n={len(val_df)} validation observations."
        ),
    }
