"""
External Validation and Prediction Evaluation Framework (Phase 9)

Provides tools for rigorous evaluation of survival, multi-state, and joint
prediction models on external data, with focus on calibration, discrimination,
and transportability diagnostics.

Core capabilities implemented:
- Integrated Brier Score (IBS) approximation at multiple time points
- Calibration slope & intercept for survival predictions (using LP)
- Harrell's C-index on validation set
- Simple transportability diagnostics (covariate shift, baseline risk shift)
- Comparison of performance between development and validation cohorts

All functions return immutable dicts. Designed to integrate with existing
model_validation.py and the new joint/multi-state modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from services.model_validation import (
    competing_risks_calibration,
    compute_calibration_slope_intercept,
    flexible_calibration_curve,
)


def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v



def _ipcw_weights(duration: np.ndarray, event: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Compute IPCW weights using Kaplan-Meier censoring distribution (simple version)."""
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter()
    km.fit(duration, 1 - event)  # censoring indicator
    surv_cens = km.survival_function_.reindex(times, method="nearest").values.flatten()
    weights = 1.0 / np.clip(surv_cens, 1e-6, 1.0)
    return weights


def _prediction_to_probability(lp: np.ndarray) -> np.ndarray:
    lp = np.asarray(lp, dtype=float)
    if np.nanmin(lp) >= 0 and np.nanmax(lp) <= 1:
        return np.clip(lp, 1e-6, 1 - 1e-6)
    return np.clip(1.0 / (1.0 + np.exp(-lp)), 1e-6, 1 - 1e-6)


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / max(np.sum(w), 1e-12))


def _numeric_share(raw: pd.Series, coerced: pd.Series) -> float:
    """Share of the recorded values that parse as numbers.

    Measured over the non-missing entries only, so a column's type never
    depends on how complete it is.
    """
    observed = raw.notna()
    if not observed.any():
        return 0.0
    return float(coerced[observed].notna().mean())


def _standardized_mean_difference(dev: np.ndarray, val: np.ndarray) -> float:
    dev = np.asarray(dev, dtype=float)
    val = np.asarray(val, dtype=float)
    pooled = np.sqrt((np.nanvar(dev, ddof=1) + np.nanvar(val, ddof=1)) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((np.nanmean(val) - np.nanmean(dev)) / pooled)


def estimate_transport_weights(
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    covariate_cols: List[str],
) -> Dict[str, Any]:
    """
    Estimate validation-cohort IPTW/transport weights using a cohort-membership
    propensity model P(validation | X), following the usual Steingrimsson-style
    transportability decomposition: detect covariate shift, then reweight the
    validation sample toward the development covariate distribution.
    """
    try:
        from sklearn.linear_model import LogisticRegression

        covariate_cols = [c for c in covariate_cols if c in dev_df.columns and c in val_df.columns]
        if not covariate_cols:
            return {"available": False, "reason": "No shared covariates supplied."}
        dev = dev_df[covariate_cols].copy()
        val = val_df[covariate_cols].copy()
        combined = pd.concat([dev, val], axis=0, ignore_index=True)
        X = pd.get_dummies(combined, drop_first=True).apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        cohort = np.r_[np.zeros(len(dev)), np.ones(len(val))]
        model = LogisticRegression(max_iter=1000)
        model.fit(X, cohort)
        ps_val = np.clip(model.predict_proba(X.iloc[len(dev):])[:, 1], 1e-4, 1 - 1e-4)
        weights = (1.0 - ps_val) / ps_val
        weights = weights / np.mean(weights)
        ess = float((np.sum(weights) ** 2) / np.sum(weights ** 2))
        return {
            "available": True,
            "weights": weights,
            "propensity_summary": {
                "mean": round(float(np.mean(ps_val)), 5),
                "min": round(float(np.min(ps_val)), 5),
                "max": round(float(np.max(ps_val)), 5),
            },
            "weight_summary": {
                "mean": round(float(np.mean(weights)), 5),
                "min": round(float(np.min(weights)), 5),
                "max": round(float(np.max(weights)), 5),
                "effective_sample_size": round(ess, 2),
            },
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def transportability_analysis(
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    covariate_cols: List[str],
    *,
    duration_col: Optional[str] = None,
    event_col: Optional[str] = None,
    predicted_lp_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transportability report: covariate shift, baseline outcome shift, and
    IPTW summaries for weighted validation.
    """
    covariate_rows = []
    for col in covariate_cols:
        if col not in dev_df.columns or col not in val_df.columns:
            continue
        dev_col = dev_df[col]
        val_col = val_df[col]
        dev_num = pd.to_numeric(dev_col, errors="coerce")
        val_num = pd.to_numeric(val_col, errors="coerce")
        # Whether a covariate is numeric is a question about the values that
        # are there, not about how many are missing. The rule used to be
        # notna().mean() >= 0.7 over the whole column, so a lab value recorded
        # for only half the validation cohort was analysed as a categorical:
        # every distinct measurement became its own level, and the largest
        # level shift came out near 1/n — "no shift" for a covariate that may
        # have moved a long way.
        if _numeric_share(dev_col, dev_num) >= 0.7 and _numeric_share(val_col, val_num) >= 0.7:
            smd = _standardized_mean_difference(dev_num.dropna().to_numpy(), val_num.dropna().to_numpy())
            dev_missing = float(dev_num.isna().mean()) if len(dev_num) else 0.0
            val_missing = float(val_num.isna().mean()) if len(val_num) else 0.0
            covariate_rows.append({
                "covariate": col,
                "type": "numeric",
                "dev_mean": round(float(dev_num.mean()), 5),
                "val_mean": round(float(val_num.mean()), 5),
                "standardized_mean_difference": round(smd, 5),
                "flag_large_shift": bool(abs(smd) > 0.1),
                # The means and the SMD come from the non-missing rows only,
                # so state how many rows that was.
                "n_dev": int(dev_num.notna().sum()),
                "n_val": int(val_num.notna().sum()),
                "dev_missing_rate": round(dev_missing, 5),
                "val_missing_rate": round(val_missing, 5),
                "missingness_shift": round(val_missing - dev_missing, 5),
                "flag_missingness_shift": bool(abs(val_missing - dev_missing) > 0.1),
            })
        else:
            # Drop missing before stringifying. astype(str) renders NaN as the
            # literal "nan", which value_counts then reports as a level: a
            # cohort that merely recorded this covariate less often would show
            # up as a distribution shift, and normalising over the full length
            # would shrink every real level's proportion by the missing rate.
            # Missingness is a genuine transport concern, so it is reported —
            # as its own number, not smuggled in as a category.
            dev_props = dev_col.dropna().astype(str).value_counts(normalize=True)
            val_props = val_col.dropna().astype(str).value_counts(normalize=True)
            levels = sorted(set(dev_props.index) | set(val_props.index))
            max_abs = max(abs(float(val_props.get(k, 0)) - float(dev_props.get(k, 0))) for k in levels) if levels else 0.0
            dev_missing = float(dev_col.isna().mean()) if len(dev_col) else 0.0
            val_missing = float(val_col.isna().mean()) if len(val_col) else 0.0
            covariate_rows.append({
                "covariate": col,
                "type": "categorical",
                "max_absolute_level_shift": round(max_abs, 5),
                "flag_large_shift": bool(max_abs > 0.1),
                "n_dev": int(dev_col.notna().sum()),
                "n_val": int(val_col.notna().sum()),
                "dev_missing_rate": round(dev_missing, 5),
                "val_missing_rate": round(val_missing, 5),
                "missingness_shift": round(val_missing - dev_missing, 5),
                "flag_missingness_shift": bool(abs(val_missing - dev_missing) > 0.1),
            })

    weights = estimate_transport_weights(dev_df, val_df, covariate_cols)
    baseline_shift: Dict[str, Any] = {}
    if event_col and event_col in dev_df.columns and event_col in val_df.columns:
        dev_event = pd.to_numeric(dev_df[event_col], errors="coerce")
        val_event = pd.to_numeric(val_df[event_col], errors="coerce")
        baseline_shift["event_rate_dev"] = round(float(dev_event.mean()), 5)
        baseline_shift["event_rate_val"] = round(float(val_event.mean()), 5)
        baseline_shift["event_rate_difference"] = round(float(val_event.mean() - dev_event.mean()), 5)
    if predicted_lp_col and predicted_lp_col in val_df.columns:
        risk = pd.to_numeric(val_df[predicted_lp_col], errors="coerce").dropna()
        if len(risk):
            baseline_shift["validation_predicted_risk_mean"] = round(float(_prediction_to_probability(risk.to_numpy()).mean()), 5)

    shifted_missingness = [
        r["covariate"] for r in covariate_rows if r.get("flag_missingness_shift")
    ]
    transport_warnings = [
        "Large covariate shift can make unweighted external validation pessimistic or optimistic.",
        "Very small effective sample size after IPTW indicates weak overlap/positivity problems.",
    ]
    if shifted_missingness:
        transport_warnings.append(
            "The two cohorts record these covariates at different rates: "
            + ", ".join(shifted_missingness)
            + ". Shift is measured on the rows that have a value, so a cohort "
            "that simply collects a covariate less often will not show up as a "
            "distribution shift — but it still limits transportability."
        )

    return {
        "framework": "Steingrimsson-style external validity: covariate shift, outcome shift, and transport-weighted validation.",
        "covariate_shift": covariate_rows,
        "n_large_covariate_shifts": int(sum(1 for r in covariate_rows if r.get("flag_large_shift"))),
        "n_missingness_shifts": len(shifted_missingness),
        "baseline_shift": baseline_shift,
        "iptw": {k: v for k, v in weights.items() if k != "weights"},
        "weights": weights.get("weights") if weights.get("available") else None,
        "warnings": transport_warnings,
    }


def time_dependent_auc(
    duration: np.ndarray,
    event: np.ndarray,
    linear_predictor: np.ndarray,
    time_points: np.ndarray,
) -> List[Dict[str, float]]:
    """Time-dependent AUC(t) using risk-set ranking (Harrell-style at each t)."""
    results = []
    for t in time_points:
        at_risk = duration >= t
        if at_risk.sum() < 5:
            continue
        events_by_t = (duration <= t) & (event == 1) & at_risk
        if events_by_t.sum() < 2:
            continue

        # Among those at risk at t, rank by LP
        lp_risk = linear_predictor[at_risk]
        status = ((duration <= t) & (event == 1))[at_risk].astype(int)

        try:
            auc_t = concordance_index(
                np.ones_like(lp_risk),  # dummy times
                -lp_risk,
                status
            )
        except Exception:
            auc_t = None

        if auc_t:
            results.append({
                "time": round(float(t), 2),
                "auc": round(float(auc_t), 4),
                "n_at_risk": int(at_risk.sum()),
                "n_events": int(status.sum()),
            })
    return results


def integrated_brier_score(
    duration: np.ndarray,
    event: np.ndarray,
    survival_probs_at_times: np.ndarray,  # shape (n_samples, n_times)
    time_points: np.ndarray,
) -> Dict[str, float]:
    """
    Proper IPCW-weighted Integrated Brier Score (IBS) over the given time grid.
    Lower is better (0 = perfect).
    """
    if len(time_points) < 2:
        return {"ibs": None, "error": "Need at least 2 time points"}

    survival_probs_at_times = np.asarray(survival_probs_at_times)
    weights = _ipcw_weights(duration, event, time_points)

    brier_sum = 0.0

    for i, t in enumerate(time_points):
        # Observed status at t (1 if event by t, 0 otherwise, weighted by IPCW)
        obs_status = (duration <= t) & (event == 1)
        w = weights[i]

        # Predicted prob of event by t = 1 - S(t)
        pred_event_prob = 1.0 - survival_probs_at_times[:, i]

        # IPCW Brier at this time
        brier_t = np.mean(w * (obs_status.astype(float) - pred_event_prob) ** 2)
        brier_sum += brier_t

    ibs = brier_sum / len(time_points)

    return {
        "ibs": round(float(ibs), 5),
        "n_time_points": len(time_points),
        "time_range": [round(float(time_points[0]), 2), round(float(time_points[-1]), 2)],
    }


def evaluate_external_validation(
    val_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    predicted_lp_col: str,           # linear predictor or risk score on validation
    survival_probs: Optional[np.ndarray] = None,  # (n, len(time_points)) if available
    time_points: Optional[List[float]] = None,
    dev_metrics: Optional[Dict[str, float]] = None,
    sample_weight: Optional[np.ndarray] = None,
    flexible_calibration: bool = False,
    calibration_time_horizon: Optional[float] = None,
    competing_risk_status_col: Optional[str] = None,
    competing_risk_event_code: int = 1,
    predicted_cif_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enhanced external validation with proper time-dependent metrics when
    survival probabilities at multiple times are provided.
    """
    if predicted_lp_col not in val_df.columns:
        raise ValueError(f"predicted_lp_col '{predicted_lp_col}' not found in val_df")

    df = val_df[[duration_col, event_col, predicted_lp_col]].copy()
    df["__pos__"] = np.arange(len(val_df))
    df = df.dropna().copy()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna()

    if len(df) < 20:
        return {"error": "Too few complete observations in validation data"}

    duration = df[duration_col].values
    event = df[event_col].astype(int).values
    lp = df[predicted_lp_col].values
    probs = _prediction_to_probability(lp)
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if len(weights) != len(val_df):
            weights = None
        else:
            weights = weights[df["__pos__"].to_numpy(dtype=int)]
            weights = weights / max(float(np.mean(weights)), 1e-12)

    # Discrimination
    try:
        val_c = concordance_index(duration, -lp, event)
    except Exception:
        val_c = None

    # Calibration (using LP)
    try:
        cal = compute_calibration_slope_intercept(
            y=event, probs_or_lp=lp, duration=duration if np.any(event == 0) else None
        )
        val_slope = cal.get("calibration_slope")
        val_intercept = cal.get("calibration_intercept")
    except Exception:
        val_slope = val_intercept = None

    result = {
        "n_validation": int(len(df)),
        "validation_c_index": round(float(val_c), 4) if val_c else None,
        "validation_calibration_slope": round(float(val_slope), 4) if val_slope else None,
        "validation_calibration_intercept": round(float(val_intercept), 4) if val_intercept else None,
    }
    if weights is not None:
        event_by_median = ((duration <= np.median(duration)) & (event == 1)).astype(float)
        result["iptw_weighted_validation"] = {
            "weighted_observed_event_by_median_time": round(_weighted_mean(event_by_median, weights), 5),
            "weighted_predicted_risk_mean": round(_weighted_mean(probs, weights), 5),
            "effective_sample_size": round(float((np.sum(weights) ** 2) / np.sum(weights ** 2)), 2),
        }
    if flexible_calibration:
        result["flexible_calibration"] = flexible_calibration_curve(
            event,
            probs,
            duration=duration,
            time_horizon=calibration_time_horizon,
        )
    if competing_risk_status_col and predicted_cif_col and competing_risk_status_col in val_df.columns and predicted_cif_col in val_df.columns:
        result["competing_risks_calibration"] = competing_risks_calibration(
            val_df,
            duration_col=duration_col,
            status_col=competing_risk_status_col,
            predicted_cif_col=predicted_cif_col,
            event_code=competing_risk_event_code,
            time_horizon=calibration_time_horizon,
        )

    # Advanced metrics if survival probs provided
    if survival_probs is not None and time_points is not None:
        time_points = np.asarray(time_points)
        surv_probs_arr = np.asarray(survival_probs)
        if len(surv_probs_arr) == len(val_df):
            surv_probs_arr = surv_probs_arr[df["__pos__"].to_numpy(dtype=int)]
        ibs = integrated_brier_score(duration, event, surv_probs_arr, time_points)
        td_auc = time_dependent_auc(duration, event, lp, time_points)

        result["integrated_brier_score"] = ibs
        result["time_dependent_auc"] = td_auc

    # Performance drop vs dev
    if dev_metrics:
        drop = {}
        if dev_metrics.get("c_index") and result.get("validation_c_index"):
            drop["c_index_drop"] = round(dev_metrics["c_index"] - result["validation_c_index"], 4)
        if dev_metrics.get("calibration_slope") and result.get("validation_calibration_slope"):
            drop["calibration_slope_shift"] = round(
                result["validation_calibration_slope"] - dev_metrics["calibration_slope"], 4
            )
        if drop:
            result["performance_vs_dev"] = drop

    result["note"] = "Use survival_probs (n_samples x n_times) for accurate IBS and tdAUC."
    return result


def transportability_diagnostics(
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    covariate_cols: List[str],
) -> Dict[str, Any]:
    """
    Simple diagnostics for transportability / dataset shift between
    development and validation cohorts.
    """
    diagnostics = []
    for col in covariate_cols:
        if col not in dev_df.columns or col not in val_df.columns:
            continue
        dev_mean = float(dev_df[col].mean())
        val_mean = float(val_df[col].mean())
        shift = val_mean - dev_mean
        diagnostics.append({
            "covariate": col,
            "dev_mean": round(dev_mean, 4),
            "val_mean": round(val_mean, 4),
            "absolute_shift": round(shift, 4),
            "relative_shift": round(shift / (abs(dev_mean) + 1e-8), 4),
        })

    return {
        "covariate_shifts": diagnostics,
        "overall_shift_magnitude": round(float(np.mean([abs(d["absolute_shift"]) for d in diagnostics])), 4),
    }
