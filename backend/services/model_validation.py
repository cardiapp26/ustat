"""
Model Validation & Calibration Module (Phase 4)

Provides tools for proper internal validation of prediction models,
with focus on:
- Bootstrapped performance metrics (AUC, Brier, calibration slope/intercept)
- Optimism-corrected estimates
- Calibration assessment

This module is designed to support mid-to-advanced biostatistical practice.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def _to_logits(values: np.ndarray, scale: str) -> np.ndarray:
    """Logit-scale predictor from probabilities or a linear predictor.

    `scale` is "prob", "lp", or "auto". "auto" keeps the legacy guess (values
    wholly inside [0, 1] are probabilities) and exists only for callers that
    pass `probs_or_lp`; a linear predictor that happens to lie in [0, 1] is
    misread by it, so new callers must state the scale.
    """
    if scale not in ("prob", "lp", "auto"):
        raise ValueError(f"scale must be 'prob', 'lp' or 'auto'. Got: {scale}")
    values = np.asarray(values, dtype=float).ravel()
    outside = np.nanmin(values) < 0 or np.nanmax(values) > 1
    if scale == "prob" and outside:
        raise ValueError("probs must lie in [0, 1]; pass linear_predictor= for a linear predictor.")
    if scale == "lp" or (scale == "auto" and outside):
        return values
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def compute_calibration_slope_intercept(
    y: np.ndarray,
    probs: Optional[np.ndarray] = None,
    probs_or_lp: Optional[np.ndarray] = None,
    duration: Optional[np.ndarray] = None,
    *,
    linear_predictor: Optional[np.ndarray] = None,
    scale: Optional[str] = None,
) -> Dict[str, float]:
    """
    Logistic calibration: regress the observed outcome on the logit of the
    prediction; the coefficient is the calibration slope.

    The scale of the prediction is explicit:
    - `probs`: probabilities in [0, 1] (anything else raises).
    - `linear_predictor`: already on the logit scale, used as is.
    - `probs_or_lp`: legacy argument, scale guessed from the range unless
      `scale` ("prob" or "lp") says which it is.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y).ravel()
    del duration  # accepted for survival-call compatibility; binary calibration uses y/probs.
    if linear_predictor is not None:
        logits = _to_logits(linear_predictor, scale or "lp")
    elif probs is not None:
        logits = _to_logits(probs, scale or "prob")
    elif probs_or_lp is not None:
        logits = _to_logits(probs_or_lp, scale or "auto")
    else:
        raise ValueError("Provide probs, linear_predictor or probs_or_lp")

    X = logits.reshape(-1, 1)
    cal_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    cal_model.fit(X, y)

    intercept = float(cal_model.intercept_[0])
    slope = float(cal_model.coef_[0][0])

    return {
        "calibration_intercept": round(intercept, 4),
        "calibration_slope": round(slope, 4),
    }


IPCW_NOTE = (
    "Observed risk at the horizon is estimated with inverse probability of "
    "censoring weights (Kaplan-Meier of the censoring distribution). Subjects "
    "censored before the horizon have unknown status and get weight 0; they "
    "are not counted as non-events."
)


def ipcw_horizon_weights(
    duration: np.ndarray,
    observed: np.ndarray,
    horizon: float,
) -> np.ndarray:
    """IPCW weights for the binary endpoint "event by `horizon`".

    `observed` is 1 when the subject's follow-up ended with any event
    (including a competing one) and 0 when it was censored. Subjects whose
    status at the horizon is known get 1 / G(min(T, horizon)), with G the
    Kaplan-Meier estimate of remaining uncensored (left limit at event
    times); subjects censored before the horizon get 0.
    """
    from lifelines import KaplanMeierFitter

    duration = np.asarray(duration, dtype=float).ravel()
    observed = np.asarray(observed).ravel().astype(int)
    kmf = KaplanMeierFitter().fit(duration, event_observed=1 - observed)
    g_times = kmf.survival_function_.index.to_numpy(dtype=float)
    g_vals = kmf.survival_function_.iloc[:, 0].to_numpy(dtype=float)

    def g_at(t: np.ndarray, left: bool) -> np.ndarray:
        idx = np.searchsorted(g_times, t, side="left" if left else "right") - 1
        return g_vals[np.clip(idx, 0, len(g_vals) - 1)]

    had_event = (observed == 1) & (duration <= horizon)
    beyond = duration > horizon
    weights = np.zeros(len(duration))
    weights[had_event] = 1.0 / np.clip(g_at(duration[had_event], left=True), 1e-6, None)
    weights[beyond] = 1.0 / max(float(g_at(np.array([horizon]), left=False)[0]), 1e-6)
    return weights


def flexible_calibration_curve(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    duration: Optional[np.ndarray] = None,
    time_horizon: Optional[float] = None,
    n_knots: int = 4,
    n_grid: int = 50,
) -> Dict[str, Any]:
    """
    Flexible calibration using a natural cubic spline of logit(p).

    For survival predictions, pass duration + time_horizon: the endpoint is
    the event by the horizon, weighted by IPCW so that subjects censored
    before the horizon are not counted as non-events.
    """
    import statsmodels.api as sm
    from patsy import build_design_matrices, dmatrix

    y = np.asarray(y).ravel().astype(float)
    probs = np.clip(np.asarray(probs).ravel().astype(float), 1e-6, 1 - 1e-6)
    weights = np.ones(len(y))
    censoring_adjusted = duration is not None and time_horizon is not None
    if len(y) != len(probs):
        return {"available": False, "reason": "Outcome and predictions differ in length."}
    if censoring_adjusted:
        duration = np.asarray(duration).ravel().astype(float)
        weights = ipcw_horizon_weights(duration, y, float(time_horizon))
        y = ((duration <= float(time_horizon)) & (y == 1)).astype(float)
    keep = weights > 0
    y, probs, weights = y[keep], probs[keep], weights[keep]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {"available": False, "reason": "Need at least 20 rows with both outcome classes."}

    lp = np.log(probs / (1 - probs))
    df = pd.DataFrame({"y": y, "lp": lp})
    spline_df = max(3, int(n_knots))
    try:
        X = dmatrix(f"cr(lp, df={spline_df})", df, return_type="dataframe")
        model = sm.GLM(df["y"], X, family=sm.families.Binomial(), freq_weights=weights).fit()
        grid_p = np.linspace(max(0.001, float(np.percentile(probs, 1))), min(0.999, float(np.percentile(probs, 99))), n_grid)
        grid_lp = np.log(grid_p / (1 - grid_p))
        # Same knots as the fit: re-running dmatrix on the grid would place
        # the knots at the grid's own quantiles and evaluate another basis.
        Xg = build_design_matrices([X.design_info], {"lp": grid_lp}, return_type="dataframe")[0]
        observed = np.clip(model.predict(Xg), 0, 1)
        ici = float(np.average(np.abs(model.predict(X) - probs), weights=weights))
        result = {
            "available": True,
            "method": "restricted_cubic_spline_logistic_calibration",
            "n": int(len(y)),
            "ici": round(ici, 5),
            "curve": [
                {"predicted": round(float(p), 5), "observed": round(float(o), 5)}
                for p, o in zip(grid_p, observed)
            ],
        }
        if censoring_adjusted:
            result["censoring_adjustment"] = "ipcw"
            result["n_censored_before_horizon"] = int(np.sum(~keep))
            result["note"] = IPCW_NOTE
        return result
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def competing_risks_calibration(
    df: pd.DataFrame,
    duration_col: str,
    status_col: str,
    predicted_cif_col: str,
    *,
    event_code: int = 1,
    time_horizon: Optional[float] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calibration table for predicted CIF against observed cause-specific event by horizon."""
    if time_horizon is None:
        time_horizon = float(np.percentile(pd.to_numeric(df[duration_col], errors="coerce").dropna(), 75))
    work = df[[duration_col, status_col, predicted_cif_col]].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work[status_col] = pd.to_numeric(work[status_col], errors="coerce")
    work[predicted_cif_col] = pd.to_numeric(work[predicted_cif_col], errors="coerce")
    work = work.dropna()
    if len(work) < 20:
        return {"available": False, "reason": "Need at least 20 complete rows."}
    pred = np.clip(work[predicted_cif_col].to_numpy(dtype=float), 0, 1)
    dur = work[duration_col].to_numpy(dtype=float)
    status = work[status_col].to_numpy(dtype=int)
    obs = ((dur <= time_horizon) & (status == event_code)).astype(float)
    # Any non-zero status ends follow-up with a known outcome; 0 is censoring.
    weights = ipcw_horizon_weights(dur, (status != 0).astype(int), float(time_horizon))
    try:
        bins = pd.qcut(pred, q=min(n_bins, len(np.unique(pred))), duplicates="drop")
    except Exception:
        bins = pd.cut(pred, bins=min(n_bins, max(2, len(np.unique(pred)))), duplicates="drop")
    frame = pd.DataFrame({"pred": pred, "obs": obs, "w": weights, "bin": bins})
    rows = []
    for _, g in frame.groupby("bin", observed=False):
        w_sum = float(g["w"].sum())
        if len(g) == 0 or w_sum <= 0:
            continue
        observed_cif = float(np.average(g["obs"], weights=g["w"]))
        rows.append({
            "n": int(len(g)),
            "n_censored_before_horizon": int((g["w"] == 0).sum()),
            "mean_predicted_cif": round(float(g["pred"].mean()), 5),
            "observed_cumulative_incidence": round(observed_cif, 5),
            "absolute_error": round(float(abs(observed_cif - g["pred"].mean())), 5),
        })
    return {
        "available": True,
        "event_code": int(event_code),
        "time_horizon": round(float(time_horizon), 4),
        "bins": rows,
        "ici": round(float(np.mean([r["absolute_error"] for r in rows])), 5) if rows else None,
        "censoring_adjustment": "ipcw",
        "note": IPCW_NOTE + " Status 0 is treated as censoring and every other status as an observed event.",
    }


def compute_cox_calibration_slope(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    linear_predictor: np.ndarray
) -> Dict[str, float]:
    """
    For Cox models: regress the original data on the linear predictor
    from the original model. The coefficient on the LP is the calibration slope.
    """
    from lifelines import CoxPHFitter

    work = df[[duration_col, event_col]].copy()
    work["lp"] = np.asarray(linear_predictor, dtype=float).ravel()

    try:
        cph = CoxPHFitter()
        cph.fit(work, duration_col=duration_col, event_col=event_col)
        slope = float(cph.params_["lp"])
        return {
            "calibration_slope": round(slope, 4),
        }
    except Exception:
        return {"calibration_slope": None}


def bootstrap_performance(
    y: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 500,
    metrics: List[str] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap performance metrics with 95% CI.

    Supported metrics: 'auc', 'brier', 'calibration_slope', 'calibration_intercept'
    """
    if metrics is None:
        metrics = ['auc', 'brier', 'calibration_slope', 'calibration_intercept']

    y = np.asarray(y).ravel()
    probs = np.asarray(probs).ravel()
    n = len(y)

    rng = np.random.default_rng(random_state)
    results = {m: [] for m in metrics}

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb = y[idx]
        pb = probs[idx]

        if 'auc' in metrics:
            try:
                from sklearn.metrics import roc_auc_score
                results['auc'].append(roc_auc_score(yb, pb))
            except Exception:
                results['auc'].append(np.nan)

        if 'brier' in metrics:
            brier = np.mean((pb - yb) ** 2)
            results['brier'].append(brier)

        if 'calibration_slope' in metrics or 'calibration_intercept' in metrics:
            try:
                cal = compute_calibration_slope_intercept(yb, pb)
                if 'calibration_slope' in metrics:
                    results['calibration_slope'].append(cal['calibration_slope'])
                if 'calibration_intercept' in metrics:
                    results['calibration_intercept'].append(cal['calibration_intercept'])
            except Exception:
                if 'calibration_slope' in metrics:
                    results['calibration_slope'].append(np.nan)
                if 'calibration_intercept' in metrics:
                    results['calibration_intercept'].append(np.nan)

    summary = {}
    for m in metrics:
        vals = np.array(results[m])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 10:
            summary[m] = {
                "mean": round(float(np.mean(vals)), 4),
                "ci_low": round(float(np.percentile(vals, 2.5)), 4),
                "ci_high": round(float(np.percentile(vals, 97.5)), 4),
            }
        else:
            summary[m] = {"mean": None, "ci_low": None, "ci_high": None}

    return summary


FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _logistic_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Maximum-likelihood logistic regression: fit on train, predict test."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def _auc_and_slope(y: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    slope = compute_calibration_slope_intercept(y, probs=np.clip(probs, 0.0, 1.0))["calibration_slope"]
    return float(roc_auc_score(y, probs)), float(slope)


def optimism_corrected_metrics(
    y: np.ndarray,
    X: np.ndarray,
    n_boot: int = 200,
    random_state: int = 42,
    *,
    fit_predict: Optional[FitPredict] = None,
) -> Dict[str, Any]:
    """
    Harrell's bootstrap optimism correction for AUC and calibration slope.

    The model is refitted on every bootstrap sample; optimism is its
    performance on that sample minus its performance on the original data.
    This needs the design matrix `X` (n, p): predicted probabilities alone
    cannot be used, because a fixed set of predictions has nothing to refit.

    `fit_predict(X_train, y_train, X_test)` returns predicted probabilities
    for `X_test`; the default is maximum-likelihood logistic regression.
    """
    y = np.asarray(y).ravel().astype(int)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError(
            "optimism_corrected_metrics needs the design matrix X with one row per "
            "outcome; predicted probabilities cannot be optimism-corrected without refitting."
        )
    fit_predict = fit_predict or _logistic_fit_predict

    apparent_auc, apparent_slope = _auc_and_slope(y, fit_predict(X, y, X))

    rng = np.random.default_rng(random_state)
    n = len(y)
    optimism_auc: List[float] = []
    optimism_slope: List[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boot_auc, boot_slope = _auc_and_slope(y[idx], fit_predict(X[idx], y[idx], X[idx]))
            orig_auc, orig_slope = _auc_and_slope(y, fit_predict(X[idx], y[idx], X))
        except Exception:
            continue
        optimism_auc.append(boot_auc - orig_auc)
        optimism_slope.append(boot_slope - orig_slope)

    opt_auc = float(np.mean(optimism_auc)) if optimism_auc else 0.0
    opt_slope = float(np.mean(optimism_slope)) if optimism_slope else 0.0

    return {
        "apparent_auc": round(apparent_auc, 4),
        "optimism_auc": round(opt_auc, 4),
        "optimism_corrected_auc": round(apparent_auc - opt_auc, 4),
        "apparent_calibration_slope": round(apparent_slope, 4),
        "optimism_calibration_slope": round(opt_slope, 4),
        "optimism_corrected_calibration_slope": round(apparent_slope - opt_slope, 4),
        "n_boot": n_boot,
        "n_boot_used": len(optimism_auc),
        "method": "Harrell bootstrap optimism (model refitted in every resample)",
    }


OPTIMISM_NEEDS_REFIT = (
    "Optimism correction needs the model refitted in every bootstrap sample, "
    "which is impossible from a column of predicted probabilities. Run the "
    "validation with `predictors` to get refit-based optimism."
)


def add_validation_to_result(
    result: dict,
    y: np.ndarray,
    probs: np.ndarray,
    include_optimism: bool = True,
    n_boot: int = 300,
    X: Optional[np.ndarray] = None,
) -> dict:
    """Attach validation metrics to a copy of `result`.

    Optimism correction is computed only when the design matrix `X` is given.
    """
    out = {**result, "bootstrap_performance": bootstrap_performance(y, probs, n_boot=n_boot)}
    if include_optimism:
        out["optimism_correction"] = (
            optimism_corrected_metrics(y, X, n_boot=min(n_boot, 200))
            if X is not None
            else {"available": False, "reason": OPTIMISM_NEEDS_REFIT}
        )
    return out
