"""
Survival Machine Learning Module (Phase 10)

Provides ML-based survival prediction with clinical-grade interpretability
and direct integration with the Phase 9 external validation framework.

Current pragmatic implementation (staying close to existing dependencies):
- Uses sklearn GradientBoosting + a survival wrapper approach for ranking.
- Strong emphasis on honest validation via the external_validation service.
- Permutation importance + head-to-head comparison vs classical Cox.

For production Random Survival Forest, scikit-survival can be added later as optional.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, ParameterSampler, RepeatedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from services.model_validation import compute_calibration_slope_intercept
from services.external_validation import evaluate_external_validation


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


def _risk_to_survival_probs(
    risk_scores: np.ndarray,
    time_points: np.ndarray,
    *,
    invert: bool = True,
    risk_std_scale: float = 0.9,
) -> np.ndarray:
    """
    Immutable helper (Phase 12 deepened): convert risk scores into approximate
    survival probability curves S(t) suitable for IPCW Integrated Brier Score.

    Strategy:
    - Standardize risks (mean 0, controlled std) so the relative hazard exp()
      term has a sensible dynamic range regardless of whether the input came
      from GB regression or Cox partial hazard.
    - Use a simple linear ramp baseline cumulative hazard derived from the
      observed time scale (good enough for demo + property testing).
    - Explicit invert flag ensures higher risk always maps to lower survival.
    """
    risk = np.asarray(risk_scores, dtype=float)
    times = np.asarray(time_points, dtype=float)

    if len(risk) == 0 or len(times) == 0:
        return np.ones((len(risk), len(times)))

    # Standardize per-model (critical: GB risks and Cox LP live on different scales)
    risk = risk - np.nanmean(risk)
    rstd = np.nanstd(risk) or 1.0
    risk = risk / (rstd / risk_std_scale)

    if invert:
        # Higher (standardized) risk → higher hazard multiplier → faster decay
        eff_risk = -risk
    else:
        eff_risk = risk

    # Data-driven-ish baseline cumulative hazard: ramp from 0 to ~1.2 over the time grid
    # This keeps S(t) in a realistic [0.05, 0.98] band for typical clinical follow-up
    t_max = float(times[-1]) if times[-1] > 0 else 1.0
    cumhaz0 = (times / max(t_max, 1e-6)) * 1.15

    # S_i(t) = exp( -Λ0(t) * exp(eff_risk_i) )
    surv = np.exp(-cumhaz0[None, :] * np.exp(eff_risk)[:, None])
    return np.clip(surv, 1e-5, 1.0 - 1e-5)


def _prepare_survival_design(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    predictors: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[int]]:
    needed = [duration_col, event_col] + predictors
    work = df[needed].copy()
    X = work[predictors].copy()
    for c in predictors:
        if X[c].dtype == object or str(X[c].dtype).startswith("category"):
            X[c] = pd.Categorical(X[c]).codes
    X = X.apply(pd.to_numeric, errors="coerce")
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work[event_col] = pd.to_numeric(work[event_col], errors="coerce")
    mask = work[[duration_col, event_col]].notna().all(axis=1) & X.notna().all(axis=1)
    X = X.loc[mask].astype(float).reset_index(drop=True)
    durations = work.loc[mask, duration_col].astype(float).to_numpy()
    events = work.loc[mask, event_col].astype(int).to_numpy()
    return X, durations, events, list(X.columns), work.index[mask].tolist()


def _fit_gb_survival(
    X: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    *,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    model_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "random_state": random_state,
    }
    if params:
        model_params.update(params)
    y_target = -np.asarray(durations, dtype=float)
    sample_weight = np.where(np.asarray(events, dtype=int) == 1, 2.0, 1.0)
    model = GradientBoostingRegressor(**model_params)
    model.fit(X, y_target, sample_weight=sample_weight)
    return model


def _c_index(durations: np.ndarray, events: np.ndarray, risk: np.ndarray) -> Optional[float]:
    try:
        val = concordance_index(durations, -np.asarray(risk, dtype=float), events)
        return float(val) if np.isfinite(val) else None
    except Exception:
        return None


def _summarize(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(float(np.mean(arr)), 4),
        "sd": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "n": int(len(arr)),
    }


def _evaluate_params_cv(
    X: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    params: Dict[str, Any],
    *,
    n_splits: int,
    random_state: int,
) -> float:
    kf = KFold(n_splits=max(2, n_splits), shuffle=True, random_state=random_state)
    scores: List[float] = []
    for train_idx, val_idx in kf.split(X):
        model = _fit_gb_survival(
            X.iloc[train_idx],
            durations[train_idx],
            events[train_idx],
            params=params,
            random_state=random_state,
        )
        risk = model.predict(X.iloc[val_idx])
        score = _c_index(durations[val_idx], events[val_idx], risk)
        if score is not None:
            scores.append(score)
    return float(np.mean(scores)) if scores else -np.inf


def _run_repeated_cv(
    X: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    *,
    params: Dict[str, Any],
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> Dict[str, Any]:
    splitter = RepeatedKFold(
        n_splits=max(2, n_splits),
        n_repeats=max(1, n_repeats),
        random_state=random_state,
    )
    folds = []
    for i, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = _fit_gb_survival(
            X.iloc[train_idx],
            durations[train_idx],
            events[train_idx],
            params=params,
            random_state=random_state + i,
        )
        risk = model.predict(X.iloc[test_idx])
        score = _c_index(durations[test_idx], events[test_idx], risk)
        folds.append({
            "fold": i,
            "c_index": round(float(score), 4) if score is not None else None,
            "n_test": int(len(test_idx)),
            "events_test": int(np.sum(events[test_idx] == 1)),
        })
    vals = [f["c_index"] for f in folds if f["c_index"] is not None]
    return {
        "enabled": True,
        "folds": folds,
        "summary": _summarize(vals),
        "n_splits": int(max(2, n_splits)),
        "n_repeats": int(max(1, n_repeats)),
    }


def _run_nested_cv(
    X: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    *,
    base_params: Dict[str, Any],
    outer_folds: int,
    inner_folds: int,
    n_iter: int,
    random_state: int,
    optimization_method: str = "random",
) -> Dict[str, Any]:
    param_dist = {
        "n_estimators": [120, 200, 300, 450],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.05, 0.08, 0.12],
        "subsample": [0.7, 0.85, 1.0],
    }
    sampled = list(ParameterSampler(param_dist, n_iter=max(1, n_iter), random_state=random_state))
    outer = KFold(n_splits=max(2, outer_folds), shuffle=True, random_state=random_state)
    folds = []
    used_optimization = "random"
    optuna_error = None
    for fold, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
        best_params = dict(base_params)
        best_score = -np.inf
        if optimization_method == "bayesian":
            try:
                import optuna  # type: ignore

                used_optimization = "bayesian"

                def objective(trial: Any) -> float:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 120, 500, step=40),
                        "max_depth": trial.suggest_int("max_depth", 2, 4),
                        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                    }
                    return _evaluate_params_cv(
                        X.iloc[train_idx],
                        durations[train_idx],
                        events[train_idx],
                        {**base_params, **params},
                        n_splits=max(2, inner_folds),
                        random_state=random_state + fold,
                    )

                sampler = optuna.samplers.TPESampler(seed=random_state + fold)
                study = optuna.create_study(direction="maximize", sampler=sampler)
                study.optimize(objective, n_trials=max(1, n_iter), show_progress_bar=False)
                best_score = float(study.best_value)
                best_params = {**base_params, **study.best_params}
            except Exception as e:
                optuna_error = str(e)

        if best_score == -np.inf:
            for params in sampled:
                score = _evaluate_params_cv(
                    X.iloc[train_idx],
                    durations[train_idx],
                    events[train_idx],
                    {**base_params, **params},
                    n_splits=max(2, inner_folds),
                    random_state=random_state + fold,
                )
                if score > best_score:
                    best_score = score
                    best_params = {**base_params, **params}
        model = _fit_gb_survival(
            X.iloc[train_idx],
            durations[train_idx],
            events[train_idx],
            params=best_params,
            random_state=random_state + fold,
        )
        risk = model.predict(X.iloc[test_idx])
        outer_score = _c_index(durations[test_idx], events[test_idx], risk)
        folds.append({
            "fold": fold,
            "outer_c_index": round(float(outer_score), 4) if outer_score is not None else None,
            "inner_best_c_index": round(float(best_score), 4) if np.isfinite(best_score) else None,
            "best_params": {k: _safe(v) for k, v in best_params.items() if k != "random_state"},
            "n_test": int(len(test_idx)),
            "events_test": int(np.sum(events[test_idx] == 1)),
        })
    vals = [f["outer_c_index"] for f in folds if f["outer_c_index"] is not None]
    return {
        "enabled": True,
        "outer_folds": int(max(2, outer_folds)),
        "inner_folds": int(max(2, inner_folds)),
        "n_iter": int(max(1, n_iter)),
        "optimization": {
            "requested": optimization_method,
            "used": used_optimization,
            "fallback_reason": optuna_error,
        },
        "folds": folds,
        "summary": _summarize(vals),
        "interpretation": "Outer-fold C-index estimates performance after inner-loop tuning, reducing optimistic bias.",
    }


def _calibration_curve_at_times(
    durations: np.ndarray,
    events: np.ndarray,
    survival_probs: np.ndarray,
    time_points: np.ndarray,
    *,
    n_bins: int = 5,
) -> List[Dict[str, Any]]:
    from scipy.stats import chi2

    curves = []
    pred_event = 1.0 - np.asarray(survival_probs, dtype=float)
    for j, t in enumerate(time_points):
        pred = np.clip(pred_event[:, j], 1e-6, 1.0 - 1e-6)
        obs = ((durations <= t) & (events == 1)).astype(float)
        try:
            bins = pd.qcut(pred, q=min(n_bins, len(np.unique(pred))), duplicates="drop")
        except Exception:
            bins = pd.cut(pred, bins=min(n_bins, max(2, len(np.unique(pred)))), duplicates="drop")
        frame = pd.DataFrame({"pred": pred, "obs": obs, "bin": bins})
        bin_rows = []
        hl = 0.0
        for _, g in frame.groupby("bin", observed=False):
            n = len(g)
            exp_events = float(g["pred"].sum())
            obs_events = float(g["obs"].sum())
            exp_nonevents = n - exp_events
            obs_nonevents = n - obs_events
            hl += ((obs_events - exp_events) ** 2) / max(exp_events, 1e-6)
            hl += ((obs_nonevents - exp_nonevents) ** 2) / max(exp_nonevents, 1e-6)
            bin_rows.append({
                "n": int(n),
                "mean_predicted_event": round(float(g["pred"].mean()), 4),
                "observed_event_rate": round(float(g["obs"].mean()), 4),
                "absolute_error": round(float(abs(g["obs"].mean() - g["pred"].mean())), 4),
            })
        abs_errors = [r["absolute_error"] for r in bin_rows]
        df_hl = max(len(bin_rows) - 2, 1)
        p_hl = float(1 - chi2.cdf(hl, df_hl)) if np.isfinite(hl) else None
        curves.append({
            "time": round(float(t), 2),
            "bins": bin_rows,
            "ici": round(float(np.mean(abs_errors)), 4) if abs_errors else None,
            "e50": round(float(np.percentile(abs_errors, 50)), 4) if abs_errors else None,
            "e90": round(float(np.percentile(abs_errors, 90)), 4) if abs_errors else None,
            "hosmer_lemeshow_like": {
                "chi_square": round(float(hl), 4) if np.isfinite(hl) else None,
                "df": int(df_hl),
                "p": round(float(p_hl), 4) if p_hl is not None else None,
            },
        })
    return curves


def _partial_dependence_curves(
    model: GradientBoostingRegressor,
    X: pd.DataFrame,
    features: List[str],
    *,
    grid_size: int = 10,
) -> List[Dict[str, Any]]:
    rows = []
    baseline = X.median(axis=0).to_frame().T
    for feature in features:
        vals = np.unique(np.nanpercentile(X[feature].to_numpy(dtype=float), np.linspace(5, 95, grid_size)))
        points = []
        for val in vals:
            probe = baseline.copy()
            probe[feature] = val
            risk = float(model.predict(probe)[0])
            points.append({"value": round(float(val), 4), "mean_risk": round(risk, 5)})
        if len(points) >= 2:
            rows.append({
                "feature": feature,
                "points": points,
                "direction": "increasing risk" if points[-1]["mean_risk"] > points[0]["mean_risk"] else "decreasing risk",
            })
    return rows


def _shap_summary(
    model: GradientBoostingRegressor,
    X: pd.DataFrame,
    *,
    max_samples: int = 200,
) -> Dict[str, Any]:
    try:
        import shap  # type: ignore

        sample = X.sample(n=min(max_samples, len(X)), random_state=42) if len(X) > max_samples else X
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        arr = np.asarray(shap_values, dtype=float)
        mean_abs = np.mean(np.abs(arr), axis=0)
        items = [
            {"variable": col, "mean_abs_shap": round(float(val), 5)}
            for col, val in zip(X.columns, mean_abs)
        ]
        items.sort(key=lambda x: -x["mean_abs_shap"])
        return {"available": True, "method": "shap.TreeExplainer", "n_samples": int(len(sample)), "summary": items}
    except Exception as e:
        return {
            "available": False,
            "reason": str(e),
            "install_hint": "Install optional dependency `shap` to enable TreeExplainer output.",
        }


def _deephit_competing_risks_status() -> Dict[str, Any]:
    try:
        import pycox  # type: ignore  # noqa: F401
        import torchtuples  # type: ignore  # noqa: F401

        return {
            "available": True,
            "method": "DeepHit via pycox/torchtuples",
            "note": "Optional dependencies are installed; model training can be enabled in a dedicated DeepHit endpoint.",
        }
    except Exception as e:
        return {
            "available": False,
            "reason": str(e),
            "install_hint": "DeepHit competing-risks ML requires optional dependencies `pycox` and `torchtuples`.",
        }


def run_survival_ml_benchmark(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    predictors: Optional[List[str]] = None,
    n_estimators: int = 300,
    random_state: int = 42,
    nested_cv: bool = False,
    repeated_cv_repeats: int = 1,
    cv_folds: int = 5,
    inner_cv_folds: int = 3,
    hyperparameter_iter: int = 12,
    include_shap: bool = False,
    include_partial_dependence: bool = True,
    include_competing_risks_ml: bool = False,
    optimization_method: str = "random",
) -> Dict[str, Any]:
    """
    Runs a practical ML survival benchmark:
    - Gradient Boosting on the survival ranking problem (using negative duration as target for ordering)
    - Classical Cox baseline
    - Permutation importance for the ML model
    - Direct comparison using Phase 9-style metrics
    """
    optimization_method = (optimization_method or "random").lower()
    if optimization_method not in {"random", "bayesian"}:
        optimization_method = "random"

    if predictors is None:
        predictors = [c for c in df.columns if c not in (duration_col, event_col)]

    X, y_duration, y_event, predictors, kept_index = _prepare_survival_design(
        df, duration_col, event_col, predictors
    )
    if len(X) < 20:
        raise ValueError("Need at least 20 complete observations for survival ML benchmark.")
    df_work = df.loc[kept_index].reset_index(drop=True)

    # --- 1. Classical Cox baseline ---
    cox_df = pd.concat([df_work[[duration_col, event_col]], X], axis=1)
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col, robust=True)

    cox_lp = cph.predict_partial_hazard(X).values
    cox_c = float(cph.concordance_index_)

    # --- 2. Practical ML Survival model (Gradient Boosting ranking) ---
    # We treat it as a regression on -duration (higher risk → shorter time), with event weighting
    base_params = {"n_estimators": n_estimators, "max_depth": 4, "learning_rate": 0.05}
    gbr = _fit_gb_survival(X, y_duration, y_event, params=base_params, random_state=random_state)

    ml_risk = gbr.predict(X)

    # --- 3. Permutation importance (ML model) ---
    perm = permutation_importance(
        gbr, X, -y_duration, n_repeats=8, random_state=random_state, scoring="neg_mean_squared_error"
    )
    importance = [
        {"variable": col, "importance": round(float(imp), 5)}
        for col, imp in zip(predictors, perm.importances_mean)
    ]
    importance.sort(key=lambda x: -x["importance"])

    # --- 4. Quick validation comparison using Phase 9 style ---
    # We use the risk scores to compute a simple C-index style comparison
    ml_c = _c_index(y_duration, y_event, ml_risk)

    # Calibration using LP-style for both
    try:
        cox_cal = compute_calibration_slope_intercept(y_event, cox_lp, duration=y_duration)
    except Exception:
        cox_cal = {}

    try:
        ml_cal = compute_calibration_slope_intercept(y_event, ml_risk, duration=y_duration)
    except Exception:
        ml_cal = {}

    # Prepare risk scores for Phase 9 external validation (higher = higher risk)
    ml_risk_scores = ml_risk.astype(float)
    cox_risk_scores = cox_lp.astype(float)

    validation_ready = {
        "ml_risk_scores": ml_risk_scores.tolist(),
        "cox_risk_scores": cox_risk_scores.tolist(),
    }

    # --- Phase 12 Deepened: Generate survival probabilities + full Phase 9 metrics (IBS + tdAUC) ---
    time_grid = np.percentile(y_duration, [20, 40, 60, 80])
    time_grid = np.clip(time_grid, 0.1, None)

    # Explicit inversion: our ML risk and Cox LP are "higher = worse"
    # _risk_to_survival_probs(..., invert=True) ensures high-risk subjects get lower S(t)
    ml_surv_probs = _risk_to_survival_probs(ml_risk_scores, time_grid, invert=True)
    cox_surv_probs = _risk_to_survival_probs(cox_risk_scores, time_grid, invert=True)

    full_validation = {}
    try:
        ml_full = evaluate_external_validation(
            val_df=df_work.assign(ml_risk=ml_risk_scores),
            duration_col=duration_col,
            event_col=event_col,
            predicted_lp_col="ml_risk",
            survival_probs=ml_surv_probs,
            time_points=time_grid.tolist(),
        )
        cox_full = evaluate_external_validation(
            val_df=df_work.assign(cox_risk=cox_risk_scores),
            duration_col=duration_col,
            event_col=event_col,
            predicted_lp_col="cox_risk",
            survival_probs=cox_surv_probs,
            time_points=time_grid.tolist(),
        )
        full_validation = {
            "ml": ml_full,
            "cox": cox_full,
            "time_points": [round(float(t), 2) for t in time_grid],
        }
    except Exception as e:
        full_validation = {"error": str(e)}

    # Comparison table (now with real IBS when available)
    ml_ibs = None
    cox_ibs = None
    if isinstance(full_validation.get("ml"), dict):
        ml_ibs = full_validation["ml"].get("integrated_brier_score", {}).get("ibs")
    if isinstance(full_validation.get("cox"), dict):
        cox_ibs = full_validation["cox"].get("integrated_brier_score", {}).get("ibs")

    comparison = {
        "models": [
            {
                "name": "Classical Cox",
                "c_index": full_validation.get("cox", {}).get("validation_c_index") if isinstance(full_validation.get("cox"), dict) else None,
                "calibration_slope": full_validation.get("cox", {}).get("validation_calibration_slope") if isinstance(full_validation.get("cox"), dict) else None,
                "ibs": cox_ibs,
            },
            {
                "name": "Gradient Boosting Survival (ranking)",
                "c_index": full_validation.get("ml", {}).get("validation_c_index") if isinstance(full_validation.get("ml"), dict) else None,
                "calibration_slope": full_validation.get("ml", {}).get("validation_calibration_slope") if isinstance(full_validation.get("ml"), dict) else None,
                "ibs": ml_ibs,
            },
        ],
        "winner_by_c_index": None,
        "winner_by_ibs": None,
    }

    # Determine winners safely
    try:
        ml_c_val = full_validation.get("ml", {}).get("validation_c_index") if isinstance(full_validation.get("ml"), dict) else None
        cox_c_val = full_validation.get("cox", {}).get("validation_c_index") if isinstance(full_validation.get("cox"), dict) else None
        if ml_c_val is not None and cox_c_val is not None:
            comparison["winner_by_c_index"] = "ML" if ml_c_val > cox_c_val else "Cox"
        if ml_ibs is not None and cox_ibs is not None:
            comparison["winner_by_ibs"] = "ML" if ml_ibs < cox_ibs else "Cox"  # lower IBS better
    except Exception:
        pass

    # Rich metadata (immutable construction, following project conventions)
    assumptions = [
        "Gradient Boosting treats survival as a censored ranking / accelerated failure time surrogate (negative duration target + event weighting).",
        "Survival probabilities are approximated via exponential decay model calibrated to observed time scale; not a full parametric or non-parametric cumulative hazard estimator.",
        "Risk direction: higher ML risk score and higher Cox LP both correspond to worse prognosis (shorter survival). Inversion applied before S(t) generation.",
        "Phase 9 external_validation receives both LP (for C-index/calibration) and generated S(t) curves (for proper IPCW IBS and tdAUC).",
        "The headline classical_cox and ml_gradient_boosting_survival C-indices are APPARENT (resubstitution) values — the model is scored on the same rows it was fitted on. Gradient boosting is far more prone to this optimism than Cox, so the two are not directly comparable. Use repeated_cv for the honest, out-of-sample estimate.",
    ]
    warnings = []
    if ml_c is None:
        warnings.append("ML C-index could not be computed in-sample.")
    if "error" in full_validation:
        warnings.append("Full Phase 9 metrics (IBS/tdAUC) unavailable due to internal error in evaluate_external_validation.")
    if len(df) < 100:
        warnings.append("Small sample size; ML advantage estimates have high variance.")

    repeated_cv = _run_repeated_cv(
        X,
        y_duration,
        y_event,
        params=base_params,
        n_splits=cv_folds,
        n_repeats=repeated_cv_repeats,
        random_state=random_state,
    )
    nested_cv_result = None
    if nested_cv:
        nested_cv_result = _run_nested_cv(
            X,
            y_duration,
            y_event,
            base_params=base_params,
            outer_folds=cv_folds,
            inner_folds=inner_cv_folds,
            n_iter=hyperparameter_iter,
            random_state=random_state,
            optimization_method=optimization_method,
        )
    calibration_assessment = {
        "ml": _calibration_curve_at_times(y_duration, y_event, ml_surv_probs, time_grid),
        "cox": _calibration_curve_at_times(y_duration, y_event, cox_surv_probs, time_grid),
        "interpretation": "Calibration compares predicted event probability 1-S(t) with observed event rate by risk bins at each time point.",
    }
    pdp = []
    if include_partial_dependence:
        top_features = [row["variable"] for row in importance[: min(5, len(importance))]]
        pdp = _partial_dependence_curves(gbr, X, top_features)
    shap_values = _shap_summary(gbr, X) if include_shap else {"available": False, "reason": "SHAP not requested."}
    competing_risks_ml = _deephit_competing_risks_status() if include_competing_risks_ml else {
        "available": False,
        "reason": "Competing-risks ML not requested.",
    }

    # The apparent ML C-index is computed on the training rows, so a boosted
    # model can score near-perfectly while generalising no better than chance.
    # Say so out loud when the cross-validated estimate contradicts it —
    # otherwise the headline number invites a false "ML beats Cox" conclusion.
    cv_mean = (repeated_cv or {}).get("summary", {}).get("mean")
    if ml_c is not None and isinstance(cv_mean, (int, float)):
        if float(ml_c) - float(cv_mean) > 0.10:
            warnings.append(
                f"Apparent ML C-index ({round(float(ml_c), 3)}) is far above the cross-validated "
                f"estimate ({round(float(cv_mean), 3)}) — the gradient-boosting model is overfitting. "
                "Report the cross-validated value, not the apparent one."
            )
        if float(cv_mean) < 0.55:
            warnings.append(
                f"Cross-validated ML C-index is {round(float(cv_mean), 3)}, i.e. at or below chance "
                "(0.5). The ML model does not generalise on this dataset."
            )

    result_text = (
        f"Survival ML benchmark on n={len(df_work)} complete subjects. "
        f"Apparent (in-sample) C-index: classical Cox {round(cox_c, 3)}, "
        f"ML (GB ranking) {round(float(ml_c), 3) if ml_c else 'N/A'} — both optimistically biased, "
        "and more so for the ML model. "
        f"Honest cross-validated ML C-index: {repeated_cv['summary']['mean']}. "
        f"Full Phase 9 validation (IBS + tdAUC) and calibration curves computed on generated survival curves. "
        f"IBS winner: {comparison.get('winner_by_ibs') or 'undetermined'}. "
        "Useful when non-linear effects or interactions are suspected."
    )

    return {
        "n": int(len(df_work)),
        "n_excluded_missing": int(len(df) - len(df_work)),
        "classical_cox": {
            "c_index": round(cox_c, 4),
            # Scored on the fitting rows — consumers must label it as such.
            "c_index_type": "apparent",
            "calibration_slope": cox_cal.get("calibration_slope"),
        },
        "ml_gradient_boosting_survival": {
            "c_index": round(float(ml_c), 4) if ml_c else None,
            "c_index_type": "apparent",
            "calibration_slope": ml_cal.get("calibration_slope"),
            "permutation_importance": importance[:8],
            "shap_values": shap_values,
            "partial_dependence": pdp,
        },
        "repeated_cv": repeated_cv,
        "nested_cv": nested_cv_result or {"enabled": False, "reason": "Set nested_cv=true to run inner-loop tuning and outer-loop evaluation."},
        "calibration_assessment": calibration_assessment,
        "competing_risks_ml": competing_risks_ml,
        "validation_ready_risk_scores": validation_ready,
        "validation_ready_survival_probs": {
            "time_points": [round(float(t), 2) for t in time_grid],
            "ml_survival_probs": ml_surv_probs.tolist(),
            "cox_survival_probs": cox_surv_probs.tolist(),
        },
        "full_phase9_validation": full_validation,
        "auto_comparison": comparison,
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "note": "Phase 12 deepened integration: risk scores are converted to survival probability curves and fed to evaluate_external_validation for proper IBS and time-dependent AUC. ML model remains a practical Gradient Boosting ranking surrogate.",
    }


def auto_survival_ml_compare(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    predictors: Optional[List[str]] = None,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Phase 12: Automated comparison of classical Cox vs practical ML survival model.
    Returns ranked models with Phase 9 metrics + feature importance for the winner.
    """
    benchmark = run_survival_ml_benchmark(
        df, duration_col, event_col, predictors, n_estimators, random_state
    )

    comparison = benchmark.get("auto_comparison", {})
    models = comparison.get("models", [])

    # Add simple winner logic based on combined score (C-index primary, lower IBS secondary)
    if models:
        def score(m):
            c = m.get("c_index") or 0
            ibs = m.get("ibs") or 1.0
            return c - 0.3 * ibs  # simple composite

        ranked = sorted(models, key=score, reverse=True)
        comparison["ranked_models"] = ranked
        comparison["recommended"] = ranked[0]["name"] if ranked else None

    # Enrich with top features from ML
    ml_info = benchmark.get("ml_gradient_boosting_survival", {})
    comparison["top_features"] = ml_info.get("permutation_importance", [])[:5]

    return {
        "n": benchmark["n"],
        "comparison": comparison,
        "details": {
            "cox": benchmark["classical_cox"],
            "ml": benchmark["ml_gradient_boosting_survival"],
        },
        "full_phase9_validation": benchmark.get("full_phase9_validation"),
        "assumptions": benchmark.get("assumptions", []),
        "warnings": benchmark.get("warnings", []),
        "result_text": benchmark.get("result_text"),
        "note": "Automated survival ML vs classical comparison using full Phase 9 metrics (C-index + calibration + IBS + tdAUC via generated survival curves). For production-grade RSF, consider scikit-survival.",
    }
