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

from services.model_validation import IPCW_NOTE, compute_cox_calibration_slope, ipcw_horizon_weights
from services.external_validation import evaluate_external_validation


# Ridge penalty used only when the unpenalised Cox fit does not converge.
FALLBACK_PENALIZER = 0.05


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


def _survival_probs_from_risk(
    risk_scores: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    time_points: np.ndarray,
) -> np.ndarray:
    """S(t | risk) on `time_points` from a risk score, estimated from the data.

    A one-covariate Cox model is fitted on the standardised score, and its
    Breslow baseline gives S_i(t) = S0(t) ** exp(b * z_i). The direction and
    the size of the score's effect, and the baseline, all come from the
    observed follow-up; nothing is assumed about the scale of the score. With
    b > 0 (a score where higher means worse) a higher score always gives a
    lower survival probability.

    Returns an array of shape (n, len(time_points)).
    """
    risk = np.asarray(risk_scores, dtype=float)
    times = np.asarray(time_points, dtype=float)
    if len(risk) == 0 or len(times) == 0:
        return np.ones((len(risk), len(times)))

    spread = float(np.nanstd(risk))
    z = (risk - float(np.nanmean(risk))) / (spread if spread > 0 else 1.0)
    frame = pd.DataFrame({"z": z, "_d_": np.asarray(durations, dtype=float), "_e_": np.asarray(events, dtype=int)})
    cph = CoxPHFitter().fit(frame, duration_col="_d_", event_col="_e_")
    surv = cph.predict_survival_function(frame[["z"]], times=times).to_numpy().T
    return np.clip(surv, 1e-5, 1.0 - 1e-5)


def _dummy_code(series: pd.Series, name: str) -> pd.DataFrame:
    """Dummy columns for a categorical predictor, first level as reference."""
    return pd.get_dummies(series.astype(str), prefix=name, drop_first=True).astype(float)


def _is_categorical(series: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series))


def _prepare_survival_design(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    predictors: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[int]]:
    """Complete-case design matrix for the survival benchmark.

    Rows with a missing value in any column, or a duration that is not
    positive, are dropped. Categorical predictors are dummy-coded against
    their first level; integer category codes would enter the Cox model as
    one linear term and would turn a missing value into a level of its own.

    Raises:
        ValueError: when the event column holds anything other than 0 and 1.
    """
    work = df[[duration_col, event_col] + predictors].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work[event_col] = pd.to_numeric(work[event_col], errors="coerce")
    categorical = [c for c in predictors if _is_categorical(work[c])]
    for c in predictors:
        if c not in categorical:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    work = work[work[duration_col] > 0]

    bad_events = sorted(set(work[event_col].unique()) - {0, 1})
    if bad_events:
        raise ValueError(
            f"Event column '{event_col}' must be coded 0 (censored) / 1 (event). Found: {bad_events[:5]}"
        )

    parts = [
        _dummy_code(work[c], c) if c in categorical else work[c].astype(float).to_frame(c)
        for c in predictors
    ]
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=work.index)
    kept_index = work.index.tolist()
    durations = work[duration_col].astype(float).to_numpy()
    events = work[event_col].astype(int).to_numpy()
    return X.reset_index(drop=True), durations, events, list(X.columns), kept_index


def _fit_cox_baseline(cox_df: pd.DataFrame, duration_col: str, event_col: str) -> Tuple[CoxPHFitter, float]:
    """Unpenalised Cox fit; a ridge penalty is used only if that fails.

    Returns the fitted model and the penalizer that was applied (0.0 for the
    ordinary partial-likelihood fit).
    """
    try:
        cph = CoxPHFitter().fit(cox_df, duration_col=duration_col, event_col=event_col, robust=True)
        return cph, 0.0
    except Exception:
        cph = CoxPHFitter(penalizer=FALLBACK_PENALIZER).fit(
            cox_df, duration_col=duration_col, event_col=event_col, robust=True
        )
        return cph, FALLBACK_PENALIZER


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
        # Subjects censored before t have unknown status: weight 0, and the
        # rest are up-weighted so each bin estimates the event proportion.
        weights = ipcw_horizon_weights(durations, events, float(t))
        try:
            bins = pd.qcut(pred, q=min(n_bins, len(np.unique(pred))), duplicates="drop")
        except Exception:
            bins = pd.cut(pred, bins=min(n_bins, max(2, len(np.unique(pred)))), duplicates="drop")
        frame = pd.DataFrame({"pred": pred, "obs": obs, "w": weights, "bin": bins})
        bin_rows = []
        hl = 0.0
        for _, g in frame.groupby("bin", observed=False):
            known = g[g["w"] > 0]
            if len(known) == 0:
                continue
            n = len(known)
            obs_rate = float(np.average(known["obs"], weights=known["w"]))
            pred_rate = float(np.average(known["pred"], weights=known["w"]))
            exp_events, obs_events = n * pred_rate, n * obs_rate
            hl += ((obs_events - exp_events) ** 2) / max(exp_events, 1e-6)
            hl += ((obs_events - exp_events) ** 2) / max(n - exp_events, 1e-6)
            bin_rows.append({
                "n": int(len(g)),
                "n_censored_before_time": int(len(g) - n),
                "mean_predicted_event": round(pred_rate, 4),
                "observed_event_rate": round(obs_rate, 4),
                "absolute_error": round(abs(obs_rate - pred_rate), 4),
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
    cox_df = X.assign(**{duration_col: y_duration, event_col: y_event})
    cph, cox_penalizer = _fit_cox_baseline(cox_df, duration_col, event_col)

    # Linear predictor (log partial hazard), not the partial hazard exp(LP).
    cox_lp = cph.predict_log_partial_hazard(X).to_numpy(dtype=float)
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

    # Calibration slope of a survival model: the coefficient of its linear
    # predictor in a Cox model on the same data. The GB score is a predicted
    # negative duration, not a log hazard, so it has no calibration slope.
    surv_frame = pd.DataFrame({duration_col: y_duration, event_col: y_event})
    cox_cal = compute_cox_calibration_slope(surv_frame, duration_col, event_col, cox_lp)
    ml_cal: Dict[str, Any] = {"calibration_slope": None}

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

    # Both scores are "higher = worse". Each is mapped to S(t) through a Cox
    # model on the score with a Breslow baseline estimated from these data.
    ml_surv_probs = _survival_probs_from_risk(ml_risk_scores, y_duration, y_event, time_grid)
    cox_surv_probs = _survival_probs_from_risk(cox_risk_scores, y_duration, y_event, time_grid)

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
                "calibration_slope": cox_cal.get("calibration_slope"),
                "ibs": cox_ibs,
            },
            {
                "name": "Gradient Boosting Survival (ranking)",
                "c_index": full_validation.get("ml", {}).get("validation_c_index") if isinstance(full_validation.get("ml"), dict) else None,
                "calibration_slope": ml_cal.get("calibration_slope"),
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
        "Survival probabilities S(t) come from a Cox model on each model's standardised risk score with a Breslow baseline hazard, both estimated on the analysed rows. They are in-sample (apparent) predictions.",
        "Risk direction: higher ML risk score and higher Cox linear predictor both correspond to worse prognosis, so a higher score gives a lower S(t).",
        "Calibration slope is the coefficient of the linear predictor in a Cox model on the same data (1 = no over- or under-fitting; exactly 1 in-sample for an unpenalised Cox model). The gradient-boosting score is a predicted negative duration, not a log hazard, so no calibration slope is reported for it.",
        "Calibration bins compare predicted 1 - S(t) with the observed event proportion estimated by inverse probability of censoring weights; subjects censored before t are not counted as event-free.",
        "Phase 9 external_validation receives both LP (for C-index/calibration) and generated S(t) curves (for proper IPCW IBS and tdAUC).",
        "The headline classical_cox and ml_gradient_boosting_survival C-indices are APPARENT (resubstitution) values — the model is scored on the same rows it was fitted on. Gradient boosting is far more prone to this optimism than Cox, so the two are not directly comparable. Use repeated_cv for the honest, out-of-sample estimate.",
    ]
    warnings = []
    if cox_penalizer > 0:
        warnings.append(
            f"The unpenalised Cox model did not converge; the Cox baseline uses a ridge penalty of {cox_penalizer}, "
            "which shrinks its coefficients towards zero."
        )
    n_dropped = int(len(df) - len(df_work))
    if n_dropped:
        warnings.append(f"{n_dropped} row(s) with a missing value or a non-positive duration were excluded.")
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
        "interpretation": "Calibration compares predicted event probability 1-S(t) with the IPCW-estimated observed event proportion by risk bins at each time point.",
        "censoring_adjustment": "ipcw",
        "note": IPCW_NOTE,
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
            "calibration_slope_method": "Cox regression on the linear predictor",
            "penalizer": cox_penalizer,
        },
        "ml_gradient_boosting_survival": {
            "c_index": round(float(ml_c), 4) if ml_c else None,
            "c_index_type": "apparent",
            "calibration_slope": ml_cal.get("calibration_slope"),
            "calibration_slope_note": "Not defined: the score is a predicted negative duration, not a log hazard.",
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
