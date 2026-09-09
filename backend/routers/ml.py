"""
Machine-learning predictive-modeling router.

Endpoints
---------
POST /random_forest      — Random Forest classifier / regressor
POST /gradient_boosting  — Gradient Boosting classifier / regressor
POST /feature_importance — permutation importance for a chosen model

Design notes
------------
- Pure scikit-learn (already a dependency). No xgboost / shap required for v1;
  feature importance uses sklearn's impurity importance plus model-agnostic
  permutation importance.
- Honest performance: classification metrics (AUC, calibration, confusion)
  come from out-of-fold predictions so they are not optimistic in-sample
  numbers.

NOTHING IS LEARNED FROM THE VALIDATION FOLD. Every step that estimates
something from data - the missing-value imputer and the one-hot encoder both
- lives inside a scikit-learn ``Pipeline`` that is refitted on each training
fold. Imputing (or encoding) the whole dataset once and cross-validating
afterwards leaks the held-out rows into the numbers that are then reported as
out-of-sample, which is the first pitfall scikit-learn's own guide names:
https://scikit-learn.org/stable/common_pitfalls.html

Two consequences worth stating outright, because both were wrong before:

- **The outcome is never imputed.** A row whose outcome is missing is excluded
  from the analysis, not filled in. Filling it invents the very thing being
  predicted, and with a binary outcome an imputed 0.5128 is not even a class.
  ``n`` in the response counts observed outcomes only.
- **Permutation importance is measured on held-out rows.** Permuting a column
  on the data the forest memorised measures how much of the training set the
  model can recite, which for a deep forest is "all of it" regardless of
  whether the feature predicts anything. The reported value is pooled over
  each fold's own validation rows, and it is per *source column*, not per
  dummy: permuting one level of a categorical while its siblings stay put is
  not a question anyone is asking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store
from services.impute import MICE_SINGLE, STRATEGY_LABELS

router = APIRouter()

_Z95 = 1.959963984540054

#: Strategies that fit something and therefore must be refitted per fold.
_FITTED_STRATEGIES = frozenset({"median", "mean", "mice"})


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _safe(v: Any) -> Any:
    """JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _downsample_curve(fpr: np.ndarray, tpr: np.ndarray, max_pts: int = 300) -> List[dict]:
    n = len(fpr)
    step = max(1, n // max_pts)
    idx = list(range(0, n, step))
    if (n - 1) not in idx:
        idx.append(n - 1)
    return [{"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6)} for i in idx]


# -- Column roles -----------------------------------------------------------


def _column_roles(raw: pd.DataFrame, predictors: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Split predictors into numeric and categorical, coercing where obvious.

    This is a decision about a column's *type*, taken from the column alone and
    never from the outcome, so it is settled once for the whole design rather
    than refitted per fold - the same way a variable's declared kind is.
    """
    out = raw.copy()
    numeric: List[str] = []
    categorical: List[str] = []
    for c in predictors:
        col = out[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric.append(c)
            continue
        coerced = pd.to_numeric(col, errors="coerce")
        if coerced.notna().mean() >= 0.8 and coerced.dropna().nunique() > 2:
            out[c] = coerced
            numeric.append(c)
        else:
            categorical.append(c)
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out, numeric, categorical


def _source_column(transformed: str, numeric: List[str], categorical: List[str]) -> Optional[str]:
    """Map a ColumnTransformer output name back to the predictor it came from.

    ``num__age`` -> ``age``; ``cat__sex_female`` -> ``sex``. Categorical column
    names may themselves contain underscores, so the longest matching source
    name wins rather than the first.
    """
    _, _, bare = transformed.partition("__")
    if bare in numeric:
        return bare
    for col in sorted(categorical, key=len, reverse=True):
        if bare == col or bare.startswith(f"{col}_"):
            return col
    return None


# -- Request model ----------------------------------------------------------


class MLRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    task: Optional[str] = "auto"            # auto | classification | regression
    # Hyper-parameters (sensible clinical defaults)
    n_estimators: int = 300
    max_depth: Optional[int] = None          # None = grow until pure
    min_samples_leaf: int = 1
    learning_rate: float = 0.1               # gradient boosting only
    # Evaluation
    cv_folds: int = 5
    class_weight_balanced: bool = True       # classification imbalance
    n_permutation_repeats: int = 10
    random_state: int = 42
    imputation: Optional[str] = "listwise"


# -- Design -----------------------------------------------------------------


@dataclass
class _Design:
    """The modelling frame plus the record of who was left out and why."""

    X: pd.DataFrame
    y: np.ndarray
    task: str
    numeric: List[str]
    categorical: List[str]
    strategy: str
    n_missing_outcome: int
    n_incomplete_categorical: int
    n_incomplete_numeric: int

    @property
    def imputation_report(self) -> Dict[str, Any]:
        applied = MICE_SINGLE if self.strategy == "mice" else self.strategy
        fitted = self.strategy in _FITTED_STRATEGIES
        return {
            "requested": self.strategy,
            "applied": applied,
            "label": STRATEGY_LABELS.get(applied, applied),
            "fitted_per_fold": fitted,
            "outcome_imputed": False,
            "n_imputations": 1,
            "pooled": False,
            "n_excluded_missing_outcome": self.n_missing_outcome,
            "n_excluded_incomplete_predictors": (
                self.n_incomplete_categorical + self.n_incomplete_numeric
            ),
        }


def _resolve_task(req: MLRequest, y: pd.Series) -> str:
    if req.task in ("classification", "regression"):
        return req.task
    # auto: binary / few-level integer -> classification, else regression
    nun = y.nunique(dropna=True)
    if nun <= 2:
        return "classification"
    if pd.api.types.is_numeric_dtype(y) and nun > 10:
        return "regression"
    return "classification"


def _prepare(req: MLRequest) -> _Design:
    df = _get_df(req.session_id)
    for c in [req.outcome, *req.predictors]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.predictors:
        raise HTTPException(status_code=422, detail="Select at least one predictor.")
    if req.outcome in req.predictors:
        raise HTTPException(status_code=422, detail="Outcome cannot also be a predictor.")

    strategy = (req.imputation or "listwise") or "listwise"
    if strategy in ("none", ""):
        strategy = "listwise"

    work = df[[req.outcome, *req.predictors]].copy().reset_index(drop=True)

    # 1. Rows without an observed outcome leave the analysis. They are never
    #    filled: the model would be scored against a value it invented.
    observed = work[req.outcome].notna()
    n_missing_outcome = int((~observed).sum())
    work = work[observed].reset_index(drop=True)

    work, numeric, categorical = _column_roles(work, req.predictors)

    # 2. Categorical predictors are not imputed (no mode-filling behind the
    #    user's back), so an incomplete one is complete-case deleted - the
    #    behaviour every strategy had before, now stated rather than implied.
    if categorical:
        complete_cat = work[categorical].notna().all(axis=1)
        n_incomplete_categorical = int((~complete_cat).sum())
        work = work[complete_cat].reset_index(drop=True)
    else:
        n_incomplete_categorical = 0

    # 3. Numeric predictors: dropped under listwise, kept (with their gaps) for
    #    the fitted strategies, whose imputer is refitted inside every fold.
    n_incomplete_numeric = 0
    if numeric and strategy not in _FITTED_STRATEGIES:
        complete_num = work[numeric].notna().all(axis=1)
        n_incomplete_numeric = int((~complete_num).sum())
        work = work[complete_num].reset_index(drop=True)

    # A predictor that is missing everywhere has nothing for any imputer to
    # learn from; drop it rather than fail deep inside a fold.
    for col in list(numeric):
        if work[col].isna().all():
            numeric.remove(col)
            work = work.drop(columns=[col])

    X = work[[c for c in req.predictors if c in numeric or c in categorical]]
    y_raw = work[req.outcome]

    if len(X) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough complete rows (need ≥ 20, got {len(X)}).")
    if X.shape[1] == 0:
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")

    task = _resolve_task(req, y_raw)
    y = pd.to_numeric(y_raw, errors="coerce").values if task == "regression" else y_raw.values
    if task == "regression" and np.isnan(y).any():
        raise HTTPException(status_code=422, detail="Regression outcome must be numeric.")

    return _Design(
        X=X, y=y, task=task, numeric=numeric, categorical=categorical,
        strategy=strategy,
        n_missing_outcome=n_missing_outcome,
        n_incomplete_categorical=n_incomplete_categorical,
        n_incomplete_numeric=n_incomplete_numeric,
    )


# -- Preprocessing, refitted per fold ---------------------------------------


def _numeric_imputer(strategy: str, random_state: int):
    if strategy in ("median", "mean"):
        from sklearn.impute import SimpleImputer
        return SimpleImputer(strategy=strategy)
    if strategy == "mice":
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        except ImportError:
            pass
        from sklearn.impute import IterativeImputer
        return IterativeImputer(random_state=random_state, max_iter=10, verbose=0)
    return None


def _make_pipeline(design: _Design, estimator, req: MLRequest):
    """Preprocessing + estimator as one estimator, so ``fit`` sees one fold."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    blocks = []
    if design.numeric:
        imputer = _numeric_imputer(design.strategy, req.random_state)
        num_block = (
            Pipeline([("impute", imputer)]) if imputer is not None else "passthrough"
        )
        blocks.append(("num", num_block, design.numeric))
    if design.categorical:
        # drop='first' keeps the dummy names identical to the pandas encoding
        # the models.py endpoints use; handle_unknown='ignore' is what makes a
        # level that appears only in the validation fold survivable.
        blocks.append((
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            design.categorical,
        ))

    prep = ColumnTransformer(blocks, remainder="drop")
    return Pipeline([("prep", prep), ("model", estimator)])


def _cv_permutation(pipeline, design: _Design, y: np.ndarray, cv, scoring: str, req: MLRequest):
    """One CV pass returning out-of-fold predictions and held-out importances.

    Both come from the same loop on purpose: the alternative is
    ``cross_val_predict`` plus a second round of fits for the importances,
    which doubles the work to answer the same question twice.
    """
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    X = design.X
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    per_fold = []
    for train_idx, test_idx in cv.split(X, y):
        fitted = clone(pipeline)
        fitted.fit(X.iloc[train_idx], y[train_idx])
        if scoring == "roc_auc":
            oof[test_idx] = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        else:
            oof[test_idx] = fitted.predict(X.iloc[test_idx])
        result = permutation_importance(
            fitted, X.iloc[test_idx], y[test_idx],
            n_repeats=req.n_permutation_repeats,
            random_state=req.random_state, scoring=scoring, n_jobs=1)
        per_fold.append(np.asarray(result.importances, dtype=float))
    # (n_features, n_folds x n_repeats): every held-out measurement pooled.
    return oof, np.concatenate(per_fold, axis=1)


def _impurity_by_source(pipeline, design: _Design) -> Dict[str, Optional[float]]:
    """Impurity importance of the full-data refit, summed back to source columns."""
    est = pipeline.named_steps["model"]
    vec = getattr(est, "feature_importances_", None)
    if vec is None:
        return {c: None for c in design.X.columns}
    names = pipeline.named_steps["prep"].get_feature_names_out()
    agg: Dict[str, float] = {c: 0.0 for c in design.X.columns}
    for name, value in zip(names, np.asarray(vec, dtype=float).ravel()):
        src = _source_column(str(name), design.numeric, design.categorical)
        if src in agg:
            agg[src] += float(value)
    return {k: float(v) for k, v in agg.items()}


def _importance_rows(design: _Design, perm: np.ndarray, impurity: Dict[str, Optional[float]]) -> List[dict]:
    rows = []
    for i, name in enumerate(design.X.columns):
        imp = impurity.get(name)
        rows.append({
            "feature": str(name),
            "impurity": round(float(imp), 6) if imp is not None else None,
            "permutation": round(float(perm[i].mean()), 6),
            "permutation_sd": round(float(perm[i].std(ddof=0)), 6),
        })
    rows.sort(key=lambda d: (d["permutation"] if d["permutation"] is not None else -1), reverse=True)
    return rows


def _estimator_label(est) -> str:
    name = type(est).__name__
    return {
        "RandomForestClassifier": "Random forest",
        "RandomForestRegressor": "Random forest",
        "GradientBoostingClassifier": "Gradient boosting",
        "GradientBoostingRegressor": "Gradient boosting",
    }.get(name, name)


# -- Core evaluators --------------------------------------------------------


def _eval_classifier(estimator, design: _Design, req: MLRequest) -> dict:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss

    y = design.y
    classes = np.unique(y)
    if len(classes) != 2:
        raise HTTPException(status_code=422,
            detail=f"Classification v1 supports a binary outcome (0/1). Found {len(classes)} classes: {classes.tolist()}.")
    # Map to 0/1 preserving the larger code as the positive event if it's {0,1}
    y01 = (y == classes.max()).astype(int)

    n = len(y01)
    n_pos = int(y01.sum())
    n_neg = n - n_pos
    folds = max(2, min(req.cv_folds, n_pos, n_neg))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=req.random_state)

    pipeline = _make_pipeline(design, estimator, req)
    # Out-of-fold probabilities -> honest AUC / calibration / confusion, and
    # permutation importance measured on each fold's own held-out rows.
    proba, perm = _cv_permutation(pipeline, design, y01, skf, "roc_auc", req)

    auc = float(roc_auc_score(y01, proba))
    # Bootstrap percentile CI for AUC over the OOF probabilities.
    rng = np.random.default_rng(req.random_state)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, n, n)
        yb = y01[idx]
        if yb.min() == yb.max():
            continue
        boot.append(roc_auc_score(yb, proba[idx]))
    ci_low = float(np.percentile(boot, 2.5)) if boot else None
    ci_high = float(np.percentile(boot, 97.5)) if boot else None

    fpr, tpr, _ = roc_curve(y01, proba)
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y01 == 1)).sum())
    tn = int(((pred == 0) & (y01 == 0)).sum())
    fp = int(((pred == 1) & (y01 == 0)).sum())
    fn = int(((pred == 0) & (y01 == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    ppv = tp / (tp + fp) if (tp + fp) else None
    npv = tn / (tn + fn) if (tn + fn) else None
    acc = (tp + tn) / n
    brier = float(brier_score_loss(y01, proba))

    # Calibration bins (10 quantile bins of predicted probability).
    cal = []
    try:
        bins = pd.qcut(proba, q=min(10, len(np.unique(proba))), duplicates="drop")
        cal_df = pd.DataFrame({"p": proba, "y": y01, "bin": bins})
        for _, g in cal_df.groupby("bin", observed=True):
            cal.append({"pred": round(float(g["p"].mean()), 4),
                        "obs": round(float(g["y"].mean()), 4),
                        "n": int(len(g))})
    except Exception:
        cal = []

    # Full-data refit for the descriptive impurity ranking only; every number
    # above it is out-of-fold.
    pipeline.fit(design.X, y01)
    importance = _importance_rows(design, perm, _impurity_by_source(pipeline, design))

    interp = (
        f"{_estimator_label(estimator)} classifier on n = {n} "
        f"({n_pos} events, {n_neg} non-events), {folds}-fold cross-validated. "
        f"AUC = {auc:.3f}"
        + (f" (95% CI {ci_low:.3f}–{ci_high:.3f})" if ci_low is not None else "")
        + f". Accuracy {acc*100:.1f}%, sensitivity {sens*100:.1f}%, specificity {spec*100:.1f}% "
          f"at the 0.5 cutoff. Brier score {brier:.3f}. "
          f"Top predictor by permutation importance (held-out folds): {importance[0]['feature']}."
    )

    return {
        "task": "classification",
        "n": n, "n_events": n_pos, "n_non_events": n_neg, "cv_folds": folds,
        "auc": round(auc, 4), "auc_ci_low": round(ci_low, 4) if ci_low is not None else None,
        "auc_ci_high": round(ci_high, 4) if ci_high is not None else None,
        "accuracy": round(acc, 4), "sensitivity": _safe(round(sens, 4) if sens is not None else None),
        "specificity": _safe(round(spec, 4) if spec is not None else None),
        "ppv": _safe(round(ppv, 4) if ppv is not None else None),
        "npv": _safe(round(npv, 4) if npv is not None else None),
        "brier": round(brier, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "roc_curve": _downsample_curve(fpr, tpr),
        "calibration": cal,
        "importance": importance,
        "importance_scope": "held-out folds",
        "importance_metric": "ΔAUC",
        "interpretation": interp,
    }


def _eval_regressor(estimator, design: _Design, req: MLRequest) -> dict:
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    y = design.y
    n = len(y)
    folds = max(2, min(req.cv_folds, n))
    kf = KFold(n_splits=folds, shuffle=True, random_state=req.random_state)

    pipeline = _make_pipeline(design, estimator, req)
    pred, perm = _cv_permutation(pipeline, design, y, kf, "r2", req)

    r2 = float(r2_score(y, pred))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    resid = (y - pred)

    pipeline.fit(design.X, y)
    importance = _importance_rows(design, perm, _impurity_by_source(pipeline, design))

    # Predicted-vs-actual scatter (downsample to 500 points for payload).
    if n > 500:
        rng = np.random.default_rng(req.random_state)
        sel = rng.choice(n, 500, replace=False)
    else:
        sel = np.arange(n)
    scatter = [{"actual": round(float(y[i]), 4), "predicted": round(float(pred[i]), 4)} for i in sel]

    interp = (
        f"{_estimator_label(estimator)} regressor on n = {n}, {folds}-fold "
        f"cross-validated. R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}. "
        f"Top predictor by permutation importance (held-out folds): {importance[0]['feature']}."
    )

    return {
        "task": "regression",
        "n": n, "cv_folds": folds,
        "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
        "resid_mean": round(float(resid.mean()), 4), "resid_sd": round(float(resid.std(ddof=1)), 4),
        "scatter": scatter,
        "importance": importance,
        "importance_scope": "held-out folds",
        "importance_metric": "ΔR²",
        "interpretation": interp,
    }


def _run(req: MLRequest, kind: str) -> dict:
    design = _prepare(req)
    task = design.task
    cw = "balanced" if req.class_weight_balanced else None
    md = req.max_depth if (req.max_depth and req.max_depth > 0) else None

    if kind == "random_forest":
        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier
            est = RandomForestClassifier(
                n_estimators=req.n_estimators, max_depth=md,
                min_samples_leaf=req.min_samples_leaf, class_weight=cw,
                random_state=req.random_state, n_jobs=-1, oob_score=False)
        else:
            from sklearn.ensemble import RandomForestRegressor
            est = RandomForestRegressor(
                n_estimators=req.n_estimators, max_depth=md,
                min_samples_leaf=req.min_samples_leaf,
                random_state=req.random_state, n_jobs=-1)
        model_name = "Random Forest"
    else:  # gradient_boosting
        if task == "classification":
            from sklearn.ensemble import GradientBoostingClassifier
            est = GradientBoostingClassifier(
                n_estimators=req.n_estimators, max_depth=md or 3,
                min_samples_leaf=req.min_samples_leaf,
                learning_rate=req.learning_rate, random_state=req.random_state)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            est = GradientBoostingRegressor(
                n_estimators=req.n_estimators, max_depth=md or 3,
                min_samples_leaf=req.min_samples_leaf,
                learning_rate=req.learning_rate, random_state=req.random_state)
        model_name = "Gradient Boosting"

    result = _eval_classifier(est, design, req) if task == "classification" else _eval_regressor(est, design, req)
    result["model"] = model_name
    result["outcome"] = req.outcome
    result["predictors"] = req.predictors
    result["n_features"] = int(design.X.shape[1])
    result["imputation"] = design.imputation_report

    try:
        store.log_action(req.session_id, kind, {
            "outcome": req.outcome, "n_predictors": len(req.predictors),
            "task": task, "n_estimators": req.n_estimators,
            "imputation": design.imputation_report["applied"],
        })
    except Exception:
        pass
    return result


@router.post("/random_forest")
def random_forest(req: MLRequest):
    return _run(req, "random_forest")


@router.post("/gradient_boosting")
def gradient_boosting(req: MLRequest):
    return _run(req, "gradient_boosting")


@router.post("/feature_importance")
def feature_importance(req: MLRequest):
    """Permutation importance only (no curves) for a quick screen."""
    res = _run(req, "random_forest")
    return {
        "model": res["model"], "task": res["task"], "n": res["n"],
        "outcome": res["outcome"], "importance": res["importance"],
        "importance_scope": res["importance_scope"],
        "importance_metric": res["importance_metric"],
        "imputation": res["imputation"],
        "interpretation": res["interpretation"],
    }
