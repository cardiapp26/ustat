"""Small shared utilities for regression-related code."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_vif(X: pd.DataFrame) -> dict:
    """Variance Inflation Factor per predictor, as R's car::vif defines it.

    The intercept has to be IN the matrix the auxiliary regressions are run
    on, and out of the results. Dropping the constant column first — which is
    what this used to do, and what four copies of it in the model routers
    also did — makes every auxiliary regression pass through the origin, so
    its R-squared absorbs the mean structure and the VIF comes out inflated.

    On the audit dataset that reported age at 21.17 and bmi at 21.69 where
    car::vif gives 1.008 and 1.012. Ten is the conventional "severe
    collinearity, drop this predictor" threshold, so the numbers were telling
    users to throw away sound predictors on data with no collinearity at all.

    Returns {column_name: vif_float}, intercept excluded from the output.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xn = X.copy().astype(float)
    if "const" not in Xn.columns:
        Xn.insert(0, "const", 1.0)
    predictors = [c for c in Xn.columns if c != "const"]
    if len(predictors) < 2:
        return {c: 1.0 for c in predictors}

    arr = Xn.values
    cols = list(Xn.columns)
    out: dict = {}
    for col in predictors:
        try:
            v = float(variance_inflation_factor(arr, cols.index(col)))
            if not np.isfinite(v):
                v = None
        except Exception:
            v = None
        out[str(col)] = v
    return out


# ── Stepwise selection helpers (pure functions) ────────────────────────────────

def _p_for_pred(pred: str, pvalues) -> float:
    if pred in pvalues.index:
        return float(pvalues[pred])
    dummy_cols = [c for c in pvalues.index if c != "const" and c.startswith(pred + "_")]
    if dummy_cols:
        return float(pvalues[dummy_cols].max())
    return 1.0


def _uni_p_for_pred(pred: str, uni_results: dict) -> float:
    if pred in uni_results:
        return uni_results[pred]["p"]
    matching = [v["p"] for k, v in uni_results.items() if k.startswith(pred + "_")]
    return min(matching) if matching else 1.0


def _compute_aic(model) -> float:
    try:
        return float(model.aic)
    except Exception:
        return float("nan")


def stepwise_forward(y, df: pd.DataFrame, pred_list: list, p_enter: float = 0.05) -> list:
    """Pure functional forward stepwise selection. Does not mutate inputs."""
    from statsmodels import api as sm
    selected: list = []
    remaining = list(pred_list)

    while remaining:
        best_var, best_p = None, p_enter
        for var in remaining:
            candidate = selected + [var]
            X_enc = pd.get_dummies(df[candidate], drop_first=True).astype(float)
            X_const = sm.add_constant(X_enc, has_constant="add")
            try:
                m = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
                p = _p_for_pred(var, m.pvalues)
                if p < best_p:
                    best_p, best_var = p, var
            except Exception:
                pass
        if best_var is None:
            break
        selected = selected + [best_var]
        remaining = [v for v in remaining if v != best_var]
    return selected


def stepwise_backward(y, df: pd.DataFrame, pred_list: list, p_remove: float = 0.10) -> list:
    """Pure functional backward stepwise selection. Does not mutate inputs."""
    from statsmodels import api as sm
    selected = list(pred_list)

    while selected:
        X_enc = pd.get_dummies(df[selected], drop_first=True).astype(float)
        X_const = sm.add_constant(X_enc, has_constant="add")
        try:
            m = sm.Logit(y, X_const).fit(disp=False, maxiter=200)
        except Exception:
            break
        worst_var, worst_p = None, p_remove
        for var in selected:
            p = _p_for_pred(var, m.pvalues)
            if p > worst_p:
                worst_p, worst_var = p, var
        if worst_var is None:
            break
        selected = [v for v in selected if v != worst_var]
    return selected
