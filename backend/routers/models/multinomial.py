"""Multinomial (baseline-category) logistic regression.

For a nominal outcome with three or more unordered categories (cause of
death, stent type, AF pattern) or an ordinal one whose proportional-odds
assumption fails. One equation per non-reference category, each comparing
that category with the reference: exp(beta) is a relative risk ratio (RRR).

Fitted with statsmodels MNLogit; matches R's nnet::multinom on coefficients,
SEs and the per-predictor likelihood-ratio tests (see
backend/tests/test_multinomial_vs_r.py).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel
from scipy import stats as scipy_stats

from services.impute import apply_imputation
from services.level_order import level_key, resolve_level_order
from services.regression import constant_column_warnings, drop_constant_columns

from .glm import _clean_predictor_categories, _get_df, _sanitize, _sanitize_model_error

router = APIRouter()

MAX_CATEGORIES = 10
# Below this many cases per estimated parameter in an equation the Wald
# intervals of a multinomial fit become unreliable (Peduzzi-style rule).
MIN_EVENTS_PER_PARAMETER = 10
# A coefficient this large on the log scale, or an SE this large, means the
# likelihood has no finite maximum in that direction (separation).
SEPARATION_COEF = 15.0
SEPARATION_SE = 1e3


class MultinomialRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    # The baseline category every other one is compared with. Defaults to the
    # first in Data Dictionary / numeric / alphabetical order.
    reference: Optional[str] = None
    imputation: Optional[str] = "listwise"


def _outcome_levels(y_raw: pd.Series, outcome: str, session_id: str) -> tuple[list[str], str]:
    """Level keys in a stable order, and where that order came from."""
    order = resolve_level_order(y_raw, outcome, session_id=session_id)
    if order is not None:
        return list(order.keys), order.source
    keys = sorted({level_key(v) for v in y_raw.dropna()})
    return keys, "alphabetical"


def _design(df: pd.DataFrame, predictors: List[str]) -> tuple[pd.DataFrame, list]:
    X = pd.get_dummies(df[predictors], drop_first=True).astype(float)
    X, dropped = drop_constant_columns(X)
    return X, dropped


def _columns_of(predictor: str, X: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """The design columns a predictor expands into (itself, or its dummies)."""
    if predictor in X.columns and pd.api.types.is_numeric_dtype(df[predictor]):
        return [predictor]
    prefix = f"{predictor}_"
    return [c for c in X.columns if c == predictor or c.startswith(prefix)]


def _fit(y: np.ndarray, X: pd.DataFrame):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.MNLogit(y, Xc)
    try:
        return model.fit(method="newton", maxiter=200, disp=False)
    except np.linalg.LinAlgError:
        return model.fit(method="bfgs", maxiter=2000, disp=False)


def _lr_tests(y, X, df, predictors, full_llf, n_eq) -> list[dict]:
    """Likelihood-ratio test of each predictor: drop all its columns from
    every equation. Equivalent to car::Anova(type = 2) for main-effects
    models and to anova() of nested nnet::multinom fits."""
    out = []
    for p in predictors:
        cols = _columns_of(p, X, df)
        if not cols:
            continue
        reduced = X.drop(columns=cols)
        try:
            llf_r = float(_fit(y, reduced).llf)
        except Exception:
            logger.exception("Multinomial reduced fit failed for {}", p)
            continue
        lr = max(0.0, 2.0 * (full_llf - llf_r))
        dof = len(cols) * n_eq
        out.append({
            "variable": p,
            "lr_chi2": lr,
            "df": dof,
            "p": float(scipy_stats.chi2.sf(lr, dof)),
        })
    return out


def _results_text(outcome, levels, reference, n, lr_model, lr_df, lr_p, r2, lr_terms) -> str:
    others = [lv for lv in levels if lv != reference]
    p_txt = "p < 0.001" if lr_p < 0.001 else f"p = {lr_p:.3f}"
    text = (
        f"A multinomial logistic regression of {outcome} ({len(levels)} categories; "
        f"reference category '{reference}', compared with {', '.join(repr(o) for o in others)}) "
        f"was fitted to {n} observations. Against the intercept-only model, "
        f"likelihood-ratio χ²({lr_df}) = {lr_model:.2f}, {p_txt}; McFadden R² = {r2:.3f}."
    )
    sig = [t for t in lr_terms if t["p"] < 0.05]
    if sig:
        parts = [
            f"{t['variable']} (χ²({t['df']}) = {t['lr_chi2']:.2f}, "
            + ("p < 0.001" if t["p"] < 0.001 else f"p = {t['p']:.3f}") + ")"
            for t in sig
        ]
        text += " Predictors associated with the outcome across its categories: " + "; ".join(parts) + "."
    else:
        text += " No predictor reached p < 0.05 in its likelihood-ratio test."
    return text


@router.post("/multinomial")
def multinomial_regression(req: MultinomialRequest):
    """Baseline-category multinomial logistic regression.

    Returns, for every non-reference category, each predictor's log-RRR, SE,
    Wald z and p, and RRR with 95% CI; a likelihood-ratio test per predictor
    across all equations; the model LR test, McFadden R², AIC/BIC; and the
    classification table of predicted against observed categories.
    """
    if not req.predictors:
        raise HTTPException(status_code=422, detail="Choose at least one predictor.")
    if req.outcome in req.predictors:
        raise HTTPException(status_code=422, detail="The outcome cannot also be a predictor.")

    df_full = _get_df(req.session_id)
    missing = [c for c in [req.outcome, *req.predictors] if c not in df_full.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Column(s) not found: {missing}")
    cols = [req.outcome] + req.predictors
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    df = df.dropna(subset=[req.outcome])
    df, cat_warnings = _clean_predictor_categories(df, req.predictors)
    n_excluded = len(df_full) - len(df)

    y_raw = df[req.outcome]
    levels, order_source = _outcome_levels(y_raw, req.outcome, req.session_id)
    if len(levels) < 3:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{req.outcome}' has {len(levels)} categories. A multinomial model needs "
                "three or more; for two, use logistic regression."
            ),
        )
    if len(levels) > MAX_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{req.outcome}' has {len(levels)} categories. Above {MAX_CATEGORIES} the "
                "outcome is unlikely to be nominal; check it is not a continuous or ID column."
            ),
        )

    reference = levels[0]
    if req.reference is not None:
        ref_key = level_key(req.reference)
        if ref_key not in levels:
            raise HTTPException(
                status_code=422,
                detail=f"Reference '{req.reference}' is not a category of '{req.outcome}': {levels}",
            )
        reference = ref_key
    ordered = [reference] + [lv for lv in levels if lv != reference]

    y = pd.Categorical(y_raw.map(level_key), categories=ordered).codes
    X, dropped_const = _design(df, req.predictors)
    if X.shape[1] == 0:
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")
    X = X.reset_index(drop=True)
    y = np.asarray(y)

    try:
        fit = _fit(y, X)
    except Exception as e:
        logger.exception("Multinomial fit failed")
        raise HTTPException(status_code=422, detail=_sanitize_model_error(e, "multinomial logistic"))

    if not np.isfinite(np.asarray(fit.params, dtype=float)).any():
        raise HTTPException(
            status_code=422,
            detail=(
                "The multinomial model has no finite estimates: the predictors separate the "
                "outcome categories completely. Merge sparse categories or drop the separating predictor."
            ),
        )

    n = int(len(y))
    n_eq = len(ordered) - 1
    counts = {lv: int((y == i).sum()) for i, lv in enumerate(ordered)}
    params, bse, pvals = fit.params, fit.bse, fit.pvalues
    conf = fit.conf_int()
    terms = list(params.index)

    warnings = list(cat_warnings) + constant_column_warnings(dropped_const)
    equations = []
    separation = []
    for j, category in enumerate(ordered[1:]):
        lo_hi = conf.loc[str(j + 1)] if str(j + 1) in conf.index.get_level_values(0) else conf.xs(j + 1, level=0)
        rows = []
        for term in terms:
            b = float(params.iloc[terms.index(term), j])
            se = float(bse.iloc[terms.index(term), j])
            lo, hi = float(lo_hi.loc[term].iloc[0]), float(lo_hi.loc[term].iloc[1])
            if term != "const" and not (math.isfinite(b) and math.isfinite(se) and abs(b) <= SEPARATION_COEF and se <= SEPARATION_SE):
                separation.append(f"{term} in '{category}' vs '{reference}'")
            rows.append({
                "variable": "Intercept" if term == "const" else str(term),
                "log_rrr": b,
                "se": se,
                "z": b / se if se and math.isfinite(se) else None,
                "p": float(pvals.iloc[terms.index(term), j]),
                "rrr": math.exp(b) if math.isfinite(b) and b < 700 else None,
                "rrr_ci_low": math.exp(lo) if math.isfinite(lo) and lo < 700 else None,
                "rrr_ci_high": math.exp(hi) if math.isfinite(hi) and hi < 700 else None,
            })
        equations.append({"category": category, "vs": reference, "n": counts[category], "coefficients": rows})

    if separation:
        warnings.append(
            "Possible complete or quasi-complete separation (very large coefficient or SE): "
            + "; ".join(separation[:5])
            + ". Those RRRs and intervals are not trustworthy; merge sparse categories or drop the predictor."
        )
    n_params = X.shape[1] + 1
    thin = [lv for lv, c in counts.items() if c / n_params < MIN_EVENTS_PER_PARAMETER]
    if thin:
        warnings.append(
            f"Few cases per estimated parameter ({n_params} per equation) in categor"
            f"{'y' if len(thin) == 1 else 'ies'} {', '.join(repr(t) for t in thin)}; "
            "intervals may be unreliable."
        )
    if not bool(fit.mle_retvals.get("converged", True)):
        warnings.append("The fit did not report convergence; treat the estimates with caution.")

    lr_terms = _lr_tests(y, X, df.reset_index(drop=True), req.predictors, float(fit.llf), n_eq)

    predicted = np.asarray(fit.predict()).argmax(axis=1)
    table = [[int(((y == i) & (predicted == k)).sum()) for k in range(len(ordered))] for i in range(len(ordered))]
    accuracy = float((predicted == y).mean())

    lr_model, lr_df, lr_p = float(fit.llr), int(fit.df_model), float(fit.llr_pvalue)
    r2 = float(fit.prsquared)
    res = {
        "model": "Multinomial Logistic Regression",
        "outcome": req.outcome,
        "categories": ordered,
        "reference": reference,
        "level_order_source": order_source,
        "category_counts": counts,
        "n": n,
        "n_excluded": int(n_excluded),
        "equations": equations,
        "lr_tests": lr_terms,
        "model_lr_chi2": lr_model,
        "model_lr_df": lr_df,
        "model_lr_p": lr_p,
        "pseudo_r2": r2,
        "log_likelihood": float(fit.llf),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "classification_table": {"labels": ordered, "table": table, "accuracy": accuracy},
        "warnings": warnings,
    }
    res["result_text"] = _results_text(
        req.outcome, ordered, reference, n, lr_model, lr_df, lr_p, r2, lr_terms,
    )
    return _sanitize(res)
