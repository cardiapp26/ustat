from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services import store
from services.impute import apply_imputation
from services.regression import constant_column_warnings, design_with_constant

router = APIRouter()


class MultiOutcomeRegressionRequest(BaseModel):
    session_id: str
    outcomes: List[str]
    predictors: List[str]
    covariates: List[str] = Field(default_factory=list)
    standardize: Optional[bool] = True
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _validate_continuous_outcome(series: pd.Series) -> None:
    non_missing = series.dropna()
    numeric = pd.to_numeric(non_missing, errors="coerce")
    if numeric.isna().any():
        raise HTTPException(status_code=422, detail="Continuous outcomes only")

    vals = numeric.dropna().unique()
    if len(vals) == 0:
        return
    if set(vals.tolist()) <= {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail="Continuous outcomes only")


def _round_or_none(value: Any, digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, digits)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _label_term(term: str) -> str:
    return "(Intercept)" if term == "const" else str(term)


def _fit_outcome(
    df_full: pd.DataFrame,
    outcome: str,
    rhs_cols: List[str],
    imputation: str,
    robust_se: bool,
    standardize: bool,
) -> dict:
    df = apply_imputation(df_full, [outcome] + rhs_cols, imputation)
    X_enc = pd.get_dummies(df[rhs_cols], drop_first=True).astype(float)
    X, dropped_const = design_with_constant(X_enc)
    y = pd.to_numeric(df[outcome], errors="coerce").astype(float)

    n = int(len(df))
    k = int(X_enc.shape[1])
    if n < k + 10:
        raise HTTPException(
            status_code=400,
            detail=f"Outcome '{outcome}' has n={n}, which is too small for k={k} predictors.",
        )

    model = sm.OLS(y, X).fit(cov_type="HC3" if robust_se else "nonrobust")
    ci = model.conf_int()
    y_std = float(y.std())

    coefficients: dict[str, dict] = {}
    for term in model.params.index:
        label = _label_term(str(term))
        beta = None
        if standardize and term != "const" and y_std != 0:
            x_std = float(X[str(term)].std())
            beta = float(model.params[term]) * (x_std / y_std)

        coefficients[label] = {
            "B": _round_or_none(model.params[term]),
            "SE": _round_or_none(model.bse[term]),
            "beta": _round_or_none(beta),
            "ci": [
                _round_or_none(ci.loc[term, 0]),
                _round_or_none(ci.loc[term, 1]),
            ],
            "p": _float_or_none(model.pvalues[term]),
        }

    return {
        "n": n,
        "k": k,
        "terms": [_label_term(str(term)) for term in model.params.index],
        "coefficients": coefficients,
        "warnings": constant_column_warnings(dropped_const),
        "fit": {
            "r2": _round_or_none(model.rsquared),
            "adj_r2": _round_or_none(model.rsquared_adj),
            "f": _round_or_none(model.fvalue),
            "f_p": _float_or_none(model.f_pvalue),
            "n": n,
            "k": k,
        },
    }


def _result_text(
    outcomes: List[str],
    n_by_outcome: dict[str, int],
    model_fit: dict[str, dict],
    rows: List[dict],
) -> str:
    parts = []
    for outcome in outcomes:
        n = n_by_outcome[outcome]
        k = model_fit[outcome]["k"]
        significant = [
            row["predictor"]
            for row in rows
            if row["predictor"] != "(Intercept)"
            and row["by_outcome"].get(outcome)
            and row["by_outcome"][outcome].get("p") is not None
            and row["by_outcome"][outcome]["p"] < 0.05
        ]
        if significant:
            sig_text = ", ".join(significant)
        else:
            sig_text = "no predictors"
        parts.append(
            f"For {outcome}, the model used n={n} observations and k={k} predictors; "
            f"significant predictors were {sig_text}."
        )
    return " ".join(parts)


@router.post("/multi_outcome_regression")
def multi_outcome_regression(req: MultiOutcomeRegressionRequest):
    if not req.outcomes:
        raise HTTPException(status_code=400, detail="At least one outcome is required.")
    if not req.predictors:
        raise HTTPException(status_code=400, detail="At least one predictor is required.")

    df_full = _get_df(req.session_id)
    imputation = req.imputation or "listwise"
    if imputation == "mice":
        raise HTTPException(
            status_code=400,
            detail="MICE pooling not supported in multi-outcome endpoint yet",
        )

    rhs_cols = list(req.predictors) + list(req.covariates or [])
    missing = [c for c in list(req.outcomes) + rhs_cols if c not in df_full.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown column(s): {', '.join(missing)}")

    for outcome in req.outcomes:
        _validate_continuous_outcome(df_full[outcome])

    fitted: dict[str, dict] = {}
    predictors_order: List[str] = []
    for outcome in req.outcomes:
        res = _fit_outcome(
            df_full=df_full,
            outcome=outcome,
            rhs_cols=rhs_cols,
            imputation=imputation,
            robust_se=bool(req.robust_se),
            standardize=bool(req.standardize),
        )
        fitted[outcome] = res
        for term in res["terms"]:
            if term not in predictors_order:
                predictors_order.append(term)

    rows = []
    for term in predictors_order:
        rows.append({
            "predictor": term,
            "by_outcome": {
                outcome: fitted[outcome]["coefficients"].get(term)
                for outcome in req.outcomes
            },
        })

    n_by_outcome = {outcome: fitted[outcome]["n"] for outcome in req.outcomes}
    model_fit = {outcome: fitted[outcome]["fit"] for outcome in req.outcomes}

    return {
        "test": "Multi-outcome linear regression",
        "outcomes": req.outcomes,
        "predictors_order": predictors_order,
        "n_by_outcome": n_by_outcome,
        "rows": rows,
        "model_fit": model_fit,
        "result_text": _result_text(req.outcomes, n_by_outcome, model_fit, rows),
    }
