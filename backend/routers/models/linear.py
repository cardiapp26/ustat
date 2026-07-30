from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from services import store
from services.category_health import clean_two_level, rare_level_warnings
from services.regression import constant_column_warnings, design_with_constant
from services.impute import apply_imputation
from services.missing_data import (
    mice_multiple,
    missing_pattern_summary,
    pool_linear_results,
    add_missing_data_diagnostics,
)
from services.regression import (
    stepwise_forward as _stepwise_forward,
    stepwise_backward as _stepwise_backward,
    _compute_aic,
)
from services.assumptions import (
    check_linear_assumptions,
    add_assumption_warnings_to_result,
)

router = APIRouter()

# ── Helpers (shared across regression endpoints) ───────────────────────────────

def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _sanitize_model_error(err: Exception, context: str = "model fitting") -> str:
    """Prevent leaking internal library details to the client."""
    msg = str(err)
    if "Singular" in msg or "perfect separation" in msg.lower():
        return "The model encountered perfect separation or singular matrix. Try removing highly correlated predictors."
    if "convergence" in msg.lower() or "failed to converge" in msg.lower():
        return f"{context.capitalize()} failed to converge. Consider increasing iterations or simplifying the model."
    return f"{context.capitalize()} failed. Please check your data and predictors."


def _compute_vif(X: pd.DataFrame) -> dict:
    """Delegates to the one correct implementation.

    This file used to carry its own copy, which dropped the intercept before
    calling statsmodels and therefore reported inflated VIFs — four routers
    each held the same copy of the same mistake, while /api/diagnostics
    computed it correctly, so the same quantity had two answers depending on
    which screen you were on.
    """
    from services.regression import compute_vif

    return compute_vif(X)


def information_criteria(model, *, n_params: int, n_obs: int) -> tuple[float, float]:
    """AIC and BIC counting every estimated parameter, scale included.

    statsmodels' OLS and Gamma-family GLM results leave the residual variance
    (or the dispersion) out of the parameter count, so their AIC sat exactly
    2 below R's and their BIC log(n) below it — on the audit frame 2155.81
    against R's 2157.81, and 2174.33 against 2180.04. A criterion is only
    meaningful next to another criterion, so a fixed offset is not harmless:
    it makes this model look better than one someone computed elsewhere.

    Families with a fixed scale (binomial, Poisson) already agree with R and
    must not be adjusted, so the caller passes the count it wants.
    """
    llf = float(model.llf)
    return (-2.0 * llf + 2.0 * n_params,
            -2.0 * llf + float(np.log(n_obs)) * n_params)


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


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


def _add_pairwise_interactions(
    enc: pd.DataFrame,
    interactions: Optional[List[List[str]]],
    requested_predictors: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Append pairwise interaction columns to a design matrix."""
    if not interactions:
        return enc, []

    out = enc.copy()
    added: List[str] = []

    def _members(name: str) -> List[str]:
        if name in out.columns:
            return [name]
        prefix = f"{name}_"
        return [c for c in out.columns if c.startswith(prefix)]

    requested_set = set(requested_predictors)
    for pair in interactions:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise HTTPException(status_code=422, detail=f"Each interaction must be a [colA, colB] pair. Got: {pair}")
        a_name, b_name = pair
        for nm in (a_name, b_name):
            if nm not in requested_set:
                raise HTTPException(
                    status_code=422,
                    detail=f"Interaction '{a_name} × {b_name}': '{nm}' must already be in the predictors list."
                )
        a_members = _members(a_name)
        b_members = _members(b_name)
        if not a_members or not b_members:
            raise HTTPException(
                status_code=422,
                detail=f"Interaction '{a_name} × {b_name}': one or both columns did not survive dummy encoding."
            )
        for a in a_members:
            for b in b_members:
                col = f"{a}:{b}"
                if col in out.columns:
                    continue
                out[col] = out[a] * out[b]
                added.append(col)
    return out, added


# ── OLS Linear Regression ──────────────────────────────────────────────────────────

class LinearRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False
    interactions: Optional[List[List[str]]] = None


@router.post("/linear")
def linear_regression(req: LinearRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    imputation_method = req.imputation or "listwise"

    use_mice_pooled = imputation_method == "mice"

    if use_mice_pooled:
        imp_result = mice_multiple(
            df_full, [req.outcome] + req.predictors, n_imputations=5
        )
        imputed_dfs = imp_result.imputed_datasets

        individual_results = []
        for df_imp in imputed_dfs:
            X_enc = pd.get_dummies(df_imp[req.predictors], drop_first=True).astype(float)
            X_enc, _ = _add_pairwise_interactions(X_enc, req.interactions, req.predictors)
            X, _dropped_const = design_with_constant(X_enc)
            y_imp = df_imp[req.outcome].astype(float)
            # use_t=True on every branch. statsmodels keeps the t on residual
            # df for an ordinary OLS fit but silently switches to a normal the
            # moment cov_type is set, so asking for robust standard errors
            # also changed the reference distribution. HC3 is itself a
            # small-sample correction — pairing it with the large-sample
            # distribution takes the correction back out. At n = 30 the
            # robust p for age read 0.016 where R's coeftest gives 0.025.
            m = sm.OLS(y_imp, X).fit(cov_type="HC3" if req.robust_se else "nonrobust", use_t=True)
            individual_results.append({
                "coefficients": [
                    {"variable": str(var), "estimate": float(m.params[var]), "se": float(m.bse[var])}
                    for var in m.params.index
                ],
                "r_squared": float(m.rsquared),
            })

        pooled = pool_linear_results(individual_results)

        df = imputed_dfs[0]
        n_excluded = n_total - len(df)
        X_enc = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
        X_enc, ix_added = _add_pairwise_interactions(X_enc, req.interactions, req.predictors)
        X, dropped_const = design_with_constant(X_enc)
        y = df[req.outcome].astype(float)
        model = sm.OLS(y, X).fit(cov_type="HC3" if req.robust_se else "nonrobust", use_t=True)
    else:
        df = apply_imputation(df_full, [req.outcome] + req.predictors, imputation_method)
        n_excluded = n_total - len(df)
        X_enc = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
        X_enc, ix_added = _add_pairwise_interactions(X_enc, req.interactions, req.predictors)
        X, dropped_const = design_with_constant(X_enc)
        y = df[req.outcome].astype(float)
        model = sm.OLS(y, X).fit(cov_type="HC3" if req.robust_se else "nonrobust", use_t=True)

    vifs = _compute_vif(X)

    r_squared_val = None
    pooled_from_imputations = False

    if use_mice_pooled and 'pooled' in locals():
        coefs = pooled.get("coefficients", [])
        for c in coefs:
            c["vif"] = vifs.get(c["variable"])
        r_squared_val = pooled.get("r_squared")
        pooled_from_imputations = True
    else:
        ci = model.conf_int()
        coefs = []
        for var in model.params.index:
            coefs.append({
                "variable": str(var),
                "estimate": float(model.params[var]),
                "se": float(model.bse[var]),
                "t": float(model.tvalues[var]),
                "p": float(model.pvalues[var]),
                "ci_low": float(ci.loc[var, 0]),
                "ci_high": float(ci.loc[var, 1]),
                "vif": vifs.get(str(var)),
            })

    predictor_info: dict = {}
    for col in req.predictors:
        if col not in df_full.columns:
            continue
        s = df_full[col].dropna()
        if len(s) == 0:
            continue
        if pd.api.types.is_numeric_dtype(s):
            predictor_info[col] = {
                "type": "numeric",
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "median": float(s.median()),
            }
        else:
            vc = s.value_counts()
            predictor_info[col] = {
                "type": "categorical",
                "categories": vc.index.astype(str).tolist(),
                "counts": [int(v) for v in vc.values],
            }

    residuals = model.resid.values
    fitted = model.fittedvalues.values

    assumption_report = check_linear_assumptions(
        residuals=residuals,
        fitted_values=fitted,
        X=X_enc,
        y=y,
        model=model,
    )

    # OLS leaves the residual variance out of its parameter count.
    _ic = information_criteria(
        model, n_params=int(model.df_model) + 2, n_obs=int(model.nobs))
    result = {
        "model": f"Linear Regression (OLS){' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "r_squared": r_squared_val if r_squared_val is not None else float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_p": float(model.f_pvalue),
        "aic": _ic[0],
        "bic": _ic[1],
        "coefficients": coefs,
        "residual_se": float(np.sqrt(model.mse_resid)),
        "df_resid": int(model.df_resid),
        "predictors": req.predictors,
        "predictor_info": predictor_info,
        "result_text": _linear_results_text(req.outcome, coefs, model),
    }

    if pooled_from_imputations:
        result["pooled_from_imputations"] = True

    result = add_assumption_warnings_to_result(result, assumption_report)

    if dropped_const:
        result["warnings"] = (result.get("warnings") or []) + constant_column_warnings(dropped_const)

    missing_info = missing_pattern_summary(df_full, [req.outcome] + req.predictors)
    result = add_missing_data_diagnostics(result, missing_info)

    return result


def _linear_results_text(outcome, coefs, model):
    sig = [c for c in coefs if c["variable"] != "const" and c["p"] < 0.05]
    f_p = "<0.001" if model.f_pvalue < 0.001 else f"{model.f_pvalue:.3f}"
    parts = [
        f"Multiple linear regression was performed to predict {outcome}. "
        f"The overall model was {'statistically significant' if model.f_pvalue < 0.05 else 'not significant'} "
        f"(F({int(model.df_model)},{int(model.df_resid)}) = {model.fvalue:.3f}, p = {f_p}), "
        f"explaining {model.rsquared*100:.1f}% of the variance (R² = {model.rsquared:.3f}, adjusted R² = {model.rsquared_adj:.3f})."
    ]
    if sig:
        preds = []
        for c in sig:
            p_s = "<0.001" if c["p"] < 0.001 else f'{c["p"]:.3f}'
            preds.append(f'{c["variable"]} (B = {c["estimate"]:.3f}, SE = {c["se"]:.3f}, p = {p_s})')
        parts.append("Significant predictors: " + "; ".join(preds) + ".")
    return " ".join(parts)


# ── Stepwise Selection Endpoint ───────────────────────────────────────────

class StepwiseRequest(BaseModel):
    session_id: str
    model_type: str = "logistic"   # "logistic" | "linear"
    outcome: str
    candidates: List[str]
    direction: str = "both"        # "forward" | "backward" | "both"
    criterion: str = "p"           # "p" or "aic"
    p_enter: float = 0.05
    p_remove: float = 0.10
    imputation: Optional[str] = "listwise"


@router.post("/stepwise")
def stepwise_selection(req: StepwiseRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    cols_needed = [req.outcome] + req.candidates
    df = apply_imputation(df_full, cols_needed, req.imputation or "listwise")
    df, cat_warnings = _clean_predictor_categories(df, req.candidates)
    n_excluded = n_total - len(df)

    if len(df) < 10:
        raise HTTPException(status_code=422, detail="Too few rows after imputation for stepwise selection.")

    y = df[req.outcome]
    if req.model_type == "logistic":
        if y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            y = pd.to_numeric(y, errors="coerce")
        unique_vals = sorted(pd.Series(y).dropna().unique())
        if set(unique_vals) - {0, 1, 0.0, 1.0}:
            raise HTTPException(status_code=422, detail=f"Logistic stepwise requires binary 0/1 outcome. Found: {unique_vals[:8]}")
        y = pd.Series(y).astype(float)
    else:
        y = pd.to_numeric(y, errors="coerce")

    Xdf = df[req.candidates].copy()

    pred_list = [c for c in req.candidates if c in Xdf.columns]

    trace: list = []
    selected: list = []

    mt = req.model_type.lower()
    direction = req.direction.lower()

    def _fit_and_aic(vars_in: list):
        if not vars_in:
            return None, float("inf")
        X_enc = pd.get_dummies(Xdf[vars_in], drop_first=True).astype(float)
        Xc = sm.add_constant(X_enc, has_constant="add")
        try:
            if mt == "logistic":
                m = sm.Logit(y, Xc).fit(disp=False, maxiter=200)
            else:
                m = sm.OLS(y, Xc).fit()
            return m, _compute_aic(m)
        except Exception:
            logger.exception("Stepwise model fit failed")
            return None, float("inf")

    if direction in ("forward", "both"):
        sel = _stepwise_forward(y, Xdf, pred_list, p_enter=req.p_enter)
        selected = sel
        m, aic = _fit_and_aic(selected)
        trace.append({"step": 1, "action": "forward", "selected": list(selected), "aic": aic})

    if direction in ("backward", "both"):
        if not selected:
            selected = list(pred_list)
        sel = _stepwise_backward(y, Xdf, selected, p_remove=req.p_remove)
        selected = sel
        m, aic = _fit_and_aic(selected)
        trace.append({"step": len(trace)+1, "action": "backward", "selected": list(selected), "aic": aic})

    if direction == "both":
        for i in range(2):
            before = list(selected)
            forward_candidates = [v for v in pred_list if v not in selected]
            new_forward = _stepwise_forward(y, Xdf, forward_candidates, p_enter=req.p_enter)
            if new_forward:
                selected = list(dict.fromkeys(selected + new_forward))
            selected = _stepwise_backward(y, Xdf, selected, p_remove=req.p_remove)
            if selected == before:
                break
            m, aic = _fit_and_aic(selected)
            trace.append({"step": len(trace)+1, "action": "refine_both", "selected": list(selected), "aic": aic})

    final_model, final_aic = _fit_and_aic(selected)
    if not selected:
        return _sanitize({
            "model_type": mt,
            "direction": direction,
            "criterion": req.criterion,
            "selected": [],
            "n_selected": 0,
            "final_aic": None,
            "n_obs": int(len(df)),
            "n_excluded": int(n_excluded),
            "warnings": cat_warnings + ["No predictors met the stepwise entry/removal criteria."],
            "trace": trace,
            "result_text": _stepwise_results_text(mt, [], float("nan"), direction),
        })
    if final_model is None:
        raise HTTPException(status_code=422, detail=_sanitize_model_error(Exception("convergence failed"), "stepwise selection"))

    return _sanitize({
        "model_type": mt,
        "direction": direction,
        "criterion": req.criterion,
        "selected": selected,
        "n_selected": len(selected),
        "final_aic": round(final_aic, 2) if np.isfinite(final_aic) else None,
        "n_obs": int(len(df)),
        "n_excluded": int(n_excluded),
        "warnings": cat_warnings,
        "trace": trace,
        "result_text": _stepwise_results_text(mt, selected, final_aic, direction),
    })


def _stepwise_results_text(model_type, selected, final_aic, direction):
    n = len(selected)
    aic_str = f"{final_aic:.1f}" if final_aic and np.isfinite(final_aic) else "N/A"
    return (
        f"Stepwise {direction} selection ({model_type}) retained {n} predictor(s) "
        f"with final AIC = {aic_str}."
    )


# ── Polynomial Regression ───────────────────────────────────────────────────

class PolynomialRequest(BaseModel):
    session_id: str
    outcome: str
    predictor: str
    degree: int = 2
    covariates: List[str] = []
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/polynomial")
def polynomial_regression(req: PolynomialRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    cols = [req.outcome, req.predictor] + req.covariates
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    if req.degree < 1 or req.degree > 10:
        raise HTTPException(status_code=422, detail="Polynomial degree must be between 1 and 10.")

    x = df[req.predictor].astype(float)
    n_unique = x.nunique()
    if n_unique <= req.degree:
        raise HTTPException(status_code=422, detail=f"Predictor has only {n_unique} unique values — need more than degree ({req.degree}) for polynomial fit.")

    X_parts = {"const": np.ones(len(df))}
    for d in range(1, req.degree + 1):
        X_parts[f"{req.predictor}^{d}"] = x ** d
    X = pd.DataFrame(X_parts, index=df.index)
    if req.covariates:
        # Covariates were cast straight to float, so a categorical one — the
        # ordinary case of adjusting a dose-response curve for treatment arm —
        # raised ValueError and came back as an unhandled 500. Every other
        # model endpoint dummy-codes its predictors; this one did not.
        covs = pd.get_dummies(df[req.covariates], drop_first=True)
        non_numeric = [
            c for c in covs.columns
            if not pd.api.types.is_numeric_dtype(covs[c])
            and not pd.api.types.is_bool_dtype(covs[c])
        ]
        if non_numeric:
            raise HTTPException(
                status_code=422,
                detail=(
                    "These covariates could not be used as numbers or as "
                    f"categories: {', '.join(non_numeric)}."
                ),
            )
        X = pd.concat([X, covs.astype(float)], axis=1)
    y = df[req.outcome].astype(float)

    base = sm.OLS(y, X)
    model = base.fit(cov_type="HC3" if req.robust_se else "nonrobust", use_t=True) if req.robust_se else base.fit()
    ci = model.conf_int()

    coefs = []
    for var in model.params.index:
        coefs.append({
            "variable": str(var),
            "estimate": float(model.params[var]),
            "se": float(model.bse[var]),
            "t": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
        })

    x_lo, x_hi = float(x.min()), float(x.max())
    xs = np.linspace(x_lo, x_hi, 200)
    X_curve = np.column_stack([xs ** d for d in range(0, req.degree + 1)])
    # Hold every covariate at its mean along the curve. Averaging the ENCODED
    # columns, not the raw ones — the raw column may be a category, and taking
    # its mean raised TypeError and came back as a 500. For a dummy the mean
    # is the level's prevalence, which is the usual "average subject" curve.
    cov_cols = [c for c in X.columns if c not in
                {"const", *(f"{req.predictor}^{d}" for d in range(1, req.degree + 1))}]
    if cov_cols:
        cov_means = [float(X[c].mean()) for c in cov_cols]
        X_curve = np.hstack([X_curve, np.tile(cov_means, (len(xs), 1))])
    pred = model.get_prediction(X_curve)
    yhat = pred.predicted_mean
    ci_df = pred.conf_int()

    # OLS leaves the residual variance out of its parameter count.
    _ic = information_criteria(
        model, n_params=int(model.df_model) + 2, n_obs=int(model.nobs))
    return {
        "model": f"Polynomial Regression (degree {req.degree}){' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "predictor": req.predictor,
        "degree": req.degree,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": _ic[0],
        "bic": _ic[1],
        "residual_se": float(np.sqrt(model.mse_resid)),
        "coefficients": coefs,
        "curve": {
            "x": xs.tolist(),
            "y": yhat.tolist(),
            "ci_low":  ci_df[:, 0].tolist(),
            "ci_high": ci_df[:, 1].tolist(),
        },
        "scatter": {
            "x": x.tolist()[:2000],
            "y": y.tolist()[:2000],
        },
    }


# ── Linear Mixed Model (LMM) ──────────────────────────────────────────────

class LMMRequest(BaseModel):
    session_id: str
    outcome: str
    fixed_effects: List[str]
    group_col: str
    imputation: Optional[str] = "listwise"


def _is_id_like(col: str, series: pd.Series) -> bool:
    """Heuristic: column is likely a patient/subject identifier."""
    name_lower = col.lower()
    name_match = any(name_lower == tok or name_lower.endswith(tok) or name_lower.startswith(tok)
                     for tok in ("id", "no", "num", "number", "patient", "subject", "case", "record"))
    if name_match:
        return True
    n = len(series.dropna())
    if n < 5:
        return False
    try:
        nunique = series.nunique()
        return (nunique / n) > 0.95 and pd.api.types.is_integer_dtype(series)
    except Exception:
        return False


@router.post("/lmm")
def linear_mixed_model(req: LMMRequest):
    import re
    import statsmodels.formula.api as smf

    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    id_in_fe = [c for c in req.fixed_effects if _is_id_like(c, df_full.get(c, pd.Series()))]
    if id_in_fe:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Column(s) {id_in_fe} look like patient/subject identifiers and cannot be fixed effects. "
                "Assign them as the Grouping variable (random intercept) instead."
            ),
        )

    cols = [req.outcome, req.group_col] + req.fixed_effects
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    outcome_vals = df[req.outcome].dropna().unique()
    is_binary = set(outcome_vals.tolist()) <= {0, 1, 0.0, 1.0}

    def safe(c: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_]", "_", c)

    rename = {c: safe(c) for c in cols}
    df_r = df.rename(columns=rename)
    outcome_s = safe(req.outcome)
    group_s   = safe(req.group_col)
    fe_s      = [safe(f) for f in req.fixed_effects]
    formula   = f"{outcome_s} ~ " + (" + ".join(fe_s) if fe_s else "1")

    if is_binary:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Independence

        gee_model = GEE.from_formula(
            formula, group_s, data=df_r,
            family=Binomial(),
            cov_struct=Independence(),
        )
        result = gee_model.fit()
        ci = result.conf_int()
        coefs = []
        for var in result.params.index:
            p_val = float(result.pvalues[var])
            est   = float(result.params[var])
            coefs.append({
                "variable": str(var),
                "estimate": round(est, 6),
                "exp_estimate": round(float(np.exp(est)), 4),
                "se": round(float(result.bse[var]), 6),
                "z": round(float(result.tvalues[var]), 4),
                "p": round(p_val, 6),
                "ci_low":  round(float(ci.loc[var, 0]), 4),
                "ci_high": round(float(ci.loc[var, 1]), 4),
                "or_low":  round(float(np.exp(ci.loc[var, 0])), 4),
                "or_high": round(float(np.exp(ci.loc[var, 1])), 4),
            })
        return {
            "model": "GEE — Binomial/Logit (Binary outcome)",
            "model_type": "gee_binomial",
            "note": (
                "Binary outcome detected. Fitted using Generalized Estimating Equations (GEE) "
                "with Binomial family and logit link — the population-averaged equivalent of a GLMM. "
                "Estimates are log-odds (logit scale); exp(β) = Odds Ratio."
            ),
            "outcome": req.outcome,
            "group": req.group_col,
            "n": int(result.nobs),
            "n_groups": int(df[req.group_col].nunique()),
            "n_excluded": n_excluded,
            "aic": float(result.aic) if hasattr(result, "aic") else None,
            "bic": float(result.bic) if hasattr(result, "bic") else None,
            "log_likelihood": float(result.llf) if hasattr(result, "llf") else None,
            "random_effect_variance": None,
            "residual_variance": None,
            "icc": None,
            "coefficients": coefs,
        }

    model = smf.mixedlm(formula, df_r, groups=df_r[group_s]).fit(reml=True)

    fe_ci = model.conf_int()
    coefs = []
    for var in model.fe_params.index:
        coefs.append({
            "variable": str(var),
            "estimate": float(model.fe_params[var]),
            "se": float(model.bse_fe[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(fe_ci.loc[var, 0]),
            "ci_high": float(fe_ci.loc[var, 1]),
        })

    re_var = float(model.cov_re.iloc[0, 0]) if model.cov_re is not None and model.cov_re.size > 0 else None
    residual_var = float(model.scale)

    return {
        "model": "Linear Mixed Model (REML)",
        "model_type": "lmm",
        "outcome": req.outcome,
        "group": req.group_col,
        "n": int(model.nobs),
        "n_groups": int(df[req.group_col].nunique()),
        "n_excluded": n_excluded,
        # statsmodels MixedLM under REML leaves AIC/BIC undefined (NaN); coerce
        # any non-finite information criterion to None so the response is
        # JSON-serialisable instead of crashing FastAPI's encoder.
        "aic": (float(model.aic) if np.isfinite(model.aic) else None),
        "bic": (float(model.bic) if np.isfinite(model.bic) else None),
        "log_likelihood": (float(model.llf) if np.isfinite(model.llf) else None),
        "random_effect_variance": re_var,
        "residual_variance": residual_var,
        "icc": (re_var / (re_var + residual_var)) if re_var is not None else None,
        "coefficients": coefs,
    }


# ── Wide → Long Melt Reshape ──────────────────────────────────────────────

class MeltRequest(BaseModel):
    session_id: str
    id_col: str
    value_cols: List[str]
    time_var_name: str = "TimePoint"
    value_var_name: str = "Value"
    time_labels: Optional[List[str]] = None


@router.post("/melt")
def melt_wide_to_long(req: MeltRequest):
    """Reshape wide-format repeated measures into long format and save back to session."""
    df = _get_df(req.session_id)
    missing = [c for c in [req.id_col] + req.value_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing}")
    if len(req.value_cols) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 value columns to melt")

    labels = req.time_labels if req.time_labels and len(req.time_labels) == len(req.value_cols) \
             else req.value_cols

    other_cols = [c for c in df.columns if c not in req.value_cols and c != req.id_col]
    keep = [req.id_col] + req.value_cols + other_cols[:20]
    df_sub = df[[c for c in keep if c in df.columns]].copy()

    df_long = df_sub.melt(
        id_vars=[c for c in df_sub.columns if c not in req.value_cols],
        value_vars=req.value_cols,
        var_name=req.time_var_name,
        value_name=req.value_var_name,
    )
    label_map = dict(zip(req.value_cols, labels))
    df_long[req.time_var_name] = df_long[req.time_var_name].map(label_map)

    store.save(req.session_id, df_long)

    # Sanitise the preview: NaN/Inf in value columns (e.g. a melted column
    # that had missing values, like cholesterol) crash FastAPI's JSON
    # serializer (allow_nan=False) and surface as a misleading "non-finite
    # statistic" 400. Walk the records and turn non-finite floats into None
    # (→ JSON null). (.where(notna, None) does NOT work on float dtype — it
    # coerces the placeholder back to NaN, so an explicit walk is required.)
    import math as _math
    _raw_preview = df_long.head(10).to_dict(orient="records")
    preview = [
        {k: (None if isinstance(v, float) and not _math.isfinite(v) else v)
         for k, v in rec.items()}
        for rec in _raw_preview
    ]

    return {
        "rows": len(df_long),
        "columns": list(df_long.columns),
        "time_var": req.time_var_name,
        "value_var": req.value_var_name,
        "time_points": labels,
        "preview": preview,
    }


# ── Linear Regression Diagnostics ─────────────────────────────────────────

class DiagRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"


@router.post("/linear_diag")
def linear_diagnostics(req: DiagRequest):
    from scipy import stats as scipy_stats

    df_full = _get_df(req.session_id)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X, _dropped_const = design_with_constant(X)
    y = df[req.outcome].astype(float)
    model = sm.OLS(y, X).fit()

    fitted   = model.fittedvalues.values
    resid    = model.resid.values
    std_res  = model.get_influence().resid_studentized_internal
    sqrt_abs = np.sqrt(np.abs(std_res))

    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(resid, dist="norm")
    qq_x_line = np.array([min(osm), max(osm)])
    qq_y_line  = slope * qq_x_line + intercept

    N = min(len(fitted), 2000)
    idx = np.random.choice(len(fitted), N, replace=False) if len(fitted) > N else np.arange(N)

    return {
        "residuals_fitted": {
            "x": fitted[idx].tolist(),
            "y": resid[idx].tolist(),
        },
        "qq": {
            "theoretical": osm[idx[:len(osm)]].tolist() if len(osm) > N else osm.tolist(),
            "sample":      osr[idx[:len(osr)]].tolist() if len(osr) > N else osr.tolist(),
            "line_x":      qq_x_line.tolist(),
            "line_y":      qq_y_line.tolist(),
        },
        "scale_location": {
            "x": fitted[idx].tolist(),
            "y": sqrt_abs[idx].tolist(),
        },
        "r_squared": float(model.rsquared),
        "residual_se": float(np.sqrt(model.mse_resid)),
        "n": int(model.nobs),
    }
