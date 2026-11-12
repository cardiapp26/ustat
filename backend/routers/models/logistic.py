from __future__ import annotations

from typing import Annotated, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from loguru import logger

from services import store
from services.impute import apply_imputation
from services.missing_data import (
    mice_multiple,
    pool_rubin_terms,
    missing_pattern_summary,
    add_missing_data_diagnostics,
)
from services.category_health import rare_level_warnings
from services.regression import (
    add_pairwise_interactions as _add_pairwise_interactions,
    constant_column_warnings,
    design_with_constant,
    dummy_column_map,
    encode_binary_outcome,
    outcome_level_sentence,
    stepwise_forward as _stepwise_forward,
    stepwise_backward as _stepwise_backward,
    _uni_p_for_pred,
)
from services.assumptions import (
    check_logistic_assumptions,
    add_assumption_warnings_to_result,
)

router = APIRouter()

# ── Helpers (shared across regression endpoints) ───────────────────────────────

def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _apply_scaling(df: pd.DataFrame, predictors: List[str], scale_factors: Optional[dict]):
    if not scale_factors:
        return df, predictors
    df = df.copy()
    new_predictors = []
    for pred in predictors:
        factor = scale_factors.get(pred)
        factor_f = float(factor) if factor is not None else 1.0
        if factor_f and factor_f != 1.0 and pred in df.columns:
            factor_label = int(factor_f) if factor_f == int(factor_f) else factor_f
            new_name = f"{pred} (per {factor_label} units)"
            df[new_name] = df[pred] / factor_f
            new_predictors.append(new_name)
        else:
            new_predictors.append(pred)
    return df, new_predictors


def _design(df: pd.DataFrame, pred_list: List[str], interactions) -> pd.DataFrame:
    """Dummy-coded predictors plus the requested pairwise interactions."""
    X = pd.get_dummies(df[pred_list], drop_first=True).astype(float)
    X, _ = _add_pairwise_interactions(X, interactions, pred_list, dummy_column_map(df, pred_list))
    return X


def _binary_outcome(y: pd.Series, label: str) -> tuple[pd.Series, dict]:
    """0/1 integer outcome and its level info; 422 when it has missing values."""
    coded, info = encode_binary_outcome(y, label)
    if coded.isna().any():
        raise HTTPException(
            status_code=422,
            detail=f"{label}: the outcome has {int(coded.isna().sum())} missing value(s) after imputation.",
        )
    if coded.nunique() < 2:
        raise HTTPException(status_code=422, detail="Outcome column has only one unique value; logistic regression requires both 0 and 1.")
    return coded.astype(int), info


# Logit and GLM results in statsmodels have no leverage-based correction:
# asking for HC1, HC2 or HC3 there silently gives HC0 (White). Request and
# report what is actually computed.
LOGIT_ROBUST_COV = "HC0"

MICE_FIT_NOTE = (
    "Coefficients are pooled across imputations with Rubin's rules. Model-fit "
    "statistics (omnibus test, R², Hosmer-Lemeshow, AUC, classification, AIC/BIC) "
    "are from the first imputed dataset and are not pooled."
)


def _pooled_logistic_coefs(per_imputation: list, vifs: dict) -> list:
    """Rubin-pooled coefficient rows in the same shape as a single fit."""
    rows = []
    for var, t in pool_rubin_terms(per_imputation).items():
        rows.append({
            "variable": var,
            "B": t["coef"],
            "log_odds": t["coef"],
            "se": t["se"],
            "wald": round(t["t"] ** 2, 4),
            "df": 1,
            "p": t["p"],
            "odds_ratio": _finite_exp(t["coef"]),
            "z": t["t"],
            "or_ci_low": _finite_exp(t["ci_low"]),
            "or_ci_high": _finite_exp(t["ci_high"]),
            "vif": vifs.get(var),
            "rubin_df": t["df"],
            "fmi": t["fmi"],
        })
    return rows


def _finite_exp(value: float) -> Optional[float]:
    out = float(np.exp(value))
    return out if np.isfinite(out) else None


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


# ── Standard Logistic Regression ───────────────────────────────────────────

class LogisticRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    scale_factors: Optional[dict] = None
    selection: Optional[str] = "all"
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False
    interactions: Optional[List[List[str]]] = None
    use_firth: Optional[bool] = False


@router.post("/logistic")
def logistic_regression(
    req: LogisticRequest,
    saveable_predictions: Annotated[bool, Header(alias="X-Ustat-Save-Probabilities")] = False,
):
    df_full, scope = store.get_filtered_with_scope(req.session_id)
    if df_full is None or scope is None:
        raise HTTPException(status_code=404, detail="Session not found")
    n_total = len(df_full)
    imputation_method = req.imputation or "listwise"
    cov_type = LOGIT_ROBUST_COV if req.robust_se else "nonrobust"

    if imputation_method == "mice":
        imp_result = mice_multiple(
            df_full, [req.outcome] + req.predictors, n_imputations=5
        )
        imputed_dfs = imp_result.imputed_datasets

        per_imputation = []
        for df_imp in imputed_dfs:
            df_imp, pred_list_imp = _apply_scaling(df_imp, req.predictors, req.scale_factors)
            Xc, _dropped_const = design_with_constant(_design(df_imp, pred_list_imp, req.interactions))
            y_imp, _ = _binary_outcome(df_imp[req.outcome], "Logistic regression")
            try:
                m = sm.Logit(y_imp, Xc).fit(disp=False, maxiter=100, cov_type=cov_type)
                per_imputation.append({str(var): (float(m.params[var]), float(m.bse[var])) for var in m.params.index})
            except Exception:
                logger.exception("MICE Logit fit iteration failed")
                continue

        df = imputed_dfs[0]
        n_excluded = n_total - len(df)
        df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)
        X_const, dropped_const = design_with_constant(_design(df, pred_list, req.interactions))
        use_mice_pooled = bool(per_imputation)
    else:
        per_imputation = []
        df = apply_imputation(df_full, [req.outcome] + req.predictors, imputation_method)
        n_excluded = n_total - len(df)
        df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)
        X_const, dropped_const = design_with_constant(_design(df, pred_list, req.interactions))
        use_mice_pooled = False

    y, outcome_info = _binary_outcome(df[req.outcome], "Logistic regression")

    model = sm.Logit(y, X_const).fit(disp=False, cov_type=cov_type)

    vifs = _compute_vif(X_const)
    coefs = []
    ci = model.conf_int()
    for var in model.params.index:
        est = float(model.params[var])
        se_val = float(model.bse[var])
        z_val = float(model.tvalues[var])
        wald = z_val ** 2
        coefs.append({
            "variable": str(var),
            "B": est,
            "log_odds": est,
            "se": se_val,
            "wald": round(wald, 4),
            "df": 1,
            "p": float(model.pvalues[var]),
            "odds_ratio": float(np.exp(est)) if np.isfinite(np.exp(est)) else None,
            "z": z_val,
            "or_ci_low": float(np.exp(ci.loc[var, 0])) if np.isfinite(np.exp(ci.loc[var, 0])) else None,
            "or_ci_high": float(np.exp(ci.loc[var, 1])) if np.isfinite(np.exp(ci.loc[var, 1])) else None,
            "vif": vifs.get(str(var)),
        })

    if use_mice_pooled:
        coefs = _pooled_logistic_coefs(per_imputation, vifs)

    from scipy.stats import chi2 as chi2_dist
    from sklearn.metrics import roc_auc_score, confusion_matrix

    n = float(model.nobs)
    llf = float(model.llf)
    llnull = float(model.llnull)

    omnibus_chi2 = -2 * (llnull - llf)
    omnibus_df = len(model.params) - 1
    omnibus_p = float(1 - chi2_dist.cdf(omnibus_chi2, omnibus_df)) if omnibus_df > 0 else 1.0

    minus2ll = -2 * llf
    cox_snell_r2 = 1 - np.exp((2 / n) * (llnull - llf))
    max_r2 = 1 - np.exp((2 / n) * llnull)
    nagelkerke_r2 = float(cox_snell_r2 / max_r2) if max_r2 != 0 else 0.0

    # Plain arrays: predict() returns a Series carrying the frame's index, and
    # indexing that with argsort positions looked rows up by label. After any
    # dropped row or case filter the labels no longer run 0..n-1, the lookup
    # raised, and the Hosmer-Lemeshow test silently vanished from the result.
    pred_probs = np.asarray(model.predict(X_const), dtype=float)
    y_arr = np.asarray(y, dtype=int)

    try:
        order = np.argsort(pred_probs)
        groups = np.array_split(order, 10)
        hl_chi2_val = 0.0
        for grp in groups:
            obs_1 = y_arr[grp].sum()
            obs_0 = len(grp) - obs_1
            exp_1 = pred_probs[grp].sum()
            exp_0 = len(grp) - exp_1
            if exp_1 > 0:
                hl_chi2_val += (obs_1 - exp_1) ** 2 / exp_1
            if exp_0 > 0:
                hl_chi2_val += (obs_0 - exp_0) ** 2 / exp_0
        hl_df = 8
        hl_p = float(1 - chi2_dist.cdf(hl_chi2_val, hl_df))
        hosmer_lemeshow = {"chi2": round(hl_chi2_val, 4), "df": hl_df, "p": round(hl_p, 6)}
    except Exception:
        logger.exception("Hosmer-Lemeshow calculation failed")
        hosmer_lemeshow = None

    y_pred = (pred_probs >= 0.5).astype(int)
    try:
        cm = confusion_matrix(y_arr, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = float((tp + tn) / (tp + tn + fp + fn))
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        classification = {
            "accuracy": round(accuracy, 4), "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4), "ppv": round(ppv, 4), "npv": round(npv, 4),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    except Exception:
        logger.exception("Confusion matrix computation failed")
        classification = None

    try:
        auc = float(roc_auc_score(y_arr, pred_probs))
    except Exception:
        auc = None

    logistic_assumption_report = check_logistic_assumptions(
        y=y_arr,
        pred_probs=pred_probs,
        hosmer_lemeshow=hosmer_lemeshow,
        model=model,
    )

    result = {
        "model": f"Logistic Regression{' [Robust SE]' if req.robust_se else ''}",
        "robust_se_type": LOGIT_ROBUST_COV if req.robust_se else None,
        "outcome": req.outcome,
        "outcome_event_level": outcome_info["event_level"],
        "outcome_reference_level": outcome_info["reference_level"],
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "minus2ll": round(minus2ll, 4),
        "cox_snell_r2": round(float(cox_snell_r2), 4),
        "nagelkerke_r2": round(float(nagelkerke_r2), 4),
        "pseudo_r2": float(model.prsquared),
        "pseudo_r2_type": "McFadden",
        "mcfadden_r2": float(model.prsquared),
        "log_likelihood": llf,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "omnibus": {"chi2": round(omnibus_chi2, 4), "df": omnibus_df, "p": round(omnibus_p, 6)},
        "hosmer_lemeshow": hosmer_lemeshow,
        "classification": classification,
        "auc": round(auc, 4) if auc is not None else None,
        "coefficients": coefs,
        "result_text": (
            _logistic_results_text(req.outcome, coefs, omnibus_chi2, omnibus_df, omnibus_p, nagelkerke_r2, hosmer_lemeshow, classification, auc)
            + outcome_level_sentence(req.outcome, outcome_info)
            + (" " + MICE_FIT_NOTE if use_mice_pooled else "")
        ),
    }
    ignored = _ignored_option_warnings(req)
    if ignored:
        result["warnings"] = ignored

    result = add_assumption_warnings_to_result(result, logistic_assumption_report)

    if dropped_const:
        result["warnings"] = (result.get("warnings") or []) + constant_column_warnings(dropped_const)

    # Dirty / very rare categorical levels silently become extra dummy
    # predictors with n<5 — that's how a typo'd 'sex' column ("M", "F", "x",
    # "Female") destabilises the fit. Surface them so the user can decide.
    cat_warnings = rare_level_warnings(df, req.predictors)
    if cat_warnings:
        existing = result.setdefault("warnings", [])
        if not isinstance(existing, list):
            existing = []
            result["warnings"] = existing
        for w in cat_warnings:
            existing.append(w["note"])
        result["rare_categories"] = cat_warnings

    if use_mice_pooled:
        result["pooled_from_imputations"] = True
        result["imputation"] = "mice (pooled)"
        result["n_imputations_pooled"] = len(per_imputation)
        result["fit_statistics_source"] = "first imputed dataset"
        result["warnings"] = (result.get("warnings") or []) + [MICE_FIT_NOTE]
    elif imputation_method == "mice":
        result["warnings"] = (result.get("warnings") or []) + [
            "No imputed dataset could be fitted; results are from the first imputed dataset only and are not pooled."
        ]

    missing_info = missing_pattern_summary(df_full, [req.outcome] + req.predictors)
    result = add_missing_data_diagnostics(result, missing_info)

    if saveable_predictions and np.isfinite(pred_probs).all():
        values = pd.Series(pred_probs, index=X_const.index)
        token = store.remember_fitted_probabilities(req.session_id, scope, req.outcome, values)
        if token is not None:
            result["probability_fit"] = {
                "token": token,
                "source": "first imputed dataset" if imputation_method == "mice" else "fitted sample",
                "event_level": outcome_info["event_level"],
            }
    return result


class SaveLogisticProbabilitiesRequest(BaseModel):
    session_id: str
    fit_token: str


@router.post("/logistic/save_probabilities")
def save_logistic_probabilities(req: SaveLogisticProbabilitiesRequest):
    from routers.session import _session_preview

    df, name, undo_depth = store.append_fitted_probabilities(req.session_id, req.fit_token)
    values = df[name]
    n_computed = int(values.notna().sum())
    return {
        **_session_preview(df, req.session_id),
        "name": name,
        "n_computed": n_computed,
        "n_missing": len(values) - n_computed,
        "undo_depth": undo_depth,
    }


def _ignored_option_warnings(req: "LogisticRequest") -> list:
    """Options this endpoint accepts in its request but does not apply."""
    notes = []
    if (req.selection or "all").strip().lower() != "all":
        notes.append(
            f"selection='{req.selection}' is not applied by this endpoint: every listed predictor "
            "is in the model. Variable selection is available in the logistic OR table."
        )
    if req.use_firth:
        notes.append(
            "use_firth is not applied by this endpoint: this is an ordinary maximum-likelihood fit. "
            "Use Firth logistic regression for the penalised fit."
        )
    return notes


def _logistic_results_text(outcome, coefs, chi2_val, df, chi2_p, nagelkerke, hl, classification, auc):
    sig_coefs = [
        c for c in coefs
        if c["variable"] != "const" and c["p"] is not None and c["p"] < 0.05
        and c["odds_ratio"] is not None and c["or_ci_low"] is not None and c["or_ci_high"] is not None
    ]
    p_str = "<0.001" if chi2_p < 0.001 else f"{chi2_p:.3f}"
    parts = [
        f"A binary logistic regression was performed to predict {outcome}. "
        f"The omnibus test indicated the model was {'statistically significant' if chi2_p < 0.05 else 'not statistically significant'} "
        f"(χ²({df}) = {chi2_val:.3f}, p = {p_str})."
    ]
    if classification:
        parts.append(f"The model explained {nagelkerke*100:.1f}% of the variance (Nagelkerke R²) "
                     f"and correctly classified {classification['accuracy']*100:.1f}% of cases.")
    if hl:
        hl_p_str = "<0.001" if hl["p"] < 0.001 else f'{hl["p"]:.3f}'
        parts.append(f"Hosmer-Lemeshow test indicated {'adequate' if hl['p'] >= 0.05 else 'poor'} model fit (p = {hl_p_str}).")
    if auc:
        parts.append(f"The area under the ROC curve was {auc:.3f}.")
    if sig_coefs:
        pred_strs = []
        for c in sig_coefs:
            p_s = "<0.001" if c["p"] < 0.001 else f'{c["p"]:.3f}'
            pred_strs.append(f'{c["variable"]} (OR = {c["odds_ratio"]:.2f}, 95% CI: {c["or_ci_low"]:.2f}–{c["or_ci_high"]:.2f}, p = {p_s})')
        parts.append("Significant predictors were: " + "; ".join(pred_strs) + ".")
    else:
        parts.append("No predictor reached statistical significance at the 0.05 level.")
    return " ".join(p for p in parts if p)


# ── Firth Penalized Logistic Regression ───────────────────────────────────────

class FirthLogisticRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    scale_factors: Optional[dict] = None
    imputation: Optional[str] = "listwise"
    max_iter: int = 50
    tol: float = 1e-6
    interactions: Optional[List[List[str]]] = None


# Firth CIs and p-values here are Wald. R's logistf, the usual reference,
# defaults to penalised PROFILE likelihood, so the two agree on coefficients
# yet part on intervals, most near separation, which is the method's reason
# to exist. Every Firth response says so next to its numbers and in the
# paragraph that gets pasted into a manuscript.
FIRTH_CI_METHOD = "Wald"
FIRTH_WALD_NOTE = (
    "Confidence intervals and p-values are Wald (estimate ± 1.96·SE), not the "
    "penalised profile-likelihood intervals R's logistf reports by default. "
    "Coefficients agree with logistf; the intervals can differ noticeably near "
    "separation, where Wald intervals are the less reliable of the two."
)
FIRTH_WALD_SENTENCE = (
    " 95% confidence intervals and p-values are Wald-based, not penalised "
    "profile likelihood."
)


def _firth_fit(X: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-6):
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        eta = np.clip(X @ beta, -30, 30)
        pi = 1.0 / (1.0 + np.exp(-eta))
        W = pi * (1.0 - pi)
        XtWX = (X.T * W) @ X
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.inv(XtWX + 1e-8 * np.eye(p))
        sqrtW = np.sqrt(W)
        XW = X * sqrtW[:, None]
        H = XW @ XtWX_inv @ XW.T
        h = np.diag(H)
        score = X.T @ ((y - pi) + h * (0.5 - pi))
        step = XtWX_inv @ score
        beta_new = beta + step
        for _ in range(10):
            eta_new = np.clip(X @ beta_new, -30, 30)
            pi_new = 1.0 / (1.0 + np.exp(-eta_new))
            ll_new = float(np.sum(y * eta_new - np.log1p(np.exp(eta_new))))
            try:
                sign, logdet = np.linalg.slogdet((X.T * (pi_new * (1.0 - pi_new))) @ X)
                pen_ll_new = ll_new + 0.5 * logdet
            except Exception:
                pen_ll_new = ll_new
            ll_old = float(np.sum(y * eta - np.log1p(np.exp(eta))))
            try:
                sign_o, logdet_o = np.linalg.slogdet(XtWX)
                pen_ll_old = ll_old + 0.5 * logdet_o
            except Exception:
                pen_ll_old = ll_old
            if pen_ll_new >= pen_ll_old - 1e-10 or np.linalg.norm(step) < tol:
                break
            step = step / 2.0
            beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30)
    pi = 1.0 / (1.0 + np.exp(-eta))
    W = pi * (1.0 - pi)
    XtWX = (X.T * W) @ X
    try:
        vcov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        vcov = np.linalg.inv(XtWX + 1e-8 * np.eye(p))
    ll = float(np.sum(y * eta - np.log1p(np.exp(eta))))
    try:
        sign, logdet = np.linalg.slogdet(XtWX)
        penalized_ll = ll + 0.5 * logdet
    except Exception:
        penalized_ll = ll
    return beta, vcov, ll, penalized_ll, n_iter, converged, pi


@router.post("/firth_logistic")
def firth_logistic_regression(req: FirthLogisticRequest):
    from scipy.stats import chi2 as chi2_dist
    from scipy.stats import norm as sp_norm
    from sklearn.metrics import roc_auc_score, confusion_matrix

    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)
    df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)
    X_df = _design(df, pred_list, req.interactions)
    X_const_df = sm.add_constant(X_df, has_constant="add")
    y, outcome_info = _binary_outcome(df[req.outcome], "Firth logistic regression")

    X = X_const_df.values.astype(float)
    y_arr = np.asarray(y, dtype=int)

    beta, vcov, ll, penalized_ll, n_iter, converged, pi = _firth_fit(
        X, y_arr, max_iter=req.max_iter, tol=req.tol
    )
    se = np.sqrt(np.diag(vcov))

    X_null = np.ones((len(y_arr), 1))
    _, _, ll0, pen_ll0, _, _, _ = _firth_fit(X_null, y_arr, max_iter=req.max_iter, tol=req.tol)

    z = beta / np.where(se > 0, se, np.nan)
    p_wald = 2.0 * (1.0 - sp_norm.cdf(np.abs(z)))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    # Wald, and knowingly so. logistf reports a PENALISED profile likelihood
    # p-value, which is not the same as refitting with the column deleted:
    # deleting a column also changes the Jeffreys penalty, because the penalty
    # is a function of the design matrix. A naive drop-column LRT was tried
    # here and moved the age p-value from 0.0077 to 5.6e-05 against R's
    # 0.0065 — further away, not closer. The correct test constrains the
    # coefficient to zero while evaluating the penalty on the full design,
    # which needs a constrained Firth fit rather than a smaller one.
    p_two = p_wald

    coefs = []
    for i, var in enumerate(X_const_df.columns):
        coefs.append({
            "variable": str(var),
            "B": float(beta[i]),
            "log_odds": float(beta[i]),
            "se": float(se[i]),
            "wald": round(float(z[i] ** 2), 4),
            "df": 1,
            "p": float(p_two[i]),
            "p_method": "Wald",
            "odds_ratio": float(np.exp(beta[i])),
            "z": float(z[i]),
            "or_ci_low": float(np.exp(ci_low[i])),
            "or_ci_high": float(np.exp(ci_high[i])),
            "vif": None,
        })

    n = len(y_arr)
    minus2ll = -2.0 * ll
    omnibus_chi2 = float(2.0 * (penalized_ll - pen_ll0))
    omnibus_df = X.shape[1] - 1
    omnibus_p = float(1 - chi2_dist.cdf(omnibus_chi2, omnibus_df)) if omnibus_df > 0 else 1.0

    cox_snell_r2 = float(1 - np.exp((2 / n) * (ll0 - ll))) if n > 0 else 0.0
    max_r2 = float(1 - np.exp((2 / n) * ll0)) if n > 0 else 1.0
    nagelkerke_r2 = float(cox_snell_r2 / max_r2) if max_r2 != 0 else 0.0

    k = X.shape[1]
    aic = float(-2.0 * penalized_ll + 2 * k)
    bic = float(-2.0 * penalized_ll + np.log(n) * k)

    y_pred = (pi >= 0.5).astype(int)
    try:
        cm = confusion_matrix(y_arr, y_pred)
        tn, fp, fn, tp = cm.ravel()
        classification = {
            "accuracy": round(float((tp + tn) / (tp + tn + fp + fn)), 4),
            "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
            "specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
            "ppv": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4),
            "npv": round(float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0, 4),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    except Exception:
        classification = None
    try:
        auc = float(roc_auc_score(y_arr, pi))
    except Exception:
        auc = None

    return {
        "model": "Firth Penalized Logistic Regression",
        "outcome": req.outcome,
        "outcome_event_level": outcome_info["event_level"],
        "outcome_reference_level": outcome_info["reference_level"],
        "n": int(n),
        "n_events": int(y_arr.sum()),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "converged": bool(converged),
        "iterations": int(n_iter),
        "minus2ll": round(minus2ll, 4),
        "penalized_ll": round(float(penalized_ll), 4),
        "log_likelihood": float(ll),
        "cox_snell_r2": round(cox_snell_r2, 4),
        "nagelkerke_r2": round(nagelkerke_r2, 4),
        # Kept as Cox-Snell for existing clients; /logistic reports McFadden
        # under the same key, so the type is stated and McFadden given too.
        "pseudo_r2": round(cox_snell_r2, 4),
        "pseudo_r2_type": "Cox-Snell",
        "mcfadden_r2": round(float(1.0 - ll / ll0), 4) if ll0 != 0 else None,
        "aic": round(aic, 4),
        "bic": round(bic, 4),
        "omnibus": {"chi2": round(omnibus_chi2, 4), "df": omnibus_df, "p": round(omnibus_p, 6)},
        "auc": round(auc, 4) if auc is not None else None,
        "classification": classification,
        "coefficients": coefs,
        "ci_method": FIRTH_CI_METHOD,
        "method_note": (
            "Firth (1993) bias-corrected logistic regression with Jeffreys-prior "
            "penalty, for rare events or (quasi-)separated data (Heinze & "
            "Schemper, Stat Med 2002). " + FIRTH_WALD_NOTE
        ),
        "result_text": (
            f"Firth penalized logistic regression was used to model {req.outcome} "
            f"(n = {n}, events = {int(y_arr.sum())}). The model "
            f"{'converged' if converged else 'did not fully converge'} in {n_iter} "
            f"iterations. Omnibus penalized likelihood ratio χ² = {omnibus_chi2:.3f} "
            f"on {omnibus_df} df, p = "
            f"{'<0.001' if omnibus_p < 0.001 else f'{omnibus_p:.3f}'}. "
            f"Nagelkerke R² = {nagelkerke_r2:.3f}"
            + (f", AUC = {auc:.3f}." if auc is not None else ".")
            + FIRTH_WALD_SENTENCE
            + outcome_level_sentence(req.outcome, outcome_info)
        ),
    }


# ── Logistic OR Table Endpoint ───────────────────────────────────────────────

@router.post("/logistic_table")
def logistic_or_table(req: LogisticRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    n_excluded = n_total - len(df)

    df, pred_list = _apply_scaling(df, req.predictors, req.scale_factors)

    y_coded, outcome_info = encode_binary_outcome(df[req.outcome], "Logistic regression")
    y = y_coded.values
    valid_mask = ~pd.isna(y)
    y = np.array(y[valid_mask], dtype=int)
    df = df.loc[valid_mask].reset_index(drop=True)
    if len(set(y)) < 2:
        raise HTTPException(status_code=422, detail="Outcome has only one unique value — needs both 0 and 1.")

    use_firth = bool(getattr(req, "use_firth", False))
    from scipy.stats import norm as _sp_norm

    def _fit_row(X_df, variable_names, return_model=False):
        X_enc = pd.get_dummies(X_df, drop_first=True).astype(float)
        combined = X_enc.copy()
        combined["__y__"] = y
        combined = combined.dropna()
        if len(combined) < 10:
            raise HTTPException(status_code=422, detail=f"Insufficient data after removing missing values ({len(combined)} rows)")
        y_clean = combined["__y__"].values.astype(int)
        X_clean = combined.drop(columns=["__y__"])
        if len(set(y_clean)) < 2:
            raise HTTPException(status_code=422, detail="After NaN removal, outcome has only one unique value.")
        X_const = sm.add_constant(X_clean, has_constant="add")

        if use_firth:
            try:
                X_arr = X_const.values.astype(float)
                y_arr_in = y_clean.astype(int)
                beta, vcov, ll, penalized_ll, n_iter, converged, pi = _firth_fit(X_arr, y_arr_in)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Firth fit failed: {exc}")
            se = np.sqrt(np.diag(vcov))
            z = beta / np.where(se > 0, se, np.nan)
            p_two = 2.0 * (1.0 - _sp_norm.cdf(np.abs(z)))
            ci_lo_arr = beta - 1.95996 * se
            ci_hi_arr = beta + 1.95996 * se
            rows = {}
            for i, var in enumerate(X_const.columns):
                if var == "const":
                    continue
                or_val = float(np.exp(beta[i]))
                ci_lo = float(np.exp(ci_lo_arr[i]))
                ci_hi = float(np.exp(ci_hi_arr[i]))
                p_val = float(p_two[i])
                if not np.isfinite(or_val):
                    or_val = 9999.0
                if not np.isfinite(ci_lo):
                    ci_lo = 0.0
                if not np.isfinite(ci_hi):
                    ci_hi = 9999.0
                rows[var] = {"or": or_val, "ci_low": ci_lo, "ci_high": ci_hi, "p": p_val}
            if return_model:
                X_null = np.ones((len(y_arr_in), 1))
                _, _, ll0, _, _, _, _ = _firth_fit(X_null, y_arr_in)
                class _FirthResults:
                    def __init__(self, beta, vcov, ll, ll0, pi, X_const, n_params):
                        self.params = pd.Series(beta, index=X_const.columns)
                        self.bse = pd.Series(np.sqrt(np.diag(vcov)), index=X_const.columns)
                        self.tvalues = self.params / self.bse.replace(0, np.nan)
                        self.pvalues = pd.Series(p_two, index=X_const.columns)
                        self.llf = float(ll)
                        self.llnull = float(ll0)
                        self.nobs = float(len(pi))
                        self.aic = float(-2.0 * ll + 2 * n_params)
                        self.bic = float(-2.0 * ll + np.log(self.nobs) * n_params)
                        self.prsquared = float(1.0 - ll / ll0) if ll0 != 0 else 0.0
                        self._pi = pi
                    def predict(self, _X):
                        return self._pi
                    def conf_int(self):
                        return pd.DataFrame({0: ci_lo_arr, 1: ci_hi_arr}, index=X_const.columns)
                m = _FirthResults(beta, vcov, ll, ll0, pi, X_const, X_arr.shape[1])
                return rows, m, X_const, y_clean
            return rows

        try:
            m = sm.Logit(y_clean, X_const).fit(disp=False, maxiter=200)
        except np.linalg.LinAlgError:
            raise HTTPException(status_code=422, detail="Perfect separation detected — model cannot converge. Try removing collinear predictors or switch on Firth's penalised likelihood (use_firth=true).")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Model convergence error: {exc}")
        rows = {}
        ci = m.conf_int()
        for var in m.params.index:
            if var == "const":
                continue
            or_val = float(np.exp(m.params[var]))
            ci_lo = float(np.exp(ci.loc[var, 0]))
            ci_hi = float(np.exp(ci.loc[var, 1]))
            p_val = float(m.pvalues[var])
            if not np.isfinite(or_val):
                or_val = 9999.0
            if not np.isfinite(ci_lo):
                ci_lo = 0.0
            if not np.isfinite(ci_hi):
                ci_hi = 9999.0
            rows[var] = {"or": or_val, "ci_low": ci_lo, "ci_high": ci_hi, "p": p_val}
        if return_model:
            return rows, m, X_const, y_clean
        return rows

    column_map = dummy_column_map(df, pred_list)
    uni_results: dict = {}
    skipped: list = []
    for pred in pred_list:
        try:
            rows = _fit_row(df[[pred]], [pred])
            for var, vals in rows.items():
                uni_results[var] = vals
        except HTTPException as he:
            skipped.append(f"{pred}: {he.detail}")
        except Exception as exc:
            logger.exception("Univariate logit fit failed")
            skipped.append(f"{pred}: {exc}")

    sel = (req.selection or "all").strip().lower()

    if sel == "p05":
        multi_pred_list = [p for p in pred_list if _uni_p_for_pred(p, uni_results, column_map) < 0.05]
        selection_label = "Univariate p < 0.05"
    elif sel == "p10":
        multi_pred_list = [p for p in pred_list if _uni_p_for_pred(p, uni_results, column_map) < 0.10]
        selection_label = "Univariate p < 0.10"
    elif sel == "forward":
        multi_pred_list = _stepwise_forward(y, df, pred_list, p_enter=0.05)
        selection_label = "Stepwise Forward (p_enter=0.05)"
    elif sel == "backward":
        multi_pred_list = _stepwise_backward(y, df, pred_list, p_remove=0.10)
        selection_label = "Stepwise Backward (p_remove=0.10)"
    else:
        multi_pred_list = pred_list
        selection_label = "All variables (Enter)"

    multi_results: dict = {}
    multi_error = None
    model_stats = None
    if multi_pred_list:
        try:
            multi_results, multi_model, multi_X, multi_y = _fit_row(df[multi_pred_list], multi_pred_list, return_model=True)
            from scipy.stats import chi2 as chi2_dist
            from sklearn.metrics import roc_auc_score, confusion_matrix as _cm

            n_m = float(multi_model.nobs)
            llf_m = float(multi_model.llf)
            llnull_m = float(multi_model.llnull)
            omnibus_chi2 = -2 * (llnull_m - llf_m)
            omnibus_df = len(multi_model.params) - 1
            omnibus_p = float(1 - chi2_dist.cdf(omnibus_chi2, omnibus_df)) if omnibus_df > 0 else 1.0
            minus2ll = -2 * llf_m
            cox_snell = 1 - np.exp((2 / n_m) * (llnull_m - llf_m))
            max_r2 = 1 - np.exp((2 / n_m) * llnull_m)
            nagelkerke = float(cox_snell / max_r2) if max_r2 != 0 else 0.0
            pred_probs = np.asarray(multi_model.predict(multi_X), dtype=float)
            try:
                auc_val = float(roc_auc_score(multi_y, pred_probs))
            except Exception:
                auc_val = None
            try:
                order = np.argsort(pred_probs)
                groups = np.array_split(order, 10)
                hl_chi2_val = 0.0
                for grp in groups:
                    o1 = multi_y[grp].sum()
                    o0 = len(grp) - o1
                    e1 = pred_probs[grp].sum()
                    e0 = len(grp) - e1
                    if e1 > 0:
                        hl_chi2_val += (o1 - e1)**2 / e1
                    if e0 > 0:
                        hl_chi2_val += (o0 - e0)**2 / e0
                hl_p = float(1 - chi2_dist.cdf(hl_chi2_val, 8))
                hl = {"chi2": round(hl_chi2_val, 4), "df": 8, "p": round(hl_p, 6)}
            except Exception:
                hl = None
            try:
                y_pred = (pred_probs >= 0.5).astype(int)
                tn, fp, fn, tp = _cm(multi_y, y_pred).ravel()
                classification = {
                    "accuracy": round(float((tp+tn)/(tp+tn+fp+fn)), 4),
                    "sensitivity": round(float(tp/(tp+fn)), 4) if (tp+fn) > 0 else 0,
                    "specificity": round(float(tn/(tn+fp)), 4) if (tn+fp) > 0 else 0,
                    "ppv": round(float(tp/(tp+fp)), 4) if (tp+fp) > 0 else 0,
                    "npv": round(float(tn/(tn+fn)), 4) if (tn+fn) > 0 else 0,
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                }
            except Exception:
                classification = None

            model_stats = {
                "omnibus": {"chi2": round(omnibus_chi2, 4), "df": omnibus_df, "p": round(omnibus_p, 6)},
                "minus2ll": round(minus2ll, 4),
                "cox_snell_r2": round(float(cox_snell), 4),
                "nagelkerke_r2": round(nagelkerke, 4),
                "pseudo_r2": round(float(multi_model.prsquared), 4),
                "pseudo_r2_type": "McFadden",
                "auc": round(auc_val, 4) if auc_val else None,
                "hosmer_lemeshow": hl,
                "classification": classification,
            }
        except HTTPException as he:
            multi_error = he.detail
        except Exception as exc:
            logger.exception("Multivariate logit fit failed")
            multi_error = str(exc)

    all_vars = list(dict.fromkeys(list(uni_results.keys()) + list(multi_results.keys())))
    table = []
    for var in all_vars:
        u = uni_results.get(var, {})
        m = multi_results.get(var, {})
        table.append({
            "variable": var,
            "uni_or": u.get("or"),
            "uni_ci_low": u.get("ci_low"),
            "uni_ci_high": u.get("ci_high"),
            "uni_p": u.get("p"),
            "multi_or": m.get("or"),
            "multi_ci_low": m.get("ci_low"),
            "multi_ci_high": m.get("ci_high"),
            "multi_p": m.get("p"),
        })

    result_text = _ortable_results_text(req.outcome, table, model_stats, selection_label)
    return {
        "model": ("Firth Penalised Logistic OR Table" if use_firth else "Logistic OR Table"),
        "use_firth": bool(use_firth),
        **({"ci_method": FIRTH_CI_METHOD, "method_note": FIRTH_WALD_NOTE} if use_firth else {}),
        "outcome": req.outcome,
        "outcome_event_level": outcome_info["event_level"],
        "outcome_reference_level": outcome_info["reference_level"],
        "n": len(df),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "selection_method": selection_label,
        "n_multi": len(multi_pred_list),
        "n_total": int(n_total),
        "n_predictors": len(pred_list),
        "table": table,
        "model_stats": model_stats,
        "result_text": (
            result_text + (FIRTH_WALD_SENTENCE if use_firth else "")
            + outcome_level_sentence(req.outcome, outcome_info)
        ),
        "warnings": (skipped if skipped else []) + ([f"Multivariate: {multi_error}"] if multi_error else []),
    }


def _ortable_results_text(outcome, table, stats, selection):
    parts = []
    uni_sig = [r for r in table if r.get("uni_p") is not None and r["uni_p"] < 0.05]
    multi_sig = [r for r in table if r.get("multi_p") is not None and r["multi_p"] < 0.05]

    parts.append(f"Univariate logistic regression identified {len(uni_sig)} of {len(table)} variables "
                 f"as significantly associated with {outcome} (p < 0.05).")

    if stats:
        om = stats.get("omnibus")
        if om:
            p_s = "<0.001" if om["p"] < 0.001 else f'{om["p"]:.3f}'
            parts.append(f"The multivariate model ({selection}) was {'significant' if om['p'] < 0.05 else 'not significant'} "
                         f"(χ²({om['df']}) = {om['chi2']:.3f}, p = {p_s}), "
                         f"Nagelkerke R² = {stats.get('nagelkerke_r2', 0):.3f}.")
        if stats.get("auc"):
            parts.append(f"AUC = {stats['auc']:.3f}.")
        cl = stats.get("classification")
        if cl:
            parts.append(f"Overall classification accuracy was {cl['accuracy']*100:.1f}%.")

    if multi_sig:
        pred_strs = []
        for r in multi_sig:
            p_s = "<0.001" if r["multi_p"] < 0.001 else f'{r["multi_p"]:.3f}'
            pred_strs.append(f'{r["variable"]} (OR = {r["multi_or"]:.2f}, 95% CI: {r["multi_ci_low"]:.2f}–{r["multi_ci_high"]:.2f}, p = {p_s})')
        parts.append("Independent predictors: " + "; ".join(pred_strs) + ".")

    return " ".join(parts)
