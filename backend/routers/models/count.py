"""Zero-inflated count models: POST /api/models/zip and /api/models/zinb.

A zero-inflated model is a mixture of two processes: a logit "structural zero"
process (the inflation part, odds ratios) and an ordinary count process (the
count part, incidence rate ratios). The Vuong test compares the zero-inflated
fit with the corresponding standard model (Poisson for ZIP, negative binomial
for ZINB) observation by observation, the way R's pscl::vuong does.
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel
from scipy import stats as scipy_stats
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedPoisson,
)

from services.impute import apply_imputation
from services.regression import constant_column_warnings, design_with_constant

from .glm import (
    _clean_predictor_categories,
    _get_df,
    _sanitize,
    _sanitize_model_error,
    _validate_exposure,
)

router = APIRouter()

Z_CRIT = float(scipy_stats.norm.ppf(0.975))
VUONG_CRIT = Z_CRIT

# ── Request ──────────────────────────────────────────────────────────────────


class ZeroInflatedRequest(BaseModel):
    session_id: str
    outcome: str
    # Count-part predictors.
    predictors: List[str]
    # Zero-inflation (logit) predictors. None or empty means intercept only.
    inflation_predictors: Optional[List[str]] = None
    imputation: Optional[str] = "listwise"
    # Follow-up time / person-years, entering the count part as an offset.
    exposure_col: Optional[str] = None


# ── Validation and design ────────────────────────────────────────────────────


def _check_columns(df_full: pd.DataFrame, req: ZeroInflatedRequest, infl_cols: List[str]) -> None:
    if req.outcome not in df_full.columns:
        raise HTTPException(status_code=422, detail=f"Outcome column '{req.outcome}' not found.")
    if not req.predictors:
        raise HTTPException(status_code=422, detail="Select at least one count-part predictor.")
    for col in list(req.predictors) + infl_cols:
        if col not in df_full.columns:
            raise HTTPException(status_code=422, detail=f"Predictor column '{col}' not found.")
        if col == req.outcome:
            raise HTTPException(status_code=422, detail="The outcome cannot also be a predictor.")
    col = req.exposure_col
    if col is not None and str(col).strip() != "" and col in infl_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Exposure column '{col}' is also listed as an inflation predictor. The exposure "
                   "enters the model as an offset, never as a predictor; remove it from the predictors.",
        )


def _validate_counts(y: pd.Series, label: str) -> None:
    if y.isna().all():
        raise HTTPException(status_code=422, detail="Outcome column has no numeric values.")
    if (y.dropna() < 0).any():
        raise HTTPException(status_code=422, detail=f"{label} requires non-negative integer counts. Negative values found.")
    if (y.dropna() % 1 != 0).any():
        raise HTTPException(status_code=422, detail=f"{label} requires integer counts. Fractional values found.")
    if not (y == 0).any():
        raise HTTPException(
            status_code=422,
            detail=f"{label} needs at least one zero count; the outcome has no zeros, so there is nothing to inflate.",
        )
    if not (y > 0).any():
        raise HTTPException(
            status_code=422,
            detail=f"{label} needs at least one non-zero count; the outcome is zero for every row.",
        )


def _design(df: pd.DataFrame, cols: List[str]):
    """Dummy-coded design with an intercept, as (matrix, names, dropped constants)."""
    if cols:
        X = pd.get_dummies(df[cols], drop_first=True)
        X, dropped = design_with_constant(X)
    else:
        X = pd.DataFrame({"const": np.ones(len(df))}, index=df.index)
        dropped = []
    return X.to_numpy(dtype=float), [str(c) for c in X.columns], dropped


# ── Fitting ──────────────────────────────────────────────────────────────────


def _valid_fit(res, zinb: bool) -> bool:
    params = np.asarray(res.params, dtype=float)
    if not np.isfinite(params).all() or not np.isfinite(float(res.llf)):
        return False
    if zinb and params[-1] <= 0:
        return False
    return True


def _fit_zero_inflated(model, zinb: bool, zip_start: Optional[np.ndarray] = None):
    """Fit with a few optimisers and keep the best valid likelihood.

    The zero-inflated likelihood is flat and awkward: a single optimiser can
    run off to absurd parameters (especially ZINB on data that are close to
    Poisson, where alpha heads to the boundary), so several are tried and
    the highest finite log-likelihood with a positive alpha wins.
    """
    stage1 = [("lbfgs", None), ("bfgs", None)]
    stage2 = [("nm", None)]
    if zip_start is not None:
        stage2 = [("bfgs", zip_start), ("lbfgs", zip_start), ("nm", zip_start), ("nm", None)]
    candidates = []
    for stage in (stage1, stage2):
        for method, start in stage:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = model.fit(start_params=start, method=method, maxiter=2000, disp=0)
            except Exception as exc:  # optimiser failure on one path is not fatal
                logger.debug(f"zero-inflated fit via {method} failed: {exc}")
                continue
            if _valid_fit(res, zinb):
                converged = bool(res.mle_retvals.get("converged", False)) if hasattr(res, "mle_retvals") else False
                candidates.append((converged, float(res.llf), res))
        if any(c[0] for c in candidates):
            break
    if not candidates:
        return None, False
    pool = [c for c in candidates if c[0]] or candidates
    converged, _, best = max(pool, key=lambda c: c[1])
    return best, converged


def _fit_standard(y: np.ndarray, X: np.ndarray, exposure, zinb: bool):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if zinb:
            return sm.NegativeBinomial(y, X, exposure=exposure).fit(disp=0, maxiter=200)
        return sm.Poisson(y, X, exposure=exposure).fit(disp=0, maxiter=200)


# ── Prediction helpers ───────────────────────────────────────────────────────


def _mu(X: np.ndarray, beta: np.ndarray, exposure) -> np.ndarray:
    mu = np.exp(X @ beta)
    return mu * exposure if exposure is not None else mu


def _prob_zero_count(mu: np.ndarray, alpha: Optional[float]) -> np.ndarray:
    if alpha is None:
        return np.exp(-mu)
    return np.power(1.0 + alpha * mu, -1.0 / alpha)


# ── Vuong test ───────────────────────────────────────────────────────────────


def _vuong(ll_zi: np.ndarray, ll_std: np.ndarray, k_zi: int, k_std: int) -> Optional[dict]:
    """Vuong (1989) non-nested test, ZI model against the standard model.

    m_i = ll_zi_i - ll_std_i; V = sqrt(n) * mean(m) / sd(m). The AIC and BIC
    corrections subtract the parameter penalty from mean(m) first, exactly as
    pscl::vuong does: (k_zi - k_std) / n and (k_zi - k_std) * log(n) / (2n).
    Under H0 (both models equally close to the truth) V is standard normal;
    V > 0 favours the zero-inflated model.
    """
    m = np.asarray(ll_zi, dtype=float) - np.asarray(ll_std, dtype=float)
    n = len(m)
    if n < 2 or not np.isfinite(m).all():
        return None
    sd = float(np.std(m, ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return None
    bar = float(np.mean(m))
    root_n = float(np.sqrt(n))
    stats_ = {
        "raw": root_n * bar / sd,
        "aic": root_n * (bar - (k_zi - k_std) / n) / sd,
        "bic": root_n * (bar - (k_zi - k_std) * np.log(n) / (2.0 * n)) / sd,
    }
    return {
        "statistic": {k: float(v) for k, v in stats_.items()},
        "p": {k: float(2.0 * scipy_stats.norm.sf(abs(v))) for k, v in stats_.items()},
        "favours": {k: _vuong_direction(v) for k, v in stats_.items()},
        "n": int(n),
        "k_zi": int(k_zi),
        "k_standard": int(k_std),
    }


def _vuong_direction(v: float) -> str:
    if v > VUONG_CRIT:
        return "zero_inflated"
    if v < -VUONG_CRIT:
        return "standard"
    return "neither"


# ── Shared endpoint body ─────────────────────────────────────────────────────


def _coef_rows(names: List[str], params, bse, ratio_key: str, est_key: str):
    rows = []
    for name, b, se in zip(names, params, bse):
        b, se = float(b), float(se)
        z = b / se if np.isfinite(se) and se > 0 else float("nan")
        p = float(2.0 * scipy_stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
        lo, hi = b - Z_CRIT * se, b + Z_CRIT * se
        rows.append({
            "variable": name,
            est_key: b,
            ratio_key: float(np.exp(b)),
            "se": se,
            "z": z,
            "p": p,
            "ci_low": lo,
            "ci_high": hi,
            f"{ratio_key}_ci_low": float(np.exp(lo)),
            f"{ratio_key}_ci_high": float(np.exp(hi)),
        })
    return rows


def _run(req: ZeroInflatedRequest, zinb: bool) -> dict:
    label = "Zero-inflated negative binomial" if zinb else "Zero-inflated Poisson"
    std_label = "negative binomial" if zinb else "Poisson"
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    infl_cols = list(req.inflation_predictors or [])
    _check_columns(df_full, req, infl_cols)
    exposure_col = _validate_exposure(df_full, req)

    all_pred = list(dict.fromkeys(list(req.predictors) + infl_cols))
    impute_cols = [req.outcome] + all_pred + ([exposure_col] if exposure_col else [])
    df = apply_imputation(df_full, impute_cols, req.imputation or "listwise")
    df, cat_warnings = _clean_predictor_categories(df, all_pred)
    y_ser = pd.to_numeric(df[req.outcome], errors="coerce")
    df = df[y_ser.notna()]
    y_ser = y_ser[y_ser.notna()]
    exposure = None
    if exposure_col:
        exp_ser = pd.to_numeric(df[exposure_col], errors="coerce")
        keep = exp_ser.notna()
        df, y_ser, exp_ser = df[keep], y_ser[keep], exp_ser[keep]
        exposure = exp_ser.to_numpy(dtype=float)
        if (exposure <= 0).any():
            raise HTTPException(status_code=422, detail=f"Exposure column '{exposure_col}' must be strictly positive (> 0) after imputation.")
    if len(df) == 0:
        raise HTTPException(status_code=422, detail="No complete rows remain after handling missing values.")
    _validate_counts(y_ser, label)
    n_excluded = n_total - len(df)

    y = y_ser.to_numpy(dtype=float)
    X, x_names, dropped_x = _design(df, list(req.predictors))
    Z, z_names, dropped_z = _design(df, infl_cols)
    n = len(y)
    if n <= X.shape[1] + Z.shape[1] + (1 if zinb else 0):
        raise HTTPException(status_code=422, detail=f"Too few rows ({n}) for the number of parameters in the {label.lower()} model.")

    # Reference fit (also the start values for a hard ZINB fit).
    try:
        std_res = _fit_standard(y, X, exposure, zinb)
    except Exception as exc:
        logger.warning(f"{std_label} reference fit failed: {exc}")
        std_res = None

    zip_start = None
    if zinb:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zip_res = ZeroInflatedPoisson(y, X, exog_infl=Z, exposure=exposure).fit(
                    method="bfgs", maxiter=1000, disp=0)
            p0 = np.asarray(zip_res.params, dtype=float)
            if np.isfinite(p0).all():
                zip_start = np.r_[p0, 0.5]
        except Exception as exc:
            logger.debug(f"ZIP start fit for ZINB failed: {exc}")

    try:
        if zinb:
            model = ZeroInflatedNegativeBinomialP(y, X, exog_infl=Z, exposure=exposure, p=2)
        else:
            model = ZeroInflatedPoisson(y, X, exog_infl=Z, exposure=exposure)
        res, converged = _fit_zero_inflated(model, zinb, zip_start)
    except Exception as exc:
        logger.warning(f"{label} fit raised: {exc}")
        raise HTTPException(status_code=422, detail=f"{label} model did not fit. {_sanitize_model_error(exc, 'fitting')}")
    if res is None:
        raise HTTPException(
            status_code=422,
            detail=f"{label} model did not converge. The zero-inflation part may be unidentifiable "
                   "with these data; try fewer inflation predictors (or intercept only).",
        )

    params = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)
    k_infl, k_main = Z.shape[1], X.shape[1]
    infl_p, main_p = params[:k_infl], params[k_infl:k_infl + k_main]
    infl_se, main_se = bse[:k_infl], bse[k_infl:k_infl + k_main]
    alpha = float(params[-1]) if zinb else None
    alpha_se = float(bse[-1]) if zinb else None

    warns = list(cat_warnings) + constant_column_warnings(dropped_x)
    warns += [w.replace("predictor", "inflation predictor") for w in constant_column_warnings(dropped_z)]
    if not converged:
        warns.append(f"The {label.lower()} optimiser did not report convergence; interpret the estimates with caution.")
    if not np.isfinite(bse).all():
        warns.append("Some standard errors could not be computed (singular Hessian); the model may be over-parameterised for these data.")
    if zinb and alpha is not None and alpha < 1e-3:
        warns.append("The estimated dispersion alpha is close to zero: the counts behave like a zero-inflated Poisson.")

    count_coefs = _coef_rows(x_names, main_p, main_se, "irr", "log_irr")
    infl_coefs = _coef_rows(z_names, infl_p, infl_se, "or", "logit")

    # Observed against model-predicted zeros.
    pi = 1.0 / (1.0 + np.exp(-(Z @ infl_p)))
    mu = _mu(X, main_p, exposure)
    p0_count = _prob_zero_count(mu, alpha)
    p0 = pi + (1.0 - pi) * p0_count
    n_zeros = int((y == 0).sum())
    expected_zeros = float(p0.sum())

    # Vuong test against the standard model.
    vuong = None
    expected_zeros_std = None
    aic_std = bic_std = None
    if std_res is not None and np.isfinite(np.asarray(std_res.params, dtype=float)).all():
        try:
            ll_zi = np.asarray(model.loglikeobs(res.params), dtype=float)
            ll_std = np.asarray(std_res.model.loglikeobs(std_res.params), dtype=float)
            k_std = len(np.asarray(std_res.params))
            vuong = _vuong(ll_zi, ll_std, len(params), k_std)
            sp = np.asarray(std_res.params, dtype=float)
            if zinb:
                mu_std = _mu(X, sp[:-1], exposure)
                expected_zeros_std = float(_prob_zero_count(mu_std, float(sp[-1])).sum())
            else:
                expected_zeros_std = float(_prob_zero_count(_mu(X, sp, exposure), None).sum())
            aic_std, bic_std = float(std_res.aic), float(std_res.bic)
        except Exception as exc:
            logger.warning(f"Vuong test failed: {exc}")
            vuong = None
    if vuong is None:
        warns.append(f"The Vuong test against the standard {std_label} model could not be computed.")
    else:
        vuong["standard_model"] = "Negative binomial" if zinb else "Poisson"
        vuong["preferred"] = vuong["favours"]["aic"]
        vuong["preferred_label"] = {
            "zero_inflated": label,
            "standard": f"Standard {std_label}",
            "neither": "Neither (not distinguishable)",
        }[vuong["preferred"]]
        vuong["note"] = (
            "Vuong (1989) non-nested test on per-observation log-likelihood differences; the AIC-corrected "
            "statistic (as in R pscl::vuong) is used for the preference. |V| > 1.96 is significant at 0.05; "
            "V > 0 favours the zero-inflated model."
        )

    return {
        "model": label,
        "outcome": req.outcome,
        "n": n,
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "aic": float(res.aic),
        "bic": float(res.bic),
        "loglik": float(res.llf),
        "converged": bool(converged),
        "exposure_col": exposure_col,
        "rate_model": bool(exposure_col),
        "inflation_predictors": infl_cols,
        "n_zeros": n_zeros,
        "observed_zero_fraction": n_zeros / n,
        "expected_zeros": expected_zeros,
        "expected_zero_fraction": expected_zeros / n,
        "expected_zeros_standard": expected_zeros_std,
        "standard_model_aic": aic_std,
        "standard_model_bic": bic_std,
        "alpha": alpha,
        "alpha_se": alpha_se,
        "theta": float(1.0 / alpha) if alpha is not None and alpha > 0 else None,
        "vuong": vuong,
        "warnings": warns,
        "count_coefficients": count_coefs,
        "inflation_coefficients": infl_coefs,
        "result_text": _results_text(
            label, std_label, req.outcome, exposure_col, count_coefs, infl_coefs,
            n_zeros, n, expected_zeros, vuong,
        ),
    }


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _results_text(label, std_label, outcome, exposure_col, count_coefs, infl_coefs,
                  n_zeros, n, expected_zeros, vuong) -> str:
    parts = [f"{label} regression was performed to model {outcome}."]
    if exposure_col:
        parts.append(f"{exposure_col} entered the count part as the exposure (follow-up time) offset; IRRs are rate ratios.")
    parts.append(
        f"Observed zeros: {n_zeros}/{n} ({100.0 * n_zeros / n:.1f}%); "
        f"model-predicted zeros: {expected_zeros:.1f} ({100.0 * expected_zeros / n:.1f}%)."
    )
    sig = [c for c in count_coefs if c["variable"] != "const" and np.isfinite(c["p"]) and c["p"] < 0.05]
    if sig:
        items = [
            f'{c["variable"]} (IRR = {c["irr"]:.2f}, 95% CI: {c["irr_ci_low"]:.2f}–{c["irr_ci_high"]:.2f}, p = {_fmt_p(c["p"])})'
            for c in sig
        ]
        parts.append("Count part, significant predictors: " + "; ".join(items) + ".")
    else:
        parts.append("Count part: no predictor reached statistical significance.")
    sig_i = [c for c in infl_coefs if c["variable"] != "const" and np.isfinite(c["p"]) and c["p"] < 0.05]
    if sig_i:
        items = [
            f'{c["variable"]} (OR = {c["or"]:.2f}, 95% CI: {c["or_ci_low"]:.2f}–{c["or_ci_high"]:.2f}, p = {_fmt_p(c["p"])})'
            for c in sig_i
        ]
        parts.append("Zero-inflation part (odds of a structural zero), significant predictors: " + "; ".join(items) + ".")
    elif len(infl_coefs) > 1:
        parts.append("Zero-inflation part: no predictor reached statistical significance.")
    if vuong is not None:
        v = vuong["statistic"]["aic"]
        p = vuong["p"]["aic"]
        parts.append(
            f"Vuong test against the standard {std_label} model (AIC-corrected): V = {v:.2f}, p = {_fmt_p(p)}; "
            f"preferred model: {vuong['preferred_label']}."
        )
    return " ".join(parts)


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/zip")
def zero_inflated_poisson(req: ZeroInflatedRequest):
    return _sanitize(_run(req, zinb=False))


@router.post("/zinb")
def zero_inflated_negbin(req: ZeroInflatedRequest):
    return _sanitize(_run(req, zinb=True))
