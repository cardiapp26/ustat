from __future__ import annotations

from typing import List, Optional
import numpy as np
from scipy import stats as scipy_stats
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from services import store
from services.category_health import clean_two_level, rare_level_warnings
from services.impute import apply_imputation
from services.regression import (
    constant_column_warnings,
    design_with_constant,
    drop_constant_columns,
)
from services.assumptions import (
    check_gee_assumptions_placeholder,
    check_ordinal_assumptions_placeholder,
    add_assumption_warnings_to_result,
)

router = APIRouter()

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _sanitize_model_error(err: Exception, context: str = "model fitting") -> str:
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


# ── Poisson Regression ───────────────────────────────────────────────────────

class PoissonRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/poisson")
def poisson_regression(req: PoissonRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    df, cat_warnings = _clean_predictor_categories(df, req.predictors)
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X, dropped_const = design_with_constant(X)
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if y.isna().all():
        raise HTTPException(status_code=422, detail="Outcome column has no numeric values.")
    if (y.dropna() < 0).any():
        raise HTTPException(status_code=422, detail="Poisson regression requires non-negative integer counts. Negative values found.")
    if (y.dropna() % 1 != 0).any():
        raise HTTPException(status_code=422, detail="Poisson regression requires integer counts. Fractional values found — consider Gamma regression instead.")
    cov_type = "HC3" if req.robust_se else "nonrobust"
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type=cov_type)
    ci = model.conf_int()
    vifs = _compute_vif(X)
    coefs = []
    for var in model.params.index:
        est = float(model.params[var])
        coefs.append({
            "variable": str(var),
            "log_irr": est,
            "irr": float(np.exp(est)),
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
            "irr_ci_low":  float(np.exp(ci.loc[var, 0])),
            "irr_ci_high": float(np.exp(ci.loc[var, 1])),
            "vif": vifs.get(str(var)),
        })
    return _sanitize({
        "model": f"Poisson Regression{' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "aic": float(model.aic),
        "bic": float(model.bic),
        "warnings": cat_warnings + constant_column_warnings(dropped_const),
        "coefficients": coefs,
        "result_text": _poisson_results_text(req.outcome, coefs),
    })


def _poisson_results_text(outcome, coefs):
    sig = [c for c in coefs if c["variable"] != "const" and c["p"] < 0.05]
    parts = [f"Poisson regression was performed to model {outcome}."]
    if sig:
        preds = []
        for c in sig:
            p_s = "<0.001" if c["p"] < 0.001 else f'{c["p"]:.3f}'
            preds.append(f'{c["variable"]} (IRR = {c["irr"]:.2f}, 95% CI: {c["irr_ci_low"]:.2f}–{c["irr_ci_high"]:.2f}, p = {p_s})')
        parts.append("Significant predictors: " + "; ".join(preds) + ".")
    else:
        parts.append("No predictor reached statistical significance.")
    return " ".join(parts)


# ── Gamma GLM ─────────────────────────────────────────────────────────────────

class GammaRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    link: str = "log"
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/gamma")
def gamma_regression(req: GammaRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    df, cat_warnings = _clean_predictor_categories(df, req.predictors)
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X, dropped_const = design_with_constant(X)
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if (y.dropna() <= 0).any():
        raise HTTPException(status_code=422, detail="Gamma regression requires strictly positive outcomes (> 0). Non-positive values found.")

    valid_links = {"log", "identity", "inverse"}
    if req.link and req.link not in valid_links:
        raise HTTPException(status_code=422, detail=f"Invalid link function '{req.link}'. Valid: {valid_links}")
    link_map = {"log": sm.families.links.Log(), "identity": sm.families.links.Identity(), "inverse": sm.families.links.InversePower()}
    family = sm.families.Gamma(link=link_map.get(req.link, sm.families.links.Log()))
    cov_type = "HC3" if req.robust_se else "nonrobust"
    model = sm.GLM(y, X, family=family).fit(cov_type=cov_type)
    ci = model.conf_int()

    vifs = _compute_vif(X)
    # The Gamma dispersion is estimated from the data, not fixed at 1 as it is
    # for binomial and Poisson, so each coefficient is tested against a t on
    # the residual degrees of freedom — that is what R does. statsmodels
    # defaults to a normal z, which is the large-sample limit and is always
    # the smaller p: on the audit frame the intercept came out at 0.000239
    # where R gives 0.000284.
    df_resid = int(model.df_resid)
    coefs = []
    for var in model.params.index:
        b = float(model.params[var])
        se = float(model.bse[var])
        stat = b / se if se > 0 else float("nan")
        p_t = float(2 * scipy_stats.t.sf(abs(stat), df_resid)) if df_resid > 0 \
            else float(model.pvalues[var])
        t_crit = float(scipy_stats.t.ppf(0.975, df_resid)) if df_resid > 0 else 1.96
        coefs.append({
            "variable": str(var),
            "estimate": b,
            "exp_estimate": float(np.exp(b)) if req.link == "log" else None,
            "se": se,
            "t": stat,
            "z": stat,
            "df": df_resid,
            "p": p_t,
            "ci_low": b - t_crit * se,
            "ci_high": b + t_crit * se,
            "vif": vifs.get(str(var)),
        })

    return _sanitize({
        "model": f"Gamma GLM (link={req.link}){' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "link": req.link,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        # The dispersion counts as an estimated parameter; statsmodels leaves
        # it out and lands exactly 2 below R's AIC.
        "aic": float(-2.0 * model.llf + 2.0 * (len(model.params) + 1)),
        "bic": float(-2.0 * model.llf + np.log(int(model.nobs)) * (len(model.params) + 1)),
        "deviance": float(model.deviance),
        "scale": float(model.scale),
        "dispersion": float(model.scale),
        "df_residual": df_resid,
        "dispersion_note": (
            "The Gamma dispersion is estimated, so coefficients are tested "
            "with a t on the residual degrees of freedom. R's AIC additionally "
            "estimates the shape by maximum likelihood, so its value can "
            "differ slightly from this one."
        ),
        "warnings": cat_warnings + constant_column_warnings(dropped_const),
        "coefficients": coefs,
    })


# ── Negative Binomial GLM ─────────────────────────────────────────────────────

class NegBinomRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    robust_se: Optional[bool] = False


@router.post("/negbinom")
def negative_binomial_regression(req: NegBinomRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.outcome] + req.predictors, req.imputation or "listwise")
    df, cat_warnings = _clean_predictor_categories(df, req.predictors)
    n_excluded = n_total - len(df)
    X = pd.get_dummies(df[req.predictors], drop_first=True)
    X, dropped_const = design_with_constant(X)
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    if (y.dropna() < 0).any():
        raise HTTPException(status_code=422, detail="Negative binomial requires non-negative integer counts.")
    if (y.dropna() % 1 != 0).any():
        raise HTTPException(status_code=422, detail="Negative binomial requires integer counts. Fractional values found.")
    cov_type = "HC3" if req.robust_se else "nonrobust"
    # The dispersion is estimated by maximum likelihood jointly with the
    # coefficients, the way R's MASS::glm.nb does it.
    #
    # It used to be estimated from the Pearson residuals of a Poisson fit and
    # then handed to a GLM as if it were KNOWN. Two things went wrong: the
    # moment estimator missed (theta 3.28 against a true 2.78 on the audit
    # frame), and fixing an estimated parameter ignores its own uncertainty,
    # so every standard error came out about 4% too small. That is one-sided:
    # it can only make results look more significant than they are. On the
    # audit frame the age coefficient carried p = 1.0e-05 where the correct
    # value is 2.3e-05.
    try:
        model = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=200)
        if cov_type != "nonrobust":
            model = sm.NegativeBinomial(y, X).fit(
                disp=0, maxiter=200, cov_type=cov_type)
        alpha_est = float(model.params["alpha"])
        alpha_se = float(model.bse["alpha"])
        converged = bool(getattr(model.mle_retvals, "get", lambda *_: True)("converged", True))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Negative binomial model did not converge: {exc}",
        )
    ci = model.conf_int()
    vifs = _compute_vif(X)

    coefs = []
    for var in [v for v in model.params.index if v != "alpha"]:
        b = float(model.params[var])
        coefs.append({
            "variable": str(var),
            "log_irr": b,
            "irr": float(np.exp(b)),
            "se": float(model.bse[var]),
            "z": float(model.tvalues[var]),
            "p": float(model.pvalues[var]),
            "ci_low": float(ci.loc[var, 0]),
            "ci_high": float(ci.loc[var, 1]),
            "irr_ci_low":  float(np.exp(ci.loc[var, 0])),
            "irr_ci_high": float(np.exp(ci.loc[var, 1])),
            "vif": vifs.get(str(var)),
        })

    return _sanitize({
        "model": f"Negative Binomial Regression{' [Robust SE]' if req.robust_se else ''}",
        "outcome": req.outcome,
        "n": int(model.nobs),
        "n_excluded": n_excluded,
        "aic": float(model.aic),
        "bic": float(model.bic),
        # The dispersion the model actually estimated, so the reader can see
        # how far the counts sit from Poisson. It was never reported at all.
        "alpha": alpha_est,
        "alpha_se": alpha_se,
        "theta": float(1.0 / alpha_est) if alpha_est > 0 else None,
        "converged": converged,
        "dispersion_note": (
            "alpha is the negative-binomial dispersion estimated by maximum "
            "likelihood; theta = 1 / alpha is the same quantity in R's "
            "MASS::glm.nb parameterisation. alpha near 0 means the counts are "
            "close to Poisson."
        ),
        "warnings": cat_warnings + constant_column_warnings(dropped_const),
        "coefficients": coefs,
    })


# ── Standalone GEE (Generalized Estimating Equations) ──────────────────────────

class GEERequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    group_col: str
    family: str = "gaussian"       # gaussian | binomial | poisson
    cov_struct: str = "independence"  # independence | exchangeable | ar
    imputation: Optional[str] = "listwise"


@router.post("/gee")
def gee_regression(req: GEERequest):
    from statsmodels.genmod.cov_struct import Independence, Exchangeable, Autoregressive

    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    cols = [req.outcome] + req.predictors + [req.group_col]
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    df, cat_warnings = _clean_predictor_categories(df, req.predictors)
    n_excluded = n_total - len(df)

    if req.group_col not in df.columns:
        raise HTTPException(status_code=422, detail=f"group_col '{req.group_col}' not found")

    y = pd.to_numeric(df[req.outcome], errors="coerce")
    X = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    Xc = sm.add_constant(X, has_constant="add")
    groups = df[req.group_col]

    fam_map = {
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
    }
    if req.family not in fam_map:
        raise HTTPException(status_code=422, detail=f"Unsupported family: {req.family}")

    cov_map = {
        "independence": Independence(),
        "exchangeable": Exchangeable(),
        "ar": Autoregressive(),
    }
    if req.cov_struct not in cov_map:
        raise HTTPException(status_code=422, detail=f"Unsupported cov_struct: {req.cov_struct}")

    cov_fallback = None
    try:
        model = sm.GEE(y, Xc, groups=groups, family=fam_map[req.family],
                       cov_struct=cov_map[req.cov_struct])
        result = model.fit()
    except Exception as e:
        # statsmodels' autoregressive working correlation solves for the
        # correlation by bracketing, and on ordinary balanced data it can fail
        # to find a bracket — "unable to find right bracket" — which used to
        # come back as a flat 422 with no way forward. The working correlation
        # is a nuisance structure: the coefficients are consistent under any
        # of them and only the efficiency changes, so fall back to
        # exchangeable and say so rather than refusing the analysis.
        if req.cov_struct == "ar":
            logger.warning("GEE autoregressive fit failed (%s); falling back to exchangeable", e)
            try:
                model = sm.GEE(y, Xc, groups=groups, family=fam_map[req.family],
                               cov_struct=Exchangeable())
                result = model.fit()
                cov_fallback = (
                    "The autoregressive working correlation could not be "
                    "estimated on this data (statsmodels could not bracket the "
                    "correlation parameter), so an exchangeable structure was "
                    "used instead. GEE coefficients are consistent under any "
                    "working correlation; only the efficiency changes."
                )
            except Exception as e2:
                logger.exception("GEE fit failed after fallback")
                raise HTTPException(status_code=422, detail=_sanitize_model_error(e2, "GEE"))
        else:
            logger.exception("GEE fit failed")
            raise HTTPException(status_code=422, detail=_sanitize_model_error(e, "GEE"))

    coefs = []
    for name in result.params.index:
        # The intercept used to be dropped from the response entirely. It is a
        # fitted parameter like any other, and its absence makes the table
        # impossible to line up against R or any other package.
        coefs.append({
            "variable": "const" if name == "const" else name,
            # Not rounded to 6 decimals — that is coarser than the numbers
            # themselves and turns anything below 5e-7 into a flat zero.
            "estimate": float(result.params[name]),
            "se": float(result.bse[name]),
            "z": float(result.tvalues[name]) if name in result.tvalues.index else None,
            "p": float(result.pvalues[name]) if name in result.pvalues.index else None,
            "ci_low": float(result.conf_int().loc[name, 0]),
            "ci_high": float(result.conf_int().loc[name, 1]),
        })

    n_clusters = int(df[req.group_col].nunique())

    res = {
        "n_obs": int(result.nobs),
        "n_clusters": n_clusters,
        "n_excluded": int(n_excluded),
        "family": req.family,
        "cov_struct": req.cov_struct,
        "cov_struct_used": ("exchangeable" if cov_fallback else req.cov_struct),
        "coefficients": coefs,
        "warnings": (cat_warnings + [cov_fallback]) if cov_fallback else cat_warnings,
        "result_text": _gee_results_text(req.family, req.cov_struct, n_clusters, result.nobs),
    }

    gee_report = check_gee_assumptions_placeholder(req.family, req.cov_struct)
    res = add_assumption_warnings_to_result(res, gee_report)
    return _sanitize(res)


def _gee_results_text(family, cov_struct, n_clusters, n_obs):
    return (
        f"GEE model with {family} family and {cov_struct} correlation structure "
        f"was fit on {n_clusters} clusters ({n_obs} observations)."
    )


# ── Ordinal Logistic Regression ───────────────────────────────────────────────

class OrdinalRequest(BaseModel):
    session_id: str
    outcome: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"


def _brant_test(y_codes: np.ndarray, X: pd.DataFrame) -> dict:
    """Brant (1990) test of the proportional-odds assumption.

    Fits the J−1 binary logits Pr(Y > j) and tests whether the slope vector is
    constant across the cut-points. Returns an omnibus χ² (H0: proportional
    odds holds for every predictor jointly) plus a per-predictor χ². A small
    p-value flags a violation — that predictor's effect differs across the
    ordinal thresholds, so a single shared OR is misleading.

    Reference: Brant R. Assessing proportionality in the proportional odds
    model for ordinal logistic regression. Biometrics 1990;46:1171-8.
    """
    from scipy import stats as _sps

    codes = np.asarray(y_codes, dtype=int)
    Xv = np.asarray(X, dtype=float)
    n, k = Xv.shape
    cut_codes = sorted(set(codes))
    cuts = cut_codes[:-1]  # J−1 cut-points (every level except the top)
    m = len(cuts)
    if m < 2:
        return {"computed": False, "reason": "Need ≥3 ordinal categories for the Brant test."}
    if n <= k + 1:
        return {"computed": False, "reason": "Too few observations for the Brant test."}

    Xc = np.column_stack([np.ones(n), Xv])  # design with intercept
    betas: list[np.ndarray] = []   # slope vectors (intercept dropped)
    pis: list[np.ndarray] = []     # fitted Pr(Y > j)
    A: list[np.ndarray] = []       # (X'W_j X)^{-1}, full (k+1)
    try:
        for c in cuts:
            z = (codes > c).astype(float)
            if z.sum() == 0 or z.sum() == n:
                return {"computed": False, "reason": "A cut-point is degenerate (all one class)."}
            fit = sm.Logit(z, Xc).fit(disp=False, maxiter=100)
            pi = np.clip(np.asarray(fit.predict(Xc), dtype=float), 1e-10, 1 - 1e-10)
            w = pi * (1 - pi)
            XtWX = Xc.T @ (Xc * w[:, None])
            A.append(np.linalg.pinv(XtWX))
            betas.append(np.asarray(fit.params, dtype=float)[1:])
            pis.append(pi)
    except Exception:  # pragma: no cover - numerical failure path
        logger.exception("Brant test: binary logit fit failed")
        return {"computed": False, "reason": "A binary logit at one cut-point failed to converge."}

    beta = np.concatenate(betas)  # length m*k
    # Brant covariance of the stacked slope vector. Diagonal blocks are the
    # ordinary GLM variance; off-diagonal block (j<l) uses W_{jl}=π_l(1−π_j).
    V = np.zeros((m * k, m * k))
    for j in range(m):
        for q in range(m):
            if j <= q:
                w_jq = pis[q] * (1 - pis[j])  # j<q; j==q → π_j(1−π_j) = W_j
                XtWX = Xc.T @ (Xc * w_jq[:, None])
                block = (A[j] @ XtWX @ A[q])[1:, 1:]  # drop intercept row/col
            else:
                block = V[q * k:(q + 1) * k, j * k:(j + 1) * k].T
            V[j * k:(j + 1) * k, q * k:(q + 1) * k] = block

    def _wald(contrast: np.ndarray, b: np.ndarray, cov: np.ndarray):
        Db = contrast @ b
        DVD = contrast @ cov @ contrast.T
        stat = float(Db.T @ np.linalg.pinv(DVD) @ Db)
        dof = int(np.linalg.matrix_rank(contrast))
        return stat, dof, float(_sps.chi2.sf(stat, dof))

    # Omnibus: every slope equal across the m binary fits (compare each to the
    # first). (m−1)·k constraints.
    rows = []
    for r in range(1, m):
        D = np.zeros((k, m * k))
        D[:, 0:k] = np.eye(k)
        D[:, r * k:(r + 1) * k] = -np.eye(k)
        rows.append(D)
    Domni = np.vstack(rows)
    chi2, df, pval = _wald(Domni, beta, V)

    # Per-predictor: just that predictor's m values equal across fits.
    per = []
    for ki, name in enumerate(X.columns):
        rows_k = []
        for r in range(1, m):
            row = np.zeros(m * k)
            row[ki] = 1.0
            row[r * k + ki] = -1.0
            rows_k.append(row)
        Dk = np.vstack(rows_k)
        s_k, df_k, p_k = _wald(Dk, beta, V)
        per.append({
            "variable": str(name),
            "chi2": round(s_k, 4),
            "df": df_k,
            "p": round(p_k, 6),
            "violation": bool(p_k < 0.05),
        })

    return {
        "computed": True,
        "omnibus": {
            "chi2": round(chi2, 4),
            "df": df,
            "p": round(pval, 6),
            "violation": bool(pval < 0.05),
        },
        "by_predictor": per,
        "note": ("Brant test of proportional odds. A significant omnibus χ² (p<0.05) "
                 "means the assumption is violated — at least one predictor's effect "
                 "is not constant across the ordinal thresholds; prefer a partial-"
                 "proportional-odds or multinomial model for those predictors."),
    }


@router.post("/ordinal")
def ordinal_regression(req: OrdinalRequest):
    """Proportional-odds ordinal logistic regression (statsmodels OrderedModel).

    One odds ratio per predictor (shared across the cumulative thresholds) — the
    proportional-odds assumption — rather than a separate effect per category as
    a multinomial model would give. Returns OR (95% CI) + p per predictor, the
    cumulative cut-points, and McFadden's pseudo-R².
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError:
        raise HTTPException(status_code=501, detail="statsmodels OrderedModel unavailable.")

    df_full = _get_df(req.session_id)
    cols = [req.outcome] + req.predictors
    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    df, cat_warnings = _clean_predictor_categories(df, req.predictors)
    n_excluded = len(df_full) - len(df)

    y_raw = df[req.outcome]
    # Order categories: numeric sort when the codes are numeric (e.g. 1/2/3),
    # otherwise lexical. Preserves the clinical ordering for numeric-coded
    # ordinal variables (NYHA, Killip, LDL groups, …).
    uniq = list(pd.Series(y_raw.dropna().unique()))
    num = pd.to_numeric(pd.Series(uniq), errors="coerce")
    if num.notna().all():
        cats = [u for _, u in sorted(zip(num.tolist(), uniq))]
    else:
        cats = sorted(uniq, key=lambda v: str(v))
    if len(cats) < 3:
        raise HTTPException(status_code=422, detail="Ordinal outcome must have at least 3 ordered categories.")

    y = pd.Categorical(y_raw, categories=cats, ordered=True).codes
    X = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    X, dropped_const = drop_constant_columns(X)
    if X.shape[1] == 0:
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")
    X = X.reset_index(drop=True)
    y = pd.Series(y, name=req.outcome).reset_index(drop=True)

    try:
        model = OrderedModel(y, X, distr="logit")
        result = model.fit(method="bfgs", disp=False, maxiter=200)
    except Exception as e:
        logger.exception("Ordinal regression fit failed")
        raise HTTPException(status_code=422, detail=_sanitize_model_error(e, "ordinal logistic"))

    exog_names = list(X.columns)
    conf = result.conf_int()

    def _ci_row(name):
        try:
            row = conf.loc[name]
            return float(row[0]), float(row[1])
        except Exception:
            return None, None

    coefs = []
    for name in exog_names:
        beta = float(result.params[name])
        se = float(result.bse[name])
        p = float(result.pvalues[name])
        lo, hi = _ci_row(name)
        import math
        coefs.append({
            "variable": name,
            # Not rounded on the way out. Six decimals is coarser than the
            # numbers themselves — a p of 1.9142727e-04 was served as
            # 0.000191, and anything below 5e-7 would have arrived as a flat
            # zero. Formatting is the display layer's job.
            "log_odds": beta,
            "se": se,
            "z": (beta / se) if se else None,
            "p": p,
            "odds_ratio": math.exp(beta),
            "or_ci_low": math.exp(lo) if lo is not None else None,
            "or_ci_high": math.exp(hi) if hi is not None else None,
        })

    # Cumulative cut-points (thresholds) — params after the predictor betas.
    thresholds = []
    for name in result.params.index:
        if name not in exog_names:
            thresholds.append({"boundary": str(name), "coef": float(result.params[name])})

    # McFadden pseudo-R² against the intercept-only (category-frequency) model.
    pseudo_r2 = None
    try:
        counts = np.bincount(np.asarray(y), minlength=len(cats)).astype(float)
        probs = counts / counts.sum()
        ll_null = float(np.sum(counts * np.log(probs + 1e-12)))
        if ll_null != 0:
            pseudo_r2 = round(1.0 - (float(result.llf) / ll_null), 4)
    except Exception:
        pseudo_r2 = None

    res = {
        "model": "Ordinal Logistic (proportional odds)",
        "outcome": req.outcome,
        "categories_in_rank_order": [str(c) for c in cats],
        "n": int(len(df)),
        "n_obs": int(len(df)),
        "n_excluded": int(n_excluded),
        "coefficients": coefs,
        "thresholds": thresholds,
        "pseudo_r2": pseudo_r2,
        "aic": float(result.aic) if result.aic is not None else None,
        "bic": float(result.bic) if result.bic is not None else None,
        "brant_proportional_odds": _brant_test(np.asarray(y), X),
        "warnings": cat_warnings + constant_column_warnings(dropped_const),
        "result_text": "",
    }
    res["result_text"] = _ordinal_results_text(len(cats), len(df), res["brant_proportional_odds"])

    ordinal_report = check_ordinal_assumptions_placeholder()
    res = add_assumption_warnings_to_result(res, ordinal_report)
    return _sanitize(res)


def _ordinal_results_text(n_categories, n_obs, brant: dict | None = None):
    text = (
        f"Ordinal logistic regression was performed on {n_categories} ordered categories "
        f"({n_obs} observations)."
    )
    if brant and brant.get("computed") and brant.get("omnibus"):
        om = brant["omnibus"]
        if om["violation"]:
            bad = [b["variable"] for b in brant.get("by_predictor", []) if b["violation"]]
            text += (
                f" The Brant test rejected the proportional-odds assumption "
                f"(χ²={om['chi2']}, df={om['df']}, p={om['p']:.3g})"
            )
            text += f"; flagged predictor(s): {', '.join(bad)}." if bad else "."
        else:
            text += (
                f" The proportional-odds assumption was supported by the Brant test "
                f"(χ²={om['chi2']}, df={om['df']}, p={om['p']:.3g})."
            )
    else:
        text += " Note: the Brant proportional-odds test could not be computed for this fit."
    return text
