"""Bayesian Statistics Router.

Computes JZS Bayes Factors for t-tests, correlations, and linear regressions
with prior/posterior curve coordinates and equivalent R code.
"""

import math
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp
from scipy.integrate import quad

from services import store
from services.category_health import clean_two_level
from services.stat_utils import sorted_groups
from services.impute import apply_imputation


def _bf_out(bf: float) -> float:
    """A Bayes factor to five significant figures.

    These used to be rounded to four DECIMALS below 10,000, so any BF under
    5e-5 came back as exactly 0.0: BF01 = 1/BF10 for a strong effect read as
    "no evidence for the null at all" instead of 1.3e-07. Significant
    figures keep the magnitude at both ends.
    """
    if not math.isfinite(bf) or bf == 0:
        return bf
    return float(f"{bf:.5g}")

router = APIRouter()


class BayesianRequest(BaseModel):
    session_id: str
    analysis_type: str                  # ttest_one | ttest_ind | ttest_paired | correlation | regression
    outcome: str                        # outcome variable or first variable
    predictor: Optional[str] = None     # grouping variable (ttest) or second variable (correlation)
    predictors: Optional[List[str]] = None # for multiple regression
    mu: float = 0.0                     # test value for one-sample t-test
    imputation: str = "listwise"


def _safe(v):
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ── JZS Bayes Factor Integrands ──

def jzs_integrand_t(g, t, n, v):
    """Integrand for JZS Bayes Factor (t-test) with r = 0.707 prior scale."""
    if g <= 0:
        return 0.0
    # Inv-Gamma(1/2, 0.25) prior for g (equivalent to Cauchy prior on delta)
    prior = 0.5 / np.sqrt(np.pi) * g**(-1.5) * np.exp(-0.25 / g)
    # Likelihood ratio relative to null
    val = (1.0 + t**2 / v) / (1.0 + t**2 / ((1.0 + n * g) * v))
    if val <= 0:
        return 0.0
    return prior * (1.0 + n * g)**(-0.5) * (val**((v + 1) / 2.0))


def jzs_integrand_corr(g, r, n):
    """Integrand for JZS Bayes Factor (correlation)."""
    if g <= 0:
        return 0.0
    prior = 0.5 / np.sqrt(np.pi) * g**(-1.5) * np.exp(-0.25 / g)
    val = 1.0 - (r**2) * g / (1.0 + g)
    if val <= 0:
        return 0.0
    return prior * (1.0 + g)**(-0.5) * (val**(-(n - 1) / 2.0))


def interpret_bf(bf10: float) -> str:
    """Interpret Bayes Factor according to standard Jeffreys/Lee-Wagenmakers rules."""
    if bf10 > 1.0:
        val = bf10
        target = "alternative hypothesis (H₁)"
    else:
        val = 1.0 / bf10 if bf10 > 0 else 99999.0
        target = "null hypothesis (H₀)"
        
    if val >= 150:
        strength = "Extreme"
    elif val >= 30:
        strength = "Very Strong"
    elif val >= 10:
        strength = "Strong"
    elif val >= 3:
        strength = "Moderate"
    elif val >= 1:
        strength = "Anecdotal"
    else:
        strength = "No"
        
    return f"{strength} evidence in favor of the {target}"


# ── Analysis Handlers ──

def run_bayesian_ttest_one(df: pd.DataFrame, req: BayesianRequest):
    x = pd.to_numeric(df[req.outcome], errors="coerce").dropna().values
    n = len(x)
    if n < 3:
        raise HTTPException(400, "Need at least 3 cases for t-test.")
        
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    
    t_stat = (mean - req.mu) / se if se > 0 else 0.0
    v = n - 1
    
    # Compute Bayes Factor
    bf10, _ = quad(jzs_integrand_t, 0, np.inf, args=(t_stat, n, v))
    bf01 = 1.0 / bf10 if bf10 > 0 else float("inf")
    
    # Effect size (Cohen's d)
    cohen_d = (mean - req.mu) / sd if sd > 0 else 0.0
    
    # Generate Prior vs Posterior coordinates
    grid_x = np.linspace(-3.0, 3.0, 200)
    # Cauchy prior: scale = 0.707
    prior_density = 1.0 / (np.pi * (1.0 + (grid_x / 0.707)**2) * 0.707)
    
    # Posterior proportional to prior * likelihood
    posterior = []
    for val in grid_x:
        # non-centrality parameter delta = val * sqrt(n)
        dens = sp.nct.pdf(t_stat, df=v, nc=val * np.sqrt(n))
        posterior.append(dens * (1.0 / (np.pi * (1.0 + (val / 0.707)**2) * 0.707)))
        
    posterior = np.array(posterior)
    area = np.trapz(posterior, grid_x)
    if area > 0:
        posterior_density = (posterior / area).tolist()
    else:
        posterior_density = prior_density.tolist()
        
    plot_coords = [
        {"x": float(grid_x[i]), "prior": float(prior_density[i]), "posterior": float(posterior_density[i])}
        for i in range(len(grid_x))
    ]
    
    r_code = f'library(BayesFactor)\nttestBF(data${req.outcome}, mu = {req.mu})'
    
    return {
        "analysis": "Bayesian One-Sample t-test",
        "statistic_label": "t",
        "statistic_value": round(t_stat, 4),
        "df": v,
        "n": n,
        "effect_size_label": "Cohen's d",
        "effect_size_value": round(cohen_d, 4),
        "bf10": _bf_out(bf10),
        "bf01": _bf_out(bf01),
        "interpretation": interpret_bf(bf10),
        "plot_coords": plot_coords,
        "r_code": r_code
    }


def run_bayesian_ttest_paired(df: pd.DataFrame, req: BayesianRequest):
    if not req.predictor:
        raise HTTPException(400, "Predictor (second paired column) required.")
    pair = df[[req.outcome, req.predictor]].dropna()
    x1 = pd.to_numeric(pair[req.outcome], errors="coerce").values
    x2 = pd.to_numeric(pair[req.predictor], errors="coerce").values
    n = len(x1)
    if n < 3:
        raise HTTPException(400, "Need at least 3 paired cases.")
        
    diff = x1 - x2
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    se_diff = sd_diff / np.sqrt(n)
    
    t_stat = mean_diff / se_diff if se_diff > 0 else 0.0
    v = n - 1
    
    # Compute Bayes Factor
    bf10, _ = quad(jzs_integrand_t, 0, np.inf, args=(t_stat, n, v))
    bf01 = 1.0 / bf10 if bf10 > 0 else float("inf")
    
    # Cohen's d for paired
    cohen_d = mean_diff / sd_diff if sd_diff > 0 else 0.0
    
    # Prior/Posterior Plotly coordinates
    grid_x = np.linspace(-3.0, 3.0, 200)
    prior_density = 1.0 / (np.pi * (1.0 + (grid_x / 0.707)**2) * 0.707)
    
    posterior = []
    for val in grid_x:
        dens = sp.nct.pdf(t_stat, df=v, nc=val * np.sqrt(n))
        posterior.append(dens * (1.0 / (np.pi * (1.0 + (val / 0.707)**2) * 0.707)))
        
    posterior = np.array(posterior)
    area = np.trapz(posterior, grid_x)
    if area > 0:
        posterior_density = (posterior / area).tolist()
    else:
        posterior_density = prior_density.tolist()
        
    plot_coords = [
        {"x": float(grid_x[i]), "prior": float(prior_density[i]), "posterior": float(posterior_density[i])}
        for i in range(len(grid_x))
    ]
    
    r_code = f'library(BayesFactor)\nttestBF(x = data${req.outcome}, y = data${req.predictor}, paired = TRUE)'
    
    return {
        "analysis": "Bayesian Paired t-test",
        "statistic_label": "t",
        "statistic_value": round(t_stat, 4),
        "df": v,
        "n": n,
        "effect_size_label": "Cohen's d_z",
        "effect_size_value": round(cohen_d, 4),
        "bf10": _bf_out(bf10),
        "bf01": _bf_out(bf01),
        "interpretation": interpret_bf(bf10),
        "plot_coords": plot_coords,
        "r_code": r_code
    }


def run_bayesian_ttest_ind(df: pd.DataFrame, req: BayesianRequest):
    if not req.predictor:
        raise HTTPException(400, "Grouping predictor variable required.")
    df_clean = df[[req.outcome, req.predictor]].dropna()
    cleaned = clean_two_level(df_clean[req.predictor])
    df_clean[req.predictor] = cleaned.series
    df_clean = df_clean.dropna()
    groups = sorted_groups(df_clean[req.predictor])
    if len(groups) != 2:
        raise HTTPException(400, f"Grouping variable must have exactly 2 groups. Found: {groups}")
        
    g1 = pd.to_numeric(df_clean[df_clean[req.predictor] == groups[0]][req.outcome], errors="coerce").dropna().values
    g2 = pd.to_numeric(df_clean[df_clean[req.predictor] == groups[1]][req.outcome], errors="coerce").dropna().values
    
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        raise HTTPException(400, "Each group must have at least 2 observations.")
        
    m1, m2 = g1.mean(), g2.mean()
    v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
    
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    se = pooled_sd * np.sqrt(1.0 / n1 + 1.0 / n2)
    
    t_stat = (m1 - m2) / se if se > 0 else 0.0
    v = n1 + n2 - 2
    n_eff = (n1 * n2) / (n1 + n2)
    
    # Compute Bayes Factor using Rouder JZS independent sample t-test formula
    bf10, _ = quad(jzs_integrand_t, 0, np.inf, args=(t_stat, n_eff, v))
    bf01 = 1.0 / bf10 if bf10 > 0 else float("inf")
    
    # Cohen's d (pooled)
    cohen_d = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0
    
    # Prior/Posterior Plotly coordinates
    grid_x = np.linspace(-3.0, 3.0, 200)
    prior_density = 1.0 / (np.pi * (1.0 + (grid_x / 0.707)**2) * 0.707)
    
    posterior = []
    for val in grid_x:
        # non-centrality parameter delta = val * sqrt(N_eff)
        dens = sp.nct.pdf(t_stat, df=v, nc=val * np.sqrt(n_eff))
        posterior.append(dens * (1.0 / (np.pi * (1.0 + (val / 0.707)**2) * 0.707)))
        
    posterior = np.array(posterior)
    area = np.trapz(posterior, grid_x)
    if area > 0:
        posterior_density = (posterior / area).tolist()
    else:
        posterior_density = prior_density.tolist()
        
    plot_coords = [
        {"x": float(grid_x[i]), "prior": float(prior_density[i]), "posterior": float(posterior_density[i])}
        for i in range(len(grid_x))
    ]
    
    r_code = f'library(BayesFactor)\nttestBF(formula = {req.outcome} ~ {req.predictor}, data = data)'
    
    return {
        "analysis": "Bayesian Independent-Samples t-test",
        "statistic_label": "t",
        "statistic_value": round(t_stat, 4),
        "df": v,
        "n": n1 + n2,
        "effect_size_label": "Cohen's d",
        "effect_size_value": round(cohen_d, 4),
        "bf10": _bf_out(bf10),
        "bf01": _bf_out(bf01),
        "interpretation": interpret_bf(bf10),
        "plot_coords": plot_coords,
        "warnings": cleaned.warnings,
        "r_code": r_code
    }


def run_bayesian_correlation(df: pd.DataFrame, req: BayesianRequest):
    if not req.predictor:
        raise HTTPException(400, "Predictor variable required.")
    pair = df[[req.outcome, req.predictor]].dropna()
    x = pd.to_numeric(pair[req.outcome], errors="coerce").values
    y = pd.to_numeric(pair[req.predictor], errors="coerce").values
    n = len(x)
    if n < 4:
        raise HTTPException(400, "Need at least 4 observations for correlation.")
        
    r, _ = sp.pearsonr(x, y)
    
    # Compute Bayes Factor via numerical integration over Cauchy prior
    bf10, _ = quad(jzs_integrand_corr, 0, np.inf, args=(r, n))
    bf01 = 1.0 / bf10 if bf10 > 0 else float("inf")
    
    # Prior/Posterior Plotly coordinates on Pearson correlation rho
    grid_x = np.linspace(-0.99, 0.99, 200)
    # Uniform prior on rho: prior_density = 0.5
    prior_density = np.ones_like(grid_x) * 0.5
    
    # Fisher z approximation for Pearson r distribution
    # z_r ~ N( arctanh(rho), 1/sqrt(n-3) )
    z_r = np.arctanh(r)
    sd = 1.0 / np.sqrt(n - 3)
    
    posterior = []
    for val in grid_x:
        z_rho = np.arctanh(val)
        dens = sp.norm.pdf(z_r, loc=z_rho, scale=sd)
        posterior.append(dens * 0.5)
        
    posterior = np.array(posterior)
    area = np.trapz(posterior, grid_x)
    if area > 0:
        posterior_density = (posterior / area).tolist()
    else:
        posterior_density = prior_density.tolist()
        
    plot_coords = [
        {"x": float(grid_x[i]), "prior": float(prior_density[i]), "posterior": float(posterior_density[i])}
        for i in range(len(grid_x))
    ]
    
    r_code = f'library(BayesFactor)\ncorrelationBF(data${req.outcome}, data${req.predictor})'
    
    return {
        "analysis": "Bayesian Correlation Analysis (Pearson)",
        "statistic_label": "r",
        "statistic_value": round(r, 4),
        "n": n,
        "effect_size_label": "r",
        "effect_size_value": round(r, 4),
        "bf10": _bf_out(bf10),
        "bf01": _bf_out(bf01),
        "interpretation": interpret_bf(bf10),
        "plot_coords": plot_coords,
        "r_code": r_code
    }


def run_bayesian_regression(df: pd.DataFrame, req: BayesianRequest):
    """Bayesian Multiple Linear Regression using BIC approximation.

    BF10 = exp((BIC_null - BIC_alternative) / 2)
    """
    if not req.predictors:
        raise HTTPException(400, "Predictor columns required for regression.")
        
    cols = [req.outcome, *req.predictors]
    df_clean = df[cols].dropna()
    
    # Coerce to numeric
    for c in cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    df_clean = df_clean.dropna()
    
    n = len(df_clean)
    if n < len(req.predictors) + 5:
        raise HTTPException(400, "Too few observations for Bayesian regression.")
        
    # Fit null model (intercept only)
    y = df_clean[req.outcome].values
    rss_null = np.sum((y - y.mean())**2)
    # BIC_null
    k_null = 1
    # BIC = n * ln(RSS / n) + k * ln(n)
    bic_null = n * np.log(rss_null / n) + k_null * np.log(n)
    
    # Fit full model
    X_alt = np.column_stack([np.ones(n), df_clean[req.predictors].values])
    beta, residuals, _, _ = np.linalg.lstsq(X_alt, y, rcond=None)
    rss_alt = np.sum((y - X_alt @ beta)**2)
    k_alt = len(req.predictors) + 1
    bic_alt = n * np.log(rss_alt / n) + k_alt * np.log(n)
    
    # Bayes Factor approximation
    bf10 = np.exp((bic_null - bic_alt) / 2.0)
    bf01 = 1.0 / bf10 if bf10 > 0 else float("inf")
    
    r2 = 1.0 - (rss_alt / rss_null) if rss_null > 0 else 0.0
    
    r_code_preds = " + ".join(req.predictors)
    r_code = f'library(BayesFactor)\nregressionBF({req.outcome} ~ {r_code_preds}, data = data)'
    
    return {
        "analysis": "Bayesian Multiple Linear Regression (BIC Approx)",
        "statistic_label": "R²",
        "statistic_value": round(r2, 4),
        "n": n,
        "effect_size_label": "Adjusted R²",
        "effect_size_value": round(r2 - (1 - r2) * (k_alt - 1) / (n - k_alt), 4),
        "bf10": _bf_out(bf10),
        "bf01": _bf_out(bf01),
        "interpretation": interpret_bf(bf10),
        "plot_coords": [],  # High dimensional model prior/posterior not simplified on 2D
        "r_code": r_code
    }


# ── Route Entry Point ──

@router.post("")
def run_bayesian(req: BayesianRequest):
    df = store.get_filtered(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Resolve the imputation method if needed
    cols_to_check = [req.outcome]
    if req.predictor:
        cols_to_check.append(req.predictor)
    if req.predictors:
        cols_to_check.extend(req.predictors)
        
    missing = [c for c in cols_to_check if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")
        
    if req.analysis_type == "ttest_ind" and req.predictor:
        numeric_cols = [req.outcome]
        df_sub = df[cols_to_check].copy()
        df_sub[req.outcome] = pd.to_numeric(df_sub[req.outcome], errors="coerce")
        df_sub = apply_imputation(df_sub, numeric_cols, req.imputation).dropna(subset=cols_to_check)
    else:
        df_sub = df[cols_to_check].apply(pd.to_numeric, errors="coerce")
        df_sub = apply_imputation(df_sub, cols_to_check, req.imputation)
    
    if req.analysis_type == "ttest_one":
        return run_bayesian_ttest_one(df_sub, req)
    elif req.analysis_type == "ttest_paired":
        return run_bayesian_ttest_paired(df_sub, req)
    elif req.analysis_type == "ttest_ind":
        if not req.predictor:
            raise HTTPException(400, "Grouping predictor variable required.")
        # Grouping variable shouldn't be coerced to numeric for grouping!
        df_sub[req.predictor] = df[req.predictor]
        return run_bayesian_ttest_ind(df_sub, req)
    elif req.analysis_type == "correlation":
        return run_bayesian_correlation(df_sub, req)
    elif req.analysis_type == "regression":
        return run_bayesian_regression(df_sub, req)
    else:
        raise HTTPException(status_code=422, detail=f"Unknown analysis type: {req.analysis_type}")
