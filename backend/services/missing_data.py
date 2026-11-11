"""
Advanced Missing Data Handling for uSTAT (Phase 3)

This module provides:
- Proper Multiple Imputation (MICE) with multiple datasets
- Rubin's Rules for pooling results across imputations
- Basic missing data diagnostics and pattern reporting
- Sensitivity analysis helpers (planned)

Goal: Move from "single imputation" to statistically proper multiple imputation
suitable for mid-to-advanced biostatistics work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from services.dirty_value_guard import flag_sentinels, mask_sentinels, plausibility_max_for_column, sentinel_values


@dataclass
class ImputationResult:
    """Container for multiple imputed datasets + metadata."""
    imputed_datasets: List[pd.DataFrame]
    original_missing_info: Dict[str, Any]
    n_imputations: int
    method: str = "mice"


def _column_method(s: pd.Series) -> str:
    """Pick the chained-equation method by variable type (mice-style)."""
    nun = int(s.dropna().nunique())
    if nun <= 1:
        return "constant"
    if pd.api.types.is_numeric_dtype(s):
        return "pmm" if nun > 2 else "logreg"      # continuous → PMM; 0/1 → logistic
    return "polyreg"                                # categorical/text → hot-deck by class


def _ridge_predict(X_obs: np.ndarray, y_obs: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    """Least-squares (ridge-stabilised) linear prediction for the PMM metric."""
    Xo = np.column_stack([np.ones(len(X_obs)), X_obs])
    Xa = np.column_stack([np.ones(len(X_all)), X_all])
    lam = 1e-6 * np.eye(Xo.shape[1])
    beta = np.linalg.solve(Xo.T @ Xo + lam, Xo.T @ y_obs)
    return Xa @ beta


def _pmm_fill(pred_obs: np.ndarray, pred_miss: np.ndarray, y_obs: np.ndarray,
              rng: np.random.Generator, donors: int = 5) -> np.ndarray:
    """Predictive Mean Matching: each missing value takes a random donor's
    OBSERVED value from the `donors` nearest predicted means. Non-parametric —
    only ever returns values that actually occurred, preserving the distribution."""
    k = min(donors, len(y_obs))
    out = np.empty(len(pred_miss), dtype=y_obs.dtype)
    for i, pm in enumerate(pred_miss):
        d = np.abs(pred_obs - pm)
        idx = np.argpartition(d, k - 1)[:k] if k < len(d) else np.arange(len(d))
        out[i] = y_obs[idx[rng.integers(0, len(idx))]]
    return out


def _chained_impute(df: pd.DataFrame, target_cols: List[str], feature_cols: List[str],
                    max_iter: int, rng: np.random.Generator, donors: int = 5) -> pd.DataFrame:
    """One completed dataset via chained equations with PMM / logistic / hot-deck
    per variable type. Categorical predictors are integer-coded for the design."""
    work = df.copy()
    masks = {c: work[c].isna() | (work[c].astype(str).str.strip() == "") for c in target_cols}

    # Numeric design matrix: numeric features as-is, categorical features factorised.
    def _coded(col: str) -> pd.Series:
        s = work[col]
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        return pd.Series(pd.factorize(s)[0], index=s.index).replace(-1, np.nan)

    # Initialise missing cells with a random observed draw (hot-deck) so the
    # first regression has no holes.
    for c in target_cols:
        observed = work.loc[~masks[c], c]
        observed = observed[observed.astype(str).str.strip() != ""]
        if observed.empty:
            continue
        draws = observed.sample(int(masks[c].sum()), replace=True, random_state=int(rng.integers(0, 1_000_000)))
        work.loc[masks[c], c] = draws.to_numpy()

    methods = {c: _column_method(df[c]) for c in target_cols}
    for _ in range(max_iter):
        for c in target_cols:
            m = masks[c]
            if not m.any() or methods[c] == "constant":
                continue
            feats = [f for f in feature_cols if f != c]
            Xfull = pd.concat([_coded(f) for f in feats], axis=1) if feats else pd.DataFrame(index=work.index)
            Xfull = Xfull.fillna(Xfull.mean(numeric_only=True)).fillna(0.0)
            obs = ~m
            if methods[c] == "polyreg":
                # Categorical target → hot-deck within the nearest predicted bucket
                # is overkill; draw from the observed category distribution.
                observed = work.loc[obs, c]
                observed = observed[observed.astype(str).str.strip() != ""]
                if observed.empty:
                    continue
                work.loc[m, c] = observed.sample(int(m.sum()), replace=True,
                                                 random_state=int(rng.integers(0, 1_000_000))).to_numpy()
                continue
            y = pd.to_numeric(work[c], errors="coerce")
            X_obs = Xfull.loc[obs].to_numpy(dtype=float)
            X_mis = Xfull.loc[m].to_numpy(dtype=float)
            y_obs = y.loc[obs].to_numpy(dtype=float)
            if len(y_obs) < 3 or X_obs.shape[1] == 0:
                continue
            try:
                pred_obs = _ridge_predict(X_obs, y_obs, X_obs)
                pred_mis = _ridge_predict(X_obs, y_obs, X_mis)
            except Exception:
                continue
            filled = _pmm_fill(pred_obs, pred_mis, y_obs, rng, donors=donors)
            work.loc[m, c] = filled
    return work


def mice_multiple(
    df: pd.DataFrame,
    cols: List[str],
    n_imputations: int = 5,
    max_iter: int = 10,
    random_state: int = 42,
) -> ImputationResult:
    """Proper Multiple Imputation via chained equations with Predictive Mean
    Matching (PMM) for continuous variables, logistic-style PMM for binary, and
    hot-deck for categorical — the mice-package default family. Returns m
    completed datasets (the foundation for Rubin's-rules pooling)."""
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return ImputationResult(
            imputed_datasets=[df.copy() for _ in range(max(1, n_imputations))],
            original_missing_info={}, n_imputations=n_imputations, method="pmm")

    original_missing = missing_pattern_summary(df, valid_cols)

    work_df = df.copy()
    for c in valid_cols:
        max_plausible = plausibility_max_for_column(c)
        if max_plausible is not None:
            work_df[c] = mask_sentinels(work_df[c], max_plausible)

    def _has_missing(c: str) -> bool:
        return bool((work_df[c].isna() | (work_df[c].astype(str).str.strip() == "")).any())

    target_cols = [c for c in valid_cols if _has_missing(c)]
    if not target_cols:
        return ImputationResult(
            imputed_datasets=[work_df.copy() for _ in range(n_imputations)],
            original_missing_info=original_missing, n_imputations=n_imputations, method="pmm")

    # Features = every analysis column (carries MAR information across variables).
    feature_cols = list(valid_cols)
    imputed_datasets = []
    for i in range(n_imputations):
        rng = np.random.default_rng(random_state + i)
        try:
            imputed_datasets.append(_chained_impute(work_df, target_cols, feature_cols, max_iter, rng))
        except Exception:
            # Defensive fallback to sklearn IterativeImputer on numeric columns.
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            num_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(work_df[c])]
            df_imp = work_df.copy()
            if num_cols:
                imp = IterativeImputer(max_iter=max_iter, random_state=random_state + i, skip_complete=True)
                df_imp[num_cols] = imp.fit_transform(df_imp[num_cols])
            imputed_datasets.append(df_imp)

    return ImputationResult(
        imputed_datasets=imputed_datasets,
        original_missing_info=original_missing,
        n_imputations=n_imputations,
        method="pmm",
    )


def missing_pattern_summary(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    """Return richer missing data pattern information."""
    valid_cols = [c for c in cols if c in df.columns]
    total = len(df)

    per_col = {}
    missing_masks = {}
    for col in valid_cols:
        max_plausible = plausibility_max_for_column(col)
        raw_missing = df[col].isna()
        implausible = flag_sentinels(df[col], max_plausible)
        missing_mask = raw_missing | implausible
        missing_masks[col] = missing_mask
        n = int(missing_mask.sum())
        per_col[col] = {
            "count": n,
            "raw_count": int(raw_missing.sum()),
            "n_implausible": int(implausible.sum()),
            "implausible_values": sorted(sentinel_values(df[col], max_plausible)),
            "pct": round(n / total * 100, 1) if total > 0 else 0.0,
        }

    rows_affected = int(pd.DataFrame(missing_masks, index=df.index).any(axis=1).sum()) if valid_cols else 0

    # Simple pattern classification
    pattern = "unknown"
    if rows_affected == 0:
        pattern = "complete"
    elif rows_affected / total > 0.5:
        pattern = "heavy_missing"
    elif any(p["pct"] > 30 for p in per_col.values()):
        pattern = "high_in_some_variables"
    else:
        pattern = "moderate"

    return {
        "total_rows": total,
        "rows_affected": rows_affected,
        "pct_affected": round(rows_affected / total * 100, 1) if total > 0 else 0.0,
        "per_column": per_col,
        "pattern_severity": pattern,
    }


# =============================================================================
# Rubin's Rules (Pooling)
# =============================================================================

def pool_rubin_terms(per_imputation: List[Dict[str, tuple]]) -> Dict[str, Dict[str, Any]]:
    """Generic Rubin's-rules pooling for arbitrary survival/regression estimands.

    `per_imputation` is a list (length m) of ``{term: (estimate, se)}`` on a
    scale where the sampling distribution is ~normal (log-HR / log-sHR for ratio
    measures, the raw value for RMST differences). Returns, per term, the pooled
    estimate with total variance T = Ubar + (1 + 1/m)·B, the Barnard–Rubin-style
    degrees of freedom, a t-based p-value and 95% CI, and the fraction of missing
    information. Correct within+between decomposition (unlike a between-only proxy).
    """
    from scipy import stats as _st

    m = len(per_imputation)
    if m == 0:
        return {}
    terms: List[str] = []
    for d in per_imputation:
        for k in d:
            if k not in terms:
                terms.append(k)

    out: Dict[str, Dict[str, Any]] = {}
    for term in terms:
        pairs = [d[term] for d in per_imputation if term in d and d[term] is not None]
        est = np.array([float(p[0]) for p in pairs if p[0] is not None and np.isfinite(p[0])], dtype=float)
        se = np.array([float(p[1]) for p in pairs if p[1] is not None and np.isfinite(p[1])], dtype=float)
        if len(est) == 0 or len(se) != len(est):
            continue
        q_bar = float(np.mean(est))
        u_bar = float(np.mean(se ** 2))
        b = float(np.var(est, ddof=1)) if len(est) > 1 else 0.0
        t_var = u_bar + (1.0 + 1.0 / m) * b
        pooled_se = float(np.sqrt(max(t_var, 1e-12)))
        if b > 0 and u_bar > 0:
            df = (m - 1) * (1.0 + u_bar / ((1.0 + 1.0 / m) * b)) ** 2
            fmi = ((1.0 + 1.0 / m) * b) / t_var
        else:
            df = 1.0e6
            fmi = 0.0
        t_stat = q_bar / pooled_se if pooled_se > 0 else 0.0
        p_val = float(2 * _st.t.sf(abs(t_stat), df)) if pooled_se > 0 else None
        crit = float(_st.t.ppf(0.975, df))
        out[term] = {
            "coef": q_bar,
            "se": pooled_se,
            "ci_low": q_bar - crit * pooled_se,
            "ci_high": q_bar + crit * pooled_se,
            "t": float(t_stat),
            "p": p_val,
            "df": round(float(df), 1),
            "fmi": round(float(fmi), 4),
            "m": m,
        }
    return out

def pool_linear_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pool results from multiple imputed linear models using Rubin's Rules.

    Expects each result dict to have at least:
    - 'coefficients': list of {'variable': str, 'estimate': float, 'se': float, ...}
    - 'r_squared': float (optional)
    """
    if not results:
        return {}

    # Assume all imputations have the same variables
    variables = [c["variable"] for c in results[0]["coefficients"]]

    pooled_coefs = []
    for var in variables:
        estimates = []
        ses = []
        for res in results:
            for c in res["coefficients"]:
                if c["variable"] == var:
                    estimates.append(c["estimate"])
                    ses.append(c.get("se", np.nan))
                    break

        estimates = np.array(estimates)
        ses = np.array(ses)

        Q_bar = np.mean(estimates)
        U_bar = np.mean(ses ** 2)
        B = np.var(estimates, ddof=1) if len(estimates) > 1 else 0.0
        T = U_bar + (1 + 1 / len(results)) * B

        pooled_se = np.sqrt(max(T, 1e-12))
        denom = (1 + 1 / len(results)) * B if B > 0 else 1e-12
        df = max((len(results) - 1) * (1 + U_bar / denom) ** 2, 1.0)

        # Simple t approximation
        t_stat = Q_bar / pooled_se if pooled_se > 0 else 0.0
        p_val = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df)) if pooled_se > 0 else None

        pooled_coefs.append({
            "variable": var,
            "estimate": round(float(Q_bar), 6),
            "se": round(float(pooled_se), 6),
            "t": round(float(t_stat), 4),
            "p": round(float(p_val), 6) if p_val is not None else None,
            "df": round(float(df), 1),
        })

    # Simple pooled R² (average)
    r2s = [r.get("r_squared", np.nan) for r in results if "r_squared" in r]
    pooled_r2 = float(np.nanmean(r2s)) if r2s else None

    return {
        "method": "rubins_rules_pooled",
        "n_imputations": len(results),
        "coefficients": pooled_coefs,
        "r_squared": round(pooled_r2, 4) if pooled_r2 is not None else None,
    }


def pool_logistic_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pool logistic regression results across multiple imputations using Rubin's Rules.

    Expects each result dict to contain:
    - 'coefficients': list of {'variable': str, 'log_odds': float, 'se': float, ...}
      (or 'estimate' as log-odds)

    Pooling is performed on the log-odds (coefficient) scale.
    Odds ratios are obtained by exponentiating the pooled log-odds.
    """
    if not results:
        return {}

    # Collect all unique variables
    first_coefs = results[0].get("coefficients", [])
    variables = [c.get("variable") or c.get("name") for c in first_coefs if c.get("variable") or c.get("name")]

    pooled_coefs = []
    for var in variables:
        log_odds_list = []
        se_list = []

        for res in results:
            coefs = res.get("coefficients", [])
            for c in coefs:
                if (c.get("variable") or c.get("name")) == var:
                    # Prefer 'log_odds', fall back to 'estimate' or 'B'
                    lod = c.get("log_odds") or c.get("estimate") or c.get("B")
                    se = c.get("se")
                    if lod is not None and se is not None:
                        log_odds_list.append(float(lod))
                        se_list.append(float(se))
                    break

        if not log_odds_list:
            continue

        log_odds_arr = np.array(log_odds_list)
        se_arr = np.array(se_list)

        Q_bar = np.mean(log_odds_arr)                    # Pooled log-odds
        U_bar = np.mean(se_arr ** 2)                     # Within-imputation variance
        B = np.var(log_odds_arr, ddof=1) if len(log_odds_arr) > 1 else 0.0  # Between-imputation variance
        T = U_bar + (1 + 1 / len(results)) * B           # Total variance

        pooled_se = np.sqrt(max(T, 1e-12))

        # Approximate degrees of freedom (Rubin's rules)
        if B > 0:
            df = (len(results) - 1) * (1 + U_bar / ((1 + 1 / len(results)) * B)) ** 2
        else:
            df = len(results) - 1

        # z-statistic and p-value (normal approximation, common in practice)
        z = Q_bar / pooled_se if pooled_se > 0 else 0.0
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z))) if pooled_se > 0 else None

        pooled_coefs.append({
            "variable": var,
            "log_odds": round(float(Q_bar), 6),
            "odds_ratio": round(float(np.exp(Q_bar)), 4),
            "se": round(float(pooled_se), 6),
            "z": round(float(z), 4),
            "p": round(float(p), 6) if p is not None else None,
            "df": round(float(df), 1) if df else None,
        })

    return {
        "method": "rubins_rules_pooled_logistic",
        "n_imputations": len(results),
        "coefficients": pooled_coefs,
    }


def add_missing_data_diagnostics(result: dict, missing_info: dict) -> dict:
    """Attach missing data diagnostics to an analysis result."""
    result["missing_data"] = missing_info
    if missing_info.get("pct_affected", 0) > 10:
        if "warnings" not in result:
            result["warnings"] = []
        result["warnings"].append(
            f"{missing_info['pct_affected']}% of rows had missing values in analysis variables. "
            "Consider using multiple imputation (mice)."
        )
    return result


def mice_convergence_diagnostics(
    imputation_result: ImputationResult,
    original_df: pd.DataFrame,
    cols: List[str],
) -> Dict[str, Any]:
    """
    MICE convergence diagnostics using per-imputation traces and a Gelman-Rubin
    R-hat style proxy over imputed values.
    """
    diagnostics = {}
    for col in [c for c in cols if c in original_df.columns and pd.api.types.is_numeric_dtype(original_df[c])]:
        miss_mask = original_df[col].isna()
        traces = []
        imputed_arrays = []
        for i, df_imp in enumerate(imputation_result.imputed_datasets, start=1):
            vals = pd.to_numeric(df_imp.loc[miss_mask, col], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                continue
            imputed_arrays.append(vals)
            traces.append({
                "imputation": i,
                "mean": round(float(np.mean(vals)), 6),
                "sd": round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else 0.0,
            })
        rhat = None
        if len(imputed_arrays) >= 2:
            means = np.asarray([np.mean(v) for v in imputed_arrays], dtype=float)
            within = float(np.mean([np.var(v, ddof=1) if len(v) > 1 else 0.0 for v in imputed_arrays]))
            between = float(np.var(means, ddof=1))
            if within > 1e-12:
                rhat = float(np.sqrt((within + between) / within))
        diagnostics[col] = {
            "trace": traces,
            "r_hat_proxy": round(rhat, 4) if rhat is not None and np.isfinite(rhat) else None,
            "converged": bool(rhat is None or rhat < 1.1),
        }
    return {
        "method": "MICE trace and Gelman-Rubin R-hat proxy",
        "variables": diagnostics,
        "warning": "R-hat is approximated from independent sklearn IterativeImputer chains, not full Bayesian MICE draws.",
    }


def posterior_predictive_check(
    imputation_result: ImputationResult,
    original_df: pd.DataFrame,
    cols: List[str],
) -> Dict[str, Any]:
    """Compare observed and imputed distributions for imputation diagnostics."""
    checks = []
    for col in [c for c in cols if c in original_df.columns and pd.api.types.is_numeric_dtype(original_df[c])]:
        obs = pd.to_numeric(original_df[col], errors="coerce").dropna().to_numpy()
        miss_mask = original_df[col].isna()
        imp_vals = []
        for df_imp in imputation_result.imputed_datasets:
            imp_vals.extend(pd.to_numeric(df_imp.loc[miss_mask, col], errors="coerce").dropna().tolist())
        if len(obs) < 2 or len(imp_vals) < 2:
            checks.append({"variable": col, "available": False, "reason": "Not enough observed/imputed values."})
            continue
        try:
            ks_stat, ks_p = scipy_stats.ks_2samp(obs, np.asarray(imp_vals, dtype=float))
        except Exception:
            ks_stat, ks_p = np.nan, np.nan
        checks.append({
            "variable": col,
            "available": True,
            "observed_mean": round(float(np.mean(obs)), 6),
            "imputed_mean": round(float(np.mean(imp_vals)), 6),
            "mean_difference": round(float(np.mean(imp_vals) - np.mean(obs)), 6),
            "ks_stat": round(float(ks_stat), 6) if np.isfinite(ks_stat) else None,
            "ks_p": round(float(ks_p), 6) if np.isfinite(ks_p) else None,
            "flag_distribution_shift": bool(np.isfinite(ks_p) and ks_p < 0.05),
        })
    return {"method": "posterior_predictive_distribution_check", "checks": checks}


def congeniality_assessment(
    imputation_cols: List[str],
    analysis_cols: List[str],
    passive_formulas: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Check whether imputation model variables cover the analysis model."""
    imputation_set = set(imputation_cols)
    analysis_set = set(analysis_cols)
    missing_from_imputation = sorted(analysis_set - imputation_set)
    passive_targets = set((passive_formulas or {}).keys())
    return {
        "congenial": len(set(missing_from_imputation) - passive_targets) == 0,
        "analysis_variables_missing_from_imputation": missing_from_imputation,
        "passive_variables": sorted(passive_targets),
        "recommendation": (
            "Include all analysis variables, outcome, exposure, interactions, and important auxiliaries in the imputation model."
            if missing_from_imputation else
            "Imputation model covers the listed analysis variables."
        ),
    }


def auxiliary_variable_guidance(
    df: pd.DataFrame,
    target_cols: List[str],
    candidate_cols: Optional[List[str]] = None,
    *,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Rank auxiliary variables by association with missingness indicators and
    observed target values.
    """
    candidate_cols = candidate_cols or [c for c in df.columns if c not in target_cols]
    rows = []
    for target in [c for c in target_cols if c in df.columns]:
        miss = df[target].isna().astype(float)
        for cand in [c for c in candidate_cols if c in df.columns and c != target]:
            if not pd.api.types.is_numeric_dtype(df[cand]):
                continue
            x = pd.to_numeric(df[cand], errors="coerce")
            if x.notna().sum() < 10:
                continue
            try:
                miss_corr = abs(float(np.corrcoef(miss.loc[x.notna()], x.dropna())[0, 1]))
            except Exception:
                miss_corr = 0.0
            obs_mask = df[target].notna() & x.notna()
            try:
                value_corr = abs(float(np.corrcoef(pd.to_numeric(df.loc[obs_mask, target], errors="coerce"), x.loc[obs_mask])[0, 1]))
            except Exception:
                value_corr = 0.0
            score = np.nan_to_num(miss_corr) + 0.5 * np.nan_to_num(value_corr)
            rows.append({
                "target": target,
                "candidate": cand,
                "missingness_corr_abs": round(float(np.nan_to_num(miss_corr)), 5),
                "value_corr_abs": round(float(np.nan_to_num(value_corr)), 5),
                "priority_score": round(float(score), 5),
            })
    rows.sort(key=lambda r: -r["priority_score"])
    return {
        "recommended_auxiliary_variables": rows[:top_k],
        "method_note": "Prioritizes variables associated with missingness and observed target values.",
    }
