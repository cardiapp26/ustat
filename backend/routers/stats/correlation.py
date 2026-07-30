from __future__ import annotations

from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException
from pydantic import AliasChoices, BaseModel, Field
from loguru import logger

from services import store
from services.impute import apply_imputation

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None in dicts/lists."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _clean_corr_matrix(corr: pd.DataFrame) -> dict:
    """Serialise a correlation matrix to a {col: {row: value|None}} dict,
    turning every NaN/Inf into a real JSON null.

    The naive ``corr.where(pd.notnull(corr), None)`` does NOT work: the
    DataFrame keeps float64 dtype, so the ``None`` placeholder is coerced
    back to NaN on the way out. A constant/degenerate column (std=0) makes
    Spearman/Kendall produce NaN, and that leaked NaN then trips the global
    "non-finite float" handler and returns a 500. Casting to object dtype
    first lets the None survive to JSON."""
    return (
        corr.round(4)
        .replace([np.inf, -np.inf], np.nan)
        .astype(object)
        .where(pd.notnull(corr), None)
        .to_dict()
    )


# ── 1. GET Correlation Matrix ──────────────────────────────────────────────────

@router.get("/{session_id}/correlation")
def correlation(session_id: str, method: str = "pearson"):
    df = _get_df(session_id)
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method=method)
    p_values = {}
    for c1 in corr.columns:
        p_values[c1] = {}
        for c2 in corr.columns:
            if c1 == c2:
                p_values[c1][c2] = 0.0
            else:
                pair = num_df[[c1, c2]].dropna()
                if len(pair) < 3 or pair[c1].std() == 0 or pair[c2].std() == 0:
                    p_values[c1][c2] = None
                    continue
                s1, s2 = pair.values.T
                try:
                    if method == "pearson":
                        _, p = scipy_stats.pearsonr(s1, s2)
                    elif method == "spearman":
                        _, p = scipy_stats.spearmanr(s1, s2)
                    else:
                        _, p = scipy_stats.kendalltau(s1, s2)
                    p_values[c1][c2] = float(p)
                except Exception:
                    logger.exception("Correlation matrix p-value calculation failed")
                    p_values[c1][c2] = None
    return {
        "method": method,
        "columns": corr.columns.tolist(),
        "matrix": _clean_corr_matrix(corr),
        "p_values": p_values,
    }


# ── 2. POST Correlation Pair ───────────────────────────────────────────────────

class CorrelationPairRequest(BaseModel):
    session_id: str
    var1: str
    var2: str
    method: Optional[str] = "auto"
    imputation: Optional[str] = "listwise"


@router.post("/correlation_pair")
def correlation_pair(req: CorrelationPairRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    df = apply_imputation(df_full, [req.var1, req.var2], req.imputation or "listwise")
    x = df[req.var1].astype(float).values
    y = df[req.var2].astype(float).values
    n = len(x)
    n_excluded = n_total - n
    if n < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 observations")

    def _assess_normality(arr: np.ndarray) -> dict:
        _n = len(arr)
        skewness = float(scipy_stats.skew(arr))

        if _n < 50:
            stat, p_val = scipy_stats.shapiro(arr)
            return {
                "statistic": float(stat),
                "p": float(p_val),
                "normal": bool(p_val >= 0.05),
                "skewness": skewness,
                "test": "Shapiro-Wilk",
                "bypass": None,
            }

        if _n <= 2000:
            from statsmodels.stats.diagnostic import lilliefors as _lilliefors
            stat, p_val = _lilliefors(arr, dist="norm")
            return {
                "statistic": float(stat),
                "p": float(p_val),
                "normal": bool(p_val >= 0.05),
                "skewness": skewness,
                "test": "Kolmogorov-Smirnov (Lilliefors)",
                "bypass": None,
            }

        if abs(skewness) <= 1.5:
            return {
                "statistic": None,
                "p": None,
                "normal": True,
                "skewness": skewness,
                "test": "Skewness (CLT bypass)",
                "bypass": "clt_skew",
            }

        from statsmodels.stats.diagnostic import lilliefors as _lilliefors
        stat, p_val = _lilliefors(arr, dist="norm")
        return {
            "statistic": float(stat),
            "p": float(p_val),
            "normal": bool(p_val >= 0.05),
            "skewness": skewness,
            "test": "Kolmogorov-Smirnov (Lilliefors)",
            "bypass": None,
        }

    norm1 = _assess_normality(x)
    norm2 = _assess_normality(y)
    normal1 = norm1["normal"]
    normal2 = norm2["normal"]

    _tests_used = {norm1["test"], norm2["test"]}
    if any("Kolmogorov" in t or "Lilliefors" in t for t in _tests_used):
        norm_test_name = "Kolmogorov-Smirnov (Lilliefors)"
    elif "Shapiro-Wilk" in _tests_used:
        norm_test_name = "Shapiro-Wilk"
    else:
        norm_test_name = "Skewness (CLT bypass)"

    method = (req.method or "auto").lower()
    # Anything that was not "pearson" used to fall through to Spearman without
    # a word — including "kendall", which the matrix tab offers and which this
    # endpoint then answered with Spearman's rho: 0.372 against Kendall's tau
    # of 0.242 on this data, a different statistic under the requested name.
    # A misspelling was answered the same way.
    if method not in ("auto", "pearson", "spearman", "kendall"):
        raise HTTPException(
            status_code=422,
            detail=(f"Unknown correlation method '{req.method}'. "
                    "Use auto, pearson, spearman or kendall."),
        )
    if method == "auto":
        method = "pearson" if (normal1 and normal2) else "spearman"

    if method == "pearson":
        r, p = scipy_stats.pearsonr(x, y)
        method_used, label = "pearson", "r"
    elif method == "kendall":
        r, p = scipy_stats.kendalltau(x, y)
        method_used, label = "kendall", "τ"
    else:
        r, p = scipy_stats.spearmanr(x, y)
        method_used, label = "spearman", "ρ"

    if abs(r) < 1.0:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        # qnorm(0.975), not 1.96. The rounded constant shifted the Fisher-z
        # limits in the sixth decimal against R's cor.test.
        z_crit = float(scipy_stats.norm.ppf(0.975))
        ci_low = float(np.tanh(z - z_crit * se))
        ci_high = float(np.tanh(z + z_crit * se))
    else:
        ci_low, ci_high = float(r), float(r)

    scatter_x = x.tolist()
    scatter_y = y.tolist()

    slope, intercept, *_ = scipy_stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    x_mean = x.mean()
    ss_x = np.sum((x - x_mean) ** 2)
    residuals = y - (slope * x + intercept)
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    t_crit = scipy_stats.t.ppf(0.975, df=n - 2)
    ci_band = t_crit * s_err * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_x)

    p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
    strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak" if abs(r) >= 0.2 else "negligible"
    direction = "positive" if r > 0 else "negative"

    return {
        "method": method_used,
        "label": label,
        "n": n,
        "n_excluded": n_excluded,
        "imputation": req.imputation or "listwise",
        "r": float(r),
        "p": float(p),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "normality_test": norm_test_name,
        "normality": {
            req.var1: norm1,
            req.var2: norm2,
        },
        "scatter": {"x": scatter_x, "y": scatter_y},
        "regression_line": {
            "x": x_line.tolist(),
            "y": y_line.tolist(),
            "slope": float(slope),
            "intercept": float(intercept),
        },
        "ci_band": {
            "x": x_line.tolist(),
            "y_upper": (y_line + ci_band).tolist(),
            "y_lower": (y_line - ci_band).tolist(),
        },
        "result_text": (
            f"{'Pearson' if method_used == 'pearson' else 'Spearman'} correlation analysis revealed a "
            f"{strength} {direction} {'correlation' if p < 0.05 else 'but non-significant correlation'} "
            f"between {req.var1} and {req.var2} ({label} = {r:.3f}, 95% CI: {ci_low:.3f}–{ci_high:.3f}, "
            f"p = {p_str}, n = {n})."
        ),
    }


# ── 3. POST Correlation Matrix ─────────────────────────────────────────────────

class CorrelationMatrixRequest(BaseModel):
    session_id: str
    variables: List[str]
    method: Optional[str] = "pearson"
    imputation: Optional[str] = "listwise"


@router.post("/correlation_matrix")
def correlation_matrix_post(req: CorrelationMatrixRequest):
    raw = _get_df(req.session_id)[req.variables].apply(pd.to_numeric, errors="coerce")
    df = apply_imputation(raw, req.variables, req.imputation or "listwise")
    if len(req.variables) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 variables")

    method = req.method or "pearson"
    corr = df.corr(method=method)

    p_matrix: dict = {}
    for c1 in req.variables:
        p_matrix[c1] = {}
        for c2 in req.variables:
            if c1 == c2:
                p_matrix[c1][c2] = None
            else:
                pair = df[[c1, c2]].dropna()
                if len(pair) < 3 or pair[c1].std() == 0 or pair[c2].std() == 0:
                    p_matrix[c1][c2] = None
                    continue
                try:
                    if method == "spearman":
                        _, pv = scipy_stats.spearmanr(pair[c1], pair[c2])
                    elif method == "kendall":
                        _, pv = scipy_stats.kendalltau(pair[c1], pair[c2])
                    else:
                        _, pv = scipy_stats.pearsonr(pair[c1], pair[c2])
                    p_matrix[c1][c2] = float(pv)
                except Exception:
                    logger.exception("Correlation matrix post p-value calculation failed")
                    p_matrix[c1][c2] = None

    warnings = []
    vars_list = req.variables
    for i in range(len(vars_list)):
        for j in range(i + 1, len(vars_list)):
            r_val = corr.loc[vars_list[i], vars_list[j]]
            if abs(r_val) >= 0.70:
                warnings.append({
                    "var1": vars_list[i],
                    "var2": vars_list[j],
                    "r": float(r_val),
                    "severity": "high" if abs(r_val) >= 0.90 else "moderate",
                })

    matrix_dict = {c: {r: (float(corr.loc[r, c])
                           if pd.notna(corr.loc[r, c]) and np.isfinite(corr.loc[r, c])
                           else None)
                        for r in req.variables} for c in req.variables}

    return {
        "method": method,
        "variables": req.variables,
        "n": len(df),
        "matrix": matrix_dict,
        "p_matrix": p_matrix,
        "multicollinearity_warnings": warnings,
    }


# ── 4. ICC ─────────────────────────────────────────────────────────────────────

class ICCRequest(BaseModel):
    session_id: str
    rater1_col: str = Field(
        validation_alias=AliasChoices("rater1_col", "rater1_column"),
    )
    rater2_col: str = Field(
        validation_alias=AliasChoices("rater2_col", "rater2_column"),
    )


@router.post("/icc")
def icc_endpoint(req: ICCRequest):
    df = _get_df(req.session_id).dropna(subset=[req.rater1_col, req.rater2_col])
    r1 = df[req.rater1_col].astype(float).values
    r2 = df[req.rater2_col].astype(float).values
    n = len(r1)
    k = 2
    if n < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 observations")

    grand_mean = np.mean(np.stack([r1, r2]))
    subject_means = (r1 + r2) / 2.0
    rater_means = np.array([r1.mean(), r2.mean()])

    SS_b = k * np.sum((subject_means - grand_mean) ** 2)
    SS_r = n * np.sum((rater_means - grand_mean) ** 2)
    SS_total = np.sum((r1 - grand_mean) ** 2) + np.sum((r2 - grand_mean) ** 2)
    SS_e = SS_total - SS_b - SS_r

    df_b = n - 1
    df_r = k - 1
    df_e = (n - 1) * (k - 1)

    MS_b = SS_b / df_b
    MS_r = SS_r / df_r if df_r > 0 else 0.0
    MS_e = SS_e / df_e if df_e > 0 else 1e-9

    icc_val = (MS_b - MS_e) / (MS_b + (k - 1) * MS_e + k * (MS_r - MS_e) / n)
    icc_val = float(np.clip(icc_val, -1.0, 1.0))

    F_obs = MS_b / MS_e if MS_e > 0 else 0.0

    # The interval has to belong to the same ICC as the estimate.
    #
    # The point estimate above is ICC(A,1): two-way random, ABSOLUTE
    # agreement — the rater mean square is in the denominator, so a
    # systematic offset between raters counts against agreement. The interval
    # used to be the consistency one, (F/F_crit - 1)/(F/F_crit + k - 1),
    # which ignores that term entirely. On the audit frame, where rater B
    # scores 2.1 points high by construction, that reported
    # [0.814, 0.955] where the agreement interval is [0.800, 0.952] — a
    # narrower interval than the estimate supports, in the direction that
    # pushes an ICC over a reporting threshold it has not earned.
    #
    # McGraw & Wong (1996), Table 7, ICC(A,1).
    ci_low, ci_high = -1.0, 1.0
    if MS_e > 0 and 0 < icc_val < 1 and n > 1 and k > 1:
        a = k * icc_val / (n * (1 - icc_val))
        b_coef = 1 + k * icc_val * (n - 1) / (n * (1 - icc_val))
        denom_v = (a * MS_r) ** 2 / df_r + (b_coef * MS_e) ** 2 / df_e
        if denom_v > 0:
            v = (a * MS_r + b_coef * MS_e) ** 2 / denom_v
            F_lower = scipy_stats.f.ppf(0.975, df_b, v)
            F_upper = scipy_stats.f.ppf(0.975, v, df_b)
            spread = k * MS_r + (k * n - k - n) * MS_e
            lo_den = F_lower * spread + n * MS_b
            hi_den = spread + n * F_upper * MS_b
            if lo_den != 0:
                ci_low = float(n * (MS_b - F_lower * MS_e) / lo_den)
            if hi_den != 0:
                ci_high = float(n * (F_upper * MS_b - MS_e) / hi_den)
    ci_low = float(np.clip(ci_low, -1.0, 1.0))
    ci_high = float(np.clip(ci_high, -1.0, 1.0))

    f_p = float(scipy_stats.f.sf(F_obs, df_b, df_e))

    if icc_val >= 0.90:
        interp = "Excellent"
    elif icc_val >= 0.75:
        interp = "Good"
    elif icc_val >= 0.50:
        interp = "Moderate"
    else:
        interp = "Poor"

    means = ((r1 + r2) / 2).tolist()
    diffs = (r1 - r2).tolist()
    mean_diff = float(np.mean(r1 - r2))
    sd_diff = float(np.std(r1 - r2, ddof=1))
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    return {
        "icc": icc_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "f_stat": float(F_obs),
        "f_p": f_p,
        "n": n,
        "interpretation": interp,
        "bland_altman": {
            "means": means,
            "diffs": diffs,
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "loa_upper": float(loa_upper),
            "loa_lower": float(loa_lower),
        },
    }


# ── 5. Cohen's Kappa ───────────────────────────────────────────────────────────

class KappaRequest(BaseModel):
    session_id: str
    rater1_col: str = Field(
        validation_alias=AliasChoices("rater1_col", "rater1_column"),
    )
    rater2_col: str = Field(
        validation_alias=AliasChoices("rater2_col", "rater2_column"),
    )


@router.post("/cohens_kappa")
def cohens_kappa(req: KappaRequest):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix as sk_confusion

    df = _get_df(req.session_id).dropna(subset=[req.rater1_col, req.rater2_col])
    r1 = df[req.rater1_col].astype(str).values
    r2 = df[req.rater2_col].astype(str).values
    n = len(r1)
    if n < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 observations")

    kappa = float(cohen_kappa_score(r1, r2))

    labels = sorted(set(r1) | set(r2))
    cm = sk_confusion(r1, r2, labels=labels)
    po = float(np.trace(cm) / n)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    po_denom = n ** 2
    pe = float(np.sum(row_sums * col_sums) / po_denom) if po_denom > 0 else 0.0
    se_denom = n * (1 - pe) ** 2
    se = float(np.sqrt(po * (1 - po) / se_denom)) if se_denom > 0 else 0.0
    # Kappa is bounded above by 1 by construction. The unclipped normal
    # interval reported an upper limit of 1.011 on this data.
    crit = float(scipy_stats.norm.ppf(0.975))
    ci_low = float(np.clip(kappa - crit * se, -1.0, 1.0))
    ci_high = float(np.clip(kappa + crit * se, -1.0, 1.0))

    # Test against no agreement. The variance under H0 is not the one used
    # for the interval — the interval is around the estimate, the test is
    # around zero — and there was no test at all in the response before.
    se_null = 0.0
    if se_denom > 0:
        row_p = row_sums / n
        col_p = col_sums / n
        se_null = float(np.sqrt(
            (pe + pe ** 2 - float(np.sum(row_p * col_p * (row_p + col_p))))
            / se_denom))
    z_stat = float(kappa / se_null) if se_null > 0 else float("nan")
    p_value = (float(2 * scipy_stats.norm.sf(abs(z_stat)))
               if se_null > 0 else None)

    if kappa >= 0.81:
        interp = "Almost Perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    elif kappa >= 0.0:
        interp = "Slight"
    else:
        interp = "Poor (< chance)"

    return {
        "kappa": kappa,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "se_null": se_null,
        "z": z_stat,
        "p": p_value,
        "n": n,
        "po": po,
        # This used to return `po` — the observed agreement, labelled as the
        # expected one. A reader comparing "observed 0.90" against "expected
        # 0.90" would conclude the raters agreed no better than chance, on
        # data where chance agreement is 0.33 and kappa is 0.85.
        "pe": pe,
        "interpretation": interp,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


# ── 6. Fleiss Kappa ────────────────────────────────────────────────────────────

class FleissKappaRequest(BaseModel):
    session_id: str
    rater_cols: List[str] = Field(
        validation_alias=AliasChoices("rater_cols", "rater_columns"),
    )


@router.post("/fleiss_kappa")
def fleiss_kappa_endpoint(req: FleissKappaRequest):
    from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
    if len(req.rater_cols) < 3:
        raise HTTPException(status_code=422, detail="Fleiss κ requires ≥3 raters. Use Cohen's κ for 2 raters.")
    df = _get_df(req.session_id).dropna(subset=req.rater_cols)
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 subjects with complete ratings across all raters.")

    raters = df[req.rater_cols].astype(str).values
    table, categories = aggregate_raters(raters)

    kappa = float(fleiss_kappa(table, method="fleiss"))
    n_subjects, k_cats = table.shape
    n_raters = int(table.sum(axis=1).mean())
    p_j = table.sum(axis=0) / (n_subjects * n_raters)
    p_e = float(np.sum(p_j ** 2))
    # Fleiss (1971), the variance irr::kappam.fleiss uses:
    #
    #   var = 2 / (S² · N · n · (n-1)) · [ S² - Σ pj qj (qj - pj) ],  S = Σ pj qj
    #
    # The form previously here — pe - (2n-3)pe² + 2(n-2)Σpj³ — is a different
    # published null variance (Fleiss, Nee & Landis 1979) and gives a
    # noticeably different answer: 0.0813 against 0.0792 on this data, a 2.6%
    # wider interval and z = 9.07 against 9.30. Anyone checking the result in
    # R sees the discrepancy, so this matches the reference implementation.
    q_j = 1.0 - p_j
    s_pq = float(np.sum(p_j * q_j))
    if s_pq > 0 and n_subjects > 0 and n_raters > 1:
        var_k = (2.0 / (s_pq ** 2 * n_subjects * n_raters * (n_raters - 1))) * (
            s_pq ** 2 - float(np.sum(p_j * q_j * (q_j - p_j)))
        )
        se = float(np.sqrt(max(var_k, 0.0)))
    else:
        se = 0.0
    # Same two corrections as Cohen's: the exact normal quantile rather than
    # 1.96, and a bound at 1, which kappa cannot exceed.
    crit = float(scipy_stats.norm.ppf(0.975))
    ci_low = float(np.clip(kappa - crit * se, -1.0, 1.0))
    ci_high = float(np.clip(kappa + crit * se, -1.0, 1.0))
    z_stat = float(kappa / se) if se > 0 else float("nan")
    p_value = float(2 * scipy_stats.norm.sf(abs(z_stat))) if se > 0 else None

    if kappa >= 0.81:
        interp = "Almost Perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    elif kappa >= 0.0:
        interp = "Slight"
    else:
        interp = "Poor (< chance)"

    per_category = []
    for j, cat in enumerate(categories):
        p_j_val = float(p_j[j])
        num = float(np.sum(table[:, j] * (table[:, j] - 1)))
        den = float(np.sum(table.sum(axis=1) * (table.sum(axis=1) - 1)))
        p_jbar = num / den if den > 0 else 0.0
        if p_j_val > 0 and p_j_val < 1:
            kj = (p_jbar - p_j_val ** 2) / (p_j_val * (1 - p_j_val))
        else:
            kj = None
        per_category.append({
            "category": str(cat),
            "kappa": round(kj, 4) if kj is not None else None,
            "prevalence": round(p_j_val, 4),
        })

    return {
        "test": "Fleiss' κ",
        # Not rounded on the way out. Four decimals is coarser than the number
        # itself and formatting is the display layer's job.
        "kappa": kappa,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "z": z_stat,
        "p": p_value,
        "n_subjects": int(n_subjects),
        "n_raters": int(n_raters),
        "n_categories": int(k_cats),
        "categories": [str(c) for c in categories],
        "per_category": per_category,
        "interpretation": interp,
        "result_text": (
            f"Fleiss' κ for {n_raters} raters on {n_subjects} subjects = {kappa:.3f} "
            f"(95% CI {ci_low:.3f} to {ci_high:.3f}) — {interp.lower()} agreement (Landis & Koch)."
        ),
    }
