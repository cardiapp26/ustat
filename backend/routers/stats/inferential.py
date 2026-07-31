from __future__ import annotations

from typing import Literal, Optional
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException
from pydantic import AliasChoices, BaseModel, Field
from loguru import logger

from services import store
from services.category_health import clean_two_level, rare_level_warnings
from services.impute import apply_imputation
from services.text_generators import (
    methods_ttest_ind, methods_ttest_one, methods_chisquare, methods_fisher, methods_anova,
    results_ttest_ind, results_ttest_one, results_chisquare,
    results_fisher, results_anova,
    r_ttest_ind, r_ttest_one, r_chisquare, r_fisher, r_anova,
)

# Fix possible import issue by falling back
try:
    from services.text_generators import results_chisquare
except ImportError:
    def results_chisquare(ret):
        return ret.get("interpretation", "")

from services.stat_utils import (
    cohen_d, cohen_d_one_sample, eta_squared, omega_squared,
    cramers_v, odds_ratio_effect,
    check_normality, check_equal_variances, group_summary,
    tukey_hsd, games_howell, sorted_groups,
    welch_satterthwaite_df,
)

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


def _two_level_work(df: pd.DataFrame, value_col: str, group_col: str) -> tuple[pd.DataFrame, list]:
    work = df[[value_col, group_col]].copy()
    cleaned = clean_two_level(work[group_col])
    work[group_col] = cleaned.series
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    return work.dropna(), cleaned.warnings


def _clean_crosstab_work(df: pd.DataFrame, row_col: str, col_col: str) -> tuple[pd.DataFrame, list]:
    work = df[[row_col, col_col]].copy()
    warnings = []
    for col in (row_col, col_col):
        cleaned = clean_two_level(work[col])
        work[col] = cleaned.series
        warnings.extend(cleaned.warnings)
    return work.dropna(), warnings


# ── 1. T-Test ──────────────────────────────────────────────────────────────────

class TTestRequest(BaseModel):
    session_id: str
    column: str
    group_column: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("group_column", "group_col"),
    )
    mu: Optional[float] = 0.0
    # "auto" lets Levene pick Student vs Welch; the other two force it.
    method: Literal["auto", "student", "welch"] = "auto"
    # Legacy alias, kept so existing callers keep working. None means "not
    # supplied" — it used to default to True while the handler ignored it
    # entirely and always let Levene decide.
    equal_var: Optional[bool] = None


@router.post("/ttest")
def ttest(req: TTestRequest):
    df = _get_df(req.session_id)
    col = df[req.column].dropna()

    if req.group_column:
        work, warnings = _two_level_work(df, req.column, req.group_column)
        groups = sorted_groups(work[req.group_column])
        if len(groups) != 2:
            raise HTTPException(status_code=400, detail="Group column must have exactly 2 groups")
        g1 = work[work[req.group_column] == groups[0]][req.column].values.astype(float)
        g2 = work[work[req.group_column] == groups[1]][req.column].values.astype(float)

        # Assumption checks
        assumptions = [check_normality(g1, str(groups[0])), check_normality(g2, str(groups[1])),
                       check_equal_variances([g1, g2], [str(groups[0]), str(groups[1])],
                                             on_violation="Welch correction applied")]
        # Precedence: explicit method > legacy equal_var > Levene.
        if req.method == "welch":
            use_welch, chosen_by = True, "request (method)"
        elif req.method == "student":
            use_welch, chosen_by = False, "request (method)"
        elif req.equal_var is not None:
            use_welch, chosen_by = (not req.equal_var), "request (equal_var)"
        else:
            use_welch, chosen_by = (not assumptions[2]["met"]), "auto (Levene)"
        stat, p = scipy_stats.ttest_ind(g1, g2, equal_var=not use_welch)
        sig = bool(p < 0.05)
        es = cohen_d(g1, g2)
        p_str = '<0.001' if p < 0.001 else f'{p:.4f}'

        ret = {
            "test": f"Independent samples t-test{' (Welch)' if use_welch else ''}",
            "group1": str(groups[0]), "n1": len(g1), "mean1": float(g1.mean()),
            "group2": str(groups[1]), "n2": len(g2), "mean2": float(g2.mean()),
            # df must match the test that produced t and p. Welch uses the
            # fractional Satterthwaite df; only the pooled test uses n1+n2-2.
            "t": float(stat), "p": float(p),
            "df": welch_satterthwaite_df(g1, g2) if use_welch else float(len(g1) + len(g2) - 2),
            "df_method": "welch_satterthwaite" if use_welch else "pooled",
            "variance_assumption": "welch" if use_welch else "student",
            "variance_assumption_selected_by": chosen_by,
            "significant": sig,
            "effect_sizes": [es],
            "assumptions": assumptions,
            "summary": {str(groups[0]): group_summary(g1, str(groups[0])),
                        str(groups[1]): group_summary(g2, str(groups[1]))},
            "interpretation": f"{'Significant' if sig else 'No significant'} difference between groups (t = {stat:.3f}, p = {p_str}, Hedges' g = {es['value']:.3f} [{es['magnitude']}])",
            "methods_text": methods_ttest_ind(req.column, req.group_column, use_welch),
            "r_code": r_ttest_ind(req.column, req.group_column, use_welch),
        }
        if warnings:
            ret["warnings"] = warnings
        ret["result_text"] = results_ttest_ind(ret)
        return _sanitize(ret)
    else:
        x = col.astype(float).values
        stat, p = scipy_stats.ttest_1samp(x, req.mu)
        sig = bool(p < 0.05)
        es = cohen_d_one_sample(x, req.mu)
        p_str = '<0.001' if p < 0.001 else f'{p:.4f}'

        ret = {
            "test": "One-sample t-test",
            "mu": req.mu, "n": len(x),
            "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "t": float(stat), "p": float(p), "df": int(len(x) - 1),
            "significant": sig,
            "effect_sizes": [es],
            "assumptions": [check_normality(x, req.column)],
            "summary": {"sample": group_summary(x, "Sample")},
            "interpretation": f"Mean {'differs from' if sig else 'does not differ from'} {req.mu} (t = {stat:.3f}, p = {p_str}, Cohen's d = {es['value']:.3f} [{es['magnitude']}])",
            "methods_text": methods_ttest_one(req.column, req.mu),
            "r_code": r_ttest_one(req.column, req.mu),
        }
        ret["result_text"] = results_ttest_one(ret)
        return _sanitize(ret)


# ── 2. Chi-Square ──────────────────────────────────────────────────────────────

class ChiSqRequest(BaseModel):
    session_id: str
    row_column: str = Field(
        validation_alias=AliasChoices("row_column", "row_col"),
    )
    col_column: str = Field(
        validation_alias=AliasChoices("col_column", "col_col"),
    )


@router.post("/chisquare")
def chisquare(req: ChiSqRequest):
    df = _get_df(req.session_id)
    work, warnings = _clean_crosstab_work(df, req.row_column, req.col_column)
    ct = pd.crosstab(work[req.row_column], work[req.col_column])
    # chi2_contingency answers a one-row or one-column table with dof 0 and
    # p exactly 1.0. Returned as-is that reads as a tested, non-significant
    # association, so a variable that never varies — or that only exists in
    # one group — looked like evidence of no difference.
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{req.row_column}' × '{req.col_column}' has only "
                f"{ct.shape[0]} row(s) and {ct.shape[1]} column(s) after "
                "dropping missing values. An association needs at least two "
                "of each; there is nothing to test."
            ),
        )
    chi2, p, dof, expected = scipy_stats.chi2_contingency(ct)
    sig = bool(p < 0.05)
    n = ct.values.sum()
    min_dim = min(ct.shape)
    es = cramers_v(chi2, n, min_dim)

    effect_sizes = [es]
    if ct.shape == (2, 2):
        effect_sizes.append(odds_ratio_effect(ct.values))

    warnings.extend(rare_level_warnings(work, [req.row_column, req.col_column]))
    if (expected < 5).any():
        warnings.append("Some expected cell counts < 5. Consider Fisher's exact test instead.")
    p_str = '<0.001' if p < 0.001 else f'{p:.4f}'
    ret = {
        "test": "Chi-square test of independence",
        "chi2": float(chi2), "p": float(p), "dof": int(dof), "n": int(n),
        "significant": sig,
        "effect_sizes": effect_sizes,
        "warnings": warnings,
        "crosstab": ct.to_dict(),
        "interpretation": f"{'Significant' if sig else 'No significant'} association (χ²({dof}) = {chi2:.2f}, p = {p_str}, Cramer's V = {es['value']:.3f} [{es['magnitude']}])",
        "methods_text": methods_chisquare(req.row_column, req.col_column),
        "r_code": r_chisquare(req.row_column, req.col_column),
    }
    ret["result_text"] = results_chisquare(ret)
    return _sanitize(ret)


# ── 3. Fisher's Exact Test ─────────────────────────────────────────────────────

class FisherRequest(BaseModel):
    session_id: str
    row_column: str = Field(
        validation_alias=AliasChoices("row_column", "row_col"),
    )
    col_column: str = Field(
        validation_alias=AliasChoices("col_column", "col_col"),
    )


@router.post("/fisher")
def fisher_exact(req: FisherRequest):
    df = _get_df(req.session_id)
    work, warnings = _clean_crosstab_work(df, req.row_column, req.col_column)
    ct = pd.crosstab(work[req.row_column], work[req.col_column])
    if ct.shape != (2, 2):
        raise HTTPException(status_code=400, detail="Fisher's exact test requires a 2×2 table")
    table = ct.values.tolist()
    or_val, p = scipy_stats.fisher_exact(ct.values)
    sig = bool(p < 0.05)
    es = odds_ratio_effect(ct.values)
    p_str = '<0.001' if p < 0.001 else f'{p:.4f}'
    ret = {
        "test": "Fisher's exact test",
        "odds_ratio": float(or_val), "p": float(p),
        "significant": sig,
        "effect_sizes": [es],
        "table": table,
        "row_labels": ct.index.tolist(),
        "col_labels": ct.columns.tolist(),
        "warnings": warnings,
        "interpretation": f"{'Significant' if sig else 'No significant'} association (p = {p_str}, OR = {es['value']:.2f}, 95% CI: {es['ci_low']:.2f}–{es['ci_high']:.2f})",
        "methods_text": methods_fisher(req.row_column, req.col_column),
        "r_code": r_fisher(req.row_column, req.col_column),
    }
    ret["result_text"] = results_fisher(ret)
    return _sanitize(ret)


# ── 4. ANOVA ───────────────────────────────────────────────────────────────────

class AnovaRequest(BaseModel):
    session_id: str
    column: str
    group_column: str = Field(
        validation_alias=AliasChoices("group_column", "group_col"),
    )


@router.post("/anova")
def anova(req: AnovaRequest):
    df = _get_df(req.session_id)
    grp_dict = {str(name): g[req.column].dropna().astype(float).values
                for name, g in df.groupby(req.group_column)}
    group_arrays = list(grp_dict.values())
    group_names = list(grp_dict.keys())
    if len(group_arrays) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 groups")

    # Levene decides the omnibus, the same way it decides Student vs Welch in
    # the two-group case. This used to be the classic equal-variance F no
    # matter what, with only the post-hoc switching to Games-Howell: under
    # unequal variances and unequal group sizes that F is not valid, and the
    # user was handed a robust post-hoc hanging off a non-robust omnibus.
    levene = check_equal_variances(
        group_arrays, group_names,
        on_violation="omnibus switched to Welch's ANOVA; post-hoc uses Games-Howell",
    )
    use_welch = not levene["met"]
    if use_welch:
        from statsmodels.stats.oneway import anova_oneway
        welch = anova_oneway(group_arrays, use_var="unequal")
        stat, p = float(welch.statistic), float(welch.pvalue)
        welch_df_den = float(welch.df[1])
        omnibus_name = "Welch's ANOVA (unequal variances)"
    else:
        stat, p = scipy_stats.f_oneway(*group_arrays)
        welch_df_den = None
        omnibus_name = "One-way ANOVA"
    sig = bool(p < 0.05)
    k = len(group_arrays)
    n_total = sum(len(g) for g in group_arrays)
    df_between = k - 1
    df_within = n_total - k

    ss_within = sum(np.sum((g - g.mean())**2) for g in group_arrays)
    ms_within = ss_within / df_within if df_within > 0 else 1

    es_eta = eta_squared(float(stat), df_between, df_within)
    es_omega = omega_squared(float(stat), df_between, df_within, ms_within)

    assumptions = [levene]
    for name, arr in grp_dict.items():
        assumptions.append(check_normality(arr, name))

    # Post-hoc tests
    posthoc = []
    posthoc_method = None
    if sig and k > 2:
        equal_var = assumptions[0]["met"]
        if equal_var:
            posthoc = tukey_hsd(grp_dict)
            posthoc_method = "Tukey HSD"
        else:
            posthoc = games_howell(grp_dict)
            posthoc_method = "Games-Howell (unequal variances)"

    p_str = '<0.001' if p < 0.001 else f'{p:.4f}'
    group_stats = df.groupby(req.group_column)[req.column].agg(["count", "mean", "std"]).reset_index()
    # Welch's F carries fractional denominator degrees of freedom; reporting
    # it against the pooled within-groups df would understate the tail.
    df_den_report = welch_df_den if welch_df_den is not None else float(df_within)
    ret = {
        "test": omnibus_name,
        "F": float(stat), "p": float(p),
        "df_between": df_between, "df_within": df_within,
        "df_denominator": round(df_den_report, 3),
        "variance_assumption": "welch" if use_welch else "equal",
        "significant": sig,
        "effect_sizes": [es_eta, es_omega],
        "assumptions": assumptions,
        "posthoc": posthoc,
        "posthoc_method": posthoc_method,
        "groups": [
            {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in row.items()}
            for row in group_stats.to_dict(orient="records")
        ],
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} difference across groups "
            f"({omnibus_name}: F({df_between},{df_den_report:.4g}) = {stat:.2f}, "
            f"p = {p_str}, \u03B7\u00B2 = {es_eta['value']:.3f} [{es_eta['magnitude']}])"
        ),
        "methods_text": methods_anova(req.column, req.group_column),
        "r_code": r_anova(req.column, req.group_column),
    }
    ret["result_text"] = results_anova(ret)
    return _sanitize(ret)


# ── 5. TOST Equivalence ────────────────────────────────────────────────────────

class TOSTRequest(BaseModel):
    session_id: str
    column: str
    group_column: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("group_column", "group_col"),
    )
    paired_column: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("paired_column", "paired_col"),
    )
    low: float
    high: float
    mu: Optional[float] = 0.0
    test_type: str = "independent"


@router.post("/tost")
def tost(req: TOSTRequest):
    from statsmodels.stats.weightstats import ttost_ind, ttost_paired

    df = _get_df(req.session_id)
    if req.low >= req.high:
        raise HTTPException(status_code=422, detail="low must be < high")

    test_type = req.test_type
    n1 = n2 = 0
    mean1 = mean2 = std1 = std2 = None
    warnings = []

    if test_type == "independent":
        if not req.group_column:
            raise HTTPException(status_code=422, detail="independent TOST requires group_column.")
        sub, warnings = _two_level_work(df, req.column, req.group_column)
        groups = sorted_groups(sub[req.group_column])
        if len(groups) != 2:
            raise HTTPException(status_code=422, detail=f"group_column must have exactly 2 levels, found {len(groups)}.")
        a = sub.loc[sub[req.group_column] == groups[0], req.column].astype(float)
        b = sub.loc[sub[req.group_column] == groups[1], req.column].astype(float)
        n1, n2 = int(len(a)), int(len(b))
        if n1 < 2 or n2 < 2:
            raise HTTPException(status_code=400, detail="Each group needs ≥2 observations.")
        mean1, mean2 = float(a.mean()), float(b.mean())
        std1, std2 = float(a.std(ddof=1)), float(b.std(ddof=1))
        p_overall, (t_low, p_low, _df_low), (t_high, p_high, _df_high) = ttost_ind(a, b, low=req.low, upp=req.high, usevar="pooled")
        diff = mean1 - mean2
        group_labels = [str(groups[0]), str(groups[1])]
    elif test_type == "paired":
        if not req.paired_column:
            raise HTTPException(status_code=422, detail="paired TOST requires paired_column.")
        sub = df[[req.column, req.paired_column]].dropna()
        a = sub[req.column].astype(float)
        b = sub[req.paired_column].astype(float)
        n1 = n2 = int(len(a))
        if n1 < 2:
            raise HTTPException(status_code=400, detail="Need ≥2 paired observations.")
        mean1, mean2 = float(a.mean()), float(b.mean())
        std1, std2 = float(a.std(ddof=1)), float(b.std(ddof=1))
        p_overall, (t_low, p_low, _df_low), (t_high, p_high, _df_high) = ttost_paired(a, b, low=req.low, upp=req.high)
        diff = mean1 - mean2
        group_labels = [req.column, req.paired_column]
    elif test_type == "one_sample":
        from scipy.stats import t as _t
        col = df[req.column].dropna().astype(float)
        n1 = int(len(col))
        if n1 < 2:
            raise HTTPException(status_code=400, detail="Need ≥2 observations.")
        mean1 = float(col.mean())
        std1 = float(col.std(ddof=1))
        se = std1 / np.sqrt(n1)
        mu = float(req.mu or 0.0)

        t_low = (mean1 - mu - req.low) / se if se > 0 else float("inf")
        p_low = float(_t.sf(t_low, df=n1 - 1))
        t_high = (mean1 - mu - req.high) / se if se > 0 else float("-inf")
        p_high = float(_t.cdf(t_high, df=n1 - 1))
        p_overall = max(p_low, p_high)
        diff = mean1 - mu
        group_labels = [req.column, f"μ₀ = {mu}"]
        warnings = []
    else:
        raise HTTPException(status_code=422, detail=f"Unknown test_type '{test_type}'")

    equivalent = p_overall < 0.05
    interp = (
        f"Equivalence demonstrated (both one-sided p < 0.05) — observed difference is statistically "
        f"within the [{req.low}, {req.high}] margin."
        if equivalent else
        f"Equivalence NOT demonstrated (max of two one-sided p = {p_overall:.4f}) — cannot conclude "
        f"the difference lies within [{req.low}, {req.high}]."
    )
    return _sanitize({
        "test": f"TOST ({test_type})",
        "test_type": test_type,
        "n1": n1, "n2": n2,
        "mean1": mean1, "mean2": mean2,
        "std1": std1, "std2": std2,
        "difference": float(diff),
        "low_bound": float(req.low),
        "high_bound": float(req.high),
        "t_low": float(t_low), "p_low": float(p_low),
        "t_high": float(t_high), "p_high": float(p_high),
        "p_overall": float(p_overall),
        "equivalent": bool(equivalent),
        "group_labels": group_labels,
        "warnings": warnings,
        "interpretation": interp,
        "result_text": (
            f"Two One-Sided Tests for equivalence within [{req.low}, {req.high}]. "
            f"Lower bound test: t = {t_low:.3f}, p = {p_low:.4f}. "
            f"Upper bound test: t = {t_high:.3f}, p = {p_high:.4f}. "
            f"{interp}"
        ),
    })


# ── 6. Non-inferiority ─────────────────────────────────────────────────────────

class NonInferiorityRequest(BaseModel):
    session_id: str
    outcome_col: str = Field(
        validation_alias=AliasChoices("outcome_col", "outcome_column"),
    )
    group_col: str = Field(
        validation_alias=AliasChoices("group_col", "group_column"),
    )
    test_group: Optional[str] = None
    ref_group: Optional[str] = None
    outcome_type: str = "binary"
    effect: str = "RR"
    margin: float = 1.20
    bound: str = "upper"
    alpha: float = 0.05
    imputation: Optional[str] = "listwise"


@router.post("/noninferiority")
def noninferiority(req: NonInferiorityRequest):
    df = _get_df(req.session_id)
    for c in [req.outcome_col, req.group_col]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not (0.0 < req.alpha < 0.5):
        raise HTTPException(status_code=422, detail="One-sided alpha must be in (0, 0.5).")
    if req.bound not in ("upper", "lower"):
        raise HTTPException(status_code=422, detail="bound must be 'upper' or 'lower'.")

    cols = [req.outcome_col, req.group_col]
    work = apply_imputation(df[cols], cols, req.imputation or "listwise").dropna()
    cleaned = clean_two_level(work[req.group_col])
    work[req.group_col] = cleaned.series
    work = work.dropna()
    groups = work[req.group_col].astype(str)
    levels = sorted(groups.unique().tolist())
    if len(levels) != 2:
        raise HTTPException(status_code=422,
            detail=f"Group column must have exactly 2 levels; found {len(levels)}: {levels}")
    test_g = str(req.test_group) if req.test_group is not None else levels[1]
    ref_g = str(req.ref_group) if req.ref_group is not None else levels[0]
    if test_g not in levels or ref_g not in levels or test_g == ref_g:
        raise HTTPException(status_code=422, detail=f"test_group / ref_group must be the 2 distinct levels {levels}.")

    z_one = float(scipy_stats.norm.ppf(1 - req.alpha))
    ci_level = round((1 - 2 * req.alpha) * 100, 1)
    log_margin = None
    is_log = False
    # Set only by the continuous branch, whose interval and p-value are t-based.
    welch_df: Optional[float] = None

    if req.outcome_type == "binary":
        y = pd.to_numeric(work[req.outcome_col], errors="coerce")
        if set(pd.unique(y.dropna())) - {0.0, 1.0}:
            raise HTTPException(status_code=422, detail="Binary outcome must be coded 0/1.")
        t = y[groups == test_g]
        r = y[groups == ref_g]
        n1, n2 = int(t.notna().sum()), int(r.notna().sum())
        x1, x2 = int(t.sum()), int(r.sum())
        p1, p2 = x1 / n1, x2 / n2
        eff = req.effect.upper()
        if eff == "RD":
            est = p1 - p2
            se = float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
            lo, hi = est - z_one * se, est + z_one * se
            est_disp, lo_disp, hi_disp = est, lo, hi
        elif eff == "OR":
            is_log = True
            a, b, c, d = x1, n1 - x1, x2, n2 - x2
            if min(a, b, c, d) == 0:
                a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            le = np.log((a * d) / (b * c))
            se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
            lo, hi = le - z_one * se, le + z_one * se
            est_disp, lo_disp, hi_disp = float(np.exp(le)), float(np.exp(lo)), float(np.exp(hi))
            log_margin = float(np.log(req.margin))
        else:  # RR
            is_log = True
            if x1 == 0 or x2 == 0:
                x1a, x2a = x1 + 0.5, x2 + 0.5
                n1a, n2a = n1 + 0.5, n2 + 0.5
            else:
                x1a, x2a, n1a, n2a = x1, x2, n1, n2
            r1, r2 = x1a / n1a, x2a / n2a
            le = np.log(r1 / r2)
            se = float(np.sqrt((1 - r1) / x1a + (1 - r2) / x2a))
            lo, hi = le - z_one * se, le + z_one * se
            est_disp, lo_disp, hi_disp = float(np.exp(le)), float(np.exp(lo)), float(np.exp(hi))
            log_margin = float(np.log(req.margin))
        detail = {"n_test": n1, "n_ref": n2, "events_test": x1, "events_ref": x2,
                  "p_test": round(p1, 4), "p_ref": round(p2, 4)}
        scale_point = le if is_log else est
        scale_se = se
    elif req.outcome_type == "continuous":
        from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
        y = pd.to_numeric(work[req.outcome_col], errors="coerce")
        t = y[groups == test_g].dropna().values.astype(float)
        r = y[groups == ref_g].dropna().values.astype(float)
        if len(t) < 2 or len(r) < 2:
            raise HTTPException(status_code=422, detail="Each arm needs ≥ 2 observations.")
        cm = CompareMeans(DescrStatsW(t), DescrStatsW(r))
        est = float(t.mean() - r.mean())
        lo, hi = cm.tconfint_diff(alpha=2 * req.alpha, usevar="unequal")
        # Take the standard error from the comparison itself. Recovering it as
        # (hi - lo) / (2 * z_one) divided a t-based interval by a normal
        # quantile, so the two scales were mixed and the SE came out too large
        # — on the report's example 2.628 against a true 2.493.
        se = float(cm.std_meandiff_separatevar)
        welch_df = float(cm.dof_satt())
        est_disp, lo_disp, hi_disp = est, float(lo), float(hi)
        detail = {"n_test": len(t), "n_ref": len(r),
                  "mean_test": round(float(t.mean()), 4), "mean_ref": round(float(r.mean()), 4)}
        scale_point, scale_se = est, se
        eff = "Mean difference"
    else:
        raise HTTPException(status_code=422, detail="outcome_type must be 'binary' or 'continuous'.")

    # One test statistic, one direction of evidence: how far the estimate sits
    # BELOW the margin (upper bound) or ABOVE it (lower bound). Both branches
    # used to build z the other way round and then take norm.cdf, which
    # returns the complement of the p-value: the report's example demonstrated
    # non-inferiority on the CI rule and printed p = 0.99999 next to it, where
    # the correct p is 1.4e-05. The decision and its p contradicted each other
    # on every call.
    m_scale = (log_margin if is_log else req.margin)
    if req.bound == "upper":
        # H0: effect >= margin. Evidence against it is an estimate below it.
        non_inferior = hi_disp < req.margin
        z = (scale_point - m_scale) / scale_se if scale_se > 0 else 0.0
        rule = f"upper {ci_level}% CI bound ({round(hi_disp, 4)}) < margin ({req.margin})"
    else:
        # H0: effect <= margin. Evidence against it is an estimate above it.
        non_inferior = lo_disp > req.margin
        z = (m_scale - scale_point) / scale_se if scale_se > 0 else 0.0
        rule = f"lower {ci_level}% CI bound ({round(lo_disp, 4)}) > margin ({req.margin})"
    # The continuous interval is a Welch t interval, so its p-value is a t
    # tail. The binary effects carry a large-sample normal SE and stay normal.
    if welch_df is not None:
        p_ni = float(scipy_stats.t.cdf(z, welch_df))
        stat_name = "t"
    else:
        p_ni = float(scipy_stats.norm.cdf(z))
        stat_name = "z"

    interp = (
        f"Non-inferiority test ({eff}, {test_g} vs {ref_g}). "
        f"{eff} = {round(est_disp, 4)} ({ci_level}% CI {round(lo_disp, 4)}–{round(hi_disp, 4)}); "
        f"prespecified margin = {req.margin}. One-sided α = {req.alpha} "
        f"(equivalently a two-sided {ci_level}% CI). "
        + (f"Non-inferiority DEMONSTRATED — {rule}, p = {'<0.001' if p_ni < 0.001 else round(p_ni, 4)}."
           if non_inferior else
           f"Non-inferiority NOT demonstrated — {rule} fails (p = {round(p_ni, 4)}).")
    )

    try:
        store.log_action(req.session_id, "noninferiority", {
            "outcome_col": req.outcome_col, "group_col": req.group_col,
            "effect": eff, "margin": req.margin, "bound": req.bound, "alpha": req.alpha,
        })
    except Exception:
        logger.exception("Logging non-inferiority action failed")

    return _sanitize({
        "test": "Non-inferiority (margin) test",
        "outcome_type": req.outcome_type,
        "effect": eff,
        "test_group": test_g, "ref_group": ref_g,
        "estimate": round(est_disp, 5),
        "ci_level": ci_level,
        "ci_low": round(lo_disp, 5),
        "ci_high": round(hi_disp, 5),
        "margin": req.margin,
        "bound": req.bound,
        "alpha_one_sided": req.alpha,
        "non_inferior": bool(non_inferior),
        # Not rounded to 6 dp: a decisive non-inferiority p of 1.5e-14 printed
        # as 0.0, which reads as an impossible certainty rather than a very
        # small number.
        "p_noninferiority": float(p_ni),
        # The statistic behind the p, so the CI rule and the p can be checked
        # against each other rather than taken on trust.
        "statistic": round(float(z), 5),
        "statistic_name": stat_name,
        "df": round(welch_df, 3) if welch_df is not None else None,
        "warnings": cleaned.warnings,
        **detail,
        "assumptions": [
            {"name": "Analysis population", "met": True,
              "detail": "Provide the ITT (or per-protocol) dataset — the test runs on the loaded rows as supplied."},
            {"name": "One-sided ↔ CI correspondence", "met": True,
              "detail": f"One-sided α = {req.alpha} corresponds to a two-sided {ci_level}% CI (regulatory convention)."},
            {"name": "Large-sample normal approx.", "met": (detail.get('n_test', 99) >= 10 and detail.get('n_ref', 99) >= 10),
              "detail": "Wald / log-Wald intervals assume adequate per-arm counts."},
        ],
        "result_text": interp,
        "interpretation": interp,
        "export_rows": [
            ["Metric", "Value"],
            [f"{eff} ({test_g} vs {ref_g})", round(est_disp, 5)],
            [f"{ci_level}% CI", f"{round(lo_disp, 4)} – {round(hi_disp, 4)}"],
            ["Margin", req.margin],
            ["Bound tested", req.bound],
            ["One-sided alpha", req.alpha],
            ["Non-inferior", "Yes" if non_inferior else "No"],
            ["p (non-inferiority)", "<0.000001" if 0 < p_ni < 1e-6 else round(p_ni, 6)],
        ],
        "r_code": (
            "# Non-inferiority: one-sided alpha = "
            f"{req.alpha} ↔ two-sided {ci_level}% CI\n"
            + ("library(epitools); riskratio(table)  # RR + CI\n" if req.effect.upper() == 'RR' and req.outcome_type == 'binary' else "")
            + f"# Non-inferior if {req.bound} {ci_level}% CI bound vs margin {req.margin}."
        ),
    })


# ── 7. Power Analysis ──────────────────────────────────────────────────────────

class PowerRequest(BaseModel):
    test: str
    solve_for: str
    alpha: float = 0.05
    power: Optional[float] = None
    effect_size: Optional[float] = None
    n: Optional[int] = None
    tails: int = 2
    k_groups: int = 3
    ratio: float = 1.0
    p1: Optional[float] = None
    p2: Optional[float] = None
    log_or: Optional[float] = None
    p_event: Optional[float] = None
    r2_other: Optional[float] = 0.0
    hr: Optional[float] = None
    event_rate: Optional[float] = None
    p_exposed: Optional[float] = 0.5


@router.post("/power")
def run_power(req: PowerRequest):
    import numpy as np
    from scipy.stats import norm
    from statsmodels.stats.power import (
        TTestIndPower, TTestPower,
        FTestAnovaPower, NormalIndPower, GofChisquarePower,
    )

    alt = "two-sided" if req.tails == 2 else "larger"
    a   = req.alpha

    # ── Upfront field validation for the statsmodels-backed power classes ──
    # solve_power() raises a bare ValueError ("need exactly one keyword that
    # is None") when the caller omits a required field. Left unhandled, that
    # crashes with a 500 instead of a usable client error. Validate the
    # solve_for-appropriate required fields before dispatching.
    if req.test in ("t_two", "t_one", "anova", "chi2", "proportion"):
        missing = []
        if req.solve_for == "n":
            if req.power is None:
                missing.append("power")
            if req.test != "proportion" and req.effect_size is None:
                missing.append("effect_size")
        elif req.solve_for == "power":
            if req.n is None:
                missing.append("n")
            if req.test != "proportion" and req.effect_size is None:
                missing.append("effect_size")
        elif req.solve_for == "effect_size":
            if req.n is None:
                missing.append("n")
            if req.power is None:
                missing.append("power")
        if missing:
            raise HTTPException(
                400,
                f"Power analysis for test='{req.test}', solve_for='{req.solve_for}' "
                f"requires: {', '.join(missing)}.",
            )

    def _ceil(x): return int(np.ceil(float(x)))

    def _curve(pw_fn, n_end, n_start=4, steps=80):
        pts, step = [], max(1, (n_end - n_start) // steps)
        for n in range(n_start, n_end + 1, step):
            try:
                pwr = float(pw_fn(n))
                if 0 <= pwr <= 1:
                    pts.append({"n": n, "power": round(pwr, 4)})
            except Exception as exc:
                logger.debug("Power curve point failed for n={}: {}", n, exc)
        return pts

    result, label, curve = None, "", []

    # ── Two-sample t-test ──
    if req.test == "t_two":
        ana = TTestIndPower()
        ratio = req.ratio or 1.0
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt)

        if req.solve_for == "n":
            n1 = _ceil(ana.solve_power(effect_size=req.effect_size, nobs1=None, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            result = n1
            label  = f"n₁ = {n1},  n₂ = {_ceil(n1*ratio)},  total N = {n1 + _ceil(n1*ratio)}"
            curve  = _curve(pw, max(n1 * 4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs1=req.n, alpha=a, power=None, ratio=ratio, alternative=alt))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs1=req.n, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            label  = f"Minimum detectable Cohen's d = {result:.4f}"
            d = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=d, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt), max(int(req.n)*4, 100))

    # ── One-sample / paired t-test ──
    elif req.test == "t_one":
        ana = TTestPower()
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, alternative=alt)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, alternative=alt))
            result, label, curve = n, f"n = {n}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, alternative=alt))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, alternative=alt))
            label  = f"Minimum detectable Cohen's d = {result:.4f}"
            d = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=d, nobs=n, alpha=a, power=None, alternative=alt), max(int(req.n)*4, 100))

    # ── One-way ANOVA ──
    elif req.test == "anova":
        # statsmodels' FTestAnovaPower takes `nobs` as the TOTAL sample size.
        # The panel asks the user for participants PER GROUP — its own field
        # help says so and its own label reports "n/group" — and the two were
        # never translated. With four groups that made every answer wrong by a
        # factor of k, in both directions and in the dangerous one:
        #
        #   power for f = 0.25, 52/group, k = 4 → reported 0.275, truth 0.864
        #   n for f = 0.25, 80% power, k = 4    → reported 179/group (716 total)
        #                                         where 45/group (179 total) does
        #
        # So a properly powered study was called hopeless, and a study needing
        # 179 participants was told to recruit 716. Both agree with
        # pwr.anova.test once the per-group count is converted here.
        ana, k = FTestAnovaPower(), req.k_groups

        def pw(n_per_group):
            return ana.solve_power(effect_size=req.effect_size,
                                   nobs=n_per_group * k, alpha=a, power=None,
                                   k_groups=k)

        if req.solve_for == "n":
            total = ana.solve_power(effect_size=req.effect_size, nobs=None,
                                    alpha=a, power=req.power, k_groups=k)
            n = _ceil(total / k)
            result, label, curve = n, f"n/group = {n},  total N = {n*k}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(pw(int(req.n)))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=int(req.n) * k,
                                           alpha=a, power=req.power, k_groups=k))
            label  = f"Minimum detectable Cohen's f = {result:.4f}"
            f_es = result
            curve  = _curve(
                lambda n_per_group: ana.solve_power(
                    effect_size=f_es, nobs=n_per_group * k, alpha=a,
                    power=None, k_groups=k),
                max(int(req.n)*4, 100))

    # ── Pearson correlation (Fisher-z) ──
    elif req.test == "correlation":
        tails = req.tails

        def corr_power(r, n):
            """Cohen (1988) as pwr.r.test implements it.

            Two things were missing from the plain Fisher-z version that was
            here. The critical value came from a normal, where the test that
            will actually be run is a t on n-2 df; and the Fisher z of a
            sample correlation is biased upward by r/(2(n-1)), which the
            transform is normally applied with. Both matter most at the small
            n a power calculation is usually about: at n = 12, r = 0.5 this
            reported 0.378 against pwr.r.test's 0.400, and at n = 85, r = 0.3
            it reported 0.800 — landing exactly on the 80% convention that
            decides whether a study goes ahead — where the correct value is
            0.804.
            """
            if abs(r) >= 1 or n <= 3:
                return float("nan")
            r = abs(r)
            tside = 1 if tails == 1 else 2
            t_c = scipy_stats.t.ppf(1 - a / tside, df=n - 2)
            r_c = np.sqrt(t_c ** 2 / (t_c ** 2 + n - 2))
            z_r = np.arctanh(r) + r / (2 * (n - 1))
            z_rc = np.arctanh(r_c)
            power = norm.cdf((z_r - z_rc) * np.sqrt(n - 3))
            if tails == 2:
                power += norm.cdf((-z_r - z_rc) * np.sqrt(n - 3))
            return float(power)

        def corr_solve_n(r, pwr):
            for n in range(4, 100001):
                if corr_power(r, n) >= pwr:
                    return n
            return 100001

        def corr_solve_r(n, pwr):
            from scipy.optimize import brentq
            try:
                return float(brentq(lambda r: corr_power(r, n) - pwr, 1e-6, 1 - 1e-6))
            except Exception as exc:
                logger.debug("Correlation power root solve failed: {}", exc)
                return None

        r_es = req.effect_size
        if req.solve_for == "n":
            n = corr_solve_n(r_es, req.power)
            result, label = n, f"n = {n}"
            curve = _curve(lambda n: corr_power(r_es, n), max(n*4, 100))
        elif req.solve_for == "power":
            result = corr_power(r_es, req.n)
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(lambda n: corr_power(r_es, n), max(int(req.n)*4, 100))
        else:
            r_sol = corr_solve_r(req.n, req.power)
            result = r_sol
            label  = f"Minimum detectable r = {r_sol:.4f}" if r_sol else "Could not converge"
            if r_sol:
                curve = _curve(lambda n: corr_power(r_sol, n), max(int(req.n)*4, 100))

    # ── Two proportions (Cohen's h) ──
    elif req.test == "proportion":
        ana   = NormalIndPower()
        ratio = req.ratio or 1.0
        p1    = req.p1 if req.p1 is not None else 0.5
        p2    = req.p2 if req.p2 is not None else 0.3
        h_from_p = abs(float(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))))

        if req.solve_for == "effect_size":
            eff = float(ana.solve_power(effect_size=None, nobs1=req.n, alpha=a, power=req.power, ratio=ratio, alternative=alt))
            result = abs(eff)
            label  = f"Minimum detectable Cohen's h = {result:.4f}"
            h_sol = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=h_sol, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt), max(int(req.n)*4, 100))
        else:
            eff = req.effect_size if req.effect_size is not None else h_from_p
            def pw(n): return ana.solve_power(effect_size=eff, nobs1=n, alpha=a, power=None, ratio=ratio, alternative=alt)
            if req.solve_for == "n":
                n1 = _ceil(ana.solve_power(effect_size=eff, nobs1=None, alpha=a, power=req.power, ratio=ratio, alternative=alt))
                result, label, curve = n1, f"n₁ = {n1},  n₂ = {_ceil(n1*ratio)},  total N = {n1+_ceil(n1*ratio)}", _curve(pw, max(n1*4, 100))
            else:
                result = float(ana.solve_power(effect_size=eff, nobs1=req.n, alpha=a, power=None, ratio=ratio, alternative=alt))
                label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
                curve  = _curve(pw, max(int(req.n)*4, 100))

    # ── Logistic regression ──
    elif req.test == "logistic":
        from scipy.stats import norm as _norm

        def _required_n(log_or, p_event, power_target, alpha_target, r2_other, tails):
            z_a = _norm.ppf(1 - alpha_target / (2 if tails == 2 else 1))
            z_b = _norm.ppf(power_target)
            return float(((z_a + z_b) ** 2) / (p_event * (1 - p_event) * (log_or ** 2) * (1 - (r2_other or 0.0))))

        def _power_from_n(log_or, p_event, n_total, alpha_target, r2_other, tails):
            z_a = _norm.ppf(1 - alpha_target / (2 if tails == 2 else 1))
            se = float(np.sqrt(1.0 / (n_total * p_event * (1 - p_event) * (1 - (r2_other or 0.0)))))
            z = abs(log_or) / se if se > 0 else 0.0
            return float(_norm.cdf(z - z_a))

        if req.solve_for == "effect_size":
            log_or = None
        elif not req.log_or and req.effect_size is not None:
            log_or = float(np.log(req.effect_size))
        elif req.log_or is not None:
            log_or = float(req.log_or) if req.log_or <= 0 else float(np.log(req.log_or))
        else:
            raise HTTPException(400, "Logistic power needs 'log_or' (or 'effect_size' = OR).")
        if req.p_event is None or not (0 < req.p_event < 1):
            raise HTTPException(400, "Logistic power needs 'p_event' in (0, 1).")
        r2 = req.r2_other if req.r2_other is not None else 0.0

        def pw(n_): return _power_from_n(log_or, req.p_event, n_, a, r2, req.tails)
        if req.solve_for == "n":
            n_req = _ceil(_required_n(log_or, req.p_event, req.power or 0.8, a, r2, req.tails))
            result, label = n_req, f"n = {n_req}"
            curve = _curve(pw, max(n_req * 4, 200))
        elif req.solve_for == "power":
            result = float(pw(int(req.n)))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 200))
        else:
            from scipy.optimize import brentq
            if req.n is None:
                raise HTTPException(400, "Solving minimum detectable OR needs 'n'.")
            try:
                or_solved = brentq(
                    lambda lo: _power_from_n(lo, req.p_event, int(req.n), a, r2, req.tails) - (req.power or 0.8),
                    1e-3, 5.0,
                )
                result = float(np.exp(or_solved))
                label  = f"Minimum detectable OR = {result:.3f}"
                ll = float(or_solved)
                curve = _curve(lambda n_: _power_from_n(ll, req.p_event, n_, a, r2, req.tails), max(int(req.n)*4, 200))
            except Exception:
                logger.exception("Solving OR in power analysis failed")
                result = None
                label = "Could not solve for OR — try different power / n combination."

    # ── Cox PH ──
    elif req.test == "survival_cox":
        from scipy.stats import norm as _norm

        if req.solve_for != "effect_size" and (req.hr is None or req.hr <= 0):
            raise HTTPException(400, "Cox power needs 'hr' > 0.")
        if req.event_rate is None or not (0 < req.event_rate < 1):
            raise HTTPException(400, "Cox power needs 'event_rate' in (0, 1).")
        p_exp = req.p_exposed if req.p_exposed is not None else 0.5
        if not (0 < p_exp < 1):
            raise HTTPException(400, "'p_exposed' must be in (0, 1).")
        r2 = req.r2_other or 0.0
        log_hr = float(np.log(req.hr)) if req.hr is not None else None

        def _events_required(power_target):
            z_a = _norm.ppf(1 - a / (2 if req.tails == 2 else 1))
            z_b = _norm.ppf(power_target)
            return ((z_a + z_b) ** 2) / (p_exp * (1 - p_exp) * (log_hr ** 2))

        def _n_required(power_target):
            d = _events_required(power_target)
            return d / (req.event_rate * (1 - r2))

        def _power_from_n(n_total):
            z_a = _norm.ppf(1 - a / (2 if req.tails == 2 else 1))
            d = n_total * req.event_rate * (1 - r2)
            if d <= 0:
                return 0.0
            se = float(np.sqrt(1.0 / (d * p_exp * (1 - p_exp))))
            z = abs(log_hr) / se if se > 0 else 0.0
            return float(_norm.cdf(z - z_a))

        def pw(n_): return _power_from_n(n_)
        if req.solve_for == "n":
            n_req = _ceil(_n_required(req.power or 0.8))
            d_req = _ceil(_events_required(req.power or 0.8))
            result, label = n_req, f"n = {n_req} (events = {d_req})"
            curve = _curve(pw, max(n_req * 4, 200))
        elif req.solve_for == "power":
            result = float(pw(int(req.n)))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n) * 4, 200))
        else:
            d_total = int(req.n) * req.event_rate * (1 - r2)
            if d_total > 0:
                z_a = _norm.ppf(1 - a / (2 if req.tails == 2 else 1))
                z_b = _norm.ppf(req.power or 0.8)
                lh = (z_a + z_b) / np.sqrt(d_total * p_exp * (1 - p_exp))
                result = float(np.exp(lh))
                label  = f"Minimum detectable HR = {result:.3f}"
                def _power_from_n_at_lh(n_total):
                    d = n_total * req.event_rate * (1 - r2)
                    if d <= 0:
                        return 0.0
                    se = float(np.sqrt(1.0 / (d * p_exp * (1 - p_exp))))
                    z = abs(lh) / se if se > 0 else 0.0
                    return float(_norm.cdf(z - z_a))
                curve = _curve(_power_from_n_at_lh, max(int(req.n) * 4, 200))
            else:
                result, label = None, "Insufficient events to solve for HR."

    # ── Chi-square GOF ──
    elif req.test == "chi2":
        ana    = GofChisquarePower()
        n_bins = req.k_groups
        def pw(n): return ana.solve_power(effect_size=req.effect_size, nobs=n, alpha=a, power=None, n_bins=n_bins)

        if req.solve_for == "n":
            n = _ceil(ana.solve_power(effect_size=req.effect_size, nobs=None, alpha=a, power=req.power, n_bins=n_bins))
            result, label, curve = n, f"n = {n}", _curve(pw, max(n*4, 100))
        elif req.solve_for == "power":
            result = float(ana.solve_power(effect_size=req.effect_size, nobs=req.n, alpha=a, power=None, n_bins=n_bins))
            label  = f"Power (1-β) = {result:.4f}  ({result*100:.1f}%)"
            curve  = _curve(pw, max(int(req.n)*4, 100))
        else:
            result = float(ana.solve_power(effect_size=None, nobs=req.n, alpha=a, power=req.power, n_bins=n_bins))
            label  = f"Minimum detectable Cohen's w = {result:.4f}"
            w_es = result
            curve  = _curve(lambda n: ana.solve_power(effect_size=w_es, nobs=n, alpha=a, power=None, n_bins=n_bins), max(int(req.n)*4, 100))
    else:
        raise HTTPException(400, f"Unknown test: {req.test}")

    result_text = _power_result_text(req, result)
    return {"result": float(result) if result is not None else None, "label": label, "curve": curve, "result_text": result_text}


def _power_result_text(req, result) -> str:
    if result is None:
        return ""

    a_str = f"{req.alpha}" if req.alpha else "0.05"
    pw_pct = int((req.power or 0.8) * 100)

    # Regression powers have their own effect metric (OR / HR) and design inputs.
    if req.test == "logistic":
        orr = req.log_or if (req.log_or and req.log_or > 0) else None
        or_txt = f"OR = {orr}" if orr else "the specified odds ratio"
        if req.solve_for == "n":
            return (f"You need a total N = {int(np.ceil(result))} for a logistic regression to "
                    f"detect {or_txt} (event prevalence {req.p_event}) with {pw_pct}% power at "
                    f"alpha = {a_str}.")
        if req.solve_for == "power":
            return (f"With N = {req.n} and event prevalence {req.p_event}, your logistic regression "
                    f"has {round(result*100,1)}% power to detect {or_txt} at alpha = {a_str}.")
        return (f"With N = {req.n} at {pw_pct}% power (event prevalence {req.p_event}, alpha = {a_str}), "
                f"the smallest detectable odds ratio is {result:.3f}.")
    if req.test == "survival_cox":
        hr_txt = f"HR = {req.hr}" if req.hr else "the specified hazard ratio"
        if req.solve_for == "n":
            return (f"You need a total N = {int(np.ceil(result))} (event rate {req.event_rate}, "
                    f"exposed fraction {req.p_exposed}) for a Cox model to detect {hr_txt} with "
                    f"{pw_pct}% power at alpha = {a_str}.")
        if req.solve_for == "power":
            return (f"With N = {req.n} (event rate {req.event_rate}), your Cox model has "
                    f"{round(result*100,1)}% power to detect {hr_txt} at alpha = {a_str}.")
        return (f"With N = {req.n} at {pw_pct}% power (event rate {req.event_rate}, alpha = {a_str}), "
                f"the smallest detectable hazard ratio is {result:.3f}.")

    test_names = {
        "t_two": "two-sample t-test", "t_one": "one-sample/paired t-test",
        "anova": "one-way ANOVA", "correlation": "correlation test",
        "proportion": "two-proportion z-test", "chi2": "chi-square test",
    }
    test_name = test_names.get(req.test, req.test)

    if req.solve_for == "n":
        n = int(np.ceil(result))
        total = n * 2 if req.test in ("t_two", "proportion") else n
        ratio_note = f" (ratio {req.ratio}:1)" if hasattr(req, "ratio") and req.ratio and req.ratio != 1 else ""
        return (
            f"You need {n} participants per group{ratio_note} (total N = {total}) "
            f"for a {test_name} to detect an effect size of {req.effect_size} "
            f"with {int((req.power or 0.8) * 100)}% power at alpha = {a_str}."
        )
    elif req.solve_for == "power":
        pwr = round(result * 100, 1)
        return (
            f"With n = {req.n} per group and effect size = {req.effect_size}, "
            f"your {test_name} has {pwr}% power to detect a real effect at alpha = {a_str}. "
            f"{'This exceeds the 80% minimum standard.' if result >= 0.8 else 'This is below the 80% minimum — consider increasing your sample size.'}"
        )
    elif req.solve_for == "effect_size":
        return (
            f"With n = {req.n} per group at {int((req.power or 0.8) * 100)}% power (alpha = {a_str}), "
            f"your {test_name} can detect a minimum effect size of {result:.3f}. "
            f"Effects smaller than this will likely be missed."
        )
    return ""
