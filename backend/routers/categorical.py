"""Categorical tests: binomial, proportion z-tests, McNemar, Cochran Q, Mantel-Haenszel."""
import numpy as np
import pandas as pd
from scipy import stats as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services import store
from services.category_health import clean_two_level
from services.stat_utils import (
    cohens_h, adjust_pvalues, kendalls_w, sorted_groups, sanitize_nonfinite,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _clean_binary_frame(df: pd.DataFrame, columns: List[str]) -> tuple[pd.DataFrame, list]:
    work = df[columns].copy()
    warnings = []
    for col in columns:
        cleaned = clean_two_level(work[col])
        work[col] = cleaned.series
        warnings.extend(cleaned.warnings)
    return work.dropna(), warnings


def _binary_success(col: pd.Series, name: str) -> tuple[int, str, list]:
    """Count successes in a column that must actually be binary.

    These tests are defined on a two-outcome variable. The code used to fall
    back to "count the most frequent value" for anything that was not 0/1, so
    a three-level column was silently collapsed into "the commonest level vs
    everything else" and a proportion test was reported for a variable that
    had no proportion to test. Say no instead, and name what was found.
    """
    obs = col.dropna()
    if obs.empty:
        raise HTTPException(400, f"No non-null values in '{name}'.")
    uniq = list(pd.unique(obs))
    if set(uniq).issubset({0, 1, 0.0, 1.0, True, False}):
        return int((obs.astype(float) == 1).sum()), "1", []
    if len(uniq) != 2:
        shown = ", ".join(repr(str(v)) for v in sorted(map(str, uniq))[:6])
        raise HTTPException(
            422,
            f"'{name}' has {len(uniq)} distinct values ({shown}"
            + (", …" if len(uniq) > 6 else "")
            + "). This test needs a binary variable — recode it to two "
              "outcomes, or use a chi-square test of independence instead.",
        )
    # A genuine two-level variable. Order is fixed so "success" does not
    # change with the sample.
    levels = sorted(map(str, uniq))
    success = levels[1]
    return (
        int((obs.astype(str) == success).sum()),
        success,
        [f"'{name}' is not coded 0/1; '{success}' was counted as the event "
         f"and '{levels[0]}' as the non-event."],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BINOMIAL TEST
# ═══════════════════════════════════════════════════════════════════════════════

class BinomialRequest(BaseModel):
    session_id: str
    column: str
    expected_proportion: float = 0.5
    alpha: float = 0.05


@router.post("/binomial")
def binomial_test(req: BinomialRequest):
    df = _get_df(req.session_id)
    if req.column not in df.columns:
        raise HTTPException(400, f"Column '{req.column}' not found.")
    col = df[req.column].dropna()
    if len(col) < 1:
        raise HTTPException(400, "No non-null values in column.")

    n = len(col)
    k, success_label, binary_warnings = _binary_success(col, req.column)

    result = sp.binomtest(k, n, req.expected_proportion)
    p = float(result.pvalue)
    sig = bool(p < req.alpha)
    observed_prop = k / n
    ps = _p_str(p)

    ci = result.proportion_ci(confidence_level=0.95)
    ci_low = round(float(ci.low), 4)
    ci_high = round(float(ci.high), 4)

    es = cohens_h(observed_prop, req.expected_proportion)

    return {
        "test": "Binomial test",
        "k": k, "n": n, "observed_proportion": round(observed_prop, 4),
        "expected_proportion": req.expected_proportion,
        "p": p,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "ci_proportion": {"low": ci_low, "high": ci_high},
        "summary": {
            "success_value": success_label,
            "k": k, "n": n,
            "observed_proportion": round(observed_prop, 4),
            "expected_proportion": req.expected_proportion,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} difference from expected proportion "
            f"(observed = {observed_prop:.3f}, expected = {req.expected_proportion:.3f}, p = {ps})"
        ),
        "result_text": (
            f"A binomial test compared the observed proportion of '{success_label}' in {req.column} "
            f"({k}/{n} = {observed_prop:.3f}) against the expected proportion of {req.expected_proportion:.3f}. "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(p = {ps}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]). "
            f"Cohen's h = {es['value']:.3f} [{es['magnitude']}]."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["k (successes)", k],
            ["n (total)", n],
            ["Observed proportion", round(observed_prop, 4)],
            ["Expected proportion", req.expected_proportion],
            ["p", round(p, 6)],
            ["95% CI lower", ci_low],
            ["95% CI upper", ci_high],
            ["Cohen's h", es["value"]],
        ],
        "r_code": f"binom.test({k}, {n}, p = {req.expected_proportion})",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ONE-SAMPLE PROPORTION Z-TEST
# ═══════════════════════════════════════════════════════════════════════════════

class OneProportionRequest(BaseModel):
    session_id: str
    column: str
    null_proportion: float = 0.5
    alpha: float = 0.05


@router.post("/one_proportion")
def one_proportion_ztest(req: OneProportionRequest):
    from statsmodels.stats.proportion import proportions_ztest

    df = _get_df(req.session_id)
    if req.column not in df.columns:
        raise HTTPException(400, f"Column '{req.column}' not found.")
    col = df[req.column].dropna()
    if len(col) < 1:
        raise HTTPException(400, "No non-null values in column.")

    n = len(col)
    k, success_label, binary_warnings = _binary_success(col, req.column)

    z_stat, p = proportions_ztest(k, n, value=req.null_proportion)
    z_stat = float(z_stat)
    p = float(p)
    sig = bool(p < req.alpha)
    observed_prop = k / n
    ps = _p_str(p)

    es = cohens_h(observed_prop, req.null_proportion)

    # Wald CI
    se = np.sqrt(observed_prop * (1 - observed_prop) / n) if n > 0 else 0
    ci_low = round(max(0, observed_prop - 1.96 * se), 4)
    ci_high = round(min(1, observed_prop + 1.96 * se), 4)

    return {
        "test": "One-sample proportion z-test",
        "z": round(z_stat, 4), "p": p,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "summary": {
            "success_value": success_label,
            "k": k, "n": n,
            "observed_proportion": round(observed_prop, 4),
            "null_proportion": req.null_proportion,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} difference from null proportion "
            f"(z = {z_stat:.3f}, p = {ps}, h = {es['value']:.3f} [{es['magnitude']}])"
        ),
        "result_text": (
            f"A one-sample proportion z-test compared the observed proportion of '{success_label}' in {req.column} "
            f"({k}/{n} = {observed_prop:.3f}) against the null proportion of {req.null_proportion:.3f}. "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(z = {z_stat:.3f}, p = {ps}). "
            f"95% CI for the proportion: [{ci_low:.3f}, {ci_high:.3f}]. "
            f"Cohen's h = {es['value']:.3f} [{es['magnitude']}]."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["z", round(z_stat, 4)],
            ["p", round(p, 6)],
            ["k (successes)", k],
            ["n (total)", n],
            ["Observed proportion", round(observed_prop, 4)],
            ["Null proportion", req.null_proportion],
            ["95% CI lower", ci_low],
            ["95% CI upper", ci_high],
            ["Cohen's h", es["value"]],
        ],
        "r_code": f"prop.test({k}, {n}, p = {req.null_proportion})",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TWO-SAMPLE PROPORTION Z-TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TwoProportionsRequest(BaseModel):
    session_id: str
    column: str
    group_column: str
    alpha: float = 0.05


@router.post("/two_proportions")
def two_proportions_ztest(req: TwoProportionsRequest):
    from statsmodels.stats.proportion import proportions_ztest

    df = _get_df(req.session_id)
    for c in [req.column, req.group_column]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.column, req.group_column]].copy()
    cleaned_group = clean_two_level(sub[req.group_column])
    sub[req.group_column] = cleaned_group.series
    sub = sub.dropna()
    groups = sorted_groups(sub[req.group_column])
    if len(groups) != 2:
        raise HTTPException(400, f"Group column must have exactly 2 groups, found {len(groups)}.")

    g1_data = sub[sub[req.group_column] == groups[0]][req.column]
    g2_data = sub[sub[req.group_column] == groups[1]][req.column]

    all_vals = sub[req.column]
    _, success_label, binary_warnings = _binary_success(all_vals, req.column)
    if success_label == "1":
        k1 = int((g1_data.astype(float) == 1).sum())
        k2 = int((g2_data.astype(float) == 1).sum())
    else:
        k1 = int((g1_data.astype(str) == success_label).sum())
        k2 = int((g2_data.astype(str) == success_label).sum())

    n1, n2 = len(g1_data), len(g2_data)
    p1, p2 = k1 / n1 if n1 > 0 else 0, k2 / n2 if n2 > 0 else 0

    z_stat, p = proportions_ztest([k1, k2], [n1, n2])
    z_stat = float(z_stat)
    p = float(p)
    sig = bool(p < req.alpha)
    ps = _p_str(p)

    es = cohens_h(p1, p2)

    return {
        "test": "Two-sample proportion z-test",
        "z": round(z_stat, 4), "p": p,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "summary": {
            str(groups[0]): {"n": n1, "k": k1, "proportion": round(p1, 4)},
            str(groups[1]): {"n": n2, "k": k2, "proportion": round(p2, 4)},
            "success_value": success_label,
        },
        "warnings": cleaned_group.warnings,
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} difference between proportions "
            f"({groups[0]}: {p1:.3f} vs {groups[1]}: {p2:.3f}, z = {z_stat:.3f}, p = {ps}, "
            f"h = {es['value']:.3f} [{es['magnitude']}])"
        ),
        "result_text": (
            f"A two-sample proportion z-test compared '{success_label}' rates between "
            f"{groups[0]} ({k1}/{n1} = {p1:.3f}) and {groups[1]} ({k2}/{n2} = {p2:.3f}). "
            f"The difference was {'statistically significant' if sig else 'not statistically significant'} "
            f"(z = {z_stat:.3f}, p = {ps}). "
            f"Cohen's h = {es['value']:.3f} [{es['magnitude']}]."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["z", round(z_stat, 4)],
            ["p", round(p, 6)],
            [f"{groups[0]}: k/n", f"{k1}/{n1}"],
            [f"{groups[0]}: proportion", round(p1, 4)],
            [f"{groups[1]}: k/n", f"{k2}/{n2}"],
            [f"{groups[1]}: proportion", round(p2, 4)],
            ["Cohen's h", es["value"]],
        ],
        "r_code": f"prop.test(c({k1}, {k2}), c({n1}, {n2}))",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. McNEMAR'S TEST
# ═══════════════════════════════════════════════════════════════════════════════

class McnemarRequest(BaseModel):
    session_id: str
    col1: str
    col2: str
    alpha: float = 0.05


@router.post("/mcnemar")
def mcnemar_test(req: McnemarRequest):
    from statsmodels.stats.contingency_tables import mcnemar

    df = _get_df(req.session_id)
    for c in [req.col1, req.col2]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub, warnings = _clean_binary_frame(df, [req.col1, req.col2])
    if len(sub) < 5:
        raise HTTPException(400, "Need at least 5 paired observations.")

    # Build 2x2 contingency table from paired observations
    ct = pd.crosstab(sub[req.col1], sub[req.col2])
    if ct.shape != (2, 2):
        raise HTTPException(400, f"Expected 2x2 table from binary variables, got {ct.shape}.")

    table = ct.values
    # crosstab sorts its levels ascending, so row/column 0 is the LOW level.
    # The cells used to be unpacked as a, b, c, d and then labelled "a (both
    # +)", "b (+ to -)" and so on — every label was the wrong way round, and
    # the discordant odds ratio printed beside them was the reciprocal of what
    # the text claimed. The p-value is symmetric in the two discordant cells
    # and was never affected; the reported direction was.
    neg_level, pos_level = (str(v) for v in ct.index)
    both_neg = int(table[0][0])
    neg_to_pos = int(table[0][1])   # low on col1, high on col2
    pos_to_neg = int(table[1][0])   # high on col1, low on col2
    both_pos = int(table[1][1])
    a, b, c, d = both_pos, pos_to_neg, neg_to_pos, both_neg

    # Use exact test when discordant pairs < 25
    exact = bool((b + c) < 25)
    result = mcnemar(table, exact=exact)
    stat = float(result.statistic)
    p = float(result.pvalue)
    sig = bool(p < req.alpha)
    ps = _p_str(p)

    # Odds of moving positive -> negative against negative -> positive.
    or_val = float(b / c) if c > 0 else float('inf')
    or_str = f"{or_val:.3f}" if np.isfinite(or_val) else "Inf"
    es = {"name": "odds_ratio_discordant", "value": round(or_val, 4) if np.isfinite(or_val) else None,
          "ci_low": None, "ci_high": None, "magnitude": ""}
    if np.isfinite(or_val):
        from services.stat_utils import _es_magnitude
        es["magnitude"] = _es_magnitude("odds_ratio", or_val)

    return sanitize_nonfinite({
        "test": "McNemar's test",
        "statistic": round(stat, 4), "p": p,
        "exact": exact,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "contingency_table": {"a": int(a), "b": int(b), "c": int(c), "d": int(d)},
        # The cells named by what they actually are, so the direction cannot
        # be misread off the letters.
        "cells": {
            "levels": {"negative": neg_level, "positive": pos_level},
            "both_positive": both_pos,
            "positive_to_negative": pos_to_neg,
            "negative_to_positive": neg_to_pos,
            "both_negative": both_neg,
        },
        "summary": {
            "discordant_b": int(b), "discordant_c": int(c),
            "concordant_a": int(a), "concordant_d": int(d),
            "n": int(len(sub)),
        },
        "warnings": warnings,
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} change between {req.col1} and {req.col2} "
            f"({'exact' if exact else 'chi-squared'} statistic = {stat:.3f}, p = {ps}, OR = {or_str})"
        ),
        "result_text": (
            f"McNemar's test ({'exact' if exact else 'asymptotic'}) assessed the change between "
            f"{req.col1} and {req.col2} (n = {len(sub)} pairs). "
            f"Discordant pairs: {req.col1} '{pos_level}' -> {req.col2} "
            f"'{neg_level}' = {pos_to_neg}; {req.col1} '{neg_level}' -> "
            f"{req.col2} '{pos_level}' = {neg_to_pos}. "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(statistic = {stat:.3f}, p = {ps}). "
            f"Odds ratio of discordant pairs = {or_str}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Test statistic", round(stat, 4)],
            ["p", round(p, 6)],
            ["Exact test", exact],
            [f"Both '{pos_level}'", both_pos],
            [f"'{pos_level}' -> '{neg_level}'", pos_to_neg],
            [f"'{neg_level}' -> '{pos_level}'", neg_to_pos],
            [f"Both '{neg_level}'", both_neg],
            ["OR (discordant)", or_str],
        ],
        "r_code": "mcnemar.test(table)",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COCHRAN'S Q TEST
# ═══════════════════════════════════════════════════════════════════════════════

class CochranQRequest(BaseModel):
    session_id: str
    columns: List[str]
    alpha: float = 0.05


@router.post("/cochran_q")
def cochran_q_test(req: CochranQRequest):
    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_fn

    if len(req.columns) < 3:
        raise HTTPException(400, "Cochran's Q test requires at least 3 binary columns.")

    df = _get_df(req.session_id)
    for c in req.columns:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[req.columns].dropna()
    if len(sub) < 5:
        raise HTTPException(400, "Need at least 5 complete subjects.")

    # Cochran's Q is defined on 0/1 outcomes. The matrix used to be cast to
    # float with no check at all, so continuous columns produced a negative Q
    # and "proportions" above 1 — arithmetic that looks like a result.
    mat = sub.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(mat).all():
        raise HTTPException(
            422,
            "Cochran's Q needs numeric 0/1 columns; some values could not be "
            "read as numbers.",
        )
    offenders = [
        c for c in req.columns
        if not set(np.unique(sub[c].astype(float))).issubset({0.0, 1.0})
    ]
    if offenders:
        raise HTTPException(
            422,
            "Cochran's Q compares binary outcomes across conditions, but "
            + ", ".join(f"'{c}'" for c in offenders)
            + " hold values other than 0 and 1. Recode them to 0/1, or use "
              "repeated-measures ANOVA / Friedman for continuous outcomes.",
        )
    n, k = mat.shape

    # Manual Cochran's Q calculation
    # Gj = column sums, Li = row sums, T = grand total
    Gj = mat.sum(axis=0)  # column sums (k values)
    Li = mat.sum(axis=1)  # row sums (n values)
    T = mat.sum()

    numerator = (k - 1) * (k * np.sum(Gj ** 2) - T ** 2)
    denominator = k * T - np.sum(Li ** 2)

    if denominator == 0:
        raise HTTPException(400, "Cannot compute Q: all rows are identical.")

    Q = numerator / denominator
    df_q = k - 1
    p = float(1 - sp.chi2.cdf(Q, df_q))
    sig = bool(p < req.alpha)
    ps = _p_str(p)

    # Effect size: Kendall's W
    es = kendalls_w(float(Q), n, k)

    # Post-hoc: pairwise McNemar with Holm correction (if significant)
    posthoc = []
    if sig:
        raw_ps = []
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        for i, j in pairs:
            ct = pd.crosstab(sub[req.columns[i]], sub[req.columns[j]])
            # Ensure 2x2
            if ct.shape == (2, 2):
                table = ct.values
                exact = bool((table[0, 1] + table[1, 0]) < 25)
                try:
                    res = mcnemar_fn(table, exact=exact)
                    pv = float(res.pvalue)
                except Exception:
                    pv = 1.0
            else:
                pv = 1.0
            posthoc.append({
                "group1": req.columns[i], "group2": req.columns[j],
                "p": round(pv, 6),
            })
            raw_ps.append(pv)

        adj = adjust_pvalues(raw_ps, "holm")
        for idx, ph in enumerate(posthoc):
            ph["p_adj"] = round(adj[idx], 6)
            ph["significant"] = adj[idx] < req.alpha
            ph["correction"] = "holm"

    col_props = {c: round(float(Gj[i]) / n, 4) for i, c in enumerate(req.columns)}

    return {
        "test": "Cochran's Q test",
        "Q": round(float(Q), 4), "df": df_q, "p": p,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "posthoc": posthoc,
        "posthoc_method": "Pairwise McNemar (Holm correction)" if posthoc else None,
        "summary": {
            "n_subjects": n, "k_conditions": k,
            "proportions": col_props,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} difference across {k} conditions "
            f"(Q({df_q}) = {Q:.2f}, p = {ps}, Kendall's W = {es['value']:.3f} [{es['magnitude']}])"
        ),
        "result_text": (
            f"Cochran's Q test assessed differences across {k} related binary conditions "
            f"(n = {n} subjects). The test was {'statistically significant' if sig else 'not statistically significant'} "
            f"(Q({df_q}) = {Q:.2f}, p = {ps}). "
            f"Kendall's W = {es['value']:.3f} [{es['magnitude']}]."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Cochran's Q", round(float(Q), 4)],
            ["df", df_q],
            ["p", round(p, 6)],
            ["Kendall's W", es["value"]],
            ["n subjects", n],
            ["k conditions", k],
            *[[f"Proportion ({c})", col_props[c]] for c in req.columns],
        ],
        "r_code": (
            f"library(RVAideMemoire)\n"
            f"cochran.qtest(cbind({', '.join(req.columns)}) ~ 1, data = data)"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COCHRAN-MANTEL-HAENSZEL TEST
# ═══════════════════════════════════════════════════════════════════════════════

class MantelHaenszelRequest(BaseModel):
    session_id: str
    row_col: str
    col_col: str
    strata_col: str
    alpha: float = 0.05


@router.post("/mantel_haenszel")
def mantel_haenszel_test(req: MantelHaenszelRequest):
    from statsmodels.stats.contingency_tables import StratifiedTable

    df = _get_df(req.session_id)
    for c in [req.row_col, req.col_col, req.strata_col]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.row_col, req.col_col, req.strata_col]].copy()
    cleaned_row = clean_two_level(sub[req.row_col])
    cleaned_col = clean_two_level(sub[req.col_col])
    sub[req.row_col] = cleaned_row.series
    sub[req.col_col] = cleaned_col.series
    sub = sub.dropna()
    warnings = cleaned_row.warnings + cleaned_col.warnings
    if len(sub) < 10:
        raise HTTPException(400, "Need at least 10 observations.")

    row_levels = sorted(sub[req.row_col].unique())
    col_levels = sorted(sub[req.col_col].unique())
    if len(row_levels) != 2 or len(col_levels) != 2:
        raise HTTPException(400, "Row and column variables must each have exactly 2 levels for CMH test.")

    strata = sorted(sub[req.strata_col].unique())
    if len(strata) < 2:
        raise HTTPException(400, "Need at least 2 strata.")

    # Build list of 2x2 tables per stratum
    tables = []
    stratum_info = []
    for s in strata:
        s_data = sub[sub[req.strata_col] == s]
        ct = pd.crosstab(s_data[req.row_col], s_data[req.col_col])
        # Ensure 2x2 with correct ordering
        ct = ct.reindex(index=row_levels, columns=col_levels, fill_value=0)
        tables.append(ct.values.astype(float))
        stratum_info.append({"stratum": str(s), "n": int(len(s_data)),
                             "table": ct.values.tolist()})

    try:
        st = StratifiedTable(tables)
        result = st.test_null_odds()
        stat = float(result.statistic)
        p = float(result.pvalue)
    except Exception as exc:
        raise HTTPException(400, f"CMH test failed: {exc}")

    sig = bool(p < req.alpha)
    ps = _p_str(p)

    # Common odds ratio. A stratum with a zero cell sends the pooled estimate
    # to infinity, and one infinite number used to make the whole response
    # unserialisable — the global handler turned it into a 400 and the valid
    # CMH statistic and p-value went with it. Report the estimate as absent
    # and name the strata responsible, rather than losing the test.
    try:
        common_or = float(st.oddsratio_pooled)
    except Exception:
        common_or = None
    if common_or is not None and not np.isfinite(common_or):
        zero_cell = [
            info["stratum"] for info, tab in zip(stratum_info, tables)
            if float(np.asarray(tab).min()) == 0
        ]
        warnings.append(
            "The common odds ratio is not finite because "
            + (f"stratum/strata {', '.join(zero_cell)} contain "
               if zero_cell else "a stratum contains ")
            + "a zero cell. The CMH test itself is unaffected and is reported "
            "above; collapse or drop the empty strata for an interpretable "
            "common odds ratio."
        )
        common_or = None
    or_str = f"{common_or:.3f}" if common_or is not None else "N/A"

    return sanitize_nonfinite({
        "test": "Cochran-Mantel-Haenszel test",
        "statistic": round(stat, 4), "p": p,
        "significant": sig,
        # `if common_or` (not `is not None`) reported an odds ratio of exactly
        # 0 as absent.
        "effect_sizes": [{"name": "common_odds_ratio",
                          "value": round(common_or, 4) if common_or is not None else None,
                          "ci_low": None, "ci_high": None, "magnitude": ""}],
        "assumptions": [],
        "summary": {
            "n_strata": len(strata),
            "n_total": int(len(sub)),
            "strata": stratum_info,
        },
        "warnings": warnings,
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} association between {req.row_col} and {req.col_col} "
            f"after stratifying by {req.strata_col} "
            f"(CMH statistic = {stat:.3f}, p = {ps}, common OR = {or_str})"
        ),
        "result_text": (
            f"A Cochran-Mantel-Haenszel test examined the association between {req.row_col} and {req.col_col} "
            f"across {len(strata)} strata of {req.strata_col} (n = {len(sub)}). "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(CMH statistic = {stat:.3f}, p = {ps}). "
            f"Common odds ratio = {or_str}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["CMH statistic", round(stat, 4)],
            ["p", round(p, 6)],
            ["Common OR", or_str],
            ["Number of strata", len(strata)],
            ["Total n", int(len(sub))],
        ],
        "r_code": "mantelhaen.test(table_array)",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 7. COCHRAN-ARMITAGE TREND TEST
# ═══════════════════════════════════════════════════════════════════════════════
# Tests for a linear trend in proportions across K ordered groups
# (e.g. dose levels 0,1,2,3 vs adverse event 0/1). Standard reference:
# Agresti, "Categorical Data Analysis" 3e §3.2.4. The test statistic is:
#   Z = Σ_k w_k (n_{k1} - n_k p̂) / sqrt( p̂ (1-p̂) Σ_k n_k (w_k - w̄)² )
# where w_k are the user-supplied scores (default = 0,1,…,K-1), n_{k1} is
# the number of successes in row k, n_k is the row total, p̂ is the pooled
# success proportion, and w̄ = Σ n_k w_k / N. Z is N(0,1) under H₀.

class CochranArmitageRequest(BaseModel):
    session_id: str
    ordinal_col: str          # the ordered exposure / dose column
    event_col: str            # binary 0/1 outcome column
    scores: Optional[List[float]] = None  # custom scores per level; default = ranks 0..K-1
    success_value: Optional[str] = None   # which value of event_col counts as "success"
    alpha: float = 0.05
    # Explicit low→high ordering for non-numeric levels. Without it the levels
    # can only be sorted alphabetically, which silently reverses e.g.
    # Low/Medium/High and inverts the trend.
    level_order: Optional[List[str]] = None


@router.post("/cochran_armitage")
def cochran_armitage(req: CochranArmitageRequest):
    df = _get_df(req.session_id)
    for c in (req.ordinal_col, req.event_col):
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.ordinal_col, req.event_col]].dropna()
    if len(sub) < 5:
        raise HTTPException(422, "Need at least 5 non-null rows.")

    # Determine "success" coding. Default: 1 if binary 0/1, else most-frequent.
    ev = sub[req.event_col]
    if req.success_value is not None:
        try:
            success = type(ev.iloc[0])(req.success_value)  # best-effort cast
        except Exception:
            success = req.success_value
    else:
        unique_vals = ev.unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            success = 1
        else:
            if len(unique_vals) != 2:
                raise HTTPException(422,
                    f"Event column must be binary; got {len(unique_vals)} unique values.")
            success = ev.value_counts().idxmax()

    ev_bin = (ev == success).astype(int)

    # Build K-row contingency table ordered by the ordinal column.
    present = list(sub[req.ordinal_col].unique())

    def _is_numeric_level(x) -> bool:
        return (isinstance(x, (int, float, np.integer, np.floating))
                or (isinstance(x, str)
                    and x.replace(".", "", 1).replace("-", "", 1).isdigit()))

    ca_warnings: List[str] = []
    if req.level_order is not None:
        # Caller stated the ordering explicitly — honour it, but insist it
        # matches the data exactly so a typo can't silently drop a level.
        wanted = [str(v) for v in req.level_order]
        have = [str(v) for v in present]
        if sorted(wanted) != sorted(have):
            raise HTTPException(422,
                f"level_order must list every level of '{req.ordinal_col}' exactly once. "
                f"Got {wanted}; the data has {sorted(have)}.")
        by_str = {str(v): v for v in present}
        levels = [by_str[v] for v in wanted]
        order_source = "caller-supplied level_order"
    else:
        levels = sorted(present, key=lambda x: (0, float(x)) if _is_numeric_level(x) else (1, str(x)))
        if all(_is_numeric_level(v) for v in present):
            order_source = "numeric value"
        else:
            # Alphabetical order is an assumption, not a fact: "Low, Medium,
            # High" becomes "High, Low, Medium" and the trend flips sign. The
            # test still runs, but the caller has to be told what was assumed.
            order_source = "alphabetical (assumed)"
            ca_warnings.append(
                f"'{req.ordinal_col}' has non-numeric levels, so they were ordered "
                f"alphabetically: {[str(v) for v in levels]}. If that is not the "
                "true low-to-high order, the trend direction is wrong — pass "
                "level_order (or scores) to state the ordering explicitly."
            )
    K = len(levels)
    if K < 3:
        raise HTTPException(422,
            "Cochran-Armitage requires at least 3 ordered groups; "
            f"got {K}. Use a 2x2 chi-square / Fisher's test instead.")

    n_k = np.array([int((sub[req.ordinal_col] == lev).sum()) for lev in levels], dtype=float)
    s_k = np.array([int(ev_bin[sub[req.ordinal_col] == lev].sum()) for lev in levels], dtype=float)
    if np.any(n_k == 0):
        raise HTTPException(422, "At least one ordinal level has zero observations.")

    # Scores: default = 0..K-1, but accept user-supplied (must match K).
    if req.scores is not None:
        if len(req.scores) != K:
            raise HTTPException(422,
                f"Custom scores must match the number of levels ({K}); got {len(req.scores)}.")
        w = np.asarray(req.scores, dtype=float)
    else:
        w = np.arange(K, dtype=float)

    N = float(n_k.sum())
    p_hat = float(s_k.sum() / N)
    if not (0.0 < p_hat < 1.0):
        raise HTTPException(422,
            "Cannot test trend: outcome is constant (all successes or all failures).")

    w_bar = float(np.sum(n_k * w) / N)
    numerator = float(np.sum(w * (s_k - n_k * p_hat)))
    variance = float(p_hat * (1.0 - p_hat) * np.sum(n_k * (w - w_bar) ** 2))
    if variance <= 0:
        raise HTTPException(422, "Trend variance is zero; scores may all be equal.")

    z = numerator / np.sqrt(variance)
    p_two = 2.0 * (1.0 - sp.norm.cdf(abs(z)))  # two-sided z-test
    sig = bool(p_two < req.alpha)
    ps = _p_str(p_two)

    # Direction: sign of Z indicates whether p̂_k increases (+) or decreases (-) with scores.
    direction = "increasing" if z > 0 else "decreasing" if z < 0 else "flat"

    # Per-level summary for the UI table.
    level_rows = []
    for lev, nk, sk, wk in zip(levels, n_k, s_k, w):
        pk = float(sk / nk) if nk > 0 else float("nan")
        level_rows.append({
            "level": str(lev),
            "score": float(wk),
            "n": int(nk),
            "successes": int(sk),
            "proportion": round(pk, 4),
        })

    return {
        "test": "Cochran-Armitage trend test",
        "z": round(z, 4),
        "statistic": round(z, 4),
        "p": p_two,
        "significant": sig,
        "effect_sizes": [],
        "warnings": ca_warnings,
        "level_order_source": order_source,
        "assumptions": [
            "Ordered (ordinal) exposure with ≥3 levels",
            "Binary outcome",
            "Independence between observations",
            f"Level ordering taken from: {order_source}",
        ],
        "summary": {
            "n": int(N),
            "n_successes": int(s_k.sum()),
            "pooled_proportion": round(p_hat, 4),
            "n_levels": K,
            "direction": direction,
            "scores": list(w.astype(float)),
            "levels": level_rows,
        },
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} linear trend in the "
            f"proportion of {req.event_col} (success = {success}) across {K} "
            f"ordered levels of {req.ordinal_col} "
            f"(Z = {z:.3f}, p = {ps}; direction: {direction})."
        ),
        "result_text": (
            f"A Cochran-Armitage trend test assessed whether the proportion of "
            f"{req.event_col} = {success} changed linearly across {K} ordered "
            f"levels of {req.ordinal_col} (n = {int(N)}). The trend was "
            f"{'statistically significant' if sig else 'not statistically significant'} "
            f"(Z = {z:.3f}, p = {ps}), with a {direction} trend in proportions."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Z", round(z, 4)],
            ["p", round(p_two, 6)],
            ["Levels", K],
            ["Pooled proportion", round(p_hat, 4)],
            ["Direction", direction],
            ["Total n", int(N)],
        ],
        "r_code": (
            f"# DescTools::CochranArmitageTest(table({req.ordinal_col}, {req.event_col}))\n"
            f"prop.trend.test(c{tuple(int(s) for s in s_k)}, c{tuple(int(n) for n in n_k)})"
        ),
    }
