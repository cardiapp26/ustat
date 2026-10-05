"""Categorical tests: binomial, chi-square goodness of fit, proportion z-tests, McNemar, Cochran Q, Mantel-Haenszel."""
import math

import numpy as np
import pandas as pd
from scipy import stats as sp
from fastapi import APIRouter, HTTPException
from pydantic import AliasChoices, BaseModel, Field
from typing import Dict, List, Optional

from services import store
from services.level_order import SOURCE_RECOGNISED, resolve_level_order
from services.category_health import clean_two_level
from services.diagnostic_ci import wilson_ci
from services.risk_measures import (
    compute_risk_measures, risk_measures_export_rows, risk_measures_text,
)
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
    expected_proportion: float = Field(
        default=0.5,
        validation_alias=AliasChoices("expected_proportion", "p"),
    )
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
# 1b. CHI-SQUARE GOODNESS OF FIT (one sample vs theoretical proportions)
# ═══════════════════════════════════════════════════════════════════════════════
# Tests whether the category frequencies of ONE variable match a stated
# distribution (Mendelian 9:3:3:1, a uniform split, a census mix). Pearson
# chi-square with df = k - 1 (no parameters estimated from the data), plus an
# exact multinomial p for small samples where the chi-square approximation
# is not trustworthy.

# Exact enumeration walks every way of splitting n observations over k
# categories: C(n + k - 1, k - 1) outcomes. Above this size the call would be
# slow and memory hungry, so the exact p is reported as unavailable instead.
_GOF_EXACT_MAX_CATEGORIES = 4
_GOF_EXACT_MAX_OUTCOMES = 2_000_000


class ChiSquareGofRequest(BaseModel):
    session_id: str
    column: str
    # Category label -> proportion or weight (9, 3, 3, 1 is as good as
    # 0.5625, 0.1875, ...). None means equal proportions across the observed
    # categories.
    expected_proportions: Optional[Dict[str, float]] = Field(
        default=None,
        validation_alias=AliasChoices("expected_proportions", "proportions"),
    )
    alpha: float = Field(default=0.05, gt=0, lt=1)


def _gof_label(value) -> str:
    """Text label of a category, with 3.0 and 3 reading as the same level."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _gof_compositions(n: int, k: int) -> np.ndarray:
    """Every k-tuple of non-negative integers summing to n, one per row."""
    if k == 1:
        return np.array([[n]], dtype=np.int32)
    if k == 2:
        first = np.arange(n + 1, dtype=np.int32)
        return np.column_stack([first, n - first])
    blocks = []
    for first in range(n + 1):
        rest = _gof_compositions(n - first, k - 1)
        blocks.append(np.column_stack([np.full(len(rest), first, dtype=np.int32), rest]))
    return np.vstack(blocks)


def _exact_multinomial_p(observed: np.ndarray, probs: np.ndarray) -> dict:
    """Exact multinomial goodness-of-fit p by full enumeration.

    The p is the total probability of every outcome that is no more likely
    than the observed one (the same ordering R's XNomial::xmulti uses with
    its probability statistic). Returns ``{"p": None, "note": ...}`` when the
    enumeration would be too large.
    """
    n = int(observed.sum())
    k = len(observed)
    n_outcomes = math.comb(n + k - 1, k - 1)
    if k > _GOF_EXACT_MAX_CATEGORIES or n_outcomes > _GOF_EXACT_MAX_OUTCOMES:
        return {
            "p": None,
            "n_outcomes": n_outcomes,
            "note": (
                f"The exact multinomial test was not computed: enumerating "
                f"{n_outcomes:,} possible outcomes (n = {n}, k = {k}) exceeds the "
                f"limit (k <= {_GOF_EXACT_MAX_CATEGORIES}, at most "
                f"{_GOF_EXACT_MAX_OUTCOMES:,} outcomes)."
            ),
        }
    outcomes = _gof_compositions(n, k)
    log_pmf = sp.multinomial.logpmf(outcomes, n, probs)
    log_obs = float(sp.multinomial.logpmf(observed, n, probs))
    # Relative tolerance of 1e-7 on the probability, so outcomes that tie with
    # the observed one up to floating-point noise are counted as tied.
    p = float(np.exp(log_pmf[log_pmf <= log_obs + 1e-7]).sum())
    return {"p": min(1.0, p), "n_outcomes": n_outcomes, "note": None}


def _gof_magnitude(w: float) -> str:
    """Cohen's (1988) conventions for w: .1 small, .3 medium, .5 large."""
    if w < 0.10:
        return "negligible"
    if w < 0.30:
        return "small"
    if w < 0.50:
        return "medium"
    return "large"


def _r_num(v: float) -> str:
    return f"{float(v):.10g}"


@router.post("/chisquare_gof")
def chisquare_gof(req: ChiSquareGofRequest):
    df = _get_df(req.session_id)
    if req.column not in df.columns:
        raise HTTPException(400, f"Column '{req.column}' not found.")

    # Same missing-value handling as the sibling tests: blanks and
    # missing-looking tokens drop out and are reported. The cleaned series is
    # used only to decide which rows survive; the labels the user typed in
    # expected_proportions are matched against the original text, not against
    # the yes/no -> 1/0 recoding the two-level cleaner applies.
    raw_col = df[req.column]
    cleaned = clean_two_level(raw_col)
    labels = raw_col[cleaned.series.notna()].map(_gof_label)
    n_total_rows = int(len(raw_col))
    n = int(len(labels))
    n_excluded = n_total_rows - n
    if n < 1:
        raise HTTPException(400, f"No non-missing values in '{req.column}'.")

    warnings: list = []
    for item in cleaned.warnings:
        warnings.append(item.get("note", str(item)) if isinstance(item, dict) else str(item))
    if n_excluded:
        warnings.append(
            f"{n_excluded} of {n_total_rows} row(s) with a missing value in "
            f"'{req.column}' were excluded; the test uses n = {n}."
        )

    counts = labels.value_counts()
    observed_map = {str(lbl): int(c) for lbl, c in counts.items()}

    supplied_sum = None
    proportions_normalised = False
    if req.expected_proportions is None:
        categories = sorted(
            observed_map,
            key=lambda x: (0, float(x), x) if _is_number_text(x) else (1, 0.0, x),
        )
        weights = {c: 1.0 for c in categories}
    else:
        weights = {}
        for key, value in req.expected_proportions.items():
            label = str(key).strip()
            if label in weights:
                raise HTTPException(
                    422, f"Category '{label}' appears more than once in expected_proportions."
                )
            if not np.isfinite(value) or value <= 0:
                raise HTTPException(
                    422,
                    f"Expected proportion for '{label}' must be a positive finite "
                    f"number (got {value}).",
                )
            weights[label] = float(value)
        missing = [c for c in observed_map if c not in weights]
        if missing:
            shown = ", ".join(repr(m) for m in sorted(missing)[:10])
            raise HTTPException(
                422,
                f"expected_proportions does not cover every observed category of "
                f"'{req.column}'. Missing: {shown}"
                + (", ..." if len(missing) > 10 else "")
                + ". Give a proportion for each one (or recode the variable).",
            )
        categories = list(weights)
        supplied_sum = float(sum(weights.values()))
        proportions_normalised = not math.isclose(supplied_sum, 1.0, abs_tol=1e-9)

    k = len(categories)
    if k < 2:
        raise HTTPException(
            422,
            f"'{req.column}' has only {k} category after dropping missing values; "
            "a goodness-of-fit test needs at least 2.",
        )

    weight_arr = np.array([weights[c] for c in categories], dtype=float)
    exp_prop = weight_arr / weight_arr.sum()
    obs = np.array([observed_map.get(c, 0) for c in categories], dtype=float)
    exp_count = exp_prop * n

    unobserved = [c for c in categories if observed_map.get(c, 0) == 0]
    if unobserved:
        warnings.append(
            "Category(ies) with an expected proportion but no observations: "
            + ", ".join(repr(c) for c in unobserved)
            + ". They are kept in the test with an observed count of 0."
        )
    if proportions_normalised:
        warnings.append(
            f"The supplied proportions summed to {supplied_sum:.6g}, not 1; they "
            "were rescaled to sum to 1 (a ratio such as 9:3:3:1 is fine)."
        )

    chi2_stat, p_raw = sp.chisquare(obs, exp_count)
    chi2_stat = float(chi2_stat)
    p = float(p_raw)
    dof = k - 1
    sig = bool(p < req.alpha)
    ps = _p_str(p)

    w = float(np.sqrt(chi2_stat / n))
    es = {
        "name": "cohens_w",
        "value": round(w, 4),
        "ci_low": None,
        "ci_high": None,
        "magnitude": _gof_magnitude(w),
    }

    pearson = (obs - exp_count) / np.sqrt(exp_count)
    adjusted = (obs - exp_count) / np.sqrt(exp_count * (1 - exp_prop))
    table = [
        {
            "category": c,
            "observed": int(obs[i]),
            "expected_count": round(float(exp_count[i]), 4),
            "expected_proportion": round(float(exp_prop[i]), 6),
            "observed_proportion": round(float(obs[i] / n), 6),
            "pearson_residual": round(float(pearson[i]), 4),
            "adjusted_residual": round(float(adjusted[i]), 4),
        }
        for i, c in enumerate(categories)
    ]

    n_small = int((exp_count < 5).sum())
    frac_small = n_small / k
    assumption_met = n_small == 0
    min_expected = float(exp_count.min())
    assumptions = [{
        "name": "Expected counts >= 5",
        "met": assumption_met,
        "detail": (
            f"All {k} expected counts are at least 5 (minimum {min_expected:.2f})."
            if assumption_met else
            f"{n_small} of {k} expected counts ({100 * frac_small:.0f}%) are below 5 "
            f"(minimum {min_expected:.2f})."
        ),
    }]
    if not assumption_met:
        extra = " More than 20% of the cells are affected." if frac_small > 0.20 else ""
        warnings.append(
            f"{n_small} of {k} expected counts are below 5 (minimum "
            f"{min_expected:.2f}), so the chi-square approximation may be unreliable."
            f"{extra} Prefer the exact multinomial test, or merge sparse categories "
            "if that is meaningful."
        )

    exact = _exact_multinomial_p(obs.astype(int), exp_prop)
    if exact["note"]:
        if not assumption_met:
            warnings.append(exact["note"])
    exact_p = exact["p"]
    exact_text = (
        f" Exact multinomial p = {_p_str(exact_p)}." if exact_p is not None else ""
    )

    obs_ints = [int(v) for v in obs]
    x_vec = ", ".join(str(v) for v in obs_ints)
    cat_comment = "# categories (in order): " + ", ".join(
        c.replace("\n", " ").replace("\r", " ") for c in categories
    ) + "\n"
    if req.expected_proportions is None:
        r_code = cat_comment + f"chisq.test(x = c({x_vec}))"
    elif proportions_normalised:
        r_code = (
            cat_comment
            + f"chisq.test(x = c({x_vec}), p = c({', '.join(_r_num(weights[c]) for c in categories)}), "
            "rescale.p = TRUE)"
        )
    else:
        r_code = (
            cat_comment
            + f"chisq.test(x = c({x_vec}), p = c({', '.join(_r_num(v) for v in exp_prop)}))"
        )
    if not assumption_met:
        r_code += (
            "\n# Exact multinomial test (small expected counts):\n"
            "# XNomial::xmulti(obs = c(" + x_vec + "), expr = c("
            + ", ".join(_r_num(v) for v in exp_prop) + "), statName = \"Prob\")"
        )

    expected_desc = (
        "equal proportions across the observed categories"
        if req.expected_proportions is None
        else "the stated expected proportions"
    )
    methods_text = (
        f"The distribution of {req.column} across {k} categories was compared with "
        f"{expected_desc} using Pearson's chi-square goodness-of-fit test "
        f"(df = {dof}). Effect size was expressed as Cohen's w."
        + (
            " Because one or more expected counts were below 5, an exact multinomial "
            "test is also reported." if not assumption_met else
            " All expected counts were 5 or greater."
        )
    )

    return sanitize_nonfinite({
        "test": "Chi-square goodness-of-fit test",
        "chi2": round(chi2_stat, 6),
        "df": dof,
        "p": p,
        "significant": sig,
        "n": n,
        "k": k,
        "n_excluded": n_excluded,
        "categories": table,
        "proportions_normalised": proportions_normalised,
        "supplied_proportion_sum": supplied_sum,
        "exact_multinomial": exact,
        "effect_sizes": [es],
        "assumptions": assumptions,
        "summary": {
            "column": req.column,
            "n": n, "k": k, "df": dof,
            "n_excluded": n_excluded,
            "min_expected_count": round(min_expected, 4),
            "n_cells_expected_below_5": n_small,
            "expected": "uniform" if req.expected_proportions is None else "user-specified",
        },
        "warnings": warnings,
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} departure from the expected "
            f"distribution (chi-square({dof}) = {chi2_stat:.3f}, p = {ps}, "
            f"Cohen's w = {es['value']:.3f} [{es['magnitude']}])"
            + (f", exact multinomial p = {_p_str(exact_p)}" if exact_p is not None else "")
        ),
        "result_text": (
            f"A chi-square goodness-of-fit test compared the distribution of {req.column} "
            f"(n = {n}, {k} categories) with {expected_desc}. "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(chi-square({dof}) = {chi2_stat:.3f}, p = {ps}). "
            f"Cohen's w = {es['value']:.3f} [{es['magnitude']}]."
            f"{exact_text}"
        ),
        "methods_text": methods_text,
        "export_rows": [
            ["Category", "Observed", "Expected", "Expected proportion",
             "Observed proportion", "Pearson residual", "Adjusted residual"],
            *[
                [r["category"], r["observed"], r["expected_count"],
                 r["expected_proportion"], r["observed_proportion"],
                 r["pearson_residual"], r["adjusted_residual"]]
                for r in table
            ],
            ["Statistic", "Value"],
            ["Chi-square", round(chi2_stat, 4)],
            ["df", dof],
            ["p", round(p, 6)],
            ["n", n],
            ["Cohen's w", es["value"]],
            ["Exact multinomial p",
             round(exact_p, 6) if exact_p is not None else "N/A"],
        ],
        "r_code": r_code,
    })


def _is_number_text(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ONE-SAMPLE PROPORTION Z-TEST
# ═══════════════════════════════════════════════════════════════════════════════

class OneProportionRequest(BaseModel):
    session_id: str
    column: str
    null_proportion: float = Field(
        default=0.5, gt=0, lt=1,
        validation_alias=AliasChoices("null_proportion", "p0"),
    )
    alpha: float = Field(default=0.05, gt=0, lt=1)


@router.post("/one_proportion")
def one_proportion_ztest(req: OneProportionRequest):
    from statsmodels.stats.proportion import proportions_ztest, proportion_confint

    df = _get_df(req.session_id)
    if req.column not in df.columns:
        raise HTTPException(400, f"Column '{req.column}' not found.")
    col = df[req.column].dropna()
    if len(col) < 1:
        raise HTTPException(400, "No non-null values in column.")

    n = len(col)
    k, success_label, binary_warnings = _binary_success(col, req.column)

    z_stat, p = proportions_ztest(k, n, value=req.null_proportion, prop_var=req.null_proportion)
    z_stat = float(z_stat)
    p = float(p)
    sig = bool(p < req.alpha)
    observed_prop = k / n
    ps = _p_str(p)

    es = cohens_h(observed_prop, req.null_proportion)

    ci_low, ci_high = proportion_confint(k, n, alpha=req.alpha, method="wilson")
    ci_low, ci_high = float(ci_low), float(ci_high)

    return {
        "test": "One-sample proportion z-test",
        "ci_proportion": {"low": ci_low, "high": ci_high, "method": "Wilson", "confidence_level": 1 - req.alpha},
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
            f"{100 * (1 - req.alpha):g}% Wilson CI for the proportion: [{ci_low:.3f}, {ci_high:.3f}]. "
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
            [f"{100 * (1 - req.alpha):g}% Wilson CI lower", ci_low],
            [f"{100 * (1 - req.alpha):g}% Wilson CI upper", ci_high],
            ["Cohen's h", es["value"]],
        ],
        "r_code": f"prop.test({k}, {n}, p = {req.null_proportion}, correct = FALSE, conf.level = {1 - req.alpha})",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TWO-SAMPLE PROPORTION Z-TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TwoProportionsRequest(BaseModel):
    session_id: str
    column: str
    group_column: str = Field(
        validation_alias=AliasChoices("group_column", "group_col")
    )
    alpha: float = Field(default=0.05, gt=0, lt=1)


@router.post("/two_proportions")
def two_proportions_ztest(req: TwoProportionsRequest):
    from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

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
    ci_low, ci_high = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb", alpha=req.alpha)
    z_stat = float(z_stat)
    p = float(p)
    sig = bool(p < req.alpha)
    ps = _p_str(p)

    es = cohens_h(p1, p2)

    # Clinical measures for group[0] (exposed) against group[1] (reference),
    # reusing the Newcombe interval above for the absolute risk difference.
    risk_measures = compute_risk_measures(
        k1, n1, k2, n2, alpha=req.alpha,
        event=str(success_label),
        exposed=str(groups[0]), reference=str(groups[1]),
        ard_ci=(float(ci_low), float(ci_high)),
    )

    return sanitize_nonfinite({
        "test": "Two-sample proportion z-test",
        "z": round(z_stat, 4), "p": p,
        "diff_prop": float(p1 - p2),
        "arr": risk_measures["arr"],
        "risk_measures": risk_measures,
        "ci_diff_low": float(ci_low), "ci_diff_high": float(ci_high),
        "ci_diff_method": "Newcombe (unpooled score)",
        "ci_confidence_level": float(1 - req.alpha),
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
            f"Cohen's h = {es['value']:.3f} [{es['magnitude']}]. "
            f"{risk_measures_text(risk_measures)}"
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
            *risk_measures_export_rows(risk_measures),
        ],
        "r_code": (f"prop.test(c({k1}, {k2}), c({n1}, {n2}), correct = FALSE, conf.level = {1 - req.alpha})\n"
                   "# Replicate the unpooled Newcombe interval reported by uSTAT:\n"
                   f"DescTools::BinomDiffCI({k1}, {n1}, {k2}, {n2}, method = 'score', conf.level = {1 - req.alpha})\n"
                   "# Risk ratio (Katz log CI), RRR = 1 - RR, NNT/NNH = 1 / |ARD| rounded up (Altman 1998):\n"
                   f"DescTools::RelRisk(matrix(c({k1}, {n1 - k1}, {k2}, {n2 - k2}), 2, byrow = TRUE), "
                   f"conf.level = {1 - req.alpha}, method = 'wald')"),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 4. McNEMAR'S TEST
# ═══════════════════════════════════════════════════════════════════════════════

def newcombe_paired_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> dict:
    """Newcombe (1998) method 10: score CI for a paired proportion difference.

    Cells: a = both positive, b = column 1 positive and column 2 negative,
    c = column 1 negative and column 2 positive, d = both negative.
    p1 = (a + b) / n is the marginal positive proportion of column 1 and
    p2 = (a + c) / n that of column 2, so the difference is
    p1 - p2 = (b - c) / n. The two Wilson intervals are combined with the
    phi correlation correction (phi = 0 when a margin is empty).
    """
    n = a + b + c + d
    if n <= 0:
        raise ValueError("empty table")
    p1, p2 = (a + b) / n, (a + c) / n
    diff = (b - c) / n
    l1, u1 = wilson_ci(a + b, n, alpha)
    l2, u2 = wilson_ci(a + c, n, alpha)
    margins = float(a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / np.sqrt(margins) if margins > 0 else 0.0
    lo_gap = (p1 - l1, u2 - p2)
    hi_gap = (u1 - p1, p2 - l2)
    low = diff - np.sqrt(max(lo_gap[0] ** 2 + lo_gap[1] ** 2 - 2 * phi * lo_gap[0] * lo_gap[1], 0.0))
    high = diff + np.sqrt(max(hi_gap[0] ** 2 + hi_gap[1] ** 2 - 2 * phi * hi_gap[0] * hi_gap[1], 0.0))
    return {
        "estimate": float(diff),
        "ci_low": float(max(low, -1.0)),
        "ci_high": float(min(high, 1.0)),
        "proportion_col1": float(p1),
        "proportion_col2": float(p2),
        "phi": float(phi),
    }


def conditional_or_ci(b: int, c: int, alpha: float = 0.05) -> Optional[tuple[float, float]]:
    """Exact conditional CI for the discordant odds ratio b / c.

    Given b + c discordant pairs, b ~ Binomial(b + c, pi) with OR = pi / (1 - pi),
    so the Clopper-Pearson interval for pi is transformed by p / (1 - p).
    None when b + c == 0 or c == 0 (the odds ratio is undefined or infinite).
    """
    if b + c == 0 or c == 0:
        return None
    pi_low = 0.0 if b == 0 else float(sp.beta.ppf(alpha / 2, b, c + 1))
    pi_high = float(sp.beta.ppf(1 - alpha / 2, b + 1, c))
    return pi_low / (1 - pi_low), pi_high / (1 - pi_high)


class McnemarRequest(BaseModel):
    session_id: str
    col1: str = Field(validation_alias=AliasChoices("col1", "column1"))
    col2: str = Field(validation_alias=AliasChoices("col2", "column2"))
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
    ci_alpha = req.alpha if 0.0 < req.alpha < 1.0 else 0.05
    conf_pct = f"{(1 - ci_alpha) * 100:g}%"
    es = {"name": "odds_ratio_discordant", "value": round(or_val, 4) if np.isfinite(or_val) else None,
          "ci_low": None, "ci_high": None, "magnitude": ""}
    if np.isfinite(or_val):
        from services.stat_utils import _es_magnitude
        es["magnitude"] = _es_magnitude("odds_ratio", or_val)
    or_ci = conditional_or_ci(int(b), int(c), ci_alpha)
    if or_ci is not None:
        es["ci_low"], es["ci_high"] = round(or_ci[0], 4), round(or_ci[1], 4)
    npc = newcombe_paired_ci(int(a), int(b), int(c), int(d), ci_alpha)
    paired_difference = {
        "estimate": npc["estimate"],
        "ci_low": npc["ci_low"],
        "ci_high": npc["ci_high"],
        "method": "Newcombe (1998) method 10: Wilson score intervals with phi correlation correction",
        "confidence_level": float(1 - ci_alpha),
        "positive_level": pos_level,
        "proportion_col1": npc["proportion_col1"],
        "proportion_col2": npc["proportion_col2"],
        "phi": npc["phi"],
        "note": (
            f"Difference = proportion '{pos_level}' in {req.col1} minus proportion "
            f"'{pos_level}' in {req.col2} = (b - c) / n, with b = {req.col1} '{pos_level}' -> "
            f"{req.col2} '{neg_level}' ({int(b)}) and c = {req.col1} '{neg_level}' -> "
            f"{req.col2} '{pos_level}' ({int(c)})."
        ),
    }
    pd_text = (
        f" The paired difference in proportion '{pos_level}' ({req.col1} minus {req.col2}) was "
        f"{npc['estimate']:.3f} ({conf_pct} Newcombe CI [{npc['ci_low']:.3f}, {npc['ci_high']:.3f}])."
    )
    or_ci_text = (
        f" ({conf_pct} exact conditional CI [{or_ci[0]:.3f}, {or_ci[1]:.3f}])" if or_ci is not None else ""
    )

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
        "paired_difference": paired_difference,
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
            f"Odds ratio of discordant pairs = {or_str}{or_ci_text}."
            f"{pd_text}"
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
            *([[f"OR (discordant) {conf_pct} CI lower", round(or_ci[0], 4)],
               [f"OR (discordant) {conf_pct} CI upper", round(or_ci[1], 4)]] if or_ci is not None else []),
            [f"Paired difference in '{pos_level}' proportion ({req.col1} - {req.col2})", round(npc["estimate"], 6)],
            [f"Paired difference {conf_pct} CI lower (Newcombe)", round(npc["ci_low"], 6)],
            [f"Paired difference {conf_pct} CI upper (Newcombe)", round(npc["ci_high"], 6)],
        ],
        "r_code": (
            "mcnemar.test(table)\n"
            "# Paired proportion difference (b - c) / n with the Newcombe (1998) method 10 score CI.\n"
            "# DescTools::BinomDiffCI is for INDEPENDENT samples; for paired data compute it by hand:\n"
            f"a <- {int(a)}; b <- {int(b)}; c <- {int(c)}; d <- {int(d)}; n <- a + b + c + d\n"
            f"w1 <- prop.test(a + b, n, correct = FALSE, conf.level = {1 - ci_alpha:g})$conf.int  # P('{pos_level}') in {req.col1}\n"
            f"w2 <- prop.test(a + c, n, correct = FALSE, conf.level = {1 - ci_alpha:g})$conf.int  # P('{pos_level}') in {req.col2}\n"
            "p1 <- (a + b) / n; p2 <- (a + c) / n\n"
            "phi <- if (min(a + b, c + d, a + c, b + d) > 0) (a * d - b * c) / sqrt((a + b) * (c + d) * (a + c) * (b + d)) else 0\n"
            "c(diff = p1 - p2,\n"
            "  lower = (p1 - p2) - sqrt((p1 - w1[1])^2 + (w2[2] - p2)^2 - 2 * phi * (p1 - w1[1]) * (w2[2] - p2)),\n"
            "  upper = (p1 - p2) + sqrt((w1[2] - p1)^2 + (p2 - w2[1])^2 - 2 * phi * (w1[2] - p1) * (p2 - w2[1])))\n"
            "# Conditional exact CI for the discordant odds ratio b / c (Clopper-Pearson on b / (b + c)):\n"
            f"bt <- binom.test(b, b + c, conf.level = {1 - ci_alpha:g})$conf.int; bt / (1 - bt)"
        ),
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

    from statsmodels.stats.contingency_tables import cochrans_q
    Gj = mat.sum(axis=0)
    if np.all(np.ptp(mat, axis=1) == 0):
        raise HTTPException(400, "Cannot compute Q: all rows are identical.")
    q_result = cochrans_q(mat)
    Q, df_q, p = float(q_result.statistic), int(q_result.df), float(q_result.pvalue)
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
            ct = pd.crosstab(sub[req.columns[i]], sub[req.columns[j]]).reindex(
                index=[0, 1], columns=[0, 1], fill_value=0)
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
    row_col: str = Field(
        validation_alias=AliasChoices("row_col", "row_column")
    )
    col_col: str = Field(
        validation_alias=AliasChoices("col_col", "col_column")
    )
    strata_col: str = Field(
        validation_alias=AliasChoices("strata_col", "strata_column")
    )
    alpha: float = Field(default=0.05, gt=0, lt=1)


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

    # Estimate the common odds ratio and its alpha-aware normal-approximation
    # interval from the same StratifiedTable used for the CMH test. Zero cells
    # can produce valid finite estimates, but complete separation can produce
    # Inf/NaN; those boundary results must not sink the otherwise valid test.
    common_or = None
    or_ci_low = None
    or_ci_high = None
    with np.errstate(all="ignore"):
        try:
            common_or = float(st.oddsratio_pooled)
        except Exception:
            pass
        try:
            ci = st.oddsratio_pooled_confint(alpha=req.alpha)
            or_ci_low, or_ci_high = (float(ci[0]), float(ci[1]))
        except Exception:
            pass

    zero_cell = [
        info["stratum"] for info, tab in zip(stratum_info, tables)
        if float(np.asarray(tab).min()) == 0
    ]
    if common_or is not None and not np.isfinite(common_or):
        warnings.append(
            "The common odds ratio is not finite because "
            + (f"stratum/strata {', '.join(zero_cell)} contain "
               if zero_cell else "a stratum contains ")
            + "a zero cell. The CMH test itself is unaffected and is reported "
            "above; collapse or drop the empty strata for an interpretable "
            "common odds ratio."
        )
        common_or = None
        or_ci_low = None
        or_ci_high = None
    elif (
        or_ci_low is None
        or or_ci_high is None
        or not np.isfinite(or_ci_low)
        or not np.isfinite(or_ci_high)
    ):
        warnings.append(
            "The common odds ratio confidence interval could not be estimated"
            + (
                f" because stratum/strata {', '.join(zero_cell)} contain a zero cell."
                if zero_cell else "."
            )
        )
        or_ci_low = None
        or_ci_high = None

    # Breslow-Day test of homogeneity of the stratum odds ratios with Tarone's
    # adjustment. A significant result means the effect differs across strata
    # (effect modification), so one pooled OR can mislead. Singular strata
    # (zero margins) can make the statistic undefined; that must not sink the
    # CMH result, so the failure is reported as a note instead.
    homogeneity_test = None
    homogeneity_note = None
    with np.errstate(all="ignore"):
        try:
            bd = st.test_equal_odds(adjust=True)
            bd_stat = float(bd.statistic)
            bd_p = float(bd.pvalue)
            if np.isfinite(bd_stat) and np.isfinite(bd_p):
                homogeneity_test = {
                    "name": "Breslow-Day test (Tarone adjusted)",
                    "statistic": round(bd_stat, 4),
                    "df": int(len(tables) - 1),
                    "p": bd_p,
                    "adjusted": True,
                    "homogeneous": bool(bd_p >= req.alpha),
                }
            else:
                homogeneity_note = (
                    "The Breslow-Day homogeneity test is undefined for these "
                    "strata (a stratum has a zero margin or a degenerate table)."
                )
        except Exception as exc:
            homogeneity_note = (
                f"The Breslow-Day homogeneity test could not be computed: {exc}"
            )
    if homogeneity_test is not None and not homogeneity_test["homogeneous"]:
        warnings.append(
            f"The odds ratios differ across strata of {req.strata_col} "
            f"(Breslow-Day test, Tarone adjusted, p = {_p_str(homogeneity_test['p'])}). "
            "A single pooled odds ratio may be misleading because of effect "
            "modification; report the stratum-specific odds ratios as well."
        )
    if homogeneity_note is not None:
        warnings.append(homogeneity_note)
    if homogeneity_test is not None:
        homogeneity_text = (
            f" Homogeneity of odds ratios (Breslow-Day, Tarone adjusted): "
            f"chi-square({homogeneity_test['df']}) = {homogeneity_test['statistic']:.3f}, "
            f"p = {_p_str(homogeneity_test['p'])}"
            f" ({'homogeneous' if homogeneity_test['homogeneous'] else 'heterogeneous, interpret the pooled OR with caution'})."
        )
    else:
        homogeneity_text = " Homogeneity of odds ratios could not be assessed (Breslow-Day test unavailable)."

    confidence_pct = 100 * (1 - req.alpha)
    confidence_label = f"{confidence_pct:g}% CI"
    or_str = f"{common_or:.3f}" if common_or is not None else "N/A"
    or_ci_str = (
        f"{or_ci_low:.3f}–{or_ci_high:.3f}"
        if or_ci_low is not None and or_ci_high is not None
        else "unavailable"
    )

    return sanitize_nonfinite({
        "test": "Cochran-Mantel-Haenszel test",
        "statistic": round(stat, 4), "p": p,
        "significant": sig,
        # `if common_or` (not `is not None`) reported an odds ratio of exactly
        # 0 as absent.
        "effect_sizes": [{"name": "common_odds_ratio",
                          "value": round(common_or, 4) if common_or is not None else None,
                          "ci_low": round(or_ci_low, 4) if or_ci_low is not None else None,
                          "ci_high": round(or_ci_high, 4) if or_ci_high is not None else None,
                          "ci_level": round(1 - req.alpha, 6),
                          "magnitude": ""}],
        "assumptions": [],
        "homogeneity_test": homogeneity_test,
        "homogeneity_note": homogeneity_note,
        "summary": {
            "n_strata": len(strata),
            "n_total": int(len(sub)),
            "strata": stratum_info,
        },
        "warnings": warnings,
        "interpretation": (
            f"{'Significant' if sig else 'No significant'} association between {req.row_col} and {req.col_col} "
            f"after stratifying by {req.strata_col} "
            f"(CMH statistic = {stat:.3f}, p = {ps}, common OR = {or_str}, "
            f"{confidence_label} [{or_ci_str}])"
        ),
        "result_text": (
            f"A Cochran-Mantel-Haenszel test examined the association between {req.row_col} and {req.col_col} "
            f"across {len(strata)} strata of {req.strata_col} (n = {len(sub)}). "
            f"The result was {'statistically significant' if sig else 'not statistically significant'} "
            f"(CMH statistic = {stat:.3f}, p = {ps}). "
            f"Common odds ratio = {or_str} ({confidence_label} [{or_ci_str}])."
            f"{homogeneity_text}"
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["CMH statistic", round(stat, 4)],
            ["p", round(p, 6)],
            ["Common OR", or_str],
            [f"Common OR {confidence_label} lower",
             round(or_ci_low, 4) if or_ci_low is not None else "N/A"],
            [f"Common OR {confidence_label} upper",
             round(or_ci_high, 4) if or_ci_high is not None else "N/A"],
            ["Number of strata", len(strata)],
            ["Total n", int(len(sub))],
            *(
                [
                    ["Breslow-Day statistic (Tarone adjusted)", homogeneity_test["statistic"]],
                    ["Breslow-Day df", homogeneity_test["df"]],
                    ["Breslow-Day p", round(homogeneity_test["p"], 6)],
                    ["Odds ratios homogeneous", "yes" if homogeneity_test["homogeneous"] else "no"],
                ]
                if homogeneity_test is not None
                else [["Breslow-Day test (Tarone adjusted)", "unavailable"]]
            ),
        ],
        "r_code": (
            "mantelhaen.test(table_array)\n"
            "# Homogeneity of odds ratios (Breslow-Day with Tarone adjustment):\n"
            "DescTools::BreslowDayTest(table_array, correct = TRUE)"
        ),
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
    ordinal_col: str = Field(  # the ordered exposure / dose column
        validation_alias=AliasChoices("ordinal_col", "ordinal_column")
    )
    event_col: str = Field(  # binary 0/1 outcome column
        validation_alias=AliasChoices("event_col", "event_column")
    )
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
        # Without custom scores, the order comes from the shared resolver
        # (Data Dictionary, numeric codes, a known grading scale). Custom
        # scores keep mapping onto the order below, as they always have.
        order = (
            resolve_level_order(present, req.ordinal_col, session_id=req.session_id)
            if req.scores is None else None
        )
        if order is not None:
            levels = list(order.levels)
            order_source = order.source
            if order.source == SOURCE_RECOGNISED:
                ca_warnings.append(
                    f"'{req.ordinal_col}' was ordered as "
                    + " < ".join(str(v) for v in levels)
                    + " from its labels. Set the order in the Data Dictionary, "
                    "or pass level_order, to state it explicitly."
                )
        elif all(_is_numeric_level(v) for v in present):
            order_source = "numeric value"
        else:
            # Alphabetical order is an assumption, not a fact: "Low, Medium,
            # High" becomes "High, Low, Medium" and the trend flips sign. The
            # test still runs, but the caller has to be told what was assumed.
            order_source = "alphabetical (assumed)"
            ca_warnings.append(
                f"'{req.ordinal_col}' has non-numeric levels, so they were ordered "
                f"alphabetically: {[str(v) for v in levels]}. If that is not the "
                "true low-to-high order, the trend direction is wrong. Set the "
                "order in the Data Dictionary, or pass level_order (or scores), "
                "to state it explicitly."
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


class PairedCategoricalRequest(BaseModel):
    session_id: str
    col1: str
    col2: str
    method: str = "bowker"
    alpha: float = Field(default=0.05, gt=0, lt=1)


@router.post("/paired_categorical")
def paired_categorical(req: PairedCategoricalRequest):
    """Symmetry or marginal homogeneity for paired multicategory observations."""
    from statsmodels.stats.contingency_tables import SquareTable
    if req.method not in {"bowker", "stuart_maxwell"}:
        raise HTTPException(422, "Method must be bowker or stuart_maxwell.")
    df = _get_df(req.session_id)
    if req.col1 == req.col2 or any(c not in df for c in [req.col1, req.col2]):
        raise HTTPException(422, "Select two distinct existing columns.")
    from services.number_format import level_key
    pair = df[[req.col1, req.col2]].dropna().apply(lambda col: col.map(level_key))
    levels = sorted(set(pair[req.col1]) | set(pair[req.col2]))
    if len(pair) < 2 or len(levels) < 2:
        raise HTTPException(422, "Need complete pairs and at least two categories.")
    table = pd.crosstab(pair[req.col1], pair[req.col2]).reindex(index=levels, columns=levels, fill_value=0)
    square = SquareTable(table.to_numpy(), shift_zeros=False)
    try:
        result = square.symmetry() if req.method == "bowker" else square.homogeneity()
    except np.linalg.LinAlgError:
        raise HTTPException(422, "Marginal homogeneity covariance is singular; combine sparse categories.")
    statistic, p = float(result.statistic), float(result.pvalue)
    if not np.isfinite(statistic) or not np.isfinite(p):
        raise HTTPException(422, "Test is undefined for this paired table; combine sparse categories.")
    name = "Bowker symmetry test" if req.method == "bowker" else "Stuart-Maxwell marginal homogeneity test"
    return {"test": name, "chi2": statistic, "df": int(result.df), "p": p,
            "significant": p < req.alpha, "n": len(pair), "table": table.values.tolist(),
            "row_labels": levels, "col_labels": levels, "effect_sizes": [], "assumptions": [],
            "interpretation": f"{name}: chi-square({int(result.df)}) = {statistic:.4f}, p = {_p_str(p)}.",
            "result_text": f"{name} assessed {len(pair)} complete paired observations across {len(levels)} categories.",
            "export_rows": [["Statistic", "Value"], ["Chi-square", statistic], ["df", int(result.df)], ["p", p], ["Complete pairs", len(pair)]]}
