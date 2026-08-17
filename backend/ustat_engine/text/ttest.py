"""Methods, Results and R-code prose for the two t-tests.

Moved verbatim from `services/text_generators.py`; that module re-exports all
six, so nothing that already imported them changed.

`r_ttest_ind` is worth noticing: it emits `var.equal = FALSE` only when the run
actually took the Welch branch. The snippet is the reproducibility claim, so it
has to be generated from the same `use_welch` the test used -- which is why it
lives here rather than being reconstructed by whoever renders the result.
"""
from __future__ import annotations

from .common import _df_str, _es_str, _p_str


def methods_ttest_ind(col: str, group_col: str, welch: bool = False) -> str:
    variant = "Welch's" if welch else "Student's"
    return (
        f"Group differences in {col} were compared between levels of {group_col} "
        f"using an independent-samples {variant} t-test. "
        f"Effect size was quantified with Hedges' g and its 95% confidence interval. "
        f"Normality of each group was assessed with the Shapiro-Wilk test (n < 50), "
        f"the Lilliefors-corrected Kolmogorov-Smirnov test (50 ≤ n ≤ 2000), "
        f"or a skewness/CLT criterion (n > 2000). "
        f"Homogeneity of variances was checked with Levene's test."
    )


def methods_ttest_one(col: str, mu: float) -> str:
    return (
        f"A one-sample t-test was used to compare the mean of {col} "
        f"against the hypothesized value of {mu}. "
        f"Effect size was quantified with Cohen's d."
    )


def results_ttest_ind(result: dict) -> str:
    g1, g2 = result.get("group1", "Group 1"), result.get("group2", "Group 2")
    t = result.get("t", 0)
    p = result.get("p", 1)
    df = _df_str(result.get("df", ""))
    m1, m2 = result.get("mean1", 0), result.get("mean2", 0)
    sig = result.get("significant", False)
    es_list = result.get("effect_sizes", [])
    es_text = f", {_es_str(es_list[0])}" if es_list else ""

    return (
        f"The {g1} group (M = {m1:.2f}) {'significantly differed from' if sig else 'did not significantly differ from'} "
        f"the {g2} group (M = {m2:.2f}), t({df}) = {t:.3f}, p = {_p_str(p)}{es_text}."
    )


def results_ttest_one(result: dict) -> str:
    mu = result.get("mu", 0)
    mean = result.get("mean", 0)
    t = result.get("t", 0)
    p = result.get("p", 1)
    df = result.get("df", "")
    sig = result.get("significant", False)
    es_list = result.get("effect_sizes", [])
    es_text = f", {_es_str(es_list[0])}" if es_list else ""

    return (
        f"The sample mean (M = {mean:.2f}) {'was significantly different from' if sig else 'did not significantly differ from'} "
        f"the test value of {mu}, t({df}) = {t:.3f}, p = {_p_str(p)}{es_text}."
    )


def r_ttest_ind(col: str, group_col: str, welch: bool = False) -> str:
    """R equivalent of the test that was actually run.

    var.equal = TRUE is Student's pooled-variance test; Welch needs FALSE.
    Emitting TRUE for a Welch result made the snippet irreproducible.
    """
    var_equal = "FALSE" if welch else "TRUE"
    return f't.test({col} ~ {group_col}, data = data, var.equal = {var_equal})'


def r_ttest_one(col: str, mu: float) -> str:
    return f't.test(data${col}, mu = {mu})'


__all__ = [
    "methods_ttest_ind",
    "methods_ttest_one",
    "r_ttest_ind",
    "r_ttest_one",
    "results_ttest_ind",
    "results_ttest_one",
]
