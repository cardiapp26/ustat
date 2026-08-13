"""Journal-style methods and results text generators for uSTAT analyses."""


def _p_str(p: float) -> str:
    from services.number_format import format_p
    return format_p(p)


def _es_str(es: dict) -> str:
    """Format effect size dict as inline text."""
    name = es.get("name", "").replace("_", " ")
    val = es.get("value", 0)
    mag = es.get("magnitude", "")
    ci_lo = es.get("ci_low")
    ci_hi = es.get("ci_high")
    s = f"{name} = {val:.3f}"
    if ci_lo is not None and ci_hi is not None:
        s += f" (95% CI: {ci_lo:.3f}\u2013{ci_hi:.3f})"
    if mag:
        s += f" [{mag}]"
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# METHODS TEXT — describes what was done (for Methods section)
# ═══════════════════════════════════════════════════════════════════════════════

def methods_ttest_ind(col: str, group_col: str, welch: bool = False) -> str:
    variant = "Welch's" if welch else "Student's"
    return (
        f"Group differences in {col} were compared between levels of {group_col} "
        f"using an independent-samples {variant} t-test. "
        f"Effect size was quantified with Hedges' g and its 95% confidence interval. "
        f"Normality of each group was assessed with the Shapiro-Wilk test (n < 50), "
        f"the Lilliefors-corrected Kolmogorov-Smirnov test (50 \u2264 n \u2264 2000), "
        f"or a skewness/CLT criterion (n > 2000). "
        f"Homogeneity of variances was checked with Levene's test."
    )


def methods_ttest_one(col: str, mu: float) -> str:
    return (
        f"A one-sample t-test was used to compare the mean of {col} "
        f"against the hypothesized value of {mu}. "
        f"Effect size was quantified with Cohen's d."
    )


def methods_chisquare(row_col: str, col_col: str, exact: str | None = None) -> str:
    """Methods sentence for a crosstab test.

    ``exact`` names the test that actually produced the p when the expected
    counts ruled out the chi-square. A methods section that describes a
    chi-square while the results report an exact p is not reproducible, and a
    reviewer checking the two against each other would find them inconsistent.
    """
    if exact:
        return (
            f"The association between {row_col} and {col_col} was assessed "
            f"using {exact}, as one or more expected cell counts were below 5. "
            f"Effect size was measured with Cramer's V, with a 95% confidence "
            f"interval from the noncentral chi-square distribution."
        )
    return (
        f"The association between {row_col} and {col_col} was assessed "
        f"using Pearson's chi-square test of independence. "
        f"Effect size was measured with Cramer's V, with a 95% confidence "
        f"interval from the noncentral chi-square distribution. "
        f"All expected cell counts were 5 or greater."
    )


def methods_mannwhitney(col: str, group_col: str) -> str:
    return (
        f"Group differences in {col} between levels of {group_col} were tested "
        f"using the Mann-Whitney U test (non-parametric alternative to the t-test). "
        f"Effect size was quantified with the rank-biserial correlation (r)."
    )


def methods_fisher(row_col: str, col_col: str) -> str:
    return (
        f"The association between {row_col} and {col_col} was assessed "
        f"using Fisher's exact test (appropriate for small samples or low expected cell counts). "
        f"The odds ratio with 95% confidence interval was computed."
    )


def methods_kruskal(col: str, group_col: str) -> str:
    return (
        f"Differences in {col} across levels of {group_col} were tested "
        f"using the Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA). "
        f"Effect size was quantified with epsilon-squared. "
        f"When significant, pairwise post-hoc comparisons were performed using Dunn's test "
        f"with Holm correction for multiplicity."
    )


def methods_anova(col: str, group_col: str) -> str:
    return (
        f"Differences in {col} across levels of {group_col} were tested "
        f"using one-way analysis of variance (ANOVA). "
        f"Effect sizes were reported as eta-squared (\u03B7\u00B2) and omega-squared (\u03C9\u00B2). "
        f"Normality was checked per group; Levene's test assessed variance homogeneity. "
        f"Post-hoc pairwise comparisons used Tukey's HSD (equal variances) or "
        f"Games-Howell (unequal variances) with familywise error control."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TEXT — reports what was found (for Results section)
# ═══════════════════════════════════════════════════════════════════════════════

def _df_str(df) -> str:
    """Whole df prints bare; a Welch–Satterthwaite df keeps 2 decimals."""
    if not isinstance(df, (int, float)):
        return str(df)
    f = float(df)
    if f != f:  # NaN
        return ""
    return str(int(f)) if float(f).is_integer() else f"{f:.2f}"


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


def results_chisquare(result: dict) -> str:
    chi2 = result.get("chi2", 0)
    p = result.get("p", 1)
    dof = result.get("dof", 1)
    n = result.get("n", 0)
    sig = result.get("significant", False)
    exact = result.get("exact_test")
    es_list = result.get("effect_sizes", [])
    es_text = f", {_es_str(es_list[0])}" if es_list else ""

    lead = (
        "A significant association was found" if sig
        else "No significant association was found"
    )
    if exact:
        # Attaching the exact p to the chi-square statistic reads as though
        # that statistic produced it. The chi-square is still quoted, but as
        # the descriptive quantity it is here, not as the test.
        return (
            f"{lead} ({exact}, p = {_p_str(p)}; N = {n}, "
            f"\u03C7\u00B2({dof}) = {chi2:.2f}){es_text}."
        )
    return (
        f"{lead}, "
        f"\u03C7\u00B2({dof}, N = {n}) = {chi2:.2f}, p = {_p_str(p)}{es_text}."
    )


def results_mannwhitney(result: dict) -> str:
    g1, g2 = result.get("group1", "Group 1"), result.get("group2", "Group 2")
    u = result.get("U", 0)
    p = result.get("p", 1)
    med1 = result.get("median1", 0)
    med2 = result.get("median2", 0)
    sig = result.get("significant", False)
    es_list = result.get("effect_sizes", [])
    es_text = f", {_es_str(es_list[0])}" if es_list else ""

    return (
        f"A Mann-Whitney U test indicated that {g1} (Mdn = {med1:.2f}) "
        f"{'significantly differed from' if sig else 'did not significantly differ from'} "
        f"{g2} (Mdn = {med2:.2f}), U = {u:.1f}, p = {_p_str(p)}{es_text}."
    )


def results_fisher(result: dict) -> str:
    or_val = result.get("odds_ratio", 1)
    p = result.get("p", 1)
    sig = result.get("significant", False)
    es_list = result.get("effect_sizes", [])
    ci_text = ""
    if es_list and es_list[0].get("ci_low") is not None:
        ci_text = f" (95% CI: {es_list[0]['ci_low']:.2f}\u2013{es_list[0]['ci_high']:.2f})"

    return (
        f"Fisher's exact test {'revealed a significant association' if sig else 'showed no significant association'}, "
        f"p = {_p_str(p)}, OR = {or_val:.2f}{ci_text}."
    )


def results_kruskal(result: dict) -> str:
    h = result.get("H", 0)
    p = result.get("p", 1)
    sig = result.get("significant", False)
    es_list = result.get("effect_sizes", [])
    es_text = f", {_es_str(es_list[0])}" if es_list else ""

    return (
        f"A Kruskal-Wallis test {'showed a significant difference' if sig else 'showed no significant difference'} "
        f"across groups, H = {h:.2f}, p = {_p_str(p)}{es_text}."
    )


def results_anova(result: dict) -> str:
    f_val = result.get("F", 0)
    p = result.get("p", 1)
    df_b = result.get("df_between", "")
    df_w = result.get("df_within", "")
    sig = result.get("significant", False)
    es_list = result.get("effect_sizes", [])
    es_text = f", {_es_str(es_list[0])}" if es_list else ""

    return (
        f"A one-way ANOVA {'revealed a significant effect' if sig else 'revealed no significant effect'}, "
        f"F({df_b}, {df_w}) = {f_val:.2f}, p = {_p_str(p)}{es_text}."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# R CODE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def r_ttest_ind(col: str, group_col: str, welch: bool = False) -> str:
    """R equivalent of the test that was actually run.

    var.equal = TRUE is Student's pooled-variance test; Welch needs FALSE.
    Emitting TRUE for a Welch result made the snippet irreproducible.
    """
    var_equal = "FALSE" if welch else "TRUE"
    return f't.test({col} ~ {group_col}, data = data, var.equal = {var_equal})'

def r_ttest_one(col: str, mu: float) -> str:
    return f't.test(data${col}, mu = {mu})'

def r_chisquare(row_col: str, col_col: str, exact: str | None = None) -> str:
    tbl = f'table(data${row_col}, data${col_col})'
    if exact and exact.startswith("Fisher-Freeman-Halton"):
        # R's exact r x c network algorithm, not simulate.p.value: setting a
        # seed in R would not reproduce uSTAT's p anyway, since the two
        # permutation streams are unrelated. The exact p is what uSTAT's
        # resampling estimates, so it is the right thing to check against —
        # and the comment says so rather than implying the numbers will match
        # to the last digit.
        return (
            f'# uSTAT estimates this p by resampling (5000 permutations);\n'
            f'# R computes it exactly. At 5000 permutations the Monte Carlo\n# standard error is about 0.007, so expect agreement to ~0.015.\n'
            f'fisher.test({tbl})'
        )
    if exact:
        return f'fisher.test({tbl})'
    return f'chisq.test({tbl})'

def r_mannwhitney(col: str, group_col: str) -> str:
    return f'wilcox.test({col} ~ {group_col}, data = data)'

def r_fisher(row_col: str, col_col: str) -> str:
    return f'fisher.test(table(data${row_col}, data${col_col}))'

def r_kruskal(col: str, group_col: str) -> str:
    return f'kruskal.test({col} ~ {group_col}, data = data)'

def r_anova(col: str, group_col: str) -> str:
    return (
        f'model <- aov({col} ~ {group_col}, data = data)\n'
        f'summary(model)\n'
        f'TukeyHSD(model)'
    )
