"""Comparing groups or repeated measures: t-tests, ANOVA, rank tests, normality.

Endpoints under /api/stats and /api/repeated. Where SciPy (what uSTAT runs)
and R pick exact vs asymptotic p-values at different sample sizes, the R
call is pinned to SciPy's choice and the comment says where they still part.
"""
from __future__ import annotations

from ._common import (
    columns, complete_r, field, groups_py, lit, number, pcol, plist, rcol,
    rname, rvec, two_groups_py,
)


def _variance_rule(b: dict) -> str:
    """Student, Welch or Levene-decided, in the precedence uSTAT applies:
    an explicit method, then the legacy equal_var flag, then Levene."""
    method = b.get("method") or "auto"
    if method in ("welch", "student"):
        return method
    if b.get("equal_var") is not None:
        return "student" if b["equal_var"] else "welch"
    return "levene"


def sign_test(b: dict) -> dict:
    y = field(b, "column", default="outcome")
    other = field(b, "comparison_column")
    mu = number(b.get("mu"), 0)
    alternative = b.get("alternative", "two-sided")
    alpha = number(b.get("alpha"), 0.05)
    selected = [y, other] if other else [y]
    py_difference = f"d[{lit(y)}] - d[{lit(other)}]" if other else f"d[{lit(y)}] - {mu}"
    r_difference = f"{rcol(y, 'd')} - {rcol(other, 'd')}" if other else f"{rcol(y, 'd')} - {mu}"
    return {"title": "Exact sign test",
            "python": "import pandas as pd\nfrom scipy import stats\n" + f"d = df[{plist(selected)}].apply(pd.to_numeric, errors='coerce').dropna()\ndifference = {py_difference}\ndifference = difference[difference != 0]\n" + f"res = stats.binomtest(int((difference > 0).sum()), len(difference), p=0.5, alternative={lit(alternative)})\nres.proportion_ci(confidence_level=1-{alpha}, method='exact')",
            "r": f"d <- na.omit(df[, {rvec(selected)}, drop=FALSE])\ndifference <- {r_difference}\ndifference <- difference[difference != 0]\n" + f"binom.test(sum(difference > 0), length(difference), p=0.5, alternative={lit(alternative)}, conf.level=1-{alpha})"}


def wilcoxon_onesample(b: dict) -> dict:
    y = field(b, "column", default="outcome")
    mu = number(b.get("mu"), 0)
    alternative = b.get("alternative", "two-sided")
    return {"title": "One-sample Wilcoxon signed-rank test",
            "python": "import pandas as pd\nfrom scipy import stats\n" + f"x = pd.to_numeric(df[{lit(y)}], errors='coerce').dropna()\nstats.wilcoxon(x-{mu}, alternative={lit(alternative)})",
            "r": f"x <- na.omit({rcol(y)})\nwilcox.test(x, mu={mu}, alternative={lit(alternative)}, correct=FALSE)"}


_LEVENE_COMMENT = (
    "# uSTAT's rule: Levene's test (median-centred, the default in both SciPy\n"
    "# and car) at p < 0.05 switches to Welch.\n"
)


def ttest(b: dict) -> dict:
    y = field(b, "column", default="outcome")
    g = field(b, "group_column", "group_col")
    if not g:
        mu = number(b.get("mu"), 0)
        return {
            "title": f"One-sample t-test: {y} vs mu = {mu}",
            "python": f"from scipy import stats\n\nstats.ttest_1samp({pcol(y)}.dropna(), popmean={mu})",
            "r": f"t.test({rcol(y)}, mu = {mu})",
        }
    rule = _variance_rule(b)
    fml = f"{rname(y)} ~ {rname(g)}"
    if rule == "levene":
        py_call = (_LEVENE_COMMENT + "equal_var = stats.levene(a, b).pvalue >= 0.05\n"
                   "stats.ttest_ind(a, b, equal_var=equal_var)")
        r_call = (_LEVENE_COMMENT
                  + f"lev <- car::leveneTest({rname(y)} ~ factor({rname(g)}), data = d)\n"
                  f't.test({fml}, data = d, var.equal = lev[1, "Pr(>F)"] >= 0.05)')
    else:
        equal = rule == "student"
        py_call = f"stats.ttest_ind(a, b, equal_var={equal})  # {rule.title()}, as requested"
        r_call = f"t.test({fml}, data = d, var.equal = {str(equal).upper()})"
    return {
        "title": f"Independent t-test: {y} by {g}",
        "python": "from scipy import stats\n\n" + two_groups_py(y, g) + py_call,
        "r": complete_r([y, g]) + r_call,
    }


def paired_ttest(b: dict) -> dict:
    c1 = field(b, "col1", "column1", default="before")
    c2 = field(b, "col2", "column2", default="after")
    return {
        "title": f"Paired t-test: {c1} vs {c2}",
        "python": (
            "from scipy import stats\n\n"
            f"d = df[{plist([c1, c2])}].dropna()  # complete pairs\n"
            f"stats.ttest_rel(d[{lit(c1)}], d[{lit(c2)}])"
        ),
        "r": f"t.test({rcol(c1)}, {rcol(c2)}, paired = TRUE)",
    }


def anova(b: dict) -> dict:
    y = field(b, "column", default="outcome")
    g = field(b, "group_column", "group_col", default="group")
    choice = str(b.get("posthoc") or "auto").lower()
    control = b.get("control_group")
    forced = bool(b.get("force_posthoc"))
    gate = ("# Post hoc, requested even without a significant omnibus (planned contrasts):\n"
            if forced else
            "# Post hoc, run only when p < 0.05 with 3+ groups:\n")
    if choice == "dunnett":
        ctrl = str(control) if control is not None else "the first group"
        py_ph = (gate
                 + f"# Dunnett's many-to-one comparisons against the control ({ctrl}).\n"
                 f"by_name = {{str(k): x.dropna() for k, x in df.groupby({lit(g)})[{lit(y)}]}}\n"
                 + (f"control = by_name.pop({str(control)!r})\n" if control is not None
                    else "control = by_name.pop(sorted(by_name)[0])\n")
                 + "stats.dunnett(*by_name.values(), control=control)")
        r_ph = (gate
                + f"# DescTools::DunnettTest, control = the reference arm.\n"
                f'DescTools::DunnettTest({rname(y)} ~ factor({rname(g)}), data = df'
                + (f', control = "{control}"' if control is not None else "") + ")")
    elif choice == "scheffe":
        py_ph = gate + "import scikit_posthocs as sp\nsp.posthoc_scheffe(groups)"
        r_ph = gate + f"DescTools::ScheffeTest(aov({rname(y)} ~ factor({rname(g)}), data = df))"
    elif choice == "bonferroni":
        py_ph = gate + "import scikit_posthocs as sp\nsp.posthoc_ttest(groups, pool_sd=True, p_adjust='bonferroni')"
        r_ph = gate + f"pairwise.t.test(df[[{lit(y)}]], df[[{lit(g)}]], p.adjust.method='bonferroni', pool.sd=TRUE)"
    elif choice == "games_howell":
        py_ph = gate + "# Games-Howell (not in SciPy; see the R comment).\n"
        r_ph = gate + f"rstatix::games_howell_test(df, {rname(y)} ~ {rname(g)})"
    elif choice == "none":
        py_ph = "# Post hoc disabled in the request.\n"
        r_ph = "# Post hoc disabled in the request.\n"
    elif choice == "tukey":
        py_ph = gate + "stats.tukey_hsd(*groups)"
        r_ph = gate + f"TukeyHSD(aov({rname(y)} ~ factor({rname(g)}), data = df))"
    else:  # auto: the Levene-driven switch
        py_ph = (gate
                 + "# Tukey HSD, or Games-Howell after Welch (not in SciPy; see the R comment).\n"
                 "stats.tukey_hsd(*groups)")
        r_ph = (gate
                + f"# Games-Howell after Welch (rstatix::games_howell_test({rname(y)} ~ {rname(g)})).\n"
                f"TukeyHSD(aov({rname(y)} ~ factor({rname(g)}), data = df))")
    return {
        "title": f"One-way ANOVA: {y} by {g}",
        "python": (
            "from scipy import stats\n"
            "from statsmodels.stats.oneway import anova_oneway\n\n"
            + groups_py(y, g) + _LEVENE_COMMENT
            + "if stats.levene(*groups).pvalue < 0.05:\n"
            '    res = anova_oneway(groups, use_var="unequal")  # Welch\'s ANOVA\n'
            "else:\n"
            "    res = stats.f_oneway(*groups)\n"
            "res\n"
            + py_ph
        ),
        "r": (
            _LEVENE_COMMENT
            + f"lev <- car::leveneTest({rname(y)} ~ factor({rname(g)}), data = df)\n"
            f'oneway.test({rname(y)} ~ factor({rname(g)}), data = df, var.equal = lev[1, "Pr(>F)"] >= 0.05)\n'
            + r_ph
        ),
    }


def mannwhitney(b: dict) -> dict:
    y = field(b, "column", default="outcome")
    g = field(b, "group_column", "group_col", default="group")
    return {
        "title": f"Mann-Whitney U: {y} by {g}",
        "python": (
            "from scipy import stats\n\n" + two_groups_py(y, g)
            + "# method=\"auto\": exact when a group has 8 or fewer values and no ties,\n"
            "# else the normal approximation with continuity and tie corrections.\n"
            'stats.mannwhitneyu(a, b, alternative="two-sided")'
        ),
        "r": (
            "# exact = FALSE: the normal approximation with continuity correction that\n"
            "# SciPy uses unless a group has 8 or fewer values and no ties (R alone\n"
            "# would go exact up to n = 50). W is U for the first group, as in SciPy.\n"
            f"wilcox.test({rname(y)} ~ {rname(g)}, data = df, exact = FALSE, correct = TRUE)"
        ),
    }


def wilcoxon_signed_rank(b: dict) -> dict:
    c1 = field(b, "col1", "column1", default="before")
    c2 = field(b, "col2", "column2", default="after")
    return {
        "title": f"Wilcoxon signed-rank: {c1} vs {c2}",
        "python": (
            "from scipy import stats\n\n"
            f"d = df[{plist([c1, c2])}].dropna()  # complete pairs\n"
            "# Zero differences are dropped; no continuity correction.\n"
            f'stats.wilcoxon(d[{lit(c1)}], d[{lit(c2)}], alternative="two-sided")'
        ),
        "r": (
            "# correct = FALSE matches SciPy's normal approximation. Both go exact up\n"
            "# to about 50 pairs without ties or zero differences; with ties or zeros\n"
            "# SciPy permutes exactly up to 13 pairs where R stays approximate.\n"
            "# R's V is the positive-rank sum; uSTAT's W is the smaller of the two.\n"
            f"wilcox.test({rcol(c1)}, {rcol(c2)}, paired = TRUE, correct = FALSE)"
        ),
    }


def kruskal(b: dict) -> dict:
    y = field(b, "column", default="outcome")
    g = field(b, "group_column", "group_col", default="group")
    corr = (b.get("posthoc_correction") or "holm").lower()
    if corr not in ("holm", "bonferroni", "fdr", "none"):
        corr = "holm"
    return {
        "title": f"Kruskal-Wallis: {y} by {g}",
        "python": (
            "from scipy import stats\n\n" + groups_py(y, g)
            + "stats.kruskal(*groups)  # tie-corrected H\n"
            f"# Post hoc when p < 0.05 with 3+ groups: Dunn's z on mean ranks ({corr} adjustment)\n"
            "# (no SciPy routine; uSTAT's version applies no tie correction)."
        ),
        "r": (
            f"kruskal.test({rname(y)} ~ factor({rname(g)}), data = df)\n"
            "# Post hoc when p < 0.05 with 3+ groups. FSA's Dunn test corrects for\n"
            "# ties and uSTAT's does not, so tied data give slightly different p.\n"
            f'# FSA::dunnTest({rname(y)} ~ factor({rname(g)}), data = df, method = "{corr}")'
        ),
    }


def friedman(b: dict) -> dict:
    cols = columns(b, "columns") or ["t1", "t2", "t3"]
    return {
        "title": f"Friedman test: {', '.join(cols)}",
        "python": (
            "from itertools import combinations\n"
            "from scipy import stats\n"
            "from statsmodels.stats.multitest import multipletests\n\n"
            f"cols = {plist(cols)}\n"
            "d = df[cols].dropna()  # subjects complete on every condition\n"
            "stats.friedmanchisquare(*[d[c] for c in cols])\n"
            "# Post hoc when p < alpha: pairwise Wilcoxon signed-rank, Holm-adjusted.\n"
            "pairs = list(combinations(cols, 2))\n"
            "raw = [stats.wilcoxon(d[x], d[y]).pvalue for x, y in pairs]\n"
            'multipletests(raw, method="holm")[1]'
        ),
        "r": (
            f"cols <- {rvec(cols)}\n"
            "d <- df[complete.cases(df[, cols]), cols]\n"
            "friedman.test(as.matrix(d))\n"
            "# Post hoc when p < alpha: pairwise Wilcoxon signed-rank, Holm-adjusted.\n"
            "pairs <- combn(cols, 2, simplify = FALSE)\n"
            "raw <- sapply(pairs, function(p) wilcox.test(d[[p[1]]], d[[p[2]]], paired = TRUE, correct = FALSE)$p.value)\n"
            'p.adjust(raw, method = "holm")'
        ),
    }


def jonckheere(b: dict) -> dict:
    y = field(b, "column", default="outcome")
    g = field(b, "group_column", "group_col", default="group")
    scores = b.get("scores")
    order_note = (
        "# The request's scores reorder the levels; only their ranking matters.\n"
        if scores else ""
    )
    return {
        "title": f"Jonckheere-Terpstra trend: {y} across {g}",
        "python": "# No Jonckheere-Terpstra test in scipy or statsmodels; see the R version.",
        "r": (
            "library(DescTools)\n\n" + complete_r([y, g])
            + "# The trend is measured across this order, low to high. uSTAT takes it\n"
            "# from the Data Dictionary, numeric codes or a recognised scale and lists\n"
            "# it in the result; set the same order here.\n" + order_note
            + f"lv <- sort(unique(d${rname(g)}))\n"
            "# exact = FALSE: uSTAT's normal approximation, with no tie correction.\n"
            f"JonckheereTerpstraTest(d${rname(y)}, ordered(d${rname(g)}, levels = lv), exact = FALSE)"
        ),
    }


def normality(b: dict) -> dict:
    cols = columns(b, "variables") or ["x"]
    g = field(b, "group_column", "group_col")
    py_loop = (f"for col in {plist(cols)}:\n"
               + (f"    for g, x in df.groupby({lit(g)})[col]:\n        x = x.dropna()\n"
                  if g else "    x = df[col].dropna()\n"))
    ind = "        " if g else "    "
    py_body = "".join(ind + line + "\n" for line in (
        "print(stats.shapiro(x))  # uSTAT's primary test up to n = 5000",
        "print(stats.anderson(x))  # A2 only; uSTAT's p is nortest::ad.test's",
        '# p from a simulated table; uSTAT reproduces nortest::lillie.test instead.',
        'print(lilliefors(x, dist="norm"))',
        "print(stats.normaltest(x), stats.jarque_bera(x))  # n >= 20 and n >= 30",
        "print(stats.skew(x, bias=False), stats.kurtosis(x, bias=False))  # G1, G2",
    ))
    r_head = (f"for (col in {rvec(cols)}) {{\n"
              + (f"  for (x in split(df[[col]], df${rname(g)})) {{\n    x <- na.omit(x)\n"
                 if g else "  x <- na.omit(df[[col]])\n"))
    rind = "    " if g else "  "
    r_body = "".join(rind + line + "\n" for line in (
        "print(shapiro.test(x))",
        "print(nortest::ad.test(x))",
        "print(nortest::lillie.test(x))",
        "print(moments::jarque.test(x))  # D'Agostino-Pearson K2: fBasics::dagoTest(x)",
        "print(c(e1071::skewness(x, type = 2), e1071::kurtosis(x, type = 2)))",
    ))
    return {
        "title": f"Normality: {', '.join(cols)}" + (f" by {g}" if g else ""),
        "python": ("from scipy import stats\n"
                   "from statsmodels.stats.diagnostic import lilliefors\n\n"
                   + py_loop + py_body.rstrip("\n")),
        "r": r_head + r_body + ("  }\n}" if g else "}"),
    }


ENDPOINTS = {
    "/api/stats/sign_test": sign_test,
    "/api/stats/wilcoxon_onesample": wilcoxon_onesample,
    "/api/stats/ttest": ttest,
    "/api/repeated/paired_ttest": paired_ttest,
    "/api/stats/anova": anova,
    "/api/stats/mannwhitney": mannwhitney,
    "/api/repeated/wilcoxon_signed_rank": wilcoxon_signed_rank,
    "/api/stats/kruskal": kruskal,
    "/api/repeated/friedman": friedman,
    "/api/stats/jonckheere_terpstra": jonckheere,
    "/api/stats/normality": normality,
}
