"""Tests on counts: chi-square, Fisher, McNemar, Cochran's Q, Cochran-Armitage.

Endpoints under /api/stats and /api/categorical. The small-sample switches
uSTAT makes (Fisher below expected counts of 5, exact McNemar below 25
discordant pairs) are written out as the same rule, so the code reports the
p the result shows rather than only the textbook one.
"""
from __future__ import annotations

from ._common import (
    columns, complete_r, field, lit, number, nvec, pcol, plist, rcol, rname, rvec,
)


def _crosstab_cols(b: dict) -> tuple:
    return (field(b, "row_column", "row_col", default="row"),
            field(b, "col_column", "col_col", default="col"))


def chisquare(b: dict) -> dict:
    r, c = _crosstab_cols(b)
    return {
        "title": f"Chi-square test of independence: {r} x {c}",
        "python": (
            "import pandas as pd\n"
            "from scipy import stats\n\n"
            f"tab = pd.crosstab({pcol(r)}, {pcol(c)})\n"
            "res = stats.chi2_contingency(tab)  # Yates' correction on a 2x2, as chisq.test\n"
            "# uSTAT reports another p when any expected count is below 5:\n"
            "if (res.expected_freq < 5).any():\n"
            "    if tab.shape == (2, 2):\n"
            "        res = stats.fisher_exact(tab)\n"
            "    else:\n"
            "        # Labelled Fisher-Freeman-Halton (MC) in uSTAT, but it is a 5000-draw\n"
            "        # permutation test of the uncorrected chi-square (SciPy >= 1.15).\n"
            "        res = stats.chi2_contingency(\n"
            "            tab, correction=False, method=stats.PermutationMethod(n_resamples=5000))\n"
            "res"
        ),
        "r": (
            f"tab <- table({rcol(r)}, {rcol(c)})\n"
            "res <- chisq.test(tab)  # Yates' correction on a 2x2, as SciPy\n"
            "# uSTAT reports another p when any expected count is below 5:\n"
            "if (any(res$expected < 5)) {\n"
            "  # Labelled Fisher-Freeman-Halton (MC) in uSTAT, but ordered by the\n"
            "  # chi-square: R's Monte Carlo chi-square is the same test, up to the draws.\n"
            "  res <- if (all(dim(tab) == 2)) fisher.test(tab) else\n"
            "    chisq.test(tab, simulate.p.value = TRUE, B = 5000)\n"
            "}\n"
            "res"
        ),
    }


def fisher(b: dict) -> dict:
    r, c = _crosstab_cols(b)
    return {
        "title": f"Fisher's exact test: {r} x {c}",
        "python": (
            "import pandas as pd\n"
            "from scipy import stats\n\n"
            f"tab = pd.crosstab({pcol(r)}, {pcol(c)})  # must be 2x2\n"
            "stats.fisher_exact(tab)  # the sample odds ratio ad/bc, as uSTAT reports"
        ),
        "r": (
            f"tab <- table({rcol(r)}, {rcol(c)})\n"
            "# Same p. R's estimate is the conditional MLE odds ratio with an exact CI;\n"
            "# uSTAT reports the sample odds ratio ad/bc with a Woolf interval.\n"
            "fisher.test(tab)"
        ),
    }


def mcnemar(b: dict) -> dict:
    c1 = field(b, "col1", "column1", default="before")
    c2 = field(b, "col2", "column2", default="after")
    return {
        "title": f"McNemar's test: {c1} vs {c2}",
        "python": (
            "import pandas as pd\n"
            "from statsmodels.stats.contingency_tables import mcnemar\n\n"
            f"tab = pd.crosstab({pcol(c1)}, {pcol(c2)}).values\n"
            "# uSTAT's rule: the exact binomial test below 25 discordant pairs,\n"
            "# else the chi-square with continuity correction.\n"
            "mcnemar(tab, exact=bool(tab[0, 1] + tab[1, 0] < 25))"
        ),
        "r": (
            f"tab <- table({rcol(c1)}, {rcol(c2)})\n"
            "# uSTAT's rule: the exact binomial test below 25 discordant pairs,\n"
            "# else the chi-square with continuity correction.\n"
            "if (tab[1, 2] + tab[2, 1] < 25) {\n"
            "  binom.test(tab[1, 2], tab[1, 2] + tab[2, 1])  # exact McNemar\n"
            "} else {\n"
            "  mcnemar.test(tab, correct = TRUE)\n"
            "}"
        ),
    }


def cochran_q(b: dict) -> dict:
    cols = columns(b, "columns") or ["c1", "c2", "c3"]
    return {
        "title": f"Cochran's Q test: {', '.join(cols)}",
        "python": (
            "from statsmodels.stats.contingency_tables import cochrans_q\n\n"
            f"cochrans_q(df[{plist(cols)}].dropna())\n"
            "# Post hoc when p < alpha: pairwise McNemar (same exact rule), Holm-adjusted."
        ),
        "r": (
            "library(DescTools)\n\n"
            f"cols <- {rvec(cols)}\n"
            "CochranQTest(as.matrix(df[complete.cases(df[, cols]), cols]))\n"
            "# Post hoc when p < alpha: pairwise McNemar (same exact rule), Holm-adjusted."
        ),
    }


def _success(value) -> str:
    """The event value as a literal: numeric when it reads as a number, the
    way uSTAT casts it to the column's type."""
    if value is None:
        return "1"
    try:
        float(value)
    except (TypeError, ValueError):
        return lit(value)
    return number(value)


def cochran_armitage(b: dict) -> dict:
    x = field(b, "ordinal_col", "ordinal_column", default="dose")
    ev = field(b, "event_col", "event_column", default="event")
    order = b.get("level_order")
    scores = b.get("scores")
    success = _success(b.get("success_value"))
    order_note = "" if order else (
        "# Low to high. uSTAT takes the order from the Data Dictionary, numeric\n"
        "# codes or a recognised scale and lists it in the result; set it here.\n")
    ev_note = "" if b.get("success_value") is not None else (
        "# uSTAT counts 1 as the event in a 0/1 column; in any other two-valued\n"
        "# column it takes the most frequent value.\n")
    # level_order holds the levels as text, which is how uSTAT matches them.
    py_x = f"d[{lit(x)}].astype(str)" if order else f"d[{lit(x)}]"
    py_levels = plist(order) if order else f"sorted(d[{lit(x)}].unique())"
    r_levels = rvec(order) if order else f"sort(unique(d${rname(x)}))"
    py_scores = nvec(scores, "py") if scores else "range(len(levels))"
    r_scores = nvec(scores, "r") if scores else "seq_along(lv) - 1"
    return {
        "title": f"Cochran-Armitage trend: {ev} across {x}",
        "python": (
            "import numpy as np\n"
            "import pandas as pd\n"
            "from statsmodels.stats.contingency_tables import Table\n\n"
            f"d = df[{plist([x, ev])}].dropna()\n"
            + order_note + f"levels = {py_levels}\n" + ev_note
            + f"event = (d[{lit(ev)}] == {success}).astype(int)\n"
            f"tab = pd.crosstab(pd.Categorical({py_x}, categories=levels, ordered=True), event)\n"
            "# statsmodels' linear-by-linear test conditions on the margins, so its z is\n"
            "# uSTAT's times sqrt((N - 1) / N); prop.trend.test in R matches uSTAT exactly.\n"
            f"Table(tab).test_ordinal_association(row_scores=np.array({py_scores}, dtype=float),\n"
            "                                    col_scores=np.array([0.0, 1.0]))"
        ),
        "r": (
            complete_r([x, ev]) + order_note + f"lv <- {r_levels}\n" + ev_note
            + f"tab <- table(factor(d${rname(x)}, levels = lv), d${rname(ev)} == {success})\n"
            "# Chi-square = the Cochran-Armitage z squared; same two-sided p as uSTAT.\n"
            f'prop.trend.test(tab[, "TRUE"], rowSums(tab), score = {r_scores})'
        ),
    }


ENDPOINTS = {
    "/api/stats/chisquare": chisquare,
    "/api/stats/fisher": fisher,
    "/api/categorical/mcnemar": mcnemar,
    "/api/categorical/cochran_q": cochran_q,
    "/api/categorical/cochran_armitage": cochran_armitage,
}
