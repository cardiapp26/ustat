"""Correlation and agreement: Pearson / Spearman / Kendall, ICC, Cronbach's alpha.

Endpoints under /api/stats and /api/reliability. R's cor.test would go exact
for small rank-correlation samples where SciPy uses its large-sample p, so
the rank methods are pinned with exact = FALSE and the comment says when
SciPy itself goes exact.
"""
from __future__ import annotations

from ._common import columns, field, imputation_note, lit, plist, rcol, rvec

_METHODS = ("pearson", "spearman", "kendall")
_PY_TEST = {"pearson": "pearsonr", "spearman": "spearmanr", "kendall": "kendalltau"}
_R_EXACT = {"pearson": "", "spearman": ", exact = FALSE", "kendall": ", exact = FALSE"}
_RANK_NOTE = {
    "pearson": "",
    "spearman": "# exact = FALSE: SciPy's t approximation for rho.\n",
    "kendall": ("# exact = FALSE: SciPy's normal approximation for tau-b; SciPy itself goes\n"
                "# exact below 34 pairs without ties, where R would up to 49.\n"),
}


def _method(b: dict, default: str) -> str:
    m = (b.get("method") or default).lower()
    return m if m in _METHODS or m == "auto" else default


def correlation_pair(b: dict) -> dict:
    x = field(b, "var1", default="x")
    y = field(b, "var2", default="y")
    method = _method(b, "auto")
    chosen = ["pearson", "spearman"] if method == "auto" else [method]
    auto_note = (
        "# method = \"auto\": uSTAT uses Pearson when both variables pass its normality\n"
        "# check (Shapiro-Wilk under n = 50, Lilliefors up to 2000), else Spearman.\n"
        "# The result names the one it used.\n" if method == "auto" else "")
    ci_note = ("" if chosen == ["pearson"] else
               "# uSTAT puts the Fisher z interval (se = 1 / sqrt(n - 3)) on rho and tau as\n"
               "# well; neither library reports one for them.\n")
    note = imputation_note(b.get("imputation"))
    py_calls = "\n".join(
        f"stats.{_PY_TEST[m]}(d[{lit(x)}], d[{lit(y)}])"
        + ("  # .confidence_interval(): Fisher z, as uSTAT" if m == "pearson" else "")
        for m in chosen)
    r_calls = "".join(
        _RANK_NOTE[m] + f'cor.test({rcol(x, "d")}, {rcol(y, "d")}, method = "{m}"{_R_EXACT[m]})\n'
        for m in chosen)
    return {
        "title": f"Correlation: {x} and {y}",
        "python": ("from scipy import stats\n\n" + note
                   + f"d = df[{plist([x, y])}].dropna()\n" + auto_note + ci_note + py_calls),
        "r": (note + f"d <- df[complete.cases(df[, {rvec([x, y])}]), ]\n"
              + auto_note + ci_note + r_calls.rstrip("\n")),
    }


def _matrix(cols_py: str, cols_r: str, method: str, complete: bool) -> dict:
    """Pairwise r and p. `complete` drops rows missing any variable first
    (the POST endpoint); otherwise each pair uses its own complete rows."""
    py_frame = ("d = df[cols].apply(pd.to_numeric, errors=\"coerce\").dropna()\n" if complete
                else "d = df.select_dtypes(\"number\")\n")
    r_frame = ("d <- na.omit(df[, cols])\n" if complete
               else "d <- df[, sapply(df, is.numeric)]\n")
    use = "" if complete else ', use = "pairwise.complete.obs"'
    pair_py = "d[[a, b]].dropna()" if not complete else "d"
    return {
        "python": (
            "from itertools import combinations\n"
            "import pandas as pd\n"
            "from scipy import stats\n\n"
            + cols_py + py_frame
            + f'd.corr(method="{method}")\n'
            + (_RANK_NOTE[method] if method != "pearson" else "")
            + f"{{(a, b): stats.{_PY_TEST[method]}({pair_py}[a], {pair_py}[b]).pvalue\n"
            " for a, b in combinations(d.columns, 2)}"
        ),
        "r": (
            cols_r + r_frame
            + f'cor(d, method = "{method}"{use})\n'
            + _RANK_NOTE[method]
            + "pairs <- combn(names(d), 2, simplify = FALSE)\n"
            f'sapply(pairs, function(p) cor.test(d[[p[1]]], d[[p[2]]], method = "{method}"{_R_EXACT[method]})$p.value)'
        ),
    }


def correlation_matrix(b: dict) -> dict:
    cols = columns(b, "variables") or ["x", "y"]
    method = _method(b, "pearson")
    if method == "auto":
        method = "pearson"
    note = imputation_note(b.get("imputation"))
    out = _matrix(f"cols = {plist(cols)}\n", f"cols <- {rvec(cols)}\n", method, complete=True)
    return {
        "title": f"Correlation matrix ({method}): {', '.join(cols)}",
        "python": note + "# Complete cases on every variable, as uSTAT.\n" + out["python"],
        "r": note + "# Complete cases on every variable, as uSTAT.\n" + out["r"],
    }


def correlation_all_numeric(b: dict) -> dict:
    method = _method(b, "pearson")
    if method == "auto":
        method = "pearson"
    out = _matrix("", "", method, complete=False)
    return {
        "title": f"Correlation matrix ({method}): every numeric column",
        "python": "# Each pair on its own complete rows, as pandas and uSTAT.\n" + out["python"],
        "r": "# Each pair on its own complete rows, as pandas and uSTAT.\n" + out["r"],
    }


def icc(b: dict) -> dict:
    r1 = field(b, "rater1_col", "rater1_column", default="rater1")
    r2 = field(b, "rater2_col", "rater2_column", default="rater2")
    return {
        "title": f"Intraclass correlation ICC(A,1): {r1} and {r2}",
        "python": ("# No intraclass correlation in scipy or statsmodels; see the R version.\n"
                   "# uSTAT reports ICC(A,1) with McGraw and Wong's interval and F test."),
        "r": (
            "library(irr)\n\n"
            "# Two-way random effects, absolute agreement, single rater: ICC(A,1),\n"
            "# with McGraw and Wong's interval and F test, as uSTAT reports.\n"
            f"icc(na.omit(df[, {rvec([r1, r2])}]), model = \"twoway\", type = \"agreement\", unit = \"single\")"
        ),
    }


def cronbach(b: dict) -> dict:
    items = columns(b, "items") or ["q1", "q2", "q3"]
    return {
        "title": f"Cronbach's alpha: {', '.join(items)}",
        "python": ("# No Cronbach's alpha in scipy or statsmodels; see the R version.\n"
                   "# uSTAT computes it on complete cases, with corrected item-total r\n"
                   "# and alpha if an item is dropped."),
        "r": (
            "library(psych)\n\n"
            f"items <- na.omit(df[, {rvec(items)}])  # complete cases, as uSTAT\n"
            "a <- alpha(items)\n"
            "a$total$raw_alpha     # Cronbach's alpha\n"
            "a$item.stats$r.drop   # corrected item-total r\n"
            "a$alpha.drop$raw_alpha  # alpha if the item is dropped\n"
            "# uSTAT's omega comes from a one-factor ML fit and the model-implied total\n"
            "# variance; psych's omega total uses the observed one, so they differ a little.\n"
            'omega(items, nfactors = 1, fm = "ml", plot = FALSE)$omega.tot'
        ),
    }


ENDPOINTS = {
    "/api/stats/correlation_pair": correlation_pair,
    "/api/stats/correlation_matrix": correlation_matrix,
    "/api/stats/{sid}/correlation": correlation_all_numeric,
    "/api/stats/icc": icc,
    "/api/reliability/cronbach": cronbach,
}
