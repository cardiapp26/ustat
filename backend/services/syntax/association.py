"""Correlation and agreement: Pearson / Spearman / Kendall, ICC, Cronbach's alpha.

Endpoints under /api/stats and /api/reliability. R's cor.test would go exact
for small rank-correlation samples where SciPy uses its large-sample p, so
the rank methods are pinned with exact = FALSE and the comment says when
SciPy itself goes exact.
"""
from __future__ import annotations

from ._common import columns, field, imputation_note, lit, plist, rcol, rvec

_METHODS = ("pearson", "spearman", "kendall", "pointbiserial")
_PY_TEST = {"pearson": "pearsonr", "spearman": "spearmanr", "kendall": "kendalltau", "pointbiserial": "pointbiserialr"}
_R_EXACT = {"pearson": "", "spearman": ", exact = FALSE", "kendall": ", exact = FALSE", "pointbiserial": ""}
_RANK_NOTE = {
    "pearson": "",
    "pointbiserial": "# One variable must be binary; point-biserial equals Pearson on its 0/1 codes.\n",
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
    if method == "pointbiserial":
        return {"title": f"Point-biserial correlation: {x} and {y}",
                "python": "from scipy import stats\n" + f"d = df[{plist([x,y])}].dropna()\n" + "binary_col = next(c for c in d if d[c].nunique() == 2)\ncontinuous_col = next(c for c in d if c != binary_col)\nlevels = sorted(d[binary_col].unique())\nbinary = d[binary_col] == levels[-1]\nstats.pointbiserialr(binary, d[continuous_col])",
                "r": f"d <- na.omit(df[, {rvec([x,y])}])\n" + "binary_col <- names(d)[sapply(d, function(x) length(unique(x)) == 2)][1]\ncontinuous_col <- setdiff(names(d), binary_col)[1]\nlevels <- sort(unique(d[[binary_col]]))\nbinary <- as.numeric(d[[binary_col]] == tail(levels, 1))\ncor.test(binary, d[[continuous_col]], method = \"pearson\")"}
    ci_note = ("" if chosen == ["pearson"] else
               "# uSTAT puts the Fisher z interval (se = 1 / sqrt(n - 3)) on rho and tau as\n"
               "# well; neither library reports one for them.\n")
    note = imputation_note(b.get("imputation"))
    py_calls = "\n".join(
        f"stats.{_PY_TEST[m]}(d[{lit(x)}], d[{lit(y)}])"
        + ("  # .confidence_interval(): Fisher z, as uSTAT" if m == "pearson" else "")
        for m in chosen)
    r_calls = "".join(
        _RANK_NOTE[m] + f'cor.test({rcol(x, "d")}, {rcol(y, "d")}, method = "{dict(pointbiserial="pearson").get(m, m)}"{_R_EXACT[m]})\n'
        for m in chosen)
    return {
        "title": f"Correlation: {x} and {y}",
        "python": ("from scipy import stats\n\n" + note
                   + f"d = df[{plist([x, y])}].dropna()\n" + auto_note + ci_note + py_calls),
        "r": (note + f"d <- df[complete.cases(df[, {rvec([x, y])}]), ]\n"
              + auto_note + ci_note + r_calls.rstrip("\n")),
    }


def partial_correlation(b: dict) -> dict:
    x = field(b, "var1", default="x")
    y = field(b, "var2", default="y")
    controls = columns(b, "controls") or ["z"]
    m = (b.get("method") or "pearson").lower()
    m = m if m in ("pearson", "spearman") else "pearson"
    note = imputation_note(b.get("imputation"))
    cols = [x, y] + list(controls)
    return {
        "title": f"Partial correlation: {x} and {y} | {', '.join(controls)}",
        "python": (
            "import pingouin as pg\n\n" + note
            + f"d = df[{plist(cols)}].dropna()\n"
            "# Same construction as uSTAT: correlation of the OLS residuals after\n"
            "# regressing each variable on the controls; t on n - 2 - k df,\n"
            "# Fisher z CI with se = 1 / sqrt(n - 3 - k).\n"
            f"pg.partial_corr(data=d, x={lit(x)}, y={lit(y)}, covar={plist(list(controls))}, method=\"{m}\")"
        ),
        "r": (
            "library(ppcor)\n\n" + note
            + f"d <- df[complete.cases(df[, {rvec(cols)}]), ]\n"
            f'pcor.test({rcol(x, "d")}, {rcol(y, "d")}, d[, {rvec(list(controls))}], method = "{m}")'
        ),
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
    raters = columns(b, "rater_cols") or [field(b, "rater1_col", "rater1_column", default="rater1"), field(b, "rater2_col", "rater2_column", default="rater2")]
    agreement = "consistency" if b.get("agreement") == "consistency" else "agreement"
    unit = "average" if b.get("unit") == "average" else "single"
    return {"title": f"ICC ({agreement}, {unit}): {', '.join(raters)}",
            "python": "# See R irr::icc for two-way ICC and its confidence interval.",
            "r": "library(irr)\n" + f'icc(na.omit(df[, {rvec(raters)}]), model = "twoway", type = "{agreement}", unit = "{unit}")'}


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
            "a <- alpha(items, check.keys = FALSE)\n"
            "a$total$raw_alpha     # Cronbach's alpha\n"
            "a$total$std.alpha     # Standardized Cronbach's alpha\n"
            "a$item.stats$r.drop   # corrected item-total r\n"
            "a$alpha.drop$raw_alpha  # alpha if the item is dropped\n"
            "# psych::omega uses its own factor model and keying defaults; this is\n"
            "# an independent comparison, not exact replication of uSTAT omega.\n"
            'omega(items, nfactors = 1, fm = "ml", plot = FALSE)$omega.tot'
        ),
    }


def cohens_kappa(b: dict) -> dict:
    x = field(b, "rater1_col", "rater1_column", default="rater1")
    y = field(b, "rater2_col", "rater2_column", default="rater2")
    weights = b.get("weights") if b.get("weights") in ("linear", "quadratic") else None
    order = columns(b, "level_order")
    return {"title": "Cohen's kappa", "python": "from sklearn.metrics import cohen_kappa_score\nimport numbers\n" + f"d = df[{plist([x,y])}].dropna()\n" + "def level_key(v):\n    if isinstance(v, numbers.Real) and float(v).is_integer():\n        return str(int(v))\n    return str(v)\nd = d.apply(lambda col: col.map(level_key))\n" + (f"levels = {plist(order)}\n" if order else "# Set levels to the ordinal order shown in the uSTAT result.\nlevels = None\n") + f"cohen_kappa_score(d[{lit(x)}], d[{lit(y)}], labels=levels, weights={repr(weights)})",
            "r": "library(irr)\n" + f"d <- na.omit(df[, {rvec([x,y])}])\n" + "d[] <- lapply(d, as.character)\n" + (f"d[] <- lapply(d, factor, levels = {rvec(order)}, ordered = TRUE)\n" if order else "# For weighted kappa, factor levels must match the order shown in uSTAT.\n") + f'kappa2(d, weight = "{dict(linear="equal", quadratic="squared").get(weights,"unweighted")}")'}


def ordinal_association(b: dict) -> dict:
    x = field(b, "row_column", default="x")
    y = field(b, "col_column", default="y")
    rows, cols = columns(b, "row_order"), columns(b, "col_order")
    py_order = (f"row_levels = {plist(rows)}\n" if rows else "# Replace row_levels with the ordinal order shown in uSTAT.\nrow_levels = sorted(d.iloc[:, 0].unique())\n") + (f"col_levels = {plist(cols)}\n" if cols else "# Replace col_levels with the ordinal order shown in uSTAT.\ncol_levels = sorted(d.iloc[:, 1].unique())\n")
    r_order = (f"rows <- {rvec(rows)}\n" if rows else "# Replace rows with the ordinal order shown in uSTAT.\nrows <- sort(unique(d[[1]]))\n") + (f"cols <- {rvec(cols)}\n" if cols else "# Replace cols with the ordinal order shown in uSTAT.\ncols <- sort(unique(d[[2]]))\n")
    return {"title": "Ordinal association", "python": "import pandas as pd\nfrom scipy.stats import somersd\n" + f"d = df[{plist([x,y])}].dropna()\n" + py_order + f"table = pd.crosstab(d[{lit(x)}], d[{lit(y)}]).reindex(index=row_levels, columns=col_levels, fill_value=0)\n" + "somersd(table.to_numpy())  # D(column | row)\nsomersd(table.to_numpy().T)  # D(row | column)\n# Gamma: use the R reference below; SciPy has no gamma API.",
            "r": "library(DescTools)\n" + f"d <- na.omit(df[, {rvec([x,y])}])\n" + r_order + f"tab <- table(factor({rcol(x,'d')}, rows), factor({rcol(y,'d')}, cols))\n" + 'GoodmanKruskalGamma(tab)\nSomersDelta(tab, direction = "column")\nSomersDelta(tab, direction = "row")'}


def paired_categorical(b: dict) -> dict:
    x = field(b, "col1", default="before")
    y = field(b, "col2", default="after")
    bowker = b.get("method", "bowker") == "bowker"
    return {"title": "Paired categorical test", "python": "import pandas as pd\nfrom statsmodels.stats.contingency_tables import SquareTable\n" + f"d = df[{plist([x,y])}].dropna()\n" + f"levels = sorted(set(d[{lit(x)}]) | set(d[{lit(y)}]))\n" + f"table = pd.crosstab(d[{lit(x)}], d[{lit(y)}]).reindex(index=levels, columns=levels, fill_value=0)\n" + f"SquareTable(table, shift_zeros=False).{'symmetry' if bowker else 'homogeneity'}()",
            "r": ("" if bowker else "library(DescTools)\n") + f"d <- na.omit(df[, {rvec([x,y])}])\n" + f"levels <- sort(unique(c({rcol(x,'d')}, {rcol(y,'d')})))\n" + f"tab <- table(factor({rcol(x,'d')}, levels), factor({rcol(y,'d')}, levels))\n" + ('mcnemar.test(tab, correct = FALSE) # Bowker extension for square tables' if bowker else 'StuartMaxwellTest(tab)')}


ENDPOINTS = {
    "/api/stats/correlation_pair": correlation_pair,
    "/api/stats/partial_correlation": partial_correlation,
    "/api/stats/correlation_matrix": correlation_matrix,
    "/api/stats/{sid}/correlation": correlation_all_numeric,
    "/api/stats/icc": icc,
    "/api/stats/cohens_kappa": cohens_kappa,
    "/api/stats/ordinal_association": ordinal_association,
    "/api/categorical/paired_categorical": paired_categorical,
    "/api/reliability/cronbach": cronbach,
}
