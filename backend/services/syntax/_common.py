"""Helpers every endpoint template shares.

Two jobs: read a recorded request body the way the API reads it (field
aliases, optional values), and spell column names and literals so the
generated code parses in both languages whatever the column is called.
Request values are data: numbers pass through float(), strings become quoted
literals, so nothing from a body lands in the code unescaped.
"""
from __future__ import annotations

import json
import keyword
import math
import re
from typing import Iterable, List, Optional, Sequence

_R_SYNTACTIC = re.compile(r"^(?:[A-Za-z]|\.[A-Za-z_.])[A-Za-z0-9_.]*$")
_R_RESERVED = frozenset({
    "if", "else", "repeat", "while", "function", "for", "in", "next", "break",
    "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "NA_integer_", "NA_real_",
    "NA_character_", "NA_complex_",
})


def field(body: dict, *names: str, default=None):
    """The first of `names` that carries a value. The API accepts aliases
    (group_column / group_col), and a recorded body holds whichever spelling
    the client sent."""
    for name in names:
        value = body.get(name)
        if value is not None and value != "" and value != []:
            return value
    return default


def columns(body: dict, *names: str) -> List[str]:
    value = field(body, *names, default=[])
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and v != ""]
    return [str(value)]


def number(value, default: float = 0) -> str:
    """A numeric literal valid in both languages."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        f = float(default)
    if not math.isfinite(f):
        f = float(default)
    return str(int(f)) if f.is_integer() else repr(f)


def lit(value) -> str:
    """A double-quoted string literal valid in both Python and R."""
    return json.dumps(str(value), ensure_ascii=False)


def plist(values: Iterable) -> str:
    return "[" + ", ".join(lit(v) for v in values) + "]"


def rvec(values: Iterable) -> str:
    return "c(" + ", ".join(lit(v) for v in values) + ")"


def nvec(values: Iterable, lang: str) -> str:
    items = ", ".join(number(v) for v in values)
    return f"[{items}]" if lang == "py" else f"c({items})"


def rname(col) -> str:
    """A column as an R name: bare when syntactic, else backticked."""
    c = str(col)
    if _R_SYNTACTIC.match(c) and c not in _R_RESERVED:
        return c
    return "`" + c.replace("\\", "\\\\").replace("`", "\\`") + "`"


def rcol(col, frame: str = "df") -> str:
    return f"{frame}${rname(col)}"


def pterm(col) -> str:
    """A column as a patsy formula term: bare when an identifier, else Q().
    Q takes a single-quoted literal so the formula, itself a double-quoted
    string, stays readable."""
    c = str(col)
    return c if c.isidentifier() and not keyword.iskeyword(c) else f"Q({c!r})"


def pcol(col, frame: str = "df") -> str:
    return f"{frame}[{lit(col)}]"


def rhs(terms: Sequence[str]) -> str:
    return " + ".join(terms) if terms else "1"


def interaction_terms(body: dict, spell) -> List[str]:
    """`a:b` terms for the [a, b] pairs in body["interactions"]."""
    out = []
    for pair in body.get("interactions") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            out.append(f"{spell(pair[0])}:{spell(pair[1])}")
    return out


def imputation_note(value: Optional[str], pooled: bool = False) -> str:
    """Comment lines for a missing-data strategy other than complete cases.

    `pooled` is for the endpoints that fit every one of five imputations and
    pool them; elsewhere uSTAT's "mice" is a single completed dataset.
    """
    strategy = (value or "listwise").lower()
    if strategy == "listwise":
        return ""
    if strategy == "mice" and pooled:
        return (
            "# uSTAT pooled 5 imputations with Rubin's rules; the code below is the\n"
            "# complete-case fit. mice::mice(df, m = 5) with pool() is the nearest\n"
            "# R route, with its own imputation model.\n"
        )
    if strategy == "mice":
        return (
            "# uSTAT filled missing values with one iterative (MICE-style) completion,\n"
            "# not pooled multiple imputation; the code below fits complete cases.\n"
        )
    # Only known names reach the comment: a request string with a line break
    # in it would otherwise end the comment and become code.
    fill = strategy if strategy in ("mean", "median") else "strategy it was given"
    return (
        f"# uSTAT filled missing values ({fill}) first; the code below fits\n"
        "# complete cases.\n"
    )


def two_groups_py(y: str, g: str) -> str:
    """Complete rows, then the two groups in uSTAT's order (by value)."""
    return (
        f"d = df[{plist([y, g])}].dropna()\n"
        f"lv = sorted(d[{lit(g)}].unique())  # uSTAT orders groups by value\n"
        f"a = d.loc[d[{lit(g)}] == lv[0], {lit(y)}]\n"
        f"b = d.loc[d[{lit(g)}] == lv[1], {lit(y)}]\n"
    )


def groups_py(y: str, g: str) -> str:
    return f"groups = [x.dropna() for _, x in df.groupby({lit(g)})[{lit(y)}]]\n"


def complete_r(cols: Sequence[str], name: str = "d") -> str:
    return f"{name} <- df[complete.cases(df[, {rvec(cols)}]), ]\n"
