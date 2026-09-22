"""ROC analysis: one score's AUC with a DeLong interval, and DeLong's paired test.

uSTAT computes the AUC with scikit-learn and the DeLong variance itself;
pROC reproduces both. The one visible difference is the reported cutoff:
scikit-learn's thresholds are observed scores, pROC's sit midway between
neighbouring scores, so the "best" cutoff prints differently while its
sensitivity and specificity agree.
"""
from __future__ import annotations

from ._common import complete_r, field, imputation_note, lit, number, plist, rname

# uSTAT's direction -> pROC's: "<" means higher scores in the events.
_PROC_DIRECTION = {"auto": "auto", "higher": "<", "lower": ">"}


def _direction(value) -> str:
    d = (value or "auto").lower()
    return d if d in _PROC_DIRECTION else "auto"


def _flip_note(direction: str) -> str:
    if direction == "lower":
        return "s = -s  # direction \"lower\": lower scores predict the event\n"
    if direction == "auto":
        return ("# direction \"auto\": uSTAT negates the score when its AUC is below 0.5.\n"
                "s = -s if roc_auc_score(y, s) < 0.5 else s\n")
    return ""


def _roc_py(score: str, y: str, direction: str, frame: str) -> str:
    return (
        f"y, s = {frame}[{lit(y)}].astype(int), {frame}[{lit(score)}].astype(float)\n"
        + _flip_note(direction)
        + "print(roc_auc_score(y, s))\n"
        "fpr, tpr, thr = roc_curve(y, s)\n"
        "print(thr[(tpr - fpr).argmax()])  # Youden-optimal cutoff, an observed score\n"
    )


def _roc_r(score: str, y: str, direction: str, cutoff, frame: str) -> str:
    return (
        f"r <- roc({frame}${rname(y)}, {frame}${rname(score)}, levels = c(0, 1),\n"
        f"         direction = \"{_PROC_DIRECTION[direction]}\")\n"
        "print(ci.auc(r, method = \"delong\"))  # DeLong 95% CI, as uSTAT\n"
        'print(coords(r, "best", best.method = "youden"))\n'
        + (f'print(coords(r, x = {number(cutoff)}, input = "threshold"))  # the manual cutoff\n'
           if cutoff is not None else "")
    )


def _indent(code: str, pad: str) -> str:
    return "".join(pad + line + "\n" for line in code.rstrip("\n").split("\n"))


def roc(b: dict) -> dict:
    score = field(b, "score_column", "score_col", default="score")
    y = field(b, "outcome_column", "outcome_col", default="outcome")
    direction = _direction(b.get("direction"))
    strat = field(b, "stratify_by")
    cutoff = b.get("manual_cutoff")
    note = imputation_note(b.get("imputation"))
    cols = [score, y] + ([strat] if strat else [])
    py = ("from sklearn.metrics import roc_auc_score, roc_curve\n\n" + note
          + f"d = df[{plist(cols)}].dropna()\n")
    r = "library(pROC)\n\n" + note + complete_r(cols)
    if strat:
        per = "# One ROC per stratum; uSTAT skips strata with fewer than 20 rows.\n"
        py += (per + f"for level, part in d.groupby({lit(strat)}):\n"
               + _indent(_roc_py(score, y, direction, "part"), "    "))
        r += (per + f"for (part in split(d, d${rname(strat)})) {{\n"
              + _indent(_roc_r(score, y, direction, cutoff, "part"), "  ") + "}\n")
    else:
        py += _roc_py(score, y, direction, "d")
        r += _roc_r(score, y, direction, cutoff, "d")
    py += ("# The DeLong interval and uSTAT's z test of AUC = 0.5 have no scipy or\n"
           "# statsmodels routine; see the R version.")
    return {
        "title": f"ROC analysis: {score} for {y}" + (f", by {strat}" if strat else ""),
        "python": py,
        "r": r.rstrip("\n"),
    }


def roc_compare(b: dict) -> dict:
    s1 = field(b, "score_column_1", "score_col_1", default="score1")
    s2 = field(b, "score_column_2", "score_col_2", default="score2")
    y = field(b, "outcome_column", "outcome_col", default="outcome")
    d1, d2 = _direction(b.get("direction_1")), _direction(b.get("direction_2"))
    return {
        "title": f"DeLong test: AUC of {s1} vs {s2} for {y}",
        "python": (
            "from sklearn.metrics import roc_auc_score\n\n"
            f"d = df[{plist([s1, s2, y])}].dropna()  # rows complete on both scores\n"
            "# uSTAT negates a score whose AUC is below 0.5 (direction \"auto\") first.\n"
            f"roc_auc_score(d[{lit(y)}], d[{lit(s1)}]), roc_auc_score(d[{lit(y)}], d[{lit(s2)}])\n"
            "# DeLong's paired test has no scipy or statsmodels routine; see the R version."
        ),
        "r": (
            "library(pROC)\n\n" + complete_r([s1, s2, y])
            + "# direction \"auto\" flips a score whose AUC is below 0.5, as uSTAT does.\n"
            f"r1 <- roc(d${rname(y)}, d${rname(s1)}, levels = c(0, 1), direction = \"{_PROC_DIRECTION[d1]}\")\n"
            f"r2 <- roc(d${rname(y)}, d${rname(s2)}, levels = c(0, 1), direction = \"{_PROC_DIRECTION[d2]}\")\n"
            'roc.test(r1, r2, method = "delong", paired = TRUE)'
        ),
    }


ENDPOINTS = {
    "/api/stats/roc": roc,
    "/api/stats/roc_compare": roc_compare,
}
