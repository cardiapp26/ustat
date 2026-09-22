"""Propensity scores: matching (MatchIt) and inverse probability weighting
(WeightIt + survey).

Neither R package reproduces uSTAT's numbers exactly, and the comments say
why rather than implying otherwise. The biggest single reason is the score
model: uSTAT fits an L2-penalised logistic regression (scikit-learn, C = 1)
on standardised covariates, where MatchIt and WeightIt fit an unpenalised
glm, so the scores (and with them the matched set or the weights) differ
slightly. Everything downstream follows the same specification.
"""
from __future__ import annotations

from ._common import columns, complete_r, field, imputation_note, lit, number, plist, rhs, rname

_PS_PY = (
    "import pandas as pd\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.preprocessing import StandardScaler\n\n"
    "{note}d = df[{cols}].dropna()\n"
    "X = StandardScaler().fit_transform(pd.get_dummies(d[{covs}], drop_first=True).astype(float))\n"
    "# uSTAT's score model: L2-penalised logistic regression (C = 1) on\n"
    "# standardised covariates. MatchIt / WeightIt fit an unpenalised glm.\n"
    "ps = LogisticRegression(C=1.0, max_iter=1000).fit(X, d[{treat}]).predict_proba(X)[:, 1]\n"
)
_SCORE_NOTE = {
    "probit": "# score_method = probit: uSTAT's score is a statsmodels probit fit.\n",
    "gbm": ("# score_method = gbm: uSTAT uses scikit-learn gradient boosting (300 trees,\n"
            "# depth 3, rate 0.05, subsample 0.8), not the package default.\n"),
}


def _setup(b: dict, extra: list) -> tuple:
    treat = field(b, "treatment_col", default="treated")
    covs = columns(b, "covariates")
    cols = list(dict.fromkeys([treat] + covs + [c for c in extra if c]))
    method = (b.get("score_method") or "logistic").lower()
    return treat, covs, cols, method


def _outcome_cols(b: dict) -> tuple:
    kind = (b.get("outcome_type") or "binary").lower()
    if kind == "survival":
        return kind, [field(b, "survival_duration_col"), field(b, "survival_event_col")]
    return kind, [field(b, "outcome_col")]


def _caliper_note(caliper: str, logit_scale: bool, method: str, optimal: bool) -> str:
    if optimal:
        return "# MatchIt's optimal matching takes no caliper; uSTAT's applies one.\n"
    scale = "the logit of the score" if logit_scale else "the score"
    probit = (" uSTAT keeps\n# that logit caliper for a probit score, where MatchIt's is on the score."
              if method == "probit" and logit_scale else "")
    return (f"# Caliper = {caliper} SD of {scale}, as uSTAT (whose SD divides by n,\n"
            f"# not n - 1).{probit}\n")


def psm(b: dict) -> dict:
    kind, out_cols = _outcome_cols(b)
    exact = columns(b, "exact_match")
    treat, covs, cols, method = _setup(b, out_cols + exact)
    note = imputation_note(b.get("imputation"))
    logit_scale = (b.get("caliper_scale") or "logit").lower() == "logit"
    distance = {"probit": '"glm", link = "probit"', "gbm": '"gbm"'}.get(
        method, f'"glm", link = "{"linear.logit" if logit_scale else "logit"}"')
    optimal = (b.get("matching_method") or "greedy").lower() == "optimal" and number(b.get("ratio"), 1) == "1"
    args = [
        f"{rname(treat)} ~ {rhs([rname(c) for c in covs])}", "data = d",
        f'method = "{"optimal" if optimal else "nearest"}"', f"distance = {distance}",
        f"ratio = {number(b.get('ratio'), 1)}",
    ]
    caliper = number(b.get("caliper"), 0.2)
    if not optimal:
        args += [f"caliper = {caliper}", "std.caliper = TRUE", 'm.order = "largest"']
    if exact:
        args.append(f"exact = ~ {rhs([rname(c) for c in exact])}")
    if b.get("trim_common_support"):
        args.append('discard = "both"')
    y = out_cols[0] if kind == "binary" else None
    outcome_r = ""
    if kind == "survival" and all(out_cols):
        outcome_r = (f"coxph(Surv({rname(out_cols[0])}, {rname(out_cols[1])}) ~ {rname(treat)} + strata(subclass),\n"
                     "      data = md)  # Cox stratified by matched set, as uSTAT\n")
    elif y:
        outcome_r = (f"clogit({rname(y)} ~ {rname(treat)} + strata(subclass), data = md)"
                     "  # conditional logistic, as uSTAT\n")
    return {
        "title": f"Propensity score matching: {treat} on {', '.join(covs) or 'no covariates'}",
        "python": (
            _PS_PY.format(note=note, cols=plist(cols), covs=plist(covs), treat=lit(treat))
            + _SCORE_NOTE.get(method, "")
            + "# The matching step (greedy nearest neighbour, largest score first, without\n"
            "# replacement, caliper in SDs of logit(ps)) has no scipy or statsmodels\n"
            "# routine; see the R version. On the matched set, uSTAT's outcome model is\n"
            "# statsmodels' ConditionalLogit (binary) or lifelines' CoxPHFitter with the\n"
            "# matched-set id as strata (survival)."
        ),
        "r": (
            "library(MatchIt)\nlibrary(survival)\n\n" + note + complete_r(cols)
            + _SCORE_NOTE.get(method, "")
            + _caliper_note(caliper, logit_scale, method, optimal)
            + "# uSTAT balances with the pooled SD before matching; summary() standardises\n"
            "# by the treated group's SD.\n"
            + "m <- matchit(" + ",\n             ".join(args) + ")\n"
            "summary(m)\n"
            "md <- match.data(m)  # subclass = the matched set\n" + outcome_r
        ).rstrip("\n"),
    }


_ESTIMAND = {"ate": "ATE", "att": "ATT", "overlap": "ATO"}


def _weight_trim_r(b: dict) -> str:
    mode = (b.get("weight_truncation") or "none").lower()
    if mode == "percentile":
        lo, hi = number(b.get("weight_truncation_lo"), 0.01), number(b.get("weight_truncation_hi"), 0.99)
        return (f"q <- quantile(d$w, c({lo}, {hi}))  # type 7, as numpy's percentile\n"
                "d$w <- pmin(pmax(d$w, q[1]), q[2])\n")
    if mode == "hard":
        return f"d$w <- pmin(d$w, {number(b.get('weight_truncation_max'), 10)})\n"
    return ""


def iptw(b: dict) -> dict:
    kind, out_cols = _outcome_cols(b)
    treat, covs, cols, method = _setup(b, out_cols)
    note = imputation_note(b.get("imputation"))
    estimand = _ESTIMAND.get((b.get("estimand") or "ate").lower(), "ATE")
    asked = bool(b.get("stabilize", True))
    # uSTAT stabilises ATE weights by P(treatment), as WeightIt does; its ATT
    # "stabilisation" is a constant factor and overlap weights take none.
    stabilize = asked and estimand == "ATE"
    wmethod = {"gbm": '"gbm"'}.get(method, '"glm", link = "probit"' if method == "probit" else '"glm"')
    outcome_r = ""
    if kind == "survival" and all(out_cols):
        outcome_r = (f"coxph(Surv({rname(out_cols[0])}, {rname(out_cols[1])}) ~ {rname(treat)}, data = d,\n"
                     "      weights = w, robust = TRUE)  # weighted Cox, robust SE, as uSTAT\n")
    elif out_cols[0]:
        outcome_r = (
            "library(survey)\n"
            "des <- svydesign(ids = ~1, weights = ~w, data = d)\n"
            f"fit <- svyglm({rname(out_cols[0])} ~ {rname(treat)}, design = des, family = quasibinomial())\n"
            "# Given the same weights this is uSTAT's estimate and SE; uSTAT's t uses\n"
            "# n - 1 df where summary.svyglm uses n - 2 (small samples only).\n"
            "summary(fit)\n"
            "exp(cbind(OR = coef(fit), confint(fit)))\n")
    return {
        "title": f"IPTW ({estimand}): {treat} on {', '.join(covs) or 'no covariates'}",
        "python": (
            _PS_PY.format(note=note, cols=plist(cols), covs=plist(covs), treat=lit(treat))
            + _SCORE_NOTE.get(method, "")
            + "# The weights and the weighted outcome model follow in the R version\n"
            "# (WeightIt + survey); scipy and statsmodels have no IPTW routine."
        ),
        "r": (
            "library(WeightIt)\nlibrary(survival)\n\n" + note + complete_r(cols)
            + _SCORE_NOTE.get(method, "")
            + f"W <- weightit({rname(treat)} ~ {rhs([rname(c) for c in covs])}, data = d,\n"
            f"              method = {wmethod}, estimand = \"{estimand}\", stabilize = {str(stabilize).upper()})\n"
            + ("# uSTAT also multiplies ATT weights by P(treated) when stabilising; a\n"
               "# constant factor, so estimates are unchanged.\n" if estimand == "ATT" and asked else "")
            + ("# uSTAT drops units outside the common support of the scores first.\n"
               if b.get("trim_common_support") else "")
            + "summary(W)\n"
            "d$w <- W$weights\n" + _weight_trim_r(b) + outcome_r
        ).rstrip("\n"),
    }


ENDPOINTS = {
    "/api/models/psm": psm,
    "/api/models/iptw": iptw,
}
