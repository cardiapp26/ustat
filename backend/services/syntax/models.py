"""Regression models under /api/models: linear, logistic, count and Gamma GLMs,
ordinal and Firth logistic.

uSTAT dummy-codes a text predictor against its first level in sorted order,
which is what both formula interfaces do by default, so the formulas below
need no contrasts. Two things differ from each package's defaults and are
set explicitly: intervals are Wald (confint.default, not R's profile
confint), and "robust" SEs are what statsmodels computes, which for every
model except OLS is the HC0 sandwich even when HC3 is asked for.
"""
from __future__ import annotations

from ._common import (
    columns, field, imputation_note, interaction_terms, lit, number, plist,
    pterm, rhs, rname, rvec,
)

_ROBUST_HC0 = (
    "# Robust SEs: uSTAT asks statsmodels for HC3, which outside OLS it computes\n"
    "# as the HC0 sandwich (no leverage adjustment); HC0 here says so plainly.\n"
)


def _terms(b: dict, preds: list, spell, scale: bool = False) -> list:
    """Predictors (rescaled per uSTAT's scale_factors) plus a:b interactions."""
    factors = (b.get("scale_factors") or {}) if scale else {}
    out = []
    for p in preds:
        f = number(factors.get(p), 1)
        # "age (per 10 units)" in uSTAT's output is age / 10 in the formula.
        out.append(spell(p) if f in ("0", "1") else f"I({spell(p)} / {f})")
    return out + interaction_terms(b, spell)


def _formulas(b: dict, scale: bool = False) -> tuple:
    y = field(b, "outcome", default="outcome")
    preds = columns(b, "predictors")
    py = f"{pterm(y)} ~ {rhs(_terms(b, preds, pterm, scale))}"
    r = f"{rname(y)} ~ {rhs(_terms(b, preds, rname, scale))}"
    return py, r


def linear(b: dict) -> dict:
    py_f, r_f = _formulas(b)
    robust = bool(b.get("robust_se"))
    note = imputation_note(b.get("imputation"), pooled=True)
    fit_args = 'cov_type="HC3", use_t=True' if robust else ""
    return {
        "title": f"Linear regression: {r_f}" + (" (robust SE)" if robust else ""),
        "python": (
            "import statsmodels.formula.api as smf\n\n" + note
            + (("# HC3 standard errors tested on t, as uSTAT (and lmtest::coeftest).\n") if robust else "")
            + f'fit = smf.ols({lit(py_f)}, data=df).fit({fit_args})\n'
            "fit.summary()"
        ),
        "r": (
            note + f"fit <- lm({r_f}, data = df)\n"
            + ("library(lmtest)\nlibrary(sandwich)\n"
               'coeftest(fit, vcov = vcovHC(fit, type = "HC3"))\n'
               'coefci(fit, vcov = vcovHC(fit, type = "HC3"))' if robust
               else "summary(fit)\nconfint(fit)")
        ),
    }


def _glm(b: dict, title: str, py_family: str, r_family: str, ratio: str) -> dict:
    """Logistic and Poisson: z tests, Wald intervals, exp() for the ratio."""
    scale = py_family == "logit"  # only the logistic endpoints take scale_factors
    py_f, r_f = _formulas(b, scale=scale)
    robust = bool(b.get("robust_se"))
    note = imputation_note(b.get("imputation"), pooled=scale)
    cov = '(cov_type="HC0")' if robust else "()"
    if scale:
        py_fit = f'fit = smf.logit({lit(py_f)}, data=df).fit{cov}\n'
    else:
        py_fit = f'fit = smf.glm({lit(py_f)}, data=df, family=sm.families.Poisson()).fit{cov}\n'
    r_ci = ('coefci(fit, vcov = vcovHC(fit, type = "HC0"))' if robust
            else "confint.default(fit)")
    r_ci_note = "" if robust else "# Wald intervals, as uSTAT; confint() would profile.\n"
    outcome_note = ("# A text outcome is coded 0/1 in sorted order: the later value is the event.\n"
                    if scale else "")
    return {
        "title": f"{title}: {r_f}" + (" (robust SE)" if robust else ""),
        "python": (
            "import numpy as np\n"
            + ("" if scale else "import statsmodels.api as sm\n")
            + "import statsmodels.formula.api as smf\n\n" + note + outcome_note
            + (_ROBUST_HC0 if robust else "") + py_fit
            + "fit.summary()\n"
            f"np.exp(fit.params), np.exp(fit.conf_int())  # {ratio}s, Wald 95% CI"
        ),
        "r": (
            note + outcome_note
            + f"fit <- glm({r_f}, data = df, family = {r_family})\n"
            + (_ROBUST_HC0 + "library(lmtest)\nlibrary(sandwich)\n"
               'coeftest(fit, vcov = vcovHC(fit, type = "HC0"))\n' if robust else "summary(fit)\n")
            + r_ci_note + f"exp(cbind({ratio} = coef(fit), {r_ci}))"
        ),
    }


def logistic(b: dict) -> dict:
    return _glm(b, "Logistic regression", "logit", "binomial", "OR")


def poisson(b: dict) -> dict:
    return _glm(b, "Poisson regression", "poisson", "poisson", "IRR")


def negbinom(b: dict) -> dict:
    py_f, r_f = _formulas(b)
    robust = bool(b.get("robust_se"))
    note = imputation_note(b.get("imputation"))
    return {
        "title": f"Negative binomial regression: {r_f}" + (" (robust SE)" if robust else ""),
        "python": (
            "import numpy as np\n"
            "import statsmodels.formula.api as smf\n\n" + note
            + "# NB2 with alpha estimated jointly by maximum likelihood; theta = 1 / alpha.\n"
            + (_ROBUST_HC0 if robust else "")
            + f'fit = smf.negativebinomial({lit(py_f)}, data=df).fit(maxiter=200, disp=0'
            + (', cov_type="HC0")\n' if robust else ")\n")
            + "fit.summary()\n"
            "np.exp(fit.params.drop(\"alpha\")), np.exp(fit.conf_int().drop(\"alpha\"))  # IRRs"
        ),
        "r": (
            "library(MASS)\n\n" + note
            + f"fit <- glm.nb({r_f}, data = df)\n"
            "summary(fit)  # theta = 1 / uSTAT's alpha\n"
            "# Coefficients and theta agree. glm.nb's SEs hold theta fixed and use the\n"
            "# expected information; statsmodels (uSTAT) uses the observed information\n"
            "# of the joint fit, so SEs and p can differ by a few percent.\n"
            + ("# uSTAT's robust sandwich has no exact counterpart here either.\n" if robust else "")
            + "exp(cbind(IRR = coef(fit), confint.default(fit)))"
        ),
    }


_GAMMA_LINKS = {"log": ("Log", "log"), "identity": ("Identity", "identity"),
                "inverse": ("InversePower", "inverse")}


def gamma(b: dict) -> dict:
    py_f, r_f = _formulas(b)
    link = b.get("link") if b.get("link") in _GAMMA_LINKS else "log"
    py_link, r_link = _GAMMA_LINKS[link]
    robust = bool(b.get("robust_se"))
    note = imputation_note(b.get("imputation"))
    exp_line_py = "\nnp.exp(fit.params), np.exp(fit.conf_int())" if link == "log" else ""
    return {
        "title": f"Gamma GLM (link = {link}): {r_f}" + (" (robust SE)" if robust else ""),
        "python": (
            ("import numpy as np\n" if link == "log" else "")
            + "import statsmodels.api as sm\n"
            "import statsmodels.formula.api as smf\n\n" + note
            + "# The dispersion is estimated, so tests and intervals use t on the\n"
            "# residual df (use_t=True), as uSTAT and R's summary.glm do.\n"
            f'fam = sm.families.Gamma(link=sm.families.links.{py_link}())\n'
            f'fit = smf.glm({lit(py_f)}, data=df, family=fam).fit(use_t=True'
            + (', cov_type="HC0")' if robust else ")")
            + "\nfit.summary()" + exp_line_py
        ),
        "r": (
            note + f'fit <- glm({r_f}, data = df, family = Gamma(link = "{r_link}"))\n'
            "summary(fit)  # t tests on the residual df\n"
            + ("# uSTAT's robust SEs (statsmodels' sandwich for a Gamma GLM) do not match\n"
               "# sandwich::vcovHC for this family; the intervals below are model-based.\n"
               if robust else "")
            + "ci <- lmtest::coefci(fit, df = df.residual(fit))  # t-based Wald, as uSTAT\n"
            + ("exp(cbind(estimate = coef(fit), ci))" if link == "log" else "cbind(estimate = coef(fit), ci)")
        ),
    }


def ordinal(b: dict) -> dict:
    y = field(b, "outcome", default="outcome")
    preds = columns(b, "predictors")
    order = b.get("level_order")
    r_rhs = rhs([rname(p) for p in preds])
    levels_r = f", levels = {rvec(order)}" if order else ""
    return {
        "title": f"Ordinal logistic regression (proportional odds): {y} ~ {r_rhs}",
        "python": (
            "import pandas as pd\n"
            "from statsmodels.miscmodels.ordinal_model import OrderedModel\n\n"
            f"d = df[{plist([y] + preds)}].dropna()\n"
            + (f"cats = {[str(v) for v in order]!r}\n" if order else
               "# Set the categories low to high; uSTAT takes them from the Data Dictionary.\n"
               f"cats = sorted(d[{lit(y)}].unique())\n")
            + f"y = pd.Series(pd.Categorical(d[{lit(y)}].astype(str), categories=cats, ordered=True),\n"
            "              index=d.index)\n"
            f"X = pd.get_dummies(d[{plist(preds)}], drop_first=True).astype(float)\n"
            'fit = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False, maxiter=200)\n'
            "fit.summary()"
        ),
        "r": (
            "library(MASS)\n\n"
            f"df${rname(y)} <- factor(df${rname(y)}{levels_r}, ordered = TRUE)"
            + ("" if order else "  # set levels = low to high")
            + f"\nfit <- polr({rname(y)} ~ {r_rhs}, data = df, Hess = TRUE)\nsummary(fit)\n"
            "exp(cbind(OR = coef(fit), confint.default(fit)))"
        ),
    }


def firth(b: dict) -> dict:
    _, r_f = _formulas(b, scale=True)
    return {
        "title": f"Firth penalised logistic regression: {r_f}",
        "python": (
            "# No Firth logistic regression in statsmodels; see the R version.\n"
            "# uSTAT's own fit matches logistf's coefficients."
        ),
        "r": (
            "library(logistf)\n\n"
            "# pl = FALSE: Wald intervals and p-values, which is what uSTAT reports.\n"
            "# logistf's default (pl = TRUE) is penalised profile likelihood.\n"
            f"fit <- logistf({r_f}, data = df, pl = FALSE)\n"
            "summary(fit)\n"
            "exp(cbind(OR = coef(fit), confint(fit)))"
        ),
    }


ENDPOINTS = {
    "/api/models/linear": linear,
    "/api/models/logistic": logistic,
    "/api/models/poisson": poisson,
    "/api/models/negbinom": negbinom,
    "/api/models/gamma": gamma,
    "/api/models/ordinal": ordinal,
    "/api/models/firth_logistic": firth,
}
