"""Time-to-event: Kaplan-Meier with log-rank, Cox, RMST, competing risks.

lifelines (what uSTAT runs) and R's survival package agree on the defaults
that matter here once two are pinned: KM intervals are lifelines' exponential
Greenwood, which is survfit's conf.type = "log-log" (not its "log" default),
and both Cox fits use Efron ties.
"""
from __future__ import annotations

from ._common import (
    columns, complete_r, field, imputation_note, interaction_terms, lit, number,
    nvec, plist, pterm, rhs, rname,
)


def _surv(b: dict) -> tuple:
    return (field(b, "duration_col", default="time"), field(b, "event_col", default="event"))


def _km_py(t: str, e: str, g, times: list, frame: str) -> str:
    """KM fits (per group when there is one) and the log-rank on `frame`."""
    at = (f"print(km.survival_function_at_times({nvec(times, 'py')}))\n"
          "print(km.confidence_interval_survival_function_)  # exponential Greenwood\n"
          if times else "")
    fit = "print(km.median_survival_time_)\n" + at
    if not g:
        return f"km = KaplanMeierFitter().fit({frame}[{lit(t)}], {frame}[{lit(e)}])\n" + fit
    return (f"for level, sub in {frame}.groupby({lit(g)}):\n"
            f"    km = KaplanMeierFitter().fit(sub[{lit(t)}], sub[{lit(e)}], label=str(level))\n"
            + _indent(fit)
            + "# Log-rank (for two groups this equals lifelines' logrank_test).\n"
            f"print(multivariate_logrank_test({frame}[{lit(t)}], {frame}[{lit(g)}], {frame}[{lit(e)}]).summary)\n")


def _km_r(t: str, e: str, g, times: list, frame: str) -> str:
    surv = f"Surv({rname(t)}, {rname(e)})"
    return (f"fit <- survfit({surv} ~ {rname(g) if g else '1'}, data = {frame}, conf.type = \"log-log\")\n"
            "print(fit)  # n, events, median\n"
            + (f"print(summary(fit, times = {nvec(times, 'r')}))  # survival at the landmarks\n" if times else "")
            + (f"print(survdiff({surv} ~ {rname(g)}, data = {frame}))  # log-rank\n" if g else ""))


def _indent(code: str, pad: str = "    ") -> str:
    return "".join(pad + line + "\n" if line else "\n" for line in code.rstrip("\n").split("\n"))


def km(b: dict) -> dict:
    t, e = _surv(b)
    g = field(b, "group_col")
    strat = field(b, "stratify_col")
    times = b.get("survival_times") or []
    note = imputation_note(b.get("imputation"))
    cols = [t, e] + [c for c in (g, strat) if c]
    py = ("from lifelines import KaplanMeierFitter\n"
          + ("from lifelines.statistics import multivariate_logrank_test\n" if g else "")
          + "\n" + note + f"d = df[{plist(cols)}].dropna()\n")
    r = ("library(survival)\n\n" + note + complete_r(cols)
         + "# conf.type = \"log-log\" is lifelines' exponential Greenwood interval.\n")
    if strat:
        # uSTAT fits each stratum separately: curves and log-rank within it.
        py += f"for s, part in d.groupby({lit(strat)}):\n" + _indent(_km_py(t, e, g, times, "part"))
        r += (f"for (part in split(d, d${rname(strat)})) {{\n"
              + _indent(_km_r(t, e, g, times, "part"), "  ") + "}\n")
    else:
        py += _km_py(t, e, g, times, "d")
        r += _km_r(t, e, g, times, "d")
    if g and b.get("pairwise") and not strat:
        corr = (b.get("pairwise_correction") or "none").lower()
        corr = corr if corr in ("none", "bonferroni", "holm", "bh") else "none"
        py += ("# Pairwise log-rank; uSTAT then adjusts the p with multipletests.\n"
               "from lifelines.statistics import pairwise_logrank_test\n"
               f"pairwise_logrank_test(d[{lit(t)}], d[{lit(g)}], d[{lit(e)}]).summary\n")
        r += (f"survminer::pairwise_survdiff(Surv({rname(t)}, {rname(e)}) ~ {rname(g)}, data = d,\n"
              f"                            p.adjust.method = \"{'BH' if corr == 'bh' else corr}\")\n")
    title = (f"Kaplan-Meier: {t}, {e}" + (f" by {g}" if g else "")
             + (f", within {strat}" if strat else ""))
    return {"title": title, "python": py.rstrip("\n"), "r": r.rstrip("\n")}


def cox(b: dict) -> dict:
    t, e = _surv(b)
    preds = columns(b, "predictors")
    py_terms = rhs([pterm(p) for p in preds] + interaction_terms(b, pterm))
    r_terms = rhs([rname(p) for p in preds] + interaction_terms(b, rname))
    note = imputation_note(b.get("imputation"), pooled=True)
    zph = ("# The PH test: lifelines (uSTAT) uses the approximation survival < 3.0 used;\n"
           "# current cox.zph computes the score test exactly, so p can differ, and\n"
           "# uSTAT's global test is the sum of the per-term chi-squares.\n")
    return {
        "title": f"Cox proportional hazards: Surv({t}, {e}) ~ {r_terms}",
        "python": (
            "from lifelines import CoxPHFitter\n"
            "from lifelines.statistics import proportional_hazard_test\n\n" + note
            + f"d = df[{plist([t, e] + preds)}].dropna()\n"
            "# Text predictors are coded against their first level, as uSTAT; Efron ties.\n"
            f"cph = CoxPHFitter().fit(d, duration_col={lit(t)}, event_col={lit(e)}, formula={lit(py_terms)})\n"
            "cph.print_summary()\n" + zph
            + 'proportional_hazard_test(cph, d, time_transform="rank").summary'
        ),
        "r": (
            "library(survival)\n\n" + note
            + f"fit <- coxph(Surv({rname(t)}, {rname(e)}) ~ {r_terms}, data = df)  # Efron ties\n"
            "summary(fit)\n" + zph
            + 'cox.zph(fit, transform = "rank")'
        ),
    }


def _rmst_r(t: str, e: str, g, tau: str) -> str:
    if not g:
        return (f"# One group: survival's restricted mean, with the same Greenwood-type SE.\n"
                f"print(survfit(Surv({rname(t)}, {rname(e)}) ~ 1, data = d), rmean = {tau})")
    return (
        f"lv <- sort(unique(d${rname(g)}))\n"
        "# arm = 1 marks the earlier group, so rmst2's (arm=1)-(arm=0) is uSTAT's\n"
        "# first-minus-second difference. rmst2 wants tau within every group's\n"
        "# follow-up; uSTAT checks it against the longest overall.\n"
        "for (p in combn(seq_along(lv), 2, simplify = FALSE)) {\n"
        f"  pair <- d[d${rname(g)} %in% lv[p], ]\n"
        f"  print(rmst2(pair${rname(t)}, pair${rname(e)}, arm = as.integer(pair${rname(g)} == lv[p[1]]), tau = {tau}))\n"
        "}"
    )


def rmst(b: dict) -> dict:
    t, e = _surv(b)
    g = field(b, "group_col")
    tau = number(b.get("tau"), 0)
    cols = [t, e] + ([g] if g else [])
    note = imputation_note(b.get("imputation"), pooled=bool(g))
    py_fit = (f"for level, sub in d.groupby({lit(g)}):\n"
              f"    km = KaplanMeierFitter().fit(sub[{lit(t)}], sub[{lit(e)}])\n"
              f"    print(level, restricted_mean_survival_time(km, t={tau}))\n" if g else
              f"km = KaplanMeierFitter().fit(d[{lit(t)}], d[{lit(e)}])\n"
              f"restricted_mean_survival_time(km, t={tau})\n")
    return {
        "title": f"Restricted mean survival time at tau = {tau}" + (f" by {g}" if g else ""),
        "python": (
            "from lifelines import KaplanMeierFitter\n"
            "from lifelines.utils import restricted_mean_survival_time\n\n" + note
            + f"d = df[{plist(cols)}].dropna()\n" + py_fit
            + "# The point estimate matches uSTAT. Its SE and the group contrast use the\n"
            "# Greenwood-type variance, which lifelines does not report (return_variance\n"
            "# is the variance of min(T, tau), not of the estimate); see the R version."
        ),
        "r": ("library(survival)\n" + ("library(survRM2)\n" if g else "") + "\n" + note
              + complete_r(cols) + _rmst_r(t, e, g, tau)),
    }


def fine_gray(b: dict) -> dict:
    dur = field(b, "duration_col", default="time")
    ev = field(b, "event_col", default="status")
    k = number(b.get("event_of_interest"), 1)
    grp = field(b, "group_col")
    preds = columns(b, "predictors")
    py = (
        "from lifelines import AalenJohansenFitter\n"
        "from lifelines.statistics import multivariate_logrank_test\n\n"
        f"# Cumulative incidence of event {k} (Aalen-Johansen)\n"
        + (f"for g, d in df.groupby({lit(grp)}):\n"
           f"    AalenJohansenFitter().fit(d[{lit(dur)}], d[{lit(ev)}], event_of_interest={k}).plot(label=str(g))\n"
           if grp else
           f"AalenJohansenFitter().fit(df[{lit(dur)}], df[{lit(ev)}], event_of_interest={k}).plot()\n")
    )
    r = (
        "library(cmprsk)\nlibrary(survival)\n\n"
        f"cif <- cuminc(df${rname(dur)}, df${rname(ev)}" + (f", group = df${rname(grp)}" if grp else "") + ")\n"
        "plot(cif)\n"
    )
    if grp:
        py += (
            "\n# The group p uSTAT reports: cause-specific log-rank, competing events\n"
            "# censored. It compares cause-specific hazards, NOT cumulative incidence.\n"
            f"multivariate_logrank_test(df[{lit(dur)}], df[{lit(grp)}], (df[{lit(ev)}] == {k}).astype(int))\n"
            "# Gray's test (equal cumulative incidence) has no Python equivalent here;\n"
            "# see the R version (cmprsk::cuminc $Tests).\n"
        )
        r += (
            "cif$Tests  # Gray's test: equal cumulative incidence\n\n"
            "# The group p uSTAT reports: cause-specific log-rank, competing events censored\n"
            f"survdiff(Surv({rname(dur)}, {rname(ev)} == {k}) ~ {rname(grp)}, data = df)\n"
        )
    if preds:
        r += (
            "\n# Fine-Gray subdistribution hazards\n"
            f"fg <- crr(df${rname(dur)}, df${rname(ev)}, cov1 = model.matrix(~ {rhs([rname(p) for p in preds])}, df)[, -1],"
            f" failcode = {k})\nsummary(fg)\n"
        )
        py += "\n# Fine-Gray regression: uSTAT fits it by Geskus IPCW weighting; see the R version.\n"
    return {"title": f"Competing risks: event {k}" + (f" by {grp}" if grp else ""),
            "python": py.rstrip("\n"), "r": r.rstrip("\n")}


ENDPOINTS = {
    "/api/models/survival/km": km,
    "/api/models/survival/cox": cox,
    "/api/survival_advanced/rmst": rmst,
    "/api/survival_advanced/fine_gray": fine_gray,
}
