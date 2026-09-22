"""Meta-analysis (/api/meta/analyze): fixed and random-effects pooling.

The studies travel in the request, so the template carries them as data.
Both routes compute each study's effect the way uSTAT does: a 2x2 table goes
through the package's effect-size helper with the continuity correction
added to every cell of a study that has a zero cell (metafor's
to = "only0"); an effect with a CI or SE is taken on the log scale for ratio
measures.
"""
from __future__ import annotations

from ._common import number, nvec, plist, rvec

_LOG = ("OR", "RR", "HR")
_SM_STAT = {"OR": "odds-ratio", "RR": "risk-ratio", "RD": "diff"}


def _split(studies: list) -> tuple:
    """Studies by input kind, in uSTAT's precedence: 2x2, then effect + SE,
    then effect + CI. Anything else uSTAT rejects, so it is left out."""
    tab, se, ci = [], [], []
    for s in studies:
        if not isinstance(s, dict):
            continue
        if all(s.get(k) is not None for k in ("e1", "n1", "e2", "n2")):
            tab.append(s)
        elif s.get("effect") is not None and s.get("se") is not None:
            se.append(s)
        elif all(s.get(k) is not None for k in ("effect", "ci_low", "ci_high")):
            ci.append(s)
    return tab, se, ci


def _vals(rows: list, key: str, lang: str) -> str:
    return nvec([r.get(key) for r in rows], lang)


def _python(tab, se, ci, measure, log, cc, method) -> str:
    tr = "np.log" if log else ""
    parts, ys, vs, names = [], [], [], []
    if tab:
        # statsmodels reads a zero correction of 0 as "unsupported"; None is none.
        zc = f"zero_correction={cc}" if cc != "0" else "zero_correction=None"
        parts.append(
            f"y_tab, v_tab = effectsize_2proportions(\n"
            f"    np.array({_vals(tab, 'e1', 'py')}), np.array({_vals(tab, 'n1', 'py')}),\n"
            f"    np.array({_vals(tab, 'e2', 'py')}), np.array({_vals(tab, 'n2', 'py')}),\n"
            f"    statistic=\"{_SM_STAT[measure]}\", {zc})\n")
        ys.append("y_tab")
        vs.append("v_tab")
        names.extend(s.get("label") for s in tab)
    if se:
        parts.append(f"y_se = {tr}(np.array({_vals(se, 'effect', 'py')}))\n"
                     f"v_se = np.array({_vals(se, 'se', 'py')}) ** 2\n")
        ys.append("y_se")
        vs.append("v_se")
        names.extend(s.get("label") for s in se)
    if ci:
        parts.append(
            f"lo, hi = {tr}(np.array({_vals(ci, 'ci_low', 'py')})), {tr}(np.array({_vals(ci, 'ci_high', 'py')}))\n"
            f"y_ci = {tr}(np.array({_vals(ci, 'effect', 'py')}))\n"
            "v_ci = ((hi - lo) / (2 * stats.norm.ppf(0.975))) ** 2  # SE from the 95% CI\n")
        ys.append("y_ci")
        vs.append("v_ci")
        names.extend(s.get("label") for s in ci)
    return (
        "import numpy as np\n"
        + ("from scipy import stats\n" if ci else "")
        + "from statsmodels.stats.meta_analysis import combine_effects"
        + (", effectsize_2proportions" if tab else "") + "\n\n"
        + "".join(parts)
        + f"res = combine_effects(np.concatenate([{', '.join(ys)}]), np.concatenate([{', '.join(vs)}]),\n"
        f"                      method_re=\"{method.lower()}\", row_names={plist(names)})\n"
        "res.summary_frame()" + ("  # log scale: np.exp() for the ratio" if log else "")
    )


def _r(tab, se, ci, measure, log, cc, method) -> str:
    tr = "log" if log else ""
    frames, parts = [], []
    if tab:
        parts.append(
            f"tab <- data.frame(study = {rvec(s.get('label') for s in tab)},\n"
            f"                  e1 = {_vals(tab, 'e1', 'r')}, n1 = {_vals(tab, 'n1', 'r')},\n"
            f"                  e2 = {_vals(tab, 'e2', 'r')}, n2 = {_vals(tab, 'n2', 'r')})\n"
            f"tab <- escalc(measure = \"{measure}\", ai = e1, n1i = n1, ci = e2, n2i = n2, data = tab,\n"
            f"              add = {cc}, to = \"only0\")\n")
        frames.append("data.frame(study = tab$study, yi = as.numeric(tab$yi), vi = as.numeric(tab$vi))")
    if se:
        parts.append(f"se_rows <- data.frame(study = {rvec(s.get('label') for s in se)},\n"
                     f"                      yi = {tr}({_vals(se, 'effect', 'r')}), vi = {_vals(se, 'se', 'r')}^2)\n")
        frames.append("se_rows")
    if ci:
        parts.append(
            f"ci_rows <- data.frame(study = {rvec(s.get('label') for s in ci)},\n"
            f"                      yi = {tr}({_vals(ci, 'effect', 'r')}),\n"
            f"                      vi = (({tr}({_vals(ci, 'ci_high', 'r')}) - {tr}({_vals(ci, 'ci_low', 'r')}))"
            " / (2 * qnorm(0.975)))^2)\n")
        frames.append("ci_rows")
    back = ", transf = exp" if log else ""
    return (
        "library(metafor)\n\n" + "".join(parts)
        + f"dat <- rbind({', '.join(frames)})\n"
        'fe <- rma(yi, vi, data = dat, method = "FE", slab = study)\n'
        f're <- rma(yi, vi, data = dat, method = "{method}", slab = study)\n'
        "summary(re)  # Q, tau^2, I^2\n"
        + ("# With PM, metafor derives I^2 from tau^2; uSTAT's I^2 is (Q - df) / Q,\n"
           "# which equals metafor's only for DL.\n" if method == "PM" else "")
        + "# uSTAT's 95% prediction interval uses t with k - 2 df (Higgins 2009);\n"
        "# predict() uses the normal quantile unless the model was fitted with test = \"t\".\n"
        f"predict(fe{back})\npredict(re{back})"
    )


def analyze(b: dict) -> dict:
    measure = str(b.get("measure") or "OR").upper()
    method = "PM" if str(b.get("tau2_method") or "DL").upper() == "PM" else "DL"
    cc = number(b.get("cc"), 0.5)
    tab, se, ci = _split(b.get("studies") or [])
    if measure not in _SM_STAT:
        tab = []  # uSTAT accepts 2x2 input only for OR, RR and RD
    log = measure in _LOG
    title = f"Meta-analysis ({measure}, {method} tau^2): {len(tab) + len(se) + len(ci)} studies"
    if not (tab or se or ci):
        return {"title": title,
                "python": "# No usable study rows in the request (need a 2x2, effect + SE or effect + CI).",
                "r": "# No usable study rows in the request (need a 2x2, effect + SE or effect + CI)."}
    return {"title": title,
            "python": _python(tab, se, ci, measure, log, cc, method),
            "r": _r(tab, se, ci, measure, log, cc, method)}


ENDPOINTS = {"/api/meta/analyze": analyze}
