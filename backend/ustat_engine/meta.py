"""Meta-analysis: fixed/random-effects pooling, subgroups, meta-regression, bias.

Extracted from routers/meta.py (all four endpoint bodies and their private
helpers). The computational body is verbatim: every expression, branch and
constant is the one that was there before, because this module's whole
purpose is to be provably the same arithmetic as the code it replaced, now
runnable in either runtime.

Second analysis to move, for the same reason `stats/power.py` was the first:
every number these four endpoints need comes in the request body, so there is
no dataset/session state to marshal across the runtime boundary.

Translation from the FastAPI original, and nothing else:
  - `class MetaStudy(BaseModel)` became the `_S` shim below, and
    `class MetaRequest(BaseModel)` became the `_R` shim, applying the same
    field defaults, so the copied body still says `s.<field>` / `req.<field>`.
    Keeping those references intact is what lets this file be diffed against
    its source.
  - `raise HTTPException(status_code=X, detail=Y)` became
    `raise EngineError(Y, status_hint=X)`, detail strings unchanged.
  - The local `_safe` helper (shallow — it only ever unwrapped a bare scalar,
    and every call site here passed it a dict, so it was a no-op on every one
    of these endpoints) is replaced by `sanitize` from `.jsonsafe`, which
    actually recurses. A strict superset of what `_safe` did, not a behaviour
    change.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy import stats as st

from .errors import EngineError
from .jsonsafe import sanitize
from .registry import register
from .spec import AnalysisSpec

_Z95 = 1.959963984540054


class _S:
    """Shim standing in for one pydantic `MetaStudy`, applying the field
    defaults declared on `MetaStudy` exactly, so the copied body can keep
    using `s.<field>` attribute access unchanged."""

    def __init__(self, params: dict):
        if "label" not in params:
            raise EngineError("Field 'label' is required.", status_hint=422)
        self.label = params["label"]
        # Option A: pre-computed effect + CI
        self.effect = params.get("effect", None)
        self.ci_low = params.get("ci_low", None)
        self.ci_high = params.get("ci_high", None)
        # Option B: pre-computed effect + SE
        self.se = params.get("se", None)
        # Option C: raw 2×2 — treated (e1/n1) vs control (e2/n2)
        self.e1 = params.get("e1", None)
        self.n1 = params.get("n1", None)
        self.e2 = params.get("e2", None)
        self.n2 = params.get("n2", None)
        # Subgroup / moderator
        self.subgroup = params.get("subgroup", None)
        self.moderator = params.get("moderator", None)


class _R:
    """Shim standing in for the pydantic `MetaRequest`, so the copied
    function bodies below can keep using `req.<field>` unchanged. Defaults
    mirror the field declarations of `MetaRequest` exactly."""

    def __init__(self, params: dict):
        if "studies" not in params:
            raise EngineError("Field 'studies' is required.", status_hint=422)
        self.studies = [_S(s) for s in params["studies"]]
        self.measure = params.get("measure", "OR")            # OR | RR | RD | SMD | MD | generic
        self.tau2_method = params.get("tau2_method", "DL")    # DL | PM
        self.cc = params.get("cc", 0.5)                        # continuity correction for zero cells (2×2)


# ─────────────────────────────────────────────────────────────────────────
# Everything below this line is copied verbatim (module-level HTTPException
# raises rewritten to EngineError; `req`/`s` now bound to the `_R`/`_S` shims
# instead of pydantic models) from routers/meta.py.
# ─────────────────────────────────────────────────────────────────────────

def _log_scale(measure: str) -> bool:
    return measure.upper() in ("OR", "RR", "HR")


def _study_effect(s: "_S", measure: str, cc: float) -> tuple:
    """Return (effect_on_analysis_scale, se_on_analysis_scale, display_effect).

    For OR/RR the analysis scale is the natural log; display_effect is the
    exponentiated value. For RD/MD/SMD/generic the analysis scale == display.
    """
    log = _log_scale(measure)

    # Option C — raw 2×2
    if s.e1 is not None and s.n1 is not None and s.e2 is not None and s.n2 is not None:
        a, n1, c, n2 = float(s.e1), float(s.n1), float(s.e2), float(s.n2)
        b, d = n1 - a, n2 - c
        if min(a, b, c, d) == 0:
            a, b, c, d = a + cc, b + cc, c + cc, d + cc
            n1, n2 = a + b, c + d
        if measure.upper() == "OR":
            eff = math.log((a * d) / (b * c))
            se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            return eff, se, math.exp(eff)
        if measure.upper() == "RR":
            r1, r2 = a / n1, c / n2
            eff = math.log(r1 / r2)
            se = math.sqrt((1 - r1) / a + (1 - r2) / c)
            return eff, se, math.exp(eff)
        if measure.upper() == "RD":
            r1, r2 = a / n1, c / n2
            eff = r1 - r2
            se = math.sqrt(r1 * (1 - r1) / n1 + r2 * (1 - r2) / n2)
            return eff, se, eff
        raise EngineError(f"2×2 input not supported for measure '{measure}'.", status_hint=422)

    # Option B — effect + SE
    if s.effect is not None and s.se is not None:
        eff_disp = float(s.effect)
        if log and eff_disp <= 0:
            raise EngineError(f"Study '{s.label}' has non-positive {measure}; log-scale measures must be > 0.", status_hint=422)
        if log:
            return math.log(max(eff_disp, 1e-12)), float(s.se), eff_disp
        return eff_disp, float(s.se), eff_disp

    # Option A — effect + CI
    if s.effect is not None and s.ci_low is not None and s.ci_high is not None:
        if log and (s.effect <= 0 or s.ci_low <= 0 or s.ci_high <= 0):
            raise EngineError(f"Study '{s.label}' has non-positive effect/CI; log-scale measures must be > 0.", status_hint=422)
        if log:
            le, llo, lhi = (math.log(max(v, 1e-12)) for v in (s.effect, s.ci_low, s.ci_high))
            return le, (lhi - llo) / (2 * _Z95), float(s.effect)
        return float(s.effect), (s.ci_high - s.ci_low) / (2 * _Z95), float(s.effect)

    raise EngineError(
        f"Study '{s.label}' lacks usable inputs (need effect+CI, effect+SE, or a 2×2 table).",
        status_hint=422,
    )


def _tau2_DL(y: np.ndarray, v: np.ndarray) -> float:
    w = 1.0 / v
    mu = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mu) ** 2))
    df = len(y) - 1
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    return max(0.0, (q - df) / c) if c > 0 else 0.0


def _tau2_PM(y: np.ndarray, v: np.ndarray, max_iter: int = 100) -> float:
    """Paule-Mandel iterative τ²."""
    df = len(y) - 1
    if df <= 0:
        return 0.0
    tau2 = _tau2_DL(y, v)
    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        mu = np.sum(w * y) / np.sum(w)
        f = float(np.sum(w * (y - mu) ** 2)) - df
        if abs(f) < 1e-6:
            break
        deriv = -float(np.sum((w ** 2) * (y - mu) ** 2))
        if deriv == 0:
            break
        tau2 = max(0.0, tau2 - f / deriv)
    return float(tau2)


def _hetero(y: np.ndarray, v: np.ndarray) -> dict:
    w = 1.0 / v
    mu = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mu) ** 2))
    df = len(y) - 1
    i2 = max(0.0, (q - df) / q * 100.0) if q > 0 else 0.0
    h2 = (q / df) if df > 0 else None
    q_p = float(1 - st.chi2.cdf(q, df)) if df > 0 else 1.0
    return {"Q": round(q, 4), "Q_df": df, "Q_p": round(q_p, 5),
            "I2_pct": round(i2, 2), "H2": round(h2, 4) if h2 is not None else None}


def _pool(y: np.ndarray, v: np.ndarray, tau2: float) -> dict:
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    var = float(1.0 / np.sum(w))
    se = math.sqrt(var)
    return {"mu": mu, "se": se, "ci_low": mu - _Z95 * se, "ci_high": mu + _Z95 * se, "weights": w}


def _prep(req: "_R"):
    if len(req.studies) < 2:
        raise EngineError("Need at least 2 studies.", status_hint=422)
    rows = []
    for s in req.studies:
        eff, se, disp = _study_effect(s, req.measure, req.cc)
        if not math.isfinite(se) or se <= 0:
            raise EngineError(f"Study '{s.label}' has a non-positive SE.", status_hint=422)
        rows.append({"label": s.label, "y": eff, "se": se, "v": se * se,
                     "disp": disp, "subgroup": s.subgroup, "moderator": s.moderator})
    return rows


def _back(measure: str, x: float) -> float:
    return math.exp(x) if _log_scale(measure) else x


# ── 1. Main analysis ─────────────────────────────────────────────────────────


def analyze(params: dict) -> dict:
    req = _R(params)
    rows = _prep(req)
    y = np.array([r["y"] for r in rows])
    v = np.array([r["v"] for r in rows])
    k = len(rows)
    log = _log_scale(req.measure)

    tau2 = _tau2_PM(y, v) if req.tau2_method.upper() == "PM" else _tau2_DL(y, v)
    fe = _pool(y, v, 0.0)
    re = _pool(y, v, tau2)
    het = _hetero(y, v)

    # 95% prediction interval (Higgins 2009): mu ± t_{k-2} * sqrt(tau2 + se_re^2)
    pi_low = pi_high = None
    if k >= 3 and tau2 > 0:
        t = float(st.t.ppf(0.975, k - 2))
        spread = math.sqrt(tau2 + re["se"] ** 2)
        pi_low = re["mu"] - t * spread
        pi_high = re["mu"] + t * spread

    # Per-study rows with RE weight %
    w_re = re["weights"]
    w_pct = 100.0 * w_re / w_re.sum()
    study_rows = []
    for i, r in enumerate(rows):
        lo = r["y"] - _Z95 * r["se"]
        hi = r["y"] + _Z95 * r["se"]
        study_rows.append({
            "label": r["label"],
            "effect": round(_back(req.measure, r["y"]), 4),
            "ci_low": round(_back(req.measure, lo), 4),
            "ci_high": round(_back(req.measure, hi), 4),
            "se": round(r["se"], 4),
            "weight_pct": round(float(w_pct[i]), 2),
        })

    def _fmt(p):
        return _back(req.measure, p["mu"]), _back(req.measure, p["ci_low"]), _back(req.measure, p["ci_high"])
    fe_e, fe_lo, fe_hi = _fmt(fe)
    re_e, re_lo, re_hi = _fmt(re)

    interp = (
        f"Random-effects meta-analysis ({req.tau2_method.upper()} τ²; k = {k} studies). "
        f"Pooled {req.measure} = {re_e:.3f} (95% CI {re_lo:.3f}–{re_hi:.3f}). "
        f"Heterogeneity: Q({het['Q_df']}) = {het['Q']:.2f}, p = {het['Q_p']:.4f}, "
        f"I² = {het['I2_pct']:.1f}%, τ² = {tau2:.4f}."
    )
    if pi_low is not None:
        interp += f" 95% prediction interval {_back(req.measure, pi_low):.3f}–{_back(req.measure, pi_high):.3f}."

    export = [["Study", req.measure, "CI low", "CI high", "Weight %"]]
    for s in study_rows:
        export.append([s["label"], s["effect"], s["ci_low"], s["ci_high"], s["weight_pct"]])
    export.append(["Pooled (RE)", round(re_e, 4), round(re_lo, 4), round(re_hi, 4), 100.0])

    return sanitize({
        "test": "Meta-analysis",
        "measure": req.measure, "k": k, "log_scale": log,
        "tau2_method": req.tau2_method.upper(), "tau2": round(tau2, 6),
        "studies": study_rows,
        "fixed": {"effect": round(fe_e, 4), "ci_low": round(fe_lo, 4), "ci_high": round(fe_hi, 4)},
        "random": {"effect": round(re_e, 4), "ci_low": round(re_lo, 4), "ci_high": round(re_hi, 4)},
        "prediction_low": round(_back(req.measure, pi_low), 4) if pi_low is not None else None,
        "prediction_high": round(_back(req.measure, pi_high), 4) if pi_high is not None else None,
        **het,
        "null_line": 1.0 if log else 0.0,
        "interpretation": interp, "result_text": interp,
        "export_rows": export,
    })


# ── 2. Subgroup analysis ─────────────────────────────────────────────────────


def subgroup(params: dict) -> dict:
    req = _R(params)
    rows = _prep(req)
    if not any(r["subgroup"] for r in rows):
        raise EngineError("No 'subgroup' values supplied on the studies.", status_hint=422)
    log = _log_scale(req.measure)
    groups: Dict[str, list] = {}
    for r in rows:
        groups.setdefault(str(r["subgroup"] or "—"), []).append(r)

    sub_results = []
    mus, vars_ = [], []
    for name, grp in groups.items():
        if len(grp) < 1:
            continue
        y = np.array([r["y"] for r in grp])
        v = np.array([r["v"] for r in grp])
        tau2 = (_tau2_PM(y, v) if req.tau2_method.upper() == "PM" else _tau2_DL(y, v)) if len(grp) >= 2 else 0.0
        pooled = _pool(y, v, tau2)
        het = _hetero(y, v) if len(grp) >= 2 else {"Q": 0, "Q_df": 0, "Q_p": None, "I2_pct": 0, "H2": None}
        mus.append(pooled["mu"])
        vars_.append(pooled["se"] ** 2)
        sub_results.append({
            "subgroup": name, "k": len(grp),
            "effect": round(_back(req.measure, pooled["mu"]), 4),
            "ci_low": round(_back(req.measure, pooled["ci_low"]), 4),
            "ci_high": round(_back(req.measure, pooled["ci_high"]), 4),
            "tau2": round(tau2, 6), "I2_pct": het["I2_pct"],
        })

    # Between-subgroup heterogeneity: Q_between on the subgroup pooled means.
    q_between = q_between_p = None
    if len(mus) >= 2:
        mus_a = np.array(mus)
        w = 1.0 / np.array(vars_)
        grand = float(np.sum(w * mus_a) / np.sum(w))
        q_between = float(np.sum(w * (mus_a - grand) ** 2))
        df_b = len(mus) - 1
        q_between_p = float(1 - st.chi2.cdf(q_between, df_b)) if df_b > 0 else None

    interp = (
        f"Subgroup meta-analysis across {len(sub_results)} subgroups. "
        + (f"Between-subgroup Q = {q_between:.2f}, p = {q_between_p:.4f} — "
           + ("subgroups differ significantly." if (q_between_p is not None and q_between_p < 0.05)
              else "no significant subgroup difference.")
           if q_between is not None else "")
    )

    return sanitize({
        "test": "Subgroup meta-analysis", "measure": req.measure,
        "subgroups": sub_results,
        "q_between": round(q_between, 4) if q_between is not None else None,
        "q_between_df": len(mus) - 1 if len(mus) >= 2 else None,
        "q_between_p": round(q_between_p, 5) if q_between_p is not None else None,
        "null_line": 1.0 if log else 0.0,
        "interpretation": interp, "result_text": interp,
    })


# ── 3. Meta-regression ───────────────────────────────────────────────────────


def meta_regression(params: dict) -> dict:
    req = _R(params)
    rows = _prep(req)
    if not all(r["moderator"] is not None for r in rows):
        raise EngineError("Every study needs a numeric 'moderator' value.", status_hint=422)
    if len(rows) < 3:
        raise EngineError("Need ≥ 3 studies for meta-regression.", status_hint=422)
    import statsmodels.api as sm

    y = np.array([r["y"] for r in rows])
    v = np.array([r["v"] for r in rows])
    x = np.array([float(r["moderator"]) for r in rows])
    tau2 = _tau2_PM(y, v) if req.tau2_method.upper() == "PM" else _tau2_DL(y, v)
    w = 1.0 / (v + tau2)

    X = sm.add_constant(x)
    model = sm.WLS(y, X, weights=w).fit()
    intercept, slope = float(model.params[0]), float(model.params[1])
    se_slope = float(model.bse[1])
    p_slope = float(model.pvalues[1])
    ci = model.conf_int()
    slope_lo, slope_hi = float(ci[1][0]), float(ci[1][1])

    # Residual heterogeneity + weighted model R². This is the WLS fit R², not
    # a direct "proportion of tau² explained" estimate.
    tau2_resid = max(0.0, tau2 * (1 - model.rsquared)) if tau2 > 0 else 0.0
    r2_analog = round(model.rsquared * 100, 2)

    interp = (
        f"Meta-regression of {req.measure} on the moderator (k = {len(rows)}). "
        f"Slope = {slope:.4f} (95% CI {slope_lo:.4f}–{slope_hi:.4f}), p = "
        f"{'<0.001' if p_slope < 0.001 else f'{p_slope:.3f}'}. "
        + ("The moderator significantly explains effect-size variation."
           if p_slope < 0.05 else "No significant moderator effect.")
    )

    # Bubble-plot points (moderator vs effect, size ∝ weight)
    wmax = float(w.max())
    points = [{
        "moderator": round(float(x[i]), 4),
        "effect": round(_back(req.measure, y[i]), 4),
        "size": round(8 + 22 * float(w[i]) / wmax, 1),
        "label": rows[i]["label"],
    } for i in range(len(rows))]
    line_x = [float(x.min()), float(x.max())]
    line_y = [round(_back(req.measure, intercept + slope * xx), 4) for xx in line_x]

    return sanitize({
        "test": "Meta-regression", "measure": req.measure, "k": len(rows),
        "intercept": round(intercept, 5), "slope": round(slope, 5),
        "slope_se": round(se_slope, 5), "slope_ci_low": round(slope_lo, 5),
        "slope_ci_high": round(slope_hi, 5), "slope_p": round(p_slope, 6),
        "r2_pct": r2_analog,
        "r2_label": "Weighted least-squares R² (%)",
        "r2_note": "WLS model R² from the weighted meta-regression; not a direct proportion of tau² explained.",
        "tau2": round(tau2, 6), "tau2_resid": round(tau2_resid, 6),
        "points": points, "line_x": line_x, "line_y": line_y,
        "log_scale": _log_scale(req.measure),
        "interpretation": interp, "result_text": interp,
    })


# ── 4. Publication-bias diagnostics ──────────────────────────────────────────


def bias(params: dict) -> dict:
    req = _R(params)
    rows = _prep(req)
    if len(rows) < 3:
        raise EngineError("Need ≥ 3 studies for bias diagnostics.", status_hint=422)
    import statsmodels.api as sm

    y = np.array([r["y"] for r in rows])
    se = np.array([r["se"] for r in rows])
    v = se ** 2

    # Egger's regression test: standardized effect (y/se) ~ precision (1/se);
    # the intercept tests funnel asymmetry.
    z = y / se
    prec = 1.0 / se
    Xe = sm.add_constant(prec)
    egger = sm.OLS(z, Xe).fit()
    egger_int = float(egger.params[0])
    egger_p = float(egger.pvalues[0])

    # Begg's rank correlation: Kendall τ between standardized effect and variance.
    mu_fe = float(np.sum((1 / v) * y) / np.sum(1 / v))
    v_star = v - 1.0 / np.sum(1 / v)
    star = (y - mu_fe) / np.sqrt(np.where(v_star > 0, v_star, np.nan))
    mask = np.isfinite(star)
    begg_tau, begg_p = (float("nan"), float("nan"))
    if mask.sum() >= 3:
        bt, bp = st.kendalltau(star[mask], v[mask])
        begg_tau, begg_p = float(bt), float(bp)

    # Trim-and-fill (L0, random side opposite the pooled mean) — estimate of
    # the number of missing studies.
    log = _log_scale(req.measure)
    order = np.argsort(y)
    y_sorted = y[order]
    mu = float(np.mean(y_sorted))
    centered = y_sorted - mu
    ranks = st.rankdata(np.abs(centered))
    signs = np.sign(centered)
    # Tweedie L0 estimator
    Tn = float(np.sum(ranks[signs > 0]))
    n = len(y_sorted)
    l0 = (4 * Tn - n * (n + 1)) / (2 * n - 1)
    k0 = max(0, int(round(l0)))
    if egger_p < 0.05 and k0 == 0:
        side_imbalance = abs(int(np.sum(y > mu_fe)) - int(np.sum(y < mu_fe)))
        k0 = max(1, side_imbalance)

    funnel = [{
        "effect": round(_back(req.measure, y[i]), 4),
        "se": round(float(se[i]), 4),
        "label": rows[i]["label"],
    } for i in range(len(rows))]
    # Pooled mean + pseudo 95% funnel guides (for the plot)
    se_max = float(se.max())

    interp = (
        f"Publication-bias diagnostics (k = {len(rows)}). "
        f"Egger's intercept = {egger_int:.3f}, p = "
        f"{'<0.001' if egger_p < 0.001 else f'{egger_p:.3f}'} "
        + ("(funnel asymmetry present)." if egger_p < 0.05 else "(no significant asymmetry).")
        + (f" Begg's τ = {begg_tau:.3f}, p = {begg_p:.3f}." if np.isfinite(begg_tau) else "")
        + f" Trim-and-fill estimates {k0} potentially missing study(ies)."
    )

    return sanitize({
        "test": "Publication bias", "measure": req.measure, "k": len(rows),
        "egger_intercept": round(egger_int, 4), "egger_p": round(egger_p, 5),
        "begg_tau": round(begg_tau, 4) if np.isfinite(begg_tau) else None,
        "begg_p": round(begg_p, 5) if np.isfinite(begg_p) else None,
        "trim_fill_missing": k0,
        "funnel": funnel,
        "pooled_effect": round(_back(req.measure, mu_fe), 4),
        "se_max": round(se_max, 4),
        "log_scale": log,
        "interpretation": interp, "result_text": interp,
    })


register(
    AnalysisSpec(
        id="meta.analyze",
        fn=lambda params: analyze(params),
        needs_frame=False,
        deps=("numpy", "scipy"),
        required_columns=lambda params: [],
        cost_key="meta.analyze",
        doc="Fixed + random-effects meta-analysis pooling (DL / PM τ²), "
            "Q / I² / H² heterogeneity, 95% prediction interval, per-study weights.",
        tags=("meta-analysis", "pooling"),
    )
)

register(
    AnalysisSpec(
        id="meta.subgroup",
        fn=lambda params: subgroup(params),
        needs_frame=False,
        deps=("numpy", "scipy"),
        required_columns=lambda params: [],
        cost_key="meta.subgroup",
        doc="Subgroup meta-analysis pooling with between-group heterogeneity (Q_between).",
        tags=("meta-analysis", "subgroup"),
    )
)

register(
    AnalysisSpec(
        id="meta.regression",
        fn=lambda params: meta_regression(params),
        needs_frame=False,
        deps=("numpy", "statsmodels"),
        required_columns=lambda params: [],
        cost_key="meta.regression",
        doc="Meta-regression (effect ~ moderator) via weighted least squares.",
        tags=("meta-analysis", "regression"),
    )
)

register(
    AnalysisSpec(
        id="meta.bias",
        fn=lambda params: bias(params),
        needs_frame=False,
        deps=("numpy", "scipy", "statsmodels"),
        required_columns=lambda params: [],
        cost_key="meta.bias",
        doc="Publication-bias diagnostics: Egger's test, Begg rank test, "
            "funnel points, trim-and-fill missing-study estimate.",
        tags=("meta-analysis", "bias"),
    )
)
