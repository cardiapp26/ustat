"""Threshold (two-piecewise) regression — where does the effect change?

A restricted cubic spline answers "is this relationship curved?" and stops
there. The question a clinical paper actually asks next is "curved where, and
how much on each side?" — the table every NHANES-style analysis prints:
inflection point K, the slope below it, the slope above it, and a test that
two lines fit better than one.

The model is

    y ~ x + (x - K)_+ + covariates

so the coefficient on x is the slope below K and the sum of the two
coefficients is the slope above it. K is not estimated by the usual machinery:
the likelihood is not differentiable in it, so it is found by profiling — fit
the model at every candidate K on a grid and keep the best. That is the same
quantity R's `segmented` converges to by Muggeo's iterative method, reached a
different way.

The p-value comparing one line against two is reported as a likelihood-ratio
test on 1 df, which is what the literature prints. It deserves the caveat that
travels with it, and the response carries that caveat rather than leaving it
to be remembered: K is chosen by maximising the very likelihood being tested,
so the null distribution is not really chi-square on 1 df and the p-value runs
optimistic. It is evidence, not proof, of a threshold.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel
from scipy import stats as scipy_stats

from services import store
from services.dirty_value_guard import coerce_numeric
from services.number_format import format_p
from services.stat_utils import sanitize_nonfinite

router = APIRouter()

# Candidate breakpoints are searched between these quantiles of the exposure.
# Outside them a "segment" is a handful of observations and its slope is noise
# fitted to the tail, which is how a threshold analysis produces an inflection
# point sitting on top of one outlier.
SEARCH_LO, SEARCH_HI = 0.10, 0.90
GRID_N = 100
# Each side of the split needs enough observations to estimate a slope at all.
MIN_SIDE = 10


class ThresholdRequest(BaseModel):
    session_id: str
    outcome: str
    exposure: str
    outcome_kind: str = "continuous"          # continuous | binary | survival
    time_col: Optional[str] = None            # survival only
    covariates: List[str] = []
    categorical: List[str] = []               # covariates to dummy-code
    grid_n: int = GRID_N


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _finite(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _encode_covariates(df: pd.DataFrame, covariates: List[str], categorical: List[str]) -> pd.DataFrame:
    """Numeric covariates as-is, declared categoricals dummy-coded.

    A covariate that is categorical but left undeclared would be fitted as a
    number — treating hospital ward 3 as three times ward 1 — so anything
    non-numeric is dummy-coded whether or not it was declared.
    """
    if not covariates:
        return pd.DataFrame(index=df.index)
    raw = df[covariates].copy()
    num_cols, cat_cols = [], []
    for c in covariates:
        if c in categorical or not pd.api.types.is_numeric_dtype(raw[c]):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    num_part = raw[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame(index=raw.index)
    cat_part = pd.get_dummies(raw[cat_cols].astype("object"), drop_first=True, dummy_na=False) if cat_cols else pd.DataFrame(index=raw.index)
    return pd.concat([num_part, cat_part], axis=1).astype(float)


# ── model fitting ──────────────────────────────────────────────────────────────


class _Fit:
    """One fitted model: its log-likelihood and the exposure coefficients."""

    def __init__(self, loglik: float, params: pd.Series, bse: pd.Series, pvalues: pd.Series,
                 cov: np.ndarray, names: List[str], n_params: int, df_resid: Optional[float] = None):
        self.loglik = loglik
        # Residual df for an OLS fit, None where the reference distribution is
        # normal (GLM, Cox). Which one applies is a property of the model, so
        # it is recorded here rather than re-derived at every call site.
        self.df_resid = df_resid
        self.params = params
        self.bse = bse
        self.pvalues = pvalues
        self.cov = cov
        self.names = names
        self.n_params = n_params


def _fit(kind: str, y: pd.Series, X: pd.DataFrame, time: Optional[pd.Series]) -> Optional[_Fit]:
    """Fit the model for the requested outcome type, or None if it fails.

    A failure here is ordinary — a candidate breakpoint can make a design
    matrix singular — so the caller drops that grid point rather than aborting
    the whole search.
    """
    try:
        if kind == "continuous":
            design = sm.add_constant(X, has_constant="add")
            res = sm.OLS(y, design).fit()
            return _Fit(float(res.llf), res.params, res.bse, res.pvalues,
                        np.asarray(res.cov_params()), list(design.columns), int(design.shape[1]) + 1,
                        df_resid=float(res.df_resid))
        if kind == "binary":
            design = sm.add_constant(X, has_constant="add")
            # statsmodels' default IRLS tolerance (1e-8) stops one iterate short,
            # and the standard error is then computed from that iterate's
            # weights — 1.1e-7 off the analytic sqrt(diag((X'WX)^-1)) at the
            # fitted coefficients. At 1e-12 it is exact to 6e-14, for the cost
            # of a couple of iterations. (R's glm has the same behaviour at its
            # own default and keeps it even at epsilon = 1e-12, which is where
            # the last residual disagreement with R comes from.)
            res = sm.GLM(y, design, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-12)
            return _Fit(float(res.llf), res.params, res.bse, res.pvalues,
                        np.asarray(res.cov_params()), list(design.columns), int(design.shape[1]))
        if kind == "survival":
            from lifelines import CoxPHFitter

            data = X.copy()
            data["_t"] = np.asarray(time, dtype=float)
            data["_e"] = np.asarray(y, dtype=float)
            cph = CoxPHFitter()
            cph.fit(data, duration_col="_t", event_col="_e")
            return _Fit(float(cph.log_likelihood_), cph.params_, cph.standard_errors_,
                        cph.summary["p"], np.asarray(cph.variance_matrix_),
                        list(cph.params_.index), int(len(cph.params_)))
    except Exception as exc:  # singular design, separation, non-convergence
        logger.debug(f"threshold fit failed ({kind}): {exc}")
        return None
    return None


def _design(x: pd.Series, cov: pd.DataFrame, k: Optional[float]) -> pd.DataFrame:
    """Exposure (plus its hinge when k is given) beside the covariates."""
    out = pd.DataFrame({"exposure": x.astype(float)}, index=x.index)
    if k is not None:
        out["exposure_above"] = np.clip(x.astype(float) - k, 0.0, None)
    if not cov.empty:
        out = pd.concat([out, cov], axis=1)
    return out


def _effect(fit: _Fit, kind: str, names: List[str], weights: List[float]) -> dict:
    """A linear combination of exposure coefficients with its CI and p.

    The slope above the breakpoint is beta1 + beta2, so its standard error
    needs the covariance between them — adding the two standard errors in
    quadrature would ignore it and, since the two are strongly negatively
    correlated here, would report an interval far wider than the truth.
    """
    idx = [fit.names.index(n) for n in names if n in fit.names]
    if len(idx) != len(names):
        return {}
    w = np.array(weights, dtype=float)
    beta = float(np.dot(w, fit.params.iloc[idx].to_numpy(dtype=float)))
    sub = fit.cov[np.ix_(idx, idx)]
    var = float(w @ sub @ w)
    if not np.isfinite(var) or var < 0:
        return {}
    se = float(np.sqrt(var))
    stat = beta / se if se > 0 else np.nan
    # A linear model tests its coefficients against t on the residual df, not
    # against the normal. With a few hundred rows the two agree to the third
    # decimal at p = 0.05 and diverge by twenty orders of magnitude far out in
    # the tail, and at small n the interval is simply too narrow — 1.96 where
    # t(20) wants 2.09. GLM and Cox are normal-based and keep z.
    if fit.df_resid is not None and fit.df_resid > 0:
        p = float(2 * scipy_stats.t.sf(abs(stat), fit.df_resid)) if np.isfinite(stat) else None
        crit = float(scipy_stats.t.ppf(0.975, fit.df_resid))
    else:
        p = float(2 * scipy_stats.norm.sf(abs(stat))) if np.isfinite(stat) else None
        crit = 1.959963984540054
    lo, hi = beta - crit * se, beta + crit * se
    out = {"beta": _finite(beta), "se": _finite(se), "p": _finite(p),
           "ci_low": _finite(lo), "ci_high": _finite(hi)}
    if kind in ("binary", "survival"):
        out["ratio"] = _finite(np.exp(beta))
        out["ratio_ci_low"] = _finite(np.exp(lo))
        out["ratio_ci_high"] = _finite(np.exp(hi))
    return out


# ── endpoint ───────────────────────────────────────────────────────────────────


@router.post("/analyze")
def threshold_analysis(req: ThresholdRequest):
    """Fit one line, then two, and report where the second one starts."""
    df = _get_df(req.session_id)

    kind = (req.outcome_kind or "continuous").strip().lower()
    if kind not in ("continuous", "binary", "survival"):
        raise HTTPException(status_code=400, detail="outcome_kind must be continuous, binary or survival")
    for col in [req.outcome, req.exposure, *req.covariates]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column not found: {col}")
    if kind == "survival":
        if not req.time_col or req.time_col not in df.columns:
            raise HTTPException(status_code=400, detail="A survival analysis needs a time column")

    x_all = coerce_numeric(df[req.exposure]).replace([np.inf, -np.inf], np.nan)
    if x_all.notna().sum() == 0:
        raise HTTPException(status_code=400, detail=f"'{req.exposure}' has no numeric values to search over")
    y_all = coerce_numeric(df[req.outcome]).replace([np.inf, -np.inf], np.nan)
    cov_all = _encode_covariates(df, req.covariates, req.categorical)
    t_all = coerce_numeric(df[req.time_col]).replace([np.inf, -np.inf], np.nan) if kind == "survival" else None

    # Complete cases across everything the model touches, so the one-line and
    # two-line models are compared on the same rows — fitted on different
    # subsets their likelihoods are not comparable and the test is meaningless.
    parts = [x_all, y_all] + ([t_all] if t_all is not None else [])
    keep = pd.concat(parts + ([cov_all] if not cov_all.empty else []), axis=1).notna().all(axis=1)
    n_used = int(keep.sum())
    n_dropped = int(len(df) - n_used)
    if n_used < 3 * MIN_SIDE:
        raise HTTPException(
            status_code=400,
            detail=f"Only {n_used} complete rows — too few to estimate a breakpoint and a slope on each side")

    x, y, cov = x_all[keep], y_all[keep], cov_all[keep] if not cov_all.empty else pd.DataFrame(index=x_all[keep].index)
    t = t_all[keep] if t_all is not None else None

    if kind == "binary":
        levels = set(pd.unique(y.dropna()))
        if not levels <= {0.0, 1.0} or len(levels) < 2:
            raise HTTPException(status_code=400, detail="A binary outcome must be coded 0 / 1 and contain both")

    linear = _fit(kind, y, _design(x, cov, None), t)
    if linear is None:
        raise HTTPException(status_code=400, detail="The single-line model did not converge on these variables")

    # Profile the likelihood over candidate breakpoints. K is not a regular
    # parameter — the likelihood has a kink at every observed x — so it is
    # searched rather than solved for.
    lo, hi = float(x.quantile(SEARCH_LO)), float(x.quantile(SEARCH_HI))
    grid_n = max(20, min(int(req.grid_n or GRID_N), 400))
    grid = np.unique(np.linspace(lo, hi, grid_n))
    profile: list[dict] = []
    best: Optional[tuple[float, _Fit]] = None
    for k in grid:
        below, above = int((x <= k).sum()), int((x > k).sum())
        if below < MIN_SIDE or above < MIN_SIDE:
            continue
        fit = _fit(kind, y, _design(x, cov, float(k)), t)
        if fit is None:
            continue
        profile.append({"k": _finite(k), "loglik": _finite(fit.loglik)})
        if best is None or fit.loglik > best[1].loglik:
            best = (float(k), fit)

    if best is None or not profile:
        raise HTTPException(
            status_code=400,
            detail=f"No breakpoint left at least {MIN_SIDE} observations on each side with a model that converged")

    k_hat, seg = best
    below = _effect(seg, kind, ["exposure"], [1.0])
    above = _effect(seg, kind, ["exposure", "exposure_above"], [1.0, 1.0])
    difference = _effect(seg, kind, ["exposure_above"], [1.0])
    single = _effect(linear, kind, ["exposure"], [1.0])

    lr_stat = float(2.0 * (seg.loglik - linear.loglik))
    lr_p = float(scipy_stats.chi2.sf(lr_stat, 1)) if lr_stat > 0 else 1.0

    # Profile-likelihood interval for K: every candidate within 1.92 units of
    # the maximum, which is the 95% cut for a chi-square on 1 df. It is wide
    # when the profile is flat, which is exactly when the breakpoint should
    # not be quoted to three decimals.
    cutoff = seg.loglik - 1.920729
    inside = [p["k"] for p in profile if p["loglik"] is not None and p["loglik"] >= cutoff]
    k_ci = {"low": _finite(min(inside)), "high": _finite(max(inside))} if inside else {}
    flat = bool(inside and (max(inside) - min(inside)) > 0.5 * (hi - lo))

    # Fitted curve, drawn with every covariate held at its mean so the two
    # segments are comparable along the exposure rather than confounded by it.
    curve_x = np.linspace(float(x.min()), float(x.max()), 120)
    curve_df = pd.DataFrame({"exposure": curve_x})
    curve_df["exposure_above"] = np.clip(curve_x - k_hat, 0.0, None)
    for c in cov.columns:
        curve_df[c] = float(cov[c].mean())
    if kind == "survival":
        eta = curve_df[seg.names].to_numpy(dtype=float) @ seg.params.reindex(seg.names).to_numpy(dtype=float)
    else:
        design = sm.add_constant(curve_df[[c for c in seg.names if c != "const"]], has_constant="add")
        eta = design[seg.names].to_numpy(dtype=float) @ seg.params.reindex(seg.names).to_numpy(dtype=float)
    # A log-odds or log-hazard curve is centred at the breakpoint so it reads
    # as "relative to the inflection point", the convention these figures use.
    if kind in ("binary", "survival"):
        eta = eta - float(np.interp(k_hat, curve_x, eta))

    unit = {"continuous": "Mean difference", "binary": "Odds ratio", "survival": "Hazard ratio"}[kind]
    verdict = ("A two-segment model fits better than a straight line"
               if lr_p < 0.05 else
               "A straight line is not clearly improved on by a breakpoint")

    return sanitize_nonfinite({
        "outcome": req.outcome, "exposure": req.exposure, "outcome_kind": kind,
        "n_used": n_used, "n_dropped": n_dropped,
        "breakpoint": _finite(k_hat), "breakpoint_ci": k_ci,
        "search_range": {"low": _finite(lo), "high": _finite(hi), "n_candidates": len(profile)},
        "effect_below": below, "effect_above": above, "effect_difference": difference,
        "effect_single_line": single,
        "loglik_single": _finite(linear.loglik), "loglik_segmented": _finite(seg.loglik),
        "lr_stat": _finite(lr_stat), "lr_p": _finite(lr_p),
        "effect_label": unit,
        "profile": profile,
        "curve": {"x": [_finite(v) for v in curve_x], "y": [_finite(v) for v in eta]},
        "verdict": verdict,
        "result_text": _results_text(req, kind, k_hat, below, above, difference, lr_p, n_used, unit),
        "warnings": _warnings(flat, k_ci, lo, hi, k_hat, n_dropped, len(df)),
        # Carried in the payload, not only in the UI, so an exported result
        # keeps the caveat attached to the number it qualifies.
        "caveat": (
            "The breakpoint is chosen by maximising the same likelihood the test then uses, "
            "so the likelihood-ratio p-value is optimistic and the chi-square reference is "
            "approximate. Read it as evidence of a threshold, not proof of one, and check "
            "that the profile has a clear peak rather than a flat ridge."
        ),
    })


def _results_text(req, kind, k, below, above, diff, lr_p, n, unit) -> str:
    """A sentence that can go straight into a results section."""
    def eff(e):
        if not e:
            return "—"
        if kind == "continuous":
            return f"{e['beta']:.3f} (95% CI {e['ci_low']:.3f} to {e['ci_high']:.3f})"
        return f"{e['ratio']:.2f} (95% CI {e['ratio_ci_low']:.2f} to {e['ratio_ci_high']:.2f})"

    word = "mean difference" if kind == "continuous" else ("odds ratio" if kind == "binary" else "hazard ratio")
    return (
        f"A two-piecewise model identified an inflection point of {req.exposure} at {k:.3g}. "
        f"Below it the {word} per unit of {req.exposure} was {eff(below)}; above it, {eff(above)} "
        f"(n = {n}, log-likelihood ratio test {format_p(lr_p, prefix=True)}). "
        f"{unit} is reported per one-unit increase in {req.exposure}."
    )


def _warnings(flat: bool, k_ci: dict, lo: float, hi: float, k: float, n_dropped: int, n_total: int) -> list[str]:
    out: list[str] = []
    if flat:
        out.append(
            "The profile likelihood is flat: breakpoints across most of the searched range fit "
            "almost as well as the best one, so the inflection point is not well identified.")
    if k_ci and (k_ci.get("low") is not None) and abs(k - lo) < 1e-9:
        out.append("The best breakpoint sits at the low edge of the search range; the true one may lie below it.")
    if k_ci and (k_ci.get("high") is not None) and abs(k - hi) < 1e-9:
        out.append("The best breakpoint sits at the high edge of the search range; the true one may lie above it.")
    if n_dropped:
        pct = 100.0 * n_dropped / n_total if n_total else 0.0
        out.append(f"{n_dropped} of {n_total} rows ({pct:.1f}%) were dropped for missing values in the model variables.")
    return out
