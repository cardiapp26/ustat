from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
from scipy.stats import chi2

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from services import store
from services.stat_utils import sorted_groups
from services.impute import apply_imputation
from services.assumptions import (
    check_cox_assumptions_from_ph_test,
    add_assumption_warnings_to_result,
)
from services.missing_data import (
    mice_multiple,
    pool_cox_results,
    missing_pattern_summary,
    add_missing_data_diagnostics,
)
from services.rcs_basis import (
    KNOT_PERCENTILES as _KNOT_PERCENTILES,
    rcs_basis as _rcs_basis,
    resolve_knots as _resolve_knots,
)
from services.survival_validation import validate_survival_inputs

router = APIRouter()

# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    """Return float or None for inf/nan values that aren't JSON-serializable."""
    try:
        f = float(v)
        if np.isfinite(f):
            return f
        return None
    except (TypeError, ValueError):
        return None


def _clean(arr):
    return [_safe_float(v) for v in arr]


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _drop_invalid_survival_rows(df: pd.DataFrame, duration_col: str, event_col: str):
    res = validate_survival_inputs(df, duration_col, event_col, mode="drop_with_warning")
    return res.df, res.warnings, res.n_excluded


def _add_survival_warnings(result: dict, warnings: list[str], n_invalid: int) -> dict:
    if warnings:
        result["warnings"] = [*result.get("warnings", []), *warnings]
    if n_invalid:
        result["n_invalid_survival"] = int(n_invalid)
    return result


_MIN_AT_RISK = 10  # below this, a landmark estimate is flagged unreliable


def _surv_at(kmf: KaplanMeierFitter, t: float,
             durations: Optional[pd.Series] = None) -> dict:
    """Survival estimate (+ 95% CI) at an exact time point.

    Used for landmark statements like 'estimated 5-year survival was 77%'.
    The CI is read from lifelines' confidence_interval_survival_function_,
    stepped to the last event time at or before t. When ``durations`` is
    given, also reports n-at-risk and a ``reliable`` flag — False when t
    exceeds the longest follow-up or fewer than _MIN_AT_RISK remain, which
    is the guard against over-reading the tail (e.g. a '5-year' estimate
    when only 3 patients are still at risk).
    """
    out: dict = {"time": _safe_float(t), "survival": None, "ci_low": None,
                 "ci_high": None, "n_at_risk": None, "reliable": None}
    try:
        out["survival"] = _safe_float(float(kmf.survival_function_at_times(t).iloc[0]))
    except Exception:
        return out
    try:
        ci = kmf.confidence_interval_survival_function_
        sub = ci[ci.index <= t]
        row = sub.iloc[-1] if len(sub) else ci.iloc[0]
        out["ci_low"] = _safe_float(float(row.iloc[0]))
        out["ci_high"] = _safe_float(float(row.iloc[1]))
    except Exception:
        pass
    if durations is not None:
        d = durations.dropna().astype(float)
        n_risk = int((d >= float(t)).sum())
        tmax = float(d.max()) if len(d) else 0.0
        out["n_at_risk"] = n_risk
        out["reliable"] = bool(t <= tmax and n_risk >= _MIN_AT_RISK)
    return out


def _median_follow_up(time: pd.Series, event: pd.Series) -> Optional[dict]:
    """Median follow-up via reverse Kaplan–Meier (flip the event indicator).

    The naive median of observed times underestimates potential follow-up
    because deaths truncate it. Reverse KM treats censoring as the 'event'
    and estimates the censoring (= follow-up) distribution.

    The quantile is read off by **linear interpolation** of the curve
    crossing — matching R's ``quantile.survfit`` (the convention used in
    most clinical papers) rather than lifelines' step ``.percentile``
    (smallest t with S(t) ≤ p), which runs a few days higher. Returns
    median + IQR in the duration unit.
    """
    t = pd.to_numeric(time, errors="coerce")
    e = pd.to_numeric(event, errors="coerce")
    m = t.notna() & e.notna()
    t = t[m].astype(float)
    e = e[m].astype(int)
    if len(t) < 2 or e.nunique() < 1:
        return None
    try:
        kmf = KaplanMeierFitter().fit(t, event_observed=(1 - e))
    except Exception:
        logger.exception("Reverse-KM median follow-up failed")
        return None

    try:
        S = kmf.survival_function_.values.ravel().astype(float)
        T = kmf.survival_function_.index.values.astype(float)
        order = np.argsort(S)            # ascending S for np.interp
        Ss, Ts = S[order], T[order]

        def _pc(p: float) -> Optional[float]:
            v = float(np.interp(p, Ss, Ts))
            return v if np.isfinite(v) else None

        return {"median": _safe_float(_pc(0.50)),
                "q1": _safe_float(_pc(0.75)),   # 25th pct of follow-up
                "q3": _safe_float(_pc(0.25))}   # 75th pct of follow-up
    except Exception:
        logger.exception("Reverse-KM quantile interpolation failed")
        return None


def _sorted_groups(series: pd.Series) -> list:
    """Stable group order for KM curves / at-risk rows / legend.

    Sort by the underlying value code (1, 2, 3 …) numerically when every group
    is numeric-coercible, else lexicographically. Without this, groups follow
    their order of appearance in the data, so the curves/legend/at-risk rows
    come out scrambled relative to the value labels (1=<100, 2=100–130, 3=>130).
    """
    vals = list(pd.Series(series).dropna().unique())
    try:
        return sorted(vals, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def _km_fit_groups(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: Optional[str],
    survival_times: Optional[List[float]] = None,
    risk_times: Optional[List[float]] = None,
    include_censors: bool = False,
) -> list:
    groups = _sorted_groups(df[group_col]) if group_col else [None]
    results = []
    for grp in groups:
        subset = df[df[group_col] == grp] if group_col else df
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(
                subset[duration_col].astype(float),
                subset[event_col].astype(int),
                label=str(grp) if grp is not None else "All",
            )
        except Exception as exc:
            logger.exception("KM fitting failed")
            raise HTTPException(status_code=400, detail=f"KM fitting error: {exc}")
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["time", "survival"]
        curve = [
            {"time": _safe_float(row["time"]), "survival": _safe_float(row["survival"])}
            for _, row in sf.iterrows()
        ]
        row_out = {
            "group": str(grp) if grp is not None else "All",
            "n": int(len(subset)),
            "events": int(subset[event_col].sum()),
            "median_survival": _safe_float(kmf.median_survival_time_),
            "curve": curve,
        }
        if survival_times:
            _dur = subset[duration_col].astype(float)
            row_out["survival_at"] = [_surv_at(kmf, float(t), _dur) for t in survival_times]
        if risk_times:
            # Standard number-at-risk: subjects with follow-up ≥ t (at risk
            # at the start of the interval beginning at t).
            dur = subset[duration_col].astype(float)
            row_out["at_risk"] = [int((dur >= float(t)).sum()) for t in risk_times]
        if include_censors:
            # Censoring tick marks: time + the step survival value there, so
            # the frontend can drop a '+' on the curve at each censoring.
            cens_t = subset.loc[subset[event_col].astype(int) == 0, duration_col].astype(float)
            cens_t = sorted(set(round(float(t), 4) for t in cens_t.dropna()))
            if cens_t:
                try:
                    ys = kmf.survival_function_at_times(cens_t)
                    row_out["censors"] = [
                        {"time": _safe_float(t), "survival": _safe_float(float(ys.iloc[k]))}
                        for k, t in enumerate(cens_t)
                    ]
                except Exception:
                    logger.exception("Censor-point survival lookup failed")
        results.append(row_out)
    return results


def _km_pairwise(df: pd.DataFrame, duration_col: str, event_col: str,
                 group_col: str, correction: str = "none") -> Optional[dict]:
    """All pairwise log-rank comparisons, optionally multiplicity-adjusted.

    Lets the user state which specific group pair drives an overall
    difference (e.g. '<100 vs 100–130, p=0.003').
    """
    from itertools import combinations
    groups = _sorted_groups(df[group_col])
    if len(groups) < 3:
        return None  # pairwise only meaningful with ≥3 groups
    comparisons = []
    pvals: list[float] = []
    for a, b in combinations(groups, 2):
        ga = df[df[group_col] == a]
        gb = df[df[group_col] == b]
        try:
            lr = logrank_test(
                ga[duration_col], gb[duration_col],
                event_observed_A=ga[event_col].astype(int),
                event_observed_B=gb[event_col].astype(int),
            )
            p = float(lr.p_value)
        except Exception:
            logger.exception("Pairwise log-rank failed for %s vs %s", a, b)
            p = float("nan")
        comparisons.append({"group_a": str(a), "group_b": str(b), "p": _safe_float(p)})
        pvals.append(p)
    if correction and correction != "none" and any(np.isfinite(pvals)):
        try:
            from statsmodels.stats.multitest import multipletests
            method = {"bonferroni": "bonferroni", "holm": "holm", "bh": "fdr_bh"}.get(correction, "bonferroni")
            finite_idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
            _, padj, _, _ = multipletests([pvals[i] for i in finite_idx], method=method)
            for j, i in enumerate(finite_idx):
                comparisons[i]["p_adj"] = _safe_float(padj[j])
        except Exception:
            logger.exception("Pairwise multiplicity correction failed")
    return {"correction": correction, "comparisons": comparisons}


def _km_logrank(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str) -> Optional[dict]:
    groups = _sorted_groups(df[group_col])
    if len(groups) < 2:
        return None
    try:
        if len(groups) == 2:
            g0 = df[df[group_col] == groups[0]]
            g1 = df[df[group_col] == groups[1]]
            lr = logrank_test(
                g0[duration_col], g1[duration_col],
                event_observed_A=g0[event_col].astype(int),
                event_observed_B=g1[event_col].astype(int),
            )
            # χ² and df travel with p: journals cite "log-rank χ²(1) = 0.70,
            # p = 0.40", and p alone cannot be checked for consistency.
            return {
                "test": "Log-rank",
                "chi2": _safe_float(lr.test_statistic),
                "df": 1,
                "p": _safe_float(lr.p_value),
            }
        else:
            lr = multivariate_logrank_test(df[duration_col], df[group_col], df[event_col].astype(int))
            return {
                "test": "Log-rank (multivariate)",
                "chi2": _safe_float(lr.test_statistic),
                "df": len(groups) - 1,
                "p": _safe_float(lr.p_value),
            }
    except Exception:
        logger.exception("KM logrank test failed")
        return None


def _compute_vif(X: pd.DataFrame) -> dict:
    """Delegates to the one correct implementation.

    This file used to carry its own copy, which dropped the intercept before
    calling statsmodels and therefore reported inflated VIFs — four routers
    each held the same copy of the same mistake, while /api/diagnostics
    computed it correctly, so the same quantity had two answers depending on
    which screen you were on.
    """
    from services.regression import compute_vif

    return compute_vif(X)


# ── Kaplan-Meier ───────────────────────────────────────────────────────────────

class KMRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    group_col: Optional[str] = None
    stratify_col: Optional[str] = None
    imputation: Optional[str] = "listwise"
    # Landmark survival probabilities at these time points (Duration unit),
    # e.g. [1825] for 5-year survival on a days-coded column.
    survival_times: Optional[List[float]] = None
    # Pairwise log-rank comparisons (≥3 groups) + multiplicity correction.
    pairwise: bool = False
    pairwise_correction: str = "none"  # none | bonferroni | holm | bh
    # Number-at-risk table: subjects still at risk per group at these times
    # (typically the x-axis ticks), for the journal-style risk table.
    risk_times: Optional[List[float]] = None
    # Censoring tick marks per group (time + survival value at that time).
    include_censors: bool = False


@router.post("/survival/km")
def kaplan_meier(req: KMRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col] = pd.to_numeric(df_full[req.event_col], errors="coerce")

    km_cols = [req.duration_col, req.event_col]
    df = apply_imputation(df_full, km_cols, req.imputation)
    df, survival_warnings, n_invalid_survival = _drop_invalid_survival_rows(df, req.duration_col, req.event_col)
    n_excluded = n_total - len(df)

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after coercing duration/event columns to numeric. Check that both columns contain numbers.")

    event_vals = sorted(df[req.event_col].dropna().unique())
    if set(event_vals) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1 (0=censored, 1=event). Found: {event_vals[:10]}")

    if req.stratify_col:
        if req.stratify_col not in df_full.columns:
            raise HTTPException(status_code=422, detail=f"Stratify column '{req.stratify_col}' not found.")
        strata_vals = sorted(df[req.stratify_col].dropna().unique(), key=lambda x: (isinstance(x, str), x))
        strata_out = []
        for sv in strata_vals:
            sub = df[df[req.stratify_col] == sv].copy()
            if len(sub) == 0:
                continue
            grp_results = _km_fit_groups(sub, req.duration_col, req.event_col, req.group_col, req.survival_times, req.risk_times, req.include_censors)
            lr = _km_logrank(sub, req.duration_col, req.event_col, req.group_col) if req.group_col else None
            strata_out.append({"label": str(sv), "n": int(len(sub)), "groups": grp_results, "logrank": lr})
        return _add_survival_warnings({
            "model": "Kaplan-Meier",
            "strata": strata_out,
            "stratify_col": req.stratify_col,
            "n_total": n_total,
            "n_excluded": n_excluded,
            "imputation": req.imputation,
        }, survival_warnings, n_invalid_survival)

    results = _km_fit_groups(df, req.duration_col, req.event_col, req.group_col, req.survival_times, req.risk_times, req.include_censors)
    logrank = _km_logrank(df, req.duration_col, req.event_col, req.group_col) if req.group_col else None
    pairwise = (
        _km_pairwise(df, req.duration_col, req.event_col, req.group_col, req.pairwise_correction)
        if (req.pairwise and req.group_col) else None
    )

    return _add_survival_warnings({
        "model": "Kaplan-Meier",
        "groups": results,
        "logrank": logrank,
        "pairwise": pairwise,
        "survival_times": req.survival_times or [],
        "risk_times": req.risk_times or [],
        "median_follow_up": _median_follow_up(df[req.duration_col], df[req.event_col]),
        "n_total": n_total,
        "n_excluded": n_excluded,
        "imputation": req.imputation,
    }, survival_warnings, n_invalid_survival)


# ── Cox Proportional Hazards ───────────────────────────────────────────────────

class CoxRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"
    interactions: Optional[List[List[str]]] = None


@router.post("/survival/cox")
async def cox_regression(req: CoxRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col] = pd.to_numeric(df_full[req.event_col], errors="coerce")

    cox_cols = [req.duration_col, req.event_col] + req.predictors
    imputation_method = req.imputation or "listwise"
    use_mice_pooled = False

    if imputation_method == "mice":
        imp_result = mice_multiple(df_full, cox_cols, n_imputations=5)
        imputed_dfs = imp_result.imputed_datasets

        individual_results = []
        first_validation = None
        for df_imp in imputed_dfs:
            try:
                val_df, val_warnings, val_invalid = _drop_invalid_survival_rows(df_imp, req.duration_col, req.event_col)
                if first_validation is None:
                    first_validation = (val_df, val_warnings, val_invalid)
                cph_imp = CoxPHFitter()
                await asyncio.to_thread(cph_imp.fit, val_df[cox_cols], duration_col=req.duration_col, event_col=req.event_col)
                loghrs = {var: float(np.log(cph_imp.hazard_ratios_.get(var, 1.0))) for var in req.predictors}
                individual_results.append({"coefficients": loghrs})
            except Exception:
                logger.exception("Cox fit failed for one imputation step")
                continue

        pooled = pool_cox_results(individual_results) if individual_results else {}

        if first_validation is not None:
            df, survival_warnings, n_invalid_survival = first_validation
        else:
            df, survival_warnings, n_invalid_survival = _drop_invalid_survival_rows(imputed_dfs[0], req.duration_col, req.event_col)
        n_excluded = n_total - len(df)
        use_mice_pooled = True
    else:
        df = apply_imputation(df_full, cox_cols, imputation_method)
        df, survival_warnings, n_invalid_survival = _drop_invalid_survival_rows(df, req.duration_col, req.event_col)
        n_excluded = n_total - len(df)
        use_mice_pooled = False
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after coercing duration/event columns to numeric.")

    event_vals = sorted(df[req.event_col].dropna().unique())
    if set(event_vals) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1. Found: {event_vals[:10]}")

    pred_raw = df[req.predictors].copy()
    numeric_pred: list[str] = []
    cat_pred: list[str] = []
    for c in req.predictors:
        col = pred_raw[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric_pred.append(c)
        else:
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().mean() >= 0.8 and len(coerced.dropna().unique()) > 2:
                pred_raw[c] = coerced
                numeric_pred.append(c)
            else:
                cat_pred.append(c)

    num_part = pred_raw[numeric_pred].apply(pd.to_numeric, errors="coerce") if numeric_pred else pd.DataFrame(index=pred_raw.index)
    cat_part = pd.get_dummies(pred_raw[cat_pred], drop_first=True, dummy_na=False) if cat_pred else pd.DataFrame(index=pred_raw.index)
    enc = pd.concat([num_part, cat_part], axis=1).astype(float)

    interaction_cols: list[str] = []
    if req.interactions:
        def _members(name: str) -> list[str]:
            if name in enc.columns:
                return [name]
            prefix = f"{name}_"
            return [c for c in enc.columns if c.startswith(prefix)]

        for pair in req.interactions:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise HTTPException(status_code=422, detail=f"Each interaction must be a [colA, colB] pair. Got: {pair}")
            a_members = _members(pair[0])
            b_members = _members(pair[1])
            if not a_members or not b_members:
                raise HTTPException(status_code=422, detail=f"Interaction '{pair[0]} × {pair[1]}': one or both columns are not in the predictor list.")
            for a in a_members:
                for b in b_members:
                    new_col = f"{a}:{b}"
                    enc[new_col] = enc[a] * enc[b]
                    interaction_cols.append(new_col)

    fit_df = pd.concat([df[[req.duration_col, req.event_col]], enc], axis=1).dropna()
    if len(fit_df) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows after encoding (need ≥ 10, got {len(fit_df)}).")

    cph = CoxPHFitter()
    try:
        await asyncio.to_thread(cph.fit, fit_df, duration_col=req.duration_col, event_col=req.event_col)
    except Exception as exc:
        logger.exception("Cox fitting failed")
        raise HTTPException(status_code=400, detail=f"Cox fitting error: {exc}")

    summary = cph.summary.reset_index()
    vifs = _compute_vif(enc)
    coefs = []
    for _, row in summary.iterrows():
        name = str(row["covariate"])
        coefs.append({
            "variable": name,
            "log_hr": _safe_float(row["coef"]),
            "hr": _safe_float(row["exp(coef)"]),
            "se": _safe_float(row["se(coef)"]),
            "z": _safe_float(row["z"]),
            "p": _safe_float(row["p"]),
            "hr_ci_low": _safe_float(row["exp(coef) lower 95%"]),
            "hr_ci_high": _safe_float(row["exp(coef) upper 95%"]),
            "vif": vifs.get(name),
        })

    ph_test = None
    try:
        from lifelines.statistics import proportional_hazard_test
        ph_res = proportional_hazard_test(cph, fit_df, time_transform="rank")
        ph_summary = ph_res.summary.reset_index() if hasattr(ph_res.summary, "reset_index") else ph_res.summary
        per_term = []
        for _, row in ph_summary.iterrows():
            per_term.append({
                "variable": str(row.get("index", row.get("covariate", ""))),
                "test_stat": _safe_float(row.get("test_statistic")),
                "p": _safe_float(row.get("p")),
            })
        from scipy.stats import chi2 as _chi2
        chi_vals = [t["test_stat"] for t in per_term if t["test_stat"] is not None]
        if chi_vals:
            global_chi = float(sum(chi_vals))
            global_df = len(chi_vals)
            global_p = float(1 - _chi2.cdf(global_chi, global_df)) if global_df > 0 else None
        else:
            global_chi, global_df, global_p = None, None, None
        ph_test = {"global": {"chi2": global_chi, "df": global_df, "p": global_p}, "per_term": per_term}
    except Exception as exc:
        logger.exception("Proportional hazards test failed")
        ph_test = {"error": str(exc)}

    result = {
        "model": "Cox Proportional Hazards",
        "n": int(cph.event_observed.sum()),
        "n_analyzed": int(len(fit_df)),
        "n_total": n_total,
        "n_excluded": n_excluded,
        "imputation": req.imputation,
        "log_likelihood": _safe_float(cph.log_likelihood_),
        "concordance": _safe_float(cph.concordance_index_),
        "coefficients": coefs,
        "interactions_used": interaction_cols,
        "ph_test": ph_test,
    }

    if use_mice_pooled and 'pooled' in locals() and pooled:
        result["coefficients"] = pooled.get("coefficients", result.get("coefficients", []))
        result["pooled_from_imputations"] = True
        result["imputation"] = "mice (pooled)"

    cox_assumption_report = check_cox_assumptions_from_ph_test(ph_test)
    result = add_assumption_warnings_to_result(result, cox_assumption_report)

    missing_info = missing_pattern_summary(df_full, cox_cols)
    result = add_missing_data_diagnostics(result, missing_info)

    return _add_survival_warnings(result, survival_warnings, n_invalid_survival)


# ── Cox time-horizon sensitivity (forest) ──────────────────────────────────────
# Runs the SAME Cox model at several administrative-censoring horizons
# (e.g. 1-year, 2-year, full follow-up) so the user can see how the hazard
# ratio for one predictor moves with the time window. Output is directly
# consumable by the Forest Builder.

class CoxHorizonsRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictor: str                              # variable whose HR is tracked
    covariates: Optional[List[str]] = None      # optional adjustment set
    horizons: List[float]                       # cut-points in the time unit (e.g. [365, 730])
    horizon_labels: Optional[List[str]] = None  # display labels, 1:1 with horizons
    include_full: bool = True                   # append an un-censored "Full follow-up" row
    full_label: str = "Full follow-up"
    imputation: Optional[str] = "listwise"


def _encode_predictors(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Numeric-vs-categorical split + dummy coding (reference = first level).

    Mirrors the encoding used by the main Cox endpoint so the horizon
    model is identical to a single-window fit at full follow-up.
    """
    raw = df[cols].copy()
    numeric_pred: List[str] = []
    cat_pred: List[str] = []
    for c in cols:
        col = raw[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric_pred.append(c)
        else:
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().mean() >= 0.8 and len(coerced.dropna().unique()) > 2:
                raw[c] = coerced
                numeric_pred.append(c)
            else:
                cat_pred.append(c)
    num_part = raw[numeric_pred].apply(pd.to_numeric, errors="coerce") if numeric_pred else pd.DataFrame(index=raw.index)
    cat_part = pd.get_dummies(raw[cat_pred], drop_first=True, dummy_na=False) if cat_pred else pd.DataFrame(index=raw.index)
    enc = pd.concat([num_part, cat_part], axis=1).astype(float)
    return enc, numeric_pred, cat_pred


@router.post("/survival/cox_horizons")
async def cox_horizons(req: CoxHorizonsRequest):
    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    # Validate columns up-front.
    cov = req.covariates or []
    needed = [req.duration_col, req.event_col, req.predictor] + cov
    missing = [c for c in needed if c not in df_full.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing}")
    if not req.horizons:
        raise HTTPException(status_code=422, detail="Provide at least one horizon cut-point.")
    if req.horizon_labels is not None and len(req.horizon_labels) != len(req.horizons):
        raise HTTPException(status_code=422, detail="horizon_labels must match horizons length.")

    # Coerce + impute the full predictor set once; per-horizon we only
    # recompute event/time, never re-impute, so every window uses the
    # identical analysis sample.
    df_full = df_full.copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col] = pd.to_numeric(df_full[req.event_col], errors="coerce")
    work_cols = [req.duration_col, req.event_col, req.predictor] + cov
    df = apply_imputation(df_full, work_cols, req.imputation or "listwise")
    df, survival_warnings, n_invalid_survival = _drop_invalid_survival_rows(df, req.duration_col, req.event_col)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after coercing/imputing.")

    event_vals = sorted(df[req.event_col].dropna().unique())
    if set(event_vals) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1. Found: {event_vals[:10]}")

    # Encode predictors once — the encoded design is reused across windows.
    enc, _num, _cat = _encode_predictors(df, [req.predictor] + cov)
    # Which encoded columns belong to the tracked predictor? (a binary/
    # numeric predictor → one column; a multi-level categorical → several).
    if req.predictor in enc.columns:
        pred_terms = [req.predictor]
    else:
        prefix = f"{req.predictor}_"
        pred_terms = [c for c in enc.columns if c.startswith(prefix)]
    if not pred_terms:
        raise HTTPException(status_code=400, detail=f"Predictor '{req.predictor}' produced no usable terms after encoding.")

    base_time = df[req.duration_col].astype(float)
    base_event = df[req.event_col].astype(float)

    # Build the window list: each requested horizon, then optionally full.
    windows: List[Tuple[str, Optional[float]]] = []
    labels = req.horizon_labels or [f"≤ {h:g}" for h in req.horizons]
    for lab, h in zip(labels, req.horizons):
        windows.append((lab, float(h)))
    if req.include_full:
        windows.append((req.full_label, None))

    horizon_results: List[dict] = []
    forest_rows: List[dict] = []

    for label, tau in windows:
        if tau is None:
            t_h = base_time
            e_h = base_event
        else:
            # Administrative censoring at tau: anyone whose event/censor is
            # after tau is censored AT tau; events at/before tau are kept.
            t_h = base_time.clip(upper=tau)
            e_h = base_event.where(base_time <= tau, other=0.0)

        fit_df = pd.concat(
            [t_h.rename(req.duration_col), e_h.rename(req.event_col), enc],
            axis=1,
        ).dropna()
        n_events = int(fit_df[req.event_col].sum())
        n_win = int(len(fit_df))

        # Need enough events to fit; skip windows that are too sparse but
        # report why so the forest doesn't silently drop rows.
        if n_win < 10 or n_events < 3:
            horizon_results.append({
                "label": label, "tau": tau, "n": n_win, "n_events": n_events,
                "terms": [], "skipped": "too few events at this horizon",
            })
            continue

        cph = CoxPHFitter()
        try:
            await asyncio.to_thread(cph.fit, fit_df, duration_col=req.duration_col, event_col=req.event_col)
        except Exception as exc:
            logger.exception("Cox horizon fit failed at tau=%s", tau)
            horizon_results.append({
                "label": label, "tau": tau, "n": n_win, "n_events": n_events,
                "terms": [], "skipped": f"fit error: {exc}",
            })
            continue

        summary = cph.summary
        terms = []
        for term in pred_terms:
            if term not in summary.index:
                continue
            row = summary.loc[term]
            terms.append({
                "variable": term,
                "hr": _safe_float(row["exp(coef)"]),
                "hr_ci_low": _safe_float(row["exp(coef) lower 95%"]),
                "hr_ci_high": _safe_float(row["exp(coef) upper 95%"]),
                "p": _safe_float(row["p"]),
            })

        horizon_results.append({
            "label": label, "tau": tau, "n": n_win, "n_events": n_events, "terms": terms,
        })

        # Forest convenience rows — one per predictor term. When there's a
        # single tracked term (the common binary case) the label is just
        # the horizon; with multi-level predictors we disambiguate.
        for t in terms:
            row_label = label if len(pred_terms) == 1 else f"{label} · {t['variable']}"
            forest_rows.append({
                "label": row_label,
                "est": t["hr"],
                "ci_low": t["hr_ci_low"],
                "ci_high": t["hr_ci_high"],
                "p": t["p"],
                "extra": f"({horizon_results[-1]['n_events']} events)",
            })

    if not forest_rows:
        raise HTTPException(status_code=400, detail="No horizon produced an estimable HR (too few events in every window).")

    cov_txt = (" + ".join(cov)) if cov else "none (unadjusted)"
    horizons_txt = ", ".join(f"{lab} (tau={tau})" for lab, tau in windows)
    interpretation = (
        f"Cox PH hazard ratio for '{req.predictor}' across {len(windows)} time horizons "
        f"[{horizons_txt}], adjustment: {cov_txt}. Each window applies administrative "
        f"censoring at its cut-point; the full-follow-up row uses all events. Widening "
        f"confidence intervals at short horizons reflect fewer accrued events."
    )

    r_code = (
        "library(survival)\n"
        "horizons <- c(" + ", ".join(f"{h:g}" for h in req.horizons) + ")\n"
        "rows <- lapply(horizons, function(tau) {\n"
        f"  d <- dat; d$ev <- ifelse(d${req.duration_col} <= tau & d${req.event_col}==1, 1, 0)\n"
        f"  d$t <- pmin(d${req.duration_col}, tau)\n"
        f"  fit <- coxph(Surv(t, ev) ~ {req.predictor}"
        + ("".join(f' + {c}' for c in cov)) + ", data = d)\n"
        "  s <- summary(fit); c(HR=s$conf.int[1,1], lo=s$conf.int[1,3], hi=s$conf.int[1,4])\n"
        "})\n"
        f"# Full follow-up: coxph(Surv({req.duration_col}, {req.event_col}) ~ {req.predictor}"
        + ("".join(f' + {c}' for c in cov)) + ", data = dat)\n"
    )

    return _add_survival_warnings({
        "model": "Cox PH — time-horizon sensitivity",
        "predictor": req.predictor,
        "covariates": cov,
        "n_total": n_total,
        "n_analyzed": int(len(df)),
        "n_excluded": int(n_total - len(df)),
        "horizons": horizon_results,
        "forest_rows": forest_rows,
        "interpretation": interpretation,
        "r_code": r_code,
    }, survival_warnings, n_invalid_survival)


# ── Cox with time-varying covariates ───────────────────────────────────────────

class CoxTVRequest(BaseModel):
    session_id: str
    id_col: str
    start_col: str
    stop_col: str
    event_col: str
    predictors: List[str]
    imputation: Optional[str] = "listwise"


@router.post("/survival/cox_tv")
async def cox_time_varying(req: CoxTVRequest):
    from lifelines import CoxTimeVaryingFitter

    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    cols = [req.id_col, req.start_col, req.stop_col, req.event_col] + req.predictors
    missing = [c for c in cols if c not in df_full.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing}")
    df = apply_imputation(df_full, cols, req.imputation or "listwise")

    for c in [req.start_col, req.stop_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[req.event_col] = pd.to_numeric(df[req.event_col], errors="coerce")
    df = df.dropna(subset=[req.start_col, req.stop_col, req.event_col, req.id_col])
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Need ≥10 valid (subject, interval) rows.")
    if (df[req.stop_col] <= df[req.start_col]).any():
        raise HTTPException(status_code=422, detail="Found rows with stop ≤ start; each interval must have stop > start.")
    if set(df[req.event_col].unique()) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail="Event column must be 0/1.")
    n_excluded = n_total - len(df)

    enc = pd.get_dummies(df[req.predictors], drop_first=True).astype(float)
    if enc.shape[1] == 0:
        raise HTTPException(status_code=422, detail="No usable predictor columns after encoding.")
    fit_df = pd.concat([df[[req.id_col, req.start_col, req.stop_col, req.event_col]], enc], axis=1)

    ctv = CoxTimeVaryingFitter()
    try:
        await asyncio.to_thread(ctv.fit, fit_df, id_col=req.id_col, start_col=req.start_col, stop_col=req.stop_col, event_col=req.event_col)
    except Exception as exc:
        logger.exception("Cox-TV fitting failed")
        raise HTTPException(status_code=400, detail=f"Cox-TV fitting failed: {exc}")

    summary = ctv.summary.reset_index()
    vifs = _compute_vif(enc)
    coefs = []
    for _, row in summary.iterrows():
        name = str(row["covariate"])
        coefs.append({
            "variable": name,
            "log_hr": _safe_float(row["coef"]),
            "hr": _safe_float(row["exp(coef)"]),
            "se": _safe_float(row["se(coef)"]),
            "z": _safe_float(row["z"]),
            "p": _safe_float(row["p"]),
            "hr_ci_low": _safe_float(row["exp(coef) lower 95%"]),
            "hr_ci_high": _safe_float(row["exp(coef) upper 95%"]),
            "vif": vifs.get(name),
        })

    n_subjects = int(df[req.id_col].nunique())
    n_events = int(df[req.event_col].sum())
    return {
        "model": "Cox Proportional Hazards (time-varying covariates)",
        "n_intervals": int(len(df)),
        "n_subjects": n_subjects,
        "n_events": n_events,
        "n_total": int(n_total),
        "n_excluded": int(n_excluded),
        "imputation": req.imputation or "listwise",
        "log_likelihood": _safe_float(ctv.log_likelihood_),
        "concordance": _safe_float(getattr(ctv, "concordance_index_", None)),
        "coefficients": coefs,
        "result_text": (
            f"Cox regression with time-varying covariates on {n_subjects} subjects "
            f"({len(df)} interval rows, {n_events} events; {n_excluded} excluded). "
            f"Predictors: {', '.join(req.predictors)}."
        ),
    }


# ── Restricted Cubic Splines (RCS) ───────────────────────────────────────────

class RCSRequest(BaseModel):
    session_id: str
    predictor: str
    outcome: Optional[str] = None
    covariates: List[str] = []
    n_knots: int = 4
    ref_value: Optional[float] = None
    model_type: str = "logistic"
    imputation: str = "listwise"
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    knot_positions: Optional[List[float]] = None
    interaction_covariates: Optional[List[str]] = None


@router.post("/rcs")
async def rcs_regression(req: RCSRequest):
    import statsmodels.api as sm_api
    if req.n_knots not in _KNOT_PERCENTILES:
        raise HTTPException(status_code=422, detail=f"n_knots must be 3, 4, or 5. Got: {req.n_knots}")

    model_type = (req.model_type or "logistic").lower()
    if model_type not in ("logistic", "linear", "cox"):
        raise HTTPException(status_code=422, detail=f"Unknown model_type: {req.model_type}")

    is_cox = model_type == "cox"

    if is_cox:
        if not req.duration_col or not req.event_col:
            raise HTTPException(status_code=422, detail="duration_col and event_col are required when model_type='cox'.")
        cols_needed = [req.predictor, req.duration_col, req.event_col] + req.covariates
    else:
        if not req.outcome:
            raise HTTPException(status_code=422, detail="outcome is required when model_type is 'logistic' or 'linear'.")
        cols_needed = [req.predictor, req.outcome] + req.covariates

    df_full = _get_df(req.session_id)
    missing_cols = [c for c in cols_needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found in session: {missing_cols}")

    df = df_full[cols_needed].copy()
    n_total = len(df)

    required_numeric = [req.predictor]
    if is_cox:
        required_numeric += [req.duration_col, req.event_col]
    else:
        required_numeric += [req.outcome]
    for c in required_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cov_df = None
    if req.covariates:
        cov_raw = df[req.covariates].copy()
        numeric_cov: list[str] = []
        cat_cov: list[str] = []
        for c in req.covariates:
            col = cov_raw[c]
            if pd.api.types.is_numeric_dtype(col):
                numeric_cov.append(c)
            else:
                coerced = pd.to_numeric(col, errors="coerce")
                if coerced.notna().mean() >= 0.8 and len(coerced.dropna().unique()) > 2:
                    cov_raw[c] = coerced
                    numeric_cov.append(c)
                else:
                    cat_cov.append(c)
        num_part = cov_raw[numeric_cov].apply(pd.to_numeric, errors="coerce") if numeric_cov else pd.DataFrame(index=cov_raw.index)
        cat_part = pd.get_dummies(cov_raw[cat_cov], drop_first=True, dummy_na=False) if cat_cov else pd.DataFrame(index=cov_raw.index)
        cov_df = pd.concat([num_part, cat_part], axis=1).astype(float)
        df = pd.concat([df.drop(columns=req.covariates), cov_df], axis=1)
    df = df.dropna()
    n = len(df)
    if n < 10:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows (need ≥ 10). Got {n} after dropping rows with missing predictor / outcome / covariates.")

    x_raw = df[req.predictor].values.astype(float)

    n_unique_x = len(np.unique(x_raw))
    if n_unique_x < req.n_knots + 2:
        raise HTTPException(status_code=422, detail=f"Predictor '{req.predictor}' has only {n_unique_x} unique values — need ≥ {req.n_knots + 2} for {req.n_knots}-knot spline.")

    if is_cox:
        duration = df[req.duration_col].values.astype(float)
        event = df[req.event_col].values.astype(float)
        if np.any(duration < 0):
            raise HTTPException(status_code=422, detail=f"duration_col '{req.duration_col}' must be ≥ 0.")
        unique_e = sorted(set(event.tolist()))
        if set(unique_e) - {0.0, 1.0}:
            raise HTTPException(status_code=422, detail=f"event_col '{req.event_col}' must be binary 0/1.")
    else:
        y = df[req.outcome].values.astype(float)
        if model_type == "logistic":
            unique_y = sorted(set(y.tolist()))
            if set(unique_y) - {0.0, 1.0}:
                raise HTTPException(status_code=422, detail="Logistic RCS requires binary 0/1 outcome.")

    try:
        knots = _resolve_knots(x_raw, req.n_knots, req.knot_positions, req.predictor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    spline_cols = _rcs_basis(x_raw, knots)

    if cov_df is not None and cov_df.shape[1] > 0:
        cov_names = list(cov_df.columns)
        cov_mat = df[cov_names].values.astype(float)
    else:
        cov_names = []
        cov_mat = None

    interaction_cov_names: list[str] = []
    interaction_extra_names: list[str] = []
    interaction_extra: list[np.ndarray] = []
    interaction_extra_meta: list[tuple[str, int]] = []
    if req.interaction_covariates:
        def _resolve_cov(name: str) -> list[str]:
            if name in cov_names:
                return [name]
            prefix = f"{name}_"
            return [c for c in cov_names if c.startswith(prefix)]

        spline_design = np.column_stack([x_raw, spline_cols])
        spline_part_names = ["lin"] + [f"sp{i+1}" for i in range(spline_cols.shape[1])]
        for cov in req.interaction_covariates:
            members = _resolve_cov(cov)
            if not members:
                raise HTTPException(status_code=422, detail=f"interaction_covariates entry '{cov}' is not in the selected covariates list.")
            interaction_cov_names.append(cov)
            for m in members:
                cov_vec = df[m].values.astype(float)
                for i in range(spline_design.shape[1]):
                    interaction_extra.append(spline_design[:, i] * cov_vec)
                    interaction_extra_names.append(f"{m}:{req.predictor}_{spline_part_names[i]}")
                    interaction_extra_meta.append((m, i))

    if interaction_extra:
        interaction_mat = np.column_stack(interaction_extra)
    else:
        interaction_mat = None

    try:
        if is_cox:
            feat_cols = ["_x_lin"] + [f"_spl_{i}" for i in range(spline_cols.shape[1])]
            fit_df = pd.DataFrame(
                np.column_stack([x_raw, spline_cols]),
                columns=feat_cols,
                index=df.index,
            )
            for c in cov_names:
                fit_df[c] = df[c].values
            for i, ix_name in enumerate(interaction_extra_names):
                fit_df[ix_name] = interaction_extra[i]
            fit_df["_dur_"] = duration
            fit_df["_evt_"] = event
            cph = CoxPHFitter()
            await asyncio.to_thread(cph.fit, fit_df, duration_col="_dur_", event_col="_evt_")
            design_cols = feat_cols + cov_names + interaction_extra_names
            params = cph.params_.reindex(design_cols).values
            cov_params = cph.variance_matrix_.reindex(index=design_cols, columns=design_cols).values
            aic_val = None
            try:
                aic_val = float(getattr(cph, "AIC_partial_", np.nan))
                if np.isnan(aic_val):
                    aic_val = None
            except Exception:
                aic_val = None
            log_lik = float(cph.log_likelihood_)
            concordance = float(cph.concordance_index_)
            n_events = int(np.sum(event))
        else:
            X_parts = [np.ones(n), x_raw, spline_cols]
            if cov_mat is not None:
                X_parts.append(cov_mat)
            if interaction_mat is not None:
                X_parts.append(interaction_mat)
            X = np.column_stack(X_parts)
            if model_type == "logistic":
                result = await asyncio.to_thread(lambda: sm_api.Logit(y, X).fit(disp=0, maxiter=200))
            else:
                result = await asyncio.to_thread(lambda: sm_api.OLS(y, X).fit())
            params = result.params
            cov_params = result.cov_params()
            try:
                aic_val = float(result.aic)
                if np.isnan(aic_val) or np.isinf(aic_val):
                    aic_val = None
            except Exception:
                aic_val = None
            log_lik = float(getattr(result, "llf", np.nan))
            if np.isnan(log_lik) or np.isinf(log_lik):
                log_lik = None
            concordance = None
            n_events = int(y.sum()) if model_type == "logistic" else None
    except Exception as exc:
        logger.exception("RCS Model fitting failed")
        raise HTTPException(status_code=400, detail=f"Model fitting error: {exc}")

    interaction_result = None
    if interaction_extra_names:
        try:
            if is_cox:
                reduced_df = fit_df.drop(columns=interaction_extra_names)
                cph_red = CoxPHFitter()
                await asyncio.to_thread(cph_red.fit, reduced_df, duration_col="_dur_", event_col="_evt_")
                ll_red = float(cph_red.log_likelihood_)
                ll_full = float(log_lik)
            else:
                X_red_parts = [np.ones(n), x_raw, spline_cols]
                if cov_mat is not None:
                    X_red_parts.append(cov_mat)
                X_red = np.column_stack(X_red_parts)
                if model_type == "logistic":
                    res_red = await asyncio.to_thread(lambda: sm_api.Logit(y, X_red).fit(disp=0, maxiter=200))
                else:
                    res_red = await asyncio.to_thread(lambda: sm_api.OLS(y, X_red).fit())
                ll_red = float(getattr(res_red, "llf", np.nan))
                ll_full = float(log_lik) if log_lik is not None else float(getattr(result, "llf", np.nan))
            lr_stat = 2.0 * (ll_full - ll_red)
            df_lr = len(interaction_extra_names)
            p_lr = float(chi2.sf(lr_stat, df=df_lr))
            interaction_result = {
                "covariates": interaction_cov_names,
                "lr_stat": round(lr_stat, 4),
                "df": df_lr,
                "p": round(p_lr, 6),
                "log_lik_full": round(ll_full, 4),
                "log_lik_reduced": round(ll_red, 4),
            }
        except Exception as exc:
            logger.exception("RCS Interaction LR test failed")
            interaction_result = {"covariates": interaction_cov_names, "error": str(exc)}

    x_lo, x_hi = float(np.percentile(x_raw, 1)), float(np.percentile(x_raw, 99))
    x_syn = np.linspace(x_lo, x_hi, 200)
    sp_syn = _rcs_basis(x_syn, knots)

    cov_means_by_name: dict[str, float] = {}
    if cov_names:
        cov_means = cov_mat.mean(axis=0)
        for cn, mean_val in zip(cov_names, cov_means):
            cov_means_by_name[cn] = float(mean_val)

    spline_design_syn = np.column_stack([x_syn, sp_syn])

    if is_cox:
        if cov_mat is not None:
            X_syn = np.column_stack([x_syn, sp_syn, np.tile(cov_means, (200, 1))])
        else:
            X_syn = np.column_stack([x_syn, sp_syn])
    else:
        if cov_mat is not None:
            X_syn = np.column_stack([np.ones(200), x_syn, sp_syn, np.tile(cov_means, (200, 1))])
        else:
            X_syn = np.column_stack([np.ones(200), x_syn, sp_syn])

    if interaction_extra_meta:
        ix_syn = np.column_stack([
            spline_design_syn[:, spi] * cov_means_by_name.get(member, 0.0)
            for (member, spi) in interaction_extra_meta
        ])
        X_syn = np.column_stack([X_syn, ix_syn])

    lp_syn = X_syn @ params

    ref_val = req.ref_value if req.ref_value is not None else float(np.median(x_raw))
    ref_val = float(np.clip(ref_val, x_lo, x_hi))
    ref_idx = int(np.argmin(np.abs(x_syn - ref_val)))
    lp_ref = lp_syn[ref_idx]
    rel_lp = lp_syn - lp_ref

    diffs = X_syn - X_syn[ref_idx]
    var_lp = np.einsum("ij,jk,ik->i", diffs, cov_params, diffs)
    se_lp = np.sqrt(np.maximum(var_lp, 0))
    z95 = 1.96

    if is_cox or model_type == "logistic":
        or_vals = np.exp(rel_lp)
        ci_low = np.exp(rel_lp - z95 * se_lp)
        ci_high = np.exp(rel_lp + z95 * se_lp)
    else:
        or_vals = rel_lp
        ci_low = rel_lp - z95 * se_lp
        ci_high = rel_lp + z95 * se_lp

    effect_type = "HR" if is_cox else ("OR" if model_type == "logistic" else "mean_diff")

    cov_summary = []
    if cov_names:
        n_pre_cov = (0 if is_cox else 2) + (1 + spline_cols.shape[1])
        for offset, name in enumerate(cov_names):
            i = n_pre_cov + offset
            beta = float(params[i]) if i < len(params) else None
            se = float(np.sqrt(max(cov_params[i, i], 0.0))) if i < len(params) else None
            cov_summary.append({
                "name": name,
                "coef": round(beta, 6) if beta is not None else None,
                "effect": round(float(np.exp(beta)), 4) if (is_cox or model_type == "logistic") and beta is not None else (round(beta, 4) if beta is not None else None),
                "se": round(se, 6) if se is not None else None,
            })

    nonlin_p = None
    nonlin_wald = None
    nonlin_df = None
    try:
        lin_idx = 0 if is_cox else 1
        nl_start = lin_idx + 1
        nl_end = nl_start + spline_cols.shape[1]
        if nl_end > nl_start:
            idx = np.arange(nl_start, nl_end)
            beta_nl = np.asarray(params)[idx]
            cov_nl = np.asarray(cov_params)[np.ix_(idx, idx)]
            wald = float(beta_nl @ np.linalg.solve(cov_nl, beta_nl))
            df_nl = int(len(idx))
            nonlin_wald = round(wald, 4)
            nonlin_df = df_nl
            nonlin_p = round(float(chi2.sf(wald, df=df_nl)), 6)
    except Exception:
        logger.exception("Nonlinearity test failed")

    crude_block = None
    if cov_names or interaction_extra_names:
        try:
            if is_cox:
                feat_cols_c = ["_x_lin"] + [f"_spl_{i}" for i in range(spline_cols.shape[1])]
                fit_df_c = pd.DataFrame(
                    np.column_stack([x_raw, spline_cols]),
                    columns=feat_cols_c,
                    index=df.index,
                )
                fit_df_c["_dur_"] = duration
                fit_df_c["_evt_"] = event
                cph_c = CoxPHFitter()
                await asyncio.to_thread(cph_c.fit, fit_df_c, duration_col="_dur_", event_col="_evt_")
                design_cols_c = feat_cols_c
                params_c = cph_c.params_.reindex(design_cols_c).values
                cov_params_c = cph_c.variance_matrix_.reindex(index=design_cols_c, columns=design_cols_c).values
                X_syn_c = np.column_stack([x_syn, sp_syn])
            else:
                X_parts_c = [np.ones(n), x_raw, spline_cols]
                X_c = np.column_stack(X_parts_c)
                if model_type == "logistic":
                    res_c = await asyncio.to_thread(lambda: sm_api.Logit(y, X_c).fit(disp=0, maxiter=200))
                else:
                    res_c = await asyncio.to_thread(lambda: sm_api.OLS(y, X_c).fit())
                params_c = res_c.params
                cov_params_c = res_c.cov_params()
                X_syn_c = np.column_stack([np.ones(200), x_syn, sp_syn])

            lp_syn_c = X_syn_c @ params_c
            ref_idx_c = int(np.argmin(np.abs(x_syn - ref_val)))
            rel_lp_c = lp_syn_c - lp_syn_c[ref_idx_c]
            diffs_c = X_syn_c - X_syn_c[ref_idx_c]
            var_lp_c = np.einsum("ij,jk,ik->i", diffs_c, cov_params_c, diffs_c)
            se_lp_c = np.sqrt(np.maximum(var_lp_c, 0))
            if is_cox or model_type == "logistic":
                or_vals_c = np.exp(rel_lp_c)
                ci_low_c = np.exp(rel_lp_c - z95 * se_lp_c)
                ci_high_c = np.exp(rel_lp_c + z95 * se_lp_c)
            else:
                or_vals_c = rel_lp_c
                ci_low_c = rel_lp_c - z95 * se_lp_c
                ci_high_c = rel_lp_c + z95 * se_lp_c

            crude_nl_p = None
            try:
                lin_idx_c = 0 if is_cox else 1
                nl_start_c = lin_idx_c + 1
                nl_end_c = nl_start_c + spline_cols.shape[1]
                idx_c = np.arange(nl_start_c, nl_end_c)
                beta_nl_c = np.asarray(params_c)[idx_c]
                cov_nl_c = np.asarray(cov_params_c)[np.ix_(idx_c, idx_c)]
                w_c = float(beta_nl_c @ np.linalg.solve(cov_nl_c, beta_nl_c))
                crude_nl_p = round(float(chi2.sf(w_c, df=int(len(idx_c)))), 6)
            except Exception:
                logger.exception("Crude RCS nonlinearity test failed")

            crude_block = {
                "x_values": _clean(x_syn),
                "or_values": _clean(or_vals_c),
                "ci_low": _clean(ci_low_c),
                "ci_high": _clean(ci_high_c),
                "nonlinearity_p": crude_nl_p,
            }
        except Exception:
            logger.exception("Crude RCS fit failed")
            crude_block = None

    return {
        "predictor": req.predictor,
        "outcome": req.outcome,
        "duration_col": req.duration_col,
        "event_col": req.event_col,
        "model_type": model_type,
        "effect_type": effect_type,
        "n": n,
        "n_total": n_total,
        "n_excluded": n_total - n,
        "n_events": n_events,
        "n_knots": req.n_knots,
        "knots": [round(float(kn), 2) for kn in knots],
        "knot_positions_custom": req.knot_positions is not None,
        "ref_value": round(ref_val, 4),
        "aic": _safe_float(aic_val),
        "log_likelihood": _safe_float(log_lik),
        "concordance": _safe_float(concordance),
        "covariates_requested": list(req.covariates or []),
        "covariates_used": cov_names,
        "covariates_summary": cov_summary,
        "interaction": interaction_result,
        "interaction_terms": interaction_extra_names,
        "nonlinearity_wald": nonlin_wald,
        "nonlinearity_df": nonlin_df,
        "nonlinearity_p": nonlin_p,
        "x_values": _clean(x_syn),
        "or_values": _clean(or_vals),
        "ci_low": _clean(ci_low),
        "ci_high": _clean(ci_high),
        "x_data": _clean(x_raw[:500]),
        "crude": crude_block,
    }


# ── Multivariable Cox-RCS Endpoint ─────────────────────────────────────────────

class SplineTerm(BaseModel):
    column: str
    n_knots: int = 4
    knot_positions: Optional[List[float]] = None
    ref_value: Optional[float] = None


class CoxRCSRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    spline_terms: List[SplineTerm]
    covariates: List[str] = []
    include_interaction: bool = False
    imputation: Optional[str] = "listwise"
    grid_size: int = 50


@router.post("/survival/cox_rcs")
async def cox_rcs(req: CoxRCSRequest):
    if not (1 <= len(req.spline_terms) <= 2):
        raise HTTPException(status_code=422, detail="spline_terms must contain 1 or 2 entries.")

    for term in req.spline_terms:
        if term.n_knots not in _KNOT_PERCENTILES:
            raise HTTPException(status_code=422, detail=f"n_knots for '{term.column}' must be 3, 4, or 5. Got: {term.n_knots}")

    if req.include_interaction and len(req.spline_terms) != 2:
        raise HTTPException(status_code=422, detail="include_interaction requires exactly 2 spline_terms.")

    df_full = _get_df(req.session_id)
    spline_cols = [t.column for t in req.spline_terms]
    cols_needed = list(dict.fromkeys(spline_cols + [req.duration_col, req.event_col] + req.covariates))
    missing_cols = [c for c in cols_needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found in session: {missing_cols}")

    df = df_full[cols_needed].copy()
    # Only the spline terms and the survival columns have to be numeric. The
    # covariates are dummy-coded further down, so coercing them here turned
    # every categorical value into NaN, dropped every row with it, and the
    # analysis then failed as "not enough complete rows" — pointing the user
    # at their data for what was this line's doing.
    numeric_cols = list(dict.fromkeys(spline_cols + [req.duration_col, req.event_col]))
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if req.imputation and req.imputation != "listwise":
        df = apply_imputation(df, cols_needed, req.imputation)
    else:
        df = df.dropna()
    df, survival_warnings, n_invalid_survival = _drop_invalid_survival_rows(df, req.duration_col, req.event_col)
    n = len(df)
    if n < 15:
        raise HTTPException(status_code=400, detail="Not enough complete rows (need ≥ 15).")

    duration = df[req.duration_col].values.astype(float)
    event = df[req.event_col].values.astype(float)
    if set(sorted(set(event.tolist()))) - {0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"event_col '{req.event_col}' must be binary 0/1.")
    if event.sum() < 5:
        raise HTTPException(status_code=400, detail="Need ≥ 5 events to fit a Cox model.")

    term_info = []
    for ti, term in enumerate(req.spline_terms):
        x_raw = df[term.column].values.astype(float)
        n_unique = len(np.unique(x_raw))
        if n_unique < term.n_knots + 2:
            raise HTTPException(status_code=422, detail=f"Spline term '{term.column}' has only {n_unique} unique values — need ≥ {term.n_knots + 2} for {term.n_knots}-knot spline.")
        try:
            knots = _resolve_knots(x_raw, term.n_knots, term.knot_positions, term.column)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        sp = _rcs_basis(x_raw, knots)
        lin_col = f"t{ti}_{term.column}_lin"
        sp_cols = [f"t{ti}_{term.column}_sp{i}" for i in range(sp.shape[1])]
        full_cols = [lin_col] + sp_cols
        term_info.append({
            "column": term.column,
            "knots": knots,
            "x_raw": x_raw,
            "design": np.column_stack([x_raw, sp]),
            "col_names": full_cols,
            "ref_value": term.ref_value if term.ref_value is not None else float(np.median(x_raw)),
        })

    feat_arrays = []
    feat_names: List[str] = []
    for ti in term_info:
        feat_arrays.append(ti["design"])
        feat_names.extend(ti["col_names"])

    cov_df = pd.DataFrame(index=df.index)
    if req.covariates:
        cov_raw = df_full.loc[df.index, req.covariates].copy()
        cov_df = pd.get_dummies(cov_raw, drop_first=True, dummy_na=False)
        for c in cov_df.columns:
            cov_df[c] = pd.to_numeric(cov_df[c], errors="coerce")
        cov_df = cov_df.dropna()
        df_aligned = df.loc[cov_df.index]
        duration = df_aligned[req.duration_col].values.astype(float)
        event = df_aligned[req.event_col].values.astype(float)
        for ti, term in enumerate(req.spline_terms):
            x_raw = df_aligned[term.column].values.astype(float)
            sp = _rcs_basis(x_raw, term_info[ti]["knots"])
            term_info[ti]["design"] = np.column_stack([x_raw, sp])
            term_info[ti]["x_raw"] = x_raw
        feat_arrays = [ti["design"] for ti in term_info]
        n = len(df_aligned)
        if n < 15:
            raise HTTPException(status_code=400, detail="Not enough complete rows after covariate handling (need ≥ 15).")

    main_design = np.column_stack(feat_arrays) if feat_arrays else np.empty((n, 0))

    interaction_design = None
    interaction_names: List[str] = []
    if req.include_interaction:
        a = term_info[0]["design"]
        b = term_info[1]["design"]
        a_names = term_info[0]["col_names"]
        b_names = term_info[1]["col_names"]
        ix_cols = []
        for i in range(a.shape[1]):
            for j in range(b.shape[1]):
                ix_cols.append(a[:, i] * b[:, j])
                interaction_names.append(f"ix_{a_names[i]}_x_{b_names[j]}")
        interaction_design = np.column_stack(ix_cols)

    full_design_arrays = [main_design]
    full_names = list(feat_names)
    if interaction_design is not None:
        full_design_arrays.append(interaction_design)
        full_names = full_names + interaction_names
    cov_names = list(cov_df.columns)
    if cov_names:
        full_design_arrays.append(cov_df.values.astype(float))
        full_names = full_names + cov_names

    full_design = np.column_stack(full_design_arrays) if full_design_arrays else np.empty((n, 0))

    full_df = pd.DataFrame(full_design, columns=full_names, index=range(n))
    full_df["_dur_"] = duration
    full_df["_evt_"] = event

    try:
        cph_full = CoxPHFitter()
        await asyncio.to_thread(cph_full.fit, full_df, duration_col="_dur_", event_col="_evt_")
    except Exception as exc:
        logger.exception("Cox-RCS fitting failed")
        raise HTTPException(status_code=400, detail=f"Cox-RCS fitting error: {exc}")

    params_full = cph_full.params_.reindex(full_names).values
    cov_full = cph_full.variance_matrix_.reindex(index=full_names, columns=full_names).values
    se_full = cph_full.standard_errors_.reindex(full_names).values
    p_full = None
    try:
        p_full = cph_full.summary["p"].reindex(full_names).values
    except Exception:
        logger.exception("Cox-RCS p-value extraction failed")
    ci_low_full = cph_full.confidence_intervals_.iloc[:, 0].reindex(full_names).values
    ci_high_full = cph_full.confidence_intervals_.iloc[:, 1].reindex(full_names).values
    log_lik_full = float(cph_full.log_likelihood_)

    coefs = []
    for i, name in enumerate(full_names):
        coef = float(params_full[i])
        se = float(se_full[i])
        z = coef / se if se > 0 else None
        p = float(p_full[i]) if (p_full is not None and not np.isnan(p_full[i])) else None
        coefs.append({
            "name": name,
            "coef": coef,
            "hr": float(np.exp(coef)),
            "se": se,
            "z": z,
            "p": p,
            "ci_low": float(np.exp(ci_low_full[i])),
            "ci_high": float(np.exp(ci_high_full[i])),
        })

    nonlinearity = {}
    for ti, term in enumerate(req.spline_terms):
        sp_names = term_info[ti]["col_names"][1:]
        idx = [full_names.index(n) for n in sp_names]
        if not idx:
            continue
        b = params_full[idx]
        cv = cov_full[np.ix_(idx, idx)]
        try:
            wald = float(b @ np.linalg.solve(cv, b))
            p_nl = float(chi2.sf(wald, df=len(idx)))
        except Exception:
            logger.exception("Cox-RCS nonlinearity Wald test failed")
            wald = None
            p_nl = None
        nonlinearity[term.column] = {
            "wald": wald,
            "df": len(idx),
            "p": p_nl,
        }

    interaction_result = None
    if req.include_interaction and interaction_design is not None:
        reduced_names = [n for n in full_names if n not in interaction_names]
        reduced_df = full_df[reduced_names + ["_dur_", "_evt_"]].copy()
        try:
            cph_red = CoxPHFitter()
            await asyncio.to_thread(cph_red.fit, reduced_df, duration_col="_dur_", event_col="_evt_")
            ll_red = float(cph_red.log_likelihood_)
            lr_stat = 2.0 * (log_lik_full - ll_red)
            df_lr = len(interaction_names)
            p_lr = float(chi2.sf(lr_stat, df=df_lr))
            interaction_result = {
                "lr_stat": lr_stat,
                "df": df_lr,
                "p": p_lr,
                "log_lik_full": log_lik_full,
                "log_lik_reduced": ll_red,
            }
        except Exception as exc:
            logger.exception("Cox-RCS interaction LR test failed")
            interaction_result = {"error": f"interaction LR fit failed: {exc}"}

    curves_1d = []
    cov_means = cov_df.values.astype(float).mean(axis=0) if cov_names else np.array([])

    for ti, term in enumerate(req.spline_terms):
        x_raw = term_info[ti]["x_raw"]
        x_lo, x_hi = float(np.percentile(x_raw, 1)), float(np.percentile(x_raw, 99))
        x_syn = np.linspace(x_lo, x_hi, 200)
        sp_syn = _rcs_basis(x_syn, term_info[ti]["knots"])
        this_design = np.column_stack([x_syn, sp_syn])
        other_idx = 1 - ti if len(term_info) == 2 else None
        other_design = None
        if other_idx is not None:
            other_term = term_info[other_idx]
            ref_x = other_term["ref_value"]
            ref_sp = _rcs_basis(np.array([ref_x]), other_term["knots"]).flatten()
            other_vec = np.concatenate([[ref_x], ref_sp])
            other_design = np.tile(other_vec, (200, 1))

        if ti == 0:
            main_syn = this_design if other_design is None else np.column_stack([this_design, other_design])
        else:
            main_syn = other_design if other_design is None else np.column_stack([other_design, this_design]) if other_idx == 0 else None
            if main_syn is None:
                main_syn = np.column_stack([other_design, this_design])

        if req.include_interaction and interaction_design is not None:
            a_syn = main_syn[:, :term_info[0]["design"].shape[1]]
            b_syn = main_syn[:, term_info[0]["design"].shape[1]:term_info[0]["design"].shape[1] + term_info[1]["design"].shape[1]]
            ix_syn = np.column_stack([a_syn[:, i] * b_syn[:, j]
                                       for i in range(a_syn.shape[1])
                                       for j in range(b_syn.shape[1])])
            main_syn = np.column_stack([main_syn, ix_syn])

        if cov_names:
            main_syn = np.column_stack([main_syn, np.tile(cov_means, (200, 1))])

        lp_syn = main_syn @ params_full

        own_ref = term_info[ti]["ref_value"]
        ref_idx_syn = int(np.argmin(np.abs(x_syn - own_ref)))
        ref_row = main_syn[ref_idx_syn].copy()

        diffs = main_syn - ref_row
        var_lp = np.einsum("ij,jk,ik->i", diffs, cov_full, diffs)
        se_lp = np.sqrt(np.maximum(var_lp, 0))
        rel_lp = lp_syn - lp_syn[ref_idx_syn]
        hr = np.exp(rel_lp)
        ci_low = np.exp(rel_lp - 1.96 * se_lp)
        ci_high = np.exp(rel_lp + 1.96 * se_lp)

        curves_1d.append({
            "column": term.column,
            "x": _clean(x_syn),
            "hr": _clean(hr),
            "lower": _clean(ci_low),
            "upper": _clean(ci_high),
            "knots": [round(float(k), 2) for k in term_info[ti]["knots"]],
            "ref": round(float(own_ref), 4),
        })

    surface_2d = None
    if req.include_interaction and interaction_design is not None and len(term_info) == 2:
        g = max(10, min(int(req.grid_size or 50), 100))
        xa = term_info[0]["x_raw"]
        xb = term_info[1]["x_raw"]
        a_lo, a_hi = float(np.percentile(xa, 1)), float(np.percentile(xa, 99))
        b_lo, b_hi = float(np.percentile(xb, 1)), float(np.percentile(xb, 99))
        a_grid = np.linspace(a_lo, a_hi, g)
        b_grid = np.linspace(b_lo, b_hi, g)
        A, B = np.meshgrid(a_grid, b_grid)
        a_flat = A.flatten()
        b_flat = B.flatten()
        a_basis = np.column_stack([a_flat, _rcs_basis(a_flat, term_info[0]["knots"])])
        b_basis = np.column_stack([b_flat, _rcs_basis(b_flat, term_info[1]["knots"])])
        ix_flat = np.column_stack([a_basis[:, i] * b_basis[:, j]
                                    for i in range(a_basis.shape[1])
                                    for j in range(b_basis.shape[1])])
        cov_block = np.tile(cov_means, (a_flat.size, 1)) if cov_names else np.empty((a_flat.size, 0))
        design = np.column_stack([a_basis, b_basis, ix_flat, cov_block])
        lp = design @ params_full
        ref_a = term_info[0]["ref_value"]
        ref_b = term_info[1]["ref_value"]
        ra_basis = np.column_stack([[ref_a], _rcs_basis(np.array([ref_a]), term_info[0]["knots"])])
        rb_basis = np.column_stack([[ref_b], _rcs_basis(np.array([ref_b]), term_info[1]["knots"])])
        rix = np.column_stack([ra_basis[:, i] * rb_basis[:, j]
                                for i in range(ra_basis.shape[1])
                                for j in range(rb_basis.shape[1])])
        rcov = np.tile(cov_means, (1, 1)) if cov_names else np.empty((1, 0))
        ref_design = np.column_stack([ra_basis, rb_basis, rix, rcov])
        lp_ref = float((ref_design @ params_full)[0])
        hr_flat = np.exp(lp - lp_ref)
        hr_grid = hr_flat.reshape(B.shape)

        def _gclean(mat):
            out = []
            for row in mat:
                rrow = []
                for v in row:
                    fv = float(v)
                    rrow.append(None if (np.isnan(fv) or np.isinf(fv)) else round(fv, 4))
                out.append(rrow)
            return out

        surface_2d = {
            "x_col": term_info[0]["column"],
            "y_col": term_info[1]["column"],
            "x": [round(float(v), 4) for v in a_grid],
            "y": [round(float(v), 4) for v in b_grid],
            "hr": _gclean(hr_grid),
            "ref": {term_info[0]["column"]: round(float(ref_a), 4),
                      term_info[1]["column"]: round(float(ref_b), 4)},
        }

    aic_partial = None
    try:
        aic_partial = float(getattr(cph_full, "AIC_partial_", np.nan))
        if np.isnan(aic_partial):
            aic_partial = None
    except Exception:
        logger.exception("Cox-RCS partial AIC extraction failed")

    return _add_survival_warnings({
        "n": int(n),
        "n_events": int(event.sum()),
        "concordance": float(cph_full.concordance_index_),
        "log_likelihood": log_lik_full,
        "aic": aic_partial,
        "spline_terms": [
            {
                "column": t.column,
                "n_knots": t.n_knots,
                "knots": [round(float(k), 2) for k in term_info[i]["knots"]],
                "knot_positions_custom": t.knot_positions is not None,
                "ref": round(float(term_info[i]["ref_value"]), 4),
            }
            for i, t in enumerate(req.spline_terms)
        ],
        "covariates": req.covariates,
        "include_interaction": req.include_interaction,
        "coefficients": coefs,
        "nonlinearity": nonlinearity,
        "interaction": interaction_result,
        "curves_1d": curves_1d,
        "surface_2d": surface_2d,
    }, survival_warnings, n_invalid_survival)


# ── Unadjusted vs Adjusted Cox forest (paired univariable + multivariable) ─────

class CoxUniMultiRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    predictors: List[str]
    # Optional per-predictor reference level (value code as string) for
    # categorical predictors. Default = lowest value code.
    references: Optional[dict] = None
    # Optional subset of `predictors` fitted together as the "parsimonious"
    # multivariable model (publication Table 3 middle column). When omitted,
    # only the univariable and fully-adjusted columns are produced.
    parsimonious: Optional[List[str]] = None


def _encode_cox_predictor(series: pd.Series, name: str, force_categorical: bool = False,
                          reference: Optional[str] = None) -> Tuple[pd.DataFrame, list]:
    """Encode one predictor for Cox: numeric → single column; categorical →
    indicator columns vs the first (reference) level. Returns (design, terms)
    where each term carries display metadata for the forest row.

    force_categorical: treat as multi-level categorical even if the dtype is
    numeric (set when the column has value labels or a categorical kind
    override, e.g. LDL groups coded 1/2/3). Binary numeric columns (≤2 levels,
    e.g. sex/diabetes) stay a single HR term regardless.
    """
    numeric = pd.api.types.is_numeric_dtype(series)
    if not numeric:
        coerced = pd.to_numeric(series, errors="coerce")
        if not force_categorical and coerced.notna().mean() >= 0.8 and coerced.dropna().nunique() > 2:
            series = coerced
            numeric = True

    treat_cat = (not numeric) or (force_categorical and series.dropna().nunique() > 2)

    if numeric and not treat_cat:
        col = pd.to_numeric(series, errors="coerce").astype(float)
        design = pd.DataFrame({name: col})
        return design, [{
            "term": name, "predictor": name, "kind": "numeric",
            "category": None, "reference": None,
        }]

    def _lvl(v) -> str:
        # Normalise integer-valued codes so "1.0" matches value-label key "1".
        try:
            f = float(v)
            if f.is_integer():
                return str(int(f))
        except (TypeError, ValueError):
            pass
        return str(v)

    levels = [_lvl(v) for v in sorted_groups(series)]
    if len(levels) < 2:
        return pd.DataFrame(index=series.index), []
    ref = _lvl(reference) if reference is not None and _lvl(reference) in levels else levels[0]
    contrasts = [lvl for lvl in levels if lvl != ref]
    s = series.map(_lvl)
    design = pd.DataFrame(index=series.index)
    terms = []
    for lvl in contrasts:
        term = f"{name}={lvl}"
        design[term] = (s == lvl).astype(float)
        terms.append({
            "term": term, "predictor": name, "kind": "category",
            "category": lvl, "reference": ref,
        })
    return design, terms


def _cox_term_stats(cph: "CoxPHFitter", term: str) -> Optional[dict]:
    try:
        row = cph.summary.loc[term]
    except (KeyError, Exception):
        return None
    return {
        "hr": _safe_float(row["exp(coef)"]),
        "hr_ci_low": _safe_float(row["exp(coef) lower 95%"]),
        "hr_ci_high": _safe_float(row["exp(coef) upper 95%"]),
        "p": _safe_float(row["p"]),
    }


@router.post("/survival/cox_uni_multi")
async def cox_uni_multi(req: CoxUniMultiRequest):
    """Paired unadjusted (univariable) vs adjusted (multivariable) Cox HRs —
    one row per model term, for the publication 'Figure 4' forest plot."""
    df_full = _get_df(req.session_id).copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col] = pd.to_numeric(df_full[req.event_col], errors="coerce")

    df_full, survival_warnings, n_invalid_survival = _drop_invalid_survival_rows(df_full, req.duration_col, req.event_col)
    ev = sorted(df_full[req.event_col].dropna().unique())
    if set(ev) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1. Found: {ev[:10]}")

    # Columns with value labels or a categorical kind override are treated as
    # multi-level categoricals (e.g. LDL groups 1/2/3 → contrast rows), even
    # when stored as integer codes. Binary columns stay a single HR term.
    meta = store.get_metadata(req.session_id) or {}
    kinds = store.get_kind_overrides(req.session_id) or {}

    # Encode every predictor once; reuse the design for both passes.
    encoded: dict[str, pd.DataFrame] = {}
    all_terms: list[dict] = []
    for p in req.predictors:
        if p not in df_full.columns:
            raise HTTPException(status_code=422, detail=f"Predictor '{p}' not in dataset.")
        force_cat = bool((meta.get(p, {}) or {}).get("value_labels")) or kinds.get(p) == "categorical"
        ref = (req.references or {}).get(p)
        design, terms = _encode_cox_predictor(df_full[p], p, force_categorical=force_cat, reference=ref)
        if not terms:
            continue
        encoded[p] = design
        all_terms.extend(terms)

    if not all_terms:
        raise HTTPException(status_code=400, detail="No usable predictors after encoding.")

    base = df_full[[req.duration_col, req.event_col]]

    # ── Univariable pass: one Cox per predictor ──
    uni_stats: dict[str, dict] = {}
    for p, design in encoded.items():
        fit_df = pd.concat([base, design], axis=1).dropna()
        if len(fit_df) < 10:
            continue
        cph = CoxPHFitter()
        try:
            await asyncio.to_thread(cph.fit, fit_df, duration_col=req.duration_col, event_col=req.event_col)
        except Exception:
            logger.exception("Univariable Cox failed for %s", p)
            continue
        for t in [t for t in all_terms if t["predictor"] == p]:
            st = _cox_term_stats(cph, t["term"])
            if st:
                uni_stats[t["term"]] = st

    # ── Multivariable pass: all predictors together ──
    full_design = pd.concat([encoded[p] for p in encoded], axis=1)
    fit_full = pd.concat([base, full_design], axis=1).dropna()
    adj_stats: dict[str, dict] = {}
    n_used = 0
    n_events = 0
    if len(fit_full) >= 10:
        cph_full = CoxPHFitter()
        try:
            await asyncio.to_thread(cph_full.fit, fit_full, duration_col=req.duration_col, event_col=req.event_col)
            n_used = len(fit_full)
            n_events = int(fit_full[req.event_col].sum())
            for t in all_terms:
                st = _cox_term_stats(cph_full, t["term"])
                if st:
                    adj_stats[t["term"]] = st
        except Exception:
            logger.exception("Multivariable Cox failed")

    # ── Parsimonious pass: a user-chosen subset fitted together ──
    # Only the selected predictors enter this model; every other predictor's
    # parsimonious cell stays empty (null) so the table renders a blank, not a
    # borrowed estimate. Mirrors the fully-adjusted pass on a smaller design.
    pars_set = [p for p in (req.parsimonious or []) if p in encoded]
    pars_stats: dict[str, dict] = {}
    n_pars = 0
    n_events_pars = 0
    if pars_set:
        pars_design = pd.concat([encoded[p] for p in pars_set], axis=1)
        fit_pars = pd.concat([base, pars_design], axis=1).dropna()
        if len(fit_pars) >= 10:
            cph_pars = CoxPHFitter()
            try:
                await asyncio.to_thread(cph_pars.fit, fit_pars, duration_col=req.duration_col, event_col=req.event_col)
                n_pars = len(fit_pars)
                n_events_pars = int(fit_pars[req.event_col].sum())
                for t in all_terms:
                    if t["predictor"] in pars_set:
                        st = _cox_term_stats(cph_pars, t["term"])
                        if st:
                            pars_stats[t["term"]] = st
            except Exception:
                logger.exception("Parsimonious Cox failed")

    rows = []
    for t in all_terms:
        rows.append({
            **t,
            "unadjusted": uni_stats.get(t["term"]),
            "parsimonious": pars_stats.get(t["term"]),
            "adjusted": adj_stats.get(t["term"]),
        })

    return _add_survival_warnings({
        "duration_col": req.duration_col,
        "event_col": req.event_col,
        "n": n_used,
        "n_events": n_events,
        "n_pars": n_pars,
        "n_events_pars": n_events_pars,
        "rows": rows,
    }, survival_warnings, n_invalid_survival)


# ── Multiple model-specification forest (one exposure across adjustment sets) ───

class _ModelSpec(BaseModel):
    label: str
    covariates: List[str] = []


class CoxModelSpecsRequest(BaseModel):
    session_id: str
    duration_col: str
    event_col: str
    exposure: str                          # the predictor of interest (HR tracked)
    exposure_reference: Optional[str] = None
    specs: List[_ModelSpec]                # named adjustment sets
    include_unadjusted: bool = True


@router.post("/survival/cox_model_specs")
async def cox_model_specs(req: CoxModelSpecsRequest):
    """One exposure's adjusted HR across several model specifications.

    Backs the 'sensitivity analyses' forest: the same exposure (e.g. LDL<100)
    fitted under different adjustment sets (parsimonious / alternative / full),
    each row = the exposure's HR (95% CI, p) in that model.
    """
    df_full = _get_df(req.session_id).copy()
    df_full[req.duration_col] = pd.to_numeric(df_full[req.duration_col], errors="coerce")
    df_full[req.event_col] = pd.to_numeric(df_full[req.event_col], errors="coerce")

    df_full, survival_warnings, n_invalid_survival = _drop_invalid_survival_rows(df_full, req.duration_col, req.event_col)
    ev = sorted(df_full[req.event_col].dropna().unique())
    if set(ev) - {0, 1, 0.0, 1.0}:
        raise HTTPException(status_code=422, detail=f"Event column must be binary 0/1. Found: {ev[:10]}")
    if req.exposure not in df_full.columns:
        raise HTTPException(status_code=422, detail=f"Exposure '{req.exposure}' not in dataset.")

    meta = store.get_metadata(req.session_id) or {}
    kinds = store.get_kind_overrides(req.session_id) or {}

    def _force_cat(col: str) -> bool:
        return bool((meta.get(col, {}) or {}).get("value_labels")) or kinds.get(col) == "categorical"

    exp_design, exp_terms = _encode_cox_predictor(
        df_full[req.exposure], req.exposure,
        force_categorical=_force_cat(req.exposure), reference=req.exposure_reference)
    if not exp_terms:
        raise HTTPException(status_code=400, detail=f"Exposure '{req.exposure}' could not be encoded (needs ≥2 levels).")

    base = df_full[[req.duration_col, req.event_col]]

    specs = list(req.specs)
    if req.include_unadjusted:
        specs = [_ModelSpec(label="Unadjusted", covariates=[])] + specs

    out_specs = []
    for spec in specs:
        cov_designs = []
        for c in spec.covariates:
            if c == req.exposure or c not in df_full.columns:
                continue
            d, _ = _encode_cox_predictor(df_full[c], c, force_categorical=_force_cat(c))
            if d.shape[1]:
                cov_designs.append(d)
        design = pd.concat([exp_design, *cov_designs], axis=1)
        fit_df = pd.concat([base, design], axis=1).dropna()
        spec_terms = []
        n_used = 0
        n_events = 0
        if len(fit_df) >= 10:
            cph = CoxPHFitter()
            try:
                await asyncio.to_thread(cph.fit, fit_df, duration_col=req.duration_col, event_col=req.event_col)
                n_used = len(fit_df)
                n_events = int(fit_df[req.event_col].sum())
                for t in exp_terms:
                    st = _cox_term_stats(cph, t["term"])
                    spec_terms.append({**t, **(st or {"hr": None, "hr_ci_low": None, "hr_ci_high": None, "p": None})})
            except Exception:
                logger.exception("Cox model-spec fit failed for %s", spec.label)
        out_specs.append({
            "label": spec.label,
            "covariates": list(spec.covariates),
            "n": n_used,
            "n_events": n_events,
            "terms": spec_terms,
        })

    return _add_survival_warnings({
        "duration_col": req.duration_col,
        "event_col": req.event_col,
        "exposure": req.exposure,
        "exposure_terms": exp_terms,
        "specs": out_specs,
    }, survival_warnings, n_invalid_survival)
