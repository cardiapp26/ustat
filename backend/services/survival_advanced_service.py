"""Advanced-survival service: statistical orchestration for the
/api/survival_advanced endpoints.

Extracted from routers/survival_advanced.py so the router is a thin dispatcher.
Each fit_* function takes the validated request, retrieves the session frame,
runs the analysis, and returns a plain dict (a few also persist results via the
store service, which is an allowed service-to-service dependency).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from services import store


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df




def _safe(v: Any) -> Any:
    """Make a value JSON-safe."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ── 1. MICE Multiple Imputation ─────────────────────────────────────────────




def _fine_gray_fit(df: pd.DataFrame, duration: str, event: str,
                   cause: int, predictors: List[str]) -> dict:
    """Fit a Fine-Gray subdistribution hazard regression via Geskus's
    (2011) IPCW-weighted Cox reformulation. Mathematically equivalent
    to Fine & Gray (1999) and free of external dependencies beyond
    lifelines.

    Algorithm:
      1. Estimate Ĝ(t) — Kaplan-Meier of the censoring distribution
         (event = 1 iff e == 0).
      2. Build an augmented long-format dataset:
         • cause-of-interest subject  → (0, t_i, evt=1, w=1)
         • competing-event subject    → pseudo-rows (0, s, evt=0,
             w = Ĝ(s)/Ĝ(t_i)) at every cause-of-interest event time
             s > t_i; weight is 0 once Ĝ(s) collapses to 0.
         • censored subject           → (0, t_i, evt=0, w=1)
      3. Fit lifelines.CoxPHFitter on this dataset with weights_col
         and robust=True (Lin-Wei sandwich SE).
    """
    from lifelines import CoxPHFitter, KaplanMeierFitter

    # Encode predictors: numeric stays numeric, categorical → dummies
    # (drop_first=True). This matches the existing Cox endpoint in
    # backend/routers/models.py so the user sees consistent dummy names.
    pred_raw = df[predictors].copy()
    numeric_pred: List[str] = []
    cat_pred: List[str] = []
    for c in predictors:
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

    work = pd.concat(
        [df[[duration, event]].reset_index(drop=True), enc.reset_index(drop=True)],
        axis=1,
    ).dropna()
    if len(work) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows after dropping NA in predictors / duration / event (need ≥ 10, got {len(work)}).")

    t = work[duration].values.astype(float)
    e = work[event].values.astype(int)
    cov_cols = list(enc.columns)
    if not cov_cols:
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")

    # Censoring KM Ĝ(t): event indicator is "censored" (e == 0).
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(t, (e == 0).astype(int))
    g_at = kmf_c.survival_function_.iloc[:, 0]  # Series indexed by time
    g_times = g_at.index.values.astype(float)
    g_vals = g_at.values.astype(float)

    def _g(tau: float) -> float:
        """Right-continuous Ĝ(tau): largest indexed time ≤ tau."""
        idx = np.searchsorted(g_times, tau, side="right") - 1
        if idx < 0:
            return 1.0
        return float(g_vals[idx])

    cause_event_times = np.sort(np.unique(t[e == cause]))
    if len(cause_event_times) == 0:
        raise HTTPException(status_code=422, detail=f"No subjects experienced the event of interest (code {cause}).")

    # Cap on augmented row count — defends against pathological inputs where
    # N_competing × K_event_times explodes (e.g. ten thousand subjects all
    # with a competing event and a thousand distinct cause-of-interest times).
    n_competing = int(((e != cause) & (e != 0)).sum())
    if n_competing * len(cause_event_times) > 500_000:
        raise HTTPException(
            status_code=422,
            detail=(
                "Augmented Fine-Gray dataset would exceed 500 000 rows "
                f"({n_competing} competing-event subjects × "
                f"{len(cause_event_times)} unique event times of interest). "
                "Reduce the dataset (e.g. coarser duration units, fewer subjects) "
                "or fit on a stratified subset."
            ),
        )

    # Build augmented dataset
    rows: List[dict] = []
    cov_arr = work[cov_cols].values
    for i in range(len(work)):
        ti = float(t[i])
        ei = int(e[i])
        cov_i = cov_arr[i]
        if ei == cause:
            row = {"_stop_": ti, "_event_": 1, "_w_": 1.0}
            for cj, name in enumerate(cov_cols):
                row[name] = float(cov_i[cj])
            rows.append(row)
        elif ei == 0:
            row = {"_stop_": ti, "_event_": 0, "_w_": 1.0}
            for cj, name in enumerate(cov_cols):
                row[name] = float(cov_i[cj])
            rows.append(row)
        else:
            g_ti = _g(ti)
            if g_ti <= 0:
                continue
            future_ev = cause_event_times[cause_event_times > ti]
            for s in future_ev:
                w_s = _g(float(s)) / g_ti
                if w_s <= 0:
                    break
                row = {"_stop_": float(s), "_event_": 0, "_w_": float(w_s)}
                for cj, name in enumerate(cov_cols):
                    row[name] = float(cov_i[cj])
                rows.append(row)

    aug = pd.DataFrame(rows)
    if aug["_event_"].sum() == 0:
        raise HTTPException(status_code=422, detail="No cause-of-interest events present after building the augmented dataset.")

    cph = CoxPHFitter()
    cph.fit(aug, duration_col="_stop_", event_col="_event_",
            weights_col="_w_", robust=True)

    ci = cph.confidence_intervals_
    coefs: List[dict] = []
    for var in cph.params_.index:
        beta = float(cph.params_[var])
        se_v = float(cph.standard_errors_[var])
        try:
            p_v = float(cph.summary["p"].loc[var])
        except Exception:
            p_v = None
        try:
            lo = float(ci.loc[var, ci.columns[0]])
            hi = float(ci.loc[var, ci.columns[1]])
        except Exception:
            lo = hi = float("nan")
        coefs.append({
            "variable": str(var),
            "estimate": round(beta, 6),
            "shr": round(float(np.exp(beta)), 4),
            "se": round(se_v, 6),
            "z": round(beta / se_v, 4) if se_v > 0 else None,
            "p": round(p_v, 6) if p_v is not None else None,
            "ci_low": round(lo, 4) if math.isfinite(lo) else None,
            "ci_high": round(hi, 4) if math.isfinite(hi) else None,
            "shr_low": round(float(np.exp(lo)), 4) if math.isfinite(lo) else None,
            "shr_high": round(float(np.exp(hi)), 4) if math.isfinite(hi) else None,
        })

    return {
        "method": "fine_gray_regression",
        "model": "Fine-Gray subdistribution hazards (IPCW-weighted Cox, Lin-Wei robust SE)",
        "n": int(len(work)),
        "n_events_of_interest": int((e == cause).sum()),
        "n_competing": int(((e != cause) & (e != 0)).sum()),
        "n_censored": int((e == 0).sum()),
        "n_augmented_rows": int(len(aug)),
        "concordance": round(float(cph.concordance_index_), 4),
        "coefficients": coefs,
        "method_note": (
            "Fine-Gray subdistribution hazard regression fit via the Geskus "
            "(2011) IPCW-weighted Cox reformulation — mathematically equivalent "
            "to Fine & Gray (1999). Competing-event subjects stay at risk past "
            "their event time with weights Ĝ(s)/Ĝ(t_i) from the Kaplan-Meier "
            "estimate of the censoring distribution. Lin-Wei sandwich estimator "
            "for the standard errors. Output sHR = exp(β) is the subdistribution "
            "hazard ratio for the cause of interest."
        ),
    }




def _two_sided_normal_p(z: float) -> float:
    """Return a numerically stable two-sided standard-normal p-value."""
    from scipy.stats import norm

    tail = float(norm.sf(abs(z)))
    if tail > 0:
        return 2 * tail
    return float(math.exp(math.log(2.0) + float(norm.logsf(abs(z)))))


def _rmst_one_group_raw(t: np.ndarray, e: np.ndarray, tau: float) -> Dict[str, float]:
    """KM-based RMST estimator with Greenwood-style SE.

    Algorithm:
      1. Fit KaplanMeierFitter on (t, e).
      2. Integrate the right-continuous KM step function from 0 to τ.
      3. SE via Greenwood's variance:
           Var[RMST(τ)] = Σ_j A_j² d_j / (n_j (n_j - d_j)),
           where A_j = ∫_{t_j}^{τ} S(u) du.
      4. 95% CI = Wald on RMST.
    """
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(t, e.astype(int))

    # Step-function on the unique observed times. surv[k] = S(t_k+) for
    # t in [t_k, t_{k+1}).
    sf = kmf.survival_function_.iloc[:, 0]
    sf_times = sf.index.values.astype(float)
    sf_surv = sf.values.astype(float)
    # lifelines already seeds the timeline with the origin (t=0, S=1). Only
    # prepend it ourselves when it is missing — otherwise we create a
    # duplicate zero, the first trapezoid piece has zero width, and the loop
    # below breaks immediately, collapsing RMST to 0.
    if len(sf_times) and sf_times[0] == 0.0:
        times = sf_times
        surv = sf_surv
    else:
        times = np.concatenate(([0.0], sf_times))
        surv = np.concatenate(([1.0], sf_surv))

    et = kmf.event_table  # columns: removed, observed, censored, entrance, at_risk
    et_index = et.index.values.astype(float)
    d_arr = et["observed"].values.astype(float)
    n_arr = et["at_risk"].values.astype(float)

    # KM step-function rectangles between consecutive time points, capped at tau.
    pieces_t: List[float] = []  # left edges
    pieces_h: List[float] = []  # heights (S at left edge)
    pieces_w: List[float] = []  # widths (capped at tau)
    for k in range(len(times) - 1):
        a = times[k]
        b = min(times[k + 1], tau)
        if b <= a:
            break
        pieces_t.append(a)
        pieces_h.append(surv[k])
        pieces_w.append(b - a)
    pieces_t_arr = np.array(pieces_t)
    pieces_h_arr = np.array(pieces_h)
    pieces_w_arr = np.array(pieces_w)
    cum_area = np.concatenate(([0.0], np.cumsum(pieces_h_arr * pieces_w_arr)))
    total_area = cum_area[-1] if len(cum_area) > 0 else 0.0
    rmst = float(total_area)

    # Greenwood-style SE: Var(RMST(τ)) = Σ_{j: t_j ≤ τ} A_j^2 * d_j / (n_j (n_j - d_j))
    # with A_j = ∫_{t_j}^τ S(u) du.
    se_var = 0.0
    for j, tj in enumerate(et_index):
        if tj > tau:
            break
        d_j = d_arr[j]
        n_j = n_arr[j]
        if d_j <= 0 or n_j - d_j <= 0:
            continue
        # Include the interval that starts at the event time.  Greenwood's
        # RMST variance uses the remaining area from t_j through tau; using
        # side="right" dropped that first interval and understated the SE.
        idx = np.searchsorted(pieces_t_arr, tj, side="left")
        area_before = cum_area[idx] if idx < len(cum_area) else total_area
        A_j = total_area - area_before
        se_var += (A_j ** 2) * d_j / (n_j * (n_j - d_j))

    se = float(np.sqrt(se_var)) if se_var > 0 else 0.0
    n = int(len(t))
    n_events = int(e.astype(int).sum())
    z95 = 1.959963984540054
    return {
        "n": n,
        "n_events": n_events,
        "rmst": float(rmst),
        "se": se,
        "ci_low": float(rmst - z95 * se),
        "ci_high": float(rmst + z95 * se),
    }


def _format_rmst_result(result: Dict[str, float]) -> Dict[str, float]:
    """Round an internal RMST estimate for the public response contract."""
    return {
        "n": int(result["n"]),
        "n_events": int(result["n_events"]),
        "rmst": round(result["rmst"], 4),
        "se": round(result["se"], 4),
        "ci_low": round(result["ci_low"], 4),
        "ci_high": round(result["ci_high"], 4),
    }


def _rmst_one_group(t: np.ndarray, e: np.ndarray, tau: float) -> Dict[str, float]:
    """Return the public, four-decimal RMST result for one group."""
    return _format_rmst_result(_rmst_one_group_raw(t, e, tau))




def _mcf(intervals: pd.DataFrame, start: str, stop: str, event: str) -> List[dict]:
    """Nonparametric mean cumulative function (Nelson 1995) for recurrent
    events on counting-process intervals: MCF(t) = Σ_{t_k ≤ t} d_k / n_k where
    d_k = events at t_k and n_k = subjects under observation at t_k."""
    ev_times = np.sort(np.unique(intervals.loc[intervals[event] == 1, stop].values.astype(float)))
    starts = intervals[start].values.astype(float)
    stops = intervals[stop].values.astype(float)
    evs = intervals[event].values.astype(int)
    pts = [{"t": 0.0, "mcf": 0.0}]
    cum = 0.0
    for t in ev_times:
        d = int(np.sum((stops == t) & (evs == 1)))
        n = int(np.sum((starts < t) & (stops >= t)))
        if n > 0:
            cum += d / n
        pts.append({"t": round(float(t), 4), "mcf": round(float(cum), 5)})
    return pts


def _rate_points_from_mcf(mcf: List[dict]) -> List[dict]:
    pts: List[dict] = []
    for prev, cur in zip(mcf[:-1], mcf[1:]):
        dt = float(cur["t"]) - float(prev["t"])
        if dt <= 0:
            continue
        rate = (float(cur["mcf"]) - float(prev["mcf"])) / dt
        pts.append({"t": cur["t"], "rate": round(float(rate), 5)})
    return pts


def _recurrent_event_order(work: pd.DataFrame, id_col: str, stop_col: str, event_col: str, event_order_col: Optional[str]) -> pd.Series:
    if event_order_col and event_order_col in work.columns:
        order = pd.to_numeric(work[event_order_col], errors="coerce").fillna(1).astype(int)
        return order.clip(lower=1)
    ordered = work.sort_values([id_col, stop_col]).copy()
    prior_events = ordered.groupby(id_col)[event_col].cumsum() - ordered[event_col].astype(int)
    event_order = prior_events + 1
    event_order = event_order.astype(int).clip(lower=1)
    return event_order.reindex(work.index).fillna(1).astype(int)


def _apply_recurrent_time_scale(work: pd.DataFrame, start_col: str, stop_col: str, time_scale: str) -> pd.DataFrame:
    out = work.copy()
    scale = (time_scale or "total").lower()
    if scale == "gap":
        out["__re_start__"] = 0.0
        out["__re_stop__"] = pd.to_numeric(out[stop_col], errors="coerce") - pd.to_numeric(out[start_col], errors="coerce")
    else:
        out["__re_start__"] = pd.to_numeric(out[start_col], errors="coerce")
        out["__re_stop__"] = pd.to_numeric(out[stop_col], errors="coerce")
    return out[out["__re_stop__"] > out["__re_start__"]]


def _cox_recurrent_coefficients(cph: Any) -> List[dict]:
    summ = cph.summary
    coefs: List[dict] = []
    for var in cph.params_.index:
        beta = float(cph.params_[var])
        coefs.append({
            "variable": str(var),
            "estimate": round(beta, 6),
            "rate_ratio": round(float(np.exp(beta)), 4),
            "robust_se": round(float(summ.loc[var, "se(coef)"]), 6),
            "z": round(float(summ.loc[var, "z"]), 4),
            "p": round(float(summ.loc[var, "p"]), 6),
            "rr_low": round(float(summ.loc[var, "exp(coef) lower 95%"]), 4),
            "rr_high": round(float(summ.loc[var, "exp(coef) upper 95%"]), 4),
        })
    return coefs


def _negative_binomial_recurrent_counts(
    work: pd.DataFrame,
    id_col: str,
    start_col: str,
    stop_col: str,
    event_col: str,
    enc: pd.DataFrame,
) -> Dict[str, Any]:
    try:
        import statsmodels.api as sm

        aligned = pd.concat([
            work[[id_col, start_col, stop_col, event_col]].reset_index(drop=True),
            enc.reset_index(drop=True),
        ], axis=1).dropna()
        cov_cols = list(enc.columns)
        grouped = aligned.groupby(id_col)
        counts = grouped[event_col].sum().astype(float)
        followup = (grouped[stop_col].max() - grouped[start_col].min()).clip(lower=1e-8).astype(float)
        X = grouped[cov_cols].first().astype(float)
        X = sm.add_constant(X, has_constant="add")
        model = sm.GLM(counts, X, family=sm.families.NegativeBinomial(), offset=np.log(followup))
        fit = model.fit()
        rows = []
        for var in fit.params.index:
            beta = float(fit.params[var])
            rows.append({
                "variable": str(var),
                "estimate": round(beta, 6),
                "rate_ratio": round(float(np.exp(beta)), 4),
                "se": round(float(fit.bse[var]), 6),
                "p": round(float(fit.pvalues[var]), 6),
            })
        return {
            "available": True,
            "model": "negative_binomial_event_count",
            "n_subjects": int(len(counts)),
            "coefficients": rows,
            "aic": round(float(fit.aic), 4),
            "method_note": "Subject-level event count model with log follow-up offset; useful as a robustness check, not a replacement for event-time models.",
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _informative_censoring_diagnostics(
    work: pd.DataFrame,
    id_col: str,
    start_col: str,
    stop_col: str,
    event_col: str,
    terminal_time_col: Optional[str],
    terminal_event_col: Optional[str],
) -> Dict[str, Any]:
    if not terminal_time_col or not terminal_event_col:
        return {"available": False, "reason": "terminal_time_col and terminal_event_col are required."}
    if terminal_time_col not in work.columns or terminal_event_col not in work.columns:
        return {"available": False, "reason": "Terminal-event columns are not present in the analysis dataframe."}
    try:
        from scipy.stats import mannwhitneyu, spearmanr

        per_subject = work.groupby(id_col).agg(
            recurrent_events=(event_col, "sum"),
            followup=(stop_col, "max"),
            start=(start_col, "min"),
            terminal_time=(terminal_time_col, "max"),
            terminal_event=(terminal_event_col, "max"),
        )
        per_subject["followup"] = (per_subject["followup"] - per_subject["start"]).clip(lower=1e-8)
        per_subject["recurrent_rate"] = per_subject["recurrent_events"] / per_subject["followup"]
        terminal_yes = per_subject.loc[per_subject["terminal_event"] == 1, "recurrent_rate"]
        terminal_no = per_subject.loc[per_subject["terminal_event"] == 0, "recurrent_rate"]
        p_group = None
        if len(terminal_yes) >= 3 and len(terminal_no) >= 3:
            p_group = float(mannwhitneyu(terminal_yes, terminal_no, alternative="two-sided").pvalue)
        corr, corr_p = spearmanr(per_subject["terminal_time"], per_subject["recurrent_rate"], nan_policy="omit")
        terminal_event_rate = float(per_subject["terminal_event"].mean())
        return {
            "available": True,
            "method": "Ghosh-Lin style screening diagnostics",
            "terminal_event_rate": round(terminal_event_rate, 4),
            "recurrent_rate_by_terminal_event": {
                "terminal_event": round(float(terminal_yes.mean()), 5) if len(terminal_yes) else None,
                "no_terminal_event": round(float(terminal_no.mean()), 5) if len(terminal_no) else None,
                "mann_whitney_p": round(p_group, 6) if p_group is not None else None,
            },
            "terminal_time_recurrent_rate_spearman": {
                "rho": round(float(corr), 5) if np.isfinite(corr) else None,
                "p": round(float(corr_p), 6) if np.isfinite(corr_p) else None,
            },
            "warning": "A small p-value suggests recurrent-event intensity may be associated with terminal events; consider joint frailty.",
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _recurrent_specific_diagnostics(
    work: pd.DataFrame,
    id_col: str,
    event_col: str,
) -> Dict[str, Any]:
    event_rows = work[work[event_col] == 1].copy()
    gap_times = (event_rows["__re_stop__"] - event_rows["__re_start__"]).astype(float)
    event_counts = work.groupby(id_col)[event_col].sum().astype(float)
    expected_count = float(event_counts.mean()) if len(event_counts) else 0.0
    residuals = event_counts - expected_count
    order_counts = event_rows["__event_order__"].value_counts().sort_index()
    return {
        "gap_time": {
            "n_event_gaps": int(len(gap_times)),
            "mean": round(float(gap_times.mean()), 5) if len(gap_times) else None,
            "median": round(float(gap_times.median()), 5) if len(gap_times) else None,
            "min": round(float(gap_times.min()), 5) if len(gap_times) else None,
            "max": round(float(gap_times.max()), 5) if len(gap_times) else None,
        },
        "event_order_distribution": {str(int(k)): int(v) for k, v in order_counts.items()},
        "subject_event_count_residuals": {
            "expected_events_per_subject": round(expected_count, 5),
            "mean": round(float(residuals.mean()), 5) if len(residuals) else None,
            "sd": round(float(residuals.std(ddof=1)), 5) if len(residuals) > 1 else 0.0,
            "outlier_count_abs_gt_2sd": int(np.sum(np.abs(residuals) > 2.0 * residuals.std(ddof=1))) if len(residuals) > 2 and residuals.std(ddof=1) > 0 else 0,
        },
    }




def _missing_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask of missing, blank, or sentinel values."""
    mask = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        mask = mask | series.astype(str).str.strip().eq("")
    from services.dirty_value_guard import flag_sentinels, plausibility_max_for_column
    mask = mask | flag_sentinels(series, plausibility_max_for_column(series.name))
    return mask


def _is_numeric_target(df: pd.DataFrame, c: str) -> bool:
    """Whether column c should be treated as numeric for MICE/PMM."""
    if pd.api.types.is_numeric_dtype(df[c]):
        return True
    coerced = pd.to_numeric(df[c], errors="coerce")
    return bool(df[c].notna().any() and coerced.notna().mean() >= 0.8)


def _compute_mice(df: pd.DataFrame, columns: List[str], max_iter: int, random_state: int):
    """Compute MICE/PMM imputation and return working frame + metadata.

    Returns
    -------
    df_work : pd.DataFrame
        Copy of ``df`` with missing values in ``columns`` filled.
    missing_masks : dict[str, pd.Series]
        Mask used to identify missing cells per target column.
    cols_with_missing : list[str]
        Target columns that actually had missing values.
    num_targets : list[str]
        Numeric targets imputed by PMM.
    cat_targets : list[str]
        Non-numeric targets imputed by mode.
    col_summaries : list[dict]
        Summary rows for each imputed column.
    """
    from services.missing_data import mice_multiple

    missing_masks = {c: _missing_mask(df[c]) for c in columns}
    cols_with_missing = [c for c in columns if missing_masks[c].any()]
    if not cols_with_missing:
        raise HTTPException(status_code=422, detail="No missing values in selected columns")

    num_targets = [c for c in cols_with_missing if _is_numeric_target(df, c)]
    cat_targets = [c for c in cols_with_missing if c not in num_targets]

    df_work = df.copy()
    for c in cols_with_missing:
        df_work.loc[missing_masks[c], c] = np.nan
    pre_missing = {c: int(missing_masks[c].sum()) for c in cols_with_missing}
    col_summaries: list = []

    # ── Numeric targets via MICE ──
    if num_targets:
        # Numeric feature columns with at least one observed value (an all-NaN
        # feature makes IterativeImputer drop/realign columns and crash).
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
        ]
        for t in num_targets:
            if t not in numeric_cols:
                df_work[t] = pd.to_numeric(df_work[t], errors="coerce")
                if df_work[t].notna().any():
                    numeric_cols.append(t)
        if len(numeric_cols) < 2:
            # Not enough features for chained equations → per-column median.
            for t in num_targets:
                med = pd.to_numeric(df_work[t], errors="coerce").median()
                df_work[t] = pd.to_numeric(df_work[t], errors="coerce").fillna(med)
        else:
            # Single Predictive Mean Matching imputation (chained equations).
            # PMM draws a real observed donor — preserving the distribution —
            # instead of averaging several parametric draws into a synthetic
            # value (the old IterativeImputer behaviour underestimated variance
            # and could yield impossible values).
            subset = df_work[numeric_cols].apply(pd.to_numeric, errors="coerce")
            imp = mice_multiple(subset, numeric_cols, n_imputations=1,
                                max_iter=max_iter, random_state=random_state)
            subset_filled = imp.imputed_datasets[0]
            for t in num_targets:
                df_work[t] = subset_filled[t]
        for c in num_targets:
            imputed_vals = pd.to_numeric(df_work.loc[missing_masks[c], c], errors="coerce")
            col_summaries.append({
                "column": c, "method": "PMM", "n_imputed": pre_missing[c],
                "mean_imputed": _safe(round(float(imputed_vals.mean()), 4)) if len(imputed_vals) > 0 else None,
                "min_imputed": _safe(round(float(imputed_vals.min()), 4)) if len(imputed_vals) > 0 else None,
                "max_imputed": _safe(round(float(imputed_vals.max()), 4)) if len(imputed_vals) > 0 else None,
            })

    # ── Categorical / text targets via most-frequent (mode) ──
    for c in cat_targets:
        observed = df_work[c].dropna()
        observed = observed[observed.astype(str).str.strip() != ""]
        fill_val = observed.mode().iloc[0] if not observed.empty else None
        if fill_val is not None:
            df_work[c] = df_work[c].fillna(fill_val)
        col_summaries.append({
            "column": c, "method": "mode", "n_imputed": pre_missing[c],
            "mode_imputed": _safe(fill_val), "mean_imputed": None, "min_imputed": None, "max_imputed": None,
        })

    return df_work, missing_masks, cols_with_missing, num_targets, cat_targets, col_summaries


def _build_mice_response(
    df: pd.DataFrame,
    req,
    df_work: pd.DataFrame,
    cols_with_missing: List[str],
    num_targets: List[str],
    cat_targets: List[str],
    col_summaries: List[dict],
    new_column_map: Dict[str, str],
    *,
    preview_rows: Optional[List[dict]] = None,
) -> dict:
    """Build the JSON response shared by apply/preview MICE endpoints."""
    total_imputed = sum(s["n_imputed"] for s in col_summaries)

    mech = req.mechanism.upper()
    mech_label = {"UNKNOWN": "Unknown", "MCAR": "MCAR", "MAR": "MAR", "MNAR": "MNAR"}.get(mech, "Unknown")
    mech_ok = mech != "MNAR"
    assumptions = [
        {"name": "Missing mechanism",
         "met": mech_ok,
         "detail": f"Assumed {mech_label}. Imputation is valid under MAR/MCAR."
                   + (" MNAR may produce biased estimates — consider sensitivity analysis." if not mech_ok else "")},
        {"name": "Imputation methods", "met": True,
         "detail": f"{len(num_targets)} numeric column(s) via Predictive Mean Matching (chained "
                   f"equations, {req.max_iter} iterations); "
                   f"{len(cat_targets)} categorical column(s) via most-frequent (mode)."},
        {"name": "Single completed dataset", "met": True,
         "detail": (
             "Original columns are preserved and one PMM-completed column is created for each selected variable. "
             if new_column_map
             else "The session is filled with one PMM-completed dataset (single imputation). "
         )
                   + "For variance-correct inference use the model panels' MICE option "
                   + "(m datasets pooled by Rubin's rules)."},
    ]

    method_bits = []
    if num_targets:
        method_bits.append(f"{len(num_targets)} numeric variable(s) via Predictive Mean Matching ({req.max_iter} iterations)")
    if cat_targets:
        method_bits.append(f"{len(cat_targets)} categorical variable(s) via most-frequent value")
    preserved_text = (
        "Original columns were preserved; completed values were written to "
        + ", ".join(f"{src} → {dst}" for src, dst in new_column_map.items()) + ". "
        if new_column_map else ""
    )
    result_text = (
        f"Single imputation was performed assuming a {mech_label} mechanism: "
        + "; ".join(method_bits) + ". "
        + f"{total_imputed} missing values were imputed across "
        + f"{len(cols_with_missing)} variable(s): {', '.join(cols_with_missing)}. "
        + preserved_text
        + "For inference, prefer the model panels' pooled MICE (Rubin's rules)."
    )

    methods_text = (
        f"Missing values in {', '.join(cols_with_missing)} were handled using a single completed "
        f"dataset under an assumed {mech_label} missing-data mechanism. "
    )
    if num_targets:
        methods_text += (
            f"Numeric variables ({', '.join(num_targets)}) were imputed by chained equations "
            f"with predictive mean matching (PMM; {req.max_iter} iterations; random seed "
            f"{req.random_state}); PMM selected observed donor values with similar model-based "
            "predictions. "
        )
    if cat_targets:
        methods_text += (
            f"Categorical variables ({', '.join(cat_targets)}) were imputed using the most "
            "frequent observed category. "
        )
    if new_column_map:
        methods_text += (
            "The original variables were retained and the completed values were stored in new "
            "variables: "
            + ", ".join(f"{src} as {dst}" for src, dst in new_column_map.items())
            + ". "
        )
    methods_text += (
        f"In total, {total_imputed} missing values were imputed. This procedure produced a single "
        "completed dataset and therefore does not incorporate between-imputation uncertainty; "
        "pooled multiple imputation with Rubin's rules should be used for variance-correct inference."
    )

    export_rows = [["Column", "Method", "N Imputed", "Mean / Mode", "Min", "Max"]]
    for s in col_summaries:
        center = s.get("mode_imputed") if s.get("method") == "mode" else s.get("mean_imputed")
        export_rows.append([s["column"], s.get("method", "MICE"), s["n_imputed"], center, s.get("min_imputed"), s.get("max_imputed")])

    n_imputations = getattr(req, "n_imputations", 1)
    r_code = (
        f"library(mice)\n"
        f"imp <- mice(data[, c({', '.join(repr(c) for c in req.columns)})],\n"
        f"            m = {n_imputations}, maxit = {req.max_iter}, method = 'pmm', seed = {req.random_state})\n"
        f"completed_data <- complete(imp, action = 'long')\n"
        f"# Pool estimates with Rubin's rules:\n"
        f"# fit <- with(imp, lm(outcome ~ predictors))\n"
        f"# pooled <- pool(fit)"
    )

    store.log_action(req.session_id, "mice", {
        "columns": cols_with_missing,
        "numeric_columns": num_targets,
        "categorical_columns": cat_targets,
        "new_column_map": new_column_map,
        "total_imputed": total_imputed,
        "max_iter": req.max_iter,
        "random_state": req.random_state,
        "mechanism": mech_label,
        "single_imputation": True,
        "methods_text": methods_text,
    })

    return {
        "test": "MICE Multiple Imputation",
        "n_total": len(df),
        "total_imputed": total_imputed,
        "columns": col_summaries,
        "n_imputations": n_imputations,
        "max_iter": req.max_iter,
        "new_column_map": new_column_map,
        "preserved_originals": bool(new_column_map),
        "assumptions": assumptions,
        "result_text": result_text,
        "methods_text": methods_text,
        "export_rows": export_rows,
        "r_code": r_code,
        "preview_rows": preview_rows,
    }


def fit_mice(req):
    df = _get_df(req.session_id)

    # Validate columns
    missing_cols = [c for c in req.columns if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing_cols}")

    df_work, missing_masks, cols_with_missing, num_targets, cat_targets, col_summaries = _compute_mice(
        df, req.columns, req.max_iter, req.random_state
    )

    # Keep originals + write imputed values to new "<col>_imp" columns.
    new_column_map: dict = {}
    if getattr(req, "new_columns", False):
        for s in col_summaries:
            c = s["column"]
            newname = f"{c}_imp"
            k, base = 2, newname
            while newname in df_work.columns:
                newname = f"{base}_{k}"
                k += 1
            source_pos = list(df_work.columns).index(c)
            df_work.insert(source_pos + 1, newname, df_work[c].copy())
            df_work[c] = df[c]              # restore the original column
            new_column_map[c] = newname
            s["source_column"] = c
            s["column"] = newname

    store.save(req.session_id, df_work)

    return _build_mice_response(
        df, req, df_work, cols_with_missing, num_targets, cat_targets,
        col_summaries, new_column_map
    )


def mice_preview(req):
    df = _get_df(req.session_id)

    missing_cols = [c for c in req.columns if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing_cols}")

    df_work, missing_masks, cols_with_missing, num_targets, cat_targets, col_summaries = _compute_mice(
        df, req.columns, req.max_iter, req.random_state
    )

    preview_rows = []
    for c in cols_with_missing:
        for idx in missing_masks[c][missing_masks[c]].index:
            preview_rows.append({
                "row_index": int(idx),
                "column": c,
                "imputed_value": _safe(df_work.at[idx, c]),
            })

    resp = _build_mice_response(
        df, req, df_work, cols_with_missing, num_targets, cat_targets,
        col_summaries, {}, preview_rows=preview_rows
    )
    # Prefix result text so the UI clearly shows this is a preview.
    resp["result_text"] = "Preview: " + resp["result_text"]
    resp["preview_only"] = True
    return resp


def mice_transfer(req):
    df = _get_df(req.session_id)

    preview_rows = getattr(req, "preview_rows", None)
    if not preview_rows:
        raise HTTPException(status_code=422, detail="No preview rows provided for transfer")

    n_imputed: Dict[str, int] = {}
    for row in preview_rows:
        col = row.get("column")
        idx = row.get("row_index")
        value = row.get("imputed_value")
        if col is None or idx is None or col not in df.columns:
            continue
        if not _missing_mask(df[col].loc[[idx]]).iloc[0]:
            continue
        df.at[idx, col] = value
        n_imputed[col] = n_imputed.get(col, 0) + 1

    if not n_imputed:
        raise HTTPException(status_code=422, detail="No missing cells matched the previewed rows")

    store.save(req.session_id, df)

    total = sum(n_imputed.values())
    return {
        "n_imputed": n_imputed,
        "total_imputed": total,
        "columns": list(n_imputed.keys()),
        "result_text": f"Transferred {total} previewed value(s) into original columns: {', '.join(n_imputed.keys())}.",
    }


# ── 2. Fine-Gray Competing Risks ────────────────────────────────────────────


def _fine_gray_mi_pooled(df_cols, cols_needed, dur, event, cause, predictors,
                         coerce_fn, n_imputations: int = 10):
    """Fit the Fine-Gray sHR model on m multiply-imputed datasets and pool
    log(sHR) with Rubin's rules. Returns the same shape as _fine_gray_fit so the
    UI renders it unchanged, plus mi_pooled / fmi metadata."""
    import math as _math
    from services.missing_data import mice_multiple, pool_rubin_terms

    imp = mice_multiple(df_cols, cols_needed, n_imputations=n_imputations)
    per: list = []
    base = None
    for dfi in imp.imputed_datasets:
        rr = _fine_gray_fit(coerce_fn(dfi), dur, event, cause, list(predictors))
        if rr is None:
            continue
        base = base or rr
        terms = {c["variable"]: (c.get("estimate"), c.get("se"))
                 for c in rr["coefficients"] if c.get("estimate") is not None and c.get("se")}
        if terms:
            per.append(terms)
    if not per or base is None:
        return None

    pooled = pool_rubin_terms(per)
    coefs = []
    for var, pv in pooled.items():
        beta, se = pv["coef"], pv["se"]
        coefs.append({
            "variable": var,
            "estimate": round(beta, 6),
            "shr": round(_math.exp(beta), 4),
            "se": round(se, 6),
            "z": round(pv["t"], 4),
            "p": round(pv["p"], 6) if pv["p"] is not None else None,
            "ci_low": round(pv["ci_low"], 4),
            "ci_high": round(pv["ci_high"], 4),
            "shr_low": round(_math.exp(pv["ci_low"]), 4),
            "shr_high": round(_math.exp(pv["ci_high"]), 4),
            "fmi": pv["fmi"],
        })
    out = dict(base)
    out["coefficients"] = coefs
    out["mi_pooled"] = True
    out["n_imputations"] = len(per)
    out["method_note"] = (
        f"Multiple imputation: {len(per)} chained-PMM datasets pooled by Rubin's "
        "rules on the log-subdistribution-hazard scale. " + str(base.get("method_note", ""))
    )
    return out


def fit_fine_gray(req):
    df = _get_df(req.session_id)

    # Reject impossible follow-up times / non-binary event flag up front. The
    # fitter previously accepted fu_days = -10 (and plotted a CIF from it),
    # which the QA wave 1 audit flagged as CRITICAL — competing-risks output
    # was silently wrong on cohorts with one stray bad row.
    from services.survival_validation import validate_survival_inputs
    validate_survival_inputs(df, req.duration_col, req.event_col, require_binary_event=False)
    if req.group_col and req.group_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.group_col}' not found")

    from lifelines import AalenJohansenFitter

    work = df[[req.duration_col, req.event_col] + ([req.group_col] if req.group_col else [])].dropna()
    durations = work[req.duration_col].values.astype(float)
    events = work[req.event_col].values.astype(int)

    # Unique event types (0 = censored, others are event types). Cast to
    # plain int — numpy.int64 is not JSON-serialisable by FastAPI's encoder.
    event_types = sorted(int(e) for e in np.unique(events) if e != 0)
    if req.event_of_interest not in event_types:
        raise HTTPException(status_code=422, detail=f"Event of interest {req.event_of_interest} not found. Available: {event_types}")

    # Build CIF curves
    cif_data = {}
    event_counts = {}

    if req.group_col:
        groups = sorted(work[req.group_col].unique())
    else:
        groups = ["All"]

    colors = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]

    traces = []
    for gi, group in enumerate(groups):
        if req.group_col:
            mask = work[req.group_col] == group
            dur_g = durations[mask]
            ev_g = events[mask]
        else:
            dur_g = durations
            ev_g = events

        ajf = AalenJohansenFitter()
        ajf.fit(dur_g, ev_g, event_of_interest=req.event_of_interest)

        cif = ajf.cumulative_density_
        col_name = cif.columns[0]
        times = cif.index.tolist()
        probs = cif[col_name].tolist()

        label = f"CIF - {group}" if req.group_col else "CIF"
        color = colors[gi % len(colors)]

        traces.append({
            "x": [_safe(t) for t in times],
            "y": [_safe(p) for p in probs],
            "type": "scatter",
            "mode": "lines",
            "name": label,
            "line": {"color": color, "width": 2},
        })

        n_event = int(np.sum(ev_g == req.event_of_interest))
        n_competing = int(np.sum((ev_g != 0) & (ev_g != req.event_of_interest)))
        n_censored = int(np.sum(ev_g == 0))
        event_counts[str(group)] = {
            "n": len(dur_g),
            "event_of_interest": n_event,
            "competing_events": n_competing,
            "censored": n_censored,
        }

        cif_data[str(group)] = {
            "cif_at_max": _safe(round(probs[-1], 4)) if probs else None,
        }

    # Gray's test (K-sample comparison) — approximate using log-rank on sub-events
    gray_p = None
    if req.group_col and len(groups) == 2:
        try:
            from lifelines.statistics import logrank_test
            g1_mask = work[req.group_col] == groups[0]
            g2_mask = work[req.group_col] == groups[1]
            # Create binary event: 1 if event of interest, 0 otherwise
            ev_binary = (events == req.event_of_interest).astype(int)
            lr = logrank_test(
                durations[g1_mask], durations[g2_mask],
                ev_binary[g1_mask], ev_binary[g2_mask],
            )
            gray_p = _safe(round(float(lr.p_value), 6))
        except Exception:
            gray_p = None

    plot = {
        "data": traces,
        "layout": {
            "title": f"Cumulative Incidence Function (Event={req.event_of_interest})",
            "xaxis": {"title": req.duration_col, "gridcolor": "#e5e7eb"},
            "yaxis": {"title": "Cumulative Incidence", "range": [0, 1], "gridcolor": "#e5e7eb"},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "#ffffff",
            "font": {"color": "#374151", "size": 12},
            "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.98},
        },
    }

    assumptions = [
        {"name": "Independent censoring", "met": True, "detail": "Competing risks model assumes censoring is non-informative."},
        {"name": "Event types", "met": True, "detail": f"Event types found: {event_types}. Event of interest: {req.event_of_interest}."},
        {"name": "Subdistribution vs cause-specific", "met": True,
         "detail": "Fine-Gray sHR has direct interpretation for cumulative incidence but is not a cause-specific hazard ratio."},
    ]

    n_total = len(work)
    warnings = []
    n_event_interest = int((events == req.event_of_interest).sum())
    if n_event_interest < 20:
        warnings.append(f"Only {n_event_interest} events of interest — Fine-Gray estimates may be unstable.")
    if n_total < 100:
        warnings.append("Small overall sample size for competing risks analysis.")
    result_text = (
        f"Competing risks analysis was performed on {n_total} subjects. "
        f"The cumulative incidence function (CIF) was estimated using the Aalen-Johansen estimator "
        f"for event type {req.event_of_interest}."
    )
    if gray_p is not None:
        result_text += f" Gray's test p = {gray_p}."
    if warnings:
        result_text += " " + " ".join(warnings)

    export_rows = [["Group", "N", "Events of Interest", "Competing Events", "Censored", "CIF at Max Time"]]
    for g in groups:
        ec = event_counts[str(g)]
        export_rows.append([str(g), ec["n"], ec["event_of_interest"], ec["competing_events"], ec["censored"],
                            cif_data[str(g)]["cif_at_max"]])

    r_code = (
        f"library(cmprsk)\n"
        f"library(tidycmprsk)\n\n"
        f"# Cumulative incidence\n"
        f"cif <- cuminc(ftime = data${req.duration_col},\n"
        f"              fstatus = data${req.event_col}"
        + (f",\n              group = data${req.group_col}" if req.group_col else "")
        + f")\n"
        f"plot(cif)\n\n"
        f"# Fine-Gray regression\n"
        f"fg <- crr(ftime = data${req.duration_col},\n"
        f"          fstatus = data${req.event_col},\n"
        f"          failcode = {req.event_of_interest},\n"
        f"          cov1 = model.matrix(~ predictors, data)[,-1])\n"
        f"summary(fg)"
    )

    # ── Subdistribution hazard regression (Fine-Gray 1999 / Geskus 2011) ──
    regression_result: Optional[dict] = None
    if req.predictors:
        missing_preds = [c for c in req.predictors if c not in df.columns]
        if missing_preds:
            raise HTTPException(status_code=422, detail=f"Predictor columns not found: {missing_preds}")
        # Reuse the imputation infrastructure used by other survival endpoints
        from services.impute import apply_imputation
        cols_needed = [req.duration_col, req.event_col] + req.predictors

        def _coerce(d):
            d = d.reset_index(drop=True)
            d[req.duration_col] = pd.to_numeric(d[req.duration_col], errors="coerce")
            d[req.event_col] = pd.to_numeric(d[req.event_col], errors="coerce")
            return d

        if (req.imputation or "listwise") == "mice":
            # Proper multiple imputation: fit the Fine-Gray sHR model on each of
            # m completed datasets and pool log(sHR) with Rubin's rules.
            regression_result = _fine_gray_mi_pooled(
                df[cols_needed], cols_needed, req.duration_col, req.event_col,
                int(req.event_of_interest), list(req.predictors), _coerce,
            )
        else:
            df_reg = _coerce(apply_imputation(df[cols_needed], cols_needed, req.imputation or "listwise"))
            regression_result = _fine_gray_fit(
                df_reg, req.duration_col, req.event_col, int(req.event_of_interest), list(req.predictors),
            )
        if regression_result is not None:
            result_text = (
                result_text
                + f" Fine-Gray subdistribution-hazard regression on {len(req.predictors)} "
                  f"predictor(s) — see the sHR table for details."
            )

    # Audit
    try:
        store.log_action(req.session_id, "fine_gray", {
            "duration_col": req.duration_col,
            "event_col": req.event_col,
            "event_of_interest": int(req.event_of_interest),
            "group_col": req.group_col,
            "n_predictors": len(req.predictors or []),
            "ran_regression": regression_result is not None,
        })
    except Exception:
        pass

    return {
        "test": "Fine-Gray Competing Risks",
        "n": n_total,
        "event_types": event_types,
        "event_of_interest": req.event_of_interest,
        "event_counts": event_counts,
        "cif_data": cif_data,
        "gray_p": gray_p,
        "plot": plot,
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
        "regression_result": regression_result,
    }


# ── 3. E-value ──────────────────────────────────────────────────────────────


def fit_evalue(req):
    # DEPRECATION NOTICE (Phase 6)
    # This endpoint is legacy. Please migrate to the much richer implementation:
    #   POST /api/model_diagnostics/causal_sensitivity
    # which supports better OR→RR conversion, QBA, and consistent response shape.
    est = req.estimate
    ci_lo = req.ci_low
    ci_hi = req.ci_high
    p0 = req.baseline_risk

    # Convert to RR scale
    if req.measure_type.upper() == "OR":
        # OR → RR approximation: RR ≈ OR / (1 - p0 + p0 * OR)
        rr = est / (1 - p0 + p0 * est)
        rr_lo = ci_lo / (1 - p0 + p0 * ci_lo)
        rr_hi = ci_hi / (1 - p0 + p0 * ci_hi)
    elif req.measure_type.upper() == "HR":
        # HR ≈ RR for rare outcomes; use directly
        rr = est
        rr_lo = ci_lo
        rr_hi = ci_hi
    else:
        rr = est
        rr_lo = ci_lo
        rr_hi = ci_hi

    def _evalue(rr_val: float) -> Optional[float]:
        if rr_val is None or rr_val <= 0:
            return None
        if rr_val < 1:
            rr_val = 1 / rr_val
        return round(rr_val + math.sqrt(rr_val * (rr_val - 1)), 4)

    e_point = _evalue(rr)

    # E-value for CI: use the bound closer to 1 (more conservative)
    ci_bound = rr_hi if rr < 1 else rr_lo
    if rr < 1:
        ci_bound = rr_hi
    else:
        ci_bound = rr_lo

    # If CI crosses 1, E-value for CI is 1
    if (rr_lo <= 1 <= rr_hi) or (ci_lo <= 1 <= ci_hi):
        e_ci = 1.0
    else:
        e_ci = _evalue(ci_bound)

    interpretation = (
        f"The E-value is {e_point}. To explain away the observed {req.measure_type} of {est}, "
        f"an unmeasured confounder would need to be associated with both the treatment and outcome "
        f"by a risk ratio of at least {e_point}-fold each, above and beyond the measured covariates."
    )
    if e_ci and e_ci > 1:
        interpretation += (
            f" For the confidence interval limit, the E-value is {e_ci}, "
            f"meaning a confounder of this strength could shift the CI to include the null."
        )

    assumptions = [
        {"name": "Sufficient adjustment", "met": True,
         "detail": "E-value quantifies the minimum confounding strength needed to explain away the result."},
        {"name": "Rare outcome", "met": req.measure_type.upper() != "OR" or p0 < 0.15,
         "detail": f"OR→RR conversion accuracy depends on baseline risk ({p0}). Best when <15%."},
    ]

    result_text = (
        f"The E-value for the observed {req.measure_type} of {est} "
        f"(95% CI: {ci_lo}–{ci_hi}) was {e_point}. "
        f"The E-value for the confidence interval bound was {e_ci}."
    )

    export_rows = [
        ["Metric", "Value"],
        [f"Observed {req.measure_type}", est],
        ["95% CI", f"{ci_lo} – {ci_hi}"],
        ["RR (converted)", round(rr, 4)],
        ["E-value (point)", e_point],
        ["E-value (CI)", e_ci],
    ]

    r_code = (
        f"library(EValue)\n\n"
        f"# E-value calculation\n"
        f'evalues.{req.measure_type}({est}, lo = {ci_lo}, hi = {ci_hi}'
        + (', rare = FALSE)' if req.measure_type.upper() == "OR" else ')')
        + f"\n\n"
        f"# Interpretation: An unmeasured confounder would need\n"
        f"# RR >= {e_point} with both treatment & outcome to explain\n"
        f"# away the observed effect."
    )

    return {
        "test": "E-value (Unmeasured Confounding)",
        "estimate": est,
        "measure_type": req.measure_type,
        "ci": [ci_lo, ci_hi],
        "rr_converted": _safe(round(rr, 4)),
        "evalue_point": e_point,
        "evalue_ci": e_ci,
        "e_value_point_estimate": e_point,
        "e_value_ci": e_ci,
        "interpretation": interpretation,
        "assumptions": assumptions,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ── 3b. Causal Sensitivity Analysis (E-value + QBA + bounds) ────────────────


def fit_causal_sensitivity(req):
    """
    Causal sensitivity suite: E-value, QBA, Manski bounds, Rosenbaum bounds,
    multi-confounder scenarios, SMD E-value, and negative-control screening.
    """
    measure = (req.measure or "rr").lower()
    if measure not in {"rr", "or", "hr"}:
        raise HTTPException(status_code=422, detail="measure must be rr, or, or hr")
    if req.ci_low is not None and req.ci_high is not None and req.ci_low >= req.ci_high:
        raise HTTPException(status_code=400, detail="ci_low must be < ci_high")

    from services.causal_sensitivity import (
        e_value,
        e_value_for_smd,
        manski_bounds_binary,
        manski_bounds_from_data,
        multi_confounder_sensitivity,
        negative_control_analysis,
        quantitative_bias_analysis,
        rosenbaum_bounds_from_matched_data,
    )

    ev = e_value(
        estimate=req.observed_estimate,
        ci_low=req.ci_low,
        ci_high=req.ci_high,
        measure=measure,
        rare_outcome=req.rare_outcome,
        baseline_risk=req.baseline_risk,
    )
    qba = quantitative_bias_analysis(
        observed_estimate=req.observed_estimate,
        measure=measure,
        confounding_strength=req.confounding_strength,
        prevalence_exposed=req.prevalence_exposed,
        prevalence_unexposed=req.prevalence_unexposed,
    )
    multi = multi_confounder_sensitivity(
        req.observed_estimate,
        req.unmeasured_confounders,
        measure=measure,
    ) if req.unmeasured_confounders else {"available": False, "reason": "No unmeasured_confounders array supplied."}
    smd_ev = e_value_for_smd(req.smd, baseline_risk=req.baseline_risk or 0.1) if req.smd is not None else {
        "available": False,
        "reason": "No SMD supplied.",
    }

    manski: Dict[str, Any]
    rosenbaum: Dict[str, Any] = {"applicable": False, "reason": "No matched data columns supplied."}
    negative_control: Dict[str, Any] = {"available": False, "reason": "No negative control outcome supplied."}
    df = None
    if req.session_id:
        df = _get_df(req.session_id)

    if df is not None and req.treatment_col and req.outcome_col:
        needed = [req.treatment_col, req.outcome_col]
        if req.match_id_col:
            needed.append(req.match_id_col)
        if req.negative_control_outcome_col:
            needed.append(req.negative_control_outcome_col)
        needed.extend(req.negative_control_covariates or [])
        missing = [c for c in set(needed) if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {sorted(missing)}")
        from services.impute import apply_imputation
        work = apply_imputation(df[list(dict.fromkeys(needed))], list(dict.fromkeys(needed)), req.imputation or "listwise")
        manski = manski_bounds_from_data(
            work,
            req.treatment_col,
            req.outcome_col,
            monotone_treatment_response=req.monotone_treatment_response,
        )
        if req.match_id_col:
            rosenbaum = rosenbaum_bounds_from_matched_data(
                work,
                req.match_id_col,
                req.treatment_col,
                req.outcome_col,
                gamma_max=req.rosenbaum_gamma_max,
                n_gamma=req.rosenbaum_n_gamma,
            )
        if req.negative_control_outcome_col:
            negative_control = negative_control_analysis(
                work,
                req.treatment_col,
                req.negative_control_outcome_col,
                covariates=req.negative_control_covariates,
            )
    elif req.p_y1_treated is not None and req.p_y1_control is not None and req.p_treated is not None:
        manski = manski_bounds_binary(
            p_y1_treated=req.p_y1_treated,
            p_y1_control=req.p_y1_control,
            p_treated=req.p_treated,
            monotone_treatment_response=req.monotone_treatment_response,
        )
        manski["available"] = True
    else:
        manski = {"available": False, "reason": "Supply session_id+treatment_col+outcome_col or Manski probabilities."}

    warnings = []
    if ev.get("e_value_point_estimate", 99) < 2:
        warnings.append("Low E-value (<2); result is sensitive to weak unmeasured confounding.")
    if rosenbaum.get("critical_gamma") is not None and rosenbaum.get("critical_gamma", 99) < 1.5:
        warnings.append("Rosenbaum critical gamma is low; matched result may be sensitive to small hidden bias.")
    if negative_control.get("flag_residual_bias"):
        warnings.append("Negative control outcome is associated with treatment; residual bias signal detected.")

    result_text = (
        f"Causal sensitivity suite: E-value {ev.get('e_value_point_estimate')} for observed "
        f"{measure.upper()}={req.observed_estimate}. QBA corrected estimate {qba.get('bias_corrected_estimate')}."
    )
    if manski.get("available"):
        result_text += f" Manski ATE bounds: {manski.get('ate_bounds')}."

    return {
        "test": "Causal Sensitivity Analysis (E-value + QBA + Partial Identification)",
        "e_value": ev,
        "e_value_smd": smd_ev,
        "quantitative_bias_analysis": qba,
        "multi_confounder_sensitivity": multi,
        "manski_bounds": manski,
        "rosenbaum_bounds": rosenbaum,
        "negative_control_analysis": negative_control,
        "warnings": warnings,
        "assumptions": [
            {"name": "No unmeasured confounding", "met": False,
             "detail": "Sensitivity methods quantify how violations could change inference."},
            {"name": "Manski partial identification", "met": manski.get("available", False),
             "detail": "Bounds avoid exchangeability assumptions but can be wide."},
            {"name": "Rosenbaum matched-pair sensitivity", "met": rosenbaum.get("applicable", False),
             "detail": "Requires clean 1:1 matched binary-outcome pairs."},
        ],
        "result_text": result_text,
        "export_rows": [
            ["Metric", "Value"],
            ["Observed estimate", req.observed_estimate],
            ["Measure", measure],
            ["E-value point", ev.get("e_value_point_estimate")],
            ["E-value CI", ev.get("e_value_ci")],
            ["QBA bias factor", qba.get("bias_factor")],
            ["QBA corrected estimate", qba.get("bias_corrected_estimate")],
            ["Manski ATE bounds", manski.get("ate_bounds")],
            ["Rosenbaum critical gamma", rosenbaum.get("critical_gamma")],
            ["Negative control p", negative_control.get("p")],
        ],
        "r_code": (
            "library(EValue)\n"
            "# E-values: EValue::evalues.RR(est, lo, hi)\n"
            "# Rosenbaum bounds: rbounds::psens(...)\n"
            "# Partial identification: report Manski lower/upper bounds."
        ),
    }


# ── 4. Landmark Analysis ────────────────────────────────────────────────────


def _landmark_cox_mi(df_needed, needed, req, n_imputations: int = 10):
    """Landmark Cox on m multiply-imputed datasets, pooled (Rubin, log-HR).
    Returns (cox_results_list, mi_note) — same row shape as the single-fit path,
    with a per-coefficient FMI added. None if no dataset converged."""
    import math as _math
    from lifelines import CoxPHFitter
    from services.missing_data import mice_multiple, pool_rubin_terms

    preds = list(req.predictors)
    cat_cols = preds + ([req.group_col] if req.group_col and req.group_col not in preds else [])
    cox_cols = [req.duration_col, req.event_col] + preds
    if req.group_col and req.group_col not in preds:
        cox_cols.append(req.group_col)

    imp = mice_multiple(df_needed, needed, n_imputations=n_imputations)
    per: list = []
    for dfi in imp.imputed_datasets:
        w = dfi.copy()
        w[req.duration_col] = pd.to_numeric(w[req.duration_col], errors="coerce")
        w = w[w[req.duration_col] >= req.landmark_time].copy()
        if len(w) < 10:
            continue
        w[req.duration_col] = w[req.duration_col] - req.landmark_time
        cdf = w[[c for c in cox_cols if c in w.columns]].copy()
        for c in cat_cols:
            if c in cdf.columns and cdf[c].dtype == object:
                cdf[c] = pd.Categorical(cdf[c]).codes
        cdf = cdf.dropna()
        try:
            cph = CoxPHFitter()
            cph.fit(cdf, duration_col=req.duration_col, event_col=req.event_col)
        except Exception:
            continue
        terms = {str(var): (float(row["coef"]), float(row["se(coef)"]))
                 for var, row in cph.summary.iterrows()}
        if terms:
            per.append(terms)
    if not per:
        return None, None

    pooled = pool_rubin_terms(per)
    out = [{
        "variable": var,
        "HR": round(_math.exp(pv["coef"]), 4),
        "ci_low": round(_math.exp(pv["ci_low"]), 4),
        "ci_high": round(_math.exp(pv["ci_high"]), 4),
        "p": round(pv["p"], 6) if pv["p"] is not None else None,
        "fmi": pv["fmi"],
    } for var, pv in pooled.items()]
    note = (f"Cox HRs pooled across {len(per)} chained-PMM imputations "
            "(Rubin's rules, log-HR scale).")
    return out, note


def fit_landmark(req):
    df = _get_df(req.session_id)

    for c in [req.duration_col, req.event_col]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")

    needed = [req.duration_col, req.event_col]
    if req.group_col:
        if req.group_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{req.group_col}' not found")
        needed.append(req.group_col)
    if req.predictors:
        for p in req.predictors:
            if p not in df.columns:
                raise HTTPException(status_code=400, detail=f"Predictor '{p}' not found")
            needed.append(p)

    work = df[needed].dropna().copy()
    work[req.duration_col] = pd.to_numeric(work[req.duration_col], errors="coerce")
    work[req.event_col] = pd.to_numeric(work[req.event_col], errors="coerce")
    n_total = len(work)
    from services.survival_validation import validate_survival_inputs
    surv = validate_survival_inputs(work, req.duration_col, req.event_col, mode="drop_with_warning")
    work = surv.df

    # Apply landmark: exclude subjects who had event before landmark
    lm_mask = work[req.duration_col] >= req.landmark_time
    lm_df = work[lm_mask].copy()
    n_landmark_excluded = len(work) - len(lm_df)
    n_excluded = n_total - len(lm_df)

    if len(lm_df) < 10:
        raise HTTPException(status_code=422,
                            detail=f"Only {len(lm_df)} subjects survived beyond landmark time {req.landmark_time}. Need at least 10.")

    # Shift time origin
    lm_df[req.duration_col] = lm_df[req.duration_col] - req.landmark_time

    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test

    colors = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
    traces = []
    km_summaries = {}

    if req.group_col:
        groups = sorted(lm_df[req.group_col].unique())
    else:
        groups = ["All"]

    for gi, group in enumerate(groups):
        if req.group_col:
            g_df = lm_df[lm_df[req.group_col] == group]
        else:
            g_df = lm_df

        kmf = KaplanMeierFitter()
        kmf.fit(g_df[req.duration_col], g_df[req.event_col])

        sf = kmf.survival_function_
        col_name = sf.columns[0]
        times = sf.index.tolist()
        probs = sf[col_name].tolist()

        label = f"KM - {group}" if req.group_col else "KM"
        color = colors[gi % len(colors)]

        traces.append({
            "x": [_safe(t) for t in times],
            "y": [_safe(p) for p in probs],
            "type": "scatter",
            "mode": "lines",
            "name": label,
            "line": {"color": color, "width": 2},
        })

        median_surv = kmf.median_survival_time_
        km_summaries[str(group)] = {
            "n": len(g_df),
            "events": int(g_df[req.event_col].sum()),
            "median_survival": _safe(round(float(median_surv), 2)) if not math.isinf(median_surv) else None,
        }

    # Log-rank test between groups
    logrank_p = None
    if req.group_col and len(groups) == 2:
        g1 = lm_df[lm_df[req.group_col] == groups[0]]
        g2 = lm_df[lm_df[req.group_col] == groups[1]]
        try:
            lr = logrank_test(
                g1[req.duration_col], g2[req.duration_col],
                g1[req.event_col], g2[req.event_col],
            )
            logrank_p = _safe(round(float(lr.p_value), 6))
        except Exception:
            logrank_p = None

    # Cox regression if predictors given
    cox_results = None
    cox_mi_note = None
    if req.predictors and (getattr(req, "imputation", None) or "listwise") == "mice":
        # Proper multiple imputation: pool the landmark Cox over m datasets.
        cox_results, cox_mi_note = _landmark_cox_mi(df[needed], needed, req)
    if req.predictors and cox_results is None:
        try:
            cox_cols = [req.duration_col, req.event_col] + req.predictors
            if req.group_col and req.group_col not in req.predictors:
                cox_cols.append(req.group_col)
            cox_df = lm_df[cox_cols].copy()

            # Encode categorical predictors
            for c in req.predictors + ([req.group_col] if req.group_col else []):
                if c in cox_df.columns and cox_df[c].dtype == object:
                    cox_df[c] = pd.Categorical(cox_df[c]).codes

            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col=req.duration_col, event_col=req.event_col)

            cox_results = []
            for _, row in cph.summary.iterrows():
                cox_results.append({
                    "variable": row.name,
                    "HR": _safe(round(float(row["exp(coef)"]), 4)),
                    "ci_low": _safe(round(float(row["exp(coef) lower 95%"]), 4)),
                    "ci_high": _safe(round(float(row["exp(coef) upper 95%"]), 4)),
                    "p": _safe(round(float(row["p"]), 6)),
                })
        except Exception as exc:
            cox_results = [{"error": str(exc)}]

    plot = {
        "data": traces,
        "layout": {
            "title": f"Landmark Analysis (t ≥ {req.landmark_time})",
            "xaxis": {"title": f"Time from landmark ({req.landmark_time})", "gridcolor": "#e5e7eb"},
            "yaxis": {"title": "Survival Probability", "range": [0, 1.05], "gridcolor": "#e5e7eb"},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "#ffffff",
            "font": {"color": "#374151", "size": 12},
            "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.02, "yanchor": "bottom"},
        },
    }

    assumptions = [
        {"name": "Landmark exclusion", "met": True,
         "detail": f"{n_landmark_excluded} subjects excluded (event or censored before t={req.landmark_time})."},
        {"name": "Sufficient sample", "met": len(lm_df) >= 20,
         "detail": f"{len(lm_df)} subjects remain after landmark."},
        {"name": "Landmark selection bias", "met": True,
         "detail": "Landmark analysis conditions on survival to the landmark time; results do not apply to patients who failed earlier."},
    ]
    if logrank_p is not None:
        assumptions.append({"name": "Log-rank test", "met": logrank_p < 0.05,
                            "detail": f"p = {logrank_p}"})

    result_text = (
        f"Landmark analysis at t = {req.landmark_time}: {n_excluded} subjects were excluded "
        f"(event or censored before landmark), leaving {len(lm_df)} subjects for analysis."
    )
    if logrank_p is not None:
        result_text += f" Log-rank test p = {logrank_p}."

    export_rows = [["Group", "N", "Events", "Median Survival"]]
    for g in groups:
        s = km_summaries[str(g)]
        export_rows.append([str(g), s["n"], s["events"], s["median_survival"]])

    preds_str = " + ".join(req.predictors) if req.predictors else "group"
    r_code = (
        f"library(survival)\n"
        f"library(survminer)\n\n"
        f"# Landmark at t = {req.landmark_time}\n"
        f"lm_data <- data[data${req.duration_col} >= {req.landmark_time}, ]\n"
        f"lm_data${req.duration_col} <- lm_data${req.duration_col} - {req.landmark_time}\n\n"
        f"# KM curves\n"
        f"fit <- survfit(Surv({req.duration_col}, {req.event_col}) ~ "
        + (req.group_col if req.group_col else "1")
        + f", data = lm_data)\n"
        f"ggsurvplot(fit, data = lm_data, risk.table = TRUE)\n\n"
        f"# Cox regression\n"
        f"cox <- coxph(Surv({req.duration_col}, {req.event_col}) ~ {preds_str}, data = lm_data)\n"
        f"summary(cox)"
    )

    return {
        "test": "Landmark Survival Analysis",
        "landmark_time": req.landmark_time,
        "n_total": n_total,
        "n_excluded": n_excluded,
        "n_invalid_survival": int(surv.n_excluded),
        "warnings": surv.warnings,
        "n_landmark": len(lm_df),
        "km_summaries": km_summaries,
        "logrank_p": logrank_p,
        "cox_results": cox_results,
        "cox_mi_note": cox_mi_note,
        "plot": plot,
        "assumptions": assumptions,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ── 5. Restricted Mean Survival Time (RMST) ─────────────────────────────────
#
# Robust alternative to the hazard-ratio framework — does NOT require the
# proportional-hazards assumption. RMST(τ) is the area under the survival
# curve from 0 to τ, equal to the average event-free time over the horizon
# (e.g. "average years alive in 0-5 years"). For two groups the difference
# Δ = RMST_A(τ) − RMST_B(τ) is a clinically interpretable summary with a
# proper SE (Royston & Parmar 2013) → z-test + 95 % CI + p.


def _rmst_mi_pool(df_full, req, n_imputations: int = 10):
    """Pool RMST(τ) per group and ΔRMST contrasts over m imputations of the
    GROUP covariate. The outcome (time/event) is kept complete-case — outcome
    values are never imputed (Royston/White caution). Returns
    (rmst_by_group, contrasts, note) or (None, None, None) when there is nothing
    to multiply-impute."""
    from services.missing_data import mice_multiple, pool_rubin_terms

    gc = req.group_col
    base = df_full[[req.duration_col, req.event_col, gc]].copy()
    base[req.duration_col] = pd.to_numeric(base[req.duration_col], errors="coerce")
    base[req.event_col] = pd.to_numeric(base[req.event_col], errors="coerce")
    base = base.dropna(subset=[req.duration_col, req.event_col])  # outcome complete-case
    from services.survival_validation import validate_survival_inputs
    base = validate_survival_inputs(base, req.duration_col, req.event_col, mode="drop_with_warning").df
    if not (base[gc].isna() | (base[gc].astype(str).str.strip() == "")).any():
        return None, None, None  # group fully observed → nothing to MI

    imp = mice_multiple(base, [gc], n_imputations=n_imputations)
    per_rmst: list = []
    per_contrast: list = []
    counts: dict = {}  # group → list of (n, n_events) across imputations
    for dfi in imp.imputed_datasets:
        d = dfi.dropna(subset=[req.duration_col, req.event_col, gc])
        t = d[req.duration_col].to_numpy(dtype=float)
        e = d[req.event_col].to_numpy(dtype=int)
        gvals = d[gc].to_numpy()
        gs = sorted(pd.unique(d[gc].dropna()).tolist(), key=lambda x: (isinstance(x, str), x))
        rg = {
            str(g): _rmst_one_group_raw(
                t[gvals == g],
                e[gvals == g],
                float(req.tau),
            )
            for g in gs
        }
        for k, v in rg.items():
            counts.setdefault(k, []).append((v["n"], v["n_events"]))
        per_rmst.append({k: (v["rmst"], v["se"]) for k, v in rg.items()})
        cdict = {}
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                a, b = rg[str(gs[i])], rg[str(gs[j])]
                cdict[f"{gs[i]}|{gs[j]}"] = (a["rmst"] - b["rmst"],
                                             float(np.sqrt(a["se"] ** 2 + b["se"] ** 2)))
        per_contrast.append(cdict)

    pooled_rmst = pool_rubin_terms(per_rmst)
    rmst_by_group = {}
    for g, pv in pooled_rmst.items():
        ns = counts.get(g, [(0, 0)])
        rmst_by_group[g] = {
            "n": int(round(float(np.mean([c[0] for c in ns])))),
            "n_events": int(round(float(np.mean([c[1] for c in ns])))),
            "rmst": round(pv["coef"], 4), "se": round(pv["se"], 4),
            "ci_low": round(pv["ci_low"], 4), "ci_high": round(pv["ci_high"], 4),
            "fmi": pv["fmi"],
        }
    contrasts = []
    for key, pv in pool_rubin_terms(per_contrast).items():
        a, b = key.split("|")
        contrasts.append({"group_a": a, "group_b": b, "delta_rmst": round(pv["coef"], 4),
                          "se": round(pv["se"], 4), "z": round(pv["t"], 4),
                          "p": round(pv["p"], 6) if pv["p"] is not None else None,
                          "p_raw": pv["p"],
                          "ci_low": round(pv["ci_low"], 4), "ci_high": round(pv["ci_high"], 4),
                          "fmi": pv["fmi"]})
    note = (f"RMST and ΔRMST pooled across {len(imp.imputed_datasets)} imputations of the "
            "group covariate (Rubin's rules; time/event kept complete-case).")
    return rmst_by_group, contrasts, note


def fit_rmst(req):
    if req.tau is None or req.tau <= 0:
        raise HTTPException(status_code=422, detail="tau must be > 0.")

    df_full = _get_df(req.session_id)
    for c in [req.duration_col, req.event_col]:
        if c not in df_full.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if req.group_col and req.group_col not in df_full.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.group_col}' not found")

    cols_needed = [req.duration_col, req.event_col] + ([req.group_col] if req.group_col else [])
    from services.impute import apply_imputation
    # For MICE the descriptive KM plot + group set stay complete-case (single
    # imputation would turn a discrete group into fractional values); the pooled
    # RMST/ΔRMST numbers come from _rmst_mi_pool below.
    desc_strategy = "listwise" if (req.imputation or "listwise") == "mice" else (req.imputation or "listwise")
    df = apply_imputation(df_full[cols_needed], cols_needed, desc_strategy).reset_index(drop=True)
    df[req.duration_col] = pd.to_numeric(df[req.duration_col], errors="coerce")
    df[req.event_col] = pd.to_numeric(df[req.event_col], errors="coerce")
    df = df.dropna()
    from services.survival_validation import validate_survival_inputs
    surv = validate_survival_inputs(df, req.duration_col, req.event_col, mode="drop_with_warning")
    df = surv.df
    if len(df) < 5:
        raise HTTPException(status_code=400, detail=f"Not enough complete rows (need ≥ 5, got {len(df)}).")

    rmst_warnings: list[str] = list(surv.warnings)

    t_all = df[req.duration_col].values.astype(float)
    if req.tau > float(t_all.max()):
        raise HTTPException(
            status_code=422,
            detail=f"tau = {req.tau} exceeds the maximum observed time ({t_all.max():.3f}). "
                   "Pick a horizon within the observed follow-up.",
        )
    e_all = df[req.event_col].values.astype(int)
    if set(np.unique(e_all)) - {0, 1}:
        raise HTTPException(status_code=422, detail="Event column must be binary 0/1.")

    groups: List[Any] = []
    if req.group_col:
        groups = sorted(df[req.group_col].dropna().unique().tolist(), key=lambda x: (isinstance(x, str), x))

    rmst_by_group: Dict[str, dict] = {}
    raw_rmst_by_group: Dict[str, dict] = {}
    if not groups:
        raw_rmst_by_group["All"] = _rmst_one_group_raw(
            t_all, e_all, float(req.tau)
        )
        rmst_by_group["All"] = _format_rmst_result(
            raw_rmst_by_group["All"]
        )
    else:
        for g in groups:
            mask = df[req.group_col] == g
            raw_rmst_by_group[str(g)] = _rmst_one_group_raw(
                t_all[mask], e_all[mask], float(req.tau)
            )
            rmst_by_group[str(g)] = _format_rmst_result(
                raw_rmst_by_group[str(g)]
            )

    # Pairwise contrasts when groups present
    contrasts: List[dict] = []
    if len(groups) >= 2:
        z95 = 1.959963984540054
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a = raw_rmst_by_group[str(groups[i])]
                b = raw_rmst_by_group[str(groups[j])]
                diff = a["rmst"] - b["rmst"]
                se = float(np.sqrt(a["se"] ** 2 + b["se"] ** 2))
                if se <= 0:
                    p = None
                    lo = hi = diff
                    z = None
                else:
                    z = diff / se
                    p = _two_sided_normal_p(z)
                    lo = diff - z95 * se
                    hi = diff + z95 * se
                contrasts.append({
                    "group_a": str(groups[i]),
                    "group_b": str(groups[j]),
                    "delta_rmst": round(diff, 4),
                    "se": round(se, 4),
                    "z": round(z, 4) if z is not None else None,
                    "p": round(p, 6) if p is not None else None,
                    "p_raw": p,
                    "ci_low": round(lo, 4),
                    "ci_high": round(hi, 4),
                })

    # Multiple imputation of the group covariate → pooled RMST / ΔRMST (the KM
    # plot above stays complete-case / single-dataset and is descriptive only).
    rmst_mi_note = None
    if req.group_col and (req.imputation or "listwise") == "mice":
        mi_rmst, mi_contrasts, rmst_mi_note = _rmst_mi_pool(df_full, req)
        if mi_rmst is not None:
            rmst_by_group = mi_rmst
            contrasts = mi_contrasts

    # Build plot (KM curves capped at tau, with shaded area = RMST per group)
    from lifelines import KaplanMeierFitter
    palette = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
    traces = []
    if not groups:
        kmf = KaplanMeierFitter()
        kmf.fit(t_all, e_all)
        sf = kmf.survival_function_.iloc[:, 0]
        traces.append({
            "x": [0] + [float(t) for t in sf.index.tolist()],
            "y": [1.0] + [float(v) for v in sf.values.tolist()],
            "type": "scatter", "mode": "lines",
            "line": {"color": palette[0], "width": 2, "shape": "hv"},
            "name": f"All (RMST = {rmst_by_group['All']['rmst']})",
        })
    else:
        for gi, g in enumerate(groups):
            mask = df[req.group_col] == g
            kmf = KaplanMeierFitter()
            kmf.fit(t_all[mask], e_all[mask])
            sf = kmf.survival_function_.iloc[:, 0]
            label = f"{g} (RMST = {rmst_by_group[str(g)]['rmst']})"
            traces.append({
                "x": [0] + [float(t) for t in sf.index.tolist()],
                "y": [1.0] + [float(v) for v in sf.values.tolist()],
                "type": "scatter", "mode": "lines",
                "line": {"color": palette[gi % len(palette)], "width": 2, "shape": "hv"},
                "name": label,
            })

    # Vertical line at tau
    shapes = [{
        "type": "line", "xref": "x", "yref": "paper",
        "x0": float(req.tau), "x1": float(req.tau), "y0": 0, "y1": 1,
        "line": {"color": "#9ca3af", "dash": "dash", "width": 1.5},
    }]
    annotations = [{
        "xref": "x", "yref": "paper", "x": float(req.tau), "y": 1.0,
        "xanchor": "left", "yanchor": "top",
        "text": f" τ = {req.tau}",
        "showarrow": False, "font": {"size": 11, "color": "#374151"},
    }]

    plot = {
        "data": traces,
        "layout": {
            "title": f"Restricted Mean Survival Time (τ = {req.tau})",
            "xaxis": {"title": req.duration_col, "gridcolor": "#e5e7eb"},
            "yaxis": {"title": "Survival probability", "range": [0, 1.05], "gridcolor": "#e5e7eb"},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "#ffffff",
            "font": {"color": "#374151", "size": 12},
            "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.05},
            "shapes": shapes,
            "annotations": annotations,
        },
    }

    n_total = len(df)
    if groups and len(groups) == 2 and contrasts:
        c0 = contrasts[0]
        result_text = (
            f"Restricted mean survival time on n = {n_total} subjects at τ = {req.tau}. "
            f"{c0['group_a']}: {rmst_by_group[c0['group_a']]['rmst']} "
            f"(95% CI {rmst_by_group[c0['group_a']]['ci_low']}–{rmst_by_group[c0['group_a']]['ci_high']}). "
            f"{c0['group_b']}: {rmst_by_group[c0['group_b']]['rmst']} "
            f"(95% CI {rmst_by_group[c0['group_b']]['ci_low']}–{rmst_by_group[c0['group_b']]['ci_high']}). "
            f"ΔRMST = {c0['delta_rmst']} (95% CI {c0['ci_low']}–{c0['ci_high']}), p = "
            f"{'<0.001' if (c0['p'] is not None and c0['p'] < 0.001) else (round(c0['p'], 3) if c0['p'] is not None else 'N/A')}."
        )
    else:
        result_text = (
            f"Restricted mean survival time on n = {n_total} subjects at τ = {req.tau}. "
            "RMST per group is reported below; the difference is interpretable as the "
            "average event-free time lived during the first τ time units."
        )

    if rmst_warnings:
        result_text += " " + " ".join(rmst_warnings)

    assumptions = [
        {"name": "Censoring at random",      "met": True,
         "detail": "RMST assumes censoring is independent of the event process within each group."},
        {"name": "τ within observed range",  "met": True,
         "detail": f"τ = {req.tau} is at or below the maximum observed time ({float(t_all.max()):.3f})."},
        {"name": "No proportional-hazards required", "met": True,
         "detail": "RMST is a robust, PH-free summary (recommended when hazards cross or treatment effect is delayed)."},
    ]

    if req.tau > 0.9 * float(t_all.max()):
        rmst_warnings.append("τ is close to the maximum follow-up — RMST estimate has higher uncertainty in the tail.")
    if len(df) < 50:
        rmst_warnings.append("Small sample size — consider reporting median survival or Kaplan-Meier curves alongside RMST.")

    export_rows = [["Group", "n", "Events", "RMST (τ)", "SE", "95% CI low", "95% CI high"]]
    for g_label, gv in rmst_by_group.items():
        export_rows.append([
            g_label, gv["n"], gv["n_events"], gv["rmst"], gv["se"], gv["ci_low"], gv["ci_high"],
        ])
    if contrasts:
        export_rows.append([])
        export_rows.append(["Group A", "Group B", "ΔRMST", "SE", "95% CI low", "95% CI high", "p"])
        for c in contrasts:
            export_rows.append([
                c["group_a"], c["group_b"], c["delta_rmst"], c["se"], c["ci_low"], c["ci_high"], c["p"],
            ])

    r_code = (
        "library(survRM2)\n"
        f"# data: time={req.duration_col}, status={req.event_col}"
        + (f", group={req.group_col}" if req.group_col else "")
        + f", tau={req.tau}\n"
        + (f"rmst2(time = data${req.duration_col}, status = data${req.event_col}, "
           f"arm = data${req.group_col}, tau = {req.tau})\n"
           if req.group_col else
           f"library(survival)\nfit <- survfit(Surv({req.duration_col}, {req.event_col}) ~ 1, data = data)\n"
           f"# RMST via integration of S(t) on [0, {req.tau}]\n")
    )

    try:
        store.log_action(req.session_id, "rmst", {
            "duration_col": req.duration_col,
            "event_col": req.event_col,
            "tau": float(req.tau),
            "group_col": req.group_col,
            "n_groups": len(groups) if groups else 1,
        })
    except Exception:
        pass

    return {
        "test": "Restricted Mean Survival Time",
        "n": n_total,
        "n_invalid_survival": int(surv.n_excluded),
        "tau": float(req.tau),
        "rmst_by_group": rmst_by_group,
        "contrasts": contrasts,
        "rmst_mi_note": rmst_mi_note,
        "plot": plot,
        "assumptions": assumptions,
        "warnings": rmst_warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ── 6. Recurrent events — LWYY (Lin-Wei-Yang-Ying) model ─────────────────────
#
# The LWYY marginal rate/mean model is a modified Andersen-Gill Cox model for
# RECURRENT events (e.g. repeat heart-failure hospitalisations): a single Cox
# fit on the counting-process (start, stop, event) data with a robust
# cluster-sandwich variance estimator clustered on subject id. The point
# estimate equals Andersen-Gill; the Lin-Wei-Yang-Ying (2000) contribution is
# the cluster-robust SE that accounts for within-subject correlation of events.
#
# Implemented with lifelines CoxPHFitter using entry_col = start, duration_col
# = stop, cluster_col = id, robust = True — mathematically the LWYY estimator.
# exp(beta) is interpreted as a rate ratio (ratio of the event RATES / mean
# cumulative functions), not a single-event hazard ratio.


def fit_recurrent_lwyy(req):
    from lifelines import CoxPHFitter

    df = _get_df(req.session_id)
    model_type = (req.model_type or "lwyy").lower()
    if model_type not in {"lwyy", "wlw", "both", "mcf_only"}:
        raise HTTPException(status_code=422, detail="model_type must be lwyy, wlw, both, or mcf_only.")
    time_scale = (req.time_scale or "total").lower()
    if time_scale not in {"total", "gap", "calendar"}:
        raise HTTPException(status_code=422, detail="time_scale must be total, gap, or calendar.")

    needed = [req.id_col, req.start_col, req.stop_col, req.event_col, *req.predictors]
    if req.group_col:
        needed.append(req.group_col)
    if req.event_order_col:
        needed.append(req.event_order_col)
    if req.terminal_time_col:
        needed.append(req.terminal_time_col)
    if req.terminal_event_col:
        needed.append(req.terminal_event_col)
    needed = list(dict.fromkeys(needed))
    for c in needed:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found")
    if not req.predictors and model_type != "mcf_only":
        raise HTTPException(status_code=422, detail="Select at least one predictor.")

    from services.impute import apply_imputation
    work = apply_imputation(df[needed], needed, req.imputation or "listwise").reset_index(drop=True)

    # Coerce the counting-process columns.
    numeric_cols = [req.start_col, req.stop_col, req.event_col]
    if req.event_order_col:
        numeric_cols.append(req.event_order_col)
    if req.terminal_time_col:
        numeric_cols.append(req.terminal_time_col)
    if req.terminal_event_col:
        numeric_cols.append(req.terminal_event_col)
    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[req.id_col, req.start_col, req.stop_col, req.event_col])
    work = work[work[req.stop_col] > work[req.start_col]]
    work = _apply_recurrent_time_scale(work, req.start_col, req.stop_col, time_scale)
    if len(work) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough usable intervals (need ≥ 10, got {len(work)}).")
    evset = set(np.unique(work[req.event_col].astype(int)))
    if evset - {0, 1}:
        raise HTTPException(status_code=422, detail="Event column must be 0/1 per interval (1 = event at the interval's stop time).")
    work["__event_order__"] = _recurrent_event_order(
        work, req.id_col, req.stop_col, req.event_col, req.event_order_col
    )

    # Encode predictors: numeric stays, categorical → dummies (drop_first).
    pred_raw = work[req.predictors].copy()
    numeric_pred, cat_pred = [], []
    for c in req.predictors:
        col = pred_raw[c]
        if pd.api.types.is_numeric_dtype(col):
            numeric_pred.append(c)
        else:
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().mean() >= 0.8 and coerced.dropna().nunique() > 2:
                pred_raw[c] = coerced
                numeric_pred.append(c)
            else:
                cat_pred.append(c)
    num_part = pred_raw[numeric_pred].apply(pd.to_numeric, errors="coerce") if numeric_pred else pd.DataFrame(index=pred_raw.index)
    cat_part = pd.get_dummies(pred_raw[cat_pred], drop_first=True, dummy_na=False) if cat_pred else pd.DataFrame(index=pred_raw.index)
    enc = pd.concat([num_part, cat_part], axis=1).astype(float)
    cov_cols = [str(c) for c in enc.columns]
    if not cov_cols and model_type != "mcf_only":
        raise HTTPException(status_code=422, detail="No usable predictors after encoding.")

    fit_df = pd.concat([
        work[[req.id_col, "__re_start__", "__re_stop__", req.event_col, "__event_order__"]].reset_index(drop=True),
        enc.reset_index(drop=True),
    ], axis=1).dropna()

    lwyy_result: Optional[Dict[str, Any]] = None
    wlw_result: Optional[Dict[str, Any]] = None
    cph: Optional[Any] = None
    if model_type in {"lwyy", "both"}:
        cph = CoxPHFitter()
        try:
            cph.fit(
                fit_df[[req.id_col, "__re_start__", "__re_stop__", req.event_col] + cov_cols],
                duration_col="__re_stop__",
                event_col=req.event_col,
                entry_col="__re_start__",
                cluster_col=req.id_col,
                robust=True,
                show_progress=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"LWYY fit failed: {exc}")
        lwyy_result = {
            "model": "Lin-Wei-Yang-Ying marginal rate/means model",
            "time_scale": time_scale,
            "coefficients": _cox_recurrent_coefficients(cph),
            "concordance": round(float(cph.concordance_index_), 4),
        }

    if model_type in {"wlw", "both"}:
        wlw_cph = CoxPHFitter()
        try:
            wlw_cph.fit(
                fit_df[[req.id_col, "__re_start__", "__re_stop__", req.event_col, "__event_order__"] + cov_cols],
                duration_col="__re_stop__",
                event_col=req.event_col,
                entry_col="__re_start__",
                cluster_col=req.id_col,
                strata=["__event_order__"],
                robust=True,
                show_progress=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"WLW fit failed: {exc}")
        wlw_result = {
            "model": "Wei-Lin-Weissfeld marginal model",
            "time_scale": time_scale,
            "event_order_source": req.event_order_col or "computed from subject event history",
            "n_event_strata": int(fit_df["__event_order__"].nunique()),
            "coefficients": _cox_recurrent_coefficients(wlw_cph),
            "concordance": round(float(wlw_cph.concordance_index_), 4),
            "method_note": "WLW fits event-order-specific marginal risk sets with robust subject-clustered variance.",
        }
        if cph is None:
            cph = wlw_cph

    coefs = (lwyy_result or wlw_result or {}).get("coefficients", [])

    n_subjects = int(work[req.id_col].nunique())
    n_events = int((work[req.event_col] == 1).sum())
    # per-subject event counts
    ev_per = work.groupby(req.id_col)[req.event_col].sum()
    # total follow-up = sum over subjects of (max stop − min start)
    grp_fu = work.groupby(req.id_col)
    fu = grp_fu["__re_stop__"].max() - grp_fu["__re_start__"].min()
    total_fu = float(fu.sum())

    # Mean cumulative function — overall + by group.
    palette = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
    traces = []
    rate_traces = []
    mcf_by_group: Dict[str, List[dict]] = {}
    if req.group_col:
        groups = sorted(work[req.group_col].dropna().unique().tolist(), key=lambda x: (isinstance(x, str), x))
        for gi, g in enumerate(groups):
            sub = work[work[req.group_col] == g]
            pts = _mcf(sub, "__re_start__", "__re_stop__", req.event_col)
            mcf_by_group[str(g)] = pts
            traces.append({
                "x": [p["t"] for p in pts], "y": [p["mcf"] for p in pts],
                "type": "scatter", "mode": "lines", "name": f"{req.group_col} = {g}",
                "line": {"color": palette[gi % len(palette)], "width": 2, "shape": "hv"},
            })
            rate_pts = _rate_points_from_mcf(pts)
            rate_traces.append({
                "x": [p["t"] for p in rate_pts], "y": [p["rate"] for p in rate_pts],
                "type": "scatter", "mode": "lines+markers", "name": f"{req.group_col} = {g}",
                "line": {"color": palette[gi % len(palette)], "width": 2},
            })
    else:
        pts = _mcf(work, "__re_start__", "__re_stop__", req.event_col)
        mcf_by_group["overall"] = pts
        traces.append({
            "x": [p["t"] for p in pts], "y": [p["mcf"] for p in pts],
            "type": "scatter", "mode": "lines", "name": "MCF",
            "line": {"color": palette[0], "width": 2, "shape": "hv"},
        })
        rate_pts = _rate_points_from_mcf(pts)
        rate_traces.append({
            "x": [p["t"] for p in rate_pts], "y": [p["rate"] for p in rate_pts],
            "type": "scatter", "mode": "lines+markers", "name": "Rate",
            "line": {"color": palette[0], "width": 2},
        })

    plot = {
        "data": traces,
        "layout": {
            "title": f"Mean cumulative function ({time_scale} time)",
            "xaxis": {"title": f"{time_scale} time", "gridcolor": "#e5e7eb"},
            "yaxis": {"title": "Mean cumulative events", "gridcolor": "#e5e7eb"},
            "paper_bgcolor": "transparent", "plot_bgcolor": "#ffffff",
            "font": {"color": "#374151", "size": 12},
            "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
            "showlegend": bool(req.group_col),
            "legend": {"x": 0.02, "y": 0.98},
        },
    }
    rate_plot = {
        "data": rate_traces,
        "layout": {
            "title": f"Recurrent event rate function ({time_scale} time)",
            "xaxis": {"title": f"{time_scale} time", "gridcolor": "#e5e7eb"},
            "yaxis": {"title": "Incremental event rate", "gridcolor": "#e5e7eb"},
            "paper_bgcolor": "transparent", "plot_bgcolor": "#ffffff",
            "font": {"color": "#374151", "size": 12},
            "margin": {"t": 40, "r": 20, "b": 50, "l": 60},
            "showlegend": bool(req.group_col),
        },
    }

    if coefs:
        primary = coefs[0]
        interp = (
            f"Recurrent-event {model_type.upper()} analysis on {n_subjects} subjects with {n_events} events "
            f"({ev_per.mean():.2f} events/subject; {n_events / total_fu * 100:.2f} events per 100 "
            f"time-units of follow-up, {time_scale} time scale). {primary['variable']}: rate ratio = {primary['rate_ratio']} "
            f"(95% CI {primary['rr_low']}–{primary['rr_high']}, p = "
            f"{'<0.001' if primary['p'] < 0.001 else round(primary['p'], 3)})."
        )
    else:
        interp = (
            f"MCF-only recurrent-event analysis on {n_subjects} subjects with {n_events} events "
            f"({ev_per.mean():.2f} events/subject; {n_events / total_fu * 100:.2f} events per 100 "
            f"time-units of follow-up, {time_scale} time scale)."
        )

    assumptions = [
        {"name": "Recurrent-event structure", "met": True,
         "detail": f"Counting-process intervals (start, stop]; {len(work)} intervals across {n_subjects} subjects."},
        {"name": "Calendar/gap/total time scale", "met": True,
         "detail": f"Model time scale set to '{time_scale}'."},
        {"name": "Rate-ratio interpretation", "met": True,
         "detail": "exp(β) is the ratio of event rates (mean cumulative functions), not a single-event hazard ratio."},
    ]
    if lwyy_result:
        assumptions.append({"name": "Robust variance (LWYY)", "met": True,
                            "detail": "Cluster-robust sandwich SE clustered on subject id accounts for within-subject event correlation."})
    if wlw_result:
        assumptions.append({"name": "WLW marginal risk sets", "met": True,
                            "detail": "Event-order strata are used for Wei-Lin-Weissfeld marginal modeling."})

    export_rows = [["Variable", "Rate ratio", "95% CI low", "95% CI high", "β", "Robust SE", "z", "p"]]
    for c in coefs:
        export_rows.append([c["variable"], c["rate_ratio"], c["rr_low"], c["rr_high"],
                            c["estimate"], c["robust_se"], c["z"], c["p"]])

    negative_binomial = (
        _negative_binomial_recurrent_counts(work, req.id_col, "__re_start__", "__re_stop__", req.event_col, enc)
        if req.include_negative_binomial and cov_cols
        else {"available": False, "reason": "Negative binomial model not requested or no covariates available."}
    )
    informative_censoring = _informative_censoring_diagnostics(
        work,
        req.id_col,
        "__re_start__",
        "__re_stop__",
        req.event_col,
        req.terminal_time_col,
        req.terminal_event_col,
    )
    recurrent_diagnostics = _recurrent_specific_diagnostics(work, req.id_col, req.event_col)
    joint_frailty_spec = {"available": False, "reason": "Joint frailty frailtypack spec not requested."}
    if req.include_joint_frailty_spec:
        if req.terminal_time_col and req.terminal_event_col:
            from services.frailty import build_joint_frailty_frailtypack_spec
            joint_frailty_spec = build_joint_frailty_frailtypack_spec(
                id_col=req.id_col,
                start_col=req.start_col,
                stop_col=req.stop_col,
                recurrent_event_col=req.event_col,
                terminal_time_col=req.terminal_time_col,
                terminal_event_col=req.terminal_event_col,
                predictors=req.predictors,
            )
        else:
            joint_frailty_spec = {
                "available": False,
                "reason": "terminal_time_col and terminal_event_col are required for joint frailty.",
            }

    r_code = (
        "library(survival)\n"
        f"# LWYY marginal rates/means model\n"
        f"fit_lwyy <- coxph(Surv({req.start_col}, {req.stop_col}, {req.event_col}) ~ "
        f"{' + '.join(req.predictors) if req.predictors else '1'} + cluster({req.id_col}), data = data)\n"
        f"summary(fit_lwyy)\n\n"
        f"# WLW marginal model with event-order strata\n"
        f"fit_wlw <- coxph(Surv({req.start_col}, {req.stop_col}, {req.event_col}) ~ "
        f"{' + '.join(req.predictors) if req.predictors else '1'} + strata(event_order) + cluster({req.id_col}), data = data)\n"
        f"summary(fit_wlw)"
    )

    try:
        store.log_action(req.session_id, "recurrent_lwyy", {
            "id_col": req.id_col, "event_col": req.event_col,
            "n_predictors": len(req.predictors), "n_subjects": n_subjects, "n_events": n_events,
        })
    except Exception:
        pass

    return _safe({
        "test": "Recurrent events analysis",
        "model": model_type,
        "time_scale": time_scale,
        "n_subjects": n_subjects,
        "n_events": n_events,
        "n_intervals": int(len(work)),
        "events_per_subject": round(float(ev_per.mean()), 4),
        "total_followup": round(total_fu, 4),
        "concordance": round(float(cph.concordance_index_), 4) if cph is not None else None,
        "coefficients": coefs,
        "lwyy": lwyy_result,
        "wlw": wlw_result,
        "mcf": mcf_by_group,
        "negative_binomial": negative_binomial,
        "informative_censoring_diagnostics": informative_censoring,
        "recurrent_diagnostics": recurrent_diagnostics,
        "joint_frailty": joint_frailty_spec,
        "plot": plot,
        "rate_plot": rate_plot,
        "assumptions": assumptions,
        "result_text": interp,
        "interpretation": interp,
        "export_rows": export_rows,
        "r_code": r_code,
    })


# ── Phase 7: Multi-State Models ──────────────────────────────────────────────
#
# Basic support for illness-death and other multi-state processes.
# Accepts long-format transition data and returns transition-specific models.


def fit_multistate(req):
    df_full = _get_df(req.session_id)

    needed = [
        req.id_col, req.from_state_col, req.to_state_col,
        req.entry_col, req.exit_col, req.event_col
    ] + req.predictors

    missing = [c for c in needed if c not in df_full.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")

    from services.impute import apply_imputation
    work = apply_imputation(df_full[needed], needed, req.imputation or "listwise")

    from services.multistate import fit_multistate_transitions
    result = fit_multistate_transitions(
        work,
        id_col=req.id_col,
        from_state_col=req.from_state_col,
        to_state_col=req.to_state_col,
        entry_col=req.entry_col,
        exit_col=req.exit_col,
        event_col=req.event_col,
        predictors=req.predictors,
        transition_model_type=req.transition_model_type or "cox",
    )

    result["test"] = "Multi-State Transition Models (Phase 7)"
    result["result_text"] = (
        f"Multi-state analysis on {len(work)} transition records. "
        f"Estimated {len(result.get('transitions_estimated', []))} transition(s) using {result.get('model_type', 'cox')} models."
    )
    return result


def fit_dynamic_prediction(req):
    """
    Dynamic multi-state prediction from a landmark time, conditional on being in a specific state.
    Builds on the existing Landmark infrastructure + multi-state transition models (Phase 7).
    """
    df_full = _get_df(req.session_id)

    needed = [
        req.id_col, req.from_state_col, req.to_state_col,
        req.entry_col, req.exit_col, req.event_col
    ] + req.predictors

    missing = [c for c in needed if c not in df_full.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")

    from services.impute import apply_imputation
    work = apply_imputation(df_full[needed], needed, "listwise")

    from services.multistate import dynamic_prediction_from_landmark
    horizon_times = np.linspace(req.landmark_time, req.landmark_time + req.horizon, req.n_points)

    result = dynamic_prediction_from_landmark(
        work,
        landmark_time=req.landmark_time,
        current_state=req.current_state,
        predictors=req.predictors,
        id_col=req.id_col,
        from_state_col=req.from_state_col,
        to_state_col=req.to_state_col,
        entry_col=req.entry_col,
        exit_col=req.exit_col,
        event_col=req.event_col,
        horizon_times=horizon_times,
        transition_model_type=req.transition_model_type or "cox",
        run_bootstrap=req.run_bootstrap or False,
        n_bootstrap=req.n_bootstrap or 50,
        run_microsimulation=req.run_microsimulation or False,
        n_simulations=req.n_simulations or 1000,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Enrich response (Phase 7 A)
    model_name = "Cox (Markov)" if req.transition_model_type == "cox" else "Weibull (Semi-Markov)"
    assumptions = [
        {"name": "Landmark conditioning", "met": True,
         "detail": f"Predictions conditional on being in state {req.current_state} at t={req.landmark_time}."},
        {"name": "Transition Model Type", "met": True,
         "detail": f"Model is fitted using {model_name} framework."},
    ]

    warnings = []
    n_at_risk = result.get("n_at_risk", 0)
    if n_at_risk < 30:
        warnings.append(f"Only {n_at_risk} subjects at risk at landmark — dynamic predictions have high uncertainty.")

    err = result.get("prediction_error", {})
    if err and "overall_mean_error" in err:
        if err["overall_mean_error"] > 1.5:
            warnings.append("High overall prediction error — consider adding more covariates or checking model fit.")

    result_text = (
        f"Dynamic multi-state prediction at landmark t={req.landmark_time} from state {req.current_state}. "
        f"Horizon = {req.horizon} time units using {n_at_risk} subjects still at risk. "
        f"Model type: {model_name}. "
    )
    if err and "overall_mean_error" in err:
        result_text += f"Overall mean squared prediction error = {err['overall_mean_error']}."

    result["test"] = "Dynamic Multi-State Prediction (Phase 7)"
    result["assumptions"] = assumptions
    result["warnings"] = warnings
    result["result_text"] = result_text
    return result


# ── Phase 8: Joint Longitudinal-Survival Models ──────────────────────────────


def fit_joint_model(req):
    """
    Advanced Two-stage joint longitudinal-survival model (Phase 8 Enhanced).
    Supports time-varying counting-process association, multivariate longitudinal,
    and joint latent class modeling.
    """
    long_df = _get_df(req.session_id_long)

    if req.session_id_surv:
        surv_df = _get_df(req.session_id_surv)
    else:
        # Single-session use: the survival part needs ONE row per subject.
        # Passing the long frame through unchanged made every repeated
        # measurement look like a separate patient — it inflated n_subjects and
        # the BIC sample size, and fitted the Cox model on duplicated subjects.
        # duration/event/baseline covariates are subject-constant, so the first
        # record per id is the right collapse.
        if req.id_col not in long_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"id_col '{req.id_col}' not found in the longitudinal session.",
            )
        surv_df = long_df.drop_duplicates(subset=[req.id_col], keep="first").reset_index(drop=True)

    if req.latent_classes > 0:
        from services.joint_model import fit_latent_class_joint_model
        result = fit_latent_class_joint_model(
            long_df,
            surv_df,
            id_col=req.id_col,
            time_col=req.time_col,
            y_cols=req.y_cols,
            long_predictors=req.long_predictors or [],
            surv_predictors=req.surv_predictors or [],
            duration_col=req.duration_col,
            event_col=req.event_col,
            latent_classes=req.latent_classes
        )
        result["test"] = "Joint Latent Class Model (Phase 8 Enhanced)"
        result["result_text"] = (
            f"Joint latent class model fitted on {result['n_subjects']} subjects "
            f"with {result['n_classes']} classes. "
            f"AIC: {result['aic']}, BIC: {result['bic']}."
        )
    else:
        from services.joint_model import fit_time_varying_joint_model
        result = fit_time_varying_joint_model(
            long_df,
            surv_df,
            id_col=req.id_col,
            time_col=req.time_col,
            y_cols=req.y_cols,
            long_predictors=req.long_predictors or [],
            surv_predictors=req.surv_predictors or [],
            duration_col=req.duration_col,
            event_col=req.event_col,
            association=req.association,
            time_spline=req.time_spline
        )
        result["test"] = "Time-Varying Joint Model (Phase 8 Enhanced)"
        assoc_str = ", ".join(req.association)
        result["result_text"] = (
            f"Time-varying joint model fitted on {result['n_subjects']} subjects. "
            f"Association structures: {assoc_str}. "
            f"AIC: {result['aic']}, BIC: {result['bic']}."
        )

    return result


# ── Phase 9: External Validation & Calibration Framework ─────────────────────


def fit_external_validation(req):
    """
    External validation with time-dependent metrics (Phase 9).
    Pass survival_probs (list of lists) at time_points for full IBS + tdAUC.
    """
    df = _get_df(req.session_id)
    # Range guard — same impossible-times check the other survival endpoints use.
    from services.survival_validation import validate_survival_inputs
    validate_survival_inputs(df, req.duration_col, req.event_col)

    surv_probs_arr = np.array(req.survival_probs) if req.survival_probs else None

    from services.external_validation import evaluate_external_validation
    try:
        result = evaluate_external_validation(
            val_df=df,
            duration_col=req.duration_col,
            event_col=req.event_col,
            predicted_lp_col=req.predicted_lp_col,
            survival_probs=surv_probs_arr,
            time_points=np.array(req.time_points) if req.time_points else None,
            dev_metrics=req.dev_metrics,
        )
    except ValueError as exc:
        # evaluate_external_validation signals unusable input with ValueError —
        # here that is a predicted_lp_col the session does not have. Bad column
        # name is the caller's mistake, so surface the reason as a 400 instead
        # of a detail-less 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["test"] = "External Validation & Calibration (Phase 9 - Enhanced)"
    if "error" not in result:
        assumptions = [
            {"name": "Independent censoring", "met": True, "detail": "IPCW assumes non-informative censoring."},
            {"name": "Model transportability", "met": result.get("performance_vs_dev") is None or abs(result.get("performance_vs_dev", {}).get("c_index_drop", 0)) < 0.1, "detail": "Performance drop between dev and val indicates poor transportability."},
        ]
        warnings = []
        if result.get("integrated_brier_score", {}).get("ibs", 0) > 0.25:
            warnings.append("High Integrated Brier Score — predictions may be poorly calibrated on this population.")
        # Without `dev_metrics` (or a second held-out cohort) there is no
        # external comparison: the C-index/calibration numbers describe
        # in-sample fit and are easily mistaken for transportability.
        if not req.dev_metrics:
            warnings.append(
                "No development metrics supplied — reported numbers describe "
                "in-sample fit on the validation_cohort columns provided, NOT "
                "true external validation. Pass `dev_metrics` (or load a "
                "held-out cohort separately) to compute transportability."
            )

        result["assumptions"] = assumptions
        result["warnings"] = warnings
        result["result_text"] = (
            f"External validation on n={result['n_validation']}. "
            f"C-index={result.get('validation_c_index')}. "
            f"IBS={result.get('integrated_brier_score', {}).get('ibs') if 'integrated_brier_score' in result else 'N/A (provide survival_probs)'}."
        )
    return result


# ── Phase 10: Survival ML Benchmark (initial) ────────────────────────────────


def fit_ml_survival_benchmark(req):
    """
    Phase 10 starter: Head-to-head of classical Cox vs practical ML survival model
    (Gradient Boosting ranking) with permutation importance.
    """
    df = _get_df(req.session_id)

    preds = req.predictors or [c for c in df.columns if c not in (req.duration_col, req.event_col)]

    from services.survival_ml import run_survival_ml_benchmark
    try:
        result = run_survival_ml_benchmark(
            df,
            duration_col=req.duration_col,
            event_col=req.event_col,
            predictors=preds,
            n_estimators=req.n_estimators,
            nested_cv=req.nested_cv,
            repeated_cv_repeats=req.repeated_cv_repeats,
            cv_folds=req.cv_folds,
            inner_cv_folds=req.inner_cv_folds,
            hyperparameter_iter=req.hyperparameter_iter,
            include_shap=req.include_shap,
            include_partial_dependence=req.include_partial_dependence,
            include_competing_risks_ml=req.include_competing_risks_ml,
            optimization_method=req.optimization_method,
        )
    except ValueError as exc:
        # The benchmark signals unusable input with ValueError — too few
        # complete rows once the chosen duration/event/predictor columns are
        # combined. That is the caller's data, not a server fault, so return
        # the reason as a 400 rather than a bare 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["test"] = "Survival ML Benchmark (Phase 13 - nested CV, calibration, interpretability)"
    # Prefer the rich result_text produced by the service (Phase 12)
    if not result.get("result_text"):
        result["result_text"] = (
            f"ML vs Cox benchmark on n={result.get('n')} subjects with full Phase 9 integration. "
            "Useful when strong non-linear effects or interactions are suspected."
        )
    return result


# ── 7. Shared Frailty Cox (Phase 6) ─────────────────────────────────────────

#
# Shared frailty for clustered / correlated survival data.
# See services/frailty.py for the implementation details.


def fit_shared_frailty(req):
    df_full = _get_df(req.session_id)

    extra_cluster_cols = [c for c in (req.nested_cluster_cols or []) if c]
    if req.correlated_cluster_col:
        extra_cluster_cols.append(req.correlated_cluster_col)
    needed = [req.duration_col, req.event_col, req.cluster_col] + extra_cluster_cols + req.predictors
    needed = list(dict.fromkeys(needed))
    missing = [c for c in needed if c not in df_full.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")

    from services.impute import apply_imputation
    df = apply_imputation(df_full[needed], needed, req.imputation or "listwise")

    from services.frailty import fit_shared_gamma_frailty
    try:
        result = fit_shared_gamma_frailty(
            df,
            duration_col=req.duration_col,
            event_col=req.event_col,
            cluster_col=req.cluster_col,
            predictors=req.predictors,
            penalizer=req.penalizer,
            frailty_distribution=req.frailty_distribution,
            estimation_method=req.estimation_method,
            nested_cluster_cols=req.nested_cluster_cols or [],
            correlated_cluster_col=req.correlated_cluster_col,
            baseline_hazard=req.baseline_hazard,
            include_diagnostics=req.include_diagnostics,
        )
    except ValueError as exc:
        # The frailty routine signals unusable input (too few clusters, no
        # complete rows, bad distribution name) with ValueError. Those are the
        # caller's problem, not a server fault — surface the reason as a 400 so
        # the UI can show it instead of a bare "failed".
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # The baseline CoxPH fit inside the frailty routine is re-raised as
        # RuntimeError when lifelines cannot converge — collinear, constant or
        # otherwise degenerate predictor columns. Same class of problem: the
        # data the caller chose, so report the reason as a 400.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Make everything JSON-safe (cluster keys can be numpy.int64 etc.)
    def _safe(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, dict):
            return {str(k): _safe(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_safe(x) for x in v]
        return v

    result = _safe(result)

    # Add a small plot of the frailty distribution (posterior means)
    frailties = list(result["cluster_frailties"].values())
    dist_label = result.get("frailty_distribution", req.frailty_distribution).replace("_", " ").title()
    frailty_plot = {
        "data": [{
            "x": sorted(frailties),
            "type": "histogram",
            "name": "Posterior frailties",
            "nbinsx": min(20, max(5, len(frailties) // 3)),
        }],
        "layout": {
            "title": f"Estimated Cluster Frailties ({dist_label} model)",
            "xaxis": {"title": "Frailty multiplier (mean=1)"},
            "yaxis": {"title": "Number of clusters"},
        },
    }

    result["plot"] = frailty_plot
    scatter = result.get("diagnostics", {}).get("frailty_diagnostics", {}).get("scatter", [])
    result["diagnostic_plots"] = {
        "frailty_scatter": {
            "data": [{
                "x": [p["events"] for p in scatter],
                "y": [p["frailty"] for p in scatter],
                "text": [p["cluster"] for p in scatter],
                "type": "scatter",
                "mode": "markers",
                "name": "Cluster frailty",
            }],
            "layout": {
                "title": "Frailty Diagnostics",
                "xaxis": {"title": "Cluster events"},
                "yaxis": {"title": "Frailty multiplier"},
            },
        }
    }
    if result.get("correlated_frailty") and result["correlated_frailty"].get("pairs"):
        pairs = result["correlated_frailty"]["pairs"]
        result["diagnostic_plots"]["correlated_frailty_scatter"] = {
            "data": [{
                "x": [p["frailty_primary"] for p in pairs],
                "y": [p["frailty_secondary"] for p in pairs],
                "text": [p["cluster"] for p in pairs],
                "type": "scatter",
                "mode": "markers",
                "name": "Bivariate frailty",
            }],
            "layout": {
                "title": "Correlated Frailty Scatter",
                "xaxis": {"title": req.cluster_col},
                "yaxis": {"title": req.correlated_cluster_col},
            },
        }
    result["test"] = f"Shared {dist_label} Frailty Cox Model"
    return result
