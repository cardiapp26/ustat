from __future__ import annotations

import asyncio
import traceback
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from loguru import logger

from services import store
from services.category_health import clean_two_level, rare_level_warnings
from services.impute import apply_imputation
from services.psm import (
    _compute_smd,
    _fit_propensity_scores,
    _ks_p,
    _match_greedy,
    _match_optimal,
    _pooled_sd,
    _rosenbaum_bounds,
    _variance_ratio,
)

router = APIRouter()

# ── Request Models ─────────────────────────────────────────────────────────────

class PSMRequest(BaseModel):
    session_id: str
    treatment_col: str
    covariates: List[str]
    outcome_col: Optional[str] = None
    caliper: Optional[float] = 0.2
    caliper_scale: Optional[str] = "logit"
    ratio: Optional[int] = 1
    imputation: Optional[str] = "listwise"
    trim_common_support: Optional[bool] = False
    random_state: Optional[int] = 42
    score_method: Optional[str] = "logistic"
    matching_method: Optional[str] = "greedy"
    exact_match: Optional[List[str]] = None
    outcome_type: Optional[str] = "binary"
    survival_duration_col: Optional[str] = None
    survival_event_col: Optional[str] = None
    compute_rosenbaum: Optional[bool] = False
    rosenbaum_gamma_max: Optional[float] = 3.0


class IPTWRequest(BaseModel):
    session_id: str
    treatment_col: str
    covariates: List[str]
    estimand: str = "ate"
    stabilize: bool = True
    weight_truncation: str = "none"
    weight_truncation_lo: float = 0.01
    weight_truncation_hi: float = 0.99
    weight_truncation_max: float = 10.0
    outcome_type: str = "binary"
    outcome_col: Optional[str] = None
    survival_duration_col: Optional[str] = None
    survival_event_col: Optional[str] = None
    se_method: str = "robust"
    imputation: Optional[str] = "listwise"
    score_method: Optional[str] = "logistic"
    random_state: Optional[int] = 42
    trim_common_support: Optional[bool] = False

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _sanitize_model_error(err: Exception, context: str = "model fitting") -> str:
    msg = str(err)
    if "Singular" in msg or "perfect separation" in msg.lower():
        return "The model encountered perfect separation or singular matrix. Try removing highly correlated predictors."
    if "convergence" in msg.lower() or "failed to converge" in msg.lower():
        return f"{context.capitalize()} failed to converge. Consider increasing iterations or simplifying the model."
    return f"{context.capitalize()} failed. Please check your data and predictors."


def _clean_covariate_categories(df: pd.DataFrame, covariates: List[str]) -> tuple[pd.DataFrame, list]:
    work = df.copy()
    warnings = []
    for col in covariates:
        if col not in work.columns or pd.api.types.is_numeric_dtype(work[col]):
            continue
        cleaned = clean_two_level(work[col])
        work[col] = cleaned.series
        warnings.extend(cleaned.warnings)
    work = work.dropna(subset=[c for c in covariates if c in work.columns])
    warnings.extend(rare_level_warnings(work, covariates))
    return work, warnings


def _run_match_strata(
    df: pd.DataFrame,
    treated_idx: np.ndarray,
    control_idx: np.ndarray,
    distance_vec: np.ndarray,
    caliper_dist: float,
    ratio: int,
    method: str,
    exact_match_cols: Optional[List[str]],
) -> Tuple[list[int], list[int]]:
    match = _match_optimal if method == "optimal" else _match_greedy
    if not exact_match_cols:
        if method == "optimal" and ratio > 1:
            return _match_greedy(treated_idx, control_idx, distance_vec, caliper_dist, ratio)
        if method == "optimal":
            return match(treated_idx, control_idx, distance_vec, caliper_dist)
        return match(treated_idx, control_idx, distance_vec, caliper_dist, ratio)

    keys = df[exact_match_cols].astype(str).agg("||".join, axis=1).values
    matched_t: list[int] = []
    matched_c: list[int] = []
    treated_keys = keys[treated_idx]
    control_keys = keys[control_idx]
    for key in pd.unique(treated_keys):
        t_sub = treated_idx[treated_keys == key]
        c_sub = control_idx[control_keys == key]
        if len(c_sub) == 0:
            continue
        if method == "optimal" and ratio == 1:
            mt, mc = match(t_sub, c_sub, distance_vec, caliper_dist)
        else:
            mt, mc = _match_greedy(t_sub, c_sub, distance_vec, caliper_dist, ratio)
        matched_t.extend(mt)
        matched_c.extend(mc)
    return matched_t, matched_c


# ── PSM Endpoint ──────────────────────────────────────────────────────────────

@router.post("/psm")
async def propensity_score_matching(req: PSMRequest):
    """Main PSM endpoint (mounted at /api/models/psm)."""
    try:
        return await asyncio.to_thread(_run_psm, req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PSM analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _run_psm(req: PSMRequest):
    df_full = _get_df(req.session_id)
    outcome_type = (req.outcome_type or "binary").lower()
    if outcome_type not in ("binary", "survival"):
        raise HTTPException(status_code=422, detail="outcome_type must be 'binary' or 'survival'.")

    extra_outcome_cols: List[str] = []
    if outcome_type == "binary" and req.outcome_col:
        extra_outcome_cols.append(req.outcome_col)
    if outcome_type == "survival":
        if not req.survival_duration_col or not req.survival_event_col:
            raise HTTPException(
                status_code=422,
                detail="Survival outcome requires survival_duration_col and survival_event_col.",
            )
        extra_outcome_cols.extend([req.survival_duration_col, req.survival_event_col])

    exact_match_cols = list(req.exact_match or [])
    needed = list(
        dict.fromkeys(
            [req.treatment_col] + req.covariates + extra_outcome_cols + exact_match_cols
        )
    )
    missing_cols = [c for c in needed if c not in df_full.columns]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"Columns not found: {missing_cols}")

    df_imputed_temp = apply_imputation(df_full[needed], needed, req.imputation or "listwise")
    df_full_imputed = df_full.loc[df_imputed_temp.index].copy().reset_index(drop=True)
    df = df_imputed_temp.reset_index(drop=True)
    df, cat_warnings = _clean_covariate_categories(df, req.covariates)
    df_full_imputed = df_full_imputed.loc[df.index].copy().reset_index(drop=True)
    df = df.reset_index(drop=True)

    treat_vals = df[req.treatment_col].astype(float)
    if not set(treat_vals.unique().tolist()) <= {0, 1, 0.0, 1.0}:
        raise HTTPException(
            status_code=422,
            detail=f"Treatment column '{req.treatment_col}' must be binary (0 = control, 1 = treated).",
        )

    X = pd.get_dummies(df[req.covariates], drop_first=True).astype(float)
    y = treat_vals.astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    score_method = (req.score_method or "logistic").lower()
    try:
        ps = _fit_propensity_scores(X_scaled, y, score_method, req.random_state)
    except ValueError as exc:
        # score_method is a free-form string on the request; an unrecognised
        # value makes the propensity fitter raise ValueError. Bad option string
        # is caller input, so answer 400 with the reason rather than a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ps_clip = np.clip(ps, 1e-6, 1 - 1e-6)
    logit_ps = np.log(ps_clip / (1.0 - ps_clip))

    df = df.copy()
    df["_ps_"] = ps
    df["_logit_ps_"] = logit_ps
    df["_treat_"] = y

    treated_idx_all = np.where(y == 1)[0]
    control_idx_all = np.where(y == 0)[0]
    if len(treated_idx_all) == 0 or len(control_idx_all) == 0:
        raise HTTPException(status_code=422, detail="Need both treated (1) and control (0) patients.")

    support_lo, support_hi = float(ps.min()), float(ps.max())
    n_trimmed = 0
    keep_mask = np.ones_like(y, dtype=bool)
    if req.trim_common_support:
        support_lo = max(ps[treated_idx_all].min(), ps[control_idx_all].min())
        support_hi = min(ps[treated_idx_all].max(), ps[control_idx_all].max())
        keep_mask = (ps >= support_lo) & (ps <= support_hi)
        n_trimmed = int((~keep_mask).sum())

    treated_idx = np.where((y == 1) & keep_mask)[0]
    control_idx = np.where((y == 0) & keep_mask)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise HTTPException(
            status_code=422,
            detail="No units remain after common-support trim. Disable trimming or widen the support.",
        )

    scale = (req.caliper_scale or "logit").lower()
    if scale not in ("logit", "raw"):
        raise HTTPException(status_code=422, detail="caliper_scale must be 'logit' or 'raw'.")
    distance_vec = logit_ps if scale == "logit" else ps
    caliper_sd = float(distance_vec[keep_mask].std())
    caliper_dist = (req.caliper or 0.2) * caliper_sd
    ratio = max(1, req.ratio or 1)

    matching_method = (req.matching_method or "greedy").lower()
    if matching_method not in ("greedy", "optimal"):
        raise HTTPException(status_code=422, detail="matching_method must be 'greedy' or 'optimal'.")
    matching_warning: Optional[str] = None
    if matching_method == "optimal" and ratio != 1:
        matching_warning = "Optimal (Hungarian) matching supports 1:1 only. Falling back to greedy for ratio > 1."
        matching_method = "greedy"

    if exact_match_cols:
        bad = [c for c in exact_match_cols if c not in df.columns]
        if bad:
            raise HTTPException(status_code=422, detail=f"exact_match columns not found: {bad}")

    matched_treated, matched_controls = _run_match_strata(
        df=df,
        treated_idx=treated_idx,
        control_idx=control_idx,
        distance_vec=distance_vec,
        caliper_dist=caliper_dist,
        ratio=ratio,
        method=matching_method,
        exact_match_cols=exact_match_cols if exact_match_cols else None,
    )

    n_matched_treated = len(matched_treated)
    n_matched_controls = len(matched_controls)

    if n_matched_treated == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No matches found within caliper {req.caliper}. "
            "Try widening the caliper or check that treatment groups overlap in covariate space.",
        )
    if n_matched_treated + n_matched_controls < 10 and len(df) >= 20:
        raise HTTPException(
            status_code=422,
            detail="Common-support/caliper settings left fewer than 10 matched units. Disable trimming or widen the caliper.",
        )

    matched_all_idx = matched_treated + matched_controls
    df_matched = df_full_imputed.iloc[matched_all_idx].copy()
    df_matched["_treat_"] = df["_treat_"].iloc[matched_all_idx].values
    df_matched["_ps_"] = df["_ps_"].iloc[matched_all_idx].values
    df_matched["_logit_ps_"] = df["_logit_ps_"].iloc[matched_all_idx].values

    match_ids = []
    for i, ti in enumerate(matched_treated):
        match_ids.append(i)
    for i in range(len(matched_controls)):
        match_ids.append(i // ratio)
    df_matched["_match_id_"] = match_ids

    smd_before, smd_after = {}, {}
    var_ratio_before, var_ratio_after = {}, {}
    ks_before, ks_after = {}, {}
    treat_mask = df["_treat_"].values
    for cov in req.covariates:
        col = df[cov]
        col_m = df_matched[cov]
        denom = _pooled_sd(col[treat_mask == 1], col[treat_mask == 0])
        smd_before[cov] = round(
            _compute_smd(col[treat_mask == 1], col[treat_mask == 0], denom_sd=denom), 4
        )
        smd_after[cov] = round(
            _compute_smd(
                col_m[df_matched["_treat_"] == 1],
                col_m[df_matched["_treat_"] == 0],
                denom_sd=denom,
            ),
            4,
        )
        var_ratio_before[cov] = _variance_ratio(col[treat_mask == 1], col[treat_mask == 0])
        var_ratio_after[cov] = _variance_ratio(
            col_m[df_matched["_treat_"] == 1], col_m[df_matched["_treat_"] == 0]
        )
        ks_before[cov] = _ks_p(col[treat_mask == 1], col[treat_mask == 0])
        ks_after[cov] = _ks_p(
            col_m[df_matched["_treat_"] == 1], col_m[df_matched["_treat_"] == 0]
        )

    avg_smd_before = float(np.mean(list(smd_before.values())))
    avg_smd_after = float(np.mean(list(smd_after.values())))
    reduction_pct = (
        float((avg_smd_before - avg_smd_after) / avg_smd_before * 100)
        if avg_smd_before > 0
        else 0.0
    )

    n_all_treated = int((y == 1).sum())
    n_all_control = int((y == 0).sum())
    n_unmatched = n_all_treated - n_matched_treated

    var_ratios_ok = all(
        (v is None) or (0.5 <= v <= 2.0) for v in var_ratio_after.values()
    )
    balance_achieved = bool(all(v < 0.10 for v in smd_after.values()) and var_ratios_ok)

    ps_dist = {
        "treated_unmatched": ps[treated_idx].tolist(),
        "control_unmatched": ps[control_idx].tolist(),
        "treated_matched": ps[matched_treated].tolist(),
        "control_matched": ps[matched_controls].tolist(),
    }

    outcome_result = None
    rosenbaum_result = None

    if outcome_type == "survival":
        try:
            from lifelines import CoxPHFitter

            dur = pd.to_numeric(df_matched[req.survival_duration_col], errors="coerce")
            evt = pd.to_numeric(df_matched[req.survival_event_col], errors="coerce")
            if np.any(dur.dropna() < 0):
                outcome_result = {
                    "error": f"survival_duration_col '{req.survival_duration_col}' must be ≥ 0."
                }
            elif set(evt.dropna().unique().tolist()) - {0.0, 1.0}:
                outcome_result = {
                    "error": f"survival_event_col '{req.survival_event_col}' must be binary 0/1."
                }
            else:
                cox_df = pd.DataFrame(
                    {
                        "_dur_": dur.values.astype(float),
                        "_evt_": evt.values.astype(int),
                        req.treatment_col: pd.to_numeric(
                            df_matched[req.treatment_col], errors="coerce"
                        )
                        .astype(float)
                        .values,
                        "_match_id_": df_matched["_match_id_"].values.astype(int),
                    }
                ).dropna()
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col="_dur_", event_col="_evt_", strata=["_match_id_"])
                coef = float(cph.params_.iloc[0])
                se = float(cph.standard_errors_.iloc[0])
                ci = cph.confidence_intervals_.iloc[0]
                try:
                    p_val = float(cph.summary["p"].iloc[0])
                except Exception as exc:
                    logger.debug("PSM stratified Cox p-value extraction failed: {}", exc)
                    p_val = None
                outcome_result = {
                    "type": "stratified_cox",
                    "model": "Cox PH stratified by matched set",
                    "n": int(len(cox_df)),
                    "n_events": int(cox_df["_evt_"].sum()),
                    "concordance": round(float(cph.concordance_index_), 4),
                    "coefficients": [
                        {
                            "variable": req.treatment_col,
                            "estimate": round(coef, 6),
                            "hr": round(float(np.exp(coef)), 4),
                            "se": round(se, 6),
                            "z": round(coef / se, 4) if se > 0 else None,
                            "p": round(p_val, 6) if p_val is not None else None,
                            "ci_low": round(float(ci.iloc[0]), 4),
                            "ci_high": round(float(ci.iloc[1]), 4),
                            "hr_low": round(float(np.exp(ci.iloc[0])), 4),
                            "hr_high": round(float(np.exp(ci.iloc[1])), 4),
                        }
                    ],
                }
        except Exception as ex:
            logger.exception("PSM Stratified Cox outcome fit failed")
            outcome_result = {"error": f"Stratified Cox failed: {ex}"}

    elif req.outcome_col and req.outcome_col in df_matched.columns:
        try:
            y_out = pd.to_numeric(df_matched[req.outcome_col], errors="coerce")
            out_vals = set(y_out.dropna().unique().tolist())

            if not out_vals <= {0, 1, 0.0, 1.0}:
                outcome_result = {
                    "error": f"Outcome must be binary 0/1 for matched analysis. Found: {sorted(out_vals)[:10]}"
                }
            else:
                from statsmodels.discrete.conditional_models import ConditionalLogit

                df_out = df_matched[[req.treatment_col, req.outcome_col, "_match_id_"]].copy()
                df_out[req.outcome_col] = y_out.astype(int)
                df_out[req.treatment_col] = pd.to_numeric(
                    df_out[req.treatment_col], errors="coerce"
                ).astype(float)
                df_out = df_out.dropna()

                grp_y = df_out.groupby("_match_id_")[req.outcome_col]
                informative_ids = grp_y.nunique().loc[lambda s: s > 1].index
                n_informative_pairs = int(len(informative_ids))
                df_clogit = df_out[df_out["_match_id_"].isin(informative_ids)].copy()

                if n_informative_pairs == 0:
                    outcome_result = {
                        "error": "No informative (discordant) matched sets — every pair has identical outcomes. Conditional logistic cannot fit.",
                    }
                else:
                    try:
                        X_out = df_clogit[[req.treatment_col]].astype(float).values
                        y_arr_out = df_clogit[req.outcome_col].astype(int).values
                        grp_arr = df_clogit["_match_id_"].values
                        mod_cl = ConditionalLogit(y_arr_out, X_out, groups=grp_arr)
                        res_cl = mod_cl.fit(disp=False)
                        coef = float(res_cl.params[0])
                        se = float(res_cl.bse[0])
                        p_val = float(res_cl.pvalues[0])
                        ci_lo = coef - 1.959963984540054 * se
                        ci_hi = coef + 1.959963984540054 * se
                        coefs_out = [
                            {
                                "variable": req.treatment_col,
                                "estimate": round(coef, 6),
                                "or": round(float(np.exp(coef)), 4),
                                "se": round(se, 6),
                                "z": round(coef / se, 4) if se > 0 else None,
                                "p": round(p_val, 6),
                                "ci_low": round(ci_lo, 4),
                                "ci_high": round(ci_hi, 4),
                                "or_low": round(float(np.exp(ci_lo)), 4),
                                "or_high": round(float(np.exp(ci_hi)), 4),
                            }
                        ]
                        outcome_result = {
                            "type": "conditional_logistic",
                            "model": "Conditional logistic regression (matched-set stratification)",
                            "n": int(len(df_clogit)),
                            "n_matched_sets": int(df_out["_match_id_"].nunique()),
                            "n_informative_sets": n_informative_pairs,
                            "n_uninformative_sets": int(df_out["_match_id_"].nunique())
                            - n_informative_pairs,
                            "coefficients": coefs_out,
                            "log_likelihood": round(float(res_cl.llf), 4),
                            "method_note": (
                                "Conditional likelihood treats each matched set as a stratum. "
                                "Uninformative (concordant) sets contribute 0 to the likelihood and are dropped. "
                                "For 1:1 matching with treatment as the only covariate this is equivalent to McNemar's test."
                            ),
                        }
                    except Exception as cl_exc:
                        logger.exception("Conditional logistic fit failed, falling back to robust logit")
                        X_out = sm.add_constant(df_out[[req.treatment_col]].astype(float))
                        m_out = sm.Logit(
                            df_out[req.outcome_col].astype(int).values, X_out
                        ).fit(disp=False, cov_type="HC1")
                        ci_out = m_out.conf_int()
                        coefs_out = []
                        for var in m_out.params.index:
                            est = float(m_out.params[var])
                            coefs_out.append(
                                {
                                    "variable": str(var),
                                    "estimate": round(est, 6),
                                    "or": round(float(np.exp(est)), 4),
                                    "se": round(float(m_out.bse[var]), 6),
                                    "z": round(float(m_out.tvalues[var]), 4),
                                    "p": round(float(m_out.pvalues[var]), 6),
                                    "ci_low": round(float(ci_out.loc[var, 0]), 4),
                                    "ci_high": round(float(ci_out.loc[var, 1]), 4),
                                    "or_low": round(float(np.exp(ci_out.loc[var, 0])), 4),
                                    "or_high": round(float(np.exp(ci_out.loc[var, 1])), 4),
                                }
                            )
                        outcome_result = {
                            "type": "logistic_robust",
                            "model": "Logistic Regression [Robust SE] (matched cohort, clogit fallback)",
                            "n": int(len(df_matched)),
                            "coefficients": coefs_out,
                            "aic": round(float(m_out.aic), 2),
                            "bic": round(float(m_out.bic), 2),
                            "method_note": f"Conditional logistic fit failed ({cl_exc}); fell back to logistic with robust SE.",
                        }
        except Exception as ex:
            logger.exception("PSM outcome fit failed")
            outcome_result = {"error": str(ex)}

    if (
        req.compute_rosenbaum
        and outcome_type == "binary"
        and ratio == 1
        and req.outcome_col
        and req.outcome_col in df_matched.columns
    ):
        try:
            y_out = pd.to_numeric(df_matched[req.outcome_col], errors="coerce")
            out_vals = set(y_out.dropna().unique().tolist())
            if not out_vals <= {0, 1, 0.0, 1.0}:
                rosenbaum_result = {
                    "applicable": False,
                    "reason": "Rosenbaum bounds require binary 0/1 outcome.",
                }
            else:
                pair_pairs: list[tuple[int, int]] = []
                df_rb = df_matched.copy()
                df_rb[req.outcome_col] = y_out.astype(int)
                df_rb[req.treatment_col] = pd.to_numeric(
                    df_rb[req.treatment_col], errors="coerce"
                ).astype(int)
                for mid, grp in df_rb.groupby("_match_id_"):
                    t_rows = grp[grp[req.treatment_col] == 1]
                    c_rows = grp[grp[req.treatment_col] == 0]
                    if len(t_rows) == 1 and len(c_rows) == 1:
                        pair_pairs.append(
                            (
                                int(t_rows[req.outcome_col].iloc[0]),
                                int(c_rows[req.outcome_col].iloc[0]),
                            )
                        )
                if not pair_pairs:
                    rosenbaum_result = {
                        "applicable": False,
                        "reason": "No clean 1:1 matched pairs available.",
                    }
                else:
                    rosenbaum_result = _rosenbaum_bounds(
                        pair_pairs,
                        gamma_max=float(req.rosenbaum_gamma_max or 3.0),
                    )
        except Exception as ex:
            logger.exception("Rosenbaum bounds calculation failed")
            rosenbaum_result = {"applicable": False, "reason": f"Rosenbaum bounds failed: {ex}"}

    df_export = df_matched.drop(columns=["_ps_", "_logit_ps_", "_treat_"], errors="ignore")
    df_export = df_export.rename(columns={"_match_id_": "match_set_id"})
    store.save(req.session_id + "_psm", df_export)

    try:
        parent_metadata = store.get_metadata(req.session_id)
        if parent_metadata:
            store.save_metadata(req.session_id + "_psm", parent_metadata)

        parent_kinds = store.get_kind_overrides(req.session_id)
        if parent_kinds:
            kinds = {**parent_kinds, "match_set_id": "categorical"}
            store.set_kind_overrides(req.session_id + "_psm", kinds)

        parent_decimals = store.get_decimals(req.session_id)
        if parent_decimals:
            store.save_decimals(req.session_id + "_psm", parent_decimals)
    except Exception:
        logger.exception("Saving PSM child-session metadata failed")

    return {
        "n_total": int(len(df)),
        "n_treated": n_all_treated,
        "n_control": n_all_control,
        "n_matched_pairs": n_matched_treated,
        "n_matched_controls": n_matched_controls,
        "n_unmatched": n_unmatched,
        "n_trimmed_common_support": n_trimmed,
        "score_method": score_method,
        "matching_method": matching_method,
        "matching_warning": matching_warning,
        "exact_match": exact_match_cols,
        "outcome_type": outcome_type,
        "caliper_scale": scale,
        "caliper_used": round(float(caliper_dist), 6),
        "caliper_sd": round(caliper_sd, 6),
        "common_support": {"lo": round(support_lo, 6), "hi": round(support_hi, 6)},
        "balance_achieved": balance_achieved,
        "avg_smd_before": round(avg_smd_before, 4),
        "avg_smd_after": round(avg_smd_after, 4),
        "reduction_pct": round(reduction_pct, 1),
        "smd_before": smd_before,
        "smd_after": smd_after,
        "variance_ratio_before": var_ratio_before,
        "variance_ratio_after": var_ratio_after,
        "ks_p_before": ks_before,
        "ks_p_after": ks_after,
        "ps_distribution": ps_dist,
        "outcome_result": outcome_result,
        "rosenbaum": rosenbaum_result,
        "warnings": cat_warnings,
        "matched_session_id": req.session_id + "_psm",
    }


# ── IPTW Endpoint ──────────────────────────────────────────────────────────────

@router.post("/iptw")
async def iptw_analysis(req: IPTWRequest):
    try:
        return await asyncio.to_thread(_run_iptw, req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("IPTW analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _run_iptw(req: IPTWRequest):
    df_full = _get_df(req.session_id)
    needed = [req.treatment_col] + req.covariates
    if req.outcome_type == "binary" and req.outcome_col:
        needed.append(req.outcome_col)
    if req.outcome_type == "survival":
        needed += [req.survival_duration_col, req.survival_event_col]

    df = apply_imputation(df_full, [c for c in needed if c], req.imputation or "listwise")
    df, cat_warnings = _clean_covariate_categories(df, req.covariates)

    treat = pd.to_numeric(df[req.treatment_col], errors="coerce").astype(int)
    if set(treat.dropna().unique()) - {0, 1}:
        raise HTTPException(
            status_code=422,
            detail=f"Treatment column '{req.treatment_col}' must be binary (0 = control, 1 = treated).",
        )
    X = pd.get_dummies(df[req.covariates], drop_first=True).astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        ps = _fit_propensity_scores(
            X_scaled, treat.values,
            method=(req.score_method or "logistic").lower(),
            random_state=req.random_state,
        )
    except ValueError as exc:
        # Same as the PSM endpoint: score_method is an unconstrained string, so
        # an unknown value is caller input and deserves a readable 400.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ps = np.clip(ps, 1e-5, 1.0 - 1e-5)

    # Common-support trimming: drop units whose PS falls outside the
    # overlap region [max of group minima, min of group maxima].
    n_trimmed = 0
    if req.trim_common_support:
        ps_t, ps_c = ps[treat.values == 1], ps[treat.values == 0]
        if len(ps_t) and len(ps_c):
            lo_cs = max(ps_t.min(), ps_c.min())
            hi_cs = min(ps_t.max(), ps_c.max())
            keep = (ps >= lo_cs) & (ps <= hi_cs)
            n_trimmed = int((~keep).sum())
            if n_trimmed:
                df = df.loc[keep]
                X = X.loc[keep]
                treat = treat.loc[keep]
                ps = ps[keep]

    eps = 1e-6
    if req.estimand == "ate":
        w = treat / (ps + eps) + (1 - treat) / (1 - ps + eps)
    elif req.estimand == "att":
        w = treat + (1 - treat) * ps / (1 - ps + eps)
    elif req.estimand == "overlap":
        w = treat * (1 - ps) + (1 - treat) * ps
    else:
        raise HTTPException(status_code=422, detail=f"Unknown estimand: {req.estimand}")

    if req.stabilize:
        p_treat = float(treat.mean())
        if req.estimand == "ate":
            w = w * (p_treat * treat + (1 - p_treat) * (1 - treat))
        elif req.estimand == "att":
            w = w * p_treat

    finite_mask = np.isfinite(w)
    n_nonfinite = int((~finite_mask).sum())
    if n_nonfinite > 0:
        w = np.where(finite_mask, w, 0.0)

    n_truncated = 0
    if req.weight_truncation == "percentile":
        p_lo = float(np.clip(req.weight_truncation_lo, 0.0, 1.0)) * 100.0
        p_hi = float(np.clip(req.weight_truncation_hi, 0.0, 1.0)) * 100.0
        if p_hi <= p_lo:
            p_lo, p_hi = 1.0, 99.0
        lo, hi = np.percentile(w, [p_lo, p_hi])
        before = w.copy()
        w = np.clip(w, lo, hi)
        n_truncated = int((before != w).sum())
    elif req.weight_truncation == "hard":
        max_w = float(req.weight_truncation_max)
        before = w.copy()
        w = np.clip(w, 0, max_w)
        n_truncated = int((before != w).sum())

    w = np.maximum(w, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)

    # Weighted SMD — Austin (2009) eqs 1 & 2: numerator uses weighted means,
    # denominator uses the *unweighted* pooled SD so the metric stays
    # comparable to the pre-weight value. Before this fix the "after" mean
    # was computed unweighted, so smd_before always equalled smd_after and
    # the panel claimed perfect balance for any weighting.
    # Per-covariate dicts (not scalars): the Love plot and SMD balance table
    # render one row per covariate, mirroring the PSM response contract.
    smd_before: dict[str, float] = {}
    smd_after: dict[str, float] = {}
    wt = pd.Series(w, index=X.index)
    wt_t = wt.loc[treat == 1]
    wt_c = wt.loc[treat == 0]
    sw_t = float(wt_t.sum())
    sw_c = float(wt_c.sum())
    for col in X.columns:
        s_t = X.loc[treat == 1, col]
        s_c = X.loc[treat == 0, col]
        smd_b = _compute_smd(s_t, s_c)
        denom = _pooled_sd(s_t, s_c) + 1e-9
        if sw_t > 0 and sw_c > 0 and denom > 0:
            mean_t = float((s_t * wt_t).sum() / sw_t)
            mean_c = float((s_c * wt_c).sum() / sw_c)
            smd_a = abs(mean_t - mean_c) / denom
        else:
            smd_a = float("nan")
        if np.isfinite(smd_b) and np.isfinite(smd_a):
            smd_before[col] = round(float(smd_b), 4)
            smd_after[col] = round(float(smd_a), 4)

    avg_smd_before = float(np.mean(list(smd_before.values()))) if smd_before else 0.0
    avg_smd_after = float(np.mean(list(smd_after.values()))) if smd_after else 0.0
    reduction_pct = (
        float((avg_smd_before - avg_smd_after) / avg_smd_before * 100)
        if avg_smd_before > 0 else 0.0
    )
    balance_achieved = bool(smd_after) and all(v < 0.10 for v in smd_after.values())

    # The weighted outcome model regresses the outcome on the TREATMENT.
    #
    # It used to regress it on the covariate matrix X — the propensity-score
    # design — with the treatment column absent entirely. The response
    # therefore carried coefficients for age, bmi and sex and no treatment
    # effect at all, which is the one number an IPTW analysis exists to
    # produce. A reader would take "age p = 0.018" for the result.
    #
    # Weighting is what removes the confounding, so the outcome model is
    # y ~ treatment on the weighted sample — the same specification as R's
    # svyglm(y ~ treat, design). Standard errors are robust; model-based ones
    # understate the uncertainty in a weighted fit.
    treat_name = str(req.treatment_col)
    outcome_result = None
    if req.outcome_type == "binary" and req.outcome_col:
        y = pd.to_numeric(df[req.outcome_col], errors="coerce")
        X_treat = sm.add_constant(
            pd.DataFrame({treat_name: np.asarray(treat, dtype=float)}, index=df.index),
            has_constant="add")
        try:
            glm = sm.GLM(y, X_treat, family=sm.families.Binomial(),
                         var_weights=w).fit(cov_type="HC0")
            b = float(glm.params[treat_name])
            se = float(glm.bse[treat_name])
            lo, hi = b - 1.959963984540054 * se, b + 1.959963984540054 * se
            outcome_result = {
                "type": "weighted_glm",
                "model": "Weighted logistic regression (IPTW), robust SE",
                "outcome": req.outcome_col,
                "treatment": treat_name,
                "coefficients": [{
                    "variable": treat_name,
                    "estimate": b, "se": se,
                    "z": float(glm.tvalues[treat_name]),
                    "p": float(glm.pvalues[treat_name]),
                    "ci_low": lo, "ci_high": hi,
                    "odds_ratio": float(np.exp(b)),
                    "or_ci_low": float(np.exp(lo)), "or_ci_high": float(np.exp(hi)),
                }],
                "method_note": (
                    "The outcome is modelled on the treatment alone; the IPTW "
                    "weights carry the adjustment for the covariates. Standard "
                    "errors are robust (HC0)."
                ),
            }
        except Exception as e:
            logger.exception("IPTW Weighted GLM outcome fit failed")
            outcome_result = {"type": "weighted_glm", "error": _sanitize_model_error(e, "weighted GLM")}

    elif req.outcome_type == "survival" and req.survival_duration_col and req.survival_event_col:
        try:
            from lifelines import CoxPHFitter
            surv_df = df[[req.survival_duration_col, req.survival_event_col]].copy()
            surv_df[treat_name] = np.asarray(treat, dtype=float)
            surv_df["w"] = w
            surv_df[req.survival_event_col] = pd.to_numeric(surv_df[req.survival_event_col], errors="coerce")
            surv_df[req.survival_duration_col] = pd.to_numeric(surv_df[req.survival_duration_col], errors="coerce")
            cph = CoxPHFitter()
            cph.fit(surv_df, duration_col=req.survival_duration_col,
                    event_col=req.survival_event_col, weights_col="w", robust=True)
            summ = cph.summary
            rows = []
            for v in cph.params_.index:
                r_ = summ.loc[v]
                rows.append({
                    "variable": str(v),
                    "log_hr": float(cph.params_[v]),
                    "hr": float(cph.hazard_ratios_[v]),
                    # The HR alone was all this returned — no uncertainty at
                    # all, so the estimate could not be read as a result.
                    "se": float(r_["se(coef)"]),
                    "z": float(r_["z"]),
                    "p": float(r_["p"]),
                    "hr_ci_low": float(np.exp(r_["coef lower 95%"])),
                    "hr_ci_high": float(np.exp(r_["coef upper 95%"])),
                })
            outcome_result = {
                "type": "weighted_cox",
                "model": "Weighted Cox regression (IPTW), robust SE",
                "treatment": treat_name,
                "coefficients": rows,
                "method_note": (
                    "The hazard is modelled on the treatment alone; the IPTW "
                    "weights carry the adjustment for the covariates."
                ),
            }
        except Exception as e:
            logger.exception("IPTW Weighted Cox outcome fit failed")
            outcome_result = {"type": "weighted_cox", "error": _sanitize_model_error(e, "weighted Cox model")}

    weight_sum = float(np.sum(w))
    effective_n = float(np.sum(w) ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0

    def _ess(arr: np.ndarray) -> float:
        denom = float(np.sum(arr ** 2))
        return float(np.sum(arr) ** 2 / denom) if denom > 0 else 0.0

    w_treated = np.asarray(w)[treat.values == 1]
    w_control = np.asarray(w)[treat.values == 0]
    n_treated = int((treat.values == 1).sum())
    n_control = int((treat.values == 0).sum())

    warnings = list(cat_warnings)
    if n_nonfinite > 0:
        warnings.append(f"{n_nonfinite} non-finite weights were replaced with 0")
    if n_truncated > 0:
        warnings.append(f"{n_truncated} weights were truncated")
    if weight_sum == 0:
        warnings.append("All weights became zero after cleaning — results will be unreliable")
    if effective_n < 10:
        warnings.append(f"Very low effective sample size ({effective_n:.1f})")
    if len(w) and float(np.max(w)) > 10:
        warnings.append(f"Extreme IPTW weight detected (max={float(np.max(w)):.2f}); consider truncation or overlap weights.")

    # Weighted-cohort child session: full covariate rows + the IPTW weight,
    # so the user can export it or load it as the active dataset (same flow
    # as the PSM "_psm" child session).
    weighted_sid = req.session_id + "_iptw"
    df_export = df.copy()
    df_export["iptw_weight"] = np.round(np.asarray(w, dtype=float), 6)
    store.save(weighted_sid, df_export)
    try:
        parent_metadata = store.get_metadata(req.session_id)
        if parent_metadata:
            store.save_metadata(weighted_sid, parent_metadata)
        parent_kinds = store.get_kind_overrides(req.session_id)
        if parent_kinds:
            store.set_kind_overrides(weighted_sid, {**parent_kinds, "iptw_weight": "numeric"})
        parent_decimals = store.get_decimals(req.session_id)
        if parent_decimals:
            store.save_decimals(weighted_sid, parent_decimals)
    except Exception:
        logger.exception("Saving IPTW child-session metadata failed")

    return {
        "method": "iptw",
        "estimand": req.estimand,
        "stabilize": bool(req.stabilize),
        "se_method": req.se_method,
        "score_method": (req.score_method or "logistic").lower(),
        "n": int(len(df)),
        "n_total": int(len(df)),
        "n_treated": n_treated,
        "n_control": n_control,
        "n_trimmed_common_support": n_trimmed,
        "weight_summary": {
            "mean": round(float(np.mean(w)), 4),
            "median": round(float(np.median(w)), 4),
            "max": round(float(np.max(w)), 4),
            "min": round(float(np.min(w)), 4),
            "sum": round(weight_sum, 2),
            "effective_n": round(effective_n, 1),
            "ess_treated": round(_ess(w_treated), 1),
            "ess_control": round(_ess(w_control), 1),
            "n_truncated": n_truncated,
        },
        "weight_distribution": {
            "treated": np.round(w_treated, 4).tolist(),
            "control": np.round(w_control, 4).tolist(),
        },
        "ps_distribution": {
            "treated_unmatched": np.round(ps[treat.values == 1], 4).tolist(),
            "control_unmatched": np.round(ps[treat.values == 0], 4).tolist(),
            "treated_matched": [],
            "control_matched": [],
        },
        "smd_before": smd_before,
        "smd_after": smd_after,
        "avg_smd_before": round(avg_smd_before, 4),
        "avg_smd_after": round(avg_smd_after, 4),
        "reduction_pct": round(reduction_pct, 1),
        "balance_achieved": balance_achieved,
        "matched_session_id": weighted_sid,
        "warnings": warnings,
        "outcome_result": outcome_result,
        "result_text": _iptw_results_text(req.estimand, outcome_result, warnings),
    }


def _iptw_results_text(estimand, outcome_result, warnings):
    parts = [f"IPTW analysis was performed for the {estimand.upper()} estimand."]
    if outcome_result and "error" not in outcome_result:
        if outcome_result.get("type") == "weighted_glm":
            parts.append("A weighted logistic/linear model was fit on the inverse probability weights.")
        elif outcome_result.get("type") == "weighted_cox":
            parts.append("A weighted Cox model was fit using the stabilized/truncated weights.")
    if warnings:
        parts.append("Numerical warnings were raised during weight calculation.")
    return " ".join(parts)
