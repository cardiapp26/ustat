"""
Causal Sensitivity Analysis Module (Phase 5)

Provides tools for assessing robustness of causal effect estimates
to unmeasured confounding.

Core methods:
- E-value (VanderWeele & Ding 2017) for RR, OR, HR
- Improved OR -> RR conversion (supports baseline risk / prevalence)
- Simple bias factor / quantitative bias analysis (Greenland / Schlesselman style)
- Integration helpers for PSM/IPTW results

All functions are pure and return new immutable dictionaries.
"""

from __future__ import annotations

from typing import Dict, Any, List, Literal, Optional
import math

import numpy as np
import pandas as pd


def _safe_sqrt(x: float) -> float:
    return math.sqrt(max(x, 0.0))


def _or_to_rr(or_val: float, baseline_risk: float = 0.1) -> float:
    """
    Convert odds ratio to approximate risk ratio using a baseline risk.
    Formula: RR ≈ OR / ((1 - p0) + p0 * OR)   (Zhang & Yu style approximation)
    """
    if or_val <= 0:
        return 1.0
    p0 = max(0.001, min(0.99, baseline_risk))
    return or_val / ((1.0 - p0) + p0 * or_val)


def _e_value_from_rr(rr: float) -> float:
    """VanderWeele & Ding: RR + sqrt(RR (RR - 1)) for RR > 1, else 1."""
    if not math.isfinite(rr):
        return float("inf")
    return rr + _safe_sqrt(rr * (rr - 1.0)) if rr > 1.0 else 1.0


def _to_rr_scale(
    estimate: float,
    ci_low: Optional[float],
    ci_high: Optional[float],
    measure: str,
    rare_outcome: bool,
    baseline_risk: Optional[float],
) -> tuple[float, Optional[List[float]], str]:
    """Approximate risk ratio for the estimate and its CI limits.

    VanderWeele & Ding (2017), Table 2:
    - RR: used as is.
    - OR or HR with a rare outcome (< 15%): treated as a risk ratio.
    - OR with a common outcome: RR ~ sqrt(OR); with a supplied baseline risk
      the Zhang & Yu conversion is used instead.
    - HR with a common outcome:
      RR ~ (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)).
    """
    if measure == "rr" or rare_outcome:
        convert, label = (lambda v: float(v)), ("none" if measure == "rr" else "rare_outcome")
    elif measure == "or":
        if baseline_risk is not None:
            convert, label = (lambda v: _or_to_rr(float(v), baseline_risk)), "baseline_risk"
        else:
            convert, label = (lambda v: math.sqrt(float(v))), "sqrt_or"
    else:
        convert, label = _hr_to_rr_common, "hr_common_outcome"
    rr_ci = None
    if ci_low is not None and ci_high is not None:
        if ci_low <= 0 or ci_high <= 0:
            raise ValueError("confidence limits must be positive")
        rr_ci = [convert(ci_low), convert(ci_high)]
    return convert(estimate), rr_ci, label


def _hr_to_rr_common(hr: float) -> float:
    root = math.sqrt(float(hr))
    return (1.0 - 0.5 ** root) / (1.0 - 0.5 ** (1.0 / root))


def e_value(
    estimate: float,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    measure: Literal["rr", "or", "hr"] = "rr",
    rare_outcome: bool = False,
    baseline_risk: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate the E-value (VanderWeele & Ding) for a point estimate and optional CI.

    For OR with common outcomes, supply baseline_risk (0 < p0 < 1) for a better
    approximation than the rare-disease assumption.

    Returns a new immutable dict with e_value_point_estimate, e_value_ci, interpretation.
    """
    if measure not in ("rr", "or", "hr"):
        raise ValueError("measure must be 'rr', 'or', or 'hr'")
    if estimate <= 0:
        raise ValueError("estimate must be positive")

    rr, rr_ci, conversion = _to_rr_scale(estimate, ci_low, ci_high, measure, rare_outcome, baseline_risk)

    # A protective estimate is inverted so both it and its CI sit above 1.
    if rr < 1.0:
        rr = 1.0 / rr
        if rr_ci:
            rr_ci = [1.0 / x if x > 0 else float("inf") for x in rr_ci]

    ev_point = _e_value_from_rr(rr)

    # E-value for the CI uses the limit closest to the null: the lower limit
    # once the estimate is on the >= 1 side. A CI that includes 1 gives 1.
    ev_ci = None
    if rr_ci:
        ev_ci = _e_value_from_rr(min(rr_ci))

    interpretation = (
        f"An unmeasured confounder would need to be associated with both the exposure and the outcome "
        f"by a risk ratio of at least {ev_point:.2f} (point estimate)"
    )
    if ev_ci and ev_ci > 1.0:
        interpretation += f" and at least {ev_ci:.2f} (CI) to explain away the observed association."

    return {
        "measure": measure,
        "e_value_point_estimate": round(ev_point, 3) if ev_point > 1.0 else 1.0,
        "e_value_ci": round(ev_ci, 3) if ev_ci and ev_ci > 1.0 else 1.0,
        "interpretation": interpretation,
        "baseline_risk_used": baseline_risk if conversion == "baseline_risk" else None,
        # How the estimate was put on the risk-ratio scale: none, rare_outcome,
        # sqrt_or, baseline_risk or hr_common_outcome.
        "rr_conversion": conversion,
        "rr_used": round(rr, 4),
    }


def e_value_for_smd(
    smd: float,
    *,
    baseline_risk: float = 0.1,
) -> Dict[str, Any]:
    """
    Approximate E-value for a standardized mean difference.

    Uses the Chinn conversion from standardized mean difference to odds ratio:
    OR ~= exp(pi * SMD / sqrt(3)), then converts OR to RR using baseline risk.
    """
    d = abs(float(smd))
    odds_ratio = math.exp(math.pi * d / math.sqrt(3.0))
    ev = e_value(
        estimate=odds_ratio,
        measure="or",
        rare_outcome=False,
        baseline_risk=baseline_risk,
    )
    return {
        "smd": round(float(smd), 5),
        "absolute_smd": round(d, 5),
        "converted_or": round(float(odds_ratio), 5),
        "baseline_risk_used": baseline_risk,
        "e_value": ev,
        "method_note": "SMD converted to OR via Chinn approximation, then to RR for E-value.",
    }


def manski_bounds_binary(
    *,
    p_y1_treated: float,
    p_y1_control: float,
    p_treated: float,
    monotone_treatment_response: bool = False,
) -> Dict[str, Any]:
    """
    Manski no-assumptions bounds for binary outcomes.

    Inputs are observed P(Y=1|T=1), observed P(Y=1|T=0), and P(T=1).
    Bounds are for E[Y(1)], E[Y(0)], and ATE = E[Y(1)-Y(0)].
    """
    p1 = min(max(float(p_y1_treated), 0.0), 1.0)
    p0 = min(max(float(p_y1_control), 0.0), 1.0)
    pt = min(max(float(p_treated), 0.0), 1.0)
    pc = 1.0 - pt

    ey1_low = p1 * pt
    ey1_high = ey1_low + pc
    ey0_low = p0 * pc
    ey0_high = ey0_low + pt
    ate_low = ey1_low - ey0_high
    ate_high = ey1_high - ey0_low

    if monotone_treatment_response:
        ate_low = max(ate_low, 0.0)
        ey1_low = max(ey1_low, ey0_low)

    return {
        "assumptions": "Manski no-assumptions bounds" + (" + monotone treatment response" if monotone_treatment_response else ""),
        "inputs": {
            "p_y1_treated": round(p1, 5),
            "p_y1_control": round(p0, 5),
            "p_treated": round(pt, 5),
        },
        "ey1_bounds": [round(ey1_low, 5), round(ey1_high, 5)],
        "ey0_bounds": [round(ey0_low, 5), round(ey0_high, 5)],
        "ate_bounds": [round(ate_low, 5), round(ate_high, 5)],
        "identified_sign": "positive" if ate_low > 0 else "negative" if ate_high < 0 else "not identified",
        "interpretation": "If the ATE interval crosses 0, the effect sign is not identified without stronger assumptions.",
    }


def manski_bounds_from_data(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    *,
    monotone_treatment_response: bool = False,
) -> Dict[str, Any]:
    work = df[[treatment_col, outcome_col]].copy()
    work[treatment_col] = pd.to_numeric(work[treatment_col], errors="coerce")
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna()
    if len(work) < 10:
        return {"available": False, "reason": "Need at least 10 complete rows."}
    if not set(work[treatment_col].unique()).issubset({0, 1, 0.0, 1.0}):
        return {"available": False, "reason": "Treatment must be binary 0/1."}
    if not set(work[outcome_col].unique()).issubset({0, 1, 0.0, 1.0}):
        return {"available": False, "reason": "Outcome must be binary 0/1."}
    treated = work[work[treatment_col] == 1]
    control = work[work[treatment_col] == 0]
    if len(treated) == 0 or len(control) == 0:
        return {"available": False, "reason": "Need both treated and control rows."}
    res = manski_bounds_binary(
        p_y1_treated=float(treated[outcome_col].mean()),
        p_y1_control=float(control[outcome_col].mean()),
        p_treated=float(work[treatment_col].mean()),
        monotone_treatment_response=monotone_treatment_response,
    )
    res["available"] = True
    res["n"] = int(len(work))
    return res


def bias_factor(
    observed_rr: float,
    confounding_rr_exposure_outcome: float,
    confounding_prevalence_exposed: float = 0.5,
    confounding_prevalence_unexposed: float = 0.5,
) -> float:
    """
    Approximate bias factor by which the observed RR may be inflated
    due to an unmeasured binary confounder (Greenland/Schlesselman formulation).
    Returns a new float (immutable result).
    """
    if observed_rr <= 0 or confounding_rr_exposure_outcome <= 0:
        return float("nan")

    pe = max(0.0, min(1.0, confounding_prevalence_exposed))
    pu = max(0.0, min(1.0, confounding_prevalence_unexposed))

    numerator = confounding_rr_exposure_outcome * pe + (1.0 - pe)
    denominator = confounding_rr_exposure_outcome * pu + (1.0 - pu)

    if denominator < 1e-12:
        return float("inf")

    return round(numerator / denominator, 3)


def quantitative_bias_analysis(
    observed_estimate: float,
    measure: Literal["rr", "or", "hr"] = "rr",
    confounding_strength: float = 2.0,
    prevalence_exposed: float = 0.5,
    prevalence_unexposed: float = 0.5,
) -> Dict[str, Any]:
    """
    Simple quantitative bias analysis (QBA) for unmeasured confounding.
    Returns a fresh dict; never mutates inputs.
    """
    if observed_estimate <= 0:
        observed_estimate = 1.0

    bias = bias_factor(
        observed_rr=observed_estimate if measure != "or" else max(observed_estimate, 0.1),
        confounding_rr_exposure_outcome=confounding_strength,
        confounding_prevalence_exposed=prevalence_exposed,
        confounding_prevalence_unexposed=prevalence_unexposed,
    )

    if bias <= 0 or math.isnan(bias):
        bias = 1.0

    if measure == "rr":
        corrected = observed_estimate / bias
    else:
        corrected = observed_estimate / bias   # rough on the ratio scale

    direction = "away from the null" if bias > 1.01 else "toward the null (or none)"

    return {
        "observed_estimate": round(observed_estimate, 3),
        "assumed_confounder_risk_ratio": confounding_strength,
        "prevalence_exposed": prevalence_exposed,
        "prevalence_unexposed": prevalence_unexposed,
        "bias_factor": bias,
        "bias_corrected_estimate": round(corrected, 3),
        "bias_direction": direction,
        "interpretation": (
            f"Under an unmeasured confounder with RR={confounding_strength} and prevalences "
            f"{prevalence_exposed:.0%} vs {prevalence_unexposed:.0%}, bias factor ≈ {bias}. "
            f"Corrected estimate ≈ {round(corrected, 3)} ({direction})."
        ),
    }


def multi_confounder_sensitivity(
    observed_estimate: float,
    confounders: List[Dict[str, float]],
    *,
    measure: Literal["rr", "or", "hr"] = "rr",
) -> Dict[str, Any]:
    """
    Array-based QBA for multiple unmeasured confounders.

    Assumes multiplicative bias factors across confounders; this is a pragmatic
    sensitivity grid rather than a full structural model.
    """
    corrected = float(observed_estimate if observed_estimate > 0 else 1.0)
    rows = []
    cumulative_bias = 1.0
    for i, spec in enumerate(confounders, start=1):
        strength = float(spec.get("rr", spec.get("strength", 1.0)))
        pe = float(spec.get("prevalence_exposed", spec.get("pe", 0.5)))
        pu = float(spec.get("prevalence_unexposed", spec.get("pu", 0.5)))
        bf = bias_factor(corrected, strength, pe, pu)
        if not np.isfinite(bf) or bf <= 0:
            bf = 1.0
        cumulative_bias *= bf
        corrected = corrected / bf
        rows.append({
            "confounder": spec.get("name", f"U{i}"),
            "rr_with_outcome": round(strength, 5),
            "prevalence_exposed": round(pe, 5),
            "prevalence_unexposed": round(pu, 5),
            "bias_factor": round(float(bf), 5),
            "cumulative_bias_factor": round(float(cumulative_bias), 5),
            "corrected_estimate_after_step": round(float(corrected), 5),
        })

    return {
        "observed_estimate": round(float(observed_estimate), 5),
        "measure": measure,
        "n_confounders": int(len(confounders)),
        "confounder_steps": rows,
        "combined_bias_factor": round(float(cumulative_bias), 5),
        "bias_corrected_estimate": round(float(corrected), 5),
        "method_note": "Bias factors are multiplied across unmeasured confounders; interpret as scenario analysis.",
    }


def negative_control_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    negative_control_outcome_col: str,
    covariates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Negative control outcome screen.

    A non-null treatment association with an outcome known not to be caused by
    treatment suggests residual confounding, selection bias, or measurement bias.
    """
    import statsmodels.api as sm

    covariates = covariates or []
    cols = [treatment_col, negative_control_outcome_col] + covariates
    work = df[cols].copy().dropna()
    if len(work) < 20:
        return {"available": False, "reason": "Need at least 20 complete rows."}
    y = pd.to_numeric(work[negative_control_outcome_col], errors="coerce")
    t = pd.to_numeric(work[treatment_col], errors="coerce")
    X_raw = work[[treatment_col] + covariates].copy()
    X_raw[treatment_col] = t
    X = pd.get_dummies(X_raw, drop_first=True).apply(pd.to_numeric, errors="coerce")
    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(frame) < 20:
        return {"available": False, "reason": "Need at least 20 numeric/encoded complete rows."}
    y_arr = frame["__y__"]
    X = sm.add_constant(frame.drop(columns=["__y__"]), has_constant="add")
    try:
        binary = set(y_arr.unique()).issubset({0, 1, 0.0, 1.0})
        model = sm.Logit(y_arr, X).fit(disp=False, maxiter=100) if binary else sm.OLS(y_arr, X).fit()
        term = treatment_col if treatment_col in model.params.index else str(treatment_col)
        beta = float(model.params.get(term, np.nan))
        se = float(model.bse.get(term, np.nan))
        p = float(model.pvalues.get(term, np.nan))
        effect = math.exp(beta) if binary and np.isfinite(beta) else beta
        return {
            "available": True,
            "model": "logistic" if binary else "linear",
            "n": int(len(frame)),
            "negative_control_outcome": negative_control_outcome_col,
            "treatment_effect": round(float(effect), 5) if np.isfinite(effect) else None,
            "coefficient": round(beta, 5) if np.isfinite(beta) else None,
            "se": round(se, 5) if np.isfinite(se) else None,
            "p": round(p, 6) if np.isfinite(p) else None,
            "flag_residual_bias": bool(np.isfinite(p) and p < 0.05),
            "interpretation": "Significant association with a negative control outcome suggests residual bias." if np.isfinite(p) and p < 0.05 else "No clear negative-control signal detected.",
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def rosenbaum_bounds_from_matched_data(
    df: pd.DataFrame,
    match_id_col: str,
    treatment_col: str,
    outcome_col: str,
    *,
    gamma_max: float = 3.0,
    n_gamma: int = 60,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Rosenbaum sensitivity bounds for clean 1:1 matched binary outcomes."""
    try:
        from services.psm import _rosenbaum_bounds

        work = df[[match_id_col, treatment_col, outcome_col]].copy().dropna()
        work[treatment_col] = pd.to_numeric(work[treatment_col], errors="coerce")
        work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
        work = work.dropna()
        if not set(work[treatment_col].unique()).issubset({0, 1, 0.0, 1.0}):
            return {"applicable": False, "reason": "Treatment must be binary 0/1."}
        if not set(work[outcome_col].unique()).issubset({0, 1, 0.0, 1.0}):
            return {"applicable": False, "reason": "Outcome must be binary 0/1."}
        pair_outcomes: List[tuple[int, int]] = []
        skipped = 0
        for _, grp in work.groupby(match_id_col):
            t_rows = grp[grp[treatment_col] == 1]
            c_rows = grp[grp[treatment_col] == 0]
            if len(t_rows) == 1 and len(c_rows) == 1:
                pair_outcomes.append((int(t_rows[outcome_col].iloc[0]), int(c_rows[outcome_col].iloc[0])))
            else:
                skipped += 1
        if not pair_outcomes:
            return {"applicable": False, "reason": "No clean 1:1 matched pairs available."}
        res = _rosenbaum_bounds(pair_outcomes, gamma_max=gamma_max, n_gamma=n_gamma, alpha=alpha)
        res["n_pairs_used"] = int(len(pair_outcomes))
        res["n_pairs_skipped"] = int(skipped)
        res["method_note"] = "Rosenbaum bounds assess hidden bias in 1:1 matched binary-outcome analyses."
        return res
    except Exception as exc:
        return {"applicable": False, "reason": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# Integration helper for PSM / IPTW results (Phase 5 requirement)
# ═══════════════════════════════════════════════════════════════════════════════

def e_value_from_psm_or_iptw(
    treatment_effect_on_ratio_scale: float,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    measure: Literal["rr", "or", "hr"] = "rr",
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute E-value directly from a marginal treatment effect
    typically obtained after PSM or IPTW (e.g. marginal RR or HR).

    This is the main integration point with the existing PSM/IPTW modules.
    """
    return e_value(
        estimate=treatment_effect_on_ratio_scale,
        ci_low=ci_low,
        ci_high=ci_high,
        measure=measure,
        rare_outcome=False,
    )
