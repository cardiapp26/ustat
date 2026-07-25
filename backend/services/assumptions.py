"""
Model Assumption Checking Module

Systematic, extensible assumption checks for the statistical models uSTAT
fits.

Purpose:
- Tell the user how well each model assumption is met
- Emit automatic warnings and suggestions
- Improve the quality of the analysis

Model types supported today:
- linear
- logistic
- cox
- ordinal (partial)

Planned:
- gee, mixed models, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AssumptionCheck:
    """Result of a single assumption check."""
    name: str
    passed: bool | None  # None = could not be checked
    p_value: float | None = None
    statistic: float | None = None
    message: str = ""
    severity: Literal["ok", "warning", "critical"] = "ok"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssumptionReport:
    """All assumption checks for one model."""
    model_type: str
    checks: list[AssumptionCheck] = field(default_factory=list)
    overall_severity: Literal["ok", "warning", "critical"] = "ok"
    summary: str = ""

    def to_dict(self) -> dict:
        def _clean(val):
            if isinstance(val, (np.bool_, bool)):
                return bool(val)
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.floating,)):
                return float(val)
            return val

        return {
            "model_type": self.model_type,
            "overall_severity": self.overall_severity,
            "summary": self.summary,
            "checks": [
                {
                    "name": c.name,
                    "passed": _clean(c.passed),
                    "p_value": _clean(c.p_value),
                    "statistic": _clean(c.statistic),
                    "message": c.message,
                    "severity": c.severity,
                    "details": {k: _clean(v) for k, v in c.details.items()},
                }
                for c in self.checks
            ],
        }


def _severity_from_pvalue(p: float | None, critical_threshold: float = 0.01) -> str:
    if p is None:
        return "warning"
    if p < critical_threshold:
        return "critical"
    if p < 0.05:
        return "warning"
    return "ok"


# =============================================================================
# LINEAR REGRESSION ASSUMPTIONS
# =============================================================================

def check_linear_assumptions(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    model,
) -> AssumptionReport:
    """
    Core assumption checks for linear regression.
    """
    checks: list[AssumptionCheck] = []

    # 1. Normality of residuals (Shapiro-Wilk for small n, Jarque-Bera otherwise)
    n = len(residuals)
    if n < 5000:
        stat, p = stats.shapiro(residuals)
        checks.append(
            AssumptionCheck(
                name="residual_normality",
                passed=p > 0.05,
                p_value=p,
                statistic=stat,
                message="Residuals are approximately normal." if p > 0.05 else "Residuals deviate significantly from normality.",
                severity=_severity_from_pvalue(p),
            )
        )
    else:
        # Jarque-Bera is the better fit at large n
        stat, p = stats.jarque_bera(residuals)
        checks.append(
            AssumptionCheck(
                name="residual_normality",
                passed=p > 0.05,
                p_value=p,
                statistic=stat,
                message="Residuals are approximately normal (Jarque-Bera)." if p > 0.05 else "Residuals show significant non-normality.",
                severity=_severity_from_pvalue(p),
            )
        )

    # 2. Homoscedasticity (Breusch-Pagan)
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_test = het_breuschpagan(residuals, model.model.exog)
        bp_p = bp_test[1]
        checks.append(
            AssumptionCheck(
                name="homoscedasticity",
                passed=bp_p > 0.05,
                p_value=bp_p,
                message="Homoscedasticity assumption holds." if bp_p > 0.05 else "Evidence of heteroscedasticity detected.",
                severity=_severity_from_pvalue(bp_p),
            )
        )
    except Exception:
        checks.append(
            AssumptionCheck(
                name="homoscedasticity",
                passed=None,
                message="Could not perform Breusch-Pagan test.",
                severity="warning",
            )
        )

    # 3. Linearity (Rainbow test)
    try:
        from statsmodels.stats.diagnostic import linear_rainbow
        # Safer call
        rainbow_stat, rainbow_p = linear_rainbow(model.model)
        checks.append(
            AssumptionCheck(
                name="linearity",
                passed=rainbow_p > 0.05,
                p_value=rainbow_p,
                statistic=rainbow_stat,
                message="Linear relationship appears reasonable." if rainbow_p > 0.05 else "Possible nonlinearity detected.",
                severity=_severity_from_pvalue(rainbow_p),
            )
        )
    except Exception:
        checks.append(
            AssumptionCheck(
                name="linearity",
                passed=None,
                message="Linearity test could not be performed.",
                severity="warning",
            )
        )

    # 4. Multicollinearity (VIF) - computed separately; only commented on here
    # Kept separate because VIF is already returned by the endpoint.

    # Overall assessment
    critical_count = sum(1 for c in checks if c.severity == "critical")
    warning_count = sum(1 for c in checks if c.severity == "warning")

    if critical_count >= 2:
        overall = "critical"
        summary = "Multiple critical assumption violations detected. Results should be interpreted with great caution."
    elif critical_count == 1 or warning_count >= 2:
        overall = "warning"
        summary = "Some model assumptions may be violated. Review diagnostics carefully."
    else:
        overall = "ok"
        summary = "No major violations of linear regression assumptions detected."

    return AssumptionReport(
        model_type="linear",
        checks=checks,
        overall_severity=overall,
        summary=summary,
    )


# =============================================================================
# LOGISTIC REGRESSION ASSUMPTIONS
# =============================================================================

def check_logistic_assumptions(
    y: np.ndarray,
    pred_probs: np.ndarray,
    hosmer_lemeshow: dict | None = None,
    model=None,
) -> AssumptionReport:
    """
    Assumption and goodness-of-fit checks for binary logistic regression.
    """
    checks: list[AssumptionCheck] = []

    # 1. Linearity of logit
    checks.append(
        AssumptionCheck(
            name="linearity_of_logit",
            passed=None,
            message="Linearity of the logit should be assessed (e.g., Box-Tidwell test or component + residual plots).",
            severity="warning",
        )
    )

    # 2. Hosmer-Lemeshow Goodness-of-Fit
    if hosmer_lemeshow and hosmer_lemeshow.get("p") is not None:
        hl_p = hosmer_lemeshow["p"]
        checks.append(
            AssumptionCheck(
                name="hosmer_lemeshow",
                passed=hl_p >= 0.05,
                p_value=hl_p,
                message="Hosmer-Lemeshow test indicates adequate fit." if hl_p >= 0.05 
                        else "Hosmer-Lemeshow test suggests poor calibration.",
                severity="ok" if hl_p >= 0.05 else "warning",
                details=hosmer_lemeshow,
            )
        )
    else:
        checks.append(
            AssumptionCheck(
                name="hosmer_lemeshow",
                passed=None,
                message="Hosmer-Lemeshow test could not be computed.",
                severity="warning",
            )
        )

    # 3. Separation risk (basit heuristik)
    if model is not None:
        try:
            max_coef = np.max(np.abs(model.params.values))
            if max_coef > 10:
                checks.append(
                    AssumptionCheck(
                        name="separation_risk",
                        passed=False,
                        message=f"Very large coefficients detected (max |coef| ≈ {max_coef:.1f}). Possible complete or quasi-complete separation.",
                        severity="critical",
                    )
                )
        except Exception:
            pass

    critical_count = sum(1 for c in checks if c.severity == "critical")
    warning_count = sum(1 for c in checks if c.severity == "warning")

    if critical_count > 0:
        overall = "critical"
        summary = "Critical issues detected (possible separation or poor fit)."
    elif warning_count > 1:
        overall = "warning"
        summary = "Some assumptions or fit issues require attention."
    else:
        overall = "ok"
        summary = "No major red flags from available logistic diagnostics."

    return AssumptionReport(
        model_type="logistic",
        checks=checks,
        overall_severity=overall,
        summary=summary,
    )


# =============================================================================
# COX PROPORTIONAL HAZARDS
# =============================================================================

def check_cox_assumptions_from_ph_test(ph_test: dict | None) -> AssumptionReport:
    """
    Survival router'da zaten hesaplanan proportional hazards test sonucunu
    Converts to the AssumptionReport format.
    """
    checks: list[AssumptionCheck] = []

    if not ph_test or "error" in ph_test:
        checks.append(
            AssumptionCheck(
                name="proportional_hazards",
                passed=None,
                message="Proportional hazards test could not be computed.",
                severity="warning",
            )
        )
    else:
        per_term = ph_test.get("per_term", [])
        global_p = ph_test.get("global", {}).get("p")

        for term in per_term:
            p = term.get("p")
            var = term.get("variable", "unknown")
            checks.append(
                AssumptionCheck(
                    name=f"proportional_hazards_{var}",
                    passed=p > 0.05 if p is not None else None,
                    p_value=p,
                    message=f"PH assumption holds for {var}." if (p is not None and p > 0.05) 
                            else f"Possible PH violation for {var}.",
                    severity="ok" if (p is not None and p > 0.05) else ("critical" if p is not None and p < 0.01 else "warning"),
                )
            )

        if global_p is not None:
            checks.append(
                AssumptionCheck(
                    name="proportional_hazards_global",
                    passed=global_p > 0.05,
                    p_value=global_p,
                    message="Global proportional hazards assumption appears reasonable." if global_p > 0.05 
                            else "Global test suggests violation of proportional hazards.",
                    severity="ok" if global_p > 0.05 else "warning",
                )
            )

    critical = any(c.severity == "critical" for c in checks)
    warning = any(c.severity == "warning" for c in checks) and not critical

    return AssumptionReport(
        model_type="cox",
        checks=checks,
        overall_severity="critical" if critical else ("warning" if warning else "ok"),
        summary="Proportional hazards assumption should be verified before interpreting hazard ratios.",
    )


# =============================================================================
# GEE & ORDINAL (Placeholder - Phase 1)
# =============================================================================

def check_gee_assumptions_placeholder(family: str, cov_struct: str) -> AssumptionReport:
    checks = [
        AssumptionCheck(
            name="gee_assumptions",
            passed=None,
            message=f"GEE ({family}, {cov_struct}): Working correlation structure and marginal model assumptions should be validated externally.",
            severity="warning",
        )
    ]
    return AssumptionReport(
        model_type="gee",
        checks=checks,
        overall_severity="warning",
        summary="GEE results depend heavily on correct specification of the working correlation structure.",
    )


def check_ordinal_assumptions_placeholder() -> AssumptionReport:
    checks = [
        AssumptionCheck(
            name="proportional_odds",
            passed=None,
            message="Proportional odds assumption should be checked (e.g., Brant test). Not yet automated in this endpoint.",
            severity="warning",
        )
    ]
    return AssumptionReport(
        model_type="ordinal",
        checks=checks,
        overall_severity="warning",
        summary="Ordinal logistic results assume proportional odds across all cutpoints.",
    )


# =============================================================================
# COX PROPORTIONAL HAZARDS
# =============================================================================

def check_cox_assumptions(
    cph_model,  # lifelines CoxPHFitter
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
) -> AssumptionReport:
    """
    Assumption checks for the Cox PH model.
    The Schoenfeld residuals test is already run via lifelines.
    """
    checks: list[AssumptionCheck] = []

    try:
        from lifelines.statistics import proportional_hazard_test

        results = proportional_hazard_test(cph_model, df, time_transform="rank")
        p_values = results.summary["p"]

        for var, p in p_values.items():
            checks.append(
                AssumptionCheck(
                    name=f"proportional_hazards_{var}",
                    passed=p > 0.05,
                    p_value=p,
                    message=f"Proportional hazards assumption holds for {var}." if p > 0.05 else f"Proportional hazards violation detected for {var}.",
                    severity=_severity_from_pvalue(p),
                )
            )
    except Exception as e:
        checks.append(
            AssumptionCheck(
                name="proportional_hazards",
                passed=None,
                message=f"Could not perform proportional hazards test: {str(e)}",
                severity="warning",
            )
        )

    critical = any(c.severity == "critical" for c in checks)

    return AssumptionReport(
        model_type="cox",
        checks=checks,
        overall_severity="critical" if critical else "ok",
        summary="Proportional hazards assumption should be verified before interpreting hazard ratios.",
    )


# =============================================================================
# Helper functions
# =============================================================================

def add_assumption_warnings_to_result(result: dict, report: AssumptionReport) -> dict:
    """Attach the assumption report to an existing result dict."""
    result["assumptions"] = report.to_dict()

    if report.overall_severity in ("warning", "critical"):
        if "warnings" not in result:
            result["warnings"] = []
        result["warnings"].append(f"Model assumption check: {report.summary}")

    return result
