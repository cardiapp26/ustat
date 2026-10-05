"""Evidence-based-medicine risk measures for a 2x2 table (pure functions).

Given the event counts in an exposed group and a reference group this returns
the quantities a clinician reads next to a proportion test: the absolute risk
difference with a CI, the risk ratio (Katz log CI), the relative risk
reduction, and the number needed to treat / harm with the Altman (1998)
confidence interval convention.

Conventions
  * ARD is risk_exposed - risk_reference. `arr` is its negative (reference
    minus exposed), the absolute risk reduction when the exposure protects.
  * NNT is reported when the exposed risk is lower than the reference risk,
    NNH when it is higher. The point estimate and the interval limits are
    rounded UP to the next whole number (Altman 1998).
  * When the ARD CI spans zero the NNT interval is not a single interval:
    it runs from the point estimate's side limit to infinity and then
    continues from infinity back to the other kind, e.g. "NNH 8 to infinity
    to NNT 25". That form is returned both as text and as structured fields.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from scipy import stats as sp


def _finite(x: Optional[float]) -> bool:
    return x is not None and math.isfinite(x)


def _ceil_int(x: float) -> int:
    """Round up to a whole number, tolerant of float noise (1/0.2 -> 5)."""
    return int(math.ceil(x - 1e-9))


def newcombe_ard_ci(
    k_exposed: int, n_exposed: int, k_reference: int, n_reference: int, alpha: float
) -> tuple[Optional[float], Optional[float]]:
    """Newcombe (unpooled score) CI for p_exposed - p_reference."""
    from statsmodels.stats.proportion import confint_proportions_2indep

    if n_exposed <= 0 or n_reference <= 0:
        return None, None
    try:
        lo, hi = confint_proportions_2indep(
            k_exposed, n_exposed, k_reference, n_reference,
            method="newcomb", alpha=alpha,
        )
    except Exception:
        return None, None
    lo, hi = float(lo), float(hi)
    return (lo, hi) if _finite(lo) and _finite(hi) else (None, None)


def risk_ratio_katz(
    k_exposed: int, n_exposed: int, k_reference: int, n_reference: int, alpha: float
) -> dict[str, Optional[float]]:
    """Risk ratio with the log-based Katz CI. Undefined cells give None."""
    out: dict[str, Optional[float]] = {"rr": None, "ci_low": None, "ci_high": None}
    if n_exposed <= 0 or n_reference <= 0 or k_reference <= 0:
        return out
    rr = (k_exposed / n_exposed) / (k_reference / n_reference)
    out["rr"] = float(rr)
    if k_exposed <= 0:
        return out
    se = math.sqrt(
        1.0 / k_exposed - 1.0 / n_exposed + 1.0 / k_reference - 1.0 / n_reference
    )
    z = float(sp.norm.ppf(1 - alpha / 2))
    out["ci_low"] = float(math.exp(math.log(rr) - z * se))
    out["ci_high"] = float(math.exp(math.log(rr) + z * se))
    return out


def number_needed(ard: float, ard_ci: tuple[Optional[float], Optional[float]]) -> dict[str, Any]:
    """NNT / NNH from an ARD (exposed minus reference) and its CI (Altman 1998).

    Negative ARD (exposed risk lower) -> NNT; positive -> NNH. A zero ARD has
    an infinite number needed and returns value None.
    """
    lo, hi = ard_ci
    out: dict[str, Any] = {
        "kind": None, "value": None, "low": None, "high": None,
        "ci_spans_zero": False, "other_kind": None, "other_low": None,
        "ci_text": None,
    }
    if not _finite(ard):
        return out
    if ard == 0:
        out["ci_text"] = "Number needed is infinite (no risk difference)"
        return out

    kind = "NNT" if ard < 0 else "NNH"
    other = "NNH" if kind == "NNT" else "NNT"
    out["kind"] = kind
    out["value"] = _ceil_int(1.0 / abs(ard))

    if not (_finite(lo) and _finite(hi)):
        return out

    if lo < 0 < hi:
        # The interval spans no effect: two-part form through infinity.
        out["ci_spans_zero"] = True
        out["other_kind"] = other
        # Benefit side is the negative limit, harm side the positive limit.
        point_limit = abs(lo) if kind == "NNT" else hi
        other_limit = hi if kind == "NNT" else abs(lo)
        out["low"] = _ceil_int(1.0 / point_limit) if point_limit > 0 else None
        out["other_low"] = _ceil_int(1.0 / other_limit) if other_limit > 0 else None
        parts = []
        if out["low"] is not None:
            parts.append(f"{kind} {out['low']} to infinity")
        else:
            parts.append(f"{kind} infinity")
        if out["other_low"] is not None:
            parts.append(f"{other} {out['other_low']}")
        else:
            parts.append(f"{other} infinity")
        out["ci_text"] = " to ".join(parts)
        return out

    near, far = sorted((abs(lo), abs(hi)))
    out["low"] = _ceil_int(1.0 / far) if far > 0 else None
    out["high"] = _ceil_int(1.0 / near) if near > 0 else None
    out["ci_text"] = (
        f"{kind} {out['low']} to "
        + (str(out["high"]) if out["high"] is not None else "infinity")
    )
    return out


def compute_risk_measures(
    k_exposed: int,
    n_exposed: int,
    k_reference: int,
    n_reference: int,
    alpha: float = 0.05,
    event: str = "",
    exposed: str = "",
    reference: str = "",
    ard_ci: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> dict[str, Any]:
    """Assemble the `risk_measures` payload for one 2x2 comparison."""
    risk_e = k_exposed / n_exposed if n_exposed > 0 else float("nan")
    risk_r = k_reference / n_reference if n_reference > 0 else float("nan")
    ard = risk_e - risk_r
    if ard_ci is None:
        ard_ci = newcombe_ard_ci(k_exposed, n_exposed, k_reference, n_reference, alpha)

    rr = risk_ratio_katz(k_exposed, n_exposed, k_reference, n_reference, alpha)
    rrr = rrr_lo = rrr_hi = None
    if rr["rr"] is not None:
        rrr = 1.0 - rr["rr"]
        if rr["ci_low"] is not None and rr["ci_high"] is not None:
            rrr_lo, rrr_hi = 1.0 - rr["ci_high"], 1.0 - rr["ci_low"]

    return {
        "event": event,
        "exposed": exposed,
        "reference": reference,
        "ci_level": round(1 - alpha, 6),
        "risk_exposed": float(risk_e),
        "risk_reference": float(risk_r),
        "ard": float(ard),
        "ard_ci": [ard_ci[0], ard_ci[1]],
        "ard_ci_method": "Newcombe (unpooled score)",
        "arr": float(-ard),
        "rr": rr["rr"],
        "rr_ci": [rr["ci_low"], rr["ci_high"]],
        "rr_ci_method": "Katz log",
        "rrr": rrr,
        "rrr_ci": [rrr_lo, rrr_hi],
        "nnt_nnh": number_needed(ard, ard_ci),
    }


def _fmt(x: Optional[float], digits: int = 3) -> str:
    return f"{x:.{digits}f}" if _finite(x) else "N/A"


def risk_measures_text(rm: dict[str, Any]) -> str:
    """One or two sentences for result_text."""
    pct = f"{100 * rm['ci_level']:g}%"
    ard_lo, ard_hi = rm["ard_ci"]
    text = (
        f"Absolute risk difference ({rm['exposed']} minus {rm['reference']}) = "
        f"{_fmt(rm['ard'])} ({pct} CI {_fmt(ard_lo)} to {_fmt(ard_hi)}); "
        f"risk ratio = {_fmt(rm['rr'])} ({pct} CI {_fmt(rm['rr_ci'][0])} to "
        f"{_fmt(rm['rr_ci'][1])}); relative risk reduction = {_fmt(rm['rrr'])}."
    )
    nn = rm["nnt_nnh"]
    if nn["value"] is not None:
        ci = f" ({pct} CI {nn['ci_text']})" if nn["ci_text"] else ""
        text += f" {nn['kind']} = {nn['value']}{ci}."
    elif nn["ci_text"]:
        text += f" {nn['ci_text']}."
    return text


def risk_measures_export_rows(rm: dict[str, Any]) -> list[list[Any]]:
    """Rows to append to an export table."""
    pct = f"{100 * rm['ci_level']:g}%"
    nn = rm["nnt_nnh"]

    def r(x: Optional[float]) -> Any:
        return round(x, 4) if _finite(x) else "N/A"

    rows: list[list[Any]] = [
        [f"Risk ({rm['exposed']})", r(rm["risk_exposed"])],
        [f"Risk ({rm['reference']})", r(rm["risk_reference"])],
        ["ARD (exposed minus reference)", r(rm["ard"])],
        [f"ARD {pct} CI lower", r(rm["ard_ci"][0])],
        [f"ARD {pct} CI upper", r(rm["ard_ci"][1])],
        ["Risk ratio", r(rm["rr"])],
        [f"Risk ratio {pct} CI lower", r(rm["rr_ci"][0])],
        [f"Risk ratio {pct} CI upper", r(rm["rr_ci"][1])],
        ["Relative risk reduction", r(rm["rrr"])],
        [f"RRR {pct} CI lower", r(rm["rrr_ci"][0])],
        [f"RRR {pct} CI upper", r(rm["rrr_ci"][1])],
    ]
    if nn["value"] is not None:
        rows.append([nn["kind"], nn["value"]])
        rows.append([f"{nn['kind']} {pct} CI", nn["ci_text"] or "N/A"])
    else:
        rows.append(["NNT/NNH", "infinite"])
    return rows
