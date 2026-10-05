"""
Rate standardisation and incidence-rate comparison (pure functions).

Takes inline stratum tables, never a session dataframe, so it works without an
uploaded dataset. Everything here raises ValueError on invalid input; the
router turns that into a 422.

Methods
-------
* Exact Poisson (Garwood) CI for a count k over exposure t:
      lower = chi2(a/2, 2k) / 2 / t,   upper = chi2(1 - a/2, 2k + 2) / 2 / t
* Direct standardisation: DSR = sum(w_i * rate_i), w_i = standard_pop_i / total.
  CI by the gamma method of Fay & Feuer (1997), identical to R
  `epitools::ageadjust.direct`:
      var  = sum(w_i^2 * events_i / pt_i^2),  wm = max(w_i / pt_i)
      lower = qgamma(a/2,   shape = DSR^2 / var,                scale = var / DSR)
      upper = qgamma(1-a/2, shape = (DSR+wm)^2 / (var + wm^2),  scale = (var + wm^2) / (DSR+wm))
* Indirect standardisation: E = sum(pt_i * ref_rate_i), SMR = O / E.
  Byar (1977) approximation and the exact Poisson CI for O given E, and the
  exact two-sided Poisson test of H0: SMR = 1.
* Rate ratio / difference for two groups, log-method Wald CI plus a
  conditional exact (binomial) test and CI.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

MAX_STRATA = 1000


# ── helpers ──────────────────────────────────────────────────────────────────


def _clean(x: Optional[float]) -> Optional[float]:
    """Finite float or None (JSON-safe)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _check_alpha(alpha: float) -> float:
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be a finite number strictly between 0 and 1.")
    return float(alpha)


def _check_multiplier(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0.")
    return float(value)


def _num(value: Any, field: str, where: str, *, positive: bool = False) -> float:
    """Finite, non-negative number (strictly positive when `positive`)."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{where}: '{field}' is required.")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: '{field}' must be a number.")
    if not math.isfinite(x):
        raise ValueError(f"{where}: '{field}' must be finite.")
    if x < 0:
        raise ValueError(f"{where}: '{field}' must not be negative.")
    if positive and x <= 0:
        raise ValueError(f"{where}: '{field}' must be greater than 0.")
    return x


def _count(value: Any, field: str, where: str) -> int:
    """Non-negative integer event count (floats with integral value accepted)."""
    x = _num(value, field, where)
    if abs(x - round(x)) > 1e-9:
        raise ValueError(f"{where}: '{field}' must be a whole number (got {value}).")
    return int(round(x))


def _fmt(x: Optional[float], digits: int = 2) -> str:
    return "NA" if x is None else f"{x:.{digits}f}"


def _pct(level: float) -> str:
    return f"{level * 100:g}%"


def _check_strata_count(strata: Sequence[Any]) -> None:
    if not strata:
        raise ValueError("Provide at least one stratum.")
    if len(strata) > MAX_STRATA:
        raise ValueError(f"At most {MAX_STRATA} strata are allowed.")


def _label(item: Dict[str, Any], i: int) -> str:
    lab = item.get("label")
    return str(lab) if lab not in (None, "") else f"Stratum {i + 1}"


def poisson_exact_ci(k: int, exposure: float, alpha: float = 0.05) -> Dict[str, Optional[float]]:
    """Exact (Garwood) CI for the rate k / exposure. Lower is 0 when k = 0."""
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2.0, 2 * k) / 2.0 / exposure
    hi = stats.chi2.ppf(1.0 - alpha / 2.0, 2 * k + 2) / 2.0 / exposure
    return {"low": float(lo), "high": float(hi)}


def _poisson_count_ci(k: int, alpha: float) -> tuple[float, float]:
    """Exact CI for a Poisson mean given the observed count k."""
    lo = 0.0 if k == 0 else float(stats.chi2.ppf(alpha / 2.0, 2 * k)) / 2.0
    hi = float(stats.chi2.ppf(1.0 - alpha / 2.0, 2 * k + 2)) / 2.0
    return lo, hi


def byar_ci(observed: int, alpha: float = 0.05) -> tuple[float, float]:
    """Byar (1977) CI for a Poisson count (to be divided by E for the SMR)."""
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    o = float(observed)
    if observed == 0:
        lo = 0.0
    else:
        lo = o * (1.0 - 1.0 / (9.0 * o) - z / (3.0 * math.sqrt(o))) ** 3
        lo = max(lo, 0.0)
    o1 = o + 1.0
    hi = o1 * (1.0 - 1.0 / (9.0 * o1) + z / (3.0 * math.sqrt(o1))) ** 3
    return lo, hi


def poisson_two_sided_p(observed: int, expected: float) -> float:
    """Exact two-sided Poisson p-value (twice the smaller tail, capped at 1)."""
    lower = float(stats.poisson.cdf(observed, expected))
    upper = float(stats.poisson.sf(observed - 1, expected)) if observed > 0 else 1.0
    return float(min(1.0, 2.0 * min(lower, upper)))


# ── 1. direct standardisation ────────────────────────────────────────────────


def direct_standardisation(
    strata: List[Dict[str, Any]],
    multiplier: float = 100000.0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Directly standardised rate with Fay-Feuer gamma CI.

    Each stratum: label, events, person_time (or population), standard_population.
    """
    alpha = _check_alpha(alpha)
    multiplier = _check_multiplier(multiplier, "multiplier")
    _check_strata_count(strata)

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for i, s in enumerate(strata):
        lab = _label(s, i)
        where = f"Stratum '{lab}'"
        events = _count(s.get("events"), "events", where)
        pt_raw = s.get("person_time")
        if pt_raw is None:
            pt_raw = s.get("population")
        pt = _num(pt_raw, "person_time", where)
        std = _num(s.get("standard_population"), "standard_population", where)
        if pt == 0 and events > 0:
            raise ValueError(f"{where}: events > 0 but person_time is 0 (rate undefined).")
        rows.append({"label": lab, "events": events, "person_time": pt, "standard_population": std})

    included: List[int] = []
    for i, r in enumerate(rows):
        if r["person_time"] == 0:
            warnings.append(
                f"Stratum '{r['label']}' has zero person-time; it is excluded and the "
                "standard weights are renormalised over the remaining strata."
            )
        elif r["standard_population"] == 0:
            warnings.append(
                f"Stratum '{r['label']}' has zero standard population; its weight is 0 "
                "and it does not contribute to the standardised rate."
            )
            included.append(i)
        else:
            included.append(i)

    total_std = sum(rows[i]["standard_population"] for i in included)
    if total_std <= 0:
        raise ValueError(
            "No usable stratum: need at least one stratum with person_time > 0 and "
            "standard_population > 0."
        )

    events_arr = np.array([rows[i]["events"] for i in included], dtype=float)
    pt_arr = np.array([rows[i]["person_time"] for i in included], dtype=float)
    std_arr = np.array([rows[i]["standard_population"] for i in included], dtype=float)
    w = std_arr / total_std
    rate = events_arr / pt_arr

    dsr = float(np.sum(w * rate))
    dsr_var = float(np.sum(w ** 2 * events_arr / pt_arr ** 2))
    wm = float(np.max(w / pt_arr))

    if dsr_var > 0 and dsr > 0:
        lo = float(stats.gamma.ppf(alpha / 2.0, a=dsr ** 2 / dsr_var, scale=dsr_var / dsr))
    else:
        lo = 0.0
    hi = float(stats.gamma.ppf(
        1.0 - alpha / 2.0,
        a=(dsr + wm) ** 2 / (dsr_var + wm ** 2),
        scale=(dsr_var + wm ** 2) / (dsr + wm),
    ))

    total_events = int(sum(r["events"] for r in rows))
    total_pt = float(sum(r["person_time"] for r in rows))
    if total_pt <= 0:
        raise ValueError("Total person_time must be greater than 0.")
    crude = total_events / total_pt
    crude_ci = poisson_exact_ci(total_events, total_pt, alpha)

    table: List[Dict[str, Any]] = []
    inc_pos = {idx: pos for pos, idx in enumerate(included)}
    for i, r in enumerate(rows):
        if i in inc_pos:
            p = inc_pos[i]
            r_i = float(rate[p])
            ci = poisson_exact_ci(r["events"], r["person_time"], alpha)
            contrib = float(w[p] * r_i)
            table.append({
                "label": r["label"],
                "events": r["events"],
                "person_time": r["person_time"],
                "standard_population": r["standard_population"],
                "rate": r_i * multiplier,
                "ci_low": ci["low"] * multiplier,
                "ci_high": ci["high"] * multiplier,
                "weight": float(w[p]),
                "contribution": contrib * multiplier,
                "contribution_pct": (contrib / dsr * 100.0) if dsr > 0 else None,
                "excluded": False,
            })
        else:
            table.append({
                "label": r["label"],
                "events": r["events"],
                "person_time": r["person_time"],
                "standard_population": r["standard_population"],
                "rate": None, "ci_low": None, "ci_high": None,
                "weight": None, "contribution": None, "contribution_pct": None,
                "excluded": True,
            })

    level = _pct(1.0 - alpha)
    result = {
        "test": "Direct standardisation",
        "alpha": alpha,
        "conf_level": 1.0 - alpha,
        "multiplier": multiplier,
        "n_strata": len(rows),
        "total_events": total_events,
        "total_person_time": total_pt,
        "crude_rate": crude * multiplier,
        "crude_ci_low": crude_ci["low"] * multiplier,
        "crude_ci_high": crude_ci["high"] * multiplier,
        "standardised_rate": dsr * multiplier,
        "standardised_ci_low": lo * multiplier,
        "standardised_ci_high": hi * multiplier,
        "standardised_se": math.sqrt(dsr_var) * multiplier,
        "ci_method": "Fay & Feuer (1997) gamma interval (epitools::ageadjust.direct)",
        "strata": table,
        "warnings": warnings,
    }
    mult_txt = f"{multiplier:g}"
    result["result_text"] = (
        f"Rates were directly standardised to the supplied standard population across "
        f"{len(rows)} strata. The crude rate was {_fmt(result['crude_rate'])} per {mult_txt} "
        f"person-time ({level} exact Poisson CI {_fmt(result['crude_ci_low'])} to "
        f"{_fmt(result['crude_ci_high'])}); the directly standardised rate was "
        f"{_fmt(result['standardised_rate'])} per {mult_txt} ({level} CI "
        f"{_fmt(result['standardised_ci_low'])} to {_fmt(result['standardised_ci_high'])}, "
        "gamma method of Fay and Feuer)."
    )
    result["r_code"] = _direct_r_code(rows, multiplier, alpha)
    return result


def _r_vec(values: Sequence[float]) -> str:
    return "c(" + ", ".join(f"{v:g}" for v in values) + ")"


def _direct_r_code(rows: List[Dict[str, Any]], multiplier: float, alpha: float) -> str:
    return (
        "library(epitools)\n"
        f"count  <- {_r_vec([r['events'] for r in rows])}\n"
        f"pop    <- {_r_vec([r['person_time'] for r in rows])}\n"
        f"stdpop <- {_r_vec([r['standard_population'] for r in rows])}\n"
        f"res <- ageadjust.direct(count = count, pop = pop, stdpop = stdpop, "
        f"conf.level = {1.0 - alpha:g})\n"
        f"res * {multiplier:g}   # crude.rate and adj.rate with CI, per {multiplier:g}\n"
        f"pois.exact(sum(count), sum(pop), conf.level = {1.0 - alpha:g})  # crude rate, exact CI"
    )


# ── 2. indirect standardisation ──────────────────────────────────────────────


def indirect_standardisation(
    strata: List[Dict[str, Any]],
    reference_multiplier: float = 1.0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """SMR with Byar and exact Poisson CIs.

    Each stratum: label, observed (events in the study group), person_time (study
    group), and EITHER reference_rate (per `reference_multiplier` person-time
    units, e.g. 50 with reference_multiplier = 100000) OR reference_events plus
    reference_person_time from which the rate is derived.
    """
    alpha = _check_alpha(alpha)
    reference_multiplier = _check_multiplier(reference_multiplier, "reference_multiplier")
    _check_strata_count(strata)

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for i, s in enumerate(strata):
        lab = _label(s, i)
        where = f"Stratum '{lab}'"
        obs = _count(s.get("observed"), "observed", where)
        pt = _num(s.get("person_time"), "person_time", where)
        if s.get("reference_rate") is not None:
            rate = _num(s["reference_rate"], "reference_rate", where) / reference_multiplier
            source = "reference_rate"
        elif s.get("reference_events") is not None or s.get("reference_person_time") is not None:
            ref_ev = _count(s.get("reference_events"), "reference_events", where)
            ref_pt = _num(s.get("reference_person_time"), "reference_person_time", where, positive=True)
            rate = ref_ev / ref_pt
            source = "reference_events/reference_person_time"
        else:
            raise ValueError(
                f"{where}: provide 'reference_rate' or both 'reference_events' and "
                "'reference_person_time'."
            )
        if pt == 0 and obs > 0:
            raise ValueError(f"{where}: observed > 0 but person_time is 0.")
        if pt == 0:
            warnings.append(f"Stratum '{lab}' has zero person-time and contributes nothing.")
        rows.append({"label": lab, "observed": obs, "person_time": pt, "rate": rate, "source": source})

    observed = int(sum(r["observed"] for r in rows))
    expected_parts = [r["person_time"] * r["rate"] for r in rows]
    expected = float(sum(expected_parts))
    if expected <= 0:
        raise ValueError(
            "Expected count is 0 (all reference rates or person-times are zero); the SMR is undefined."
        )

    smr = observed / expected
    byar_lo, byar_hi = byar_ci(observed, alpha)
    ex_lo, ex_hi = _poisson_count_ci(observed, alpha)
    p_exact = poisson_two_sided_p(observed, expected)

    table = []
    for r, e_i in zip(rows, expected_parts):
        table.append({
            "label": r["label"],
            "observed": r["observed"],
            "person_time": r["person_time"],
            "reference_rate": r["rate"] * reference_multiplier,
            "expected": float(e_i),
            "ratio": (r["observed"] / e_i) if e_i > 0 else None,
        })

    level = _pct(1.0 - alpha)
    if observed < 10:
        warnings.append(
            "Fewer than 10 observed events: prefer the exact Poisson interval over Byar's approximation."
        )
    result = {
        "test": "Indirect standardisation (SMR)",
        "alpha": alpha,
        "conf_level": 1.0 - alpha,
        "reference_multiplier": reference_multiplier,
        "n_strata": len(rows),
        "observed": observed,
        "expected": expected,
        "smr": smr,
        "smr_x100": smr * 100.0,
        "byar_ci_low": byar_lo / expected,
        "byar_ci_high": byar_hi / expected,
        "byar_ci_low_x100": byar_lo / expected * 100.0,
        "byar_ci_high_x100": byar_hi / expected * 100.0,
        "exact_ci_low": ex_lo / expected,
        "exact_ci_high": ex_hi / expected,
        "exact_ci_low_x100": ex_lo / expected * 100.0,
        "exact_ci_high_x100": ex_hi / expected * 100.0,
        "p_value": p_exact,
        "p_method": "Exact two-sided Poisson (twice the smaller tail) for H0: SMR = 1",
        "strata": table,
        "warnings": warnings,
    }
    result["result_text"] = (
        f"Indirect standardisation against the reference rates gave {observed} observed and "
        f"{expected:.2f} expected events (SMR = {smr:.2f}, i.e. {smr * 100:.0f} when "
        f"expressed per 100; {level} exact Poisson CI {result['exact_ci_low']:.2f} to "
        f"{result['exact_ci_high']:.2f}; Byar CI {result['byar_ci_low']:.2f} to "
        f"{result['byar_ci_high']:.2f}). The exact two-sided Poisson test of SMR = 1 gave "
        f"p = {_p_txt(p_exact)}."
    )
    result["r_code"] = (
        "library(epitools)\n"
        f"observed <- {observed}\n"
        f"expected <- {expected:.10g}   # sum(person_time * reference_rate)\n"
        f"# SMR with exact Poisson CI (rate = O / E, exposure = E)\n"
        f"pois.exact(x = observed, pt = expected, conf.level = {1.0 - alpha:g})\n"
        f"poisson.test(observed, T = expected, r = 1)$p.value   # exact test of SMR = 1"
    )
    return result


def _p_txt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ── 3. rate ratio / difference ───────────────────────────────────────────────


def rate_ratio(
    events1: Any,
    person_time1: Any,
    events2: Any,
    person_time2: Any,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two-group incidence-rate comparison (group 1 relative to group 2)."""
    alpha = _check_alpha(alpha)
    e1 = _count(events1, "events1", "Group 1")
    e2 = _count(events2, "events2", "Group 2")
    t1 = _num(person_time1, "person_time1", "Group 1", positive=True)
    t2 = _num(person_time2, "person_time2", "Group 2", positive=True)

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    r1, r2 = e1 / t1, e2 / t2
    ci1 = poisson_exact_ci(e1, t1, alpha)
    ci2 = poisson_exact_ci(e2, t2, alpha)
    warnings: List[str] = []

    # rate difference (Wald)
    diff = r1 - r2
    se_diff = math.sqrt(e1 / t1 ** 2 + e2 / t2 ** 2)
    diff_lo, diff_hi = diff - z * se_diff, diff + z * se_diff

    # rate ratio, log method
    rr = rr_lo = rr_hi = se_log = p_wald = None
    if e1 > 0 and e2 > 0:
        rr = r1 / r2
        se_log = math.sqrt(1.0 / e1 + 1.0 / e2)
        rr_lo = math.exp(math.log(rr) - z * se_log)
        rr_hi = math.exp(math.log(rr) + z * se_log)
        p_wald = float(2.0 * stats.norm.sf(abs(math.log(rr) / se_log)))
    elif e1 == 0 and e2 == 0:
        warnings.append("No events in either group: the rate ratio is undefined.")
    else:
        rr = r1 / r2 if e2 > 0 else None
        warnings.append(
            "A group has zero events: the log-method CI is undefined; use the exact "
            "conditional CI."
        )

    # conditional exact inference: e1 | n ~ Binomial(n, t1 / (t1 + t2)) under H0
    n = e1 + e2
    p_exact = p_mid = rr_ex_lo = rr_ex_hi = None
    if n > 0:
        p0 = t1 / (t1 + t2)
        p_exact = float(stats.binomtest(e1, n, p0).pvalue)
        pmf = float(stats.binom.pmf(e1, n, p0))
        low_tail = float(stats.binom.cdf(e1, n, p0)) - pmf / 2.0
        up_tail = float(stats.binom.sf(e1 - 1, n, p0)) - pmf / 2.0
        p_mid = float(min(1.0, 2.0 * min(low_tail, up_tail)))
        # Clopper-Pearson CI on the binomial proportion, mapped to the rate ratio
        pl = 0.0 if e1 == 0 else float(stats.beta.ppf(alpha / 2.0, e1, n - e1 + 1))
        pu = 1.0 if e1 == n else float(stats.beta.ppf(1.0 - alpha / 2.0, e1 + 1, n - e1))
        rr_ex_lo = (pl / (1.0 - pl)) * (t2 / t1)
        rr_ex_hi = (pu / (1.0 - pu)) * (t2 / t1) if pu < 1.0 else None

    level = _pct(1.0 - alpha)
    result = {
        "test": "Rate ratio and rate difference",
        "alpha": alpha,
        "conf_level": 1.0 - alpha,
        "group1": {"events": e1, "person_time": t1, "rate": r1,
                   "ci_low": ci1["low"], "ci_high": ci1["high"]},
        "group2": {"events": e2, "person_time": t2, "rate": r2,
                   "ci_low": ci2["low"], "ci_high": ci2["high"]},
        "rate_ratio": _clean(rr),
        "rr_ci_low": _clean(rr_lo),
        "rr_ci_high": _clean(rr_hi),
        "rr_se_log": _clean(se_log),
        "rr_ci_method": "Log method (Wald on log rate ratio)",
        "rr_exact_ci_low": _clean(rr_ex_lo),
        "rr_exact_ci_high": _clean(rr_ex_hi),
        "rate_difference": float(diff),
        "rd_ci_low": float(diff_lo),
        "rd_ci_high": float(diff_hi),
        "rd_se": float(se_diff),
        "p_value": _clean(p_exact),
        "p_value_midp": _clean(p_mid),
        "p_value_wald": _clean(p_wald),
        "p_method": "Conditional exact (binomial on events1 given total events), two-sided",
        "warnings": warnings,
    }
    if rr is not None and rr_lo is not None:
        rr_part = (
            f"The rate ratio was {rr:.2f} ({level} CI {rr_lo:.2f} to {rr_hi:.2f}, log method)"
        )
    elif rr is not None:
        rr_part = f"The rate ratio was {rr:.2f} (log-method CI undefined with a zero count)"
    else:
        rr_part = "The rate ratio could not be estimated"
    result["result_text"] = (
        f"Incidence rates were {r1:.4g} (events {e1}, person-time {t1:g}; {level} exact CI "
        f"{ci1['low']:.4g} to {ci1['high']:.4g}) in group 1 and {r2:.4g} (events {e2}, "
        f"person-time {t2:g}; {level} exact CI {ci2['low']:.4g} to {ci2['high']:.4g}) in group 2. "
        f"{rr_part}; the rate difference was {diff:.4g} ({level} CI {diff_lo:.4g} to "
        f"{diff_hi:.4g}). "
        + (f"The conditional exact test gave p = {_p_txt(p_exact)} "
           f"(mid-p = {_p_txt(p_mid)})." if p_exact is not None else "No test was possible.")
    )
    result["r_code"] = (
        "library(epitools)\n"
        f"pois.exact(x = c({e1}, {e2}), pt = c({t1:g}, {t2:g}), conf.level = {1.0 - alpha:g})\n"
        "# Rate ratio (group 1 vs group 2) with conditional exact CI and p-value\n"
        f"poisson.test(c({e1}, {e2}), T = c({t1:g}, {t2:g}), conf.level = {1.0 - alpha:g})\n"
        "# epitools::rateratio(...) gives the same comparison with mid-p, Fisher or Wald methods"
    )
    return result
