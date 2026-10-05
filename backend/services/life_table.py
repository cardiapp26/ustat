"""Actuarial (Cutler-Ederer) life table for interval-grouped follow-up.

Pure functions, no I/O. A life table groups follow-up into intervals
[t_i, t_{i+1}) and, per interval, works with counts rather than exact times:

    l_i   subjects entering the interval (alive and under observation)
    w_i   withdrawn (censored) during the interval
    d_i   events during the interval
    l'_i  effective number at risk = l_i - w_i / 2   (censoring spread evenly)
    q_i   conditional probability of the event = d_i / l'_i
    p_i   conditional probability of surviving = 1 - q_i
    S_i   cumulative survival at the END of interval i = prod_{j<=i} p_j

Hazard rate (Cutler-Ederer, at the interval midpoint):

    h_i(mid) = 2 q_i / (width_i (1 + p_i)) = d_i / (width_i (l'_i - d_i / 2))

Greenwood's formula for life tables:

    SE(S_i) = S_i * sqrt( sum_{j<=i} d_j / (l'_j (l'_j - d_j)) )
    SE(q_i) = sqrt( q_i p_i / l'_i )

Confidence interval for S_i: normal theory, S_i +/- z_{1-alpha/2} SE(S_i),
clipped to [0, 1] (the same intervals SAS PROC LIFETEST and R KMsurv::lifetab
users usually build by hand; no log or log-log transform is applied).

Median survival: the time at which S crosses 0.5, by linear interpolation of
the cumulative survival inside the interval where it does so (S is taken as 1
at time 0). None if S never reaches 0.5.

Conventions
-----------
* An interval is half-open, [lower, upper). A time exactly on a break belongs
  to the interval that starts at that break.
* The last interval also accepts times equal to the last break: an event there
  is an event of the last interval, a censoring there is a withdrawal.
* Times beyond the last break (only possible with explicit breaks or a fixed
  number of intervals) are administratively censored at the last break: they
  count as withdrawn in the last interval, whether or not they were events.
* When nobody is at risk (l'_i = 0) the interval and every later one carry
  null estimates. When d_i = l'_i the survival is 0 and its standard error is
  reported as 0 (Greenwood's term is infinite there; S = 0 is exact).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

MAX_INTERVALS = 200
_NICE = (1.0, 2.0, 2.5, 5.0, 10.0)
_ROUND = 10  # decimals kept on generated breaks, to hide k * width float noise


class LifeTableInputError(ValueError):
    """Bad breaks / width / n_intervals. The router turns this into a 422."""


# ── Breaks ───────────────────────────────────────────────────────────────────


def nice_width(raw: float) -> float:
    """Round a positive width UP to 1, 2, 2.5, 5 or 10 times a power of ten."""
    if not np.isfinite(raw) or raw <= 0:
        return 1.0
    exp = 10.0 ** math.floor(math.log10(raw))
    frac = raw / exp
    for step in _NICE:
        if frac <= step * (1.0 + 1e-9):
            return round(step * exp, _ROUND)
    return round(10.0 * exp, _ROUND)


def default_width(t_max: float, target_intervals: int = 10) -> float:
    """About ``target_intervals`` intervals over the follow-up, on a round width."""
    if not np.isfinite(t_max) or t_max <= 0:
        return 1.0
    return nice_width(t_max / float(target_intervals))


def _regular_breaks(width: float, n: int) -> List[float]:
    return [round(k * width, _ROUND) for k in range(n + 1)]


def validate_breaks(breaks: Sequence[float]) -> List[float]:
    try:
        arr = [float(b) for b in breaks]
    except (TypeError, ValueError):
        raise LifeTableInputError("breaks must be a list of numbers.")
    if len(arr) < 2:
        raise LifeTableInputError("breaks needs at least two values (0 and one upper limit).")
    if not all(np.isfinite(arr)):
        raise LifeTableInputError("breaks must be finite numbers.")
    if arr[0] != 0.0:
        raise LifeTableInputError("The first break must be 0.")
    if any(b2 <= b1 for b1, b2 in zip(arr, arr[1:])):
        raise LifeTableInputError("breaks must be strictly increasing.")
    if len(arr) - 1 > MAX_INTERVALS:
        raise LifeTableInputError(f"At most {MAX_INTERVALS} intervals are allowed; got {len(arr) - 1}.")
    return arr


def resolve_breaks(
    t_max: float,
    interval_width: Optional[float] = None,
    n_intervals: Optional[int] = None,
    breaks: Optional[Sequence[float]] = None,
) -> Tuple[List[float], bool, List[str]]:
    """Choose the interval limits.

    Returns (breaks, extended_automatically, warnings). ``extended_automatically``
    is True when the limits were built to cover the longest follow-up (default
    mode, or a width without an interval count), so no subject lies beyond them.
    """
    warns: List[str] = []
    if breaks is not None:
        if interval_width is not None or n_intervals is not None:
            raise LifeTableInputError("Give either breaks or interval_width / n_intervals, not both.")
        return validate_breaks(breaks), False, warns

    if interval_width is not None and (not np.isfinite(interval_width) or interval_width <= 0):
        raise LifeTableInputError("interval_width must be a positive number.")
    if n_intervals is not None and n_intervals < 1:
        raise LifeTableInputError("n_intervals must be at least 1.")
    if n_intervals is not None and n_intervals > MAX_INTERVALS:
        raise LifeTableInputError(f"n_intervals can be at most {MAX_INTERVALS}.")

    if n_intervals is not None:
        width = float(interval_width) if interval_width is not None else default_width(t_max, n_intervals)
        return _regular_breaks(width, int(n_intervals)), False, warns

    width = float(interval_width) if interval_width is not None else default_width(t_max)
    # Cover the longest time strictly: the last break lies beyond t_max.
    n = int(math.floor(t_max / width + 1e-9)) + 1
    if n > MAX_INTERVALS:
        raise LifeTableInputError(
            f"interval_width {width:g} would need {n} intervals to cover follow-up up to {t_max:g} "
            f"(maximum {MAX_INTERVALS}). Use a wider interval."
        )
    return _regular_breaks(width, n), True, warns


def regular_width(breaks: Sequence[float]) -> Optional[float]:
    """The common interval width when the breaks are equally spaced, else None."""
    diffs = np.diff(np.asarray(breaks, dtype=float))
    if len(diffs) and np.allclose(diffs, diffs[0], rtol=1e-9, atol=1e-12):
        return float(diffs[0])
    return None


# ── Counting ─────────────────────────────────────────────────────────────────


def count_intervals(
    time: np.ndarray, event: np.ndarray, breaks: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-interval (entering, withdrawn, events) and the number beyond the last break."""
    t = np.asarray(time, dtype=float)
    ev = np.asarray(event, dtype=float) == 1.0
    b = np.asarray(breaks, dtype=float)
    k = len(b) - 1
    last = b[-1]

    beyond = t > last
    ev = ev & ~beyond  # administratively censored at the last break
    idx = np.clip(np.searchsorted(b, t, side="right") - 1, 0, k - 1)

    events = np.bincount(idx[ev], minlength=k).astype(float)
    withdrawn = np.bincount(idx[~ev], minlength=k).astype(float)
    leaving_before = np.concatenate(([0.0], np.cumsum(events + withdrawn)))[:k]
    entering = float(len(t)) - leaving_before
    return entering, withdrawn, events, int(beyond.sum())


# ── Estimation ───────────────────────────────────────────────────────────────


def _f(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def compute_life_table(
    time: Sequence[float],
    event: Sequence[float],
    breaks: Sequence[float],
    alpha: float = 0.05,
) -> Dict:
    """Cutler-Ederer table for one sample. See the module docstring for the formulas."""
    b = [float(x) for x in breaks]
    entering, withdrawn, events, n_beyond = count_intervals(np.asarray(time), np.asarray(event), b)
    k = len(b) - 1
    z = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))

    rows: List[Dict] = []
    surv_prev = 1.0
    green = 0.0
    median: Optional[float] = None
    estimable = True

    for i in range(k):
        l_i, w_i, d_i = float(entering[i]), float(withdrawn[i]), float(events[i])
        lo, hi = b[i], b[i + 1]
        h = hi - lo
        eff = l_i - w_i / 2.0
        row: Dict = {
            "interval_start": lo,
            "interval_end": hi,
            "n_entering": int(l_i),
            "withdrawn": int(w_i),
            "events": int(d_i),
            "effective_at_risk": eff,
            "q": None, "p": None, "cumulative_survival": None, "se_cumulative": None,
            "ci_low": None, "ci_high": None, "hazard": None, "se_q": None,
        }
        if not estimable or eff <= 0:
            estimable = False
            rows.append(row)
            continue

        q = d_i / eff
        p = 1.0 - q
        surv = surv_prev * p
        row["q"], row["p"] = q, p
        row["se_q"] = math.sqrt(max(q * p, 0.0) / eff)
        row["hazard"] = d_i / (h * (eff - d_i / 2.0))

        if eff - d_i <= 0:
            surv = 0.0
            se = 0.0
        else:
            green += d_i / (eff * (eff - d_i))
            se = surv * math.sqrt(green)
        row["cumulative_survival"] = surv
        row["se_cumulative"] = se
        row["ci_low"] = max(0.0, surv - z * se)
        row["ci_high"] = min(1.0, surv + z * se)

        if median is None and surv <= 0.5 < surv_prev:
            median = lo + h * (surv_prev - 0.5) / (surv_prev - surv)
        surv_prev = surv
        rows.append(row)

    return {
        "intervals": rows,
        "median_survival": _f(median),
        "n_beyond_last_break": n_beyond,
        "n": int(len(np.asarray(time))),
        "events": int(np.sum(np.asarray(event, dtype=float) == 1.0)),
    }


# ── Text, R code, export ─────────────────────────────────────────────────────


def _num(x: Optional[float], digits: int = 3) -> str:
    return "NA" if x is None else f"{x:.{digits}f}"


def _g(x: float) -> str:
    return f"{x:g}"


def result_text(
    groups: List[Dict], group_col: Optional[str], breaks: Sequence[float], alpha: float
) -> str:
    conf = f"{100.0 * (1.0 - alpha):g}%"
    parts = [
        f"An actuarial (Cutler-Ederer) life table was constructed over {len(breaks) - 1} follow-up "
        f"intervals from {_g(breaks[0])} to {_g(breaks[-1])}."
    ]
    for g in groups:
        label = f"In {group_col} = {g['group']}, " if group_col else ""
        usable = [r for r in g["intervals"] if r["cumulative_survival"] is not None]
        sentence = f"{label}{g['n']} subjects contributed {g['events']} events ({g['censored']} censored)"
        if usable:
            last = usable[-1]
            sentence += (
                f"; cumulative survival at {_g(last['interval_end'])} was {_num(last['cumulative_survival'])} "
                f"({conf} CI {_num(last['ci_low'])}-{_num(last['ci_high'])})"
            )
        if g["median_survival"] is not None:
            sentence += f"; median survival {_g(round(g['median_survival'], 3))}"
        else:
            sentence += "; median survival was not reached"
        parts.append(sentence + ".")
    return " ".join(parts)


def methods_text(alpha: float, grouped: bool) -> str:
    conf = f"{100.0 * (1.0 - alpha):g}%"
    return (
        "Survival was summarised with an actuarial (Cutler-Ederer) life table. Follow-up was divided into "
        "intervals; in each interval the effective number at risk was the number entering minus half of those "
        "withdrawn (censored), the conditional probability of the event was the number of events divided by "
        "the effective number at risk, and cumulative survival was the product of the conditional survival "
        "probabilities. The standard error of cumulative survival followed Greenwood's formula for life tables, "
        f"and {conf} confidence intervals were normal-theory intervals clipped to [0, 1]. The hazard rate was "
        "estimated at the interval midpoint as 2q/(width(1+p)). Median survival was obtained by linear "
        "interpolation within the interval where cumulative survival crossed 0.5."
        + (" A separate table was computed for each group." if grouped else "")
    )


def r_code(
    time_col: str, event_col: str, group_col: Optional[str], breaks: Sequence[float], alpha: float
) -> str:
    """Plain base-R reproduction of the table (no KMsurv dependency)."""
    brk = ", ".join(_g(x) for x in breaks)
    run = (
        f"res <- do.call(rbind, lapply(split(dat, dat[[\"{group_col}\"]]), function(s) {{\n"
        f"  cbind(group = s[[\"{group_col}\"]][1], life_table(s[[\"{time_col}\"]], s[[\"{event_col}\"]], breaks, alpha))\n"
        "}))\n"
        if group_col
        else f"res <- life_table(dat[[\"{time_col}\"]], dat[[\"{event_col}\"]], breaks, alpha)\n"
    )
    return (
        "# Cutler-Ederer (actuarial) life table, base R\n"
        f"breaks <- c({brk})\n"
        f"alpha <- {_g(alpha)}\n"
        "life_table <- function(time, event, breaks, alpha = 0.05) {\n"
        "  k <- length(breaks) - 1\n"
        "  ev <- ifelse(time > breaks[k + 1], 0, event)   # beyond the last break: withdrawn\n"
        "  idx <- pmin(findInterval(time, breaks), k)     # intervals are [lower, upper)\n"
        "  d <- tabulate(idx[ev == 1], nbins = k)\n"
        "  w <- tabulate(idx[ev == 0], nbins = k)\n"
        "  l <- length(time) - c(0, cumsum(d + w))[seq_len(k)]\n"
        "  h <- diff(breaks)\n"
        "  eff <- l - w / 2\n"
        "  q <- ifelse(eff > 0, d / eff, NA)\n"
        "  p <- 1 - q\n"
        "  S <- cumprod(p)\n"
        "  green <- cumsum(ifelse(eff > 0, d / (eff * (eff - d)), NA))\n"
        "  se <- ifelse(is.na(S), NA, ifelse(S == 0, 0, S * sqrt(green)))\n"
        "  z <- qnorm(1 - alpha / 2)\n"
        "  data.frame(\n"
        "    interval_start = breaks[-(k + 1)], interval_end = breaks[-1],\n"
        "    n_entering = l, withdrawn = w, events = d, effective_at_risk = eff,\n"
        "    q = q, p = p, cumulative_survival = S, se_cumulative = se,\n"
        "    ci_low = pmax(0, S - z * se), ci_high = pmin(1, S + z * se),\n"
        "    hazard = ifelse(eff > 0, d / (h * (eff - d / 2)), NA),\n"
        "    se_q = sqrt(q * p / eff)\n"
        "  )\n"
        "}\n"
        + run
        + "print(res)\n"
    )


EXPORT_HEADER = [
    "Interval start", "Interval end", "Entering", "Withdrawn", "Events", "Effective at risk",
    "q (event)", "p (survival)", "Cumulative survival", "SE", "CI low", "CI high", "Hazard", "SE(q)",
]


def _rounded(x: Optional[float], digits: int = 4):
    return None if x is None else round(float(x), digits)


def export_rows(groups: List[Dict], group_col: Optional[str]) -> List[List]:
    header = ([group_col] if group_col else []) + EXPORT_HEADER
    out: List[List] = [header]
    for g in groups:
        for r in g["intervals"]:
            row = [
                r["interval_start"], r["interval_end"], r["n_entering"], r["withdrawn"], r["events"],
                _rounded(r["effective_at_risk"], 2), _rounded(r["q"]), _rounded(r["p"]),
                _rounded(r["cumulative_survival"]), _rounded(r["se_cumulative"]),
                _rounded(r["ci_low"]), _rounded(r["ci_high"]), _rounded(r["hazard"], 5),
                _rounded(r["se_q"]),
            ]
            out.append(([g["group"]] if group_col else []) + row)
    return out
