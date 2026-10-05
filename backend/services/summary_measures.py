"""Summary-measures analysis of serial (longitudinal) data.

Matthews, Altman, Campbell & Royston (1990) BMJ 300:230-235: collapse each
subject's repeated measurements into one number (area under the curve, peak,
time to peak, slope, ...), then compare groups on that number with an
ordinary two-sample or one-way method. Each subject contributes exactly one
value per measure, so the usual independence assumption holds.

All functions are pure. `analyse` takes a long-format DataFrame and raises
ValueError on invalid input (the router turns that into a 422).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp

MEASURE_LABELS: Dict[str, str] = {
    "auc": "AUC (trapezoid)",
    "auc_per_time": "Time-averaged AUC",
    "auc_above_baseline": "Incremental AUC (above baseline)",
    "peak": "Peak (Ymax)",
    "time_to_peak": "Time to peak (Tmax)",
    "slope": "Slope (OLS per subject)",
    "mean": "Mean of observations",
    "first": "First observation",
    "last": "Last observation",
    "change_last_first": "Change (last minus first)",
}
VALID_MEASURES = tuple(MEASURE_LABELS)
DEFAULT_MEASURES = ["auc_per_time", "peak", "time_to_peak", "slope"]
_AUC_FAMILY = {"auc", "auc_per_time", "auc_above_baseline"}


# ───────────────────────────── per-subject measures ─────────────────────────────

def trapezoid_auc(t: np.ndarray, y: np.ndarray) -> float:
    """Trapezoid-rule area under (t, y); t must be sorted ascending."""
    if len(t) < 2:
        return float("nan")
    return float(np.sum(np.diff(t) * (y[:-1] + y[1:]) / 2.0))


def collapse_duplicate_times(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Sort by time and average the values observed at the same time."""
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    uniq, inverse = np.unique(t, return_inverse=True)
    if len(uniq) == len(t):
        return t.astype(float), y.astype(float), False
    sums = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse)
    return uniq.astype(float), sums / counts, True


def _ols_slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2:
        return float("nan")
    tc = t - t.mean()
    denom = float(np.sum(tc * tc))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(tc * (y - y.mean())) / denom)


def subject_measures(
    t: np.ndarray,
    y: np.ndarray,
    measures: Sequence[str],
    min_points: int = 2,
    missing: str = "omit",
    grid: Optional[np.ndarray] = None,
) -> dict:
    """Summary measures for one subject.

    `t`, `y` are the non-missing observations (already sorted, unique times).
    AUC-family measures and the slope need at least `min_points` observations
    (and never fewer than 2); the others need one (change needs two). With
    missing='interpolate', interior gaps on the shared visit `grid` are filled
    by linear interpolation before the slope and the mean are computed; the
    trapezoid AUC is unchanged by linear interpolation, and peak, Tmax, first
    and last always use the observed points only.
    """
    n = len(t)
    need = max(int(min_points), 2)
    nan = float("nan")
    out: Dict[str, float] = {m: nan for m in measures}
    n_interp = 0
    if n == 0:
        return {"n_points": 0, "n_interpolated": 0, "values": out}

    ts, ys = t, y
    if missing == "interpolate" and grid is not None and n >= 2:
        extra = grid[(grid > t[0]) & (grid < t[-1]) & ~np.isin(grid, t)]
        if len(extra):
            n_interp = int(len(extra))
            ts = np.concatenate([t, extra])
            order = np.argsort(ts, kind="stable")
            ts = ts[order]
            ys = np.interp(ts, t, y)

    span = float(t[-1] - t[0])
    ok_auc = n >= need
    auc = trapezoid_auc(t, y) if ok_auc else nan
    for m in measures:
        if m == "auc":
            out[m] = auc
        elif m == "auc_per_time":
            out[m] = auc / span if ok_auc and span > 0 else nan
        elif m == "auc_above_baseline":
            out[m] = auc - float(y[0]) * span if ok_auc else nan
        elif m == "peak":
            out[m] = float(np.max(y))
        elif m == "time_to_peak":
            out[m] = float(t[int(np.argmax(y))])
        elif m == "slope":
            out[m] = _ols_slope(ts, ys) if n >= need else nan
        elif m == "mean":
            out[m] = float(np.mean(ys))
        elif m == "first":
            out[m] = float(y[0])
        elif m == "last":
            out[m] = float(y[-1])
        elif m == "change_last_first":
            out[m] = float(y[-1] - y[0]) if n >= 2 else nan
    return {"n_points": n, "n_interpolated": n_interp, "values": out}


# ───────────────────────────── descriptives and tests ─────────────────────────────

def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None,
                "q1": None, "q3": None, "iqr": None, "min": None, "max": None}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": n,
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if n > 1 else None,
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _magnitude_d(d: Optional[float]) -> Optional[str]:
    if d is None:
        return None
    a = abs(d)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def _magnitude_eta(e: Optional[float]) -> Optional[str]:
    if e is None:
        return None
    return "negligible" if e < 0.01 else "small" if e < 0.06 else "medium" if e < 0.14 else "large"


def _finite_or_none(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def normality_of_residuals(groups: List[np.ndarray]) -> dict:
    """Shapiro-Wilk on the pooled within-group residuals."""
    resid = np.concatenate([g - g.mean() for g in groups]) if groups else np.array([])
    n = len(resid)
    base = {"test": "Shapiro-Wilk (pooled residuals)", "n": n, "p": None, "normal": None}
    if n < 3 or n > 5000 or float(np.ptp(resid)) == 0.0:
        base["note"] = "Normality could not be assessed (n < 3 or no variation)."
        return base
    p = float(sp.shapiro(resid).pvalue)
    base["p"] = p
    base["normal"] = bool(p >= 0.05)
    return base


def compare_two(a: np.ndarray, b: np.ndarray, alpha: float) -> dict:
    n1, n2 = len(a), len(b)
    m1, m2 = float(a.mean()), float(b.mean())
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    diff = m1 - m2
    se2 = v1 / n1 + v2 / n2
    res = {
        "mean_difference": diff, "t": None, "df": None, "p": None,
        "ci_low": None, "ci_high": None,
        "nonparametric_test": "Mann-Whitney U", "u": None, "p_nonparametric": None,
        "effect_size": {"name": "Cohen's d", "value": None, "magnitude": None},
    }
    if se2 > 0:
        t_res = sp.ttest_ind(a, b, equal_var=False)
        df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        tcrit = float(sp.t.ppf(1 - alpha / 2, df))
        res.update(
            t=_finite_or_none(t_res.statistic), df=float(df), p=_finite_or_none(t_res.pvalue),
            ci_low=diff - tcrit * math.sqrt(se2), ci_high=diff + tcrit * math.sqrt(se2),
        )
    sp_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if sp_pooled > 0:
        d = diff / sp_pooled
        res["effect_size"] = {"name": "Cohen's d", "value": d, "magnitude": _magnitude_d(d)}
    try:
        # Exact p when there are no ties and both groups are small, as R's
        # wilcox.test does; scipy's default would switch to the normal
        # approximation above 8 per group.
        exact = len(np.unique(np.concatenate([a, b]))) == n1 + n2 and n1 < 50 and n2 < 50
        mw = sp.mannwhitneyu(a, b, alternative="two-sided",
                             method="exact" if exact else "asymptotic")
        res["u"] = _finite_or_none(mw.statistic)
        res["p_nonparametric"] = _finite_or_none(mw.pvalue)
    except ValueError:
        pass
    return res


def compare_many(groups: List[np.ndarray]) -> dict:
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    res = {
        "f": None, "df_between": k - 1, "df_within": n_total - k, "p": None,
        "nonparametric_test": "Kruskal-Wallis", "h": None, "p_nonparametric": None,
        "effect_size": {"name": "eta squared", "value": None, "magnitude": None},
    }
    allv = np.concatenate(groups)
    grand = float(allv.mean())
    ss_total = float(np.sum((allv - grand) ** 2))
    ss_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    ss_within = ss_total - ss_between
    if ss_total > 0:
        eta = ss_between / ss_total
        res["effect_size"] = {"name": "eta squared", "value": eta, "magnitude": _magnitude_eta(eta)}
    if ss_within > 1e-12 * max(ss_total, 1.0) and n_total > k:
        f_res = sp.f_oneway(*groups)
        res["f"] = _finite_or_none(f_res.statistic)
        res["p"] = _finite_or_none(f_res.pvalue)
    try:
        kw = sp.kruskal(*groups)
        res["h"] = _finite_or_none(kw.statistic)
        res["p_nonparametric"] = _finite_or_none(kw.pvalue)
    except ValueError:
        pass
    return res


# ───────────────────────────── text builders ─────────────────────────────

def _p_str(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _num(v: Optional[float], nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def build_result_text(measures, comparisons, descriptives, group_labels, alpha, has_groups) -> str:
    parts: List[str] = []
    if not has_groups:
        for m in measures:
            d = next((r for r in descriptives if r["measure"] == m), None)
            if d:
                parts.append(f"{MEASURE_LABELS[m]}: M = {_num(d['mean'])}, SD = {_num(d['sd'])}, "
                             f"median = {_num(d['median'])} (IQR {_num(d['q1'])} to {_num(d['q3'])}), n = {d['n']}.")
        return ("Each subject's repeated measurements were summarised into one value per measure. "
                + " ".join(parts))
    for c in comparisons:
        label = MEASURE_LABELS[c["measure"]]
        if c.get("skipped"):
            parts.append(f"{label}: comparison not possible ({c['skipped']}).")
            continue
        if c["test"] == "Welch t-test":
            sig = "a significant" if (c["p"] is not None and c["p"] < alpha) else "no significant"
            ci = f"{100 * (1 - alpha):.0f}% CI {_num(c['ci_low'], 3)} to {_num(c['ci_high'], 3)}"
            parts.append(
                f"{label}: {sig} difference between {c['groups'][0]} and {c['groups'][1]} "
                f"(mean difference = {_num(c['mean_difference'], 3)}, {ci}; Welch t({_num(c['df'], 1)}) = "
                f"{_num(c['t'], 3)}, p = {_p_str(c['p'])}; Cohen's d = {_num(c['effect_size']['value'], 2)}). "
                f"Sensitivity analysis, Mann-Whitney p = {_p_str(c['p_nonparametric'])}.")
        else:
            sig = "a significant" if (c["p"] is not None and c["p"] < alpha) else "no significant"
            parts.append(
                f"{label}: {sig} difference across groups (one-way ANOVA F({c['df_between']}, "
                f"{c['df_within']}) = {_num(c['f'], 3)}, p = {_p_str(c['p'])}; eta squared = "
                f"{_num(c['effect_size']['value'], 3)}). Sensitivity analysis, Kruskal-Wallis p = "
                f"{_p_str(c['p_nonparametric'])}.")
    head = ("Each subject's repeated measurements were summarised into one value per measure and groups "
            "were compared on these subject-level values. The primary test is the Welch t-test "
            "(two groups) or one-way ANOVA (more than two groups); the rank-based p value "
            "(Mann-Whitney or Kruskal-Wallis) is given as a sensitivity analysis. ")
    return head + " ".join(parts)


def build_methods_text(measures, has_groups, n_groups, missing, min_points, alpha) -> str:
    desc = {
        "auc": "the area under the curve",
        "auc_per_time": "the time-averaged area under the curve (AUC divided by the follow-up span)",
        "auc_above_baseline": "the incremental area under the curve above the baseline value",
        "peak": "the peak value",
        "time_to_peak": "the time to peak",
        "slope": "the least-squares slope of value on time",
        "mean": "the mean of the observations",
        "first": "the first observation",
        "last": "the last observation",
        "change_last_first": "the change from first to last observation",
    }
    names = [desc[m] for m in measures]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    gap = ("Interior gaps were filled by linear interpolation before the slope and mean were computed."
           if missing == "interpolate" else
           "Missing observations were omitted and the trapezoid was computed over the observed points only.")
    if not has_groups:
        comp = "Summary measures are reported descriptively."
    elif n_groups == 2:
        comp = (f"The two groups were compared on each summary measure with Welch's t-test (mean difference "
                f"with {100 * (1 - alpha):.0f}% confidence interval; Cohen's d) and, as a sensitivity "
                "analysis, the Mann-Whitney U test.")
    else:
        comp = (f"The {n_groups} groups were compared on each summary measure with one-way analysis of variance "
                "(eta squared as effect size) and, as a sensitivity analysis, the Kruskal-Wallis test.")
    return (
        "Serial measurements were analysed with the summary-measures approach (Matthews et al., 1990, "
        "BMJ 300:230-235): each subject's repeated measurements were reduced to a single value, namely "
        f"{listed}. The area under the curve was obtained with the trapezoid rule over the observed visit "
        f"times. Subjects with fewer than {max(min_points, 2)} observations were excluded from the "
        f"area and slope measures. {gap} {comp} Statistical significance was set at alpha = {alpha}."
    )


def _r_name(s: str) -> str:
    return "`" + s.replace("`", "\\`") + "`"


def build_r_code(subject_col, time_col, value_col, group_col, measures, min_points, missing,
                 group_labels, alpha) -> str:
    need = max(min_points, 2)
    auc_expr = "sum(diff(t) * (head(y, -1) + tail(y, -1)) / 2)"
    lines = [
        "library(dplyr)",
        "",
        "# d: long-format data frame, one row per subject and visit",
        "# Duplicate times are averaged; trapezoid AUC over the observed points.",
    ]
    if missing == "interpolate":
        lines += ["# The Python side also interpolated interior gaps for the slope and the mean;",
                  "# fill them first (e.g. with approx()) to reproduce those two measures exactly."]
    sel = f"id = {_r_name(subject_col)}, t = {_r_name(time_col)}, y = {_r_name(value_col)}"
    if group_col:
        sel += f", grp = {_r_name(group_col)}"
    keys = "id, grp" if group_col else "id"
    spec = {
        "auc": f"auc = if (n() >= {need}) {auc_expr} else NA_real_",
        "auc_per_time": f"auc_per_time = if (n() >= {need}) {auc_expr} / (max(t) - min(t)) else NA_real_",
        "auc_above_baseline": (f"auc_above_baseline = if (n() >= {need}) {auc_expr} "
                               "- first(y) * (max(t) - min(t)) else NA_real_"),
        "peak": "peak = max(y)",
        "time_to_peak": "time_to_peak = t[which.max(y)]",
        "slope": f"slope = if (n() >= {need}) unname(coef(lm(y ~ t))[2]) else NA_real_",
        "mean": "mean = mean(y)",
        "first": "first = first(y)",
        "last": "last = last(y)",
        "change_last_first": "change_last_first = if (n() >= 2) last(y) - first(y) else NA_real_",
    }
    args = ["n_points = n()"] + [spec[m] for m in measures] + [".groups = \"drop\""]
    lines += [
        "subj <- d %>%",
        f"  transmute({sel}) %>%",
        "  filter(!is.na(y), !is.na(t)) %>%",
        f"  group_by({keys}, t) %>% summarise(y = mean(y), .groups = \"drop\") %>%",
        "  arrange(id, t) %>%",
        f"  group_by({keys}) %>%",
        "  summarise(",
        *["    " + a_ + ("," if i < len(args) - 1 else "") for i, a_ in enumerate(args)],
        "  )",
        "",
    ]
    if group_col:
        lines.append("subj$grp <- factor(subj$grp)")
        two = len(group_labels) == 2
        for m in measures:
            if two:
                lines.append(f"t.test({m} ~ grp, data = subj, conf.level = {1 - alpha:g})  # Welch (default)")
                lines.append(f"wilcox.test({m} ~ grp, data = subj)  # sensitivity")
            else:
                lines.append(f"summary(aov({m} ~ grp, data = subj))")
                lines.append(f"kruskal.test({m} ~ grp, data = subj)  # sensitivity")
    else:
        lines.append("summary(subj)")
    return "\n".join(lines)


# ───────────────────────────── orchestration ─────────────────────────────

def _label(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _native(v):
    return v.item() if hasattr(v, "item") else v


def _sorted_labels(labels: List[str], raw: Dict[str, object]) -> List[str]:
    try:
        return sorted(labels, key=lambda s: float(raw[s]))
    except (TypeError, ValueError):
        return sorted(labels)


def analyse(
    df: pd.DataFrame,
    subject_col: str,
    time_col: str,
    value_col: str,
    group_col: Optional[str] = None,
    measures: Optional[Sequence[str]] = None,
    min_points: int = 2,
    missing: str = "omit",
    alpha: float = 0.05,
) -> dict:
    measures = list(dict.fromkeys(measures if measures else DEFAULT_MEASURES))
    bad = [m for m in measures if m not in MEASURE_LABELS]
    if bad:
        raise ValueError(f"Unknown measure(s): {', '.join(bad)}. Valid: {', '.join(VALID_MEASURES)}.")
    if missing not in ("omit", "interpolate"):
        raise ValueError("missing must be 'omit' or 'interpolate'.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if min_points < 1:
        raise ValueError("min_points must be at least 1.")

    cols = [subject_col, time_col, value_col] + ([group_col] if group_col else [])
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    if len(set(cols)) != len(cols):
        raise ValueError("Subject, time, value and group columns must all be different.")
    for c, what in ((time_col, "Time"), (value_col, "Value")):
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            raise ValueError(f"{what} column '{c}' must be numeric.")

    warnings: List[str] = []
    data = df[cols].copy()
    data = data[data[subject_col].notna() & data[time_col].notna()]
    n_missing_val = int(data[value_col].isna().sum())
    data = data[data[value_col].notna()]
    data = data[np.isfinite(data[time_col].astype(float)) & np.isfinite(data[value_col].astype(float))]
    if n_missing_val:
        warnings.append(f"{n_missing_val} observation(s) with a missing value were left out "
                        f"({'interior gaps interpolated' if missing == 'interpolate' else 'trapezoid over observed points only'}).")

    grid = np.sort(data[time_col].astype(float).unique()) if len(data) else np.array([])

    subjects: List[dict] = []
    raw_groups: Dict[str, object] = {}
    short: List[str] = []
    dup_subjects: List[str] = []
    no_group: List[str] = []
    for sid, sub in data.groupby(subject_col, sort=False):
        sname = _label(_native(sid))
        grp_label = None
        if group_col:
            gvals = sub[group_col].dropna().unique()
            if len(gvals) > 1:
                raise ValueError(f"Subject '{sname}' belongs to more than one group in '{group_col}'; "
                                 "the group must be a between-subject factor.")
            if len(gvals):
                grp_label = _label(_native(gvals[0]))
                raw_groups.setdefault(grp_label, _native(gvals[0]))
            else:
                no_group.append(sname)
        t, y, dup = collapse_duplicate_times(
            sub[time_col].to_numpy(dtype=float), sub[value_col].to_numpy(dtype=float))
        if dup:
            dup_subjects.append(sname)
        res = subject_measures(t, y, measures, min_points, missing, grid)
        if res["n_points"] < max(min_points, 2) and (set(measures) & (_AUC_FAMILY | {"slope"})):
            short.append(f"{sname} (n = {res['n_points']})")
        row = {"subject": _native(sid), "group": grp_label, "n_points": res["n_points"]}
        if missing == "interpolate":
            row["n_interpolated"] = res["n_interpolated"]
        for m in measures:
            row[m] = _finite_or_none(res["values"][m])
        subjects.append(row)

    valid_subjects = [s for s in subjects if s["n_points"] > 0]
    if len(valid_subjects) < 2:
        raise ValueError("At least 2 subjects with observed values are required.")
    if dup_subjects:
        warnings.append(f"Duplicate visit times were averaged for {len(dup_subjects)} subject(s): "
                        + ", ".join(dup_subjects[:10]) + (", ..." if len(dup_subjects) > 10 else "") + ".")
    if short:
        warnings.append(f"Fewer than {max(min_points, 2)} observed points (AUC and slope set to missing): "
                        + ", ".join(short[:10]) + (", ..." if len(short) > 10 else "") + ".")
    if no_group:
        warnings.append(f"{len(no_group)} subject(s) have no group value and are excluded from the comparison.")

    group_labels: List[str] = []
    if group_col:
        counts: Dict[str, int] = {}
        for s in valid_subjects:
            if s["group"] is not None:
                counts[s["group"]] = counts.get(s["group"], 0) + 1
        group_labels = _sorted_labels(list(counts), raw_groups)
        if len(group_labels) < 2:
            raise ValueError("The group column must have at least 2 groups.")
        small = [g for g in group_labels if counts[g] < 2]
        if small:
            raise ValueError("Each group needs at least 2 subjects; too few in: " + ", ".join(small) + ".")

    def values(measure: str, group: Optional[str]) -> np.ndarray:
        v = [s[measure] for s in valid_subjects
             if s[measure] is not None and (group is None or s["group"] == group)]
        return np.asarray(v, dtype=float)

    descriptives: List[dict] = []
    for m in measures:
        if group_col:
            for g in group_labels:
                descriptives.append({"measure": m, "group": g, **describe(values(m, g))})
        else:
            descriptives.append({"measure": m, "group": "All", **describe(values(m, None))})

    comparisons: List[dict] = []
    if group_col:
        for m in measures:
            usable = [g for g in group_labels if len(values(m, g)) >= 2]
            entry: dict = {"measure": m, "label": MEASURE_LABELS[m], "groups": usable,
                           "n": {g: int(len(values(m, g))) for g in group_labels}}
            if len(usable) < 2:
                entry.update(test=None, skipped="fewer than 2 groups with at least 2 valid subject values")
                comparisons.append(entry)
                continue
            arrays = [values(m, g) for g in usable]
            if len(usable) == 2:
                entry.update(test="Welch t-test", **compare_two(arrays[0], arrays[1], alpha))
            else:
                entry.update(test="One-way ANOVA", **compare_many(arrays))
            entry["normality"] = normality_of_residuals(arrays)
            notes: List[str] = []
            if len(usable) < len(group_labels):
                notes.append("Groups with fewer than 2 valid values were left out: "
                             + ", ".join(g for g in group_labels if g not in usable) + ".")
            if entry["normality"]["normal"] is False:
                notes.append("Residuals deviate from normality (Shapiro-Wilk p < 0.05); "
                             "prefer the rank-based p value.")
            elif entry["normality"]["normal"] is None:
                notes.append(entry["normality"].get("note", ""))
            if m == "time_to_peak":
                notes.append("Time to peak takes few distinct values (visit times); "
                             "the rank-based p value is usually the safer choice.")
            entry["notes"] = [n for n in notes if n]
            comparisons.append(entry)

    result_text = build_result_text(measures, comparisons, descriptives, group_labels, alpha, bool(group_col))
    methods_text = build_methods_text(measures, bool(group_col), len(group_labels), missing, min_points, alpha)
    r_code = build_r_code(subject_col, time_col, value_col, group_col, measures, min_points, missing,
                          group_labels, alpha)

    export_rows: List[list] = [["Measure", "Group", "n", "Mean", "SD", "Median", "Q1", "Q3"]]
    export_rows += [[MEASURE_LABELS[d["measure"]], d["group"], d["n"], d["mean"], d["sd"],
                     d["median"], d["q1"], d["q3"]] for d in descriptives]
    if comparisons:
        export_rows.append(["Measure", "Primary test", "Statistic", "p", "Nonparametric test",
                            "p (nonparametric)", "Effect size", "Mean difference", "CI low", "CI high"])
        for c in comparisons:
            if c.get("skipped"):
                export_rows.append([c["label"], "not computed", None, None, None, None, None, None, None, None])
                continue
            stat = c["t"] if c["test"] == "Welch t-test" else c["f"]
            export_rows.append([c["label"], c["test"], stat, c["p"], c["nonparametric_test"],
                                c["p_nonparametric"], c["effect_size"]["value"],
                                c.get("mean_difference"), c.get("ci_low"), c.get("ci_high")])

    times = grid
    return {
        "test": "Summary measures of serial data",
        "reference": "Matthews JNS, Altman DG, Campbell MJ, Royston P (1990) BMJ 300:230-235",
        "measures": measures,
        "measure_labels": {m: MEASURE_LABELS[m] for m in measures},
        "subjects": subjects,
        "descriptives": descriptives,
        "comparisons": comparisons,
        "summary": {
            "n_subjects": len(valid_subjects),
            "n_observations": int(len(data)),
            "groups": group_labels,
            "time_min": float(times.min()) if len(times) else None,
            "time_max": float(times.max()) if len(times) else None,
            "n_visit_times": int(len(times)),
            "min_points": max(min_points, 2),
            "missing": missing,
            "alpha": alpha,
        },
        "warnings": warnings,
        "interpretation": (
            "Primary comparison uses the subject-level Welch t-test (two groups) or one-way ANOVA; "
            "the Mann-Whitney / Kruskal-Wallis p value is a sensitivity analysis."
            if group_col else "Descriptive subject-level summary measures."),
        "result_text": result_text,
        "methods_text": methods_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }
