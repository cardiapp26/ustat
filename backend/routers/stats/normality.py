"""Normality assessment — the whole cohort and, optionally, within each group.

Every source that tells clinicians how to check normality says the same thing:
never decide on a single p-value. Ghasemi & Zahediasl (Int J Endocrinol Metab
2012;10:486-9) and Kim (Restor Dent Endod 2013;38:52-4) both recommend
combining (a) a formal test, (b) the shape statistics — skewness and excess
kurtosis with their standard errors — and (c) a look at the Q-Q plot and
histogram. The reason is that the formal test alone is misleading at both ends
of the sample-size range: under ~20 observations it has almost no power, so a
non-significant p is not evidence of normality, and above a few hundred it
turns clinically irrelevant departures into p < 0.001.

So this endpoint returns all three, and a verdict that reads them together
rather than thresholding one of them.

The shape statistics follow the SPSS / e1071-type-2 definitions (G1, G2) with
the SPSS standard errors, because those are the numbers Kim's z-score cutoffs
are defined on and the ones a reader recognises from their own output.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as scipy_stats

from services import store
from services.dirty_value_guard import coerce_numeric
from services.number_format import format_p
from services.stat_utils import sanitize_nonfinite, sorted_groups

router = APIRouter()

# Shapiro-Wilk's Royston approximation is defined up to n = 5000; scipy warns
# beyond it. Above that Anderson-Darling takes over as the primary test.
SHAPIRO_MAX_N = 5000
# Q-Q and histogram payloads are drawn, not read, so a cohort of 100 000 does
# not need 100 000 points crossing the wire to look the same on screen.
MAX_PLOT_POINTS = 2000


class NormalityRequest(BaseModel):
    session_id: str
    variables: List[str]
    group_column: Optional[str] = None
    alpha: float = 0.05


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


# ── shape statistics ───────────────────────────────────────────────────────────


def _shape(x: np.ndarray) -> dict:
    """Skewness and excess kurtosis as SPSS reports them, with z-scores.

    ``bias=False`` gives G1/G2 — the sample estimators SPSS, Excel and
    ``e1071::skewness(type = 2)`` print, and the ones the published standard
    errors below are derived for. scipy's default (bias=True) is the g1/g2
    population form and would make the z-scores mildly wrong at small n, which
    is exactly where they get used.
    """
    n = int(len(x))
    out: dict = {
        "n": n,
        "skewness": None, "skew_se": None, "skew_z": None,
        "kurtosis": None, "kurt_se": None, "kurt_z": None,
    }
    if n < 3 or float(np.std(x)) == 0.0:
        return out

    skew = _finite(scipy_stats.skew(x, bias=False))
    se_skew = float(np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))))
    out["skewness"] = skew
    out["skew_se"] = se_skew
    out["skew_z"] = _finite(skew / se_skew) if skew is not None and se_skew > 0 else None

    if n >= 4:
        kurt = _finite(scipy_stats.kurtosis(x, fisher=True, bias=False))
        out["kurtosis"] = kurt
        if n > 3:
            se_kurt = float(
                2 * se_skew * np.sqrt((n * n - 1) / ((n - 3) * (n + 5)))
            )
            out["kurt_se"] = se_kurt
            out["kurt_z"] = (
                _finite(kurt / se_kurt) if kurt is not None and se_kurt > 0 else None
            )
    return out


def _shape_flag(shape: dict, n: int) -> Optional[bool]:
    """Kim (2013): which shape rule applies depends on the sample size.

    Under 50 the z-scores are compared against 1.96, from 50 to 300 against
    3.29 (the z-test itself over-detects once n grows), and beyond 300 the
    z-scores are abandoned for the absolute values, because at that size any
    non-zero skew is "significant" while a skew of 0.3 is invisible in a
    histogram.
    """
    sk, ku = shape.get("skewness"), shape.get("kurtosis")
    if sk is None:
        return None
    if n > 300:
        return bool(abs(sk) > 2.0 or (ku is not None and abs(ku) > 7.0))
    crit = 1.96 if n < 50 else 3.29
    zs, zk = shape.get("skew_z"), shape.get("kurt_z")
    if zs is None:
        return None
    return bool(abs(zs) > crit or (zk is not None and abs(zk) > crit))


# ── formal tests ───────────────────────────────────────────────────────────────


def _anderson_darling(x: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """A² and its p-value, matching R's ``nortest::ad.test``.

    scipy returns the statistic with a table of critical values but no p, and
    a table cannot be pasted into a paper. These are the D'Agostino-Stephens
    fits that nortest uses, so the number lines up with what a reviewer gets
    in R rather than being close to it.
    """
    n = int(len(x))
    if n < 8:
        return None, None
    xs = np.sort(x)
    sd = float(np.std(xs, ddof=1))
    if sd == 0:
        return None, None
    z = (xs - float(np.mean(xs))) / sd
    logp1 = scipy_stats.norm.logcdf(z)
    logp2 = scipy_stats.norm.logcdf(-z)
    i = np.arange(1, n + 1)
    h = (2 * i - 1) * (logp1 + logp2[::-1])
    a2 = float(-n - np.mean(h))
    if not np.isfinite(a2):
        return None, None
    adj = (1 + 0.75 / n + 2.25 / n**2) * a2
    if adj < 0.2:
        p = 1 - np.exp(-13.436 + 101.14 * adj - 223.73 * adj**2)
    elif adj < 0.34:
        p = 1 - np.exp(-8.318 + 42.796 * adj - 59.938 * adj**2)
    elif adj < 0.6:
        p = np.exp(0.9177 - 4.279 * adj - 1.38 * adj**2)
    elif adj < 10:
        p = np.exp(1.2937 - 5.709 * adj + 0.0186 * adj**2)
    else:
        p = 3.7e-24
    return a2, float(min(max(p, 0.0), 1.0))


def _lilliefors(x: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Lilliefors D and its p-value, matching R's ``nortest::lillie.test``.

    statsmodels has this test, but its p-value comes from a simulated table
    and differs from R's by up to ~0.02 on samples this size — enough for a
    reviewer re-running the analysis to get a different number in the third
    decimal. The statistic is identical either way; only the p-value
    approximation differs, so what is used here is the one R uses (Dallal &
    Wilkinson's fit, with Stephens' upper-tail correction above p = 0.1).
    """
    n = int(len(x))
    if n < 5:
        return None, None
    xs = np.sort(x)
    sd = float(np.std(xs, ddof=1))
    if sd == 0:
        return None, None
    p = scipy_stats.norm.cdf((xs - float(np.mean(xs))) / sd)
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - p), np.max(p - (i - 1) / n)))

    kd, nd = (d, n) if n <= 100 else (d * ((n / 100.0) ** 0.49), 100)
    pval = float(np.exp(
        -7.01256 * kd**2 * (nd + 2.78019)
        + 2.99587 * kd * np.sqrt(nd + 2.78019)
        - 0.122119 + 0.974598 / np.sqrt(nd) + 1.67997 / nd
    ))
    if pval > 0.1:
        kk = (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n)) * d
        if kk <= 0.302:
            pval = 1.0
        elif kk <= 0.5:
            pval = (2.76773 - 19.828315 * kk + 80.709644 * kk**2
                    - 138.55152 * kk**3 + 81.218052 * kk**4)
        elif kk <= 0.9:
            pval = (-4.901232 + 40.662806 * kk - 97.490286 * kk**2
                    + 94.029866 * kk**3 - 32.355711 * kk**4)
        elif kk <= 1.31:
            pval = (6.198765 - 19.558097 * kk + 23.186922 * kk**2
                    - 12.234627 * kk**3 + 2.423045 * kk**4)
        else:
            pval = 0.0
    return d, float(min(max(pval, 0.0), 1.0))


def _run_tests(x: np.ndarray) -> list[dict]:
    """Every applicable test, each carrying why it did or did not run."""
    n = int(len(x))
    tests: list[dict] = []

    # A column with no variance has no distribution to test. scipy and
    # statsmodels will still return a number here — Shapiro warns and hands
    # back a W of 1, Lilliefors divides by a zero SD — and a "p = 1.000, looks
    # normal" printed for a constant is worse than saying nothing.
    if n > 1 and float(np.std(x)) == 0.0:
        return [
            {"id": tid, "name": name, "stat": None, "p": None, "applicable": False,
             "note": "constant column — no distribution to test"}
            for tid, name in (
                ("shapiro", "Shapiro-Wilk"),
                ("anderson", "Anderson-Darling"),
                ("lilliefors", "Kolmogorov-Smirnov (Lilliefors)"),
                ("dagostino", "D'Agostino-Pearson K²"),
                ("jarque_bera", "Jarque-Bera"),
            )
        ]

    def add(tid, name, stat, p, note=""):
        tests.append({
            "id": tid, "name": name,
            "stat": _finite(stat) if stat is not None else None,
            "p": _finite(p) if p is not None else None,
            "applicable": p is not None,
            "note": note,
        })

    # Shapiro-Wilk — most powerful across the shapes met in clinical data
    # (Razali & Wah 2011), and the default in R.
    if n < 3:
        add("shapiro", "Shapiro-Wilk", None, None, "needs n ≥ 3")
    elif n > SHAPIRO_MAX_N:
        add("shapiro", "Shapiro-Wilk", None, None, f"defined up to n = {SHAPIRO_MAX_N}")
    else:
        try:
            w, p = scipy_stats.shapiro(x)
            add("shapiro", "Shapiro-Wilk", w, p)
        except Exception as exc:  # constant column, degenerate input
            add("shapiro", "Shapiro-Wilk", None, None, str(exc)[:80])

    a2, p_ad = _anderson_darling(x)
    add("anderson", "Anderson-Darling", a2, p_ad,
        "" if p_ad is not None else "needs n ≥ 8")

    d, p_ll = _lilliefors(x)
    add("lilliefors", "Kolmogorov-Smirnov (Lilliefors)", d, p_ll,
        "" if p_ll is not None else "needs n ≥ 5")

    # D'Agostino-Pearson K² reads skewness and kurtosis together, which is the
    # omnibus a textbook reaches for when the question is *how* it departs.
    if n >= 20:
        try:
            k2, p = scipy_stats.normaltest(x)
            add("dagostino", "D'Agostino-Pearson K²", k2, p)
        except Exception as exc:
            add("dagostino", "D'Agostino-Pearson K²", None, None, str(exc)[:80])
    else:
        add("dagostino", "D'Agostino-Pearson K²", None, None, "needs n ≥ 20")

    # Jarque-Bera is asymptotic; under ~30 its χ² reference is simply wrong,
    # so it is shown rather than silently trusted.
    if n >= 30:
        try:
            jb = scipy_stats.jarque_bera(x)
            add("jarque_bera", "Jarque-Bera", jb.statistic, jb.pvalue)
        except Exception as exc:
            add("jarque_bera", "Jarque-Bera", None, None, str(exc)[:80])
    else:
        add("jarque_bera", "Jarque-Bera", None, None, "asymptotic, needs n ≥ 30")

    return tests


def _primary_id(n: int) -> Optional[str]:
    if n < 3:
        return None
    return "shapiro" if n <= SHAPIRO_MAX_N else "anderson"


# ── plot payloads ──────────────────────────────────────────────────────────────


def _thin(a: np.ndarray) -> np.ndarray:
    if len(a) <= MAX_PLOT_POINTS:
        return a
    idx = np.linspace(0, len(a) - 1, MAX_PLOT_POINTS).round().astype(int)
    return a[idx]


def _qq(x: np.ndarray) -> dict:
    """Q-Q points on R's convention, with R's ``qqline``.

    ``ppoints`` switches its offset at n = 10 and ``qqline`` is drawn through
    the two quartiles rather than through mean ± SD — a line fitted to the
    ends would follow the outliers it is meant to expose.
    """
    n = int(len(x))
    if n < 3:
        return {}
    xs = np.sort(x)
    a = 3.0 / 8.0 if n <= 10 else 0.5
    probs = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
    theo = scipy_stats.norm.ppf(probs)

    line = {}
    y1, y3 = np.quantile(xs, [0.25, 0.75])
    x1, x3 = scipy_stats.norm.ppf([0.25, 0.75])
    if x3 != x1 and np.isfinite(y1) and np.isfinite(y3):
        slope = float((y3 - y1) / (x3 - x1))
        line = {"slope": slope, "intercept": float(y1 - slope * x1)}

    return {
        "theoretical": [_finite(v) for v in _thin(theo)],
        "sample": [_finite(v) for v in _thin(xs)],
        "line": line,
        "thinned": bool(n > MAX_PLOT_POINTS),
    }


def _histogram(x: np.ndarray) -> dict:
    n = int(len(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n < 3 or sd == 0:
        return {}
    edges = np.histogram_bin_edges(x, bins="auto")
    if len(edges) > 60:  # "auto" can go wild on heavy tails
        edges = np.histogram_bin_edges(x, bins=30)
    counts, edges = np.histogram(x, bins=edges)
    width = float(edges[1] - edges[0])
    grid = np.linspace(float(edges[0]), float(edges[-1]), 120)
    curve = scipy_stats.norm.pdf(grid, float(np.mean(x)), sd) * n * width
    return {
        "bin_edges": [_finite(v) for v in edges],
        "counts": [int(c) for c in counts],
        "curve_x": [_finite(v) for v in grid],
        "curve_y": [_finite(v) for v in curve],
    }


# ── verdict ────────────────────────────────────────────────────────────────────


def _verdict(n: int, p: Optional[float], flag: Optional[bool], alpha: float) -> dict:
    """Read the test and the shape statistics together, and say when neither
    can settle it.

    A test and a shape rule that agree give a clean answer. When they
    disagree the honest label is "borderline", not whichever one happened to
    cross its threshold — that disagreement is the whole reason the sources
    tell you to look at both.
    """
    notes: list[str] = []
    if n < 3:
        return {"code": "undetermined", "label": "Too few observations",
                "reason": "Fewer than 3 non-missing values.", "notes": notes}

    if n < 20:
        notes.append(
            "n < 20: the test has little power here, so a non-significant p is "
            "not evidence of normality. Judge from the Q-Q plot."
        )
    if n >= 300:
        notes.append(
            "n ≥ 300: formal tests flag departures too small to matter. The "
            "shape statistics and the Q-Q plot carry more weight than p."
        )
    if n >= 30:
        notes.append(
            "n ≥ 30: means-based tests (t-test, ANOVA) are robust to mild "
            "non-normality here by the central limit theorem."
        )

    sig = None if p is None else bool(p < alpha)
    if sig is None and flag is None:
        return {"code": "undetermined", "label": "Not assessable",
                "reason": "No applicable test and no shape statistics.",
                "notes": notes}

    if sig and flag:
        return {"code": "non_normal", "label": "Clear departure from normal",
                "reason": "The test rejects normality and the shape statistics agree.",
                "notes": notes}
    if sig is False and flag is False:
        return {"code": "normal", "label": "Consistent with normal",
                "reason": "The test does not reject normality and the shape statistics are within range.",
                "notes": notes}
    if sig is None:
        return {"code": "non_normal" if flag else "normal",
                "label": "Skewed shape" if flag else "Shape within range",
                "reason": "Judged from the shape statistics; no formal test applied.",
                "notes": notes}
    if flag is None:
        return {"code": "non_normal" if sig else "normal",
                "label": "Clear departure from normal" if sig else "Consistent with normal",
                "reason": "Judged from the formal test; shape statistics unavailable.",
                "notes": notes}
    return {
        "code": "borderline", "label": "Borderline",
        "reason": (
            "The test rejects normality but the shape statistics are within range."
            if sig else
            "The test does not reject normality but the shape statistics are out of range."
        ),
        "notes": notes,
    }


def _sentence(var: str, label: str, block: dict) -> str:
    """A line that can be pasted into the methods or results section."""
    n = block["n"]
    where = f"{var}" if label == "All (pooled)" else f"{var} in {label}"
    prim = block.get("primary")
    if not prim or prim.get("p") is None or prim.get("stat") is None:
        return f"{where}: n = {n}, normality not formally assessable."
    verb = {
        "normal": "did not depart from",
        "non_normal": "departed from",
        "borderline": "showed a borderline departure from",
    }.get(block["verdict"]["code"], "was assessed against")
    stat_label = "W" if prim["id"] == "shapiro" else "A²"
    return (
        f"{where} {verb} a normal distribution "
        f"({prim['name']} {stat_label} = {prim['stat']:.3f}, "
        f"{format_p(prim['p'], prefix=True)}, n = {n})."
    )


# ── per-sample block ───────────────────────────────────────────────────────────


def _describe(var: str, label: str, raw: pd.Series, alpha: float) -> dict:
    """Everything said about one variable in one sample (cohort or group)."""
    numeric = coerce_numeric(raw).replace([np.inf, -np.inf], np.nan)
    x = numeric.dropna().astype(float).to_numpy()
    n = int(len(x))
    n_missing = int(len(raw) - n)

    block: dict = {
        "label": label,
        "n": n,
        "n_missing": n_missing,
        "n_total": int(len(raw)),
        "constant": bool(n > 1 and float(np.std(x)) == 0.0),
    }

    if n > 0:
        q1, q3 = (float(v) for v in np.quantile(x, [0.25, 0.75]))
        block.update({
            "mean": _finite(np.mean(x)),
            "sd": _finite(np.std(x, ddof=1)) if n > 1 else None,
            "median": _finite(np.median(x)),
            "q1": q1, "q3": q3,
            "min": _finite(np.min(x)), "max": _finite(np.max(x)),
        })

    shape = _shape(x) if n else {}
    block["shape"] = shape
    block["shape_flag"] = _shape_flag(shape, n) if shape else None

    tests = _run_tests(x) if n else []
    block["tests"] = tests
    primary_id = _primary_id(n)
    primary = next((t for t in tests if t["id"] == primary_id and t["applicable"]), None)
    block["primary"] = primary
    block["verdict"] = _verdict(
        n, primary["p"] if primary else None, block["shape_flag"], alpha
    )
    block["qq"] = _qq(x) if n else {}
    block["histogram"] = _histogram(x) if n else {}
    block["sentence"] = _sentence(var, label, block)
    return block


@router.post("/normality")
def normality(req: NormalityRequest):
    """Normality assessment for the whole cohort and, optionally, per group."""
    df = _get_df(req.session_id)
    alpha = float(req.alpha) if 0 < float(req.alpha) < 1 else 0.05

    variables = [v for v in (req.variables or []) if v in df.columns]
    if not variables:
        raise HTTPException(status_code=400, detail="No valid variables selected")

    group_column = req.group_column if req.group_column in df.columns else None
    group_levels: list = []
    warnings: list[str] = []
    if group_column:
        group_levels = sorted_groups(df[group_column])
        if len(group_levels) < 2:
            warnings.append(
                f"'{group_column}' has fewer than 2 levels — reporting the pooled sample only."
            )
            group_column, group_levels = None, []
        elif len(group_levels) > 20:
            warnings.append(
                f"'{group_column}' has {len(group_levels)} levels; that looks like an "
                "identifier rather than a grouping variable."
            )
            group_column, group_levels = None, []

    results = []
    for var in variables:
        entry: dict = {
            "variable": var,
            # The pooled sample is always reported, even when grouping is on:
            # it is the right thing to check for a one-sample summary, and the
            # wrong thing to check before a t-test. Showing both side by side
            # is what makes the difference visible instead of a hidden default.
            "overall": _describe(var, "All (pooled)", df[var], alpha),
            "groups": [],
        }
        if group_column:
            for lvl in group_levels:
                sub = df.loc[df[group_column] == lvl, var]
                entry["groups"].append(_describe(var, str(lvl), sub, alpha))
            codes = {g["verdict"]["code"] for g in entry["groups"]}
            entry["group_summary"] = (
                "All groups consistent with normal" if codes == {"normal"}
                else "At least one group departs from normal"
                if "non_normal" in codes else "Mixed / borderline across groups"
            )
        results.append(entry)

    return sanitize_nonfinite({
        "alpha": alpha,
        "group_column": group_column,
        "group_levels": [str(g) for g in group_levels],
        "variables": results,
        "warnings": warnings,
        # Stated in the response rather than only in the UI so an exported
        # result carries the caveat with it.
        "guidance": (
            "Assess normality from the test, the shape statistics and the Q-Q plot "
            "together — no one of them decides it (Ghasemi & Zahediasl 2012; Kim 2013). "
            "Where groups are compared, the assumption behind the t-test and ANOVA is "
            "normality within each group, not in the pooled sample."
        ),
    })
