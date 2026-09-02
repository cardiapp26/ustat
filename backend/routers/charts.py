import json as _json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from services import store, plot_render
from services.dirty_value_guard import (
    coerce_numeric,
    mask_sentinels,
    plausibility_max_for_column,
)
from services.number_format import level_key
from services.stat_utils import (
    sorted_groups,
    _categorical_p_with_rule,
    pairwise_t_tests,
    pairwise_wilcoxon,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _plausibility_warnings(col: str, series: pd.Series) -> list[dict]:
    numeric = coerce_numeric(series)
    key = str(col).strip().lower()
    mask = pd.Series(False, index=series.index)
    rule = None
    if key == "age":
        mask = numeric.notna() & ((numeric < 0) | (numeric > 120))
        rule = "expected 0 <= age <= 120"
    elif key in {"bmi", "body_mass_index"} or "bmi" in key:
        mask = numeric.notna() & ((numeric <= 10) | (numeric >= 100))
        rule = "expected 10 < bmi < 100"
    elif key in {"fu_days", "followup_days", "follow_up_days"}:
        mask = numeric.notna() & (numeric <= 0)
        rule = "expected fu_days > 0"
    if not mask.any():
        return []
    return [
        {
            "variable": col,
            "n_implausible": int(mask.sum()),
            "implausible_values": sorted(
                {float(v) for v in numeric[mask].dropna().unique()}
            ),
            "rule": rule,
            "note": "Values were retained for display but should be reviewed.",
        }
    ]


class ChartRequest(BaseModel):
    session_id: str
    x: str
    y: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None
    bins: int = 20
    # Scatter-only. Log axes are what make an agreement plot readable when the
    # values span orders of magnitude (p-values, concentrations, counts).
    log_x: bool = False
    log_y: bool = False
    # Confidence ellipse per group (ggpubr's stat_conf_ellipse) and marginal
    # histograms (ggscatterhist). Both describe the cloud rather than the fit.
    ellipse: bool = False
    ellipse_level: float = 0.95
    marginal: bool = False
    marginal_bins: int = 24
    # y = x reference. Only meaningful when both axes carry the same quantity —
    # a reported value against a recomputed one, a method against a reference.
    identity_line: bool = False
    # Bar-only. "percentage" reports the share of each group meeting a
    # condition on y rather than y's mean — the form a risk-factor figure
    # needs, and the one a 0/1 mean silently gives at the wrong scale.
    y_mode: Optional[str] = "mean"
    target_value: Optional[str] = None
    # Bar-only, mean mode. Whisker on each bar — ggplot2's geom_col with
    # stat_summary's errorbar. sd | se | ci; None draws bare bars.
    error: Optional[str] = None
    # Column whose value labels each point (e.g. the variable name per row).
    label: Optional[str] = None
    # Scatter-only — geom_smooth. "lm" is the straight line with its CI band,
    # "loess" the local curve (statsmodels lowess), "none" the bare cloud.
    fit: str = "lm"
    fit_per_group: bool = False
    loess_span: float = 0.75


def _mean_bars(sub: pd.DataFrame, x: str, y: str, spread: Optional[str]) -> list[dict]:
    """One bar per level of ``x`` at the mean of ``y``, with the whisker asked for.

    A bar chart of means without a spread is the figure reviewers call a
    dynamite plot; with the whisker it at least says how firm each mean is.
    Same arithmetic as the error plot: SD describes the sample, SE and the
    t-based CI describe the estimate.
    """
    if spread not in (None, "", "sd", "se", "ci"):
        raise HTTPException(status_code=400, detail="error must be sd, se or ci")
    rows: list[dict] = []
    values = coerce_numeric(sub[y]).replace([np.inf, -np.inf], np.nan)
    block = pd.DataFrame({"x": sub[x], "y": values}).dropna(subset=["y"])
    for level in sorted_groups(block["x"]):
        vals = block.loc[block["x"] == level, "y"].astype(float).to_numpy()
        n = int(len(vals))
        if n == 0:
            continue
        mean = float(np.mean(vals))
        row: dict = {"label": level_key(level), "value": mean, "n": n}
        if spread:
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            se = sd / np.sqrt(n)
            if spread == "sd":
                half = sd
            elif spread == "se":
                half = se
            else:
                half = float(scipy_stats.t.ppf(0.975, n - 1)) * se if n > 1 else 0.0
            row.update({"sd": sd, "se": se, "lower": mean - half, "upper": mean + half})
        rows.append(row)
    return rows


_BAR_ERROR_LABEL = {"sd": "mean ± SD", "se": "mean ± SE", "ci": "mean with 95% CI"}


@router.post("/histogram")
def histogram(req: ChartRequest):
    df = _get_df(req.session_id)
    if req.x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.x}' not found")
    s = coerce_numeric(df[req.x]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        raise HTTPException(
            status_code=400, detail="Need at least 2 numeric values for a histogram."
        )
    counts, edges = np.histogram(s, bins=req.bins)
    kde_x = np.linspace(s.min(), s.max(), 200)
    kde_points = []
    if len(s) >= 3 and float(s.std()) > 0:
        kde = scipy_stats.gaussian_kde(s)
        kde_points = [
            {"x": float(kx), "y": float(ky)} for kx, ky in zip(kde_x, kde(kde_x))
        ]
    return {
        "type": "histogram",
        "x": req.x,
        "bins": [
            {"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(len(counts))
        ],
        "kde": kde_points,
        "stats": {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
        },
        "warnings": _plausibility_warnings(req.x, df[req.x]),
    }


def _empty_fit(note: Optional[str], method: str) -> dict:
    return {
        "method": method,
        "slope": None, "intercept": None, "r": None, "r2": None, "p": None,
        "se": None, "line_x": [], "line_y": [], "band": {},
        "spearman": {"rho": None, "p": None},
        "note": note,
    }


def _finite_or_none(v) -> Optional[float]:
    # scipy answers n = 2 with rho = 1 and p = nan. A nan reaching the
    # response is not a small annoyance: the endpoint's non-finite guard
    # rejects the WHOLE payload with a 400, so a two-point group would have
    # cost the user the entire scatter.
    fv = float(v)
    return fv if np.isfinite(fv) else None


def _fit_curve(
    x_raw: pd.Series,
    y_raw: pd.Series,
    log_x: bool,
    log_y: bool,
    method: str,
    span: float,
) -> dict:
    """Trend through a cloud — ggplot2's geom_smooth(method = lm | loess).

    Fitted in the space the reader sees: on a log axis a fit computed from raw
    values renders as a curve matching no visible trend, so the transformed
    variables are fitted and back-transformed with the line. Pearson and
    Spearman are reported for every method, including "none" — the caption of
    a skewed clinical scatter quotes rho whether or not a line is drawn.
    """
    if method not in {"lm", "loess", "none"}:
        raise HTTPException(status_code=400, detail="fit must be lm, loess or none")
    if not (0.1 <= span <= 1.0):
        raise HTTPException(status_code=400, detail="loess_span must be between 0.1 and 1")
    fit_x = np.log10(x_raw) if log_x else x_raw
    fit_y = np.log10(y_raw) if log_y else y_raw
    fx = np.asarray(fit_x, dtype=float)
    fy = np.asarray(fit_y, dtype=float)
    n_fit = int(len(fx))
    space = (
        "log10-log10" if log_x and log_y
        else "log10-x" if log_x
        else "log10-y" if log_y
        else "linear"
    )

    def _back_x(v: float) -> float:
        return float(10 ** v) if log_x else float(v)

    def _back_y(v: float) -> float:
        return float(10 ** v) if log_y else float(v)

    try:
        slope, intercept, r, p, se = scipy_stats.linregress(fx.tolist(), fy.tolist())
        if np.isnan(r) or np.isinf(r):
            raise ValueError("degenerate")
    except Exception:
        return _empty_fit("Regression unavailable (constant or degenerate data)", method)

    try:
        rho, p_rho = scipy_stats.spearmanr(fx, fy)
        spearman = {"rho": _finite_or_none(rho), "p": _finite_or_none(p_rho)}
    except Exception:
        spearman = {"rho": None, "p": None}

    out: dict = {
        "method": method,
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "r2": float(r ** 2),
        "p": float(p),
        "se": float(se),
        "line_x": [],
        "line_y": [],
        "band": {},
        "spearman": spearman,
        "n": n_fit,
        "space": space,
    }
    fit_lo, fit_hi = float(fx.min()), float(fx.max())

    if method == "lm":
        out["line_x"] = [_back_x(fit_lo), _back_x(fit_hi)]
        out["line_y"] = [_back_y(slope * v + intercept) for v in (fit_lo, fit_hi)]
        # Confidence band for the fitted LINE (not a prediction interval):
        #   se_fit(x) = s * sqrt(1/n + (x - x̄)² / Sxx)
        # A bare line says nothing about how well the slope is pinned down,
        # and at the ends of the x range — where readers extrapolate — it is
        # pinned down least.
        if n_fit > 2:
            sxx = float(((fx - fx.mean()) ** 2).sum())
            resid = fy - (slope * fx + intercept)
            dof = n_fit - 2
            s_err = float(np.sqrt((resid ** 2).sum() / dof))
            if sxx > 0 and np.isfinite(s_err):
                t_crit = float(scipy_stats.t.ppf(0.975, dof))
                grid = np.linspace(fit_lo, fit_hi, 60)
                centre = slope * grid + intercept
                half = t_crit * s_err * np.sqrt(1.0 / n_fit + (grid - fx.mean()) ** 2 / sxx)
                out["band"] = {
                    "x": [_back_x(v) for v in grid],
                    "lo": [_back_y(v) for v in (centre - half)],
                    "hi": [_back_y(v) for v in (centre + half)],
                    "level": 0.95,
                }
    elif method == "loess":
        # statsmodels' lowess is the local regression ggplot2 draws; no
        # closed-form band, so none is claimed.
        out["span"] = float(span)
        if n_fit < 4 or fit_lo == fit_hi:
            out["note"] = "LOESS needs at least 4 points spread along x"
        else:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            curve = lowess(fy, fx, frac=span, it=3, return_sorted=True)
            # One point per distinct x, on the fitting scale, then back.
            xs = [_back_x(float(v)) for v in curve[:, 0]]
            ys = [_back_y(float(v)) for v in curve[:, 1]]
            seen: dict[float, float] = {}
            for xv, yv in zip(xs, ys):
                seen[xv] = yv
            if any(not np.isfinite(v) for v in seen.values()):
                out["note"] = "LOESS did not converge on these data"
            else:
                out["line_x"] = list(seen.keys())
                out["line_y"] = list(seen.values())
    return out


@router.post("/scatter")
def scatter(req: ChartRequest):
    df = _get_df(req.session_id)

    # Build deduplicated column list
    needed = [req.x, req.y]
    if req.color and req.color not in needed:
        needed.append(req.color)
    if req.shape and req.shape not in needed:
        needed.append(req.shape)
    if req.label and req.label not in needed:
        needed.append(req.label)

    for col in needed:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    # Clean: replace inf→nan on numeric cols only, then drop missing
    sub = df[needed].copy()
    for col in needed:
        if sub[col].dtype.kind in ("f", "i", "u"):
            sub[col] = sub[col].replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna()

    if len(sub) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough non-missing data points to draw scatter (need ≥ 2)",
        )

    # Regression only when both axes are numeric
    x_numeric = df[req.x].dtype.kind in ("f", "i", "u")
    y_numeric = df[req.y].dtype.kind in ("f", "i", "u")

    # A log axis cannot show zero or negative values. Drop them here rather
    # than letting the browser silently omit the points, and say how many went
    # — a scatter quietly missing a third of its data is worse than an error.
    axis_warnings: list[dict] = []
    if req.log_x or req.log_y:
        if not (x_numeric and y_numeric):
            raise HTTPException(
                status_code=400,
                detail="Log axes require both axes to be numeric.",
            )
        before = len(sub)
        keep = pd.Series(True, index=sub.index)
        if req.log_x:
            keep &= sub[req.x].astype(float) > 0
        if req.log_y:
            keep &= sub[req.y].astype(float) > 0
        sub = sub[keep]
        dropped = before - len(sub)
        if dropped:
            axes = " and ".join(
                [a for a, on in ((req.x, req.log_x), (req.y, req.log_y)) if on]
            )
            axis_warnings.append(
                {
                    "type": "log_axis_nonpositive",
                    "n_dropped": int(dropped),
                    "message": (
                        f"{dropped} of {before} points had a zero or negative value on "
                        f"{axes} and cannot be placed on a log axis. They are omitted."
                    ),
                }
            )
        if len(sub) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Fewer than 2 points remain with positive values on the log axis — "
                    "nothing can be drawn. Turn the log axis off, or filter the data."
                ),
            )

    # The fit — ggplot2's geom_smooth. Overall, and per group when asked,
    # each computed in the space the reader sees (see _fit_curve).
    if x_numeric and y_numeric:
        reg = _fit_curve(
            sub[req.x].astype(float), sub[req.y].astype(float),
            req.log_x, req.log_y, req.fit, req.loess_span,
        )
    else:
        reg = _empty_fit("Regression requires two numeric axes", req.fit)
    group_fits: list[dict] = []
    if req.color and req.fit_per_group and x_numeric and y_numeric and req.fit != "none":
        for level in sorted_groups(sub[req.color]):
            rows = sub[sub[req.color] == level]
            group_fits.append({
                "group": str(level),
                "n": int(len(rows)),
                **_fit_curve(
                    rows[req.x].astype(float), rows[req.y].astype(float),
                    req.log_x, req.log_y, req.fit, req.loess_span,
                ),
            })

    # y = x over the span both axes share, so the line stops where the data
    # stops instead of stretching the plot to an empty corner.
    identity: dict = {}
    if req.identity_line:
        if not (x_numeric and y_numeric):
            raise HTTPException(
                status_code=400,
                detail="A y = x reference line requires both axes to be numeric.",
            )
        lo = min(float(sub[req.x].min()), float(sub[req.y].min()))
        hi = max(float(sub[req.x].max()), float(sub[req.y].max()))
        if lo == hi:
            identity = {"line_x": [], "line_y": [], "note": "Degenerate range"}
        else:
            identity = {"line_x": [lo, hi], "line_y": [lo, hi]}
            n_below = int((sub[req.y].astype(float) < sub[req.x].astype(float)).sum())
            identity["n_below"] = n_below
            identity["n_above"] = int(len(sub) - n_below)

    # Confidence ellipses. The 2-df chi-square quantile is what makes this a
    # containment region for a bivariate normal rather than a decorative oval;
    # groups with fewer than 3 points or a singular covariance get none.
    ellipses: list[dict] = []
    if req.ellipse:
        if not (x_numeric and y_numeric):
            raise HTTPException(
                status_code=400,
                detail="A confidence ellipse requires both axes to be numeric.",
            )
        if not (0 < req.ellipse_level < 1):
            raise HTTPException(
                status_code=400, detail="ellipse_level must be between 0 and 1"
            )
        radius = float(np.sqrt(scipy_stats.chi2.ppf(req.ellipse_level, df=2)))
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.vstack([np.cos(theta), np.sin(theta)])
        buckets = (
            {str(g): sub[sub[req.color] == g] for g in sorted_groups(sub[req.color])}
            if req.color
            else {"All": sub}
        )
        for name, rows in buckets.items():
            pts = rows[[req.x, req.y]].astype(float).to_numpy()
            if len(pts) < 3:
                ellipses.append(
                    {"group": name, "x": [], "y": [], "note": "needs at least 3 points"}
                )
                continue
            cov = np.cov(pts, rowvar=False)
            centre = pts.mean(axis=0)
            try:
                vals, vecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                ellipses.append(
                    {"group": name, "x": [], "y": [], "note": "singular covariance"}
                )
                continue
            if np.any(vals <= 0):
                ellipses.append(
                    {"group": name, "x": [], "y": [], "note": "degenerate spread"}
                )
                continue
            transform = vecs @ np.diag(np.sqrt(vals)) * radius
            pathpts = (transform @ circle).T + centre
            ellipses.append(
                {
                    "group": name,
                    "n": int(len(pts)),
                    "x": [float(v) for v in pathpts[:, 0]],
                    "y": [float(v) for v in pathpts[:, 1]],
                }
            )

    # Marginal histograms — ggscatterhist. Counts only; the frontend places them.
    marginal: dict = {}
    if req.marginal:
        if not (x_numeric and y_numeric):
            raise HTTPException(
                status_code=400,
                detail="Marginal histograms require both axes to be numeric.",
            )
        if not (3 <= req.marginal_bins <= 200):
            raise HTTPException(
                status_code=400, detail="marginal_bins must be between 3 and 200"
            )
        for axis, col in (("x", req.x), ("y", req.y)):
            counts, edges = np.histogram(
                sub[col].astype(float).to_numpy(), bins=req.marginal_bins
            )
            marginal[axis] = [
                {
                    "centre": float((edges[i] + edges[i + 1]) / 2),
                    "x0": float(edges[i]),
                    "x1": float(edges[i + 1]),
                    "count": int(c),
                }
                for i, c in enumerate(counts)
            ]

    # Serialize points safely (NaN → null via json round-trip)
    points = _json.loads(
        sub.to_json(
            orient="records", default_handler=str, date_format="iso", date_unit="s"
        )
    )

    return {
        "type": "scatter",
        "x": req.x,
        "y": req.y,
        "points": points,
        "regression": reg,
        "fit": req.fit,
        "fit_per_group": bool(group_fits),
        "regressions": group_fits,
        "color": req.color,
        "shape": req.shape,
        "label": req.label,
        "log_x": req.log_x,
        "log_y": req.log_y,
        "identity": identity,
        "ellipses": ellipses,
        "ellipse_level": req.ellipse_level if req.ellipse else None,
        "marginal": marginal,
        "warnings": axis_warnings,
    }


STAR_CUTOFFS = ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (5e-2, "*"))


def _stars(p: float) -> str:
    """ggpubr's convention: **** <=1e-4, *** <=1e-3, ** <=1e-2, * <=0.05, else ns."""
    if p is None or not np.isfinite(p):
        return "ns"
    for cutoff, mark in STAR_CUTOFFS:
        if p <= cutoff:
            return mark
    return "ns"


class CompareMeansRequest(BaseModel):
    session_id: str
    y: str                                   # the numeric variable being compared
    group: str                               # the categorical axis
    method: str = "auto"                     # auto | t | welch | wilcoxon
    p_adjust: str = "holm"                   # holm | bonferroni | fdr | none
    ref_group: Optional[str] = None          # compare everything against this level
    label: str = "stars"                     # stars | p


@router.post("/compare_means")
def compare_means(req: CompareMeansRequest):
    """Pairwise group comparisons, positioned for drawing as brackets on a plot.

    The equivalent of ggpubr's stat_compare_means. Two things it does that a
    bare p-value list does not: it says which test produced each number and
    why, and it reports the adjusted p separately from the raw one so a figure
    cannot silently show unadjusted values from a dozen comparisons.
    """
    df = _get_df(req.session_id)
    for col in (req.y, req.group):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    if req.method not in {"auto", "t", "welch", "wilcoxon"}:
        raise HTTPException(
            status_code=400,
            detail=f"method must be auto, t, welch or wilcoxon — got '{req.method}'",
        )
    if req.p_adjust not in {"holm", "bonferroni", "fdr", "none"}:
        raise HTTPException(
            status_code=400,
            detail=f"p_adjust must be holm, bonferroni, fdr or none — got '{req.p_adjust}'",
        )
    if req.label not in {"stars", "p"}:
        raise HTTPException(
            status_code=400, detail=f"label must be stars or p — got '{req.label}'"
        )

    sub = df[[req.y, req.group]].copy()
    sub[req.y] = coerce_numeric(sub[req.y]).replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna()

    groups: dict[str, np.ndarray] = {}
    for name in sorted_groups(sub[req.group]):
        vals = sub.loc[sub[req.group] == name, req.y].astype(float).to_numpy()
        if len(vals) >= 2:
            groups[str(name)] = vals

    if len(groups) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least two levels of '{req.group}' with 2 or more non-missing "
                f"'{req.y}' values; found {len(groups)}."
            ),
        )
    if req.ref_group is not None and req.ref_group not in groups:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reference group '{req.ref_group}' is not one of the usable levels: "
                + ", ".join(groups)
            ),
        )

    # Test choice. "auto" screens every group for normality and falls to the
    # rank test if any fails — the same rule a careful analyst applies by hand,
    # stated here so the figure legend can repeat it.
    chosen, why = req.method, "requested"
    if req.method == "auto":
        normal = True
        for vals in groups.values():
            if len(vals) < 3:
                normal = False
                break
            if float(np.std(vals, ddof=1)) == 0:
                continue
            if scipy_stats.shapiro(vals[:5000]).pvalue < 0.05:
                normal = False
                break
        chosen = "welch" if normal else "wilcoxon"
        why = (
            "auto: all groups passed Shapiro-Wilk"
            if normal
            else "auto: at least one group failed Shapiro-Wilk"
        )

    if chosen == "wilcoxon":
        rows = pairwise_wilcoxon(groups, correction=req.p_adjust)
        test_name = "Mann-Whitney U"
    else:
        rows = pairwise_t_tests(
            groups, correction=req.p_adjust, equal_var=(chosen == "t")
        )
        test_name = "Student t-test" if chosen == "t" else "Welch t-test"

    if req.ref_group is not None:
        rows = [
            r for r in rows if req.ref_group in (r["group1"], r["group2"])
        ]
        if not rows:
            raise HTTPException(
                status_code=400,
                detail=f"No comparisons involve reference group '{req.ref_group}'.",
            )

    # Bracket geometry. Height is left to the caller (it depends on the axis),
    # but the ordering — shortest span lowest — is what keeps them from
    # crossing, so it belongs with the data.
    order = {name: i for i, name in enumerate(groups)}
    for r in rows:
        r["x1"] = order[r["group1"]]
        r["x2"] = order[r["group2"]]
        r["span"] = abs(r["x2"] - r["x1"])
        shown = r["p"] if req.p_adjust == "none" else r["p_adj"]
        r["p_shown"] = float(shown)
        r["stars"] = _stars(float(shown))
        r["label"] = (
            r["stars"] if req.label == "stars"
            else ("p < 0.001" if shown < 1e-3 else f"p = {shown:.3f}")
        )
    rows.sort(key=lambda r: (r["span"], min(r["x1"], r["x2"])))
    for level, r in enumerate(rows):
        r["level"] = level  # stacking order, bottom-up

    # Omnibus, so a figure with three or more groups can carry the overall test
    # rather than implying the pairwise set is the whole analysis.
    arrays = list(groups.values())
    omnibus: dict = {}
    if len(arrays) >= 3:
        if chosen == "wilcoxon":
            h, p_om = scipy_stats.kruskal(*arrays)
            omnibus = {"test": "Kruskal-Wallis", "statistic": float(h), "p": float(p_om)}
        else:
            f, p_om = scipy_stats.f_oneway(*arrays)
            omnibus = {
                "test": "One-way ANOVA",
                "statistic": float(f),
                "p": float(p_om),
                "note": (
                    "Classic equal-variance F. It is not Welch-corrected even when the "
                    "pairwise tests are."
                ),
            }

    return {
        "type": "compare_means",
        "y": req.y,
        "group": req.group,
        "levels": list(groups),
        "n_per_group": {k: int(len(v)) for k, v in groups.items()},
        "test": test_name,
        "test_selected_by": why,
        "p_adjust": req.p_adjust,
        "p_shown_is_adjusted": req.p_adjust != "none",
        "comparisons": rows,
        "omnibus": omnibus,
    }


class LinePlotRequest(BaseModel):
    session_id: str
    x: str                       # the ordered axis — visit, dose, time point
    y: str
    group: Optional[str] = None
    centre: str = "mean"         # mean | median
    spread: str = "ci"           # ci | se | sd | iqr | none
    ci_level: float = 0.95


@router.post("/lineplot")
def lineplot(req: LinePlotRequest):
    """Group means across an ordered axis, with a band — cnsplots' lineplot.

    The repeated-measures figure: one line per arm across visits. Each point
    carries the n behind it, because in longitudinal data the n almost always
    falls over time and a line that thins out looks identical to one that does
    not.
    """
    df = _get_df(req.session_id)
    for col in (req.x, req.y):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    if req.group and req.group not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.group}' not found")
    if req.centre not in {"mean", "median"}:
        raise HTTPException(status_code=400, detail="centre must be mean or median")
    if req.spread not in {"ci", "se", "sd", "iqr", "none"}:
        raise HTTPException(
            status_code=400, detail="spread must be ci, se, sd, iqr or none"
        )
    if req.centre == "median" and req.spread in {"ci", "se", "sd"}:
        raise HTTPException(
            status_code=400,
            detail="Pair median with iqr (or none); ci / se / sd describe a mean.",
        )
    if req.centre == "mean" and req.spread == "iqr":
        raise HTTPException(
            status_code=400, detail="An IQR band belongs with the median."
        )

    cols = [req.x, req.y] + ([req.group] if req.group else [])
    sub = df[list(dict.fromkeys(cols))].copy()
    sub[req.y] = coerce_numeric(sub[req.y]).replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=[req.x, req.y])
    if sub.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No rows have both '{req.x}' and a numeric '{req.y}'.",
        )

    x_levels = sorted_groups(sub[req.x])
    series = []
    for gname in (sorted_groups(sub[req.group]) if req.group else ["All"]):
        block = sub[sub[req.group] == gname] if req.group else sub
        pts = []
        for xv in x_levels:
            vals = block.loc[block[req.x] == xv, req.y].astype(float).to_numpy()
            if len(vals) == 0:
                continue
            n = int(len(vals))
            if req.centre == "median":
                centre = float(np.median(vals))
                if req.spread == "iqr":
                    lo, hi = (float(v) for v in np.percentile(vals, [25, 75]))
                else:
                    lo = hi = centre
            else:
                centre = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                se = sd / np.sqrt(n) if n else 0.0
                if req.spread == "sd":
                    lo, hi = centre - sd, centre + sd
                elif req.spread == "se":
                    lo, hi = centre - se, centre + se
                elif req.spread == "ci" and n > 1:
                    t = float(scipy_stats.t.ppf(0.5 + req.ci_level / 2, n - 1))
                    lo, hi = centre - t * se, centre + t * se
                else:
                    lo = hi = centre
            pts.append(
                {"x": str(xv), "n": n, "centre": centre,
                 "lower": float(lo), "upper": float(hi)}
            )
        if pts:
            series.append({"group": str(gname), "points": pts})

    # A shrinking n is the thing readers miss in a longitudinal line.
    warnings: list[dict] = []
    for s in series:
        ns = [p["n"] for p in s["points"]]
        if ns and min(ns) < max(ns) * 0.5:
            warnings.append(
                {
                    "type": "attrition",
                    "group": s["group"],
                    "message": (
                        f"'{s['group']}' falls from n = {max(ns)} to n = {min(ns)} "
                        "across the axis. Later points rest on far fewer observations."
                    ),
                }
            )

    return {
        "type": "lineplot",
        "x": req.x, "y": req.y, "group": req.group,
        "centre": req.centre, "spread": req.spread,
        "x_levels": [str(v) for v in x_levels],
        "series": series,
        "warnings": warnings,
    }


class SlopePlotRequest(BaseModel):
    session_id: str
    before: str
    after: str
    group: Optional[str] = None
    label: Optional[str] = None
    test: str = "auto"           # auto | paired_t | wilcoxon | none


@router.post("/slopeplot")
def slopeplot(req: SlopePlotRequest):
    """Before-and-after, one line per subject — cnsplots' slopeplot.

    Rows missing either measurement are excluded and counted: a paired
    comparison computed on whoever happened to have both values is a different
    analysis from the one the figure implies.
    """
    df = _get_df(req.session_id)
    for col in (req.before, req.after):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    for col in (req.group, req.label):
        if col and col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    if req.before == req.after:
        raise HTTPException(
            status_code=400, detail="The two measurements must be different columns."
        )
    if req.test not in {"auto", "paired_t", "wilcoxon", "none"}:
        raise HTTPException(
            status_code=400, detail="test must be auto, paired_t, wilcoxon or none"
        )

    cols = [req.before, req.after] + [c for c in (req.group, req.label) if c]
    sub = df[list(dict.fromkeys(cols))].copy()
    for col in (req.before, req.after):
        sub[col] = coerce_numeric(sub[col]).replace([np.inf, -np.inf], np.nan)
    complete = sub.dropna(subset=[req.before, req.after])
    n_incomplete = int(len(sub) - len(complete))
    if len(complete) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Fewer than 2 rows have both measurements; there is nothing to pair."
            ),
        )

    pairs = [
        {
            "before": float(r[req.before]),
            "after": float(r[req.after]),
            "change": float(r[req.after] - r[req.before]),
            "group": str(r[req.group]) if req.group else None,
            "label": str(r[req.label]) if req.label else None,
        }
        for _, r in complete.iterrows()
    ]

    b = complete[req.before].astype(float).to_numpy()
    a = complete[req.after].astype(float).to_numpy()
    diff = a - b
    chosen, why = req.test, "requested"
    if req.test == "auto":
        normal = len(diff) >= 3 and (
            float(np.std(diff, ddof=1)) == 0
            or scipy_stats.shapiro(diff[:5000]).pvalue >= 0.05
        )
        chosen = "paired_t" if normal else "wilcoxon"
        why = (
            "auto: the differences passed Shapiro-Wilk"
            if normal
            else "auto: the differences failed Shapiro-Wilk"
        )

    test_result: dict = {}
    if chosen == "paired_t":
        t, p = scipy_stats.ttest_rel(a, b)
        test_result = {
            "test": "Paired t-test", "statistic": float(t), "p": float(p),
            "df": int(len(diff) - 1), "selected_by": why,
        }
    elif chosen == "wilcoxon":
        if np.all(diff == 0):
            test_result = {
                "test": "Wilcoxon signed-rank", "statistic": None, "p": None,
                "selected_by": why,
                "note": "Every difference is zero; the test is undefined.",
            }
        else:
            w, p = scipy_stats.wilcoxon(a, b)
            test_result = {
                "test": "Wilcoxon signed-rank", "statistic": float(w),
                "p": float(p), "selected_by": why,
            }

    warnings: list[dict] = []
    if n_incomplete:
        warnings.append(
            {
                "type": "incomplete_pairs",
                "n_dropped": n_incomplete,
                "message": (
                    f"{n_incomplete} rows lack one of the two measurements and are "
                    f"excluded. The comparison covers {len(complete)} complete pairs."
                ),
            }
        )

    return {
        "type": "slopeplot",
        "before": req.before, "after": req.after,
        "group": req.group, "label": req.label,
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_incomplete": n_incomplete,
        "mean_change": float(np.mean(diff)),
        "median_change": float(np.median(diff)),
        "n_increased": int((diff > 0).sum()),
        "n_decreased": int((diff < 0).sum()),
        "n_unchanged": int((diff == 0).sum()),
        "test_result": test_result,
        "warnings": warnings,
    }


class SankeyRequest(BaseModel):
    session_id: str
    stages: List[str]            # two or more columns, read left to right
    value: Optional[str] = None  # omit to count rows
    min_flow: int = 0            # drop links thinner than this


@router.post("/sankey")
def sankey(req: SankeyRequest):
    """Flow between successive states — cnsplots' sankeyplot.

    Stage names are prefixed with their column, so a level that appears at two
    stages ("Alive" at baseline and at follow-up) becomes two nodes rather
    than one node with a loop back into itself.
    """
    df = _get_df(req.session_id)
    if len(req.stages) < 2:
        raise HTTPException(
            status_code=400, detail="A Sankey needs at least two stage columns."
        )
    if len(set(req.stages)) != len(req.stages):
        raise HTTPException(
            status_code=400, detail="Each stage column must be listed only once."
        )
    for col in req.stages + ([req.value] if req.value else []):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    if req.min_flow < 0:
        raise HTTPException(status_code=400, detail="min_flow cannot be negative")

    cols = list(dict.fromkeys(req.stages + ([req.value] if req.value else [])))
    sub = df[cols].copy()
    if req.value:
        sub[req.value] = coerce_numeric(sub[req.value]).replace(
            [np.inf, -np.inf], np.nan
        )
    sub = sub.dropna()
    if sub.empty:
        raise HTTPException(
            status_code=400,
            detail="No rows have every stage present.",
        )

    node_index: dict[str, int] = {}
    labels: list[str] = []
    stage_of: list[int] = []

    def node(stage_i: int, level: str) -> int:
        key = f"{req.stages[stage_i]}={level}"
        if key not in node_index:
            node_index[key] = len(labels)
            labels.append(str(level))
            stage_of.append(stage_i)
        return node_index[key]

    links = []
    dropped = 0
    for i in range(len(req.stages) - 1):
        src_col, dst_col = req.stages[i], req.stages[i + 1]
        if req.value:
            grouped = sub.groupby([src_col, dst_col], dropna=True)[req.value].sum()
        else:
            grouped = sub.groupby([src_col, dst_col], dropna=True).size()
        for (s, t), v in grouped.items():
            if float(v) <= req.min_flow:
                dropped += 1
                continue
            links.append(
                {
                    "source": node(i, str(s)),
                    "target": node(i + 1, str(t)),
                    "value": float(v),
                    "from": str(s),
                    "to": str(t),
                    "stage": i,
                }
            )

    if not links:
        raise HTTPException(
            status_code=400,
            detail=(
                "No flows survive the min_flow threshold; nothing would be drawn."
            ),
        )

    return {
        "type": "sankey",
        "stages": req.stages,
        "value": req.value,
        "measure": "sum" if req.value else "count",
        "labels": labels,
        "node_stage": stage_of,
        "links": links,
        "n_rows": int(len(sub)),
        "n_links_dropped": dropped,
    }


class StackPlotRequest(BaseModel):
    session_id: str
    x: str                    # the axis category
    fill: str                 # what is stacked within each bar
    value: Optional[str] = None
    normalize: bool = False   # each bar to 100%


@router.post("/stackplot")
def stackplot(req: StackPlotRequest):
    """Composition within each category — cnsplots' stackplot.

    Counts and percentages both come back. A 100% stacked bar hides how many
    observations each bar rests on, so the raw n per bar is returned even in
    normalised mode and is meant to be shown.
    """
    df = _get_df(req.session_id)
    for col in (req.x, req.fill):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    if req.value and req.value not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.value}' not found")
    if req.x == req.fill:
        raise HTTPException(
            status_code=400, detail="The axis and the stacked variable must differ."
        )

    cols = [req.x, req.fill] + ([req.value] if req.value else [])
    sub = df[list(dict.fromkeys(cols))].copy()
    if req.value:
        sub[req.value] = coerce_numeric(sub[req.value]).replace(
            [np.inf, -np.inf], np.nan
        )
        sub = sub.dropna()
        if (sub[req.value] < 0).any():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{req.value}' contains negative values; a stacked bar cannot "
                    "represent them as parts of a total."
                ),
            )
        table = sub.pivot_table(
            index=req.x, columns=req.fill, values=req.value, aggfunc="sum", fill_value=0
        )
    else:
        sub = sub.dropna()
        table = pd.crosstab(sub[req.x], sub[req.fill])

    if table.empty:
        raise HTTPException(status_code=400, detail="Nothing to stack.")

    totals = table.sum(axis=1)
    series = [
        {
            "fill": str(col),
            "x": [str(i) for i in table.index],
            "value": [float(v) for v in table[col]],
            "percent": [
                float(v) / float(t) * 100.0 if t else 0.0
                for v, t in zip(table[col], totals)
            ],
        }
        for col in table.columns
    ]
    return {
        "type": "stackplot",
        "x": req.x, "fill": req.fill, "value": req.value,
        "normalize": req.normalize,
        "measure": "sum" if req.value else "count",
        "x_levels": [str(i) for i in table.index],
        "totals": {str(i): float(t) for i, t in totals.items()},
        "series": series,
    }


class RidgePlotRequest(BaseModel):
    session_id: str
    x: str
    group: str
    points: int = 200


@router.post("/ridgeplot")
def ridgeplot(req: RidgePlotRequest):
    """One density per group, stacked — cnsplots' ridgeplot.

    Every group is evaluated on the same grid, so the curves are comparable;
    per-group grids would make distributions of different width look alike.
    Densities are scaled to a common peak for legibility, and the n is
    returned because a smooth curve over six observations looks as
    authoritative as one over six hundred.
    """
    df = _get_df(req.session_id)
    for col in (req.x, req.group):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    if not (20 <= req.points <= 1000):
        raise HTTPException(status_code=400, detail="points must be between 20 and 1000")

    sub = df[[req.x, req.group]].copy()
    sub[req.x] = coerce_numeric(sub[req.x]).replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna()
    if sub.empty:
        raise HTTPException(
            status_code=400, detail=f"No numeric values in '{req.x}'."
        )

    lo, hi = float(sub[req.x].min()), float(sub[req.x].max())
    if lo == hi:
        raise HTTPException(
            status_code=400,
            detail=f"'{req.x}' is constant; a density curve cannot be drawn.",
        )
    pad = (hi - lo) * 0.05
    grid = np.linspace(lo - pad, hi + pad, req.points)

    ridges = []
    warnings: list[dict] = []
    for name in sorted_groups(sub[req.group]):
        vals = sub.loc[sub[req.group] == name, req.x].astype(float).to_numpy()
        n = int(len(vals))
        if n < 3 or float(np.std(vals, ddof=1)) == 0:
            warnings.append(
                {
                    "type": "no_density",
                    "group": str(name),
                    "message": (
                        f"'{name}' has {n} values with no usable spread; no curve is "
                        "drawn for it."
                    ),
                }
            )
            continue
        dens = scipy_stats.gaussian_kde(vals)(grid)
        ridges.append(
            {
                "group": str(name),
                "n": n,
                "x": [float(v) for v in grid],
                "density": [float(v) for v in dens],
                "peak": float(dens.max()),
                "median": float(np.median(vals)),
            }
        )
    if not ridges:
        raise HTTPException(
            status_code=400,
            detail="No group has enough spread for a density curve.",
        )
    thin = [r["group"] for r in ridges if r["n"] < 10]
    if thin:
        warnings.append(
            {
                "type": "thin_groups",
                "message": (
                    "Fewer than 10 observations in: " + ", ".join(thin)
                    + ". A smooth curve there is mostly the kernel, not the data."
                ),
            }
        )
    return {
        "type": "ridgeplot",
        "x": req.x, "group": req.group,
        "ridges": ridges,
        "max_peak": max(r["peak"] for r in ridges),
        "warnings": warnings,
    }


class SetsRequest(BaseModel):
    session_id: str
    columns: List[str]        # membership columns, read as truthy
    max_sets: int = 6


@router.post("/sets")
def sets(req: SetsRequest):
    """Overlap between membership columns — Venn and UpSet.

    Returns every non-empty intersection with its exclusive size, which is
    what both a Venn region and an UpSet bar show. Set count is capped
    because the number of regions doubles with each column.
    """
    df = _get_df(req.session_id)
    if len(req.columns) < 2:
        raise HTTPException(status_code=400, detail="Give at least two columns.")
    if len(set(req.columns)) != len(req.columns):
        raise HTTPException(status_code=400, detail="Columns must be distinct.")
    if len(req.columns) > req.max_sets:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{len(req.columns)} sets would make {2 ** len(req.columns) - 1} "
                f"regions; the limit is {req.max_sets} columns."
            ),
        )
    for col in req.columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    truthy = {"1", "1.0", "true", "yes", "y", "evet", "var", "positive", "pos"}

    def as_member(series: pd.Series) -> np.ndarray:
        numeric = coerce_numeric(series)
        if numeric.notna().sum() >= max(1, int(0.8 * len(series))):
            return (numeric.fillna(0) != 0).to_numpy()
        return series.astype(str).str.strip().str.lower().isin(truthy).to_numpy()

    masks = {c: as_member(df[c]) for c in req.columns}
    sizes = {c: int(m.sum()) for c, m in masks.items()}
    empty = [c for c, s in sizes.items() if s == 0]
    if len(empty) == len(req.columns):
        raise HTTPException(
            status_code=400,
            detail=(
                "No column has any member. Values are read as truthy when numeric "
                "and non-zero, or one of: " + ", ".join(sorted(truthy))
            ),
        )

    from itertools import combinations

    intersections = []
    n_cols = len(req.columns)
    for size in range(1, n_cols + 1):
        for combo in combinations(req.columns, size):
            mask = np.ones(len(df), dtype=bool)
            for c in req.columns:
                mask &= masks[c] if c in combo else ~masks[c]
            count = int(mask.sum())
            if count:
                intersections.append(
                    {"sets": list(combo), "degree": size, "count": count}
                )
    intersections.sort(key=lambda r: (-r["count"], r["degree"]))

    return {
        "type": "sets",
        "columns": req.columns,
        "set_sizes": sizes,
        "n_rows": int(len(df)),
        "n_in_no_set": int(len(df) - sum(r["count"] for r in intersections)),
        "intersections": intersections,
        "empty_columns": empty,
        # A Venn stays readable to three sets; past that the UpSet form is the
        # honest rendering rather than an unreadable ellipse pile.
        "renderable_as_venn": n_cols <= 3,
    }


class FacetRequest(BaseModel):
    session_id: str
    kind: str                      # boxplot | scatter
    x: Optional[str] = None
    y: Optional[str] = None        # scatter only
    facet: Optional[str] = None    # the column split into panels
    # One panel per VARIABLE instead of per level of a column: the layout of a
    # published multi-panel figure, where QT, QRS and an index sit side by side
    # over the same two groups. Boxplot only.
    variables: Optional[List[str]] = None
    color: Optional[str] = None
    max_panels: int = 12


def _facet_by_variable(df: pd.DataFrame, req: FacetRequest) -> dict:
    """One panel per variable, each split by the same grouping column.

    The axis is deliberately NOT shared here. Sharing it is right when every
    panel shows the same measurement — that is what stops small multiples
    lying — but these panels are different measurements: QT in milliseconds
    next to a unitless index, forced onto one scale, flattens the index into a
    line at the bottom. Each panel gets its own range and the response says so.
    """
    if req.kind != "boxplot":
        raise HTTPException(
            status_code=400,
            detail="A panel per variable is a boxplot layout; scatter panels "
                   "need a facet column.",
        )
    variables = list(dict.fromkeys(req.variables or []))
    if len(variables) < 1:
        raise HTTPException(status_code=400, detail="Name at least one variable.")
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Column(s) not found: {', '.join(missing)}"
        )
    if req.color and req.color not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.color}' not found")
    if req.max_panels < 1:
        raise HTTPException(status_code=400, detail="max_panels must be at least 1")

    dropped_panels = max(0, len(variables) - req.max_panels)
    variables = variables[: req.max_panels]

    names = sorted_groups(df[req.color]) if req.color else ["All"]
    panels: list[dict] = []
    empty_vars: list[str] = []
    for var in variables:
        values = coerce_numeric(df[var]).replace([np.inf, -np.inf], np.nan)
        block = pd.DataFrame({"v": values})
        if req.color:
            block["g"] = df[req.color].values
        block = block.dropna(subset=["v"])
        if block.empty:
            empty_vars.append(var)
            continue
        groups = []
        for gname in names:
            arm = block[block["g"] == gname] if req.color else block
            vals = arm["v"].astype(float).tolist()
            if vals:
                groups.append({"group": str(gname), "values": vals})
        panels.append({
            "panel": var,
            "n": int(len(block)),
            "groups": groups,
            # Per-panel range, since the panels are not on one scale.
            "range": [float(block["v"].min()), float(block["v"].max())],
        })

    if not panels:
        raise HTTPException(
            status_code=400,
            detail="No numeric values remain in any of the named variables.",
        )

    warnings: list[dict] = []
    if dropped_panels:
        warnings.append({
            "type": "panels_truncated",
            "n_dropped": dropped_panels,
            "message": (
                f"{dropped_panels} variable(s) beyond the {req.max_panels}-panel "
                "limit are not drawn."
            ),
        })
    if empty_vars:
        warnings.append({
            "type": "empty_variables",
            "message": (
                "No numeric values in: " + ", ".join(empty_vars) + "."
            ),
        })

    return {
        "type": "facet",
        "kind": "boxplot",
        "facet_by": "variable",
        "x": None,
        "y": None,
        "facet": None,
        "color": req.color,
        "panels": panels,
        # Explicitly empty: the frontend must not fall back to a shared range.
        "shared_range": {},
        "warnings": warnings,
    }


@router.post("/facet")
def facet(req: FacetRequest):
    """Split one plot into a panel per level — ggpubr's facet().

    Panels share the data-driven axis range, computed here across all of them,
    because per-panel autoscaling is what makes small multiples lie: two
    panels look alike while their axes differ by an order of magnitude.
    """
    df = _get_df(req.session_id)
    if req.kind not in {"boxplot", "scatter"}:
        raise HTTPException(
            status_code=400, detail="kind must be boxplot or scatter"
        )
    if req.variables:
        return _facet_by_variable(df, req)
    if not req.x or not req.facet:
        raise HTTPException(
            status_code=400,
            detail="Faceting needs an x column and a facet column, or a list "
                   "of variables to put one per panel.",
        )
    needed = [req.x, req.facet]
    if req.kind == "scatter":
        if not req.y:
            raise HTTPException(status_code=400, detail="scatter facets need a y column")
        needed.append(req.y)
    if req.color:
        needed.append(req.color)
    for col in needed:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    if req.max_panels < 1:
        raise HTTPException(status_code=400, detail="max_panels must be at least 1")

    sub = df[list(dict.fromkeys(needed))].copy()
    numeric_cols = [req.x] if req.kind == "boxplot" else [req.x, req.y]
    for col in numeric_cols:
        sub[col] = coerce_numeric(sub[col]).replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=numeric_cols + [req.facet])
    if sub.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No rows remain after dropping missing values in '{req.facet}'.",
        )

    levels = sorted_groups(sub[req.facet])
    dropped_panels = 0
    if len(levels) > req.max_panels:
        dropped_panels = len(levels) - req.max_panels
        levels = levels[: req.max_panels]

    panels = []
    for lv in levels:
        block = sub[sub[req.facet] == lv]
        if block.empty:
            continue
        if req.kind == "boxplot":
            groups = []
            names = sorted_groups(block[req.color]) if req.color else ["All"]
            for gname in names:
                vals = (
                    block[block[req.color] == gname] if req.color else block
                )[req.x].astype(float).tolist()
                if vals:
                    groups.append({"group": str(gname), "values": vals})
            panels.append({"panel": str(lv), "n": int(len(block)), "groups": groups})
        else:
            panels.append(
                {
                    "panel": str(lv),
                    "n": int(len(block)),
                    "x": block[req.x].astype(float).tolist(),
                    "y": block[req.y].astype(float).tolist(),
                    "color": (
                        block[req.color].astype(str).tolist() if req.color else None
                    ),
                }
            )

    shared: dict = {
        "x": [float(sub[req.x].min()), float(sub[req.x].max())],
    }
    if req.kind == "scatter":
        shared["y"] = [float(sub[req.y].min()), float(sub[req.y].max())]

    warnings: list[dict] = []
    if dropped_panels:
        warnings.append(
            {
                "type": "panels_truncated",
                "n_dropped": dropped_panels,
                "message": (
                    f"'{req.facet}' has {dropped_panels} more levels than the "
                    f"{req.max_panels}-panel limit; those panels are not drawn."
                ),
            }
        )

    return {
        "type": "facet",
        "kind": req.kind,
        "facet_by": "level",
        "x": req.x,
        "y": req.y,
        "facet": req.facet,
        "color": req.color,
        "panels": panels,
        "shared_range": shared,
        "warnings": warnings,
    }


class PieRequest(BaseModel):
    session_id: str
    category: str
    value: Optional[str] = None   # omit to count rows
    sort: str = "value"           # value | category
    max_slices: int = 12          # the rest are folded into "Other"


@router.post("/pie")
def pie(req: PieRequest):
    """Composition of one categorical variable — ggpubr's ggpie / ggdonutchart.

    Percentages are returned alongside counts because a pie without them is
    hard to read past three slices, and a long tail of thin slices is folded
    into a named "Other" rather than drawn as unreadable slivers.
    """
    df = _get_df(req.session_id)
    if req.category not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.category}' not found")
    if req.value and req.value not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.value}' not found")
    if req.sort not in {"value", "category"}:
        raise HTTPException(status_code=400, detail="sort must be value or category")
    if req.max_slices < 2:
        raise HTTPException(status_code=400, detail="max_slices must be at least 2")

    sub = df[[req.category] + ([req.value] if req.value else [])].copy()
    if req.value:
        sub[req.value] = coerce_numeric(sub[req.value]).replace(
            [np.inf, -np.inf], np.nan
        )
        sub = sub.dropna()
        if (sub[req.value] < 0).any():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{req.value}' contains negative values. A pie chart divides a "
                    "whole into parts, which negative quantities cannot do."
                ),
            )
        agg = sub.groupby(req.category, dropna=True)[req.value].sum()
    else:
        sub = sub.dropna()
        agg = sub[req.category].value_counts()

    if agg.empty or float(agg.sum()) <= 0:
        raise HTTPException(
            status_code=400, detail=f"Nothing to plot for '{req.category}'."
        )

    agg = agg.sort_values(ascending=False) if req.sort == "value" else agg.sort_index()
    folded = 0
    if len(agg) > req.max_slices:
        keep = agg.iloc[: req.max_slices - 1]
        folded = int(len(agg) - len(keep))
        other = float(agg.iloc[req.max_slices - 1 :].sum())
        agg = pd.concat([keep, pd.Series({"Other": other})])

    total = float(agg.sum())
    slices = [
        {
            # level_key, so a float64 code arrives as "0" — the key its value
            # labels are stored under. str() gave "0.0" and the pie drew raw
            # codes for a fully labelled column.
            "label": level_key(k),
            "value": float(v),
            "percent": float(v) / total * 100.0,
        }
        for k, v in agg.items()
    ]
    return {
        "type": "pie",
        "category": req.category,
        "value": req.value,
        "measure": "sum" if req.value else "count",
        "total": total,
        "slices": slices,
        "n_folded_into_other": folded,
    }


class BalloonRequest(BaseModel):
    session_id: str
    row: str
    col: str


@router.post("/balloon")
def balloon(req: BalloonRequest):
    """Contingency table as sized, coloured dots — ggpubr's ggballoonplot.

    Size carries the count and colour carries the standardised residual, so a
    cell that is merely large is visually distinct from one that departs from
    independence. Without the residual the plot only restates the marginals.
    """
    df = _get_df(req.session_id)
    for col in (req.row, req.col):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
    if req.row == req.col:
        raise HTTPException(
            status_code=400, detail="Row and column variables must differ."
        )

    sub = df[[req.row, req.col]].dropna()
    if sub.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No rows have both '{req.row}' and '{req.col}' present.",
        )
    ct = pd.crosstab(sub[req.row], sub[req.col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "A balloon plot needs at least two levels on each axis; "
                f"got {ct.shape[0]} x {ct.shape[1]}."
            ),
        )

    observed = ct.to_numpy(dtype=float)
    chi2, p, dof, expected = scipy_stats.chi2_contingency(observed)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.where(expected > 0, (observed - expected) / np.sqrt(expected), 0.0)

    cells = []
    for i, r in enumerate(ct.index):
        for j, c in enumerate(ct.columns):
            cells.append(
                {
                    "row": str(r),
                    "col": str(c),
                    "count": int(observed[i, j]),
                    "expected": float(expected[i, j]),
                    "residual": float(resid[i, j]),
                }
            )

    min_expected = float(expected.min())
    warnings: list[dict] = []
    if min_expected < 5:
        warnings.append(
            {
                "type": "low_expected_count",
                "min_expected": min_expected,
                "message": (
                    f"The smallest expected count is {min_expected:.2f}. The chi-square "
                    "approximation is unreliable below 5; treat the p-value with care."
                ),
            }
        )

    return {
        "type": "balloon",
        "row": req.row,
        "col": req.col,
        "rows": [str(r) for r in ct.index],
        "cols": [str(c) for c in ct.columns],
        "cells": cells,
        "n": int(observed.sum()),
        "chi2": float(chi2),
        "df": int(dof),
        "p": float(p),
        "warnings": warnings,
    }


class SummaryStatsRequest(BaseModel):
    session_id: str
    y: str
    group: Optional[str] = None


@router.post("/summary_stats")
def summary_stats(req: SummaryStatsRequest):
    """Per-group descriptives to print under a plot — ggpubr's ggsummarystats."""
    df = _get_df(req.session_id)
    if req.y not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.y}' not found")
    if req.group and req.group not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.group}' not found")

    cols = [req.y] + ([req.group] if req.group else [])
    sub = df[cols].copy()
    sub[req.y] = coerce_numeric(sub[req.y]).replace([np.inf, -np.inf], np.nan)
    # Missing counts are per group and are reported, not silently dropped: "n"
    # in a figure caption means the rows that contributed to it.
    levels = sorted_groups(sub[req.group]) if req.group else ["All"]
    rows = []
    for name in levels:
        block = sub[sub[req.group] == name] if req.group else sub
        vals = block[req.y].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            continue
        q1, med, q3 = (float(v) for v in np.percentile(vals, [25, 50, 75]))
        rows.append(
            {
                "group": str(name),
                "n": int(len(vals)),
                "n_missing": int(len(block) - len(vals)),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": med,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )
    if not rows:
        raise HTTPException(
            status_code=400, detail=f"No non-missing numeric values in '{req.y}'."
        )
    return {"type": "summary_stats", "y": req.y, "group": req.group, "rows": rows}


class ErrorPlotRequest(BaseModel):
    session_id: str
    y: str
    group: Optional[str] = None
    centre: str = "mean"       # mean | median
    spread: str = "ci"         # sd | se | ci | iqr
    ci_level: float = 0.95


@router.post("/errorplot")
def errorplot(req: ErrorPlotRequest):
    """Centre and spread per group — ggpubr's ggerrorplot.

    Which spread is drawn changes what the figure claims: SD describes the
    sample, SE and CI describe the estimate, and they differ by sqrt(n). The
    choice is returned so the caption can name it instead of leaving the
    reader to guess from the whisker length.
    """
    df = _get_df(req.session_id)
    if req.y not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.y}' not found")
    if req.group and req.group not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.group}' not found")
    if req.centre not in {"mean", "median"}:
        raise HTTPException(status_code=400, detail="centre must be mean or median")
    if req.spread not in {"sd", "se", "ci", "iqr"}:
        raise HTTPException(status_code=400, detail="spread must be sd, se, ci or iqr")
    if not (0 < req.ci_level < 1):
        raise HTTPException(status_code=400, detail="ci_level must be between 0 and 1")
    if req.centre == "median" and req.spread in {"sd", "se", "ci"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "A median with an SD, SE or CI whisker mixes two different summaries. "
                "Pair median with iqr, or mean with sd / se / ci."
            ),
        )
    if req.centre == "mean" and req.spread == "iqr":
        raise HTTPException(
            status_code=400,
            detail="An IQR whisker belongs with the median, not the mean.",
        )

    cols = [req.y] + ([req.group] if req.group else [])
    sub = df[cols].copy()
    sub[req.y] = coerce_numeric(sub[req.y]).replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna()

    levels = sorted_groups(sub[req.group]) if req.group else ["All"]
    rows = []
    for name in levels:
        vals = (
            sub.loc[sub[req.group] == name, req.y] if req.group else sub[req.y]
        ).astype(float).to_numpy()
        if len(vals) == 0:
            continue
        n = int(len(vals))
        if req.centre == "median":
            centre = float(np.median(vals))
            q1, q3 = (float(v) for v in np.percentile(vals, [25, 75]))
            lo, hi = q1, q3
        else:
            centre = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            se = sd / np.sqrt(n) if n > 0 else 0.0
            if req.spread == "sd":
                lo, hi = centre - sd, centre + sd
            elif req.spread == "se":
                lo, hi = centre - se, centre + se
            else:  # ci — t-based, so small groups get the wider interval they deserve
                if n > 1:
                    tcrit = float(scipy_stats.t.ppf(0.5 + req.ci_level / 2, n - 1))
                    lo, hi = centre - tcrit * se, centre + tcrit * se
                else:
                    lo = hi = centre
        rows.append(
            {
                "group": str(name),
                "n": n,
                "centre": centre,
                "lower": float(lo),
                "upper": float(hi),
            }
        )

    if not rows:
        raise HTTPException(
            status_code=400, detail=f"No non-missing numeric values in '{req.y}'."
        )

    spread_label = {
        "sd": "mean ± SD",
        "se": "mean ± SE",
        "ci": f"mean with {int(round(req.ci_level * 100))}% CI",
        "iqr": "median with IQR",
    }[req.spread]
    return {
        "type": "errorplot",
        "y": req.y,
        "group": req.group,
        "centre": req.centre,
        "spread": req.spread,
        "ci_level": req.ci_level,
        "spread_label": spread_label,
        "rows": rows,
    }


class EcdfRequest(BaseModel):
    session_id: str
    x: str
    group: Optional[str] = None


@router.post("/ecdf")
def ecdf(req: EcdfRequest):
    """Empirical cumulative distribution — ggpubr's ggecdf.

    Shows the whole distribution without the binning choice a histogram
    forces, and lets two groups be compared at every quantile at once, which
    is what a Kolmogorov-Smirnov statistic measures.
    """
    df = _get_df(req.session_id)
    if req.x not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.x}' not found")
    if req.group and req.group not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.group}' not found")

    cols = [req.x] + ([req.group] if req.group else [])
    sub = df[cols].copy()
    sub[req.x] = coerce_numeric(sub[req.x]).replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna()
    if len(sub) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 non-missing numeric values in '{req.x}'.",
        )

    levels = sorted_groups(sub[req.group]) if req.group else ["All"]
    curves = []
    arrays: dict[str, np.ndarray] = {}
    for name in levels:
        vals = (
            sub.loc[sub[req.group] == name, req.x] if req.group else sub[req.x]
        ).astype(float).to_numpy()
        if len(vals) == 0:
            continue
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        arrays[str(name)] = xs
        curves.append(
            {
                "group": str(name),
                "n": int(len(xs)),
                "x": [float(v) for v in xs],
                "y": [float(v) for v in ys],
            }
        )

    # With exactly two groups the vertical gap between the curves *is* the KS
    # statistic, so reporting it turns the picture into a test.
    ks: dict = {}
    if len(arrays) == 2:
        a, b = list(arrays.values())
        stat, p = scipy_stats.ks_2samp(a, b)
        ks = {
            "test": "Two-sample Kolmogorov-Smirnov",
            "statistic": float(stat),
            "p": float(p),
            "note": "D is the largest vertical distance between the two curves.",
        }

    return {
        "type": "ecdf",
        "x": req.x,
        "group": req.group,
        "curves": curves,
        "ks": ks,
    }


class DumbbellRequest(BaseModel):
    session_id: str
    category: str          # one row per level — the variable names down the axis
    start: str             # open marker: the reference / expected value
    end: str               # filled marker: the observed / recomputed value
    group: Optional[str] = None   # optional colour band per row
    sort: str = "gap"      # gap | end | start | category


@router.post("/dumbbell")
def dumbbell(req: DumbbellRequest):
    """Paired values per category, drawn as two markers joined by a line.

    The shape answers "how far apart are these two numbers, for each of these
    things, ranked" — an expected value against an observed one, a figure
    implied by a reported statistic against one computed from the raw data. A
    grouped bar chart can carry the same numbers but buries the gap, which is
    the quantity of interest.
    """
    df = _get_df(req.session_id)

    needed = [req.category, req.start, req.end]
    if req.group and req.group not in needed:
        needed.append(req.group)
    for col in needed:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    if req.sort not in {"gap", "end", "start", "category"}:
        raise HTTPException(
            status_code=400,
            detail=f"sort must be one of gap, end, start, category — got '{req.sort}'",
        )

    sub = df[needed].copy()
    for col in (req.start, req.end):
        sub[col] = coerce_numeric(sub[col]).replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=[req.category, req.start, req.end])

    if sub.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No rows have all of '{req.category}', '{req.start}' and '{req.end}' "
                "present and numeric."
            ),
        )

    # One marker pair per category. Collapsing duplicates silently would draw a
    # plot that looks fine and means something else, so name them instead.
    dupes = sub[req.category].astype(str).value_counts()
    repeated = dupes[dupes > 1]
    if len(repeated) > 0:
        shown = ", ".join(str(v) for v in repeated.index[:5])
        more = "" if len(repeated) <= 5 else f" (and {len(repeated) - 5} more)"
        raise HTTPException(
            status_code=400,
            detail=(
                f"A dumbbell chart draws one row per '{req.category}' value, but "
                f"these appear more than once: {shown}{more}. Aggregate the data "
                "first, or pick a column with one row per category."
            ),
        )

    sub["_gap"] = sub[req.end].astype(float) - sub[req.start].astype(float)
    sort_key = {
        "gap": sub["_gap"].abs(),
        "end": sub[req.end].astype(float),
        "start": sub[req.start].astype(float),
        "category": sub[req.category].astype(str),
    }[req.sort]
    sub = sub.assign(_k=sort_key).sort_values(
        "_k", ascending=(req.sort == "category")
    )

    rows = [
        {
            "category": str(r[req.category]),
            "start": float(r[req.start]),
            "end": float(r[req.end]),
            "gap": float(r["_gap"]),
            "group": (str(r[req.group]) if req.group else None),
        }
        for _, r in sub.iterrows()
    ]

    gaps = np.array([r["gap"] for r in rows], dtype=float)
    return {
        "type": "dumbbell",
        "category": req.category,
        "start": req.start,
        "end": req.end,
        "group": req.group,
        "sort": req.sort,
        "rows": rows,
        "summary": {
            "n": len(rows),
            "mean_gap": float(np.mean(gaps)),
            "median_abs_gap": float(np.median(np.abs(gaps))),
            "max_abs_gap": float(np.max(np.abs(gaps))),
            "largest_gap_category": rows[int(np.argmax(np.abs(gaps)))]["category"],
            "n_end_above_start": int((gaps > 0).sum()),
            "n_end_below_start": int((gaps < 0).sum()),
        },
    }


@router.post("/boxplot")
def boxplot(req: ChartRequest):
    df = _get_df(req.session_id)
    if req.color:
        result = []
        for grp, sub in df.groupby(req.color):
            mask = sub[req.x].notna()
            vals = sub.loc[mask, req.x].tolist()
            indices = sub.loc[mask].index.tolist()
            result.append({"group": str(grp), "values": vals, "row_indices": indices})
    else:
        mask = df[req.x].notna()
        vals = df.loc[mask, req.x].tolist()
        indices = df.loc[mask].index.tolist()
        result = [{"group": "All", "values": vals, "row_indices": indices}]
    # The grouping column travels with the data: the client resolves each
    # group's value labels from it, and without the name it was looking them
    # up in an empty map — so a labelled histology column drew its raw codes
    # on every box, violin, raincloud and strip chart.
    return {"type": "boxplot", "x": req.x, "color": req.color, "groups": result}


class PairedBoxRequest(BaseModel):
    session_id: str
    y: str
    group: str
    pair_id: str


@router.post("/paired_box")
def paired_box(req: PairedBoxRequest):
    """Matched-pair box plot: one box per group plus a connector line joining
    each pair's two values (e.g. PSM-matched cohorts, before/after cases).
    Groups are looked up via `pair_id` (typically PSM's `match_set_id` or a
    per-case ID column) rather than row position, so unsorted or filtered
    data still pairs correctly."""
    df = _get_df(req.session_id)
    for col in (req.y, req.group, req.pair_id):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    y = coerce_numeric(df[req.y]).replace([np.inf, -np.inf], np.nan)
    work = pd.DataFrame({
        "y": y,
        "group": df[req.group].astype(str),
        "pair_id": df[req.pair_id].astype(str),
    }, index=df.index)
    work = work.dropna(subset=["y"])

    levels = sorted_groups(df.loc[work.index, req.group])
    levels = [str(lvl) for lvl in levels]
    if len(levels) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Group column '{req.group}' must have exactly 2 levels (found {len(levels)}).",
        )

    groups_out = []
    side: dict[str, pd.Series] = {}
    for lvl in levels:
        sub = work[work["group"] == lvl]
        groups_out.append({
            "group": lvl, "values": sub["y"].tolist(), "row_indices": sub.index.tolist(),
            # Parallel to values/row_indices — lets the frontend seed each
            # point's jitter from the same pair_id used in `pairs`, so a
            # connector line lands exactly on its two marker positions.
            "pair_ids": sub["pair_id"].tolist(),
        })
        # First occurrence per pair_id — duplicates within a side can't be
        # unambiguously paired, so only the first is used as the pair partner.
        side[lvl] = sub.dropna(subset=["pair_id"]).drop_duplicates(subset="pair_id", keep="first").set_index("pair_id")["y"]

    g0, g1 = levels
    common_ids = sorted(set(side[g0].index) & set(side[g1].index))
    pairs = [
        {"pair_id": pid, "y0": float(side[g0][pid]), "y1": float(side[g1][pid])}
        for pid in common_ids
    ]

    return {
        "type": "paired_box",
        "y": req.y,
        "group": req.group,
        "pair_id": req.pair_id,
        "groups": groups_out,
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_unpaired": len(work) - 2 * len(pairs),
    }


class SplomRequest(BaseModel):
    session_id: str
    variables: List[str]
    color: Optional[str] = None


@router.post("/splom")
def splom(req: SplomRequest):
    df = _get_df(req.session_id)

    if len(req.variables) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 variables")

    needed = list(req.variables)
    if req.color and req.color not in needed:
        needed.append(req.color)

    for col in needed:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    sub = df[needed].replace([np.inf, -np.inf], np.nan).dropna()

    if len(sub) < 3:
        raise HTTPException(
            status_code=400,
            detail="Not enough data after removing missing values (need ≥ 3 rows)",
        )

    # Build column arrays
    data_cols = {col: sub[col].tolist() for col in req.variables}
    color_values = sub[req.color].tolist() if req.color else None

    # Pairwise Pearson r matrix
    corr: dict = {}
    for a in req.variables:
        for b in req.variables:
            if a == b:
                corr[f"{a}||{b}"] = 1.0
            else:
                key = f"{a}||{b}"
                try:
                    r, _ = scipy_stats.pearsonr(
                        sub[a].astype(float), sub[b].astype(float)
                    )
                    corr[key] = (
                        round(float(r), 4) if not (np.isnan(r) or np.isinf(r)) else None
                    )
                except Exception:
                    corr[key] = None

    return {
        "variables": req.variables,
        "n": len(sub),
        "data": data_cols,
        "color": req.color,
        "color_values": color_values,
        "corr": corr,
    }


def _bar_series(sub: pd.DataFrame, req: ChartRequest, mode: str) -> list[dict]:
    """One bar per level of req.x, under whichever of the three modes applies."""
    if req.y and mode == "percentage":
        target = req.target_value
        if target is None or str(target).strip() == "":
            hit = coerce_numeric(sub[req.y]).fillna(0) != 0
        else:
            hit = sub[req.y].map(level_key) == level_key(target)
        grouped = sub.assign(_hit=hit).groupby(req.x)["_hit"]
        return [
            {"label": level_key(label), "value": round(int(g.sum()) / int(g.size) * 100, 1) if g.size else 0.0,
             "n": int(g.size), "k": int(g.sum())}
            for label, g in grouped
        ]
    if req.y:
        return _mean_bars(sub, req.x, req.y, req.error)
    # sorted_groups, not value_counts' own order: that is frequency-descending,
    # so tertile 3 came before tertile 1 on the axis, and two series of a
    # grouped chart could order their bars differently from each other.
    counts = sub[req.x].value_counts()
    return [{"label": level_key(k), "value": int(counts[k])} for k in sorted_groups(sub[req.x])]


def _grouped_bar(df: pd.DataFrame, req: ChartRequest, mode: str) -> dict:
    """A bar chart split by a second categorical column.

    The Color / Group selector was offered on this chart and silently ignored:
    the request carried the column and the handler never read it, so choosing
    one changed nothing and said nothing. Splitting is what the control claims
    to do, so it now does it.
    """
    cols = [req.x, req.color] + ([req.y] if req.y else [])
    sub = df[cols].dropna(subset=[req.x, req.color])
    if sub.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No rows with both '{req.x}' and '{req.color}' present.",
        )
    series = []
    for level in sorted_groups(sub[req.color]):
        part = sub[sub[req.color] == level]
        if req.y:
            part = part.dropna(subset=[req.y])
        if part.empty:
            continue
        series.append({"group": level_key(level), "data": _bar_series(part, req, mode)})
    return {
        "type": "bar",
        "x": req.x,
        "y": (f"% {req.y}" + (f" = {req.target_value}" if req.target_value not in (None, "") else ""))
             if (req.y and mode == "percentage") else (req.y or "count"),
        "y_mode": mode if req.y else "count",
        "color": req.color,
        "series": series,
    }


@router.post("/bar")
def bar(req: ChartRequest):
    """Bar chart: count, group mean, or the percentage of each group meeting a
    condition on the y column.

    The percentage mode exists because "what fraction of this tertile was
    malignant" is the question a risk-factor figure asks, and computing it as
    a mean of a 0/1 column gives 0.37 where the figure needs 37% — a rescale
    the caller then has to remember, and label, on their own.
    """
    df = _get_df(req.session_id)
    mode = (req.y_mode or "mean").lower()
    if mode not in {"mean", "percentage"}:
        raise HTTPException(
            status_code=400, detail="y_mode must be mean or percentage"
        )

    if req.color:
        if req.color not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{req.color}' not found")
        return _grouped_bar(df, req, mode)

    if req.y and mode == "percentage":
        if req.y not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{req.y}' not found")
        sub = df[[req.x, req.y]].dropna()
        if sub.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No rows with both '{req.x}' and '{req.y}' present.",
            )
        target = req.target_value
        if target is None or str(target).strip() == "":
            # No target named: treat the column as a 0/1 flag and report the
            # positives. Anything that parses as a non-zero number counts.
            hit = coerce_numeric(sub[req.y]).fillna(0) != 0
        else:
            # Compare on the canonical level string, so "1" matches a float64
            # 1.0 the same way the value-label lookup does.
            hit = sub[req.y].map(level_key) == level_key(target)
        sub = sub.assign(_hit=hit)
        grouped = sub.groupby(req.x)["_hit"]
        data = []
        for label, series in grouped:
            n = int(series.size)
            k = int(series.sum())
            data.append({
                "label": level_key(label),
                "value": round(k / n * 100, 1) if n else 0.0,
                # The counts travel with the percentage: 37% of 8 and 37% of
                # 800 are the same bar and not the same finding.
                "n": n,
                "k": k,
            })
        return {
            "type": "bar",
            "x": req.x,
            "y": f"% {req.y}" + (f" = {target}" if target not in (None, "") else ""),
            "y_mode": "percentage",
            "data": data,
        }

    if req.y:
        grp = df.groupby(req.x)[req.y].mean().reset_index()
        return {
            "type": "bar",
            "x": req.x,
            "y": req.y,
            "y_mode": "mean",
            "error": req.error or None,
            "error_label": _BAR_ERROR_LABEL.get(req.error or "", None),
            "data": _mean_bars(df, req.x, req.y, req.error),
        }
    counts = df[req.x].value_counts()
    return {
        "type": "bar",
        "x": req.x,
        "y": "count",
        "y_mode": "count",
        "data": [{"label": level_key(k), "value": int(counts[k])}
                 for k in sorted_groups(df[req.x].dropna())],
    }


# ── Forest plot ─────────────────────────────────────────────────────────────────


class ForestRow(BaseModel):
    label: str
    est: float
    ci_low: float
    ci_high: float
    weight: Optional[float] = None  # for meta-analysis weighting
    group: Optional[str] = None  # optional group label (sub-heading)
    n: Optional[int] = None  # optional sample-size annotation


class ForestRequest(BaseModel):
    rows: List[ForestRow]
    effect_label: str = "OR"  # OR / HR / RR / β / Mean difference
    x_axis: str = "log"  # "log" for OR/HR/RR, "linear" for β/diff
    null_line: float = 1.0  # reference value (1.0 for log-scale, 0 for linear)
    title: Optional[str] = None
    sort_by: Optional[str] = None  # "effect" | "p" | None (preserve order)
    # Meta-analysis (optional):
    do_meta: bool = False
    meta_method: str = "DL"  # DerSimonian-Laird random-effects


@router.post("/forest")
def forest_plot(req: ForestRequest):
    """Forest plot data + optional DerSimonian-Laird meta-analysis pool.

    Accepts a flat row array of {label, est, ci_low, ci_high, weight?, group?}
    and returns Plotly-ready traces + (when do_meta=True) a pooled diamond
    with I² heterogeneity and τ². Same backend serves two UI hooks:
    univariate-OR screening (from logistic_table) and study-level
    meta-analysis (free-form upload).
    """
    rows = [r.dict() for r in req.rows]
    if not rows:
        raise HTTPException(status_code=422, detail="rows array is empty.")
    if req.sort_by == "effect":
        rows.sort(key=lambda r: r["est"])
    # SE inferred from CI assuming symmetric on the log/linear scale.
    log_scale = req.x_axis == "log"
    for r in rows:
        if log_scale:
            if r["est"] <= 0 or r["ci_low"] <= 0 or r["ci_high"] <= 0:
                raise HTTPException(
                    status_code=422,
                    detail="Log-scale forest plots require positive est, ci_low, and ci_high values.",
                )
            r["log_est"] = float(np.log(max(r["est"], 1e-12)))
            r["log_low"] = float(np.log(max(r["ci_low"], 1e-12)))
            r["log_high"] = float(np.log(max(r["ci_high"], 1e-12)))
            r["se"] = (r["log_high"] - r["log_low"]) / (2 * 1.96)
        else:
            r["se"] = (r["ci_high"] - r["ci_low"]) / (2 * 1.96)

    meta = None
    if req.do_meta and len(rows) >= 2:
        ests = np.array(
            [r["log_est"] if log_scale else r["est"] for r in rows], dtype=float
        )
        ses = np.array([r["se"] for r in rows], dtype=float)
        wts_fe = 1.0 / (ses**2)
        wts_fe = wts_fe / wts_fe.sum()  # normalise
        # Fixed-effect mean
        mu_fe = float(np.sum(wts_fe * ests))
        # Cochran Q and τ² (DerSimonian-Laird)
        q = float(np.sum((1.0 / (ses**2)) * (ests - mu_fe) ** 2))
        dfree = len(rows) - 1
        c = float(
            np.sum(1.0 / (ses**2))
            - np.sum((1.0 / (ses**2)) ** 2) / np.sum(1.0 / (ses**2))
        )
        tau2 = max(0.0, (q - dfree) / c if c > 0 else 0.0)
        # Random-effects re-weighting
        wts_re = 1.0 / (ses**2 + tau2)
        mu_re = float(np.sum(wts_re * ests) / np.sum(wts_re))
        var_re = float(1.0 / np.sum(wts_re))
        se_re = float(np.sqrt(var_re))
        ci_low_re = float(mu_re - 1.96 * se_re)
        ci_high_re = float(mu_re + 1.96 * se_re)
        i2 = max(0.0, (q - dfree) / q * 100.0) if q > 0 else 0.0
        from scipy.stats import chi2 as _chi2

        q_p = float(1 - _chi2.cdf(q, dfree)) if dfree > 0 else 1.0
        if log_scale:
            pooled_est = float(np.exp(mu_re))
            pooled_low = float(np.exp(ci_low_re))
            pooled_high = float(np.exp(ci_high_re))
        else:
            pooled_est = float(mu_re)
            pooled_low = float(ci_low_re)
            pooled_high = float(ci_high_re)
        meta = {
            "method": req.meta_method,
            "pooled_est": round(pooled_est, 4),
            "pooled_ci_low": round(pooled_low, 4),
            "pooled_ci_high": round(pooled_high, 4),
            "tau2": round(tau2, 6),
            "Q": round(q, 4),
            "Q_df": dfree,
            "Q_p": round(q_p, 4),
            "I_squared_pct": round(i2, 2),
            "k_studies": len(rows),
            "result_text": (
                f"DerSimonian-Laird random-effects meta-analysis (k = {len(rows)} studies). "
                f"Pooled {req.effect_label} = {pooled_est:.3f} (95% CI {pooled_low:.3f}–{pooled_high:.3f}). "
                f"Heterogeneity Q({dfree}) = {q:.2f}, p = {q_p:.4f}, I² = {i2:.1f}%, τ² = {tau2:.4f}."
            ),
        }

    return {
        "type": "forest",
        "effect_label": req.effect_label,
        "x_axis": req.x_axis,
        "null_line": req.null_line,
        "title": req.title,
        "rows": rows,
        "meta": meta,
    }


class SubgroupBarRequest(BaseModel):
    session_id: str
    y_col: str
    subgroup_col: str
    xaxis_col: str
    color_col: Optional[str] = None
    y_mode: str = "mean"  # "mean" or "percentage"
    target_value: Optional[str] = None
    error_type: str = "ci"  # "ci", "se", "sd", "none"


@router.post("/subgroup_bar")
def subgroup_bar(req: SubgroupBarRequest):
    df = _get_df(req.session_id)

    # Check if selected columns exist
    for col in [req.y_col, req.subgroup_col, req.xaxis_col]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    if req.color_col and req.color_col not in df.columns:
        raise HTTPException(
            status_code=400, detail=f"Column '{req.color_col}' not found"
        )

    # Get subset of columns
    cols_to_use = [req.y_col, req.subgroup_col, req.xaxis_col]
    if req.color_col:
        cols_to_use.append(req.color_col)

    sub = df[cols_to_use].copy()

    # Drop missing in grouping columns, but handle Y missing per cell safely
    sub = sub.dropna(
        subset=[req.subgroup_col, req.xaxis_col]
        + ([req.color_col] if req.color_col else [])
    )

    if len(sub) == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid data points found after dropping missing values in grouping variables.",
        )

    # Get unique groups, ordered by value code (numeric when coercible, else
    # string) so multi-digit codes (1, 2, 10) don't sort as 1, 10, 2.
    subgroups = sorted_groups(sub[req.subgroup_col])
    x_vals = sorted_groups(sub[req.xaxis_col])
    color_groups = sorted_groups(sub[req.color_col]) if req.color_col else x_vals
    warnings = []
    if req.y_mode == "mean":
        max_plausible = plausibility_max_for_column(req.y_col)
        raw_y = sub[req.y_col]
        masked_y = mask_sentinels(raw_y, max_plausible)
        if masked_y.isna().sum() > coerce_numeric(raw_y).isna().sum():
            warnings.append(
                {
                    "variable": req.y_col,
                    "note": "Implausible high sentinel values were treated as missing for mean bars.",
                }
            )
        sub[req.y_col] = masked_y

    # ── Percentage "success" level — resolved ONCE over the whole subset, not
    # per cell. Picking it per cell (the old behaviour) let different bars
    # measure different levels, so the chart was not comparable.
    pct_target = None
    if req.y_mode == "percentage":
        pct_target = req.target_value
        if pct_target is None:
            levels = sorted(str(v) for v in sub[req.y_col].dropna().unique())
            if "1" in levels:
                pct_target = "1"
            elif "1.0" in levels:
                pct_target = "1.0"
            elif levels:
                pct_target = levels[-1]  # deterministic fallback
            else:
                pct_target = "1"

    Z = 1.959963984540054  # 95% normal quantile

    def _wilson_pct(successes: int, n: int) -> tuple:
        """Wilson score interval (×100). Returns (point%, low%, high%)."""
        if n == 0:
            return 0.0, 0.0, 0.0
        p = successes / n
        denom = 1.0 + Z * Z / n
        center = (p + Z * Z / (2 * n)) / denom
        half = (Z / denom) * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
        return (
            p * 100.0,
            max(0.0, center - half) * 100.0,
            min(1.0, center + half) * 100.0,
        )

    traces = []
    for cg in color_groups:
        tr = {
            "name": str(cg),
            "x_subgroup": [],
            "x_xaxis": [],
            "y": [],
            "error": [],
            "error_low": [],
            "error_high": [],
            "ns": [],
        }
        for sg in subgroups:
            iter_x_vals = x_vals if req.color_col else [cg]
            for xv in iter_x_vals:
                mask = (sub[req.subgroup_col] == sg) & (sub[req.xaxis_col] == xv)
                if req.color_col:
                    mask = mask & (sub[req.color_col] == cg)
                cell = sub.loc[mask, req.y_col].dropna()
                n = int(len(cell))

                val, e_low, e_high = 0.0, 0.0, 0.0
                if n > 0 and req.y_mode == "percentage":
                    successes = int((cell.astype(str) == str(pct_target)).sum())
                    p = successes / n
                    val, lo, hi = _wilson_pct(successes, n)
                    se = np.sqrt(p * (1 - p) / n) * 100.0
                    sd = np.sqrt(p * (1 - p)) * 100.0
                    if req.error_type == "ci":
                        e_low, e_high = (
                            max(0.0, val - lo),
                            max(0.0, hi - val),
                        )  # asymmetric (Wilson)
                    elif req.error_type == "se":
                        e_low = e_high = se
                    elif req.error_type == "sd":
                        e_low = e_high = sd
                elif n > 0:
                    nums = pd.to_numeric(cell, errors="coerce").dropna()
                    m = int(len(nums))
                    if m > 0:
                        val = float(nums.mean())
                        sd = float(nums.std(ddof=1)) if m > 1 else 0.0
                        se = sd / np.sqrt(m)
                        if req.error_type == "ci":
                            tcrit = (
                                float(scipy_stats.t.ppf(0.975, m - 1)) if m > 1 else 0.0
                            )
                            e_low = e_high = tcrit * se  # t-distribution CI half-width
                        elif req.error_type == "se":
                            e_low = e_high = se
                        elif req.error_type == "sd":
                            e_low = e_high = sd

                tr["x_subgroup"].append(str(sg))
                tr["x_xaxis"].append(str(xv))
                tr["y"].append(val)
                tr["error"].append(e_high)  # legacy symmetric field (= upper offset)
                tr["error_low"].append(e_low)
                tr["error_high"].append(e_high)
                tr["ns"].append(n)
        traces.append(tr)

    _err_label = {
        "ci": "95% CI",
        "se": "± 1 SE",
        "sd": "± 1 SD",
        "none": "no error bars",
    }.get(req.error_type, req.error_type)
    return {
        "type": "subgroup_bar",
        "y_col": req.y_col,
        "subgroup_col": req.subgroup_col,
        "xaxis_col": req.xaxis_col,
        "color_col": req.color_col,
        "y_mode": req.y_mode,
        "target_value": pct_target if req.y_mode == "percentage" else req.target_value,
        "error_type": req.error_type,
        "traces": traces,
        "warnings": warnings,
        "method_note": (
            "Means use a t-distribution CI (t_{n−1}); percentages use the Wilson "
            "score interval (bounded to 0–100%, accurate for small n and extreme "
            f"proportions). Error bars show {_err_label}."
        ),
    }


class ScoreFigureSpec(BaseModel):
    score_col: str
    label: Optional[str] = None
    components: List[str]
    component_labels: Optional[Dict[str, str]] = None


class ScoreCompositeRequest(BaseModel):
    session_id: str
    group_col: str
    scores: List[ScoreFigureSpec]
    group_order: Optional[List[str]] = None
    bins: int = 8
    title: Optional[str] = None
    positive_values: List[str] = ["1", "true", "yes", "y", "present", "positive"]


def _format_p_value(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "p = NA"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _score_group_pvalue(
    score: pd.Series, groups: pd.Series, group_levels: list[str]
) -> Optional[float]:
    samples = [
        pd.to_numeric(score[groups == g], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        for g in group_levels
    ]
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2:
        return None
    try:
        if len(samples) == 2:
            return float(
                scipy_stats.mannwhitneyu(
                    samples[0], samples[1], alternative="two-sided"
                ).pvalue
            )
        return float(scipy_stats.kruskal(*samples).pvalue)
    except Exception:
        return None


def _component_positive(series: pd.Series, positive_values: list[str]) -> pd.Series:
    non_missing = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_positive = numeric.notna() & (numeric > 0)
    normalized = series.astype(str).str.strip().str.lower()
    string_positive = normalized.isin({str(v).strip().lower() for v in positive_values})
    return non_missing & (numeric_positive | string_positive)


def _component_pvalue(
    values: pd.Series,
    groups: pd.Series,
    group_levels: list[str],
    positive_values: list[str],
) -> tuple[Optional[float], Optional[str]]:
    valid = values.notna() & groups.notna()
    if not valid.any():
        return None, None
    pos = _component_positive(values[valid], positive_values)
    valid_groups = groups[valid]
    table = []
    for g in group_levels:
        mask = valid_groups == g
        positives = int(pos[mask].sum())
        total = int(mask.sum())
        table.append([positives, max(total - positives, 0)])
    if len(table) < 2 or any(sum(row) == 0 for row in table):
        return None, None
    try:
        p_value, test_name = _categorical_p_with_rule(np.array(table))
        return float(p_value), test_name
    except Exception:
        return None, None


def _score_xbins(values: pd.Series, bins: int) -> dict:
    clean = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if clean.empty:
        return {"nbinsx": max(2, int(bins))}
    arr = clean.to_numpy(dtype=float)
    is_integer = np.all(np.isclose(arr, np.round(arr)))
    if is_integer and len(np.unique(arr)) <= 50:
        return {
            "xbins": {
                "start": float(np.floor(arr.min()) - 0.5),
                "end": float(np.ceil(arr.max()) + 0.5),
                "size": 1,
            }
        }
    return {"nbinsx": max(2, int(bins))}


@router.post("/score_composite")
def score_composite(req: ScoreCompositeRequest):
    """Build a manuscript-style score distribution + component prevalence figure.

    The endpoint is intentionally generic: any two score columns can be compared
    across a grouping column, with each score paired to its own binary component
    columns. It returns both computed summaries and a Plotly-ready 5-panel
    figure matching the common clinical manuscript layout:
    score histograms, score overlap boxplots, and grouped component prevalence
    bars.
    """
    df = _get_df(req.session_id)
    if req.group_col not in df.columns:
        raise HTTPException(
            status_code=400, detail=f"Column '{req.group_col}' not found"
        )
    if len(req.scores) != 2:
        raise HTTPException(
            status_code=400,
            detail="Select exactly two score columns for a 5-panel score-composite figure.",
        )

    needed = [req.group_col]
    for spec in req.scores:
        if spec.score_col not in df.columns:
            raise HTTPException(
                status_code=400, detail=f"Column '{spec.score_col}' not found"
            )
        if not spec.components:
            raise HTTPException(
                status_code=400,
                detail=f"Select at least one component for '{spec.score_col}'.",
            )
        needed.append(spec.score_col)
        for comp in spec.components:
            if comp not in df.columns:
                raise HTTPException(
                    status_code=400, detail=f"Column '{comp}' not found"
                )
            needed.append(comp)

    sub = df[list(dict.fromkeys(needed))].copy()
    sub["_score_group"] = sub[req.group_col].astype(str)
    if req.group_order:
        group_levels = [
            str(g)
            for g in req.group_order
            if str(g) in set(sub["_score_group"].dropna())
        ]
    else:
        group_levels = [str(g) for g in pd.unique(sub["_score_group"].dropna())]
    if len(group_levels) < 2:
        raise HTTPException(
            status_code=400, detail="Need at least two non-missing groups."
        )

    colors = ["#4f86c6", "#dd7b6e", "#6fbf73", "#9b6ec8", "#d19a2e"]
    axis_ids = [
        ("x", "y"),
        ("x2", "y2"),
        ("x3", "y3"),
        ("x4", "y4"),
        ("x5", "y5"),
    ]
    trace_data: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    shapes: list[dict[str, Any]] = []
    score_summaries: list[dict[str, Any]] = []
    component_ticks: list[dict[str, list[Any]]] = []

    panel_titles = [
        f"A  {req.scores[0].label or req.scores[0].score_col} Score Distribution",
        f"B  {req.scores[1].label or req.scores[1].score_col} Score Distribution",
        "C  Score Overlap (Box Plots)",
        f"D  {req.scores[0].label or req.scores[0].score_col} Component Prevalence",
        f"E  {req.scores[1].label or req.scores[1].score_col} Component Prevalence",
    ]
    title_positions = [
        (0.0, 1.06),
        (0.37, 1.06),
        (0.74, 1.06),
        (0.0, 0.47),
        (0.74, 0.47),
    ]
    for text, (xpos, ypos) in zip(panel_titles, title_positions):
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": xpos,
                "y": ypos,
                "text": f"<b>{text}</b>",
                "showarrow": False,
                "xanchor": "left",
                "font": {"size": 14, "color": "#111827"},
            }
        )

    for score_idx, spec in enumerate(req.scores):
        label = spec.label or spec.score_col
        score_values = pd.to_numeric(sub[spec.score_col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        p_value = _score_group_pvalue(score_values, sub["_score_group"], group_levels)
        n_by_group = {
            g: int(score_values[sub["_score_group"] == g].dropna().shape[0])
            for g in group_levels
        }
        score_summaries.append(
            {
                "score_col": spec.score_col,
                "label": label,
                "p_value": p_value,
                "p_text": _format_p_value(p_value),
                "n_by_group": n_by_group,
                "components": [],
            }
        )

        xref, yref = axis_ids[score_idx]
        bin_kwargs = _score_xbins(score_values, req.bins)
        for group_idx, group in enumerate(group_levels):
            vals = (
                score_values[sub["_score_group"] == group]
                .dropna()
                .astype(float)
                .tolist()
            )
            trace = {
                "type": "histogram",
                "x": vals,
                "name": f"{group} (n={len(vals)})",
                "marker": {
                    "color": colors[group_idx % len(colors)],
                    "line": {"color": "white", "width": 0.5},
                },
                "opacity": 0.78,
                "showlegend": score_idx == 0,
                "legendgroup": group,
                "xaxis": xref,
                "yaxis": yref,
            }
            trace.update(bin_kwargs)
            trace_data.append(trace)

            box_x = [f"{group}<br>{label}"] * len(vals)
            trace_data.append(
                {
                    "type": "box",
                    "x": box_x,
                    "y": vals,
                    "name": group,
                    "marker": {"color": colors[group_idx % len(colors)]},
                    "line": {"color": "#111827", "width": 1},
                    "boxpoints": "outliers" if len(vals) < 500 else False,
                    "showlegend": False,
                    "legendgroup": group,
                    "xaxis": "x3",
                    "yaxis": "y3",
                }
            )

        annotations.append(
            {
                "xref": f"{xref} domain",
                "yref": f"{yref} domain",
                "x": 0.86,
                "y": 0.9,
                "text": _format_p_value(p_value),
                "showarrow": False,
                "font": {"size": 12, "color": "#4b5563"},
            }
        )

    box_max_candidates = [
        float(
            pd.to_numeric(sub[spec.score_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .max()
        )
        for spec in req.scores
    ]
    box_max_candidates = [v for v in box_max_candidates if np.isfinite(v)]
    box_y_max = max(box_max_candidates or [1.0])
    for score_idx, spec in enumerate(req.scores):
        label = spec.label or spec.score_col
        x0 = f"{group_levels[0]}<br>{label}"
        x1 = f"{group_levels[-1]}<br>{label}"
        y = box_y_max * (0.94 if score_idx == 0 else 0.88)
        shapes.append(
            {
                "type": "line",
                "xref": "x3",
                "yref": "y3",
                "x0": x0,
                "x1": x1,
                "y0": y,
                "y1": y,
                "line": {"color": "#9ca3af", "width": 1},
            }
        )
        annotations.append(
            {
                "xref": "x3",
                "yref": "y3",
                "x": x0 if score_idx == 0 else x1,
                "y": y + max(box_y_max * 0.04, 0.5),
                "text": score_summaries[score_idx]["p_text"],
                "showarrow": False,
                "font": {"size": 12, "color": "#4b5563"},
            }
        )

    for score_idx, spec in enumerate(req.scores):
        xref, yref = axis_ids[3 + score_idx]
        labels = [
            spec.component_labels.get(c, c) if spec.component_labels else c
            for c in spec.components
        ]
        component_ticks.append(
            {"tickvals": list(range(len(labels))), "ticktext": labels}
        )
        component_rows = []
        bar_width = 0.8 / max(len(group_levels), 1)
        for group_idx, group in enumerate(group_levels):
            y_values = []
            text_values = []
            for comp in spec.components:
                mask = sub["_score_group"] == group
                raw = sub.loc[mask, comp]
                valid = raw.notna()
                n = int(valid.sum())
                positives = (
                    int(_component_positive(raw[valid], req.positive_values).sum())
                    if n
                    else 0
                )
                prevalence = (positives / n * 100.0) if n else 0.0
                y_values.append(prevalence)
                text_values.append(f"{positives}/{n}")
            x_positions = [
                comp_idx - 0.4 + (group_idx + 0.5) * bar_width
                for comp_idx in range(len(labels))
            ]
            trace_data.append(
                {
                    "type": "bar",
                    "x": x_positions,
                    "y": y_values,
                    "width": [bar_width * 0.92] * len(labels),
                    "name": group,
                    "text": text_values,
                    "textposition": "none",
                    "customdata": labels,
                    "hovertemplate": "%{customdata}<br>%{fullData.name}: %{y:.1f}% (%{text})<extra></extra>",
                    "marker": {"color": colors[group_idx % len(colors)]},
                    "showlegend": score_idx == 1,
                    "legendgroup": group,
                    "xaxis": xref,
                    "yaxis": yref,
                }
            )

        for comp_idx, (comp, label) in enumerate(zip(spec.components, labels)):
            p_value, test_name = _component_pvalue(
                sub[comp], sub["_score_group"], group_levels, req.positive_values
            )
            component_rows.append(
                {
                    "component": comp,
                    "label": label,
                    "p_value": p_value,
                    "p_text": _format_p_value(p_value),
                    "test": test_name,
                }
            )
            annotations.append(
                {
                    "xref": xref,
                    "yref": yref,
                    "x": comp_idx,
                    "y": 103,
                    "text": "ns"
                    if p_value is not None and p_value >= 0.05
                    else _format_p_value(p_value),
                    "showarrow": False,
                    "font": {"size": 11, "color": "#9ca3af"},
                }
            )
        score_summaries[score_idx]["components"] = component_rows

    layout = {
        "title": {
            "text": req.title
            or "Score Distributions and Component Prevalence by Group",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": "#111827"},
        },
        "height": 760,
        "barmode": "overlay",
        "boxmode": "group",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "Arial, sans-serif", "size": 12, "color": "#111827"},
        "margin": {"l": 70, "r": 30, "t": 90, "b": 90},
        "legend": {
            "orientation": "v",
            "x": 0.98,
            "y": 0.98,
            "xanchor": "right",
            "yanchor": "top",
        },
        "annotations": annotations,
        "shapes": shapes,
        "xaxis": {
            "domain": [0.0, 0.29],
            "anchor": "y",
            "title": {"text": req.scores[0].label or req.scores[0].score_col},
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        "yaxis": {
            "domain": [0.57, 1.0],
            "anchor": "x",
            "title": {"text": "Number of Patients"},
            "gridcolor": "#e5e7eb",
            "rangemode": "tozero",
        },
        "xaxis2": {
            "domain": [0.36, 0.65],
            "anchor": "y2",
            "title": {"text": req.scores[1].label or req.scores[1].score_col},
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        "yaxis2": {
            "domain": [0.57, 1.0],
            "anchor": "x2",
            "title": {"text": "Number of Patients"},
            "gridcolor": "#e5e7eb",
            "rangemode": "tozero",
        },
        "xaxis3": {
            "domain": [0.72, 1.0],
            "anchor": "y3",
            "tickangle": 0,
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        "yaxis3": {
            "domain": [0.57, 1.0],
            "anchor": "x3",
            "title": {"text": "Score Value"},
            "gridcolor": "#e5e7eb",
            "rangemode": "tozero",
        },
        "xaxis4": {
            "domain": [0.0, 0.68],
            "anchor": "y4",
            "tickangle": 0,
            "gridcolor": "#e5e7eb",
            "zeroline": False,
            **component_ticks[0],
        },
        "yaxis4": {
            "domain": [0.0, 0.42],
            "anchor": "x4",
            "title": {"text": "Prevalence (%)"},
            "range": [0, 108],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        "xaxis5": {
            "domain": [0.76, 1.0],
            "anchor": "y5",
            "tickangle": 0,
            "gridcolor": "#e5e7eb",
            "zeroline": False,
            **component_ticks[1],
        },
        "yaxis5": {
            "domain": [0.0, 0.42],
            "anchor": "x5",
            "title": {"text": "Prevalence (%)"},
            "range": [0, 108],
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
    }

    return {
        "type": "score_composite",
        "group_col": req.group_col,
        "groups": group_levels,
        "scores": score_summaries,
        "figure": {"data": trace_data, "layout": layout},
        "method_note": (
            "Score comparisons use Mann-Whitney U for two groups or Kruskal-Wallis for more groups. "
            "Component prevalence uses chi-square when all expected cells are ≥ 5; otherwise Fisher exact "
            "for 2×2 tables or Fisher-Freeman-Halton Monte Carlo for larger r×c tables."
        ),
    }


# ── Kaplan-Meier composite (NEJM-style multi-endpoint cumulative incidence) ──


class KMEndpointSpec(BaseModel):
    duration_col: str
    event_col: str
    label: Optional[str] = None


class KMCompositeRequest(BaseModel):
    session_id: str
    group_col: str  # treatment arm / comparison column
    endpoints: List[KMEndpointSpec]  # 1-4 endpoints, one panel each
    risk_times: Optional[List[float]] = None  # x-axis ticks for No.-at-risk
    group_order: Optional[List[str]] = None
    # 1 - S(t) climbing from 0 (event accrual) vs S(t) falling from 1.
    as_cumulative_incidence: bool = True
    inset: bool = True  # magnified zoom sub-panel per endpoint
    inset_max_pct: Optional[float] = None  # inset y-max in %; None = auto
    as_percent: bool = True  # y in % (0-100) vs proportion (0-1)
    imputation: str = "listwise"
    title: Optional[str] = None


_KM_COMPOSITE_COLORS = ["#9ca3af", "#1f6f8b", "#dd7b6e", "#6fbf73", "#9b6ec8"]
_PANEL_LETTERS = ["A", "B", "C", "D"]


def _km_composite_grid(n_panels: int) -> tuple[int, int]:
    """Column/row count for the panel grid (max 2 columns, NEJM-style)."""
    ncols = 1 if n_panels == 1 else 2
    nrows = (n_panels + ncols - 1) // ncols
    return ncols, nrows


@router.post("/km_composite")
def km_composite(req: KMCompositeRequest):
    """NEJM-style composite Kaplan-Meier figure.

    Runs one KM analysis per endpoint against a shared grouping (arm) column
    and lays the panels out in a 2-column grid. Each panel plots cumulative
    incidence (1 - S(t)) as step curves, optionally with a magnified zoom
    inset and a per-panel No.-at-risk table, matching the common trial
    primary-endpoint figure. Curves and log-rank p-values are computed with
    the same lifelines helpers as the main survival panel.
    """
    from routers.models.cox import (
        _km_fit_groups,
        _km_logrank,
        _drop_invalid_survival_rows,
    )
    from services.impute import apply_imputation

    df_full = _get_df(req.session_id)
    if req.group_col not in df_full.columns:
        raise HTTPException(
            status_code=400, detail=f"Column '{req.group_col}' not found"
        )
    if not (1 <= len(req.endpoints) <= 4):
        raise HTTPException(status_code=400, detail="Select between 1 and 4 endpoints.")
    for spec in req.endpoints:
        for col in (spec.duration_col, spec.event_col):
            if col not in df_full.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")

    scale = 100.0 if req.as_percent else 1.0
    ncols, nrows = _km_composite_grid(len(req.endpoints))

    # Grid geometry (paper fractions). Rows leave a gap beneath each panel for
    # the No.-at-risk rows; columns leave a left gutter for the y-axis title.
    left_pad, right_pad, col_gap = 0.07, 0.02, 0.09
    top_pad, bottom_pad, row_gap = 0.05, 0.16, 0.19
    # Vertical offsets for the No.-at-risk block below each panel. `risk_gap`
    # clears the axis tick labels; `risk_row_h` is the per-arm row pitch.
    risk_gap, risk_row_h = 0.055, 0.028
    usable_w = 1.0 - left_pad - right_pad - (ncols - 1) * col_gap
    cell_w = usable_w / ncols
    usable_h = 1.0 - top_pad - bottom_pad - (nrows - 1) * row_gap
    cell_h = usable_h / nrows

    colors = _KM_COMPOSITE_COLORS
    trace_data: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    layout_axes: dict[str, Any] = {}
    endpoint_summaries: list[dict[str, Any]] = []

    # Establish a stable arm order once (from the first endpoint's clean data),
    # so colors/legend/at-risk rows line up across every panel.
    group_levels: Optional[list[str]] = None

    for idx, spec in enumerate(req.endpoints):
        label = spec.label or spec.event_col
        cols = [spec.duration_col, spec.event_col, req.group_col]
        sub = df_full[cols].copy()
        sub[spec.duration_col] = pd.to_numeric(sub[spec.duration_col], errors="coerce")
        sub[spec.event_col] = pd.to_numeric(sub[spec.event_col], errors="coerce")
        sub = apply_imputation(sub, [spec.duration_col, spec.event_col], req.imputation)
        sub, _w, _ni = _drop_invalid_survival_rows(
            sub, spec.duration_col, spec.event_col
        )
        sub = sub.dropna(subset=[req.group_col])
        if sub.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No valid rows for endpoint '{label}' after cleaning duration/event.",
            )
        ev_vals = sorted(
            pd.to_numeric(sub[spec.event_col], errors="coerce").dropna().unique()
        )
        if set(ev_vals) - {0, 1, 0.0, 1.0}:
            raise HTTPException(
                status_code=422,
                detail=f"Event column '{spec.event_col}' must be binary 0/1. Found: {ev_vals[:8]}",
            )

        groups = _km_fit_groups(
            sub,
            spec.duration_col,
            spec.event_col,
            req.group_col,
            survival_times=None,
            risk_times=req.risk_times,
            include_censors=False,
        )
        present = [g["group"] for g in groups]
        if group_levels is None:
            if req.group_order:
                ordered = [str(g) for g in req.group_order if str(g) in set(present)]
                group_levels = ordered or present
            else:
                group_levels = present
            if len(group_levels) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Need at least two non-missing groups in the arm column.",
                )
        by_group = {g["group"]: g for g in groups}

        logrank = _km_logrank(sub, spec.duration_col, spec.event_col, req.group_col)
        p_value = logrank.get("p") if logrank else None

        main_ref = idx + 1
        inset_ref = idx + 5
        main_x = "x" if main_ref == 1 else f"x{main_ref}"
        main_y = "y" if main_ref == 1 else f"y{main_ref}"
        inset_x = f"x{inset_ref}"
        inset_y = f"y{inset_ref}"

        r, c = divmod(idx, ncols)
        x0 = left_pad + c * (cell_w + col_gap)
        x1 = x0 + cell_w
        y1 = 1.0 - top_pad - r * (cell_h + row_gap)
        y0 = y1 - cell_h

        # Curve max time (shared x range for at-risk positioning).
        tmax = 0.0
        curves: dict[str, dict[str, list[float]]] = {}
        for grp in group_levels:
            g = by_group.get(grp)
            pts = g["curve"] if g else []
            xs = [p["time"] for p in pts if p["time"] is not None]
            ys = [
                (1.0 - p["survival"]) * scale
                if req.as_cumulative_incidence
                else p["survival"] * scale
                for p in pts
                if p["survival"] is not None
            ]
            curves[grp] = {"x": xs, "y": ys}
            if xs:
                tmax = max(tmax, max(xs))
        tmax = tmax or 1.0

        final_by_group: dict[str, float] = {}
        n_by_group: dict[str, int] = {}
        inset_peak = 0.0
        for gi, grp in enumerate(group_levels):
            xs = curves[grp]["x"]
            ys = curves[grp]["y"]
            g = by_group.get(grp)
            n_by_group[grp] = int(g["n"]) if g else 0
            final_by_group[grp] = round(ys[-1], 1) if ys else 0.0
            inset_peak = max(inset_peak, ys[-1] if ys else 0.0)
            color = colors[gi % len(colors)]
            trace_data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": xs,
                    "y": ys,
                    "line": {"color": color, "width": 2, "shape": "hv"},
                    "name": str(grp),
                    "legendgroup": str(grp),
                    "showlegend": idx == 0,
                    "xaxis": main_x,
                    "yaxis": main_y,
                    "hovertemplate": f"{grp}<br>%{{x}}: %{{y:.1f}}<extra></extra>",
                }
            )
            if req.inset:
                trace_data.append(
                    {
                        "type": "scatter",
                        "mode": "lines",
                        "x": xs,
                        "y": ys,
                        "line": {"color": color, "width": 1.6, "shape": "hv"},
                        "name": str(grp),
                        "legendgroup": str(grp),
                        "showlegend": False,
                        "xaxis": inset_x,
                        "yaxis": inset_y,
                        "hoverinfo": "skip",
                    }
                )

        # Full-range y for the main panel.
        y_full = 100.0 if req.as_percent else 1.0
        # Inset y-max: explicit, else a little above the peak accrual.
        if req.inset:
            if req.inset_max_pct is not None:
                inset_top = req.inset_max_pct / (100.0 / scale)
            else:
                inset_top = max(inset_peak * 1.25, scale * 0.02)

        # Main axes.
        main_x_key = "xaxis" if main_ref == 1 else f"xaxis{main_ref}"
        main_y_key = "yaxis" if main_ref == 1 else f"yaxis{main_ref}"
        is_bottom_row = r == nrows - 1
        risk_times = req.risk_times or []
        # The built-in axis title would land on the No.-at-risk numbers, so when
        # a risk table is drawn we suppress it and add our own annotation below
        # the table instead.
        show_axis_title = is_bottom_row and not risk_times
        layout_axes[main_x_key] = {
            "domain": [x0, x1],
            "anchor": main_y,
            "range": [0, tmax],
            "title": {"text": "Months since Randomization"}
            if show_axis_title
            else None,
            "gridcolor": "#eef1f4",
            "zeroline": False,
        }
        layout_axes[main_y_key] = {
            "domain": [y0, y1],
            "anchor": main_x,
            "range": [0, y_full],
            "title": {"text": "Percentage of Patients"} if c == 0 else None,
            "gridcolor": "#eef1f4",
            "zeroline": False,
        }
        if req.inset:
            ix0 = x0 + 0.34 * cell_w
            ix1 = x1 - 0.02 * cell_w
            iy0 = y0 + 0.40 * cell_h
            iy1 = y1 - 0.02 * cell_h
            layout_axes[f"xaxis{inset_ref}"] = {
                "domain": [ix0, ix1],
                "anchor": inset_y,
                "range": [0, tmax],
                "showgrid": False,
                "zeroline": False,
                "tickfont": {"size": 8},
                "ticklen": 2,
            }
            layout_axes[f"yaxis{inset_ref}"] = {
                "domain": [iy0, iy1],
                "anchor": inset_x,
                "range": [0, inset_top],
                "showgrid": False,
                "zeroline": False,
                "tickfont": {"size": 8},
                "ticklen": 2,
            }
            # Final cumulative-incidence value label at the end of each inset curve.
            for gi, grp in enumerate(group_levels):
                if curves[grp]["y"]:
                    annotations.append(
                        {
                            "xref": inset_x,
                            "yref": inset_y,
                            "x": tmax,
                            "y": curves[grp]["y"][-1],
                            "text": f"{final_by_group[grp]:.1f}",
                            "showarrow": False,
                            "xanchor": "left",
                            "xshift": 2,
                            "font": {"size": 9, "color": colors[gi % len(colors)]},
                        }
                    )

        # Panel letter + endpoint title.
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": x0,
                "y": min(y1 + 0.035, 1.0),
                "text": f"<b>{_PANEL_LETTERS[idx]}</b>  {label}",
                "showarrow": False,
                "xanchor": "left",
                "font": {"size": 12, "color": "#111827"},
            }
        )
        # Log-rank p on the main panel.
        annotations.append(
            {
                "xref": main_x,
                "yref": main_y,
                "x": tmax * 0.5,
                "y": y_full * 0.9,
                "text": _format_p_value(p_value),
                "showarrow": False,
                "font": {"size": 11, "color": "#4b5563"},
            }
        )

        # No.-at-risk rows beneath the panel.
        if risk_times:
            annotations.append(
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": x0,
                    "y": y0 - risk_gap,
                    "text": "<b>No. at Risk</b>",
                    "showarrow": False,
                    "xanchor": "left",
                    "font": {"size": 9, "color": "#374151"},
                }
            )
            for gi, grp in enumerate(group_levels):
                g = by_group.get(grp)
                at_risk = g.get("at_risk") if g else None
                row_y = y0 - risk_gap - (gi + 1) * risk_row_h
                # Arm name in the left gutter so it never collides with the
                # t=0 count that sits at the panel's left edge.
                annotations.append(
                    {
                        "xref": "paper",
                        "yref": "paper",
                        "x": max(x0 - 0.06, 0.002),
                        "y": row_y,
                        "text": str(grp),
                        "showarrow": False,
                        "xanchor": "left",
                        "font": {"size": 9, "color": colors[gi % len(colors)]},
                    }
                )
                if at_risk:
                    for ti, t in enumerate(risk_times):
                        px = x0 + (float(t) / tmax) * cell_w if tmax else x0
                        annotations.append(
                            {
                                "xref": "paper",
                                "yref": "paper",
                                "x": min(max(px, x0), x1),
                                "y": row_y,
                                "text": str(at_risk[ti]) if ti < len(at_risk) else "",
                                "showarrow": False,
                                "xanchor": "center",
                                "font": {"size": 9, "color": "#374151"},
                            }
                        )
            if is_bottom_row:
                # X-axis title below the at-risk table (suppressed on the axis).
                annotations.append(
                    {
                        "xref": "paper",
                        "yref": "paper",
                        "x": (x0 + x1) / 2.0,
                        "y": y0 - risk_gap - (len(group_levels) + 1) * risk_row_h,
                        "text": "Months since Randomization",
                        "showarrow": False,
                        "xanchor": "center",
                        "font": {"size": 12, "color": "#111827"},
                    }
                )

        endpoint_summaries.append(
            {
                "label": label,
                "duration_col": spec.duration_col,
                "event_col": spec.event_col,
                "p_value": p_value,
                "p_text": _format_p_value(p_value),
                "final_by_group": final_by_group,
                "n_by_group": n_by_group,
            }
        )

    # Drop None-valued axis titles (Plotly rejects title:{text:None} noisily).
    for axis in layout_axes.values():
        if axis.get("title") is None:
            axis.pop("title", None)

    layout = {
        "title": {
            "text": req.title
            or "Composite Primary End Point and Individual Components",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 15, "color": "#111827"},
        },
        "height": 760 if nrows > 1 else 440,
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "Arial, sans-serif", "size": 12, "color": "#111827"},
        "margin": {"l": 60, "r": 30, "t": 70, "b": 40},
        # Curves are keyed by the arm-name labels in the No.-at-risk rows and
        # the colored end-of-inset value labels, matching the trial-figure
        # convention, so no separate legend box is drawn.
        "showlegend": False,
        "annotations": annotations,
        **layout_axes,
    }

    return {
        "type": "km_composite",
        "group_col": req.group_col,
        "groups": group_levels or [],
        "endpoints": endpoint_summaries,
        "as_cumulative_incidence": req.as_cumulative_incidence,
        "figure": {"data": trace_data, "layout": layout},
        "method_note": (
            "Cumulative incidence (1 - Kaplan-Meier survival) by group; "
            "p-values from the log-rank test. Number at risk shown beneath each panel."
        ),
    }


# ── Server-side static rendering (headless / API / reports) ──────────────────
# The other chart endpoints return Plotly trace data for the browser to draw.
# This one renders a full figure spec to a static image on the server (kaleido),
# so non-browser callers can obtain a PNG/SVG/PDF directly. Styling stays in the
# caller's figure — the server never builds traces, so there is no drift from
# what the frontend shows. plotly/kaleido are optional; absent → 503.


class RenderRequest(BaseModel):
    figure: Dict[str, Any]  # Plotly figure: {"data": [...], "layout": {...}}
    format: str = "png"  # png | svg | jpeg | pdf | webp
    width: Optional[int] = None  # px; None → figure's own layout size
    height: Optional[int] = None
    scale: float = 2.0  # device-pixel multiplier (print/retina)


@router.post("/render")
def render_chart(req: RenderRequest):
    """Render a Plotly figure spec to a static image and return the raw bytes."""
    try:
        image = plot_render.render_figure(
            req.figure,
            fmt=req.format,
            width=req.width,
            height=req.height,
            scale=req.scale,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except plot_render.RenderUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    media_type = plot_render.MIME_TYPES.get(
        req.format.lower(), "application/octet-stream"
    )
    return Response(content=image, media_type=media_type)
