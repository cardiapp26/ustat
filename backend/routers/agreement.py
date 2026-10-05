"""Method agreement tests: Bland-Altman, Deming regression, Passing-Bablok, Lin's CCC."""
import numpy as np
import pandas as pd
from scipy import stats as sp
from fastapi import APIRouter, HTTPException
from pydantic import AliasChoices, BaseModel, Field

from services import store
from services.stat_utils import lins_ccc, group_summary

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BLAND-ALTMAN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class BlandAltmanRequest(BaseModel):
    session_id: str
    method1: str = Field(validation_alias=AliasChoices("method1", "column1"))
    method2: str = Field(validation_alias=AliasChoices("method2", "column2"))
    alpha: float = 0.05


@router.post("/bland_altman")
def bland_altman(req: BlandAltmanRequest):
    df = _get_df(req.session_id)
    for c in [req.method1, req.method2]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.method1, req.method2]].dropna()
    if len(sub) < 3:
        raise HTTPException(400, "Need at least 3 paired observations.")

    x = sub[req.method1].astype(float).values
    y = sub[req.method2].astype(float).values
    n = len(x)

    means = (x + y) / 2
    diffs = x - y
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1))

    # Limits of agreement are the conventional mean +/- 1.96 SD (Bland & Altman
    # 1986), whatever alpha is; alpha only sets the confidence level of the
    # intervals around the bias and around each limit.
    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff

    # Student t critical value on n-1 df (not 1.96): with n = 30 the normal
    # value makes every interval about 4% too narrow.
    alpha = float(req.alpha)
    if not 0.0 < alpha < 1.0:
        raise HTTPException(400, "alpha must be between 0 and 1.")
    conf_pct = f"{(1 - alpha) * 100:g}"
    t_crit = float(sp.t.ppf(1 - alpha / 2, n - 1))

    # CI for the mean difference
    se_mean = sd_diff / np.sqrt(n)
    ci_mean_low = mean_diff - t_crit * se_mean
    ci_mean_high = mean_diff + t_crit * se_mean

    # CIs for the limits of agreement (Bland & Altman 1986, 1999):
    # SE(LoA) = sd * sqrt(1/n + 1.96^2 / (2 (n - 1))), interval = LoA +/- t * SE.
    se_loa = sd_diff * float(np.sqrt(1.0 / n + 1.96 ** 2 / (2.0 * (n - 1))))
    ci_loa_lower = {"low": round(loa_lower - t_crit * se_loa, 4), "high": round(loa_lower + t_crit * se_loa, 4)}
    ci_loa_upper = {"low": round(loa_upper - t_crit * se_loa, 4), "high": round(loa_upper + t_crit * se_loa, 4)}

    # Proportional bias: regression of diffs on means
    slope, intercept, r_val, p_bias, se_slope = sp.linregress(means, diffs)
    slope = float(slope)
    intercept = float(intercept)
    p_bias = float(p_bias)
    bias_sig = bool(p_bias < req.alpha)
    ps_bias = _p_str(p_bias)

    return {
        "test": "Bland-Altman analysis",
        "statistic": round(mean_diff, 4),
        "p": p_bias,
        "significant": bias_sig,
        "effect_sizes": [],
        "assumptions": [],
        "plot_data": {
            "means": means.tolist(),
            "diffs": diffs.tolist(),
        },
        "summary": {
            req.method1: group_summary(x, req.method1),
            req.method2: group_summary(y, req.method2),
            "mean_diff": round(mean_diff, 4),
            "sd_diff": round(sd_diff, 4),
            "n": n,
        },
        "limits_of_agreement": {
            "lower": round(loa_lower, 4),
            "upper": round(loa_upper, 4),
            "mean_diff": round(mean_diff, 4),
            "ci_loa_lower": ci_loa_lower,
            "ci_loa_upper": ci_loa_upper,
            "se_loa": round(se_loa, 4),
            "multiplier": 1.96,
        },
        "ci_loa_lower": ci_loa_lower,
        "ci_loa_upper": ci_loa_upper,
        "ci_mean_diff": {
            "low": round(ci_mean_low, 4),
            "high": round(ci_mean_high, 4),
        },
        "bias_regression": {
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "p": p_bias,
            "significant": bias_sig,
        },
        "interpretation": (
            f"Mean difference = {mean_diff:.3f} (SD = {sd_diff:.3f}). "
            f"LOA: [{loa_lower:.3f}, {loa_upper:.3f}] "
            f"(lower {conf_pct}% CI [{ci_loa_lower['low']:.3f}, {ci_loa_lower['high']:.3f}], "
            f"upper {conf_pct}% CI [{ci_loa_upper['low']:.3f}, {ci_loa_upper['high']:.3f}]). "
            f"Proportional bias: {'present' if bias_sig else 'absent'} (slope = {slope:.4f}, p = {ps_bias})."
        ),
        "result_text": (
            f"Bland-Altman analysis compared {req.method1} and {req.method2} (n = {n} pairs). "
            f"The mean difference (bias) was {mean_diff:.3f} (SD = {sd_diff:.3f}), "
            f"{conf_pct}% CI {ci_mean_low:.3f} to {ci_mean_high:.3f}, "
            f"with 95% limits of agreement from {loa_lower:.3f} to {loa_upper:.3f} "
            f"(lower limit {conf_pct}% CI {ci_loa_lower['low']:.3f} to {ci_loa_lower['high']:.3f}; "
            f"upper limit {conf_pct}% CI {ci_loa_upper['low']:.3f} to {ci_loa_upper['high']:.3f}). "
            f"Proportional bias was {'detected' if bias_sig else 'not detected'} "
            f"(regression slope = {slope:.4f}, p = {ps_bias})."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Mean difference (bias)", round(mean_diff, 4)],
            ["SD of differences", round(sd_diff, 4)],
            ["Lower LOA", round(loa_lower, 4)],
            ["Upper LOA", round(loa_upper, 4)],
            [f"{conf_pct}% CI mean diff (lower)", round(ci_mean_low, 4)],
            [f"{conf_pct}% CI mean diff (upper)", round(ci_mean_high, 4)],
            [f"Lower LOA {conf_pct}% CI (lower)", ci_loa_lower["low"]],
            [f"Lower LOA {conf_pct}% CI (upper)", ci_loa_lower["high"]],
            [f"Upper LOA {conf_pct}% CI (lower)", ci_loa_upper["low"]],
            [f"Upper LOA {conf_pct}% CI (upper)", ci_loa_upper["high"]],
            ["Proportional bias slope", round(slope, 4)],
            ["Proportional bias p", round(p_bias, 6)],
            ["n", n],
        ],
        "r_code": f"library(BlandAltmanLeh)\nbland.altman.plot(data${req.method1}, data${req.method2})",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DEMING REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class DemingRequest(BaseModel):
    session_id: str
    method1: str = Field(validation_alias=AliasChoices("method1", "column1"))
    method2: str = Field(validation_alias=AliasChoices("method2", "column2"))
    error_ratio: float = 1.0
    alpha: float = 0.05


@router.post("/deming")
def deming_regression(req: DemingRequest):
    df = _get_df(req.session_id)
    for c in [req.method1, req.method2]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.method1, req.method2]].dropna()
    if len(sub) < 5:
        raise HTTPException(400, "Need at least 5 paired observations.")

    x = sub[req.method1].astype(float).values
    y = sub[req.method2].astype(float).values
    n = len(x)
    lam = req.error_ratio

    # Deming regression formula
    mx, my = x.mean(), y.mean()
    Sxx = float(np.sum((x - mx) ** 2)) / (n - 1)
    Syy = float(np.sum((y - my) ** 2)) / (n - 1)
    Sxy = float(np.sum((x - mx) * (y - my))) / (n - 1)

    diff = Syy - lam * Sxx
    slope = (diff + np.sqrt(diff ** 2 + 4 * lam * Sxy ** 2)) / (2 * Sxy) if Sxy != 0 else 0.0
    intercept = my - slope * mx

    # Standard errors via jackknife
    slopes_jk = []
    intercepts_jk = []
    for i in range(n):
        xj = np.delete(x, i)
        yj = np.delete(y, i)
        mxj, myj = xj.mean(), yj.mean()
        Sxxj = float(np.sum((xj - mxj) ** 2)) / (n - 2)
        Syyj = float(np.sum((yj - myj) ** 2)) / (n - 2)
        Sxyj = float(np.sum((xj - mxj) * (yj - myj))) / (n - 2)
        dj = Syyj - lam * Sxxj
        sj = (dj + np.sqrt(dj ** 2 + 4 * lam * Sxyj ** 2)) / (2 * Sxyj) if Sxyj != 0 else 0.0
        ij = myj - sj * mxj
        slopes_jk.append(sj)
        intercepts_jk.append(ij)

    slopes_jk = np.array(slopes_jk)
    intercepts_jk = np.array(intercepts_jk)
    se_slope = float(np.sqrt((n - 1) / n * np.sum((slopes_jk - slopes_jk.mean()) ** 2)))
    se_intercept = float(np.sqrt((n - 1) / n * np.sum((intercepts_jk - intercepts_jk.mean()) ** 2)))

    ci_slope_low = slope - 1.96 * se_slope
    ci_slope_high = slope + 1.96 * se_slope
    ci_int_low = intercept - 1.96 * se_intercept
    ci_int_high = intercept + 1.96 * se_intercept

    # Test if slope differs from 1 and intercept from 0
    slope_differs = not (ci_slope_low <= 1 <= ci_slope_high)
    intercept_differs = not (ci_int_low <= 0 <= ci_int_high)

    return {
        "test": "Deming regression",
        "statistic": round(float(slope), 4),
        "p": None,
        "significant": slope_differs or intercept_differs,
        "effect_sizes": [],
        "assumptions": [],
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "se_slope": round(se_slope, 4),
        "se_intercept": round(se_intercept, 4),
        "ci_slope": {"low": round(ci_slope_low, 4), "high": round(ci_slope_high, 4)},
        "ci_intercept": {"low": round(ci_int_low, 4), "high": round(ci_int_high, 4)},
        "error_ratio": lam,
        "summary": {
            req.method1: group_summary(x, req.method1),
            req.method2: group_summary(y, req.method2),
            "n": n,
        },
        "interpretation": (
            f"Deming regression (lambda = {lam}): y = {intercept:.4f} + {slope:.4f}x. "
            f"Slope {'differs from' if slope_differs else 'includes'} 1 (95% CI [{ci_slope_low:.4f}, {ci_slope_high:.4f}]). "
            f"Intercept {'differs from' if intercept_differs else 'includes'} 0 (95% CI [{ci_int_low:.4f}, {ci_int_high:.4f}])."
        ),
        "result_text": (
            f"Deming regression (error ratio = {lam}) compared {req.method1} and {req.method2} (n = {n}). "
            f"Slope = {slope:.4f} (SE = {se_slope:.4f}, 95% CI [{ci_slope_low:.4f}, {ci_slope_high:.4f}]). "
            f"Intercept = {intercept:.4f} (SE = {se_intercept:.4f}, 95% CI [{ci_int_low:.4f}, {ci_int_high:.4f}]). "
            f"{'Proportional bias detected (slope CI excludes 1).' if slope_differs else 'No proportional bias (slope CI includes 1).'} "
            f"{'Constant bias detected (intercept CI excludes 0).' if intercept_differs else 'No constant bias (intercept CI includes 0).'}"
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Slope", round(float(slope), 4)],
            ["SE (slope)", round(se_slope, 4)],
            ["95% CI slope (lower)", round(ci_slope_low, 4)],
            ["95% CI slope (upper)", round(ci_slope_high, 4)],
            ["Intercept", round(float(intercept), 4)],
            ["SE (intercept)", round(se_intercept, 4)],
            ["95% CI intercept (lower)", round(ci_int_low, 4)],
            ["95% CI intercept (upper)", round(ci_int_high, 4)],
            ["Error ratio (lambda)", lam],
            ["n", n],
        ],
        "r_code": f"library(deming)\ndeming(data${req.method2} ~ data${req.method1}, data = data)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PASSING-BABLOK REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class PassingBablokRequest(BaseModel):
    session_id: str
    method1: str = Field(validation_alias=AliasChoices("method1", "column1"))
    method2: str = Field(validation_alias=AliasChoices("method2", "column2"))
    alpha: float = 0.05


# Above this many pairs the O(n^2) pairwise slope set no longer fits comfortably
# in memory (n = 5000 already gives 12.5 million slopes).
PB_MAX_N = 5000


def _pb_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """All pairwise slopes S_ij (i < j) of Passing & Bablok (1983), sorted.

    Pairs with dx = dy = 0 are dropped. dx = 0 with dy != 0 gives +/-inf as in
    the paper. Slopes of exactly -1 are dropped (they would sit on the
    boundary of the K shift). Built in row blocks to bound memory.
    """
    n = len(x)
    chunks = []
    block = max(1, int(4_000_000 // max(n, 1)))
    for start in range(0, n - 1, block):
        stop = min(start + block, n - 1)
        xi = x[start:stop, None]
        yi = y[start:stop, None]
        dx = x[None, :] - xi
        dy = y[None, :] - yi
        # keep only j > i
        jj = np.arange(n)[None, :]
        ii = np.arange(start, stop)[:, None]
        mask = jj > ii
        dx = dx[mask]
        dy = dy[mask]
        keep = ~((dx == 0) & (dy == 0))
        dx = dx[keep]
        dy = dy[keep]
        with np.errstate(divide="ignore", invalid="ignore"):
            sl = np.where(dx == 0, np.sign(dy) * np.inf, dy / np.where(dx == 0, 1.0, dx))
        chunks.append(sl[sl != -1.0])
    out = np.concatenate(chunks) if chunks else np.array([], dtype=float)
    out.sort()
    return out


def _pb_median_shifted(s_sorted: np.ndarray, k: int) -> float:
    """Median of the sorted slopes shifted by K (Passing & Bablok 1983, eq. 8)."""
    big_n = len(s_sorted)
    if big_n % 2 == 1:
        return float(s_sorted[(big_n + 1) // 2 + k - 1])
    lo = s_sorted[big_n // 2 + k - 1]
    hi = s_sorted[big_n // 2 + k]
    return float((lo + hi) / 2.0)


def _pb_cusum(x: np.ndarray, y: np.ndarray, a: float, b: float):
    """Cusum linearity test of Passing & Bablok (1983).

    Residual signs, ordered by the position of each point's projection along
    the fitted line, are weighted so the cusum starts and ends at zero:
    sqrt(M/L) for the L points above the line and -sqrt(L/M) for the M points
    below. H = max|C_i| / sqrt(L + M) follows (asymptotically) the Kolmogorov
    distribution. Points exactly on the line are left out. Returns
    (h_statistic, p_value); None values when the test cannot be computed.
    """
    if not np.isfinite(a) or not np.isfinite(b):
        return None, None
    resid = y - (a + b * x)
    scale = max(float(np.max(np.abs(y))), 1.0)
    on_line = np.abs(resid) <= 1e-12 * scale
    pos = (resid > 0) & ~on_line
    neg = (resid < 0) & ~on_line
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        # every point is on or on one side of the line: nothing departs from it
        return 0.0, 1.0
    order = np.argsort(x + b * y, kind="stable")
    f = np.zeros(len(x))
    f[pos] = np.sqrt(n_neg / n_pos)
    f[neg] = -np.sqrt(n_pos / n_neg)
    f = f[order][(pos | neg)[order]]
    cusum = np.cumsum(f)
    h = float(np.max(np.abs(cusum)) / np.sqrt(n_pos + n_neg))
    return h, float(sp.kstwobign.sf(h))


def _pb_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    """Passing-Bablok regression of y on x (1983 algorithm)."""
    n = len(x)
    s = _pb_slopes(x, y)
    big_n = len(s)
    if big_n == 0:
        raise HTTPException(400, "Passing-Bablok regression is undefined: every pair of points is identical.")
    k = int(np.sum(s < -1.0))
    if big_n + k < 1 or ((big_n % 2 == 0) and (big_n // 2 + k + 1 > big_n)):
        raise HTTPException(400, "Passing-Bablok regression is undefined for these data (the slopes are mostly below -1; the methods are negatively related).")
    slope = _pb_median_shifted(s, k)
    intercept = float(np.median(y - slope * x)) if np.isfinite(slope) else float("nan")

    # Confidence bounds for the slope (C_gamma, M1, M2 as in the paper)
    z = float(sp.norm.ppf(1 - alpha / 2))
    c_gamma = z * float(np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0))
    m1 = int(np.round((big_n - c_gamma) / 2.0))
    m2 = big_n - m1 + 1
    lo_idx = min(max(m1 + k, 1), big_n)
    hi_idx = min(max(m2 + k, 1), big_n)
    slope_low = float(s[lo_idx - 1])
    slope_high = float(s[hi_idx - 1])

    # Intercept bounds come from the slope bounds
    def _med_int(b: float) -> float:
        return float(np.median(y - b * x)) if np.isfinite(b) else float("nan")

    int_low = _med_int(slope_high)
    int_high = _med_int(slope_low)
    h_stat, cusum_p = _pb_cusum(x, y, intercept, slope)
    return {
        "slope": slope, "intercept": intercept,
        "slope_low": slope_low, "slope_high": slope_high,
        "int_low": int_low, "int_high": int_high,
        "n_slopes": big_n, "K": k,
        "cusum_h": h_stat, "cusum_p": cusum_p,
    }


def _r4(v):
    """Round to 4 dp, mapping non-finite values to None (valid JSON)."""
    return round(float(v), 4) if v is not None and np.isfinite(v) else None


def _fmt4(v) -> str:
    return f"{v:.4f}" if v is not None and np.isfinite(v) else "NA"


@router.post("/passing_bablok")
def passing_bablok(req: PassingBablokRequest):
    df = _get_df(req.session_id)
    for c in [req.method1, req.method2]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")
    if not 0.0 < req.alpha < 1.0:
        raise HTTPException(400, "alpha must be between 0 and 1.")

    sub = df[[req.method1, req.method2]].dropna()
    if len(sub) < 10:
        raise HTTPException(400, "Need at least 10 paired observations for Passing-Bablok regression.")
    if len(sub) > PB_MAX_N:
        raise HTTPException(
            400,
            f"Passing-Bablok regression uses all n(n-1)/2 pairwise slopes; n = {len(sub)} exceeds the limit of {PB_MAX_N} pairs.",
        )

    x = sub[req.method1].astype(float).values
    y = sub[req.method2].astype(float).values
    n = len(x)

    fit = _pb_fit(x, y, req.alpha)
    slope = fit["slope"]
    intercept = fit["intercept"]
    ci_slope_low, ci_slope_high = fit["slope_low"], fit["slope_high"]
    ci_int_low, ci_int_high = fit["int_low"], fit["int_high"]
    cusum_p = fit["cusum_p"]

    linearity_ok = None if cusum_p is None else bool(cusum_p >= req.alpha)
    cusum_str = "NA" if cusum_p is None else _p_str(cusum_p)
    linearity_word = "could not be assessed" if linearity_ok is None else ("met" if linearity_ok else "violated")

    slope_differs = not (ci_slope_low <= 1 <= ci_slope_high)
    intercept_differs = not (ci_int_low <= 0 <= ci_int_high)
    conf_pct = f"{(1 - req.alpha) * 100:g}"

    return {
        "test": "Passing-Bablok regression",
        "method": "Passing-Bablok (1983)",
        "statistic": _r4(slope),
        "p": None,
        "significant": slope_differs or intercept_differs,
        "effect_sizes": [],
        "assumptions": [
            {"name": "Linearity (Cusum test)", "met": linearity_ok,
             "detail": f"Cusum p = {cusum_str}"},
        ],
        "slope": _r4(slope),
        "intercept": _r4(intercept),
        "ci_slope": {"low": _r4(ci_slope_low), "high": _r4(ci_slope_high)},
        "ci_intercept": {"low": _r4(ci_int_low), "high": _r4(ci_int_high)},
        "cusum_p": None if cusum_p is None else round(cusum_p, 4),
        "cusum_h": None if fit["cusum_h"] is None else round(fit["cusum_h"], 4),
        "n_slopes": fit["n_slopes"],
        "K": fit["K"],
        "summary": {
            req.method1: group_summary(x, req.method1),
            req.method2: group_summary(y, req.method2),
            "n": n,
        },
        "interpretation": (
            f"Passing-Bablok regression: y = {_fmt4(intercept)} + {_fmt4(slope)}x. "
            f"Slope {'differs from' if slope_differs else 'includes'} 1 ({conf_pct}% CI [{_fmt4(ci_slope_low)}, {_fmt4(ci_slope_high)}]). "
            f"Intercept {'differs from' if intercept_differs else 'includes'} 0 ({conf_pct}% CI [{_fmt4(ci_int_low)}, {_fmt4(ci_int_high)}]). "
            f"Linearity: {'not assessed' if linearity_ok is None else ('OK' if linearity_ok else 'violated')} (Cusum p = {cusum_str})."
        ),
        "result_text": (
            f"Passing-Bablok regression (1983) compared {req.method1} and {req.method2} (n = {n}, {fit['n_slopes']} pairwise slopes, K = {fit['K']}). "
            f"Slope = {_fmt4(slope)} ({conf_pct}% CI [{_fmt4(ci_slope_low)}, {_fmt4(ci_slope_high)}]). "
            f"Intercept = {_fmt4(intercept)} ({conf_pct}% CI [{_fmt4(ci_int_low)}, {_fmt4(ci_int_high)}]). "
            f"{'Proportional bias detected (slope CI excludes 1).' if slope_differs else 'No proportional bias (slope CI includes 1).'} "
            f"{'Constant bias detected (intercept CI excludes 0).' if intercept_differs else 'No constant bias (intercept CI includes 0).'} "
            f"Cusum linearity test: p = {cusum_str}, linearity assumption {linearity_word}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Slope", _r4(slope)],
            [f"{conf_pct}% CI slope (lower)", _r4(ci_slope_low)],
            [f"{conf_pct}% CI slope (upper)", _r4(ci_slope_high)],
            ["Intercept", _r4(intercept)],
            [f"{conf_pct}% CI intercept (lower)", _r4(ci_int_low)],
            [f"{conf_pct}% CI intercept (upper)", _r4(ci_int_high)],
            ["Cusum linearity p", None if cusum_p is None else round(cusum_p, 4)],
            ["Pairwise slopes (N)", fit["n_slopes"]],
            ["K (slopes < -1)", fit["K"]],
            ["n", n],
        ],
        "r_code": f'library(mcr)\nmcreg(data${req.method1}, data${req.method2}, method.reg = "PaBa")',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LIN'S CONCORDANCE CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

class ConcordanceRequest(BaseModel):
    session_id: str
    method1: str = Field(validation_alias=AliasChoices("method1", "column1"))
    method2: str = Field(validation_alias=AliasChoices("method2", "column2"))
    alpha: float = 0.05


@router.post("/concordance")
def concordance(req: ConcordanceRequest):
    df = _get_df(req.session_id)
    for c in [req.method1, req.method2]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.method1, req.method2]].dropna()
    if len(sub) < 5:
        raise HTTPException(400, "Need at least 5 paired observations.")

    x = sub[req.method1].astype(float).values
    y = sub[req.method2].astype(float).values
    n = len(x)

    ccc = lins_ccc(x, y)
    ccc_val = ccc["value"]
    ci_low = ccc["ci_low"]
    ci_high = ccc["ci_high"]
    precision = ccc["precision"]
    accuracy = ccc["accuracy"]

    # Interpretation thresholds for CCC
    if abs(ccc_val) >= 0.99:
        interp = "almost perfect"
    elif abs(ccc_val) >= 0.95:
        interp = "substantial"
    elif abs(ccc_val) >= 0.90:
        interp = "moderate"
    else:
        interp = "poor"

    return {
        "test": "Lin's concordance correlation coefficient",
        "statistic": ccc_val,
        "p": None,
        "significant": None,
        "effect_sizes": [ccc],
        "assumptions": [],
        "ccc": ccc_val,
        "ci": {"low": ci_low, "high": ci_high},
        "precision": precision,
        "accuracy": accuracy,
        "interpretation_label": interp,
        "summary": {
            req.method1: group_summary(x, req.method1),
            req.method2: group_summary(y, req.method2),
            "n": n,
        },
        "interpretation": (
            f"Lin's CCC = {ccc_val:.4f} (95% CI [{ci_low:.4f}, {ci_high:.4f}]) — {interp} agreement. "
            f"Precision (Pearson r) = {precision:.4f}, Accuracy (bias correction) = {accuracy:.4f}."
        ),
        "result_text": (
            f"Lin's concordance correlation coefficient assessed agreement between {req.method1} and {req.method2} "
            f"(n = {n}). CCC = {ccc_val:.4f} (95% CI [{ci_low:.4f}, {ci_high:.4f}]), indicating {interp} agreement. "
            f"Precision (Pearson r) = {precision:.4f}. Accuracy (bias correction factor) = {accuracy:.4f}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["CCC", ccc_val],
            ["95% CI lower", ci_low],
            ["95% CI upper", ci_high],
            ["Precision (Pearson r)", precision],
            ["Accuracy (Cb)", accuracy],
            ["Interpretation", interp],
            ["n", n],
        ],
        "r_code": f"library(DescTools)\nCCC(data${req.method1}, data${req.method2})",
    }
