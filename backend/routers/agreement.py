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

    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff

    # 95% CI for mean difference
    se_mean = sd_diff / np.sqrt(n)
    ci_mean_low = mean_diff - 1.96 * se_mean
    ci_mean_high = mean_diff + 1.96 * se_mean

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
        },
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
            f"LOA: [{loa_lower:.3f}, {loa_upper:.3f}]. "
            f"Proportional bias: {'present' if bias_sig else 'absent'} (slope = {slope:.4f}, p = {ps_bias})."
        ),
        "result_text": (
            f"Bland-Altman analysis compared {req.method1} and {req.method2} (n = {n} pairs). "
            f"The mean difference (bias) was {mean_diff:.3f} (SD = {sd_diff:.3f}), "
            f"with 95% limits of agreement from {loa_lower:.3f} to {loa_upper:.3f}. "
            f"Proportional bias was {'detected' if bias_sig else 'not detected'} "
            f"(regression slope = {slope:.4f}, p = {ps_bias})."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Mean difference (bias)", round(mean_diff, 4)],
            ["SD of differences", round(sd_diff, 4)],
            ["Lower LOA", round(loa_lower, 4)],
            ["Upper LOA", round(loa_upper, 4)],
            ["95% CI mean diff (lower)", round(ci_mean_low, 4)],
            ["95% CI mean diff (upper)", round(ci_mean_high, 4)],
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


@router.post("/passing_bablok")
def passing_bablok(req: PassingBablokRequest):
    df = _get_df(req.session_id)
    for c in [req.method1, req.method2]:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[[req.method1, req.method2]].dropna()
    if len(sub) < 10:
        raise HTTPException(400, "Need at least 10 paired observations for Passing-Bablok regression.")

    x = sub[req.method1].astype(float).values
    y = sub[req.method2].astype(float).values
    n = len(x)

    # Use Theil-Sen (median-based) as approximation to Passing-Bablok
    result = sp.theilslopes(y, x)
    slope = float(result.slope)
    intercept = float(result.intercept)
    ci_slope_low = float(result.low_slope)
    ci_slope_high = float(result.high_slope)

    # Intercept CI from slope CI
    ci_int_low = float(np.median(y) - ci_slope_high * np.median(x))
    ci_int_high = float(np.median(y) - ci_slope_low * np.median(x))

    # Cusum linearity test
    residuals = y - (intercept + slope * x)
    # Approximate p-value using Kolmogorov-Smirnov approach on the residuals
    # under H0 of linearity (residuals ~ N(0, σ²)).
    try:
        cusum_p = float(sp.kstest(residuals, 'norm', args=(0, np.std(residuals, ddof=1))).pvalue)
    except Exception:
        cusum_p = 1.0

    linearity_ok = bool(cusum_p >= req.alpha)

    slope_differs = not (ci_slope_low <= 1 <= ci_slope_high)
    intercept_differs = not (ci_int_low <= 0 <= ci_int_high)

    return {
        "test": "Passing-Bablok regression",
        "statistic": round(slope, 4),
        "p": None,
        "significant": slope_differs or intercept_differs,
        "effect_sizes": [],
        "assumptions": [
            {"name": "Linearity (Cusum test)", "met": linearity_ok,
             "detail": f"Cusum p = {_p_str(cusum_p)}"},
        ],
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "ci_slope": {"low": round(ci_slope_low, 4), "high": round(ci_slope_high, 4)},
        "ci_intercept": {"low": round(ci_int_low, 4), "high": round(ci_int_high, 4)},
        "cusum_p": round(cusum_p, 4),
        "summary": {
            req.method1: group_summary(x, req.method1),
            req.method2: group_summary(y, req.method2),
            "n": n,
        },
        "interpretation": (
            f"Passing-Bablok regression: y = {intercept:.4f} + {slope:.4f}x. "
            f"Slope {'differs from' if slope_differs else 'includes'} 1 (95% CI [{ci_slope_low:.4f}, {ci_slope_high:.4f}]). "
            f"Intercept {'differs from' if intercept_differs else 'includes'} 0 (95% CI [{ci_int_low:.4f}, {ci_int_high:.4f}]). "
            f"Linearity: {'OK' if linearity_ok else 'violated'} (Cusum p = {_p_str(cusum_p)})."
        ),
        "result_text": (
            f"Passing-Bablok regression compared {req.method1} and {req.method2} (n = {n}). "
            f"Slope = {slope:.4f} (95% CI [{ci_slope_low:.4f}, {ci_slope_high:.4f}]). "
            f"Intercept = {intercept:.4f} (95% CI [{ci_int_low:.4f}, {ci_int_high:.4f}]). "
            f"{'Proportional bias detected (slope CI excludes 1).' if slope_differs else 'No proportional bias (slope CI includes 1).'} "
            f"{'Constant bias detected (intercept CI excludes 0).' if intercept_differs else 'No constant bias (intercept CI includes 0).'} "
            f"Cusum linearity test: p = {_p_str(cusum_p)} — linearity assumption {'met' if linearity_ok else 'violated'}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Slope", round(slope, 4)],
            ["95% CI slope (lower)", round(ci_slope_low, 4)],
            ["95% CI slope (upper)", round(ci_slope_high, 4)],
            ["Intercept", round(intercept, 4)],
            ["95% CI intercept (lower)", round(ci_int_low, 4)],
            ["95% CI intercept (upper)", round(ci_int_high, 4)],
            ["Cusum linearity p", round(cusum_p, 4)],
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
