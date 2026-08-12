"""Progressive adjustment — one exposure, several models, one table.

The staple of observational epidemiology: pick the exposure the paper is
about, then show its effect crude, then adjusted for demographics, then for
demographics plus clinical covariates, and so on. Readers judge confounding by
watching the estimate move across those columns, which only works if every
column is the same estimate of the same thing computed the same way.

Building that table by hand means fitting each model separately and copying
numbers between them, which is where transcription errors come from and where
the models quietly stop being comparable — one fitted on 812 rows, the next on
784, and the reader has no way to tell.

So this endpoint fits them together. Complete cases are taken across the union
of every model's variables, so the estimate that moves across the row moves
because of adjustment and not because the sample changed underneath it. That
is stricter than fitting each model on its own rows, and it is the only way
the comparison means anything; the cost in dropped rows is reported rather
than hidden.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel
from scipy import stats as scipy_stats

from services import store
from services.dirty_value_guard import coerce_numeric
from services.number_format import format_p
from services.stat_utils import sanitize_nonfinite

router = APIRouter()


class ModelSpec(BaseModel):
    label: str
    covariates: List[str] = []


class MultiModelRequest(BaseModel):
    session_id: str
    outcome: str
    exposure: str
    models: List[ModelSpec]
    outcome_kind: str = "continuous"      # continuous | binary | survival
    time_col: Optional[str] = None
    categorical: List[str] = []
    # Treat the exposure as categorical and report one row per level against a
    # reference — the quartile tables these figures usually accompany.
    exposure_categorical: bool = False


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


def _encode(df: pd.DataFrame, cols: List[str], categorical: List[str]) -> pd.DataFrame:
    """Covariates as the model sees them: numbers as-is, everything else dummied."""
    if not cols:
        return pd.DataFrame(index=df.index)
    raw = df[cols].copy()
    num, cat = [], []
    for c in cols:
        if c in categorical or not pd.api.types.is_numeric_dtype(raw[c]):
            cat.append(c)
        else:
            num.append(c)
    num_part = raw[num].apply(pd.to_numeric, errors="coerce") if num else pd.DataFrame(index=raw.index)
    cat_part = (pd.get_dummies(raw[cat].astype("object"), drop_first=True, dummy_na=False)
                if cat else pd.DataFrame(index=raw.index))
    return pd.concat([num_part, cat_part], axis=1).astype(float)


def _fit(kind: str, y: pd.Series, X: pd.DataFrame, time: Optional[pd.Series]):
    """Return (params, cov, names, df_resid) or None. df_resid is None where
    the coefficient's reference distribution is normal rather than t."""
    try:
        if kind == "continuous":
            design = sm.add_constant(X, has_constant="add")
            res = sm.OLS(y, design).fit()
            return res.params, np.asarray(res.cov_params()), list(design.columns), float(res.df_resid)
        if kind == "binary":
            design = sm.add_constant(X, has_constant="add")
            # statsmodels' default IRLS tolerance (1e-8) stops one iterate short,
            # and the standard error is then computed from that iterate's
            # weights — 1.1e-7 off the analytic sqrt(diag((X'WX)^-1)) at the
            # fitted coefficients. At 1e-12 it is exact to 6e-14, for the cost
            # of a couple of iterations. (R's glm has the same behaviour at its
            # own default and keeps it even at epsilon = 1e-12, which is where
            # the last residual disagreement with R comes from.)
            res = sm.GLM(y, design, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-12)
            return res.params, np.asarray(res.cov_params()), list(design.columns), None
        if kind == "survival":
            from lifelines import CoxPHFitter

            data = X.copy()
            data["_t"] = np.asarray(time, dtype=float)
            data["_e"] = np.asarray(y, dtype=float)
            cph = CoxPHFitter()
            cph.fit(data, duration_col="_t", event_col="_e")
            return cph.params_, np.asarray(cph.variance_matrix_), list(cph.params_.index), None
    except Exception as exc:
        logger.debug(f"multimodel fit failed ({kind}): {exc}")
        return None
    return None


def _term(params, cov, names, df_resid, term: str, kind: str) -> dict:
    """One coefficient with its interval and p, on the right reference curve."""
    if term not in names:
        return {}
    i = names.index(term)
    beta = float(params.iloc[i] if hasattr(params, "iloc") else params[i])
    var = float(cov[i, i])
    if not np.isfinite(beta) or not np.isfinite(var) or var < 0:
        return {}
    se = float(np.sqrt(var))
    stat = beta / se if se > 0 else np.nan
    # A linear model tests coefficients against t on the residual df; GLM and
    # Cox are normal-based. Using z everywhere would quietly narrow every
    # linear-model interval, most at the small n where it matters most.
    if df_resid is not None and df_resid > 0:
        p = float(2 * scipy_stats.t.sf(abs(stat), df_resid)) if np.isfinite(stat) else None
        crit = float(scipy_stats.t.ppf(0.975, df_resid))
    else:
        p = float(2 * scipy_stats.norm.sf(abs(stat))) if np.isfinite(stat) else None
        crit = 1.959963984540054
    lo, hi = beta - crit * se, beta + crit * se
    out = {"beta": _finite(beta), "se": _finite(se), "p": _finite(p),
           "ci_low": _finite(lo), "ci_high": _finite(hi)}
    if kind in ("binary", "survival"):
        out.update({"ratio": _finite(np.exp(beta)),
                    "ratio_ci_low": _finite(np.exp(lo)),
                    "ratio_ci_high": _finite(np.exp(hi))})
    return out


@router.post("/analyze")
def multi_model(req: MultiModelRequest):
    """One exposure across progressively adjusted models."""
    df = _get_df(req.session_id)
    kind = (req.outcome_kind or "continuous").strip().lower()
    if kind not in ("continuous", "binary", "survival"):
        raise HTTPException(status_code=400, detail="outcome_kind must be continuous, binary or survival")
    if not req.models:
        raise HTTPException(status_code=400, detail="Add at least one model")

    every = {req.outcome, req.exposure} | {c for m in req.models for c in m.covariates}
    if kind == "survival":
        if not req.time_col or req.time_col not in df.columns:
            raise HTTPException(status_code=400, detail="A survival analysis needs a time column")
        every.add(req.time_col)
    missing = [c for c in every if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Column not found: {', '.join(sorted(missing))}")
    if req.exposure in {c for m in req.models for c in m.covariates}:
        raise HTTPException(status_code=400, detail="The exposure cannot also be an adjustment variable")

    y = coerce_numeric(df[req.outcome]).replace([np.inf, -np.inf], np.nan)
    t = coerce_numeric(df[req.time_col]).replace([np.inf, -np.inf], np.nan) if kind == "survival" else None

    # The exposure as the table will show it: one row per level against a
    # reference, or a single per-unit row.
    exp_raw = df[req.exposure]
    if req.exposure_categorical:
        levels = _sorted_levels(exp_raw)
        if len(levels) < 2:
            raise HTTPException(status_code=400, detail=f"'{req.exposure}' has fewer than 2 levels")
        exp_cat = exp_raw.astype("object")
        exp_design = pd.get_dummies(exp_cat, prefix="exp", drop_first=False, dummy_na=False)
        exp_design = exp_design[[f"exp_{lv}" for lv in levels]].iloc[:, 1:].astype(float)
        exp_present = exp_cat.notna()
    else:
        exp_num = coerce_numeric(exp_raw).replace([np.inf, -np.inf], np.nan)
        exp_design = pd.DataFrame({"exp": exp_num.astype(float)})
        exp_present = exp_num.notna()
        levels = []

    # Complete cases across the union of every model's variables. Fitting each
    # model on its own rows would let the estimate move because the sample
    # changed rather than because of adjustment, which is the one thing this
    # table is read for.
    union_cov = _encode(df, sorted({c for m in req.models for c in m.covariates}), req.categorical)
    frames = [y.rename("_y"), exp_present.rename("_x")] + ([t.rename("_t")] if t is not None else [])
    if not union_cov.empty:
        frames.append(union_cov)
    keep = pd.concat(frames, axis=1).notna().all(axis=1)
    n_used, n_dropped = int(keep.sum()), int(len(df) - int(keep.sum()))
    if n_used < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Only {n_used} rows are complete across every model's variables — too few to fit")

    if kind == "binary":
        lv = set(pd.unique(y[keep].dropna()))
        if not lv <= {0.0, 1.0} or len(lv) < 2:
            raise HTTPException(status_code=400, detail="A binary outcome must be coded 0 / 1 and contain both")

    # Trend across ordered levels: the exposure replaced by one number per
    # level, so a single coefficient carries the monotone trend. Numeric
    # levels use their own values; text levels fall back to their rank, which
    # assumes the ordering is meaningful and is reported as such.
    trend_series, trend_basis = _trend_score(exp_raw, levels) if req.exposure_categorical else (None, None)

    rows, warnings = [], []
    for spec in req.models:
        cov = _encode(df, spec.covariates, req.categorical)
        X = pd.concat([exp_design, cov], axis=1)[keep] if not cov.empty else exp_design[keep]
        fit = _fit(kind, y[keep], X.astype(float), t[keep] if t is not None else None)
        if fit is None:
            warnings.append(f"'{spec.label}' did not converge and is omitted.")
            continue
        params, cov_m, names, df_resid = fit

        if req.exposure_categorical:
            effects = [{"level": str(levels[0]), "reference": True}]
            for lv_name in levels[1:]:
                e = _term(params, cov_m, names, df_resid, f"exp_{lv_name}", kind)
                effects.append({"level": str(lv_name), "reference": False, **e})
        else:
            effects = [{"level": "per unit", "reference": False,
                        **_term(params, cov_m, names, df_resid, "exp", kind)}]

        entry = {"label": spec.label, "covariates": spec.covariates, "effects": effects}
        if trend_series is not None:
            tX = pd.concat([trend_series.rename("exp_trend"), cov], axis=1)[keep] if not cov.empty \
                else pd.DataFrame({"exp_trend": trend_series})[keep]
            tf = _fit(kind, y[keep], tX.astype(float), t[keep] if t is not None else None)
            if tf is not None:
                tp, tc, tn, tdf = tf
                entry["trend"] = _term(tp, tc, tn, tdf, "exp_trend", kind)
        rows.append(entry)

    if not rows:
        raise HTTPException(status_code=400, detail="No model converged on these variables")

    if trend_basis == "rank":
        warnings.append(
            f"P for trend uses the rank of each level of '{req.exposure}' (1, 2, 3 …) because its "
            "levels are not numeric. That assumes the levels are equally spaced and in the right order.")
    if n_dropped:
        warnings.append(
            f"{n_dropped} of {len(df)} rows ({100.0 * n_dropped / len(df):.1f}%) were dropped. Every model "
            "is fitted on the same complete cases, so the estimate moves across the row because of "
            "adjustment rather than because the sample changed.")

    label = {"continuous": "Mean difference", "binary": "Odds ratio", "survival": "Hazard ratio"}[kind]
    return sanitize_nonfinite({
        "outcome": req.outcome, "exposure": req.exposure, "outcome_kind": kind,
        "effect_label": label, "exposure_categorical": bool(req.exposure_categorical),
        "levels": [str(lv) for lv in levels],
        "trend_basis": trend_basis,
        "n_used": n_used, "n_dropped": n_dropped,
        "models": rows,
        "warnings": warnings,
        "result_text": _results_text(req, kind, label, rows, n_used),
    })


def _sorted_levels(series: pd.Series) -> list:
    """Level order for the table: numerically when the codes are numbers, else
    lexicographically — the same rule the rest of the app uses so a quartile
    column does not come out 1, 10, 2."""
    from services.stat_utils import sorted_groups

    return sorted_groups(series)


def _trend_score(series: pd.Series, levels: list) -> tuple[Optional[pd.Series], Optional[str]]:
    """One number per level, for the P-for-trend refit.

    Numeric levels score as themselves — for quantile bins carrying their own
    codes that is the conventional choice. Text levels have no spacing to
    read, so they score by rank, and the caller says so.
    """
    numeric = pd.to_numeric(pd.Series(levels), errors="coerce")
    if numeric.notna().all():
        mapping = {lv: float(v) for lv, v in zip(levels, numeric)}
        basis = "level value"
    else:
        mapping = {lv: float(i + 1) for i, lv in enumerate(levels)}
        basis = "rank"
    scored = series.astype("object").map(mapping)
    return pd.to_numeric(scored, errors="coerce"), basis


def _results_text(req, kind, label, rows, n) -> str:
    word = "mean difference" if kind == "continuous" else ("odds ratio" if kind == "binary" else "hazard ratio")
    last = rows[-1]
    parts = []
    for e in last["effects"]:
        if e.get("reference") or e.get("beta") is None:
            continue
        val = (f"{e['beta']:.3f} (95% CI {e['ci_low']:.3f} to {e['ci_high']:.3f})" if kind == "continuous"
               else f"{e['ratio']:.2f} (95% CI {e['ratio_ci_low']:.2f} to {e['ratio_ci_high']:.2f})")
        parts.append(f"{e['level']}: {val}")
    trend = last.get("trend") or {}
    trend_txt = f" P for trend {format_p(trend['p'], prefix=True)}." if trend.get("p") is not None else ""
    return (
        f"In the fully adjusted model ({last['label']}), the {word} for {req.exposure} was "
        f"{'; '.join(parts) if parts else '—'} (n = {n}).{trend_txt} "
        f"All models were fitted on the same {n} complete cases."
    )
