"""Subgroup analysis: the effect within each stratum, and whether it differs.

The forest plot at the back of every clinical paper. One row per subgroup
level carrying the exposure's effect inside that stratum, and beside them a
P for interaction saying whether the effect really differs across strata at
all.

The two halves come from different models, and that is the point rather than
an inconvenience. The per-stratum estimates come from fitting the model inside
each stratum separately, which is what a reader wants to see. The interaction
p-value does NOT come from comparing those estimates or their intervals — it
comes from one model on the whole sample with an exposure x subgroup
interaction, tested by likelihood ratio against the same model without it.

Reading significance off the strata instead is the classic subgroup error:
two strata can sit either side of the null with non-overlapping-looking
intervals while the interaction test is nowhere near significant, because a
difference in significance is not a significant difference. The panel reports
both and says which one answers the question.
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
from services.stat_utils import sanitize_nonfinite, sorted_groups

router = APIRouter()

# Below this a stratum's estimate is fitted to a handful of points and its
# interval is uninformative. It is still reported — dropping it silently would
# hide the imbalance — but it carries a flag.
THIN_STRATUM = 20


class SubgroupRequest(BaseModel):
    session_id: str
    outcome: str
    exposure: str
    subgroups: List[str]
    outcome_kind: str = "continuous"
    time_col: Optional[str] = None
    covariates: List[str] = []
    categorical: List[str] = []


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
    if not cols:
        return pd.DataFrame(index=df.index)
    raw = df[cols].copy()
    num, cat = [], []
    for c in cols:
        (cat if (c in categorical or not pd.api.types.is_numeric_dtype(raw[c])) else num).append(c)
    num_part = raw[num].apply(pd.to_numeric, errors="coerce") if num else pd.DataFrame(index=raw.index)
    cat_part = (pd.get_dummies(raw[cat].astype("object"), drop_first=True, dummy_na=False)
                if cat else pd.DataFrame(index=raw.index))
    return pd.concat([num_part, cat_part], axis=1).astype(float)


def _fit(kind: str, y: pd.Series, X: pd.DataFrame, time: Optional[pd.Series]):
    """(params, cov, names, df_resid, loglik) or None."""
    try:
        if kind == "continuous":
            design = sm.add_constant(X, has_constant="add")
            r = sm.OLS(y, design).fit()
            return r.params, np.asarray(r.cov_params()), list(design.columns), float(r.df_resid), float(r.llf)
        if kind == "binary":
            design = sm.add_constant(X, has_constant="add")
            # statsmodels' default IRLS tolerance (1e-8) stops one iterate short,
            # and the standard error is then computed from that iterate's
            # weights — 1.1e-7 off the analytic sqrt(diag((X'WX)^-1)) at the
            # fitted coefficients. At 1e-12 it is exact to 6e-14, for the cost
            # of a couple of iterations. (R's glm has the same behaviour at its
            # own default and keeps it even at epsilon = 1e-12, which is where
            # the last residual disagreement with R comes from.)
            r = sm.GLM(y, design, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-12)
            return r.params, np.asarray(r.cov_params()), list(design.columns), None, float(r.llf)
        if kind == "survival":
            from lifelines import CoxPHFitter

            data = X.copy()
            data["_t"] = np.asarray(time, dtype=float)
            data["_e"] = np.asarray(y, dtype=float)
            cph = CoxPHFitter()
            cph.fit(data, duration_col="_t", event_col="_e")
            return (cph.params_, np.asarray(cph.variance_matrix_), list(cph.params_.index),
                    None, float(cph.log_likelihood_))
    except Exception as exc:
        logger.debug(f"subgroup fit failed ({kind}): {exc}")
        return None
    return None


def _term(params, cov, names, df_resid, term: str, kind: str) -> dict:
    if term not in names:
        return {}
    i = names.index(term)
    beta = float(params.iloc[i] if hasattr(params, "iloc") else params[i])
    var = float(cov[i, i])
    if not np.isfinite(beta) or not np.isfinite(var) or var < 0:
        return {}
    se = float(np.sqrt(var))
    stat = beta / se if se > 0 else np.nan
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
        out.update({"ratio": _finite(np.exp(beta)), "ratio_ci_low": _finite(np.exp(lo)),
                    "ratio_ci_high": _finite(np.exp(hi))})
    return out


@router.post("/analyze")
def subgroup_analysis(req: SubgroupRequest):
    """The exposure's effect inside each stratum, plus P for interaction."""
    df = _get_df(req.session_id)
    kind = (req.outcome_kind or "continuous").strip().lower()
    if kind not in ("continuous", "binary", "survival"):
        raise HTTPException(status_code=400, detail="outcome_kind must be continuous, binary or survival")
    if not req.subgroups:
        raise HTTPException(status_code=400, detail="Choose at least one subgroup variable")

    needed = {req.outcome, req.exposure, *req.subgroups, *req.covariates}
    if kind == "survival":
        if not req.time_col or req.time_col not in df.columns:
            raise HTTPException(status_code=400, detail="A survival analysis needs a time column")
        needed.add(req.time_col)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Column not found: {', '.join(sorted(missing))}")
    if req.exposure in req.subgroups:
        raise HTTPException(status_code=400, detail="The exposure cannot also be a subgroup variable")

    y = coerce_numeric(df[req.outcome]).replace([np.inf, -np.inf], np.nan)
    x = coerce_numeric(df[req.exposure]).replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        raise HTTPException(status_code=400, detail=f"'{req.exposure}' has no numeric values")
    t = coerce_numeric(df[req.time_col]).replace([np.inf, -np.inf], np.nan) if kind == "survival" else None
    if kind == "binary":
        lv = set(pd.unique(y.dropna()))
        if not lv <= {0.0, 1.0} or len(lv) < 2:
            raise HTTPException(status_code=400, detail="A binary outcome must be coded 0 / 1 and contain both")

    cov_all = _encode(df, req.covariates, req.categorical)
    overall = _overall(kind, y, x, cov_all, t, req)
    blocks, warnings = [], []

    for sg in req.subgroups:
        g = df[sg]
        levels = sorted_groups(g)
        if len(levels) < 2:
            warnings.append(f"'{sg}' has fewer than 2 levels and was skipped.")
            continue
        if len(levels) > 10:
            warnings.append(f"'{sg}' has {len(levels)} levels — that looks like an identifier, not a subgroup.")
            continue

        # One complete-case rule per subgroup variable, so an unrelated
        # variable's gaps do not shrink this one's strata.
        frames = [y.rename("_y"), x.rename("_x"), g.rename("_g")] + ([t.rename("_t")] if t is not None else [])
        if not cov_all.empty:
            frames.append(cov_all)
        keep = pd.concat(frames, axis=1).notna().all(axis=1)

        rows = []
        for lv_name in levels:
            sel = keep & (g == lv_name)
            n_lv = int(sel.sum())
            row = {"level": str(lv_name), "n": n_lv}
            if kind in ("binary", "survival"):
                row["events"] = int(y[sel].sum())
            if n_lv < 3:
                row["note"] = "too few observations to fit"
                rows.append(row)
                continue
            X = pd.concat([x.rename("exp"), cov_all], axis=1)[sel] if not cov_all.empty else pd.DataFrame({"exp": x})[sel]
            # A covariate constant inside a stratum carries no information
            # there and makes the design singular — sex inside "female", say.
            X = X.loc[:, X.nunique(dropna=False) > 1]
            if "exp" not in X.columns:
                row["note"] = "the exposure does not vary in this stratum"
                rows.append(row)
                continue
            fit = _fit(kind, y[sel], X.astype(float), t[sel] if t is not None else None)
            if fit is None:
                row["note"] = "the model did not converge in this stratum"
            else:
                params, cov_m, names, df_resid, _ = fit
                row.update(_term(params, cov_m, names, df_resid, "exp", kind))
                if n_lv < THIN_STRATUM:
                    row["thin"] = True
            rows.append(row)

        p_int, int_note = _interaction(kind, y, x, g, levels, cov_all, t, keep)
        blocks.append({"variable": sg, "levels": [str(v) for v in levels], "rows": rows,
                       "p_interaction": p_int, "interaction_note": int_note,
                       "n_used": int(keep.sum())})
        if any(r.get("thin") for r in rows):
            thin = [r["level"] for r in rows if r.get("thin")]
            warnings.append(
                f"'{sg}': {', '.join(thin)} has fewer than {THIN_STRATUM} observations, so its estimate "
                "is fitted to very little and its interval is wide for that reason rather than an informative one.")

    if not blocks:
        raise HTTPException(status_code=400, detail="No usable subgroup variable")

    label = {"continuous": "Mean difference", "binary": "Odds ratio", "survival": "Hazard ratio"}[kind]
    return sanitize_nonfinite({
        "outcome": req.outcome, "exposure": req.exposure, "outcome_kind": kind,
        "effect_label": label, "null_value": 0.0 if kind == "continuous" else 1.0,
        "overall": overall, "subgroups": blocks, "warnings": warnings,
        "result_text": _results_text(req, kind, label, overall, blocks),
        # Shipped with the result so an export cannot lose it.
        "caveat": (
            "Significance within a stratum is not evidence that the effect differs between strata — "
            "two subgroups can fall either side of the null while the interaction test is nowhere near "
            "significant. Only P for interaction answers that question, and with several subgroups it "
            "is being asked several times."
        ),
    })


def _overall(kind, y, x, cov, t, req) -> dict:
    """The effect in everybody, as the reference line of the forest plot."""
    frames = [y.rename("_y"), x.rename("_x")] + ([t.rename("_t")] if t is not None else [])
    if not cov.empty:
        frames.append(cov)
    keep = pd.concat(frames, axis=1).notna().all(axis=1)
    X = pd.concat([x.rename("exp"), cov], axis=1)[keep] if not cov.empty else pd.DataFrame({"exp": x})[keep]
    fit = _fit(kind, y[keep], X.astype(float), t[keep] if t is not None else None)
    out = {"n": int(keep.sum())}
    if kind in ("binary", "survival"):
        out["events"] = int(y[keep].sum())
    if fit is None:
        out["note"] = "the overall model did not converge"
        return out
    params, cov_m, names, df_resid, _ = fit
    out.update(_term(params, cov_m, names, df_resid, "exp", kind))
    return out


def _interaction(kind, y, x, g, levels, cov, t, keep) -> tuple[Optional[float], str]:
    """P for interaction, by likelihood ratio on the whole sample.

    One model with exposure x subgroup against the same model without it. The
    per-stratum estimates say nothing about this: comparing them, or eyeballing
    whether their intervals overlap, is a different and much weaker question.
    """
    gd = pd.get_dummies(g.astype("object"), prefix="g", drop_first=True, dummy_na=False).astype(float)
    if gd.empty:
        return None, "the subgroup has only one level"
    base = pd.concat([x.rename("exp"), gd, cov], axis=1) if not cov.empty else pd.concat([x.rename("exp"), gd], axis=1)
    inter = base.copy()
    for c in gd.columns:
        inter[f"exp_x_{c}"] = base["exp"] * base[c]

    f0 = _fit(kind, y[keep], base[keep].astype(float), t[keep] if t is not None else None)
    f1 = _fit(kind, y[keep], inter[keep].astype(float), t[keep] if t is not None else None)
    if f0 is None or f1 is None:
        return None, "the interaction model did not converge"
    stat = 2.0 * (f1[4] - f0[4])
    dfree = int(len(gd.columns))
    if not np.isfinite(stat) or stat < 0:
        return None, "the interaction test was not estimable"
    return _finite(scipy_stats.chi2.sf(stat, dfree)), f"likelihood-ratio test on {dfree} df"


def _results_text(req, kind, label, overall, blocks) -> str:
    word = "mean difference" if kind == "continuous" else ("odds ratio" if kind == "binary" else "hazard ratio")

    def val(e):
        if not e or e.get("beta") is None:
            return "—"
        return (f"{e['beta']:.3f} (95% CI {e['ci_low']:.3f} to {e['ci_high']:.3f})" if kind == "continuous"
                else f"{e['ratio']:.2f} (95% CI {e['ratio_ci_low']:.2f} to {e['ratio_ci_high']:.2f})")

    named = "; ".join(
        f"{b['variable']} {format_p(b['p_interaction'], prefix=True)}" if b["p_interaction"] is not None
        else f"{b['variable']} not estimable"
        for b in blocks)
    return (
        f"Overall the {word} for {req.exposure} was {val(overall)} (n = {overall.get('n')}). "
        f"The effect was examined across {len(blocks)} subgroup "
        f"{'variable' if len(blocks) == 1 else 'variables'}; P for interaction: {named}. "
        f"{label} is reported per one-unit increase in {req.exposure}."
    )
