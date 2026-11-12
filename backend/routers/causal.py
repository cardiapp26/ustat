"""Causal-inference methods that complement PSM/IPTW/E-value:

  * /iv_2sls          — instrumental-variable estimation (2SLS) + diagnostics.
  * /mediation        — linear causal mediation (ACME/ADE/proportion mediated).
  * /target_trial     — target-trial emulation: eligibility → IPTW-ATE on the
                        emulated cohort + the 7-component protocol scaffold.
  * /did              — difference-in-differences (2×2) with the interaction
                        estimate, cell means, and parallel-trends note.
  * /rdd              — sharp regression-discontinuity (local-linear LATE at the
                        cutoff with a triangular kernel).
  * /dag_adjustment   — backdoor analysis of a user-specified DAG: roles
                        (confounder/mediator/collider) + a minimal adjustment set.

(Additive router — does not touch the existing PSM/IPTW/sensitivity panels.)
"""
from __future__ import annotations

import functools
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store
from services.impute import apply_imputation

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _numeric_or_two_level(col: pd.Series, name: str) -> tuple[pd.Series, Optional[Dict[str, str]]]:
    """A numeric column as floats, or a two-level text column coded 0/1.

    `pd.to_numeric(..., errors="coerce")` turned a text exposure such as
    control/treat into all-NaN, and the request then failed with "not enough
    complete observations". Two-level text is coded in sorted order (second
    level = 1) and the coding is returned so the response can report it.
    """
    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.notna().sum() == col.notna().sum():
        return numeric.astype(float), None
    levels = sorted({str(v) for v in col.dropna()})
    if len(levels) != 2:
        raise HTTPException(
            422,
            f"'{name}' must be numeric or have exactly two levels; found {len(levels)}: {levels[:6]}",
        )
    coded = col.map(lambda v: np.nan if pd.isna(v) else float(str(v) == levels[1]))
    return coded.astype(float), {"column": name, "coded_0": levels[0], "coded_1": levels[1]}



def _states_exposure_coding(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Append how a text exposure was coded to the result sentence.

    The coding was returned as `exposure_coding` but no panel printed it, so a
    reader of "IV effect = 3.40" could not tell whether that was treat minus
    control or the reverse. In the sentence it also reaches copies and exports.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> dict:
        result = fn(*args, **kwargs)
        coding = result.get("exposure_coding") if isinstance(result, dict) else None
        if coding and isinstance(result.get("result_text"), str):
            result["result_text"] = (
                f"{result['result_text'].rstrip()} {coding['column']} was coded "
                f"1 = '{coding['coded_1']}', 0 = '{coding['coded_0']}'."
            )
        return result
    return wrapper

def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _design(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Numeric design block (categoricals → dummies, drop_first)."""
    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[cols], drop_first=True).astype(float)


class IV2SLSRequest(BaseModel):
    session_id: str
    outcome: str                       # continuous
    endogenous: str                    # the treatment/exposure suspected endogenous
    instruments: List[str]             # ≥1 instrument(s)
    covariates: List[str] = []         # exogenous controls
    imputation: str = "listwise"


@router.post("/iv_2sls")
@_states_exposure_coding
def iv_2sls(req: IV2SLSRequest):
    """Two-stage least squares IV estimator for a continuous outcome with one
    endogenous regressor. Reports the IV (2SLS) effect with correct SE, the
    first-stage weak-instrument F, the Wu-Hausman endogeneity test, the Sargan
    over-identification test (when over-identified), and the naive OLS estimate
    for contrast.
    """
    if not req.instruments:
        raise HTTPException(400, "Provide at least one instrument.")
    if req.endogenous in req.instruments:
        raise HTTPException(400, "The endogenous variable cannot also be an instrument.")
    if req.endogenous in req.covariates:
        raise HTTPException(400, "The endogenous variable cannot also be a covariate.")

    df_full = _get_df(req.session_id)
    cols = list(dict.fromkeys([req.outcome, req.endogenous] + req.instruments + req.covariates))
    missing = [c for c in cols if c not in df_full.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")

    df = apply_imputation(df_full, cols, req.imputation)
    df[req.outcome] = pd.to_numeric(df[req.outcome], errors="coerce")
    df[req.endogenous], exposure_coding = _numeric_or_two_level(df[req.endogenous], req.endogenous)
    df = df.dropna(subset=cols)
    n = len(df)
    if n < len(cols) + 10:
        raise HTTPException(400, "Not enough complete observations for IV estimation.")

    y = df[req.outcome].astype(float).values
    X = df[[req.endogenous]].astype(float).values            # n×1 endogenous
    W = _design(df, req.covariates)                           # exogenous controls
    Z = _design(df, req.instruments)                          # instruments
    n_instr = Z.shape[1]
    if n_instr < 1:
        raise HTTPException(400, "Instruments produced no usable columns.")

    Wc = sm.add_constant(W, has_constant="add")               # add intercept
    exog_2nd_names = list(Wc.columns) + [req.endogenous]

    # ── First stage: endogenous ~ controls + instruments ─────────────────────
    fs_exog = pd.concat([Wc, Z], axis=1).astype(float)
    fs = sm.OLS(X.ravel(), fs_exog).fit()
    Xhat = fs.fittedvalues.values
    v = fs.resid.values                                       # first-stage residuals

    # Weak-instrument test: joint F for the excluded instruments.
    try:
        f_res = fs.f_test([f"{c} = 0" for c in Z.columns])
        first_stage_F = float(np.ravel(f_res.fvalue)[0])
        first_stage_F_p = float(f_res.pvalue)
    except Exception:
        first_stage_F, first_stage_F_p = float("nan"), float("nan")
    weak = bool(np.isfinite(first_stage_F) and first_stage_F < 10.0)

    # ── Second stage: outcome ~ controls + fitted endogenous ─────────────────
    ss_exog = pd.concat([Wc, pd.Series(Xhat, index=df.index, name=req.endogenous)], axis=1).astype(float)
    ss = sm.OLS(y, ss_exog).fit()
    beta = ss.params.values

    # Correct 2SLS covariance: residuals use the ACTUAL endogenous, not fitted.
    ss_actual = pd.concat([Wc, df[[req.endogenous]].astype(float)], axis=1).astype(float)
    resid = y - ss_actual.values @ beta
    k = ss_exog.shape[1]
    sigma2 = float(resid @ resid) / (n - k)
    XtX = ss_exog.values.T @ ss_exog.values
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        raise HTTPException(400, "Design matrix is singular — check for collinear instruments/covariates.")
    Vbeta = sigma2 * XtX_inv
    se = np.sqrt(np.clip(np.diag(Vbeta), 0, None))
    tvals = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = 2 * sp.t.sf(np.abs(tvals), df=n - k)
    tcrit = float(sp.t.ppf(0.975, n - k))

    coefs = []
    endo_row = None
    for i, name in enumerate(exog_2nd_names):
        row = {
            "variable": name,
            "estimate": round(float(beta[i]), 6),
            "se": round(float(se[i]), 6),
            "t": round(float(tvals[i]), 4),
            "p": float(pvals[i]),
            "ci_low": round(float(beta[i] - tcrit * se[i]), 6),
            "ci_high": round(float(beta[i] + tcrit * se[i]), 6),
        }
        coefs.append(row)
        if name == req.endogenous:
            endo_row = row

    # ── Naive OLS (for contrast — biased if the regressor is endogenous) ─────
    ols = sm.OLS(y, ss_actual).fit()
    ols_b = float(ols.params.get(req.endogenous, np.nan))
    ols_se = float(ols.bse.get(req.endogenous, np.nan))
    ols_p = float(ols.pvalues.get(req.endogenous, np.nan))

    # ── Wu-Hausman endogeneity test (control-function form) ──────────────────
    aug = pd.concat([ss_actual, pd.Series(v, index=df.index, name="_resid_v_")], axis=1).astype(float)
    haus = sm.OLS(y, aug).fit()
    wu_p = float(haus.pvalues.get("_resid_v_", np.nan))
    wu_t = float(haus.tvalues.get("_resid_v_", np.nan))
    endogenous_flag = bool(np.isfinite(wu_p) and wu_p < 0.05)

    # ── Sargan over-identification test (only if over-identified) ────────────
    sargan = None
    if n_instr > 1:
        sg = sm.OLS(resid, pd.concat([Wc, Z], axis=1).astype(float)).fit()
        sargan_stat = float(n * sg.rsquared)
        sargan_df = int(n_instr - 1)
        sargan_p = float(sp.chi2.sf(sargan_stat, sargan_df))
        sargan = {"stat": round(sargan_stat, 4), "df": sargan_df, "p": sargan_p,
                  "valid": bool(sargan_p >= 0.05)}

    iv_b = endo_row["estimate"]
    iv_p = endo_row["p"]
    result_text = (
        f"Instrumental-variable (2SLS) estimate of the effect of {req.endogenous} on {req.outcome} "
        f"using {', '.join(req.instruments)} as instrument(s) (n = {n}). "
        f"IV effect = {iv_b:.4f} (95% CI {endo_row['ci_low']:.4f} to {endo_row['ci_high']:.4f}, p = {_p_str(iv_p)}); "
        f"naive OLS = {ols_b:.4f} (p = {_p_str(ols_p)}). "
        f"First-stage F = {first_stage_F:.1f} "
        + ("(WEAK instruments, F < 10 — IV estimate unreliable). " if weak else "(instruments adequate, F ≥ 10). ")
        + f"Wu-Hausman endogeneity p = {_p_str(wu_p)} "
        + ("→ endogeneity present, IV preferred over OLS. " if endogenous_flag
           else "→ no strong evidence of endogeneity; OLS may suffice. ")
        + (f"Sargan over-identification p = {_p_str(sargan['p'])} "
           + ("(instruments jointly valid)." if sargan["valid"] else "(instrument validity in doubt).")
           if sargan else "Just-identified — over-identification not testable.")
    )

    return {
        "exposure_coding": exposure_coding,
        "test": "Instrumental Variable (2SLS)",
        "outcome": req.outcome,
        "endogenous": req.endogenous,
        "instruments": req.instruments,
        "covariates": req.covariates,
        "n": int(n),
        "iv_estimate": endo_row,
        "coefficients": coefs,
        "ols_estimate": {"estimate": round(ols_b, 6), "se": round(ols_se, 6), "p": ols_p},
        "first_stage": {"f_stat": round(first_stage_F, 4), "f_p": first_stage_F_p,
                        "weak_instruments": weak, "n_instruments": int(n_instr)},
        "wu_hausman": {"t": round(wu_t, 4), "p": wu_p, "endogenous": endogenous_flag},
        "sargan": sargan,
        "result_text": result_text,
        "interpretation": result_text,
        "r_code": (
            f'library(AER)\n'
            f'iv <- ivreg({req.outcome} ~ {req.endogenous}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "")
            + f' | {" + ".join(req.instruments)}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "")
            + ', data = data)\n'
            'summary(iv, diagnostics = TRUE)  # weak instruments, Wu-Hausman, Sargan'
        ),
    }


# ── Causal mediation (X → M → Y) ──────────────────────────────────────────────

class MediationRequest(BaseModel):
    session_id: str
    outcome: str                       # continuous Y
    treatment: str                     # exposure X
    mediator: str                      # continuous M
    covariates: List[str] = []
    bootstrap: int = 1000              # nonparametric bootstrap reps for the ACME CI
    imputation: str = "listwise"


@router.post("/mediation")
@_states_exposure_coding
def mediation(req: MediationRequest):
    """Linear causal mediation (Baron-Kenny / Preacher-Hayes) for a continuous
    mediator and outcome. Decomposes the total effect into the indirect effect
    through the mediator (ACME = a·b) and the direct effect (ADE = c'), with a
    percentile bootstrap CI for the indirect effect, a Sobel test, and the
    proportion mediated.
    """
    for nm, c in [("outcome", req.outcome), ("treatment", req.treatment), ("mediator", req.mediator)]:
        if not c:
            raise HTTPException(400, f"{nm} is required.")
    if len({req.outcome, req.treatment, req.mediator}) < 3:
        raise HTTPException(400, "Outcome, treatment, and mediator must be three distinct columns.")

    df_full = _get_df(req.session_id)
    cols = list(dict.fromkeys([req.outcome, req.treatment, req.mediator] + req.covariates))
    miss = [c for c in cols if c not in df_full.columns]
    if miss:
        raise HTTPException(400, f"Columns not found: {miss}")

    df = apply_imputation(df_full, cols, req.imputation)
    for c in [req.outcome, req.mediator]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[req.treatment], exposure_coding = _numeric_or_two_level(df[req.treatment], req.treatment)
    df = df.dropna(subset=cols)
    y_levels = sorted(df[req.outcome].dropna().unique().tolist())
    if len(y_levels) == 2 and set(y_levels).issubset({0, 1, 0.0, 1.0}):
        raise HTTPException(
            status_code=422,
            detail=(
                "Linear mediation requires a continuous outcome. Binary outcomes "
                "need a logistic mediation model, which is not implemented in this endpoint."
            ),
        )
    n = len(df)
    if n < len(cols) + 10:
        raise HTTPException(400, "Not enough complete observations for mediation analysis.")

    def _fit(d: pd.DataFrame):
        Wd = _design(d, req.covariates)
        Xm = sm.add_constant(pd.concat([d[[req.treatment]].astype(float), Wd], axis=1), has_constant="add")
        m_model = sm.OLS(d[req.mediator].astype(float).values, Xm).fit()
        a, a_se = float(m_model.params[req.treatment]), float(m_model.bse[req.treatment])
        Xy = sm.add_constant(pd.concat([d[[req.treatment, req.mediator]].astype(float), Wd], axis=1), has_constant="add")
        y_model = sm.OLS(d[req.outcome].astype(float).values, Xy).fit()
        b, b_se = float(y_model.params[req.mediator]), float(y_model.bse[req.mediator])
        cprime = float(y_model.params[req.treatment])
        return a, a_se, b, b_se, cprime

    a, a_se, b, b_se, cprime = _fit(df)
    acme = a * b
    ade = cprime
    total = acme + ade
    prop_med = (acme / total) if abs(total) > 1e-9 else float("nan")

    sobel_se = float(np.sqrt(b * b * a_se * a_se + a * a * b_se * b_se))
    sobel_z = float(acme / sobel_se) if sobel_se > 0 else float("nan")
    sobel_p = float(2 * sp.norm.sf(abs(sobel_z))) if np.isfinite(sobel_z) else float("nan")

    reps = int(req.bootstrap)
    acme_ci = ade_ci = total_ci = prop_ci = None
    if reps and reps >= 100:
        rng = np.random.default_rng(42)
        idx = np.arange(n)
        a_bs, ad_bs, t_bs, pm_bs = [], [], [], []
        dfr = df.reset_index(drop=True)
        for _ in range(reps):
            bi = rng.choice(idx, size=n, replace=True)
            try:
                ab, _, bb, _, cb = _fit(dfr.iloc[bi])
            except Exception:
                continue
            ind = ab * bb
            tot = ind + cb
            a_bs.append(ind)
            ad_bs.append(cb)
            t_bs.append(tot)
            if abs(tot) > 1e-9:
                pm_bs.append(ind / tot)
        if a_bs:
            def q(arr):
                return [round(float(np.quantile(arr, 0.025)), 5), round(float(np.quantile(arr, 0.975)), 5)]
            acme_ci, ade_ci, total_ci = q(a_bs), q(ad_bs), q(t_bs)
            prop_ci = q(pm_bs) if pm_bs else None

    acme_sig = bool(acme_ci is not None and (acme_ci[0] > 0 or acme_ci[1] < 0))
    result_text = (
        f"Causal mediation of {req.treatment} → {req.mediator} → {req.outcome} (n = {n}"
        + (f", adjusted for {', '.join(req.covariates)}" if req.covariates else "") + "). "
        f"Indirect effect (ACME = a·b) = {acme:.4f}"
        + (f" (95% bootstrap CI {acme_ci[0]} to {acme_ci[1]})" if acme_ci else "")
        + f"; direct effect (ADE) = {ade:.4f}; total effect = {total:.4f}. "
        f"Proportion mediated = {prop_med*100:.1f}%"
        + (f" (95% CI {prop_ci[0]*100:.1f}% to {prop_ci[1]*100:.1f}%)" if prop_ci else "") + ". "
        f"Sobel z = {sobel_z:.2f}, p = {_p_str(sobel_p)}. "
        + ("The indirect (mediated) effect is statistically significant (bootstrap CI excludes 0)."
           if acme_sig else "The indirect (mediated) effect is not statistically significant.")
    )

    return {
        "exposure_coding": exposure_coding,
        "test": "Causal Mediation (linear)",
        "outcome": req.outcome, "treatment": req.treatment, "mediator": req.mediator,
        "covariates": req.covariates, "n": int(n),
        "paths": {"a": round(a, 6), "a_se": round(a_se, 6), "b": round(b, 6),
                  "b_se": round(b_se, 6), "c_prime": round(cprime, 6)},
        "effects": {
            "acme": round(acme, 6), "acme_ci": acme_ci,
            "ade": round(ade, 6), "ade_ci": ade_ci,
            "total": round(total, 6), "total_ci": total_ci,
            "proportion_mediated": round(prop_med, 4) if np.isfinite(prop_med) else None,
            "proportion_mediated_ci": prop_ci,
        },
        "sobel": {"z": round(sobel_z, 4), "p": sobel_p, "se": round(sobel_se, 6)},
        "acme_significant": acme_sig,
        "result_text": result_text,
        "interpretation": result_text,
        "r_code": (
            f'library(mediation)\n'
            f'm.med <- lm({req.mediator} ~ {req.treatment}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "") + ', data = data)\n'
            f'm.out <- lm({req.outcome} ~ {req.treatment} + {req.mediator}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "") + ', data = data)\n'
            f'med <- mediate(m.med, m.out, treat = "{req.treatment}", mediator = "{req.mediator}", boot = TRUE)\n'
            f'summary(med)  # ACME, ADE, total effect, proportion mediated'
        ),
    }


# ── Target trial emulation ────────────────────────────────────────────────────

_OPS = {
    "eq": lambda s, v: s == v, "ne": lambda s, v: s != v,
    "gt": lambda s, v: s > v, "lt": lambda s, v: s < v,
    "gte": lambda s, v: s >= v, "lte": lambda s, v: s <= v,
}


class EligibilityCriterion(BaseModel):
    column: str
    op: str            # eq | ne | gt | lt | gte | lte
    value: float


class TargetTrialRequest(BaseModel):
    session_id: str
    treatment: str                       # binary 0/1 (arm assignment proxy)
    outcome: str                         # binary 0/1
    confounders: List[str]               # baseline covariates measured at time zero
    eligibility: List[EligibilityCriterion] = []
    strategies: List[str] = []           # arm labels [control, treated]; optional
    time_zero: str = "Baseline (cohort entry)"
    imputation: str = "listwise"
    bootstrap: int = 400


def _smd_bin_cont(t: np.ndarray, c: np.ndarray) -> float:
    if len(t) < 2 or len(c) < 2:
        return 0.0
    m1, m0 = float(np.mean(t)), float(np.mean(c))
    sd = float(np.sqrt((np.var(t, ddof=1) + np.var(c, ddof=1)) / 2))
    return float(abs(m1 - m0) / sd) if sd > 1e-9 else 0.0


@router.post("/target_trial")
@_states_exposure_coding
def target_trial(req: TargetTrialRequest):
    """Target-trial emulation: apply explicit eligibility, estimate the
    intention-to-treat-style average treatment effect on the eligible cohort via
    stabilized IPTW (propensity on the baseline confounders), report covariate
    balance before/after weighting, and return the 7-component target-trial
    protocol scaffold plus the standard emulation caveats.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if not req.confounders:
        raise HTTPException(400, "Specify the baseline confounders to adjust for.")
    df_full = _get_df(req.session_id)
    cols = list(dict.fromkeys([req.treatment, req.outcome] + req.confounders
                              + [c.column for c in req.eligibility]))
    miss = [c for c in cols if c not in df_full.columns]
    if miss:
        raise HTTPException(400, f"Columns not found: {miss}")

    df = apply_imputation(df_full, cols, req.imputation).reset_index(drop=True)
    n_screened = len(df)

    # ── Eligibility ──────────────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)
    applied = []
    for crit in req.eligibility:
        fn = _OPS.get(crit.op)
        if fn is None:
            raise HTTPException(400, f"Unknown eligibility op '{crit.op}'.")
        col = pd.to_numeric(df[crit.column], errors="coerce")
        mask &= fn(col, crit.value).fillna(False)
        applied.append(f"{crit.column} {crit.op} {crit.value}")
    elig = df[mask].reset_index(drop=True)
    n_eligible = len(elig)
    n_excluded_elig = n_screened - n_eligible
    if n_eligible < len(req.confounders) + 20:
        raise HTTPException(400, "Too few eligible patients after applying eligibility criteria.")

    t, exposure_coding = _numeric_or_two_level(elig[req.treatment], req.treatment)
    y = pd.to_numeric(elig[req.outcome], errors="coerce")
    keep = t.notna() & y.notna()
    elig, t, y = elig[keep].reset_index(drop=True), t[keep].astype(int).values, y[keep].astype(int).values
    if set(np.unique(t)) - {0, 1} or len(np.unique(t)) < 2:
        raise HTTPException(400, "Treatment must be binary 0/1 with both arms present.")
    if set(np.unique(y)) - {0, 1} or len(np.unique(y)) < 2:
        raise HTTPException(400, "Outcome must be binary 0/1 with both classes present.")
    n = len(t)

    # ── Propensity + stabilized ATE weights ──────────────────────────────────
    Xc = _design(elig, req.confounders)
    Xs = StandardScaler().fit_transform(Xc.values)
    ps = LogisticRegression(max_iter=1000, C=1.0).fit(Xs, t).predict_proba(Xs)[:, 1]
    ps = np.clip(ps, 1e-3, 1 - 1e-3)
    p_t = float(np.mean(t))
    w = np.where(t == 1, p_t / ps, (1 - p_t) / (1 - ps))           # stabilized ATE
    cap = np.quantile(w, 0.99)
    w = np.clip(w, None, cap)

    # ── Balance (SMD) before vs weighted-after ───────────────────────────────
    balance = []
    for col in Xc.columns:
        v = Xc[col].values.astype(float)
        smd_before = _smd_bin_cont(v[t == 1], v[t == 0])
        # weighted means/vars
        def wstat(arr, ww):
            m = np.sum(arr * ww) / np.sum(ww)
            var = np.sum(ww * (arr - m) ** 2) / np.sum(ww)
            return m, var
        m1, var1 = wstat(v[t == 1], w[t == 1])
        m0, var0 = wstat(v[t == 0], w[t == 0])
        sd = np.sqrt((var1 + var0) / 2)
        smd_after = float(abs(m1 - m0) / sd) if sd > 1e-9 else 0.0
        balance.append({"covariate": str(col), "smd_before": round(smd_before, 4),
                        "smd_after": round(smd_after, 4)})
    balanced = bool(all(b["smd_after"] < 0.1 for b in balance))

    # ── Weighted ATE: risk difference / ratio (bootstrap CI) ─────────────────
    def _effect(ti, yi, wi):
        r1 = np.sum(yi[ti == 1] * wi[ti == 1]) / np.sum(wi[ti == 1])
        r0 = np.sum(yi[ti == 0] * wi[ti == 0]) / np.sum(wi[ti == 0])
        rd = r1 - r0
        rr = (r1 / r0) if r0 > 1e-9 else float("nan")
        return r1, r0, rd, rr

    risk1, risk0, rd, rr = _effect(t, y, w)
    rd_ci = rr_ci = None
    reps = int(req.bootstrap)
    if reps and reps >= 100:
        rng = np.random.default_rng(42)
        idx = np.arange(n)
        rds, rrs = [], []
        for _ in range(reps):
            bi = rng.choice(idx, size=n, replace=True)
            tb, yb = t[bi], y[bi]
            if len(np.unique(tb)) < 2:
                continue
            try:
                psb = LogisticRegression(max_iter=500, C=1.0).fit(Xs[bi], tb).predict_proba(Xs[bi])[:, 1]
                psb = np.clip(psb, 1e-3, 1 - 1e-3)
                ptb = float(np.mean(tb))
                wb = np.where(tb == 1, ptb / psb, (1 - ptb) / (1 - psb))
                wb = np.clip(wb, None, np.quantile(wb, 0.99))
                _, _, rdb, rrb = _effect(tb, yb, wb)
                rds.append(rdb)
                if np.isfinite(rrb):
                    rrs.append(rrb)
            except Exception:
                continue
        if rds:
            rd_ci = [round(float(np.quantile(rds, 0.025)), 4), round(float(np.quantile(rds, 0.975)), 4)]
        if rrs:
            rr_ci = [round(float(np.quantile(rrs, 0.025)), 4), round(float(np.quantile(rrs, 0.975)), 4)]

    rd_sig = bool(rd_ci is not None and (rd_ci[0] > 0 or rd_ci[1] < 0))
    arm0 = req.strategies[0] if len(req.strategies) > 0 else "control / no treatment"
    arm1 = req.strategies[1] if len(req.strategies) > 1 else "treated"

    protocol = {
        "eligibility": applied if applied else ["(whole cohort — no eligibility criteria specified)"],
        "treatment_strategies": [arm0, arm1],
        "assignment": "Emulated from observed treatment; confounding by indication addressed via stabilized IPTW on the baseline confounders.",
        "time_zero": req.time_zero,
        "outcome": req.outcome,
        "causal_contrast": "Intention-to-treat-style average treatment effect (ATE) — risk difference and risk ratio.",
        "analysis_plan": "Propensity-score logistic model on the baseline confounders; stabilized ATE weights (truncated at the 99th percentile); weighted outcome risks; bootstrap 95% CIs.",
    }
    caveats = [
        "Unmeasured confounding is not addressed by IPTW — pair with an E-value sensitivity analysis.",
        "Immortal-time bias: ensure treatment status and eligibility are defined at the same time zero.",
        "Positivity: extreme weights (truncated here) flag regions of poor overlap.",
    ]

    result_text = (
        f"Target-trial emulation on the eligible cohort (screened {n_screened}, "
        f"eligible {n_eligible}{f', {n_excluded_elig} excluded by eligibility' if applied else ''}, "
        f"analysed {n}). After stabilized IPTW on {', '.join(req.confounders)}, covariate balance was "
        + ("achieved (all |SMD| < 0.10). " if balanced else "improved but not all |SMD| < 0.10. ")
        + f"Weighted {req.outcome} risk: {risk1:.3f} ({arm1}) vs {risk0:.3f} ({arm0}); "
        f"risk difference = {rd:+.3f}"
        + (f" (95% CI {rd_ci[0]:+.3f} to {rd_ci[1]:+.3f})" if rd_ci else "")
        + f", risk ratio = {rr:.3f}"
        + (f" (95% CI {rr_ci[0]:.3f} to {rr_ci[1]:.3f})" if rr_ci else "") + ". "
        + ("The effect is statistically significant (RD CI excludes 0)."
           if rd_sig else "The effect is not statistically significant (RD CI includes 0).")
    )

    return {
        "exposure_coding": exposure_coding,
        "test": "Target Trial Emulation",
        "treatment": req.treatment, "outcome": req.outcome, "confounders": req.confounders,
        "n_screened": int(n_screened), "n_eligible": int(n_eligible),
        "n_excluded_eligibility": int(n_excluded_elig), "n_analyzed": int(n),
        "protocol": protocol,
        "effect": {
            "risk_treated": round(float(risk1), 4), "risk_control": round(float(risk0), 4),
            "risk_difference": round(float(rd), 4), "rd_ci": rd_ci,
            "risk_ratio": round(float(rr), 4) if np.isfinite(rr) else None, "rr_ci": rr_ci,
            "significant": rd_sig,
        },
        "balance": balance, "balanced": balanced,
        "weight_summary": {"max": round(float(np.max(w)), 3), "mean": round(float(np.mean(w)), 3),
                           "truncated_at": round(float(cap), 3)},
        "caveats": caveats,
        "result_text": result_text,
        "interpretation": result_text,
        "r_code": (
            f'# Target trial emulation (IPTW ATE)\n'
            f'library(ipw); library(survey)\n'
            f'ps <- glm({req.treatment} ~ {" + ".join(req.confounders)}, family = binomial, data = elig)\n'
            f'elig$w <- ifelse(elig${req.treatment}==1, mean(elig${req.treatment})/fitted(ps),\n'
            f'                 (1-mean(elig${req.treatment}))/(1-fitted(ps)))\n'
            f'design <- svydesign(~1, weights = ~w, data = elig)\n'
            f'svyglm({req.outcome} ~ {req.treatment}, design, family = quasibinomial)'
        ),
    }


# ── Difference-in-Differences (2×2) ───────────────────────────────────────────

class DiDRequest(BaseModel):
    session_id: str
    outcome: str                       # continuous
    group_col: str                     # 0 = control, 1 = treated
    time_col: str                      # 0 = pre, 1 = post
    covariates: List[str] = []
    imputation: str = "listwise"


@router.post("/did")
@_states_exposure_coding
def difference_in_differences(req: DiDRequest):
    """Canonical 2×2 difference-in-differences for a continuous outcome. The
    treatment effect is the group×time interaction in OLS
    (Y ~ group + time + group·time [+ covariates]); the four cell means and the
    DiD estimate (with HC1 robust SE) are returned, plus a parallel-trends note.
    """
    if len({req.outcome, req.group_col, req.time_col}) < 3:
        raise HTTPException(400, "Outcome, group, and time must be three distinct columns.")
    df_full = _get_df(req.session_id)
    cols = list(dict.fromkeys([req.outcome, req.group_col, req.time_col] + req.covariates))
    miss = [c for c in cols if c not in df_full.columns]
    if miss:
        raise HTTPException(400, f"Columns not found: {miss}")

    df = apply_imputation(df_full, cols, req.imputation)
    df[req.outcome] = pd.to_numeric(df[req.outcome], errors="coerce")
    g, exposure_coding = _numeric_or_two_level(df[req.group_col], req.group_col)
    tm, _time_coding = _numeric_or_two_level(df[req.time_col], req.time_col)
    df = df.assign(_g_=g, _t_=tm).dropna(subset=[req.outcome, "_g_", "_t_"] + req.covariates)
    if set(df["_g_"].unique()) - {0.0, 1.0} or set(df["_t_"].unique()) - {0.0, 1.0}:
        raise HTTPException(400, "Group and time must both be binary 0/1 (control/treated, pre/post).")
    cell_counts = df.groupby(["_g_", "_t_"]).size()
    missing_cells = [(g, t) for g in (0.0, 1.0) for t in (0.0, 1.0) if cell_counts.get((g, t), 0) == 0]
    if missing_cells:
        raise HTTPException(422, "Difference-in-differences requires non-empty control/treated × pre/post cells.")
    n = len(df)
    if n < len(req.covariates) + 12:
        raise HTTPException(400, "Not enough complete observations for difference-in-differences.")

    df["_gt_"] = df["_g_"] * df["_t_"]
    Wd = _design(df, req.covariates)
    X = sm.add_constant(pd.concat([df[["_g_", "_t_", "_gt_"]].astype(float), Wd], axis=1), has_constant="add")
    model = sm.OLS(df[req.outcome].astype(float).values, X).fit(cov_type="HC1")

    did = float(model.params["_gt_"])
    se = float(model.bse["_gt_"])
    p = float(model.pvalues["_gt_"])
    ci = model.conf_int().loc["_gt_"]
    ci_low, ci_high = float(ci[0]), float(ci[1])

    def cell(gv, tv):
        sub = df[(df["_g_"] == gv) & (df["_t_"] == tv)][req.outcome]
        return round(float(sub.mean()), 4) if len(sub) else None
    means = {
        "control_pre": cell(0, 0), "control_post": cell(0, 1),
        "treated_pre": cell(1, 0), "treated_post": cell(1, 1),
    }
    ctrl_change = (means["control_post"] - means["control_pre"]) if None not in (means["control_post"], means["control_pre"]) else None
    trt_change = (means["treated_post"] - means["treated_pre"]) if None not in (means["treated_post"], means["treated_pre"]) else None
    sig = bool(p < 0.05)

    result_text = (
        f"Difference-in-differences for {req.outcome} (n = {n}"
        + (f", adjusted for {', '.join(req.covariates)}" if req.covariates else "") + "). "
        f"Treated change = {trt_change:+.4f}, control change = {ctrl_change:+.4f}; "
        f"DiD (group×time) = {did:+.4f} (95% CI {ci_low:+.4f} to {ci_high:+.4f}, p = {_p_str(p)}). "
        + ("The intervention had a statistically significant effect on the trend. "
           if sig else "No statistically significant differential change. ")
        + "Validity rests on the parallel-trends assumption (similar pre-period trajectories)."
    )

    return {
        "exposure_coding": exposure_coding,
        "test": "Difference-in-Differences (2×2)",
        "outcome": req.outcome, "group_col": req.group_col, "time_col": req.time_col,
        "covariates": req.covariates, "n": int(n),
        "did_estimate": round(did, 6), "se": round(se, 6), "p": p,
        "ci_low": round(ci_low, 6), "ci_high": round(ci_high, 6), "significant": sig,
        "cell_means": means,
        "control_change": round(ctrl_change, 4) if ctrl_change is not None else None,
        "treated_change": round(trt_change, 4) if trt_change is not None else None,
        "result_text": result_text, "interpretation": result_text,
        "r_code": (
            f'did <- lm({req.outcome} ~ {req.group_col} * {req.time_col}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "") + ', data = data)\n'
            'lmtest::coeftest(did, vcov = sandwich::vcovHC(did, "HC1"))  # group:time = DiD'
        ),
    }


# ── Sharp Regression Discontinuity (local linear) ─────────────────────────────

class RDDRequest(BaseModel):
    session_id: str
    outcome: str
    running: str                       # running / forcing variable
    cutoff: float
    bandwidth: Optional[float] = None  # None → IK-style rule-of-thumb
    imputation: str = "listwise"


@router.post("/rdd")
def regression_discontinuity(req: RDDRequest):
    """Sharp regression-discontinuity: estimate the local average treatment
    effect (LATE) at the cutoff by local-linear regression on each side, using a
    triangular kernel within the bandwidth and allowing different slopes either
    side of the threshold.
    """
    if req.outcome == req.running:
        raise HTTPException(400, "Outcome and running variable must differ.")
    df_full = _get_df(req.session_id)
    cols = [req.outcome, req.running]
    miss = [c for c in cols if c not in df_full.columns]
    if miss:
        raise HTTPException(400, f"Columns not found: {miss}")
    df = apply_imputation(df_full, cols, req.imputation)
    y = pd.to_numeric(df[req.outcome], errors="coerce")
    x = pd.to_numeric(df[req.running], errors="coerce")
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    n_total = len(d)
    if n_total < 40:
        raise HTTPException(400, "Need at least 40 complete observations for RDD.")

    c = float(req.cutoff)
    xc = d["x"].values - c
    # Bandwidth: rule of thumb (~1.84·SD·n^-1/5) if not supplied.
    bw = float(req.bandwidth) if req.bandwidth and req.bandwidth > 0 else float(1.84 * np.std(d["x"].values) * n_total ** (-0.2))
    inb = np.abs(xc) <= bw
    if int(inb.sum()) < 20:
        raise HTTPException(400, "Too few observations within the bandwidth — widen the bandwidth.")

    xb = xc[inb]
    yb = d["y"].values[inb]
    T = (xb >= 0).astype(float)               # treatment = above cutoff
    # Triangular kernel weights.
    w = np.clip(1.0 - np.abs(xb) / bw, 0, None)
    # Local linear with treatment, running, and interaction (different slopes).
    X = np.column_stack([np.ones_like(xb), T, xb, T * xb])
    wls = sm.WLS(yb, X, weights=w).fit(cov_type="HC1")
    late = float(wls.params[1])               # discontinuity at cutoff = T coefficient
    se = float(wls.bse[1])
    p = float(wls.pvalues[1])
    ci = wls.conf_int()[1]
    ci_low, ci_high = float(ci[0]), float(ci[1])
    n_left = int(np.sum(xb < 0))
    n_right = int(np.sum(xb >= 0))
    sig = bool(p < 0.05)

    result_text = (
        f"Sharp regression-discontinuity for {req.outcome} at {req.running} = {c:g} "
        f"(bandwidth ±{bw:.3g}, n = {int(inb.sum())}: {n_left} below / {n_right} at-or-above; triangular kernel). "
        f"LATE at the cutoff = {late:+.4f} (95% CI {ci_low:+.4f} to {ci_high:+.4f}, p = {_p_str(p)}). "
        + ("A statistically significant discontinuity — evidence of a local causal effect at the threshold. "
           if sig else "No statistically significant discontinuity at the threshold. ")
        + "Assumes units cannot precisely manipulate the running variable around the cutoff."
    )

    # Binned scatter (for a plot): mean y in equal-width running bins within 2·bw.
    plot_lo, plot_hi = c - 2 * bw, c + 2 * bw
    pm = (d["x"] >= plot_lo) & (d["x"] <= plot_hi)
    bins = np.linspace(plot_lo, plot_hi, 21)
    binned = []
    dp = d[pm]
    if len(dp):
        idxb = np.digitize(dp["x"].values, bins)
        for bidx in range(1, len(bins)):
            sel = idxb == bidx
            if sel.sum() >= 1:
                binned.append({"x": round(float(np.mean(dp["x"].values[sel])), 4),
                               "y": round(float(np.mean(dp["y"].values[sel])), 4),
                               "n": int(sel.sum())})

    return {
        "test": "Regression Discontinuity (sharp, local linear)",
        "outcome": req.outcome, "running": req.running, "cutoff": c,
        "bandwidth": round(bw, 6), "n_in_bandwidth": int(inb.sum()),
        "n_left": n_left, "n_right": n_right,
        "late": round(late, 6), "se": round(se, 6), "p": p,
        "ci_low": round(ci_low, 6), "ci_high": round(ci_high, 6), "significant": sig,
        "binned": binned,
        "result_text": result_text, "interpretation": result_text,
        "r_code": (
            f'library(rdrobust)\n'
            f'rdrobust(data${req.outcome}, data${req.running}, c = {c})  # local-linear LATE'
        ),
    }


# ── DAG backdoor analysis ─────────────────────────────────────────────────────

class DAGRequest(BaseModel):
    edges: List[List[str]]             # directed edges [[from, to], ...]
    treatment: str
    outcome: str


@router.post("/dag_adjustment")
def dag_adjustment(req: DAGRequest):
    """Analyse a user-specified causal DAG: classify each other node relative to
    the treatment→outcome effect (confounder / mediator / collider / other) and
    return a valid minimal adjustment set via the backdoor criterion. Pure graph
    logic — no dataset needed.
    """
    edges = [(str(a), str(b)) for e in req.edges if len(e) == 2 for a, b in [e]]
    nodes = set()
    children: Dict[str, set] = {}
    parents: Dict[str, set] = {}
    for a, b in edges:
        nodes.update([a, b])
        children.setdefault(a, set()).add(b)
        parents.setdefault(b, set()).add(a)
    # Validate against the edge-derived nodes BEFORE forcing them in.
    if req.treatment not in nodes or req.outcome not in nodes:
        raise HTTPException(400, "Treatment and outcome must appear in the edges.")
    if req.treatment == req.outcome:
        raise HTTPException(400, "Treatment and outcome must differ.")

    def descendants(start: str) -> set:
        seen, stack = set(), [start]
        while stack:
            u = stack.pop()
            for v in children.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def ancestors(start: str) -> set:
        seen, stack = set(), [start]
        while stack:
            u = stack.pop()
            for v in parents.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    desc_T = descendants(req.treatment)
    anc_T, anc_Y = ancestors(req.treatment), ancestors(req.outcome)
    roles = {}
    for nd in nodes - {req.treatment, req.outcome}:
        is_confounder = (nd in anc_T) and (nd in anc_Y) and (nd not in desc_T)
        is_mediator = (nd in desc_T) and (req.outcome in descendants(nd) or req.outcome in children.get(nd, set()))
        if is_confounder:
            roles[nd] = "confounder"
        elif is_mediator:
            roles[nd] = "mediator"
        elif len(parents.get(nd, set())) >= 2:
            roles[nd] = "collider"
        else:
            roles[nd] = "other"

    # Minimal adjustment set (backdoor): adjust for confounders; never for
    # mediators or colliders (or their descendants).
    adjust = sorted([nd for nd, r in roles.items() if r == "confounder"])
    do_not_adjust = sorted([nd for nd, r in roles.items() if r in ("mediator", "collider")])

    result_text = (
        f"Backdoor analysis for {req.treatment} → {req.outcome}. "
        f"Minimal adjustment set: {', '.join(adjust) if adjust else '∅ (no confounders identified)'}. "
        + (f"Do NOT adjust for {', '.join(do_not_adjust)} — "
           "adjusting for a mediator removes part of the effect; adjusting for a collider opens a spurious path. "
           if do_not_adjust else "")
        + "Estimate the effect of the treatment controlling for exactly the adjustment-set variables."
    )

    return {
        "test": "DAG Backdoor Adjustment",
        "treatment": req.treatment, "outcome": req.outcome,
        "nodes": sorted(nodes), "n_edges": len(edges),
        "roles": roles,
        "adjustment_set": adjust,
        "do_not_adjust": do_not_adjust,
        "result_text": result_text, "interpretation": result_text,
        "r_code": (
            'library(dagitty)\n'
            'g <- dagitty("dag { ' + " ; ".join(f"{a} -> {b}" for a, b in edges) + ' }")\n'
            f'adjustmentSets(g, exposure = "{req.treatment}", outcome = "{req.outcome}", type = "minimal")'
        ),
    }
