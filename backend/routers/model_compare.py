"""Model comparison: nested likelihood ratio tests and side-by-side model comparison."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp
from sklearn.metrics import brier_score_loss, roc_auc_score
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from services import store
from services.impute import apply_imputation

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _fit_model(df: pd.DataFrame, outcome: str, predictors: List[str], model_type: str):
    """Fit a logistic or linear model, return the fitted model object."""
    X = pd.get_dummies(df[predictors], drop_first=True).astype(float)
    X = sm.add_constant(X)
    y = df[outcome].astype(float)

    if model_type == "logistic":
        if y.nunique() < 2:
            raise HTTPException(400, "Outcome must be binary (0/1) with at least one event and one non-event.")
        try:
            model = sm.Logit(y, X).fit(disp=False, maxiter=100)
        except Exception as exc:
            raise HTTPException(400, f"Logistic regression failed: {exc}")
    elif model_type == "linear":
        try:
            model = sm.OLS(y, X).fit()
        except Exception as exc:
            raise HTTPException(400, f"Linear regression failed: {exc}")
    else:
        raise HTTPException(400, f"Unknown model_type '{model_type}'. Use 'logistic' or 'linear'.")

    return model, X, y


def _aic_bic(model, model_type: str) -> tuple[float, float]:
    """AIC and BIC as the model screens report them.

    The linear model counts the residual variance as a parameter (as R does
    and as /api/models/linear does); statsmodels' own OLS aic leaves it out,
    so the same model read 2 lower here than on the regression screen.
    """
    if model_type == "linear":
        from routers.models.linear import information_criteria

        return information_criteria(model, n_params=int(model.df_model) + 2, n_obs=int(model.nobs))
    return float(model.aic), float(model.bic)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NESTED LIKELIHOOD RATIO TEST
# ═══════════════════════════════════════════════════════════════════════════════

class NestedLRRequest(BaseModel):
    session_id: str
    outcome: str
    predictors_reduced: List[str]
    predictors_full: List[str]
    model_type: str = "logistic"
    imputation: str = "listwise"


@router.post("/nested_lr_test")
def nested_lr_test(req: NestedLRRequest):
    # Validate nesting: reduced predictors must be a subset of full predictors
    if not set(req.predictors_reduced).issubset(set(req.predictors_full)):
        raise HTTPException(400, "Reduced model predictors must be a subset of full model predictors.")

    if set(req.predictors_reduced) == set(req.predictors_full):
        raise HTTPException(400, "Reduced and full models have identical predictors — nothing to test.")

    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    all_cols = list(set([req.outcome] + req.predictors_full))
    df = apply_imputation(df_full, all_cols, req.imputation)
    n_excluded = n_total - len(df)

    if len(df) < len(req.predictors_full) + 10:
        raise HTTPException(400, "Not enough observations relative to number of predictors.")

    model_red, X_red, y_red = _fit_model(df, req.outcome, req.predictors_reduced, req.model_type)
    model_full, X_full, y_full = _fit_model(df, req.outcome, req.predictors_full, req.model_type)

    # ── Log-likelihoods ────────────────────────────────────────────────────
    ll_red = float(model_red.llf)
    ll_full = float(model_full.llf)

    n_params_red = int(model_red.df_model) + 1  # +1 for intercept
    n_params_full = int(model_full.df_model) + 1

    aic_red, bic_red = _aic_bic(model_red, req.model_type)
    aic_full, bic_full = _aic_bic(model_full, req.model_type)

    # ── LR test statistic ──────────────────────────────────────────────────
    lr_stat = -2 * (ll_red - ll_full)
    df_test = n_params_full - n_params_red
    if df_test <= 0:
        raise HTTPException(400, "Full model must have more parameters than reduced model.")

    p_val = float(sp.chi2.sf(lr_stat, df_test))
    sig = bool(p_val < 0.05)
    ps = _p_str(p_val)

    # ── Interpretation ─────────────────────────────────────────────────────
    added_vars = sorted(set(req.predictors_full) - set(req.predictors_reduced))
    added_str = ", ".join(added_vars)

    if sig:
        interpretation = (
            f"The full model (with {added_str}) provides a significantly better fit than the "
            f"reduced model (LR chi2({df_test}) = {lr_stat:.3f}, p = {ps}). "
            f"The additional predictor(s) contribute meaningfully to the model."
        )
    else:
        interpretation = (
            f"The full model (with {added_str}) does not significantly improve fit over the "
            f"reduced model (LR chi2({df_test}) = {lr_stat:.3f}, p = {ps}). "
            f"The simpler model may be preferred."
        )

    result_text = (
        f"Likelihood ratio test comparing nested {req.model_type} models (n = {len(df)}, {n_excluded} excluded). "
        f"Reduced model: {', '.join(req.predictors_reduced)} (LL = {ll_red:.2f}, AIC = {aic_red:.1f}). "
        f"Full model: {', '.join(req.predictors_full)} (LL = {ll_full:.2f}, AIC = {aic_full:.1f}). "
        f"LR chi2({df_test}) = {lr_stat:.3f}, p = {ps}."
    )

    return {
        "test": "Likelihood Ratio Test (nested models)",
        "reduced": {
            "predictors": req.predictors_reduced,
            "ll": round(ll_red, 4),
            "aic": round(aic_red, 4),
            "bic": round(bic_red, 4),
            "n_params": n_params_red,
        },
        "full": {
            "predictors": req.predictors_full,
            "ll": round(ll_full, 4),
            "aic": round(aic_full, 4),
            "bic": round(bic_full, 4),
            "n_params": n_params_full,
        },
        "lr_stat": round(float(lr_stat), 4),
        "df": df_test,
        "p": float(p_val),
        "significant": sig,
        "n": len(df),
        "n_excluded": n_excluded,
        "interpretation": interpretation,
        "result_text": result_text,
        "export_rows": [
            ["Statistic", "Value"],
            ["LR chi-square", round(float(lr_stat), 4)],
            ["df", df_test],
            ["p", round(float(p_val), 6)],
            ["Reduced LL", round(ll_red, 4)],
            ["Full LL", round(ll_full, 4)],
            ["Reduced AIC", round(aic_red, 4)],
            ["Full AIC", round(aic_full, 4)],
            ["Reduced BIC", round(bic_red, 4)],
            ["Full BIC", round(bic_full, 4)],
            ["Reduced n_params", n_params_red],
            ["Full n_params", n_params_full],
            ["n", len(df)],
            ["n excluded", n_excluded],
        ],
        "r_code": "anova(model_reduced, model_full, test = 'LRT')",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPARE MODELS (side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════

class CompareModelsRequest(BaseModel):
    session_id: str
    outcome: str
    model_specs: List[dict]  # [{"name": "Model 1", "predictors": ["x1", "x2"]}, ...]
    model_type: str = "logistic"
    imputation: str = "listwise"


@router.post("/compare_models")
def compare_models(req: CompareModelsRequest):
    if len(req.model_specs) < 2:
        raise HTTPException(400, "Need at least 2 model specifications to compare.")
    if len(req.model_specs) > 20:
        raise HTTPException(400, "Maximum 20 models for comparison.")

    df_full = _get_df(req.session_id)
    n_total = len(df_full)

    # Collect all predictor columns to impute uniformly
    all_pred_cols = set()
    for spec in req.model_specs:
        preds = spec.get("predictors", [])
        if not preds:
            raise HTTPException(400, f"Model '{spec.get('name', '?')}' has no predictors.")
        all_pred_cols.update(preds)
    all_cols = list(set([req.outcome]) | all_pred_cols)

    df = apply_imputation(df_full, all_cols, req.imputation)
    n_excluded = n_total - len(df)

    if len(df) < 20:
        raise HTTPException(400, "Need at least 20 complete observations for model comparison.")

    models_out = []
    for spec in req.model_specs:
        name = spec.get("name", f"Model {len(models_out) + 1}")
        predictors = spec["predictors"]

        try:
            fitted, X, y = _fit_model(df, req.outcome, predictors, req.model_type)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, f"Model '{name}' fitting failed: {exc}")

        n_params = int(fitted.df_model) + 1
        ll = round(float(fitted.llf), 4)
        aic, bic = (round(v, 4) for v in _aic_bic(fitted, req.model_type))

        entry = {
            "name": name,
            "predictors": predictors,
            "n": len(df),
            "n_params": n_params,
            "aic": aic,
            "bic": bic,
            "ll": ll,
        }

        if req.model_type == "logistic":
            # Pseudo R-squared (McFadden)
            pseudo_r2 = round(float(fitted.prsquared), 4)
            entry["r_squared"] = pseudo_r2

            # C-statistic (AUC)
            probs = fitted.predict(X).values
            y_vals = y.values
            try:
                c_stat = round(float(roc_auc_score(y_vals, probs)), 4)
            except ValueError:
                c_stat = None
            entry["c_statistic"] = c_stat

            # Brier score
            brier = round(float(brier_score_loss(y_vals, probs)), 4)
            entry["brier_score"] = brier

        elif req.model_type == "linear":
            r2 = round(float(fitted.rsquared), 4)
            r2_adj = round(float(fitted.rsquared_adj), 4)
            entry["r_squared"] = r2
            entry["r_squared_adj"] = r2_adj
            entry["c_statistic"] = None
            entry["brier_score"] = None

        models_out.append(entry)

    # ── Rank by AIC (lower = better) ───────────────────────────────────────
    aic_vals = [m["aic"] for m in models_out]
    sorted_indices = np.argsort(aic_vals)
    for rank, idx in enumerate(sorted_indices):
        models_out[idx]["rank_aic"] = rank + 1

    best_idx = int(sorted_indices[0])
    best_model = models_out[best_idx]["name"]

    # ── Result text ────────────────────────────────────────────────────────
    model_summaries = []
    for m in sorted(models_out, key=lambda x: x["rank_aic"]):
        parts = [f"{m['name']} (AIC = {m['aic']}, BIC = {m['bic']}"]
        if m.get("c_statistic") is not None:
            parts.append(f"C = {m['c_statistic']}")
        if m.get("r_squared") is not None:
            r2_label = "pseudo-R2" if req.model_type == "logistic" else "R2"
            parts.append(f"{r2_label} = {m['r_squared']}")
        model_summaries.append(", ".join(parts) + ")")

    result_text = (
        f"Model comparison of {len(req.model_specs)} {req.model_type} models "
        f"(n = {len(df)}, {n_excluded} excluded). "
        f"Best model by AIC: {best_model}. "
        + " | ".join(model_summaries)
    )

    # ── Export rows ────────────────────────────────────────────────────────
    header = ["Model", "Predictors", "n_params", "AIC", "BIC", "LL", "R2/pseudo-R2"]
    if req.model_type == "logistic":
        header += ["C-statistic", "Brier"]
    header.append("Rank (AIC)")

    export_rows = [header]
    for m in sorted(models_out, key=lambda x: x["rank_aic"]):
        row = [
            m["name"],
            ", ".join(m["predictors"]),
            m["n_params"],
            m["aic"],
            m["bic"],
            m["ll"],
            m["r_squared"],
        ]
        if req.model_type == "logistic":
            row += [m.get("c_statistic"), m.get("brier_score")]
        row.append(m["rank_aic"])
        export_rows.append(row)

    # ── R code ─────────────────────────────────────────────────────────────
    r_lines = []
    for i, spec in enumerate(req.model_specs):
        preds_str = " + ".join(spec["predictors"])
        fam = "family = binomial" if req.model_type == "logistic" else ""
        func = "glm" if req.model_type == "logistic" else "lm"
        fam_arg = f", {fam}" if fam else ""
        r_lines.append(f"m{i+1} <- {func}({req.outcome} ~ {preds_str}, data = data{fam_arg})")
    r_lines.append(f"AIC({', '.join(f'm{i+1}' for i in range(len(req.model_specs)))})")
    r_lines.append(f"BIC({', '.join(f'm{i+1}' for i in range(len(req.model_specs)))})")
    r_code = "\n".join(r_lines)

    return {
        "test": "Model Comparison",
        "models": models_out,
        "best_model": best_model,
        "n": len(df),
        "n_excluded": n_excluded,
        "result_text": result_text,
        "export_rows": export_rows,
        "r_code": r_code,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ADDED PREDICTIVE VALUE  (incremental value of a new predictor)
# ═══════════════════════════════════════════════════════════════════════════════

class AddedValueRequest(BaseModel):
    session_id: str
    outcome: str                       # binary 0/1
    base_predictors: List[str]         # known/established predictors
    new_predictors: List[str]          # candidate predictor(s) to evaluate
    model_type: str = "logistic"       # v1: logistic only
    imputation: str = "listwise"
    cv_folds: int = 0                  # 0 = apparent (in-sample); ≥5 = honest k-fold CV ΔAUC
    bootstrap: int = 400               # bootstrap reps for NRI/IDI CIs (0 = skip)


def _logit_oof_probs(X: pd.DataFrame, y: np.ndarray, cv_folds: int) -> np.ndarray:
    """Cross-validated out-of-fold predicted probabilities (statsmodels Logit per fold)."""
    from sklearn.model_selection import StratifiedKFold
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    for tr, te in skf.split(X.values, y):
        try:
            m = sm.Logit(y[tr], X.iloc[tr]).fit(disp=False, maxiter=100)
            oof[te] = np.asarray(m.predict(X.iloc[te]))
        except Exception:
            oof[te] = float(np.mean(y[tr]))  # degenerate fold → base rate
    return oof


def _nagelkerke(fitted, n: int) -> float:
    try:
        llf, lln = float(fitted.llf), float(fitted.llnull)
        cox_snell = 1.0 - np.exp((lln - llf) * 2.0 / n)
        denom = 1.0 - np.exp(lln * 2.0 / n)
        return float(cox_snell / denom) if denom > 0 else float("nan")
    except Exception:
        return float("nan")


@router.post("/added_value")
def added_value(req: AddedValueRequest):
    """Quantify the incremental predictive value of adding new predictor(s) to a
    base model — the right way to judge a "predictor", beyond a significant
    coefficient p-value. Reports ΔAUC (DeLong test), continuous NRI, IDI, the
    nested likelihood-ratio test, ΔAIC/BIC, pseudo-R² change, and the calibration
    of both models, plus an integrated narrative.
    """
    from routers.stats.nonparametric import _delong_compare
    from services.model_validation import compute_calibration_slope_intercept

    if req.model_type != "logistic":
        raise HTTPException(400, "added_value currently supports model_type='logistic' (binary outcome).")
    if not req.new_predictors:
        raise HTTPException(400, "Provide at least one new predictor to evaluate.")

    base = list(dict.fromkeys(req.base_predictors))
    new = [p for p in dict.fromkeys(req.new_predictors) if p not in base]
    if not new:
        raise HTTPException(400, "New predictor(s) are already in the base model.")
    full = base + new
    if not base:
        raise HTTPException(400, "Base model needs at least one predictor (the established factors).")

    df_full = _get_df(req.session_id)
    n_total = len(df_full)
    cols = list(dict.fromkeys([req.outcome] + full))
    missing = [c for c in cols if c not in df_full.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    df = apply_imputation(df_full, cols, req.imputation)
    n_excluded = n_total - len(df)
    if len(df) < len(full) + 10:
        raise HTTPException(400, "Not enough complete observations relative to the number of predictors.")

    # Fit both models (apparent fit drives likelihood-based metrics).
    m_base, X_base, y_s = _fit_model(df, req.outcome, base, "logistic")
    m_full, X_full, _ = _fit_model(df, req.outcome, full, "logistic")
    y = y_s.values.astype(int)
    if set(np.unique(y)) - {0, 1} or len(np.unique(y)) < 2:
        raise HTTPException(400, "Outcome must be binary 0/1 with both classes present.")

    # Predicted probabilities — apparent or cross-validated (honest).
    cv = int(req.cv_folds)
    if cv and cv >= 3:
        p_base = _logit_oof_probs(X_base, y, cv)
        p_full = _logit_oof_probs(X_full, y, cv)
        pred_basis = f"{cv}-fold cross-validated"
    else:
        p_base = np.asarray(m_base.predict(X_base))
        p_full = np.asarray(m_full.predict(X_full))
        pred_basis = "apparent (in-sample)"

    # ── Discrimination: AUCs + DeLong ΔAUC (full vs base) ────────────────────
    delong = _delong_compare(y, p_full, p_base)
    auc_base, auc_full = delong["auc_2"], delong["auc_1"]
    delta_auc = delong["difference"]
    delong_p = delong["p"]

    # ── Reclassification: continuous NRI + IDI ───────────────────────────────
    def _nri_idi(pf: np.ndarray, pb: np.ndarray) -> tuple:
        ev, ne = y == 1, y == 0
        d = pf - pb
        nri_e = (np.mean(d[ev] > 0) - np.mean(d[ev] < 0)) if ev.any() else 0.0
        nri_ne = (np.mean(d[ne] < 0) - np.mean(d[ne] > 0)) if ne.any() else 0.0
        nri = nri_e + nri_ne
        idi = (np.mean(pf[ev]) - np.mean(pb[ev])) - (np.mean(pf[ne]) - np.mean(pb[ne]))
        return float(nri), float(idi), float(nri_e), float(nri_ne)

    nri, idi, nri_ev, nri_ne = _nri_idi(p_full, p_base)

    nri_ci = idi_ci = None
    reps = int(req.bootstrap)
    if reps and reps >= 50:
        rng = np.random.default_rng(42)
        nris, idis = [], []
        idx_all = np.arange(len(y))
        for _ in range(reps):
            bi = rng.choice(idx_all, size=len(idx_all), replace=True)
            if len(np.unique(y[bi])) < 2:
                continue
            # predictions are fixed per subject; resample subjects with replacement
            ev, ne = y[bi] == 1, y[bi] == 0
            d = p_full[bi] - p_base[bi]
            nb = ((np.mean(d[ev] > 0) - np.mean(d[ev] < 0)) if ev.any() else 0.0) + \
                 ((np.mean(d[ne] < 0) - np.mean(d[ne] > 0)) if ne.any() else 0.0)
            ib = (np.mean(p_full[bi][ev]) - np.mean(p_base[bi][ev])) - \
                 (np.mean(p_full[bi][ne]) - np.mean(p_base[bi][ne]))
            nris.append(nb)
            idis.append(ib)
        if nris:
            nri_ci = [round(float(np.quantile(nris, 0.025)), 4), round(float(np.quantile(nris, 0.975)), 4)]
            idi_ci = [round(float(np.quantile(idis, 0.025)), 4), round(float(np.quantile(idis, 0.975)), 4)]

    # ── Nested likelihood-ratio test + information criteria (apparent fit) ────
    ll_base, ll_full = float(m_base.llf), float(m_full.llf)
    k_base, k_full = int(m_base.df_model) + 1, int(m_full.df_model) + 1
    lr_stat = -2.0 * (ll_base - ll_full)
    df_test = k_full - k_base
    lr_p = float(sp.chi2.sf(lr_stat, df_test)) if df_test > 0 else 1.0
    daic = float(m_full.aic) - float(m_base.aic)
    dbic = float(m_full.bic) - float(m_base.bic)
    r2_base, r2_full = _nagelkerke(m_base, len(y)), _nagelkerke(m_full, len(y))

    # ── Calibration of each model ────────────────────────────────────────────
    cal_base = compute_calibration_slope_intercept(y, probs=np.clip(p_base, 1e-6, 1 - 1e-6))
    cal_full = compute_calibration_slope_intercept(y, probs=np.clip(p_full, 1e-6, 1 - 1e-6))
    brier_base = round(float(brier_score_loss(y, p_base)), 4)
    brier_full = round(float(brier_score_loss(y, p_full)), 4)

    # ── Integrated verdict ───────────────────────────────────────────────────
    disc_better = bool(delta_auc > 0 and delong_p < 0.05)
    idi_pos = bool(idi > 0 and (idi_ci is None or idi_ci[0] > 0))
    slope_full = cal_full["calibration_slope"]
    calib_ok = bool(0.8 <= slope_full <= 1.2)
    added = bool(disc_better or idi_pos or (lr_p < 0.05 and daic < 0))
    new_str = ", ".join(new)

    verdict = (
        f"Adding {new_str} to the base model "
        + ("**improved predictive performance**" if added else "**did not meaningfully improve predictive performance**")
        + f". Discrimination: AUC {auc_base:.3f} → {auc_full:.3f} (ΔAUC = {delta_auc:+.3f}, DeLong p = {_p_str(delong_p)}). "
        + f"Reclassification: IDI = {idi:+.4f}"
        + (f" (95% CI {idi_ci[0]:+.4f} to {idi_ci[1]:+.4f})" if idi_ci else "")
        + f", continuous NRI = {nri:+.4f}"
        + (f" (95% CI {nri_ci[0]:+.4f} to {nri_ci[1]:+.4f})" if nri_ci else "")
        + f". Overall fit: LR χ²({df_test}) = {lr_stat:.2f}, p = {_p_str(lr_p)}, ΔAIC = {daic:+.1f}, "
        + f"Nagelkerke R² {r2_base:.3f} → {r2_full:.3f}. "
        + f"Calibration of the full model: slope = {slope_full:.2f}, intercept = {cal_full['calibration_intercept']:.2f}, "
        + f"Brier {brier_base:.3f} → {brier_full:.3f} "
        + ("(calibration preserved)." if calib_ok else "(calibration slope outside 0.8–1.2 — check calibration plot).")
        + " Note: a significant coefficient p-value alone does not establish added predictive value — "
        + "the evidence above (discrimination + reclassification + calibration) does."
    )

    return {
        "test": "Added Predictive Value",
        "outcome": req.outcome,
        "base_predictors": base,
        "new_predictors": new,
        "model_type": "logistic",
        "n": int(len(df)),
        "n_excluded": int(n_excluded),
        "prediction_basis": pred_basis,
        "discrimination": {
            "auc_base": auc_base, "auc_full": auc_full, "delta_auc": delta_auc,
            "delong_z": delong["z"], "delong_p": delong_p,
            "delta_auc_ci": [delong["ci_diff_low"], delong["ci_diff_high"]],
            "significant": bool(delong_p < 0.05),
        },
        "reclassification": {
            "nri": round(nri, 4), "nri_ci": nri_ci,
            "nri_events": round(nri_ev, 4), "nri_nonevents": round(nri_ne, 4),
            "idi": round(idi, 4), "idi_ci": idi_ci,
        },
        "fit": {
            "lr_stat": round(lr_stat, 4), "df": df_test, "lr_p": lr_p,
            "delta_aic": round(daic, 4), "delta_bic": round(dbic, 4),
            "nagelkerke_base": round(r2_base, 4), "nagelkerke_full": round(r2_full, 4),
        },
        "calibration": {
            "base": {**cal_base, "brier": brier_base},
            "full": {**cal_full, "brier": brier_full},
            "preserved": calib_ok,
        },
        "added_value": added,
        "interpretation": verdict,
        "result_text": verdict,
        "r_code": (
            f'library(pROC); library(PredictABEL)\n'
            f'base <- glm({req.outcome} ~ {" + ".join(base)}, family = binomial, data = data)\n'
            f'full <- glm({req.outcome} ~ {" + ".join(full)}, family = binomial, data = data)\n'
            f'roc.test(roc(data${req.outcome}, fitted(base)), roc(data${req.outcome}, fitted(full)))  # DeLong\n'
            f'anova(base, full, test = "LRT")\n'
            f'reclassification(data, cOutcome, fitted(base), fitted(full), ...)  # NRI / IDI'
        ),
    }
