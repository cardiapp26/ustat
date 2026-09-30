"""Advanced ANOVA: ANCOVA, two-way ANOVA with interaction + estimated marginal means."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats
from typing import List
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from services import store
from services.impute import apply_imputation
from services.stat_utils import (
    partial_eta_squared, check_normality, group_summary, sorted_groups,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _safe_col(name: str) -> str:
    """Make column name safe for statsmodels formula (wrap in Q())."""
    return f"Q('{name}')"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ANCOVA
# ═══════════════════════════════════════════════════════════════════════════════

class AncovaRequest(BaseModel):
    session_id: str
    outcome: str
    group_col: str
    covariates: List[str]
    alpha: float = 0.05
    imputation: str = "listwise"


@router.post("/ancova")
def ancova(req: AncovaRequest):
    df_full = _get_df(req.session_id)
    cols = [req.outcome, req.group_col] + req.covariates
    for c in cols:
        if c not in df_full.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    df = apply_imputation(df_full, cols, req.imputation)
    df[req.outcome] = pd.to_numeric(df[req.outcome], errors="coerce")
    for cov in req.covariates:
        df[cov] = pd.to_numeric(df[cov], errors="coerce")
    df = df.dropna(subset=cols)

    if len(df) < 10:
        raise HTTPException(400, "Need at least 10 complete rows.")

    groups = sorted_groups(df[req.group_col])
    if len(groups) < 2:
        raise HTTPException(400, "Group column must have at least 2 levels.")

    # Build formula: outcome ~ C(group) + cov1 + cov2 + ...
    cov_terms = " + ".join([_safe_col(c) for c in req.covariates])
    formula = f"{_safe_col(req.outcome)} ~ C({_safe_col(req.group_col)}) + {cov_terms}"

    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as exc:
        raise HTTPException(400, f"Model fitting error: {exc}")

    # Type II ANOVA table
    try:
        aov = anova_lm(model, typ=2)
    except Exception:
        aov = anova_lm(model, typ=1)

    # Extract group effect
    group_key = [k for k in aov.index if req.group_col in str(k)]
    if not group_key:
        raise HTTPException(500, "Could not find group effect in ANOVA table.")

    gk = group_key[0]
    F_val = float(aov.loc[gk, "F"])
    p_val = float(aov.loc[gk, "PR(>F)"])
    df_num = int(aov.loc[gk, "df"])
    df_den = int(aov.loc["Residual", "df"])
    sig = bool(p_val < req.alpha)
    es = partial_eta_squared(F_val, df_num, df_den)
    ps = _p_str(p_val)

    # Assumption checks
    assumptions = [check_normality(model.resid.values, "Residuals")]
    # Homogeneity of regression slopes: test group x covariate interaction
    for cov in req.covariates:
        try:
            int_formula = f"{_safe_col(req.outcome)} ~ C({_safe_col(req.group_col)}) * {_safe_col(cov)}"
            int_model = smf.ols(int_formula, data=df).fit()
            int_aov = anova_lm(int_model, typ=2)
            int_key = [k for k in int_aov.index if req.group_col in str(k) and cov in str(k)]
            if int_key:
                int_p = float(int_aov.loc[int_key[0], "PR(>F)"])
                met = bool(int_p >= 0.05)
                assumptions.append({
                    "name": f"Homogeneity of slopes ({cov})",
                    "met": met,
                    "detail": f"Interaction p = {int_p:.4f}" + (" — ANCOVA assumption violated" if not met else ""),
                })
        except Exception:
            pass

    # Estimated marginal means (EMMs)
    cov_means = {c: float(df[c].mean()) for c in req.covariates}
    emms = []
    for g in sorted(groups):
        row = {req.group_col: g, **cov_means}
        pred_df = pd.DataFrame([row])
        try:
            emm = float(model.predict(pred_df).iloc[0])
        except Exception:
            emm = None
        emms.append({"group": str(g), "emm": round(emm, 4) if emm else None})

    # Covariate effects
    cov_effects = []
    for cov in req.covariates:
        cov_key = [k for k in aov.index if cov in str(k) and req.group_col not in str(k)]
        if cov_key:
            ck = cov_key[0]
            cov_effects.append({
                "covariate": cov,
                "F": round(float(aov.loc[ck, "F"]), 4),
                "p": round(float(aov.loc[ck, "PR(>F)"]), 6),
                "significant": bool(aov.loc[ck, "PR(>F)"] < req.alpha),
            })

    # ANOVA table as export rows
    export_rows = [["Source", "SS", "df", "F", "p"]]
    for idx_name, row in aov.iterrows():
        if str(idx_name) != "Residual":
            export_rows.append([str(idx_name), round(float(row["sum_sq"]), 4),
                                int(row["df"]), round(float(row["F"]), 4), round(float(row["PR(>F)"]), 6)])
    export_rows.append(["Residual", round(float(aov.loc["Residual", "sum_sq"]), 4),
                         int(aov.loc["Residual", "df"]), "", ""])

    cov_list = ", ".join(req.covariates)
    return {
        "test": "ANCOVA",
        "F": round(F_val, 4), "df_num": df_num, "df_den": df_den, "p": float(p_val),
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": assumptions,
        "emms": emms,
        "covariate_effects": cov_effects,
        "summary": {str(g): group_summary(df[df[req.group_col] == g][req.outcome].values, str(g))
                    for g in sorted(groups)},
        "interpretation": (
            f"After controlling for {cov_list}, there was {'a significant' if sig else 'no significant'} "
            f"effect of {req.group_col} on {req.outcome} "
            f"(F({df_num},{df_den}) = {F_val:.2f}, p = {ps}, partial \u03B7\u00B2 = {es['value']:.3f} [{es['magnitude']}])"
        ),
        "result_text": (
            f"An ANCOVA was conducted with {req.outcome} as the dependent variable, {req.group_col} as the factor, "
            f"and {cov_list} as covariate(s). After controlling for the covariate(s), there was "
            f"{'a significant' if sig else 'no significant'} effect of {req.group_col} "
            f"(F({df_num},{df_den}) = {F_val:.2f}, p = {ps}, partial \u03B7\u00B2 = {es['value']:.3f} [{es['magnitude']}]). "
            f"Estimated marginal means: " + ", ".join([f"{e['group']} = {e['emm']}" for e in emms if e['emm']]) + "."
        ),
        "export_rows": export_rows,
        "r_code": (
            f'library(emmeans)\n'
            f'model <- lm({req.outcome} ~ {req.group_col} + {" + ".join(req.covariates)}, data = data)\n'
            f'anova(model)\n'
            f'emmeans(model, ~ {req.group_col})'
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TWO-WAY ANOVA
# ═══════════════════════════════════════════════════════════════════════════════

class TwoWayAnovaRequest(BaseModel):
    session_id: str
    outcome: str
    factor1: str
    factor2: str
    alpha: float = 0.05
    imputation: str = "listwise"


@router.post("/two_way_anova")
def two_way_anova(req: TwoWayAnovaRequest):
    df_full = _get_df(req.session_id)
    cols = [req.outcome, req.factor1, req.factor2]
    for c in cols:
        if c not in df_full.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    df = apply_imputation(df_full, cols, req.imputation)
    df[req.outcome] = pd.to_numeric(df[req.outcome], errors="coerce")
    df = df.dropna(subset=cols)

    if len(df) < 12:
        raise HTTPException(400, "Need at least 12 complete rows.")

    k1 = df[req.factor1].nunique()
    k2 = df[req.factor2].nunique()
    if k1 < 2 or k2 < 2:
        raise HTTPException(400, "Both factors must have at least 2 levels.")

    # Full factorial model: outcome ~ C(f1) * C(f2)
    formula = f"{_safe_col(req.outcome)} ~ C({_safe_col(req.factor1)}) * C({_safe_col(req.factor2)})"
    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as exc:
        raise HTTPException(400, f"Model fitting error: {exc}")

    try:
        aov = anova_lm(model, typ=2)
    except Exception:
        aov = anova_lm(model, typ=1)

    # Parse effects
    effects = []
    for idx_name, row in aov.iterrows():
        name = str(idx_name)
        if name == "Residual":
            continue
        F_val = float(row["F"]) if not np.isnan(row["F"]) else 0
        p_val = float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else 1
        df_n = int(row["df"])
        df_d = int(aov.loc["Residual", "df"])
        sig_e = bool(p_val < req.alpha)
        es_e = partial_eta_squared(F_val, df_n, df_d)

        # Determine readable term name
        if req.factor1 in name and req.factor2 in name:
            term_label = f"{req.factor1} \u00D7 {req.factor2} (interaction)"
        elif req.factor1 in name:
            term_label = req.factor1
        elif req.factor2 in name:
            term_label = req.factor2
        else:
            term_label = name

        effects.append({
            "term": term_label, "raw_term": name,
            "F": round(F_val, 4), "df_num": df_n, "df_den": df_d,
            "p": round(p_val, 6), "significant": sig_e, "effect_size": es_e,
        })

    # Assumptions
    assumptions = [check_normality(model.resid.values, "Residuals")]

    levels1 = sorted(df[req.factor1].unique())
    levels2 = sorted(df[req.factor2].unique())
    interaction_sig = any(e["significant"] and "interaction" in e["term"] for e in effects)

    # ── Model-based estimated marginal means ────────────────────────────────
    # In the full factorial the cell predictions equal the observed cell
    # means, but the SEs come from the pooled residual MS, and the marginal
    # EMMs are the *unweighted* average over the other factor's levels,
    # what SPSS GLM / EMMEANS and R's emmeans report, and what the raw cell
    # summary that used to sit under this key was not.
    from patsy import build_design_matrices

    cell_counts = {
        (g1, g2): int(((df[req.factor1] == g1) & (df[req.factor2] == g2)).sum())
        for g1 in levels1 for g2 in levels2
    }
    has_empty_cells = any(n == 0 for n in cell_counts.values())
    df_resid = float(model.df_resid)
    t_crit = float(stats.t.ppf(0.975, df_resid))
    params = np.asarray(model.params)
    cov = np.asarray(model.cov_params())

    design_info = model.model.data.design_info
    grid = pd.DataFrame(
        [{req.factor1: g1, req.factor2: g2} for g1 in levels1 for g2 in levels2]
    )
    X_grid = None
    if not has_empty_cells:
        try:
            (X_grid,) = build_design_matrices([design_info], grid)
            X_grid = np.asarray(X_grid)
        except Exception:
            X_grid = None

    def _lin(L: np.ndarray) -> tuple:
        """Estimate, SE and 95% CI of the linear combination L·β."""
        est = float(L @ params)
        se = float(np.sqrt(max(L @ cov @ L, 0.0)))
        return est, se, est - t_crit * se, est + t_crit * se

    emms = []
    row_of = {}
    for idx, (g1, g2) in enumerate((a, b) for a in levels1 for b in levels2):
        cell = df[(df[req.factor1] == g1) & (df[req.factor2] == g2)][req.outcome]
        entry = {
            "factor1": str(g1), "factor2": str(g2),
            "n": int(len(cell)),
            "mean": round(float(cell.mean()), 4) if len(cell) > 0 else None,
            "sd": round(float(cell.std(ddof=1)), 4) if len(cell) > 1 else None,
            "emm": None, "se": None, "ci_low": None, "ci_high": None,
        }
        if X_grid is not None:
            row_of[(g1, g2)] = X_grid[idx]
            est, se, lo, hi = _lin(X_grid[idx])
            entry.update({"emm": round(est, 4), "se": round(se, 4),
                          "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
        emms.append(entry)

    emm_marginal = []
    marginal_L = {}          # (factor_name, level) -> contrast row
    if X_grid is not None:
        for g1 in levels1:
            L = np.mean([row_of[(g1, g2)] for g2 in levels2], axis=0)
            marginal_L[(req.factor1, g1)] = L
            est, se, lo, hi = _lin(L)
            emm_marginal.append({"factor": req.factor1, "level": str(g1),
                                 "emm": round(est, 4), "se": round(se, 4),
                                 "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
        for g2 in levels2:
            L = np.mean([row_of[(g1, g2)] for g1 in levels1], axis=0)
            marginal_L[(req.factor2, g2)] = L
            est, se, lo, hi = _lin(L)
            emm_marginal.append({"factor": req.factor2, "level": str(g2),
                                 "emm": round(est, 4), "se": round(se, 4),
                                 "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

    def _tukey_pair(L_diff: np.ndarray, n_means: int) -> tuple:
        """Tukey-Kramer comparison of two EMMs: q = |diff| / (SE/√2)."""
        diff = float(L_diff @ params)
        se = float(np.sqrt(max(L_diff @ cov @ L_diff, 0.0)))
        if se == 0:
            return diff, se, 0.0, 1.0
        q = abs(diff) / (se / np.sqrt(2.0))
        p = float(stats.studentized_range.sf(q, n_means, df_resid))
        return diff, se, q, p

    # ── Post-hoc on marginal EMMs (significant, non-qualified main effects) ─
    posthoc = []
    posthoc_method = None
    posthoc_note = None
    if X_grid is not None:
        for factor, levels in ((req.factor1, levels1), (req.factor2, levels2)):
            eff = next((e for e in effects
                        if e["term"] == factor and "interaction" not in e["term"]), None)
            if eff is None or not eff["significant"]:
                continue
            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    L = marginal_L[(factor, levels[i])] - marginal_L[(factor, levels[j])]
                    diff, se, q, p = _tukey_pair(L, len(levels))
                    posthoc.append({
                        "factor": factor,
                        "group1": str(levels[i]), "group2": str(levels[j]),
                        "statistic": round(q, 4), "mean_diff": round(diff, 4),
                        "se": round(se, 4),
                        "p_adj": round(p, 6), "significant": bool(p < req.alpha),
                        "correction": "tukey_emm",
                    })
            posthoc_method = "Tukey-Kramer on model-based EMMs"
        if interaction_sig and posthoc:
            posthoc_note = ("The interaction is significant: marginal comparisons "
                            "average over the other factor and should be read "
                            "alongside the simple effects below.")

    # ── Simple effects when the interaction is significant ─────────────────
    # F test of one factor's cell means within each level of the other,
    # using the full model's residual MS (the SPSS GLM / emmeans
    # joint_tests(by=...) construction), plus Tukey-adjusted pairwise cell
    # comparisons within that level.
    simple_effects = []
    if interaction_sig and X_grid is not None:
        def _simple(effect_of: str, eff_levels: list, within: str, within_levels: list, key):
            for w in within_levels:
                rows = [row_of[key(lv, w)] for lv in eff_levels]
                L = np.array([rows[i] - rows[0] for i in range(1, len(rows))])
                LB = L @ params
                M = L @ cov @ L.T
                try:
                    F = float(LB @ np.linalg.solve(M, LB)) / L.shape[0]
                except np.linalg.LinAlgError:
                    continue
                df_num = L.shape[0]
                p = float(stats.f.sf(F, df_num, df_resid))
                simple_effects.append({
                    "effect_of": effect_of, "within_factor": within,
                    "within_level": str(w),
                    "F": round(F, 4), "df_num": df_num, "df_den": int(df_resid),
                    "p": round(p, 6), "significant": bool(p < req.alpha),
                    "partial_eta_sq": round(F * df_num / (F * df_num + df_resid), 4),
                })
                for i in range(len(eff_levels)):
                    for j in range(i + 1, len(eff_levels)):
                        Ld = row_of[key(eff_levels[i], w)] - row_of[key(eff_levels[j], w)]
                        diff, se, q, p_pair = _tukey_pair(Ld, len(eff_levels))
                        posthoc.append({
                            "factor": f"{effect_of} within {within} = {w}",
                            "group1": str(eff_levels[i]), "group2": str(eff_levels[j]),
                            "statistic": round(q, 4), "mean_diff": round(diff, 4),
                            "se": round(se, 4),
                            "p_adj": round(p_pair, 6),
                            "significant": bool(p_pair < req.alpha),
                            "correction": "tukey_emm",
                        })
        _simple(req.factor1, levels1, req.factor2, levels2, lambda lv, w: (lv, w))
        _simple(req.factor2, levels2, req.factor1, levels1, lambda lv, w: (w, lv))
        if posthoc_method is None:
            posthoc_method = "Tukey-Kramer on model-based EMMs"

    emm_note = None
    if X_grid is None:
        emm_note = ("Model-based EMMs unavailable"
                    + (" (empty factor cells)" if has_empty_cells else "")
                    + "; the table shows observed cell means only.")

    # Build interpretation
    interp_parts = []
    for e in effects:
        ps = _p_str(e["p"])
        interp_parts.append(
            f"{'significant' if e['significant'] else 'no significant'} effect of {e['term']} "
            f"(F({e['df_num']},{e['df_den']}) = {e['F']:.2f}, p = {ps}, partial \u03B7\u00B2 = {e['effect_size']['value']:.3f})"
        )

    # Export rows
    export_rows = [["Source", "SS", "df", "F", "p", "Partial eta-sq"]]
    for idx_name, row in aov.iterrows():
        name = str(idx_name)
        export_rows.append([
            name,
            round(float(row["sum_sq"]), 4),
            int(row["df"]),
            round(float(row["F"]), 4) if not np.isnan(row["F"]) else "",
            round(float(row["PR(>F)"]), 6) if not np.isnan(row["PR(>F)"]) else "",
            "",
        ])

    interp = "Two-way ANOVA: " + "; ".join(interp_parts) + "."
    if interaction_sig and simple_effects:
        sig_simple = [s for s in simple_effects if s["significant"]]
        interp += (
            f" The interaction is significant, so main effects are qualified; "
            f"simple effects were tested within each factor level "
            f"({len(sig_simple)} of {len(simple_effects)} significant)."
        )

    return {
        "test": f"Two-way ANOVA ({req.factor1} \u00D7 {req.factor2})",
        "effects": effects,
        "significant": any(e["significant"] for e in effects),
        "effect_sizes": [e["effect_size"] for e in effects],
        "assumptions": assumptions,
        "emms": emms,
        "emm_marginal": emm_marginal,
        "emm_note": emm_note,
        "simple_effects": simple_effects,
        "posthoc": posthoc,
        "posthoc_method": posthoc_method,
        "posthoc_note": posthoc_note,
        "summary": {
            f"{req.factor1}_levels": sorted(df[req.factor1].unique().tolist()),
            f"{req.factor2}_levels": sorted(df[req.factor2].unique().tolist()),
            "n": len(df),
        },
        "interpretation": interp,
        "result_text": (
            f"A two-way ANOVA examined the effects of {req.factor1} ({k1} levels) and {req.factor2} ({k2} levels) "
            f"on {req.outcome} (N = {len(df)}). " + "; ".join(interp_parts) + "."
            + (" As the interaction was significant, simple main effects were examined within each level of the other factor."
               if interaction_sig and simple_effects else "")
        ),
        "export_rows": export_rows,
        # This script reproduces the numbers on screen: emmeans(~ factor)
        # matches the marginal EMM table, joint_tests(by=...) matches the
        # simple-effect F tests, pairs(..., adjust="tukey") matches the
        # Tukey-Kramer comparisons.
        "r_code": (
            f'model <- aov({req.outcome} ~ {req.factor1} * {req.factor2}, data = data)\n'
            f'summary(model)\n'
            f'library(emmeans)\n'
            f'emmeans(model, ~ {req.factor1})   # marginal EMMs (as shown)\n'
            f'emmeans(model, ~ {req.factor2})\n'
            f'pairs(emmeans(model, ~ {req.factor1}), adjust = "tukey")\n'
            f'emm <- emmeans(model, ~ {req.factor1} | {req.factor2})\n'
            f'joint_tests(emm, by = "{req.factor2}")   # simple main effects\n'
            f'pairs(emm, adjust = "tukey")             # cell comparisons'
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MANCOVA  (multivariate analysis of covariance)
# ═══════════════════════════════════════════════════════════════════════════════

class MancovaRequest(BaseModel):
    session_id: str
    outcomes: List[str]            # ≥2 dependent variables (e.g. BDNF, GDNF, NTF3, NGF)
    group_col: str
    covariates: List[str] = []
    alpha: float = 0.05
    imputation: str = "listwise"


def _eta_magnitude(v: float) -> str:
    if v >= 0.14:
        return "large"
    if v >= 0.06:
        return "medium"
    if v >= 0.01:
        return "small"
    return "negligible"


@router.post("/mancova")
def mancova(req: MancovaRequest):
    """One-way MANCOVA: several continuous outcomes vs a grouping factor while
    controlling for covariates. Returns the four multivariate test statistics
    (Pillai's trace, Wilks' lambda, Hotelling-Lawley trace, Roy's greatest root)
    for the group effect, with an F-approximation, p-value, and a multivariate
    partial η². Intended as the omnibus step before per-outcome ANCOVAs.
    """
    from statsmodels.multivariate.manova import MANOVA

    df_full = _get_df(req.session_id)
    if len(req.outcomes) < 2:
        raise HTTPException(400, "MANCOVA needs at least 2 outcome variables.")
    cols = list(dict.fromkeys(req.outcomes + [req.group_col] + req.covariates))
    for c in cols:
        if c not in df_full.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    df = apply_imputation(df_full, cols, req.imputation)
    for c in req.outcomes + req.covariates:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)

    min_n = max(10, len(req.outcomes) + len(req.covariates) + 3)
    if len(df) < min_n:
        raise HTTPException(400, f"Need at least {min_n} complete rows for this MANCOVA.")

    groups = sorted_groups(df[req.group_col])
    if len(groups) < 2:
        raise HTTPException(400, "Group column must have at least 2 levels.")

    lhs = " + ".join(_safe_col(o) for o in req.outcomes)
    rhs = " + ".join([f"C({_safe_col(req.group_col)})"] + [_safe_col(c) for c in req.covariates])
    formula = f"{lhs} ~ {rhs}"

    try:
        mv = MANOVA.from_formula(formula, data=df)
        mvtest = mv.mv_test()
    except Exception as exc:
        raise HTTPException(400, f"MANCOVA fitting error: {exc}")

    # Locate the grouping term in the multivariate test output.
    group_term = next((k for k in mvtest.results.keys() if req.group_col in str(k)), None)
    if group_term is None:
        raise HTTPException(500, "Could not find the group effect in the MANCOVA output.")

    stat_df = mvtest.results[group_term]["stat"]
    tests_out = []
    pillai = None
    for name in stat_df.index:
        row = stat_df.loc[name]
        entry = {
            "test": str(name),
            "value": round(float(row["Value"]), 5),
            "F": round(float(row["F Value"]), 4),
            "num_df": round(float(row["Num DF"]), 1),
            "den_df": round(float(row["Den DF"]), 1),
            "p": float(row["Pr > F"]),
            "significant": bool(float(row["Pr > F"]) < req.alpha),
        }
        tests_out.append(entry)
        if "Pillai" in str(name):
            pillai = entry

    if pillai is None:
        pillai = tests_out[0]

    # Multivariate partial η² from Pillai's trace: V / s, s = min(#outcomes, df_hypothesis)
    s = max(1, min(len(req.outcomes), len(groups) - 1))
    eta2 = float(pillai["value"]) / s
    eta2 = max(0.0, min(1.0, eta2))
    magnitude = _eta_magnitude(eta2)
    sig = bool(pillai["p"] < req.alpha)
    ps = _p_str(pillai["p"])

    cov_list = ", ".join(req.covariates) if req.covariates else "none"
    out_list = ", ".join(req.outcomes)

    export_rows = [["Multivariate test", "Value", "F", "Num df", "Den df", "p"]]
    for t in tests_out:
        export_rows.append([t["test"], t["value"], t["F"], t["num_df"], t["den_df"], round(t["p"], 6)])

    return {
        "test": "MANCOVA",
        "outcomes": req.outcomes,
        "group_col": req.group_col,
        "covariates": req.covariates,
        "n": int(len(df)),
        "n_groups": len(groups),
        "groups": [str(g) for g in groups],
        "multivariate_tests": tests_out,
        "pillai": pillai,
        "significant": sig,
        "effect_size": {"name": "partial η² (multivariate, from Pillai)",
                        "value": round(eta2, 4), "magnitude": magnitude},
        "interpretation": (
            f"After controlling for {cov_list}, there was {'a significant' if sig else 'no significant'} "
            f"multivariate effect of {req.group_col} on the combined outcomes "
            f"(Pillai's Trace = {pillai['value']:.3f}, F({pillai['num_df']:.0f}, {pillai['den_df']:.0f}) "
            f"= {pillai['F']:.2f}, p = {ps}, partial η² = {eta2:.3f} [{magnitude}])."
            + (" Follow up with per-outcome ANCOVAs." if sig else "")
        ),
        "result_text": (
            f"A one-way MANCOVA was conducted with {out_list} as dependent variables and {req.group_col} as the "
            f"between-subjects factor, controlling for {cov_list}. "
            f"There was {'a statistically significant' if sig else 'no statistically significant'} "
            f"multivariate effect of {req.group_col} (Pillai's Trace = {pillai['value']:.3f}, "
            f"F({pillai['num_df']:.0f}, {pillai['den_df']:.0f}) = {pillai['F']:.2f}, p = {ps}, "
            f"partial η² = {eta2:.3f}). "
            + ("Given the significant omnibus result, separate ANCOVAs were examined for each outcome."
               if sig else "As the omnibus test was non-significant, follow-up ANCOVAs are not warranted.")
        ),
        "export_rows": export_rows,
        "r_code": (
            f'library(car)\n'
            f'model <- lm(cbind({", ".join(req.outcomes)}) ~ {req.group_col}'
            + (f' + {" + ".join(req.covariates)}' if req.covariates else "")
            + ', data = data)\n'
            'Manova(model, type = "II")  # Pillai by default'
        ),
    }
