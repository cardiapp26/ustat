"""Repeated-measures tests: paired t-test, Wilcoxon SR, Friedman, RM ANOVA, mixed ANOVA."""
import numpy as np
import pandas as pd
from scipy import stats as sp
from fastapi import APIRouter, HTTPException
from pydantic import AliasChoices, BaseModel, Field
from typing import List, Optional

from services import store
from services.stat_utils import (
    cohen_d_paired, matched_rank_biserial, kendalls_w, partial_eta_squared,
    check_normality, group_summary, adjust_pvalues,
    sanitize_nonfinite, sphericity, paired_contrast_is_degenerate,
)

router = APIRouter()


def _get_df(session_id: str) -> pd.DataFrame:
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


def _p_str(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAIRED T-TEST
# ═══════════════════════════════════════════════════════════════════════════════

class PairedTTestRequest(BaseModel):
    session_id: str
    col1: str = Field(validation_alias=AliasChoices("col1", "column1"))
    col2: str = Field(validation_alias=AliasChoices("col2", "column2"))
    alpha: float = 0.05


@router.post("/paired_ttest")
def paired_ttest(req: PairedTTestRequest):
    df = _get_df(req.session_id)
    pair = df[[req.col1, req.col2]].dropna()
    if len(pair) < 3:
        raise HTTPException(400, "Need at least 3 complete pairs.")
    x1 = pair[req.col1].astype(float).values
    x2 = pair[req.col2].astype(float).values
    d = x1 - x2
    n = len(d)

    # Every pair changing by the same amount leaves the difference with no
    # variance, and a t has nowhere to come from. This used to be handled by
    # capping an infinite t at +/-9999 and setting p to 0 — a statistic the
    # data never produced, reported as an overwhelming result. Testing
    # isfinite alone does not catch it either: on large values the subtraction
    # loses bits, the spread comes out as a few ulps instead of zero, and
    # SciPy returns a finite t of 8.6e12 with p = 0.
    degenerate = paired_contrast_is_degenerate(x1, x2)
    t_stat, p = sp.ttest_rel(x1, x2)
    degenerate_note = None
    if degenerate:
        if abs(float(d.mean())) <= 1e-12 * max(
            float(np.max(np.abs(x1))), float(np.max(np.abs(x2))), 1.0
        ):
            # Identical columns. "No difference" is a real, reportable answer.
            t_stat, p = 0.0, 1.0
        else:
            t_stat, p = None, None
            degenerate_note = (
                f"Every pair changed by the same amount "
                f"({float(d.mean()):.6g}), so the difference carries no "
                "variance and a paired t-test has nothing to estimate. The "
                "change itself is reported below."
            )
    elif np.isnan(p) or np.isnan(t_stat):
        p = 1.0
        t_stat = 0.0
    sig = bool(p is not None and p < req.alpha)
    es = cohen_d_paired(d)
    norm = check_normality(d, "Differences")

    mean_diff = float(d.mean())
    sd_diff = float(d.std(ddof=1))
    ps = _p_str(p) if p is not None else "\u2014"

    t_disp = f"{t_stat:.3f}" if t_stat is not None else "—"
    warn_list = (
        ["Differences are not normally distributed — consider Wilcoxon signed-rank test."]
        if not norm["met"] else []
    )
    if degenerate_note:
        warn_list = [degenerate_note] + warn_list

    return sanitize_nonfinite({
        "test": "Paired-samples t-test",
        "t": round(float(t_stat), 4) if t_stat is not None else None,
        "df": n - 1,
        "p": float(p) if p is not None else None,
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [norm],
        "warnings": warn_list,
        "summary": {
            req.col1: group_summary(x1, req.col1),
            req.col2: group_summary(x2, req.col2),
            "differences": {"n": n, "mean": round(mean_diff, 4), "sd": round(sd_diff, 4)},
        },
        "interpretation": (
            degenerate_note if degenerate_note else
            f"{'Significant' if sig else 'No significant'} difference between {req.col1} and {req.col2} (t({n-1}) = {t_disp}, p = {ps}, d_z = {es['value']:.3f} [{es['magnitude']}])"
        ),
        "result_text": (
            f"A paired-samples t-test compared {req.col1} (M = {x1.mean():.2f}, SD = {x1.std(ddof=1):.2f}) "
            f"and {req.col2} (M = {x2.mean():.2f}, SD = {x2.std(ddof=1):.2f}). "
            f"{'There was a significant difference' if sig else 'There was no significant difference'} "
            f"(t({n-1}) = {t_disp}, p = {ps}). The mean difference was {mean_diff:.3f} (SD = {sd_diff:.3f}), "
            f"with a {es['magnitude']} effect size (Cohen's d_z = {es['value']:.3f}, 95% CI [{es['ci_low']:.3f}, {es['ci_high']:.3f}])."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["t", round(float(t_stat), 4) if t_stat is not None else "—"],
            ["df", n - 1],
            ["p", round(float(p), 6) if p is not None else "—"],
            ["Mean difference", round(mean_diff, 4)],
            ["SD of differences", round(sd_diff, 4)],
            ["Cohen's d_z", es["value"]],
            ["95% CI lower", es["ci_low"]],
            ["95% CI upper", es["ci_high"]],
            ["Magnitude", es["magnitude"]],
        ],
        "r_code": f't.test(data${req.col1}, data${req.col2}, paired = TRUE)',
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WILCOXON SIGNED-RANK
# ═══════════════════════════════════════════════════════════════════════════════

class WilcoxonSRRequest(BaseModel):
    session_id: str
    col1: str = Field(validation_alias=AliasChoices("col1", "column1"))
    col2: str = Field(validation_alias=AliasChoices("col2", "column2"))
    alpha: float = 0.05


@router.post("/wilcoxon_signed_rank")
def wilcoxon_signed_rank(req: WilcoxonSRRequest):
    df = _get_df(req.session_id)
    pair = df[[req.col1, req.col2]].dropna()
    if len(pair) < 6:
        raise HTTPException(400, "Need at least 6 complete pairs for Wilcoxon signed-rank.")
    x1 = pair[req.col1].astype(float).values
    x2 = pair[req.col2].astype(float).values
    d = x1 - x2

    # Remove zero differences (standard Wilcoxon practice)
    nonzero = d[d != 0]
    if len(nonzero) < 3:
        raise HTTPException(400, "Too few non-zero differences for Wilcoxon test.")

    # SciPy picks the exact distribution for small samples without ties and
    # the normal approximation otherwise; zero differences are dropped
    # (Wilcoxon's own rule). None of that used to be reported, so a p that
    # differs from R's wilcox.text on tied data had no explanation attached.
    n_zero = int((d == 0).sum())
    _, tie_counts = np.unique(np.abs(nonzero), return_counts=True)
    n_ties = int((tie_counts > 1).sum())
    p_method = (
        "exact" if (len(nonzero) <= 25 and n_ties == 0) else "normal approximation"
    )
    w_stat, p = sp.wilcoxon(x1, x2, alternative="two-sided")
    sig = bool(p < req.alpha)
    # The two-sided SciPy statistic is min(W+, W-), which carries no
    # direction: differences that were all positive and differences that were
    # all negative both give W = 0, and feeding that to the rank-biserial
    # formula returned r = -1 for both. The effect size needs the sum of the
    # POSITIVE signed ranks.
    abs_ranks = sp.rankdata(np.abs(nonzero))
    w_plus = float(abs_ranks[nonzero > 0].sum())
    es = matched_rank_biserial(w_plus, len(nonzero))
    ps = _p_str(p)

    return sanitize_nonfinite({
        "test": "Wilcoxon signed-rank test",
        "W": round(float(w_stat), 4), "p": float(p), "n_nonzero": len(nonzero),
        "significant": sig,
        "effect_sizes": [es],
        "p_method": p_method,
        "n_zero_differences": n_zero,
        "n_tied_ranks": n_ties,
        "assumptions": [
            {"name": "Zero differences", "met": True,
             "detail": (f"{n_zero} pair(s) showed no change and were dropped, "
                        "as Wilcoxon's rule requires." if n_zero else
                        "No pair showed exactly zero change.")},
            {"name": "Ties", "met": n_ties == 0,
             "detail": (f"{n_ties} group(s) of tied absolute differences; the "
                        "p-value uses the normal approximation, which can "
                        "differ from an exact method." if n_ties else
                        f"No tied absolute differences; p from the {p_method}.")},
        ],
        "summary": {
            req.col1: group_summary(x1, req.col1),
            req.col2: group_summary(x2, req.col2),
        },
        "interpretation": f"{'Significant' if sig else 'No significant'} difference (W = {w_stat:.1f}, p = {ps}, r = {es['value']:.3f} [{es['magnitude']}])",
        "result_text": (
            f"A Wilcoxon signed-rank test indicated that {req.col2} scores were "
            f"{'significantly' if sig else 'not significantly'} different from {req.col1} scores "
            f"(W = {w_stat:.1f}, p = {ps}, r = {es['value']:.3f} [{es['magnitude']}]). "
            f"Median {req.col1} = {np.median(x1):.2f}, median {req.col2} = {np.median(x2):.2f}."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["W", round(float(w_stat), 4)],
            ["p", round(float(p), 6)],
            ["n (non-zero differences)", len(nonzero)],
            ["Rank-biserial r", es["value"]],
            ["95% CI lower", es["ci_low"]],
            ["95% CI upper", es["ci_high"]],
        ],
        "r_code": f'wilcox.test(data${req.col1}, data${req.col2}, paired = TRUE)',
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FRIEDMAN TEST
# ═══════════════════════════════════════════════════════════════════════════════

class FriedmanRequest(BaseModel):
    session_id: str
    columns: List[str]  # 3+ repeated measures columns (wide format)
    alpha: float = 0.05


@router.post("/friedman")
def friedman(req: FriedmanRequest):
    if len(req.columns) < 3:
        raise HTTPException(400, "Friedman test requires at least 3 repeated measures.")
    df = _get_df(req.session_id)
    sub = df[req.columns].dropna()
    if len(sub) < 5:
        raise HTTPException(400, "Need at least 5 complete subjects.")

    arrays = [sub[c].astype(float).values for c in req.columns]
    n = len(sub)
    k = len(req.columns)

    chi2, p = sp.friedmanchisquare(*arrays)
    sig = bool(p < req.alpha)
    es = kendalls_w(float(chi2), n, k)
    ps = _p_str(p)

    # Post-hoc: pairwise Wilcoxon signed-rank with Holm correction
    posthoc = []
    if sig and k > 2:
        raw_ps = []
        pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
        for i, j in pairs:
            try:
                w, pv = sp.wilcoxon(arrays[i], arrays[j], alternative="two-sided")
            except Exception:
                pv = 1.0
                w = 0
            posthoc.append({
                "group1": req.columns[i], "group2": req.columns[j],
                "statistic": round(float(w), 4), "p": round(float(pv), 6),
            })
            raw_ps.append(float(pv))
        adj = adjust_pvalues(raw_ps, "holm")
        for idx, ph in enumerate(posthoc):
            ph["p_adj"] = round(adj[idx], 6)
            ph["significant"] = adj[idx] < req.alpha
            ph["correction"] = "holm"

    return sanitize_nonfinite({
        "test": "Friedman test",
        "chi2": round(float(chi2), 4), "df": k - 1, "p": float(p),
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": [],
        "posthoc": posthoc,
        "posthoc_method": "Pairwise Wilcoxon signed-rank (Holm correction)" if posthoc else None,
        "summary": {c: group_summary(sub[c].astype(float).values, c) for c in req.columns},
        "interpretation": f"{'Significant' if sig else 'No significant'} difference across {k} conditions (\u03C7\u00B2({k-1}) = {chi2:.2f}, p = {ps}, Kendall's W = {es['value']:.3f} [{es['magnitude']}])",
        "result_text": (
            f"A Friedman test showed {'a significant' if sig else 'no significant'} difference across {k} conditions "
            f"(\u03C7\u00B2({k-1}) = {chi2:.2f}, p = {ps}, Kendall's W = {es['value']:.3f} [{es['magnitude']}]). "
            f"n = {n} subjects with complete data across all conditions."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["Chi-square", round(float(chi2), 4)],
            ["df", k - 1],
            ["p", round(float(p), 6)],
            ["Kendall's W", es["value"]],
            ["n", n],
            ["k (conditions)", k],
        ],
        "r_code": 'friedman.test(y ~ timepoint | subject, data = data_long)',
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REPEATED-MEASURES ANOVA
# ═══════════════════════════════════════════════════════════════════════════════

class RMAnovaRequest(BaseModel):
    session_id: str
    subject_col: str
    within_col: str
    value_col: str
    alpha: float = 0.05


@router.post("/rm_anova")
def rm_anova(req: RMAnovaRequest):
    from statsmodels.stats.anova import AnovaRM

    df = _get_df(req.session_id)
    cols = [req.subject_col, req.within_col, req.value_col]
    for c in cols:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found. Data must be in long format — use the Melt helper first.")

    sub = df[cols].dropna()
    sub[req.value_col] = pd.to_numeric(sub[req.value_col], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 10:
        raise HTTPException(400, "Need at least 10 rows for RM ANOVA.")

    k = sub[req.within_col].nunique()
    if k < 2:
        raise HTTPException(400, f"Within-subjects factor '{req.within_col}' must have at least 2 levels.")

    try:
        rm = AnovaRM(sub, req.value_col, req.subject_col, within=[req.within_col])
        res = rm.fit()
    except Exception as exc:
        raise HTTPException(400, f"RM ANOVA failed: {exc}")

    tbl = res.anova_table
    row = tbl.iloc[0]
    F_val = float(row["F Value"])
    p_val = float(row["Pr > F"])
    df_num = int(row["Num DF"])
    df_den = int(row["Den DF"])
    sig = bool(p_val < req.alpha)
    es = partial_eta_squared(F_val, df_num, df_den)
    ps = _p_str(p_val)

    # Sphericity. The code here used to read `res.epsilon` off the AnovaRM
    # result, which has no such attribute — so eps was always None, the block
    # never fired, and nothing about sphericity ever reached the user. With
    # three or more levels an unmet sphericity assumption makes the
    # uncorrected F anticonservative, so the epsilons are computed here and
    # the corrected p-values are reported alongside the uncorrected one.
    assumptions = []
    sphericity_out = None
    corrected = []
    if k > 2:
        wide = sub.pivot_table(
            index=req.subject_col, columns=req.within_col, values=req.value_col
        ).dropna()
        if len(wide) >= 3:
            sphericity_out = sphericity(wide.to_numpy())
            for label, key in (("Greenhouse-Geisser", "gg"), ("Huynh-Feldt", "hf")):
                eps = float(sphericity_out[key])
                dfn, dfd = df_num * eps, df_den * eps
                corrected.append({
                    "correction": label,
                    "epsilon": round(eps, 4),
                    "df_num": round(dfn, 3),
                    "df_den": round(dfd, 3),
                    "p": float(sp.f.sf(F_val, dfn, dfd)),
                })
            gg = float(sphericity_out["gg"])
            mp = sphericity_out.get("mauchly_p")
            met = mp is None or mp >= 0.05
            detail = (
                f"Mauchly W = {sphericity_out['mauchly_w']:.4f}, p = {mp:.4f}; "
                if mp is not None else ""
            ) + f"Greenhouse-Geisser \u03B5 = {gg:.3f}"
            if not met:
                detail += (
                    " \u2014 sphericity rejected; use the corrected p-value "
                    f"({corrected[0]['p']:.4g} with GG) rather than the "
                    "uncorrected one."
                )
            assumptions.append({"name": "Sphericity (Mauchly)", "met": bool(met),
                                "detail": detail})

    # Post-hoc: pairwise paired t-tests with Holm
    posthoc = []
    if sig and k > 2:
        levels = sorted(sub[req.within_col].unique())
        raw_ps = []
        for i in range(len(levels)):
            for j in range(i+1, len(levels)):
                g1 = sub[sub[req.within_col] == levels[i]].set_index(req.subject_col)[req.value_col]
                g2 = sub[sub[req.within_col] == levels[j]].set_index(req.subject_col)[req.value_col]
                common = g1.index.intersection(g2.index)
                if len(common) < 3:
                    continue
                v1 = g1.loc[common].values
                v2 = g2.loc[common].values
                # A pair whose differences are all identical has no variance,
                # so SciPy returns t = inf — one value that used to make the
                # entire response unserialisable, turning a valid omnibus
                # result into a 400. Testing isfinite is not enough on its
                # own: when the values are large, subtracting them loses bits
                # and the spread comes out as a few ulps rather than exactly
                # zero, so SciPy returns a FINITE t (8.6e12 on values of order
                # 1e6) with p = 0 and the pair is reported as significant. The
                # spread is judged against the size of the numbers instead.
                degenerate = paired_contrast_is_degenerate(v1, v2)
                t, pv = sp.ttest_rel(v1, v2)
                row = {
                    "group1": str(levels[i]), "group2": str(levels[j]),
                    "mean_diff": round(float(g1.loc[common].mean() - g2.loc[common].mean()), 4),
                }
                if not degenerate and np.isfinite(t) and np.isfinite(pv):
                    row["statistic"] = round(float(t), 4)
                    # Not rounded to 6 dp: a p of 1e-40 would print as 0.0.
                    row["p"] = float(pv)
                    raw_ps.append(float(pv))
                else:
                    row["statistic"] = None
                    row["p"] = None
                    row["note"] = (
                        "Every subject changed by effectively the same amount "
                        "between these two levels, so the difference carries no "
                        "variance and a paired t cannot be computed."
                    )
                posthoc.append(row)
        if raw_ps:
            adj = adjust_pvalues(raw_ps, "holm")
            it = iter(adj)
            for ph in posthoc:
                if ph.get("p") is None:
                    ph["p_adj"] = None
                    ph["significant"] = None
                    continue
                a = next(it)
                ph["p_adj"] = round(a, 6)
                ph["significant"] = a < req.alpha
                ph["correction"] = "holm"

    n_subj = sub[req.subject_col].nunique()
    return sanitize_nonfinite({
        "test": "Repeated-measures ANOVA",
        "F": round(F_val, 4), "df_num": df_num, "df_den": df_den, "p": float(p_val),
        "significant": sig,
        "effect_sizes": [es],
        "assumptions": assumptions,
        "sphericity": sphericity_out,
        "corrected": corrected,
        "posthoc": posthoc,
        "posthoc_method": "Pairwise paired t-tests (Holm correction)" if posthoc else None,
        "summary": {str(lv): group_summary(sub[sub[req.within_col] == lv][req.value_col].values, str(lv))
                    for lv in sorted(sub[req.within_col].unique())},
        "interpretation": f"{'Significant' if sig else 'No significant'} effect of {req.within_col} on {req.value_col} (F({df_num},{df_den}) = {F_val:.2f}, p = {ps}, partial \u03B7\u00B2 = {es['value']:.3f} [{es['magnitude']}])",
        "result_text": (
            f"A repeated-measures ANOVA with {k} levels of {req.within_col} (n = {n_subj} subjects) "
            f"showed {'a significant' if sig else 'no significant'} effect on {req.value_col} "
            f"(F({df_num},{df_den}) = {F_val:.2f}, p = {ps}, partial \u03B7\u00B2 = {es['value']:.3f} [{es['magnitude']}])."
        ),
        "export_rows": [
            ["Statistic", "Value"],
            ["F", round(F_val, 4)],
            ["df (numerator)", df_num],
            ["df (denominator)", df_den],
            ["p", round(float(p_val), 6)],
            ["Partial eta-squared", es["value"]],
            ["n subjects", n_subj],
            ["k conditions", k],
        ],
        "r_code": f'library(ez)\nezANOVA(data = data, dv = .({req.value_col}), wid = .({req.subject_col}), within = .({req.within_col}))',
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MIXED ANOVA (within + between)
# ═══════════════════════════════════════════════════════════════════════════════

class MixedAnovaRequest(BaseModel):
    session_id: str
    subject_col: str
    within_col: str
    between_col: str
    value_col: str
    alpha: float = 0.05


@router.post("/mixed_anova")
def mixed_anova(req: MixedAnovaRequest):
    """Split-plot mixed ANOVA: each effect against its own error term.

    The previous implementation fitted a plain factorial OLS —
    ``Y ~ C(within) * C(between)`` — with no subject term at all, despite
    describing the subject as a random effect. Every repeated observation of
    the same person was treated as an independent case, so all three effects
    were tested against a single pooled residual. That deflates the
    between-subject test, which has to be judged against variation BETWEEN
    subjects rather than within them.

    On the audit fixture the arm effect came out F = 8.07, p = 0.0098 —
    significant — where the correct split-plot test gives F = 2.72, p = 0.14.
    The clinical reading reversed.

    The design has two error strata and each effect belongs to one of them:

      * the between-subjects effect is tested against subject-to-subject
        variation inside each arm, which is exactly a one-way ANOVA on each
        subject's own mean;
      * the within-subjects effect and the interaction are tested against the
        residual of a model that has already absorbed each subject's level.

    For balanced complete data this reproduces R's
    ``aov(y ~ a*b + Error(subject/b))`` and ``ez::ezANOVA``.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    df = _get_df(req.session_id)
    cols = [req.subject_col, req.within_col, req.between_col, req.value_col]
    for c in cols:
        if c not in df.columns:
            raise HTTPException(400, f"Column '{c}' not found.")

    sub = df[cols].dropna()
    sub[req.value_col] = pd.to_numeric(sub[req.value_col], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 12:
        raise HTTPException(400, "Need at least 12 rows for mixed ANOVA.")

    warnings: List[str] = []
    sub = sub.copy()
    for c in (req.subject_col, req.within_col, req.between_col):
        sub[c] = sub[c].astype(str)

    # A subject belongs to exactly one between-subjects level. Anything else
    # is not a split plot and the error strata would not mean what they say.
    arms_per_subject = sub.groupby(req.subject_col)[req.between_col].nunique()
    crossed = arms_per_subject[arms_per_subject > 1].index.tolist()
    if crossed:
        raise HTTPException(
            400,
            f"'{req.between_col}' is meant to be between-subjects, but "
            f"{len(crossed)} subject(s) appear under more than one of its "
            f"levels (e.g. {crossed[0]}). Check the subject and group columns.",
        )

    within_levels = sorted(sub[req.within_col].unique().tolist())
    if len(within_levels) < 2:
        raise HTTPException(400, "The within-subjects factor needs at least 2 levels.")
    if sub[req.between_col].nunique() < 2:
        raise HTTPException(400, "The between-subjects factor needs at least 2 levels.")

    dupes = int(sub.duplicated(subset=[req.subject_col, req.within_col]).sum())
    if dupes:
        raise HTTPException(
            400,
            f"{dupes} subject x {req.within_col} combination(s) appear more than "
            "once. Mixed ANOVA needs one value per subject per level.",
        )

    # A subject missing a level contributes nothing to the within stratum.
    level_counts = sub.groupby(req.subject_col)[req.within_col].nunique()
    complete = level_counts[level_counts == len(within_levels)].index
    dropped = int(len(level_counts) - len(complete))
    if dropped:
        warnings.append(
            f"{dropped} subject(s) lack a value at every level of "
            f"'{req.within_col}' and were excluded; the analysis uses "
            f"{len(complete)} complete subject(s)."
        )
    sub = sub[sub[req.subject_col].isin(complete)]
    if sub[req.subject_col].nunique() < 3:
        raise HTTPException(400, "Need at least 3 subjects with complete data.")

    subj_means = (
        sub.groupby([req.subject_col, req.between_col], as_index=False)[req.value_col]
        .mean()
    )
    if int(subj_means.groupby(req.between_col)[req.subject_col].nunique().min()) < 2:
        raise HTTPException(
            400,
            f"Each level of '{req.between_col}' needs at least 2 subjects to "
            "estimate between-subject error.",
        )

    v, w, b, s = req.value_col, req.within_col, req.between_col, req.subject_col
    try:
        # Stratum 1 - between subjects. One row per subject, so the residual
        # is genuine subject-to-subject variation.
        between_aov = anova_lm(
            smf.ols(f"Q('{v}') ~ C(Q('{b}'))", data=subj_means).fit(), typ=2
        )
        # Stratum 2 - within subjects. The subject dummies absorb each
        # person's own level, leaving the residual that the within effect and
        # the interaction are properly judged against. The between main effect
        # is collinear with those dummies and drops out here by design; it was
        # tested in stratum 1.
        # The between main effect is written out so the interaction is the
        # genuine (a-1)(b-1) contrast. Without it the interaction term absorbs
        # the between effect too and comes out with an extra degree of freedom
        # (3 instead of 2 on the audit fixture, F 175.5 instead of 0.78). The
        # C(between) row this produces is aliased with the subject dummies and
        # is judged against the wrong error term, so it is ignored here - the
        # between effect comes from stratum 1.
        within_aov = anova_lm(
            smf.ols(
                f"Q('{v}') ~ C(Q('{s}')) + C(Q('{w}')) * C(Q('{b}'))",
                data=sub,
            ).fit(),
            typ=2,
        )
    except Exception as exc:
        raise HTTPException(400, f"Mixed ANOVA failed: {exc}")

    def _effect(aov, term: str, label: str, stratum: str) -> Optional[dict]:
        if term not in aov.index:
            return None
        row = aov.loc[term]
        f_val = float(row["F"])
        p_val = float(row["PR(>F)"])
        if not np.isfinite(f_val) or not np.isfinite(p_val):
            return None
        df_n = int(round(float(row["df"])))
        df_d = int(round(float(aov.loc["Residual", "df"])))
        return {
            "term": label,
            "F": round(f_val, 4),
            "df_num": df_n,
            "df_den": df_d,
            # Not rounded to 6 dp: a strongly significant within-subject
            # effect (3.3e-12 on the audit fixture) would print as 0.0.
            "p": p_val,
            "significant": bool(p_val < req.alpha),
            "effect_size": partial_eta_squared(f_val, df_n, df_d),
            "error_stratum": stratum,
        }

    effects = [
        e for e in (
            _effect(between_aov, f"C(Q('{b}'))", b, f"between subjects ({s})"),
            _effect(within_aov, f"C(Q('{w}'))", w, f"within subjects ({w} x {s})"),
            _effect(
                within_aov,
                f"C(Q('{w}')):C(Q('{b}'))",
                f"{w} \u00D7 {b} (interaction)",
                f"within subjects ({w} x {s})",
            ),
        ) if e is not None
    ]
    if not effects:
        raise HTTPException(400, "Mixed ANOVA produced no estimable effects.")

    n_subj = sub[req.subject_col].nunique()
    k_within = sub[req.within_col].nunique()
    k_between = sub[req.between_col].nunique()

    # Build interpretation
    interp_parts = []
    for e in effects:
        ps = _p_str(e["p"])
        interp_parts.append(
            f"{'significant' if e['significant'] else 'no significant'} effect of {e['term']} "
            f"(F({e['df_num']},{e['df_den']}) = {e['F']:.2f}, p = {ps}, partial \u03B7\u00B2 = {e['effect_size']['value']:.3f})"
        )

    return sanitize_nonfinite({
        "test": "Mixed ANOVA (within × between)",
        "effects": effects,
        "significant": any(e["significant"] for e in effects),
        "effect_sizes": [e["effect_size"] for e in effects],
        "warnings": warnings,
        "assumptions": [
            {"name": "Error strata", "met": True,
             "detail": ("Each effect is tested against its own error term: the "
                        "between-subjects effect against subject-to-subject "
                        "variation, the within-subjects effect and the "
                        "interaction against the subject x within residual.")},
            {"name": "Complete cases", "met": not warnings,
             "detail": (warnings[0] if warnings else
                        "Every subject has a value at every within-subjects level.")},
        ],
        "summary": {
            "within_levels": sorted(sub[req.within_col].unique().tolist()),
            "between_levels": sorted(sub[req.between_col].unique().tolist()),
            "n_subjects": n_subj,
        },
        "interpretation": "Mixed ANOVA: " + "; ".join(interp_parts) + ".",
        "result_text": (
            f"A mixed ANOVA with {req.within_col} ({k_within} levels, within-subjects) and "
            f"{req.between_col} ({k_between} levels, between-subjects) on {req.value_col} "
            f"(n = {n_subj} subjects) revealed: " + "; ".join(interp_parts) + "."
        ),
        "export_rows": [
            ["Term", "F", "df_num", "df_den", "p", "Partial eta-squared"],
            *[[e["term"], e["F"], e["df_num"], e["df_den"], e["p"], e["effect_size"]["value"]] for e in effects],
        ],
        "r_code": (
            f'library(ez)\n'
            f'ezANOVA(data = data, dv = .({req.value_col}), wid = .({req.subject_col}), '
            f'within = .({req.within_col}), between = .({req.between_col}))'
        ),
    })
