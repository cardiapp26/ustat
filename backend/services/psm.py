"""
Pure statistical helper functions for Propensity Score Matching (PSM).

These functions contain no FastAPI, no request models, and no side effects
on the session store. They are deliberately extracted so they can be
unit-tested in isolation and reused by future IPTW / weighting modules.

All functions are intentionally kept small and side-effect free.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _is_nominal_multilevel(s_treated: pd.Series, s_control: pd.Series) -> bool:
    """True for a text / categorical covariate with three or more levels."""
    if pd.api.types.is_bool_dtype(s_treated) or pd.api.types.is_numeric_dtype(s_treated):
        return False
    return pd.concat([s_treated, s_control]).dropna().nunique() > 2


def smd_by_level(
    s_treated: pd.Series,
    s_control: pd.Series,
    ref_treated: Optional[pd.Series] = None,
    ref_control: Optional[pd.Series] = None,
) -> dict:
    """SMD of each level's indicator for a nominal covariate.

    A nominal covariate has no single mean. Coding its levels 0, 1, 2 in
    alphabetical order and taking the SMD of those codes gave a number that
    changed when the levels were relabelled and could sit under 0.10 while
    individual levels were at 0.3. Each level is compared as a proportion
    instead (Austin 2009). With `ref_*` (the unmatched sample) the levels and
    the denominators come from there, so before and after are comparable.
    """
    base_t = s_treated if ref_treated is None else ref_treated
    base_c = s_control if ref_control is None else ref_control
    levels = sorted(pd.concat([base_t, base_c]).dropna().unique().tolist(), key=str)
    out: dict = {}
    for level in levels:
        ind_t = (s_treated.dropna() == level).astype(float)
        ind_c = (s_control.dropna() == level).astype(float)
        denom = None
        if ref_treated is not None and ref_control is not None:
            denom = _pooled_sd(
                (ref_treated.dropna() == level).astype(float),
                (ref_control.dropna() == level).astype(float),
            )
        out[str(level)] = _binary_smd(ind_t, ind_c, denom)
    return out


def _binary_smd(ind_treated: pd.Series, ind_control: pd.Series, denom_sd: Optional[float]) -> float:
    if len(ind_treated) == 0 or len(ind_control) == 0:
        return 0.0
    p1, p0 = float(ind_treated.mean()), float(ind_control.mean())
    denom = denom_sd if denom_sd is not None else np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return float(abs(p1 - p0) / denom) if denom > 1e-9 else 0.0


def smd_before_after(
    before_treated: pd.Series,
    before_control: pd.Series,
    after_treated: pd.Series,
    after_control: pd.Series,
) -> Tuple[float, float, Optional[dict]]:
    """SMD before and after matching, on the unmatched-sample denominator.

    Returns (before, after, per_level). For a nominal covariate with three or
    more levels the two numbers are the largest level SMD and `per_level` is
    {level: {"before", "after"}}; otherwise `per_level` is None.
    """
    if _is_nominal_multilevel(before_treated, before_control):
        lv_before = smd_by_level(before_treated, before_control)
        lv_after = smd_by_level(after_treated, after_control, before_treated, before_control)
        per_level = {
            lvl: {"before": round(lv_before[lvl], 4), "after": round(lv_after.get(lvl, 0.0), 4)}
            for lvl in lv_before
        }
        return max(lv_before.values()), max(lv_after.values()), per_level
    denom = _pooled_sd(before_treated, before_control)
    return (
        _compute_smd(before_treated, before_control, denom_sd=denom),
        _compute_smd(after_treated, after_control, denom_sd=denom),
        None,
    )


def _smd_columns(
    s_treated: pd.Series, s_control: pd.Series
) -> Tuple[pd.Series, pd.Series, bool]:
    """Coerce two covariate series to numeric (label-encoded if categorical).

    Returns (treated_numeric, control_numeric, is_binary).
    """
    if s_treated.dtype == object or str(s_treated.dtype).startswith("category"):
        combined = pd.concat([s_treated, s_control]).dropna()
        cats = sorted(combined.unique().tolist(), key=str)
        cat_map = {c: i for i, c in enumerate(cats)}
        s_treated = s_treated.map(cat_map)
        s_control = s_control.map(cat_map)

    s_treated = pd.to_numeric(s_treated, errors="coerce").dropna()
    s_control = pd.to_numeric(s_control, errors="coerce").dropna()

    is_binary = pd.concat([s_treated, s_control]).nunique() <= 2
    return s_treated, s_control, is_binary


def _compute_smd(
    s_treated: pd.Series,
    s_control: pd.Series,
    denom_sd: Optional[float] = None,
) -> float:
    """Standardized Mean Difference for one covariate (Austin 2011 convention).

    If `denom_sd` is supplied, it is used as the denominator for both
    before and after (so that improvement is visible in the numerator only).

    A nominal covariate with three or more levels returns its largest
    per-level SMD (see `smd_by_level`); `denom_sd` does not apply to it.
    """
    if _is_nominal_multilevel(s_treated, s_control):
        return max(smd_by_level(s_treated, s_control).values())
    s_t, s_c, is_bin = _smd_columns(s_treated, s_control)
    if len(s_t) == 0 or len(s_c) == 0:
        return 0.0

    if is_bin:
        p1 = float(s_t.mean())
        p0 = float(s_c.mean())
        denom = denom_sd if denom_sd is not None else np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
        return float(abs(p1 - p0) / denom) if denom > 1e-9 else 0.0

    m1, m0 = float(s_t.mean()), float(s_c.mean())
    if denom_sd is None:
        sd1, sd0 = float(s_t.std(ddof=1)), float(s_c.std(ddof=1))
        denom_sd = np.sqrt((sd1**2 + sd0**2) / 2)
    return float(abs(m1 - m0) / denom_sd) if denom_sd > 1e-9 else 0.0


def _pooled_sd(s_treated: pd.Series, s_control: pd.Series) -> float:
    """Pooled SD from the unmatched sample (Austin 2011 denominator)."""
    s_t, s_c, is_bin = _smd_columns(s_treated, s_control)
    if len(s_t) == 0 or len(s_c) == 0:
        return 0.0
    if is_bin:
        p1 = float(s_t.mean())
        p0 = float(s_c.mean())
        return float(np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2))
    sd1 = float(s_t.std(ddof=1))
    sd0 = float(s_c.std(ddof=1))
    return float(np.sqrt((sd1**2 + sd0**2) / 2))


def _variance_ratio(s_treated: pd.Series, s_control: pd.Series) -> Optional[float]:
    """Rubin's variance ratio (treated variance / control variance). Target 0.5–2.0."""
    if _is_nominal_multilevel(s_treated, s_control):
        return None
    s_t, s_c, is_bin = _smd_columns(s_treated, s_control)
    if is_bin or len(s_t) < 2 or len(s_c) < 2:
        return None
    v1 = float(s_t.var(ddof=1))
    v0 = float(s_c.var(ddof=1))
    if v0 <= 1e-12:
        return None
    return round(v1 / v0, 4)


def _ks_p(s_treated: pd.Series, s_control: pd.Series) -> Optional[float]:
    """Two-sample Kolmogorov-Smirnov p-value for distributional balance."""
    if _is_nominal_multilevel(s_treated, s_control):
        return None
    s_t, s_c, is_bin = _smd_columns(s_treated, s_control)
    if is_bin or len(s_t) < 2 or len(s_c) < 2:
        return None
    from scipy.stats import ks_2samp

    try:
        _, p = ks_2samp(s_t.values, s_c.values)
        return round(float(p), 6)
    except Exception:
        return None


def _fit_propensity_scores(
    X_scaled: np.ndarray, y: np.ndarray, method: str, random_state: Optional[int]
) -> np.ndarray:
    """Fit propensity scores using logistic, probit or GBM.

    Pure function — no side effects.
    """
    method = (method or "logistic").lower()
    if method == "logistic":
        from sklearn.linear_model import LogisticRegression

        # penalty=None: a propensity model is a maximum-likelihood logistic
        # regression. The scikit-learn default (L2, C=1) shrinks the
        # coefficients, which pulls the scores toward the prevalence and
        # understates the extreme weights, most in small samples.
        m = LogisticRegression(
            max_iter=5000, solver="lbfgs", penalty=None, random_state=random_state
        )
        m.fit(X_scaled, y)
        return m.predict_proba(X_scaled)[:, 1]

    if method == "probit":
        import statsmodels.api as _sm

        X_const = _sm.add_constant(X_scaled, has_constant="add")
        m = _sm.Probit(y, X_const).fit(disp=False, maxiter=200)
        return np.asarray(m.predict(X_const))

    if method == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier

        m = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        )
        m.fit(X_scaled, y)
        return m.predict_proba(X_scaled)[:, 1]

    raise ValueError(f"Unknown score_method: {method}")


def _match_greedy(
    treated_idx: np.ndarray,
    control_idx: np.ndarray,
    distance_vec: np.ndarray,
    caliper_dist: float,
    ratio: int,
) -> Tuple[list[int], list[int]]:
    """Greedy nearest-neighbour matching with caliper (hardest-first)."""
    from sklearn.neighbors import NearestNeighbors

    matched_t: list[int] = []
    matched_c: list[int] = []
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return matched_t, matched_c

    ctrl_dist = distance_vec[control_idx].reshape(-1, 1)
    n_controls = len(control_idx)

    used: set[int] = set()
    ordered = treated_idx[np.argsort(-distance_vec[treated_idx])]

    # The neighbour search used to ask for a fixed ratio * 5 candidates — five
    # of them at 1:1 — and drop the treated unit if every one of those five
    # was already taken. Greedy matching works down from the highest
    # propensity, so the units processed later are exactly the ones whose
    # nearest controls have been spent, and they were being discarded while
    # free controls sat well inside the caliper. On the audit dataset that
    # lost 17 of 138 matchable patients, a seventh of the matched sample, for
    # no statistical reason at all.
    #
    # The candidate list now grows until either enough free controls inside
    # the caliper are found, or the returned neighbours have run past the
    # caliper — at which point there genuinely are none left.
    k = min(max(ratio * 5, 10), n_controls)
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(ctrl_dist)

    for ti in ordered:
        q = np.array([[distance_vec[ti]]])
        chosen: list[int] = []
        k_search = k
        while True:
            distances, neighbours = knn.kneighbors(q, n_neighbors=k_search)
            chosen = []
            for dist, nb in zip(distances[0], neighbours[0]):
                if dist > caliper_dist:
                    break
                c_real = int(control_idx[nb])
                if c_real not in used:
                    chosen.append(c_real)
                    if len(chosen) == ratio:
                        break
            if len(chosen) == ratio or k_search >= n_controls:
                break
            # Everything returned was inside the caliper but taken; there may
            # be more beyond the current horizon.
            if float(distances[0][-1]) > caliper_dist:
                break
            k_search = min(k_search * 2, n_controls)
        if len(chosen) == ratio:
            used.update(chosen)
            matched_t.append(int(ti))
            matched_c.extend(chosen)
    return matched_t, matched_c


def _match_optimal(
    treated_idx: np.ndarray,
    control_idx: np.ndarray,
    distance_vec: np.ndarray,
    caliper_dist: float,
) -> Tuple[list[int], list[int]]:
    """Optimal 1:1 matching via Hungarian algorithm (scipy)."""
    from scipy.optimize import linear_sum_assignment

    matched_t: list[int] = []
    matched_c: list[int] = []
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return matched_t, matched_c

    t_idx = np.asarray(treated_idx)
    c_idx = np.asarray(control_idx)
    dist_matrix = np.abs(
        distance_vec[t_idx][:, None] - distance_vec[c_idx][None, :]
    )
    dist_matrix[dist_matrix > caliper_dist] = 1e9  # effectively forbidden

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] < 1e8:
            matched_t.append(int(t_idx[r]))
            matched_c.append(int(c_idx[c]))
    return matched_t, matched_c


def _rosenbaum_bounds(
    pair_outcomes: list[tuple[int, int]],
    gamma_max: float = 3.0,
    n_gamma: int = 60,
    alpha: float = 0.05,
) -> dict:
    """Rosenbaum sensitivity bounds for 1:1 matched binary outcomes.

    See original _rosenbaum_bounds in models.py for full statistical explanation.
    """
    from scipy.stats import binom

    b = sum(1 for t, c in pair_outcomes if t == 1 and c == 0)
    c = sum(1 for t, c in pair_outcomes if t == 0 and c == 1)
    discordant = b + c
    if discordant == 0:
        return {"applicable": False, "reason": "No discordant pairs.", "b": b, "c": c}

    b_obs = max(b, c)
    p_at_no_bias = float(binom.sf(b_obs - 1, discordant, 0.5))
    gammas = np.linspace(1.0, max(1.0, gamma_max), max(2, int(n_gamma)))
    rows = []
    crit_gamma: Optional[float] = None
    for g in gammas:
        p_plus = g / (1.0 + g)
        p_upper = float(binom.sf(b_obs - 1, discordant, p_plus))
        rows.append({"gamma": round(float(g), 3), "p_upper": round(p_upper, 6)})
        if crit_gamma is None and p_upper > alpha:
            crit_gamma = round(float(g), 3)

    return {
        "applicable": True,
        "b": b,
        "c": c,
        "discordant_pairs": discordant,
        "p_unbiased": round(p_at_no_bias, 6),
        "critical_gamma": crit_gamma,
        "alpha": alpha,
        "gamma_max": gamma_max,
        "curve": rows,
    }
