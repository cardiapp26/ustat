"""
Shared statistical utilities for uSTAT.

Provides:
  - AnalysisResult: standard result contract
  - Effect size calculators with CI
  - Multiplicity correction (Bonferroni, Holm, FDR)
  - Bootstrap CI
  - Pairwise comparison builders (Tukey, Games-Howell, Dunn, etc.)
"""

import numpy as np
import pandas as pd
from scipy import stats as sp
from typing import Optional
from dataclasses import dataclass, field, asdict


def sorted_groups(series: "pd.Series") -> list:
    """Stable, value-code order for grouped output (Table 1 columns, ANOVA /
    t-test / Kruskal group rows, crosstab levels, KM curves, etc.).

    Sort by the underlying value code numerically when every distinct value is
    numeric-coercible, else lexicographically by string. Without this, groups
    follow their order of appearance in the data, so results come out scrambled
    relative to the value labels (e.g. 3, 1, 2 instead of 1, 2, 3).
    """
    vals = list(pd.Series(series).dropna().unique())
    try:
        return sorted(vals, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(vals, key=str)


# ── Categorical p-value with small-cell rule ───────────────────────────────────


def _fisher_freeman_halton_mc(
    observed: np.ndarray, n_resamples: int = 5000, seed: int = 42
) -> float:
    """Monte Carlo p-value for r×c independence via Fisher-Freeman-Halton.

    Preserves row totals (categories) and permutes column assignments
    (groups). Returns a one-tailed p-value in the upper tail of the chi-square
    statistic under the null.
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.sum() <= 0:
        return float("nan")
    n_rows, n_cols = obs.shape

    cats_list: list[int] = []
    grps_list: list[int] = []
    for i in range(n_rows):
        for j in range(n_cols):
            n_ij = int(obs[i, j])
            if n_ij > 0:
                cats_list.extend([i] * n_ij)
                grps_list.extend([j] * n_ij)
    cats = np.asarray(cats_list, dtype=np.int64)
    grps = np.asarray(grps_list, dtype=np.int64)

    def _chi(ct: np.ndarray) -> float:
        rs = ct.sum(axis=1, keepdims=True)
        cs = ct.sum(axis=0, keepdims=True)
        total = ct.sum()
        if total <= 0:
            return 0.0
        e = rs * cs / total
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(((ct - e) ** 2 / np.where(e > 0, e, 1)).sum())

    obs_chi = _chi(obs)
    rng = np.random.default_rng(seed)
    minlength = n_rows * n_cols
    count = 0
    for _ in range(n_resamples):
        perm = rng.permutation(grps)
        enc = cats * n_cols + perm
        ct = np.bincount(enc, minlength=minlength).reshape(n_rows, n_cols).astype(float)
        if _chi(ct) >= obs_chi - 1e-9:
            count += 1
    return (count + 1) / (n_resamples + 1)


def _categorical_p_with_rule(ct: np.ndarray) -> tuple[float, str]:
    """Categorical association p-value with small-cell fallback rule.

    Runs a chi-square test of independence. If any expected cell count is < 5,
    fall back to Fisher's exact test for 2×2 tables or a Fisher-Freeman-Halton
    Monte Carlo permutation test for larger r×c tables.
    """
    obs = np.asarray(ct, dtype=float)
    chi2, p_chi, dof, expected = sp.chi2_contingency(obs)
    if (expected < 5).any():
        if obs.shape == (2, 2):
            _, p_fisher = sp.fisher_exact(obs)
            return float(p_fisher), "Fisher"
        return _fisher_freeman_halton_mc(obs), "Fisher-Freeman-Halton (MC)"
    return float(p_chi), "Chi-square"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SHARED RESULT CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AnalysisResult:
    """Standard result envelope returned by every analysis endpoint."""

    test: str  # e.g. "Independent t-test"
    statistic: Optional[float] = None  # test statistic (t, F, U, chi2, etc.)
    statistic_label: str = "Statistic"  # label for the statistic
    p: Optional[float] = None  # p-value
    significant: Optional[bool] = None  # p < alpha
    interpretation: str = ""  # one-sentence human-readable
    result_text: str = ""  # longer plain-English explanation

    # Effect sizes
    effect_sizes: list = field(
        default_factory=list
    )  # [{name, value, ci_low, ci_high, magnitude}]

    # Assumption checks
    assumptions: list = field(default_factory=list)  # [{name, met, detail}]

    # Warnings
    warnings: list = field(default_factory=list)  # [str]

    # Summary statistics per group
    summary: dict = field(
        default_factory=dict
    )  # {group_name: {n, mean, sd, median, ...}}

    # Post-hoc results
    posthoc: list = field(
        default_factory=list
    )  # [{group1, group2, statistic, p, p_adj, ...}]

    # Export-ready rows
    export_rows: list = field(default_factory=list)  # [[col1, col2, ...]] for CSV/Excel

    # Extra payload (test-specific)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None values and empty lists for cleaner JSON
        return {
            k: v
            for k, v in d.items()
            if v is not None and v != [] and v != {} and v != ""
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EFFECT SIZE CALCULATORS WITH CI
# ═══════════════════════════════════════════════════════════════════════════════


def _es_magnitude(name: str, val: float) -> str:
    """Cohen's magnitude label for common effect sizes."""
    v = abs(val)
    if name in ("cohen_d", "hedges_g"):
        if v < 0.2:
            return "negligible"
        if v < 0.5:
            return "small"
        if v < 0.8:
            return "medium"
        return "large"
    if name == "cohen_f":
        if v < 0.10:
            return "negligible"
        if v < 0.25:
            return "small"
        if v < 0.40:
            return "medium"
        return "large"
    if name in ("r", "pearson_r", "point_biserial_r"):
        if v < 0.10:
            return "negligible"
        if v < 0.30:
            return "small"
        if v < 0.50:
            return "medium"
        return "large"
    if name in ("eta_squared", "eta2"):
        if v < 0.01:
            return "negligible"
        if v < 0.06:
            return "small"
        if v < 0.14:
            return "medium"
        return "large"
    if name in ("cramers_v", "cramer_v"):
        if v < 0.10:
            return "negligible"
        if v < 0.30:
            return "small"
        if v < 0.50:
            return "medium"
        return "large"
    if name == "odds_ratio":
        if v < 1.5:
            return "negligible"
        if v < 2.5:
            return "small"
        if v < 4.0:
            return "medium"
        return "large"
    if name == "rank_biserial_r":
        if v < 0.10:
            return "negligible"
        if v < 0.30:
            return "small"
        if v < 0.50:
            return "medium"
        return "large"
    return ""


def welch_satterthwaite_df(g1: np.ndarray, g2: np.ndarray) -> float:
    """Welch–Satterthwaite degrees of freedom for two independent samples.

    Fractional by construction, and always ≤ the pooled n1 + n2 − 2. A Welch
    test reported against the pooled df understates the tail and overstates
    significance, so the two must never be mixed.
    """
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1 = float(np.var(g1, ddof=1))
    v2 = float(np.var(g2, ddof=1))
    a, b = v1 / n1, v2 / n2
    denom = (a ** 2) / (n1 - 1) + (b ** 2) / (n2 - 1)
    if denom <= 0:
        return float(n1 + n2 - 2)
    return float((a + b) ** 2 / denom)


def cohen_d(g1: np.ndarray, g2: np.ndarray) -> dict:
    """Cohen's d with 95% CI (Hedges-corrected = Hedges' g for small samples)."""
    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return {
            "name": "cohen_d",
            "value": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "magnitude": "negligible",
        }
    d = (m1 - m2) / sp
    # Hedges' correction for small samples
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    g = d * j
    # CI via non-central t approximation
    se = (
        np.sqrt(n1 + n2)
        / np.sqrt(n1 * n2)
        * np.sqrt(1 + g**2 * n1 * n2 / (2 * (n1 + n2)))
    )
    ci_lo = g - 1.96 * se
    ci_hi = g + 1.96 * se
    return {
        "name": "hedges_g",
        "value": round(g, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "magnitude": _es_magnitude("hedges_g", g),
    }


def eta_squared(f_stat: float, df_between: int, df_within: int) -> dict:
    """Eta-squared from ANOVA F-statistic."""
    ss_between = f_stat * df_between
    ss_total = ss_between + df_within
    eta2 = ss_between / ss_total if ss_total > 0 else 0
    # CI via F-to-R2 transformation (approximate)
    return {
        "name": "eta_squared",
        "value": round(eta2, 4),
        "ci_low": None,
        "ci_high": None,
        "magnitude": _es_magnitude("eta_squared", eta2),
    }


def partial_eta_squared(f_stat: float, df_between: int, df_within: int) -> dict:
    """Partial eta-squared."""
    peta2 = (f_stat * df_between) / (f_stat * df_between + df_within)
    return {
        "name": "partial_eta_squared",
        "value": round(peta2, 4),
        "ci_low": None,
        "ci_high": None,
        "magnitude": _es_magnitude("eta_squared", peta2),
    }


def omega_squared(
    f_stat: float, df_between: int, df_within: int, ms_within: float
) -> dict:
    """Omega-squared (less biased than eta-squared)."""
    ss_between = f_stat * df_between * ms_within
    ss_total = ss_between + df_within * ms_within
    omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    omega2 = max(0, omega2)
    return {
        "name": "omega_squared",
        "value": round(omega2, 4),
        "ci_low": None,
        "ci_high": None,
        "magnitude": _es_magnitude("eta_squared", omega2),
    }


def rank_biserial_r(u_stat: float, n1: int, n2: int) -> dict:
    """Rank-biserial correlation from Mann-Whitney U."""
    r = 1 - (2 * u_stat) / (n1 * n2)
    # CI via Fisher z-transform
    se = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
    z = np.arctanh(r)
    ci_lo = np.tanh(z - 1.96 * se)
    ci_hi = np.tanh(z + 1.96 * se)
    return {
        "name": "rank_biserial_r",
        "value": round(r, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "magnitude": _es_magnitude("rank_biserial_r", r),
    }


def cramers_v(chi2: float, n: int, min_dim: int) -> dict:
    """Cramer's V with bias correction."""
    k = max(min_dim - 1, 1)
    v = np.sqrt(chi2 / (n * k)) if n > 0 else 0
    return {
        "name": "cramers_v",
        "value": round(v, 4),
        "ci_low": None,
        "ci_high": None,
        "magnitude": _es_magnitude("cramers_v", v),
    }


def odds_ratio_effect(table: np.ndarray) -> dict:
    """Odds ratio with 95% CI from a 2x2 contingency table."""
    a, b = table[0]
    c, d = table[1]
    if b == 0 or c == 0 or d == 0 or a == 0:
        # Add 0.5 continuity correction
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (a * d) / (b * c)
    se_log = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_or = np.log(or_val)
    ci_lo = np.exp(log_or - 1.96 * se_log)
    ci_hi = np.exp(log_or + 1.96 * se_log)
    return {
        "name": "odds_ratio",
        "value": round(or_val, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "magnitude": _es_magnitude("odds_ratio", or_val),
    }


def cohen_d_one_sample(x: np.ndarray, mu: float) -> dict:
    """Cohen's d for one-sample: (mean - mu) / sd."""
    n = len(x)
    d = (x.mean() - mu) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0
    se = np.sqrt(1 / n + d**2 / (2 * n))
    ci_lo = d - 1.96 * se
    ci_hi = d + 1.96 * se
    return {
        "name": "cohen_d",
        "value": round(d, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "magnitude": _es_magnitude("cohen_d", d),
    }


def epsilon_squared(h_stat: float, n: int) -> dict:
    """Epsilon-squared for Kruskal-Wallis (rank-based eta-squared analogue)."""
    eps2 = (h_stat - 1) / (n - 1) if n > 1 else 0  # actually (H) / (n^2-1)/(n+1)
    eps2 = max(0, h_stat / ((n**2 - 1) / (n + 1)))
    return {
        "name": "epsilon_squared",
        "value": round(eps2, 4),
        "ci_low": None,
        "ci_high": None,
        "magnitude": _es_magnitude("eta_squared", eps2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MULTIPLICITY CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════


def adjust_pvalues(p_values: list[float], method: str = "holm") -> list[float]:
    """Adjust p-values for multiple comparisons.
    Methods: bonferroni, holm, fdr (Benjamini-Hochberg), none.
    """
    m = len(p_values)
    if m == 0:
        return []
    ps = np.array(p_values, dtype=float)

    if method == "bonferroni":
        return np.minimum(ps * m, 1.0).tolist()

    if method == "holm":
        order = np.argsort(ps)
        adjusted = np.empty(m)
        cummax = 0.0
        for i, idx in enumerate(order):
            val = ps[idx] * (m - i)
            cummax = max(cummax, val)
            adjusted[idx] = min(cummax, 1.0)
        return adjusted.tolist()

    if method == "fdr":
        order = np.argsort(ps)[::-1]  # descending
        adjusted = np.empty(m)
        cummin = 1.0
        for i, idx in enumerate(order):
            rank = m - i
            val = ps[idx] * m / rank
            cummin = min(cummin, val)
            adjusted[idx] = min(cummin, 1.0)
        return adjusted.tolist()

    # "none"
    return ps.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PAIRWISE COMPARISON BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════


def pairwise_t_tests(
    groups: dict[str, np.ndarray], correction: str = "holm", equal_var: bool = True
) -> list[dict]:
    """Run all pairwise t-tests with multiplicity correction."""
    names = list(groups.keys())
    pairs = [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    results = []
    raw_ps = []

    for g1_name, g2_name in pairs:
        g1, g2 = groups[g1_name], groups[g2_name]
        t_stat, p_val = sp.ttest_ind(g1, g2, equal_var=equal_var)
        d = cohen_d(g1, g2)
        results.append(
            {
                "group1": g1_name,
                "group2": g2_name,
                "statistic": round(float(t_stat), 4),
                "p": float(p_val),
                "mean_diff": round(float(g1.mean() - g2.mean()), 4),
                "effect_size": d,
            }
        )
        raw_ps.append(p_val)

    adj_ps = adjust_pvalues(raw_ps, correction)
    for i, r in enumerate(results):
        r["p_adj"] = round(adj_ps[i], 6)
        r["significant"] = adj_ps[i] < 0.05
        r["correction"] = correction
    return results


def pairwise_wilcoxon(
    groups: dict[str, np.ndarray], correction: str = "holm"
) -> list[dict]:
    """Pairwise Mann-Whitney U tests with multiplicity correction."""
    names = list(groups.keys())
    pairs = [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    results = []
    raw_ps = []

    for g1_name, g2_name in pairs:
        g1, g2 = groups[g1_name], groups[g2_name]
        u_stat, p_val = sp.mannwhitneyu(g1, g2, alternative="two-sided")
        r_es = rank_biserial_r(float(u_stat), len(g1), len(g2))
        results.append(
            {
                "group1": g1_name,
                "group2": g2_name,
                "statistic": round(float(u_stat), 4),
                "p": float(p_val),
                "effect_size": r_es,
            }
        )
        raw_ps.append(p_val)

    adj_ps = adjust_pvalues(raw_ps, correction)
    for i, r in enumerate(results):
        r["p_adj"] = round(adj_ps[i], 6)
        r["significant"] = adj_ps[i] < 0.05
        r["correction"] = correction
    return results


def tukey_hsd(groups: dict[str, np.ndarray]) -> list[dict]:
    """Tukey's HSD post-hoc test."""
    names = list(groups.keys())
    arrays = [groups[n] for n in names]
    try:
        res = sp.tukey_hsd(*arrays)
        results = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p_val = float(res.pvalue[i][j])
                d = cohen_d(arrays[i], arrays[j])
                results.append(
                    {
                        "group1": names[i],
                        "group2": names[j],
                        "statistic": round(float(res.statistic[i][j]), 4),
                        "p_adj": round(p_val, 6),
                        "significant": p_val < 0.05,
                        "mean_diff": round(
                            float(arrays[i].mean() - arrays[j].mean()), 4
                        ),
                        "effect_size": d,
                        "correction": "tukey_hsd",
                    }
                )
        return results
    except Exception:
        # Fallback to pairwise t with Bonferroni if Tukey fails
        return pairwise_t_tests(groups, correction="bonferroni")


def games_howell(groups: dict[str, np.ndarray]) -> list[dict]:
    """Games-Howell post-hoc test (unequal variances)."""
    names = list(groups.keys())
    results = []
    raw_ps = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            g1, g2 = groups[names[i]], groups[names[j]]
            n1, n2 = len(g1), len(g2)
            m1, m2 = g1.mean(), g2.mean()
            v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
            se = np.sqrt(v1 / n1 + v2 / n2)
            if se == 0:
                results.append(
                    {
                        "group1": names[i],
                        "group2": names[j],
                        "statistic": 0,
                        "p": 1.0,
                        "p_adj": 1.0,
                        "significant": False,
                        "mean_diff": 0,
                        "effect_size": cohen_d(g1, g2),
                        "correction": "games_howell",
                    }
                )
                raw_ps.append(1.0)
                continue
            t_stat = (m1 - m2) / se
            df = welch_satterthwaite_df(g1, g2)
            p_val = float(2 * sp.t.sf(abs(t_stat), df))
            d = cohen_d(g1, g2)
            results.append(
                {
                    "group1": names[i],
                    "group2": names[j],
                    "statistic": round(float(t_stat), 4),
                    "p": round(p_val, 6),
                    "mean_diff": round(float(m1 - m2), 4),
                    "effect_size": d,
                    "correction": "games_howell",
                }
            )
            raw_ps.append(p_val)

    # Games-Howell uses studentized range for correction, but
    # for simplicity we apply Holm correction here
    adj_ps = adjust_pvalues(raw_ps, "holm")
    for i, r in enumerate(results):
        r["p_adj"] = round(adj_ps[i], 6)
        r["significant"] = adj_ps[i] < 0.05
    return results


def dunn_test(groups: dict[str, np.ndarray], correction: str = "holm") -> list[dict]:
    """Dunn's test for pairwise comparisons after Kruskal-Wallis."""
    names = list(groups.keys())
    all_data = np.concatenate([groups[n] for n in names])
    ranks = sp.rankdata(all_data)
    N = len(all_data)

    # Assign ranks back to groups
    idx = 0
    group_ranks = {}
    for n in names:
        g = groups[n]
        group_ranks[n] = ranks[idx : idx + len(g)]
        idx += len(g)

    results = []
    raw_ps = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1 = len(groups[names[i]])
            n2 = len(groups[names[j]])
            r1_mean = group_ranks[names[i]].mean()
            r2_mean = group_ranks[names[j]].mean()
            se = np.sqrt((N * (N + 1) / 12) * (1 / n1 + 1 / n2))
            if se == 0:
                results.append(
                    {
                        "group1": names[i],
                        "group2": names[j],
                        "statistic": 0,
                        "p": 1.0,
                        "rank_diff": 0,
                        "correction": correction,
                    }
                )
                raw_ps.append(1.0)
                continue
            z = (r1_mean - r2_mean) / se
            p_val = float(2 * sp.norm.sf(abs(z)))
            results.append(
                {
                    "group1": names[i],
                    "group2": names[j],
                    "statistic": round(float(z), 4),
                    "p": round(p_val, 6),
                    "rank_diff": round(float(r1_mean - r2_mean), 2),
                    "correction": correction,
                }
            )
            raw_ps.append(p_val)

    adj_ps = adjust_pvalues(raw_ps, correction)
    for i, r in enumerate(results):
        r["p_adj"] = round(adj_ps[i], 6)
        r["significant"] = adj_ps[i] < 0.05
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ASSUMPTION CHECKS
# ═══════════════════════════════════════════════════════════════════════════════


def check_normality(x: np.ndarray, label: str = "Sample") -> dict:
    """Test normality using the appropriate test for sample size.

    Tier 1: n < 50  → Shapiro-Wilk (most powerful for small samples)
    Tier 2: 50 ≤ n ≤ 2000 → Kolmogorov-Smirnov with Lilliefors correction
    Tier 3: n > 2000 → CLT skewness bypass (|skew| ≤ 1.5) → Lilliefors
    """
    n = len(x)
    if n < 3:
        return {
            "name": f"Normality ({label})",
            "met": True,
            "detail": "Too few obs to test",
        }
    if np.std(x) == 0:
        return {
            "name": f"Normality ({label})",
            "met": True,
            "detail": "Constant values (no variation)",
        }

    if n < 50:
        # Small sample — Shapiro-Wilk is most powerful
        stat, p = sp.shapiro(x)
        if np.isnan(p):
            return {
                "name": f"Normality ({label})",
                "met": True,
                "detail": "Test inconclusive",
            }
        test_name = "Shapiro-Wilk"
    elif n <= 2000:
        # Medium sample — Kolmogorov-Smirnov with Lilliefors correction
        from statsmodels.stats.diagnostic import lilliefors as _lf

        stat, p = _lf(x, dist="norm")
        test_name = "Kolmogorov-Smirnov (Lilliefors)"
    else:
        # Large sample — CLT bypass if skewness is acceptable
        skew = float(sp.skew(x))
        if abs(skew) <= 1.5:
            return {
                "name": f"Normality ({label})",
                "met": True,
                "detail": f"CLT bypass (n={n}, |skewness|={abs(skew):.2f} ≤ 1.5)",
            }
        from statsmodels.stats.diagnostic import lilliefors as _lf

        stat, p = _lf(x, dist="norm")
        test_name = "Kolmogorov-Smirnov (Lilliefors)"

    return {
        "name": f"Normality ({label})",
        "met": bool(p >= 0.05),
        "detail": f"{test_name}: p = {p:.4f}",
    }


def check_equal_variances(
    groups: list[np.ndarray],
    names: list[str],
    on_violation: str = "",
) -> dict:
    """Levene's test for homogeneity of variances.

    `on_violation` states what the CALLER actually does when the assumption
    fails. It used to be hardcoded to "using Welch correction", which was true
    for the t-test but false for one-way ANOVA — that omnibus F stays the
    classic equal-variance statistic and only the post-hoc switches. Reporting
    a correction that was never applied is a false methods claim, so each
    caller now supplies its own wording.
    """
    if len(groups) < 2:
        return {"name": "Equal variances", "met": True, "detail": "Single group"}
    stat, p = sp.levene(*groups)
    violated = p < 0.05
    suffix = f" — violated, {on_violation}" if (violated and on_violation) else (
        " — violated" if violated else ""
    )
    return {
        "name": "Equal variances (Levene)",
        "met": bool(not violated),
        "detail": f"F = {stat:.3f}, p = {p:.4f}{suffix}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GROUP SUMMARY BUILDER
# ═══════════════════════════════════════════════════════════════════════════════


def group_summary(x: np.ndarray, label: str = "Sample") -> dict:
    """Standard descriptive statistics for a numeric array."""
    return {
        "label": label,
        "n": int(len(x)),
        "mean": round(float(x.mean()), 4),
        "sd": round(float(x.std(ddof=1)), 4),
        "median": round(float(np.median(x)), 4),
        "q1": round(float(np.percentile(x, 25)), 4),
        "q3": round(float(np.percentile(x, 75)), 4),
        "min": round(float(x.min()), 4),
        "max": round(float(x.max()), 4),
    }


def cohen_d_paired(d: np.ndarray) -> dict:
    """Cohen's d_z for paired differences with 95% CI."""
    n = len(d)
    sd = float(d.std(ddof=1))
    if sd == 0 or np.isnan(sd):
        return {
            "name": "cohen_d_z",
            "value": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "magnitude": "negligible",
        }
    dz = float(d.mean()) / sd
    se = np.sqrt(1 / n + dz**2 / (2 * n))
    ci_lo = dz - 1.96 * se
    ci_hi = dz + 1.96 * se
    return {
        "name": "cohen_d_z",
        "value": round(dz, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "magnitude": _es_magnitude("cohen_d", dz),
    }


def kendalls_w(chi2: float, n: int, k: int) -> dict:
    """Kendall's W concordance coefficient for Friedman test."""
    w = chi2 / (n * (k - 1)) if n > 0 and k > 1 else 0.0
    return {
        "name": "kendalls_w",
        "value": round(w, 4),
        "ci_low": None,
        "ci_high": None,
        "magnitude": _es_magnitude("eta_squared", w),
    }


def matched_rank_biserial(w_stat: float, n: int) -> dict:
    """Matched-pairs rank-biserial r from Wilcoxon signed-rank W."""
    max_w = n * (n + 1) / 2
    r = 2 * w_stat / max_w - 1 if max_w > 0 else 0.0
    se = np.sqrt((2 * n + 1) / (6 * n)) if n > 0 else 0
    ci_lo = max(-1, r - 1.96 * se)
    ci_hi = min(1, r + 1.96 * se)
    return {
        "name": "rank_biserial_r",
        "value": round(r, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "magnitude": _es_magnitude("rank_biserial_r", r),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BOOTSTRAP CI & PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════════


def bootstrap_ci(
    data: np.ndarray, statistic_fn, n_boot: int = 2000, ci: float = 0.95, seed: int = 42
) -> dict:
    """Bootstrap confidence interval for any statistic function.
    statistic_fn takes an array and returns a scalar."""
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = np.array(
        [statistic_fn(data[rng.randint(0, n, n)]) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, alpha * 100))
    hi = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return {
        "estimate": float(statistic_fn(data)),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n_boot": n_boot,
        "method": "percentile bootstrap",
    }


def bootstrap_ci_two(
    x: np.ndarray,
    y: np.ndarray,
    statistic_fn,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for a two-sample statistic (e.g. mean difference)."""
    rng = np.random.RandomState(seed)
    nx, ny = len(x), len(y)
    boot_stats = []
    for _ in range(n_boot):
        bx = x[rng.randint(0, nx, nx)]
        by = y[rng.randint(0, ny, ny)]
        boot_stats.append(statistic_fn(bx, by))
    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, alpha * 100))
    hi = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return {
        "estimate": float(statistic_fn(x, y)),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n_boot": n_boot,
        "method": "percentile bootstrap",
    }


def permutation_test(
    x: np.ndarray, y: np.ndarray, statistic_fn=None, n_perm: int = 5000, seed: int = 42
) -> dict:
    """Two-sample permutation test. Default statistic: difference of means."""
    if statistic_fn is None:

        def statistic_fn(a, b):
            return float(a.mean() - b.mean())

    rng = np.random.RandomState(seed)
    observed = statistic_fn(x, y)
    combined = np.concatenate([x, y])
    nx = len(x)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        perm_stat = statistic_fn(perm[:nx], perm[nx:])
        if abs(perm_stat) >= abs(observed):
            count += 1
    p = (count + 1) / (n_perm + 1)  # +1 to include observed
    return {
        "observed_statistic": round(observed, 4),
        "p_permutation": round(p, 6),
        "n_permutations": n_perm,
        "significant": p < 0.05,
    }


def cohens_h(p1: float, p2: float) -> dict:
    """Cohen's h for comparing two proportions."""
    h = 2 * (
        np.arcsin(np.sqrt(max(0, min(1, p1)))) - np.arcsin(np.sqrt(max(0, min(1, p2))))
    )
    return {
        "name": "cohens_h",
        "value": round(float(h), 4),
        "ci_low": None,
        "ci_high": None,
        "magnitude": _es_magnitude("cohen_d", h),
    }


def lins_ccc(x: np.ndarray, y: np.ndarray) -> dict:
    """Lin's concordance correlation coefficient with 95% CI."""
    n = len(x)
    mx, my = float(x.mean()), float(y.mean())
    sx, sy = float(x.std(ddof=1)), float(y.std(ddof=1))
    r = float(np.corrcoef(x, y)[0, 1]) if sx > 0 and sy > 0 else 0.0
    precision = r
    denom = sx**2 + sy**2 + (mx - my) ** 2
    accuracy = 2 * sx * sy / denom if denom > 0 else 0.0
    ccc = precision * accuracy
    if n > 3 and abs(ccc) < 1:
        z = np.arctanh(ccc)
        se = np.sqrt(1 / (n - 3))
        ci_lo = float(np.tanh(z - 1.96 * se))
        ci_hi = float(np.tanh(z + 1.96 * se))
    else:
        ci_lo, ci_hi = float(ccc), float(ccc)
    return {
        "name": "lins_ccc",
        "value": round(float(ccc), 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "precision": round(float(precision), 4),
        "accuracy": round(float(accuracy), 4),
        "magnitude": _es_magnitude("r", ccc),
    }
