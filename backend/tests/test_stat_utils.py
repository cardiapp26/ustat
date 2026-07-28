"""
Unit tests for services/stat_utils.py.

This is a pure service module (no FastAPI router), so we import and test the
exported helper functions directly with known inputs/outputs.

Session-id prefix for this module: "tutil".
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services import stat_utils as su

SEED = 20260531
TUTIL_PREFIX = "tutil"  # module-unique prefix (no store collisions even though unused)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="module")
def two_groups(rng):
    """Two clearly separated normal samples (large effect)."""
    g1 = rng.normal(10.0, 2.0, 60)
    g2 = rng.normal(13.0, 2.0, 60)
    return g1, g2


@pytest.fixture(scope="module")
def three_groups(rng):
    return {
        f"{TUTIL_PREFIX}_A": rng.normal(10.0, 2.0, 40),
        f"{TUTIL_PREFIX}_B": rng.normal(12.0, 2.0, 40),
        f"{TUTIL_PREFIX}_C": rng.normal(14.0, 2.0, 40),
    }


# ───────────────────────────────────────────────────────────────────────────────
# AnalysisResult contract
# ───────────────────────────────────────────────────────────────────────────────


def test_analysis_result_to_dict_strips_empties():
    res = su.AnalysisResult(
        test="t-test", statistic=2.5, p=0.01, significant=True, interpretation="sig"
    )
    d = res.to_dict()
    assert d["test"] == "t-test"
    assert d["statistic"] == 2.5
    assert d["p"] == 0.01
    assert d["significant"] is True
    # Empty defaults must be stripped out
    for empty_key in (
        "effect_sizes",
        "assumptions",
        "warnings",
        "summary",
        "posthoc",
        "export_rows",
        "extra",
        "result_text",
    ):
        assert empty_key not in d


def test_analysis_result_keeps_populated_fields():
    res = su.AnalysisResult(test="anova", warnings=["w1"], summary={"g": {"n": 3}})
    d = res.to_dict()
    assert d["warnings"] == ["w1"]
    assert d["summary"] == {"g": {"n": 3}}


# ───────────────────────────────────────────────────────────────────────────────
# Effect size calculators
# ───────────────────────────────────────────────────────────────────────────────


def test_cohen_d_shape_and_sign(two_groups):
    g1, g2 = two_groups
    out = su.cohen_d(g1, g2)
    assert set(out.keys()) == {"name", "value", "ci_low", "ci_high", "magnitude"}
    assert out["name"] == "hedges_g"
    # g1 mean < g2 mean -> negative effect
    assert out["value"] < 0
    assert out["ci_low"] <= out["value"] <= out["ci_high"]
    assert out["magnitude"] in {"negligible", "small", "medium", "large"}


def test_cohen_d_zero_pooled_sd():
    const = np.array([5.0, 5.0, 5.0, 5.0])
    out = su.cohen_d(const, const)
    assert out["value"] == 0.0
    assert out["magnitude"] == "negligible"


def test_eta_squared_range():
    out = su.eta_squared(f_stat=10.0, df_between=2, df_within=57)
    assert out["name"] == "eta_squared"
    assert 0.0 <= out["value"] <= 1.0
    assert out["magnitude"] in {"negligible", "small", "medium", "large"}


def test_partial_eta_squared_range():
    out = su.partial_eta_squared(f_stat=10.0, df_between=2, df_within=57)
    assert out["name"] == "partial_eta_squared"
    assert 0.0 <= out["value"] <= 1.0


def test_omega_squared_nonnegative():
    out = su.omega_squared(f_stat=10.0, df_between=2, df_within=57, ms_within=4.0)
    assert out["name"] == "omega_squared"
    assert out["value"] >= 0.0


def test_rank_biserial_r_range():
    out = su.rank_biserial_r(u_stat=200.0, n1=30, n2=30)
    assert out["name"] == "rank_biserial_r"
    assert -1.0 <= out["value"] <= 1.0
    assert out["ci_low"] <= out["ci_high"]


def test_cramers_v_range():
    out = su.cramers_v(chi2=12.0, n=200, min_dim=3)
    assert out["name"] == "cramers_v"
    assert 0.0 <= out["value"] <= 1.0


def test_odds_ratio_effect_known_value():
    # a=10,b=20 ; c=30,d=40 -> OR = (10*40)/(20*30) = 400/600 = 0.6667
    table = np.array([[10, 20], [30, 40]])
    out = su.odds_ratio_effect(table)
    assert out["name"] == "odds_ratio"
    assert abs(out["value"] - 0.6667) < 1e-2
    assert out["ci_low"] <= out["value"] <= out["ci_high"]


def test_odds_ratio_continuity_correction_with_zero_cell():
    # zero cell triggers +0.5 continuity correction, must not crash / div-by-zero
    table = np.array([[0, 5], [8, 12]])
    out = su.odds_ratio_effect(table)
    assert out["value"] > 0
    assert np.isfinite(out["ci_low"]) and np.isfinite(out["ci_high"])


def test_cohen_d_one_sample(rng):
    x = rng.normal(5.0, 1.0, 50)
    out = su.cohen_d_one_sample(x, mu=4.0)
    assert out["name"] == "cohen_d"
    assert out["value"] > 0  # mean(~5) > mu(4)
    assert out["ci_low"] <= out["value"] <= out["ci_high"]


def test_epsilon_squared_nonnegative():
    out = su.epsilon_squared(h_stat=15.0, n=60)
    assert out["name"] == "epsilon_squared"
    assert out["value"] >= 0.0


def test_cohen_d_paired(rng):
    diffs = rng.normal(1.5, 1.0, 40)
    out = su.cohen_d_paired(diffs)
    assert out["name"] == "cohen_d_z"
    assert out["value"] > 0
    assert out["ci_low"] <= out["ci_high"]


def test_cohen_d_paired_zero_sd():
    out = su.cohen_d_paired(np.array([2.0, 2.0, 2.0]))
    assert out["value"] == 0.0
    assert out["magnitude"] == "negligible"


def test_kendalls_w_range():
    out = su.kendalls_w(chi2=20.0, n=15, k=4)
    assert out["name"] == "kendalls_w"
    assert 0.0 <= out["value"] <= 1.0


def test_matched_rank_biserial_range():
    out = su.matched_rank_biserial(w_plus=50.0, n=20)
    assert out["name"] == "rank_biserial_r"
    assert -1.0 <= out["value"] <= 1.0
    assert -1.0 <= out["ci_low"] <= out["ci_high"] <= 1.0


def test_matched_rank_biserial_carries_the_sign():
    """The argument is the sum of POSITIVE signed ranks, so all-positive and
    all-negative differences must land at opposite ends."""
    n = 6
    max_w = n * (n + 1) / 2
    assert su.matched_rank_biserial(max_w, n)["value"] == 1.0
    assert su.matched_rank_biserial(0.0, n)["value"] == -1.0


def test_cohens_h_bounds():
    out = su.cohens_h(0.5, 0.2)
    assert out["name"] == "cohens_h"
    assert out["value"] > 0  # p1 > p2
    # h is bounded by [-pi, pi]
    assert abs(out["value"]) <= np.pi


def test_lins_ccc_high_agreement(rng):
    x = rng.normal(0, 1, 100)
    y = x + rng.normal(0, 0.05, 100)  # nearly identical
    out = su.lins_ccc(x, y)
    assert out["name"] == "lins_ccc"
    assert 0.8 < out["value"] <= 1.0
    assert out["ci_low"] <= out["value"] <= out["ci_high"]
    assert -1.0 <= out["precision"] <= 1.0
    assert 0.0 <= out["accuracy"] <= 1.0


# ───────────────────────────────────────────────────────────────────────────────
# Magnitude labels
# ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,val,expected",
    [
        ("hedges_g", 0.1, "negligible"),
        ("hedges_g", 0.3, "small"),
        ("hedges_g", 0.6, "medium"),
        ("hedges_g", 1.0, "large"),
        ("cramers_v", 0.05, "negligible"),
        ("eta_squared", 0.20, "large"),
        ("odds_ratio", 1.2, "negligible"),
        ("odds_ratio", 5.0, "large"),
        ("unknown_es", 0.5, ""),
    ],
)
def test_es_magnitude_labels(name, val, expected):
    assert su._es_magnitude(name, val) == expected


# ───────────────────────────────────────────────────────────────────────────────
# Multiplicity correction
# ───────────────────────────────────────────────────────────────────────────────


def test_adjust_pvalues_empty():
    assert su.adjust_pvalues([], "holm") == []


def test_adjust_pvalues_bonferroni():
    out = su.adjust_pvalues([0.01, 0.02, 0.5], "bonferroni")
    assert out == pytest.approx([0.03, 0.06, 1.0])  # 0.5*3=1.5 -> capped at 1.0


def test_adjust_pvalues_holm_monotone_and_capped():
    raw = [0.001, 0.02, 0.03, 0.5]
    out = su.adjust_pvalues(raw, "holm")
    assert all(0.0 <= p <= 1.0 for p in out)
    # adjusted >= raw for each
    assert all(adj >= r - 1e-9 for adj, r in zip(out, raw))


def test_adjust_pvalues_fdr_capped():
    out = su.adjust_pvalues([0.001, 0.01, 0.04, 0.9], "fdr")
    assert all(0.0 <= p <= 1.0 for p in out)


def test_adjust_pvalues_none_passthrough():
    raw = [0.1, 0.2, 0.3]
    assert su.adjust_pvalues(raw, "none") == pytest.approx(raw)


# ───────────────────────────────────────────────────────────────────────────────
# Pairwise comparison builders
# ───────────────────────────────────────────────────────────────────────────────


def _check_pairwise_shape(results, n_groups, has_padj=True):
    expected_pairs = n_groups * (n_groups - 1) // 2
    assert len(results) == expected_pairs
    for r in results:
        assert "group1" in r and "group2" in r
        assert "statistic" in r
        if has_padj:
            assert "p_adj" in r
            assert "significant" in r
            assert 0.0 <= r["p_adj"] <= 1.0


def test_pairwise_t_tests(three_groups):
    res = su.pairwise_t_tests(three_groups, correction="holm")
    _check_pairwise_shape(res, 3)
    for r in res:
        assert r["correction"] == "holm"
        assert "effect_size" in r and r["effect_size"]["name"] == "hedges_g"


def test_pairwise_wilcoxon(three_groups):
    res = su.pairwise_wilcoxon(three_groups, correction="bonferroni")
    _check_pairwise_shape(res, 3)
    for r in res:
        assert r["correction"] == "bonferroni"
        assert r["effect_size"]["name"] == "rank_biserial_r"


def test_tukey_hsd(three_groups):
    res = su.tukey_hsd(three_groups)
    _check_pairwise_shape(res, 3)
    for r in res:
        assert "p_adj" in r
        # Either tukey_hsd succeeded or it fell back to bonferroni
        assert r["correction"] in {"tukey_hsd", "bonferroni"}


def test_games_howell(three_groups):
    res = su.games_howell(three_groups)
    _check_pairwise_shape(res, 3)
    for r in res:
        assert r["correction"] == "games_howell"


def test_dunn_test(three_groups):
    res = su.dunn_test(three_groups, correction="holm")
    _check_pairwise_shape(res, 3)
    for r in res:
        assert "rank_diff" in r
        assert r["correction"] == "holm"


# ───────────────────────────────────────────────────────────────────────────────
# Assumption checks
# ───────────────────────────────────────────────────────────────────────────────


def test_check_normality_small_normal(rng):
    x = rng.normal(0, 1, 30)
    out = su.check_normality(x, "S")
    assert out["name"].startswith("Normality")
    assert isinstance(out["met"], bool)
    assert "Shapiro-Wilk" in out["detail"]


def test_check_normality_too_few():
    out = su.check_normality(np.array([1.0, 2.0]), "S")
    assert out["met"] is True
    assert "Too few" in out["detail"]


def test_check_normality_constant():
    out = su.check_normality(np.array([3.0, 3.0, 3.0, 3.0, 3.0]), "S")
    assert out["met"] is True
    assert "Constant" in out["detail"]


def test_check_normality_medium_sample(rng):
    x = rng.normal(0, 1, 200)
    out = su.check_normality(x, "Med")
    assert "Kolmogorov-Smirnov" in out["detail"]
    assert isinstance(out["met"], bool)


def test_check_normality_large_clt_bypass(rng):
    x = rng.normal(0, 1, 3000)  # symmetric -> |skew| small -> CLT bypass
    out = su.check_normality(x, "Big")
    assert out["met"] is True
    assert "CLT bypass" in out["detail"]


def test_check_equal_variances_single_group():
    out = su.check_equal_variances([np.array([1.0, 2.0, 3.0])], ["only"])
    assert out["met"] is True
    assert out["detail"] == "Single group"


def test_check_equal_variances_levene(two_groups):
    g1, g2 = two_groups
    out = su.check_equal_variances([g1, g2], ["g1", "g2"])
    assert "Levene" in out["name"]
    assert isinstance(out["met"], bool)
    assert "F =" in out["detail"]


# ───────────────────────────────────────────────────────────────────────────────
# Group summary
# ───────────────────────────────────────────────────────────────────────────────


def test_group_summary_keys_and_order():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = su.group_summary(x, "G")
    assert out["label"] == "G"
    assert out["n"] == 5
    assert out["mean"] == 3.0
    assert out["median"] == 3.0
    assert out["min"] == 1.0
    assert out["max"] == 5.0
    assert out["q1"] <= out["median"] <= out["q3"]
    assert out["sd"] > 0


# ───────────────────────────────────────────────────────────────────────────────
# Bootstrap / permutation
# ───────────────────────────────────────────────────────────────────────────────


def test_bootstrap_ci_mean(rng):
    data = rng.normal(10.0, 2.0, 100)
    out = su.bootstrap_ci(data, np.mean, n_boot=300, seed=1)
    assert out["n_boot"] == 300
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]
    assert abs(out["estimate"] - 10.0) < 1.5


def test_bootstrap_ci_two_mean_diff(rng):
    x = rng.normal(10.0, 2.0, 80)
    y = rng.normal(8.0, 2.0, 80)

    def fn(a, b):
        return float(a.mean() - b.mean())

    out = su.bootstrap_ci_two(x, y, fn, n_boot=300, seed=2)
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]
    assert out["estimate"] > 0  # x mean > y mean


def test_permutation_test_separated(rng):
    x = rng.normal(10.0, 1.0, 50)
    y = rng.normal(14.0, 1.0, 50)
    out = su.permutation_test(x, y, n_perm=500, seed=3)
    assert 0.0 < out["p_permutation"] <= 1.0
    assert out["n_permutations"] == 500
    assert out["significant"] is True  # clearly separated


def test_permutation_test_identical_distribution(rng):
    x = rng.normal(0, 1, 50)
    y = rng.normal(0, 1, 50)
    out = su.permutation_test(x, y, n_perm=500, seed=4)
    assert 0.0 < out["p_permutation"] <= 1.0


# ───────────────────────────────────────────────────────────────────────────────
# Categorical p-value with small-cell rule
# ───────────────────────────────────────────────────────────────────────────────


def test_categorical_p_with_rule_uses_chisquare_when_expected_large():
    # Balanced 2x3 table: all expected counts well above 5.
    ct = np.array([[30, 25, 20], [20, 25, 30]])
    p, test_name = su._categorical_p_with_rule(ct)
    assert test_name == "Chi-square"
    assert 0.0 <= p <= 1.0


def test_categorical_p_with_rule_falls_back_to_fisher_for_2x2():
    # Small expected counts in a 2x2 -> Fisher exact.
    ct = np.array([[1, 9], [8, 2]])
    p, test_name = su._categorical_p_with_rule(ct)
    assert test_name == "Fisher"
    assert 0.0 < p <= 1.0


def test_categorical_p_with_rule_falls_back_to_ffh_for_rxC():
    # 3x3 table with a small expected cell -> Fisher-Freeman-Halton MC.
    ct = np.array([[0, 8, 12], [1, 10, 9], [2, 7, 11]])
    p, test_name = su._categorical_p_with_rule(ct)
    assert test_name == "Fisher-Freeman-Halton (MC)"
    assert 0.0 < p <= 1.0
