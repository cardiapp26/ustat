"""A crosstab must report the test that produced its p, and V must carry a CI.

Two defects motivated this file:

  * The Tests tab ran a plain chi-square on a table too sparse for one, printed
    "Consider Fisher's exact test instead" underneath, and left the reader with
    a p that Table 1 and the publication export both disagreed with — they had
    already fallen back to the exact test on the same table.
  * Cramer's V was reported as a bare number. An effect size with no interval
    cannot be read as evidence of a small effect rather than of a small study,
    which is the whole reason it is quoted next to a non-significant p.

The interval is checked against DescTools::CramerV(x, conf.level = 0.95) in R,
which uses the same noncentral chi-square inversion.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp

from conftest import make_session
from services.stat_utils import cramers_v
from services.text_generators import r_chisquare


# ── Cramer's V interval, against DescTools ────────────────────────────────────

# table, (V, ci_low, ci_high) as printed by R:
#   library(DescTools); CramerV(x, conf.level = 0.95)
R_REFERENCE = [
    (
        [[30, 10], [10, 30]],
        (0.5000000000, 0.2808781776, 0.7191320316),
    ),
    (
        [[12, 18, 15], [20, 14, 16], [9, 22, 17], [11, 13, 25]],
        (0.1719076274, 0.0000000000, 0.2396167315),
    ),
    (
        [[5, 1, 2], [3, 8, 2], [1, 2, 9]],
        (0.5019883561, 0.1886825220, 0.7049181796),
    ),
]


@pytest.mark.parametrize("table,expected", R_REFERENCE)
def test_cramers_v_interval_matches_desctools(table, expected):
    obs = np.array(table, dtype=float)
    chi2, _, dof, _ = sp.chi2_contingency(obs, correction=False)
    got = cramers_v(chi2, int(obs.sum()), min(obs.shape), dof)
    v, lo, hi = expected
    assert got["value"] == pytest.approx(v, abs=5e-5)
    assert got["ci_low"] == pytest.approx(lo, abs=5e-5)
    assert got["ci_high"] == pytest.approx(hi, abs=5e-5)


def test_a_weak_association_gets_a_lower_limit_of_zero():
    """The bound V cannot cross is 0, and the interval has to be able to reach
    it. A symmetric interval around a V of 0.17 would have claimed the effect
    was bounded away from nothing when the chi-square was not significant."""
    obs = np.array([[12, 18, 15], [20, 14, 16], [9, 22, 17], [11, 13, 25]], float)
    chi2, p, dof, _ = sp.chi2_contingency(obs, correction=False)
    assert p > 0.05
    assert cramers_v(chi2, int(obs.sum()), min(obs.shape), dof)["ci_low"] == 0.0


def test_the_interval_is_omitted_when_the_table_shape_is_unknown():
    """V and n alone do not determine the interval — the same V on a 2x2 and on
    a 5x4 carry different uncertainty — so no interval is invented."""
    out = cramers_v(chi2=12.0, n=200, min_dim=3)
    assert out["value"] == pytest.approx(np.sqrt(12.0 / 400), abs=1e-4)
    assert out["ci_low"] is None and out["ci_high"] is None


def test_the_interval_contains_the_point_estimate():
    for table, _ in R_REFERENCE:
        obs = np.array(table, dtype=float)
        chi2, _, dof, _ = sp.chi2_contingency(obs, correction=False)
        out = cramers_v(chi2, int(obs.sum()), min(obs.shape), dof)
        assert out["ci_low"] <= out["value"] <= out["ci_high"], table


# ── the exact fallback in the Tests tab ───────────────────────────────────────


def _frame() -> pd.DataFrame:
    """A sparse 3x2 crosstab beside a dense one, in a single frame."""
    rng = np.random.default_rng(7)
    n = 120
    return pd.DataFrame(
        {
            "arm": rng.choice(["A", "B"], n).tolist(),
            # 'stage' has a level that occurs twice in total, so its expected
            # counts fall well below 5.
            "stage": (["I"] * 59 + ["II"] * 59 + ["III"] * 2),
            "sex": rng.choice(["M", "F"], n).tolist(),
        }
    )


@pytest.fixture()
def sid() -> str:
    return make_session(_frame(), "crosstab_exact")


def _chisq(client, sid, row_col):
    r = client.post(
        "/api/stats/chisquare",
        json={"session_id": sid, "row_column": row_col, "col_column": "arm"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_a_sparse_table_is_reported_as_the_exact_test(client, sid):
    out = _chisq(client, sid, "stage")
    assert out["exact_test"].startswith("Fisher-Freeman-Halton")
    assert out["test"] == out["exact_test"]
    assert out["p"] != out["p_chisquare"]
    assert any("expected cell counts are below 5" in str(w) for w in out["warnings"])


def test_a_dense_table_is_still_a_chi_square(client, sid):
    out = _chisq(client, sid, "sex")
    assert out["exact_test"] is None
    assert out["test"] == "Chi-square test of independence"
    assert out["p"] == pytest.approx(out["p_chisquare"])
    assert not any("below 5" in str(w) for w in out["warnings"])


def test_the_chi_square_p_is_kept_rather_than_discarded(client, sid):
    """Replacing the reported p is a judgement about which test applies; hiding
    the other number would stop anyone from checking that judgement."""
    out = _chisq(client, sid, "stage")
    obs = pd.crosstab(_frame()["stage"], _frame()["arm"]).values
    _, p_chi, _, _ = sp.chi2_contingency(obs)
    assert out["p_chisquare"] == pytest.approx(p_chi, rel=1e-9)


def test_the_written_result_does_not_credit_the_chi_square_with_the_exact_p(client, sid):
    """chi2(dof, N) = X, p = Y reads as though the chi-square produced Y."""
    out = _chisq(client, sid, "stage")
    text = out["result_text"]
    assert "Fisher-Freeman-Halton" in text
    assert f"p = {out['p']:.3f}" in text or "p < .001" in text
    chi_str = f"{out['chi2']:.2f}"
    assert text.index(chi_str) > text.index("Fisher-Freeman-Halton")


def test_the_methods_sentence_names_the_test_that_ran(client, sid):
    assert "Fisher-Freeman-Halton" in _chisq(client, sid, "stage")["methods_text"]
    assert "chi-square test of independence" in _chisq(client, sid, "sex")["methods_text"]


def test_the_r_code_checks_the_reported_p_without_promising_the_last_digit(client, sid):
    """Emitting fisher.test(..., simulate.p.value = TRUE, B = 5000) would look
    like a recipe for the identical number, but R's permutation stream is
    unrelated to uSTAT's, so it never is. R's exact algorithm is what the
    resampling estimates, so that is what the emitted code runs — with the
    tolerance stated."""
    code = _chisq(client, sid, "stage")["r_code"]
    assert "fisher.test" in code
    assert "simulate.p.value" not in code
    assert "resampling" in code and "Monte Carlo" in code
    assert "chisq.test" in _chisq(client, sid, "sex")["r_code"]


def test_a_two_by_two_falls_back_to_fisher_not_to_monte_carlo():
    assert r_chisquare("a", "b", "Fisher") == "fisher.test(table(data$a, data$b))"


def test_the_effect_size_interval_reaches_the_client(client, sid):
    es = _chisq(client, sid, "sex")["effect_sizes"][0]
    assert es["name"] == "cramers_v"
    assert es["ci_low"] is not None and es["ci_high"] is not None
    assert es["ci_low"] <= es["value"] <= es["ci_high"]
