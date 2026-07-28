"""Findings from the 200x20 audit that the 10x20 pass could not reach.

The two audits overlap heavily, but three things only show up at n = 200:

  * a paired contrast whose differences are constant produces t = inf at
    n = 10 and a FINITE t of 8.6e12 at n = 200, because subtracting large
    values loses bits. The infinite one was caught; the finite one was
    reported as significant.
  * the Hedges g confidence interval is wide enough to span zero beside a
    p of 1e-14, because its standard error was missing a divisor.
  * one-way ANOVA kept the equal-variance F under heteroscedasticity and
    only switched the post-hoc, so a robust Games-Howell hung off a
    non-robust omnibus.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp
from statsmodels.stats.oneway import anova_oneway

from conftest import make_session
from services.stat_utils import cohen_d, paired_contrast_is_degenerate


# ── a constant difference on large values ─────────────────────────────────────


def test_ulp_noise_counts_as_no_variance():
    """Every subject changed by the same amount, but on values of order 1e6
    the subtraction leaves a spread of a few ulps rather than exactly zero."""
    rng = np.random.default_rng(0)
    base = rng.normal(1e6, 1e5, 200)
    shifted = base + 5.0
    shifted[0] = np.nextafter(shifted[0], np.inf)

    t, _ = sp.ttest_rel(shifted, base)
    assert np.isfinite(t) and abs(t) > 1e10, "the premise: a finite, absurd t"
    assert paired_contrast_is_degenerate(shifted, base)


def test_real_data_is_not_called_degenerate():
    rng = np.random.default_rng(1)
    assert not paired_contrast_is_degenerate(
        rng.normal(10, 2, 50), rng.normal(9, 2, 50)
    )
    # A genuine but tiny effect must survive.
    x = np.arange(50.0)
    assert not paired_contrast_is_degenerate(x + rng.normal(0, 1e-3, 50), x)


def test_paired_ttest_reports_a_constant_change_instead_of_inventing_a_t(client):
    """It used to cap an infinite t at +/-9999 and set p to 0 — a statistic
    the data never produced, presented as overwhelming evidence."""
    base = np.arange(1.0, 41.0) * 1e5
    df = pd.DataFrame({"before": base, "after": base + 7.0})
    sid = make_session(df, "a200_paired_const")
    r = client.post(
        "/api/repeated/paired_ttest",
        json={"session_id": sid, "col1": "after", "col2": "before"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["t"] is None and out["p"] is None
    assert out["significant"] is False
    assert any("same amount" in str(w) for w in out["warnings"])
    # The contract still holds — the change itself is reported.
    assert out["summary"]["differences"]["mean"] == pytest.approx(7.0)
    assert isinstance(out["result_text"], str) and len(out["result_text"]) > 10


def test_identical_columns_still_report_no_difference(client):
    """Zero change for everyone is a real answer, not a degenerate one."""
    vals = np.arange(10.0, 60.0)
    df = pd.DataFrame({"a": vals, "b": vals})
    sid = make_session(df, "a200_paired_same")
    r = client.post(
        "/api/repeated/paired_ttest",
        json={"session_id": sid, "col1": "a", "col2": "b"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["significant"] is False
    assert out["p"] == pytest.approx(1.0)


def test_rm_posthoc_does_not_report_a_noise_statistic(client):
    """At n = 200 the constant contrast produced a finite t and was shown as
    significant; isfinite alone never caught it."""
    rows = []
    for s in range(200):
        base = float(s) * 1e5
        for k, bump in enumerate((0.0, 5.0, 5.0 + (s % 3))):
            rows.append({"id": f"s{s}", "time": f"t{k + 1}", "y": base + bump})
    sid = make_session(pd.DataFrame(rows), "a200_rm_noise")
    r = client.post(
        "/api/repeated/rm_anova",
        json={"session_id": sid, "subject_col": "id", "within_col": "time",
              "value_col": "y"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    frozen = [ph for ph in out["posthoc"] if ph["group1"] == "t1" and ph["group2"] == "t2"]
    assert frozen and frozen[0]["statistic"] is None
    assert frozen[0]["significant"] is None


def test_rm_posthoc_keeps_a_tiny_p(client):
    """round(p, 6) printed a p of 1e-40 as 0.0."""
    rng = np.random.default_rng(3)
    rows = []
    for s in range(60):
        base = rng.normal(0, 1)
        for k, bump in enumerate((0.0, 8.0, 16.0)):
            rows.append({"id": f"s{s}", "time": f"t{k + 1}",
                         "y": base + bump + rng.normal(0, 0.2)})
    sid = make_session(pd.DataFrame(rows), "a200_rm_tinyp")
    out = client.post(
        "/api/repeated/rm_anova",
        json={"session_id": sid, "subject_col": "id", "within_col": "time",
              "value_col": "y"},
    ).json()
    live = [ph for ph in out["posthoc"] if ph["p"] is not None]
    assert live and min(ph["p"] for ph in live) > 0.0


# ── the Hedges g interval ─────────────────────────────────────────────────────


def test_hedges_g_interval_does_not_span_zero_at_overwhelming_significance():
    """The standard error read sqrt((n1+n2)/(n1 n2) + g^2/2) — the second term
    missing its (n1 + n2) divisor, so the interval came out 5.5x too wide."""
    rng = np.random.default_rng(0)
    g1, g2 = rng.normal(1.0, 1.0, 100), rng.normal(0.0, 1.0, 100)
    out = cohen_d(g1, g2)
    p = float(sp.ttest_ind(g1, g2).pvalue)

    assert p < 1e-10
    assert out["ci_low"] > 0, (out["ci_low"], out["ci_high"], p)
    # Against the textbook Hedges-Olkin standard error.
    se = np.sqrt(200 / 10000 + out["value"] ** 2 / 400)
    crit = float(sp.t.ppf(0.975, 198))
    assert out["ci_low"] == pytest.approx(out["value"] - crit * se, abs=1e-3)


def test_hedges_g_interval_still_spans_zero_when_there_is_no_effect():
    rng = np.random.default_rng(7)
    out = cohen_d(rng.normal(0, 1, 80), rng.normal(0, 1, 80))
    assert out["ci_low"] < 0 < out["ci_high"]


# ── Welch omnibus ─────────────────────────────────────────────────────────────


def test_anova_omnibus_follows_levene(client):
    rng = np.random.default_rng(11)
    groups = [rng.normal(10, 1, 60), rng.normal(11, 6, 60), rng.normal(12, 14, 60)]
    df = pd.DataFrame({
        "y": np.concatenate(groups), "g": ["A"] * 60 + ["B"] * 60 + ["C"] * 60,
    })
    sid = make_session(df, "a200_anova_welch")
    out = client.post(
        "/api/stats/anova", json={"session_id": sid, "column": "y", "group_column": "g"}
    ).json()

    assert out["variance_assumption"] == "welch"
    expected = anova_oneway(groups, use_var="unequal")
    assert out["F"] == pytest.approx(float(expected.statistic), rel=1e-9)
    assert out["p"] == pytest.approx(float(expected.pvalue), rel=1e-9)
    assert out["df_denominator"] < out["df_within"]
    assert "Welch" in out["interpretation"]


def test_anova_equal_variance_interpretation_path_is_serializable(client):
    rng = np.random.default_rng(12)
    groups = [
        rng.normal(10, 2, 60),
        rng.normal(11, 2, 60),
        rng.normal(12, 2, 60),
    ]
    df = pd.DataFrame({
        "y": np.concatenate(groups),
        "g": ["A"] * 60 + ["B"] * 60 + ["C"] * 60,
    })
    sid = make_session(df, "a200_anova_equal")
    response = client.post(
        "/api/stats/anova",
        json={"session_id": sid, "column": "y", "group_column": "g"},
    )

    assert response.status_code == 200, response.text
    out = response.json()
    assert out["variance_assumption"] == "equal"
    assert out["df_denominator"] == pytest.approx(177.0)
    assert "F(2,177)" in out["interpretation"]


# ── things the report asked to be stated rather than assumed ──────────────────


def test_table1_returns_the_raw_p_as_well_as_the_formatted_one(client):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "arm": ["A"] * 60 + ["B"] * 60,
        "y": np.r_[rng.normal(10, 1, 60), rng.normal(13, 1, 60)],
        "cat": list(rng.choice(["x", "y"], 120)),
    })
    sid = make_session(df, "a200_rawp")
    out = client.post("/api/stats/table1", json={
        "session_id": sid, "variables": ["y", "cat"], "group_column": "arm",
    }).json()
    rows = {r["variable"]: r for r in out["rows"]}

    # The formatted p saturates at "<0.001"; the raw one does not.
    assert rows["y"]["p_value"] == "<0.001"
    assert 0 < rows["y"]["p_raw"] < 1e-3
    assert rows["cat"]["p_raw"] == pytest.approx(float(rows["cat"]["p_value"]), abs=1e-3)


def test_wilcoxon_states_its_zero_and_tie_handling(client):
    df = pd.DataFrame({
        "pre":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "post": [1.0, 4.0, 5.0, 4.0, 8.0, 9.0, 9.0, 11.0],  # two zeros, ties
    })
    sid = make_session(df, "a200_wsr_ties")
    out = client.post("/api/repeated/wilcoxon_signed_rank",
                      json={"session_id": sid, "col1": "post", "col2": "pre"}).json()
    assert out["n_zero_differences"] == 2
    assert out["n_tied_ranks"] >= 1
    assert out["p_method"] in ("exact", "normal approximation")
    assert any("dropped" in a["detail"] for a in out["assumptions"])


def test_monte_carlo_test_names_its_resample_count(client):
    """A Monte Carlo p carries sampling error; 1/(N+1) granularity has to be
    visible to whoever reads the number."""
    df = pd.DataFrame({
        "arm": ["A"] * 12 + ["B"] * 12,
        "grade": (["I"] * 5 + ["II"] * 5 + ["III"] * 2) * 2,
    })
    sid = make_session(df, "a200_ffh_name")
    out = client.post("/api/stats/table1", json={
        "session_id": sid, "variables": ["grade"], "group_column": "arm",
    }).json()
    name = out["rows"][0]["test"]
    assert "Fisher-Freeman-Halton" in name
    assert "resamples" in name


def test_non_inferiority_keeps_a_decisive_p(client):
    """round(p, 6) turned a p of 1.5e-14 into 0.0 — an impossible certainty
    where the data only said "very small"."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "arm": ["ref"] * 100 + ["test"] * 100,
        "y": np.r_[rng.normal(100, 8, 100), rng.normal(112, 8, 100)],
    })
    sid = make_session(df, "a200_ni_tinyp")
    out = client.post("/api/stats/noninferiority", json={
        "session_id": sid, "outcome_col": "y", "group_col": "arm",
        "test_group": "test", "ref_group": "ref", "outcome_type": "continuous",
        "margin": 20.0, "bound": "upper", "alpha": 0.05,
    }).json()
    assert out["non_inferior"] is True
    assert 0 < out["p_noninferiority"] < 1e-6
