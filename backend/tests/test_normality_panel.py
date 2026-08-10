"""The Normality panel: cohort-wide and within-group assessment.

The panel exists because "is it normal?" is not answered by one p-value. The
tests below pin the three things that make it trustworthy: the numbers agree
with what R prints, the verdict reads the test and the shape statistics
together, and a sample too small or too large to judge on p says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from routers.stats.normality import (
    _anderson_darling,
    _lilliefors,
    _shape,
    _shape_flag,
    _verdict,
)
from services import store


@pytest.fixture()
def cohort(client) -> str:
    """Two normal groups with different means, plus a skewed variable.

    Missing values are deliberately scattered: every count in the panel has to
    be the non-missing count, not the row count.
    """
    rng = np.random.default_rng(7)
    n = 60
    grp = np.array(["A"] * 30 + ["B"] * 30)
    normal = np.concatenate([rng.normal(10, 2, 30), rng.normal(16, 2, 30)])
    skewed = rng.lognormal(0, 1, n)
    normal[[1, 45]] = np.nan
    skewed[3] = np.nan
    store.save("norm_cohort", pd.DataFrame(
        {"grp": grp, "normal_var": normal, "skewed_var": skewed}
    ))
    return "norm_cohort"


def _run(client, sid, **extra):
    body = {"session_id": sid, "variables": ["normal_var"], **extra}
    r = client.post("/api/stats/normality", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ── numbers ────────────────────────────────────────────────────────────────────


def test_shapiro_matches_scipy_on_the_non_missing_values(client, cohort):
    df = store.get("norm_cohort")
    expected = scipy_stats.shapiro(df["normal_var"].dropna())
    block = _run(client, cohort)["variables"][0]["overall"]
    shapiro = next(t for t in block["tests"] if t["id"] == "shapiro")
    assert shapiro["stat"] == pytest.approx(float(expected.statistic))
    assert shapiro["p"] == pytest.approx(float(expected.pvalue))
    assert block["n"] == 58 and block["n_missing"] == 2


def test_anderson_darling_reproduces_the_nortest_formula():
    # Worked against R's nortest::ad.test, whose p-value comes from the
    # D'Agostino-Stephens fits — scipy returns a critical-value table and no p,
    # and a table cannot be pasted into a paper.
    # R: nortest::ad.test(as.numeric(1:20)) → A = 0.22074, p-value = 0.8064
    a2, p = _anderson_darling(np.arange(1.0, 21.0))
    assert a2 == pytest.approx(0.22074, abs=1e-4)
    assert p == pytest.approx(0.8064, abs=1e-3)


def test_anderson_darling_declines_under_eight_observations():
    assert _anderson_darling(np.arange(1.0, 8.0)) == (None, None)


def test_lilliefors_reproduces_the_nortest_p_value():
    # statsmodels has this test, but its simulated-table p differs from R's by
    # up to ~0.02 at these sample sizes — enough to change the third decimal a
    # reviewer re-running the analysis would see. The statistic is the same
    # either way; the approximation used here is R's.
    # R: nortest::lillie.test(as.numeric(1:20)) → D = 0.076564, p-value = 0.991
    d, p = _lilliefors(np.arange(1.0, 21.0))
    assert d == pytest.approx(0.076564, abs=1e-5)
    assert p == pytest.approx(0.991, abs=1e-3)


def test_lilliefors_declines_under_five_observations():
    assert _lilliefors(np.arange(1.0, 5.0)) == (None, None)


def test_skewness_is_the_spss_estimator_not_the_biased_one():
    # G1, as SPSS and e1071::skewness(type = 2) report it. The biased g1 is a
    # different number at small n, and the standard error below is derived for
    # G1 — mixing them would make the z-score wrong exactly where it is used.
    x = np.array([1.0, 2, 2, 3, 3, 3, 4, 9])
    shape = _shape(x)
    assert shape["skewness"] == pytest.approx(float(scipy_stats.skew(x, bias=False)))
    n = len(x)
    assert shape["skew_se"] == pytest.approx(
        np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    )
    assert shape["skew_z"] == pytest.approx(shape["skewness"] / shape["skew_se"])


def test_shape_thresholds_follow_the_sample_size(client):
    # Kim (2013): |z| > 1.96 under n = 50, > 3.29 to n = 300, and beyond that
    # the absolute values, because at that size every non-zero skew is
    # "significant" while a skew of 0.3 is invisible in a histogram.
    borderline = {"skewness": 0.5, "kurtosis": 0.0, "skew_z": 2.5, "kurt_z": 0.4}
    assert _shape_flag(borderline, 40) is True     # 2.5 > 1.96
    assert _shape_flag(borderline, 150) is False   # 2.5 < 3.29
    huge = {"skewness": 0.5, "kurtosis": 0.0, "skew_z": 40.0, "kurt_z": 0.4}
    assert _shape_flag(huge, 5000) is False        # |skew| 0.5 < 2, z ignored


# ── verdict ────────────────────────────────────────────────────────────────────


def test_test_and_shape_disagreeing_is_called_borderline():
    v = _verdict(n=100, p=0.01, flag=False, alpha=0.05)
    assert v["code"] == "borderline"
    assert "within range" in v["reason"]


def test_a_small_sample_carries_the_low_power_caveat():
    v = _verdict(n=12, p=0.6, flag=False, alpha=0.05)
    assert v["code"] == "normal"
    assert any("little power" in note for note in v["notes"])


def test_a_large_sample_says_p_is_oversensitive():
    v = _verdict(n=800, p=0.001, flag=False, alpha=0.05)
    assert any("too small to matter" in note for note in v["notes"])


def test_thirty_observations_earns_the_clt_note():
    assert any("central limit" in n for n in _verdict(30, 0.4, False, 0.05)["notes"])
    assert not any("central limit" in n for n in _verdict(29, 0.4, False, 0.05)["notes"])


# ── endpoint shape ─────────────────────────────────────────────────────────────


def test_grouping_reports_each_group_and_still_reports_the_pooled_sample(client, cohort):
    # Two separated groups pooled are a mixture; the pooled test can reject
    # because the groups differ rather than because either is skewed. Showing
    # both is what makes that visible instead of hiding it behind a default.
    out = _run(client, cohort, group_column="grp")
    entry = out["variables"][0]
    assert out["group_levels"] == ["A", "B"]
    assert [g["label"] for g in entry["groups"]] == ["A", "B"]
    assert entry["overall"]["n"] == 58
    assert sum(g["n"] for g in entry["groups"]) == 58
    assert all(g["verdict"]["code"] == "normal" for g in entry["groups"])
    assert entry["group_summary"] == "All groups consistent with normal"


def test_a_skewed_variable_is_called_out(client, cohort):
    out = _run(client, cohort, variables=["skewed_var"])
    block = out["variables"][0]["overall"]
    assert block["verdict"]["code"] == "non_normal"
    assert block["shape"]["skewness"] > 1


def test_qq_points_use_the_r_convention(client, cohort):
    block = _run(client, cohort)["variables"][0]["overall"]
    qq = block["qq"]
    n = block["n"]
    expected = scipy_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    assert qq["theoretical"] == pytest.approx(list(expected))
    assert qq["sample"] == sorted(qq["sample"])
    # qqline goes through the quartiles, not through mean ± SD: a line fitted
    # to the ends would follow the outliers it is meant to expose.
    assert qq["line"]["slope"] > 0


def test_a_constant_column_does_not_500(client):
    store.save("norm_const", pd.DataFrame({"flat": [5.0] * 20}))
    out = _run(client, "norm_const", variables=["flat"])
    block = out["variables"][0]["overall"]
    assert block["constant"] is True
    assert block["verdict"]["code"] in ("undetermined", "normal", "non_normal")
    assert block["histogram"] == {}


def test_a_group_too_small_to_test_reports_the_reason_not_a_p(client):
    store.save("norm_thin", pd.DataFrame({
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "g": ["A", "A", "A", "A", "A", "B", "B"],
    }))
    out = _run(client, "norm_thin", variables=["v"], group_column="g")
    thin = out["variables"][0]["groups"][1]
    assert thin["n"] == 2
    assert thin["verdict"]["code"] == "undetermined"
    assert all(t["p"] is None for t in thin["tests"])


def test_an_identifier_is_refused_as_a_grouping_column(client, cohort):
    df = store.get("norm_cohort").copy()
    df["id"] = [f"P{i}" for i in range(len(df))]
    store.save("norm_ids", df)
    out = _run(client, "norm_ids", group_column="id")
    assert out["group_column"] is None
    assert out["variables"][0]["groups"] == []
    assert any("identifier" in w for w in out["warnings"])


def test_an_unknown_variable_is_rejected_rather_than_silently_dropped(client, cohort):
    r = client.post("/api/stats/normality", json={
        "session_id": cohort, "variables": ["not_a_column"]})
    assert r.status_code == 400


def test_the_sentence_is_paste_ready(client, cohort):
    block = _run(client, cohort)["variables"][0]["overall"]
    assert "Shapiro-Wilk W =" in block["sentence"]
    assert "p=" in block["sentence"] or "p<" in block["sentence"]
    assert f"n = {block['n']}" in block["sentence"]
