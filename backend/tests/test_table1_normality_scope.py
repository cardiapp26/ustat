"""Where Table 1 tests normality when a grouping column is present.

Reported from the outside: uSTAT decided normality on the POOLED sample while
the same data checked by hand in R and in Python used Shapiro within each
group, and `per_group_normality` came back empty. The pooled sample of two
groups that differ is a mixture — it can fail normality because the groups
differ rather than because either one is skewed, and it can pass while both
are skewed in opposite directions. What the t-test and ANOVA assume is
normality WITHIN each group, so that is the default now.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from services import store


@pytest.fixture()
def bimodal(client) -> str:
    """Two tight, well-separated normal groups.

    Each group passes Shapiro on its own; pooled they are a two-humped
    mixture, which is the case that used to flip the whole table to
    Mann-Whitney.
    """
    rng = np.random.default_rng(11)
    a = rng.normal(10, 1, 25)
    b = rng.normal(30, 1, 25)
    store.save("t1n_bimodal", pd.DataFrame({
        "value": np.concatenate([a, b]),
        "grp": ["A"] * 25 + ["B"] * 25,
    }))
    return "t1n_bimodal"


def _table1(client, sid, **extra):
    r = client.post("/api/stats/table1", json={
        "session_id": sid, "variables": ["value"], "group_column": "grp",
        "variable_kinds": {"value": "numeric"}, **extra})
    assert r.status_code == 200, r.text
    return r.json()["rows"][0]


def test_the_pooled_sample_really_is_non_normal_here(bimodal):
    """Guards the fixture: without the split there is a genuine failure."""
    pooled = store.get(bimodal)["value"]
    assert scipy_stats.shapiro(pooled).pvalue < 0.05
    for g in ("A", "B"):
        arm = store.get(bimodal).query("grp == @g")["value"]
        assert scipy_stats.shapiro(arm).pvalue >= 0.05


def test_grouped_table_tests_each_group_by_default(client, bimodal):
    row = _table1(client, bimodal)
    assert row["normality_mode"] == "within_group"
    assert set(row["per_group_normality"]) == {"A", "B"}
    assert all(v["normal"] for v in row["per_group_normality"].values())
    assert row["normal"] is True


def test_the_default_picks_the_parametric_test(client, bimodal):
    """Both groups are normal, so this is a t-test — not Mann-Whitney."""
    assert _table1(client, bimodal)["test"].startswith("t-test")


def test_per_group_normality_carries_p_test_and_n(client, bimodal):
    entry = _table1(client, bimodal)["per_group_normality"]["A"]
    assert entry["n"] == 25
    assert entry["test"] == "Shapiro-Wilk"
    assert 0.0 <= entry["p"] <= 1.0


def test_overall_can_still_be_asked_for(client, bimodal):
    row = _table1(client, bimodal, normality_mode="overall")
    assert row["normality_mode"] == "overall"
    assert row["per_group_normality"] == {}
    assert row["normal"] is False          # the pooled mixture fails
    assert row["test"] == "Mann-Whitney"


def test_the_reported_mode_is_the_one_that_was_used(client, bimodal):
    """A caller that sends nothing has to be able to read back what happened."""
    assert _table1(client, bimodal)["normality_mode"] == "within_group"


def test_without_a_group_column_nothing_changes(client, bimodal):
    r = client.post("/api/stats/table1", json={
        "session_id": bimodal, "variables": ["value"],
        "variable_kinds": {"value": "numeric"}})
    assert r.status_code == 200, r.text
    row = r.json()["rows"][0]
    assert row["normality_mode"] == "overall"
    assert row["per_group_normality"] == {}


def test_one_skewed_group_is_enough_to_go_non_parametric(client):
    rng = np.random.default_rng(7)
    store.save("t1n_skew", pd.DataFrame({
        "value": np.concatenate([rng.normal(10, 1, 25), rng.exponential(2, 25)]),
        "grp": ["A"] * 25 + ["B"] * 25,
    }))
    row = _table1(client, "t1n_skew")
    assert row["per_group_normality"]["A"]["normal"] is True
    assert row["per_group_normality"]["B"]["normal"] is False
    assert row["normal"] is False
    assert row["test"] == "Mann-Whitney"


def test_the_docx_export_agrees_with_the_table(client, bimodal):
    """The export claims to match the on-screen table; that has to include
    where normality was tested, not only how."""
    from routers.pub_export import _run_table1_analysis, TableDocxRequest

    live = _table1(client, bimodal)
    exported = _run_table1_analysis(TableDocxRequest(
        session_id=bimodal, variables=["value"], group_column="grp",
        variable_kinds={"value": "numeric"}))
    row = exported["rows"][0]
    assert row["test"] == live["test"]
    # Parametric on both sides means mean ± SD on both sides.
    assert "±" in row["overall"] and "±" in live["overall"]
