"""Six chart shapes taken from cnsplots: line, slope, sankey, stack, ridge, sets.

The recurring failure in each of these is a picture that stays confident while
the data under it thins out — a longitudinal line whose n halves, a paired
test computed on whoever happened to have both measurements, a density curve
over six points, a 100% stacked bar that hides its denominator. Those are what
these tests pin.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def trial() -> str:
    rng = np.random.default_rng(77)
    rows = []
    for arm, base in (("Placebo", 10.0), ("Drug", 10.0)):
        for subj in range(30):
            for visit, drift in (("V1", 0.0), ("V2", 1.0), ("V3", 2.0)):
                # The drug arm improves; both arms lose subjects over time.
                if visit == "V3" and subj % 3 == 0:
                    continue
                eff = drift * (1.8 if arm == "Drug" else 0.2)
                rows.append(
                    {"arm": arm, "visit": visit, "subject": f"{arm}{subj}",
                     "score": base + eff + rng.normal(0, 1.2)}
                )
    return make_session(pd.DataFrame(rows), "cns_trial")


# ── line plot ───────────────────────────────────────────────────────────────

def test_lineplot_returns_one_series_per_group_with_n_at_each_point(client, trial):
    d = client.post(
        "/api/charts/lineplot",
        json={"session_id": trial, "x": "visit", "y": "score", "group": "arm"},
    ).json()
    assert {s["group"] for s in d["series"]} == {"Drug", "Placebo"}
    for s in d["series"]:
        assert [p["x"] for p in s["points"]] == ["V1", "V2", "V3"]
        assert all(p["n"] > 0 for p in s["points"])


def test_lineplot_warns_when_the_group_loses_half_its_subjects(client, trial):
    d = client.post(
        "/api/charts/lineplot",
        json={"session_id": trial, "x": "visit", "y": "score", "group": "arm"},
    ).json()
    # V3 drops a third, not a half — so no warning on this fixture.
    assert d["warnings"] == []

    rows = [{"v": "V1", "s": 1.0}] * 20 + [{"v": "V2", "s": 2.0}] * 4
    sid = make_session(pd.DataFrame(rows), "cns_attrition")
    d2 = client.post(
        "/api/charts/lineplot", json={"session_id": sid, "x": "v", "y": "s"}
    ).json()
    assert d2["warnings"][0]["type"] == "attrition"
    assert "n = 20" in d2["warnings"][0]["message"]


def test_lineplot_ci_band_brackets_the_centre(client, trial):
    d = client.post(
        "/api/charts/lineplot",
        json={"session_id": trial, "x": "visit", "y": "score", "group": "arm"},
    ).json()
    for s in d["series"]:
        for p in s["points"]:
            assert p["lower"] <= p["centre"] <= p["upper"]


def test_lineplot_refuses_a_median_with_a_ci_band(client, trial):
    r = client.post(
        "/api/charts/lineplot",
        json={"session_id": trial, "x": "visit", "y": "score",
              "centre": "median", "spread": "ci"},
    )
    assert r.status_code == 400


# ── slope plot ──────────────────────────────────────────────────────────────

@pytest.fixture()
def prepost() -> str:
    rng = np.random.default_rng(5)
    before = rng.normal(140, 12, 40)
    after = before - rng.normal(8, 5, 40)
    df = pd.DataFrame({"sbp_pre": before, "sbp_post": after,
                       "arm": ["A"] * 20 + ["B"] * 20})
    df.loc[0:4, "sbp_post"] = np.nan          # five incomplete pairs
    return make_session(df, "cns_prepost")


def test_slopeplot_counts_incomplete_pairs_and_excludes_them(client, prepost):
    d = client.post(
        "/api/charts/slopeplot",
        json={"session_id": prepost, "before": "sbp_pre", "after": "sbp_post"},
    ).json()
    assert d["n_incomplete"] == 5
    assert d["n_pairs"] == 35
    assert len(d["pairs"]) == 35
    assert d["warnings"][0]["type"] == "incomplete_pairs"
    assert "35 complete pairs" in d["warnings"][0]["message"]


def test_slopeplot_direction_counts_add_up(client, prepost):
    d = client.post(
        "/api/charts/slopeplot",
        json={"session_id": prepost, "before": "sbp_pre", "after": "sbp_post"},
    ).json()
    assert d["n_increased"] + d["n_decreased"] + d["n_unchanged"] == d["n_pairs"]
    assert d["mean_change"] < 0  # treatment lowered it


def test_slopeplot_runs_a_paired_test_and_names_it(client, prepost):
    d = client.post(
        "/api/charts/slopeplot",
        json={"session_id": prepost, "before": "sbp_pre", "after": "sbp_post"},
    ).json()
    assert d["test_result"]["test"] in {"Paired t-test", "Wilcoxon signed-rank"}
    assert "auto:" in d["test_result"]["selected_by"]
    assert d["test_result"]["p"] < 0.01


def test_slopeplot_paired_t_matches_scipy(client, prepost):
    from scipy import stats as sp

    d = client.post(
        "/api/charts/slopeplot",
        json={"session_id": prepost, "before": "sbp_pre", "after": "sbp_post",
              "test": "paired_t"},
    ).json()
    pairs = d["pairs"]
    a = np.array([p["after"] for p in pairs])
    b = np.array([p["before"] for p in pairs])
    ref = sp.ttest_rel(a, b)
    assert d["test_result"]["p"] == pytest.approx(float(ref.pvalue))
    assert d["test_result"]["df"] == len(pairs) - 1


def test_slopeplot_handles_all_zero_differences_without_crashing(client):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    sid = make_session(df, "cns_nochange")
    d = client.post(
        "/api/charts/slopeplot",
        json={"session_id": sid, "before": "a", "after": "b", "test": "wilcoxon"},
    ).json()
    assert d["test_result"]["p"] is None
    assert "undefined" in d["test_result"]["note"]


def test_slopeplot_refuses_the_same_column_twice(client, prepost):
    r = client.post(
        "/api/charts/slopeplot",
        json={"session_id": prepost, "before": "sbp_pre", "after": "sbp_pre"},
    )
    assert r.status_code == 400


# ── sankey ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def pathway() -> str:
    rng = np.random.default_rng(13)
    n = 150
    first = rng.choice(["Medical", "PCI"], n, p=[0.6, 0.4])
    second = np.where(
        first == "Medical",
        rng.choice(["Medical", "PCI", "CABG"], n, p=[0.7, 0.2, 0.1]),
        rng.choice(["PCI", "CABG"], n, p=[0.8, 0.2]),
    )
    return make_session(
        pd.DataFrame({"line1": first, "line2": second}), "cns_pathway"
    )


def test_sankey_link_totals_match_the_row_count(client, pathway):
    d = client.post(
        "/api/charts/sankey",
        json={"session_id": pathway, "stages": ["line1", "line2"]},
    ).json()
    assert sum(l["value"] for l in d["links"]) == d["n_rows"]


def test_sankey_keeps_a_repeated_level_as_two_nodes(client, pathway):
    """'Medical' at both stages must not become a self-loop."""
    d = client.post(
        "/api/charts/sankey",
        json={"session_id": pathway, "stages": ["line1", "line2"]},
    ).json()
    medical_nodes = [i for i, l in enumerate(d["labels"]) if l == "Medical"]
    assert len(medical_nodes) == 2
    assert d["node_stage"][medical_nodes[0]] != d["node_stage"][medical_nodes[1]]
    for l in d["links"]:
        assert l["source"] != l["target"]


def test_sankey_min_flow_drops_thin_links_and_counts_them(client, pathway):
    d = client.post(
        "/api/charts/sankey",
        json={"session_id": pathway, "stages": ["line1", "line2"], "min_flow": 10},
    ).json()
    assert all(l["value"] > 10 for l in d["links"])
    assert d["n_links_dropped"] > 0


def test_sankey_needs_two_stages(client, pathway):
    r = client.post(
        "/api/charts/sankey", json={"session_id": pathway, "stages": ["line1"]}
    )
    assert r.status_code == 400


def test_sankey_refuses_a_repeated_stage_column(client, pathway):
    r = client.post(
        "/api/charts/sankey",
        json={"session_id": pathway, "stages": ["line1", "line1"]},
    )
    assert r.status_code == 400


# ── stacked bar ─────────────────────────────────────────────────────────────

def test_stackplot_percentages_sum_to_100_per_bar(client, pathway):
    d = client.post(
        "/api/charts/stackplot",
        json={"session_id": pathway, "x": "line1", "fill": "line2"},
    ).json()
    for i, _ in enumerate(d["x_levels"]):
        assert sum(s["percent"][i] for s in d["series"]) == pytest.approx(100.0)


def test_stackplot_reports_the_denominator_behind_each_bar(client, pathway):
    """A 100% stacked bar hides n; it has to come back separately."""
    d = client.post(
        "/api/charts/stackplot",
        json={"session_id": pathway, "x": "line1", "fill": "line2", "normalize": True},
    ).json()
    assert set(d["totals"]) == set(d["x_levels"])
    assert sum(d["totals"].values()) == 150


def test_stackplot_refuses_negative_values(client):
    df = pd.DataFrame({"g": ["a", "b"], "f": ["x", "y"], "v": [3.0, -1.0]})
    sid = make_session(df, "cns_stack_neg")
    r = client.post(
        "/api/charts/stackplot",
        json={"session_id": sid, "x": "g", "fill": "f", "value": "v"},
    )
    assert r.status_code == 400


def test_stackplot_refuses_the_same_column_on_both_roles(client, pathway):
    r = client.post(
        "/api/charts/stackplot",
        json={"session_id": pathway, "x": "line1", "fill": "line1"},
    )
    assert r.status_code == 400


# ── ridge plot ──────────────────────────────────────────────────────────────

def test_ridgeplot_evaluates_every_group_on_one_shared_grid(client, trial):
    d = client.post(
        "/api/charts/ridgeplot",
        json={"session_id": trial, "x": "score", "group": "visit"},
    ).json()
    grids = [r["x"] for r in d["ridges"]]
    assert all(g == grids[0] for g in grids), "per-group grids would distort widths"


def test_ridgeplot_flags_groups_too_thin_to_smooth(client):
    df = pd.DataFrame(
        {"v": list(np.random.default_rng(1).normal(0, 1, 40)) + [1.0, 2.0, 3.0, 4.0],
         "g": ["big"] * 40 + ["tiny"] * 4}
    )
    sid = make_session(df, "cns_ridge_thin")
    d = client.post(
        "/api/charts/ridgeplot", json={"session_id": sid, "x": "v", "group": "g"}
    ).json()
    msgs = " ".join(w["message"] for w in d["warnings"])
    assert "tiny" in msgs


def test_ridgeplot_skips_a_constant_group_with_a_reason(client):
    df = pd.DataFrame(
        {"v": list(np.random.default_rng(2).normal(0, 1, 30)) + [5.0] * 5,
         "g": ["real"] * 30 + ["flat"] * 5}
    )
    sid = make_session(df, "cns_ridge_flat")
    d = client.post(
        "/api/charts/ridgeplot", json={"session_id": sid, "x": "v", "group": "g"}
    ).json()
    assert {r["group"] for r in d["ridges"]} == {"real"}
    assert any(w["type"] == "no_density" and w["group"] == "flat" for w in d["warnings"])


def test_ridgeplot_refuses_a_constant_variable(client):
    df = pd.DataFrame({"v": [3.0] * 10, "g": ["a"] * 5 + ["b"] * 5})
    sid = make_session(df, "cns_ridge_const")
    r = client.post(
        "/api/charts/ridgeplot", json={"session_id": sid, "x": "v", "group": "g"}
    )
    assert r.status_code == 400


# ── sets (Venn / UpSet) ─────────────────────────────────────────────────────

@pytest.fixture()
def criteria() -> str:
    rng = np.random.default_rng(23)
    n = 200
    df = pd.DataFrame(
        {
            "diabetes": rng.integers(0, 2, n),
            "hypertension": rng.integers(0, 2, n),
            "smoker": rng.integers(0, 2, n),
        }
    )
    return make_session(df, "cns_sets")


def test_set_intersections_partition_the_rows(client, criteria):
    d = client.post(
        "/api/charts/sets",
        json={"session_id": criteria,
              "columns": ["diabetes", "hypertension", "smoker"]},
    ).json()
    total = sum(r["count"] for r in d["intersections"])
    assert total + d["n_in_no_set"] == d["n_rows"]


def test_exclusive_regions_reconstruct_each_set_size(client, criteria):
    d = client.post(
        "/api/charts/sets",
        json={"session_id": criteria,
              "columns": ["diabetes", "hypertension", "smoker"]},
    ).json()
    for col, size in d["set_sizes"].items():
        from_regions = sum(r["count"] for r in d["intersections"] if col in r["sets"])
        assert from_regions == size


def test_three_sets_are_venn_renderable_four_are_not(client):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({c: rng.integers(0, 2, 60) for c in "abcd"})
    sid = make_session(df, "cns_sets4")
    three = client.post(
        "/api/charts/sets", json={"session_id": sid, "columns": ["a", "b", "c"]}
    ).json()
    four = client.post(
        "/api/charts/sets", json={"session_id": sid, "columns": ["a", "b", "c", "d"]}
    ).json()
    assert three["renderable_as_venn"] is True
    assert four["renderable_as_venn"] is False


def test_yes_no_text_columns_are_read_as_membership(client):
    df = pd.DataFrame(
        {"a": ["yes", "no", "yes", "no"], "b": ["evet", "evet", "hayır", "hayır"]}
    )
    sid = make_session(df, "cns_sets_text")
    d = client.post(
        "/api/charts/sets", json={"session_id": sid, "columns": ["a", "b"]}
    ).json()
    assert d["set_sizes"] == {"a": 2, "b": 2}


def test_too_many_sets_is_refused_with_the_region_count(client):
    rng = np.random.default_rng(4)
    cols = list("abcdefg")
    df = pd.DataFrame({c: rng.integers(0, 2, 30) for c in cols})
    sid = make_session(df, "cns_sets7")
    r = client.post("/api/charts/sets", json={"session_id": sid, "columns": cols})
    assert r.status_code == 400
    assert "127 regions" in r.json()["detail"]


def test_sets_with_no_members_anywhere_is_a_400(client):
    df = pd.DataFrame({"a": [0, 0, 0], "b": [0, 0, 0]})
    sid = make_session(df, "cns_sets_empty")
    r = client.post("/api/charts/sets", json={"session_id": sid, "columns": ["a", "b"]})
    assert r.status_code == 400
