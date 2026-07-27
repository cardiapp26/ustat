"""The remaining ggpubr shapes: pie, balloon, summary table, ellipse,
marginal histograms, and facets.

Each of these can be drawn in a way that is technically a picture of the data
and still misleads — a pie of negative numbers, a balloon plot that only
restates the marginals, facets whose axes differ per panel. Those are the
cases pinned here.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def clinic() -> str:
    rng = np.random.default_rng(31)
    n = 180
    df = pd.DataFrame(
        {
            "site": rng.choice(["Ankara", "Izmir", "Bursa"], n, p=[0.5, 0.3, 0.2]),
            "outcome": rng.choice(["Alive", "Dead"], n, p=[0.75, 0.25]),
            "sex": rng.choice(["M", "F"], n),
            "age": rng.normal(62, 11, n),
            "ldl": rng.normal(120, 28, n),
            "cost": rng.gamma(4, 300, n),
        }
    )
    return make_session(df, "ggpubr_rest")


# ── pie / donut ─────────────────────────────────────────────────────────────

def test_pie_counts_rows_and_percentages_sum_to_100(client, clinic):
    d = client.post(
        "/api/charts/pie", json={"session_id": clinic, "category": "site"}
    ).json()
    assert d["measure"] == "count"
    assert sum(s["value"] for s in d["slices"]) == pytest.approx(d["total"])
    assert sum(s["percent"] for s in d["slices"]) == pytest.approx(100.0)


def test_pie_can_sum_a_value_column_instead_of_counting(client, clinic):
    d = client.post(
        "/api/charts/pie",
        json={"session_id": clinic, "category": "site", "value": "cost"},
    ).json()
    assert d["measure"] == "sum"
    assert d["total"] > 1000  # gamma costs, not row counts


def test_pie_refuses_negative_quantities(client):
    df = pd.DataFrame({"g": ["a", "b"], "v": [5.0, -3.0]})
    sid = make_session(df, "pie_neg")
    r = client.post(
        "/api/charts/pie", json={"session_id": sid, "category": "g", "value": "v"}
    )
    assert r.status_code == 400
    assert "negative" in r.json()["detail"]


def test_pie_folds_a_long_tail_into_other(client):
    df = pd.DataFrame({"g": [f"lvl{i}" for i in range(20)]})
    sid = make_session(df, "pie_tail")
    d = client.post(
        "/api/charts/pie",
        json={"session_id": sid, "category": "g", "max_slices": 5},
    ).json()
    assert len(d["slices"]) == 5
    assert d["slices"][-1]["label"] == "Other"
    assert d["n_folded_into_other"] == 16
    assert sum(s["percent"] for s in d["slices"]) == pytest.approx(100.0)


# ── balloon plot ────────────────────────────────────────────────────────────

def test_balloon_returns_a_full_grid_with_residuals(client, clinic):
    d = client.post(
        "/api/charts/balloon",
        json={"session_id": clinic, "row": "site", "col": "outcome"},
    ).json()
    assert len(d["cells"]) == len(d["rows"]) * len(d["cols"])
    assert d["df"] == (len(d["rows"]) - 1) * (len(d["cols"]) - 1)
    assert sum(c["count"] for c in d["cells"]) == d["n"]
    # Expected counts must reproduce the observed grand total.
    assert sum(c["expected"] for c in d["cells"]) == pytest.approx(d["n"])


def test_balloon_residual_matches_the_standard_formula(client, clinic):
    d = client.post(
        "/api/charts/balloon",
        json={"session_id": clinic, "row": "site", "col": "outcome"},
    ).json()
    for c in d["cells"]:
        expected_resid = (c["count"] - c["expected"]) / np.sqrt(c["expected"])
        assert c["residual"] == pytest.approx(expected_resid)


def test_balloon_warns_when_an_expected_count_is_below_five(client):
    df = pd.DataFrame(
        {"r": ["a"] * 9 + ["b"], "c": ["x"] * 5 + ["y"] * 4 + ["y"]}
    )
    sid = make_session(df, "balloon_small")
    d = client.post(
        "/api/charts/balloon", json={"session_id": sid, "row": "r", "col": "c"}
    ).json()
    assert d["warnings"][0]["type"] == "low_expected_count"
    assert d["warnings"][0]["min_expected"] < 5


def test_balloon_needs_two_levels_on_each_axis(client):
    df = pd.DataFrame({"r": ["a", "a", "a"], "c": ["x", "y", "x"]})
    sid = make_session(df, "balloon_thin")
    r = client.post(
        "/api/charts/balloon", json={"session_id": sid, "row": "r", "col": "c"}
    )
    assert r.status_code == 400
    assert "at least two levels" in r.json()["detail"]


def test_balloon_refuses_the_same_column_twice(client, clinic):
    r = client.post(
        "/api/charts/balloon", json={"session_id": clinic, "row": "site", "col": "site"}
    )
    assert r.status_code == 400


# ── summary stats table ─────────────────────────────────────────────────────

def test_summary_stats_reports_n_and_missing_per_group(client):
    df = pd.DataFrame(
        {"g": ["a", "a", "a", "b", "b"], "v": [1.0, 2.0, np.nan, 10.0, 12.0]}
    )
    sid = make_session(df, "summary_missing")
    d = client.post(
        "/api/charts/summary_stats", json={"session_id": sid, "y": "v", "group": "g"}
    ).json()
    a = next(r for r in d["rows"] if r["group"] == "a")
    assert a["n"] == 2 and a["n_missing"] == 1
    assert a["mean"] == pytest.approx(1.5)
    b = next(r for r in d["rows"] if r["group"] == "b")
    assert b["median"] == pytest.approx(11.0)
    assert b["iqr"] == pytest.approx(b["q3"] - b["q1"])


def test_summary_stats_without_a_group_is_one_row(client, clinic):
    d = client.post(
        "/api/charts/summary_stats", json={"session_id": clinic, "y": "age"}
    ).json()
    assert len(d["rows"]) == 1 and d["rows"][0]["group"] == "All"


# ── scatter: ellipse + marginal + shape ─────────────────────────────────────

def test_confidence_ellipse_is_closed_and_grows_with_the_level(client, clinic):
    def area(level):
        d = client.post(
            "/api/charts/scatter",
            json={
                "session_id": clinic, "x": "age", "y": "ldl",
                "ellipse": True, "ellipse_level": level,
            },
        ).json()
        e = d["ellipses"][0]
        xs, ys = np.array(e["x"]), np.array(e["y"])
        # Shoelace formula on the returned polygon.
        return 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

    a95, a50 = area(0.95), area(0.50)
    assert a95 > a50, "a 95% region must enclose more than a 50% one"


def test_ellipse_is_drawn_per_group_when_a_colour_column_is_given(client, clinic):
    d = client.post(
        "/api/charts/scatter",
        json={
            "session_id": clinic, "x": "age", "y": "ldl",
            "color": "sex", "ellipse": True,
        },
    ).json()
    assert {e["group"] for e in d["ellipses"]} == {"M", "F"}
    assert all(len(e["x"]) > 50 for e in d["ellipses"])


def test_ellipse_on_a_group_too_small_to_have_one_says_so(client):
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "g": ["a", "b"]})
    sid = make_session(df, "ellipse_tiny")
    d = client.post(
        "/api/charts/scatter",
        json={"session_id": sid, "x": "x", "y": "y", "color": "g", "ellipse": True},
    ).json()
    assert all(e["x"] == [] and "note" in e for e in d["ellipses"])


def test_marginal_histograms_cover_every_point(client, clinic):
    d = client.post(
        "/api/charts/scatter",
        json={
            "session_id": clinic, "x": "age", "y": "ldl",
            "marginal": True, "marginal_bins": 15,
        },
    ).json()
    assert len(d["marginal"]["x"]) == 15
    n = len(d["points"])
    assert sum(b["count"] for b in d["marginal"]["x"]) == n
    assert sum(b["count"] for b in d["marginal"]["y"]) == n


def test_marginal_bins_are_bounded(client, clinic):
    r = client.post(
        "/api/charts/scatter",
        json={"session_id": clinic, "x": "age", "y": "ldl",
              "marginal": True, "marginal_bins": 5000},
    )
    assert r.status_code == 400


def test_shape_column_is_echoed_so_the_plot_can_vary_the_marker(client, clinic):
    d = client.post(
        "/api/charts/scatter",
        json={"session_id": clinic, "x": "age", "y": "ldl",
              "color": "sex", "shape": "outcome"},
    ).json()
    assert d["shape"] == "outcome"
    assert "outcome" in d["points"][0]


# ── facets ──────────────────────────────────────────────────────────────────

def test_facet_boxplot_returns_a_panel_per_level(client, clinic):
    d = client.post(
        "/api/charts/facet",
        json={"session_id": clinic, "kind": "boxplot", "x": "ldl",
              "facet": "site", "color": "sex"},
    ).json()
    assert {p["panel"] for p in d["panels"]} == {"Ankara", "Bursa", "Izmir"}
    for p in d["panels"]:
        assert {g["group"] for g in p["groups"]} == {"F", "M"}


def test_facet_shares_one_axis_range_across_panels(client, clinic):
    """Per-panel autoscaling is what makes small multiples misread."""
    d = client.post(
        "/api/charts/facet",
        json={"session_id": clinic, "kind": "scatter", "x": "age",
              "y": "ldl", "facet": "site"},
    ).json()
    lo, hi = d["shared_range"]["y"]
    every = [v for p in d["panels"] for v in p["y"]]
    assert lo == pytest.approx(min(every))
    assert hi == pytest.approx(max(every))


def test_facet_truncation_is_reported_not_silent(client):
    df = pd.DataFrame({"v": np.arange(40.0), "f": [f"s{i}" for i in range(40)]})
    sid = make_session(df, "facet_many")
    d = client.post(
        "/api/charts/facet",
        json={"session_id": sid, "kind": "boxplot", "x": "v",
              "facet": "f", "max_panels": 6},
    ).json()
    assert len(d["panels"]) == 6
    assert d["warnings"][0]["type"] == "panels_truncated"
    assert d["warnings"][0]["n_dropped"] == 34


def test_facet_scatter_without_y_is_a_400(client, clinic):
    r = client.post(
        "/api/charts/facet",
        json={"session_id": clinic, "kind": "scatter", "x": "age", "facet": "site"},
    )
    assert r.status_code == 400
    assert "y column" in r.json()["detail"]


def test_facet_rejects_an_unknown_kind(client, clinic):
    r = client.post(
        "/api/charts/facet",
        json={"session_id": clinic, "kind": "violin", "x": "ldl", "facet": "site"},
    )
    assert r.status_code == 400
