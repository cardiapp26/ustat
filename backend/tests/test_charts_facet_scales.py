"""Free panel scales (facet_wrap(scales=)) and a colour ramp on the scatter.

Both trade something away, so both are opt-in and both are named in the
response: a freed axis costs the comparison between panels, and a colour bar
of values costs the legend of groups.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def wards() -> str:
    rng = np.random.default_rng(9)
    # Two wards on very different scales: shared axes flatten the smaller one.
    df = pd.DataFrame(
        {
            "crp": np.concatenate([rng.normal(10, 2, 40), rng.normal(200, 30, 40)]),
            "ldl": np.concatenate([rng.normal(100, 10, 40), rng.normal(105, 10, 40)]),
            "ward": ["A"] * 40 + ["B"] * 40,
            "sex": ["F", "M"] * 40,
        }
    )
    return make_session(df, "facet_scales")


def _facet(client, sid, **kw):
    body = {"session_id": sid, "kind": "boxplot", "x": "crp", "facet": "ward"}
    body.update(kw)
    return client.post("/api/charts/facet", json=body)


def test_fixed_is_the_default_and_shares_the_value_axis(client, wards):
    d = _facet(client, wards).json()
    assert d["scales"] == "fixed"
    assert d["shared_range"]["x"]
    # Each panel still reports its own range, unused while the axis is shared.
    ranges = {p["panel"]: p["range"] for p in d["panels"]}
    assert ranges["A"][1] < ranges["B"][0]


def test_free_drops_the_shared_range_so_each_panel_keeps_its_own(client, wards):
    d = _facet(client, wards, scales="free").json()
    assert d["scales"] == "free"
    assert d["shared_range"] == {}
    assert all(p["range"] for p in d["panels"])


def test_a_box_plots_category_axis_cannot_be_freed(client, wards):
    """free_x names the axis carrying the group names; the values stay shared."""
    d = _facet(client, wards, scales="free_x").json()
    assert d["shared_range"]["x"]


def test_scatter_frees_exactly_the_axis_named(client, wards):
    free_y = _facet(client, wards, kind="scatter", x="ldl", y="crp", scales="free_y").json()
    assert "x" in free_y["shared_range"] and "y" not in free_y["shared_range"]
    free_x = _facet(client, wards, kind="scatter", x="ldl", y="crp", scales="free_x").json()
    assert "y" in free_x["shared_range"] and "x" not in free_x["shared_range"]
    for panel in free_x["panels"]:
        assert panel["range_x"] and panel["range_y"]


def test_column_count_is_echoed_and_bounded(client, wards):
    assert _facet(client, wards, ncol=2).json()["ncol"] == 2
    assert _facet(client, wards).json()["ncol"] is None
    assert _facet(client, wards, ncol=0).status_code == 400
    assert _facet(client, wards, ncol=99).status_code == 400
    assert _facet(client, wards, scales="loose").status_code == 400


def test_variable_panels_report_themselves_as_free(client, wards):
    d = client.post(
        "/api/charts/facet",
        json={"session_id": wards, "kind": "boxplot", "variables": ["crp", "ldl"]},
    ).json()
    assert d["scales"] == "free" and d["shared_range"] == {}


def test_scatter_colours_by_a_numeric_column_and_reports_its_range(client, wards):
    r = client.post(
        "/api/charts/scatter",
        json={"session_id": wards, "x": "ldl", "y": "crp", "gradient": "crp"},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["gradient"] == "crp"
    lo, hi = d["gradient_range"]
    assert lo < hi
    assert all("crp" in p for p in d["points"])


def test_a_colour_bar_and_a_group_legend_cannot_both_be_asked_for(client, wards):
    r = client.post(
        "/api/charts/scatter",
        json={"session_id": wards, "x": "ldl", "y": "crp", "gradient": "crp", "color": "sex"},
    )
    assert r.status_code == 400
    assert "colour bar" in r.json()["detail"]


def test_a_non_numeric_colour_column_is_refused(client, wards):
    r = client.post(
        "/api/charts/scatter",
        json={"session_id": wards, "x": "ldl", "y": "crp", "gradient": "ward"},
    )
    assert r.status_code == 400
    assert "numeric" in r.json()["detail"]
