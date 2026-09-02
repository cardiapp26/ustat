"""geom_bin2d: a scatter too dense to read, binned into a grid of counts.

Pinned: the grid replaces the points rather than travelling beside them, it
counts every row, it is binned in the space the reader sees on a log axis, and
it refuses to also carry a group legend or a value ramp — the heatmap is
already using colour.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def dense() -> str:
    rng = np.random.default_rng(4)
    n = 3000
    x = rng.normal(100, 15, n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": x * 0.5 + rng.normal(0, 10, n),
            "grp": rng.choice(["a", "b"], n),
        }
    )
    return make_session(df, "bin2d")


def _scatter(client, sid, **kw):
    body = {"session_id": sid, "x": "x", "y": "y"}
    body.update(kw)
    return client.post("/api/charts/scatter", json=body)


def test_off_by_default(client, dense):
    d = _scatter(client, dense).json()
    assert d["bin2d"] == {}
    assert len(d["points"]) == 3000


def test_the_grid_replaces_the_points_and_counts_them_all(client, dense):
    d = _scatter(client, dense, bin2d=True, bin2d_bins=20).json()
    grid = d["bin2d"]
    assert d["points"] == []
    assert grid["n"] == 3000
    assert len(grid["x"]) == 20 and len(grid["y"]) == 20
    # z is row-major over y: one row per y bin, one column per x bin.
    assert len(grid["z"]) == 20 and all(len(row) == 20 for row in grid["z"])
    assert sum(sum(row) for row in grid["z"]) == 3000
    assert grid["max"] == max(max(row) for row in grid["z"])


def test_the_fit_still_comes_back_with_the_grid(client, dense):
    """The line is computed from the rows, not the points that were shipped."""
    reg = _scatter(client, dense, bin2d=True).json()["regression"]
    assert reg["n"] == 3000
    assert reg["r"] > 0.5
    assert len(reg["line_x"]) == 2


def test_log_axis_bins_are_even_in_log_space(client):
    x = np.geomspace(1, 10_000, 400)
    df = pd.DataFrame({"x": x, "y": x})
    sid = make_session(df, "bin2d_log")
    r = client.post(
        "/api/charts/scatter",
        json={"session_id": sid, "x": "x", "y": "y", "bin2d": True, "bin2d_bins": 8, "log_x": True},
    )
    assert r.status_code == 200, r.text
    centres = np.array(r.json()["bin2d"]["x"])
    # Even in log space: the ratio between neighbouring centres is constant.
    ratios = centres[1:] / centres[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-6)


def test_a_grid_cannot_also_carry_a_legend_or_a_ramp(client, dense):
    assert _scatter(client, dense, bin2d=True, color="grp").status_code == 400
    r = _scatter(client, dense, bin2d=True, gradient="x")
    assert r.status_code == 400
    assert "counts" in r.json()["detail"]


def test_bin_count_is_bounded(client, dense):
    assert _scatter(client, dense, bin2d=True, bin2d_bins=2).status_code == 400
    assert _scatter(client, dense, bin2d=True, bin2d_bins=999).status_code == 400
