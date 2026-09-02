"""Scatter trend lines — geom_smooth(method = lm | loess), overall or per group.

What is pinned: a bare request still draws the straight line with its band;
"none" removes the line but keeps the correlations the caption quotes; LOESS
returns a curve without a band it cannot justify; per-group fits come one per
level, each fitted on its own rows.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def curved() -> str:
    rng = np.random.default_rng(5)
    x = np.linspace(1, 10, 80)
    df = pd.DataFrame(
        {
            "x": x,
            # A hump a straight line cannot follow.
            "y": np.sin(x / 2) * 5 + 10 + rng.normal(0, 0.5, 80),
            "grp": ["a", "b"] * 40,
        }
    )
    return make_session(df, "scatter_smooth")


def _scatter(client, sid, **kw):
    body = {"session_id": sid, "x": "x", "y": "y"}
    body.update(kw)
    return client.post("/api/charts/scatter", json=body)


def test_default_is_the_straight_line_with_its_band(client, curved):
    d = _scatter(client, curved).json()
    reg = d["regression"]
    assert d["fit"] == "lm" and reg["method"] == "lm"
    assert len(reg["line_x"]) == 2
    assert reg["band"]["level"] == 0.95
    assert d["regressions"] == []


def test_none_keeps_the_correlations_but_draws_nothing(client, curved):
    reg = _scatter(client, curved, fit="none").json()["regression"]
    assert reg["line_x"] == [] and reg["band"] == {}
    assert reg["r"] is not None
    assert reg["spearman"]["rho"] is not None


def test_loess_follows_the_hump_a_line_misses(client, curved):
    reg = _scatter(client, curved, fit="loess", loess_span=0.4).json()["regression"]
    assert reg["method"] == "loess" and reg["span"] == 0.4
    assert reg["band"] == {}
    xs, ys = np.array(reg["line_x"]), np.array(reg["line_y"])
    assert len(xs) > 10
    assert np.all(np.diff(xs) > 0)  # sorted, one point per x
    # The curve tracks the truth far better than the straight line.
    truth = np.sin(xs / 2) * 5 + 10
    line = np.polyval(np.polyfit(xs, ys, 1), xs)
    assert np.abs(ys - truth).mean() < 0.5 * np.abs(line - truth).mean()


def test_per_group_fits_come_one_per_level(client, curved):
    d = _scatter(client, curved, color="grp", fit_per_group=True).json()
    assert d["fit_per_group"] is True
    assert [g["group"] for g in d["regressions"]] == ["a", "b"]
    for g in d["regressions"]:
        assert g["n"] == 40
        assert len(g["line_x"]) == 2
        assert g["band"]["level"] == 0.95


def test_per_group_is_ignored_without_a_colour_column(client, curved):
    d = _scatter(client, curved, fit_per_group=True).json()
    assert d["fit_per_group"] is False and d["regressions"] == []


def test_loess_on_a_log_axis_is_fitted_in_log_space(client):
    x = np.array([1, 10, 100, 1000, 10000, 100000], dtype=float)
    df = pd.DataFrame({"x": x, "y": np.log10(x) * 2 + 1})
    sid = make_session(df, "scatter_smooth_log")
    r = client.post(
        "/api/charts/scatter",
        json={"session_id": sid, "x": "x", "y": "y", "fit": "loess", "log_x": True, "loess_span": 1.0},
    )
    assert r.status_code == 200, r.text
    reg = r.json()["regression"]
    assert reg["space"] == "log10-x"
    # Back-transformed to data coordinates: the curve spans the raw x range.
    assert reg["line_x"][0] == pytest.approx(1.0) and reg["line_x"][-1] == pytest.approx(1e5)


def test_bad_method_and_span_are_rejected(client, curved):
    assert _scatter(client, curved, fit="gam").status_code == 400
    assert _scatter(client, curved, fit="loess", loess_span=0.01).status_code == 400
