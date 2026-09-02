"""Grouped histogram — geom_histogram(aes(fill = group)), binwidth, rug.

Pinned: groups are binned on one shared set of edges (the comparison is
meaningless otherwise), a bin width aligns edges to multiples of itself, the
raw values come back only when a rug is asked for, and the legacy single-group
fields survive so a plain histogram is unchanged.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def two_arms() -> str:
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "ldl": np.concatenate([rng.normal(120, 20, 60), rng.normal(100, 20, 40)]),
            "arm": ["Control"] * 60 + ["Drug"] * 40,
        }
    )
    return make_session(df, "hist_groups")


def _hist(client, sid, **kw):
    body = {"session_id": sid, "x": "ldl"}
    body.update(kw)
    return client.post("/api/charts/histogram", json=body)


def test_plain_histogram_keeps_its_old_shape_and_gains_one_group(client, two_arms):
    d = _hist(client, two_arms, bins=10).json()
    assert len(d["bins"]) == 10 and d["kde"]
    assert d["color"] is None
    assert [g["group"] for g in d["groups"]] == ["All"]
    assert d["groups"][0]["n"] == 100
    assert sum(d["groups"][0]["counts"]) == 100
    assert "values" not in d["groups"][0]


def test_groups_share_edges_and_add_up_to_the_whole(client, two_arms):
    d = _hist(client, two_arms, bins=12, color="arm").json()
    assert [g["group"] for g in d["groups"]] == ["Control", "Drug"]
    assert [g["n"] for g in d["groups"]] == [60, 40]
    assert len(d["edges"]) == 13
    for g in d["groups"]:
        assert len(g["counts"]) == 12
        assert len(g["kde"]) == 200
    summed = np.add(d["groups"][0]["counts"], d["groups"][1]["counts"])
    assert summed.tolist() == [b["count"] for b in d["bins"]]


def test_binwidth_aligns_edges_to_its_multiples(client, two_arms):
    d = _hist(client, two_arms, binwidth=10).json()
    edges = np.array(d["edges"])
    assert d["bin_width"] == pytest.approx(10)
    assert np.allclose(edges % 10, 0) or np.allclose(edges % 10, 10)
    assert edges[0] <= 50 and edges[-1] >= 170


def test_absurd_binwidth_is_refused_not_exploded(client, two_arms):
    r = _hist(client, two_arms, binwidth=0.001)
    assert r.status_code == 400
    assert "limit" in r.json()["detail"]
    assert _hist(client, two_arms, binwidth=-1).status_code == 400


def test_rug_returns_the_raw_values_per_group(client, two_arms):
    d = _hist(client, two_arms, color="arm", rug=True).json()
    assert [len(g["values"]) for g in d["groups"]] == [60, 40]


def test_missing_colour_column_is_rejected(client, two_arms):
    assert _hist(client, two_arms, color="nope").status_code == 400
