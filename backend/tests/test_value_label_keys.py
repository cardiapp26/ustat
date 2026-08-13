"""Category codes must reach the client under the key their labels are stored by.

Reported: a Histopathology column labelled through the grid (0 = Benign,
1 = Papillary thyroid cancer, ... 8 = Other) still showed 0.0, 1.0, 7.0, 8.0
in the distribution chart. Value labels are keyed by what the grid displayed
when they were typed — a JSON number, so "0" — while the analysis endpoints
stringified the same code from a float64 column as "0.0". Nothing matched.

The Data Dictionary made it worse than a display problem: its inputs were
keyed off the unique-values endpoint, which had the second spelling, so a
label typed there was stored under a key nothing else looked up. One column
could end up with "0" and "0.0" both labelled and neither one winning.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session
from services.number_format import level_key


# ── the canonical form ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "0"),
        (np.float64(1.0), "1"),
        (8, "8"),
        (-3.0, "-3"),
        (1.7, "1.7"),
        (np.float64(1.70), "1.7"),
        ("  Benign ", "Benign"),
        (True, "True"),
    ],
)
def test_level_key_matches_what_the_grid_shows(value, expected):
    assert level_key(value) == expected


def test_a_missing_value_has_no_code():
    """The caller decides what to call missing; a key of "nan" is not it."""
    assert level_key(np.nan) == ""
    assert level_key(None) == ""


def test_a_fractional_code_is_left_alone():
    """Data entry produces codes like 1.7 and 1.8. Rounding them to a whole
    number would merge two categories into one."""
    assert level_key(1.7) != level_key(1.8)


# ── the endpoints that feed a label lookup ────────────────────────────────────


@pytest.fixture()
def sid() -> str:
    df = pd.DataFrame(
        {
            "Histopatoloji": [0.0] * 10 + [1.0] * 5 + [1.7] * 2 + [8.0] * 3,
            "arm": (["A", "B"] * 10)[:20],
        }
    )
    return make_session(df, "vlabel_keys")


def test_the_distribution_summary_uses_the_label_key(client, sid):
    r = client.get(
        f"/api/stats/{sid}/column_summary",
        params={"column": "Histopatoloji", "kind": "categorical"},
    )
    assert r.status_code == 200, r.text
    values = [c["value"] for c in r.json()["categories"]]
    assert set(values) == {"0", "1", "1.7", "8"}


def test_the_unique_values_endpoint_agrees_with_it(client, sid):
    """The Data Dictionary and the grid dialog write into the same map. If the
    two disagree about the key, a label typed in one is invisible in the other."""
    r = client.get(f"/api/compute/{sid}/unique/Histopatoloji")
    assert r.status_code == 200, r.text
    assert r.json()["values"] == ["0", "1", "1.7", "8"]


def test_table_one_prints_the_same_levels(client, sid):
    r = client.post(
        "/api/stats/table1",
        json={
            "session_id": sid,
            "variables": ["Histopatoloji"],
            "group_column": "arm",
            "variable_kinds": {"Histopatoloji": "categorical"},
        },
    )
    assert r.status_code == 200, r.text
    row = r.json()["rows"][0]
    assert {sr["category"] for sr in row["sub_rows"]} == {"0", "1", "1.7", "8"}


def test_the_counts_still_belong_to_the_right_level(client, sid):
    """The level string used to match the data and the one printed are now
    different strings. Crossing them would silently attach 0's count to 1."""
    r = client.post(
        "/api/stats/table1",
        json={
            "session_id": sid,
            "variables": ["Histopatoloji"],
            "variable_kinds": {"Histopatoloji": "categorical"},
        },
    )
    assert r.status_code == 200, r.text
    counts = {
        sr["category"]: int(sr["overall"].split(" ")[0])
        for sr in r.json()["rows"][0]["sub_rows"]
    }
    assert counts == {"0": 10, "1": 5, "1.7": 2, "8": 3}


def test_a_text_column_is_untouched(client):
    sid = make_session(
        pd.DataFrame({"grade": ["I", "II", "II", "III"]}), "vlabel_text"
    )
    r = client.get(
        f"/api/stats/{sid}/column_summary",
        params={"column": "grade", "kind": "categorical"},
    )
    assert {c["value"] for c in r.json()["categories"]} == {"I", "II", "III"}
