"""A column added after upload has to behave like a column.

Everything typed into a freshly added column was stored as text, because
_detect_kind calls an empty column "categorical" and a column of numeric
strings "categorical" too. The visible symptom was the grid header's range
badge: the new column showed no minimum or maximum while the columns beside
it did. The invisible one was worse — the numbers were labels to every model,
and sorting was lexicographic.
"""
from __future__ import annotations

import pandas as pd
import pytest

from services import store
from services.dirty_value_guard import values_are_numeric


@pytest.mark.parametrize("values,expected", [
    ([None, None], True),          # nothing yet contradicts a number
    (["3", "7"], True),            # numbers that happen to be stored as text
    ([3.0, "7"], True),            # half-converted, as the old writer left them
    ([1, 2, 3], True),
    (["", "  ", "5"], True),       # blanks are gaps, not values
    (["apple", "3"], False),       # one genuine word settles it
    (["3.5", "abc"], False),
])
def test_values_are_numeric(values, expected):
    assert values_are_numeric(pd.Series(values)) is expected


def test_typing_numbers_into_a_new_column_stores_numbers(client):
    store.save("newcol", pd.DataFrame({"a": [1, 2, 3]}))
    client.post("/api/compute/newcol/add_column", json={"name": "score", "position": -1})
    for i, v in enumerate(["3", "7", "11"]):
        r = client.patch("/api/sessions/newcol/cell",
                         json={"row_index": i, "column": "score", "value": v})
        assert r.status_code == 200

    # Not ["3", "7", "11"]: a column of strings sorts lexicographically and
    # reaches every model as a set of labels.
    assert store.get("newcol")["score"].tolist() == [3.0, 7.0, 11.0]


def test_the_new_column_gets_a_range_badge(client):
    store.save("badge", pd.DataFrame({"a": [1, 2, 3]}))
    client.post("/api/compute/badge/add_column", json={"name": "score", "position": -1})
    for i, v in enumerate(["3", "7", "11"]):
        client.patch("/api/sessions/badge/cell",
                     json={"row_index": i, "column": "score", "value": v})

    badge = client.get("/api/stats/badge/column_badges").json()["columns"]["score"]
    assert badge["min"] == 3.0
    assert badge["max"] == 11.0
    assert badge["n_valid"] == 3


def test_a_column_of_numeric_strings_gets_its_range_without_being_rewritten(client):
    """Sessions saved before the writer was fixed still hold strings; the
    header should report their range rather than waiting for a migration."""
    store.save("legacy", pd.DataFrame({"legacy": ["3", "7", "11"]}))
    badge = client.get("/api/stats/legacy/column_badges").json()["columns"]["legacy"]
    assert (badge["min"], badge["max"]) == (3.0, 11.0)


def test_a_text_column_is_left_alone(client):
    """Typing "3" into a column of words must not turn it numeric, and a
    minimum over "apple" and "3" is sort order, not a fact about the data."""
    store.save("txt", pd.DataFrame({"txt": ["apple", "pear", None]}))
    client.patch("/api/sessions/txt/cell",
                 json={"row_index": 2, "column": "txt", "value": "3"})

    assert store.get("txt")["txt"].tolist() == ["apple", "pear", "3"]
    badge = client.get("/api/stats/txt/column_badges").json()["columns"]["txt"]
    assert "min" not in badge and "max" not in badge


def test_an_explicit_kind_override_still_wins(client):
    """A user who marked the column categorical meant it."""
    store.save("ovr", pd.DataFrame({"grade": [None, None, None]}))
    client.post("/api/sessions/ovr/kind", json={"column": "grade", "kind": "categorical"})
    client.patch("/api/sessions/ovr/cell",
                 json={"row_index": 0, "column": "grade", "value": "3"})
    assert store.get("ovr")["grade"].tolist()[0] == "3"
