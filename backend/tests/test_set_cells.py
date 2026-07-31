"""POST /api/sessions/{id}/set_cells — write one value into a cell selection.

The counterpart of clear_cells, backing the Data grid's "Convert value" and
"Fill blanks with" on a multi-cell selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import store


@pytest.fixture()
def sid(client):
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0, 0.0],
        "txt": ["x", "", "y", "x"],
    })
    store.save("setcells", df)
    return "setcells"


def _all_cells(columns=("num", "txt"), rows=4):
    return [{"row_index": r, "column": c} for r in range(rows) for c in columns]


def test_fill_blanks_only_touches_empty_cells(client, sid):
    r = client.post(f"/api/sessions/{sid}/set_cells", json={
        "cells": _all_cells(), "value": "9", "only_blank": True})
    assert r.status_code == 200
    assert r.json() == {"changed": 2, "skipped": 6}
    df = store.get(sid)
    assert df["num"].tolist() == [1.0, 9.0, 3.0, 0.0]
    assert df["txt"].tolist() == ["x", "9", "y", "x"]


def test_a_whitespace_only_string_counts_as_blank(client):
    """It reads as a value to pandas and as a gap to the person looking at the
    sheet; "fill the blanks" has to mean the same thing to both."""
    store.save("ws", pd.DataFrame({"txt": ["a", "   ", ""]}))
    r = client.post("/api/sessions/ws/set_cells", json={
        "cells": [{"row_index": i, "column": "txt"} for i in range(3)],
        "value": "z", "only_blank": True})
    assert r.json()["changed"] == 2
    assert store.get("ws")["txt"].tolist() == ["a", "z", "z"]


def test_convert_value_changes_only_matching_cells(client, sid):
    r = client.post(f"/api/sessions/{sid}/set_cells", json={
        "cells": _all_cells(), "value": "z", "match_value": "x"})
    assert r.json() == {"changed": 2, "skipped": 6}
    assert store.get(sid)["txt"].tolist() == ["z", "", "y", "z"]


def test_convert_matches_a_number_by_what_the_cell_shows(client, sid):
    """The user types what they see; the column's dtype is not their concern."""
    r = client.post(f"/api/sessions/{sid}/set_cells", json={
        "cells": _all_cells(), "value": "7", "match_value": "0"})
    assert r.json()["changed"] == 1
    num = store.get(sid)["num"]
    # NaN never equals itself, so compare the observed values and the gap
    # separately rather than the whole list.
    assert num.dropna().tolist() == [1.0, 3.0, 7.0]
    assert num.isna().tolist() == [False, True, False, False]


def test_without_a_filter_every_selected_cell_is_written(client, sid):
    cells = [{"row_index": r, "column": "num"} for r in range(4)]
    r = client.post(f"/api/sessions/{sid}/set_cells", json={
        "cells": cells, "value": "5"})
    assert r.json()["changed"] == 4
    assert store.get(sid)["num"].tolist() == [5.0, 5.0, 5.0, 5.0]


def test_a_numeric_column_stays_numeric(client, sid):
    """Writing "3" as text would turn the column into strings and silently
    break every model built on it."""
    client.post(f"/api/sessions/{sid}/set_cells", json={
        "cells": [{"row_index": 0, "column": "num"}], "value": "42"})
    assert pd.api.types.is_numeric_dtype(store.get(sid)["num"])
    assert store.get(sid)["num"].iloc[0] == 42.0


def test_a_non_numeric_value_widens_the_column_rather_than_failing(client, sid):
    client.post(f"/api/sessions/{sid}/set_cells", json={
        "cells": [{"row_index": 0, "column": "num"}], "value": "unknown"})
    assert store.get(sid)["num"].iloc[0] == "unknown"


def test_the_two_filters_cannot_be_combined(client, sid):
    """A blank cell cannot also equal a value, so asking for both is a
    mistake worth naming rather than an empty result."""
    r = client.post(f"/api/sessions/{sid}/set_cells", json={
        "cells": _all_cells(), "value": "1",
        "only_blank": True, "match_value": "x"})
    assert r.status_code == 422


def test_out_of_range_and_unknown_columns_are_ignored(client, sid):
    r = client.post(f"/api/sessions/{sid}/set_cells", json={
        "cells": [
            {"row_index": 99, "column": "num"},
            {"row_index": 0, "column": "nope"},
            {"row_index": 0, "column": "num"},
        ],
        "value": "5"})
    assert r.json()["changed"] == 1


def test_unknown_session_is_404(client):
    r = client.post("/api/sessions/does-not-exist/set_cells", json={
        "cells": [], "value": "1"})
    assert r.status_code == 404
