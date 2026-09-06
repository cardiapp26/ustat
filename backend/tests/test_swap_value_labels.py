"""POST /api/sessions/{id}/swap_value_labels — put the labels in the cells.

A column typed as words reads "ANTERIOR = 1" the moment someone writes the code
they meant into the label box: backwards from what they want stored. Swapping
makes it "1 = ANTERIOR" — the codes live in the data, the words label them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import store


@pytest.fixture()
def sid(client):
    df = pd.DataFrame({
        "stemi": ["ANTERIOR", "LATERAL", np.nan, "ANTERIOR", "POSTERIOR"],
        "other": [1, 2, 3, 4, 5],
    })
    store.save("swaplbl", df)
    store.save_metadata("swaplbl", {"stemi": {"value_labels": {"ANTERIOR": "1"}}})
    return "swaplbl"


def _swap(client, sid, labels, column="stemi"):
    return client.post(
        f"/api/sessions/{sid}/swap_value_labels",
        json={"column": column, "labels": labels},
    )


def test_cells_become_their_labels_and_the_map_is_inverted(client, sid):
    r = _swap(client, sid, {"ANTERIOR": "1", "LATERAL": "2", "POSTERIOR": "3"})
    assert r.status_code == 200
    assert r.json()["changed"] == 4
    assert r.json()["value_labels"] == {"1": "ANTERIOR", "2": "LATERAL", "3": "POSTERIOR"}

    col = store.get(sid)["stemi"]
    assert col.dropna().tolist() == [1.0, 2.0, 1.0, 3.0]
    assert store.get_metadata(sid)["stemi"]["value_labels"] == {
        "1": "ANTERIOR", "2": "LATERAL", "3": "POSTERIOR",
    }


def test_codes_typed_as_numbers_are_stored_as_numbers(client, sid):
    """Left as text the column sorts lexicographically and every model reads
    its codes as free labels."""
    _swap(client, sid, {"ANTERIOR": "1", "LATERAL": "2", "POSTERIOR": "3"})
    assert pd.api.types.is_numeric_dtype(store.get(sid)["stemi"])


def test_a_column_that_stays_textual_is_left_as_text(client, sid):
    _swap(client, sid, {"ANTERIOR": "ANT", "LATERAL": "LAT", "POSTERIOR": "POST"})
    assert store.get(sid)["stemi"].dropna().tolist() == ["ANT", "LAT", "ANT", "POST"]


def test_unlabelled_values_keep_what_they_hold(client, sid):
    r = _swap(client, sid, {"ANTERIOR": "1", "LATERAL": ""})
    assert r.json()["value_labels"] == {"1": "ANTERIOR"}
    assert store.get(sid)["stemi"].tolist()[:2] == ["1", "LATERAL"]


def test_missing_cells_stay_missing(client, sid):
    _swap(client, sid, {"ANTERIOR": "1", "LATERAL": "2", "POSTERIOR": "3"})
    assert store.get(sid)["stemi"].isna().tolist() == [False, False, True, False, False]


def test_two_values_sharing_a_label_are_refused(client, sid):
    """Swapping would pool two distinct groups into one, and the swap is meant
    to rename values, not merge them."""
    r = _swap(client, sid, {"ANTERIOR": "1", "LATERAL": "1"})
    assert r.status_code == 422
    assert "merge" in r.json()["detail"]
    assert store.get(sid)["stemi"].iloc[0] == "ANTERIOR"   # nothing written


def test_a_label_colliding_with_an_unlabelled_value_is_refused(client, sid):
    r = _swap(client, sid, {"ANTERIOR": "POSTERIOR"})
    assert r.status_code == 422
    assert "already appears in the column" in r.json()["detail"]


def test_a_numeric_code_matches_the_float_the_cell_holds(client):
    """The user labels what the grid shows them; a 0/1 column holds 0.0."""
    store.save("swapnum", pd.DataFrame({"sex": [0.0, 1.0, 1.0]}))
    r = client.post("/api/sessions/swapnum/swap_value_labels",
                    json={"column": "sex", "labels": {"0": "female", "1": "male"}})
    assert r.status_code == 200
    assert store.get("swapnum")["sex"].tolist() == ["female", "male", "male"]
    assert r.json()["value_labels"] == {"female": "0", "male": "1"}


def test_undo_restores_the_data_and_the_labels_together(client, sid):
    """Restoring the codes under the new labels would bring the column back
    unreadable, so both go under one snapshot."""
    _swap(client, sid, {"ANTERIOR": "1", "LATERAL": "2", "POSTERIOR": "3"})
    assert client.post(f"/api/sessions/{sid}/undo").status_code == 200
    assert store.get(sid)["stemi"].iloc[0] == "ANTERIOR"
    assert store.get_metadata(sid)["stemi"]["value_labels"] == {"ANTERIOR": "1"}


def test_the_columns_kind_survives_words_becoming_digits(client, sid):
    _swap(client, sid, {"ANTERIOR": "1", "LATERAL": "2", "POSTERIOR": "3"})
    assert store.get_kind_overrides(sid).get("stemi") == "categorical"


def test_no_labels_at_all_is_refused(client, sid):
    r = _swap(client, sid, {"ANTERIOR": "  "})
    assert r.status_code == 422
    assert "Nothing to swap" in r.json()["detail"]


def test_unknown_column_and_session_are_errors(client, sid):
    assert _swap(client, sid, {"a": "1"}, column="nope").status_code == 400
    assert _swap(client, "does-not-exist", {"a": "1"}).status_code == 404
