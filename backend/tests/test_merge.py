"""Joining a second file onto the open dataset.

The failure modes here are silent, so most of these tests are about what the
endpoint refuses to do quietly rather than about the join itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import store


@pytest.fixture()
def pair(client):
    base = pd.DataFrame({
        "pid": [1001, 1002, 1003, 1004, 1005],
        "age": [61, 54, 72, 66, 48],
        "sbp": [130, 142, 128, 155, 119],
    })
    labs = pd.DataFrame({
        "id": ["1001", "1002", " 1003", "1006"],   # text keys, one padded, one unknown here
        "crp": [2.1, 8.4, 1.2, 5.5],
        "sbp": [999, 999, 999, 999],               # deliberately clashes with base
    })
    store.save("mrg_base", base)
    store.save("mrg_labs", labs)
    return "mrg_base", "mrg_labs"


def _body(**extra):
    return {"session_id": "mrg_base", "other_session_id": "mrg_labs",
            "left_on": ["pid"], "right_on": ["id"], **extra}


def _preview(client, **extra):
    r = client.post("/api/merge/preview", json=_body(**extra))
    assert r.status_code == 200, r.text
    return r.json()


def _apply(client, **extra):
    r = client.post("/api/merge/apply", json=_body(**extra))
    assert r.status_code == 200, r.text
    return r.json()


# ── keys ───────────────────────────────────────────────────────────────────────


def test_a_number_and_the_same_number_as_text_are_the_same_participant(client, pair):
    # pandas typing one column int and the other object is a fact about the
    # file, not about the patients. Matching zero rows over it would be a
    # tooling artefact.
    out = _preview(client)
    assert out["keys_matched"] == 3
    assert out["left_rows_matched"] == 3


def test_whitespace_around_a_key_does_not_lose_a_match(client, pair):
    # " 1003" is 1003.
    out = _apply(client)
    merged = store.get("mrg_base")
    assert merged.loc[merged["pid"] == 1003, "crp"].iloc[0] == pytest.approx(1.2)


def test_a_float_key_holding_whole_numbers_does_not_become_1024_point_0(client):
    store.save("mrg_f", pd.DataFrame({"pid": [1001.0, 1002.0], "x": [1, 2]}))
    store.save("mrg_t", pd.DataFrame({"id": ["1001", "1002"], "y": [7, 8]}))
    r = client.post("/api/merge/preview", json={
        "session_id": "mrg_f", "other_session_id": "mrg_t", "left_on": ["pid"], "right_on": ["id"]})
    assert r.json()["keys_matched"] == 2


def test_case_is_not_folded(client):
    # "AB12" and "ab12" may be two different participants, and quietly merging
    # them is not a decision to make on the user's behalf.
    store.save("mrg_a", pd.DataFrame({"pid": ["AB12"], "x": [1]}))
    store.save("mrg_b", pd.DataFrame({"id": ["ab12"], "y": [2]}))
    r = client.post("/api/merge/preview", json={
        "session_id": "mrg_a", "other_session_id": "mrg_b", "left_on": ["pid"], "right_on": ["id"]})
    assert r.json()["keys_matched"] == 0


def test_a_blank_key_never_matches_another_blank_key(client):
    store.save("mrg_n1", pd.DataFrame({"pid": ["1001", None, None], "x": [1, 2, 3]}))
    store.save("mrg_n2", pd.DataFrame({"id": ["1001", None], "y": [7, 8]}))
    r = client.post("/api/merge/apply", json={
        "session_id": "mrg_n1", "other_session_id": "mrg_n2", "left_on": ["pid"], "right_on": ["id"]})
    out = r.json()
    assert out["left_keys_missing"] == 2 and out["right_keys_missing"] == 1
    assert out["rows_after"] == 3  # the two blanks did not join to each other
    assert store.get("mrg_n1")["y"].notna().sum() == 1


# ── the failure that inflates a dataset ────────────────────────────────────────


def test_duplicate_keys_in_the_incoming_file_are_called_out_before_the_join(client):
    store.save("mrg_d1", pd.DataFrame({"pid": [1, 2], "x": [10, 20]}))
    store.save("mrg_d2", pd.DataFrame({"id": [1, 1, 2], "y": [7, 8, 9]}))
    r = client.post("/api/merge/preview", json={
        "session_id": "mrg_d1", "other_session_id": "mrg_d2", "left_on": ["pid"], "right_on": ["id"]})
    out = r.json()
    assert out["right_duplicate_keys"] == 1
    assert any("more rows than the dataset you started with" in w for w in out["warnings"])
    assert out["rows_after"] is None  # cannot be promised in advance


def test_a_join_that_grew_the_dataset_says_so_first(client):
    store.save("mrg_g1", pd.DataFrame({"pid": [1, 2], "x": [10, 20]}))
    store.save("mrg_g2", pd.DataFrame({"id": [1, 1, 2], "y": [7, 8, 9]}))
    r = client.post("/api/merge/apply", json={
        "session_id": "mrg_g1", "other_session_id": "mrg_g2", "left_on": ["pid"], "right_on": ["id"]})
    out = r.json()
    assert out["rows_after"] == 3 > out["rows_left"]
    assert "grew from 2 to 3 rows" in out["warnings"][0]


# ── columns ────────────────────────────────────────────────────────────────────


def test_a_clashing_column_is_renamed_rather_than_overwriting(client, pair):
    out = _apply(client)
    merged = store.get("mrg_base")
    assert "sbp_2" in merged.columns
    assert merged["sbp"].tolist() == [130, 142, 128, 155, 119]   # untouched
    assert any("renamed" in w for w in out["warnings"])


def test_only_the_chosen_columns_come_across(client, pair):
    _apply(client, columns=["crp"])
    merged = store.get("mrg_base")
    assert "crp" in merged.columns and "sbp_2" not in merged.columns


def test_the_key_column_of_the_incoming_file_is_not_added(client, pair):
    _apply(client)
    assert "id" not in store.get("mrg_base").columns


# ── reporting ──────────────────────────────────────────────────────────────────


def test_preview_changes_nothing(client, pair):
    before = store.get("mrg_base").copy()
    _preview(client)
    pd.testing.assert_frame_equal(store.get("mrg_base"), before)


def test_unmatched_rows_are_counted_both_ways(client, pair):
    out = _preview(client)
    assert out["left_rows_unmatched"] == 2      # 1004, 1005
    assert out["right_keys_unused"] == 1        # 1006
    assert any("found no match" in w for w in out["warnings"])
    assert any("choose an outer join" in w for w in out["warnings"])


def test_an_outer_join_keeps_the_rows_a_left_join_would_drop(client, pair):
    out = _apply(client, how="outer")
    assert out["rows_after"] == 6               # 5 here + 1006 from the other file


def test_nothing_matching_is_reported_as_such(client):
    store.save("mrg_x1", pd.DataFrame({"pid": [1, 2], "x": [1, 2]}))
    store.save("mrg_x2", pd.DataFrame({"visit": [9, 8], "y": [1, 2]}))
    r = client.post("/api/merge/preview", json={
        "session_id": "mrg_x1", "other_session_id": "mrg_x2", "left_on": ["pid"], "right_on": ["visit"]})
    assert any("No key matched" in w for w in r.json()["warnings"])


# ── refusals ───────────────────────────────────────────────────────────────────


def test_mismatched_key_lists_are_refused(client, pair):
    r = client.post("/api/merge/preview", json=_body(left_on=["pid"], right_on=["id", "crp"]))
    assert r.status_code == 400


def test_an_unknown_key_column_is_refused(client, pair):
    r = client.post("/api/merge/preview", json=_body(left_on=["nope"]))
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]


def test_an_unknown_join_type_is_refused(client, pair):
    assert client.post("/api/merge/preview", json=_body(how="cross")).status_code == 400


def test_a_file_with_nothing_to_add_is_refused(client):
    store.save("mrg_o1", pd.DataFrame({"pid": [1], "x": [1]}))
    store.save("mrg_o2", pd.DataFrame({"id": [1]}))
    r = client.post("/api/merge/preview", json={
        "session_id": "mrg_o1", "other_session_id": "mrg_o2", "left_on": ["pid"], "right_on": ["id"]})
    assert r.status_code == 400
    assert "no columns to add" in r.json()["detail"]


def test_a_missing_second_session_is_a_404(client, pair):
    r = client.post("/api/merge/preview", json=_body(other_session_id="not_a_session"))
    assert r.status_code == 404


def test_two_key_columns_join_on_the_pair(client):
    store.save("mrg_k1", pd.DataFrame({"pid": [1, 1, 2], "visit": [1, 2, 1], "x": [10, 11, 20]}))
    store.save("mrg_k2", pd.DataFrame({"p": [1, 1, 2], "v": [1, 2, 9], "y": [7, 8, 9]}))
    r = client.post("/api/merge/apply", json={
        "session_id": "mrg_k1", "other_session_id": "mrg_k2",
        "left_on": ["pid", "visit"], "right_on": ["p", "v"]})
    out = r.json()
    assert out["keys_matched"] == 2 and out["rows_after"] == 3
    assert store.get("mrg_k1")["y"].tolist()[:2] == [7, 8]


def test_a_join_covers_rows_that_select_cases_is_hiding_and_says_so(client):
    """The join reads the whole sheet on purpose.

    Joining the filtered view and saving the result would delete every
    excluded row — a display filter turned into permanent data loss. So the
    full dataset is joined, and the mismatch between these counts and the
    sheet on screen is stated rather than left to be discovered.
    """
    store.save("mrg_flt", pd.DataFrame({"pid": [1, 2, 3, 4], "keep": [1, 1, 0, 0], "x": [1, 2, 3, 4]}))
    store.save("mrg_flt2", pd.DataFrame({"id": [1, 2, 3, 4], "y": [9, 8, 7, 6]}))
    store.save_filter("mrg_flt", [{"column": "keep", "operator": "eq", "value": "1", "join": "AND"}])

    r = client.post("/api/merge/apply", json={
        "session_id": "mrg_flt", "other_session_id": "mrg_flt2",
        "left_on": ["pid"], "right_on": ["id"]})
    out = r.json()
    assert out["rows_left"] == 4 and out["rows_after"] == 4      # not the 2 on screen
    assert any("Select Cases is active" in w for w in out["warnings"])
    assert len(store.get("mrg_flt")) == 4                         # nothing was dropped
    store.save_filter("mrg_flt", [])
