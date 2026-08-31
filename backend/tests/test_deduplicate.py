"""Dropping duplicate rows by a chosen key.

The dangerous half of this endpoint is not the drop, it is what counts as a
duplicate. Two visits by one patient share an identity number and are not the
same record; two rows pasted twice are. So the key is the caller's choice, the
count is available before anything is deleted, and rows whose key is entirely
blank are never collapsed into each other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import store


@pytest.fixture()
def sess():
    df = pd.DataFrame({
        # 1001 appears three times, 1002 twice, 1003 once.
        "file_no": [1001, 1001, 1001, 1002, 1002, 1003],
        "name":    ["Ali", "Ali", "Ali", "Ayse", "Ayse", "Veli"],
        # Rows 0/1 are the same visit entered twice; row 2 is a second visit by
        # the same patient. Rows 3/4 differ only in a value nobody keys on.
        "visit":   ["2024-01-05", "2024-01-05", "2024-06-11", "2024-02-02", "2024-02-02", "2024-03-03"],
        "crp":     [2.1, 2.1, 4.8, 1.0, 9.9, 3.3],
    })
    store.save("dedup_s", df)
    return "dedup_s"


def _post(client, sess, **body):
    return client.post(f"/api/compute/{sess}/deduplicate", json=body)


def _df(sess):
    return store.get(sess)


class TestCounting:
    def test_dry_run_counts_without_deleting(self, client, sess):
        r = _post(client, sess, key_columns=["file_no"], dry_run=True)
        assert r.status_code == 200, r.text
        body = r.json()
        # 1001 x3 and 1002 x2 leave three rows to drop.
        assert body["duplicate_rows"] == 3
        assert body["remaining_rows"] == 3
        assert body["deleted"] == 0
        assert len(_df(sess)) == 6

    def test_whole_row_key_is_stricter_than_one_column(self, client, sess):
        whole = _post(client, sess, key_columns=[], dry_run=True).json()
        # Only rows 0/1 are identical across every column.
        assert whole["duplicate_rows"] == 1
        one_col = _post(client, sess, key_columns=["file_no"], dry_run=True).json()
        assert one_col["duplicate_rows"] == 3


class TestDeleting:
    def test_keep_first_leaves_the_earliest_of_each_key(self, client, sess):
        r = _post(client, sess, key_columns=["file_no"], keep="first")
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] == 3
        out = _df(sess)
        assert list(out["file_no"]) == [1001, 1002, 1003]
        # The first 1001 row is the one kept, so its crp comes along.
        assert out.loc[0, "crp"] == 2.1
        assert out.loc[1, "crp"] == 1.0

    def test_keep_last_leaves_the_latest_of_each_key(self, client, sess):
        r = _post(client, sess, key_columns=["file_no"], keep="last")
        assert r.status_code == 200, r.text
        out = _df(sess)
        assert list(out["file_no"]) == [1001, 1002, 1003]
        assert out.loc[0, "crp"] == 4.8
        assert out.loc[1, "crp"] == 9.9

    def test_index_is_reset_so_later_edits_address_the_right_row(self, client, sess):
        # Cell edits address rows by position. A gapped index after the drop
        # would send every edit below the first deletion to the wrong record.
        _post(client, sess, key_columns=["file_no"])
        assert list(_df(sess).index) == [0, 1, 2]

    def test_multi_column_key(self, client, sess):
        # Same patient, same visit — the real "entered twice" case.
        r = _post(client, sess, key_columns=["file_no", "visit"])
        assert r.json()["deleted"] == 2   # rows 1 and 4
        assert len(_df(sess)) == 4


class TestBlankKeys:
    def test_rows_with_an_all_blank_key_are_never_duplicates_of_each_other(self, client):
        # pandas treats NaN == NaN as equal in `duplicated`, so three rows with
        # no identity number would collapse into one and two patients would be
        # deleted for having a missing field.
        df = pd.DataFrame({
            "file_no": [np.nan, np.nan, np.nan, 7],
            "name": ["Ali", "Ayse", "Veli", "Can"],
        })
        store.save("dedup_blank", df)
        r = client.post("/api/compute/dedup_blank/deduplicate",
                        json={"key_columns": ["file_no"], "dry_run": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["duplicate_rows"] == 0
        assert body["blank_key_rows"] == 3

    def test_a_partly_blank_key_still_counts(self, client):
        # Only an ENTIRELY blank key is exempt: one filled field is a claim
        # about identity, and two rows making the same claim are duplicates.
        df = pd.DataFrame({
            "file_no": [np.nan, np.nan, 7],
            "name": ["Ali", "Ali", "Can"],
        })
        store.save("dedup_partial", df)
        r = client.post("/api/compute/dedup_partial/deduplicate",
                        json={"key_columns": ["file_no", "name"], "dry_run": True})
        body = r.json()
        assert body["duplicate_rows"] == 1
        assert body["blank_key_rows"] == 0


class TestRefusals:
    def test_unknown_key_column(self, client, sess):
        r = _post(client, sess, key_columns=["not_a_column"], dry_run=True)
        assert r.status_code == 422
        assert "not_a_column" in r.text

    def test_unknown_keep_mode(self, client, sess):
        r = _post(client, sess, key_columns=["file_no"], keep="middle")
        assert r.status_code == 422

    def test_missing_session(self, client):
        r = client.post("/api/compute/no_such_session/deduplicate",
                        json={"key_columns": [], "dry_run": True})
        assert r.status_code == 404
