"""Choosing the same column as row and column variable is a 400, not a 500.

`df[[c, c]]` returns a frame with duplicate column labels, so the cleaning
step used to fail with an AttributeError deep inside the crosstab helper.
"""
import pandas as pd
import pytest

from conftest import make_session


@pytest.mark.parametrize("path", ["/api/stats/chisquare", "/api/stats/fisher"])
def test_same_row_and_column_is_rejected_with_400(client, path):
    df = pd.DataFrame({"a": ["x", "y"] * 10, "b": ["u", "v", "v", "u"] * 5})
    sid = make_session(df, "same_col_session")
    r = client.post(path, json={"session_id": sid, "row_column": "a", "col_column": "a"})
    assert r.status_code == 400
    assert "different" in r.json()["detail"].lower()


def test_distinct_columns_still_work(client):
    df = pd.DataFrame({"a": ["x", "y"] * 10, "b": ["u", "v", "v", "u"] * 5})
    sid = make_session(df, "distinct_col_session")
    r = client.post("/api/stats/chisquare", json={"session_id": sid, "row_column": "a", "col_column": "b"})
    assert r.status_code == 200
