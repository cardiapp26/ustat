"""Per-column header badges, computed over the whole dataframe.

The grid's `preview` is capped at 2000 rows by the upload endpoint, so a
badge counted there describes the top of the file while reading as a fact
about the column. These numbers come from the full frame instead, which is
the point of the endpoint.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def wide_session() -> str:
    rng = np.random.default_rng(19)
    n = 5000  # deliberately past the 2000-row preview cap
    age = rng.integers(18, 90, n).astype(float)
    age[:100] = np.nan
    df = pd.DataFrame(
        {
            "age": age,
            "site": rng.choice(["A", "B", "C"], n),
            "score": rng.normal(50, 10, n),
        }
    )
    return make_session(df, "badges_wide")


def _badges(client, sid):
    r = client.get(f"/api/stats/{sid}/column_badges")
    assert r.status_code == 200, r.text
    return r.json()


def test_counts_come_from_the_whole_frame_not_the_preview(client, wide_session):
    d = _badges(client, wide_session)
    assert d["n_rows"] == 5000
    # All 100 blanks sit in the first 2000 rows, so a preview-based count
    # would agree on the count but not on the denominator.
    assert d["columns"]["age"]["n_missing"] == 100
    assert d["columns"]["age"]["pct_missing"] == pytest.approx(2.0)


def test_range_is_reported_for_numeric_columns(client, wide_session):
    d = _badges(client, wide_session)["columns"]
    assert d["age"]["min"] >= 18
    assert d["age"]["max"] <= 89
    assert d["age"]["n_valid"] == 4900


def test_no_range_for_a_categorical_column(client, wide_session):
    """min/max of a label is alphabetical order, not a fact about the data."""
    entry = _badges(client, wide_session)["columns"]["site"]
    assert "min" not in entry and "max" not in entry
    assert entry["n_missing"] == 0


def test_sentinels_are_excluded_from_the_range(client):
    # 999 is the classic "unknown" placeholder. Counting it as the maximum
    # heart rate would be worse than showing nothing.
    df = pd.DataFrame({"age": [30.0, 44.0, 51.0, 999.0]})
    sid = make_session(df, "badges_sentinel")
    entry = _badges(client, sid)["columns"]["age"]
    assert entry["max"] == 51.0
    # Counted as out-of-range, NOT as missing: the cell is not blank, and a
    # glucose of 348 flagged by the same rule is a real reading. The badge
    # says "check this", not "this is absent".
    assert entry["n_missing"] == 0
    assert entry["n_implausible"] == 1
    assert entry["n_valid"] == 3


def test_all_missing_column_gets_no_range(client):
    df = pd.DataFrame({"empty": [np.nan, np.nan, np.nan]})
    sid = make_session(df, "badges_empty")
    entry = _badges(client, sid)["columns"]["empty"]
    assert entry["n_missing"] == 3
    assert entry["pct_missing"] == pytest.approx(100.0)
    assert "min" not in entry


def test_infinities_do_not_become_the_range(client):
    df = pd.DataFrame({"x": [1.0, 2.0, np.inf, -np.inf]})
    sid = make_session(df, "badges_inf")
    entry = _badges(client, sid)["columns"]["x"]
    assert entry["min"] == 1.0
    assert entry["max"] == 2.0


def test_unknown_session_returns_empty_rather_than_500(client):
    r = client.get("/api/stats/no_such_session/column_badges")
    assert r.status_code == 200
    assert r.json() == {"n_rows": 0, "columns": {}}
