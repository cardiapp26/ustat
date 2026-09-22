"""Grid sort/filter/missing-only over the full dataframe.

The grid's own client-side sort/filter/missing-only only ever see `preview`,
which is capped at PREVIEW_ROWS (2000) rows -- for a file bigger than that
they silently only look at the file's head. `/grid_rows` mirrors the same
predicate/sort semantics but runs them over the whole in-memory dataframe.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session

PREVIEW_ROWS = 2000


@pytest.fixture()
def wide_df() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    n = 7000  # deliberately past the 2000-row preview cap, and large enough
    # that a single site value alone exceeds PREVIEW_ROWS
    age = rng.integers(18, 90, n).astype(float)
    site = rng.choice(["A", "B", "C"], n)
    note = np.array(["ordinary"] * n, dtype=object)
    # The one thing a preview-only filter/sort can never see: signal placed
    # entirely beyond row 2000.
    note[2500] = "needle-in-haystack"
    age[3000:3005] = np.nan
    age[4999] = 999.0  # the true max, past the preview cap
    return pd.DataFrame({"age": age, "site": site, "note": note})


def _grid_rows(client, sid, **body):
    r = client.post(f"/api/sessions/{sid}/grid_rows", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_filter_finds_a_match_past_the_preview_cap(client, wide_df):
    sid = make_session(wide_df, "grid_filter")
    d = _grid_rows(client, sid, filters={"note": "needle"})
    assert d["matched_total"] == 1
    assert d["positions"] == [2500]
    assert d["rows"][0]["note"] == "needle-in-haystack"


def test_sort_desc_surfaces_the_true_max_past_the_preview_cap(client, wide_df):
    sid = make_session(wide_df, "grid_sort")
    d = _grid_rows(client, sid, sort=[{"col": "age", "dir": "desc"}])
    assert d["positions"][0] == 4999
    assert d["rows"][0]["age"] == 999.0


def test_missing_only_reaches_rows_past_the_preview_cap(client, wide_df):
    sid = make_session(wide_df, "grid_missing")
    d = _grid_rows(client, sid, missing_only=True)
    assert d["matched_total"] == 5
    assert sorted(d["positions"]) == [3000, 3001, 3002, 3003, 3004]


def test_matched_total_is_the_full_count_even_when_the_page_is_capped(client, wide_df):
    sid = make_session(wide_df, "grid_page")
    d = _grid_rows(client, sid, filters={"site": "A"})
    full_a = int((wide_df["site"] == "A").sum())
    assert full_a > PREVIEW_ROWS
    assert d["matched_total"] == full_a
    assert len(d["rows"]) == PREVIEW_ROWS
    assert d["truncated"] is True


def test_sort_na_sinks_to_the_end_regardless_of_direction(client):
    # A small frame, fully returned in one page, so the tail of the result
    # is actually the tail of the sort -- not just cut off by the page cap.
    small = pd.DataFrame({"age": [30.0, np.nan, 10.0, np.nan, 20.0]})
    sid = make_session(small, "grid_na_sink")
    asc = _grid_rows(client, sid, sort=[{"col": "age", "dir": "asc"}])
    desc = _grid_rows(client, sid, sort=[{"col": "age", "dir": "desc"}])
    assert asc["positions"] == [2, 4, 0, 1, 3]
    assert desc["positions"] == [0, 4, 2, 1, 3]


def test_position_addresses_the_correct_row_through_cell_edit(client, wide_df):
    """A position returned for a row past the preview cap must be a valid
    row_index for PATCH /cell -- otherwise sorting/filtering a large file
    would let edits land on the wrong record."""
    sid = make_session(wide_df, "grid_edit_roundtrip")
    d = _grid_rows(client, sid, filters={"note": "needle"})
    pos = d["positions"][0]
    assert pos == 2500
    r = client.patch(f"/api/sessions/{sid}/cell", json={"row_index": pos, "column": "note", "value": "edited"})
    assert r.status_code == 200, r.text
    d2 = _grid_rows(client, sid, filters={"note": "edited"})
    assert d2["positions"] == [2500]


def test_excluded_rows_sink_to_the_bottom_when_sorting(client, wide_df):
    sid = make_session(wide_df, "grid_excluded_sink")
    r = client.post(f"/api/sessions/{sid}/select_cases", json={
        "conditions": [{"column": "site", "operator": "eq", "value": "A"}],
        "apply": True,
    })
    assert r.status_code == 200, r.text

    d = _grid_rows(client, sid, sort=[{"col": "age", "dir": "asc"}])
    positions = d["positions"]
    is_a = (wide_df["site"].to_numpy() == "A")
    included_flags = [bool(is_a[p]) for p in positions]
    # Once a False (excluded) shows up, every row after it must also be
    # excluded -- the selected block is contiguous and comes first.
    first_excluded = next((i for i, ok in enumerate(included_flags) if not ok), None)
    if first_excluded is not None:
        assert all(not ok for ok in included_flags[first_excluded:])


def test_date_column_sorts_chronologically_not_lexically(client):
    # 05/03/2024 (5 March) sorts before 25/01/2024 (25 January) lexically
    # ("05" < "25"), but chronologically 25/01 comes first. day>12 makes
    # every value unambiguously day-first, and _detect_kind picks this up
    # as a date column so the server takes the parse_series path.
    dates = pd.DataFrame({"visit": ["05/03/2024", "25/01/2024", "10/02/2024", None]})
    sid = make_session(dates, "grid_date_sort")
    d = _grid_rows(client, sid, sort=[{"col": "visit", "dir": "asc"}])
    assert d["positions"] == [1, 2, 0, 3]  # Jan, Feb, Mar, then the blank


def test_unknown_session_returns_404(client):
    r = client.post("/api/sessions/does-not-exist/grid_rows", json={})
    assert r.status_code == 404
