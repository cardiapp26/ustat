"""User-declared missing codes (99, 999, "don't know"), end to end.

The codes stay in the stored data and every analysis reads them as missing;
see services/missing_codes.py. Before this, 99 and 999 in a clinical file
were numbers in every mean and model unless the file was SPSS, and even then
the original codes were thrown away on import.
"""
import io
import os
import tempfile

import numpy as np
import pandas as pd
import pyreadstat
import pytest
from conftest import make_session

from services import store
from services.missing_codes import apply_missing_codes, suggest_missing_codes
from services.project_file import build_project, parse_project, restore_project

AGES = [34, 41, 52, 58, 63, 70, 999, 47, 999, 66]
REAL_AGES = [a for a in AGES if a != 999]


def _declare(client, sid, column, codes):
    r = client.post(f"/api/sessions/{sid}/metadata", json={"columns": {column: {"missing_codes": codes}}})
    assert r.status_code == 200, r.text


def test_declared_code_is_missing_in_analyses_but_kept_in_the_data(client):
    sid = make_session(pd.DataFrame({"age": AGES}), "mc_basic")
    before = client.get(f"/api/stats/{sid}/descriptive", params={"column": "age"}).json()["age"]
    assert before["max"] == 999  # the reported bug: 999 is an age

    _declare(client, sid, "age", ["999"])
    after = client.get(f"/api/stats/{sid}/descriptive", params={"column": "age"}).json()["age"]
    assert after["n"] == len(REAL_AGES)
    assert after["mean"] == pytest.approx(np.mean(REAL_AGES))
    assert after["max"] == max(REAL_AGES)
    # The file itself is untouched: the grid and exports still show 999.
    assert (store.get(sid)["age"] == 999).sum() == 2


def test_case_filter_does_not_select_a_missing_code(client):
    """Codes are applied before the filter, as in SPSS: age > 50 is not 999."""
    df = pd.DataFrame({"age": AGES, "y": range(len(AGES))})
    sid = make_session(df, "mc_filter")
    _declare(client, sid, "age", [999])
    store.save_filter(sid, [{"column": "age", "operator": "gt", "value": 50}])
    assert sorted(store.get_filtered(sid)["age"].tolist()) == sorted(a for a in REAL_AGES if a > 50)
    # And "is missing" selects the coded rows, like SPSS's MISSING().
    store.save_filter(sid, [{"column": "age", "operator": "missing"}])
    assert store.get_filtered(sid)["y"].tolist() == [6, 8]


def test_text_code_in_a_numeric_column_leaves_numbers_behind():
    df = pd.DataFrame({"sbp": ["120", "UNK", "135", "142", "UNK"]})
    out = apply_missing_codes(df, {"sbp": {"missing_codes": ["UNK"]}})
    assert pd.api.types.is_numeric_dtype(out["sbp"])
    assert out["sbp"].isna().sum() == 2
    assert df["sbp"].tolist()[1] == "UNK"  # the input is not mutated


def test_codes_match_whatever_spelling_the_data_uses():
    df = pd.DataFrame({"grade": [1.0, 2.0, 9.0, 3.0]})
    out = apply_missing_codes(df, {"grade": {"missing_codes": ["9"]}})
    assert out["grade"].isna().tolist() == [False, False, True, False]


def test_no_codes_means_no_copy():
    df = pd.DataFrame({"a": [1, 2]})
    assert apply_missing_codes(df, {"a": {"label": "A"}}) is df


# ── Suggestions ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("values, expected", [
    (AGES, ["999"]),                                   # 999 far above ages up to 70
    ([60, 72, 88, 95, 98, 99, 64], []),               # 99 could be a real age here
    ([72, 88, 99, 104, 99, 130, 160], []),             # heart rate 99 is ordinary
    ([1, 2, 3, 4, 5, 8, 9, 3, 2, 9], ["8", "9"]),      # Likert: don't know / no answer
    ([21.5, 24.0, 31.2, 99, 27.8], ["99"]),            # BMI 99
    ([3.1, 2.2, -99, 4.0], ["-99"]),
])
def test_suggestions(values, expected):
    got = [s["value"] for s in suggest_missing_codes(pd.Series(values))]
    assert got == expected


def test_overview_suggests_until_declared_then_counts(client):
    df = pd.DataFrame({"age": AGES, "likert": [1, 2, 3, 9, 5, 4, 2, 9, 1, 3]})
    sid = make_session(df, "mc_overview")
    first = client.get(f"/api/sessions/{sid}/missing_codes").json()
    assert first["suggestions"]["age"][0] == {
        "value": "999", "count": 2, "reason": "999 is far above every other value (largest 70)",
    }
    assert [s["value"] for s in first["suggestions"]["likert"]] == ["9"]
    assert first["counts"] == {}
    # Suggesting is all it does.
    assert store.get_filtered(sid)["age"].max() == 999

    _declare(client, sid, "age", ["999"])
    second = client.get(f"/api/sessions/{sid}/missing_codes").json()
    assert "age" not in second["suggestions"]
    assert second["counts"] == {"age": 2}


# ── Round trips ──────────────────────────────────────────────────────────────

def _read_sav(content: bytes):
    fd, path = tempfile.mkstemp(suffix=".sav")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(content)
        return pyreadstat.read_sav(path, user_missing=True)
    finally:
        os.unlink(path)


def test_sav_export_declares_the_codes_user_missing(client):
    sid = make_session(pd.DataFrame({"age": AGES}), "mc_sav")
    _declare(client, sid, "age", ["999"])
    r = client.get(f"/api/sessions/{sid}/export", params={"fmt": "sav", "filename": "mc"})
    assert r.status_code == 200, r.text
    df, meta = _read_sav(r.content)
    assert (df["age"] == 999).sum() == 2  # the code is in the file...
    assert meta.missing_ranges["age"] == [{"lo": 999.0, "hi": 999.0}]  # ...declared missing


def test_sav_export_blanks_codes_spss_cannot_declare(client):
    """SPSS holds at most three discrete user-missing values; a fourth would
    be read as a valid value, so it is written system-missing instead."""
    df = pd.DataFrame({"x": [1, 2, 3, 97, 98, 99, 999]})
    sid = make_session(df, "mc_sav_overflow")
    _declare(client, sid, "x", [97, 98, 99, 999])
    r = client.get(f"/api/sessions/{sid}/export", params={"fmt": "sav", "filename": "mc"})
    assert r.status_code == 200, r.text
    out, meta = _read_sav(r.content)
    assert [m["lo"] for m in meta.missing_ranges["x"]] == [97.0, 98.0, 99.0]
    assert out["x"].isna().sum() == 1 and 999 not in out["x"].tolist()


def test_project_file_keeps_the_codes():
    sid = make_session(pd.DataFrame({"age": AGES}), "mc_project")
    store.save_metadata(sid, {"age": {"missing_codes": ["999"]}})
    restored = restore_project(parse_project(build_project(sid)))
    assert store.get_metadata(restored)["age"]["missing_codes"] == ["999"]
    assert store.get_filtered(restored)["age"].max() == max(REAL_AGES)


def test_spss_user_missing_round_trips_as_codes(client):
    """An SPSS file's user-missing codes survive import and re-export."""
    fd, path = tempfile.mkstemp(suffix=".sav")
    os.close(fd)
    try:
        pyreadstat.write_sav(pd.DataFrame({"q": [1.0, 2.0, 8.0, 9.0]}), path, missing_ranges={"q": [8, 9]})
        with open(path, "rb") as f:
            content = f.read()
    finally:
        os.unlink(path)
    sid = client.post("/api/upload/", files={"file": ("s.sav", io.BytesIO(content), "application/octet-stream")}).json()["session_id"]
    assert store.get(sid)["q"].tolist() == [1.0, 2.0, 8.0, 9.0]
    assert store.get_filtered(sid)["q"].isna().sum() == 2
    out, meta = _read_sav(client.get(f"/api/sessions/{sid}/export", params={"fmt": "sav"}).content)
    assert out["q"].tolist() == [1.0, 2.0, 8.0, 9.0]
    assert [m["lo"] for m in meta.missing_ranges["q"]] == [8.0, 9.0]


def test_session_columns_carry_the_codes(client):
    """The Data Dictionary reads them back after a reload."""
    sid = make_session(pd.DataFrame({"age": AGES}), "mc_columns")
    _declare(client, sid, "age", ["999"])
    cols = client.get(f"/api/sessions/{sid}").json()["columns"]
    assert next(c for c in cols if c["name"] == "age")["missing_codes"] == ["999"]
