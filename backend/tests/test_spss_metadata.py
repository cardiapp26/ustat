import io
import os
import tempfile

import pandas as pd
import pyreadstat

from services import store


def _sav_bytes() -> bytes:
    fd, path = tempfile.mkstemp(suffix=".sav")
    os.close(fd)
    try:
        df = pd.DataFrame(
            {
                "Grup": [0, 1, 9],
                "Age": [45, 50, -99],
            }
        )
        pyreadstat.write_sav(
            df,
            path,
            column_labels={
                "Grup": "Patient control group",
                "Age": "Age in years",
            },
            variable_value_labels={
                "Grup": {0: "Hasta", 1: "Kontrol", 9: "Cevapsiz"},
            },
            missing_ranges={
                "Grup": [9],
                "Age": [-99],
            },
            variable_measure={
                "Grup": "nominal",
                "Age": "scale",
            },
        )
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _upload_sav(client) -> dict:
    files = {"file": ("Chadvasc.sav", io.BytesIO(_sav_bytes()), "application/octet-stream")}
    response = client.post("/api/upload/", files=files)
    assert response.status_code == 200, response.text
    return response.json()


def test_spss_upload_imports_dictionary_metadata(client):
    body = _upload_sav(client)
    sid = body["session_id"]
    by_name = {col["name"]: col for col in body["columns"]}

    assert by_name["Grup"]["kind"] == "categorical"
    assert by_name["Grup"]["label"] == "Patient control group"
    assert by_name["Grup"]["value_labels"] == {"0": "Hasta", "1": "Kontrol", "9": "Cevapsiz"}
    assert by_name["Grup"]["missing_ranges"] == [{"lo": 9, "hi": 9}]
    assert by_name["Age"]["kind"] == "numeric"
    assert by_name["Age"]["label"] == "Age in years"
    assert by_name["Age"]["missing_ranges"] == [{"lo": -99, "hi": -99}]

    assert body["preview"][2]["Grup"] is None
    assert body["preview"][2]["Age"] is None

    stored = store.get_metadata(sid)
    assert stored["Grup"]["value_labels"]["0"] == "Hasta"
    assert stored["Grup"]["measure"] == "nominal"

    refreshed = client.get(f"/api/sessions/{sid}")
    assert refreshed.status_code == 200, refreshed.text
    refreshed_grup = next(col for col in refreshed.json()["columns"] if col["name"] == "Grup")
    assert refreshed_grup["label"] == "Patient control group"
    assert refreshed_grup["value_labels"]["1"] == "Kontrol"


def test_spss_export_writes_dictionary_metadata(client):
    body = _upload_sav(client)
    sid = body["session_id"]

    response = client.get(f"/api/sessions/{sid}/export", params={"fmt": "sav", "filename": "roundtrip"})
    assert response.status_code == 200, response.text

    fd, path = tempfile.mkstemp(suffix=".sav")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(response.content)
        _, meta = pyreadstat.read_sav(path, metadataonly=True, user_missing=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    assert meta.column_names_to_labels["Grup"] == "Patient control group"
    assert meta.column_names_to_labels["Age"] == "Age in years"
    assert meta.variable_value_labels["Grup"] == {0.0: "Hasta", 1.0: "Kontrol", 9.0: "Cevapsiz"}
    assert meta.missing_ranges["Grup"] == [{"lo": 9.0, "hi": 9.0}]
    assert meta.missing_ranges["Age"] == [{"lo": -99.0, "hi": -99.0}]

    assert meta.variable_measure["Grup"] == "nominal"
    assert meta.variable_measure["Age"] == "scale"


def test_spss_export_renames_columns_that_are_spss_keywords(client):
    """readstat refuses the whole file over one reserved name, so a column
    called OR or TO used to cost the user every other column with it."""
    df = pd.DataFrame({"OR": [1, 2], "to": [3, 4], "With": [5, 6], "keep": [7, 8]})
    store.save("sav_reserved", df)

    response = client.get(
        "/api/sessions/sav_reserved/export", params={"fmt": "sav", "filename": "reserved"}
    )
    assert response.status_code == 200, response.text

    fd, path = tempfile.mkstemp(suffix=".sav")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(response.content)
        df_out, meta = pyreadstat.read_sav(path, metadataonly=True, user_missing=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    names = list(df_out.columns)
    assert names == ["OR_", "to_", "With_", "keep"]
    # The name the user typed survives as the variable label.
    assert meta.column_names_to_labels["OR_"] == "OR"
    assert meta.column_names_to_labels["to_"] == "to"
    assert "keep" not in meta.column_names_to_labels or not meta.column_names_to_labels["keep"]


def test_spss_export_sanitizes_invalid_variable_names(client):
    """Columns with spaces, long names, or leading digits must not crash SAV export."""
    df = pd.DataFrame(
        {
            "x y": [1, 2, 3],
            "a" * 70: [4, 5, 6],
            "1x": [7, 8, 9],
            "yaş aralığı": [10, 11, 12],
            "$cost": [13, 14, 15],
            "#tag": [16, 17, 18],
        }
    )
    store.save("sav_sanitize", df)

    response = client.get("/api/sessions/sav_sanitize/export", params={"fmt": "sav", "filename": "sanitized"})
    assert response.status_code == 200, response.text

    fd, path = tempfile.mkstemp(suffix=".sav")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(response.content)
        df_out, meta = pyreadstat.read_sav(path, metadataonly=True, user_missing=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    names = list(df_out.columns)
    assert "x_y" in names
    assert "v1x" in names
    # SPSS documents @, # and $ as legal openers; readstat rejects all three.
    assert "v$cost" in names
    assert "v#tag" in names
    assert all(len(n) <= 64 for n in names)
    # Original names survive as column labels
    assert meta.column_names_to_labels["x_y"] == "x y"
    assert meta.column_names_to_labels["v1x"] == "1x"
    assert meta.column_names_to_labels["ya__aral"] == "yaş aralığı"
