"""Import must not delete a clinical value without saying so.

The reported case: a column at least 98% numeric had its remaining cells
rewritten to NaN, so `<0.1` -- a result, not a gap -- vanished and the count of
observed values dropped by one with nothing on screen to explain it.

Two properties are guarded here. A value at a measurement limit, or one
carrying a unit, is never silently converted away. And when a cell *is*
blanked, the report says so, counts it, shows an example, and keeps the
original text so it can be read back.
"""
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from services import store
from services.ingest_coercion import coerce_with_report

client = TestClient(app)


def _column(report: dict, name: str) -> dict:
    return next(c for c in report["columns"] if c["column"] == name)


def _upload(df: pd.DataFrame, filename: str = "data.csv"):
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return client.post("/api/upload/", files={"file": (filename, buf, "text/csv")})


# -- Measurement limits are results, not missingness ------------------------


def test_below_limit_value_is_not_deleted():
    """The exact reported case: 100 filled cells stay 100."""
    values = ["<0.1"] + [str(round(0.5 + i * 0.1, 1)) for i in range(99)]
    df = pd.DataFrame({"CRP": values})
    out, report, _ = coerce_with_report(df)

    assert out["CRP"].notna().sum() == 100
    assert out["CRP"].iloc[0] == "<0.1"
    crp = _column(report, "CRP")
    assert crp["decision"] == "kept_text"
    assert crp["censored"]["n"] == 1
    assert crp["censored"]["operators"] == {"<": 1}
    assert crp["censored"]["limits"] == [0.1]
    assert "measurement limit" in crp["reason"]


@pytest.mark.parametrize("marker,operator", [
    ("<0.1", "<"), ("> 50", ">"), ("<=2", "<="), (">= 7,5", ">="),
    ("≤ 3", "≤"), ("≥12", "≥"),
])
def test_every_censoring_operator_is_recognised(marker, operator):
    df = pd.DataFrame({"V": [marker] + [str(i) for i in range(99)]})
    out, report, _ = coerce_with_report(df)
    assert out["V"].dtype == object, "the column must not be converted"
    assert _column(report, "V")["censored"]["operators"] == {operator: 1}


def test_unit_bearing_values_are_not_stripped():
    """Mixed units in one column is a silent tenfold error; refuse to guess."""
    df = pd.DataFrame({"GLU": ["12 mg/dL", "5,4 mmol/L"] + [str(i) for i in range(98)]})
    out, report, _ = coerce_with_report(df)
    assert out["GLU"].dtype == object
    glu = _column(report, "GLU")
    assert glu["decision"] == "kept_text"
    assert glu["units"]["n"] == 2
    assert set(glu["units"]["suffixes"]) == {"mg/dL", "mmol/L"}


def test_a_date_column_is_not_mistaken_for_a_united_measurement():
    df = pd.DataFrame({"D": ["2024-01-02", "2024-03-11", "2023-12-31"]})
    _out, report, _ = coerce_with_report(df)
    d = next((c for c in report["columns"] if c["column"] == "D"), None)
    assert d is None or not d.get("units")


# -- What is blanked is reported and recoverable ----------------------------


def test_blanked_cells_are_counted_with_examples_and_originals():
    values = [str(i) for i in range(99)] + ["see notes"]
    out, report, preserved = coerce_with_report(pd.DataFrame({"X": values}))

    assert pd.api.types.is_numeric_dtype(out["X"])
    x = _column(report, "X")
    assert x["decision"] == "numeric"
    assert x["n_discarded"] == 1
    assert x["discarded_examples"] == [{"row": 99, "value": "see notes"}]
    # The original text survives the blanking.
    assert preserved["X"] == {99: "see notes"}


def test_missing_codes_are_named_and_counted_separately_from_losses():
    df = pd.DataFrame({"LDL": ["120", "NA", "98", "n/a", "?", "150"]})
    out, report, preserved = coerce_with_report(df)

    assert out["LDL"].notna().sum() == 3
    ldl = _column(report, "LDL")
    assert ldl["n_missing_coded"] == 3
    assert ldl["missing_codes"] == {"NA": 1, "n/a": 1, "?": 1}
    # A declared missing code is not a loss: nothing to recover.
    assert ldl["n_discarded"] == 0
    assert preserved == {}


def test_comma_decimals_are_reported_as_a_rewrite():
    df = pd.DataFrame({"BMI": ["25,9", "30.6", "22,1", "28"]})
    out, report, _ = coerce_with_report(df)
    assert out["BMI"].tolist() == [25.9, 30.6, 22.1, 28.0]
    bmi = _column(report, "BMI")
    assert bmi["decimal_separator"] == ","
    assert bmi["n_changed"] == 2
    assert "comma-decimal" in bmi["reason"]


def test_clean_columns_stay_out_of_the_report():
    """A report nobody reads is a report that hides the one line that matters."""
    df = pd.DataFrame({"NOTE": ["high", "low", "middling", "elevated"]})
    _out, report, _ = coerce_with_report(df)
    assert report["columns"] == []
    assert report["needs_review"] is False


def test_leading_zero_identifiers_are_still_protected():
    df = pd.DataFrame({"ID": ["0123", "0456", "0789"]})
    out, _report, _ = coerce_with_report(df)
    assert out["ID"].dtype == object


# -- The report reaches the user --------------------------------------------


def test_upload_returns_the_report():
    df = pd.DataFrame({
        "CRP": ["<0.1"] + [str(round(0.5 + i * 0.1, 1)) for i in range(29)],
        "BMI": ["25,9"] + [str(20 + i) for i in range(29)],
    })
    r = _upload(df)
    assert r.status_code == 200, r.text
    body = r.json()
    report = body["ingest_report"]
    assert report["needs_review"] is True
    assert report["n_columns_converted"] == 1          # BMI only
    assert _column(report, "CRP")["decision"] == "kept_text"
    assert _column(report, "BMI")["decision"] == "numeric"

    store.delete(body["session_id"])


def test_report_is_readable_again_after_the_panel_is_dismissed():
    df = pd.DataFrame({"X": [str(i) for i in range(99)] + ["see notes"]})
    sid = _upload(df).json()["session_id"]

    r = client.get(f"/api/upload/{sid}/ingest_report")
    assert r.status_code == 200, r.text
    body = r.json()
    assert _column(body["report"], "X")["n_discarded"] == 1
    assert body["preserved_cells"]["X"] == [{"row": 99, "value": "see notes"}]

    store.delete(sid)


def test_unknown_session_404():
    r = client.get("/api/upload/no_such_session/ingest_report")
    assert r.status_code == 404


def test_report_and_originals_are_purged_with_the_session():
    df = pd.DataFrame({"X": [str(i) for i in range(99)] + ["see notes"]})
    sid = _upload(df).json()["session_id"]
    assert store.get_preserved_cells(sid)

    store.delete(sid)
    assert store.get_ingest_report(sid) == {}
    assert store.get_preserved_cells(sid) == {}
