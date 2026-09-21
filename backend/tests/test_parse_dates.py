"""
Robust date parsing — service (date_parser) + /parse_dates endpoint.
Converts mixed-format text columns (SPSS/Excel paste) to real datetime64
in place. Ported from the drtr Excel-date tool.
"""
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store
from services.date_parser import parse_series, parse_one

client = TestClient(app)


# ── Unit: parser ──────────────────────────────────────────────────────────────

def test_parse_one_formats():
    assert parse_one("2024-03-15") == {"y": 2024, "mo": 3, "d": 15, "ambig": False}
    assert parse_one("5 Ocak 2024") == {"y": 2024, "mo": 1, "d": 5, "ambig": False}
    assert parse_one("Jan 2, 2022") == {"y": 2022, "mo": 1, "d": 2, "ambig": False}
    # Excel serial 45000 → 2023-03-15 (44927 = 2023-01-01, +73 days)
    assert parse_one("45000") == {"y": 2023, "mo": 3, "d": 15, "ambig": False}
    # 2-digit year, threshold 50 → 1999
    assert parse_one("1.1.99")["y"] == 1999
    # garbage
    assert parse_one("not a date") is None


def test_parse_series_dmy_mdy_auto():
    # 15 > 12 in one row forces DMY for the whole column.
    s = pd.Series(["03/04/2024", "15/06/2024"])
    out, stats = parse_series(s, order="auto")
    assert stats["order_used"] == "dmy"
    assert out.iloc[0] == pd.Timestamp(2024, 4, 3)   # day=3, month=4
    assert out.iloc[1] == pd.Timestamp(2024, 6, 15)
    assert stats["n_ok"] == 2


def test_parse_series_auto_detects_a_month_first_column():
    # 03/15/2024 can only be month-first, so the column is, and 04/02/2024
    # (valid both ways) must read as 2 April. The auto scan used to look for a
    # day above 12 among the ambiguous rows, where there can never be one, so
    # it always chose DMY and read this row as 4 February.
    s = pd.Series(["03/15/2024", "04/02/2024", "12/31/2023"])
    out, stats = parse_series(s, order="auto")
    assert stats["order_used"] == "mdy"
    assert out.iloc[1] == pd.Timestamp(2024, 4, 2)


def test_parse_series_mixed_and_blanks():
    s = pd.Series(["2024-12-31", "5 Ocak 2024", "", None, "garbage"])
    out, stats = parse_series(s, order="auto")
    assert stats == {"n_total": 5, "n_ok": 2, "n_bad": 1, "n_empty": 2, "order_used": "dmy"}
    assert str(out.dtype) == "datetime64[ns]"


# ── Endpoint ──────────────────────────────────────────────────────────────────

def _seed(df, sid):
    store.save(sid, df)
    return sid


def test_parse_dates_apply_in_place():
    sid = _seed(pd.DataFrame({"dt": ["15/03/2024", "2024-12-31", "5 Ocak 2024", "45000"]}), "pd_apply")
    r = client.post(f"/api/compute/{sid}/parse_dates", json={"column": "dt", "order": "auto"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["kind"] == "date"
    assert d["stats"]["n_ok"] == 4
    df = store.get(sid)
    assert str(df["dt"].dtype) == "datetime64[ns]"
    assert df["dt"].iloc[0] == pd.Timestamp(2024, 3, 15)
    assert df["dt"].iloc[3] == pd.Timestamp(2023, 3, 15)
    # preview values are ISO strings for the grid
    assert d["preview_values"][1] == "2024-12-31"


def test_parse_dates_preview_only_does_not_mutate():
    sid = _seed(pd.DataFrame({"dt": ["1.1.2020", "2.2.2021"]}), "pd_prev")
    r = client.post(f"/api/compute/{sid}/parse_dates", json={"column": "dt", "preview_only": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "sample" in d and d["stats"]["n_ok"] == 2
    # untouched — still text
    assert str(store.get(sid)["dt"].dtype) == "object"


def test_parse_dates_errors():
    sid = _seed(pd.DataFrame({"x": [1, 2]}), "pd_err")
    assert client.post(f"/api/compute/{sid}/parse_dates", json={"column": "missing"}).status_code == 404


def test_detect_kind_labels_text_dates_not_numbers():
    from routers.upload import _detect_kind
    # Month-name dates → date label (was previously misclassified).
    assert _detect_kind(pd.Series(["5 Ocak 2024", "12 March 2023", "1 Şub 2022"])) == "date"
    # Numeric-looking text (serials / IDs) must NOT become date on import.
    assert _detect_kind(pd.Series(["45000", "45001", "45002"])) != "date"
