"""Lipid calculators: non-HDL-C, TG/HDL-C, AIP and CMI.

Every one of them has a scale that its cut-offs depend on, so the unit
handling is tested as heavily as the arithmetic.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from services import store

CHOL = 38.67   # mg/dL per mmol/L, cholesterol
TG = 88.57     # mg/dL per mmol/L, triglyceride


@pytest.fixture()
def sid(client):
    store.save("lipids", pd.DataFrame({
        "TG":     [150.0, 300.0, 90.0],
        "HDL":    [50.0, 30.0, 60.0],
        "TotChol": [200.0, 260.0, 170.0],
        "Bel":    [90.0, 110.0, 75.0],
        "Boy":    [170.0, 165.0, 180.0],
    }))
    return "lipids"


@pytest.fixture()
def sid_mmol(client):
    """The same three patients, reported in mmol/L."""
    store.save("lipids_mmol", pd.DataFrame({
        "TG":      [150.0 / TG, 300.0 / TG, 90.0 / TG],
        "HDL":     [50.0 / CHOL, 30.0 / CHOL, 60.0 / CHOL],
        "TotChol": [200.0 / CHOL, 260.0 / CHOL, 170.0 / CHOL],
        "Bel":     [90.0, 110.0, 75.0],
        "Boy":     [170.0, 165.0, 180.0],
    }))
    return "lipids_mmol"


MAP = {"tg": "TG", "hdl": "HDL", "total_cholesterol": "TotChol",
       "waist": "Bel", "height": "Boy"}


def _run(client, sid, calc, column_map=None, **extra):
    return client.post(f"/api/compute/{sid}/clinical/{calc}",
                       json={"column_map": column_map or MAP, **extra})


# ── non-HDL ──────────────────────────────────────────────────────────────────

def test_non_hdl_is_total_minus_hdl(client, sid):
    assert _run(client, sid, "non_hdl").status_code == 200
    assert store.get(sid)["NonHDL"].tolist() == [150.0, 230.0, 110.0]


def test_non_hdl_bands(client, sid):
    body = _run(client, sid, "non_hdl").json()
    assert body["n_optimal"] == 1          # 110
    assert body["n_above_optimal"] == 1     # 150
    assert body["n_very_high"] == 1         # 230


def test_non_hdl_negative_difference_is_missing(client):
    store.save("bad", pd.DataFrame({"TotChol": [40.0], "HDL": [60.0]}))
    r = client.post("/api/compute/bad/clinical/non_hdl",
                    json={"column_map": {"total_cholesterol": "TotChol", "hdl": "HDL"}})
    assert r.status_code == 200
    assert store.get("bad")["NonHDL"].isna().all()


def test_non_hdl_converts_mmol_inputs(client, sid_mmol):
    body = _run(client, sid_mmol, "non_hdl").json()
    assert body["units"] == "mmol/L"
    got = store.get(sid_mmol)["NonHDL"].tolist()
    assert got == pytest.approx([150.0, 230.0, 110.0], abs=0.1)


# ── TG/HDL ───────────────────────────────────────────────────────────────────

def test_tg_hdl_ratio_on_mgdl(client, sid):
    assert _run(client, sid, "tg_hdl_ratio").status_code == 200
    assert store.get(sid)["TG_HDL"].tolist() == [3.0, 10.0, 1.5]


def test_tg_hdl_cut_off_is_inclusive(client, sid):
    body = _run(client, sid, "tg_hdl_ratio").json()
    assert body["n_elevated"] == 2   # 3.0 counts as elevated
    assert body["n_normal"] == 1


def test_tg_hdl_is_the_mgdl_ratio_whatever_the_input_scale(client, sid_mmol):
    """The 3.0 cut-off is a mg/dL one, so mmol/L input must not shift it."""
    body = _run(client, sid_mmol, "tg_hdl_ratio").json()
    assert body["units"] == "mmol/L"
    assert store.get(sid_mmol)["TG_HDL"].tolist() == pytest.approx([3.0, 10.0, 1.5], abs=0.01)


def test_zero_hdl_is_missing_not_infinite(client):
    store.save("z", pd.DataFrame({"TG": [150.0], "HDL": [0.0]}))
    r = client.post("/api/compute/z/clinical/tg_hdl_ratio",
                    json={"column_map": {"tg": "TG", "hdl": "HDL"}})
    assert r.status_code == 200
    assert store.get("z")["TG_HDL"].isna().all()


def test_declared_units_beat_detection(client, sid):
    """mg/dL values declared as mmol/L are taken at their word."""
    body = _run(client, sid, "tg_hdl_ratio", units="mmol/L").json()
    assert body["units"] == "mmol/L"
    # 150 mmol/L TG and 50 mmol/L HDL -> ratio scaled by 88.57/38.67
    assert store.get(sid)["TG_HDL"][0] == pytest.approx(3.0 * TG / CHOL, abs=0.01)


# ── TC/HDL ───────────────────────────────────────────────────────────────────

def test_tc_hdl_ratio(client, sid):
    assert _run(client, sid, "tc_hdl_ratio").status_code == 200
    assert store.get(sid)["TC_HDL"].tolist() == [4.0, 8.67, 2.83]


def test_tc_hdl_is_scale_free(client, sid, sid_mmol):
    """Both terms are cholesterol, so the conversion factor cancels."""
    _run(client, sid, "tc_hdl_ratio")
    _run(client, sid_mmol, "tc_hdl_ratio")
    assert store.get(sid_mmol)["TC_HDL"].tolist() == pytest.approx(
        store.get(sid)["TC_HDL"].tolist(), abs=0.01)


def test_tc_hdl_thresholds_overlap_as_documented(client, sid):
    body = _run(client, sid, "tc_hdl_ratio").json()
    assert body["n_optimal"] == 1                 # 2.83
    assert body["n_above_male_threshold"] == 1    # 8.67
    assert body["n_above_female_threshold"] == 1  # 8.67 again, by design


# ── AIP ──────────────────────────────────────────────────────────────────────

def test_aip_is_the_molar_log_ratio(client, sid):
    assert _run(client, sid, "aip").status_code == 200
    expected = [round(math.log10((v / TG) / (h / CHOL)), 3)
                for v, h in [(150, 50), (300, 30), (90, 60)]]
    assert store.get(sid)["AIP"].tolist() == expected


def test_aip_does_not_use_the_mgdl_ratio(client, sid):
    """The mg/dL ratio would shift every value by log10(88.57/38.67) = 0.36."""
    aip = store.get(sid)["AIP"] if _run(client, sid, "aip").status_code else None
    naive = math.log10(150 / 50)
    assert abs(float(aip[0]) - naive) == pytest.approx(math.log10(TG / CHOL), abs=1e-3)


def test_aip_agrees_across_input_scales(client, sid, sid_mmol):
    _run(client, sid, "aip")
    _run(client, sid_mmol, "aip")
    assert store.get(sid_mmol)["AIP"].tolist() == pytest.approx(
        store.get(sid)["AIP"].tolist(), abs=0.001)


def test_calibrated_column_is_exactly_aip_plus_two(client, sid):
    body = _run(client, sid, "aip").json()
    assert body["calibrated_column"] == "AIP_100"
    df = store.get(sid)
    assert (df["AIP_100"] - df["AIP"]).round(6).eq(2.0).all()


def test_calibrated_column_travels_with_the_response(client, sid):
    """The client adds columns from the response, so a second one must be in it."""
    body = _run(client, sid, "aip").json()
    extra = body["extra_columns"]
    assert [c["name"] for c in extra] == ["AIP_100"]
    assert extra[0]["preview_values"] == [round(v + 2, 3) for v in body["preview_values"]]


def test_calibrated_column_has_no_negative_values_here(client, sid):
    _run(client, sid, "aip")
    assert (store.get(sid)["AIP_100"] > 0).all()


def test_aip_risk_bands(client, sid):
    body = _run(client, sid, "aip").json()
    assert body["n_low"] + body["n_intermediate"] + body["n_high"] == 3


def test_aip_zero_tg_is_missing_not_minus_infinity(client):
    store.save("zt", pd.DataFrame({"TG": [0.0], "HDL": [50.0]}))
    r = client.post("/api/compute/zt/clinical/aip",
                    json={"column_map": {"tg": "TG", "hdl": "HDL"}})
    assert r.status_code == 200
    assert store.get("zt")["AIP"].isna().all()


# ── CMI ──────────────────────────────────────────────────────────────────────

def test_cmi_is_molar_ratio_times_whtr(client, sid):
    assert _run(client, sid, "cmi").status_code == 200
    expected = [round(((v / TG) / (h / CHOL)) * (w / ht), 3)
                for v, h, w, ht in [(150, 50, 90, 170), (300, 30, 110, 165), (90, 60, 75, 180)]]
    assert store.get(sid)["CMI"].tolist() == expected


def test_cmi_waist_and_height_unit_cancels(client):
    store.save("m", pd.DataFrame({"TG": [150.0], "HDL": [50.0],
                                  "Bel": [0.9], "Boy": [1.7]}))
    r = client.post("/api/compute/m/clinical/cmi",
                    json={"column_map": {"tg": "TG", "hdl": "HDL",
                                         "waist": "Bel", "height": "Boy"},
                          "units": "mg/dL"})
    assert r.status_code == 200
    expected = round(((150 / TG) / (50 / CHOL)) * (0.9 / 1.7), 3)
    assert store.get("m")["CMI"].tolist() == [expected]


def test_cmi_reports_no_cut_off_bands(client, sid):
    body = _run(client, sid, "cmi").json()
    assert "n_high" not in body
    assert "sex-specific" in body["result_text"]


def test_cmi_zero_height_is_missing(client):
    store.save("h0", pd.DataFrame({"TG": [150.0], "HDL": [50.0],
                                   "Bel": [90.0], "Boy": [0.0]}))
    r = client.post("/api/compute/h0/clinical/cmi",
                    json={"column_map": {"tg": "TG", "hdl": "HDL",
                                         "waist": "Bel", "height": "Boy"}})
    assert r.status_code == 200
    assert store.get("h0")["CMI"].isna().all()


# ── shared ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("calc", ["non_hdl", "tg_hdl_ratio", "tc_hdl_ratio", "aip", "cmi"])
def test_missing_mapping_is_422(client, sid, calc):
    assert _run(client, sid, calc, column_map={"hdl": "HDL"}).status_code == 422
