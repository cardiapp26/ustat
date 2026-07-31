"""FIB-4 index: (age x AST) / (platelets x sqrt(ALT)), Sterling 2006."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from services import store


@pytest.fixture()
def sid(client):
    store.save("liver", pd.DataFrame({
        "Yas":       [60, 45, 72],
        "AST":       [40, 80, 25],
        "ALT":       [50, 30, 60],
        "Trombosit": [200, 150, 95],
    }))
    return "liver"


MAP = {"age": "Yas", "ast": "AST", "alt": "ALT", "platelets": "Trombosit"}


def _run(client, sid, column_map=None, **extra):
    r = client.post(f"/api/compute/{sid}/clinical/fib4",
                    json={"column_map": column_map or MAP, **extra})
    return r


def test_matches_the_published_worked_example(client, sid):
    """Age 60, AST 40, ALT 50, platelets 200 -> 1.70."""
    assert _run(client, sid).status_code == 200
    values = store.get(sid)["FIB4"].tolist()
    expected = [
        round(60 * 40 / (200 * math.sqrt(50)), 2),
        round(45 * 80 / (150 * math.sqrt(30)), 2),
        round(72 * 25 / (95 * math.sqrt(60)), 2),
    ]
    assert values == expected
    assert values[0] == 1.70


def test_reports_the_risk_bands(client, sid):
    j = _run(client, sid).json()
    # 1.70 and 2.45 are indeterminate, 4.38 is above the 2.67 cut-off.
    assert (j["n_low"], j["n_indeterminate"], j["n_high"]) == (0, 2, 1)
    assert "10^9/L" in j["result_text"]


def test_a_zero_denominator_is_missing_not_infinite(client):
    """A platelet count or ALT of 0 is a missing or mis-entered value. Divided
    through it would produce an infinity that reads as extreme fibrosis."""
    store.save("z", pd.DataFrame({
        "Yas": [50, 50], "AST": [30, 30], "ALT": [0, 40], "Trombosit": [180, 0]}))
    j = _run(client, "z").json()
    out = store.get("z")["FIB4"]
    assert out.isna().all()
    assert "0 of 2 rows" in j["result_text"]


def test_a_missing_mapping_is_rejected(client, sid):
    r = _run(client, sid, column_map={"age": "Yas", "ast": "AST"})
    assert r.status_code == 422
    assert "alt" in r.json()["detail"] and "platelets" in r.json()["detail"]


def test_the_output_column_can_be_named(client, sid):
    _run(client, sid, new_col="fibrosis_index")
    assert "fibrosis_index" in store.get(sid).columns
