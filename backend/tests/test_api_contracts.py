"""Response contracts for the core session endpoints.

The frontend types these responses in frontend/src/api.ts (DescriptiveStats,
FrequencyTable, SessionInfo). The backend endpoints return plain dicts, so a
renamed or dropped field sails through TypeScript unnoticed and only fails at
runtime in the browser. These tests pin the exact key sets: change a field
here AND in api.ts, or not at all.
"""
import pandas as pd
import pytest

from conftest import make_session

DESCRIPTIVE_KEYS = {
    "n", "missing", "mean", "std", "se", "min", "max", "median",
    "q1", "q3", "iqr", "skewness", "kurtosis",
    "normality_p", "normality_test", "normal", "warnings", "display_decimals",
}

FREQUENCY_KEYS = {"n", "missing", "categories"}
FREQUENCY_CATEGORY_KEYS = {"value", "count", "pct"}

SESSION_INFO_KEYS = {"session_id", "filename", "rows", "columns", "preview"}
SESSION_COLUMN_REQUIRED_KEYS = {"name", "dtype", "kind"}


@pytest.fixture
def contract_session():
    df = pd.DataFrame({
        "age": [34, 55, 61, 47, 29, 70, 58, 44],
        "sex": ["M", "F", "F", "M", "F", "M", "M", "F"],
    })
    yield make_session(df, "contract_test_session")
    store_delete("contract_test_session")


def store_delete(session_id: str) -> None:
    from services import store
    with store._lock:
        store._purge_locked(session_id)


def test_descriptive_contract(client, contract_session):
    r = client.get(f"/api/stats/{contract_session}/descriptive")
    assert r.status_code == 200
    body = r.json()
    assert "age" in body
    assert set(body["age"].keys()) == DESCRIPTIVE_KEYS


def test_frequency_contract(client, contract_session):
    r = client.get(f"/api/stats/{contract_session}/frequency", params={"column": "sex"})
    assert r.status_code == 200
    body = r.json()
    assert set(body["sex"].keys()) == FREQUENCY_KEYS
    assert body["sex"]["categories"], "expected at least one category"
    assert set(body["sex"]["categories"][0].keys()) == FREQUENCY_CATEGORY_KEYS


def test_session_info_contract(client, contract_session):
    r = client.get(f"/api/sessions/{contract_session}")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == SESSION_INFO_KEYS
    assert body["rows"] == 8
    for col in body["columns"]:
        assert SESSION_COLUMN_REQUIRED_KEYS <= set(col.keys())
