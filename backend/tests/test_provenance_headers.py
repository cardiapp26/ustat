"""Every API response says what computed it.

The bug: the header bar printed the session's engine choice ("R-based
statistics") over numbers the server's Python had computed, because only two
analyses are routed to a local engine and nothing else reported where it ran.
A reader reproducing the work in R would fail to match numbers that were never
R's.

The fix has to be un-forgettable, so it is middleware rather than a field in
each response model. These tests hold that line: the headers are on every /api
response including errors, they name real installed versions rather than the
pins, and the client is allowed to read them cross-origin.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from services import store
from services.runtime_identity import (
    HEADER_ENGINE,
    HEADER_ENGINE_FINGERPRINT,
    HEADER_ENGINE_VERSION,
    HEADER_PACKAGES,
    HEADER_PYTHON,
    HEADER_RUNTIME,
    PROVENANCE_HEADERS,
    package_versions,
    provenance_headers,
    runtime_identity,
)

client = TestClient(app)


@pytest.fixture(scope="module")
def sid():
    df = pd.DataFrame({
        "AGE": [55, 62, 48, 71, 66, 59, 44, 68, 52, 61],
        "G": list("AABBAABBAB"),
    })
    store.save("prov_sid", df)
    yield "prov_sid"
    store.delete("prov_sid")


def test_every_api_response_carries_provenance():
    r = client.get("/api/health")
    for name in PROVENANCE_HEADERS:
        assert r.headers.get(name), f"{name} missing"
    assert r.headers[HEADER_RUNTIME] == "server"
    assert r.headers[HEADER_ENGINE] == "python"


def test_an_analysis_response_carries_it_too(sid):
    r = client.post("/api/stats/ttest", json={"session_id": sid, "column": "AGE", "group_column": "G"})
    assert r.status_code == 200, r.text
    assert r.headers[HEADER_ENGINE] == "python"
    assert "scipy=" in r.headers[HEADER_PACKAGES]


def test_error_responses_carry_it(sid):
    """A 400 is still an answer from an engine, and still has to say which."""
    r = client.post("/api/stats/ttest", json={"session_id": "nope", "column": "AGE"})
    assert r.status_code >= 400
    assert r.headers[HEADER_ENGINE] == "python"


def test_non_api_paths_are_left_alone():
    """A JavaScript bundle did not compute anything."""
    r = client.get("/")
    assert HEADER_ENGINE not in r.headers


def test_versions_are_the_installed_ones_not_the_pins():
    """Reporting requirements.txt would report what SHOULD be installed."""
    import scipy

    versions = package_versions()
    assert versions["scipy"] == scipy.__version__
    assert set(versions) >= {"numpy", "scipy", "pandas", "statsmodels"}


def test_package_header_round_trips_to_a_mapping():
    header = provenance_headers()[HEADER_PACKAGES]
    parsed = dict(pair.split("=", 1) for pair in header.split(","))
    assert parsed == package_versions()


def test_engine_identity_reports_the_runtime_too():
    body = client.get("/api/engine/identity").json()
    runtime = body["runtime"]
    assert runtime["engine"] == "python"
    assert runtime["packages"]["scipy"]
    assert runtime["python"]
    # The pre-existing keys the local-compute check reads by name must survive.
    assert body["fingerprint"] and body["version"]
    assert body["python"]["fingerprint"] == body["fingerprint"]


def test_the_fingerprint_header_matches_the_identity_endpoint():
    full = client.get("/api/engine/identity").json()["fingerprint"]
    assert full.startswith(client.get("/api/health").headers[HEADER_ENGINE_FINGERPRINT])


def test_headers_are_exposed_to_cross_origin_javascript():
    """Provenance the client cannot read is provenance that does not exist."""
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    exposed = {h.strip().lower() for h in r.headers.get("access-control-expose-headers", "").split(",")}
    for name in PROVENANCE_HEADERS:
        assert name.lower() in exposed, f"{name} not exposed"


def test_the_reported_os_is_coarse():
    """A methods section needs 'Linux x86_64'; nothing needs a kernel build."""
    os_field = runtime_identity()["os"]
    assert len(os_field.split()) <= 3
    assert HEADER_PYTHON not in os_field
    assert HEADER_ENGINE_VERSION not in os_field
