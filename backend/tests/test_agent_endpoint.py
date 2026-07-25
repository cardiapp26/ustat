"""Tests for the server-side ``POST /api/agent/run`` endpoint.

These tests boot a real uvicorn backend in a subprocess, upload a fixture
through the agent endpoint, and verify that explicit plans execute without
requiring an external LLM API key.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import UstatServer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "qa" / "fixtures" / "trial.csv"
BACKEND_DIR = PROJECT_ROOT / "backend"


PLAN = json.dumps(
    [
        {
            "name": "table1",
            "args": {
                "variables": ["AGE", "SEX", "STROKE"],
                "group_column": "ARM",
                "selected_stats": ["auto"],
            },
        },
        {
            "name": "fisher",
            "args": {"row_column": "STROKE", "col_column": "ARM"},
        },
    ]
)


def _free_port() -> int:
    """Return an ephemeral TCP port that is free at the moment of the call."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server() -> UstatServer:
    """Start the real uSTAT backend on a free port and return the server."""
    port = _free_port()
    # The endpoint's internal client calls the server back on a fixed loopback
    # base (SSRF hardening — it no longer trusts the request Host). Point that
    # base at THIS test server's ephemeral port so the loopback call reaches us;
    # the Host-spoofing protection itself is covered by test_agent_ssrf.py.
    prev = os.environ.get("AGENT_INTERNAL_BASE_URL")
    os.environ["AGENT_INTERNAL_BASE_URL"] = f"http://127.0.0.1:{port}"
    server = UstatServer(host="127.0.0.1", port=port, cwd=str(BACKEND_DIR))
    server.start(timeout=120)
    try:
        yield server
    finally:
        server.stop()
        if prev is None:
            os.environ.pop("AGENT_INTERNAL_BASE_URL", None)
        else:
            os.environ["AGENT_INTERNAL_BASE_URL"] = prev


def test_agent_run_explicit_plan(live_server: UstatServer) -> None:
    """POST /api/agent/run with an explicit plan returns summary + records."""
    assert FIXTURE_PATH.is_file(), f"Fixture not found: {FIXTURE_PATH}"

    with open(FIXTURE_PATH, "rb") as fh:
        response = httpx.post(
            f"{live_server.base}/api/agent/run",
            files={"file": ("trial.csv", fh)},
            data={
                "nl_request": "Compare baseline characteristics and stroke prevalence.",
                "plan": PLAN,
            },
            timeout=120,
        )

    response.raise_for_status()
    payload = response.json()

    assert "summary" in payload
    assert "records" in payload
    assert "session_id" in payload
    assert "columns" in payload

    assert len(payload["records"]) == 2
    assert payload["records"][0]["call"]["name"] == "table1"
    assert payload["records"][1]["call"]["name"] == "fisher"

    table1 = payload["records"][0]["response"]
    assert table1["total_n"] == 60
    assert len(table1["rows"]) == 3

    fisher = payload["records"][1]["response"]
    assert fisher["test"] == "Fisher's exact test"
    assert isinstance(fisher["p"], float)

    # Summary should reference both analyses.
    summary = payload["summary"]
    assert "Table 1" in summary or "table1" in summary
    assert "Fisher" in summary or "fisher" in summary


def test_agent_run_missing_plan_and_key(live_server: UstatServer, monkeypatch) -> None:
    """A natural-language request without an explicit plan errors when no LLM key is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with open(FIXTURE_PATH, "rb") as fh:
        response = httpx.post(
            f"{live_server.base}/api/agent/run",
            files={"file": ("trial.csv", fh)},
            data={"nl_request": "Run a table 1 analysis."},
            timeout=120,
        )

    assert response.status_code == 400
    assert "OPENAI_API_KEY" in response.text
