"""End-to-end test for the agent orchestration layer.

Boots a real uvicorn backend in a subprocess, uploads ``qa/fixtures/trial.csv``,
and drives it through ``UstatRunner`` with a scripted plan. The runner's
recorded responses are compared against direct calls to the same endpoints.
"""

from __future__ import annotations

import math
import socket
import sys
from pathlib import Path

import pytest

# Allow ``from agent import ...`` when pytest is invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import UstatClient, UstatServer, UstatRunner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "qa" / "fixtures" / "trial.csv"
BACKEND_DIR = PROJECT_ROOT / "backend"


def _free_port() -> int:
    """Return an ephemeral TCP port that is free at the moment of the call."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_runner() -> UstatRunner:
    """Start the real uSTAT backend on a free port and return a runner."""
    port = _free_port()
    server = UstatServer(host="127.0.0.1", port=port, cwd=str(BACKEND_DIR))
    client = UstatClient(server.base)
    runner = UstatRunner(client, server=server)

    server.start(timeout=120)
    try:
        yield runner
    finally:
        server.stop()


PLAN = [
    {
        "name": "table1",
        "args": {
            "variables": ["AGE", "SEX", "STROKE", "CKD", "PAD"],
            "group_column": "ARM",
            "selected_stats": ["auto"],
        },
    },
    {
        "name": "fisher",
        "args": {"row_column": "STROKE", "col_column": "ARM"},
    },
]


def test_e2e_scripted_plan_matches_direct_calls(live_runner: UstatRunner) -> None:
    """Run a scripted table1 + fisher plan and assert parity with direct calls."""
    assert FIXTURE_PATH.is_file(), f"Fixture not found: {FIXTURE_PATH}"

    result = live_runner.run(
        "Compare baseline characteristics and stroke prevalence between arms.",
        str(FIXTURE_PATH),
        plan=PLAN,
    )

    session_id = result["session_id"]
    records = result["records"]
    assert len(records) == 2

    runner_table1 = records[0]["response"]
    runner_fisher = records[1]["response"]

    # Compare against fresh direct calls in the same session.
    direct_table1 = live_runner.client.call("table1", session_id, PLAN[0]["args"])
    direct_fisher = live_runner.client.call("fisher", session_id, PLAN[1]["args"])

    assert runner_table1 == direct_table1
    assert runner_fisher == direct_fisher

    # Numeric sanity checks on the public contract.
    assert runner_table1["total_n"] == 60
    assert len(runner_table1["rows"]) == 5
    assert runner_table1["group_column"] == "ARM"

    fisher = runner_fisher
    assert fisher["test"] == "Fisher's exact test"
    assert isinstance(fisher["p"], float)
    assert 0 < fisher["p"] < 1
    assert isinstance(fisher["odds_ratio"], float)
    assert math.isfinite(fisher["odds_ratio"])

    # Summary should mention both analyses.
    summary = result["summary"]
    assert "Table 1" in summary or "table1" in summary
    assert "Fisher" in summary or "fisher" in summary
