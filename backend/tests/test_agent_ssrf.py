"""Security regression tests for the agent endpoint."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import app


def test_agent_uses_loopback_base_url_not_spoofed_host(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class StubClient:
        def __init__(self, base_url: str) -> None:
            captured["base_url"] = base_url

    class StubRunner:
        def __init__(self, client: Any, server: Any, planner: Any) -> None:
            pass

        def run(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "summary": "ok",
                "records": [],
                "session_id": "sid-agent",
                "columns": ["x"],
            }

    import agent

    monkeypatch.delenv("AGENT_INTERNAL_BASE_URL", raising=False)
    monkeypatch.setattr(agent, "UstatClient", StubClient)
    monkeypatch.setattr(agent, "UstatRunner", StubRunner)

    client = TestClient(app)
    response = client.post(
        "/api/agent/run",
        headers={"Host": "evil.example.com"},
        files={"file": ("tiny.csv", b"x\n1\n", "text/csv")},
        data={"plan": json.dumps([])},
    )

    assert response.status_code == 200
    assert captured["base_url"] == "http://127.0.0.1:8000"
    assert "evil.example.com" not in captured["base_url"]


def test_agent_rejects_upload_over_size_cap(monkeypatch) -> None:
    import routers.agent as agent_router

    monkeypatch.setattr(agent_router, "MAX_UPLOAD_BYTES", 8)

    client = TestClient(app)
    response = client.post(
        "/api/agent/run",
        files={"file": ("too-large.csv", b"x\n12345678\n", "text/csv")},
        data={"plan": json.dumps([])},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File too large. Maximum upload size is 0 MB."
