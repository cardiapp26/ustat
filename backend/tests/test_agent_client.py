"""Mock-backend contract tests for agent.client and agent.tool_catalog.

These tests verify the UstatClient transport layer and the catalog schema without
starting a real uvicorn process. They are Phase 1 of the LLM backend automation plan.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent import TOOLS, UstatClient, UstatServer


@pytest.fixture
def client() -> UstatClient:
    return UstatClient("http://127.0.0.1:8000")


# ──────────────────────────────────────────────────────────────────────────────
# UstatClient.upload
# ──────────────────────────────────────────────────────────────────────────────


def test_upload_returns_metadata(client: UstatClient, tmp_path, httpx_mock) -> None:
    """upload() POSTs a multipart file and returns parsed session metadata."""
    fixture = tmp_path / "demo.csv"
    fixture.write_text("A,B\n1,x\n2,y\n")

    expected = {
        "session_id": "sess-123",
        "filename": "demo.csv",
        "rows": 2,
        "columns": [
            {"name": "A", "dtype": "int64", "kind": "numeric"},
            {"name": "B", "dtype": "object", "kind": "categorical"},
        ],
        "preview": [{"A": 1, "B": "x"}, {"A": 2, "B": "y"}],
    }
    httpx_mock.add_response(url="http://127.0.0.1:8000/api/upload/", json=expected)

    result = client.upload(str(fixture))

    assert result == expected
    request = httpx_mock.get_request()
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:8000/api/upload/"
    assert "multipart/form-data" in request.headers.get("content-type", "")


def test_upload_raises_on_http_error(client: UstatClient, tmp_path, httpx_mock) -> None:
    """upload() propagates HTTP errors from the backend."""
    fixture = tmp_path / "bad.csv"
    fixture.write_text("A\n1\n")

    httpx_mock.add_response(url="http://127.0.0.1:8000/api/upload/", status_code=413)
    with pytest.raises(Exception):  # httpx.HTTPStatusError
        client.upload(str(fixture))


# ──────────────────────────────────────────────────────────────────────────────
# UstatClient.call
# ──────────────────────────────────────────────────────────────────────────────


def test_call_body_session_includes_session_id(client: UstatClient, httpx_mock) -> None:
    """For body-session tools, session_id is merged into the JSON body."""
    httpx_mock.add_response(
        url="http://127.0.0.1:8000/api/stats/table1", json={"rows": []}
    )

    result = client.call(
        "table1",
        "sess-123",
        {"variables": ["age"], "group_column": "sex"},
    )

    assert result == {"rows": []}
    request = httpx_mock.get_request()
    assert request.method == "POST"
    body = json.loads(request.content)
    assert body == {
        "session_id": "sess-123",
        "variables": ["age"],
        "group_column": "sex",
    }


def test_call_path_session_formats_url(client: UstatClient, httpx_mock) -> None:
    """For path-session tools, session_id is substituted into the URL path."""
    httpx_mock.add_response(
        url="http://127.0.0.1:8000/api/compute/sess-abc/formula",
        json={"ok": True},
    )

    result = client.call(
        "formula",
        "sess-abc",
        {"formula": "age + 1", "new_col": "age_plus"},
    )

    assert result == {"ok": True}
    request = httpx_mock.get_request()
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:8000/api/compute/sess-abc/formula"


def test_call_get_tool(client: UstatClient, httpx_mock) -> None:
    """GET tools pass merged params as query parameters."""
    httpx_mock.add_response(
        url="http://127.0.0.1:8000/api/stats/sess-x/descriptive?column=age",
        json={"n": 100},
    )

    result = client.call("descriptive", "sess-x", {"column": "age"})

    assert result == {"n": 100}
    request = httpx_mock.get_request()
    assert request.method == "GET"


def test_call_unknown_tool(client: UstatClient) -> None:
    """Calling an undefined tool name raises before any HTTP request."""
    with pytest.raises((KeyError, ValueError)):
        client.call("not_a_tool", "sess-123", {})


def test_call_propagates_http_error(client: UstatClient, httpx_mock) -> None:
    """call() surfaces backend 4xx/5xx responses as httpx exceptions."""
    httpx_mock.add_response(
        url="http://127.0.0.1:8000/api/stats/table1", status_code=400
    )
    with pytest.raises(Exception):  # httpx.HTTPStatusError
        client.call("table1", "sess-123", {"variables": ["age"]})


# ──────────────────────────────────────────────────────────────────────────────
# UstatServer lifecycle (mocked subprocess)
# ──────────────────────────────────────────────────────────────────────────────


def test_server_start_blocks_until_healthy(httpx_mock) -> None:
    """UstatServer.start() polls /api/health and only returns once status is ok."""
    server = UstatServer(port=8123, cwd="backend")
    proc = MagicMock()

    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    httpx_mock.add_response(
        url="http://127.0.0.1:8123/api/health", json={"status": "ok"}
    )

    with patch("agent.client.subprocess.Popen", return_value=proc) as popen:
        server.start(timeout=5)

    assert server._proc is proc
    popen.assert_called_once()


def test_server_stop_terminates_process() -> None:
    """UstatServer.stop() terminates the managed subprocess and clears state."""
    server = UstatServer(port=8123)
    proc = MagicMock()
    server._proc = proc

    server.stop()

    proc.terminate.assert_called_once()
    # Implementation may call wait(10) or wait(timeout=10)
    assert proc.wait.called
    assert server._proc is None


# ──────────────────────────────────────────────────────────────────────────────
# Tool catalog contract tests
# ──────────────────────────────────────────────────────────────────────────────


def test_tools_is_populated() -> None:
    """The catalog must expose at least the foundational analysis tools."""
    names = {t["name"] for t in TOOLS}
    required = {
        "table1",
        "fisher",
        "chisquare",
        "ttest",
        "anova",
        "formula",
        "descriptive",
    }
    assert required <= names, f"Missing tools: {required - names}"


def test_tool_schema_contract() -> None:
    """Every catalog entry must carry the mandatory fields."""
    required_keys = {"name", "method", "path", "session_in", "body", "requires", "doc"}
    for tool in TOOLS:
        missing = required_keys - set(tool.keys())
        assert not missing, f"Tool {tool.get('name')} missing keys: {missing}"
        assert tool["session_in"] in {"body", "path"}
        assert tool["method"].upper() in {"GET", "POST", "PUT", "DELETE", "PATCH"}
        assert tool["path"].startswith("/api/"), (
            f"Tool {tool['name']} path must start with /api/"
        )
        assert isinstance(tool["body"], dict)
        assert isinstance(tool["requires"], dict)
        assert isinstance(tool["doc"], str) and tool["doc"]


def test_tool_names_unique() -> None:
    """Tool names must be unique so client.call(name) is unambiguous."""
    names = [t["name"] for t in TOOLS]
    assert len(names) == len(set(names)), f"Duplicate tool names: {names}"


def test_requires_reference_known_body_args() -> None:
    """Semantic guards in 'requires' should refer to args declared in 'body'."""
    for tool in TOOLS:
        # The catalog body may be a JSON schema (with properties) or a flat type-hint map.
        body_props = set(tool["body"].get("properties", {}).keys())
        body_props.update(tool["body"].keys())
        for req_arg in tool["requires"]:
            if req_arg == "variables":
                # variables is a special list arg present on several tools
                assert req_arg in body_props or any(
                    k in body_props
                    for k in (
                        "variables",
                        "column",
                        "outcome",
                        "predictor",
                        "score_column",
                    )
                )
            else:
                assert req_arg in body_props, (
                    f"Tool {tool['name']} requires unknown arg '{req_arg}'"
                )


def test_path_session_tools_have_session_placeholder() -> None:
    """Tools with session_in='path' must include {session_id} in the URL."""
    for tool in TOOLS:
        if tool["session_in"] == "path":
            assert "{session_id}" in tool["path"], (
                f"Tool {tool['name']} is path-session but missing {{session_id}} in path"
            )
