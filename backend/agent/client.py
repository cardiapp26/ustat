"""Typed Python client for the uSTAT FastAPI backend."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

import httpx

from .tool_catalog import TOOLS


class UstatServer:
    """Manage a uvicorn subprocess running the uSTAT FastAPI backend."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, cwd: str = "backend"):
        self._host = host
        self._port = port
        self._cwd = cwd
        self.base = f"http://{host}:{port}"
        self._proc: subprocess.Popen | None = None

    def start(self, timeout: float = 30) -> None:
        """Boot uvicorn and block until /api/health returns status: ok."""
        if self._proc is not None:
            return
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--workers",
            "1",
            "--log-level",
            "warning",
        ]
        self._proc = subprocess.Popen(cmd, cwd=self._cwd)
        try:
            self._wait_healthy(timeout)
        except Exception:
            self.stop()
            raise

    def _wait_healthy(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.base}/api/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
            except Exception:
                pass
            time.sleep(0.4)
        raise RuntimeError(f"uvicorn did not become healthy at {self.base}")

    def stop(self) -> None:
        """Terminate the managed uvicorn process."""
        proc = self._proc
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(10)
        finally:
            self._proc = None

    def __enter__(self) -> "UstatServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


class UstatClient:
    """Typed HTTP client for the uSTAT backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base = base_url.rstrip("/")
        self.timeout = 120

    def health(self) -> dict[str, Any]:
        """GET /api/health"""
        r = httpx.get(f"{self.base}/api/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def upload(self, path: str) -> dict[str, Any]:
        """Upload a dataset and return the backend metadata (including session_id)."""
        filename = os.path.basename(path)
        with open(path, "rb") as fh:
            r = httpx.post(
                f"{self.base}/api/upload/",
                files={"file": (filename, fh)},
                timeout=self.timeout,
            )
        r.raise_for_status()
        return r.json()

    def call(
        self,
        name: str,
        session_id: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a named tool from the catalog."""
        args = dict(args or {})
        tool = next((t for t in TOOLS if t["name"] == name), None)
        if tool is None:
            raise KeyError(f"Unknown tool '{name}'")

        method = tool["method"].upper()
        path_template = tool["path"]
        session_in = tool.get("session_in", "body")

        if session_in == "path":
            url = f"{self.base}{path_template.format(session_id=session_id)}"
            payload = args
        else:
            url = f"{self.base}{path_template}"
            payload = {"session_id": session_id, **args}

        if method == "GET":
            r = httpx.get(url, params=payload, timeout=self.timeout)
        else:
            r = httpx.post(url, json=payload, timeout=self.timeout)

        r.raise_for_status()
        return r.json()
