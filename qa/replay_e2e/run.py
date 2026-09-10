#!/usr/bin/env python3
"""Clean-environment replay check (docs/DESIGN_project_file.md, script export).

The claim under test: a replay script generated from one session's recipe,
run by a separate process against a FRESH server, rebuilds the same prepared
dataset the GUI session had. Not "similar"; the same rows and values.

    backend/.venv/bin/python qa/replay_e2e/run.py

Steps:
 1. start server A (new process, empty in-memory store);
 2. drive it the way the GUI does: upload the raw file, add a formula column,
    edit a cell, delete a column, apply a case filter;
 3. ask A for the Python replay script and for its prepared CSV (ground truth);
 4. stop A; start server B, another new process, so nothing A held survives;
 5. run the generated script as its own process against B;
 6. compare B's prepared CSV with A's. Exit code 0 only on equality.

Every server is started with SESSION_DISK_CACHE unset, so the default
RAM-only store applies and "fresh" means fresh.
"""
from __future__ import annotations

import os
import pathlib
import re
import socket
import subprocess
import sys
import tempfile
import time

import pandas as pd
import requests

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
RAW = REPO / "qa" / "models_audit" / "dataset.csv"
PYTHON = sys.executable


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """One uvicorn process; a new one per phase so state cannot leak across."""

    def __init__(self, port: int):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        env = {k: v for k, v in os.environ.items() if k != "SESSION_DISK_CACHE"}
        self.proc = subprocess.Popen(
            [PYTHON, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
            cwd=BACKEND, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(120):
            try:
                if requests.get(f"{self.base}/api/health", timeout=1).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if self.proc.poll() is not None:
                raise SystemExit(f"server on {port} died on startup:\n{self.proc.stderr.read()}")
            time.sleep(0.5)
        raise SystemExit(f"server on {port} never became healthy")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def check(r: requests.Response) -> requests.Response:
    if r.status_code >= 300:
        raise SystemExit(f"{r.request.method} {r.url} -> {r.status_code}: {r.text[:300]}")
    return r


def drive_gui_session(base: str) -> tuple[str, str, bytes]:
    """Do what a user would, return (script text, prepared CSV bytes)."""
    with open(RAW, "rb") as fh:
        sid = check(requests.post(f"{base}/api/upload/", files={"file": (RAW.name, fh)})).json()["session_id"]
    check(requests.post(f"{base}/api/compute/{sid}/formula",
                        json={"new_col": "bmi_x2", "formula": "bmi * 2"}))
    check(requests.patch(f"{base}/api/sessions/{sid}/cell",
                         json={"row_index": 3, "column": "age", "value": 61.5}))
    check(requests.delete(f"{base}/api/compute/{sid}/column/cost"))
    check(requests.post(f"{base}/api/sessions/{sid}/select_cases",
                        json={"conditions": [{"column": "age", "operator": "gt", "value": 50}], "apply": True}))
    script = check(requests.post(f"{base}/api/project/{sid}/script?lang=python", json={"ui_state": None})).text
    prepared = check(requests.get(f"{base}/api/sessions/{sid}/export/csv")).content
    return sid, script, prepared


def run_replay(script: str, base: str, workdir: pathlib.Path) -> bytes:
    """Point the generated script at the fresh server and run it as a process."""
    out = workdir / "prepared_replay.csv"
    patched = re.sub(r'^BASE = ".*?"', f'BASE = "{base}"', script, flags=re.M)
    patched = re.sub(r'^RAW_FILE = ".*?"', f'RAW_FILE = "{RAW}"', patched, flags=re.M)
    patched = re.sub(r'^OUT_FILE = ".*?"', f'OUT_FILE = "{out}"', patched, flags=re.M)
    path = workdir / "replay.py"
    path.write_text(patched)
    proc = subprocess.run([PYTHON, str(path)], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise SystemExit(f"replay script failed:\n{proc.stdout}\n{proc.stderr}")
    print(proc.stdout.strip())
    return out.read_bytes()


def compare(gui: bytes, replay: bytes) -> None:
    a = pd.read_csv(pd.io.common.BytesIO(gui))
    b = pd.read_csv(pd.io.common.BytesIO(replay))
    if a.shape != b.shape:
        raise SystemExit(f"MISMATCH shape: gui {a.shape} vs replay {b.shape}")
    if list(a.columns) != list(b.columns):
        raise SystemExit(f"MISMATCH columns: {list(a.columns)} vs {list(b.columns)}")
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=0, atol=0)
    except AssertionError as exc:
        raise SystemExit(f"MISMATCH values:\n{exc}")
    print(f"OK: replay rebuilt the prepared dataset exactly ({a.shape[0]} rows x {a.shape[1]} cols)")


def main() -> None:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ustat_replay_"))
    server_a = Server(free_port())
    try:
        sid, script, prepared_gui = drive_gui_session(server_a.base)
        (workdir / "prepared_gui.csv").write_bytes(prepared_gui)
        print(f"session A {sid[:8]}: script {len(script)} bytes, prepared {len(prepared_gui)} bytes")
    finally:
        server_a.stop()

    server_b = Server(free_port())
    try:
        # Prove the environment is clean: A's session must be unknown to B.
        r = requests.get(f"{server_b.base}/api/sessions/{sid}/export/csv")
        if r.status_code != 404:
            raise SystemExit(f"server B is not fresh: A's session answered {r.status_code}")
        prepared_replay = run_replay(script, server_b.base, workdir)
    finally:
        server_b.stop()

    compare(prepared_gui, prepared_replay)


if __name__ == "__main__":
    main()
