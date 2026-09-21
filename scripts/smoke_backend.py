"""
Smoke-test a PyInstaller-built uSTAT backend before it is shipped.

The desktop installers carry a frozen backend, and a frozen backend can fail
in ways the source tree never does: PyInstaller only bundles what it can see
imported, so a compiled submodule it misses is absent from the build and the
binary dies at startup. v3.7.0 shipped exactly that (pyreadstat's
_readstat_writer), and every installed copy showed a window that never loaded.

This starts the binary, waits for /api/health, checks the frontend is served,
uploads a CSV, an SPSS .sav and an Excel .xlsx, and runs a t-test. Any failure
exits non-zero with the tail of the backend's own output.

Usage:
    python scripts/smoke_backend.py dist/ustat-backend/ustat-backend
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

STARTUP_TIMEOUT_S = 180  # first start of a cold runner can be slow
REQUEST_TIMEOUT_S = 60


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(url: str, data: bytes | None = None, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post_json(url: str, payload: dict) -> tuple[int, dict]:
    status, body = request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    return status, json.loads(body or b"{}")


def upload(base: str, path: Path) -> dict:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    status, raw = request(
        f"{base}/api/upload/",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    if status != 200:
        raise AssertionError(f"upload {path.name}: HTTP {status}: {raw[:500]!r}")
    return json.loads(raw)


def write_fixtures(folder: Path) -> list[Path]:
    import pandas as pd
    import pyreadstat

    df = pd.DataFrame(
        {
            "age": [54.0, 61.0, 47.0, 70.0, 58.0, 66.0, 49.0, 63.0],
            "group": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        }
    )
    csv = folder / "smoke.csv"
    sav = folder / "smoke.sav"
    xlsx = folder / "smoke.xlsx"
    df.to_csv(csv, index=False)
    pyreadstat.write_sav(df, sav, variable_value_labels={"group": {1.0: "A", 2.0: "B"}})
    df.to_excel(xlsx, index=False)
    return [csv, sav, xlsx]


def wait_for_health(base: str, proc: subprocess.Popen) -> float:
    start = time.monotonic()
    while time.monotonic() - start < STARTUP_TIMEOUT_S:
        if proc.poll() is not None:
            raise AssertionError(f"backend exited with code {proc.returncode} before it was healthy")
        try:
            status, _ = request(f"{base}/api/health")
            if status == 200:
                return time.monotonic() - start
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    raise AssertionError(f"backend not healthy within {STARTUP_TIMEOUT_S}s")


def run_checks(base: str, fixtures: list[Path]) -> None:
    status, html = request(f"{base}/")
    if status != 200 or b"uSTAT" not in html:
        raise AssertionError(f"frontend not served at /: HTTP {status}")
    print("ok  frontend served at /")

    sessions = {}
    for path in fixtures:
        session = upload(base, path)
        if session.get("rows") != 8:
            raise AssertionError(f"{path.name}: expected 8 rows, got {session.get('rows')}")
        sessions[path.suffix] = session["session_id"]
        print(f"ok  uploaded {path.name}")

    status, result = post_json(
        f"{base}/api/stats/ttest",
        {"session_id": sessions[".sav"], "column": "age", "group_column": "group"},
    )
    if status != 200:
        raise AssertionError(f"t-test: HTTP {status}: {result}")
    print("ok  t-test on the .sav session")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("binary", type=Path, help="path to the built ustat-backend executable")
    args = parser.parse_args()

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        fixtures = write_fixtures(folder)
        log_path = folder / "backend.log"
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                [str(args.binary), "--port", str(port)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "USTAT_NO_BROWSER": "1", "USTAT_DESKTOP_MODE": "1"},
            )
            try:
                took = wait_for_health(base, proc)
                print(f"ok  healthy after {took:.1f}s")
                run_checks(base, fixtures)
            except AssertionError as e:
                print(f"FAIL {e}", file=sys.stderr)
                print("---- backend output (tail) ----", file=sys.stderr)
                print(log_path.read_text(errors="replace")[-6000:], file=sys.stderr)
                return 1
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
