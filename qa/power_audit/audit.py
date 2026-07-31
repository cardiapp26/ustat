"""Call /api/stats/power for every case in cases.json and record the answer.

    backend/.venv/bin/python qa/power_audit/audit.py

Writes `endpoints.json`. Nothing here judges anything; compare.py puts the
numbers next to R's.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


def main() -> None:
    cases = json.loads((HERE / "cases.json").read_text())
    client = TestClient(app)
    out: dict = {}
    errors: dict = {}

    for case in cases:
        key = case["key"]
        body = {k: v for k, v in case.items() if k != "key"}
        r = client.post("/api/stats/power", json=body)
        if r.status_code != 200:
            errors[key] = {"status": r.status_code, "detail": r.json()}
            continue
        j = r.json()
        out[key] = {
            "result": j.get("result"),
            "label": j.get("label"),
            "test": case["test"],
            "solve_for": case["solve_for"],
        }

    (HERE / "endpoints.json").write_text(
        json.dumps({"cases": out, "errors": errors}, indent=2, default=str))
    print(f"captured {len(out)} of {len(cases)}")
    for key, err in errors.items():
        print(f"  ERROR {key}: {err['status']} {str(err['detail'])[:160]}")


if __name__ == "__main__":
    main()
