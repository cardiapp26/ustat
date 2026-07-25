"""End-to-end agent demo: boot backend, upload fixture, run table1 + fisher.

Usage from repo root:
    python scripts/agent_demo.py
"""

from __future__ import annotations

import json
import os
import sys

# Allow importing backend/agent from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from agent import UstatClient, UstatServer


def main() -> int:
    fixture = os.path.join(
        os.path.dirname(__file__), "..", "qa", "fixtures", "trial.csv"
    )
    if not os.path.isfile(fixture):
        print(f"Fixture not found: {fixture}", file=sys.stderr)
        return 1

    backend_dir = os.path.join(os.path.dirname(__file__), "..", "backend")
    # Use a non-default port in case 8000 is already occupied locally.
    server = UstatServer(host="127.0.0.1", port=8123, cwd=backend_dir)
    client = UstatClient(server.base)

    server.start()
    try:
        print("Health:", client.health())

        meta = client.upload(fixture)
        sid = meta["session_id"]
        print(
            f"\nUploaded {meta['filename']}: {meta['rows']} rows, {len(meta['columns'])} columns"
        )
        print("Columns:", json.dumps(meta["columns"], indent=2))

        print("\nRunning /api/stats/table1 ...")
        t1 = client.call(
            "table1",
            sid,
            {
                "variables": ["AGE", "SEX", "STROKE", "CKD", "PAD"],
                "group_column": "ARM",
                "selected_stats": ["auto"],
            },
        )
        print("Table 1 rows:", len(t1.get("rows", [])))
        print(json.dumps(t1.get("rows", [])[:5], indent=2))

        print("\nRunning /api/stats/fisher ...")
        fx = client.call(
            "fisher",
            sid,
            {"row_column": "STROKE", "col_column": "ARM"},
        )
        print("test:", fx.get("test"))
        print("p-value:", fx.get("p"))
        print("odds_ratio:", fx.get("odds_ratio"))
        print("interpretation:", fx.get("interpretation"))

        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
