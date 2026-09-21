"""
uSTAT Desktop Backend Entry Point

This module is the entry point for the PyInstaller-bundled backend binary.
It starts the FastAPI application via uvicorn, serving both the API and
the pre-built frontend static files.

Usage:
    ustat-backend --port 18731
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
from pathlib import Path


def _resource_root() -> Path:
    """Resolve the application root directory.
    
    When running from a PyInstaller bundle, sys._MEIPASS points to the
    temporary extraction directory. Otherwise, we resolve relative to
    this file's location in the source tree.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def _find_port(preferred: int = 18731) -> int:
    """Find a free port starting from preferred."""
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No available local port found for uSTAT backend.")


def main() -> None:
    parser = argparse.ArgumentParser(description="uSTAT backend server")
    parser.add_argument(
        "--port",
        type=int,
        default=18731,
        help="Port to listen on (default: 18731)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Import every module the backend's source names, report failures, and exit",
    )
    args = parser.parse_args()

    root = _resource_root()
    backend_dir = root / "backend"
    frontend_dist = root / "frontend" / "dist"

    # Add backend to Python path so imports work
    sys.path.insert(0, str(backend_dir))

    if args.self_check:
        from frozen_self_check import run  # noqa: E402

        sys.exit(run(backend_dir))

    # Tell the FastAPI app where to find the frontend static files
    os.environ["USTAT_FRONTEND_DIST"] = str(frontend_dist)
    os.environ["USTAT_DESKTOP_MODE"] = "1"

    # Add localhost origins for the desktop app
    existing_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    desktop_origins = f"http://127.0.0.1:{args.port},http://localhost:{args.port}"
    if existing_origins:
        os.environ["CORS_ALLOWED_ORIGINS"] = f"{existing_origins},{desktop_origins}"
    else:
        os.environ["CORS_ALLOWED_ORIGINS"] = (
            f"https://ustat.drtr.uk,"
            f"http://localhost:5173,http://localhost:5174,"
            f"http://127.0.0.1:5173,http://127.0.0.1:5174,"
            f"{desktop_origins}"
        )

    # Graceful shutdown on SIGTERM (sent by Tauri on window close)
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

    import uvicorn  # noqa: E402

    from main import app  # noqa: E402

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
