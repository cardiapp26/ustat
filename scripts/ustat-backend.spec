# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for uSTAT Desktop Backend

Produces a single-directory bundle called "ustat-backend" that includes:
  - The FastAPI application (backend/)
  - The pre-built frontend (frontend/dist/)
  - All Python dependencies

Build with:
    cd <project_root>
    pyinstaller scripts/ustat-backend.spec --noconfirm
"""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # project root (scripts/../)
BACKEND = ROOT / "backend"
FRONTEND_DIST = ROOT / "frontend" / "dist"

# Collect all backend Python source files
backend_datas = []
for dirpath, dirnames, filenames in os.walk(BACKEND):
    # Skip tests, caches and hidden directories. The last matters locally:
    # a checkout's backend/.venv would otherwise be copied in whole (~180 MB).
    dirnames[:] = [
        d for d in dirnames
        if d not in ("__pycache__", "tests", "node_modules", "venv") and not d.startswith(".")
    ]
    for fn in filenames:
        # .R for backend/ustat_engine_r: those sources are not imported, they
        # are hashed. Leaving them out would make source_fingerprint() return
        # None in a desktop build, which a browser must read as "cannot prove
        # equality" and refuse to compute locally — safe, but needlessly.
        if fn.endswith((".py", ".R", ".json", ".yml", ".yaml", ".html", ".css", ".js")):
            src = os.path.join(dirpath, fn)
            dst = os.path.relpath(dirpath, ROOT)
            backend_datas.append((src, dst))

# Collect frontend dist
frontend_datas = []
if FRONTEND_DIST.exists():
    for dirpath, dirnames, filenames in os.walk(FRONTEND_DIST):
        for fn in filenames:
            src = os.path.join(dirpath, fn)
            dst = os.path.relpath(dirpath, ROOT)
            frontend_datas.append((src, dst))

a = Analysis(
    [str(BACKEND / "desktop_main.py")],
    pathex=[str(BACKEND)],
    datas=backend_datas + frontend_datas,
    hiddenimports=[
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "fastapi",
        "starlette",
        "starlette.responses",
        "starlette.routing",
        "starlette.middleware",
        "starlette.middleware.cors",
        "multipart",
        "multipart.multipart",
        "pandas",
        "numpy",
        "scipy",
        "scipy.special",
        "scipy.stats",
        "scipy.optimize",
        "scipy.interpolate",
        "statsmodels",
        "statsmodels.api",
        "statsmodels.formula.api",
        "sklearn",
        "sklearn.ensemble",
        "sklearn.linear_model",
        "sklearn.model_selection",
        "sklearn.preprocessing",
        "sklearn.metrics",
        "lifelines",
        "lifelines.fitters",
        "patsy",
        "openpyxl",
        "xlrd",
        "pyreadstat",
        "pdfplumber",
        "docx",
        "psutil",
        "loguru",
        "simpleeval",
    ]
    # Compiled code importing its own siblings is invisible to PyInstaller's
    # bytecode scan. pyreadstat's extension imports _readstat_writer at load
    # time; without this every build died on startup (v3.7.0). The release
    # workflow's `--self-check` step is what finds the next one.
    + collect_submodules("pyreadstat"),
    excludes=[
        "tkinter",
        "matplotlib",
        "PIL",
        "IPython",
        "notebook",
        "pytest",
        "hypothesis",
    ],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ustat-backend",
    debug=False,
    strip=True,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    name="ustat-backend",
)
