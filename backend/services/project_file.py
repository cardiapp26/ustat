"""The .ustat project file: build, validate, restore.

Phase 1 of docs/DESIGN_project_file.md: a zip container with named JSON
parts. This module owns the format; routers/project.py owns HTTP.

Parts written today:

    manifest.json        identity, engine provenance, per-part sha256 hashes
    data/dataset.json    records, same serializer save_session v1.2 used
    data/originals.json  ingest coercion report + verbatim blanked cells
    dictionary.json      per-column semantics (kind, labels, decimals, ...)
    prep/steps.json      replayable recipe (Phase 1: the case filter only)
    ui/state.json        the frontend's panel state: per-panel settings and
                         stamped results (Phase 2). Opaque here -- the backend
                         carries it, hashes it, and hands it back; only the
                         frontend interprets it.
    audit.json           informational audit trail

Deliberately not yet: named analyses as first-class parts (Phase 3), notes
UI (Phase 3), unknown-part preservation on re-save (Phase 3), seeds
(Phase 4).
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

from services import store
from services.runtime_identity import runtime_identity

SCHEMA_VERSION = "2.0.0"
_KIND = "ustat-project"


class ProjectFileError(ValueError):
    """A project file that cannot be honestly loaded. Message is user-facing."""


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _canonical_json(obj) -> bytes:
    # Stable key order so a byte hash means "same content", not "same dict
    # iteration order that day".
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")


def _dataset_records(df: pd.DataFrame) -> list:
    """Exactly the serializer save_session v1.2 used, for continuity."""
    return json.loads(
        df.replace([np.inf, -np.inf], np.nan).to_json(
            orient="records", date_format="iso", default_handler=str
        )
    )


# ── Build ────────────────────────────────────────────────────────────────────

def build_project(session_id: str, ui_state: Optional[dict] = None) -> bytes:
    """Serialize a live session into .ustat bytes. Raises KeyError if absent.

    ``ui_state`` is the frontend's panel state (settings + stamped results),
    passed through verbatim: the backend does not read it, but the file must
    carry it or reopening a project loses every result on screen.
    """
    df = store.get(session_id)
    if df is None:
        raise KeyError(session_id)

    from routers.upload import _detect_kind  # late: routers import services

    kind_overrides = store.get_kind_overrides(session_id)
    metadata = store.get_metadata(session_id)
    decimals = store.get_decimals(session_id)

    columns = []
    for col in df.columns:
        entry = {
            "name": col,
            "dtype": str(df[col].dtype),
            "kind": kind_overrides.get(col) or _detect_kind(df[col]),
        }
        if col in decimals:
            entry["decimals"] = decimals[col]
        if metadata.get(col):
            entry["metadata"] = metadata[col]
        columns.append(entry)

    dictionary = {
        "columns": columns,
        # Which kinds were the user's explicit choice, as opposed to detected
        # now. Restoring detection results as overrides would pin them.
        "kind_overrides": kind_overrides,
    }

    steps = []
    case_filter = store.get_filter(session_id)
    if case_filter:
        steps.append({"op": "filter", "params": {"conditions": case_filter}})

    dataset_bytes = json.dumps(_dataset_records(df), allow_nan=False, default=str).encode("utf-8")

    parts: dict[str, bytes] = {
        "data/dataset.json": dataset_bytes,
        "dictionary.json": _canonical_json(dictionary),
        "audit.json": _canonical_json(store.get_audit(session_id)),
    }
    if steps:
        parts["prep/steps.json"] = _canonical_json(steps)

    if ui_state:
        parts["ui/state.json"] = _canonical_json(ui_state)

    ingest_report = store.get_ingest_report(session_id)
    preserved = store.get_preserved_cells(session_id)
    if ingest_report or preserved:
        parts["data/originals.json"] = _canonical_json(
            {"report": ingest_report, "preserved": preserved}
        )

    now = time.time()
    manifest = {
        "kind": _KIND,
        "schema_version": SCHEMA_VERSION,
        "created": now,
        "modified": now,
        "filename": store.get_filename(session_id) or f"project_{session_id[:8]}",
        # The environment at save. Per-result provenance arrives with the
        # analyses parts in Phase 2; this is the manifest-level record.
        "engine": runtime_identity(),
        "seed": None,
        "data_hash": _sha256(dataset_bytes),
        "parts": {name: _sha256(data) for name, data in parts.items()},
    }

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", _canonical_json(manifest))
        for name, data in parts.items():
            zf.writestr(name, data)
    return out.getvalue()


# ── Parse / validate ─────────────────────────────────────────────────────────

def parse_project(content: bytes) -> dict:
    """Read and validate .ustat bytes. Returns the parsed parts.

    Validation order matters for the error a user sees: not-a-zip, then
    missing/foreign manifest, then version gate, then integrity.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise ProjectFileError("Not a uSTAT project file (not a zip archive).")

    names = set(zf.namelist())
    if "manifest.json" not in names:
        raise ProjectFileError("Not a uSTAT project file (no manifest).")
    try:
        manifest = json.loads(zf.read("manifest.json"))
    except json.JSONDecodeError:
        raise ProjectFileError("Project manifest is not valid JSON.")
    if manifest.get("kind") != _KIND:
        raise ProjectFileError("Not a uSTAT project file (wrong kind).")

    version = str(manifest.get("schema_version", ""))
    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        raise ProjectFileError(f"Unreadable project schema version: {version!r}.")
    ours = int(SCHEMA_VERSION.split(".", 1)[0])
    if major > ours:
        # Refuse outright rather than half-load: a partial project that opens
        # "successfully" is how analyses silently vanish.
        raise ProjectFileError(
            f"This project was made by a newer uSTAT (schema {version}); "
            f"this build reads up to major {ours}. Update uSTAT to open it."
        )

    if "data/dataset.json" not in names:
        raise ProjectFileError("Project file has no dataset part.")
    dataset_bytes = zf.read("data/dataset.json")
    recorded = manifest.get("data_hash")
    if recorded and recorded != _sha256(dataset_bytes):
        raise ProjectFileError(
            "Dataset does not match the hash recorded in the manifest; "
            "the file is corrupted or was modified outside uSTAT."
        )

    def _read_json(name: str, default):
        if name not in names:
            return default
        try:
            return json.loads(zf.read(name))
        except json.JSONDecodeError:
            raise ProjectFileError(f"Project part {name} is not valid JSON.")

    return {
        "manifest": manifest,
        "dataset": json.loads(dataset_bytes),
        "dictionary": _read_json("dictionary.json", {"columns": [], "kind_overrides": {}}),
        "steps": _read_json("prep/steps.json", []),
        "audit": _read_json("audit.json", []),
        "originals": _read_json("data/originals.json", {}),
        "ui_state": _read_json("ui/state.json", None),
    }


# ── Restore ──────────────────────────────────────────────────────────────────

def restore_project(parsed: dict) -> str:
    """Hydrate a fresh session from parse_project output; returns session id."""
    from routers.upload import coerce_numeric_objects

    df = coerce_numeric_objects(pd.DataFrame(parsed["dataset"]))
    session_id = str(uuid.uuid4())
    store.save(session_id, df)

    dictionary = parsed.get("dictionary") or {}
    kind_overrides = dictionary.get("kind_overrides") or {}
    if kind_overrides:
        store.set_kind_overrides(session_id, kind_overrides)

    metadata = {
        c["name"]: c["metadata"]
        for c in dictionary.get("columns", [])
        if c.get("name") and isinstance(c.get("metadata"), dict)
    }
    if metadata:
        store.save_metadata(session_id, metadata)

    decimals = {
        c["name"]: c["decimals"]
        for c in dictionary.get("columns", [])
        if c.get("name") and isinstance(c.get("decimals"), int)
    }
    if decimals:
        store.save_decimals(session_id, decimals)

    for step in parsed.get("steps") or []:
        if step.get("op") == "filter":
            conditions = (step.get("params") or {}).get("conditions") or []
            if conditions:
                store.save_filter(session_id, conditions)

    for entry in parsed.get("audit") or []:
        if isinstance(entry, dict) and entry.get("action"):
            store.log_action(session_id, entry["action"], entry.get("params"))

    originals = parsed.get("originals") or {}
    if originals.get("report") or originals.get("preserved"):
        preserved = {
            col: {int(row): value for row, value in cells.items()}
            for col, cells in (originals.get("preserved") or {}).items()
        }
        store.save_ingest_report(session_id, originals.get("report") or {}, preserved)

    filename = (parsed.get("manifest") or {}).get("filename")
    if filename:
        store.set_filename(session_id, filename)
    return session_id


def restore_legacy(payload: dict) -> Optional[str]:
    """Restore a v1.x save_session JSON as a project. None if not one."""
    if not isinstance(payload, dict) or "data" not in payload:
        return None
    parsed = {
        "manifest": {"filename": payload.get("filename")},
        "dataset": payload["data"],
        "dictionary": {
            "columns": [
                {"name": col, "metadata": meta}
                for col, meta in (payload.get("col_metadata") or {}).items()
                if isinstance(meta, dict)
            ],
            "kind_overrides": payload.get("kind_overrides") or {},
        },
        "steps": (
            [{"op": "filter", "params": {"conditions": payload["case_filter"]}}]
            if payload.get("case_filter")
            else []
        ),
        "audit": payload.get("audit") or [],
        "originals": {},
    }
    session_id = restore_project(parsed)
    decimals = payload.get("decimals_overrides") or {}
    if decimals:
        store.save_decimals(session_id, decimals)
    return session_id
