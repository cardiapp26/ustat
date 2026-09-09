"""HTTP surface of the .ustat project file (docs/DESIGN_project_file.md).

Thin by design: services/project_file.py owns the format, this router owns
status codes, filenames and the response shape the frontend hydrates from.
save_session/load_session in routers/session.py stay untouched as the legacy
v1.x path; /api/project/load also accepts those old JSON files directly.
"""
from __future__ import annotations

import json
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from services import store
from services.project_file import (
    ProjectFileError,
    build_project,
    parse_project,
    restore_legacy,
    restore_project,
)

router = APIRouter()


@router.get("/{session_id}/save")
async def save_project(session_id: str):
    """Download the session as a .ustat project file."""
    try:
        content = build_project(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    base = store.get_filename(session_id) or f"project_{session_id[:8]}"
    base = base.rsplit(".", 1)[0] if "." in base else base
    ascii_base = base.encode("ascii", errors="replace").decode("ascii")
    utf8_base = quote(base, safe="")
    return StreamingResponse(
        iter([content]),
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename=\"{ascii_base}.ustat\"; "
                f"filename*=UTF-8''{utf8_base}.ustat"
            )
        },
    )


@router.post("/load")
async def load_project(file: UploadFile = File(...)):
    """Open a .ustat file, or a legacy v1.x save_session JSON, as a session."""
    raw = await file.read()

    if raw[:2] == b"PK":
        try:
            session_id = restore_project(parse_project(raw))
        except ProjectFileError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    else:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPException(
                status_code=400,
                detail="Not a uSTAT project file (neither a .ustat archive nor session JSON).",
            )
        session_id = restore_legacy(payload)
        if session_id is None:
            raise HTTPException(
                status_code=400, detail="Missing 'data' key in session file"
            )

    df = store.get(session_id)

    from routers.upload import _detect_kind

    overrides = store.get_kind_overrides(session_id)
    metadata = store.get_metadata(session_id)
    columns = []
    for col in df.columns:
        entry = {
            "name": col,
            "dtype": str(df[col].dtype),
            "kind": overrides.get(col) or _detect_kind(df[col]),
        }
        value_labels = (metadata.get(col) or {}).get("value_labels")
        if value_labels:
            entry["value_labels"] = value_labels
        columns.append(entry)

    case_filter = store.get_filter(session_id)
    preview = json.loads(
        df.head(2000)
        .replace([float("inf"), float("-inf")], None)
        .to_json(orient="records", default_handler=str, date_format="iso", date_unit="s")
    )
    return {
        "session_id": session_id,
        "filename": store.get_filename(session_id) or file.filename,
        "rows": len(df),
        "columns": columns,
        "preview": preview,
        "case_filter": {
            "conditions": case_filter,
            "selected": len(store.get_filtered(session_id)),
            "total": len(df),
        }
        if case_filter
        else None,
    }
