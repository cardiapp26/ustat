"""HTTP surface of the .ustat project file (docs/DESIGN_project_file.md).

Thin by design: services/project_file.py owns the format, this router owns
status codes, filenames and the response shape the frontend hydrates from.
save_session/load_session in routers/session.py stay untouched as the legacy
v1.x path; /api/project/load also accepts those old JSON files directly.
"""
from __future__ import annotations

import json
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

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
async def save_project_get(session_id: str):
    """Download the session as a .ustat project file (no frontend state)."""
    return _save(session_id, None)


class SaveProjectRequest(BaseModel):
    # Opaque to the backend: per-panel settings and stamped results, written
    # into ui/state.json and handed back verbatim on load.
    ui_state: Optional[dict] = None


@router.post("/{session_id}/save")
async def save_project(session_id: str, body: SaveProjectRequest):
    """Download the session as .ustat, carrying the frontend's panel state."""
    return _save(session_id, body.ui_state)


def _save(session_id: str, ui_state: Optional[dict]):
    try:
        content = build_project(session_id, ui_state)
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


@router.post("/{session_id}/script")
async def export_script(session_id: str, body: SaveProjectRequest, lang: str = "python"):
    """The project's replay script: import, recorded prep steps, export,
    saved-analysis definitions. See services/script_export.py for why it
    replays the API rather than translating steps to pandas."""
    if lang != "python":
        raise HTTPException(status_code=400, detail="Only lang=python is supported for now.")
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    from services.script_export import generate_python_script

    ui_state = body.ui_state or {}
    analyses = ui_state.get("savedAnalyses") if isinstance(ui_state, dict) else None
    script = generate_python_script(
        steps=store.get_steps(session_id),
        analyses=analyses if isinstance(analyses, list) else None,
        project_name=store.get_filename(session_id) or session_id[:8],
    )
    return PlainTextResponse(script, media_type="text/x-python")


@router.post("/load")
async def load_project(file: UploadFile = File(...)):
    """Open a .ustat file, or a legacy v1.x save_session JSON, as a session."""
    raw = await file.read()

    ui_state = None
    if raw[:2] == b"PK":
        try:
            parsed = parse_project(raw)
            session_id = restore_project(parsed)
        except ProjectFileError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        ui_state = parsed.get("ui_state")
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
        # v1.3 autosave snapshots piggyback the frontend's panel state on the
        # legacy JSON; older files simply lack the key.
        candidate = payload.get("ui_state")
        ui_state = candidate if isinstance(candidate, dict) else None

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
        "ui_state": ui_state,
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
