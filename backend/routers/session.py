"""Session management: cell editing, dataset export, session save/load, audit."""
import io
import json
import math
import os
import re
import tempfile
import time
import uuid
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from services import store
from services.dirty_value_guard import values_are_numeric

router = APIRouter()


PLACEHOLDER_COLS = 5
PLACEHOLDER_ROWS = 10


def _sav_scalar(value: Any, numeric: bool) -> Any:
    if value is None:
        return None
    if numeric:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            return int(numeric_value)
        return numeric_value
    return str(value)


def _sav_value_labels(labels: dict, series: pd.Series) -> dict:
    if not isinstance(labels, dict):
        return {}
    numeric = pd.api.types.is_numeric_dtype(series)
    out = {}
    for raw_key, raw_label in labels.items():
        if raw_label is None or str(raw_label) == "":
            continue
        key = _sav_scalar(raw_key, numeric)
        if numeric and not isinstance(key, (int, float)):
            continue
        out[key] = str(raw_label)
    return out


def _sav_missing_ranges(ranges: Any, series: pd.Series) -> list:
    if not isinstance(ranges, list):
        return []
    numeric = pd.api.types.is_numeric_dtype(series)
    out = []
    for item in ranges:
        if isinstance(item, dict):
            lo = _sav_scalar(item.get("lo"), numeric)
            hi = _sav_scalar(item.get("hi", item.get("lo")), numeric)
            if lo is None:
                continue
            out.append({"lo": lo, "hi": hi if hi is not None else lo})
        else:
            value = _sav_scalar(item, numeric)
            if value is not None:
                out.append(value)
    return out


def _measure_for_export(kind: str, metadata: dict) -> str:
    measure = str((metadata or {}).get("measure", "")).strip().lower()
    if measure in {"nominal", "ordinal", "scale"}:
        return measure
    if kind == "ordinal":
        return "ordinal"
    if kind in {"categorical", "text"}:
        return "nominal"
    return "scale"


_SPSS_NAME_MAX_LEN = 64


def _sanitize_spss_name(name: str, used: set) -> str:
    """Return a valid SPSS variable name, preserving the original via column_labels later."""
    if not name:
        base = "var"
    else:
        # Replace spaces and any character that is not alphanumeric, @, #, $, _, or .
        base = re.sub(r"[^A-Za-z0-9@#$_.]", "_", str(name))
        # SPSS names must start with a letter or @/#/$; underscore is not allowed at the start.
        if base and base[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@#$":
            base = "v" + base

    # Truncate to leave room for a uniqueness suffix (_nnn)
    base = base[:_SPSS_NAME_MAX_LEN]

    # Remove trailing underscores/periods and ensure non-empty
    base = base.rstrip("_.") or "var"

    candidate = base
    counter = 1
    while candidate.lower() in {u.lower() for u in used}:
        suffix = f"_{counter}"
        candidate = base[: _SPSS_NAME_MAX_LEN - len(suffix)] + suffix
        counter += 1

    return candidate


@router.post("/blank")
async def create_blank_session():
    """Create a workspace seeded with placeholder columns/rows for manual entry."""
    session_id = str(uuid.uuid4())
    filename = "Untitled workspace"
    col_names = [f"Column_{i + 1}" for i in range(PLACEHOLDER_COLS)]
    df = pd.DataFrame(
        {name: [None] * PLACEHOLDER_ROWS for name in col_names}
    )
    store.save(session_id, df, track_undo=False)
    store.set_filename(session_id, filename)
    columns = [
        {"name": name, "dtype": str(df[name].dtype), "kind": "text"}
        for name in df.columns
    ]
    preview = [{name: None for name in col_names} for _ in range(PLACEHOLDER_ROWS)]
    return {
        "session_id": session_id,
        "filename": filename,
        "rows": len(df),
        "columns": columns,
        "preview": preview,
    }


# ── Cell editing ───────────────────────────────────────────────────────────────

class CellUpdate(BaseModel):
    row_index: int
    column: str
    value: Optional[Any] = None  # string, number, or null from frontend


class ClearCellsRequest(BaseModel):
    cells: list  # [{row_index: int, column: str}, ...]


class SetCellsRequest(BaseModel):
    """Write one value into a selection of cells.

    The counterpart of clear_cells: the same arbitrary cell list, but setting
    a value instead of removing one. Two filters make it cover both things a
    user wants from a selection — replacing a particular value everywhere it
    appears, and filling only the gaps:

      only_blank  write only where the cell is currently missing or empty
      match_value write only where the current value equals this
    """
    cells: list                       # [{row_index: int, column: str}, ...]
    value: Optional[str] = None       # None clears, like clear_cells
    only_blank: bool = False
    match_value: Optional[str] = None


@router.patch("/{session_id}/cell")
async def update_cell(session_id: str, body: CellUpdate):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if body.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{body.column}' not found")
    if body.row_index < 0 or body.row_index >= len(df):
        raise HTTPException(status_code=400, detail=f"Row index {body.row_index} out of range")

    col_dtype = df[body.column].dtype
    val = body.value

    # Coerce to numeric using the column's DECLARED kind, not just its current
    # pandas dtype — a numeric column that's still empty (all-NaN) has dtype
    # "object"/"float64" depending on history, so dtype alone missed typed
    # values on blank numeric columns and left them as bare strings forever
    # (they'd never respect the column's decimals display setting).
    from routers.upload import _detect_kind

    kind_override = store.get_kind_overrides(session_id).get(body.column)
    kind = kind_override or _detect_kind(df[body.column])
    # _detect_kind calls a column of numeric STRINGS "categorical", and calls
    # an empty column that too — so the first number typed into a freshly
    # added column was stored as the string "3", and every number after it as
    # well. Nothing later repairs that: the header's range badge disappears,
    # sorting is lexicographic, and any model built on the column treats its
    # numbers as labels. As long as everything already in the column is a
    # number — including when there is nothing in it at all — a typed number
    # is stored as one. A column holding a genuine word is left alone.
    numeric_so_far = values_are_numeric(df[body.column])
    is_numeric_kind = (
        col_dtype.kind in ("i", "u", "f")
        or kind == "numeric"
        or (numeric_so_far and kind_override is None)
    )

    if val is not None and val != "":
        try:
            if col_dtype.kind in ("i", "u"):
                val = int(float(str(val)))
            elif col_dtype.kind == "f" or is_numeric_kind:
                val = float(str(val).replace(",", "."))
        except (ValueError, TypeError):
            pass  # keep as string
    else:
        val = np.nan  # blank → missing

    df = df.copy()
    df.at[body.row_index, body.column] = val
    store.save(session_id, df)

    stored = df.at[body.row_index, body.column]
    if hasattr(stored, "item"):
        stored = stored.item()
    try:
        if isinstance(stored, float) and (np.isnan(stored) or np.isinf(stored)):
            stored = None
    except (TypeError, ValueError):
        pass

    return {"row_index": body.row_index, "column": body.column, "value": stored}


@router.post("/{session_id}/clear_cells")
async def clear_cells(session_id: str, body: ClearCellsRequest):
    """Clear (set to NaN) multiple cells at once."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    df = df.copy()
    cleared = 0
    for cell in body.cells:
        r = cell.get("row_index") if isinstance(cell, dict) else None
        c = cell.get("column") if isinstance(cell, dict) else None
        if r is None or c is None:
            continue
        if c not in df.columns or r < 0 or r >= len(df):
            continue
        df.at[r, c] = np.nan
        cleared += 1

    store.save(session_id, df)
    return {"cleared": cleared}


def _is_blank(value) -> bool:
    """Missing, or a string that is empty once trimmed.

    A cell holding "" or "   " reads as a value to pandas but is a gap to the
    person looking at the sheet, and "fill the blanks" has to mean the same
    thing to both.
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def _matches(current, typed: str) -> bool:
    """Does this cell hold the value the user typed?

    The user types what the grid shows them. A float column shows 0 and holds
    0.0, so comparing str(0.0) with "0" would miss every cell in the one case
    this is most used for — recoding a 0/1 column. Compare numerically when
    both sides are numbers, and as trimmed text otherwise.
    """
    typed = typed.strip()
    try:
        return float(current) == float(typed)
    except (TypeError, ValueError):
        return str(current).strip() == typed


@router.post("/{session_id}/set_cells")
async def set_cells(session_id: str, body: SetCellsRequest):
    """Write one value into a selection of cells."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if body.only_blank and body.match_value is not None:
        raise HTTPException(
            status_code=422,
            detail="Use only_blank or match_value, not both: a blank cell "
                   "cannot also equal a value.",
        )

    df = df.copy()
    changed = 0
    skipped = 0
    for cell in body.cells:
        r = cell.get("row_index") if isinstance(cell, dict) else None
        c = cell.get("column") if isinstance(cell, dict) else None
        if r is None or c is None:
            continue
        if c not in df.columns or r < 0 or r >= len(df):
            continue

        current = df.at[r, c]
        if body.only_blank and not _is_blank(current):
            skipped += 1
            continue
        if body.match_value is not None:
            if _is_blank(current) or not _matches(current, body.match_value):
                skipped += 1
                continue

        if body.value is None:
            df.at[r, c] = np.nan
            changed += 1
            continue

        # Keep a numeric column numeric where the typed value allows it;
        # writing "3" into a float column as text would turn the whole column
        # into strings and silently break every model built on it.
        written = body.value
        if pd.api.types.is_numeric_dtype(df[c]):
            try:
                written = float(body.value)
            except ValueError:
                df[c] = df[c].astype(object)
        df.at[r, c] = written
        changed += 1

    store.save(session_id, df)
    return {"changed": changed, "skipped": skipped}


@router.delete("/{session_id}/row/{row_index}")
async def delete_row(session_id: str, row_index: int):
    """Delete a specific row containing an outlier."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # row_index from the frontend is already a 0-based position (store.delete_row
    # treats it positionally). The previous "-1" silently deleted the wrong row
    # (and made the first row undeletable).
    if row_index < 0:
        raise HTTPException(status_code=400, detail="Invalid row index")

    success = store.delete_row(session_id, row_index)
    if not success:
        raise HTTPException(status_code=400, detail="Row could not be deleted")
        
    return _session_preview(store.get(session_id), session_id)


class ReorderColumnsRequest(BaseModel):
    columns: list  # ordered list of column names


@router.post("/{session_id}/reorder_columns")
async def reorder_columns(session_id: str, body: ReorderColumnsRequest):
    """Reorder DataFrame columns to match frontend drag-and-drop order."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    new_order = [c for c in body.columns if c in df.columns]
    # Append any columns that weren't in the request (safety)
    for c in df.columns:
        if c not in new_order:
            new_order.append(c)

    df = df[new_order]
    store.save(session_id, df)
    return {"columns": list(df.columns)}


# ── Export ─────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/export")
async def export_dataset(
    session_id: str,
    fmt: str = Query("csv", pattern="^(csv|tsv|xlsx|sav)$"),
    filename: str = Query("data"),
    col_kinds: str = Query("{}"),   # JSON: {"colName": "numeric"|"categorical"|"boolean"|"text"}
):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Strip any extension the user might have included
    base = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Build Content-Disposition header safely for non-ASCII filenames (Turkish, etc.)
    from urllib.parse import quote
    ascii_base = base.encode("ascii", errors="replace").decode("ascii")  # fallback for latin-1
    utf8_base = quote(base, safe="")  # RFC 5987 percent-encoded
    def _cd(ext: str) -> dict:
        return {"Content-Disposition": f"attachment; filename=\"{ascii_base}.{ext}\"; filename*=UTF-8''{utf8_base}.{ext}"}

    if fmt == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        content = buf.getvalue().encode("utf-8-sig")  # BOM for Excel compat
        return Response(content=content, media_type="text/csv", headers=_cd("csv"))

    if fmt == "tsv":
        buf = io.StringIO()
        df.to_csv(buf, index=False, sep="\t")
        content = buf.getvalue().encode("utf-8-sig")
        return Response(content=content, media_type="text/tab-separated-values", headers=_cd("tsv"))

    if fmt == "xlsx":
        col_metadata = store.get_metadata(session_id)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")

            # Build Value Labels sheet if any column has user-defined labels
            vl_rows = []
            for col in df.columns:
                user_labels = (col_metadata.get(col, {}) or {}).get("value_labels", {})
                if user_labels:
                    for val, label in sorted(user_labels.items(), key=lambda x: str(x[0])):
                        if label:  # skip empty labels
                            vl_rows.append({"Column": col, "Value": val, "Label": label})
            if vl_rows:
                vl_df = pd.DataFrame(vl_rows)
                vl_df.to_excel(writer, index=False, sheet_name="Value Labels")

        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=_cd("xlsx"),
        )

    if fmt == "sav":
        import pyreadstat

        try:
            kinds: dict = json.loads(col_kinds)
        except Exception:
            kinds = {}

        # Load user-defined value labels from session metadata
        col_metadata = store.get_metadata(session_id)

        # Build a clean copy of the dataframe suitable for pyreadstat
        df_sav = df.copy()

        from routers.upload import _detect_kind

        kind_overrides = store.get_kind_overrides(session_id)

        # SPSS variable names are restrictive. Sanitize them and keep original names as labels.
        used_names: set = set()
        name_map: dict = {}
        for col in df_sav.columns:
            sanitized = _sanitize_spss_name(col, used_names)
            used_names.add(sanitized)
            name_map[col] = sanitized

        df_sav.rename(columns=name_map, inplace=True)

        column_labels: dict = {}
        variable_measure: dict = {}
        variable_value_labels: dict = {}
        missing_ranges: dict = {}

        for original_col, sav_col in name_map.items():
            kind = kinds.get(original_col) or kind_overrides.get(original_col) or _detect_kind(df_sav[sav_col])
            if kind not in ("categorical", "text", "ordinal") and df_sav[sav_col].dtype == object:
                df_sav[sav_col] = pd.to_numeric(df_sav[sav_col], errors="coerce")

            metadata = col_metadata.get(original_col, {}) or {}
            label = metadata.get("label")
            if original_col != sav_col:
                # Preserve the original column name; append any user label after a separator.
                column_labels[sav_col] = f"{original_col} | {label}" if label else original_col
            elif label:
                column_labels[sav_col] = str(label)

            variable_measure[sav_col] = _measure_for_export(kind, metadata)

            user_labels = metadata.get("value_labels", {})
            labels = _sav_value_labels(user_labels, df_sav[sav_col])
            if labels:
                variable_value_labels[sav_col] = labels
            elif kind in ("categorical", "text", "ordinal") and pd.api.types.is_numeric_dtype(df_sav[sav_col]):
                unique_vals = sorted(df_sav[sav_col].dropna().unique())
                variable_value_labels[sav_col] = {float(v): str(v) for v in unique_vals}

            user_missing = _sav_missing_ranges(metadata.get("missing_ranges"), df_sav[sav_col])
            if user_missing:
                missing_ranges[sav_col] = user_missing

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sav")
        os.close(tmp_fd)
        try:
            pyreadstat.write_sav(
                df_sav,
                tmp_path,
                column_labels=column_labels if column_labels else None,
                variable_measure=variable_measure,
                variable_value_labels=variable_value_labels if variable_value_labels else None,
                missing_ranges=missing_ranges if missing_ranges else None,
            )
            with open(tmp_path, "rb") as f:
                content = f.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"SAV export failed: {exc}") from exc
        finally:
            os.unlink(tmp_path)

        return Response(content=content, media_type="application/octet-stream", headers=_cd("sav"))


# ── Select Cases ────────────────────────────────────────────────────────────────

class SelectCasesRequest(BaseModel):
    conditions: list  # [{column, operator, value, join}]
    apply: bool = True


@router.post("/{session_id}/select_cases")
def select_cases(session_id: str, body: SelectCasesRequest):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    from services.store import _apply_conditions, validate_conditions
    validate_conditions(df, body.conditions)
    df_filtered = _apply_conditions(df, body.conditions)
    if body.apply:
        store.save_filter(session_id, body.conditions)
        store.log_action(session_id, "case_filter", {
            "conditions": body.conditions,
            "selected": len(df_filtered),
            "total": len(df),
        })
    return {"selected": len(df_filtered), "total": len(df), "applied": body.apply}


@router.delete("/{session_id}/select_cases")
def clear_cases(session_id: str):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    store.clear_filter(session_id)
    store.log_action(session_id, "case_filter_cleared")
    return {"selected": len(df), "total": len(df)}


# ── File Export ─────────────────────────────────────────────────────────────

@router.get("/{session_id}/export/csv")
def export_csv(session_id: str, filename: str = Query("export.csv")):
    """Export session data as CSV file."""
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Stream CSV directly instead of loading into memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/{session_id}/export/xlsx")
def export_xlsx(session_id: str, filename: str = Query("export.xlsx")):
    """Export session data as XLSX file."""
    df = store.get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    import importlib.util
    if importlib.util.find_spec("openpyxl") is None:
        raise HTTPException(status_code=400, detail="XLSX export requires openpyxl")

    # Write to bytes buffer
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
    excel_buffer.seek(0)

    return StreamingResponse(
        iter([excel_buffer.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ── Session Save/Load ─────────────────────────────────────────────────────────

@router.get("/{session_id}/save_session")
async def save_session(session_id: str):
    """Export the full session as a downloadable JSON file."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build columns metadata (same shape as upload response). User-driven kind
    # overrides win over auto-detection so the dictionary classification
    # survives the save/load round-trip.
    from routers.upload import _detect_kind
    kind_overrides = store.get_kind_overrides(session_id)
    columns = []
    for col in df.columns:
        kind = kind_overrides.get(col) or _detect_kind(df[col])
        columns.append({"name": col, "dtype": str(df[col].dtype), "kind": kind})

    # User-chosen display name (set via /rename) wins over the hardcoded
    # fallback so the saved JSON round-trips the rename.
    user_filename = store.get_filename(session_id) or f"session_{session_id[:8]}.json"
    payload = {
        "version": "1.2",
        "filename": user_filename,
        "created": time.time(),
        "columns": columns,
        "col_metadata": store.get_metadata(session_id),
        "kind_overrides": kind_overrides,
        "decimals_overrides": store.get_decimals(session_id),
        "case_filter": store.get_filter(session_id),
        "audit": store.get_audit(session_id),
        "data": json.loads(
            df.replace([np.inf, -np.inf], np.nan).to_json(
                orient="records", date_format="iso", default_handler=str
            )
        ),
    }

    content = json.dumps(payload, allow_nan=False, default=str).encode("utf-8")

    # Download filename should follow the user's dataset name (e.g. "svo"),
    # not the generic session-id fallback — otherwise every re-save silently
    # renames the file back to "session_xxxxxxxx.json". Strip any existing
    # extension from the stored name so we don't double it (".json.json").
    base = user_filename.rsplit(".", 1)[0] if "." in user_filename else user_filename
    from urllib.parse import quote
    ascii_base = base.encode("ascii", errors="replace").decode("ascii")
    utf8_base = quote(base, safe="")

    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f"attachment; filename=\"{ascii_base}.json\"; filename*=UTF-8''{utf8_base}.json"
            )
        },
    )


@router.post("/load_session")
async def load_session(file: UploadFile = File(...)):
    """Restore a session from a previously saved JSON file."""
    raw = await file.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    if "data" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'data' key in session file")

    df = pd.DataFrame(payload["data"])
    # JSON round-trips serialise numbers with default_handler=str, so genuinely
    # numeric columns can come back as strings (object dtype) and then show as
    # 'text'. Losslessly restore numeric dtype where every value parses.
    from routers.upload import coerce_numeric_objects
    df = coerce_numeric_objects(df)
    new_session_id = str(uuid.uuid4())
    store.save(new_session_id, df)

    # Restore filters if present
    case_filter = payload.get("case_filter", [])
    if case_filter:
        store.save_filter(new_session_id, case_filter)

    # Restore column metadata if present
    col_metadata = payload.get("col_metadata", {})
    if col_metadata:
        store.save_metadata(new_session_id, col_metadata)

    # Restore user-driven kind overrides (v1.1+ session files). For older v1.0
    # files we fall back to the "columns" array on the payload, which already
    # carries the kind the user saw at save time.
    # Only fall back to the columns array for genuine v1.0 files that LACK a
    # kind_overrides block. v1.1+ files always carry the key (possibly an empty
    # dict when the user set no overrides) — using an empty dict as the trigger
    # would wrongly pin every auto-detected kind, including stale 'text'/
    # 'categorical' on numeric columns, defeating the lossless coercion above.
    kind_overrides = payload.get("kind_overrides")
    if kind_overrides is None and isinstance(payload.get("columns"), list):
        kind_overrides = {c["name"]: c["kind"] for c in payload["columns"] if c.get("name") and c.get("kind")}
        # Drop stale auto-detected kinds that conflict with a now-numeric column
        # (object-typed numbers were classified text/categorical at save time).
        kind_overrides = {
            k: v for k, v in kind_overrides.items()
            if not (
                v in ("text", "categorical")
                and k in df.columns
                and pd.api.types.is_numeric_dtype(df[k])
                and df[k].dropna().nunique() > 2
            )
        }
    if kind_overrides:
        store.set_kind_overrides(new_session_id, kind_overrides)

    # Restore per-column decimal-places overrides (v1.2+ session files).
    decimals_overrides = payload.get("decimals_overrides") or {}
    if decimals_overrides:
        store.save_decimals(new_session_id, decimals_overrides)

    # Restore column metadata (labels, units, value_labels set at recode
    # time) so the Data Dictionary + legends repopulate after reload.
    col_metadata = payload.get("col_metadata") or {}
    if col_metadata:
        store.save_metadata(new_session_id, col_metadata)

    # Restore user-chosen display name so subsequent save_session calls
    # keep round-tripping the rename.
    restored_filename = payload.get("filename")
    if restored_filename:
        store.set_filename(new_session_id, restored_filename)

    # Build columns info — overrides win over auto-detection.
    from routers.upload import _detect_kind
    overrides = store.get_kind_overrides(new_session_id)
    columns = []
    for col in df.columns:
        kind = overrides.get(col) or _detect_kind(df[col])
        columns.append({"name": col, "dtype": str(df[col].dtype), "kind": kind})
    _attach_value_labels(columns, new_session_id)

    preview = json.loads(
        df.head(2000).replace([np.inf, -np.inf], np.nan).to_json(
            orient="records", default_handler=str, date_format="iso", date_unit="s"
        )
    )

    return {
        "session_id": new_session_id,
        "filename": payload.get("filename", file.filename),
        "rows": len(df),
        "columns": columns,
        "preview": preview,
        "case_filter": {
            "conditions": case_filter,
            "selected": len(store.get_filtered(new_session_id)),
            "total": len(df),
        } if case_filter else None,
    }


# ── Audit ─────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/audit")
async def get_audit(session_id: str):
    """Return the audit trail for a session."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return store.get_audit(session_id)


# ── Undo / Redo ──────────────────────────────────────────────────────────────

def _attach_value_labels(columns: list, session_id: str) -> list:
    """Merge persisted per-column metadata (set at recode/import time via the
    metadata endpoint) into the column objects, so the frontend Data
    Dictionary and legends see them after a refresh / reload."""
    meta = store.get_metadata(session_id) or {}
    for c in columns:
        m = meta.get(c.get("name"), {}) or {}
        for key in (
            "label",
            "description",
            "units",
            "role",
            "value_labels",
            "missing_ranges",
            "missing_user_values",
            "measure",
        ):
            if m.get(key):
                c[key] = m.get(key)
        # Per-column flags that drive the data tab + analysis pickers.
        if m.get("analysis_excluded") is not None:
            c["analysis_excluded"] = bool(m.get("analysis_excluded"))
        if m.get("display_name"):
            c["display_name"] = m.get("display_name")
    return columns


def _session_preview(df: pd.DataFrame, session_id: str | None = None) -> dict:
    """Build a session-like response from a DataFrame for frontend state update."""
    import json as _json
    # User-set kind overrides (data-tab badge / dictionary) must win over
    # auto-detection — otherwise a GET / refresh / undo silently reverts a
    # numeric-coded categorical (e.g. LDL groups 1/2/3) back to 'numeric'.
    overrides = store.get_kind_overrides(session_id) if session_id else {}
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "datetime" in dtype or "timedelta" in dtype:
            kind = "date"
        elif dtype.startswith("int") or dtype.startswith("float"):
            unique_vals = set(df[col].dropna().unique())
            kind = "categorical" if len(unique_vals) <= 2 else "numeric"
        elif dtype == "bool":
            kind = "categorical"
        else:
            kind = "categorical" if df[col].nunique() <= 50 else "text"
        columns.append({"name": col, "dtype": dtype, "kind": overrides.get(col, kind)})
    if session_id:
        _attach_value_labels(columns, session_id)
    preview_df = df.head(2000).replace([np.inf, -np.inf], np.nan)
    preview = _json.loads(preview_df.to_json(orient="records", default_handler=str, date_format="iso", date_unit="s"))
    case_filter = None
    if session_id:
        conditions = store.get_filter(session_id)
        if conditions:
            case_filter = {
                "conditions": conditions,
                "selected": len(store.get_filtered(session_id)),
                "total": len(df),
            }
    return {
        "rows": len(df),
        "columns": columns,
        "preview": preview,
        "case_filter": case_filter,
    }


@router.post("/{session_id}/undo")
async def undo_action(session_id: str):
    """Undo the last data mutation (backend DataFrame + return refreshed preview)."""
    restored = store.undo(session_id)
    if restored is None:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    store.log_action(session_id, "undo")
    result = _session_preview(restored, session_id)
    result["undo_depth"] = store.undo_depth(session_id)
    result["redo_depth"] = store.redo_depth(session_id)
    return result


@router.post("/{session_id}/redo")
async def redo_action(session_id: str):
    """Redo the last undone mutation."""
    restored = store.redo(session_id)
    if restored is None:
        raise HTTPException(status_code=400, detail="Nothing to redo")
    store.log_action(session_id, "redo")
    result = _session_preview(restored, session_id)
    result["undo_depth"] = store.undo_depth(session_id)
    result["redo_depth"] = store.redo_depth(session_id)
    return result


# ── Column Metadata ──────────────────────────────────────────────────────────

class ColumnMetadataRequest(BaseModel):
    columns: Dict[str, dict]  # {COL_NAME: {label, units, role, value_labels, description}}


@router.post("/{session_id}/metadata")
async def save_metadata(session_id: str, body: ColumnMetadataRequest):
    """Store column-level metadata for the session."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    store.save_metadata(session_id, body.columns)
    store.log_action(session_id, "metadata_updated", {"columns": list(body.columns.keys())})

    return {"status": "ok", "columns_updated": list(body.columns.keys())}


@router.get("/{session_id}/name_suggestions")
async def name_suggestions(session_id: str):
    """Suggest readable Sentence-case names for the session's columns.

    Advisory only — the frontend shows old → new and renames on confirmation.
    Returns {column: suggested_name} omitting columns where the suggestion
    equals the current name.
    """
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    from services.naming import suggest_names
    return {"suggestions": suggest_names(list(df.columns))}


# ── Column kind override ─────────────────────────────────────────────────────

class KindOverrideRequest(BaseModel):
    column: str
    kind: str  # "numeric" | "categorical" | "ordinal" | "text" | "date" | "boolean"


@router.post("/{session_id}/kind")
async def set_column_kind(session_id: str, body: KindOverrideRequest):
    """Persist a user-driven kind change (data-tab badge / dictionary)."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if body.column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{body.column}' not in session")
    if body.kind not in ("numeric", "categorical", "ordinal", "text", "date", "boolean"):
        raise HTTPException(status_code=422, detail=f"Invalid kind '{body.kind}'")

    store.save_kind_overrides(session_id, {body.column: body.kind})
    store.log_action(session_id, "kind_override", {"column": body.column, "kind": body.kind})
    return {"status": "ok", "column": body.column, "kind": body.kind}


# ── Decimal-places override ──────────────────────────────────────────────────
# Per-column display precision the user picks via the data-tab context menu.
# Persisted so save_session round-trips the formatting choice.

class DecimalRequest(BaseModel):
    column: str
    decimals: Optional[int] = None  # None ⇒ clear the override


@router.post("/{session_id}/decimals")
async def set_column_decimals(session_id: str, body: DecimalRequest):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if body.column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{body.column}' not in session")
    if body.decimals is None:
        store.clear_decimal(session_id, body.column)
        return {"status": "ok", "column": body.column, "decimals": None}
    if not (0 <= int(body.decimals) <= 10):
        raise HTTPException(status_code=422, detail="decimals must be between 0 and 10")
    store.set_decimal(session_id, body.column, int(body.decimals))
    return {"status": "ok", "column": body.column, "decimals": int(body.decimals)}


@router.get("/{session_id}/decimals")
async def get_column_decimals(session_id: str):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return store.get_decimals(session_id)


# ── Session rename ────────────────────────────────────────────────────────────
# Lets the user choose a display name for the session — surfaced in the
# header pill, the auto-save IndexedDB record, and the round-tripped JSON.

class RenameRequest(BaseModel):
    filename: str


@router.post("/{session_id}/rename")
async def rename_session(session_id: str, body: RenameRequest):
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    name = (body.filename or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="filename cannot be empty")
    if len(name) > 200:
        raise HTTPException(status_code=422, detail="filename too long (>200 chars)")
    store.set_filename(session_id, name)
    return {"status": "ok", "filename": name}


@router.get("/{session_id}")
async def get_session_info(session_id: str):
    """Retrieve session details (filename, rows, columns, preview) for a saved session ID."""
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from routers.upload import _detect_kind
    kind_overrides = store.get_kind_overrides(session_id)
    columns = []
    for col in df.columns:
        kind = kind_overrides.get(col) or _detect_kind(df[col])
        columns.append({"name": col, "dtype": str(df[col].dtype), "kind": kind})
    _attach_value_labels(columns, session_id)

    import numpy as np
    import json as _json
    preview = _json.loads(
        df.head(2000).replace([np.inf, -np.inf], np.nan).to_json(
            orient="records", default_handler=str, date_format="iso", date_unit="s"
        )
    )

    return {
        "session_id": session_id,
        "filename": "iptw_weighted_cohort.csv" if session_id.endswith("_iptw") else "psm_matched_cohort.csv" if session_id.endswith("_psm") else f"session_{session_id[:8]}.csv",
        "rows": len(df),
        "columns": columns,
        "preview": preview,
    }
