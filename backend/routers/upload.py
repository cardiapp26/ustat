import io
import math
import os
import re
import tempfile
import uuid
from numbers import Integral, Real
from typing import Any

import pandas as pd
import pyreadstat
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from loguru import logger
from services import store
from services.ingest_coercion import coerce_with_report

router = APIRouter()

# Hard cap on a single uploaded dataset. Protects the in-memory store from
# being exhausted by an oversized (or hostile) file. Override via env.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))  # 100 MB

# Date/time patterns for auto-detection
_DATE_PATTERNS = [
    re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$"),        # 01/02/2024, 1-2-24
    re.compile(r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}$"),           # 2024-01-02
    re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$"),                     # 01:29:00, 1:29
    re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\s+\d{1,2}:\d{2}"),  # 01/02/2024 13:45
    re.compile(r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}[T ]\d{1,2}:\d{2}"),   # 2024-01-02T13:45
]


def coerce_numeric_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Restore numeric dtype for object columns whose values are numeric.

    Kept for callers that only want the frame -- notably the session-load path,
    where a JSON round-trip has turned genuine numbers back into strings and
    there is no user to show an import report to. Everything this does, and
    every reason it might decline to convert a column, lives in
    ``services.ingest_coercion``; see that module's docstring for why a value
    at a measurement limit (``<0.1``) is not treated as dirty missingness.
    """
    frame, _report, _preserved = coerce_with_report(df)
    return frame


def _detect_kind(series: pd.Series) -> str:
    """Detect column kind with date/time and binary auto-detection."""
    import datetime as _dt
    dtype = str(series.dtype)

    # Already a datetime dtype (pandas parsed it)
    if "datetime" in dtype or "timedelta" in dtype:
        return "date"

    if dtype == "bool":
        return "categorical"  # treat bool as categorical

    if dtype.startswith("int") or dtype.startswith("float"):
        # Binary detection: if only 2 unique non-null values (typically 0/1)
        # → treat as categorical (e.g. SEX, DM, EXITUS)
        unique_vals = set(series.dropna().unique())
        if len(unique_vals) <= 2:
            return "categorical"
        return "numeric"

    # Object column: check for datetime.time / datetime.date / datetime.datetime objects
    # (SPSS/SAS often store these as Python objects, not pandas datetime)
    sample_vals = series.dropna().head(20)
    if len(sample_vals) > 0:
        first_nonnull = sample_vals.iloc[0]
        if isinstance(first_nonnull, (_dt.time, _dt.date, _dt.datetime)):
            return "date"

    # For object/string columns: check if values look like dates/times.
    # Numeric-separator/ISO/time forms via regex, plus TR/EN month-name dates
    # via the date parser. Pure numbers are NOT treated as dates here so Excel
    # serial numbers / integer IDs are never mislabelled (serial parsing stays
    # opt-in through the 'Parse as date' tool).
    from services.date_parser import parse_one
    _pure_num = re.compile(r"^-?\d+(\.\d+)?$")
    sample = series.dropna().head(50).astype(str)
    if len(sample) > 0:
        def _looks_date(v: str) -> bool:
            v = v.strip()
            if _pure_num.match(v):
                return False
            return any(p.match(v) for p in _DATE_PATTERNS) or parse_one(v) is not None
        matches = sum(1 for v in sample if _looks_date(v))
        if matches / len(sample) >= 0.7:  # ≥70% match → date
            return "date"

    # String binary detection: Yes/No, True/False, M/F, etc.
    unique_str = set(series.dropna().astype(str).str.strip().str.lower().unique())
    if len(unique_str) <= 2:
        return "categorical"

    n_unique = series.nunique()
    return "categorical" if n_unique <= 50 else "text"

SUPPORTED = {
    "csv": "text/csv",
    "xlsx": "excel",
    "xls": "excel",
    "sas7bdat": "sas",
    "sav": "spss",
    "dta": "stata",
}


def _json_scalar(value: Any) -> Any:
    """Return a JSON-safe scalar while keeping numeric metadata usable."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return value
    return value


def _metadata_key(value: Any) -> str:
    value = _json_scalar(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def _labels_by_name(meta: Any) -> dict:
    raw = getattr(meta, "variable_labels", None)
    column_names = getattr(meta, "column_names", []) or []
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {name: label for name, label in zip(column_names, raw)}

    raw = getattr(meta, "column_names_to_labels", None)
    if isinstance(raw, dict):
        return raw

    raw = getattr(meta, "column_labels", None)
    if isinstance(raw, list):
        return {name: label for name, label in zip(column_names, raw)}
    return {}


def _normalise_missing_ranges(raw_ranges: Any) -> list:
    out = []
    for item in raw_ranges or []:
        if isinstance(item, dict):
            lo = item.get("lo")
            hi = item.get("hi", lo)
        else:
            lo = getattr(item, "lo", item)
            hi = getattr(item, "hi", lo)
        lo = _json_scalar(lo)
        hi = _json_scalar(hi)
        if lo is None:
            continue
        out.append({"lo": lo, "hi": hi if hi is not None else lo})
    return out


def _extract_readstat_metadata(meta: Any) -> dict[str, dict]:
    if meta is None:
        return {}

    variable_labels = _labels_by_name(meta)
    variable_value_labels = getattr(meta, "variable_value_labels", None) or {}
    missing_ranges = getattr(meta, "missing_ranges", None) or {}
    missing_user_values = getattr(meta, "missing_user_values", None) or {}
    variable_measure = getattr(meta, "variable_measure", None) or {}
    if not isinstance(variable_value_labels, dict):
        variable_value_labels = {}
    if not isinstance(missing_ranges, dict):
        missing_ranges = {}
    if not isinstance(missing_user_values, dict):
        missing_user_values = {}
    if not isinstance(variable_measure, dict):
        variable_measure = {}

    columns = set(variable_labels) | set(variable_value_labels) | set(missing_ranges) | set(missing_user_values) | set(variable_measure)
    out: dict[str, dict] = {}
    for col in columns:
        entry: dict[str, Any] = {}

        label = variable_labels.get(col)
        if label:
            entry["label"] = str(label)

        value_labels = variable_value_labels.get(col)
        if value_labels:
            entry["value_labels"] = {_metadata_key(k): str(v) for k, v in value_labels.items() if v is not None}

        ranges = _normalise_missing_ranges(missing_ranges.get(col))
        if ranges:
            entry["missing_ranges"] = ranges

        user_values = missing_user_values.get(col)
        if user_values:
            entry["missing_user_values"] = [_json_scalar(v) for v in user_values]

        measure = variable_measure.get(col)
        if measure:
            entry["measure"] = str(measure)

        if entry:
            out[col] = entry
    return out


def _kind_with_imported_metadata(series: pd.Series, metadata: dict) -> str:
    measure = str(metadata.get("measure", "")).strip().lower()
    if measure == "ordinal":
        return "ordinal"
    if measure == "nominal":
        return "categorical"
    if measure == "scale" and pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if metadata.get("value_labels") and measure != "scale":
        return "categorical"
    return _detect_kind(series)



# ── Excel cell notes as value labels ───────────────────────────────────────────

# "0:Benign", "3 = Hurthle", "2) Foliküler", "1 - Papiller". A bare hyphen only
# counts when a space follows it, so a label like "1-2 kez" is not read as the
# code 1 meaning "2 kez".
_CODE_LINE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*(?::|=|\)|\||\t|(?<=\d)\s[-\u2013]\s)\s*(\S.*?)\s*$")
# Legends live at the top of a sheet; scanning the whole thing would read every
# stray note on a 50 000-row file for nothing.
_NOTE_SCAN_ROWS = 30
# A note only becomes value labels if the codes actually describe the column.
# Otherwise a passing remark on a continuous variable would relabel it and, via
# _kind_with_imported_metadata, turn it categorical.
_COVERAGE = 0.8


def _parse_code_lines(text: str) -> dict[str, str]:
    """Pull `code: label` pairs out of a free-text note.

    Lines that do not look like a coding line are ignored rather than fought
    with — Excel puts the author's name on the first line, and people write a
    sentence above the list.
    """
    out: dict[str, str] = {}
    for line in str(text).replace("\r", "\n").split("\n"):
        m = _CODE_LINE.match(line)
        if m:
            out[_metadata_key(float(m.group(1).replace(",", ".")))] = m.group(2).strip()
    return out


def _covers_column(series: pd.Series, codes: dict[str, str]) -> bool:
    """Do these codes describe what is actually in the column?"""
    values = series.dropna()
    if values.empty:
        return False
    seen = {_metadata_key(v) for v in values.unique()}
    return len(seen & set(codes)) / len(seen) >= _COVERAGE


def _excel_note_metadata(content: bytes, df: pd.DataFrame) -> dict[str, dict]:
    """Value labels written as Excel cell notes.

    SPSS, Stata and SAS carry value labels in the file and uSTAT already reads
    them. An .xlsx has nowhere to put them, so people type the coding scheme
    into a note on the header cell — and pandas drops notes entirely, which is
    how a column of 0s and 1s arrives with nothing to say what they mean.

    Notes are matched to columns by position, which is how pandas reads them:
    an empty spreadsheet column still becomes an `Unnamed: n` column, so the
    two stay aligned.
    """
    try:
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        ws = wb.worksheets[0]
    except Exception as exc:  # .xls, encrypted, or a format openpyxl declines
        logger.debug(f"no Excel notes read: {exc}")
        return {}

    by_index: dict[int, list[tuple[int, str]]] = {}
    try:
        for row in ws.iter_rows(min_row=1, max_row=_NOTE_SCAN_ROWS):
            for cell in row:
                note = getattr(cell, "comment", None)
                if note is not None and getattr(note, "text", None):
                    by_index.setdefault(cell.column, []).append((cell.row, note.text))
    except Exception as exc:
        logger.debug(f"Excel note scan failed: {exc}")
        return {}
    finally:
        wb.close()

    out: dict[str, dict] = {}
    for position, col in enumerate(df.columns, start=1):
        notes = by_index.get(position)
        if not notes:
            continue
        # The header's own note wins; a note further down is a fallback.
        notes.sort(key=lambda rt: rt[0])
        text = notes[0][1]
        codes = _parse_code_lines(text)
        if len(codes) >= 2 and _covers_column(df[col], codes):
            out[col] = {"value_labels": codes, "measure": "nominal"}
        else:
            # Not a coding scheme — but the note still says something about the
            # variable, and a description is better kept than dropped.
            flat = " ".join(str(text).split())
            if flat:
                out[col] = {"label": flat[:300]}
    return out


def _read(filename: str, content: bytes) -> tuple[pd.DataFrame, dict[str, dict]]:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(io.BytesIO(content)), {}
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(content))
        return df, _excel_note_metadata(content, df)
    elif ext in ("sas7bdat", "sav", "dta"):
        # pyreadstat requires a real file path, not BytesIO
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            if ext == "sas7bdat":
                df, meta = pyreadstat.read_sas7bdat(tmp_path)
            elif ext == "sav":
                df, meta = pyreadstat.read_sav(tmp_path)
                _, meta = pyreadstat.read_sav(tmp_path, metadataonly=True, user_missing=True)
            elif ext == "dta":
                df, meta = pyreadstat.read_dta(tmp_path)
        finally:
            os.unlink(tmp_path)
        return df, _extract_readstat_metadata(meta)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


@router.post("/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    _max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    # Cheap pre-check on the declared size (rejects before reading the body).
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {_max_mb} MB.")
    # Hard cap on the bytes actually read — defends against a missing or spoofed
    # Content-Length. Read one byte past the limit; if we got it, it's too big.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {_max_mb} MB.")
    try:
        df, imported_metadata = _read(file.filename, content)
    except Exception as e:
        logger.exception("upload: failed to parse {}", file.filename)
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")

    # Pass over object columns: salvage numeric ones that arrived dirty
    # (comma-decimals, text-as-missing sentinels). Without this, a single
    # "30,6" cell or "NA" pinned the whole column to text and every later
    # statistical endpoint either crashed or silently dropped rows.
    #
    # The report is the other half of that bargain. The salvage rewrites cells,
    # and a rewrite nobody is told about is indistinguishable from data loss --
    # which is what it was for a value at a measurement limit. Every change,
    # its count, an example of it, and the verbatim original of anything
    # blanked come back with the upload and are kept with the session.
    df, ingest_report, preserved_cells = coerce_with_report(df)

    session_id = str(uuid.uuid4())
    store.save(session_id, df)
    store.save_ingest_report(session_id, ingest_report, preserved_cells)
    if imported_metadata:
        store.save_metadata(session_id, imported_metadata)
    # Persist the uploaded filename so subsequent save_session snapshots
    # embed it (and resume restores it). Without this, get_filename returns
    # None and save_session falls back to "session_{id}.json", which diverges
    # from the original dataset name and spawns duplicate Recent Sessions
    # cards (same data, different display name) that the name-based dedupe
    # in sessionDb.ts cannot collapse.
    store.set_filename(session_id, file.filename)

    columns = []
    kind_overrides = {}
    for col in df.columns:
        col_metadata = imported_metadata.get(col, {})
        detected_kind = _detect_kind(df[col])
        kind = _kind_with_imported_metadata(df[col], col_metadata)
        if kind != detected_kind:
            kind_overrides[col] = kind
        col_obj = {"name": col, "dtype": str(df[col].dtype), "kind": kind}
        for key in ("label", "value_labels", "missing_ranges", "missing_user_values", "measure"):
            if key in col_metadata:
                col_obj[key] = col_metadata[key]
        columns.append(col_obj)
    if kind_overrides:
        store.save_kind_overrides(session_id, kind_overrides)

    # Use pandas to_json → loads to guarantee NaN/Inf become null
    import numpy as np
    import json as _json
    preview_df = df.head(2000).replace([np.inf, -np.inf], np.nan)
    preview = _json.loads(preview_df.to_json(orient="records", default_handler=str, date_format="iso", date_unit="s"))

    return {
        "session_id": session_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": columns,
        "preview": preview,
        "ingest_report": ingest_report,
    }


@router.get("/{session_id}/ingest_report")
def ingest_report(session_id: str):
    """What the import changed, re-readable for as long as the session lives.

    The upload response carries the same object, but a user who dismissed the
    review panel (or reloaded) still has to be able to answer "which cells did
    it blank, and what did they say?" without re-uploading the file.
    """
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    preserved = store.get_preserved_cells(session_id)
    return {
        "session_id": session_id,
        "report": store.get_ingest_report(session_id),
        # {column: [{row, value}]} -- the originals of every blanked cell, so
        # the raw file value is recoverable from the session alone.
        "preserved_cells": {
            col: [{"row": int(row), "value": value} for row, value in sorted(cells.items())]
            for col, cells in preserved.items()
        },
    }
