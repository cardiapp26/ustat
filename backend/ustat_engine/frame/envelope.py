"""One filtered dataset, in a form that survives the trip to a worker.

`json.dumps(df)` is not a wire format. A DataFrame carries three things an
analysis depends on and JSON has no room for: the declared kind of each column
(which is NOT its pandas dtype -- a blank numeric column is `object` until
someone types in it), the order its categorical levels are supposed to come out
in, and which rows the active Select Cases filter kept. Drop any one of them and
the browser computes a defensible number for a different question.

So the envelope carries all three explicitly:

  - **kind decides storage, never dtype.** The declared kind is the user's
    statement of what the column is; pandas' dtype is an accident of what has
    been typed into it so far.
  - **both level orderings travel.** `levels` is the order results are
    presented in (`sorted_groups` + `level_key`); `model_levels` is the order
    `pd.get_dummies(drop_first=True)` uses, whose first entry is the reference
    category a coefficient is relative to. They are frequently different, both
    are computed here by calling the same functions the server calls, and
    neither is ever re-derived by the reader.
  - **the filter travels with a fingerprint.** A resident frame in a worker
    outlives the filter it was built under; `frame_from_envelope` stamps the
    fingerprint onto `df.attrs` so `registry.run` can refuse a frame that no
    longer matches the analysis being asked for.

Column names are copied verbatim -- spaces, parentheses, Turkish letters. They
are dictionary keys here, never identifiers.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..errors import EngineError
from ..jsonsafe import sanitize
from .levels import level_key, sorted_groups
from .select import keep_mask

SCHEMA = "ustat.frame/1"

#: Declared kind -> how its values are written into `data`.
STORAGE_BY_KIND = {
    "numeric": "f64",
    "categorical": "cat",
    "ordinal": "cat",
    "date": "f64_epoch_ms",
    "text": "str",
}


def filter_fingerprint(conditions: Optional[List[dict]]) -> str:
    """sha256 over the canonical JSON of the Select Cases conditions.

    Canonical because the same filter must hash the same however the dict that
    described it was built: keys sorted, no incidental whitespace, unicode kept
    as itself rather than escaped.
    """
    canonical = json.dumps(
        conditions or [], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):  # pragma: no cover - exotic scalars
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _dedupe(values: Iterable[str]) -> List[str]:
    """First-occurrence-wins dedupe.

    `level_key` is many-to-one -- 1 and 1.0 are both the level "1" -- so a
    column holding both spellings would otherwise get two identical levels and
    a codes array that could never round-trip.
    """
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _fallback_kind(series: pd.Series) -> str:
    """Used only when the caller's kind map has no entry for a column.

    Callers are expected to pass a complete map (the server merges the user's
    overrides over `_detect_kind`); this exists so a missing entry degrades to
    something usable rather than raising.
    """
    dtype = str(series.dtype)
    if "datetime" in dtype or "timedelta" in dtype:
        return "date"
    if dtype.startswith("int") or dtype.startswith("float"):
        return "numeric"
    return "categorical"


def _numeric_values(series: pd.Series) -> List[Optional[float]]:
    coerced = pd.to_numeric(series, errors="coerce")
    arr = np.asarray(coerced, dtype="float64")
    return [None if not np.isfinite(x) else float(x) for x in arr]


def _epoch_ms_values(series: pd.Series) -> List[Optional[float]]:
    stamps = pd.to_datetime(series, errors="coerce", utc=True)
    return [None if pd.isna(t) else float(t.value) / 1e6 for t in stamps]


def _text_values(series: pd.Series) -> List[Optional[str]]:
    return [None if _is_na(v) else str(v) for v in series]


def _categorical_payload(series: pd.Series) -> tuple[dict, List[str], List[str]]:
    """Codes + both orderings for one categorical/ordinal column."""
    levels = _dedupe(level_key(v) for v in sorted_groups(series))
    position = {lv: i for i, lv in enumerate(levels)}
    codes = [-1 if _is_na(v) else position.get(level_key(v), -1) for v in series]

    # The order `pd.get_dummies(..., drop_first=True)` would build indicators
    # in: pandas factorises through a Categorical, whose categories are the
    # sorted uniques, and drops the first. Computed rather than assumed equal
    # to `levels`, because for string codes it usually is not.
    model_levels = _dedupe(
        level_key(v) for v in pd.Categorical(series.dropna()).categories
    )
    return {"codes": codes, "levels": levels}, levels, model_levels


def build_envelope(
    df_unfiltered: pd.DataFrame,
    *,
    kinds: Optional[dict] = None,
    metadata: Optional[dict] = None,
    decimals: Optional[dict] = None,
    conditions: Optional[List[dict]] = None,
    columns: Optional[Sequence[str]] = None,
) -> dict:
    """Serialise the filtered view of `df_unfiltered` as a `ustat.frame/1` dict.

    `columns=None` sends every column. Naming the columns an analysis declared
    is what turns a t-test into a three-column transfer rather than a copy of
    the whole sheet, so callers that know should say.
    """
    kinds = kinds or {}
    metadata = metadata or {}
    decimals = decimals or {}
    conditions = list(conditions or [])

    if columns is None:
        wanted = list(df_unfiltered.columns)
    else:
        wanted = list(columns)
        missing = [c for c in wanted if c not in df_unfiltered.columns]
        if missing:
            raise EngineError(
                f"Unknown column(s): {', '.join(repr(c) for c in missing)}",
                status_hint=422,
            )

    mask = keep_mask(df_unfiltered, conditions)
    row_index = [int(i) for i in np.flatnonzero(mask)]
    filtered = df_unfiltered[mask] if conditions else df_unfiltered

    column_specs: List[dict] = []
    data: dict = {}
    for name in wanted:
        series = filtered[name]
        kind = kinds.get(name) or _fallback_kind(series)
        storage = STORAGE_BY_KIND.get(kind, "str")
        spec: dict = {"name": name, "kind": kind, "storage": storage}

        if name in decimals and decimals[name] is not None:
            spec["decimals"] = int(decimals[name])
        labels = (metadata.get(name) or {}).get("value_labels")
        if labels:
            spec["value_labels"] = labels

        if storage == "cat":
            payload, levels, model_levels = _categorical_payload(series)
            data[name] = payload
            spec["levels"] = levels
            spec["model_levels"] = model_levels
            spec["reference"] = model_levels[0] if model_levels else None
            if kind == "ordinal":
                spec["ordered"] = True
        elif storage == "f64":
            data[name] = _numeric_values(series)
        elif storage == "f64_epoch_ms":
            data[name] = _epoch_ms_values(series)
        else:
            data[name] = _text_values(series)

        column_specs.append(spec)

    envelope = {
        "schema": SCHEMA,
        "rows": int(len(filtered)),
        "row_index": row_index,
        "filter": {
            "conditions": conditions,
            "fingerprint": filter_fingerprint(conditions),
        },
        "columns": column_specs,
        "data": data,
    }
    # Nothing numpy-typed may leave here: the browser hands this straight to
    # `json.dumps`, where a numpy float64 is a TypeError rather than a number.
    return sanitize(envelope)


def frame_from_envelope(env: dict) -> pd.DataFrame:
    """Rebuild the DataFrame an envelope describes.

    The index is the envelope's `row_index` -- the rows' positions in the
    unfiltered frame -- so a result that names a row names the same row the
    grid does.
    """
    if not isinstance(env, dict) or env.get("schema") != SCHEMA:
        raise EngineError(
            f"expected a {SCHEMA} envelope, got {env.get('schema') if isinstance(env, dict) else type(env).__name__!r}",
            status_hint=422,
        )

    specs = env.get("columns") or []
    data = env.get("data") or {}
    row_index = [int(i) for i in env.get("row_index") or []]
    index = pd.Index(row_index, dtype="int64")

    frame = pd.DataFrame(index=index)
    for spec in specs:
        name = spec["name"]
        storage = spec.get("storage")
        payload = data.get(name)

        if storage == "cat":
            codes = np.asarray((payload or {}).get("codes") or [], dtype="int64")
            levels = list((payload or {}).get("levels") or spec.get("levels") or [])
            frame[name] = pd.Categorical.from_codes(
                codes, categories=levels, ordered=bool(spec.get("ordered", False))
            )
        elif storage == "f64":
            frame[name] = pd.Series(
                np.asarray(payload or [], dtype="float64"), index=index
            )
        elif storage == "f64_epoch_ms":
            frame[name] = pd.to_datetime(
                pd.Series(np.asarray(payload or [], dtype="float64"), index=index),
                unit="ms",
                utc=True,
            )
        else:
            frame[name] = pd.Series(list(payload or []), index=index, dtype="object")

    # The filter this frame was cut with, so a later run can prove it is asking
    # about the same patients. Without it a worker holding a resident frame
    # would happily answer a question about a Select Cases that has since
    # changed, and the answer would look entirely normal.
    frame.attrs["filter_fingerprint"] = (env.get("filter") or {}).get("fingerprint")
    return frame


__all__ = [
    "SCHEMA",
    "STORAGE_BY_KIND",
    "build_envelope",
    "filter_fingerprint",
    "frame_from_envelope",
]
