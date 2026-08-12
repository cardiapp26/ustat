"""Joining a second file onto the open dataset.

A follow-up sheet, a lab export, a second visit — the rows exist somewhere
else and have to be matched onto the ones already open. Doing it in a
spreadsheet with VLOOKUP is where silent damage happens: keys that look equal
but are not (" 1024" and "1024"), a lookup table with duplicate keys that
quietly multiplies rows, and no record afterwards of how many rows actually
found a partner.

So the endpoint refuses to do it blind. `preview` reports what a join would
do — how many keys match, how many duplicate on each side, how many rows the
result would have — and returns it without changing anything. `apply` performs
the same join and reports the same counts alongside it.

The one judgement call that is not the user's: keys are compared as trimmed
text. A key read as the number 1024 on one side and the string "1024" on the
other is the same participant, and matching zero rows because pandas typed two
columns differently would be a tooling artefact, not a fact about the data.
Whitespace is stripped for the same reason. Case is NOT folded — "AB12" and
"ab12" are different strings, and deciding otherwise silently could merge two
distinct participants.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store
from services.stat_utils import sanitize_nonfinite

router = APIRouter()

HOW = {"left", "inner", "outer"}


class MergeRequest(BaseModel):
    session_id: str
    other_session_id: str
    left_on: List[str]
    right_on: List[str]
    how: str = "left"
    # Columns to take from the incoming file. Empty means every column that is
    # not a key.
    columns: List[str] = []
    suffix: str = "_2"


def _filter_note(session_id: str) -> List[str]:
    """Warn when Select Cases is hiding rows this join will still touch.

    The join deliberately reads the whole sheet: joining the filtered view and
    saving the result would delete every excluded row, turning a display filter
    into permanent data loss. But the user is looking at a subset, so the row
    counts here will not match what they can see, and that needs saying.
    """
    try:
        active = store.get_filter(session_id)
    except AttributeError:
        return []
    return ([
        "Select Cases is active. The join covers the whole dataset, including the rows currently "
        "hidden, so these counts are larger than the sheet on screen."
    ] if active else [])


def _get(session_id: str, what: str) -> pd.DataFrame:
    df = store.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"{what} not found")
    return df


def _key_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Join keys as trimmed text, so 1024 and '1024' are the same participant.

    Anything missing stays missing: a blank key must not join to another blank
    key, which is how one absent identifier ends up matched to every other
    absent identifier in the file.
    """
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c]
        text = s.astype("string").str.strip()
        # A float column holding whole numbers prints as "1024.0"; the same id
        # typed by hand is "1024". Normalise the ones that are integral.
        if pd.api.types.is_float_dtype(s):
            whole = s.notna() & (s % 1 == 0)
            text = text.mask(whole, s.where(whole).astype("Float64").astype("Int64").astype("string"))
        out[f"__key_{c}"] = text.replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    return out


def _plan(left: pd.DataFrame, right: pd.DataFrame, req: MergeRequest) -> dict:
    """What the join would do, before anything is changed."""
    lk = _key_frame(left, req.left_on)
    rk = _key_frame(right, req.right_on)
    lk_cols, rk_cols = list(lk.columns), list(rk.columns)

    l_ok = lk.notna().all(axis=1)
    r_ok = rk.notna().all(axis=1)
    l_tuples = pd.MultiIndex.from_frame(lk[l_ok]) if len(lk_cols) > 1 else pd.Index(lk.loc[l_ok, lk_cols[0]])
    r_tuples = pd.MultiIndex.from_frame(rk[r_ok]) if len(rk_cols) > 1 else pd.Index(rk.loc[r_ok, rk_cols[0]])

    l_set, r_set = set(l_tuples), set(r_tuples)
    matched = l_set & r_set
    l_dup = int(len(l_tuples) - len(l_set))
    r_dup = int(len(r_tuples) - len(r_set))

    matched_rows = int(l_ok.sum() - sum(1 for k in l_tuples if k not in matched))
    return {
        "rows_left": int(len(left)), "rows_right": int(len(right)),
        "keys_matched": int(len(matched)),
        "left_rows_matched": matched_rows,
        "left_rows_unmatched": int(l_ok.sum()) - matched_rows,
        "left_keys_missing": int((~l_ok).sum()),
        "right_keys_missing": int((~r_ok).sum()),
        "left_duplicate_keys": l_dup,
        "right_duplicate_keys": r_dup,
        "right_keys_unused": int(len(r_set - matched)),
    }


def _warnings(plan: dict, req: MergeRequest, overlap: List[str]) -> List[str]:
    out: List[str] = []
    if plan["keys_matched"] == 0:
        out.append(
            "No key matched. Check that the two columns really hold the same identifier — "
            "a participant id and a visit id will not join.")
    if plan["right_duplicate_keys"]:
        # This is the one that silently inflates a dataset.
        out.append(
            f"The incoming file has {plan['right_duplicate_keys']} rows sharing a key that another row "
            "already uses. Each match will be repeated once per duplicate, so the result can have more "
            "rows than the dataset you started with.")
    if plan["left_rows_unmatched"]:
        out.append(
            f"{plan['left_rows_unmatched']} rows in the open dataset found no match and will have the "
            f"new columns left empty.")
    if plan["left_keys_missing"] or plan["right_keys_missing"]:
        out.append(
            f"{plan['left_keys_missing']} rows here and {plan['right_keys_missing']} in the incoming file "
            "have a blank key. They are never matched — a blank is not an identifier.")
    if plan["right_keys_unused"] and req.how == "left":
        out.append(
            f"{plan['right_keys_unused']} keys in the incoming file match nothing here. With a left join "
            "those rows are dropped; choose an outer join to keep them.")
    if overlap:
        out.append(
            f"Both files have a column called {', '.join(overlap)}. The incoming one is renamed with "
            f"'{req.suffix}' rather than overwriting what is already there.")
    return out


def _prepare(req: MergeRequest) -> tuple[pd.DataFrame, pd.DataFrame, List[str], dict]:
    if req.how not in HOW:
        raise HTTPException(status_code=400, detail=f"how must be one of {', '.join(sorted(HOW))}")
    if not req.left_on or not req.right_on:
        raise HTTPException(status_code=400, detail="Choose the key column on both sides")
    if len(req.left_on) != len(req.right_on):
        raise HTTPException(status_code=400, detail="The two key lists must have the same length")

    left = _get(req.session_id, "Session")
    right = _get(req.other_session_id, "The file to join")
    for c in req.left_on:
        if c not in left.columns:
            raise HTTPException(status_code=400, detail=f"Key column not in the open dataset: {c}")
    for c in req.right_on:
        if c not in right.columns:
            raise HTTPException(status_code=400, detail=f"Key column not in the incoming file: {c}")

    bring = [c for c in (req.columns or [c for c in right.columns if c not in req.right_on])
             if c in right.columns and c not in req.right_on]
    if not bring:
        raise HTTPException(status_code=400, detail="The incoming file has no columns to add besides the key")
    overlap = [c for c in bring if c in left.columns]
    return left, right, bring, {"overlap": overlap}


@router.post("/preview")
def merge_preview(req: MergeRequest):
    """What this join would do, without doing it."""
    left, right, bring, extra = _prepare(req)
    plan = _plan(left, right, req)
    plan["columns_added"] = [c + req.suffix if c in extra["overlap"] else c for c in bring]
    plan["warnings"] = _filter_note(req.session_id) + _warnings(plan, req, extra["overlap"])
    plan["rows_after"] = _rows_after(left, right, req, plan)
    return sanitize_nonfinite(plan)


def _rows_after(left: pd.DataFrame, right: pd.DataFrame, req: MergeRequest, plan: dict) -> Optional[int]:
    """Row count of the result. Exact for a left join with unique incoming
    keys; otherwise it depends on the duplicates, so it is computed rather
    than guessed."""
    if plan["right_duplicate_keys"] == 0 and req.how == "left":
        return int(len(left))
    return None


@router.post("/apply")
def merge_apply(req: MergeRequest):
    """Perform the join and replace the open dataset with the result."""
    left, right, bring, extra = _prepare(req)
    plan = _plan(left, right, req)

    lk, rk = _key_frame(left, req.left_on), _key_frame(right, req.right_on)
    key_names = [f"__key_{i}" for i in range(len(req.left_on))]
    lk.columns, rk.columns = key_names, key_names

    l = pd.concat([left.reset_index(drop=True), lk.reset_index(drop=True)], axis=1)
    r = pd.concat([right[bring].reset_index(drop=True), rk.reset_index(drop=True)], axis=1)
    # A blank key must not join to another blank key.
    r = r[rk.notna().all(axis=1).to_numpy()]

    merged = l.merge(r, on=key_names, how=req.how, suffixes=("", req.suffix))
    merged = merged.drop(columns=key_names)

    plan["rows_after"] = int(len(merged))
    plan["columns_added"] = [c for c in merged.columns if c not in left.columns]
    plan["warnings"] = _filter_note(req.session_id) + _warnings(plan, req, extra["overlap"])
    if plan["rows_after"] > plan["rows_left"]:
        plan["warnings"].insert(0, (
            f"The dataset grew from {plan['rows_left']} to {plan['rows_after']} rows. That is duplicate "
            "keys in the incoming file multiplying matches, not new participants — check the key before "
            "analysing this."))

    store.save(req.session_id, merged)
    plan["result_text"] = (
        f"Joined {plan['columns_added'] and len(plan['columns_added']) or 0} column(s) onto the dataset by "
        f"{', '.join(req.left_on)}: {plan['left_rows_matched']} of {plan['rows_left']} rows matched."
    )
    return sanitize_nonfinite(plan)
