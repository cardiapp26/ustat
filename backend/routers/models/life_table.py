"""Actuarial (Cutler-Ederer) life table: POST /api/models/survival/life_table.

Interval-grouped follow-up (years, months, days) summarised the way vital
statistics and older clinical series report it: subjects entering, withdrawn
and failing per interval, conditional and cumulative survival with Greenwood
standard errors, the interval hazard and a median survival time. The formulas
and conventions live in services/life_table.py.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import life_table as lt
from services.impute import apply_imputation
from services.stat_utils import sorted_groups

from .glm import _get_df, _sanitize

router = APIRouter()

MAX_GROUPS = 30


class LifeTableRequest(BaseModel):
    session_id: str
    time_col: str
    event_col: str
    group_col: Optional[str] = None
    # Interval layout: explicit breaks, or a width (optionally with a count).
    # With none of them the width is the follow-up maximum / 10 on a round number.
    interval_width: Optional[float] = None
    n_intervals: Optional[int] = None
    breaks: Optional[List[float]] = None
    alpha: float = 0.05
    imputation: Optional[str] = "listwise"


def _bad(detail: str, code: int = 422) -> HTTPException:
    return HTTPException(status_code=code, detail=detail)


def _prepare(req: LifeTableRequest, df_full: pd.DataFrame):
    """Validate the request and return (clean frame, warnings, n_total)."""
    if not (0.0 < req.alpha < 1.0):
        raise _bad("alpha must be between 0 and 1.")
    cols = [req.time_col, req.event_col] + ([req.group_col] if req.group_col else [])
    for c in cols:
        if c not in df_full.columns:
            raise _bad(f"Column '{c}' not found.")
    if len(set(cols)) != len(cols):
        raise _bad("time_col, event_col and group_col must be three different columns.")

    n_total = len(df_full)
    df_full = df_full.copy()
    df_full[req.time_col] = pd.to_numeric(df_full[req.time_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    df_full[req.event_col] = pd.to_numeric(df_full[req.event_col], errors="coerce")

    df = apply_imputation(df_full, cols, req.imputation or "listwise")
    df = df.dropna(subset=cols)
    if len(df) == 0:
        raise _bad(
            "No valid rows after coercing time/event columns to numeric. Check that both columns contain numbers.",
            400,
        )

    if (df[req.time_col] < 0).any():
        raise _bad(f"'{req.time_col}' contains negative follow-up times; a life table needs time >= 0.")
    ev_vals = sorted(df[req.event_col].unique())
    if set(ev_vals) - {0, 1, 0.0, 1.0}:
        raise _bad(f"Event column must be binary 0/1 (0=censored, 1=event). Found: {ev_vals[:10]}")

    warnings: List[str] = []
    n_excluded = n_total - len(df)
    if n_excluded:
        warnings.append(
            f"{n_excluded} of {n_total} rows were excluded (missing or non-numeric time, event or group)."
        )
    return df, warnings, n_total


def _run(req: LifeTableRequest) -> dict:
    df, warnings, n_total = _prepare(req, _get_df(req.session_id))
    time = df[req.time_col].to_numpy(dtype=float)
    t_max = float(time.max())

    try:
        breaks, extended, brk_warns = lt.resolve_breaks(t_max, req.interval_width, req.n_intervals, req.breaks)
    except lt.LifeTableInputError as exc:
        raise _bad(str(exc))
    warnings.extend(brk_warns)

    n_beyond = int((time > breaks[-1]).sum())
    if n_beyond:
        warnings.append(
            f"{n_beyond} subject(s) have follow-up beyond the last break ({breaks[-1]:g}); they are counted as "
            "withdrawn (censored) in the final interval, even if they had the event."
        )
    n_at_last = int((time == breaks[-1]).sum())
    if n_at_last and not extended:
        warnings.append(
            f"{n_at_last} subject(s) have follow-up exactly at the last break ({breaks[-1]:g}); they are placed "
            "in the final interval (an event there is an event, a censoring there is a withdrawal)."
        )
    if extended:
        warnings.append(
            f"Breaks were extended automatically to cover the longest follow-up ({t_max:g}): "
            f"{len(breaks) - 1} intervals of width {lt.regular_width(breaks):g}."
        )

    if req.group_col:
        levels = sorted_groups(df[req.group_col])
        if len(levels) > MAX_GROUPS:
            raise _bad(f"Group column '{req.group_col}' has {len(levels)} levels; at most {MAX_GROUPS} are allowed.")
        samples = [(lv, df[df[req.group_col] == lv]) for lv in levels]
    else:
        samples = [(None, df)]

    groups = []
    for label, sub in samples:
        table = lt.compute_life_table(sub[req.time_col], sub[req.event_col], breaks, req.alpha)
        groups.append({
            "group": None if label is None else (label.item() if hasattr(label, "item") else label),
            "n": table["n"],
            "events": table["events"],
            "censored": table["n"] - table["events"],
            "intervals": table["intervals"],
            "median_survival": table["median_survival"],
        })
        if label is not None and table["n"] < 10:
            warnings.append(f"Group '{label}' has only {table['n']} subjects; its life table is unstable.")

    return {
        "model": "Life table (Cutler-Ederer)",
        "time_col": req.time_col,
        "event_col": req.event_col,
        "group_col": req.group_col,
        "alpha": req.alpha,
        "breaks": breaks,
        "interval_width": lt.regular_width(breaks),
        "n_total": n_total,
        "n_analyzed": int(len(df)),
        "n_excluded": int(n_total - len(df)),
        "imputation": req.imputation or "listwise",
        "groups": groups,
        "warnings": warnings,
        "result_text": lt.result_text(groups, req.group_col, breaks, req.alpha),
        "methods_text": lt.methods_text(req.alpha, bool(req.group_col)),
        "r_code": lt.r_code(req.time_col, req.event_col, req.group_col, breaks, req.alpha),
        "export_rows": lt.export_rows(groups, req.group_col),
    }


@router.post("/survival/life_table")
def life_table(req: LifeTableRequest):
    return _sanitize(_run(req))
