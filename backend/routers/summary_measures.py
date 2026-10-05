"""
Summary-measures router: serial (longitudinal) data collapsed to one value per subject.

Endpoints
---------
POST /compute   per-subject AUC, time-averaged AUC, incremental AUC, peak, time to
                peak, OLS slope, mean, first, last and change (Matthews et al. 1990),
                per-group descriptives and, when a group column is given, the group
                comparison for each measure (Welch t-test or one-way ANOVA as primary,
                Mann-Whitney or Kruskal-Wallis as sensitivity analysis).

The input is long format (one row per subject and visit). The session data is
never modified; the per-subject table is returned for download. The arithmetic
lives in services.summary_measures; invalid input becomes a 422.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import store, summary_measures as sm
from services.stat_utils import sanitize_nonfinite

router = APIRouter()


class ComputeRequest(BaseModel):
    session_id: str
    subject_col: str
    time_col: str
    value_col: str
    group_col: Optional[str] = None
    measures: List[str] = list(sm.DEFAULT_MEASURES)
    min_points: int = 2
    missing: str = "omit"            # 'omit' | 'interpolate'
    alpha: float = 0.05


@router.post("/compute")
def compute(req: ComputeRequest):
    # Select Cases / missing codes apply, but Weight Cases must not: replicated
    # rows would be read as duplicate visits of the same subject.
    df = store.get_filtered(req.session_id, weighted=False)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        result = sm.analyse(
            df, req.subject_col, req.time_col, req.value_col, req.group_col,
            req.measures, req.min_points, req.missing, req.alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return sanitize_nonfinite(result)
