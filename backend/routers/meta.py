"""
Meta-analysis router.

Endpoints
---------
POST /analyze     — fixed + random-effects pooling (DL / PM τ²), Q / I² / H²,
                    95% prediction interval, per-study weights
POST /subgroup    — subgroup pooling + between-group heterogeneity (Q_between)
POST /regression  — meta-regression (effect ~ moderator) via weighted LS
POST /bias        — Egger's test, Begg rank test, funnel points, trim-and-fill

Pure numpy / scipy / statsmodels (all existing deps). Studies are supplied
either as a pre-computed effect with a CI (or SE), or as a raw 2×2 table
(events / n per arm) which is converted to log-OR / log-RR / risk difference
with a continuity correction for zero cells.

The arithmetic itself lives in ustat_engine.meta, so the browser can run the
identical code; what stays here is the wire contract. MetaStudy/MetaRequest
are still the schema clients are validated against, and `model_dump()` hands
the engine plain dicts with those same defaults already applied.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ustat_engine import meta as engine_meta
from routers.engine_adapter import adapt

router = APIRouter()


# ── Study input ──────────────────────────────────────────────────────────────


class MetaStudy(BaseModel):
    label: str
    # Option A: pre-computed effect + CI
    effect: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    # Option B: pre-computed effect + SE
    se: Optional[float] = None
    # Option C: raw 2×2 — treated (e1/n1) vs control (e2/n2)
    e1: Optional[float] = None
    n1: Optional[float] = None
    e2: Optional[float] = None
    n2: Optional[float] = None
    # Subgroup / moderator
    subgroup: Optional[str] = None
    moderator: Optional[float] = None


class MetaRequest(BaseModel):
    studies: List[MetaStudy]
    measure: str = "OR"          # OR | RR | RD | SMD | MD | generic
    tau2_method: str = "DL"      # DL | PM
    cc: float = 0.5              # continuity correction for zero cells (2×2)


@router.post("/analyze")
def analyze(req: MetaRequest):
    return adapt(engine_meta.analyze, req.model_dump())


@router.post("/subgroup")
def subgroup(req: MetaRequest):
    return adapt(engine_meta.subgroup, req.model_dump())


@router.post("/regression")
def meta_regression(req: MetaRequest):
    return adapt(engine_meta.meta_regression, req.model_dump())


@router.post("/bias")
def bias(req: MetaRequest):
    return adapt(engine_meta.bias, req.model_dump())
