"""
Epidemiology router: rate standardisation and incidence-rate comparison.

Endpoints
---------
POST /direct_standardisation    directly standardised rate with Fay-Feuer gamma CI
                                (epitools::ageadjust.direct), crude rate with
                                exact Poisson CI, per-stratum weights and rates
POST /indirect_standardisation  SMR with Byar and exact Poisson CIs and the
                                exact Poisson test of SMR = 1
POST /rate_ratio                two-group incidence rates (exact CIs), rate ratio
                                (log-method CI), rate difference, conditional
                                exact / mid-p tests

All inputs are inline stratum tables, so no uploaded dataset is needed. The
arithmetic lives in services.standardisation; invalid input becomes a 422.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import standardisation as std

router = APIRouter()


class DirectStratum(BaseModel):
    label: Optional[str] = None
    events: float
    person_time: Optional[float] = None      # or `population`
    population: Optional[float] = None
    standard_population: float


class DirectRequest(BaseModel):
    strata: List[DirectStratum]
    multiplier: float = 100000.0             # rates reported per this many person-time units
    alpha: float = 0.05


class IndirectStratum(BaseModel):
    label: Optional[str] = None
    observed: float                          # events in the study group
    person_time: float                       # study-group person-time
    # Either the reference rate (per `reference_multiplier` person-time units) ...
    reference_rate: Optional[float] = None
    # ... or reference events and person-time, from which the rate is derived.
    reference_events: Optional[float] = None
    reference_person_time: Optional[float] = None


class IndirectRequest(BaseModel):
    strata: List[IndirectStratum]
    reference_multiplier: float = 1.0        # reference_rate is per this many person-time units
    alpha: float = 0.05


class RateRatioRequest(BaseModel):
    events1: float
    person_time1: float
    events2: float
    person_time2: float
    alpha: float = 0.05


def _run(fn: Callable[[], dict]) -> dict:
    try:
        return fn()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/direct_standardisation")
def direct_standardisation(req: DirectRequest):
    return _run(lambda: std.direct_standardisation(
        [s.model_dump() for s in req.strata], req.multiplier, req.alpha))


@router.post("/indirect_standardisation")
def indirect_standardisation(req: IndirectRequest):
    return _run(lambda: std.indirect_standardisation(
        [s.model_dump() for s in req.strata], req.reference_multiplier, req.alpha))


@router.post("/rate_ratio")
def rate_ratio(req: RateRatioRequest):
    return _run(lambda: std.rate_ratio(
        req.events1, req.person_time1, req.events2, req.person_time2, req.alpha))
