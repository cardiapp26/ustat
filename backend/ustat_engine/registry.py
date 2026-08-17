"""The list of analyses both runtimes can run.

A registry rather than direct imports, because the browser has to answer
"can this run locally?" before it has loaded the code that would answer it.
Registration is explicit -- an analysis appears here when someone has written
a parity fixture for it, not because a module happened to get imported.
"""
from __future__ import annotations

from .errors import EngineError
from .spec import AnalysisSpec

ANALYSES: dict[str, AnalysisSpec] = {}


def register(spec: AnalysisSpec) -> AnalysisSpec:
    """Add `spec` to the registry, refusing to shadow an existing id.

    Silently replacing would mean two analyses answering to one name, with
    whichever module imported last deciding which of them runs.
    """
    if spec.id in ANALYSES:
        raise ValueError(f"analysis id {spec.id!r} is already registered")
    ANALYSES[spec.id] = spec
    return spec


def get(analysis_id: str) -> AnalysisSpec:
    try:
        return ANALYSES[analysis_id]
    except KeyError:
        raise EngineError(f"Unknown analysis: {analysis_id}", status_hint=404) from None


def _check_filter(frame, params: dict) -> None:
    """Refuse a frame that was cut under a different Select Cases.

    A worker keeps a frame resident between runs; the user's filter does not
    have to stay still while it does. When the caller states which filter the
    analysis assumes, the frame has to agree, because the failure mode
    otherwise is invisible: a perfectly ordinary-looking result computed over
    the wrong patients.
    """
    expected = params.get("__filter_fingerprint")
    if expected is None:
        return
    actual = getattr(frame, "attrs", {}).get("filter_fingerprint")
    if actual != expected:
        raise EngineError(
            "frame does not match the active Select Cases", status_hint=409
        )


def run(analysis_id: str, frame=None, params: dict | None = None) -> dict:
    """Run one analysis by id. The single entry point the browser calls."""
    spec = get(analysis_id)
    params = params or {}
    if spec.needs_frame:
        if frame is None:
            raise EngineError(
                f"{analysis_id} needs a dataset and none was supplied", status_hint=400
            )
        _check_filter(frame, params)
        return spec.fn(frame, params)
    return spec.fn(params)
