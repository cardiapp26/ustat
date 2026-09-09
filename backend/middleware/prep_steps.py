"""Records every data-mutating request as a replayable prep step.

The audit trail (store.log_action) is informational: each endpoint logs a
summary in whatever detail it chose, and several log only counts. A recipe
that is meant to be replayed or turned into a script needs the FULL request.
Rather than teaching 40 endpoints to log twice, this middleware watches the
mutation routes, captures the request that succeeded, and appends it to the
session's step list (store.log_step).

What counts as a mutation is an explicit allow-list of route shapes below.
Analysis endpoints (models, tests, charts) are reads: running a regression
does not belong in the recipe that rebuilds the dataset.

Known honest limitation: undo/redo are recorded as steps. A replayer cannot
"replay" an undo without implementing the whole undo stack; a script
generator must surface them as warnings instead of code. Recording them
anyway keeps the recipe truthful about the order of what happened.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from services import store

# (method, compiled path regex, op template). Groups: sid, and optionally
# extra path params folded into the step's params.
_RULES: list = [
    # Data-management endpoints: op is the route's own slug.
    ("POST", re.compile(r"^/api/compute/(?P<sid>[^/]+)/(?P<op>[a-z0-9_/]+)$"), "compute/{op}"),
    ("DELETE", re.compile(r"^/api/compute/(?P<sid>[^/]+)/column/(?P<column>.+)$"), "compute/delete_column"),
    # Cell/structure edits and the case filter, on the sessions router.
    ("PATCH", re.compile(r"^/api/sessions/(?P<sid>[^/]+)/cell$"), "sessions/edit_cell"),
    ("POST", re.compile(r"^/api/sessions/(?P<sid>[^/]+)/(?P<op>set_cells|clear_cells|swap_value_labels|reorder_columns|select_cases|undo|redo)$"), "sessions/{op}"),
    ("DELETE", re.compile(r"^/api/sessions/(?P<sid>[^/]+)/select_cases$"), "sessions/clear_cases"),
    ("DELETE", re.compile(r"^/api/sessions/(?P<sid>[^/]+)/row/(?P<row_index>\d+)$"), "sessions/delete_row"),
    ("POST", re.compile(r"^/api/merge/apply$"), "merge/apply"),
]

# compute/ ops that are reads despite living on the compute router.
_COMPUTE_READS = {"missing_diagnostics"}


def _match(method: str, path: str):
    for m, rx, op_tpl in _RULES:
        if m != method:
            continue
        match = rx.match(path)
        if not match:
            continue
        groups = match.groupdict()
        op = op_tpl.format(op=groups.get("op", ""))
        if op.startswith("compute/") and op.split("/", 1)[1] in _COMPUTE_READS:
            return None
        return op, groups
    return None


class PrepStepsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        matched = _match(request.method, request.url.path)
        if matched is None:
            return await call_next(request)

        op, groups = matched
        # Read and re-inject the body: BaseHTTPMiddleware hands the endpoint
        # the same receive channel, so consuming it here without putting it
        # back would starve the endpoint.
        body_bytes = await request.body()

        async def receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request._receive = receive  # noqa: SLF001 - the documented Starlette idiom

        response = await call_next(request)
        if 200 <= response.status_code < 300:
            params: dict = {}
            if body_bytes:
                try:
                    parsed = json.loads(body_bytes)
                    if isinstance(parsed, dict):
                        params = {k: v for k, v in parsed.items() if k != "session_id"}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    params = {}
            for key, value in groups.items():
                if key in ("sid", "op"):
                    continue
                params[key] = int(value) if value.isdigit() else value
            sid = _session_id(groups, body_bytes)
            if sid:
                store.log_step(sid, op, params)
        return response


def _session_id(groups: dict, body_bytes: bytes) -> Optional[str]:
    sid = groups.get("sid")
    if sid:
        return sid
    # merge/apply and friends carry the session in the body instead.
    try:
        parsed = json.loads(body_bytes)
        candidate = parsed.get("session_id") if isinstance(parsed, dict) else None
        return candidate if isinstance(candidate, str) else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
