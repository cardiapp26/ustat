"""Read the Split File level a request states (services/split_scope)."""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from services.split_scope import HEADER, SplitScopeError, current_split, parse_header


class SplitScopeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        raw = request.headers.get(HEADER)
        if not raw or not request.url.path.startswith("/api/"):
            return await call_next(request)
        try:
            scope = parse_header(raw)
        except SplitScopeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        token = current_split.set(scope)
        try:
            return await call_next(request)
        finally:
            current_split.reset(token)
