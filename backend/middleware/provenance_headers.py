"""Stamp every API response with what computed it.

One middleware rather than a field in two hundred response models. Response
shapes here are load-bearing -- panels destructure backend fields without
optional chaining -- so adding a key to every body is both a large diff and a
crash risk, and any endpoint missed would be an endpoint that quietly claims
whatever the session header claims. A header cannot be forgotten by an
endpoint author, because no endpoint author writes it.

Static files and the SPA shell are left alone: provenance is a property of a
computation, and a JavaScript bundle did not compute anything.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from services.runtime_identity import provenance_headers


class ProvenanceHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        if request.url.path.startswith("/api/"):
            for name, value in provenance_headers().items():
                response.headers[name] = value
        return response
