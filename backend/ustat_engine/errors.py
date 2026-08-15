"""The engine's only way to reject a request.

The engine is imported by two callers that cannot share an exception type: the
FastAPI routers on this server, and CPython-on-WebAssembly in the browser,
where `fastapi` does not exist and an HTTPException would be an ImportError
before it could ever be raised.

So the engine raises this instead, and each caller translates. The HTTP status
travels with the error rather than being re-derived at the boundary, because
the code that detected the problem is the only code that knows whether it was
the caller's input (422) or a missing session (404).
"""
from __future__ import annotations


class EngineError(Exception):
    """A rejection the caller is expected to report, not a crash.

    `status_hint` is advice for an HTTP caller, not a promise that one exists.
    A browser caller is free to ignore it and show the message.
    """

    def __init__(self, message: str, status_hint: int = 400) -> None:
        super().__init__(message)
        self.message = str(message)
        self.status_hint = int(status_hint)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message
