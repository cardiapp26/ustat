"""Diagnostics that survive being imported where loguru is not installed.

The engine carries a handful of debug calls that record why a numerical solve
gave up -- which root finder failed, on which point of a power curve. They are
worth keeping: without them a returned `None` is indistinguishable from a bug.

But the engine cannot depend on loguru being importable. It is pure Python and
could be vendored into the browser runtime later; until someone decides that
download is worth three debug lines, the browser gets no-ops and the server
gets the real thing. Neither can affect a returned value.
"""
from __future__ import annotations


class _NullLogger:
    """Accepts every loguru call and does nothing with it."""

    def debug(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


try:  # pragma: no cover - the fallback only runs where loguru is absent
    from loguru import logger
except ImportError:  # pragma: no cover
    logger = _NullLogger()

__all__ = ["logger"]
