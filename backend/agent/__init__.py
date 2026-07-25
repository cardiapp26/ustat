"""Agent-facing layer for driving the uSTAT backend programmatically.

Phase 1 deliverables:
- UstatClient: typed HTTP client over httpx
- UstatServer: managed uvicorn subprocess for headless runs
- tool_catalog: machine-readable registry of analysis endpoints
"""

from .client import UstatClient, UstatServer
from .runner import UstatRunner, UstatRunnerError
from .tool_catalog import TOOLS, Tool
from . import prompts

__all__ = [
    "UstatClient",
    "UstatServer",
    "UstatRunner",
    "UstatRunnerError",
    "TOOLS",
    "Tool",
    "prompts",
]
