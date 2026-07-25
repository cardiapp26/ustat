"""Server-side agent execution endpoint.

This router exposes ``POST /api/agent/run``, which accepts a dataset file and
either an explicit execution plan or a natural-language request, runs the
``UstatRunner`` inside the backend process, and returns the generated summary
and raw endpoint records.

When an explicit ``plan`` is supplied no external LLM is consulted, so the
endpoint works without ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from httpx import HTTPStatusError
from loguru import logger

from routers.upload import MAX_UPLOAD_BYTES


router = APIRouter()


class _NoPlannerError(Exception):
    """Raised when a natural-language request cannot be planned locally."""


def _default_planner(
    nl_request: str,
    kinds: dict[str, str],
    levels: dict[str, int],
    get_tool: Any,
) -> List[dict[str, Any]]:
    """Plan a natural-language request using OpenAI when configured.

    This planner is only invoked when the caller does not supply an explicit
    ``plan``.  It requires ``OPENAI_API_KEY`` to be set in the environment.
    """
    # Lazy import agent modules and the OpenAI client so that importing this
    # router at application startup does not create a circular import through
    # ``agent.tool_catalog`` (which needs the fully-built ``app``).
    from agent.prompts import planning_messages
    from agent.tool_catalog import TOOLS

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise _NoPlannerError(
            "No explicit plan was provided and OPENAI_API_KEY is not set. "
            "Provide an explicit 'plan' JSON array, or set OPENAI_API_KEY."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise _NoPlannerError(
            "OpenAI package is not installed; provide an explicit 'plan'."
        ) from exc

    client = OpenAI(api_key=api_key)
    messages = planning_messages(nl_request, kinds, levels, TOOLS)
    try:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content or "{}")
        calls = parsed.get("calls", [])
        if not isinstance(calls, list):
            raise ValueError("LLM response 'calls' must be a list")
        return [{"name": c["name"], "args": c.get("args", {})} for c in calls]
    except Exception as exc:
        raise RuntimeError(f"LLM planning failed: {exc}") from exc


@router.post("/run")
async def agent_run(
    request: Request,
    file: UploadFile = File(...),
    nl_request: str = Form(""),
    plan: Optional[str] = Form(None),
) -> dict[str, Any]:
    """Run an agent analysis server-side.

    Parameters
    ----------
    file:
        Dataset to upload (CSV, Excel, SPSS, SAS, Stata).
    nl_request:
        Natural-language analysis request.  Only used when ``plan`` is omitted;
        requires ``OPENAI_API_KEY`` to be configured.
    plan:
        Optional explicit plan as a JSON array of tool calls, e.g.
        ``[{"name": "table1", "args": {"variables": ["AGE"], ...}}]``.

    Returns
    -------
    ``{"summary": str, "records": [...], "session_id": str, "columns": [...]}``
    """
    if not plan and not nl_request:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'nl_request' or 'plan'.",
        )

    _max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    # Cheap pre-check on the declared size (rejects before reading the body).
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {_max_mb} MB.")
    # Hard cap on the bytes actually read — defends against a missing or spoofed
    # Content-Length. Read one byte past the limit; if we got it, it's too big.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum upload size is {_max_mb} MB.")

    suffix = os.path.splitext(file.filename or "data.csv")[1] or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Import agent modules here to avoid circular imports at router load.
        from agent import UstatClient, UstatRunner

        # Use fixed loopback by default, not request Host, to prevent SSRF.
        AGENT_INTERNAL_BASE_URL = os.environ.get("AGENT_INTERNAL_BASE_URL", "http://127.0.0.1:8000")
        client = UstatClient(AGENT_INTERNAL_BASE_URL)
        runner = UstatRunner(client, server=None, planner=_default_planner)

        explicit_plan: Optional[List[dict[str, Any]]] = None
        if plan is not None:
            try:
                parsed_plan = json.loads(plan)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail=f"Invalid plan JSON: {exc}")
            if not isinstance(parsed_plan, list):
                raise HTTPException(
                    status_code=422, detail="'plan' must be a JSON array"
                )
            explicit_plan = parsed_plan

        result = await asyncio.to_thread(
            runner.run,
            nl_request=nl_request or "Agent analysis",
            dataset_path=tmp_path,
            plan=explicit_plan,
            auto_stop=False,
        )
    except _NoPlannerError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPStatusError as exc:
        detail = f"Backend call failed: {exc}"
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        raise HTTPException(
            status_code=exc.response.status_code, detail=detail
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {
        "summary": result["summary"],
        "records": result["records"],
        "session_id": result["session_id"],
        "columns": result["columns"],
    }
