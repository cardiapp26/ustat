"""Orchestration loop for the LLM-driven uSTAT agent.

UstatRunner ties the pieces together:

1. Boot (or attach to) a uvicorn backend.
2. Upload a dataset and build a column-kind map.
3. Accept a planned list of tool calls (from an LLM, a heuristic planner, or a
   deterministic test script).
4. Validate every call against the catalog's semantic guards.
5. Execute calls through UstatClient and record raw responses.
6. Produce a concise summary and stop the server if the runner started it.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .client import UstatClient, UstatServer
from .tool_catalog import check_requires, get_tool


class UstatRunnerError(Exception):
    """Raised when the runner cannot complete a request."""


class UstatRunner:
    """End-to-end orchestrator for natural-language requests to the uSTAT API.

    Parameters
    ----------
    client:
        A configured UstatClient pointing at the backend.
    server:
        Optional managed UstatServer. When provided and not already running,
        the runner starts it before the request and stops it afterwards.
    planner:
        Optional callable ``(nl_request, kinds, levels, tools) -> plan`` that
        turns a natural-language request into a list of tool calls. If omitted,
        the caller must pass an explicit ``plan`` to ``run()``.
    """

    def __init__(
        self,
        client: UstatClient,
        server: Optional[UstatServer] = None,
        planner: Optional[Any] = None,
    ) -> None:
        self.client = client
        self.server = server
        self.planner = planner

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _column_kinds(meta: dict[str, Any]) -> dict[str, str]:
        """Build ``column -> kind`` from the upload metadata."""
        return {col["name"]: col["kind"] for col in meta.get("columns", [])}

    @staticmethod
    def _column_levels(meta: dict[str, Any]) -> dict[str, int]:
        """Build ``column -> level_count`` for categorical columns.

        Levels are taken from ``value_labels`` when present; otherwise they are
        counted from the (possibly capped) preview. This is intentionally cheap
        and only used for guard validation.
        """
        preview = meta.get("preview", [])
        levels: dict[str, int] = {}
        for col in meta.get("columns", []):
            name = col["name"]
            if col.get("kind") != "categorical":
                continue

            value_labels = col.get("value_labels")
            if isinstance(value_labels, dict):
                levels[name] = len(value_labels)
                continue

            seen: set[Any] = set()
            for row in preview:
                value = row.get(name)
                if value is not None:
                    seen.add(value)
            levels[name] = len(seen)
        return levels

    @staticmethod
    def _strip_session_id(args: dict[str, Any]) -> dict[str, Any]:
        """Remove any session_id the planner may have accidentally included."""
        return {k: v for k, v in args.items() if k != "session_id"}

    @staticmethod
    def _summarize(
        nl_request: str,
        records: List[dict[str, Any]],
        kinds: dict[str, str],
    ) -> str:
        """Build a deterministic prose summary from the raw endpoint records."""
        lines = [
            f"Request: {nl_request}",
            f"Columns: {', '.join(f'{n} ({k})' for n, k in kinds.items())}",
            "",
        ]

        for rec in records:
            call = rec["call"]
            resp = rec["response"]
            name = call["name"]

            if name == "table1":
                rows = resp.get("rows", [])
                total = resp.get("total_n")
                groups = resp.get("group_labels", [])
                lines.append(
                    f"Table 1: {len(rows)} variable rows, n={total}, groups={groups}"
                )
                for row in rows:
                    variable = row.get("variable")
                    test = row.get("test") or row.get("normality_test")
                    p = row.get("p_value")
                    if test and p is not None:
                        lines.append(f"  - {variable}: {test} p={p}")
            elif name == "roc":
                auc = resp.get("auc")
                ci_low = resp.get("ci_lower")
                ci_high = resp.get("ci_upper")
                direction = resp.get("direction_used")
                ci_text = (
                    f" (95% CI {ci_low}–{ci_high})"
                    if ci_low is not None and ci_high is not None
                    else ""
                )
                dir_note = (
                    "higher scores predict the event"
                    if direction == "higher"
                    else "higher scores indicate lower event risk (1-AUC reported)"
                    if direction == "lower"
                    else "auto direction"
                )
                lines.append(
                    f"ROC: AUC={auc}{ci_text}, direction={direction} ({dir_note})"
                )
            elif "p" in resp:
                test = resp.get("test", name)
                p = resp.get("p")
                sig = resp.get("significant")
                or_text = ""
                if "odds_ratio" in resp:
                    or_text = f", OR={resp['odds_ratio']:.2f}"
                lines.append(f"{test}: p={p}{or_text}, significant={sig}")
            else:
                lines.append(f"{name}: completed.")

        return "\n".join(lines)

    # ---------------------------------------------------------------- loop
    def run(
        self,
        nl_request: str,
        dataset_path: str,
        plan: Optional[List[dict[str, Any]]] = None,
        auto_stop: bool = True,
    ) -> dict[str, Any]:
        """Execute a natural-language request against a dataset.

        Parameters
        ----------
        nl_request:
            The user's natural-language analysis request.
        dataset_path:
            Path to a supported dataset file.
        plan:
            Optional ordered list of tool calls, e.g.
            ``[{"name": "table1", "args": {"variables": [...], ...}}]``. If
            omitted, ``self.planner`` must be provided.
        auto_stop:
            If ``True`` and the runner started the server, stop it before
            returning.

        Returns
        -------
        ``{"summary": str, "records": [...], "session_id": str,
        "columns": [...]}``
        """
        started_here = False
        if self.server is not None and getattr(self.server, "_proc", None) is None:
            self.server.start()
            started_here = True

        try:
            meta = self.client.upload(dataset_path)
            session_id = meta["session_id"]
            kinds = self._column_kinds(meta)
            levels = self._column_levels(meta)

            if plan is None:
                if self.planner is None:
                    raise UstatRunnerError(
                        "No plan provided and no planner configured."
                    )
                plan = self.planner(nl_request, kinds, levels, get_tool)

            records: List[dict[str, Any]] = []
            for call in plan:
                name = call["name"]
                args = self._strip_session_id(dict(call.get("args", {})))

                tool = get_tool(name)
                if tool is None:
                    raise UstatRunnerError(f"Unknown tool '{name}' in plan.")

                error = check_requires(tool, args, kinds, levels)
                if error:
                    raise UstatRunnerError(error)

                response = self.client.call(name, session_id, args)
                records.append(
                    {"call": {"name": name, "args": args}, "response": response}
                )

            summary = self._summarize(nl_request, records, kinds)
            return {
                "summary": summary,
                "records": records,
                "session_id": session_id,
                "columns": meta.get("columns", []),
            }
        finally:
            if auto_stop and started_here:
                self.server.stop()
