"""Prompts and tool-formatting helpers for the LLM-driven uSTAT agent.

These prompts are consumed by the orchestration loop in ``agent.runner``.
They tell the model how to plan a sequence of calls against the uSTAT backend
and how to interpret the resulting raw endpoint JSON.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .tool_catalog import Tool, TOOLS


SYSTEM_PROMPT = """You are an expert biostatistics assistant driving the uSTAT statistical backend.

Your job is to turn a natural-language analysis request into a correct sequence of HTTP endpoint calls, execute them through the provided tool catalog, and then summarise the returned statistics in plain language with the exact numbers.

Rules:
- Inspect the uploaded dataset's column names and kinds before choosing tools.
- Respect each tool's ``requires`` guard; never request a tool with the wrong column kind.
- Prefer the simplest test that answers the question (e.g. Fisher for small expected counts, chi-square for larger tables, t-test/ANOVA for numeric outcomes).
- Report exact p-values and effect sizes returned by the backend; do not round or recompute them.
- When a result is not significant, say so plainly. When it is significant, give the direction and magnitude.
""".strip()


TOOL_USE_INSTRUCTIONS = """You may call tools by emitting a single JSON object.

Format:
{
  "calls": [
    {"name": "tool_name", "args": {"arg1": "value", "arg2": ["a", "b"]}},
    ...
  ]
}

Rules for tool calls:
1. Use only tool names listed below.
2. Do NOT include ``session_id`` in args; the runner injects it automatically.
3. For optional arguments, omit them entirely rather than passing null unless the schema says otherwise.
4. Column names must exactly match those in the dataset.
5. Respect semantic guards:
   - ``categorical(2)`` means the column must be categorical with exactly 2 levels.
   - ``categorical`` means any categorical column.
   - ``numeric`` means a numeric column.
   - ``any`` means any column kind.
6. The plan is executed in order; later calls can use columns from the same session but cannot rely on computed columns unless a prior ``formula`` call created them.
7. For Table 1, include all relevant variables in ``variables`` and a categorical ``group_column``.
8. For ROC analysis, use ``direction='lower'`` when higher scores indicate a lower risk of the event (e.g. higher eGFR means lower renal-failure risk). The backend will report ``1 - AUC`` with swapped confidence-interval bounds. Use ``direction='higher'`` when higher scores indicate higher event risk, and ``direction='auto'`` when you are unsure.

Available tools:
{tools}
""".strip()


def format_tools(tools: List[Tool]) -> str:
    """Render the tool catalog as human-readable markdown-like text."""
    lines: List[str] = []
    for tool in tools:
        body = ", ".join(f"{k}: {v}" for k, v in tool["body"].items()) or "(none)"
        requires = (
            ", ".join(f"{k}: {v}" for k, v in tool["requires"].items()) or "(none)"
        )
        lines.extend(
            [
                f"- {tool['name']}  {tool['method']} {tool['path']}",
                f"  args: {body}",
                f"  requires: {requires}",
                f"  doc: {tool['doc']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def tool_use_prompt(tools: List[Tool] | None = None) -> str:
    """Return the full tool-use instructions populated with the current catalog."""
    return TOOL_USE_INSTRUCTIONS.format(tools=format_tools(tools or TOOLS))


def planning_messages(
    nl_request: str,
    kinds: Dict[str, str],
    levels: Dict[str, int] | None = None,
    tools: List[Tool] | None = None,
) -> List[Dict[str, str]]:
    """Return a ready-to-send message list for a model that accepts chat messages.

    The returned list contains the system prompt, the tool catalog, and the
    user's analysis request together with the column-kind map the model needs
    to choose appropriate tests.
    """
    levels = levels or {}
    columns_block = "\n".join(
        f"- {name}: {kind}"
        + (
            f" (levels={levels.get(name)})"
            if kind == "categorical" and name in levels
            else ""
        )
        for name, kind in kinds.items()
    )

    user_prompt = (
        f"Dataset columns:\n{columns_block}\n\n"
        f"Analysis request: {nl_request}\n\n"
        "Return the plan as a JSON object with a 'calls' array."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": tool_use_prompt(tools or TOOLS)},
        {"role": "user", "content": user_prompt},
    ]


def format_record(record: Dict[str, Any]) -> str:
    """Render one tool-call record as compact JSON for interpretation prompts."""
    return json.dumps(record, indent=2, default=str)
