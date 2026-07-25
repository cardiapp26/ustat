"""Generator / drift-checker for backend/agent/tool_catalog.py.

Usage from repo root:
    python scripts/generate_tool_catalog.py              # check for drift
    python scripts/generate_tool_catalog.py --write      # regenerate catalog + docs
    python scripts/generate_tool_catalog.py --docs FILE  # write markdown docs only

The script imports backend/main.py (so backend/ must be on sys.path), builds a
fresh catalog from app.openapi() using agent.tool_catalog.build_catalog(), and
compares it to the TOOLS value currently stored in backend/agent/tool_catalog.py.
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make the backend package importable from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main as backend_main  # noqa: E402,F401  (backend/main.py -- imported for FastAPI side effects)
from agent import tool_catalog as tc  # noqa: E402


CATALOG_PATH = BACKEND_DIR / "agent" / "tool_catalog.py"
DOCS_PATH = REPO_ROOT / "docs" / "AGENT_TOOLS.md"


def _indent(level: int) -> str:
    return "    " * level


def _serialize_value(value: Any, indent: int = 0) -> str:
    """Serialize a JSON-like value as a compact but readable Python literal."""
    prefix = _indent(indent)
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        for k, v in value.items():
            lines.append(
                f"{_indent(indent + 1)}{k!r}: {_serialize_value(v, indent + 1)},"
            )
        lines.append(f"{prefix}}}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        return (
            "[\n"
            + "\n".join(
                f"{_indent(indent + 1)}{_serialize_value(item, indent + 1)},"
                for item in value
            )
            + f"\n{prefix}]"
        )
    return repr(value)


def serialize_tools(tools: List[Dict[str, Any]]) -> str:
    """Return a Python source fragment ``TOOLS: List[Tool] = [...]``."""
    buf = io.StringIO()
    buf.write("TOOLS: List[Tool] = ")
    buf.write(_serialize_value(tools, indent=0))
    return buf.getvalue()


def load_existing_tools() -> List[Dict[str, Any]]:
    """Return the TOOLS list currently materialised by tool_catalog.py."""
    return list(tc.TOOLS)


def build_fresh_tools() -> List[Dict[str, Any]]:
    """Build a new catalog from the current FastAPI OpenAPI schema."""
    return tc.build_catalog()


def diff_tools(fresh: List[Dict[str, Any]], existing: List[Dict[str, Any]]) -> str:
    """Return a human-readable diff summary of two catalogs."""
    fresh_by_name = {t["name"]: t for t in fresh}
    existing_by_name = {t["name"]: t for t in existing}

    added = sorted(fresh_by_name.keys() - existing_by_name.keys())
    removed = sorted(existing_by_name.keys() - fresh_by_name.keys())
    changed = []
    for name in sorted(fresh_by_name.keys() & existing_by_name.keys()):
        if fresh_by_name[name] != existing_by_name[name]:
            changed.append(name)

    lines = []
    if added:
        lines.append(f"Added tools ({len(added)}): {', '.join(added)}")
    if removed:
        lines.append(f"Removed tools ({len(removed)}): {', '.join(removed)}")
    if changed:
        lines.append(f"Changed tools ({len(changed)}): {', '.join(changed)}")
    if not lines:
        lines.append("Catalogs are identical.")
    return "\n".join(lines)


def write_catalog(tools: List[Dict[str, Any]], path: Path = CATALOG_PATH) -> None:
    """Replace the ``TOOLS`` annotated assignment in tool_catalog.py.

    The replacement uses the AST to locate the assignment, so it works whether
    the current value is a one-line call or a multi-line literal.
    """
    original = path.read_text(encoding="utf-8")
    tree = ast.parse(original)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "TOOLS"
        ):
            start_line = node.lineno - 1  # 0-based, inclusive
            end_line = node.end_lineno  # 1-based, exclusive
            lines = original.splitlines()
            replacement_lines = serialize_tools(tools).splitlines()
            new_lines = lines[:start_line] + replacement_lines + lines[end_line:]
            new_source = "\n".join(new_lines)
            if original.endswith("\n"):
                new_source += "\n"
            path.write_text(new_source, encoding="utf-8")
            return

    raise RuntimeError(
        f"Could not find a 'TOOLS: List[Tool] = ...' assignment in {path}"
    )


def _format_body(body: Dict[str, str]) -> str:
    if not body:
        return "—"
    return ", ".join(f"`{k}`: `{v}`" for k, v in body.items())


def _format_requires(reqs: Dict[str, str]) -> str:
    if not reqs:
        return "—"
    return ", ".join(f"`{k}`: `{v}`" for k, v in reqs.items())


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(tools: List[Dict[str, Any]]) -> str:
    """Render the catalog as a human-readable Markdown table."""
    lines = [
        "# uSTAT Agent Tool Catalog",
        "",
        "Auto-generated reference for the analysis endpoints the LLM agent can call.",
        "Each row is one tool; `session_id` is injected by the client and is omitted from the body column.",
        "",
        "| Name | Method | Path | Body args | Requires | Description |",
        "|---|---|---|---|---|---|",
    ]
    for tool in tools:
        name = tool["name"]
        method = tool["method"]
        path = f"`{tool['path']}`"
        body = _format_body(tool.get("body", {}))
        requires = _format_requires(tool.get("requires", {}))
        doc = _escape_cell(tool.get("doc", ""))
        lines.append(f"| `{name}` | {method} | {path} | {body} | {requires} | {doc} |")
    lines.append("")
    lines.append(f"*Total: {len(tools)} tools*")
    lines.append("")
    return "\n".join(lines)


def write_docs(tools: List[Dict[str, Any]], path: Path = DOCS_PATH) -> None:
    """Write the Markdown rendering of the catalog to disk."""
    path.write_text(render_markdown(tools), encoding="utf-8")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate backend/agent/tool_catalog.py against the running FastAPI app.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite tool_catalog.py and docs/AGENT_TOOLS.md from the current OpenAPI schema.",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=DOCS_PATH,
        metavar="PATH",
        help=f"Markdown output path (default: {DOCS_PATH}).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_PATH,
        metavar="PATH",
        help=f"Python catalog path (default: {CATALOG_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    fresh = build_fresh_tools()
    existing = load_existing_tools()

    print(f"OpenAPI routes scanned: {len(fresh)} tools")
    print(f"Existing catalog tools: {len(existing)} tools")

    if args.write:
        write_catalog(fresh, path=args.catalog)
        write_docs(fresh, path=args.docs)
        print(f"\nWrote {args.catalog}")
        print(f"Wrote {args.docs}")
        return 0

    print("\n" + diff_tools(fresh, existing))

    if fresh != existing:
        print(
            "\nDrift detected. Run with --write to regenerate the catalog and docs.",
            file=sys.stderr,
        )
        return 1

    print("\nCatalog is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
