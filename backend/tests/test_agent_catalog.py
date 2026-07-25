"""Tests for the Phase-2 catalog generator and semantic guards.

These tests verify:
* check_requires rejects mistyped column selections before the API is hit.
* scripts/generate_tool_catalog.py can build, diff, serialize and overwrite the catalog.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from agent.tool_catalog import TOOLS, Tool, check_requires, get_tool

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
GENERATOR_PATH = SCRIPTS_DIR / "generate_tool_catalog.py"


def _load_generator_module() -> ModuleType:
    """Import scripts/generate_tool_catalog.py without relying on sys.path."""
    spec = importlib.util.spec_from_file_location(
        "generate_tool_catalog", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The script itself inserts backend/ onto sys.path, so execution is enough.
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_generator_module()


# ──────────────────────────────────────────────────────────────────────────────
# Semantic guards (check_requires)
# ──────────────────────────────────────────────────────────────────────────────


def _tool(name: str) -> Tool:
    tool = get_tool(name)
    assert tool is not None, f"Tool '{name}' not found in catalog"
    return tool


def test_check_requires_allows_valid_categorical_2x2() -> None:
    fisher = _tool("fisher")
    kinds = {"ARM": "categorical", "STROKE": "categorical"}
    levels = {"ARM": 2, "STROKE": 2}
    assert (
        check_requires(
            fisher, {"row_column": "STROKE", "col_column": "ARM"}, kinds, levels
        )
        is None
    )


def test_check_requires_rejects_wrong_level_count() -> None:
    fisher = _tool("fisher")
    kinds = {"ARM": "categorical", "STROKE": "categorical"}
    levels = {"ARM": 3, "STROKE": 2}
    err = check_requires(
        fisher, {"row_column": "STROKE", "col_column": "ARM"}, kinds, levels
    )
    assert err is not None
    assert "2 levels" in err
    assert "ARM" in err


def test_check_requires_rejects_wrong_kind() -> None:
    ttest = _tool("ttest")
    kinds = {"AGE": "categorical", "ARM": "categorical"}
    err = check_requires(ttest, {"column": "AGE", "group_column": "ARM"}, kinds)
    assert err is not None
    assert "numeric" in err
    assert "AGE" in err


def test_check_requires_allows_any_for_variables_list() -> None:
    table1 = _tool("table1")
    kinds = {"AGE": "numeric", "SEX": "categorical"}
    assert (
        check_requires(
            table1, {"variables": ["AGE", "SEX"], "group_column": "SEX"}, kinds
        )
        is None
    )


def test_check_requires_skips_missing_args() -> None:
    """Optional arguments that are omitted should not trigger guard errors."""
    table1 = _tool("table1")
    kinds = {"AGE": "numeric"}
    assert check_requires(table1, {"variables": ["AGE"]}, kinds) is None


def test_check_requires_numeric_pair() -> None:
    corr = _tool("correlation_pair")
    kinds = {"A": "numeric", "B": "numeric"}
    assert check_requires(corr, {"var1": "A", "var2": "B"}, kinds) is None


def test_check_requires_rejects_non_numeric_pair() -> None:
    corr = _tool("correlation_pair")
    kinds = {"A": "numeric", "B": "categorical"}
    err = check_requires(corr, {"var1": "A", "var2": "B"}, kinds)
    assert err is not None
    assert "numeric" in err


# ──────────────────────────────────────────────────────────────────────────────
# Generator helpers
# ──────────────────────────────────────────────────────────────────────────────


def test_serialize_tools_roundtrips(generator) -> None:
    """The serializer must emit valid Python that evaluates back to the input."""
    sample = [
        {
            "name": "demo",
            "method": "POST",
            "path": "/api/demo",
            "session_in": "body",
            "body": {"x": "str", "y": "list[str]?"},
            "requires": {"x": "numeric"},
            "doc": "Demo tool.",
        }
    ]
    source = generator.serialize_tools(sample)
    namespace: dict = {}
    exec(source + "\nresult = TOOLS", namespace)
    assert namespace["result"] == sample


def test_diff_tools_reports_added_removed_and_changed(generator) -> None:
    a = [{"name": "x", "method": "POST", "path": "/api/x"}]
    b = [{"name": "y", "method": "POST", "path": "/api/y"}]
    same = [{"name": "x", "method": "POST", "path": "/api/x"}]

    diff = generator.diff_tools(a, [])
    assert "Added tools (1): x" in diff

    diff = generator.diff_tools([], a)
    assert "Removed tools (1): x" in diff

    diff = generator.diff_tools(a, b)
    assert "Added tools (1): x" in diff
    assert "Removed tools (1): y" in diff

    diff = generator.diff_tools(a, same)
    assert "Catalogs are identical" in diff


def test_render_markdown_contains_headers_and_tool(generator) -> None:
    sample = [
        {
            "name": "fisher",
            "method": "POST",
            "path": "/api/stats/fisher",
            "session_in": "body",
            "body": {"row_column": "str", "col_column": "str"},
            "requires": {
                "row_column": "categorical(2)",
                "col_column": "categorical(2)",
            },
            "doc": "Fisher exact test.",
        }
    ]
    md = generator.render_markdown(sample)
    assert "# uSTAT Agent Tool Catalog" in md
    assert "| Name | Method | Path | Body args | Requires | Description |" in md
    assert "`fisher`" in md
    assert "`row_column`: `categorical(2)`" in md


def test_write_catalog_overwrites_static_tools(generator, tmp_path) -> None:
    """write_catalog must replace a TOOLS assignment while preserving surrounding code."""
    original_source = """from typing import List

class Tool: ...

# comment
TOOLS: List[Tool] = [
    {"name": "old"},
]

__all__ = ["TOOLS"]
"""
    catalog_file = tmp_path / "tool_catalog.py"
    catalog_file.write_text(original_source, encoding="utf-8")

    new_tools = [{"name": "new", "method": "POST", "path": "/api/new"}]
    generator.write_catalog(new_tools, path=catalog_file)

    new_source = catalog_file.read_text(encoding="utf-8")
    assert "# comment" in new_source
    assert "__all__" in new_source
    assert "old" not in new_source
    assert "new" in new_source

    namespace: dict = {}
    exec(new_source + "\nresult = TOOLS", namespace)
    assert namespace["result"] == new_tools


def test_generator_main_reports_no_drift(generator, monkeypatch) -> None:
    """When the existing catalog equals the freshly built one, main exits 0."""

    def fake_build() -> list:
        return list(TOOLS)

    def fake_load() -> list:
        return list(TOOLS)

    monkeypatch.setattr(generator, "build_fresh_tools", fake_build)
    monkeypatch.setattr(generator, "load_existing_tools", fake_load)

    rc = generator.main([])
    assert rc == 0


def test_generator_main_reports_drift(generator, monkeypatch) -> None:
    """When catalogs differ, main exits 1 (without --write)."""
    existing = [{"name": "old"}]
    fresh = [{"name": "new"}]

    monkeypatch.setattr(generator, "build_fresh_tools", lambda: fresh)
    monkeypatch.setattr(generator, "load_existing_tools", lambda: existing)

    rc = generator.main([])
    assert rc == 1


def test_generator_main_write_flag_regenerates(
    generator, monkeypatch, tmp_path
) -> None:
    """With --write, main writes the catalog and docs even when no drift is present."""
    catalog = tmp_path / "tool_catalog.py"
    docs = tmp_path / "AGENT_TOOLS.md"

    sample = [
        {
            "name": "x",
            "method": "GET",
            "path": "/api/x",
            "session_in": "body",
            "body": {},
            "requires": {},
            "doc": "x",
        }
    ]

    # Seed catalog with a valid TOOLS assignment.
    catalog.write_text(
        "from typing import List\nclass Tool: ...\nTOOLS: List[Tool] = []\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(generator, "build_fresh_tools", lambda: sample)
    monkeypatch.setattr(generator, "load_existing_tools", lambda: sample)

    rc = generator.main(["--write", "--catalog", str(catalog), "--docs", str(docs)])
    assert rc == 0
    assert catalog.exists()
    assert docs.exists()
    assert "x" in docs.read_text(encoding="utf-8")
