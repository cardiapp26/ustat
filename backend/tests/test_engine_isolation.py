"""The engine's constraints, enforced by reading its source.

`backend/ustat_engine/` has to import and run unchanged in two places: this server,
and CPython compiled to WebAssembly in the browser. Several of the things that
would break the browser build import perfectly well here, so nothing about a
green test run on the server would reveal them -- a `threading` import, an
`async def`, a `from fastapi import ...`. They would surface as a failure in a
user's tab, on data we cannot see, long after the commit that caused them.

So the rules are checked against the AST instead of trusted to a convention in
a docstring.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ENGINE_DIR = pathlib.Path(__file__).resolve().parent.parent / "ustat_engine"

# Import roots the engine may not depend on, and why.
FORBIDDEN_IMPORTS = {
    "fastapi": "the HTTP layer is a caller of the engine, not a dependency",
    "starlette": "same as fastapi",
    "pydantic": "pydantic-core is a Rust extension; the engine takes plain dicts",
    "threading": "the browser build is single-threaded",
    "subprocess": "there are no processes to spawn in a browser tab",
    "asyncio": "offloading is the server adapter's job, not the engine's",
    "multiprocessing": "no process pool exists in the browser",
    "psutil": "there is no host to inspect",
}

# Modules that are fine on their own but drag a forbidden dependency behind
# them. services.store starts a daemon thread at import time.
FORBIDDEN_FROM = {
    "services.store": "it starts a background thread when imported",
    "services": "import the specific pure module, not the package",
}


def _engine_sources() -> list[pathlib.Path]:
    return sorted(p for p in ENGINE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_roots(tree: ast.AST) -> list[tuple[str, int]]:
    """Every module name the file imports, with the line it appears on."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # A relative import (level > 0) is inside the engine by definition.
            if node.level == 0 and node.module:
                found.append((node.module, node.lineno))
    return found


def test_engine_package_exists_and_has_modules():
    sources = _engine_sources()
    assert sources, f"no python files under {ENGINE_DIR}"


@pytest.mark.parametrize("path", _engine_sources(), ids=lambda p: p.name)
def test_engine_module_imports_nothing_the_browser_lacks(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    problems = []

    for module, lineno in _imported_roots(tree):
        root = module.split(".")[0]
        if root in FORBIDDEN_IMPORTS:
            problems.append(f"{path.name}:{lineno} imports {module!r} — {FORBIDDEN_IMPORTS[root]}")
        for banned, why in FORBIDDEN_FROM.items():
            if module == banned or module.startswith(banned + "."):
                problems.append(f"{path.name}:{lineno} imports {module!r} — {why}")

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", _engine_sources(), ids=lambda p: p.name)
def test_engine_module_defines_no_coroutines(path: pathlib.Path):
    """An `async def` here cannot be awaited by the browser's synchronous call."""
    tree = ast.parse(path.read_text(), filename=str(path))
    coros = [
        f"{path.name}:{n.lineno} async def {n.name}"
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef)
    ]
    assert not coros, "\n".join(coros)


def test_engine_imports_without_a_server():
    """The whole package must import with nothing from the web stack loaded."""
    import ustat_engine as engine  # noqa: F401

    assert engine.ANALYSES, "no analyses registered — the registry is empty"


def test_registered_analyses_declare_what_they_need():
    import ustat_engine as engine
    import ustat_engine.stats.power  # noqa: F401  (registers itself on import)

    for analysis_id, spec in engine.ANALYSES.items():
        assert spec.id == analysis_id, f"{analysis_id} registered under a different id"
        assert callable(spec.fn), f"{analysis_id} has no callable"
        assert spec.deps, f"{analysis_id} declares no packages; the browser cannot preload"
        assert spec.doc, f"{analysis_id} has no description"


def test_registering_the_same_id_twice_is_refused():
    """Two analyses answering to one name would let import order pick a winner."""
    import ustat_engine as engine
    from ustat_engine.spec import AnalysisSpec

    existing = next(iter(engine.ANALYSES))
    with pytest.raises(ValueError, match="already registered"):
        engine.register(AnalysisSpec(id=existing, fn=lambda p: {}, doc="duplicate"))


def test_unknown_analysis_is_a_404_not_a_key_error():
    import ustat_engine as engine

    with pytest.raises(engine.EngineError) as exc:
        engine.get("stats.no_such_thing")
    assert exc.value.status_hint == 404
