"""
Import every module the backend's code names, inside the frozen binary.

PyInstaller finds imports by reading bytecode, which includes imports inside
functions, but it cannot see an import made from compiled code. A C extension
that imports a sibling module at load time (pyreadstat's .pyx importing
_readstat_writer is how v3.7.0 broke) is missing from the build, and nothing
fails until that import runs. For a module imported lazily by one analysis,
that is the first time a user runs it.

So the frozen backend can be asked to run every import its own source
contains: `ustat-backend --self-check`. The release workflow runs it on each
platform before anything is packaged.

Deliberately skipped:
- imports inside `try:` blocks that catch ImportError (optional features),
- imports under `if TYPE_CHECKING:`,
- relative imports, and the standard library, whose platform-specific modules
  (fcntl, winreg, ...) are imported behind platform checks this cannot read.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

# Hidden directories cover .venv, caches and tool state in a source checkout.
SKIP_DIRS = {"tests", "__pycache__", "node_modules", "venv"}
IMPORT_ERROR_NAMES = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(t, ast.Name) and t.id in IMPORT_ERROR_NAMES for t in types)


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


class _ImportCollector(ast.NodeVisitor):
    """Collect (module, names) for every unguarded absolute import."""

    def __init__(self) -> None:
        self.imports: set[tuple[str, tuple[str, ...]]] = set()
        self._guarded = 0

    def visit_Try(self, node: ast.Try) -> None:
        guarded = any(_catches_import_error(h) for h in node.handlers)
        self._guarded += guarded
        for stmt in node.body:
            self.visit(stmt)
        self._guarded -= guarded
        for part in (*node.handlers, *node.orelse, *node.finalbody):
            self.visit(part)

    visit_TryStar = visit_Try

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking(node.test):
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if not self._guarded:
            self.imports.update((alias.name, ()) for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not self._guarded and node.level == 0 and node.module:
            names = tuple(a.name for a in node.names if a.name != "*")
            self.imports.add((node.module, names))


def _collect(backend_dir: Path) -> set[tuple[str, tuple[str, ...]]]:
    collector = _ImportCollector()
    for path in sorted(backend_dir.rglob("*.py")):
        parts = path.relative_to(backend_dir).parts
        if SKIP_DIRS.intersection(parts) or any(p.startswith(".") for p in parts):
            continue
        # Bytes, so ast honours a file's own encoding declaration.
        collector.visit(ast.parse(path.read_bytes(), filename=str(path)))
    stdlib = sys.stdlib_module_names
    return {(m, names) for m, names in collector.imports if m.split(".")[0] not in stdlib}


def _check(module: str, names: tuple[str, ...]) -> str | None:
    """Import `module`, and each of `names` that is a submodule rather than an
    attribute. Return why it failed, or None."""
    try:
        mod = importlib.import_module(module)
        for name in names:
            if not hasattr(mod, name):
                importlib.import_module(f"{module}.{name}")
    except Exception as e:  # noqa: BLE001 -- any failure here is the finding
        return f"{type(e).__name__}: {e}"
    return None


def run(backend_dir: Path) -> int:
    imports = _collect(backend_dir)
    failures = [(m, why) for m, names in sorted(imports) if (why := _check(m, names))]
    for module, why in failures:
        print(f"FAIL {module}: {why}")
    checked = len({m for m, _ in imports})
    print(f"self-check: {checked} modules, {len(failures)} failed")
    return 1 if failures else 0
