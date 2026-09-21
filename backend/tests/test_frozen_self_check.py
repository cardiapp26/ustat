"""frozen_self_check must fail on a missing module and ignore optional ones.

It is the release workflow's guard against shipping a frozen backend that dies
on an import PyInstaller did not bundle, so a false pass is the failure mode
that matters: v3.7.0 went out with pyreadstat's compiled submodule missing.
"""

from pathlib import Path

import frozen_self_check


def _write(root: Path, name: str, source: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_passes_when_every_import_resolves(tmp_path, capsys):
    _write(tmp_path, "ok.py", "import numpy\nfrom pandas import DataFrame\n")
    assert frozen_self_check.run(tmp_path) == 0
    assert "2 modules, 0 failed" in capsys.readouterr().out


def test_fails_on_a_missing_module_even_inside_a_function(tmp_path, capsys):
    _write(tmp_path, "lazy.py", "def analysis():\n    import ustat_no_such_module_xyz\n")
    assert frozen_self_check.run(tmp_path) == 1
    assert "FAIL ustat_no_such_module_xyz" in capsys.readouterr().out


def test_fails_on_a_missing_submodule_named_in_from_import(tmp_path, capsys):
    # `from pkg import sub` where sub is a module, not an attribute: importing
    # pkg alone would not notice sub is absent.
    _write(tmp_path, "sub.py", "from numpy import ustat_no_such_submodule_xyz\n")
    assert frozen_self_check.run(tmp_path) == 1
    assert "FAIL numpy" in capsys.readouterr().out


def test_skips_optional_type_checking_relative_and_stdlib_imports(tmp_path, capsys):
    _write(
        tmp_path,
        "skipped.py",
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING\n"
        "import os, fcntl, winreg\n"
        "from . import sibling\n"
        "try:\n"
        "    import ustat_optional_xyz\n"
        "except ImportError:\n"
        "    ustat_optional_xyz = None\n"
        "if TYPE_CHECKING:\n"
        "    import ustat_typing_only_xyz\n",
    )
    assert frozen_self_check.run(tmp_path) == 0
    assert "0 modules, 0 failed" in capsys.readouterr().out


def test_ignores_tests_and_hidden_directories(tmp_path):
    _write(tmp_path, "tests/test_x.py", "import ustat_test_only_xyz\n")
    _write(tmp_path, ".venv/lib/site.py", "import ustat_venv_only_xyz\n")
    assert frozen_self_check.run(tmp_path) == 0
