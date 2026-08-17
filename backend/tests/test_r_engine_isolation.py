"""The R engine's constraints, enforced by reading its source.

`backend/ustat_engine_r/` is not imported by anything on this server -- it is
concatenated into one script and eval'd inside a webR worker in a user's tab.
Nothing about a green backend run would reveal a `library(nortest)` at file
scope, a `print()` in the middle of an analysis, or a `round()` that quietly
turns a result field into a fixture that can never be compared against the
Python engine's. Those show up as a broken tab, on data we cannot see.

So the rules are checked by reading the files, the same way
test_engine_isolation.py reads the Python engine's AST. There is no R parser
here, so these are regexes over source text -- which is why the rules are
written to be checkable that way, and why the R sources are kept in a form a
regex can read (see fingerprint.py on the ustat_register format).

Every rule below is one that has a plausible way of getting into the tree:

  library()/require() at file scope  -- works on a developer's machine where
      the package happens to be installed, and fails in the browser, where a
      package that was not in the boot plan cannot be fetched from inside a
      synchronous call. Packages are DECLARED in `packages =` and attached by
      the host; sources call through `pkg::fn`.
  install.packages/download.file/url/system/setwd/Sys.setenv/readline -- each
      either reaches the network from inside an analysis (defeating the point
      of computing locally) or blocks on something a worker cannot provide.
  `<<-` at file scope -- the bundle is one flat script with no namespaces, so a
      superassignment writes into the global environment the host also owns.
  print()/cat() -- a worker's stdout goes nowhere useful, and a bundle that
      printed on load would be indistinguishable from one that failed.
  format()/signif()/round() in analyses/ -- formatting is the client's job, and
      a rounded number in a result is a value the other engine can never be
      compared against below the rounding. runtime/stats.R is allowed them,
      because the Python engine itself rounds cohen_d and group_summary and
      those rounded values ARE the contract; nothing in analyses/ has that
      excuse. Functions whose name ends in _text are exempt: they produce prose.
  non-ASCII bytes -- the bundle is fetched, decoded and eval'd by a browser
      pipeline we do not control end to end, and R's own parser marks a string
      literal's encoding from the session locale. A \\u escape is unambiguous
      under both. See the note at the top of runtime/errors.R.
"""
from __future__ import annotations

import pathlib
import re

import pytest

R_ENGINE_DIR = pathlib.Path(__file__).resolve().parent.parent / "ustat_engine_r"
RUNTIME_DIR = R_ENGINE_DIR / "runtime"
ANALYSES_DIR = R_ENGINE_DIR / "analyses"

# Calls that must not appear anywhere in the tree, and why.
FORBIDDEN_ANYWHERE = {
    "install.packages": "the host installs declared packages; an analysis must not reach the network",
    "download.file": "same -- a local computation that downloads is not a local computation",
    "system": "there are no processes to spawn in a browser tab",
    "system2": "same as system",
    "setwd": "there is no working directory worth changing in a worker",
    "Sys.setenv": "the host owns the environment; an engine that edits it is not reproducible",
    "readline": "nothing can answer a prompt in a worker",
    "readLines": "the engine is handed its data, it does not read files",
    "print": "a worker's stdout goes nowhere a user can see",
    "cat": "same as print -- and a bundle that prints on load looks like one that failed",
    "source": "the bundle is already one flat script; nothing else is loaded at runtime",
    "eval": "nothing here needs to build code from data",
}

# Calls that must not appear at FILE SCOPE (column 0). Inside a function they
# would still be wrong for other reasons, but at file scope they run the moment
# the host eval's the bundle, before any host code can react.
FORBIDDEN_AT_FILE_SCOPE = {
    "library": "packages are declared in `packages =` and attached by the host",
    "require": "same as library, and it fails silently by returning FALSE",
}

# Formatting calls barred from analyses/, outside a *_text function.
FORMATTING_CALLS = ("format", "signif", "round")

_FUNCTION_DEF = re.compile(r"^([A-Za-z._][A-Za-z0-9._]*)\s*(?:<-|=)\s*function\b")


def _r_sources() -> list[pathlib.Path]:
    return sorted(R_ENGINE_DIR.rglob("*.R"))


def _call_re(name: str) -> re.Pattern[str]:
    """`name(` as a whole token -- so `sprintf(` never reads as `print(`."""
    return re.compile(r"(?<![A-Za-z0-9._])" + re.escape(name) + r"\s*\(")


def test_r_engine_package_exists_and_has_modules():
    sources = _r_sources()
    assert sources, f"no R files under {R_ENGINE_DIR}"
    assert (RUNTIME_DIR / "registry.R").is_file()
    assert list(ANALYSES_DIR.glob("*.R")), "no analyses registered"


@pytest.mark.parametrize("path", _r_sources(), ids=lambda p: p.name)
def test_r_module_is_pure_ascii(path: pathlib.Path):
    data = path.read_bytes()
    offenders = [
        (i, line.decode("utf-8", "replace"))
        for i, line in enumerate(data.split(b"\n"), 1)
        if any(b > 127 for b in line)
    ]
    assert not offenders, "\n".join(
        f"{path.name}:{i} has a byte above 0x7f -- write it as a \\u escape: {line}"
        for i, line in offenders
    )


@pytest.mark.parametrize("path", _r_sources(), ids=lambda p: p.name)
def test_r_module_calls_nothing_the_browser_lacks(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        code = _strip_comment(line)
        for name, why in FORBIDDEN_ANYWHERE.items():
            if _call_re(name).search(code):
                problems.append(f"{path.name}:{lineno} calls {name}() -- {why}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", _r_sources(), ids=lambda p: p.name)
def test_r_module_does_nothing_at_file_scope_but_define_and_register(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        code = _strip_comment(line)
        if not code.strip() or code[:1].isspace():
            continue  # indented: inside something, not file scope
        for name, why in FORBIDDEN_AT_FILE_SCOPE.items():
            if _call_re(name).search(code):
                problems.append(f"{path.name}:{lineno} calls {name}() at file scope -- {why}")
        if "<<-" in code:
            problems.append(
                f"{path.name}:{lineno} superassigns at file scope -- "
                "the bundle shares the host's global environment"
            )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "path", sorted(ANALYSES_DIR.glob("*.R")), ids=lambda p: p.name
)
def test_analysis_module_leaves_formatting_to_the_client(path: pathlib.Path):
    """No format/signif/round in analyses/, except inside a *_text function.

    A rounded number in a result is a value the other engine can never be
    compared against below the rounding.
    """
    problems: list[str] = []
    current = "<file scope>"
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = _strip_comment(line)
        definition = _FUNCTION_DEF.match(code)
        if definition:
            current = definition.group(1)
        if current.endswith("_text"):
            continue
        for name in FORMATTING_CALLS:
            if _call_re(name).search(code):
                problems.append(
                    f"{path.name}:{lineno} calls {name}() inside {current}() -- "
                    "formatting belongs in runtime/text.R, in a function named *_text"
                )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "path", sorted(ANALYSES_DIR.glob("*.R")), ids=lambda p: p.name
)
def test_analysis_module_ends_with_a_registration(path: pathlib.Path):
    """An analysis file that does not register is dead code the bundle still
    carries, and one whose registration is not the last statement is one whose
    fn may not be defined when it runs."""
    text = path.read_text(encoding="utf-8")
    start = text.rfind("ustat_register(")
    assert start != -1, f"{path.name}: no ustat_register(...) call"
    tail = text[:start].rstrip()
    assert not tail.endswith(")") or "function" in text[:start], (
        f"{path.name}: nothing may follow the registration"
    )
    after = text[start:]
    assert re.search(r'\bid\s*=\s*"([^"]+)"', after), (
        f'{path.name}: registration has no id = "..." literal'
    )


def test_registered_ids_are_unique_and_parsed():
    from ustat_engine_r.fingerprint import analyses

    declared = analyses()
    assert declared, "no analyses parsed out of the R sources"
    ids = [a["id"] for a in declared]
    assert len(ids) == len(set(ids))
    assert "stats.ttest" in ids


def test_every_analysis_declares_its_packages_explicitly():
    """`packages` is what the host installs before calling in. An analysis that
    silently relied on a package another one happened to pull in would work
    until it ran first."""
    from ustat_engine_r.fingerprint import analyses

    for spec in analyses():
        assert isinstance(spec["packages"], list), spec["id"]
        assert isinstance(spec["needs_frame"], bool), spec["id"]


def _strip_comment(line: str) -> str:
    """Drop a trailing R comment, respecting quotes.

    Crude but sufficient: R has no multi-line strings and no regex literals, so
    a `#` outside a quoted run starts a comment.
    """
    out: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in line:
        if quote:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out)
