"""Cross-runtime parity for `ustat_engine.stats.ttest.run_ttest`.

Same contract as `test_engine_power_parity.py`, and for the same reason: the
fixtures in `qa/parity/ttest.json` are data, so the identical file can be
replayed against the identical engine source from CPython here and from a
Pyodide harness in a browser. If both runners agree with this file they agree
with each other, which is the property the two-runtime split rests on.

One thing is different, and it is the point of P1. Power analysis reads no
dataset; a t-test does. So each fixture names a CSV, the columns it transfers
and the Select Cases conditions in force, and the frame is built the way the
browser gets one -- `build_envelope` then `frame_from_envelope` -- rather than
by `pd.read_csv` straight into the analysis. Reading the CSV directly would
test the arithmetic and skip the entire question P1 exists to answer: whether a
dataset that has been through the envelope is still the same dataset. Kinds are
merged exactly as `GET /api/sessions/{id}/frame` merges them, so a
numeric-coded categorical arrives categorical here too.

Fixtures carrying `through_registry` go via `ustat_engine.run`, which is what
the worker calls, so the 409 filter guard is exercised rather than assumed.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import ustat_engine
from ustat_engine.errors import EngineError
from ustat_engine.frame.envelope import build_envelope, frame_from_envelope
from ustat_engine.stats.ttest import run_ttest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES_PATH = REPO_ROOT / "qa" / "parity" / "ttest.json"


def _load_fixtures() -> list[dict]:
    if not FIXTURES_PATH.exists():
        return []
    return json.loads(FIXTURES_PATH.read_text())


FIXTURES = _load_fixtures()


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _parse_tolerance(tolerance: str) -> str:
    """The machine-readable part is everything before the first '(' — the
    rest is human provenance (what produced the expected value)."""
    return tolerance.split("(", 1)[0].strip()


def _frame_for(spec: dict):
    """The dataset the browser would have been handed, built the way it is.

    `_detect_kind` comes from the upload router because that is what
    `GET /api/sessions/{id}/frame` calls; re-deriving kinds here would mean the
    fixture ran on a differently-typed frame than a real session does.
    """
    from routers.upload import _detect_kind

    csv_path = REPO_ROOT / spec["csv"]
    df = pd.read_csv(csv_path)
    kinds = {col: _detect_kind(df[col]) for col in df.columns}
    envelope = build_envelope(
        df,
        kinds=kinds,
        columns=spec.get("columns"),
        conditions=spec.get("conditions") or [],
    )
    return frame_from_envelope(envelope)


def _check_value(actual, expected, kind: str, field: str, fixture_id: str) -> None:
    where = f"{fixture_id}.{field}"
    if isinstance(expected, str):
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"
        return
    if isinstance(expected, bool):
        assert actual is expected or actual == expected, (
            f"{where}: {actual!r} != {expected!r}"
        )
        return
    # An integer in the fixture is a count or a whole df: it is either right or
    # it is a different set of rows, and no tolerance applies to that.
    if isinstance(expected, int):
        assert float(actual) == float(expected), f"{where}: {actual!r} != {expected!r}"
        return

    assert actual is not None, f"{where}: is None"
    if kind == "exact":
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"
    elif kind.startswith("abs<="):
        amount = float(kind[len("abs<="):])
        assert abs(actual - expected) <= amount, (
            f"{where}: abs diff {abs(actual - expected)!r} exceeds {amount!r} "
            f"(actual={actual!r}, expected={expected!r})"
        )
    elif kind.startswith("rel<="):
        amount = float(kind[len("rel<="):])
        rel = _rel(actual, expected)
        assert rel <= amount, (
            f"{where}: relative diff {rel!r} exceeds {amount!r} "
            f"(actual={actual!r}, expected={expected!r})"
        )
    else:
        raise AssertionError(f"{fixture_id}: unrecognised tolerance kind {kind!r}")


def _invoke(fixture: dict, frame):
    if fixture.get("through_registry"):
        return ustat_engine.run("stats.ttest", frame, fixture["params"])
    return run_ttest(frame, fixture["params"])


def _run_fixture(fixture: dict) -> None:
    fixture_id = fixture["id"]
    expected = fixture["expected"]
    tolerance_kind = _parse_tolerance(fixture["tolerance"])
    frame = _frame_for(fixture["dataset"])

    if "status_code" in expected:
        with pytest.raises(EngineError) as exc_info:
            _invoke(fixture, frame)
        assert exc_info.value.status_hint == expected["status_code"], (
            f"{fixture_id}: status_hint {exc_info.value.status_hint!r} != "
            f"{expected['status_code']!r}"
        )
        if "message" in expected:
            assert exc_info.value.message == expected["message"], (
                f"{fixture_id}: {exc_info.value.message!r} != {expected['message']!r}"
            )
        assert tolerance_kind == "exact status code", (
            f"{fixture_id}: unexpected tolerance kind {tolerance_kind!r} for a "
            "status_code fixture"
        )
        return

    out = _invoke(fixture, frame)
    for field, want in expected.items():
        assert field in out, f"{fixture_id}: result has no {field!r}"
        _check_value(out[field], want, tolerance_kind, field, fixture_id)


@pytest.mark.skipif(
    not FIXTURES, reason=f"fixtures not found at {FIXTURES_PATH} (lives outside backend/)"
)
@pytest.mark.parametrize(
    "fixture", FIXTURES, ids=[f["id"] for f in FIXTURES] if FIXTURES else None
)
def test_ttest_fixture(fixture: dict):
    _run_fixture(fixture)


def test_fixture_file_was_actually_found():
    """A skipif on an empty list "passes" with zero cases collected, which would
    hide a moved or deleted fixture file."""
    if not FIXTURES_PATH.exists():
        pytest.skip(f"fixtures not found at {FIXTURES_PATH} (lives outside backend/)")
    assert FIXTURES, f"{FIXTURES_PATH} exists but contains no fixtures"


def test_the_envelope_round_trip_is_what_makes_these_fixtures_worth_anything():
    """Reading the CSV straight into the analysis must give the same numbers as
    going through the envelope.

    If it did not, every fixture above would be pinning the envelope's opinion
    of the data rather than the t-test's, and the failure would be invisible --
    both paths would still be self-consistent. This is the only test here that
    compares two paths rather than a path against a literal.
    """
    if not FIXTURES_PATH.exists():
        pytest.skip(f"fixtures not found at {FIXTURES_PATH}")
    direct = pd.read_csv(REPO_ROOT / "qa" / "models_audit" / "dataset.csv")
    params = {"column": "sbp", "group_column": "arm", "method": "welch"}

    from_csv = run_ttest(direct, params)
    from_envelope = run_ttest(
        _frame_for({"csv": "qa/models_audit/dataset.csv", "columns": ["sbp", "arm"]}),
        params,
    )
    for field in ("n1", "n2", "group1", "group2", "t", "df", "p", "mean1", "mean2"):
        assert from_envelope[field] == from_csv[field], (
            f"{field}: envelope {from_envelope[field]!r} != direct {from_csv[field]!r}"
        )


def test_registered_spec_declares_the_columns_it_will_read():
    """The `?columns=` transfer is derived from this, so a wrong answer here
    means the browser is sent the wrong slice of the patient's data -- either
    too much of it, or too little for the analysis to run."""
    spec = ustat_engine.get("stats.ttest")
    assert spec.needs_frame is True
    assert spec.columns_for({"column": "sbp", "group_column": "arm"}) == ["sbp", "arm"]
    assert spec.columns_for({"column": "sbp", "group_col": "arm"}) == ["sbp", "arm"]
    assert spec.columns_for({"column": "sbp"}) == ["sbp"]
    assert "pandas" in spec.deps
