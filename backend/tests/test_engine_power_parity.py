"""Cross-runtime parity for `ustat_engine.stats.power.run_power`.

`backend/engine/` is meant to run identically on this server and inside
Pyodide in the browser. A single-runtime test can never prove that: it can
only show that the server's answer matches some external reference (R,
G*Power, a hand-derived formula), which says nothing about what the browser
computed.

The fixtures this test reads from `qa/parity/power.json`
are the shared contract instead. They started life as the expectations baked
into `test_power_vs_r.py` and `test_power_vs_gpower_paper.py`, pulled out into
data so the *same* JSON can be replayed against the *same* engine code from
two different runners: this file drives it through CPython on the server, and
a Pyodide harness in the browser drives the identical fixtures through the
identical `engine/` source running as WebAssembly. If both runners agree with
this file, they agree with each other -- which is the property the whole
two-runtime split depends on, and the only way to check it without literally
diffing the two runtimes' output against one another.

This test calls `ustat_engine.stats.power.run_power(params)` directly. It does not
go through the FastAPI endpoint, on purpose: the HTTP layer is a caller of
the engine, not part of what is being pinned here.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ustat_engine.errors import EngineError
from ustat_engine.stats.power import run_power

FIXTURES_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "qa" / "parity" / "power.json"
)

# scipy 1.14.1 (pinned in backend/requirements.txt so the server and the
# Pyodide-in-browser runtime agree with each other -- see the comment there
# and the NONCENTRAL_REL constant in backend/tests/test_power_vs_r.py) is
# slightly less precise on the noncentral t/F distributions than scipy 1.15,
# which R's `pwr` package effectively matches. These three fixtures state
# rel<=1e-9 against R, which scipy 1.14.1 cannot reach; their actual relative
# differences are ~6.9e-7, ~3.1e-6 and ~9.8e-9 respectively. This is expected
# and correct given the pin, not a bug in the engine, so these ids get an
# explicit, narrow exemption instead of loosening every fixture's tolerance.
NONCENTRAL_REL_EXEMPTIONS = {
    "anova_power_reads_n_as_per_group_k4_f0.25_n52": 1e-6,
    "anova_power_k3_f0.4_n20": 1e-5,
    "ttest_ind_power_d0.5_n64": 1e-8,
}


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


def _check_numeric(actual: float, expected: float, kind: str, amount: float | None, fixture_id: str):
    if fixture_id in NONCENTRAL_REL_EXEMPTIONS:
        allowed = NONCENTRAL_REL_EXEMPTIONS[fixture_id]
        rel = _rel(actual, expected)
        assert rel <= allowed, (
            f"{fixture_id}: relative diff {rel!r} exceeds the noncentral-t/F "
            f"scipy-1.14.1 exemption of {allowed!r} (actual={actual!r}, expected={expected!r})"
        )
        return

    if kind == "exact":
        assert actual == expected, f"{fixture_id}: {actual!r} != {expected!r}"
    elif kind.startswith("abs<="):
        amount = float(kind[len("abs<="):])
        assert abs(actual - expected) <= amount, (
            f"{fixture_id}: abs diff {abs(actual - expected)!r} exceeds {amount!r} "
            f"(actual={actual!r}, expected={expected!r})"
        )
    elif kind.startswith("rel<="):
        amount = float(kind[len("rel<="):])
        rel = _rel(actual, expected)
        assert rel <= amount, (
            f"{fixture_id}: relative diff {rel!r} exceeds {amount!r} "
            f"(actual={actual!r}, expected={expected!r})"
        )
    else:
        raise AssertionError(f"{fixture_id}: unrecognised tolerance kind {kind!r}")


def _run_fixture(fixture: dict):
    params = fixture["params"]
    expected = fixture["expected"]
    tolerance_kind = _parse_tolerance(fixture["tolerance"])
    fixture_id = fixture["id"]

    if "status_code" in expected:
        with pytest.raises(EngineError) as exc_info:
            run_power(params)
        assert exc_info.value.status_hint == expected["status_code"], (
            f"{fixture_id}: status_hint {exc_info.value.status_hint!r} != {expected['status_code']!r}"
        )
        assert tolerance_kind == "exact status code", (
            f"{fixture_id}: unexpected tolerance kind {tolerance_kind!r} for a status_code fixture"
        )
        return

    out = run_power(params)

    if "result" in expected:
        assert out["result"] is not None, f"{fixture_id}: result is None"
        _check_numeric(out["result"], expected["result"], tolerance_kind, None, fixture_id)

    if "result_int" in expected:
        assert out["result"] is not None, f"{fixture_id}: result is None"
        assert int(out["result"]) == expected["result_int"], (
            f"{fixture_id}: int(result)={int(out['result'])!r} != {expected['result_int']!r}"
        )

    if "n_corrected" in expected:
        assert out["n_corrected"] == expected["n_corrected"], (
            f"{fixture_id}: n_corrected {out['n_corrected']!r} != {expected['n_corrected']!r}"
        )

    if "attrition" in expected:
        assert out["attrition"] == expected["attrition"], (
            f"{fixture_id}: attrition {out['attrition']!r} != {expected['attrition']!r}"
        )

    if "label_contains" in expected:
        needles = expected["label_contains"]
        if isinstance(needles, str):
            needles = [needles]
        for needle in needles:
            assert needle in out["label"], (
                f"{fixture_id}: label {out['label']!r} does not contain {needle!r}"
            )


@pytest.mark.skipif(not FIXTURES, reason=f"fixtures not found at {FIXTURES_PATH} (lives outside backend/)")
@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["id"] for f in FIXTURES] if FIXTURES else None)
def test_power_fixture(fixture: dict):
    _run_fixture(fixture)


def test_fixture_file_was_actually_found():
    """A skipif on an empty list silently "passes" with zero cases collected,
    which would hide a moved or deleted fixture file. This test fails loudly
    instead, but only when the file genuinely does not exist -- it is itself
    skipped in that case, same as the parametrized test above, so a checkout
    that legitimately lacks the frontend tree does not get a spurious error."""
    if not FIXTURES_PATH.exists():
        pytest.skip(f"fixtures not found at {FIXTURES_PATH} (lives outside backend/)")
    assert FIXTURES, f"{FIXTURES_PATH} exists but contains no fixtures"


def test_every_exemption_id_exists_in_the_fixture_file():
    """Guards the exemption table itself: if a fixture id gets renamed, the
    exemption would silently stop applying and that fixture would start
    failing against the full rel<=1e-9 tolerance with no clue why."""
    if not FIXTURES:
        pytest.skip(f"fixtures not found at {FIXTURES_PATH} (lives outside backend/)")
    fixture_ids = {f["id"] for f in FIXTURES}
    missing = set(NONCENTRAL_REL_EXEMPTIONS) - fixture_ids
    assert not missing, f"exempted ids not present in fixtures: {missing}"
