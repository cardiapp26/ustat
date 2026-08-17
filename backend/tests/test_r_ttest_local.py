"""The R engine's t-test, run under a real R, against the Python engine's.

This is the test the whole R1a exists to make possible. Everything else checks
that the bundle is what we think it is; this one checks that what it computes is
what the other engine computes, by building the bundle, handing it the same
envelope a browser would get, and running it under Rscript.

WHAT IS ASSERTED, AND WHY EACH ONE
-----------------------------------
(a) The result KEYS are set-equal to run_ttest's for the same params. This is
    the load-bearing one. HypothesisPanel destructures fields with no optional
    chaining, so an R result missing a key is a blank panel or a crash, not a
    type error -- and the panel cannot know which engine produced what it got.
(b) t / df / p / mean1 / mean2 / n1 / n2 against the Python engine at MEASURED
    tolerances, stated in TOL below rather than picked to pass.
(c) The same numbers against qa/tests_audit/reference.json, which is R 4.5.2's
    own t.test on this dataset at 17 digits -- so the chain is closed: R here
    agrees with R there agrees with Python.
(d) A stale __filter_fingerprint gets the 409 and the exact message the Python
    registry uses. Two engines that worded this differently would mean a browser
    could not tell "wrong patients" from "unknown analysis".
(e) A Turkish/space/parenthesis column name survives the round trip. read.csv
    would have mangled it; the envelope path must not.
(f) Both variance-assumption paths and the Levene-driven auto path, because
    getting the precedence wrong does not raise -- it silently runs the other
    test.

Skipped cleanly when Rscript is absent, which is the state of most CI runners.

THE ONE FIELD THAT LEGITIMATELY DIVERGES is the Lilliefors p inside
`assumptions`, and it is asserted to differ rather than ignored -- see
KNOWN_DIVERGENCE below and runtime/stats.R for why.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess

import pandas as pd
import pytest

from ustat_engine.frame.envelope import build_envelope, frame_from_envelope
from ustat_engine.stats.ttest import run_ttest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_r_bundle.py"
DATASET = REPO_ROOT / "qa" / "models_audit" / "dataset.csv"
R_REFERENCE = REPO_ROOT / "qa" / "tests_audit" / "reference.json"

RSCRIPT = shutil.which("Rscript") or "/usr/local/bin/Rscript"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(RSCRIPT).is_file(),
    reason="Rscript not installed; the R engine cannot be executed here",
)

# Measured on R 4.5.2 / jsonlite 2.0.0 against scipy 1.14.1 over the audit
# dataset, not chosen to pass. Worst observed relative difference per field:
#
#   n1 n2 group1 group2 df df_method            exact
#   mean1 mean2 effect_sizes summary            exact (the rounded ones too)
#   t                                           6.2e-16
#   std                                         3.3e-16
#   p, independent samples                      3.9e-15
#   p, one sample (p ~ 8e-14, deep in the tail) 3.3e-14
#
# The headroom is one order of magnitude on each. Anything larger is not
# floating-point noise, it is a different computation.
TOL = {"t": 1e-14, "p": 1e-12, "std": 1e-14, "mean1": 1e-15, "mean2": 1e-15,
       "mean": 1e-15, "df": 1e-15}

# scipy's Lilliefors p comes from an interpolated table of simulated critical
# values; nortest::lillie.test uses the Dallal-Wilkinson analytic form with
# Stephens' modification. Same statistic, different approximation of the same
# null distribution. Recorded in qa/parity/ttest.json.
KNOWN_DIVERGENCE = "Kolmogorov-Smirnov (Lilliefors)"

CASES = {
    "welch": {"column": "sbp", "group_column": "arm", "method": "welch"},
    "student": {"column": "sbp", "group_column": "arm", "method": "student"},
    "auto": {"column": "sbp", "group_column": "arm"},
    "equal_var_false": {"column": "sbp", "group_column": "arm", "equal_var": False},
    "equal_var_true": {"column": "sbp", "group_column": "arm", "equal_var": True},
    # 140 rather than 140.0: JSON.stringify(140) has no decimal point, so the
    # server's json module hands the Python engine an int and R a double that
    # happens to be whole. Both then print "140". See the float-mu test below
    # for the case where they do not.
    "one_sample": {"column": "sbp", "mu": 140},
    "one_sample_float_mu": {"column": "sbp", "mu": 140.0},
    "one_sample_default_mu": {"column": "sbp"},
    "three_levels": {"column": "sbp", "group_column": "stage"},
    "missing_column": {"mu": 0.0},
    "bad_method": {"column": "sbp", "method": "bogus"},
    # The frame below is unfiltered, whose fingerprint is the sha256 of "[]".
    # This is any other 64 hex digits.
    "stale_filter": {
        "column": "sbp",
        "group_column": "arm",
        "__filter_fingerprint": "0" * 64,
    },
}

DRIVER = r"""
args <- commandArgs(trailingOnly = TRUE)
source(args[1])
frame <- ustat_frame_from_envelope(jsonlite::fromJSON(args[2], simplifyVector = FALSE))
jobs <- jsonlite::fromJSON(args[3], simplifyVector = FALSE)
out <- lapply(jobs, function(j) {
  ustat_run_json("stats.ttest", as.character(ustat_to_json(j)), frame)
})
writeLines(as.character(ustat_to_json(out)), args[4], useBytes = TRUE)
"""

# The same thing, but each job carries its own envelope -- the parity fixtures
# name different column sets and different Select Cases.
FIXTURE_DRIVER = r"""
args <- commandArgs(trailingOnly = TRUE)
source(args[1])
jobs <- jsonlite::fromJSON(args[2], simplifyVector = FALSE)
out <- lapply(jobs, function(j) {
  frame <- ustat_frame_from_envelope(j$envelope)
  ustat_run_json("stats.ttest", as.character(ustat_to_json(j$params)), frame)
})
writeLines(as.character(ustat_to_json(out)), args[3], useBytes = TRUE)
"""


def _bundle(tmp_path: pathlib.Path) -> pathlib.Path:
    spec = importlib.util.spec_from_file_location("_build_r_bundle_for_local", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    manifest = module.build(tmp_path / "bundle")
    return tmp_path / "bundle" / manifest["bundle"]


def _run_r(tmp_path: pathlib.Path, envelope: dict, jobs: dict) -> dict:
    """Evaluate the bundle under Rscript and return one parsed result per job."""
    bundle = _bundle(tmp_path)
    env_path = tmp_path / "envelope.json"
    env_path.write_text(json.dumps(envelope), encoding="utf-8")
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    driver = tmp_path / "driver.R"
    driver.write_text(DRIVER, encoding="utf-8")
    out_path = tmp_path / "out.json"

    proc = subprocess.run(
        [RSCRIPT, "--vanilla", str(driver), str(bundle), str(env_path),
         str(jobs_path), str(out_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"Rscript failed ({proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # useBytes above means the file is exactly the UTF-8 jsonlite produced.
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    return {name: json.loads(payload) for name, payload in raw.items()}


def _envelope(columns: list[str], conditions: list[dict] | None = None) -> dict:
    """The dataset the browser would have been handed, built the way it is.

    `_detect_kind` comes from the upload router because that is what
    GET /api/sessions/{id}/frame calls; re-deriving kinds here would mean the
    test ran on a differently-typed frame than a real session does.
    """
    from routers.upload import _detect_kind

    df = pd.read_csv(DATASET)
    kinds = {col: _detect_kind(df[col]) for col in df.columns}
    return build_envelope(df, kinds=kinds, columns=columns, conditions=conditions or [])


@pytest.fixture(scope="module")
def audit_envelope() -> dict:
    return _envelope(["sbp", "arm", "stage"])


@pytest.fixture(scope="module")
def r_results(tmp_path_factory, audit_envelope) -> dict:
    """Every case, in one Rscript invocation. Booting R costs ~1s; the cases do
    not, so batching keeps this test honest about what it is measuring."""
    return _run_r(tmp_path_factory.mktemp("r_engine"), audit_envelope, CASES)


@pytest.fixture(scope="module")
def python_frame(audit_envelope):
    return frame_from_envelope(audit_envelope)


def _ok(r_results: dict, name: str) -> dict:
    payload = r_results[name]
    assert payload["ok"] is True, f"{name}: R refused -- {payload.get('error')}"
    return payload["result"]


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------------
# (a) the schema contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", ["welch", "student", "auto", "equal_var_false", "one_sample"]
)
def test_r_result_has_exactly_the_python_engine_s_keys(case, r_results, python_frame):
    expected = run_ttest(python_frame, CASES[case])
    actual = _ok(r_results, case)
    assert set(actual) == set(expected), (
        f"{case}: R is missing {sorted(set(expected) - set(actual))} and has "
        f"extra {sorted(set(actual) - set(expected))}"
    )
    for key in ("effect_sizes", "assumptions", "summary"):
        assert isinstance(actual[key], type(expected[key])), key
    assert len(actual["assumptions"]) == len(expected["assumptions"])
    for got, want in zip(actual["assumptions"], expected["assumptions"]):
        assert set(got) == set(want)
    assert set(actual["effect_sizes"][0]) == set(expected["effect_sizes"][0])
    assert set(actual["summary"]) == set(expected["summary"])
    first = next(iter(expected["summary"]))
    assert set(actual["summary"][first]) == set(expected["summary"][first])


# ---------------------------------------------------------------------------
# (b) and (f) the numbers, both variance paths and the Levene-driven one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["welch", "student", "auto", "equal_var_false", "equal_var_true",
     "one_sample", "one_sample_default_mu"],
)
def test_r_agrees_with_the_python_engine(case, r_results, python_frame):
    expected = run_ttest(python_frame, CASES[case])
    actual = _ok(r_results, case)

    for key, value in expected.items():
        if key == "assumptions":
            continue  # (b') below
        got = actual[key]
        if isinstance(value, float) and not isinstance(value, bool):
            tol = TOL.get(key, 0.0)
            rel = _rel(got, value)
            assert rel <= tol, f"{case}.{key}: rel {rel:.3g} > {tol:.3g} (R {got!r}, py {value!r})"
        else:
            assert got == value, f"{case}.{key}: R {got!r} != py {value!r}"


@pytest.mark.parametrize("case", ["welch", "student", "auto"])
def test_the_variance_assumption_is_chosen_the_same_way(case, r_results, python_frame):
    expected = run_ttest(python_frame, CASES[case])
    actual = _ok(r_results, case)
    assert actual["variance_assumption"] == expected["variance_assumption"]
    assert actual["variance_assumption_selected_by"] == expected["variance_assumption_selected_by"]
    assert actual["df_method"] == expected["df_method"]
    # The Levene detail is the one assumption line that must agree exactly: it
    # is what "auto" reads to pick the test.
    assert actual["assumptions"][-1] == expected["assumptions"][-1]


def test_auto_and_student_land_on_the_same_test_on_this_dataset(r_results):
    """sbp ~ arm has a Levene p of 0.62, so "auto" must reach exactly the
    Student test -- not merely a defensible one."""
    auto = _ok(r_results, "auto")
    student = _ok(r_results, "student")
    assert auto["t"] == student["t"]
    assert auto["p"] == student["p"]
    assert auto["variance_assumption_selected_by"] == "auto (Levene)"
    assert student["variance_assumption_selected_by"] == "request (method)"


def test_equal_var_alias_still_selects_the_test(r_results):
    assert _ok(r_results, "equal_var_false")["variance_assumption"] == "welch"
    assert _ok(r_results, "equal_var_true")["variance_assumption"] == "student"
    assert (
        _ok(r_results, "equal_var_false")["variance_assumption_selected_by"]
        == "request (equal_var)"
    )


# ---------------------------------------------------------------------------
# (b') the one field that legitimately differs
# ---------------------------------------------------------------------------


def test_a_float_mu_is_worded_differently_and_that_is_recorded(r_results, python_frame):
    """The second known divergence, and the smaller one: prose only.

    Python's json module splits JSON's single number type into int and float, so
    `"mu": 140.0` reaches run_ttest as a float and prints "140.0". R has one
    numeric type and cannot see the difference, so it prints "140" -- the same
    thing it prints for `"mu": 140`, which is what a JS caller actually sends
    (JSON.stringify(140) === "140"). Every number in the payload agrees; three
    sentences differ by two characters.
    """
    expected = run_ttest(python_frame, CASES["one_sample_float_mu"])
    actual = _ok(r_results, "one_sample_float_mu")

    assert expected["interpretation"].startswith("Mean differs from 140.0 ")
    assert actual["interpretation"].startswith("Mean differs from 140 ")
    assert actual["r_code"] == "t.test(data$sbp, mu = 140)"
    assert expected["r_code"] == "t.test(data$sbp, mu = 140.0)"

    for key in ("t", "p", "df", "n", "mean", "std", "mu", "significant"):
        got, want = actual[key], expected[key]
        if isinstance(want, float):
            assert _rel(got, want) <= TOL.get(key, 1e-14), key
        else:
            assert got == want, key

    # And the int spelling -- what a browser sends -- agrees exactly.
    strict = _ok(r_results, "one_sample")
    assert strict["interpretation"] == run_ttest(python_frame, CASES["one_sample"])["interpretation"]


def test_the_lilliefors_p_differs_and_that_is_recorded(r_results, python_frame):
    """Asserted to differ, not ignored. If a future nortest or statsmodels made
    these agree, this test failing is how we would find out -- and the fixture's
    `divergence` note would have to come out with it."""
    expected = run_ttest(python_frame, CASES["welch"])
    actual = _ok(r_results, "welch")

    normality_py = [a for a in expected["assumptions"] if KNOWN_DIVERGENCE in a["detail"]]
    normality_r = [a for a in actual["assumptions"] if KNOWN_DIVERGENCE in a["detail"]]
    assert len(normality_py) == 2 and len(normality_r) == 2, (
        "both groups are n >= 50, so both should reach the Lilliefors branch"
    )
    for got, want in zip(normality_r, normality_py):
        assert got["name"] == want["name"]
        # Same verdict, different p. The verdict is what a reader acts on.
        assert got["met"] == want["met"]
        assert got["detail"] != want["detail"], (
            "the two Lilliefors approximations now agree to 4 dp -- update "
            "qa/parity/ttest.json's divergence note and this test"
        )


# ---------------------------------------------------------------------------
# (c) against R 4.5.2's own t.test, recorded at 17 digits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case,branch", [("welch", "welch"), ("student", "student")])
def test_r_agrees_with_the_recorded_r_reference(case, branch, r_results):
    reference = json.loads(R_REFERENCE.read_text())[branch]
    actual = _ok(r_results, case)

    for field, want in (
        ("t", reference["statistic"]),
        ("df", reference["parameter"]),
        ("p", reference["p.value"]),
        ("mean1", reference["estimate"][0]),
        ("mean2", reference["estimate"][1]),
    ):
        rel = _rel(actual[field], want)
        assert rel <= 1e-14, f"{case}.{field}: rel {rel:.3g} vs R 4.5.2 ({actual[field]!r} vs {want!r})"

    n_by_arm = json.loads(R_REFERENCE.read_text())["meta"]["n_by_arm"]
    assert actual["n1"] == n_by_arm["control"]
    assert actual["n2"] == n_by_arm["treat"]
    assert actual["group1"] == "control"
    assert actual["group2"] == "treat"


# ---------------------------------------------------------------------------
# (d) the errors, including the 409 both engines have to word identically
# ---------------------------------------------------------------------------


def test_a_stale_filter_fingerprint_is_a_409_with_the_python_wording(r_results):
    payload = r_results["stale_filter"]
    assert payload["ok"] is False
    assert payload["error"]["status_hint"] == 409
    assert payload["error"]["message"] == "frame does not match the active Select Cases"


@pytest.mark.parametrize(
    "case,status,message",
    [
        ("three_levels", 400, "Group column must have exactly 2 groups"),
        ("missing_column", 422, "Field 'column' is required."),
        ("bad_method", 422, "Field 'method' must be one of ('auto', 'student', 'welch')."),
    ],
)
def test_refusals_match_the_python_engine(case, status, message, r_results, python_frame):
    from ustat_engine.errors import EngineError

    payload = r_results[case]
    assert payload["ok"] is False, f"{case}: R returned a result where Python refuses"
    assert payload["error"]["status_hint"] == status
    assert payload["error"]["message"] == message

    with pytest.raises(EngineError) as exc:
        run_ttest(python_frame, CASES[case])
    assert exc.value.status_hint == status
    assert exc.value.message == message


# ---------------------------------------------------------------------------
# (e) column names travel verbatim
# ---------------------------------------------------------------------------


def test_turkish_and_spaced_column_names_round_trip(tmp_path):
    """`data.frame()` would have turned these into "Ya..y.l." and "Evre.Grubu",
    and `read.csv` would have done it before R ever saw the values. The envelope
    path has to carry them as keys, not identifiers."""
    value_col = "Yaş (yıl)"
    group_col = "Evre Grubu"
    df = pd.DataFrame(
        {
            value_col: [61.0, 58.5, 70.2, 49.9, 66.1, 55.4, 72.8, 63.3],
            group_col: ["Erken", "İleri", "Erken", "İleri", "Erken", "İleri", "Erken", "İleri"],
        }
    )
    envelope = build_envelope(
        df,
        kinds={value_col: "numeric", group_col: "categorical"},
        columns=[value_col, group_col],
    )
    params = {"column": value_col, "group_column": group_col, "method": "student"}
    results = _run_r(tmp_path, envelope, {"turkish": params})
    actual = results["turkish"]
    assert actual["ok"] is True, actual.get("error")
    actual = actual["result"]

    expected = run_ttest(frame_from_envelope(envelope), params)
    assert set(actual) == set(expected)
    assert actual["group1"] == expected["group1"]
    assert actual["group2"] == expected["group2"]
    assert actual["n1"] == expected["n1"] == 4
    assert actual["n2"] == expected["n2"] == 4
    assert _rel(actual["t"], expected["t"]) <= TOL["t"]
    assert _rel(actual["p"], expected["p"]) <= TOL["p"]

    # The names themselves, as they come back out in the generated prose.
    assert value_col in actual["methods_text"]
    assert group_col in actual["methods_text"]
    assert actual["r_code"] == f"t.test({value_col} ~ {group_col}, data = data, var.equal = TRUE)"
    assert actual["methods_text"] == expected["methods_text"]
    assert set(actual["summary"]) == set(expected["summary"])


def test_the_two_level_cleaner_folds_and_warns_identically(tmp_path):
    """clean_two_level decides which rows exist, so this is n1 and n2, not prose.

    M / m / male / Male / Female / F / f / female fold to two levels; "n/a" and
    "unknown" are dropped WITH a warning naming them, because either could be a
    category the user meant to keep. An R copy that folded one spelling
    differently would not round differently, it would test a different number of
    patients and report a confidently wrong result.
    """
    df = pd.DataFrame(
        {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "g": ["M", "F", "male", "female", "n/a", "M", "Female", "unknown", "m", "f"],
        }
    )
    envelope = build_envelope(df, kinds={"v": "numeric", "g": "categorical"}, columns=["v", "g"])
    params = {"column": "v", "group_column": "g", "method": "student"}

    results = _run_r(tmp_path, envelope, {"folded": params})
    assert results["folded"]["ok"] is True, results["folded"].get("error")
    actual = results["folded"]["result"]
    expected = run_ttest(frame_from_envelope(envelope), params)

    assert actual["group1"] == expected["group1"] == "Female"
    assert actual["group2"] == expected["group2"] == "Male"
    assert actual["n1"] == expected["n1"] == 4
    assert actual["n2"] == expected["n2"] == 4
    # The warning is part of the contract: it is what tells a reader that two
    # rows left the denominator.
    assert actual["warnings"] == expected["warnings"]
    assert actual["warnings"][0]["dropped_levels"] == [
        {"level": "n/a", "n": 1},
        {"level": "unknown", "n": 1},
    ]


# ---------------------------------------------------------------------------
# Select Cases: the same rows on both sides
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The JS-facing surface
# ---------------------------------------------------------------------------

IDENTITY_DRIVER = r"""
args <- commandArgs(trailingOnly = TRUE)
source(args[1])
ustat_init()
writeLines(ustat_identity_json(), args[2], useBytes = TRUE)
"""


def test_the_bundle_reports_itself_in_shapes_javascript_can_read(tmp_path):
    """`analyses` and `packages` must be JSON ARRAYS even at length one.

    toJSON(auto_unbox = TRUE) collapses a length-1 atomic vector to a scalar, so
    with exactly one analysis registered `analyses` would arrive in JS as the
    string "stats.ttest" and `analyses.includes("stats.ttest")` would happen to
    return true -- for the wrong reason, and only until a second analysis
    landed. Caught by hand once; pinned here.
    """
    bundle = _bundle(tmp_path)
    driver = tmp_path / "identity.R"
    driver.write_text(IDENTITY_DRIVER, encoding="utf-8")
    out_path = tmp_path / "identity.json"
    proc = subprocess.run(
        [RSCRIPT, "--vanilla", str(driver), str(bundle), str(out_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

    identity = json.loads(out_path.read_text(encoding="utf-8"))
    assert identity["schema"] == "ustat.frame/1"
    assert isinstance(identity["analyses"], list)
    assert identity["analyses"] == ["stats.ttest"]
    assert isinstance(identity["packages"], list)
    assert identity["packages"] == ["moments", "nortest"]
    assert identity["r_version"].startswith("4.")

    # And the manifest the server serves says the same about the packages, since
    # that is what the browser's boot plan is built from.
    from ustat_engine_r.fingerprint import analyses as declared

    ttest = next(a for a in declared() if a["id"] == "stats.ttest")
    assert ttest["packages"] == identity["packages"]


# ---------------------------------------------------------------------------
# The shared fixtures, replayed through R
# ---------------------------------------------------------------------------

FIXTURES_PATH = REPO_ROOT / "qa" / "parity" / "ttest.json"


def _load_fixtures() -> list[dict]:
    if not FIXTURES_PATH.exists():
        return []
    return json.loads(FIXTURES_PATH.read_text())


FIXTURES = _load_fixtures()


@pytest.fixture(scope="module")
def fixture_results(tmp_path_factory) -> dict:
    """Every fixture in qa/parity/ttest.json, run through R in one invocation.

    This is what makes the fixture file worth having. test_engine_ttest_parity.py
    replays the identical file against the Python engine; if both runners agree
    with the file they agree with each other, which is a property no
    single-runtime test can check. Nothing in `expected` is a field the two
    engines legitimately differ on -- those are described in each fixture's
    `divergence` and are all nested (assumptions[], effect_sizes[]) or prose the
    fixtures deliberately avoid.
    """
    tmp_path = tmp_path_factory.mktemp("r_fixtures")
    jobs = {}
    for fixture in FIXTURES:
        spec = fixture["dataset"]
        jobs[fixture["id"]] = {
            "envelope": _envelope(spec.get("columns"), spec.get("conditions")),
            "params": fixture["params"],
        }

    bundle = _bundle(tmp_path)
    jobs_path = tmp_path / "fixture_jobs.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    driver = tmp_path / "fixture_driver.R"
    driver.write_text(FIXTURE_DRIVER, encoding="utf-8")
    out_path = tmp_path / "fixture_out.json"

    proc = subprocess.run(
        [RSCRIPT, "--vanilla", str(driver), str(bundle), str(jobs_path), str(out_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"Rscript failed\n{proc.stdout}\n{proc.stderr}"
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    return {name: json.loads(payload) for name, payload in raw.items()}


@pytest.mark.skipif(not FIXTURES, reason=f"fixtures not found at {FIXTURES_PATH}")
@pytest.mark.parametrize(
    "fixture", FIXTURES, ids=[f["id"] for f in FIXTURES] if FIXTURES else None
)
def test_r_engine_satisfies_the_shared_fixtures(fixture, fixture_results):
    payload = fixture_results[fixture["id"]]
    expected = fixture["expected"]
    kind = fixture["tolerance"].split("(", 1)[0].strip()

    if "status_code" in expected:
        assert payload["ok"] is False, f"{fixture['id']}: R returned a result"
        assert payload["error"]["status_hint"] == expected["status_code"]
        if "message" in expected:
            assert payload["error"]["message"] == expected["message"]
        return

    assert payload["ok"] is True, f"{fixture['id']}: {payload.get('error')}"
    result = payload["result"]
    for field, want in expected.items():
        assert field in result, f"{fixture['id']}: result has no {field!r}"
        got = result[field]
        if isinstance(want, (str, bool)):
            assert got == want, f"{fixture['id']}.{field}: {got!r} != {want!r}"
        elif isinstance(want, int):
            assert float(got) == float(want), f"{fixture['id']}.{field}: {got!r} != {want!r}"
        elif kind == "exact":
            assert got == want, f"{fixture['id']}.{field}: {got!r} != {want!r}"
        elif kind.startswith("rel<="):
            amount = float(kind[len("rel<="):])
            rel = _rel(got, want)
            assert rel <= amount, (
                f"{fixture['id']}.{field}: rel {rel:.3g} > {amount:.3g} "
                f"({got!r} vs {want!r})"
            )
        else:
            raise AssertionError(f"{fixture['id']}: unrecognised tolerance {kind!r}")


def test_every_fixture_names_its_authoritative_engine():
    """A fixture nobody can say where the number came from is a fixture that
    pins whatever the code happened to do the day it was written."""
    for fixture in FIXTURES:
        assert fixture.get("authoritative_engine"), fixture["id"]
        assert fixture.get("source"), fixture["id"]


def test_a_filtered_frame_reaches_the_same_rows(tmp_path):
    envelope = _envelope(["sbp", "arm"], [{"column": "age", "operator": "gt", "value": 50}])
    params = {"column": "sbp", "group_column": "arm"}
    fingerprint = envelope["filter"]["fingerprint"]

    results = _run_r(
        tmp_path,
        envelope,
        {"filtered": dict(params, __filter_fingerprint=fingerprint)},
    )
    actual = results["filtered"]
    assert actual["ok"] is True, actual.get("error")
    actual = actual["result"]

    expected = run_ttest(frame_from_envelope(envelope), params)
    assert actual["n1"] == expected["n1"] == 125
    assert actual["n2"] == expected["n2"] == 129
    assert _rel(actual["t"], expected["t"]) <= TOL["t"]
    assert _rel(actual["p"], expected["p"]) <= TOL["p"]
    assert actual["df"] == expected["df"]
