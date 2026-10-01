"""Gray's test in the R engine, run under the local Rscript.

The bundle's survival.gray is evaluated on the same envelope the browser gets,
and compared with cmprsk::cuminc called directly on the raw table in the same
R process -- so what is tested is the plumbing (frame decoding, event codes,
group order, row selection), not cmprsk. Skipped when Rscript or cmprsk is
not installed.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from ustat_engine.frame.envelope import build_envelope

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_r_bundle.py"
RSCRIPT = shutil.which("Rscript") or "/usr/local/bin/Rscript"


def _has_cmprsk() -> bool:
    if not pathlib.Path(RSCRIPT).is_file():
        return False
    proc = subprocess.run(
        [RSCRIPT, "--vanilla", "-e", "cat(requireNamespace('cmprsk', quietly = TRUE))"],
        capture_output=True, text=True, timeout=60,
    )
    return proc.stdout.strip() == "TRUE"


pytestmark = pytest.mark.skipif(not _has_cmprsk(), reason="Rscript with cmprsk not available")

DRIVER = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(cmprsk))
source(args[1])
frame <- ustat_frame_from_envelope(jsonlite::fromJSON(args[2], simplifyVector = FALSE))
raw <- read.csv(args[3], stringsAsFactors = FALSE)
jobs <- jsonlite::fromJSON(args[4], simplifyVector = FALSE)
out <- lapply(jobs, function(j) ustat_run_json("survival.gray", as.character(ustat_to_json(j)), frame))
direct <- cuminc(raw$fu, raw$status, raw$arm, cencode = 0)$Tests
out$direct <- as.character(ustat_to_json(list(
  stat = unname(direct[, "stat"]), p = unname(direct[, "pv"]), cause = rownames(direct))))
writeLines(as.character(ustat_to_json(out)), args[5], useBytes = TRUE)
"""

CASES = {
    "cause1": {"duration_col": "fu", "event_col": "status", "group_col": "arm", "event_of_interest": 1},
    "cause2": {"duration_col": "fu", "event_col": "status", "group_col": "arm", "event_of_interest": 2},
    "absent_cause": {"duration_col": "fu", "event_col": "status", "group_col": "arm", "event_of_interest": 7},
    "one_group": {"duration_col": "fu", "event_col": "status", "group_col": "site", "event_of_interest": 1},
    "no_event_field": {"duration_col": "fu", "event_col": "status", "group_col": "arm"},
}


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 240
    arm = rng.choice(["control", "drug A", "drug B"], n)
    h1 = np.select([arm == "control", arm == "drug A"], [0.10, 0.06], 0.08)
    h2 = np.select([arm == "control", arm == "drug A"], [0.05, 0.09], 0.05)
    t1, t2 = rng.exponential(1 / h1), rng.exponential(1 / h2)
    cens = rng.uniform(2, 15, n)
    t = np.minimum(np.minimum(t1, t2), cens)
    status = np.where(t == cens, 0, np.where(t == t1, 1, 2))
    # Whole months: heavy ties, the case a sloppy event-time grid gets wrong.
    return pd.DataFrame({"fu": np.ceil(t), "status": status, "arm": arm, "site": "A"})


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("r_gray")
    spec = importlib.util.spec_from_file_location("_build_r_bundle_gray", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    manifest = module.build(tmp / "bundle")
    bundle = tmp / "bundle" / manifest["bundle"]

    df = _data()
    raw = tmp / "raw.csv"
    df.to_csv(raw, index=False)
    envelope = build_envelope(
        df, kinds={"fu": "numeric", "status": "numeric", "arm": "categorical", "site": "categorical"},
        columns=list(df.columns), conditions=[],
    )
    (tmp / "env.json").write_text(json.dumps(envelope), encoding="utf-8")
    (tmp / "jobs.json").write_text(json.dumps(CASES), encoding="utf-8")
    driver = tmp / "driver.R"
    driver.write_text(DRIVER, encoding="utf-8")
    out = tmp / "out.json"
    proc = subprocess.run(
        [RSCRIPT, "--vanilla", str(driver), str(bundle), str(tmp / "env.json"), str(raw),
         str(tmp / "jobs.json"), str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    return {k: json.loads(v) for k, v in payload.items()}


@pytest.mark.parametrize("case,row", [("cause1", 0), ("cause2", 1)])
def test_matches_cuminc_called_directly(results, case, row):
    got = results[case]
    assert got["ok"] is True, got
    res = got["result"]
    direct = results["direct"]
    assert res["statistic"] == pytest.approx(direct["stat"][row], rel=1e-12)
    assert res["p"] == pytest.approx(direct["p"][row], rel=1e-12)
    assert res["df"] == 2
    assert res["test"] == "Gray's test"


def test_reports_group_counts_in_sorted_order(results):
    res = results["cause1"]["result"]
    assert [g["group"] for g in res["groups"]] == ["control", "drug A", "drug B"]
    assert sum(g["n"] for g in res["groups"]) == res["n"] == 240
    for g in res["groups"]:
        assert g["event_of_interest"] + g["competing_events"] + g["censored"] == g["n"]
    assert {c["event"] for c in res["by_cause"]} == {1, 2}


@pytest.mark.parametrize("case,fragment", [
    ("absent_cause", "does not occur"),
    ("one_group", "at least two groups"),
    ("no_event_field", "event_of_interest"),
])
def test_bad_requests_are_422_with_a_reason(results, case, fragment):
    got = results[case]
    assert got["ok"] is False
    assert got["error"]["status_hint"] == 422
    assert fragment in got["error"]["message"]
