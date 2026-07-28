#!/usr/bin/env python3
"""Read-only audit of categorical / contingency and non-inferiority tests on
the 200x20 uSTAT audit dataset. No product code is modified.

Mirrors the in-process TestClient pattern from qa/statistical_audit_10x20/audit.py:
    sys.path.insert backend, from main import app, from services import store,
    store.save(session_id, df, track_undo=False).

Compares each endpoint output against the independent R references in
/tmp/r_refs_200.txt (tab-separated name<TAB>value).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKEND = ROOT / "backend"
DATASET = HERE / "dataset.csv"
R_REFS = Path("/tmp/r_refs_200.txt")
REFERENCE_R = HERE / "reference.R"
SESSION = "audit_200x20_categorical"
OUTPUT = Path("/tmp/audit_categorical.json")

sys.path.insert(0, str(BACKEND))

from main import app  # noqa: E402
from services import store  # noqa: E402


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return proc.stdout.strip()


def _load_r_refs() -> dict[str, float]:
    if not R_REFS.exists():
        proc = subprocess.run(
            ["Rscript", str(REFERENCE_R), str(DATASET)],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        text = proc.stdout
    else:
        text = R_REFS.read_text()
    refs: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        try:
            refs[key] = float(value)
        except ValueError:
            continue
    return refs


def _compact(data: Any) -> Any:
    """Keep only the numeric / decision fields needed for the findings report."""
    if not isinstance(data, dict):
        return data
    keep = {
        "test", "p", "p_noninferiority", "F", "t", "U", "H", "J", "Z", "z",
        "chi2", "Q", "df", "dof", "statistic", "n", "n1", "n2", "exact",
        "estimate", "ci_low", "ci_high", "ci_level", "margin", "bound",
        "non_inferior", "warnings", "assumptions", "r_code",
        "observed_proportion", "null_proportion", "expected_proportion",
        "k", "contingency_table", "summary", "level_order_source",
        "effect", "test_group", "ref_group", "outcome_type", "alpha_one_sided",
        "p_test", "p_ref", "n_test", "events_test", "events_ref",
        "mean_test", "mean_ref", "common_odds_ratio",
    }
    return {k: v for k, v in data.items() if k in keep}


def _check(
    observed: float, reference: float, *, tol: float = 1e-6
) -> dict[str, Any]:
    if observed is None or (isinstance(observed, float) and math.isnan(observed)):
        diff = None
        ok = False
    else:
        diff = abs(float(observed) - float(reference))
        ok = bool(diff <= tol)
    return {
        "observed": observed,
        "reference": reference,
        "absolute_difference": diff,
        "tolerance": tol,
        "pass": ok,
    }


def main() -> None:
    raw = DATASET.read_bytes()
    df = pd.read_csv(DATASET)
    if df.shape != (200, 20):
        raise RuntimeError(f"fixture shape changed: {df.shape}, expected (200, 20)")

    store.save(SESSION, df, track_undo=False)
    r = _load_r_refs()
    client = TestClient(app)
    checks: list[dict[str, Any]] = []

    def post(
        name: str, path: str, payload: dict[str, Any],
        *, expected_status: int = 200, note: str | None = None,
        references: list[tuple[str, float, float]] | None = None,
    ) -> dict[str, Any]:
        resp = client.post(path, json=payload)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        item: dict[str, Any] = {
            "name": name, "path": path, "request": payload,
            "http_status": resp.status_code,
            "expected_http_status": expected_status,
            "status_pass": resp.status_code == expected_status,
            "response": _compact(body) if isinstance(body, dict) else body,
        }
        if note:
            item["note"] = note
        if references and resp.status_code == 200 and isinstance(body, dict):
            item["numeric_checks"] = []
            for key, ref, tol in references:
                val = body.get(key)
                if val is None:
                    item["numeric_checks"].append({
                        "observed_key": key, "observed": None,
                        "reference": ref, "absolute_difference": None,
                        "tolerance": tol, "pass": False,
                        "reason": "key missing in response",
                    })
                else:
                    item["numeric_checks"].append(
                        {**_check(float(val), ref, tol=tol), "observed_key": key}
                    )
            item["numeric_pass"] = all(c["pass"] for c in item["numeric_checks"])
        checks.append(item)
        return body if isinstance(body, dict) else {}

    sid = SESSION

    # ── chi-square 2x2 (Yates vs uncorrected) ──────────────────────────────
    post(
        "chi_square_2x2", "/api/stats/chisquare",
        {"session_id": sid, "row_column": "category_binary", "col_column": "arm"},
        references=[
            ("chi2", r["chi_square"], 1e-6),          # Yates
            ("p", r["chi_square_p"], 1e-6),           # Yates p
        ],
        note=(
            "Endpoint matches the R Yates-corrected chi2 "
            f"(R uncorrected chi2={r['chi_square_uncorrected']:.6f}, "
            f"p={r['chi_square_uncorrected_p']:.6g}). scipy chi2_contingency "
            "default correction=True applies Yates for 2x2."
        ),
    )
    # Record which chi-square (Yates vs uncorrected) the endpoint produced.
    last = checks[-1]["response"]
    chi2_ep = last.get("chi2")
    checks[-1]["yates_vs_uncorrected"] = {
        "endpoint_chi2": chi2_ep,
        "yates_chi2": r["chi_square"],
        "uncorrected_chi2": r["chi_square_uncorrected"],
        "matches_yates": math.isclose(chi2_ep, r["chi_square"], abs_tol=1e-6),
        "matches_uncorrected": math.isclose(chi2_ep, r["chi_square_uncorrected"], abs_tol=1e-6),
    }

    # ── Fisher ─────────────────────────────────────────────────────────────
    post(
        "fisher_2x2", "/api/stats/fisher",
        {"session_id": sid, "row_column": "category_binary", "col_column": "arm"},
        references=[("p", r["fisher_p"], 1e-6)],
    )

    # ── chi-square 3-level (R used MC simulation) ──────────────────────────
    post(
        "chi_square_3level", "/api/stats/chisquare",
        {"session_id": sid, "row_column": "category_three", "col_column": "arm"},
        references=[("p", r["chi_square_3level_p"], 1e-6)],
        note=(
            "R reference p_chi_square_3level_p was computed via Monte-Carlo "
            "simulate.p.value (B=100000, seed). The endpoint uses the asymptotic "
            "chi2_contingency p-value, so an exact match is not expected; "
            "report the observed delta."
        ),
    )

    # ── binomial ───────────────────────────────────────────────────────────
    binom = post(
        "binomial_test", "/api/categorical/binomial",
        {"session_id": sid, "column": "event_binary", "expected_proportion": 0.5},
    )
    # Independent binomial recomputation against R refs is via k/n; record k,n
    # so the reader can confirm against binom.test.
    checks[-1]["expected_k"] = int((df["event_binary"].dropna() == 1).sum())
    checks[-1]["expected_n"] = int(df["event_binary"].dropna().shape[0])
    checks[-1]["numeric_checks"] = [{
        "observed_key": "k",
        "observed": binom.get("k"),
        "reference": checks[-1]["expected_k"],
        "absolute_difference": abs(binom.get("k", -999) - checks[-1]["expected_k"]),
        "tolerance": 0,
        "pass": binom.get("k") == checks[-1]["expected_k"],
    }]
    checks[-1]["numeric_pass"] = all(c["pass"] for c in checks[-1]["numeric_checks"])

    # ── one_proportion z-test ──────────────────────────────────────────────
    onep = post(
        "one_proportion_z", "/api/categorical/one_proportion",
        {"session_id": sid, "column": "event_binary", "null_proportion": 0.5},
        note="statsmodels proportions_ztest two-sided; no exact-binomial fallback.",
    )
    checks[-1]["k_n"] = {"k": onep.get("summary", {}).get("k"),
                         "n": onep.get("summary", {}).get("n")}

    # ── two_proportions z-test ─────────────────────────────────────────────
    twop = post(
        "two_proportions_z", "/api/categorical/two_proportions",
        {"session_id": sid, "column": "event_binary", "group_column": "arm"},
        note="statsmodels proportions_ztest two-sided, pooled SE.",
    )
    checks[-1]["group_counts"] = twop.get("summary")

    # ── McNemar ────────────────────────────────────────────────────────────
    post(
        "mcnemar", "/api/categorical/mcnemar",
        {"session_id": sid, "col1": "paired_pre", "col2": "paired_post"},
        note="statsmodels mcnemar; exact when discordant b+c < 25.",
    )

    # ── Cochran Q ──────────────────────────────────────────────────────────
    post(
        "cochran_q", "/api/categorical/cochran_q",
        {"session_id": sid, "columns": ["paired_pre", "paired_post", "paired_third"]},
    )

    # ── Mantel-Haenszel ────────────────────────────────────────────────────
    post(
        "mantel_haenszel", "/api/categorical/mantel_haenszel",
        {"session_id": sid, "row_col": "arm", "col_col": "category_binary",
         "strata_col": "stratum"},
    )

    # ── Cochran-Armitage ───────────────────────────────────────────────────
    post(
        "cochran_armitage", "/api/categorical/cochran_armitage",
        {"session_id": sid, "ordinal_col": "ordinal_dose", "event_col": "event_binary"},
    )

    # ── Non-inferiority continuous (CRITICAL: known wrong-tail bug) ────────
    ni_cont = post(
        "noninferiority_continuous_upper", "/api/stats/noninferiority",
        {
            "session_id": sid, "outcome_col": "ancova_outcome", "group_col": "arm",
            "test_group": "B", "ref_group": "A", "outcome_type": "continuous",
            "effect": "MD", "margin": 20, "bound": "upper", "alpha": 0.05,
        },
    )
    p_ni_cont = ni_cont.get("p_noninferiority")
    correct_p_t = r["ni_cont_upper_p_t"]   # correct one-sided t p-value
    correct_p_z = r["ni_cont_upper_p_z"]   # one-sided z p-value (for reference)
    complement_t = 1.0 - correct_p_t
    checks[-1]["noninferiority_diagnostics"] = {
        "endpoint_p_noninferiority": p_ni_cont,
        "correct_one_sided_t_p": correct_p_t,
        "correct_one_sided_z_p": correct_p_z,
        "complement_of_t_p_1_minus_p": complement_t,
        "endpoint_equals_correct_t_p": (
            math.isclose(float(p_ni_cont), correct_p_t, rel_tol=0, abs_tol=1e-6)
            if p_ni_cont is not None else False
        ),
        "endpoint_equals_complement_of_t_p": (
            math.isclose(float(p_ni_cont), complement_t, rel_tol=0, abs_tol=1e-6)
            if p_ni_cont is not None else False
        ),
        "endpoint_equals_correct_z_p": (
            math.isclose(float(p_ni_cont), correct_p_z, rel_tol=0, abs_tol=1e-6)
            if p_ni_cont is not None else False
        ),
        "endpoint_estimate": ni_cont.get("estimate"),
        "reference_estimate": r["ni_cont_estimate"],
        "ci_low": ni_cont.get("ci_low"),
        "ci_high": ni_cont.get("ci_high"),
        "decision_non_inferior": ni_cont.get("non_inferior"),
        "welch_df_reference": r["ni_cont_welch_df"],
    }

    # ── Non-inferiority binary (RR, upper) ─────────────────────────────────
    ni_bin = post(
        "noninferiority_binary_rr_upper", "/api/stats/noninferiority",
        {
            "session_id": sid, "outcome_col": "event_binary", "group_col": "arm",
            "test_group": "B", "ref_group": "A", "outcome_type": "binary",
            "effect": "RR", "margin": 3, "bound": "upper", "alpha": 0.05,
        },
    )
    p_ni_bin = ni_bin.get("p_noninferiority")
    correct_p_bin = r["ni_binary_upper_p"]
    complement_bin = 1.0 - correct_p_bin
    checks[-1]["noninferiority_binary_diagnostics"] = {
        "endpoint_p_noninferiority": p_ni_bin,
        "correct_one_sided_p": correct_p_bin,
        "complement_of_p_1_minus_p": complement_bin,
        "endpoint_equals_correct_p": (
            math.isclose(float(p_ni_bin), correct_p_bin, rel_tol=0, abs_tol=1e-6)
            if p_ni_bin is not None else False
        ),
        "endpoint_equals_complement_of_p": (
            math.isclose(float(p_ni_bin), complement_bin, rel_tol=0, abs_tol=1e-6)
            if p_ni_bin is not None else False
        ),
        "endpoint_rr": ni_bin.get("estimate"),
        "reference_rr": r["ni_binary_rr"],
        "decision_non_inferior": ni_bin.get("non_inferior"),
    }

    # ── Provenance ─────────────────────────────────────────────────────────
    r_proc = subprocess.run(["Rscript", "--version"], text=True,
                            capture_output=True, check=True)
    sources = [
        "backend/routers/categorical.py",
        "backend/routers/stats/inferential.py",
        "backend/routers/stats/nonparametric.py",
        "backend/services/store.py",
    ]
    result = {
        "scope": (
            "Read-only categorical/contingency + non-inferiority method audit on "
            "the fixed 200x20 dataset. No product code modified."
        ),
        "provenance": {
            "git_head": _git("rev-parse", "HEAD"),
            "git_status": _git("status", "--short").splitlines(),
            "working_tree_diff_sha256": hashlib.sha256(
                _git("diff", "--binary").encode()
            ).hexdigest(),
            "product_source_sha256": {
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                for p in sources
            },
            "dataset_path": str(DATASET.relative_to(ROOT)),
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "dataset_shape": list(df.shape),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "r_version": (r_proc.stdout or r_proc.stderr).strip(),
            "r_refs_path": str(R_REFS),
        },
        "r_references": r,
        "checks": checks,
        "summary": {
            "total_checks": len(checks),
            "http_status_failures": [
                c["name"] for c in checks if not c["status_pass"]
            ],
            "numeric_failures": [
                c["name"] for c in checks if c.get("numeric_pass") is False
            ],
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
