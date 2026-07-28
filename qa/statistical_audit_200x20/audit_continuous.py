#!/usr/bin/env python3
"""Read-only reproducible audit of uSTAT continuous-variable inferential tests
and descriptive statistics on the deterministic 200x20 dataset.

Compares the backend's TestClient responses against independent R reference
values in /tmp/r_refs_200.txt (regenerated from reference.R if missing).

Run: cd /Users/yh/Documents/projects/wiz3 && .venv/bin/python \
        qa/statistical_audit_200x20/audit_continuous.py
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
REFERENCE_R = HERE / "reference.R"
REFS_TXT = Path("/tmp/r_refs_200.txt")
OUT_JSON = Path("/tmp/audit_continuous.json")

SESSION = "audit_200x20_continuous"

sys.path.insert(0, str(BACKEND))

from main import app  # noqa: E402
from services import store  # noqa: E402


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _r_references() -> dict[str, float]:
    """Parse /tmp/r_refs_200.txt; (re)generate from reference.R if missing."""
    if not REFS_TXT.exists():
        subprocess.run(
            ["Rscript", str(REFERENCE_R), str(DATASET)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            stdout=open(REFS_TXT, "w"),
        )
    values: dict[str, float] = {}
    for line in REFS_TXT.read_text().splitlines():
        if not line.strip():
            continue
        key, value = line.split("\t", 1)
        values[key.strip()] = float(value)
    return values


def _compact_response(data: Any) -> Any:
    """Trim response payloads to the fields we care about for the report."""
    if not isinstance(data, dict):
        return data
    keep = {
        "test",
        "p",
        "F",
        "t",
        "U",
        "H",
        "J",
        "Z",
        "z",
        "df",
        "df_between",
        "df_within",
        "df_method",
        "statistic",
        "n",
        "n1",
        "n2",
        "n_total",
        "mean",
        "mean1",
        "mean2",
        "variance_assumption",
        "variance_assumption_selected_by",
        "assumptions",
        "warnings",
        "posthoc_method",
        "effect_sizes",
        "group1",
        "group2",
        "groups",
        "normality_test",
        "normality_p",
    }
    return {key: value for key, value in data.items() if key in keep}


def _metric_check(
    response: dict[str, Any],
    observed_key: str,
    reference: float,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    if observed_key not in response or response[observed_key] is None:
        return {
            "observed_key": observed_key,
            "observed": None,
            "reference": reference,
            "absolute_difference": None,
            "tolerance": tolerance,
            "pass": False,
            "note": f"key '{observed_key}' missing or null in response",
        }
    observed = float(response[observed_key])
    difference = abs(observed - reference)
    return {
        "observed_key": observed_key,
        "observed": observed,
        "reference": reference,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "pass": bool(difference <= tolerance),
    }


def main() -> None:
    raw = DATASET.read_bytes()
    df = pd.read_csv(DATASET)
    if df.shape != (200, 20):
        raise RuntimeError(f"fixture shape changed: {df.shape}, expected (200, 20)")

    store.save(SESSION, df, track_undo=False)
    r = _r_references()
    client = TestClient(app)
    checks: list[dict[str, Any]] = []

    def post(
        name: str,
        path: str,
        payload: dict[str, Any],
        *,
        expected_status: int = 200,
        references: list[tuple[str, float, float]] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        response = client.post(path, json=payload)
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text}
        item: dict[str, Any] = {
            "name": name,
            "path": path,
            "request": payload,
            "http_status": response.status_code,
            "expected_http_status": expected_status,
            "response": _compact_response(body),
            "status_pass": response.status_code == expected_status,
        }
        if note:
            item["note"] = note
        if references and response.status_code == 200:
            item["numeric_checks"] = [
                _metric_check(body, key, ref, tolerance=tol)
                for key, ref, tol in references
            ]
            item["numeric_pass"] = all(c["pass"] for c in item["numeric_checks"])
        checks.append(item)
        return body

    sid = SESSION

    # 1. One-sample t-test: biomarker_normal vs mu=10
    post(
        "one_sample_t",
        "/api/stats/ttest",
        {"session_id": sid, "column": "biomarker_normal", "mu": 10},
        references=[
            ("t", r["one_sample_t"], 1e-4),
            ("p", r["one_sample_t_p"], 1e-6),
        ],
    )

    # 2. Independent t-test (auto / Levene) on biomarker_normal by arm.
    #    Compare to BOTH Student and Welch references; report which the
    #    endpoint actually selected via variance_assumption.
    ind = post(
        "independent_t_auto",
        "/api/stats/ttest",
        {
            "session_id": sid,
            "column": "biomarker_normal",
            "group_column": "arm",
        },
        references=[
            ("t", r["independent_student_t"], 1e-4),
            ("p", r["independent_student_p"], 1e-6),
        ],
    )
    # Extra dual-comparison record so we can see the Welch numbers too.
    if isinstance(ind, dict):
        checks[-1]["dual_reference_checks"] = {
            "endpoint_variance_assumption": ind.get("variance_assumption"),
            "endpoint_variance_assumption_selected_by": ind.get(
                "variance_assumption_selected_by"
            ),
            "endpoint_t": ind.get("t"),
            "endpoint_p": ind.get("p"),
            "endpoint_df": ind.get("df"),
            "endpoint_df_method": ind.get("df_method"),
            "student_reference_t": r["independent_student_t"],
            "student_reference_p": r["independent_student_p"],
            "welch_reference_t": r["independent_welch_t"],
            "welch_reference_p": r["independent_welch_p"],
            "delta_vs_student_t": abs(float(ind.get("t", math.nan)) - r["independent_student_t"]),
            "delta_vs_student_p": abs(float(ind.get("p", math.nan)) - r["independent_student_p"]),
            "delta_vs_welch_t": abs(float(ind.get("t", math.nan)) - r["independent_welch_t"]),
            "delta_vs_welch_p": abs(float(ind.get("p", math.nan)) - r["independent_welch_p"]),
            "matches_student": abs(float(ind.get("t", math.nan)) - r["independent_student_t"]) <= 1e-4,
            "matches_welch": abs(float(ind.get("t", math.nan)) - r["independent_welch_t"]) <= 1e-4,
        }

    # 3. One-way ANOVA: biomarker_skew by group3 (classical F).
    anova_body = post(
        "one_way_anova",
        "/api/stats/anova",
        {
            "session_id": sid,
            "column": "biomarker_skew",
            "group_column": "group3",
        },
        references=[
            ("F", r["anova_f"], 1e-4),
            ("p", r["anova_p"], 1e-6),
        ],
    )
    # Record whether the endpoint matched the classical or the Welch omnibus F.
    if isinstance(anova_body, dict):
        checks[-1]["anova_omnibus_check"] = {
            "endpoint_F": anova_body.get("F"),
            "endpoint_p": anova_body.get("p"),
            "endpoint_posthoc_method": anova_body.get("posthoc_method"),
            "classical_reference_F": r["anova_f"],
            "classical_reference_p": r["anova_p"],
            "welch_reference_F": r["anova_welch_f"],
            "welch_reference_p": r["anova_welch_p"],
            "delta_vs_classical_F": abs(float(anova_body.get("F", math.nan)) - r["anova_f"]),
            "delta_vs_welch_F": abs(float(anova_body.get("F", math.nan)) - r["anova_welch_f"]),
            "matches_classical_F": abs(float(anova_body.get("F", math.nan)) - r["anova_f"]) <= 1e-4,
            "matches_welch_F": abs(float(anova_body.get("F", math.nan)) - r["anova_welch_f"]) <= 1e-4,
        }

    # 4. Mann-Whitney U: biomarker_normal by arm.
    mw_body = post(
        "mann_whitney",
        "/api/stats/mannwhitney",
        {
            "session_id": sid,
            "column": "biomarker_normal",
            "group_column": "arm",
        },
        references=[("p", r["mann_whitney_p"], 1e-6)],
    )

    # 5. Kruskal-Wallis: biomarker_skew by group3.
    post(
        "kruskal_wallis",
        "/api/stats/kruskal",
        {
            "session_id": sid,
            "column": "biomarker_skew",
            "group_column": "group3",
            "posthoc_correction": "holm",
        },
        references=[("p", r["kruskal_p"], 1e-6)],
    )

    # Descriptive sanity: confirm the column we test on is what we think.
    desc_url = f"/api/stats/{sid}/descriptive?column=biomarker_normal"
    desc_resp = client.get(desc_url)
    try:
        desc_body = desc_resp.json()
    except Exception:
        desc_body = {"raw": desc_resp.text}
    checks.append(
        {
            "name": "descriptive_biomarker_normal",
            "path": desc_url,
            "request": None,
            "http_status": desc_resp.status_code,
            "expected_http_status": 200,
            "response": desc_body.get("biomarker_normal", desc_body) if isinstance(desc_body, dict) else desc_body,
            "status_pass": desc_resp.status_code == 200,
        }
    )

    # Provenance
    try:
        r_version_process = subprocess.run(
            ["Rscript", "--version"],
            text=True,
            capture_output=True,
            check=True,
        )
        r_version = (r_version_process.stdout or r_version_process.stderr).strip()
    except Exception as exc:  # pragma: no cover
        r_version = f"Rscript unavailable: {exc}"

    product_sources = [
        "backend/routers/stats/descriptive.py",
        "backend/routers/stats/inferential.py",
        "backend/routers/stats/nonparametric.py",
        "backend/services/stat_utils.py",
    ]
    result = {
        "scope": (
            "Read-only audit of continuous-variable inferential tests "
            "(one-sample t, independent t with auto Student/Welch, one-way "
            "ANOVA, Mann-Whitney, Kruskal-Wallis) on the fixed 200x20 dataset; "
            "no product code changes."
        ),
        "provenance": {
            "git_head": _git("rev-parse", "HEAD"),
            "git_status": _git("status", "--short").splitlines(),
            "working_tree_diff_sha256": hashlib.sha256(
                _git("diff", "--binary").encode()
            ).hexdigest(),
            "product_source_sha256": {
                path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                for path in product_sources
            },
            "dataset_path": str(DATASET.relative_to(ROOT)),
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "dataset_shape": list(df.shape),
            "missing_by_column": {
                key: int(value)
                for key, value in df.isna().sum().items()
                if int(value) > 0
            },
            "arm_counts": df["arm"].value_counts().to_dict(),
            "group3_counts": df["group3"].value_counts().to_dict(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "r_version": r_version,
            "r_refs_path": str(REFS_TXT),
        },
        "r_references": r,
        "checks": checks,
        "summary": {
            "endpoint_status_checks": len(checks),
            "unexpected_http_statuses": [
                item["name"] for item in checks if not item["status_pass"]
            ],
            "numeric_reference_failures": [
                item["name"]
                for item in checks
                if item.get("numeric_pass") is False
            ],
            "all_numeric_pass": all(
                item.get("numeric_pass", True) for item in checks
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
