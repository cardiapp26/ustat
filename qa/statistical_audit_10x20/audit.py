#!/usr/bin/env python3
"""Read-only reproducible audit of uSTAT Tests and Table tabs on 10×20 data."""

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
SOURCE_SESSION = "audit_10x20_source"
LONG_SESSION = "audit_10x20_complete_long"
CONSTANT_DIFF_SESSION = "audit_10x20_constant_diff_long"

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
    proc = subprocess.run(
        ["Rscript", str(REFERENCE_R), str(DATASET)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    values: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        key, value = line.split("\t", 1)
        values[key] = float(value)
    return values


def _compact_response(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    keep = {
        "test",
        "p",
        "p_overall",
        "p_noninferiority",
        "F",
        "t",
        "U",
        "H",
        "J",
        "Z",
        "z",
        "chi2",
        "Q",
        "df",
        "df_num",
        "df_den",
        "dof",
        "statistic",
        "n",
        "n1",
        "n2",
        "n_nonzero",
        "variance_assumption",
        "variance_assumption_selected_by",
        "warnings",
        "assumptions",
        "pillai",
        "multivariate_tests",
        "effects",
        "non_inferior",
        "estimate",
        "ci_low",
        "ci_high",
        "margin",
        "n_rejected",
        "families",
        "rows",
        "detail",
    }
    return {key: value for key, value in data.items() if key in keep}


def _metric_check(
    response: dict[str, Any],
    observed_key: str,
    reference: float,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
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


def _formatted_p_matches(raw_p: float, formatted_p: str) -> bool:
    if formatted_p.startswith("<"):
        return raw_p < float(formatted_p[1:])
    return math.isclose(raw_p, float(formatted_p), rel_tol=0, abs_tol=0.0005)


def main() -> None:
    raw = DATASET.read_bytes()
    df = pd.read_csv(DATASET)
    if df.shape != (10, 20):
        raise RuntimeError(f"fixture shape changed: {df.shape}, expected (10, 20)")

    complete_wide = df.dropna(
        subset=["pre_score", "post_score", "followup_score"]
    )
    long_df = complete_wide.melt(
        id_vars=["id", "arm"],
        value_vars=["pre_score", "post_score", "followup_score"],
        var_name="timepoint",
        value_name="score",
    )
    constant_wide = complete_wide.copy()
    constant_wide["followup_score"] = constant_wide["post_score"] + 1
    constant_long_df = constant_wide.melt(
        id_vars=["id", "arm"],
        value_vars=["pre_score", "post_score", "followup_score"],
        var_name="timepoint",
        value_name="score",
    )
    store.save(SOURCE_SESSION, df, track_undo=False)
    store.save(LONG_SESSION, long_df, track_undo=False)
    store.save(CONSTANT_DIFF_SESSION, constant_long_df, track_undo=False)

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
            item["numeric_pass"] = all(
                check["pass"] for check in item["numeric_checks"]
            )
        checks.append(item)
        return body

    sid = SOURCE_SESSION
    post(
        "one_sample_t",
        "/api/stats/ttest",
        {"session_id": sid, "column": "biomarker_normal", "mu": 10},
        references=[
            ("t", r["one_sample_t"], 1e-10),
            ("p", r["one_sample_t_p"], 1e-10),
        ],
    )
    independent = post(
        "independent_t_auto",
        "/api/stats/ttest",
        {
            "session_id": sid,
            "column": "biomarker_normal",
            "group_column": "arm",
        },
        references=[
            ("t", r["independent_student_t"], 1e-10),
            ("p", r["independent_student_p"], 1e-10),
        ],
        note="Fixture passes Levene; endpoint and current Table code select Student.",
    )
    post(
        "one_way_anova",
        "/api/stats/anova",
        {
            "session_id": sid,
            "column": "biomarker_skew",
            "group_column": "group3",
        },
        references=[
            ("F", r["anova_f"], 1e-10),
            ("p", r["anova_p"], 1e-10),
        ],
    )
    post(
        "mann_whitney",
        "/api/stats/mannwhitney",
        {
            "session_id": sid,
            "column": "biomarker_normal",
            "group_column": "arm",
        },
        references=[("p", r["mann_whitney_p"], 1e-10)],
    )
    post(
        "kruskal_wallis",
        "/api/stats/kruskal",
        {
            "session_id": sid,
            "column": "biomarker_skew",
            "group_column": "group3",
            "posthoc_correction": "holm",
        },
        references=[("p", r["kruskal_p"], 1e-10)],
    )
    post(
        "jonckheere_terpstra",
        "/api/stats/jonckheere_terpstra",
        {
            "session_id": sid,
            "column": "biomarker_skew",
            "group_column": "ordinal_dose",
        },
        note="Normal approximation only; no exact small-sample p-value is offered.",
    )
    post(
        "chi_square_sparse_2x2",
        "/api/stats/chisquare",
        {
            "session_id": sid,
            "row_column": "category_binary",
            "col_column": "arm",
        },
        references=[
            ("chi2", r["chi_square"], 1e-10),
            ("p", r["chi_square_p"], 1e-10),
        ],
        note="Numerically correct Yates chi-square, but every expected count is <5.",
    )
    post(
        "fisher_exact",
        "/api/stats/fisher",
        {
            "session_id": sid,
            "row_column": "category_binary",
            "col_column": "arm",
        },
        references=[("p", r["fisher_p"], 1e-10)],
    )
    post(
        "ancova",
        "/api/advanced_anova/ancova",
        {
            "session_id": sid,
            "outcome": "ancova_outcome",
            "group_col": "arm",
            "covariates": ["age"],
        },
        references=[
            ("F", r["ancova_f"], 5e-5),
            ("p", r["ancova_p"], 1e-10),
        ],
        note="Endpoint correctly flags violated group-by-age slope homogeneity.",
    )
    mancova = post(
        "mancova",
        "/api/advanced_anova/mancova",
        {
            "session_id": sid,
            "outcomes": ["ancova_outcome", "outcome2"],
            "group_col": "arm",
            "covariates": ["age"],
        },
    )
    if isinstance(mancova, dict) and "pillai" in mancova:
        pillai = mancova["pillai"]
        checks[-1]["numeric_checks"] = [
            {
                "observed_key": "pillai.value",
                "observed": float(pillai["value"]),
                "reference": r["mancova_pillai"],
                "absolute_difference": abs(
                    float(pillai["value"]) - r["mancova_pillai"]
                ),
                "tolerance": 5e-5,
                "pass": abs(
                    float(pillai["value"]) - r["mancova_pillai"]
                )
                <= 5e-5,
            },
            {
                "observed_key": "pillai.p",
                "observed": float(pillai["p"]),
                "reference": r["mancova_p"],
                "absolute_difference": abs(float(pillai["p"]) - r["mancova_p"]),
                "tolerance": 1e-10,
                "pass": abs(float(pillai["p"]) - r["mancova_p"]) <= 1e-10,
            },
        ]
        checks[-1]["numeric_pass"] = all(
            row["pass"] for row in checks[-1]["numeric_checks"]
        )
    post(
        "two_way_anova_expected_rejection",
        "/api/advanced_anova/two_way_anova",
        {
            "session_id": sid,
            "outcome": "ancova_outcome",
            "factor1": "arm",
            "factor2": "factor2",
        },
        expected_status=400,
        note="Correct guard: source has 10 rows; endpoint requires >=12 complete rows.",
    )

    post(
        "paired_t",
        "/api/repeated/paired_ttest",
        {"session_id": sid, "col1": "pre_score", "col2": "post_score"},
        references=[
            ("t", r["paired_t"], 5e-5),
            ("p", r["paired_t_p"], 1e-10),
        ],
    )
    post(
        "wilcoxon_signed_rank",
        "/api/repeated/wilcoxon_signed_rank",
        {"session_id": sid, "col1": "pre_score", "col2": "post_score"},
        note="SciPy exact calculation used; zeros/ties make cross-software conventions differ.",
    )
    post(
        "friedman",
        "/api/repeated/friedman",
        {
            "session_id": sid,
            "columns": ["pre_score", "post_score", "followup_score"],
        },
        references=[
            ("chi2", r["friedman_chi2"], 5e-5),
            ("p", r["friedman_p"], 1e-10),
        ],
    )
    post(
        "rm_anova_complete_case_reshape",
        "/api/repeated/rm_anova",
        {
            "session_id": LONG_SESSION,
            "subject_col": "id",
            "within_col": "timepoint",
            "value_col": "score",
        },
        references=[
            ("F", r["rm_time_f"], 5e-5),
            ("p", r["rm_time_p"], 1e-10),
        ],
        note="Derived from 9 complete source rows; source artifact remains 10x20.",
    )
    post(
        "rm_anova_constant_posthoc_difference",
        "/api/repeated/rm_anova",
        {
            "session_id": CONSTANT_DIFF_SESSION,
            "subject_col": "id",
            "within_col": "timepoint",
            "value_col": "score",
        },
        expected_status=400,
        note=(
            "Robustness probe: valid balanced RM data, but one post-hoc contrast "
            "has constant differences. Infinite t leaks into response and global "
            "JSON guard rejects the entire otherwise-valid omnibus result."
        ),
    )
    mixed = post(
        "mixed_anova_complete_case_reshape",
        "/api/repeated/mixed_anova",
        {
            "session_id": LONG_SESSION,
            "subject_col": "id",
            "within_col": "timepoint",
            "between_col": "arm",
            "value_col": "score",
        },
        note=(
            "Endpoint OLS treats repeated rows as independent. R reference uses "
            "subject and subject-by-time error strata."
        ),
    )
    if isinstance(mixed, dict) and isinstance(mixed.get("effects"), list):
        reference_by_term = {
            "timepoint": (r["mixed_time_f"], r["mixed_time_p"]),
            "arm": (r["mixed_arm_f"], r["mixed_arm_p"]),
            "timepoint × arm (interaction)": (
                r["mixed_interaction_f"],
                r["mixed_interaction_p"],
            ),
        }
        effect_checks = []
        for effect in mixed["effects"]:
            term = effect["term"]
            if term not in reference_by_term:
                continue
            ref_f, ref_p = reference_by_term[term]
            effect_checks.append(
                {
                    "term": term,
                    "observed_F": float(effect["F"]),
                    "reference_F": ref_f,
                    "observed_p": float(effect["p"]),
                    "reference_p": ref_p,
                    "p_absolute_difference": abs(float(effect["p"]) - ref_p),
                    "pass": math.isclose(
                        float(effect["p"]), ref_p, rel_tol=0, abs_tol=1e-5
                    ),
                }
            )
        checks[-1]["numeric_checks"] = effect_checks
        checks[-1]["numeric_pass"] = all(x["pass"] for x in effect_checks)

    post(
        "binomial_exact",
        "/api/categorical/binomial",
        {
            "session_id": sid,
            "column": "event_binary",
            "expected_proportion": 0.5,
        },
    )
    post(
        "one_proportion_z",
        "/api/categorical/one_proportion",
        {
            "session_id": sid,
            "column": "event_binary",
            "null_proportion": 0.5,
        },
        note="Normal approximation is exposed without a small-count warning; exact binomial is safer for n=10.",
    )
    post(
        "two_proportion_z",
        "/api/categorical/two_proportions",
        {
            "session_id": sid,
            "column": "event_binary",
            "group_column": "arm",
        },
        note="Normal approximation is exposed without an expected-count warning; each arm has n=5.",
    )
    post(
        "mcnemar_exact",
        "/api/categorical/mcnemar",
        {"session_id": sid, "col1": "paired_pre", "col2": "paired_post"},
    )
    post(
        "cochran_q",
        "/api/categorical/cochran_q",
        {
            "session_id": sid,
            "columns": ["paired_pre", "paired_post", "paired_third"],
        },
    )
    post(
        "mantel_haenszel",
        "/api/categorical/mantel_haenszel",
        {
            "session_id": sid,
            "row_col": "arm",
            "col_col": "category_binary",
            "strata_col": "stratum",
        },
    )
    post(
        "cochran_armitage",
        "/api/categorical/cochran_armitage",
        {
            "session_id": sid,
            "ordinal_col": "ordinal_dose",
            "event_col": "event_binary",
        },
    )

    skew_t = post(
        "table_parity_biomarker_skew_ttest",
        "/api/stats/ttest",
        {
            "session_id": sid,
            "column": "biomarker_skew",
            "group_column": "arm",
        },
    )
    outcome_t = post(
        "table_parity_ancova_outcome_ttest",
        "/api/stats/ttest",
        {
            "session_id": sid,
            "column": "ancova_outcome",
            "group_column": "arm",
        },
    )
    category_three_chi = post(
        "table_method_difference_category_three_chisquare",
        "/api/stats/chisquare",
        {
            "session_id": sid,
            "row_column": "category_three",
            "col_column": "arm",
        },
    )

    table_payload = {
        "session_id": sid,
        "group_column": "arm",
        "variables": [
            "biomarker_normal",
            "biomarker_skew",
            "ancova_outcome",
            "category_binary",
            "category_three",
        ],
        "variable_kinds": {
            "biomarker_normal": "numeric",
            "biomarker_skew": "numeric",
            "ancova_outcome": "numeric",
            "category_binary": "categorical",
            "category_three": "categorical",
        },
        "normality_mode": "overall",
    }
    table = post(
        "table_overall_default",
        "/api/stats/table1",
        table_payload,
        note=(
            "UI default uses pooled overall Shapiro normality. Sparse 2x2 "
            "categorical row should select Fisher, not chi-square."
        ),
    )
    table_within_payload = {**table_payload, "normality_mode": "within_group"}
    post(
        "table_within_group_toggle",
        "/api/stats/table1",
        table_within_payload,
    )
    if isinstance(table, dict) and isinstance(table.get("rows"), list):
        rows = {row["variable"]: row for row in table["rows"]}
        checks[-2]["parity_checks"] = {
            "numeric_rows": {
                "biomarker_normal": {
                    "table_test": rows["biomarker_normal"]["test"],
                    "tests_variance_assumption": independent.get(
                        "variance_assumption"
                    ),
                    "tests_p": independent.get("p"),
                    "table_formatted_p": rows["biomarker_normal"]["p_value"],
                    "p_matches": _formatted_p_matches(
                        float(independent["p"]),
                        rows["biomarker_normal"]["p_value"],
                    ),
                },
                "biomarker_skew": {
                    "table_test": rows["biomarker_skew"]["test"],
                    "tests_variance_assumption": skew_t.get(
                        "variance_assumption"
                    ),
                    "tests_p": skew_t.get("p"),
                    "table_formatted_p": rows["biomarker_skew"]["p_value"],
                    "p_matches": _formatted_p_matches(
                        float(skew_t["p"]),
                        rows["biomarker_skew"]["p_value"],
                    ),
                },
                "ancova_outcome": {
                    "table_test": rows["ancova_outcome"]["test"],
                    "tests_variance_assumption": outcome_t.get(
                        "variance_assumption"
                    ),
                    "tests_p": outcome_t.get("p"),
                    "table_formatted_p": rows["ancova_outcome"]["p_value"],
                    "p_matches": _formatted_p_matches(
                        float(outcome_t["p"]),
                        rows["ancova_outcome"]["p_value"],
                    ),
                },
            },
            "binary_test_label": rows["category_binary"]["test"],
            "binary_table_p": rows["category_binary"]["p_value"],
            "fisher_reference_p": r["fisher_p"],
            "binary_p_matches": _formatted_p_matches(
                r["fisher_p"], rows["category_binary"]["p_value"]
            ),
            "category_three_table_test": rows["category_three"]["test"],
            "category_three_table_p": rows["category_three"]["p_value"],
            "category_three_tests_chi_square_p": category_three_chi.get("p"),
            "missing_category_overall_n": rows["category_three"]["overall_n"],
            "expected_missing_category_n": 9,
            "pass": (
                rows["biomarker_normal"]["test"] == "t-test"
                and rows["biomarker_skew"]["test"] == "t-test"
                and rows["ancova_outcome"]["test"] == "t-test"
                and rows["category_binary"]["test"] == "Fisher"
                and _formatted_p_matches(
                    float(independent["p"]),
                    rows["biomarker_normal"]["p_value"],
                )
                and _formatted_p_matches(
                    float(skew_t["p"]),
                    rows["biomarker_skew"]["p_value"],
                )
                and _formatted_p_matches(
                    float(outcome_t["p"]),
                    rows["ancova_outcome"]["p_value"],
                )
                and _formatted_p_matches(
                    r["fisher_p"],
                    rows["category_binary"]["p_value"],
                )
                and rows["category_three"]["overall_n"] == 9
            ),
        }

    ni_cont = post(
        "noninferiority_continuous_upper",
        "/api/stats/noninferiority",
        {
            "session_id": sid,
            "outcome_col": "ancova_outcome",
            "group_col": "arm",
            "test_group": "B",
            "ref_group": "A",
            "outcome_type": "continuous",
            "effect": "MD",
            "margin": 20,
            "bound": "upper",
            "alpha": 0.05,
        },
    )
    checks[-1]["correct_reference"] = {
        "estimate": r["ni_cont_estimate"],
        "welch_df": r["ni_cont_welch_df"],
        "p_upper": r["ni_cont_upper_p"],
        "endpoint_p": ni_cont.get("p_noninferiority"),
        "endpoint_is_complement": math.isclose(
            float(ni_cont.get("p_noninferiority", math.nan)),
            1 - r["ni_cont_upper_p"],
            rel_tol=0,
            abs_tol=1e-6,
        ),
        "decision": ni_cont.get("non_inferior"),
    }
    ni_binary = post(
        "noninferiority_binary_rr_upper",
        "/api/stats/noninferiority",
        {
            "session_id": sid,
            "outcome_col": "event_binary",
            "group_col": "arm",
            "test_group": "B",
            "ref_group": "A",
            "outcome_type": "binary",
            "effect": "RR",
            "margin": 3,
            "bound": "upper",
            "alpha": 0.05,
        },
    )
    checks[-1]["correct_reference"] = {
        "rr": r["ni_binary_rr"],
        "p_upper": r["ni_binary_upper_p"],
        "endpoint_p": ni_binary.get("p_noninferiority"),
        "endpoint_is_complement": math.isclose(
            float(ni_binary.get("p_noninferiority", math.nan)),
            1 - r["ni_binary_upper_p"],
            rel_tol=0,
            abs_tol=1e-6,
        ),
        "decision": ni_binary.get("non_inferior"),
    }
    post(
        "gatekeeping_serial_holm",
        "/api/multiplicity/gatekeeping",
        {
            "families": [
                {
                    "name": "Primary",
                    "hypotheses": [
                        {"label": "H1", "p": 0.01},
                        {"label": "H2", "p": 0.04},
                    ],
                },
                {
                    "name": "Secondary",
                    "hypotheses": [{"label": "H3", "p": 0.02}],
                },
            ],
            "method": "holm",
            "logic": "serial",
            "alpha": 0.05,
        },
        note="Adjusted p-values use a 0.0005 grid, so they are approximate.",
    )

    r_version_process = subprocess.run(
        ["Rscript", "--version"],
        text=True,
        capture_output=True,
        check=True,
    )
    product_sources = [
        "backend/main.py",
        "backend/routers/repeated.py",
        "backend/routers/categorical.py",
        "backend/routers/stats/descriptive.py",
        "backend/routers/stats/inferential.py",
        "backend/routers/stats/nonparametric.py",
        "backend/services/store.py",
        "frontend/src/api.ts",
        "frontend/src/components/CategoricalTestsPanel.tsx",
        "frontend/src/components/HypothesisPanel.tsx",
        "frontend/src/components/Table1Panel.tsx",
    ]
    result = {
        "scope": (
            "Read-only audit of Tests subpanels and Table 1 on a fixed synthetic "
            "10-row x 20-column dataset; no product code fixes."
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
            "derived_long_shape": list(long_df.shape),
            "derived_long_rule": (
                "Drop rows missing any repeated score, then melt pre/post/followup."
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "r_version": (
                r_version_process.stdout or r_version_process.stderr
            ).strip(),
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
            "known_method_failures": [
                "mixed_anova_complete_case_reshape",
                "noninferiority_continuous_upper",
                "noninferiority_binary_rr_upper",
            ],
            "known_robustness_failures": [
                "rm_anova_constant_posthoc_difference",
            ],
            "known_ui_contract_failures": [
                "Hypothesis two-way ANOVA does not expose factor1 selector reliably.",
                "Test-type changes can leave hidden stale variable selections.",
            ],
        },
        "limitations": [
            "Ten rows cannot validate large-sample approximations or power.",
            "Two-way ANOVA correctly rejects because endpoint requires 12 rows.",
            "Monte Carlo Fisher-Freeman-Halton uses fixed seed 42 and 5000 resamples.",
            "Same-model orchestration reviewed results; R supplies independent numeric implementation for key tests.",
            "Working tree changed during audit; git_status captures uncommitted state.",
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
