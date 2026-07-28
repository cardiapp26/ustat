#!/usr/bin/env python3
"""Table 1 panel audit for the 200x20 dataset.

Verifies, against the Tests panel and R:
  (a) table1 numeric p matches the Tests /ttest p (same method, Levene-driven);
  (b) missing values are NOT counted as a category (the recent fix);
  (c) automatic test selection (Student/Welch/MW; Fisher/Chi/FFH-MC);
  (d) SMD correctness for multi-level categorical (Mahalanobis form).

Read-only: does not modify product code. Saves df into the store via
store.save(session_id, df, track_undo=False) and calls the live endpoints
through TestClient(app).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKEND = ROOT / "backend"
DATASET = HERE / "dataset.csv"
R_REFS = Path("/tmp/r_refs_200.txt")

sys.path.insert(0, str(BACKEND))

from main import app  # noqa: E402
from services import store  # noqa: E402

SESSION_ID = "audit_200x20_table1"
GROUP = "arm"


def load_r_refs() -> dict[str, float]:
    out: dict[str, float] = {}
    for line in R_REFS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        key, val = line.split("\t", 1)
        out[key] = float(val)
    return out


def fmt_p_matches(raw_p: float, formatted: str | None) -> bool:
    """Rule: '<0.001' means raw<0.001; otherwise abs diff <= 0.0005."""
    if formatted is None:
        return False
    s = str(formatted)
    if s.startswith("<"):
        return raw_p < float(s[1:])
    try:
        return math.isclose(raw_p, float(s), rel_tol=0, abs_tol=0.0005)
    except ValueError:
        return False


def parse_formatted(formatted: str | None) -> float | None:
    if formatted is None:
        return None
    s = str(formatted)
    if s.startswith("<"):
        return float(s[1:])  # upper bound implied
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    df = pd.read_csv(DATASET)
    assert df.shape == (200, 20), f"unexpected shape {df.shape}"
    r = load_r_refs()
    store.save(SESSION_ID, df, track_undo=False)
    client = TestClient(app)

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = client.post(path, json=payload)
        assert resp.status_code == 200, f"{path} -> {resp.status_code}: {resp.text}"
        return resp.json()

    # ---- 1. Raw Tests-panel p-values for parity comparison ----
    t_normal = post(
        "/api/stats/ttest",
        {"session_id": SESSION_ID, "column": "biomarker_normal", "group_column": GROUP},
    )
    t_skew = post(
        "/api/stats/ttest",
        {"session_id": SESSION_ID, "column": "biomarker_skew", "group_column": GROUP},
    )
    t_outcome = post(
        "/api/stats/ttest",
        {"session_id": SESSION_ID, "column": "ancova_outcome", "group_column": GROUP},
    )
    f_binary = post(
        "/api/stats/fisher",
        {
            "session_id": SESSION_ID,
            "row_column": "category_binary",
            "col_column": GROUP,
        },
    )

    print("=" * 78)
    print("TESTS PANEL (raw p-values)")
    print("=" * 78)
    for name, t in [
        ("biomarker_normal (independent_t)", t_normal),
        ("biomarker_skew   (independent_t)", t_skew),
        ("ancova_outcome   (independent_t)", t_outcome),
    ]:
        print(
            f"  {name}: p={t['p']:.6e}  variance_assumption="
            f"{t['variance_assumption']}  (selected_by={t.get('variance_assumption_selected_by')})"
        )
    print(f"  category_binary (fisher): p={f_binary['p']:.6e}")

    # ---- 2. Table 1 call (overall) ----
    variables = [
        "biomarker_normal",
        "biomarker_skew",
        "ancova_outcome",
        "category_binary",
        "category_three",
        "age",
        "event_binary",
    ]
    variable_kinds = {
        "biomarker_normal": "numeric",
        "biomarker_skew": "numeric",
        "ancova_outcome": "numeric",
        "category_binary": "categorical",
        "category_three": "categorical",
        "age": "numeric",
        "event_binary": "categorical",
    }

    base_payload = {
        "session_id": SESSION_ID,
        "group_column": GROUP,
        "variables": variables,
        "variable_kinds": variable_kinds,
    }

    table_overall = post(
        "/api/stats/table1", {**base_payload, "normality_mode": "overall"}
    )
    table_within = post(
        "/api/stats/table1",
        {**base_payload, "normality_mode": "within_group"},
    )

    def rows_of(payload: dict) -> dict[str, dict]:
        return {row["variable"]: row for row in payload["rows"]}

    rows_o = rows_of(table_overall)
    rows_w = rows_of(table_within)

    # ---- 3. Numeric parity checks (Table1 vs Tests) ----
    print()
    print("=" * 78)
    print("NUMERIC PARITY: Table1 vs Tests panel  (normality_mode=overall)")
    print("=" * 78)

    numeric_parity = []
    for var, tests_resp in [
        ("biomarker_normal", t_normal),
        ("biomarker_skew", t_skew),
        ("ancova_outcome", t_outcome),
    ]:
        row = rows_o[var]
        tests_va = tests_resp["variance_assumption"]  # 'welch' or 'student'
        table_test = row["test"]
        # Map the Tests-panel variance decision to what the Table label should be
        expected_table_test = (
            "t-test (Welch)" if tests_va == "welch" else "t-test"
        )
        test_label_ok = table_test == expected_table_test
        p_ok = fmt_p_matches(tests_resp["p"], row["p_value"])
        numeric_parity.append(
            {
                "variable": var,
                "table_test": table_test,
                "expected_table_test": expected_table_test,
                "tests_variance_assumption": tests_va,
                "tests_raw_p": tests_resp["p"],
                "table_formatted_p": row["p_value"],
                "test_label_ok": test_label_ok,
                "p_matches": p_ok,
                "normality_test": row.get("normality_test"),
                "normality_p": row.get("normality_p"),
                "normal": row.get("normal"),
                "overall_n": row.get("overall_n"),
            }
        )
        print(
            f"  {var}:"
            f"\n      table test  : {table_test!r}  (expected {expected_table_test!r}) -> {'OK' if test_label_ok else 'MISMATCH'}"
            f"\n      tests p     : {tests_resp['p']:.6e}  (variance={tests_va})"
            f"\n      table p_fmt : {row['p_value']!r}  -> {'OK' if p_ok else 'MISMATCH'}"
            f"\n      normality   : {row.get('normality_test')} p={row.get('normality_p')} -> normal={row.get('normal')}"
            f"\n      overall_n   : {row.get('overall_n')}"
        )

    # ---- 4. category_binary (2x2): Table1 must pick Fisher and match R ----
    print()
    print("=" * 78)
    print("CATEGORICAL 2x2: category_binary  (must select Fisher, p matches R)")
    print("=" * 78)
    cb_row = rows_o["category_binary"]
    cb_test = cb_row["test"]
    cb_table_p = cb_row["p_value"]
    fisher_ok = cb_test == "Fisher"
    p_match_r = fmt_p_matches(r["fisher_p"], cb_table_p)
    p_match_tests = fmt_p_matches(f_binary["p"], cb_table_p)
    print(f"  table test        : {cb_test!r}  -> {'OK (Fisher)' if fisher_ok else 'NOT Fisher'}")
    print(f"  table p_fmt       : {cb_table_p!r}")
    print(f"  Tests /fisher p   : {f_binary['p']:.6e}  -> {'OK' if p_match_tests else 'MISMATCH'}")
    print(f"  R fisher_p        : {r['fisher_p']:.6e}  -> {'OK' if p_match_r else 'MISMATCH'}")
    print(f"  levels            : {[sr['category'] for sr in cb_row['sub_rows']]}")
    print(f"  overall_n         : {cb_row['overall_n']}")
    # Expected counts: both arms x 100, no missing -> 200
    printed_total = 0
    for sr in cb_row["sub_rows"]:
        for cell in sr["group_stats"].values():
            printed_total += int(str(cell).split(" ")[0])
    print(f"  printed cell total: {printed_total}  (expected 200)")

    # ---- 5. category_three (3-level, HAS MISSING): the core fix ----
    print()
    print("=" * 78)
    print("MISSING-VALUE HANDLING: category_three (4 missing of 200)")
    print("=" * 78)
    ct_row = rows_o["category_three"]
    ct_levels = [sr["category"] for sr in ct_row["sub_rows"]]
    missing_is_level = any(
        str(c).strip().lower() in {"nan", "", "missing", "na"} for c in ct_levels
    )
    expected_n = 200 - 4  # 196
    overall_n_ok = ct_row["overall_n"] == expected_n
    printed_total_ct = 0
    for sr in ct_row["sub_rows"]:
        for cell in sr["group_stats"].values():
            printed_total_ct += int(str(cell).split(" ")[0])

    # R chi_square_3level_p was computed on table() which drops NAs.
    ct_table_p = ct_row["p_value"]
    ct_p_match_r = fmt_p_matches(r["chi_square_3level_p"], ct_table_p)
    print(f"  overall_n         : {ct_row['overall_n']}  (expected {expected_n}) -> {'OK' if overall_n_ok else 'FAIL'}")
    print(f"  levels            : {ct_levels}  (3 expected: Alpha/Beta/Gamma)")
    print(f"  'missing'/'nan' is a level? : {missing_is_level} -> {'FAIL' if missing_is_level else 'OK'}")
    print(f"  printed cell total: {printed_total_ct}  (expected {expected_n})")
    print(f"  table test        : {ct_row['test']}")
    print(f"  table p_fmt       : {ct_table_p!r}")
    print(f"  R chi_square_3level_p (MC, dropna): {r['chi_square_3level_p']:.6e} -> {'OK' if ct_p_match_r else 'MISMATCH'}")
    if ct_row.get("warnings") or any(
        "category_three" in str(w) for w in table_overall.get("warnings", [])
    ):
        rel = [w for w in table_overall.get("warnings", []) if "category_three" in str(w)]
        print(f"  warnings          : {rel}")

    # ---- 6. age (no missing): overall_n must be 200 ----
    print()
    print("=" * 78)
    print("AGE (no missing): overall_n must be 200")
    print("=" * 78)
    age_row = rows_o["age"]
    print(f"  age overall_n : {age_row['overall_n']}  (expected 200) -> {'OK' if age_row['overall_n']==200 else 'FAIL'}")
    print(f"  age test      : {age_row['test']}  p={age_row['p_value']}")

    # ---- 7. within_group mode: does the test/normality decision change? ----
    print()
    print("=" * 78)
    print("WITHIN_GROUP mode: per-group normality + resulting test choice")
    print("=" * 78)
    for var in ["biomarker_normal", "biomarker_skew", "ancova_outcome"]:
        row = rows_w[var]
        print(f"  {var}:")
        print(f"      overall test    : {row['test']}  p={row['p_value']}")
        print(f"      overall normal  : {row['normal']} (mode={row['normality_mode']})")
        pg = row.get("per_group_normality") or {}
        for g, info in pg.items():
            print(
                f"      group {g}: {info.get('test')} p={info.get('p')} "
                f"normal={info.get('normal')} n={info.get('n')}"
            )

    # ---- 8. SMD verification for category_three (multi-level, the SMD fix) ----
    print()
    print("=" * 78)
    print("SMD VERIFICATION: category_three (3-level multinomial Mahalanobis)")
    print("=" * 78)
    pairs = df[[ "category_three", GROUP]].dropna()
    levels = sorted(pairs["category_three"].unique())
    g1 = pairs[pairs[GROUP] == "A"]["category_three"]
    g2 = pairs[pairs[GROUP] == "B"]["category_three"]
    p1 = np.array([(g1 == c).mean() for c in levels[:-1]])
    p2 = np.array([(g2 == c).mean() for c in levels[:-1]])
    s_pool = (np.diag(p1 * (1 - p1)) + np.diag(p2 * (1 - p2))) / 2
    diff = p1 - p2
    expected_smd = float(np.sqrt(diff @ np.linalg.inv(s_pool) @ diff))
    table_smd = ct_row["smd"]
    smd_ok = (
        table_smd is not None
        and abs(table_smd - expected_smd) <= 1e-3
    )
    print(f"  levels           : {levels}")
    print(f"  p1 (A, K-1)      : {p1}")
    print(f"  p2 (B, K-1)      : {p2}")
    print(f"  expected SMD     : {expected_smd:.4f}")
    print(f"  table SMD        : {table_smd}")
    print(f"  -> {'OK' if smd_ok else 'MISMATCH'}")
    # Also check the 2x2 SMD form for category_binary
    cb_smd = cb_row["smd"]
    cb_pairs = df[["category_binary", GROUP]].dropna()
    tgt = sorted(cb_pairs["category_binary"].unique())[0]
    p1b = (cb_pairs[cb_pairs[GROUP] == "A"]["category_binary"] == tgt).mean()
    p2b = (cb_pairs[cb_pairs[GROUP] == "B"]["category_binary"] == tgt).mean()
    pooled_b = math.sqrt((p1b * (1 - p1b) + p2b * (1 - p2b)) / 2)
    expected_cb_smd = abs(p1b - p2b) / pooled_b
    cb_smd_ok = cb_smd is not None and abs(cb_smd - expected_cb_smd) <= 1e-3
    print(f"  category_binary SMD (2-level Cohen's h form): expected {expected_cb_smd:.4f}, table {cb_smd} -> {'OK' if cb_smd_ok else 'MISMATCH'}")

    # ---- 9. Numeric SMD spot-check for age ----
    print()
    print("NUMERIC SMD spot-check (age, Cohen's d pooled form):")
    age_smd = age_row["smd"]
    a = df[df[GROUP] == "A"]["age"].dropna().astype(float)
    b = df[df[GROUP] == "B"]["age"].dropna().astype(float)
    ps = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    expected_age_smd = abs(a.mean() - b.mean()) / ps
    age_smd_ok = age_smd is not None and abs(age_smd - expected_age_smd) <= 1e-3
    print(f"  expected {expected_age_smd:.4f}, table {age_smd} -> {'OK' if age_smd_ok else 'MISMATCH'}")

    # ---- 10. Summary table ----
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    overall_pass = True
    print("NUMERIC PARITY (Table1 vs Tests):")
    for p in numeric_parity:
        verdict = "PASS" if (p["test_label_ok"] and p["p_matches"]) else "FAIL"
        if verdict == "FAIL":
            overall_pass = False
        print(
            f"  [{verdict}] {p['variable']}: "
            f"test {p['table_test']!r} vs expected {p['expected_table_test']!r}, "
            f"p_match={p['p_matches']}"
        )
    print("CATEGORICAL 2x2 (category_binary vs R fisher):")
    v = "PASS" if (fisher_ok and p_match_r and p_match_tests) else "FAIL"
    if v == "FAIL":
        overall_pass = False
    print(
        f"  [{v}] Fisher selected={fisher_ok}, p_match_R={p_match_r}, "
        f"p_match_Tests={p_match_tests}"
    )
    print("MISSING-VALUE HANDLING (category_three):")
    mv = "PASS" if (overall_n_ok and not missing_is_level and printed_total_ct == expected_n) else "FAIL"
    if mv == "FAIL":
        overall_pass = False
    print(
        f"  [{mv}] overall_n={ct_row['overall_n']} (exp {expected_n}), "
        f"missing_is_level={missing_is_level}, printed_total={printed_total_ct}"
    )
    print("AGE no-missing n:")
    age_ok = age_row["overall_n"] == 200
    if not age_ok:
        overall_pass = False
    print(f"  [{'PASS' if age_ok else 'FAIL'}] overall_n={age_row['overall_n']}")
    print("SMD:")
    sv = "PASS" if (smd_ok and cb_smd_ok and age_smd_ok) else "FAIL"
    if sv == "FAIL":
        overall_pass = False
    print(
        f"  [{sv}] category_three={smd_ok}, category_binary={cb_smd_ok}, age={age_smd_ok}"
    )
    print()
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
