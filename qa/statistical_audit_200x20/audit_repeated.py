#!/usr/bin/env python3
"""Read-only method audit of repeated-measures / mixed / two-way ANOVA / ANCOVA
endpoints on the 200x20 deterministic dataset. Does NOT modify product code."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKEND = ROOT / "backend"
DATASET = HERE / "dataset.csv"
REFERENCE_R = HERE / "reference.R"
REFS_TXT = Path("/tmp/r_refs_200.txt")
OUT_JSON = Path("/tmp/audit_repeated.json")

SOURCE_SESSION = "audit_200x20_source"
LONG_SESSION = "audit_200x20_long"
CONSTANT_DIFF_SESSION = "audit_200x20_constant_diff_long"

sys.path.insert(0, str(BACKEND))

from main import app  # noqa: E402
from services import store  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Reference loader
# ═══════════════════════════════════════════════════════════════════════════════

def _r_references() -> dict[str, float]:
    if REFS_TXT.exists():
        text = REFS_TXT.read_text()
    else:
        proc = subprocess.run(
            ["Rscript", str(REFERENCE_R), str(DATASET)],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        text = proc.stdout
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        try:
            values[key] = float(value)
        except ValueError:
            pass
    return values


# ═══════════════════════════════════════════════════════════════════════════════
# Compact response (strip verbose fields) for the saved JSON artifact
# ═══════════════════════════════════════════════════════════════════════════════

_KEEP_KEYS = {
    "test", "p", "t", "W", "chi2", "df", "df_num", "df_den",
    "n_nonzero", "F", "effects", "assumptions", "posthoc",
    "posthoc_method", "warnings", "significant", "summary",
    "interpretation", "emms",
}


def _compact(data: Any) -> Any:
    if isinstance(data, dict):
        out = {k: v for k, v in data.items() if k in _KEEP_KEYS}
        # Keep only the scalar/short bits of nested effects
        if "effects" in out and isinstance(out["effects"], list):
            out["effects"] = [
                {k: e[k] for k in ("term", "F", "df_num", "df_den", "p",
                                   "significant") if k in e}
                for e in out["effects"]
            ]
        if "posthoc" in out and isinstance(out["posthoc"], list):
            out["posthoc"] = [
                {k: e[k] for k in ("group1", "group2", "statistic", "p",
                                   "p_adj", "significant") if k in e}
                for e in out["posthoc"]
            ]
        if "assumptions" in out and isinstance(out["assumptions"], list):
            out["assumptions"] = [
                {k: e[k] for k in ("name", "met", "detail") if k in e}
                for e in out["assumptions"]
            ]
        return out
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Numeric check helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _check(body: dict, key: str, ref: float, tol: float) -> dict:
    obs = float(body[key])
    delta = abs(obs - ref)
    return {
        "key": key, "observed": obs, "reference": ref,
        "absolute_difference": delta, "tolerance": tol,
        "pass": bool(delta <= tol),
    }


def _main() -> None:
    df = pd.read_csv(DATASET)
    assert df.shape == (200, 20), f"fixture shape changed: {df.shape}"

    # Long-format reshape EXACTLY as the 10x20 audit: drop rows missing any
    # of the 3 wide score columns, then melt.
    complete_wide = df.dropna(
        subset=["pre_score", "post_score", "followup_score"]
    ).copy()
    long_df = complete_wide.melt(
        id_vars=["id", "arm"],
        value_vars=["pre_score", "post_score", "followup_score"],
        var_name="timepoint",
        value_name="score",
    )

    # Probe session: followup = post + 1 for every subject -> constant paired
    # difference on the post-vs-followup post-hoc contrast (t = Inf).
    constant_wide = complete_wide.copy()
    constant_wide["followup_score"] = constant_wide["post_score"] + 1
    constant_long_df = constant_wide.melt(
        id_vars=["id", "arm"],
        value_vars=["pre_score", "post_score", "followup_score"],
        var_name="timepoint",
        value_name="score",
    )

    # Make the timepoint levels look like proper factor levels (R uses
    # pre/post/followup). The melted values are the column names
    # "pre_score"/"post_score"/"followup_score"; statsmodels handles them
    # fine as categorical levels. RM ANOVA AnovaRM only needs within_col to
    # be a factor — pandas dtype is object, which is fine.
    store.save(SOURCE_SESSION, df, track_undo=False)
    store.save(LONG_SESSION, long_df, track_undo=False)
    store.save(CONSTANT_DIFF_SESSION, constant_long_df, track_undo=False)

    r = _r_references()
    print(f"[refs] loaded {len(r)} reference values; "
          f"complete rows = {len(complete_wide)}, long rows = {len(long_df)}")

    client = TestClient(app)
    checks: list[dict[str, Any]] = []

    def post(name, path, payload, *, expected_status=200,
             references=None, note=None):
        response = client.post(path, json=payload)
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text}
        item: dict[str, Any] = {
            "name": name, "path": path, "request": payload,
            "http_status": response.status_code,
            "expected_http_status": expected_status,
            "response": _compact(body) if response.status_code == 200 else body,
            "status_pass": response.status_code == expected_status,
        }
        if note:
            item["note"] = note
        if references and response.status_code == 200:
            item["numeric_checks"] = [
                _check(body, k, ref, tol) for k, ref, tol in references
            ]
            item["numeric_pass"] = all(c["pass"]
                                       for c in item["numeric_checks"])
        checks.append(item)
        return body if response.status_code == 200 else None

    sid = SOURCE_SESSION

    # ---- paired t -----------------------------------------------------
    post(
        "paired_t",
        "/api/repeated/paired_ttest",
        {"session_id": sid, "col1": "pre_score", "col2": "post_score"},
        references=[
            ("t", r["paired_t"], 1e-5),
            ("p", r["paired_t_p"], 1e-8),
        ],
    )

    # ---- wilcoxon signed rank (method note only) ----------------------
    post(
        "wilcoxon_signed_rank",
        "/api/repeated/wilcoxon_signed_rank",
        {"session_id": sid, "col1": "pre_score", "col2": "post_score"},
        note=("SciPy wilcoxon exact mode (n<500, no ties). Endpoint strips "
              "zero differences BEFORE the call (line 124) then passes raw "
              "x1,x2 to sp.wilcoxon, which re-differences; ties handling and "
              "the exact-vs-normal convention may diverge from R."),
    )

    # ---- friedman -----------------------------------------------------
    post(
        "friedman",
        "/api/repeated/friedman",
        {"session_id": sid,
         "columns": ["pre_score", "post_score", "followup_score"]},
        references=[
            ("chi2", r["friedman_chi2"], 1e-5),
            ("p", r["friedman_p"], 1e-8),
        ],
    )

    # ---- rm_anova (within only) --------------------------------------
    post(
        "rm_anova",
        "/api/repeated/rm_anova",
        {"session_id": LONG_SESSION, "subject_col": "id",
         "within_col": "timepoint", "value_col": "score"},
        references=[
            ("F", r["rm_time_f"], 1e-5),
            ("p", r["rm_time_p"], 1e-8),
        ],
        note="statsmodels AnovaRM — no Greenhouse-Geisser correction applied.",
    )

    # ---- mixed ANOVA (KNOWN BUG: OLS ignoring subject strata) ---------
    mixed = post(
        "mixed_anova",
        "/api/repeated/mixed_anova",
        {"session_id": LONG_SESSION, "subject_col": "id",
         "within_col": "timepoint", "between_col": "arm",
         "value_col": "score"},
        note=("Endpoint runs plain factorial OLS Y ~ C(time)*C(arm) and uses "
              "the single pooled Residual as the F denominator for ALL three "
              "effects — ignoring the subject and subject×time error strata. "
              "R reference uses aov(score ~ arm*time + Error(id/timepoint)). "
              "Expected: large F/p divergence at n=200."),
    )
    # Per-effect comparison table for mixed ANOVA.
    if isinstance(mixed, dict) and isinstance(mixed.get("effects"), list):
        ref_by_term = {
            "timepoint": (r["mixed_time_f"], r["mixed_time_p"]),
            "arm": (r["mixed_arm_f"], r["mixed_arm_p"]),
            # endpoint labels interaction "within × between (interaction)"
            "timepoint × arm (interaction)": (
                r["mixed_interaction_f"], r["mixed_interaction_p"]),
        }
        effect_rows = []
        for eff in mixed["effects"]:
            term = eff["term"]
            ref_pair = ref_by_term.get(term)
            if not ref_pair:
                continue
            ref_f, ref_p = ref_pair
            obs_f = float(eff["F"])
            obs_p = float(eff["p"])
            effect_rows.append({
                "term": term,
                "observed_F": obs_f, "reference_F": ref_f,
                "F_ratio": (obs_f / ref_f) if ref_f else None,
                "observed_p": obs_p, "reference_p": ref_p,
                "p_abs_diff": abs(obs_p - ref_p),
                "observed_sig": bool(obs_p < 0.05),
                "reference_sig": bool(ref_p < 0.05),
                "conclusion_flips": (
                    bool(obs_p < 0.05) != bool(ref_p < 0.05)),
            })
        checks[-1]["effect_checks"] = effect_rows

    # ---- two-way ANOVA (now runs at n=200) ---------------------------
    # F/p live under effects[], so per-effect checks are done below.
    post(
        "two_way_anova",
        "/api/advanced_anova/two_way_anova",
        {"session_id": sid, "outcome": "ancova_outcome",
         "factor1": "arm", "factor2": "factor2"},
        note="Per-effect F/p compared against R Type-I aov table below.",
    )
    tw = checks[-1]["response"]
    if isinstance(tw, dict) and isinstance(tw.get("effects"), list):
        tw_ref = {
            "arm": (r["twoway_arm_f"], r["twoway_arm_p"]),
            "factor2": (r["twoway_factor2_f"], r["twoway_factor2_p"]),
            "arm × factor2 (interaction)": (
                r["twoway_interaction_f"], r["twoway_interaction_p"]),
        }
        tw_rows = []
        for eff in tw["effects"]:
            ref_pair = tw_ref.get(eff["term"])
            if not ref_pair:
                continue
            ref_f, ref_p = ref_pair
            obs_f = float(eff["F"])
            obs_p = float(eff["p"])
            tw_rows.append({
                "term": eff["term"],
                "observed_F": obs_f, "reference_F": ref_f,
                "F_abs_diff": abs(obs_f - ref_f),
                "observed_p": obs_p, "reference_p": ref_p,
                "p_abs_diff": abs(obs_p - ref_p),
                "pass": (abs(obs_f - ref_f) <= 1e-5
                         and abs(obs_p - ref_p) <= 1e-8),
            })
        checks[-1]["effect_checks"] = tw_rows
        checks[-1]["numeric_pass"] = all(row["pass"] for row in tw_rows)

    # ---- ANCOVA ------------------------------------------------------
    post(
        "ancova",
        "/api/advanced_anova/ancova",
        {"session_id": sid, "outcome": "ancova_outcome",
         "group_col": "arm", "covariates": ["age"]},
        references=[
            ("F", r["ancova_f"], 1e-5),
            ("p", r["ancova_p"], 1e-8),
        ],
        note="R ref uses drop1 Type-II SS; endpoint uses statsmodels typ=2.",
    )

    # ---- Robustness probe: constant post-hoc difference --------------
    post(
        "rm_anova_constant_posthoc_difference",
        "/api/repeated/rm_anova",
        {"session_id": CONSTANT_DIFF_SESSION, "subject_col": "id",
         "within_col": "timepoint", "value_col": "score"},
        expected_status=400,
        note=("Probe: balanced valid RM data, but post-vs-followup post-hoc "
              "contrast has constant differences -> t=Inf. Does the global "
              "JSON finiteness guard turn the whole omnibus response into "
              "HTTP 400?"),
    )

    OUT_JSON.write_text(json.dumps(
        {"checks": checks, "references": r,
         "n_complete_rows": int(len(complete_wide)),
         "n_long_rows": int(len(long_df))},
        indent=2, default=str))
    print(f"[done] wrote {OUT_JSON}")

    # Console summary
    print("\n=== SUMMARY ===")
    for c in checks:
        tag = "OK" if c["status_pass"] else "STATUS-MISMATCH"
        npass = c.get("numeric_pass")
        npass_str = "" if npass is None else (
            " numeric_pass=True" if npass else " numeric_pass=FALSE")
        print(f"- {c['name']}: http={c['http_status']} [{tag}]{npass_str}")
        for nc in c.get("numeric_checks", []) or []:
            mark = "OK" if nc["pass"] else "MISMATCH"
            print(f"    [{mark}] {nc['key']}: obs={nc['observed']:.6g} "
                  f"ref={nc['reference']:.6g} d={nc['absolute_difference']:.3g}")
        for ec in c.get("effect_checks", []) or []:
            print(f"    effect {ec['term']}: F_obs={ec['observed_F']:.4g} "
                  f"F_ref={ec['reference_F']:.4g} p_obs={ec['observed_p']:.4g} "
                  f"p_ref={ec['reference_p']:.4g}"
                  + ("  <<< CONCLUSION FLIPS" if ec.get("conclusion_flips")
                     else ""))


if __name__ == "__main__":
    _main()
