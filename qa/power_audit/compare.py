"""Put the Power panel's answers next to R's and report every disagreement.

    backend/.venv/bin/python qa/power_audit/compare.py

Exit code is the number of mismatches, so this can gate a check.
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# Sample sizes are integers after a ceiling, so they either match or they do
# not. Powers and effect sizes come from root-finding on both sides; 1e-6
# relative separates a real disagreement from solver tolerance.
REL = 1e-6
# Solving for the effect size is a root-find on both sides — statsmodels uses
# brentq, pwr uses uniroot — and they stop at different points inside their own
# tolerances. 1e-4 still catches a wrong formula while not reporting the
# solver's last digit as a disagreement.
REL_ROOT = 1e-4


def close(a: float, b: float, rel: float = REL) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) <= rel * scale


def main() -> int:
    ust = json.loads((HERE / "endpoints.json").read_text())
    ref = json.loads((HERE / "reference.json").read_text())["cases"]
    cases = ust["cases"]

    mismatches = 0
    for key in sorted(set(cases) | set(ref)):
        u = cases.get(key)
        r = ref.get(key)
        if u is None:
            print(f"[MISSING from uSTAT] {key}")
            mismatches += 1
            continue
        if r is None:
            print(f"[no R counterpart] {key}")
            continue
        a, b = u.get("result"), r.get("value")
        if a is None or b is None:
            print(f"[NULL] {key}: uSTAT {a!r} vs R {b!r}")
            mismatches += 1
            continue
        tol = REL_ROOT if u["solve_for"] == "effect_size" else REL
        if not close(float(a), float(b), tol):
            rel = abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1e-300)
            mismatches += 1
            print(f"[DIFF] {key} ({u['test']}/{u['solve_for']}): "
                  f"uSTAT {a!r} vs R {b!r}  rel={rel:.3g}"
                  + (f"  [{r['engine']}]" if r.get("engine") else ""))

    for key, err in (ust.get("errors") or {}).items():
        print(f"[ENDPOINT ERROR] {key}: {err['status']}")
        mismatches += 1

    print(f"\n{len(cases)} cases compared, {mismatches} mismatches")
    return mismatches


if __name__ == "__main__":
    raise SystemExit(main())
