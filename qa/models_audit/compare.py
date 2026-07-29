"""Put uSTAT's Models output next to R's and report every disagreement.

    backend/.venv/bin/python qa/models_audit/compare.py

Exit code is the number of mismatches, so this can gate a check.
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# uSTAT names dummy columns `arm_treat`; R names them `armtreat`. Same
# reference level, different spelling.
ALIASES = {
    "const": "(Intercept)",
    "Intercept": "(Intercept)",
    "arm_treat": "armtreat",
    "sex_M": "sexM",
    "arm[T.treat]": "armtreat",
    "visit[T.v2]": "visitv2",
    "visit[T.v3]": "visitv3",
    "age^1": "age",
    "age^2": "I(age^2)",
}

REL = 1e-4   # iterative fits converge to slightly different points; this
             # separates a real disagreement from optimiser noise
ABS_P = 1e-8  # absolute tolerance for p-values


def canon(term: str) -> str:
    return ALIASES.get(term, term)


def close(a, b, rel=REL, abs_=0.0) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) <= max(abs_, rel * scale)


def main() -> int:
    ust = json.loads((HERE / "endpoints.json").read_text())
    ref = json.loads((HERE / "reference.json").read_text())
    u_models, r_models = ust["models"], ref["models"]

    # uSTAT model name -> R model name
    pairs = [
        ("linear", "linear"),
        ("logistic", "logistic"),
        ("poisson", "poisson"),
        ("gamma", "gamma"),
        ("negbinom", "negbinom"),
        ("ordinal", "ordinal_polr"),
        ("cox", "cox"),
        ("firth_logistic", "firth_logistic"),
        ("lmm", "lmm"),
        ("polynomial", "polynomial"),
    ]

    problems: list[str] = []
    notes: list[str] = []

    print(f"R: {ref['meta']['r_version']}")
    print(f"packages: {ref['meta']['packages']}\n")

    for uname, rname in pairs:
        u = u_models.get(uname)
        r = r_models.get(rname)
        if u is None:
            err = ust["errors"].get(uname)
            problems.append(f"[{uname}] endpoint returned no result: {err}")
            continue
        if r is None:
            notes.append(f"[{uname}] no R counterpart")
            continue

        r_terms = {canon(t["term"]): t for t in r["terms"]}
        u_terms = {canon(t["term"]): t for t in u["terms"]}

        only_u = sorted(set(u_terms) - set(r_terms))
        only_r = sorted(set(r_terms) - set(u_terms))
        if only_u or only_r:
            problems.append(
                f"[{uname}] term mismatch — only uSTAT: {only_u}; only R: {only_r}")

        agree = 0
        for term in sorted(set(u_terms) & set(r_terms)):
            ut, rt = u_terms[term], r_terms[term]
            for field, tol_abs in (("estimate", 0.0), ("se", 0.0), ("p", ABS_P)):
                a, b = ut.get(field), rt.get(field)
                if a is None or b is None:
                    continue
                if not close(a, b, abs_=tol_abs):
                    problems.append(
                        f"[{uname}] {term}.{field}: uSTAT {a!r} vs R {b!r} "
                        f"(rel diff {abs(a - b) / max(abs(b), 1e-300):.3g})")
                else:
                    agree += 1
        print(f"{uname:<16} {len(set(u_terms) & set(r_terms))} shared terms, "
              f"{agree} values within tolerance")

    # ── model-level numbers ───────────────────────────────────────────────
    scalar_checks = [
        ("linear", "linear", "r_squared", "r_squared"),
        ("linear", "linear", "adj_r_squared", "adj_r_squared"),
        ("linear", "linear", "f_statistic", "f_statistic"),
        ("linear", "linear", "sigma", "sigma"),
        ("linear", "linear", "aic", "aic"),
        ("linear", "linear", "bic", "bic"),
        ("logistic", "logistic", "aic", "aic"),
        ("logistic", "logistic", "bic", "bic"),
        ("logistic", "logistic", "log_likelihood", "log_likelihood"),
        ("poisson", "poisson", "aic", "aic"),
        ("gamma", "gamma", "aic", "aic"),
        ("negbinom", "negbinom", "aic", "aic"),
        ("cox", "cox", "concordance", "concordance"),
        ("cox", "cox", "log_likelihood", "log_likelihood"),
    ]
    print()
    for uname, rname, ukey, rkey in scalar_checks:
        u, r = u_models.get(uname), r_models.get(rname)
        if not u or not r:
            continue
        a, b = u.get(ukey), r.get(rkey)
        if a is None:
            notes.append(f"[{uname}] does not report {ukey} (R: {b!r})")
            continue
        if not close(a, b):
            problems.append(f"[{uname}] {ukey}: uSTAT {a!r} vs R {b!r}")

    # ── things R reports that uSTAT does not ──────────────────────────────
    if "negbinom" in u_models and u_models["negbinom"].get("theta") is None \
            and u_models["negbinom"].get("alpha") is None:
        notes.append(
            f"[negbinom] no dispersion parameter in the response; "
            f"R theta = {r_models['negbinom']['theta']:.6g}")
    if "gamma" in u_models and u_models["gamma"].get("dispersion") is None:
        notes.append(
            f"[gamma] no dispersion in the response; "
            f"R dispersion = {r_models['gamma']['dispersion']:.6g}")

    print("\n" + "=" * 72)
    if problems:
        print(f"MISMATCHES ({len(problems)})")
        for p in problems:
            print("  ✗", p)
    else:
        print("No mismatches.")
    if notes:
        print(f"\nNOTES ({len(notes)})")
        for n in notes:
            print("  ·", n)
    return len(problems)


if __name__ == "__main__":
    raise SystemExit(main())
