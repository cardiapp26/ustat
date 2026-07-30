"""Put uSTAT's output next to R's and report every disagreement.

    backend/.venv/bin/python qa/models_audit_30x50/compare.py

Reads `endpoints.json` (written by audit.py) and both R reference files, and
exits with the number of mismatches so this can gate a check.
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
    "arm[T.treat]": "armtreat",
    "sex_M": "sexM",
    "sex[T.M]": "sexM",
    "rare_grp_b": "rare_grpb",
    "rare_grp_c": "rare_grpc",
    "visit_v2": "visitv2",
    "visit_v3": "visitv3",
    "visit[T.v2]": "visitv2",
    "visit[T.v3]": "visitv3",
    "age^1": "age",
    "age^2": "I(age^2)",
    "stage_II": "stageII",
    "stage_III": "stageIII",
    "smoke_former": "smokeformer",
    "smoke_never": "smokenever",
    "prior_tx": "prior_tx",
}

REL = 1e-4    # iterative fits converge to slightly different points
ABS_P = 1e-8  # absolute floor for p-values


def canon(term: str) -> str:
    return ALIASES.get(term, term)


def close(a, b, rel=REL, abs_=0.0) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) <= max(abs_, rel * scale)


def _compare_km(u_km: dict, r_km: dict) -> tuple[int, int]:
    """Log-rank, per-group medians, and survival at the requested times.

    The confidence limits are log-log (cloglog) on both sides. R's survfit
    DEFAULTS to conf.type="log", so the reference script has to ask for
    log-log explicitly or the two disagree for a reason that is a convention
    rather than an error.
    """
    checked = bad = 0
    lr = u_km.get("logrank") or {}
    for field, r_key in (("chi2", "survdiff_chisq"), ("df", "survdiff_df"),
                         ("p", "survdiff_p")):
        a, b = lr.get(field), r_km.get(r_key)
        if a is None or b is None:
            continue
        checked += 1
        if not close(a, b, abs_=ABS_P if field == "p" else 0.0):
            bad += 1
            print(f"[DIFF] km.logrank.{field}: uSTAT {a!r} vs R {b!r}")

    r_strata = {s["stratum"].split("=", 1)[-1]: s for s in (r_km.get("strata") or [])}
    r_at = {}
    for a in r_km.get("at_times") or []:
        r_at[(a["stratum"].split("=", 1)[-1], float(a["time"]))] = a

    for g in u_km.get("groups") or []:
        name = str(g.get("group"))
        rs = r_strata.get(name)
        if rs is None:
            bad += 1
            print(f"[DIFF] km: group {name!r} has no R stratum")
            continue
        for field, r_key in (("n", "n"), ("events", "n_events"),
                             ("median_survival", "median")):
            a, b = g.get(field), rs.get(r_key)
            if a is None or b is None:
                continue
            checked += 1
            if not close(a, b):
                bad += 1
                print(f"[DIFF] km.{name}.{field}: uSTAT {a!r} vs R {b!r}")

        for point in g.get("survival_at") or []:
            key = (name, float(point["time"]))
            ra = r_at.get(key)
            if ra is None:
                continue
            for field, r_key in (("survival", "surv"), ("ci_low", "ci_low"),
                                 ("ci_high", "ci_high")):
                a, b = point.get(field), ra.get(r_key)
                if a is None or b is None:
                    continue
                checked += 1
                if not close(a, b):
                    bad += 1
                    print(f"[DIFF] km.{name}@{point['time']}.{field}: "
                          f"uSTAT {a!r} vs R {b!r}")
    return checked, bad


def main() -> int:
    ust = json.loads((HERE / "endpoints.json").read_text())["models"]
    ref: dict = {}
    for name in ("reference_regression.json", "reference_advanced.json",
                 "reference_agreement.json"):
        path = HERE / name
        if path.exists():
            ref.update(json.loads(path.read_text())["models"])
        else:
            print(f"[missing] {name} — those models are not checked")

    mismatches = 0
    checked = 0
    only_ustat, only_r = [], []

    for key in sorted(set(ust) | set(ref)):
        if key not in ref:
            only_ustat.append(key)
            continue
        if key not in ust:
            only_r.append(key)
            continue
        r_model, u_model = ref[key], ust[key]
        if isinstance(r_model, dict) and "error" in r_model:
            print(f"[R could not fit] {key}: {r_model['error']}")
            continue

        r_terms = {canon(t["term"]): t for t in (r_model.get("terms") or [])}
        u_terms = {canon(t["term"]): t for t in (u_model.get("terms") or [])}
        # geepack reports a Wald chi-square where uSTAT reports the z it is
        # the square of. Same test, different convention.
        squares_z = key.startswith("gee")
        for term in sorted(set(r_terms) & set(u_terms)):
            for field in ("estimate", "se", "statistic", "p"):
                a, b = u_terms[term].get(field), r_terms[term].get(field)
                if field == "statistic" and squares_z and a is not None:
                    a = a * a
                if a is None or b is None:
                    continue
                checked += 1
                if not close(a, b, abs_=ABS_P if field == "p" else 0.0):
                    mismatches += 1
                    print(f"[DIFF] {key}.{term}.{field}: uSTAT {a!r} vs R {b!r}")
        # R reports an aliased predictor (one that is collinear with the
        # intercept, e.g. a constant column) as a row with an NA estimate.
        # uSTAT drops it and says so, which is the same decision.
        missing_terms = sorted(
            t for t in set(r_terms) - set(u_terms)
            if r_terms[t].get("estimate") is not None
        )
        if missing_terms:
            mismatches += 1
            print(f"[DIFF] {key}: terms in R but not in uSTAT: {missing_terms}")

        # The agreement models are flat: no coefficient table, just scalars.
        for field in ("r_squared", "adj_r_squared", "sigma", "aic", "bic", "n",
                      "log_likelihood", "dispersion", "theta", "concordance",
                      "r", "p", "ci_low", "ci_high", "icc", "kappa", "z", "se",
                      "po", "pe", "f_stat", "n_subjects",
                      *(k for k in u_model if "|" in str(k))):
            a, b = u_model.get(field), r_model.get(field)
            if a is None or b is None:
                continue
            # irr computes its p as 2*(1-pnorm(|z|)), which underflows to a
            # flat 0 past about z = 8.3. scipy's survival function does not.
            # A tiny p against R's 0 is agreement, not a difference.
            if field == "p" and b == 0.0 and a is not None and abs(a) < 1e-12:
                checked += 1
                continue
            checked += 1
            if not close(a, b):
                mismatches += 1
                print(f"[DIFF] {key}.{field}: uSTAT {a!r} vs R {b!r}")

    # Kaplan-Meier has no coefficient table, so the loop above matched nothing
    # for it and it silently counted as agreement. Compare it on its own terms.
    if "km" in ust and "km" in ref:
        checked_km, bad_km = _compare_km(ust["km"], ref["km"])
        checked += checked_km
        mismatches += bad_km

    if only_ustat:
        print(f"\nno R counterpart yet: {only_ustat}")
    if only_r:
        print(f"in R but not captured from uSTAT: {only_r}")
    print(f"\n{checked} values compared, {mismatches} mismatches")
    return mismatches


if __name__ == "__main__":
    raise SystemExit(main())
