"""Call every Models endpoint on the sample dataset and normalise the output.

Writes `endpoints.json` in the same shape as the R `reference.json`, so
`compare.py` can put the two side by side. Nothing here judges anything —
it only records what the product returns.

    backend/.venv/bin/python qa/models_audit/audit.py
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BACKEND = HERE.parent.parent / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "tests"))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from services import store  # noqa: E402

SID = "models_audit"
SID_LONG = "models_audit_long"


def _f(v):
    """Best-effort float; None stays None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _terms(rows, name_key, est_key, se_key, stat_key, p_key, **extra):
    out = []
    for r in rows:
        t = {
            "term": str(r.get(name_key)),
            "estimate": _f(r.get(est_key)),
            "se": _f(r.get(se_key)),
            "statistic": _f(r.get(stat_key)),
            "p": _f(r.get(p_key)),
        }
        for k, src in extra.items():
            t[k] = _f(r.get(src)) if not isinstance(r.get(src), str) else r.get(src)
        out.append(t)
    return out


def main() -> None:
    d = pd.read_csv(HERE / "dataset.csv")
    long = pd.read_csv(HERE / "dataset_long.csv")
    store.save(SID, d)
    store.save(SID_LONG, long)
    c = TestClient(app)
    models: dict = {}
    errors: dict = {}

    def call(key, path, body):
        try:
            r = c.post(path, json=body)
        except Exception as exc:  # an unhandled 500 propagates through TestClient
            errors[key] = {"status": 500, "detail": f"{type(exc).__name__}: {exc}"}
            return None
        if r.status_code != 200:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:400]
            errors[key] = {"status": r.status_code, "detail": detail}
            return None
        return r.json()

    # ── linear ────────────────────────────────────────────────────────────
    j = call("linear", "/api/models/linear", {
        "session_id": SID, "outcome": "sbp",
        "predictors": ["age", "bmi", "arm", "sex"]})
    if j:
        models["linear"] = {
            "terms": _terms(j["coefficients"], "variable", "estimate", "se", "t", "p",
                            ci_low="ci_low", ci_high="ci_high"),
            "r_squared": _f(j.get("r_squared")),
            "adj_r_squared": _f(j.get("adj_r_squared")),
            "f_statistic": _f(j.get("f_stat")),
            "f_dendf": _f(j.get("df_resid")),
            "p": _f(j.get("f_p")),
            "sigma": _f(j.get("residual_se")),
            "aic": _f(j.get("aic")), "bic": _f(j.get("bic")),
            "n": _f(j.get("n")),
        }

    # ── polynomial ────────────────────────────────────────────────────────
    j = call("polynomial_numeric_only", "/api/models/polynomial", {
        "session_id": SID, "outcome": "sbp", "predictor": "age",
        "degree": 2, "covariates": []})
    if j:
        models["polynomial_numeric_only"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "estimate", "se",
                            "t", "p"),
            "r_squared": _f(j.get("r_squared")),
        }
    j = call("polynomial", "/api/models/polynomial", {
        "session_id": SID, "outcome": "sbp", "predictor": "age",
        "degree": 2, "covariates": ["arm"]})
    if j:
        models["polynomial"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "estimate", "se",
                            "t", "p"),
            "r_squared": _f(j.get("r_squared")),
        }

    # ── stepwise ──────────────────────────────────────────────────────────
    j = call("stepwise", "/api/models/stepwise", {
        "session_id": SID, "model_type": "linear", "outcome": "sbp",
        "candidates": ["age", "bmi", "arm", "sex", "biomarker"],
        "direction": "both", "criterion": "aic"})
    if j:
        models["stepwise"] = {
            "selected": j.get("selected") or [],
            "aic": _f(j.get("final_aic")),
            "raw_keys": sorted(j.keys()),
        }

    # ── logistic ──────────────────────────────────────────────────────────
    j = call("logistic", "/api/models/logistic", {
        "session_id": SID, "outcome": "event_binary",
        "predictors": ["age", "bmi", "arm", "sex"]})
    if j:
        models["logistic"] = {
            "terms": _terms(j["coefficients"], "variable", "B", "se", "z", "p",
                            odds_ratio="odds_ratio", or_ci_low="or_ci_low",
                            or_ci_high="or_ci_high", wald="wald"),
            "aic": _f(j.get("aic")), "bic": _f(j.get("bic")),
            "n": _f(j.get("n")),
            "log_likelihood": _f(j.get("log_likelihood")),
            "minus2ll": _f(j.get("minus2ll")),
            "auc": _f(j.get("auc")),
        }

    # ── Firth ─────────────────────────────────────────────────────────────
    j = call("firth_logistic", "/api/models/firth_logistic", {
        "session_id": SID, "outcome": "event_binary", "predictors": ["age", "arm"]})
    if j:
        models["firth_logistic"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "B", "se", "z", "p",
                            odds_ratio="odds_ratio", or_ci_low="or_ci_low",
                            or_ci_high="or_ci_high"),
            "raw_keys": sorted(j.keys()),
        }

    # ── Poisson ───────────────────────────────────────────────────────────
    j = call("poisson", "/api/models/poisson", {
        "session_id": SID, "outcome": "admissions", "predictors": ["age", "arm"]})
    if j:
        models["poisson"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "log_irr", "se",
                            "z", "p", irr="irr",
                            ci_low="ci_low", ci_high="ci_high"),
            "aic": _f(j.get("aic")),
            "raw_keys": sorted(j.keys()),
        }

    # ── Gamma ─────────────────────────────────────────────────────────────
    j = call("gamma", "/api/models/gamma", {
        "session_id": SID, "outcome": "cost", "predictors": ["age", "arm"],
        "link": "log"})
    if j:
        models["gamma"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "estimate", "se",
                            "z", "p"),
            "aic": _f(j.get("aic")),
            "dispersion": _f(j.get("dispersion") if j.get("dispersion") is not None
                             else j.get("scale")),
            "raw_keys": sorted(j.keys()),
        }

    # ── Negative binomial ─────────────────────────────────────────────────
    j = call("negbinom", "/api/models/negbinom", {
        "session_id": SID, "outcome": "visits", "predictors": ["age", "arm"]})
    if j:
        models["negbinom"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "log_irr", "se",
                            "z", "p", irr="irr"),
            "aic": _f(j.get("aic")),
            "alpha": _f(j.get("alpha")),
            "theta": _f(j.get("theta")),
            "raw_keys": sorted(j.keys()),
        }

    # ── Ordinal ───────────────────────────────────────────────────────────
    j = call("ordinal", "/api/models/ordinal", {
        "session_id": SID, "outcome": "grade", "predictors": ["age", "arm"]})
    if j:
        models["ordinal"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "log_odds", "se",
                            "z", "p", odds_ratio="odds_ratio"),
            "thresholds": j.get("thresholds") or j.get("intercepts"),
            "raw_keys": sorted(j.keys()),
        }

    # ── Cox ───────────────────────────────────────────────────────────────
    j = call("cox", "/api/models/survival/cox", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "predictors": ["age", "arm", "sex"]})
    if j:
        models["cox"] = {
            "terms": _terms(j["coefficients"], "variable", "log_hr", "se", "z", "p",
                            hr="hr", hr_ci_low="hr_ci_low", hr_ci_high="hr_ci_high"),
            "concordance": _f(j.get("concordance")),
            "log_likelihood": _f(j.get("log_likelihood")),
            "n": _f(j.get("n_analyzed") or j.get("n")),
            "ph_test": j.get("ph_test"),
        }

    # ── Kaplan-Meier ──────────────────────────────────────────────────────
    j = call("km", "/api/models/survival/km", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "group_col": "arm", "survival_times": [5.0, 10.0]})
    if j:
        models["km"] = {
            "raw_keys": sorted(j.keys()),
            "logrank_p": _f((j.get("logrank") or {}).get("p")),
            "logrank": j.get("logrank"),
            "groups": j.get("groups"),
            "median_survival": j.get("median_survival"),
            "survival_at_times": j.get("survival_at_times"),
        }

    # ── Linear mixed model ────────────────────────────────────────────────
    j = call("lmm", "/api/models/lmm", {
        "session_id": SID_LONG, "outcome": "score",
        "fixed_effects": ["visit", "arm", "age"], "group_col": "pid"})
    if j:
        models["lmm"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "estimate", "se",
                            "z", "p"),
            "group_var": _f(j.get("random_effect_variance")),
            "residual_var": _f(j.get("residual_variance")),
            "raw_keys": sorted(j.keys()),
        }

    out = {"models": models, "errors": errors}
    (HERE / "endpoints.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"models captured: {sorted(models)}")
    if errors:
        print(f"endpoint errors: {json.dumps(errors, indent=2, default=str)[:2000]}")


if __name__ == "__main__":
    main()
