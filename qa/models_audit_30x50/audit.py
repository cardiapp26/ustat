"""Call every regression and survival endpoint on the 30x50 dataset and
record what came back.

Nothing here judges anything: it writes `endpoints.json` in the same shape as
the R `reference.json` so `compare.py` can put the two side by side. An
endpoint that fails is recorded under `errors` with its status and detail,
because at n = 30 a refusal can be the correct answer and a 500 never is.

    backend/.venv/bin/python qa/models_audit_30x50/audit.py
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BACKEND = HERE.parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from services import store  # noqa: E402

SID = "a3050"
SID_LONG = "a3050_long"
SID_REC = "a3050_rec"
SID_MS = "a3050_ms"
SID_EXT = "a3050_ext"
SID_AG = "a3050_raters"


def _f(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _terms(rows, name_key, est_key, se_key, stat_key, p_key, **extra):
    out = []
    for r in rows or []:
        t = {
            "term": str(r.get(name_key)),
            "estimate": _f(r.get(est_key)),
            "se": _f(r.get(se_key)),
            "statistic": _f(r.get(stat_key)),
            "p": _f(r.get(p_key)),
        }
        for k, src in extra.items():
            v = r.get(src)
            t[k] = v if isinstance(v, str) else _f(v)
        out.append(t)
    return out


def main() -> None:
    store.save(SID, pd.read_csv(HERE / "dataset.csv"))
    store.save(SID_LONG, pd.read_csv(HERE / "dataset_long.csv"))
    store.save(SID_REC, pd.read_csv(HERE / "dataset_recurrent.csv"))
    store.save(SID_MS, pd.read_csv(HERE / "dataset_multistate.csv"))
    store.save(SID_EXT, pd.read_csv(HERE / "dataset_external.csv"))
    store.save(SID_AG, pd.read_csv(HERE / "dataset_raters.csv"))

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
                detail = r.text[:500]
            errors[key] = {"status": r.status_code, "detail": detail}
            return None
        return r.json()

    M = "/api/models"
    SA = "/api/survival_advanced"

    # ══ regression ═══════════════════════════════════════════════════════

    j = call("linear", f"{M}/linear", {
        "session_id": SID, "outcome": "sbp",
        "predictors": ["age", "bmi", "arm", "sex"]})
    if j:
        models["linear"] = {
            "terms": _terms(j["coefficients"], "variable", "estimate", "se", "t", "p",
                            ci_low="ci_low", ci_high="ci_high", vif="vif"),
            "r_squared": _f(j.get("r_squared")),
            "adj_r_squared": _f(j.get("adj_r_squared")),
            "f_statistic": _f(j.get("f_stat")), "f_dendf": _f(j.get("df_resid")),
            "p": _f(j.get("f_p")), "sigma": _f(j.get("residual_se")),
            "aic": _f(j.get("aic")), "bic": _f(j.get("bic")), "n": _f(j.get("n")),
            "raw_keys": sorted(j.keys()),
        }

    # Near-collinear pair: score2 = 2 * score1 + noise. VIF must be large and
    # finite, and the fit must not silently drop a column.
    j = call("linear_collinear", f"{M}/linear", {
        "session_id": SID, "outcome": "sbp",
        "predictors": ["score1", "score2", "age"]})
    if j:
        models["linear_collinear"] = {
            "terms": _terms(j["coefficients"], "variable", "estimate", "se", "t", "p",
                            vif="vif"),
            "r_squared": _f(j.get("r_squared")),
        }

    # A predictor with no variance at all.
    j = call("linear_constant", f"{M}/linear", {
        "session_id": SID, "outcome": "sbp", "predictors": ["age", "const_num"]})
    if j:
        models["linear_constant"] = {
            "terms": _terms(j["coefficients"], "variable", "estimate", "se", "t", "p"),
            "n": _f(j.get("n")),
        }

    # A category whose third level has one observation.
    j = call("linear_rare", f"{M}/linear", {
        "session_id": SID, "outcome": "sbp", "predictors": ["age", "rare_grp"]})
    if j:
        models["linear_rare"] = {
            "terms": _terms(j["coefficients"], "variable", "estimate", "se", "t", "p"),
            "n": _f(j.get("n")),
        }

    # The MAR-heavy predictors: complete-case n drops to 16.
    j = call("linear_mar", f"{M}/linear", {
        "session_id": SID, "outcome": "sbp", "predictors": ["egfr", "crp", "age"]})
    if j:
        models["linear_mar"] = {
            "terms": _terms(j["coefficients"], "variable", "estimate", "se", "t", "p"),
            "n": _f(j.get("n")), "r_squared": _f(j.get("r_squared")),
        }

    j = call("linear_robust", f"{M}/linear", {
        "session_id": SID, "outcome": "sbp", "predictors": ["age", "bmi", "arm"],
        "robust_se": True})
    if j:
        models["linear_robust"] = {
            "terms": _terms(j["coefficients"], "variable", "estimate", "se", "t", "p"),
            "raw_keys": sorted(j.keys()),
        }

    j = call("polynomial", f"{M}/polynomial", {
        "session_id": SID, "outcome": "sbp", "predictor": "age",
        "degree": 2, "covariates": ["arm"]})
    if j:
        models["polynomial"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "estimate", "se",
                            "t", "p"),
            "r_squared": _f(j.get("r_squared")),
        }

    j = call("stepwise", f"{M}/stepwise", {
        "session_id": SID, "model_type": "linear", "outcome": "sbp",
        "candidates": ["age", "bmi", "arm", "sex", "score1"],
        "direction": "both", "criterion": "aic"})
    if j:
        models["stepwise"] = {"selected": j.get("selected") or [],
                              "aic": _f(j.get("final_aic")),
                              "raw_keys": sorted(j.keys())}

    j = call("linear_diag", f"{M}/linear_diag", {
        "session_id": SID, "outcome": "sbp", "predictors": ["age", "bmi", "arm"]})
    if j:
        models["linear_diag"] = {
            "raw_keys": sorted(j.keys()),
            "breusch_pagan_p": _f((j.get("breusch_pagan") or {}).get("p")),
            "shapiro_p": _f((j.get("shapiro") or {}).get("p")),
            "durbin_watson": _f(j.get("durbin_watson")),
            "body": j,
        }

    j = call("lmm", f"{M}/lmm", {
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

    j = call("logistic", f"{M}/logistic", {
        "session_id": SID, "outcome": "event_binary",
        "predictors": ["age", "bmi", "arm"]})
    if j:
        models["logistic"] = {
            "terms": _terms(j["coefficients"], "variable", "B", "se", "z", "p",
                            odds_ratio="odds_ratio", or_ci_low="or_ci_low",
                            or_ci_high="or_ci_high", vif="vif"),
            "aic": _f(j.get("aic")), "bic": _f(j.get("bic")), "n": _f(j.get("n")),
            "log_likelihood": _f(j.get("log_likelihood")),
            "auc": _f(j.get("auc")),
            "raw_keys": sorted(j.keys()),
        }

    # Perfect separation: sep_binary is prior_tx. Ordinary ML diverges.
    j = call("logistic_separated", f"{M}/logistic", {
        "session_id": SID, "outcome": "sep_binary", "predictors": ["prior_tx"]})
    if j:
        models["logistic_separated"] = {
            "terms": _terms(j["coefficients"], "variable", "B", "se", "z", "p",
                            odds_ratio="odds_ratio"),
            "warnings": j.get("warnings"), "raw_keys": sorted(j.keys()),
        }

    j = call("firth_separated", f"{M}/firth_logistic", {
        "session_id": SID, "outcome": "sep_binary", "predictors": ["prior_tx"]})
    if j:
        models["firth_separated"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "B", "se", "z", "p",
                            odds_ratio="odds_ratio", or_ci_low="or_ci_low",
                            or_ci_high="or_ci_high"),
            "raw_keys": sorted(j.keys()),
        }

    j = call("firth_logistic", f"{M}/firth_logistic", {
        "session_id": SID, "outcome": "event_binary", "predictors": ["age", "arm"]})
    if j:
        models["firth_logistic"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "B", "se", "z", "p",
                            odds_ratio="odds_ratio", or_ci_low="or_ci_low",
                            or_ci_high="or_ci_high"),
            "raw_keys": sorted(j.keys()),
        }

    j = call("logistic_table", f"{M}/logistic_table", {
        "session_id": SID, "outcome": "event_binary",
        "predictors": ["age", "arm", "stage"]})
    if j:
        models["logistic_table"] = {"raw_keys": sorted(j.keys()), "body": j}

    j = call("poisson", f"{M}/poisson", {
        "session_id": SID, "outcome": "admissions", "predictors": ["age", "arm"]})
    if j:
        models["poisson"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "log_irr", "se",
                            "z", "p", irr="irr", ci_low="ci_low", ci_high="ci_high"),
            "aic": _f(j.get("aic")), "raw_keys": sorted(j.keys()),
        }

    j = call("gamma", f"{M}/gamma", {
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

    j = call("negbinom", f"{M}/negbinom", {
        "session_id": SID, "outcome": "visits", "predictors": ["age", "arm"]})
    if j:
        models["negbinom"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "log_irr", "se",
                            "z", "p", irr="irr"),
            "aic": _f(j.get("aic")), "alpha": _f(j.get("alpha")),
            "theta": _f(j.get("theta")), "raw_keys": sorted(j.keys()),
        }

    j = call("gee", f"{M}/gee", {
        "session_id": SID_LONG, "outcome": "score",
        "predictors": ["visit", "arm", "age"], "group_col": "pid",
        "family": "gaussian", "cov_struct": "exchangeable"})
    if j:
        models["gee"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "estimate", "se",
                            "z", "p"),
            "cov_struct_used": j.get("cov_struct_used"),
            "raw_keys": sorted(j.keys()),
        }

    j = call("gee_binomial", f"{M}/gee", {
        "session_id": SID_LONG, "outcome": "resp",
        "predictors": ["visit", "arm"], "group_col": "pid",
        "family": "binomial", "cov_struct": "exchangeable"})
    if j:
        models["gee_binomial"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "estimate", "se",
                            "z", "p"),
        }

    j = call("ordinal", f"{M}/ordinal", {
        "session_id": SID, "outcome": "grade", "predictors": ["age", "arm"]})
    if j:
        models["ordinal"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "log_odds", "se",
                            "z", "p", odds_ratio="odds_ratio"),
            "thresholds": j.get("thresholds") or j.get("intercepts"),
            "raw_keys": sorted(j.keys()),
        }

    j = call("multi_outcome", f"{M}/multi_outcome_regression", {
        "session_id": SID, "outcomes": ["sbp", "qol"], "predictors": ["arm"],
        "covariates": ["age"], "standardize": False})
    if j:
        models["multi_outcome"] = {"raw_keys": sorted(j.keys()), "body": j}

    # ══ survival, core ═══════════════════════════════════════════════════

    j = call("km", f"{M}/survival/km", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "group_col": "arm", "survival_times": [5.0, 10.0]})
    if j:
        models["km"] = {
            "logrank": j.get("logrank"), "groups": j.get("groups"),
            "median_survival": j.get("median_survival"),
            "survival_at_times": j.get("survival_at_times"),
            "raw_keys": sorted(j.keys()),
        }

    j = call("cox", f"{M}/survival/cox", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "predictors": ["age", "arm", "sex"]})
    if j:
        models["cox"] = {
            "terms": _terms(j["coefficients"], "variable", "log_hr", "se", "z", "p",
                            hr="hr", hr_ci_low="hr_ci_low", hr_ci_high="hr_ci_high",
                            vif="vif"),
            "concordance": _f(j.get("concordance")),
            "log_likelihood": _f(j.get("log_likelihood")),
            "n": _f(j.get("n_analyzed") or j.get("n")),
            "ph_test": j.get("ph_test"), "raw_keys": sorted(j.keys()),
        }

    # Missing covariate: complete-case n = 27, and R must be told to match.
    j = call("cox_missing", f"{M}/survival/cox", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "predictors": ["age", "arm", "bmi"]})
    if j:
        models["cox_missing"] = {
            "terms": _terms(j["coefficients"], "variable", "log_hr", "se", "z", "p",
                            hr="hr"),
            "n": _f(j.get("n_analyzed") or j.get("n")),
        }

    j = call("cox_horizons", f"{M}/survival/cox_horizons", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "predictor": "arm", "covariates": ["age"], "horizons": [5.0, 10.0]})
    if j:
        models["cox_horizons"] = {"raw_keys": sorted(j.keys()), "body": j}

    j = call("cox_tv", f"{M}/survival/cox_tv", {
        "session_id": SID_REC, "id_col": "pid", "start_col": "start",
        "stop_col": "stop", "event_col": "event", "predictors": ["age", "arm"]})
    if j:
        models["cox_tv"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "log_hr", "se",
                            "z", "p", hr="hr"),
            "raw_keys": sorted(j.keys()),
        }

    j = call("cox_uni_multi", f"{M}/survival/cox_uni_multi", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "predictors": ["age", "arm", "sex"]})
    if j:
        models["cox_uni_multi"] = {"raw_keys": sorted(j.keys()), "body": j}

    j = call("rcs_logistic", f"{M}/rcs", {
        "session_id": SID, "predictor": "age", "outcome": "event_binary",
        "covariates": ["arm"], "n_knots": 3, "model_type": "logistic"})
    if j:
        models["rcs_logistic"] = {"raw_keys": sorted(j.keys()),
                                  "knots": j.get("knots"),
                                  "nonlinearity": j.get("nonlinearity_test")
                                  or j.get("nonlinearity")}

    j = call("cox_rcs", f"{M}/survival/cox_rcs", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "spline_terms": [{"column": "age", "n_knots": 3}], "covariates": ["arm"]})
    if j:
        models["cox_rcs"] = {"raw_keys": sorted(j.keys()),
                             "knots": j.get("knots"),
                             "nonlinearity": j.get("nonlinearity_test")
                             or j.get("nonlinearity")}

    j = call("psm", f"{M}/psm", {
        "session_id": SID, "treatment_col": "arm",
        "covariates": ["age", "bmi", "sex"], "outcome_col": "event_binary",
        "caliper": 0.2, "caliper_scale": "logit", "outcome_type": "binary"})
    if j:
        models["psm"] = {"raw_keys": sorted(j.keys()),
                         "n_matched": _f(j.get("n_matched_pairs")
                                         or j.get("n_matched")),
                         "balance": j.get("balance"), "effect": j.get("effect")
                         or j.get("outcome_analysis")}

    j = call("iptw", f"{M}/iptw", {
        "session_id": SID, "treatment_col": "arm",
        "covariates": ["age", "bmi", "sex"], "outcome_col": "event_binary",
        "outcome_type": "binary", "estimand": "ate", "stabilize": True})
    if j:
        models["iptw"] = {"raw_keys": sorted(j.keys()),
                          "effect": j.get("effect") or j.get("outcome_analysis"),
                          "weights_summary": j.get("weights_summary")}

    # ══ survival, advanced ═══════════════════════════════════════════════

    j = call("fine_gray", f"{SA}/fine_gray", {
        "session_id": SID, "duration_col": "time", "event_col": "cmp_status",
        "event_of_interest": 1, "predictors": ["age", "arm"]})
    if j:
        reg = j.get("regression_result") or {}
        models["fine_gray"] = {
            "terms": _terms(reg.get("coefficients", []), "variable", "estimate",
                            "se", "z", "p", shr="shr"),
            "n": _f(j.get("n")), "raw_keys": sorted(j.keys()), "body": j,
        }

    j = call("rmst", f"{SA}/rmst", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "tau": 10.0, "group_col": "arm"})
    if j:
        models["rmst"] = {"raw_keys": sorted(j.keys()), "body": j}

    j = call("landmark", f"{SA}/landmark", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "landmark_time": 5.0, "group_col": "arm", "predictors": ["age", "arm"]})
    if j:
        models["landmark"] = {"raw_keys": sorted(j.keys()), "body": j}

    j = call("frailty", f"{SA}/frailty", {
        "session_id": SID, "duration_col": "time", "event_col": "status",
        "cluster_col": "site", "predictors": ["age", "arm"]})
    if j:
        frailty_terms = _terms(j.get("coefficients", []), "variable", "estimate",
                               "se", "z", "p", hr="hr")
        models["frailty"] = {
            "terms": frailty_terms, "theta": _f(j.get("theta")),
            "raw_keys": sorted(j.keys()), "body": j,
        }
        # The endpoint reports a cluster-robust marginal fit, so that — not R's
        # frailty() — is the like-for-like reference.
        models["frailty_cluster_robust"] = {"terms": frailty_terms}

    j = call("interval_censored", f"{SA}/interval_censored", {
        "session_id": SID, "lower_col": "ic_l", "upper_col": "ic_r",
        "covariates": ["arm"]})
    if j:
        models["interval_censored"] = {"raw_keys": sorted(j.keys()), "body": j}

    j = call("recurrent_lwyy", f"{SA}/recurrent_lwyy", {
        "session_id": SID_REC, "id_col": "pid", "start_col": "start",
        "stop_col": "stop", "event_col": "event", "predictors": ["age", "arm"]})
    if j:
        models["recurrent_lwyy"] = {
            "terms": _terms(j.get("coefficients", []), "variable", "estimate",
                            "robust_se", "z", "p", rate_ratio="rate_ratio"),
            "raw_keys": sorted(j.keys()), "body": j,
        }

    j = call("multistate", f"{SA}/multistate", {
        "session_id": SID_MS, "id_col": "id", "from_state_col": "from_state",
        "to_state_col": "to_state", "entry_col": "entry", "exit_col": "exit",
        "event_col": "event", "predictors": ["age", "arm"]})
    if j:
        models["multistate"] = {"raw_keys": sorted(j.keys()), "body": j}

    j = call("dynamic_prediction", f"{SA}/dynamic_prediction", {
        "session_id": SID_MS, "landmark_time": 3.0, "current_state": 1,
        "id_col": "id", "from_state_col": "from_state", "to_state_col": "to_state",
        "entry_col": "entry", "exit_col": "exit", "event_col": "event",
        "predictors": ["age", "arm"], "horizon": 8.0, "n_points": 5})
    if j:
        models["dynamic_prediction"] = {"raw_keys": sorted(j.keys())}

    j = call("external_validation", f"{SA}/external_validation", {
        "session_id": SID_EXT, "duration_col": "time", "event_col": "status",
        "predicted_lp_col": "pred_lp", "time_points": [5.0, 10.0]})
    if j:
        models["external_validation"] = {"raw_keys": sorted(j.keys()), "body": j}

    j = call("evalue", f"{SA}/evalue", {
        "estimate": 2.5, "ci_low": 1.4, "ci_high": 4.5, "measure_type": "HR",
        "baseline_risk": 0.1})
    if j:
        models["evalue"] = {"raw_keys": sorted(j.keys()), "body": j}

    # ══ agreement: the Correlation panel's four tabs ═════════════════════
    S = "/api/stats"

    for key, method in (("corr_pearson", "pearson"), ("corr_spearman", "spearman"),
                        ("corr_kendall", "kendall")):
        j = call(key, f"{S}/correlation_pair", {
            "session_id": SID, "var1": "age", "var2": "bmi", "method": method})
        if j:
            models[key] = {"r": _f(j.get("r")), "p": _f(j.get("p")),
                           "ci_low": _f(j.get("ci_low")),
                           "ci_high": _f(j.get("ci_high")), "n": _f(j.get("n"))}

    j = call("corr_matrix", f"{S}/correlation_matrix", {
        "session_id": SID, "variables": ["age", "bmi", "sbp", "score1"],
        "method": "pearson"})
    if j:
        mat = j.get("matrix") or {}
        flat = {}
        for col, inner in mat.items():
            for row, v in (inner or {}).items():
                flat[f"{row}|{col}"] = _f(v)
        models["corr_matrix"] = flat

    j = call("icc", f"{S}/icc", {
        "session_id": SID_AG, "rater1_col": "rater_a", "rater2_col": "rater_b"})
    if j:
        models["icc"] = {k: _f(j.get(k))
                         for k in ("icc", "ci_low", "ci_high", "f_stat", "n")}

    j = call("cohens_kappa", f"{S}/cohens_kappa", {
        "session_id": SID_AG, "rater1_col": "cat_a", "rater2_col": "cat_b"})
    if j:
        models["cohens_kappa"] = {k: _f(j.get(k))
                                  for k in ("kappa", "z", "p", "n", "po", "pe")}

    j = call("fleiss_kappa", f"{S}/fleiss_kappa", {
        "session_id": SID_AG, "rater_cols": ["cat_a", "cat_b", "cat_c"]})
    if j:
        models["fleiss_kappa"] = {"kappa": _f(j.get("kappa")), "z": _f(j.get("z")),
                                  "p": _f(j.get("p")), "se": _f(j.get("se")),
                                  "n_subjects": _f(j.get("n_subjects"))}

    out = {"models": models, "errors": errors}
    (HERE / "endpoints.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"captured {len(models)}: {sorted(models)}")
    print(f"errors {len(errors)}: {sorted(errors)}")
    for k, v in errors.items():
        print(f"  {k}: {v['status']} {str(v['detail'])[:220]}")


if __name__ == "__main__":
    main()
