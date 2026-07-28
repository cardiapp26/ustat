"""200x20 dogrulama veri seti ile uSTAT endpoint testleri.

Kullanim: python run_tests.py <alan>
Alanlar: inferential, nonparametric, repeated, categorical, correlation,
         agreement, reliability, table1, descriptive
Her alan: (1) referans hesaplama (dogrudan scipy/statsmodels, bagimsiz),
(2) uSTAT endpoint cagrisi, (3) karsilastirma + tutarlilik bulgulari.
Sonuc: results/<alan>.json  (findings, reference, api)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as S

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "..", "..", "backend"))
os.chdir(os.path.join(BASE, "..", "..", "backend"))

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from services import store  # noqa: E402

AREA = sys.argv[1] if len(sys.argv) > 1 else "inferential"
client = TestClient(app)
df = pd.read_csv(os.path.join(BASE, "dataset_200x20.csv"))
store.save("qa_session", df)
SID = "qa_session"

REF = {}       # bagimsiz referans hesaplar
API = {}       # uSTAT endpoint cevaplari (ozet)
FINDINGS = []  # {test, status, detail}

def post(url, payload):
    r = client.post(url, json=payload)
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text[:500]}
    return r.status_code, body

def check(test, ok, detail):
    FINDINGS.append({"test": test, "status": "PASS" if ok else "FAIL", "detail": detail})

def close(a, b, tol=1e-4):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol

# kolonlar (ground truth)
CONT = ["age", "bmi", "sbp", "cholesterol", "glucose", "score", "biomarker", "pre", "post", "time"]
CATS = ["event", "sex", "group", "education", "smoking", "diabetes", "hypertension", "stage", "center", "response"]

d = df.copy()

# ═══════════════════════════ INFERENTIAL ═══════════════════════════
if AREA == "inferential":
    # ---- 1. t-test (score ~ group, gercek fark +5) ----
    w = d[["score", "group"]].dropna()
    gA = w[w.group == "A"].score.values
    gB = w[w.group == "B"].score.values
    lev_p = S.levene(gA, gB).pvalue
    eq = lev_p >= 0.05
    t_ref, p_ref = S.ttest_ind(gA, gB, equal_var=eq)
    REF["ttest_score_group"] = {"t": t_ref, "p": p_ref, "equal_var": bool(eq), "levene_p": lev_p,
                                "meanA": gA.mean(), "meanB": gB.mean(), "nA": len(gA), "nB": len(gB)}
    sc, r = post("/api/stats/ttest", {"session_id": SID, "column": "score", "group_column": "group"})
    API["ttest_score_group"] = {"status": sc, "t": r.get("t"), "p": r.get("p"), "df": r.get("df"),
                                "test_name": r.get("test"), "interp": r.get("interpretation"),
                                "df_method": r.get("df_method"),
                                "es": [e.get("name") for e in r.get("effect_sizes", [])]}
    check("ttest score~group HTTP200", sc == 200, f"status={sc} body={str(r)[:200]}")
    if sc == 200:
        check("ttest t degeri scipy ile ayni", close(r["t"], t_ref), f"api={r['t']:.5f} ref={t_ref:.5f}")
        check("ttest p degeri scipy ile ayni", close(r["p"], p_ref, 1e-6), f"api={r['p']:.3e} ref={p_ref:.3e}")
        check("ttest anlamli (beklenen +5 fark)", r["p"] < 0.05, f"p={r['p']:.3e}")
        check("ttest Welch/Student secimi Levene ile tutarli",
              (r["variance_assumption"] == "student") == eq,
              f"api={r['variance_assumption']} levene_p={lev_p:.4f}")
        esv = r["effect_sizes"][0]
        pooled_sd = np.sqrt(
            ((len(gA) - 1) * gA.var(ddof=1) + (len(gB) - 1) * gB.var(ddof=1))
            / (len(gA) + len(gB) - 2)
        )
        d_ref = (gA.mean() - gB.mean()) / pooled_sd
        correction = 1 - 3 / (4 * (len(gA) + len(gB) - 2) - 1)
        g_ref = d_ref * correction
        check(
            "ttest ES Hedges g olarak etiketli ve hesapli",
            esv.get("name") == "hedges_g"
            and close(esv.get("value"), g_ref, 1e-4)
            and "Hedges' g" in r.get("interpretation", ""),
            f"api={esv} ref_g={g_ref:.6f}",
        )

    # ---- 2. t-test (biomarker ~ group, gercek fark 0) ----
    w2 = d[["biomarker", "group"]].dropna()
    bA = w2[w2.group == "A"].biomarker.values
    bB = w2[w2.group == "B"].biomarker.values
    t2, p2 = S.ttest_ind(bA, bB, equal_var=(S.levene(bA, bB).pvalue >= 0.05))
    sc, r = post("/api/stats/ttest", {"session_id": SID, "column": "biomarker", "group_column": "group"})
    check("ttest biomarker anlamsiz (null dogru)", sc == 200 and r["p"] > 0.05, f"p={r.get('p')}")
    if sc == 200:
        check("ttest biomarker p scipy ile ayni", close(r["p"], p2, 1e-6), f"api={r['p']:.3e} ref={p2:.3e}")

    # ---- 3. one-sample t-test ----
    x = d.age.dropna().values
    t3, p3 = S.ttest_1samp(x, 50.0)
    sc, r = post("/api/stats/ttest", {"session_id": SID, "column": "age", "mu": 50})
    check("one-sample t HTTP200+p", sc == 200 and close(r["t"], t3) and close(r["p"], p3, 1e-6),
          f"api t={r.get('t')} p={r.get('p')} ref t={t3:.4f} p={p3:.3e}")

    # ---- 4. chi-square (sex x group: bagimsiz) ----
    ct = pd.crosstab(d.sex, d.group)
    chi2, p4, dof, exp = S.chi2_contingency(ct)
    sc, r = post("/api/stats/chisquare", {"session_id": SID, "row_column": "sex", "col_column": "group"})
    REF["chi_sex_group"] = {"chi2": chi2, "p": p4, "dof": int(dof)}
    check("chi2 sex x group", sc == 200 and close(r["chi2"], chi2) and close(r["p"], p4, 1e-6) and r["dof"] == dof,
          f"api={r.get('chi2')},{r.get('p')},{r.get('dof')} ref={chi2:.4f},{p4:.4f},{dof}")
    if sc == 200:
        check("chi2 sex x group anlamsiz (tasarim geregi)", r["p"] > 0.05, f"p={r['p']:.4f}")

    # ---- 5. chi-square (stage x group) ----
    ct2 = pd.crosstab(d.stage, d.group)
    chi2b, p5, dof2, _ = S.chi2_contingency(ct2)
    sc, r = post("/api/stats/chisquare", {"session_id": SID, "row_column": "stage", "col_column": "group"})
    check("chi2 stage x group", sc == 200 and close(r["chi2"], chi2b) and close(r["p"], p5, 1e-6),
          f"api={r.get('chi2')},{r.get('p')} ref={chi2b:.4f},{p5:.4f}")

    # ---- 6. Fisher (diabetes x group 2x2) ----
    ct3 = pd.crosstab(d.diabetes, d.group)
    orv, p6 = S.fisher_exact(ct3.values)
    sc, r = post("/api/stats/fisher", {"session_id": SID, "row_column": "diabetes", "col_column": "group"})
    check("fisher diabetes x group", sc == 200 and close(r["odds_ratio"], orv, 1e-3) and close(r["p"], p6, 1e-6),
          f"api OR={r.get('odds_ratio')} p={r.get('p')} ref OR={orv:.4f} p={p6:.4f}")

    # ---- 7. ANOVA (score ~ education) ----
    grp = {k: g.score.dropna().values for k, g in d.groupby("education")}
    lev7 = S.levene(*grp.values()).pvalue
    if lev7 < 0.05:
        from statsmodels.stats.oneway import anova_oneway
        _w7 = anova_oneway(list(grp.values()), use_var="unequal")
        F, p7 = float(_w7.statistic), float(_w7.pvalue)
    else:
        F, p7 = S.f_oneway(*grp.values())
    REF["anova_levene"] = {"levene_p": lev7, "welch": bool(lev7 < 0.05)}
    sc, r = post("/api/stats/anova", {"session_id": SID, "column": "score", "group_column": "education"})
    REF["anova_score_edu"] = {"F": F, "p": p7}
    check("anova score~education F/p", sc == 200 and close(r["F"], F) and close(r["p"], p7, 1e-6),
          f"api F={r.get('F')} p={r.get('p')} ref F={F:.4f} p={p7:.4f}")
    if sc == 200:
        check("anova posthoc method variansla tutarli", r.get("posthoc_method") in (None, "Tukey HSD", "Games-Howell (unequal variances)"),
              f"posthoc={r.get('posthoc_method')} (F {'sig' if p7<0.05 else 'ns'} oldugunda {'dolu' if p7<0.05 else 'None'} beklenir; api={'sig' if r['significant'] else 'ns'})")
        if p7 < 0.05:
            tukey = S.tukey_hsd(*grp.values())
            api_ph = r.get("posthoc") or []
            ok_ph = len(api_ph) == 3
            check("anova posthoc karsilastirma sayisi (3 pair)", ok_ph, f"n_pairs={len(api_ph)}")

    # ---- 8. TOST (biomarker, fark ~0, ±0.8 marjin) ----
    from statsmodels.stats.weightstats import ttost_ind
    po, (tl, pl, _), (th, ph, _) = ttost_ind(bA, bB, low=-0.8, upp=0.8, usevar="pooled")
    sc, r = post("/api/stats/tost", {"session_id": SID, "column": "biomarker", "group_column": "group",
                                     "low": -0.8, "high": 0.8, "test_type": "independent"})
    check("tost p_overall statsmodels ile ayni", sc == 200 and close(r["p_overall"], po, 1e-6),
          f"api={r.get('p_overall')} ref={po:.6f}")
    if sc == 200:
        check("tost esdegerlik karari tutarli", r["equivalent"] == (po < 0.05), f"equiv={r['equivalent']} ref_p={po:.5f}")

    # ---- 9. Non-inferiority (score, mean diff, marjin 4) ----
    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
    cm = CompareMeans(DescrStatsW(gA), DescrStatsW(gB))
    lo_ref, hi_ref = cm.tconfint_diff(alpha=0.10, usevar="unequal")
    sc, r = post("/api/stats/noninferiority", {"session_id": SID, "outcome_col": "score", "group_col": "group",
                                               "outcome_type": "continuous", "margin": 4.0, "bound": "upper",
                                               "alpha": 0.05, "test_group": "A", "ref_group": "B"})
    check("noninferiority HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")
    if sc == 200:
        est_api = r["estimate"]
        est_ref = gA.mean() - gB.mean()
        check("noninferiority estimate A-B dogru", close(est_api, est_ref, 1e-3), f"api={est_api} ref={est_ref:.4f}")
        check("noninferiority CI smodels ile ayni", close(r["ci_low"], lo_ref, 1e-3) and close(r["ci_high"], hi_ref, 1e-3),
              f"api=({r['ci_low']},{r['ci_high']}) ref=({lo_ref:.4f},{hi_ref:.4f})")
        check("noninferiority karar CI kurali ile tutarli", r["non_inferior"] == (r["ci_high"] < 4.0),
              f"ni={r['non_inferior']} ci_high={r['ci_high']} margin=4")

    # ---- 10. Power (t_two solve n) ----
    from statsmodels.stats.power import TTestIndPower
    n_ref = TTestIndPower().solve_power(effect_size=0.5, nobs1=None, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided")
    sc, r = post("/api/stats/power", {"test": "t_two", "solve_for": "n", "alpha": 0.05, "power": 0.8, "effect_size": 0.5, "tails": 2})
    check("power t_two n", sc == 200 and close(r["result"], np.ceil(n_ref), 0.51),
          f"api={r.get('result')} ref_ceil={np.ceil(n_ref)}")

# ═══════════════════════════ NONPARAMETRIC ═══════════════════════════
elif AREA == "nonparametric":
    # ---- Mann-Whitney (score ~ group) ----
    w = d[["score", "group"]].dropna()
    gA = w[w.group == "A"].score.values
    gB = w[w.group == "B"].score.values
    U, p_ref = S.mannwhitneyu(gA, gB, alternative="two-sided")
    sc, r = post("/api/stats/mannwhitney", {"session_id": SID, "column": "score", "group_column": "group"})
    API["mw"] = {"status": sc, "keys": list(r.keys()) if sc == 200 else r}
    check("mannwhitney HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        p_api = r.get("p")
        check("mannwhitney p scipy ile yakin", close(p_api, p_ref, 5e-3), f"api={p_api} ref={p_ref:.6f} (normal yakl. farki olabilir)")
        check("mannwhitney anlamli (gercek fark var)", p_api < 0.05, f"p={p_api}")

    # ---- Kruskal-Wallis (score ~ education) ----
    grp = [g.score.dropna().values for _, g in d.groupby("education")]
    H, pkw = S.kruskal(*grp)
    sc, r = post("/api/stats/kruskal", {"session_id": SID, "column": "score", "group_column": "education"})
    check("kruskal HTTP200+p", sc == 200 and close(r.get("H") or r.get("statistic"), H, 1e-2) and close(r.get("p"), pkw, 5e-3),
          f"api={r.get('H') or r.get('statistic')},{r.get('p')} ref={H:.4f},{pkw:.5f}")

    # ---- Jonckheere-Terpstra (score ~ stage ordinal) ----
    sc, r = post("/api/stats/jonckheere_terpstra", {"session_id": SID, "column": "score", "group_column": "stage",
                                                    "order": ["I", "II", "III", "IV"]})
    check("jonckheere HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")

    # ---- ROC (biomarker -> event) ----
    from sklearn.metrics import roc_auc_score
    wr = d[["biomarker", "event"]].dropna()
    auc_ref = roc_auc_score(wr.event, wr.biomarker)
    sc, r = post("/api/stats/roc", {"session_id": SID, "score_column": "biomarker", "outcome_column": "event"})
    check("roc HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        auc_api = r.get("auc") or (r.get("roc") or {}).get("auc")
        expected_flip = bool(auc_ref < 0.5)
        expected_auc = 1 - auc_ref if expected_flip else auc_ref
        check(
            "roc AUC sklearn ile ayni ve auto-flip acik",
            auc_api is not None
            and close(auc_api, expected_auc, 1e-2)
            and r.get("direction_requested") == "auto"
            and r.get("direction_used") == ("lower" if expected_flip else "higher")
            and r.get("direction_flipped") is expected_flip,
            f"api={auc_api} ref={auc_ref:.4f} "
            f"requested={r.get('direction_requested')} used={r.get('direction_used')} "
            f"flipped={r.get('direction_flipped')}",
        )

# ═══════════════════════════ REPEATED ═══════════════════════════
elif AREA == "repeated":
    # ---- paired t-test (pre vs post, gercek fark +3) ----
    w = d[["pre", "post"]].dropna()
    t_ref, p_ref = S.ttest_rel(w.pre, w.post)
    sc, r = post("/api/repeated/paired_ttest", {"session_id": SID, "col1": "pre", "col2": "post"})
    check("paired t HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        t_api = r.get("t")
        p_api = r.get("p")
        check("paired t degeri scipy ile ayni", close(t_api, t_ref, 1e-3), f"api={t_api} ref={t_ref:.4f}")
        check("paired p scipy ile ayni", close(p_api, p_ref, 1e-6), f"api={p_api} ref={p_ref:.3e}")
        check("paired anlamli (gercek +3 fark)", p_api < 0.05, f"p={p_api}")
        REF["paired"] = {"t": t_ref, "p": p_ref, "mean_diff": (w.pre - w.post).mean(), "n": len(w)}
        API["paired"] = {"t": t_api, "p": p_api, "n": r.get("n"), "mean_diff": r.get("mean_diff") or r.get("mean1_minus_2")}

    # ---- Wilcoxon signed-rank ----
    W, pw = S.wilcoxon(w.pre, w.post)
    sc, r = post("/api/repeated/wilcoxon_signed_rank", {"session_id": SID, "col1": "pre", "col2": "post"})
    check("wilcoxon HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")
    if sc == 200:
        check("wilcoxon p scipy ile yakin", close(r.get("p"), pw, 5e-3), f"api={r.get('p')} ref={pw:.5f}")

    # ---- Friedman (pre, post, score standardize? - ayni olcekte: pre/post/age degil.
    # Uc tekrarli olcum yok; pre/post iki. Simule ucuncu: post2 = post - 1 + noise)
    rng = np.random.default_rng(7)
    d2 = d[["pre", "post"]].copy()
    d2["post2"] = d2["post"] - 1 + rng.normal(0, 5, len(d2))
    store.save("qa_session_fr", d2.dropna())
    wf = d2.dropna()
    Ff, pf = S.friedmanchisquare(wf.pre, wf.post, wf.post2)
    sc, r = post("/api/repeated/friedman", {"session_id": "qa_session_fr", "columns": ["pre", "post", "post2"]})
    check("friedman HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")
    if sc == 200:
        stat_api = r.get("chi2") or r.get("statistic") or r.get("Q")
        check("friedman stat scipy ile yakin", close(stat_api, Ff, 1e-1), f"api={stat_api} ref={Ff:.4f}")
        check("friedman p scipy ile yakin", close(r.get("p"), pf, 5e-3), f"api={r.get('p')} ref={pf:.5f}")

    # ---- RM-ANOVA ----
    wl = wf.reset_index().melt(id_vars="index", value_vars=["pre", "post", "post2"],
                              var_name="timepoint", value_name="value").rename(columns={"index": "subject"})
    store.save("qa_session_rml", wl)
    sc, r = post("/api/repeated/rm_anova", {"session_id": "qa_session_rml", "subject_col": "subject",
                                            "within_col": "timepoint", "value_col": "value"})
    check("rm_anova HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")

# ═══════════════════════════ CATEGORICAL ═══════════════════════════
elif AREA == "categorical":
    # ---- binomial ----
    x = int(d.response.sum())
    n = int(d.response.notna().sum())
    p_ref = S.binomtest(x, n, 0.5).pvalue
    sc, r = post("/api/categorical/binomial", {"session_id": SID, "column": "response", "p": 0.5})
    check("binomial HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")
    if sc == 200:
        check("binomial p scipy ile ayni", close(r.get("p"), p_ref, 1e-4), f"api={r.get('p')} ref={p_ref:.5f}")

    # ---- one proportion ----
    sc, r = post("/api/categorical/one_proportion", {"session_id": SID, "column": "response", "p0": 0.5})
    check("one_proportion HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")

    # ---- two proportions (response ~ group) ----
    ct = pd.crosstab(d.response, d.group)
    x1, x2 = int(ct.iloc[1, 0]), int(ct.iloc[1, 1])
    n1, n2 = int(ct.iloc[:, 0].sum()), int(ct.iloc[:, 1].sum())
    from statsmodels.stats.proportion import proportions_ztest
    z_ref, p_ref = proportions_ztest([x1, x2], [n1, n2])
    sc, r = post("/api/categorical/two_proportions", {"session_id": SID, "column": "response", "group_column": "group"})
    check("two_proportions HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        p_api = r.get("p")
        check("two_proportions p yakin", p_api is not None and close(p_api, p_ref, 5e-3),
              f"api={p_api} ref={p_ref:.5f}")

    # ---- McNemar (response vs hypertension - 2 olcum degil ama format 2x2; paired senaryo: diabetes x hypertension degil...
    # Gercek paired ikili yok; McNemar'i diabetes (t1) ve yapay diabetes2 (t2) ile test et
    rng = np.random.default_rng(11)
    d2 = d[["diabetes"]].dropna().copy()
    d2["diab2"] = np.where(rng.random(len(d2)) < 0.12, 1 - d2.diabetes, d2.diabetes)
    store.save("qa_session_mcn", d2)
    ct_m = pd.crosstab(d2.diabetes, d2.diab2)
    from statsmodels.stats.contingency_tables import mcnemar
    p_ref = mcnemar(ct_m.values, exact=True).pvalue
    sc, r = post("/api/categorical/mcnemar", {"session_id": "qa_session_mcn", "col1": "diabetes", "col2": "diab2"})
    check("mcnemar HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        p_api = r.get("p")
        check("mcnemar p smodels ile yakin", p_api is not None and close(p_api, p_ref, 1e-2),
              f"api={p_api} ref={p_ref:.5f} (exact vs chi2 yaklasimi farki olabilir)")

    # ---- Cochran Q (3 olcum: diabetes, hypertension, response>0.5 degil... uc binary olcum lazim) ----
    d3 = d[["diabetes", "hypertension", "response"]].dropna()
    store.save("qa_session_cq", d3)
    from statsmodels.stats.contingency_tables import cochrans_q
    Q_ref, p_ref = cochrans_q(d3.values).statistic, cochrans_q(d3.values).pvalue
    sc, r = post("/api/categorical/cochran_q", {"session_id": "qa_session_cq", "columns": ["diabetes", "hypertension", "response"]})
    check("cochran_q HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        check("cochran_q p smodels ile yakin", close(r.get("p"), p_ref, 5e-3), f"api={r.get('p')} ref={p_ref:.5f}")

    # ---- Mantel-Haenszel (response x group, strata sex) ----
    d4 = d[["response", "group", "sex"]].dropna()
    store.save("qa_session_mh", d4)
    from statsmodels.stats.contingency_tables import StratifiedTable
    tables = [pd.crosstab(d4[d4.sex == s].response, d4[d4.sex == s].group).values for s in ["Female", "Male"]]
    st = StratifiedTable(tables)
    or_ref = st.oddsratio_pooled
    ci_ref = st.oddsratio_pooled_confint(alpha=0.05)
    p_ref = st.test_null_odds().pvalue
    sc, r = post("/api/categorical/mantel_haenszel", {"session_id": "qa_session_mh", "row_col": "response",
                                                      "col_col": "group", "strata_col": "sex"})
    check("mantel_haenszel HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        or_api = (r.get("common_odds_ratio") or r.get("odds_ratio")
                  or next((e.get("value") for e in r.get("effect_sizes", []) if "odds" in str(e.get("name", "")).lower()), None))
        check("MH pooled OR smodels ile yakin", or_api is not None and close(or_api, or_ref, 5e-2),
              f"api={or_api} ref={or_ref:.4f} es={r.get('effect_sizes')}")
        or_effect = next(
            (
                e
                for e in r.get("effect_sizes", [])
                if e.get("name") == "common_odds_ratio"
            ),
            {},
        )
        check(
            "MH pooled OR CI smodels ile yakin",
            close(or_effect.get("ci_low"), ci_ref[0], 5e-4)
            and close(or_effect.get("ci_high"), ci_ref[1], 5e-4)
            and close(or_effect.get("ci_level"), 0.95, 1e-8),
            f"api={or_effect} ref=[{ci_ref[0]:.5f}, {ci_ref[1]:.5f}]",
        )
        check("MH p smodels ile yakin", close(r.get("p"), p_ref, 5e-3), f"api={r.get('p')} ref={p_ref:.5f}")

    # ---- Cochran-Armitage trend (response x stage) ----
    sc, r = post("/api/categorical/cochran_armitage", {"session_id": SID, "event_col": "response",
                                                       "ordinal_col": "stage", "level_order": ["I", "II", "III", "IV"]})
    check("cochran_armitage HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")

# ═══════════════════════════ CORRELATION ═══════════════════════════
elif AREA == "correlation":
    # ---- Pearson (age, sbp) ----
    w = d[["age", "sbp"]].dropna()
    r_ref, p_ref = S.pearsonr(w.age, w.sbp)
    sc, r = post("/api/stats/correlation_pair", {"session_id": SID, "var1": "age", "var2": "sbp", "method": "pearson"})
    check("pearson age-sbp HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        check("pearson r scipy ile ayni", close(r.get("r"), r_ref, 1e-4), f"api={r.get('r')} ref={r_ref:.5f}")
        check("pearson p scipy ile ayni", close(r.get("p"), p_ref, 1e-6), f"api={r.get('p')} ref={p_ref:.3e}")
        check("pearson pozitif anlamli (tasarim)", r.get("r") > 0 and r.get("p") < 0.05, f"r={r.get('r'):.3f}")

    # ---- Spearman ----
    rs_ref, ps_ref = S.spearmanr(w.age, w.sbp)
    sc, r = post("/api/stats/correlation_pair", {"session_id": SID, "var1": "age", "var2": "sbp", "method": "spearman"})
    check("spearman HTTP200+r/p", sc == 200 and close(r.get("r") or r.get("rho"), rs_ref, 5e-3) and close(r.get("p"), ps_ref, 5e-4),
          f"api={r.get('r') or r.get('rho')},{r.get('p')} ref={rs_ref:.5f},{ps_ref:.5f}")

    # ---- correlation matrix ----
    sc, r = post("/api/stats/correlation_matrix", {"session_id": SID, "variables": ["age", "bmi", "sbp", "glucose"], "method": "pearson"})
    check("corr matrix HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")
    if sc == 200:
        wm = d[["age", "bmi", "sbp", "glucose"]].dropna()
        ref_m = wm.corr().loc["age", "sbp"]
        matrix = r.get("matrix")
        matrix_value = (
            matrix.get("age", {}).get("sbp")
            if isinstance(matrix, dict) and isinstance(matrix.get("age"), dict)
            else None
        )
        check(
            "corr matrix age-sbp pandas ile ayni",
            matrix_value is not None and close(matrix_value, ref_m, 1e-3),
            f"api={matrix_value} ref={ref_m:.4f}",
        )

    # ---- ICC (pre, post) ----
    w2 = d[["pre", "post"]].dropna()
    store.save("qa_session_icc", w2)
    sc, r = post("/api/stats/icc", {"session_id": "qa_session_icc", "rater1_col": "pre", "rater2_col": "post"})
    check("icc HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")

    # ---- Cohen's kappa (diabetes vs hypertension - iki rater simulasyonu) ----
    rng = np.random.default_rng(13)
    d2 = d[["diabetes"]].dropna().copy()
    d2["rater2"] = np.where(rng.random(len(d2)) < 0.15, 1 - d2.diabetes, d2.diabetes)
    store.save("qa_session_kappa", d2)
    from sklearn.metrics import cohen_kappa_score
    k_ref = cohen_kappa_score(d2.diabetes, d2.rater2)
    sc, r = post("/api/stats/cohens_kappa", {"session_id": "qa_session_kappa", "rater1_col": "diabetes", "rater2_col": "rater2"})
    check("cohens_kappa HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        k_api = r.get("kappa")
        check("kappa sklearn ile ayni", k_api is not None and close(k_api, k_ref, 1e-2), f"api={k_api} ref={k_ref:.4f}")

# ═══════════════════════════ AGREEMENT ═══════════════════════════
elif AREA == "agreement":
    w = d[["pre", "post"]].dropna()
    store.save("qa_session_ag", w)
    # ---- Bland-Altman ----
    mean_ref = ((w.pre + w.post) / 2).mean()
    diff = w.pre - w.post
    bias_ref = diff.mean()
    sd_ref = diff.std(ddof=1)
    loa_lo, loa_hi = bias_ref - 1.96 * sd_ref, bias_ref + 1.96 * sd_ref
    sc, r = post("/api/agreement/bland_altman", {"session_id": "qa_session_ag", "method1": "pre", "method2": "post"})
    check("bland_altman HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        bias_api = r.get("bias") or r.get("mean_diff") or (r.get("summary") or {}).get("mean_diff")
        check("BA bias dogru", bias_api is not None and close(bias_api, bias_ref, 1e-3), f"api={bias_api} ref={bias_ref:.4f}")
        _loa = r.get("limits_of_agreement") or {}
        lo_api = r.get("loa_lower") or _loa.get("lower")
        hi_api = r.get("loa_upper") or _loa.get("upper")
        if lo_api is not None:
            check("BA LoA dogru (±1.96 SD)", close(lo_api, loa_lo, 1e-2) and close(hi_api, loa_hi, 1e-2),
                  f"api=({lo_api},{hi_api}) ref=({loa_lo:.3f},{loa_hi:.3f})")
        check("BA bias ~ +3 (tasarim)", abs(bias_api - 3.0) < 1.5, f"bias={bias_api:.3f}")

    # ---- Deming regression ----
    sc, r = post("/api/agreement/deming", {"session_id": "qa_session_ag", "method1": "pre", "method2": "post"})
    check("deming HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        # Deming slope lambda=1 referansi (basit kapali form)
        xm, ym = w.pre.mean(), w.post.mean()
        sxx, syy = w.pre.var(ddof=1), w.post.var(ddof=1)
        sxy = w.pre.cov(w.post)
        slope_ref = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
        slope_api = r.get("slope")
        check("deming slope dogru", slope_api is not None and close(slope_api, slope_ref, 1e-2),
              f"api={slope_api} ref={slope_ref:.4f}")

    # ---- Passing-Bablok ----
    sc, r = post("/api/agreement/passing_bablok", {"session_id": "qa_session_ag", "method1": "pre", "method2": "post"})
    check("passing_bablok HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")

    # ---- CCC (Lin) ----
    from scipy.stats import pearsonr
    rho_c = pearsonr(w.pre, w.post)[0]
    ccc_ref = (2 * rho_c * w.pre.std() * w.post.std()) / (w.pre.var() + w.post.var() + (w.pre.mean() - w.post.mean()) ** 2)
    sc, r = post("/api/agreement/concordance", {"session_id": "qa_session_ag", "method1": "pre", "method2": "post"})
    check("concordance HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        ccc_api = r.get("ccc") or r.get("concordance")
        check("CCC dogru", ccc_api is not None and close(ccc_api, ccc_ref, 1e-2), f"api={ccc_api} ref={ccc_ref:.4f}")

# ═══════════════════════════ RELIABILITY ═══════════════════════════
elif AREA == "reliability":
    # 6 maddelik olcek simule et (alfa bilinen)
    rng = np.random.default_rng(21)
    trait = rng.normal(0, 1, 200)
    items = {f"item{i+1}": np.round(3 + 1.2 * trait + rng.normal(0, 0.8, 200)).clip(1, 5) for i in range(6)}
    di = pd.DataFrame(items)
    store.save("qa_session_rel", di)
    # referans alfa
    k = 6
    alpha_ref = k / (k - 1) * (1 - di.var(ddof=1).sum() / di.sum(axis=1).var(ddof=1))
    sc, r = post("/api/reliability/cronbach", {"session_id": "qa_session_rel", "items": list(items)})
    check("cronbach HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        a_api = r.get("alpha")
        check("cronbach alfa dogru", a_api is not None and close(a_api, alpha_ref, 1e-3), f"api={a_api} ref={alpha_ref:.4f}")
        check("cronbach alfa makul (>0.7, tek boyut tasarim)", a_api > 0.7, f"alpha={a_api}")

# ═══════════════════════════ TABLE1 ═══════════════════════════
elif AREA == "table1":
    vars_all = ["age", "bmi", "sbp", "cholesterol", "glucose", "score", "biomarker",
                "sex", "education", "smoking", "diabetes", "hypertension", "stage", "center", "response"]
    sc, r = post("/api/stats/table1", {"session_id": SID, "variables": vars_all, "group_column": "group"})
    check("table1 HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        rows = r.get("rows") or r.get("table") or []
        API["table1_keys"] = list(r.keys())
        API["n_rows"] = len(rows)
        check("table1 tum degiskenler icin satir var", len(rows) >= len(vars_all),
              f"rows={len(rows)} vars={len(vars_all)}")
        # score satirinin p degeri t-test ile ayni mi
        score_row = next((x for x in rows if (x.get("variable") or x.get("name") or "") == "score"), None)
        if score_row:
            import re as _re
            _raw_p = score_row.get("p") or score_row.get("p_value")
            if isinstance(_raw_p, str):
                _raw_p = _raw_p.strip()
                p_api = 0.0005 if _raw_p.startswith("<") else float(_re.sub(r"[^0-9.eE+-]", "", _raw_p) or "nan")
            else:
                p_api = _raw_p
            w = d[["score", "group"]].dropna()
            gA = w[w.group == "A"].score.values
            gB = w[w.group == "B"].score.values
            eq = S.levene(gA, gB).pvalue >= 0.05
            t_ref, p_ref = S.ttest_ind(gA, gB, equal_var=eq)
            U, p_mw = S.mannwhitneyu(gA, gB, alternative="two-sided")
            REF["table1_score"] = {"p_ttest": p_ref, "p_mw": p_mw, "meanA": gA.mean(), "meanB": gB.mean(),
                                   "sdA": gA.std(ddof=1), "sdB": gB.std(ddof=1)}
            API["table1_score_row"] = score_row
            if p_api is not None:
                is_t = abs(p_api - p_ref) < 5e-3
                is_mw = abs(p_api - p_mw) < 5e-3
                check("table1 score p'si t-test veya MW ile eslesiyor", is_t or is_mw,
                      f"table1_p={p_api} t_p={p_ref:.5f} mw_p={p_mw:.5f} -> {'t-test' if is_t else 'MW' if is_mw else 'HICBIRI!'}")
                check("table1 kullanilan test adi raporlanmis", bool(score_row.get("test") or score_row.get("test_name")),
                      f"test={score_row.get('test') or score_row.get('test_name')}")
        # sex satiri chi2 ile uyumlu mu
        sex_row = next((x for x in rows if (x.get("variable") or x.get("name") or "") == "sex"), None)
        if sex_row:
            import re as _re2
            _raw_p2 = sex_row.get("p") or sex_row.get("p_value")
            if isinstance(_raw_p2, str):
                _raw_p2 = _raw_p2.strip()
                p_api = 0.0005 if _raw_p2.startswith("<") else float(_re2.sub(r"[^0-9.eE+-]", "", _raw_p2) or "nan")
            else:
                p_api = _raw_p2
            chi2, p_ref, _, _ = S.chi2_contingency(pd.crosstab(d.sex, d.group))
            orv, p_f = S.fisher_exact(pd.crosstab(d.sex, d.group).values)
            API["table1_sex_row"] = sex_row
            if p_api is not None:
                ok = abs(p_api - p_ref) < 5e-3 or abs(p_api - p_f) < 5e-3
                check("table1 sex p'si chi2 veya Fisher ile eslesiyor", ok,
                      f"table1_p={p_api} chi2_p={p_ref:.5f} fisher_p={p_f:.5f}")
        # SMD kontrolu
        smd_rows = [x for x in rows if x.get("smd") is not None or x.get("SMD") is not None]
        check("table1 SMD raporlaniyor", len(smd_rows) > 0 or "smd" in str(r).lower(),
              f"smd_rows={len(smd_rows)}")
        # eksiklik sayilari satirlarda gorunuyor mu
        missing_rows = [x for x in rows if x.get("missing_row") is not None]
        expected_missing = sum(bool(d[var].isna().any()) for var in vars_all)
        check(
            "table1 eksiklik bilgisi var",
            len(missing_rows) == expected_missing
            and all(
                row["missing_row"].get("label") == "Missing n (%)"
                for row in missing_rows
            ),
            f"missing_rows={len(missing_rows)} expected={expected_missing}",
        )

    # ---- Table1 grupsuz ----
    sc, r = post("/api/stats/table1", {"session_id": SID, "variables": ["age", "score", "sex", "smoking"]})
    check("table1 grupsuz HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")

    # ---- pub_tables format/export ----
    sc2, r2 = post("/api/stats/table1", {"session_id": SID, "variables": ["age", "score", "sex"], "group_column": "group"})
    if sc2 == 200:
        sc3, r3 = post("/api/pub_tables/format", {"table1_result": r2})
        check("pub_tables format HTTP200", sc3 == 200, f"status={sc3} {str(r3)[:200]}")
        if sc3 == 200:
            formatted_missing = [
                row
                for row in r3.get("rows", [])
                if row.get("label") == "Missing n (%)"
            ]
            check(
                "pub_tables eksiklik satirlarini koruyor",
                len(formatted_missing) > 0,
                f"missing_rows={len(formatted_missing)}",
            )
            rr4 = client.post("/api/pub_tables/export", json={"formatted_table": r3, "format": "xlsx"})
            check("pub_tables export xlsx HTTP200", rr4.status_code == 200, f"status={rr4.status_code}")
            API["export_xlsx_bytes"] = len(rr4.content) if rr4.status_code == 200 else 0
            check("pub_tables xlsx icerik boyutu >1KB", len(rr4.content) > 1000, f"bytes={len(rr4.content)}")
            rr5 = client.post("/api/pub_tables/export", json={"formatted_table": r3, "format": "docx"})
            check("pub_tables export docx HTTP200", rr5.status_code == 200, f"status={rr5.status_code}")

    # ---- weighted descriptive ----
    dw = d[["age", "bmi"]].dropna().copy()
    dw["w"] = np.random.default_rng(3).uniform(0.5, 2.0, len(dw))
    store.save("qa_session_w", dw)
    sc, r = post("/api/stats/weighted_descriptive", {"session_id": "qa_session_w", "value_cols": ["age"], "weight_col": "w"})
    check("weighted_descriptive HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")
    if sc == 200:
        wmean_ref = np.average(dw.age, weights=dw.w)
        _res0 = (r.get("results") or [{}])[0]
        wm_api = r.get("weighted_mean") or r.get("mean") or _res0.get("w_mean") or _res0.get("mean")
        check("weighted mean dogru", wm_api is not None and close(wm_api, wmean_ref, 1e-2),
              f"api={wm_api} ref={wmean_ref:.4f}")

# ═══════════════════════════ SURVIVAL ═══════════════════════════
elif AREA == "survival":
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test

    ds = d.copy()
    ds["grpB"] = (ds.group == "B").astype(int)
    store.save("qa_session_surv", ds)
    SID = "qa_session_surv"
    ws = ds[["time", "event", "group", "age"]].dropna().copy()
    ws["grpB"] = (ws.group == "B").astype(int)

    # ---- 1. KM + log-rank (group) ----
    mA = ws.group == "A"; mB = ws.group == "B"
    lr = logrank_test(ws.time[mA], ws.time[mB], ws.event[mA], ws.event[mB])
    kmA, kmB = KaplanMeierFitter().fit(ws.time[mA], ws.event[mA]), KaplanMeierFitter().fit(ws.time[mB], ws.event[mB])
    medA, medB = kmA.median_survival_time_, kmB.median_survival_time_
    REF["km"] = {"logrank_chi2": float(lr.test_statistic), "logrank_p": float(lr.p_value),
                 "median_A": float(medA), "median_B": float(medB)}
    sc, r = post("/api/models/survival/km", {"session_id": SID, "duration_col": "time", "event_col": "event",
                                             "group_col": "group"})
    check("km HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        lr_api = r.get("logrank") or {}
        p_api = lr_api.get("p") or lr_api.get("p_value")
        stat_api = lr_api.get("chi2") or lr_api.get("statistic") or lr_api.get("test_statistic")
        check("km logrank p lifelines ile ayni", p_api is not None and close(p_api, lr.p_value, 5e-3),
              f"api={p_api} ref={lr.p_value:.5f} keys={list(lr_api.keys())}")
        check("km logrank stat lifelines ile ayni", stat_api is not None and close(stat_api, lr.test_statistic, 5e-2),
              f"api={stat_api} ref={lr.test_statistic:.4f}")
        check("km logrank anlamli (HR=1.8 tasarim)", p_api < 0.05, f"p={p_api}")
        grps = r.get("groups") or []
        med_api = {g.get("label") or g.get("group"): g.get("median_survival") for g in grps} if grps else {}
        if med_api:
            ok_med = (med_api.get("A") is not None and med_api.get("B") is not None
                      and abs((med_api.get("A") or 0) - medA) < 2 and abs((med_api.get("B") or 0) - medB) < 2)
            check("km median survival A/B dogru", ok_med,
                  f"api={med_api} ref A={medA:.2f} B={medB:.2f}")

    # ---- 2. Cox PH (group + age, gercek HR=1.8) ----
    cph = CoxPHFitter().fit(ws[["time", "event", "grpB", "age"]], "time", "event")
    hr_ref = float(np.exp(cph.params_["grpB"])); p_ref = float(cph.summary.loc["grpB", "p"])
    ci_ref = (float(np.exp(cph.confidence_intervals_.loc["grpB"].iloc[0])),
              float(np.exp(cph.confidence_intervals_.loc["grpB"].iloc[1])))
    REF["cox"] = {"HR_grpB": hr_ref, "p": p_ref, "beta_grpB": float(cph.params_["grpB"]),
                  "beta_age": float(cph.params_["age"])}
    sc, r = post("/api/models/survival/cox", {"session_id": SID, "duration_col": "time", "event_col": "event",
                                              "predictors": ["grpB", "age"]})
    check("cox HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        coefs = r.get("coefficients") or r.get("coef_table") or {}
        # hem {var: {hr, p}} hem liste formatina dayanikli oku
        def _coef(name):
            if isinstance(coefs, dict):
                c = coefs.get(name) or {}
                return c.get("hr") or c.get("HR"), c.get("p") or c.get("p_value")
            for c in (coefs if isinstance(coefs, list) else []):
                if c.get("variable") == name or c.get("name") == name:
                    return c.get("hr") or c.get("HR"), c.get("p") or c.get("p_value")
            return None, None
        hr_api, p_api = _coef("grpB")
        check("cox HR(grpB) lifelines ile ayni", hr_api is not None and close(hr_api, hr_ref, 5e-2),
              f"api={hr_api} ref={hr_ref:.4f}")
        check("cox p(grpB) lifelines ile ayni", p_api is not None and close(p_api, p_ref, 5e-3),
              f"api={p_api} ref={p_ref:.5f}")
        check("cox HR>1 ve anlamli (gercek HR=1.8)", hr_api is not None and hr_api > 1 and p_api < 0.05,
              f"HR={hr_api} p={p_api}")
        check("cox HR tahmini gercek 1.8'e yakin (CI icinde)", hr_api is not None and abs(hr_api - 1.8) < 0.9,
              f"HR={hr_api} true=1.8")
        API["cox_keys"] = list(r.keys())

    # ---- 3. Cox uni_multi ----
    sc, r = post("/api/models/survival/cox_uni_multi", {"session_id": SID, "duration_col": "time", "event_col": "event",
                                                        "predictors": ["grpB", "age", "bmi"]})
    check("cox_uni_multi HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")

    # ---- 4. RMST (tau=10) ----
    def _rmst(t, e, tau):
        km = KaplanMeierFitter().fit(t, e)
        ts = km.survival_function_.index.values
        sv = km.survival_function_.values.ravel()
        ts2 = np.append(ts[ts < tau], tau); sv2 = np.append(sv[ts < tau], km.predict(tau))
        return float(np.trapz(sv2, ts2))
    rmstA, rmstB = _rmst(ws.time[mA], ws.event[mA], 10), _rmst(ws.time[mB], ws.event[mB], 10)
    REF["rmst"] = {"A": rmstA, "B": rmstB, "diff_B_minus_A": rmstB - rmstA}
    sc, r = post("/api/survival_advanced/rmst", {"session_id": SID, "duration_col": "time", "event_col": "event",
                                                 "tau": 10, "group_col": "group"})
    check("rmst HTTP200", sc == 200, f"status={sc} {str(r)[:300]}")
    if sc == 200:
        rs = r.get("groups") or r.get("rmst_by_group") or {}
        API["rmst_keys"] = list(r.keys())
        ok = False
        if isinstance(rs, dict) and rs:
            vals = {k: (v.get("rmst") if isinstance(v, dict) else v) for k, v in rs.items()}
            ok = any(abs((vals.get(kA) or -99) - rmstA) < 0.5 for kA in vals)
        check("rmst grup degerleri lifelines ile yakin", ok,
              f"api_keys={list(r.keys())} ref A={rmstA:.3f} B={rmstB:.3f}")

    # ---- 5. E-value (HR=1.8) ----
    ev_ref = 1.8 + np.sqrt(1.8 * (1.8 - 1))
    sc, r = post("/api/survival_advanced/evalue", {"estimate": 1.8, "ci_low": ci_ref[0], "ci_high": ci_ref[1],
                                                   "measure_type": "HR"})
    check("evalue HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")
    if sc == 200:
        ev_api = r.get("evalue_point") or r.get("evalue") or r.get("e_value")
        check("evalue formulu dogru", ev_api is not None and close(ev_api, ev_ref, 5e-3),
              f"api={ev_api} ref={ev_ref:.4f}")

    # ---- 6. Fine-Gray (competing risk: event 1 olay, event=0 iken baska olay yok → risk=1) ----
    ws2 = ws.copy(); ws2["cr"] = ws2.event  # tek olay tipi; CIF KM'ye esdeger olmali
    store.save("qa_session_fg", ws2)
    sc, r = post("/api/survival_advanced/fine_gray", {"session_id": "qa_session_fg", "duration_col": "time",
                                                      "event_col": "cr", "predictors": ["grpB"]})
    check("fine_gray HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")

    # ---- 7. Landmark ----
    sc, r = post("/api/survival_advanced/landmark", {"session_id": SID, "duration_col": "time", "event_col": "event",
                                                     "landmark_time": 5, "predictors": ["grpB"]})
    check("landmark HTTP200", sc == 200, f"status={sc} {str(r)[:200]}")


# ═══════════════════════════ CIKTI ═══════════════════════════
out = {
    "area": AREA,
    "n_pass": sum(1 for f in FINDINGS if f["status"] == "PASS"),
    "n_fail": sum(1 for f in FINDINGS if f["status"] == "FAIL"),
    "findings": FINDINGS,
    "reference": {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                      for kk, vv in v.items()} for k, v in REF.items()},
    "api_summary": API,
}
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
with open(os.path.join(BASE, "results", f"{AREA}.json"), "w") as f:
    json.dump(out, f, indent=2, default=str, ensure_ascii=False)

for fd in FINDINGS:
    mark = "OK " if fd["status"] == "PASS" else "XX "
    print(f"[{fd['status']}] {fd['test']}: {fd['detail'][:180]}")
print(f"\n{out['n_pass']} PASS / {out['n_fail']} FAIL")
if out["n_fail"]:
    raise SystemExit(1)
