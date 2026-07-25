"""Cross-validate uSTAT's headline statistics against reference implementations.

A user found the Welch-df defect by comparing uSTAT against R: the reported
`df` did not belong to the test that produced `t` and `p`. These tests guard
that whole class of defect for the statistics a manuscript actually cites —
each reported number is checked against scipy / statsmodels / lifelines, and
where a test reports a statistic, its df and its p, the three are also checked
for internal consistency (p recomputed from the reported statistic and df).
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from scipy import stats as sp

SEED = 4242
N = 240


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(SEED)
    grp = rng.integers(0, 2, N)
    x = rng.normal(50, 10, N) + 4 * grp
    y = rng.normal(100, 20, N) + 0.5 * x
    out = rng.binomial(1, 1 / (1 + np.exp(-(-3 + 0.05 * x))))
    cat3 = rng.choice(["A", "B", "C"], N)
    dur = rng.exponential(5, N)
    cens = rng.uniform(1, 12, N)
    return pd.DataFrame({
        "x": np.round(x, 6),
        "y": np.round(y, 6),
        "grp": grp,
        "out": out,
        "cat3": cat3,
        "time": np.round(np.minimum(dur, cens), 6),
        "event": (dur <= cens).astype(int),
    })


@pytest.fixture(scope="module")
def sid(frame):
    return make_session(frame, "crossval_session")


def _post(client, path, **payload):
    r = client.post(path, json=payload)
    assert r.status_code == 200, r.text
    return r.json()


# ── t-test ───────────────────────────────────────────────────────────────────


def test_ttest_matches_scipy_and_is_internally_consistent(client, sid, frame):
    d = _post(client, "/api/stats/ttest", session_id=sid, column="x", group_column="grp")
    g0 = frame.loc[frame.grp == 0, "x"].to_numpy()
    g1 = frame.loc[frame.grp == 1, "x"].to_numpy()
    welch = d["df_method"] == "welch_satterthwaite"

    t_ref, p_ref = sp.ttest_ind(g0, g1, equal_var=not welch)
    assert d["t"] == pytest.approx(t_ref, rel=1e-12)
    assert d["p"] == pytest.approx(p_ref, rel=1e-12)

    if welch:
        a, b = g0.var(ddof=1) / len(g0), g1.var(ddof=1) / len(g1)
        df_ref = (a + b) ** 2 / ((a ** 2) / (len(g0) - 1) + (b ** 2) / (len(g1) - 1))
    else:
        df_ref = len(g0) + len(g1) - 2
    assert d["df"] == pytest.approx(df_ref, rel=1e-12)

    # The defect that started this: p must follow from the reported t and df.
    assert d["p"] == pytest.approx(2 * sp.t.sf(abs(d["t"]), d["df"]), rel=1e-9)


# ── Non-parametric ───────────────────────────────────────────────────────────


def test_mann_whitney_matches_scipy(client, sid, frame):
    d = _post(client, "/api/stats/mannwhitney", session_id=sid, column="x", group_column="grp")
    g0 = frame.loc[frame.grp == 0, "x"].to_numpy()
    g1 = frame.loc[frame.grp == 1, "x"].to_numpy()
    u_ref, p_ref = sp.mannwhitneyu(g0, g1, alternative="two-sided")
    assert d["p"] == pytest.approx(p_ref, rel=1e-12)
    got_u = d.get("u", d.get("U"))
    if got_u is not None:
        assert got_u == pytest.approx(u_ref)


def test_kruskal_matches_scipy(client, sid, frame):
    d = _post(client, "/api/stats/kruskal", session_id=sid, column="x", group_column="cat3")
    arrs = [frame.loc[frame.cat3 == lv, "x"].to_numpy() for lv in sorted(frame.cat3.unique())]
    _, p_ref = sp.kruskal(*arrs)
    assert d["p"] == pytest.approx(p_ref, rel=1e-12)


# ── ANOVA ────────────────────────────────────────────────────────────────────


def test_anova_matches_scipy_and_dfs_reproduce_p(client, sid, frame):
    d = _post(client, "/api/stats/anova", session_id=sid, column="x", group_column="cat3")
    arrs = [frame.loc[frame.cat3 == lv, "x"].to_numpy() for lv in sorted(frame.cat3.unique())]
    f_ref, p_ref = sp.f_oneway(*arrs)
    assert d["F"] == pytest.approx(f_ref, rel=1e-12)
    assert d["p"] == pytest.approx(p_ref, rel=1e-12)
    assert d["df_between"] == len(arrs) - 1
    assert d["df_within"] == N - len(arrs)
    assert d["p"] == pytest.approx(sp.f.sf(d["F"], d["df_between"], d["df_within"]), rel=1e-9)


# ── Correlation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method,ref", [
    ("pearson", sp.pearsonr),
    ("spearman", sp.spearmanr),
])
def test_correlation_matches_scipy(client, sid, frame, method, ref):
    d = _post(client, "/api/stats/correlation_pair", session_id=sid,
              var1="x", var2="y", method=method)
    r_ref, p_ref = ref(frame.x, frame.y)
    assert d.get("r", d.get("correlation")) == pytest.approx(r_ref, rel=1e-10)
    assert d["p"] == pytest.approx(p_ref, rel=1e-10)


# ── Contingency ──────────────────────────────────────────────────────────────


def test_chisquare_matches_scipy_and_df_reproduces_p(client, sid, frame):
    d = _post(client, "/api/stats/chisquare", session_id=sid,
              row_column="grp", col_column="out")
    table = pd.crosstab(frame.grp, frame.out).to_numpy()
    got = d.get("chi2", d.get("statistic"))
    dof = d.get("df", d.get("dof"))

    # uSTAT applies Yates for 2x2; accept whichever variant it reports, but the
    # reported chi2/df/p must be mutually consistent either way.
    uncorrected = sp.chi2_contingency(table, correction=False)
    corrected = sp.chi2_contingency(table, correction=True)
    matches_either = (
        got == pytest.approx(uncorrected[0], rel=1e-10)
        or got == pytest.approx(corrected[0], rel=1e-10)
    )
    assert matches_either, (
        f"chi2={got} matches neither uncorrected {uncorrected[0]} "
        f"nor Yates-corrected {corrected[0]}"
    )
    assert dof == uncorrected[2]
    assert d["p"] == pytest.approx(sp.chi2.sf(got, dof), rel=1e-9)


def test_fisher_matches_scipy(client, sid, frame):
    d = _post(client, "/api/stats/fisher", session_id=sid,
              row_column="grp", col_column="out")
    table = pd.crosstab(frame.grp, frame.out).to_numpy()
    _, p_ref = sp.fisher_exact(table)
    assert d["p"] == pytest.approx(p_ref, rel=1e-10)


# ── Regression / survival ────────────────────────────────────────────────────


def test_logistic_or_matches_statsmodels(client, sid, frame):
    import statsmodels.api as smapi

    d = _post(client, "/api/models/logistic", session_id=sid,
              outcome="out", predictors=["x"])
    X = smapi.add_constant(frame[["x"]].to_numpy())
    ref = smapi.Logit(frame.out.to_numpy(), X).fit(disp=0)
    row = next(c for c in d["coefficients"] if c.get("variable") == "x")
    assert row.get("or", row.get("odds_ratio")) == pytest.approx(np.exp(ref.params[1]), rel=1e-6)
    assert row["p"] == pytest.approx(ref.pvalues[1], rel=1e-5)


def test_cox_hr_matches_lifelines(client, sid, frame):
    from lifelines import CoxPHFitter

    d = _post(client, "/api/models/survival/cox", session_id=sid,
              duration_col="time", event_col="event", predictors=["x"])
    cph = CoxPHFitter().fit(frame[["time", "event", "x"]], "time", "event")
    row = next(c for c in d["coefficients"] if c.get("variable") == "x")
    assert row.get("hr", row.get("exp_coef")) == pytest.approx(
        float(np.exp(cph.params_["x"])), rel=1e-4)
    assert row["p"] == pytest.approx(float(cph.summary.loc["x", "p"]), rel=1e-4)


def test_logrank_two_group_reports_chi2_df_and_p(client, sid, frame):
    from lifelines.statistics import logrank_test

    d = _post(client, "/api/models/survival/km", session_id=sid,
              duration_col="time", event_col="event", group_col="grp")
    lg = d["logrank"]
    m0, m1 = frame.grp == 0, frame.grp == 1
    ref = logrank_test(frame.time[m0], frame.time[m1], frame.event[m0], frame.event[m1])

    assert lg["chi2"] == pytest.approx(float(ref.test_statistic), rel=1e-10)
    assert lg["df"] == 1
    assert lg["p"] == pytest.approx(float(ref.p_value), rel=1e-10)
    assert lg["p"] == pytest.approx(sp.chi2.sf(lg["chi2"], lg["df"]), rel=1e-9)


def test_logrank_multigroup_reports_correct_df(client, sid, frame):
    from lifelines.statistics import multivariate_logrank_test

    d = _post(client, "/api/models/survival/km", session_id=sid,
              duration_col="time", event_col="event", group_col="cat3")
    lg = d["logrank"]
    ref = multivariate_logrank_test(frame.time, frame.cat3, frame.event)

    assert lg["chi2"] == pytest.approx(float(ref.test_statistic), rel=1e-10)
    assert lg["df"] == frame.cat3.nunique() - 1
    assert lg["p"] == pytest.approx(float(ref.p_value), rel=1e-10)
    assert lg["p"] == pytest.approx(sp.chi2.sf(lg["chi2"], lg["df"]), rel=1e-9)
