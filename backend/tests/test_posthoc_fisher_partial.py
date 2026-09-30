"""One-way ANOVA post-hoc options (Dunnett, forced), r×c Fisher-Freeman-
Halton, and the partial-correlation endpoint."""
import numpy as np
import pandas as pd
from scipy import stats as sp
from conftest import make_session


def _arm_df(seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sbp": np.concatenate([rng.normal(120, 10, 30), rng.normal(128, 10, 30),
                               rng.normal(135, 10, 30), rng.normal(122, 10, 30)]),
        "arm": ["placebo"] * 30 + ["low"] * 30 + ["high"] * 30 + ["mid"] * 30,
    })


# ── ANOVA post-hoc options ───────────────────────────────────────────────────


def test_anova_dunnett_vs_control(client):
    sid = make_session(_arm_df(), "ph1")
    r = client.post("/api/stats/anova", json={
        "session_id": sid, "column": "sbp", "group_column": "arm",
        "posthoc": "dunnett", "control_group": "placebo"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["posthoc_method"] == "Dunnett (vs control 'placebo')"
    rows = d["posthoc"]
    assert len(rows) == 3                      # every arm vs placebo, no arm-vs-arm
    assert all(p["group2"] == "placebo" for p in rows)
    assert all("ci_low" in p and "ci_high" in p for p in rows)
    # Cross-check one p against scipy directly.
    df = _arm_df()
    by = {k: g["sbp"].values for k, g in df.groupby("arm")}
    ref = sp.dunnett(*[by[k] for k in sorted(by) if k != "placebo"], control=by["placebo"])
    got = sorted(p["p_adj"] for p in rows)
    want = sorted(float(p) for p in ref.pvalue)
    # scipy evaluates the Dunnett distribution by stochastic multivariate-t
    # quadrature, so two runs differ in the fourth decimal.
    for g, w in zip(got, want):
        assert abs(g - w) < 2e-3


def test_anova_dunnett_bad_control_422(client):
    sid = make_session(_arm_df(), "ph2")
    r = client.post("/api/stats/anova", json={
        "session_id": sid, "column": "sbp", "group_column": "arm",
        "posthoc": "dunnett", "control_group": "nope"})
    assert r.status_code == 422
    assert "nope" in r.text


def test_anova_unknown_posthoc_422(client):
    sid = make_session(_arm_df(), "ph3")
    r = client.post("/api/stats/anova", json={
        "session_id": sid, "column": "sbp", "group_column": "arm", "posthoc": "weird"})
    assert r.status_code == 422


def test_anova_force_posthoc_on_null_data(client):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(0, 1, 90),
        "g": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
    })
    sid = make_session(df, "ph4")
    base = client.post("/api/stats/anova", json={
        "session_id": sid, "column": "y", "group_column": "g"}).json()
    assert not base["significant"]
    assert base["posthoc"] == []               # default gate unchanged
    forced = client.post("/api/stats/anova", json={
        "session_id": sid, "column": "y", "group_column": "g",
        "force_posthoc": True}).json()
    assert len(forced["posthoc"]) == 3
    assert "planned contrasts" in forced["posthoc_note"]


def test_anova_posthoc_none(client):
    sid = make_session(_arm_df(), "ph5")
    d = client.post("/api/stats/anova", json={
        "session_id": sid, "column": "sbp", "group_column": "arm",
        "posthoc": "none"}).json()
    assert d["significant"]
    assert d["posthoc"] == []
    assert d["posthoc_method"] is None


# ── Fisher r×c ────────────────────────────────────────────────────────────────


def test_fisher_rxc_freeman_halton(client):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "stage": rng.choice(["I", "II", "III"], 90),
        "center": rng.choice(["A", "B", "C"], 90),
    })
    sid = make_session(df, "ffh1")
    r = client.post("/api/stats/fisher", json={
        "session_id": sid, "row_column": "stage", "col_column": "center"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "Freeman-Halton" in d["test"]
    assert "5000" in d["test"]                 # resample count is named
    assert 0 < d["p"] <= 1
    assert "odds_ratio" not in d
    assert d["effect_sizes"][0]["name"] == "cramers_v"
    assert "simulate.p.value" in d["r_code"]
    assert "Freeman-Halton" in d["result_text"]


def test_fisher_2x2_unchanged(client):
    df = pd.DataFrame({
        "exposed": ["yes"] * 12 + ["no"] * 15 + ["yes"] * 3 + ["no"] * 14,
        "event":   ["yes"] * 12 + ["yes"] * 15 + ["no"] * 3 + ["no"] * 14,
    })
    sid = make_session(df, "ffh2")
    d = client.post("/api/stats/fisher", json={
        "session_id": sid, "row_column": "exposed", "col_column": "event"}).json()
    assert d["test"] == "Fisher's exact test"
    assert "odds_ratio" in d


def test_fisher_single_level_column_400(client):
    df = pd.DataFrame({"a": ["x"] * 20, "b": ["p"] * 10 + ["q"] * 10})
    sid = make_session(df, "ffh3")
    r = client.post("/api/stats/fisher", json={
        "session_id": sid, "row_column": "a", "col_column": "b"})
    assert r.status_code == 400


# ── Partial correlation ──────────────────────────────────────────────────────


def _confounded_df(seed=11, n=200):
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    return pd.DataFrame({
        "x": 0.5 * age + rng.normal(0, 5, n),
        "y": 0.5 * age + rng.normal(0, 5, n),
        "age": age,
    })


def test_partial_correlation_removes_confounded_association(client):
    df = _confounded_df()
    sid = make_session(df, "pc1")
    raw = client.post("/api/stats/correlation_pair", json={
        "session_id": sid, "var1": "x", "var2": "y", "method": "pearson"}).json()
    part = client.post("/api/stats/partial_correlation", json={
        "session_id": sid, "var1": "x", "var2": "y", "controls": ["age"]}).json()
    assert raw["r"] > 0.4                      # confounded association
    assert abs(part["r"]) < 0.15               # gone after adjustment
    assert part["df"] == len(df) - 3           # n - 2 - k
    # Ground truth: correlation of the OLS residuals.
    Z = np.column_stack([np.ones(len(df)), df["age"].values])
    rx = df["x"].values - Z @ np.linalg.lstsq(Z, df["x"].values, rcond=None)[0]
    ry = df["y"].values - Z @ np.linalg.lstsq(Z, df["y"].values, rcond=None)[0]
    assert abs(part["r"] - float(np.corrcoef(rx, ry)[0, 1])) < 1e-3
    assert part["ci_low"] < part["r"] < part["ci_high"]
    assert "pcor.test" in part["r_code"]


def test_partial_correlation_spearman_runs(client):
    sid = make_session(_confounded_df(seed=4), "pc2")
    r = client.post("/api/stats/partial_correlation", json={
        "session_id": sid, "var1": "x", "var2": "y", "controls": ["age"],
        "method": "spearman"})
    assert r.status_code == 200
    assert r.json()["method"] == "spearman"


def test_partial_correlation_rejects_overlap_and_no_controls(client):
    sid = make_session(_confounded_df(seed=5), "pc3")
    r = client.post("/api/stats/partial_correlation", json={
        "session_id": sid, "var1": "x", "var2": "y", "controls": ["x"]})
    assert r.status_code == 422
    r = client.post("/api/stats/partial_correlation", json={
        "session_id": sid, "var1": "x", "var2": "y", "controls": []})
    assert r.status_code == 422
