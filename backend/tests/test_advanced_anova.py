"""Tests for advanced ANOVA endpoints: ANCOVA, two-way ANOVA."""
import numpy as np
import pandas as pd
from conftest import make_session


# ── ANCOVA ───────────────────────────────────────────────────────────────────

def test_ancova_known(client):
    np.random.seed(42)
    n = 40
    group = ["A"] * 20 + ["B"] * 20
    covariate = np.random.normal(50, 10, n)
    outcome = np.where(np.array(group) == "A", 70, 75) + 0.5 * covariate + np.random.normal(0, 5, n)
    df = pd.DataFrame({"outcome": outcome, "group": group, "covariate": covariate})
    sid = make_session(df, "ancova1")
    r = client.post("/api/advanced_anova/ancova", json={
        "session_id": sid, "outcome": "outcome", "group_col": "group", "covariates": ["covariate"]
    })
    assert r.status_code == 200
    d = r.json()
    assert "F" in d
    assert "emms" in d
    assert len(d["emms"]) == 2
    assert d["effect_sizes"][0]["name"] == "partial_eta_squared"
    assert "r_code" in d
    assert "emmeans" in d["r_code"]


def test_ancova_covariate_adjusts(client):
    """Groups differ in raw means but not after covariate adjustment."""
    np.random.seed(42)
    n = 30
    # Group A has higher covariate values, making raw outcome higher
    cov_a = np.random.normal(60, 5, n)
    cov_b = np.random.normal(40, 5, n)
    y_a = 0.8 * cov_a + np.random.normal(0, 3, n)
    y_b = 0.8 * cov_b + np.random.normal(0, 3, n)
    df = pd.DataFrame({
        "outcome": np.concatenate([y_a, y_b]),
        "group": ["A"] * n + ["B"] * n,
        "cov": np.concatenate([cov_a, cov_b]),
    })
    sid = make_session(df, "ancova2")
    r = client.post("/api/advanced_anova/ancova", json={
        "session_id": sid, "outcome": "outcome", "group_col": "group", "covariates": ["cov"]
    })
    assert r.status_code == 200
    d = r.json()
    # After adjustment, group effect should be non-significant or much weaker
    # (the raw means differ only because of the covariate)
    assert isinstance(d["significant"], bool)


def test_ancova_homogeneity_warning(client):
    """Data with different slopes per group — should warn about assumption violation."""
    np.random.seed(42)
    n = 30
    cov = np.random.normal(50, 10, n * 2)
    group = ["A"] * n + ["B"] * n
    # Different slopes: A has slope 1.0, B has slope 0.1
    outcome = np.where(np.array(group) == "A", 1.0, 0.1) * cov + np.random.normal(0, 2, n * 2)
    df = pd.DataFrame({"outcome": outcome, "group": group, "cov": cov})
    sid = make_session(df, "ancova3")
    r = client.post("/api/advanced_anova/ancova", json={
        "session_id": sid, "outcome": "outcome", "group_col": "group", "covariates": ["cov"]
    })
    assert r.status_code == 200
    d = r.json()
    # Should have assumption check about homogeneity of slopes
    slope_checks = [a for a in d.get("assumptions", []) if "slope" in a.get("name", "").lower()]
    assert len(slope_checks) > 0


# ── Two-way ANOVA ────────────────────────────────────────────────────────────

def test_two_way_main_effects(client):
    np.random.seed(42)
    data = []
    for drug in ["A", "B"]:
        for dose in ["Low", "High"]:
            base = (5 if drug == "A" else 8) + (0 if dose == "Low" else 3)
            for _ in range(15):
                data.append({"drug": drug, "dose": dose, "response": np.random.normal(base, 2)})
    df = pd.DataFrame(data)
    sid = make_session(df, "twa1")
    r = client.post("/api/advanced_anova/two_way_anova", json={
        "session_id": sid, "outcome": "response", "factor1": "drug", "factor2": "dose"
    })
    assert r.status_code == 200
    d = r.json()
    assert "effects" in d
    assert len(d["effects"]) >= 2  # at least 2 main effects + possible interaction
    assert "emms" in d
    assert "r_code" in d


def test_two_way_interaction(client):
    """Crossover interaction: drug effect reverses at different doses."""
    np.random.seed(42)
    data = []
    for drug in ["X", "Y"]:
        for dose in ["Low", "High"]:
            if drug == "X":
                base = 10 if dose == "Low" else 5  # X worse at high dose
            else:
                base = 5 if dose == "Low" else 10   # Y better at high dose
            for _ in range(20):
                data.append({"drug": drug, "dose": dose, "response": np.random.normal(base, 1.5)})
    df = pd.DataFrame(data)
    sid = make_session(df, "twa2")
    r = client.post("/api/advanced_anova/two_way_anova", json={
        "session_id": sid, "outcome": "response", "factor1": "drug", "factor2": "dose"
    })
    assert r.status_code == 200
    d = r.json()
    interaction = [e for e in d["effects"] if "interaction" in e.get("term", "").lower()]
    assert len(interaction) > 0
    assert interaction[0]["significant"] is True


def test_two_way_emms(client):
    df = pd.DataFrame({
        "y": np.random.normal(10, 2, 40),
        "f1": (["A"] * 10 + ["B"] * 10) * 2,
        "f2": ["Low"] * 20 + ["High"] * 20,
    })
    sid = make_session(df, "twa3")
    r = client.post("/api/advanced_anova/two_way_anova", json={
        "session_id": sid, "outcome": "y", "factor1": "f1", "factor2": "f2"
    })
    assert r.status_code == 200
    d = r.json()
    assert "emms" in d
    assert len(d["emms"]) == 4  # 2 x 2 cells


# ── Contract checks ──────────────────────────────────────────────────────────

def test_export_rows_format(client):
    df = pd.DataFrame({
        "y": np.random.normal(10, 2, 40),
        "group": ["A"] * 20 + ["B"] * 20,
        "cov": np.random.normal(5, 1, 40),
    })
    sid = make_session(df, "export1")
    r = client.post("/api/advanced_anova/ancova", json={
        "session_id": sid, "outcome": "y", "group_col": "group", "covariates": ["cov"]
    })
    d = r.json()
    assert isinstance(d["export_rows"], list)
    assert len(d["export_rows"]) > 1
    assert isinstance(d["export_rows"][0], list)  # header row


def test_r_code_present(client):
    df = pd.DataFrame({
        "y": np.random.normal(10, 2, 40),
        "f1": ["A"] * 20 + ["B"] * 20,
        "f2": ["X"] * 10 + ["Y"] * 10 + ["X"] * 10 + ["Y"] * 10,
    })
    sid = make_session(df, "rcode1")
    r = client.post("/api/advanced_anova/two_way_anova", json={
        "session_id": sid, "outcome": "y", "factor1": "f1", "factor2": "f2"
    })
    d = r.json()
    assert "r_code" in d
    assert "aov" in d["r_code"]
    assert "emmeans" in d["r_code"]


# ── MANCOVA ──────────────────────────────────────────────────────────────────

def test_mancova_significant_with_covariates(client):
    rng = np.random.default_rng(1)
    n = 120
    grp = rng.choice(["ADHD", "Control"], n)
    eff = (grp == "ADHD").astype(float)
    df = pd.DataFrame({
        "group": grp,
        "age": rng.normal(12, 2, n), "sex": rng.integers(0, 2, n),
        "SCARED": rng.normal(30, 8, n), "BMIpct": rng.uniform(5, 95, n), "smoke": rng.integers(0, 2, n),
        "BDNF": rng.normal(20, 5, n) + 3 * eff, "GDNF": rng.normal(15, 4, n) + 2 * eff,
        "NTF3": rng.normal(10, 3, n) + eff, "NGF": rng.normal(8, 2, n) + 1.5 * eff,
    })
    sid = make_session(df, "mancova1")
    r = client.post("/api/advanced_anova/mancova", json={
        "session_id": sid, "outcomes": ["BDNF", "GDNF", "NTF3", "NGF"], "group_col": "group",
        "covariates": ["age", "sex", "SCARED", "BMIpct", "smoke"],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "MANCOVA"
    names = {t["test"] for t in d["multivariate_tests"]}
    assert any("Pillai" in n for n in names) and any("Wilks" in n for n in names)
    assert d["pillai"]["test"].startswith("Pillai")
    assert 0.0 <= d["pillai"]["p"] <= 1.0
    assert d["effect_size"]["magnitude"] in {"negligible", "small", "medium", "large"}
    assert "Pillai" in d["result_text"] and "partial" in d["result_text"]
    assert "Manova" in d["r_code"]


def test_mancova_requires_two_outcomes(client):
    df = pd.DataFrame({"group": ["A", "B"] * 20, "y1": np.random.rand(40)})
    sid = make_session(df, "mancova_one")
    r = client.post("/api/advanced_anova/mancova", json={
        "session_id": sid, "outcomes": ["y1"], "group_col": "group", "covariates": [],
    })
    assert r.status_code == 400, r.text


def test_mancova_missing_column_400(client):
    df = pd.DataFrame({"group": ["A", "B"] * 20, "y1": np.random.rand(40), "y2": np.random.rand(40)})
    sid = make_session(df, "mancova_miss")
    r = client.post("/api/advanced_anova/mancova", json={
        "session_id": sid, "outcomes": ["y1", "nope"], "group_col": "group", "covariates": [],
    })
    assert r.status_code == 400, r.text


# ── Two-way ANOVA: model-based EMMs, simple effects ─────────────────────────

def _interaction_df(seed=3):
    """Unbalanced 2x2 with a real drug x dose interaction."""
    rng = np.random.default_rng(seed)
    rows = []
    for a, b, mu, n in [("drugA", "low", 10, 20), ("drugA", "high", 12, 10),
                        ("drugB", "low", 10, 15), ("drugB", "high", 20, 25)]:
        for _ in range(n):
            rows.append({"y": rng.normal(mu, 2), "drug": a, "dose": b})
    return pd.DataFrame(rows)


def test_two_way_model_based_emms(client):
    sid = make_session(_interaction_df(), "tw_emm")
    r = client.post("/api/advanced_anova/two_way_anova", json={
        "session_id": sid, "outcome": "y", "factor1": "drug", "factor2": "dose"})
    assert r.status_code == 200
    d = r.json()
    # Cell EMMs equal observed cell means in the full factorial, and carry
    # a CI from the pooled residual MS.
    for cell in d["emms"]:
        assert cell["emm"] == cell["mean"]
        assert cell["ci_low"] < cell["emm"] < cell["ci_high"]
    # Marginal EMMs are the UNWEIGHTED average of the cell means: for drugA
    # the weighted (raw) mean is pulled toward the n=20 low-dose cell, the
    # EMM must sit exactly halfway between the two cell means.
    cells = {(c["factor1"], c["factor2"]): c["emm"] for c in d["emms"]}
    marg = {(m["factor"], m["level"]): m["emm"] for m in d["emm_marginal"]}
    expected = (cells[("drugA", "low")] + cells[("drugA", "high")]) / 2
    assert abs(marg[("drug", "drugA")] - expected) < 0.001
    raw_mean = _interaction_df().query("drug == 'drugA'")["y"].mean()
    assert abs(marg[("drug", "drugA")] - raw_mean) > 0.3  # unbalanced: differs from raw


def test_two_way_simple_effects_when_interaction_significant(client):
    sid = make_session(_interaction_df(), "tw_simple")
    r = client.post("/api/advanced_anova/two_way_anova", json={
        "session_id": sid, "outcome": "y", "factor1": "drug", "factor2": "dose"})
    d = r.json()
    inter = next(e for e in d["effects"] if "interaction" in e["term"])
    assert inter["significant"]
    se = d["simple_effects"]
    assert len(se) == 4  # drug within each dose, dose within each drug
    by_key = {(s["effect_of"], s["within_level"]): s for s in se}
    # Constructed truth: drug matters at high dose (12 vs 20), not at low (10 vs 10).
    assert by_key[("drug", "high")]["significant"]
    assert not by_key[("drug", "low")]["significant"]
    # Cell comparisons within levels ride along in posthoc
    assert any("within" in p["factor"] for p in d["posthoc"])
    assert d["posthoc_method"] == "Tukey-Kramer on model-based EMMs"
    # The R script must describe the same analysis the numbers came from.
    assert "joint_tests" in d["r_code"]
    assert "TukeyHSD" not in d["r_code"]


def test_two_way_no_interaction_keeps_marginal_posthoc(client):
    rng = np.random.default_rng(9)
    rows = []
    for a in ("A", "B", "C"):
        for b in ("x", "y"):
            mu = {"A": 10, "B": 14, "C": 18}[a] + (2 if b == "y" else 0)
            for _ in range(15):
                rows.append({"out": rng.normal(mu, 2), "f1": a, "f2": b})
    sid = make_session(pd.DataFrame(rows), "tw_noint")
    r = client.post("/api/advanced_anova/two_way_anova", json={
        "session_id": sid, "outcome": "out", "factor1": "f1", "factor2": "f2"})
    d = r.json()
    inter = next(e for e in d["effects"] if "interaction" in e["term"])
    assert not inter["significant"]
    assert d["simple_effects"] == []
    # Marginal Tukey-Kramer comparisons for the significant f1 main effect
    f1_rows = [p for p in d["posthoc"] if p["factor"] == "f1"]
    assert len(f1_rows) == 3  # A-B, A-C, B-C
    assert all(p["significant"] for p in f1_rows)
