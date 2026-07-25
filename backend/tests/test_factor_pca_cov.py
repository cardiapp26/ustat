"""Coverage tests for POST /api/factor/factor_pca (routers/factor.py).

The endpoint had no test reference. Data here is generated from a known
two-factor structure (two orthogonal blocks of three correlated items), so the
correct answer is known in advance: two components above the Kaiser criterion,
each block loading on its own component, and a strongly rejected sphericity
test. Unstructured noise is used as the negative control.

The last three tests are regressions for two 500-level defects that are now
fixed: the read-only `np.diag` view that made `rotation="promax"` always fail,
and the unhandled eigen-decomposition failure on a zero-variance item.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

SEED = 20260725
PREFIX = "/api/factor"

BLOCK_A = ["A1", "A2", "A3"]
BLOCK_B = ["B1", "B2", "B3"]
ITEMS = BLOCK_A + BLOCK_B
NOISE_ITEMS = [f"N{i}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def factor_df():
    """Two orthogonal latent factors, three indicators each, plus noise items.

    Each indicator is `factor + N(0, 0.5)`, so within-block correlation is
    1 / 1.25 = 0.8 and between-block correlation is 0. The population
    correlation matrix therefore has exactly two eigenvalues of 1 + 2(0.8) =
    2.6 and four of 0.2 — i.e. exactly two components pass Kaiser's rule.
    """
    rng = np.random.default_rng(SEED)
    n = 220
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    data = {}
    for name in BLOCK_A:
        data[name] = f1 + rng.normal(0, 0.5, n)
    for name in BLOCK_B:
        data[name] = f2 + rng.normal(0, 0.5, n)
    for name in NOISE_ITEMS:
        data[name] = rng.normal(0, 1, n)
    data["CONST"] = np.ones(n)
    data["LABEL"] = np.where(rng.uniform(0, 1, n) < 0.5, "a", "b")
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def sid(factor_df):
    return make_session(factor_df, "factor_pca_main")


def _post(client, **payload):
    return client.post(f"{PREFIX}/factor_pca", json=payload)


def _dominant(loading_row, factors):
    """Name of the factor this variable loads on most strongly (by |loading|)."""
    return max(factors, key=lambda f: abs(loading_row[f]))


# ── Structure recovery ───────────────────────────────────────────────────────


def test_pca_recovers_two_components_by_kaiser(client, sid):
    r = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="varimax")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "Factor & Principal Component Analysis"
    assert d["n"] == 220 and d["p"] == 6
    assert d["extraction_method"] == "Principal Component Analysis (PCA)"
    assert d["rotation_method"] == "Varimax (Orthogonal)"

    # Kaiser criterion on a known 2-factor design must retain exactly 2.
    assert d["n_factors"] == 2
    assert d["factors"] == ["PC1", "PC2"]
    eigs = [v["eigenvalue"] for v in d["variance_explained"]]
    assert len(eigs) == 6
    assert sum(1 for e in eigs if e >= 1.0) == 2
    # Population eigenvalues are 2.6 / 2.6 / 0.2 x4.
    assert 2.2 < eigs[0] < 3.1 and 2.2 < eigs[1] < 3.1
    assert eigs[2] < 0.5


def test_variance_explained_is_monotone_and_totals_100(client, sid):
    """pct_variance is reported in PERCENT and the eigenvalues sum to p."""
    r = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="varimax")
    assert r.status_code == 200, r.text
    d = r.json()
    ve = d["variance_explained"]
    assert [v["component"] for v in ve] == [1, 2, 3, 4, 5, 6]

    eigs = [v["eigenvalue"] for v in ve]
    assert eigs == sorted(eigs, reverse=True)
    # Eigenvalues of a correlation matrix sum to the number of variables.
    assert sum(eigs) == pytest.approx(6.0, abs=1e-8)

    pcts = [v["pct_variance"] for v in ve]
    assert pcts == sorted(pcts, reverse=True)
    assert all(0.0 <= p <= 100.0 for p in pcts)
    assert sum(pcts) == pytest.approx(100.0, abs=1e-6)

    cum = [v["cum_variance"] for v in ve]
    assert cum == sorted(cum)
    assert cum[-1] == pytest.approx(100.0, abs=1e-6)
    for i, v in enumerate(ve):
        assert v["cum_variance"] == pytest.approx(sum(pcts[: i + 1]), abs=1e-6)
    # The two retained components carry the great majority of the variance.
    assert cum[1] > 80.0

    scree = d["scree_coords"]
    assert [s["eigenvalue"] for s in scree] == eigs


def test_varimax_loadings_recover_the_generating_blocks(client, sid):
    """Every item must load on the component its own latent factor built."""
    r = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="varimax")
    assert r.status_code == 200, r.text
    d = r.json()
    factors = d["factors"]
    rows = {row["variable"]: row for row in d["loadings"]}
    assert set(rows) == set(ITEMS)

    dom_a = {_dominant(rows[v], factors) for v in BLOCK_A}
    dom_b = {_dominant(rows[v], factors) for v in BLOCK_B}
    assert len(dom_a) == 1 and len(dom_b) == 1, "a block split across components"
    assert dom_a != dom_b, "both blocks collapsed onto the same component"

    # Simple structure: high on own component, near-zero on the other.
    for v in ITEMS:
        row = rows[v]
        own = _dominant(row, factors)
        other = [f for f in factors if f != own][0]
        assert abs(row[own]) > 0.80, f"{v} loads only {row[own]:.3f} on {own}"
        assert abs(row[other]) < 0.30, f"{v} cross-loads {row[other]:.3f}"
        # Communality from 2 orthogonal components on r=0.8 blocks ~ 0.87.
        assert 0.70 < row["h2"] < 1.0
        assert row["u2"] == pytest.approx(1.0 - row["h2"], abs=1e-9)
        assert row["h2"] == pytest.approx(
            sum(row[f] ** 2 for f in factors), abs=1e-9)


def test_bartlett_rejects_sphericity_on_structured_data(client, sid):
    r = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="varimax")
    assert r.status_code == 200, r.text
    s = r.json()["suitability"]
    # df = p(p-1)/2 = 15 for six items.
    assert s["bartlett_df"] == 15
    assert s["bartlett_chi2"] > 100.0
    assert s["bartlett_p"] < 1e-20
    assert s["overall_kmo"] > 0.6
    assert s["kmo_rating"] in {"Middling", "Meritorious", "Marvelous"}
    assert set(s["item_kmo"]) == set(ITEMS)
    assert all(0.0 <= v <= 1.0 for v in s["item_kmo"].values())


def test_unstructured_noise_fails_the_suitability_tests(client, sid):
    """Negative control: independent items must not look factorable."""
    r = _post(client, session_id=sid, items=NOISE_ITEMS, extraction="pca",
              rotation="varimax")
    assert r.status_code == 200, r.text
    s = r.json()["suitability"]
    assert s["bartlett_p"] > 0.05, "sphericity rejected on independent data"
    assert s["overall_kmo"] < 0.6
    assert s["kmo_rating"] in {"Unacceptable", "Miserable"}
    # No component should dominate: eigenvalues of a random correlation
    # matrix hug 1.0.
    eigs = [v["eigenvalue"] for v in r.json()["variance_explained"]]
    assert eigs[0] < 1.6


def test_explicit_n_factors_overrides_kaiser(client, sid):
    r = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="varimax", n_factors=3)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_factors"] == 3
    assert d["factors"] == ["PC1", "PC2", "PC3"]
    assert all("PC3" in row for row in d["loadings"])
    # More components can only explain more of each variable.
    two = _post(client, session_id=sid, items=ITEMS, extraction="pca",
                rotation="varimax", n_factors=2).json()
    h2_three = {r_["variable"]: r_["h2"] for r_ in d["loadings"]}
    h2_two = {r_["variable"]: r_["h2"] for r_ in two["loadings"]}
    for v in ITEMS:
        assert h2_three[v] >= h2_two[v] - 1e-9


def test_n_factors_is_capped_at_the_item_count(client, sid):
    r = _post(client, session_id=sid, items=ITEMS, n_factors=99)
    assert r.status_code == 200, r.text
    assert r.json()["n_factors"] == 6


def test_single_component_biplot_falls_back_to_zero_y(client, sid):
    r = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="varimax", n_factors=1)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["factors"] == ["PC1"]
    assert all(pt["y"] == 0.0 for pt in d["biplot"])
    assert {pt["variable"] for pt in d["biplot"]} == set(ITEMS)


def test_unrotated_pca_first_component_is_a_general_factor(client, sid):
    """Sanity check that rotation is what produces simple structure.

    Unrotated, PC1 of two equal-sized orthogonal blocks is a general factor
    that every item loads on; only after varimax does each item pick a side.
    """
    d = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="none").json()
    assert d["rotation_method"] == "Unrotated"
    rows = {row["variable"]: row for row in d["loadings"]}
    assert all(abs(rows[v]["PC1"]) > 0.4 for v in ITEMS)
    # Rotation is variance-preserving: communalities are unchanged.
    rotated = _post(client, session_id=sid, items=ITEMS, extraction="pca",
                    rotation="varimax").json()
    rot_rows = {row["variable"]: row for row in rotated["loadings"]}
    for v in ITEMS:
        assert rows[v]["h2"] == pytest.approx(rot_rows[v]["h2"], abs=1e-6)


def test_efa_extraction_also_recovers_the_blocks(client, sid):
    r = _post(client, session_id=sid, items=ITEMS, extraction="efa",
              rotation="varimax", n_factors=2)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["extraction_method"].startswith("Exploratory Factor Analysis")
    assert d["factors"] == ["Factor 1", "Factor 2"]
    rows = {row["variable"]: row for row in d["loadings"]}
    dom_a = {_dominant(rows[v], d["factors"]) for v in BLOCK_A}
    dom_b = {_dominant(rows[v], d["factors"]) for v in BLOCK_B}
    assert len(dom_a) == 1 and len(dom_b) == 1 and dom_a != dom_b
    for v in ITEMS:
        own = _dominant(rows[v], d["factors"])
        other = [f for f in d["factors"] if f != own][0]
        assert abs(rows[v][own]) > 0.7
        assert abs(rows[v][other]) < 0.3
    # NOTE: h2/u2 are deliberately NOT asserted here — sklearn's
    # FactorAnalysis is fit on the *unstandardised* data in factor.py, so the
    # reported communalities can exceed 1 (see the report accompanying these
    # tests).


def test_export_rows_mirror_the_loadings_table(client, sid):
    d = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="varimax").json()
    rows = d["export_rows"]
    assert rows[0] == ["Variable", "PC1", "PC2", "Communality (h2)",
                       "Uniqueness (u2)"]
    assert len(rows) == len(ITEMS) + 1
    assert [r_[0] for r_ in rows[1:]] == ITEMS
    for exported, loading in zip(rows[1:], d["loadings"]):
        assert exported[1] == pytest.approx(loading["PC1"], abs=1e-9)
        assert exported[2] == pytest.approx(loading["PC2"], abs=1e-9)
        assert exported[3] == pytest.approx(loading["h2"], abs=1e-9)


def test_r_code_reflects_the_requested_solution(client, sid):
    d = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="varimax").json()
    assert "principal(" in d["r_code"]
    assert "nfactors = 2" in d["r_code"]
    assert 'rotate = "varimax"' in d["r_code"]


# ── Error paths ──────────────────────────────────────────────────────────────


def test_unknown_item_400(client, sid):
    r = _post(client, session_id=sid, items=["A1", "A2", "GHOST"])
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "GHOST" in detail


def test_fewer_than_three_items_400(client, sid):
    r = _post(client, session_id=sid, items=["A1", "A2"])
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "at least 3" in detail


def test_too_few_rows_400(client, factor_df):
    tiny = make_session(factor_df.head(8), "factor_pca_tiny")
    r = _post(client, session_id=tiny, items=ITEMS)
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "at least 10" in detail


def test_non_numeric_item_is_rejected_not_silently_dropped(client, sid):
    """A text column coerces to all-NaN, leaving zero usable rows -> 400."""
    r = _post(client, session_id=sid, items=["A1", "A2", "LABEL"])
    assert r.status_code == 400, r.text
    assert r.json()["detail"].strip()


def test_unknown_session_404(client):
    r = _post(client, session_id="no_such_session", items=ITEMS)
    assert r.status_code == 404, r.text
    assert r.json()["detail"].strip()


# ── Regressions (were 500-level defects, now fixed) ──────────────────────────


def test_promax_rotation_returns_an_oblique_solution(client, sid):
    """`np.diag` on a 2-D array returns a read-only view.

    `d[d == 0] = 1e-12` on that view raised "assignment destination is
    read-only" on every call, so promax — one of the three advertised
    rotations — was a guaranteed 500.
    """
    r = _post(client, session_id=sid, items=ITEMS, extraction="pca",
              rotation="promax", n_factors=2)
    assert r.status_code == 200, r.text
    assert r.json()["rotation_method"] == "Promax (Oblique)"


def test_promax_rotation_also_works_for_efa(client, sid):
    r = _post(client, session_id=sid, items=ITEMS, extraction="efa",
              rotation="promax", n_factors=2)
    assert r.status_code == 200, r.text
    assert r.json()["rotation_method"] == "Promax (Oblique)"


def test_constant_item_is_a_readable_400_not_a_500(client, sid):
    """A zero-variance item makes df.corr() NaN and eigh fail to converge.

    Choosing a column that is constant (often after filtering) is ordinary
    user input, so it must come back as a 400 naming the column.
    """
    r = _post(client, session_id=sid, items=["A1", "A2", "CONST"])
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail.strip()
    assert "CONST" in detail, f"the 400 should name the offending item: {detail}"


def test_efa_communalities_are_valid_proportions(client, factor_df):
    """h2 is a share of variance, so it cannot exceed 1 (nor u2 drop below 0).

    EFA used to be fitted on the raw columns while KMO, Bartlett and the
    eigenvalues all used the correlation matrix. The loadings therefore carried
    the variables' own scale and h2 came back above 1 with negative uniqueness.
    """
    sid_efa = make_session(factor_df, "factor_pca_efa_scale")
    r = _post(client, session_id=sid_efa, items=ITEMS, extraction="efa",
              rotation="varimax", n_factors=2)
    assert r.status_code == 200, r.text
    for row in r.json()["loadings"]:
        assert 0.0 <= row["h2"] <= 1.0 + 1e-6, f"communality out of range: {row}"
        assert row["u2"] >= -1e-6, f"negative uniqueness: {row}"
        assert row["h2"] + row["u2"] == pytest.approx(1.0, abs=1e-6)


def test_efa_communalities_are_scale_invariant(client, factor_df):
    """Rescaling a variable must not change how much of it the factors explain."""
    scaled = factor_df.copy()
    for c in ITEMS:
        scaled[c] = scaled[c] * 1000.0

    def h2s(df, name):
        r = _post(client, session_id=make_session(df, name), items=ITEMS,
                  extraction="efa", rotation="varimax", n_factors=2)
        assert r.status_code == 200, r.text
        return [row["h2"] for row in r.json()["loadings"]]

    base = h2s(factor_df, "factor_pca_scale_base")
    blown = h2s(scaled, "factor_pca_scale_blown")
    for a, b in zip(base, blown):
        assert a == pytest.approx(b, abs=0.02), f"h2 moved with scale: {base} vs {blown}"
