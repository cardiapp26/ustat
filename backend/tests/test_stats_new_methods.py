"""Regression checks for newly exposed statistical methods."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from conftest import make_session


def test_sign_test_one_sample_and_paired(client):
    sid = make_session(pd.DataFrame({"a": [3, 4, 5, 2, 1, 3], "b": [2, 3, 5, 4, 0, 3]}), "sign_new")
    one = client.post("/api/stats/sign_test", json={"session_id": sid, "column": "a", "mu": 3}).json()
    assert (one["positive"], one["negative"], one["ties_excluded"]) == (2, 2, 2)
    assert one["p"] == pytest.approx(stats.binomtest(2, 4).pvalue)
    paired = client.post("/api/stats/sign_test", json={"session_id": sid, "column": "a", "comparison_column": "b", "alternative": "greater"}).json()
    assert (paired["positive"], paired["negative"], paired["ties_excluded"]) == (3, 1, 2)
    assert paired["p"] == pytest.approx(stats.binomtest(3, 4, alternative="greater").pvalue)


def test_reliability_extensions(client):
    values = pd.DataFrame({
        "i1": [0, 1, 1, 0, 1, 0, 1, 1],
        "i2": [0, 1, 0, 0, 1, 1, 1, 0],
        "i3": [0, 1, 1, 1, 0, 0, 1, 1],
        "i4": [0, 1, 0, 1, 1, 0, 1, 0],
    })
    sid = make_session(values, "reliability_new")
    response = client.post("/api/reliability/cronbach", json={"session_id": sid, "items": list(values)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kr20"] == pytest.approx(body["alpha"])
    assert body["kr21"] == pytest.approx(0.5555555555555556, abs=1e-4)
    # Independent R psych::alpha(check.keys=FALSE) reference.
    assert body["alpha"] == pytest.approx(0.5679012345679012, abs=1e-4)
    assert body["standardized_alpha"] == pytest.approx(0.567724, abs=1e-4)
    corr = values.corr().to_numpy()
    mean_r = corr[np.triu_indices(4, 1)].mean()
    assert body["standardized_alpha"] == pytest.approx(4 * mean_r / (1 + 3 * mean_r), abs=1e-4)
    left = values[["i1", "i3"]].sum(axis=1)
    right = values[["i2", "i4"]].sum(axis=1)
    r = left.corr(right)
    assert body["split_half"]["spearman_brown"] == pytest.approx(2 * r / (1 + r), abs=1e-4)
    assert body["split_half"]["guttman"] == pytest.approx(2 * (1 - (left.var() + right.var()) / (left + right).var()), abs=1e-4)


def test_ordinal_association_and_weighted_kappa(client):
    rows = []
    table = np.array([[5, 1, 0], [2, 4, 1], [0, 1, 6]])
    for i in range(3):
        for j in range(3):
            rows.extend([{"x": i + 1, "y": j + 1}] * int(table[i, j]))
    sid = make_session(pd.DataFrame(rows), "ordinal_new")
    response = client.post("/api/stats/ordinal_association", json={"session_id": sid, "row_column": "x", "col_column": "y"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["table"] == table.tolist()
    assert body["somers_d_col_given_row"] == pytest.approx(stats.somersd(table).statistic)
    # DescTools::GoodmanKruskalGamma reference for this contingency table.
    assert body["gamma"] == pytest.approx(0.9444444444444444)
    response = client.post("/api/stats/cohens_kappa", json={"session_id": sid, "rater1_col": "x", "rater2_col": "y", "weights": "quadratic"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kappa"] == pytest.approx(cohen_kappa_score([r["x"] for r in rows], [r["y"] for r in rows], weights="quadratic"))
    assert body["ci_low"] is None


def test_anova_scheffe_table(client):
    groups = {"a": [1, 2, 3, 2, 1], "b": [3, 4, 5, 4, 3], "c": [6, 7, 8, 7, 6]}
    sid = make_session(pd.DataFrame([{"group": key, "value": v} for key, vals in groups.items() for v in vals]), "scheffe_new")
    response = client.post("/api/stats/anova", json={"session_id": sid, "column": "value", "group_column": "group", "posthoc": "scheffe", "force_posthoc": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["posthoc_method"] == "Scheffé"
    assert len(body["posthoc"]) == 3
    assert [row["df"] for row in body["anova_table"]] == [2, 12, 14]
    values = np.concatenate(list(groups.values()))
    assert body["anova_table"][2]["ss"] == pytest.approx(np.sum((values - values.mean()) ** 2))
    # R DescTools::ScheffeTest fixture, computed independently.
    assert [r["p_adj"] for r in body["posthoc"]] == pytest.approx([0.00905250510317123, 2.765656922735e-6, 0.000403575543048498])
    response = client.post("/api/stats/anova", json={"session_id": sid, "column": "value", "group_column": "group", "posthoc": "bonferroni", "force_posthoc": True})
    # R pairwise.t.test(pool.sd=TRUE, p.adjust.method='bonferroni').
    assert [r["p_adj"] for r in response.json()["posthoc"]] == pytest.approx([0.00787582729431023, 1.97482042761604e-6, 0.000312078153903601])


def test_weighted_kappa_preserves_absent_interior_category(client):
    x = [1., 1., 3., 3., 4., 4., 1., 4.]
    y = [1., 3., 3., 4., 4., 1., 4., 3.]
    sid = make_session(pd.DataFrame({"x": x, "y": y}), "kappa_gap")
    response = client.post("/api/stats/cohens_kappa", json={"session_id": sid, "rater1_col": "x", "rater2_col": "y", "weights": "linear", "level_order": ["1", "2", "3", "4"]})
    assert response.status_code == 200, response.text
    assert response.json()["kappa"] == pytest.approx(cohen_kappa_score(x, y, labels=[1, 2, 3, 4], weights="linear"))
    assert response.json()["level_order"] == ["1", "2", "3", "4"]


def test_reliability_constant_item_remains_reportable(client):
    sid = make_session(pd.DataFrame({"a": [0, 0, 0, 0], "b": [0, 1, 0, 1], "c": [0, 1, 1, 0]}), "constant_item")
    response = client.post("/api/reliability/cronbach", json={"session_id": sid, "items": ["a", "b", "c"]})
    assert response.status_code == 200, response.text
    assert response.json()["item_stats"][0]["item_total_r"] is None
    assert response.json()["standardized_alpha"] is None


def test_one_sample_wilcoxon(client):
    values = np.array([1, 2, 4, 5, 8, 9, 3, np.nan])
    sid = make_session(pd.DataFrame({"x": values}), "one_wilcoxon")
    response = client.post("/api/stats/wilcoxon_onesample", json={"session_id": sid, "column": "x", "mu": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    differences = values[np.isfinite(values)] - 3
    assert body["p"] == pytest.approx(stats.wilcoxon(differences).pvalue)
    assert body["zeros_excluded"] == 1
    assert body["z_asymptotic"] == pytest.approx(stats.wilcoxon(differences, method="approx").zstatistic)


def test_chisquare_residuals_and_contingency_coefficient(client):
    from statsmodels.stats.contingency_tables import Table
    table = np.array([[20, 10, 5], [5, 10, 20]])
    rows = [(str(i), str(j)) for i in range(2) for j in range(3) for _ in range(table[i, j])]
    sid = make_session(pd.DataFrame(rows, columns=["x", "y"]), "cell_residuals")
    response = client.post("/api/stats/chisquare", json={"session_id": sid, "row_column": "x", "col_column": "y"})
    assert response.status_code == 200, response.text
    body = response.json()
    residuals = pd.DataFrame(body["standardized_residuals"]).to_numpy()
    assert residuals == pytest.approx(Table(table, shift_zeros=False).standardized_resids)
    assert body["contingency_coefficient"] == pytest.approx(stats.contingency.association(table, method="pearson"))


def test_correlation_method_labels_and_point_biserial(client):
    sid = make_session(pd.DataFrame({"x": [0, 0, 0, 1, 1, 1], "y": [1, 2, 4, 3, 5, 8]}), "pair_methods")
    response = client.post("/api/stats/correlation_pair", json={"session_id": sid, "var1": "x", "var2": "y", "method": "pointbiserial"})
    assert response.status_code == 200, response.text
    assert response.json()["r"] == pytest.approx(stats.pointbiserialr([0, 0, 0, 1, 1, 1], [1, 2, 4, 3, 5, 8]).statistic)
    response = client.post("/api/stats/correlation_pair", json={"session_id": sid, "var1": "x", "var2": "y", "method": "kendall"})
    assert "Kendall" in response.json()["result_text"]


def test_welch_anova_effect_size_matches_classical_ss(client):
    sid = make_session(pd.DataFrame({"g": ["a"] * 8 + ["b"] * 8 + ["c"] * 8, "x": list(range(8)) + list(np.arange(8) * 20) + list(np.arange(8) * 0.1)}), "welch_ss")
    response = client.post("/api/stats/anova", json={"session_id": sid, "column": "x", "group_column": "g", "posthoc": "none"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["variance_assumption"] == "welch"
    assert body["effect_sizes"][0]["value"] == pytest.approx(body["anova_table"][0]["ss"] / body["anova_table"][2]["ss"], abs=1e-4)


@pytest.mark.parametrize("agreement", ["absolute", "consistency"])
@pytest.mark.parametrize("unit", ["single", "average"])
def test_icc_perfect_raters_do_not_report_zero_f(client, agreement, unit):
    sid = make_session(pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [1, 2, 3, 4]}), "perfect_icc")
    response = client.post("/api/stats/icc", json={"session_id": sid, "rater_cols": ["a", "b", "c"], "agreement": agreement, "unit": unit})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["icc"] == pytest.approx(1)
    assert body["f_stat"] is None
    assert body["f_p"] == 0
    assert body["ci_low"] is None
    assert "Zero residual variance" in body["interval_note"]
