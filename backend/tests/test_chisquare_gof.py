"""One-way chi-square goodness-of-fit endpoint (/api/categorical/chisquare_gof)."""
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp
from conftest import make_session

from routers.categorical import (
    ChiSquareGofRequest,
    _exact_multinomial_p,
    _gof_compositions,
)

URL = "/api/categorical/chisquare_gof"
MENDEL = {"round_yellow": 9, "round_green": 3, "wrinkled_yellow": 3, "wrinkled_green": 1}


def _mendel_frame() -> pd.DataFrame:
    counts = dict(zip(MENDEL, (315, 108, 101, 32)))
    return pd.DataFrame({"seed": [k for k, c in counts.items() for _ in range(c)]})


def _post(client, df, sid, **body):
    make_session(df, sid)
    return client.post(URL, json={"session_id": sid, "column": "seed", **body})


def test_request_accepts_proportions_alias():
    req = ChiSquareGofRequest.model_validate(
        {"session_id": "s", "column": "x", "proportions": {"a": 1, "b": 1}}
    )
    assert req.expected_proportions == {"a": 1, "b": 1}


def test_mendel_classic_values(client):
    r = _post(client, _mendel_frame(), "gof_mendel", expected_proportions=MENDEL)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n"] == 556 and d["k"] == 4 and d["df"] == 3
    assert d["chi2"] == pytest.approx(0.470024, abs=1e-5)
    assert d["p"] == pytest.approx(0.925425, abs=1e-5)
    assert d["significant"] is False
    assert d["proportions_normalised"] is True
    assert d["supplied_proportion_sum"] == pytest.approx(16.0)
    # Hand values for the first category: E = 556 * 9/16 = 312.75
    first = d["categories"][0]
    assert first["category"] == "round_yellow"
    assert first["observed"] == 315
    assert first["expected_count"] == pytest.approx(312.75)
    assert first["expected_proportion"] == pytest.approx(0.5625)
    assert first["observed_proportion"] == pytest.approx(315 / 556, abs=1e-6)
    assert first["pearson_residual"] == pytest.approx(2.25 / np.sqrt(312.75), abs=1e-4)
    assert first["adjusted_residual"] == pytest.approx(
        2.25 / np.sqrt(312.75 * (1 - 0.5625)), abs=1e-4
    )
    w = d["effect_sizes"][0]
    assert w["name"] == "cohens_w"
    assert w["value"] == pytest.approx(np.sqrt(0.470024 / 556), abs=1e-4)
    assert w["magnitude"] == "negligible"
    assert d["assumptions"][0]["met"] is True
    assert d["exact_multinomial"]["p"] is None  # 556 observations: too large
    assert any("rescaled" in x for x in d["warnings"])
    assert "chisq.test" in d["r_code"] and "rescale.p = TRUE" in d["r_code"]
    for key in ("result_text", "interpretation", "methods_text", "export_rows"):
        assert d[key]


def test_default_is_uniform_across_observed_categories(client):
    df = pd.DataFrame({"seed": ["a"] * 30 + ["b"] * 20 + ["c"] * 10})
    d = _post(client, df, "gof_uniform").json()
    assert d["k"] == 3 and d["n"] == 60
    assert [c["category"] for c in d["categories"]] == ["a", "b", "c"]
    assert all(c["expected_count"] == pytest.approx(20.0) for c in d["categories"])
    ref = sp.chisquare([30, 20, 10])
    assert d["chi2"] == pytest.approx(ref.statistic)
    assert d["p"] == pytest.approx(ref.pvalue)
    assert d["proportions_normalised"] is False
    assert d["r_code"].endswith("chisq.test(x = c(30, 20, 10))")


def test_weights_and_proportions_give_the_same_result(client):
    df = _mendel_frame()
    weights = _post(client, df, "gof_w", expected_proportions=MENDEL).json()
    props = _post(
        client, df, "gof_p",
        expected_proportions={k: v / 16 for k, v in MENDEL.items()},
    ).json()
    assert weights["chi2"] == pytest.approx(props["chi2"])
    assert weights["p"] == pytest.approx(props["p"])
    assert props["proportions_normalised"] is False
    assert "rescale.p" not in props["r_code"]


def test_missing_category_in_expected_is_422_and_named(client):
    df = pd.DataFrame({"seed": ["a"] * 10 + ["b"] * 10 + ["c"] * 10})
    r = _post(client, df, "gof_missing", expected_proportions={"a": 1, "b": 1})
    assert r.status_code == 422
    assert "'c'" in r.text


@pytest.mark.parametrize("bad", [0, -1, float("inf")])
def test_non_positive_or_nonfinite_proportion_is_422(client, bad):
    df = pd.DataFrame({"seed": ["a"] * 10 + ["b"] * 10})
    make_session(df, "gof_bad")
    from routers.categorical import chisquare_gof
    from fastapi import HTTPException

    req = ChiSquareGofRequest(
        session_id="gof_bad", column="seed", expected_proportions={"a": 1.0, "b": bad}
    )
    with pytest.raises(HTTPException) as exc:
        chisquare_gof(req)
    assert exc.value.status_code == 422


def test_extra_expected_category_kept_with_zero_observed(client):
    df = pd.DataFrame({"seed": ["a"] * 30 + ["b"] * 30})
    d = _post(client, df, "gof_extra",
              expected_proportions={"a": 0.4, "b": 0.4, "c": 0.2}).json()
    assert d["k"] == 3 and d["df"] == 2
    c_row = next(c for c in d["categories"] if c["category"] == "c")
    assert c_row["observed"] == 0
    assert c_row["expected_count"] == pytest.approx(12.0)
    assert any("no observations" in x for x in d["warnings"])
    ref = sp.chisquare([30, 30, 0], [24, 24, 12])
    assert d["chi2"] == pytest.approx(ref.statistic)


def test_single_category_is_422(client):
    df = pd.DataFrame({"seed": ["a"] * 10})
    assert _post(client, df, "gof_one").status_code == 422


def test_unknown_column_and_all_missing(client):
    df = pd.DataFrame({"seed": [None, None, None]})
    assert _post(client, df, "gof_allna").status_code == 400
    make_session(pd.DataFrame({"x": [1, 2]}), "gof_nocol")
    r = client.post(URL, json={"session_id": "gof_nocol", "column": "seed"})
    assert r.status_code == 400


def test_missing_values_are_dropped_and_reported(client):
    df = pd.DataFrame({"seed": ["a"] * 12 + ["b"] * 8 + [None] * 3 + ["n/a"] * 2})
    d = _post(client, df, "gof_na").json()
    assert d["n"] == 20 and d["n_excluded"] == 5
    assert any("excluded" in x for x in d["warnings"])


def test_numeric_and_yes_no_labels_match_what_the_user_typed(client):
    df = pd.DataFrame({"seed": [1.0] * 30 + [2.0] * 30})
    d = _post(client, df, "gof_num", expected_proportions={"1": 1, "2": 1}).json()
    assert d["chi2"] == pytest.approx(0.0)
    df2 = pd.DataFrame({"seed": ["yes"] * 30 + ["no"] * 10})
    d2 = _post(client, df2, "gof_yn", expected_proportions={"yes": 3, "no": 1}).json()
    assert d2["chi2"] == pytest.approx(0.0)


def test_small_expected_counts_warn_and_return_exact_p(client):
    df = pd.DataFrame({"seed": ["a"] * 8 + ["b"] * 2 + ["c"] * 1})
    d = _post(client, df, "gof_small").json()
    assert d["assumptions"][0]["met"] is False
    assert any("below 5" in x and "20%" in x for x in d["warnings"])
    exact = d["exact_multinomial"]
    assert exact["p"] is not None and 0 < exact["p"] <= 1
    assert "XNomial" in d["r_code"]


def test_exact_multinomial_matches_binomial_for_two_categories():
    # With k = 2 the multinomial test is the two-sided binomial test
    # (probability-ordered), which scipy implements independently.
    obs = np.array([7, 3])
    out = _exact_multinomial_p(obs, np.array([0.5, 0.5]))
    assert out["p"] == pytest.approx(sp.binomtest(7, 10, 0.5).pvalue, abs=1e-10)
    obs = np.array([9, 1])
    out = _exact_multinomial_p(obs, np.array([0.3, 0.7]))
    assert out["p"] == pytest.approx(sp.binomtest(9, 10, 0.3).pvalue, abs=1e-10)


def test_exact_multinomial_matches_brute_force_three_categories():
    obs = np.array([6, 2, 1])
    probs = np.array([0.5, 0.3, 0.2])
    n = int(obs.sum())
    total = 0.0
    p_obs = sp.multinomial.pmf(obs, n, probs)
    for a in range(n + 1):
        for b in range(n + 1 - a):
            x = np.array([a, b, n - a - b])
            pm = sp.multinomial.pmf(x, n, probs)
            if pm <= p_obs * (1 + 1e-7):
                total += pm
    out = _exact_multinomial_p(obs, probs)
    assert out["p"] == pytest.approx(total, abs=1e-12)
    assert out["n_outcomes"] == len(_gof_compositions(n, 3)) == 55


def test_exact_multinomial_is_capped():
    out = _exact_multinomial_p(np.array([250, 250, 250, 250]), np.full(4, 0.25))
    assert out["p"] is None and "limit" in out["note"]
    out5 = _exact_multinomial_p(np.array([2, 2, 2, 2, 2]), np.full(5, 0.2))
    assert out5["p"] is None


def test_alpha_controls_significance(client):
    df = pd.DataFrame({"seed": ["a"] * 60 + ["b"] * 40})
    strict = _post(client, df, "gof_a1", alpha=0.01).json()
    loose = _post(client, df, "gof_a2", alpha=0.05).json()
    assert loose["p"] == pytest.approx(strict["p"])
    assert 0.01 < loose["p"] < 0.05
    assert loose["significant"] is True and strict["significant"] is False


def test_no_em_dash_in_response(client):
    d = _post(client, _mendel_frame(), "gof_dash", expected_proportions=MENDEL).json()
    assert "—" not in repr(d)


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")
def test_matches_r_chisq_test(client):
    df = pd.DataFrame({"seed": ["a"] * 18 + ["b"] * 25 + ["c"] * 9 + ["d"] * 4})
    d = _post(client, df, "gof_r", expected_proportions={"a": 4, "b": 3, "c": 2, "d": 1}).json()
    code = (
        "r <- chisq.test(x = c(18, 25, 9, 4), p = c(4, 3, 2, 1), rescale.p = TRUE);"
        "cat(sprintf('%.10f %.10f', r$statistic, r$p.value))"
    )
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, check=True)
    chi2_r, p_r = (float(v) for v in out.stdout.split())
    assert d["chi2"] == pytest.approx(chi2_r, abs=1e-6)
    assert d["p"] == pytest.approx(p_r, abs=1e-8)
    # The emitted r_code is runnable and agrees too.
    body = re.sub(r"^#.*\n", "", d["r_code"])
    out2 = subprocess.run(
        ["Rscript", "-e", f"r <- {body}; cat(sprintf('%.10f', r$p.value))"],
        capture_output=True, text=True, check=True,
    )
    assert float(out2.stdout) == pytest.approx(p_r, abs=1e-8)
