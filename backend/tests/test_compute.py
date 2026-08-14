"""Coverage tests for routers/compute.py.

The compute router is mounted at prefix ``/api/compute`` and addresses the
session via the URL PATH segment ``/{session_id}/...`` (NOT a body field).
Each test persists a synthetic DataFrame via ``make_session`` and exercises one
endpoint (happy path + a few validation / edge cases).
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session
from services import store

SEED = 4242
BASE = "/api/compute"


def _fresh(synth, suffix):
    """Persist a fresh copy of the synthetic frame under a unique session id.

    Many compute endpoints mutate the stored frame (add/delete columns/rows),
    so each test gets its own isolated session to avoid cross-test coupling.
    """
    sid = f"tcomp_{suffix}"
    return make_session(synth.copy(), sid)


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(SEED)
    n = 200
    age = rng.normal(60, 10, n).clip(20, 90)
    ldl = rng.normal(120, 30, n).clip(40, 250)
    weight = rng.normal(80, 15, n).clip(40, 140)
    height = rng.normal(170, 10, n).clip(140, 200)
    creat = rng.normal(1.0, 0.3, n).clip(0.4, 3.0)
    sbp = rng.normal(130, 20, n).clip(80, 220)
    dbp = rng.normal(80, 12, n).clip(40, 130)
    hr = rng.normal(75, 15, n).clip(40, 160)
    qt = rng.normal(400, 40, n).clip(300, 550)
    ef = rng.normal(45, 12, n).clip(10, 70)
    bmi = (weight / ((height / 100) ** 2)).round(1)
    sex = rng.integers(0, 2, n)
    dm = rng.integers(0, 2, n)
    htn = rng.integers(0, 2, n)
    chf = rng.integers(0, 2, n)
    stroke = rng.integers(0, 2, n)
    vasc = rng.integers(0, 2, n)
    af = rng.integers(0, 2, n)
    nyha = rng.integers(1, 5, n)
    killip = rng.integers(1, 5, n)
    group = rng.choice(["A", "B", "C"], n)
    return pd.DataFrame({
        "AGE": age, "LDL": ldl, "WEIGHT": weight, "HEIGHT": height,
        "CREAT": creat, "SBP": sbp, "DBP": dbp, "HR": hr, "QT": qt,
        "EF": ef, "BMI": bmi, "SEX": sex, "DM": dm, "HTN": htn,
        "CHF": chf, "STROKE": stroke, "VASC": vasc, "AF": af,
        "NYHA": nyha, "KILLIP": killip, "GROUP": group,
    })


# ── Formula ───────────────────────────────────────────────────────────────────

def test_formula_basic_arithmetic(client, synth):
    sid = _fresh(synth, "formula_ok")
    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "LDL / 38.67", "new_col": "LDL_mmol"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "LDL_mmol"
    assert "preview_values" in body
    assert body["n_computed"] > 0
    assert set(body) >= {"name", "dtype", "kind", "preview_values", "n_computed", "n_missing"}


def test_formula_dedupes_a_colliding_name_instead_of_overwriting(client, synth):
    """Reported: typing an existing column's name into "New column name" and
    running a formula said a new column was created — while nothing new
    appeared, because df[new_col] = result had just replaced the existing
    column of that name in place, silently, with no error and no warning."""
    sid = _fresh(synth, "formula_collide")
    before = store.get(sid)["AGE"].tolist()

    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "AGE + 1", "new_col": "AGE"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "AGE_new"
    assert any("AGE" in str(w) and "AGE_new" in str(w) for w in body["warnings"])

    df = store.get(sid)
    assert "AGE_new" in df.columns
    # The original is untouched — this is the whole point of the fix.
    assert df["AGE"].tolist() == before
    assert df["AGE_new"].tolist() == [v + 1 for v in before]


def test_formula_dedupe_keeps_incrementing_past_the_first_collision(client, synth):
    sid = _fresh(synth, "formula_collide2")
    r1 = client.post(f"{BASE}/{sid}/formula",
                     json={"formula": "AGE + 1", "new_col": "AGE"})
    assert r1.json()["name"] == "AGE_new"
    r2 = client.post(f"{BASE}/{sid}/formula",
                     json={"formula": "AGE + 2", "new_col": "AGE"})
    assert r2.json()["name"] == "AGE_new2"
    df = store.get(sid)
    assert {"AGE", "AGE_new", "AGE_new2"} <= set(df.columns)


def test_formula_no_rename_note_when_the_name_is_free(client, synth):
    sid = _fresh(synth, "formula_no_collide")
    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "AGE + 1", "new_col": "AGE_PLUS_ONE"})
    body = r.json()
    assert body["name"] == "AGE_PLUS_ONE"
    assert not body.get("warnings")


def test_transform_dedupes_a_colliding_name(client, synth):
    sid = _fresh(synth, "transform_collide")
    before = store.get(sid)["DM"].tolist()
    r = client.post(f"{BASE}/{sid}/transform",
                    json={"source_col": "AGE", "transform": "zscore", "new_col": "DM"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "DM_new"
    assert any("DM_new" in str(w) for w in r.json()["warnings"])
    assert store.get(sid)["DM"].tolist() == before


def test_transform_still_refuses_its_own_source_rather_than_deduping(client):
    """Deduping is for collision with an UNRELATED column. Colliding with the
    source column is the destructive case that has its own explicit error,
    and must keep raising it rather than being quietly renamed away."""
    sid = make_session(
        pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), "tcomp_tf_still_refuses"
    )
    r = client.post(
        f"{BASE}/{sid}/transform",
        json={"source_col": "X", "transform": "tertile", "new_col": "X"},
    )
    assert r.status_code == 422, r.text
    assert "source column" in r.json()["detail"]


def test_recode_dedupes_a_colliding_name(client, synth):
    sid = _fresh(synth, "recode_collide")
    before = store.get(sid)["DM"].tolist()
    r = client.post(f"{BASE}/{sid}/recode", json={
        "new_col": "DM",
        "rules": [{"conditions": [{"col": "AGE", "op": ">", "val": "50"}], "result": "1"}],
        "else_val": "0",
    })
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "DM_new"
    assert any("DM_new" in str(w) for w in r.json()["warnings"])
    assert store.get(sid)["DM"].tolist() == before


# ── Fill blanks: compute from other columns ───────────────────────────────────

def _nlr_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Notrofil": [6.0, 8.0, 4.0, 9.0, 5.0],
        "Lenfosit": [2.0, 4.0, 2.0, 3.0, np.nan],
        "NLR":      [99.0, np.nan, np.nan, np.nan, np.nan],
    })


def test_fill_blanks_can_compute_missing_values_from_a_formula(client):
    """A derived column's blanks are not missing data — the inputs are right
    there and the ratio simply was not computed for those rows. Imputing a
    mean into them replaces an exactly knowable number with an estimate that
    then reads as measured."""
    sid = make_session(_nlr_frame(), "fill_formula_nlr")
    r = client.post(f"{BASE}/{sid}/fill_blanks", json={
        "column": "NLR", "value": "__formula__", "formula": "Notrofil / Lenfosit",
    })
    assert r.status_code == 200, r.text
    out = store.get(sid)["NLR"].tolist()
    assert out[1] == 2.0 and out[2] == 2.0 and out[3] == 3.0
    assert r.json()["n_filled"] == 3


def test_fill_blanks_formula_never_overwrites_a_recorded_value(client):
    """Row 0 holds 99, which the formula would compute as 3. The stored number
    is the measurement; recomputing it would silently replace it."""
    sid = make_session(_nlr_frame(), "fill_formula_keeps")
    client.post(f"{BASE}/{sid}/fill_blanks", json={
        "column": "NLR", "value": "__formula__", "formula": "Notrofil / Lenfosit",
    })
    assert store.get(sid)["NLR"].iloc[0] == 99.0


def test_fill_blanks_formula_leaves_a_row_blank_when_its_inputs_are_missing(client):
    """Row 4 has no Lymphocyte count, so the ratio is undefined there. It has
    to stay blank rather than become a NaN presented as a fill."""
    sid = make_session(_nlr_frame(), "fill_formula_unresolved")
    r = client.post(f"{BASE}/{sid}/fill_blanks", json={
        "column": "NLR", "value": "__formula__", "formula": "Notrofil / Lenfosit",
    })
    assert pd.isna(store.get(sid)["NLR"].iloc[4])
    assert "left blank" in r.json()["fill_value"]


def test_fill_blanks_formula_requires_a_formula(client):
    sid = make_session(_nlr_frame(), "fill_formula_empty")
    r = client.post(f"{BASE}/{sid}/fill_blanks", json={
        "column": "NLR", "value": "__formula__",
    })
    assert r.status_code == 422, r.text
    assert "formula" in r.json()["detail"].lower()


def test_fill_blanks_formula_reports_a_bad_expression(client):
    sid = make_session(_nlr_frame(), "fill_formula_bad")
    r = client.post(f"{BASE}/{sid}/fill_blanks", json={
        "column": "NLR", "value": "__formula__", "formula": "Notrofil / NOSUCH",
    })
    assert r.status_code == 422, r.text


def test_formula_custom_if_function(client, synth):
    sid = _fresh(synth, "formula_if")
    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "IF(AGE > 65, 1, 0)", "new_col": "ELDERLY"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "ELDERLY"
    assert body["n_computed"] == 200


def test_formula_column_name_with_spaces(client, synth):
    """A column whose name isn't a Python identifier is still referencable —
    bare and backticked — instead of dying with 'invalid syntax'."""
    df = synth.copy()
    df["p wave mv_2"] = 2.0
    sid = make_session(df, "tcomp_formula_spaces")

    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "LDL / p wave mv_2", "new_col": "R_BARE"})
    assert r.status_code == 200, r.text
    assert r.json()["n_computed"] == 200

    r2 = client.post(f"{BASE}/{sid}/formula",
                     json={"formula": "LDL / `p wave mv_2`", "new_col": "R_TICK"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["n_computed"] == 200


def test_formula_prefers_longest_column_name(client, synth):
    """'p wave' must not shadow 'p wave mv_2'."""
    df = synth.copy()
    df["p wave"] = 10.0
    df["p wave mv_2"] = 2.0
    sid = make_session(df, "tcomp_formula_longest")
    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "p wave mv_2 + p wave", "new_col": "SUMP"})
    assert r.status_code == 200, r.text
    assert r.json()["preview_values"][0] == 12.0


def test_formula_syntax_error_is_actionable(client, synth):
    """The old message was 'invalid syntax (<unknown>, line 1)' — useless."""
    df = synth.copy()
    df["p wave mv_2"] = 2.0
    sid = make_session(df, "tcomp_formula_syntax")
    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "LDL / / 2", "new_col": "BAD"})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "invalid syntax" not in detail
    assert "backticks" in detail


def test_formula_backtick_unknown_column(client, synth):
    sid = _fresh(synth, "formula_tick_miss")
    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "`NOSUCH` + 1", "new_col": "BAD2"})
    assert r.status_code == 422, r.text
    assert "NOSUCH" in r.json()["detail"]


@pytest.mark.parametrize("evil", [
    "().__class__.__bases__[0].__subclasses__()",  # dunder traversal
    "AGE.__class__",                                 # attribute access
    "__import__('os').system('echo pwned')",         # import / RCE
    "AGE.to_csv('/tmp/leak.csv')",                    # method call → disk write
    "open('/etc/passwd').read()",                     # builtin file read
])
def test_formula_rejects_code_injection(client, synth, evil):
    # The formula evaluator must not execute arbitrary Python — every escape
    # attempt should be rejected as an invalid formula, never run.
    sid = _fresh(synth, "formula_evil")
    r = client.post(f"{BASE}/{sid}/formula", json={"formula": evil, "new_col": "X"})
    assert r.status_code in (400, 422), f"injection not blocked: {evil!r} -> {r.status_code}"


def test_formula_unknown_column(client, synth):
    sid = _fresh(synth, "formula_bad")
    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "NOPE * 2", "new_col": "X"})
    assert r.status_code == 422, r.text


def test_formula_empty_name_rejected(client, synth):
    sid = _fresh(synth, "formula_empty")
    r = client.post(f"{BASE}/{sid}/formula",
                    json={"formula": "AGE + 1", "new_col": "   "})
    assert r.status_code == 422, r.text


def test_formula_session_not_found(client):
    r = client.post(f"{BASE}/tcomp_nope/formula",
                    json={"formula": "AGE + 1", "new_col": "X"})
    assert r.status_code == 404, r.text


# ── Transform ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("transform", [
    "ln", "log10", "sqrt", "square", "exp", "abs",
    "zscore", "tertile", "quartile", "median_split",
])
def test_transform_all_kinds(client, synth, transform):
    sid = _fresh(synth, f"tf_{transform}")
    r = client.post(f"{BASE}/{sid}/transform",
                    json={"source_col": "LDL", "transform": transform,
                          "new_col": f"LDL_{transform}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == f"LDL_{transform}"
    assert body["n_computed"] >= 0


def test_transform_unknown_transform(client, synth):
    sid = _fresh(synth, "tf_unknown")
    r = client.post(f"{BASE}/{sid}/transform",
                    json={"source_col": "LDL", "transform": "wat", "new_col": "Y"})
    assert r.status_code == 422, r.text


def test_transform_missing_column(client, synth):
    sid = _fresh(synth, "tf_miss")
    r = client.post(f"{BASE}/{sid}/transform",
                    json={"source_col": "MISSING", "transform": "ln", "new_col": "Y"})
    assert r.status_code == 422, r.text


def test_quantile_transforms_preserve_missing_with_duplicate_edges(client):
    sid = make_session(pd.DataFrame({"X": [1, 1, 1, 2, 2, np.nan, 3, 3, 4, 4]}), "tcomp_tf_bins_missing")

    for transform in ("tertile", "quartile"):
        new_col = f"X_{transform}"
        r = client.post(f"{BASE}/{sid}/transform",
                        json={"source_col": "X", "transform": transform, "new_col": new_col})
        assert r.status_code == 200, r.text
        out = store.get(sid)[new_col]
        assert pd.isna(out.iloc[5])
        assert out.dropna().min() >= 1


def test_quantile_transforms_report_their_cut_points(client):
    """A tertile column of 1/2/3 is unreportable on its own.

    A paper has to state where the boundaries fell, and a reader cannot place
    a patient in a group without them. The transform used to return the codes
    and nothing else, so the cut points had to be reconstructed by hand from
    the descriptives.
    """
    values = [float(v) for v in range(1, 127)]  # 126 rows, 42 per tertile
    sid = make_session(pd.DataFrame({"SII": values}), "tcomp_tf_cutpoints")
    r = client.post(
        f"{BASE}/{sid}/transform",
        json={"source_col": "SII", "transform": "tertile", "new_col": "SII_t"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    # The 1/3 and 2/3 quantiles of 1..126, which are not integers.
    assert out["cut_points"] == pytest.approx([42.666667, 84.333333], abs=1e-6)

    text = out["result_text"]
    assert "Tertile groups of SII" in text
    assert "42.6667" in text and "84.3333" in text
    # Every group's size is named, and they account for the whole column.
    assert text.count("n = 42") == 3

    groups = store.get(sid)["SII_t"]
    assert groups.value_counts().sort_index().tolist() == [42, 42, 42]


def test_the_reported_cut_points_are_the_boundaries_and_not_the_range(client):
    """qcut hands back the observed minimum and maximum as outer edges. Quoted
    as cut points those would say a quartile split has five boundaries."""
    sid = make_session(pd.DataFrame({"X": [float(v) for v in range(100)]}), "tcomp_tf_edges")
    r = client.post(
        f"{BASE}/{sid}/transform",
        json={"source_col": "X", "transform": "quartile", "new_col": "X_q"},
    )
    assert r.status_code == 200, r.text
    cuts = r.json()["cut_points"]
    assert len(cuts) == 3
    assert min(cuts) > 0.0 and max(cuts) < 99.0


def test_collapsed_quantile_groups_are_declared(client):
    """Ties spanning a boundary make qcut drop the duplicate edge, so a
    'tertile' can come back with two levels. Silently that reads as a bug in
    the data rather than a property of it."""
    sid = make_session(
        pd.DataFrame({"X": [1.0] * 8 + [2.0, 3.0]}), "tcomp_tf_collapsed"
    )
    r = client.post(
        f"{BASE}/{sid}/transform",
        json={"source_col": "X", "transform": "tertile", "new_col": "X_t"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(out["cut_points"]) < 2
    assert "instead of 3" in out["result_text"]


def test_median_split_reports_where_it_split(client):
    sid = make_session(
        pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), "tcomp_tf_med_text"
    )
    r = client.post(
        f"{BASE}/{sid}/transform",
        json={"source_col": "X", "transform": "median_split", "new_col": "X_med"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["cut_points"] == [3.5]
    assert "3.5" in out["result_text"]
    assert "n = 3" in out["result_text"]


def test_plain_transforms_report_no_cut_points(client):
    sid = make_session(pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0]}), "tcomp_tf_nocuts")
    r = client.post(
        f"{BASE}/{sid}/transform",
        json={"source_col": "X", "transform": "zscore", "new_col": "X_z"},
    )
    assert r.status_code == 200, r.text
    assert "cut_points" not in r.json()


def test_median_split_preserves_missing(client):
    sid = make_session(pd.DataFrame({"X": [1, 2, np.nan, 4]}), "tcomp_tf_median_missing")
    r = client.post(f"{BASE}/{sid}/transform",
                    json={"source_col": "X", "transform": "median_split", "new_col": "X_med"})
    assert r.status_code == 200, r.text
    out = store.get(sid)["X_med"]
    assert out.tolist()[:2] == [0.0, 0.0]
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] == 1.0


# ── Recode ────────────────────────────────────────────────────────────────────

def test_recode_numeric_rules(client, synth):
    sid = _fresh(synth, "recode_num")
    payload = {
        "new_col": "AGE_CAT",
        "else_val": 0,
        "rules": [
            {"conditions": [{"col": "AGE", "op": ">=", "val": 65}], "result": 2},
            {"conditions": [{"col": "AGE", "op": ">=", "val": 50}], "result": 1},
        ],
    }
    r = client.post(f"{BASE}/{sid}/recode", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "AGE_CAT"
    assert body["n_computed"] == 200


def test_recode_string_result(client, synth):
    sid = _fresh(synth, "recode_str")
    payload = {
        "new_col": "AGE_LABEL",
        "else_val": "young",
        "rules": [
            {"conditions": [{"col": "AGE", "op": ">=", "val": 65}], "result": "old"},
        ],
    }
    r = client.post(f"{BASE}/{sid}/recode", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "AGE_LABEL"


def test_recode_missing_source_returns_jsonable_preview(client):
    sid = make_session(pd.DataFrame({"GROUP": ["A", None, "B", pd.NA]}), "tcomp_recode_missing")
    payload = {
        "new_col": "GROUP_A",
        "else_val": None,
        "rules": [
            {"conditions": [{"col": "GROUP", "op": "==", "val": "A"}], "result": "yes"},
        ],
    }
    r = client.post(f"{BASE}/{sid}/recode", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["preview_values"] == ["yes", None, None, None]
    out = store.get(sid)["GROUP_A"]
    assert out.iloc[0] == "yes"
    assert out.iloc[1:].isna().all()


def test_recode_no_rules(client, synth):
    sid = _fresh(synth, "recode_none")
    r = client.post(f"{BASE}/{sid}/recode",
                    json={"new_col": "X", "rules": []})
    assert r.status_code == 422, r.text


def test_recode_missing_column(client, synth):
    sid = _fresh(synth, "recode_miss")
    payload = {
        "new_col": "X",
        "rules": [{"conditions": [{"col": "GHOST", "op": ">", "val": 1}], "result": 1}],
    }
    r = client.post(f"{BASE}/{sid}/recode", json=payload)
    assert r.status_code == 422, r.text


# ── Clinical calculators ──────────────────────────────────────────────────────

def test_clinical_bmi(client, synth):
    sid = _fresh(synth, "bmi")
    r = client.post(f"{BASE}/{sid}/clinical/bmi",
                    json={"column_map": {"weight": "WEIGHT", "height": "HEIGHT"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "BMI"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(5 < v < 90 for v in vals)


def test_clinical_bmi_missing_mapping(client, synth):
    sid = _fresh(synth, "bmi_miss")
    r = client.post(f"{BASE}/{sid}/clinical/bmi",
                    json={"column_map": {"weight": "WEIGHT"}})
    assert r.status_code == 422, r.text


def test_clinical_egfr(client, synth):
    sid = _fresh(synth, "egfr")
    r = client.post(f"{BASE}/{sid}/clinical/egfr",
                    json={"column_map": {"age": "AGE", "sex": "SEX",
                                         "creatinine": "CREAT"},
                          "female_value": "1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "eGFR"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(0 < v < 250 for v in vals)


def test_clinical_chadsvasc(client, synth):
    sid = _fresh(synth, "chadsvasc")
    r = client.post(f"{BASE}/{sid}/clinical/chadsvasc",
                    json={"column_map": {"age": "AGE", "sex": "SEX", "chf": "CHF",
                                         "htn": "HTN", "dm": "DM", "stroke": "STROKE",
                                         "vasc": "VASC"},
                          "female_value": "1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "CHA2DS2VASc"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(0 <= v <= 9 for v in vals)


def test_clinical_chadsva(client, synth):
    sid = _fresh(synth, "chadsva")
    r = client.post(f"{BASE}/{sid}/clinical/chadsva",
                    json={"column_map": {"age": "AGE", "chf": "CHF", "htn": "HTN",
                                         "dm": "DM", "stroke": "STROKE", "vasc": "VASC"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "CHA2DS2VA"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(0 <= v <= 8 for v in vals)


def test_clinical_bsa(client, synth):
    sid = _fresh(synth, "bsa")
    r = client.post(f"{BASE}/{sid}/clinical/bsa",
                    json={"column_map": {"weight": "WEIGHT", "height": "HEIGHT"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "BSA"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(0.5 < v < 4 for v in vals)


def test_clinical_map(client, synth):
    sid = _fresh(synth, "map")
    r = client.post(f"{BASE}/{sid}/clinical/map",
                    json={"column_map": {"sbp": "SBP", "dbp": "DBP"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "MAP"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(40 < v < 200 for v in vals)


def test_clinical_hasbled(client, synth):
    sid = _fresh(synth, "hasbled")
    r = client.post(f"{BASE}/{sid}/clinical/hasbled",
                    json={"column_map": {"age": "AGE", "htn": "HTN",
                                         "stroke": "STROKE"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "HAS_BLED"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(0 <= v <= 9 for v in vals)


def test_clinical_grace(client, synth):
    sid = _fresh(synth, "grace")
    r = client.post(f"{BASE}/{sid}/clinical/grace",
                    json={"column_map": {"age": "AGE", "hr": "HR", "sbp": "SBP",
                                         "creatinine": "CREAT", "killip": "KILLIP"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "GRACE_Score"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(v >= 0 for v in vals)


def test_clinical_timi_nstemi(client, synth):
    sid = _fresh(synth, "timi_n")
    r = client.post(f"{BASE}/{sid}/clinical/timi_nstemi",
                    json={"column_map": {"age": "AGE", "known_cad": "VASC"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "TIMI_NSTEMI"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(0 <= v <= 7 for v in vals)


def test_clinical_timi_stemi(client, synth):
    sid = _fresh(synth, "timi_s")
    r = client.post(f"{BASE}/{sid}/clinical/timi_stemi",
                    json={"column_map": {"age": "AGE", "sbp": "SBP", "hr": "HR",
                                         "killip": "KILLIP", "weight": "WEIGHT"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "TIMI_STEMI"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(0 <= v <= 14 for v in vals)


def test_clinical_h2fpef(client, synth):
    sid = _fresh(synth, "h2fpef")
    r = client.post(f"{BASE}/{sid}/clinical/h2fpef",
                    json={"column_map": {"bmi": "BMI", "age": "AGE", "af": "AF"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "H2FPEF"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(0 <= v <= 9 for v in vals)


def test_clinical_maggic(client, synth):
    sid = _fresh(synth, "maggic")
    r = client.post(f"{BASE}/{sid}/clinical/maggic",
                    json={"column_map": {"age": "AGE", "sbp": "SBP", "bmi": "BMI",
                                         "creatinine": "CREAT", "ef": "EF",
                                         "nyha": "NYHA", "sex": "SEX"},
                          "female_value": "1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "MAGGIC_Score"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(v >= 0 for v in vals)


def test_clinical_qtc(client, synth):
    sid = _fresh(synth, "qtc")
    r = client.post(f"{BASE}/{sid}/clinical/qtc",
                    json={"column_map": {"qt": "QT", "hr": "HR"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "QTc_Bazett"
    vals = [v for v in body["preview_values"] if v is not None]
    assert all(v > 0 for v in vals)


def test_clinical_maggic_missing_mapping(client, synth):
    sid = _fresh(synth, "maggic_miss")
    r = client.post(f"{BASE}/{sid}/clinical/maggic",
                    json={"column_map": {"age": "AGE"}})
    assert r.status_code == 422, r.text


# ── Column / row data operations ──────────────────────────────────────────────

def test_delete_column(client, synth):
    sid = _fresh(synth, "delcol")
    r = client.delete(f"{BASE}/{sid}/column/LDL")
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == "LDL"


def test_delete_column_not_found(client, synth):
    sid = _fresh(synth, "delcol_miss")
    r = client.delete(f"{BASE}/{sid}/column/NOPE")
    assert r.status_code == 404, r.text


def test_delete_column_removes_filter_and_column_state(client, synth):
    sid = _fresh(synth, "delcol_state")
    store.save_filter(sid, [
        {"column": "LDL", "operator": "gt", "value": "100", "join": "AND"},
        {"column": "AGE", "operator": "gt", "value": "50", "join": "AND"},
    ])
    store.save_metadata(sid, {"LDL": {"label": "LDL cholesterol"}})
    store.save_kind_overrides(sid, {"LDL": "numeric"})
    store.set_decimal(sid, "LDL", 2)

    r = client.delete(f"{BASE}/{sid}/column/LDL")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["case_filter"]["conditions"] == [
        {"column": "AGE", "operator": "gt", "value": "50", "join": "AND"},
    ]
    assert body["case_filter"]["selected"] <= body["case_filter"]["total"]
    assert "LDL" not in store.get_metadata(sid)
    assert "LDL" not in store.get_kind_overrides(sid)
    assert "LDL" not in store.get_decimals(sid)


def test_column_values_returns_every_row(client, synth):
    """Copy-column must see the whole column, not just the 2000-row preview."""
    sid = _fresh(synth, "colvals")
    r = client.get(f"{BASE}/{sid}/column_values/LDL")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "LDL"
    assert body["rows"] == 200
    assert len(body["values"]) == 200


def test_column_values_not_found(client, synth):
    sid = _fresh(synth, "colvals_miss")
    r = client.get(f"{BASE}/{sid}/column_values/NOPE")
    assert r.status_code == 404, r.text


def test_paste_column_numeric(client, synth):
    sid = _fresh(synth, "pastecol")
    vals = [str(i) for i in range(200)]
    r = client.post(f"{BASE}/{sid}/paste_column", json={"name": "PASTED", "values": vals})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "PASTED"
    assert body["kind"] == "numeric"
    assert body["n_computed"] == 200
    assert body["n_truncated"] == 0 and body["n_padded"] == 0
    assert body["preview_values"][:3] == [0, 1, 2]


def test_paste_column_pads_and_truncates(client, synth):
    sid = _fresh(synth, "pastecol_short")
    r = client.post(f"{BASE}/{sid}/paste_column", json={"name": "SHORT", "values": ["1", "2", "3"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_padded"] == 197
    assert body["n_computed"] == 3

    sid2 = _fresh(synth, "pastecol_long")
    r2 = client.post(f"{BASE}/{sid2}/paste_column",
                     json={"name": "LONG", "values": [str(i) for i in range(250)]})
    assert r2.status_code == 200, r2.text
    assert r2.json()["n_truncated"] == 50


def test_paste_column_text_stays_text(client, synth):
    """A text payload must not be coerced to all-NaN numeric."""
    sid = _fresh(synth, "pastecol_txt")
    vals = ["alpha", "beta"] * 100
    r = client.post(f"{BASE}/{sid}/paste_column", json={"name": "TXT", "values": vals})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_computed"] == 200
    assert body["preview_values"][0] == "alpha"


def test_paste_column_blanks_become_missing(client, synth):
    sid = _fresh(synth, "pastecol_blank")
    vals = ["5", "", "  ", "7"] + [""] * 196
    r = client.post(f"{BASE}/{sid}/paste_column", json={"name": "SPARSE", "values": vals})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_computed"] == 2
    assert body["preview_values"][1] is None


def test_paste_column_position_inserts(client, synth):
    sid = _fresh(synth, "pastecol_pos")
    r = client.post(f"{BASE}/{sid}/paste_column",
                    json={"name": "FIRST", "values": ["1"] * 200, "position": 0})
    assert r.status_code == 200, r.text
    cols = client.get(f"/api/stats/{sid}/refresh").json()["columns"]
    assert cols[0]["name"] == "FIRST"


def test_paste_column_duplicate_name(client, synth):
    sid = _fresh(synth, "pastecol_dupe")
    r = client.post(f"{BASE}/{sid}/paste_column", json={"name": "LDL", "values": ["1"]})
    assert r.status_code == 422, r.text


def test_paste_column_empty_name(client, synth):
    sid = _fresh(synth, "pastecol_noname")
    r = client.post(f"{BASE}/{sid}/paste_column", json={"name": "  ", "values": ["1"]})
    assert r.status_code == 422, r.text


def test_delete_columns_bulk(client, synth):
    sid = _fresh(synth, "delcols")
    store.save_filter(sid, [
        {"column": "LDL", "operator": "gt", "value": "100", "join": "AND"},
        {"column": "GROUP", "operator": "eq", "value": "A", "join": "AND"},
    ])
    for column in ("LDL", "AGE", "WEIGHT"):
        store.save_metadata(sid, {column: {"label": column}})
        store.save_kind_overrides(sid, {column: "numeric"})
        store.set_decimal(sid, column, 1)

    r = client.post(f"{BASE}/{sid}/delete_columns", json={"columns": ["LDL", "AGE", "WEIGHT"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] == ["LDL", "AGE", "WEIGHT"]
    for col in ("LDL", "AGE", "WEIGHT"):
        assert col not in body["remaining_columns"]
        assert col not in store.get_metadata(sid)
        assert col not in store.get_kind_overrides(sid)
        assert col not in store.get_decimals(sid)
    assert body["case_filter"]["conditions"] == [
        {"column": "GROUP", "operator": "eq", "value": "A", "join": "AND"},
    ]


def test_delete_columns_dedupes(client, synth):
    sid = _fresh(synth, "delcols_dupe")
    r = client.post(f"{BASE}/{sid}/delete_columns", json={"columns": ["LDL", "LDL"]})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == ["LDL"]


def test_delete_columns_not_found(client, synth):
    sid = _fresh(synth, "delcols_miss")
    r = client.post(f"{BASE}/{sid}/delete_columns", json={"columns": ["LDL", "NOPE"]})
    assert r.status_code == 404, r.text


def test_delete_columns_empty(client, synth):
    sid = _fresh(synth, "delcols_empty")
    r = client.post(f"{BASE}/{sid}/delete_columns", json={"columns": []})
    assert r.status_code == 422, r.text


def test_fill_blanks_mean(client, synth):
    sid = _fresh(synth, "fill_mean")
    r = client.post(f"{BASE}/{sid}/fill_blanks",
                    json={"column": "LDL", "value": "__mean__"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["column"] == "LDL"
    assert "fill_value" in body
    assert "n_filled" in body


def test_fill_blanks_rownum_partial(client, synth):
    """Blanks get their 1-based row position; observed values untouched."""
    sid = _fresh(synth, "fill_rownum")
    df_before = store.get(sid)
    observed_mask = df_before["LDL"].notna()
    r = client.post(f"{BASE}/{sid}/fill_blanks",
                    json={"column": "LDL", "value": "__rownum__"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fill_value"] == "sequential row number (1…n)"
    df_after = store.get(sid)
    assert df_after["LDL"].notna().all()
    # Observed values unchanged
    pd.testing.assert_series_equal(
        df_after.loc[observed_mask, "LDL"], df_before.loc[observed_mask, "LDL"]
    )
    # Filled cells hold their 1-based positional index
    positions = pd.Series(range(1, len(df_after) + 1), index=df_after.index)
    filled = ~observed_mask
    assert (df_after.loc[filled, "LDL"].astype(float)
            == positions[filled].astype(float)).all()


def test_fill_blanks_rownum_all_empty_column(client, synth):
    """Fully-empty new column numbers every case 1..n as a numeric ID."""
    sid = _fresh(synth, "fill_rownum_id")
    df = store.get(sid).copy()
    df["CASE_ID"] = np.nan
    store.save(sid, df)
    r = client.post(f"{BASE}/{sid}/fill_blanks",
                    json={"column": "CASE_ID", "value": "__rownum__"})
    assert r.status_code == 200, r.text
    assert r.json()["n_filled"] == len(df)
    out = store.get(sid)["CASE_ID"]
    assert pd.api.types.is_numeric_dtype(out)
    assert out.astype(int).tolist() == list(range(1, len(df) + 1))


def test_fill_blanks_column_not_found(client, synth):
    sid = _fresh(synth, "fill_miss")
    r = client.post(f"{BASE}/{sid}/fill_blanks",
                    json={"column": "NOPE", "value": "0"})
    assert r.status_code == 404, r.text


def test_fill_blanks_new_column_must_not_overwrite(client, synth):
    sid = _fresh(synth, "fill_duplicate_target")
    r = client.post(f"{BASE}/{sid}/fill_blanks",
                    json={"column": "LDL", "value": "__mean__", "new_column": "AGE"})
    assert r.status_code == 422, r.text


def test_delete_rows(client, synth):
    sid = _fresh(synth, "delrows")
    r = client.post(f"{BASE}/{sid}/delete_rows",
                    json={"row_indices": [0, 1, 2]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] == 3
    assert body["remaining_rows"] == 197


def test_delete_rows_out_of_range(client, synth):
    sid = _fresh(synth, "delrows_oor")
    r = client.post(f"{BASE}/{sid}/delete_rows",
                    json={"row_indices": [99999]})
    assert r.status_code == 422, r.text


def test_add_row(client, synth):
    sid = _fresh(synth, "addrow")
    r = client.post(f"{BASE}/{sid}/add_row", json={"position": -1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rows"] == 201


def test_add_column(client, synth):
    sid = _fresh(synth, "addcol")
    r = client.post(f"{BASE}/{sid}/add_column",
                    json={"name": "NEWCOL", "default_value": 0})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "NEWCOL"


def test_add_column_duplicate(client, synth):
    sid = _fresh(synth, "addcol_dup")
    r = client.post(f"{BASE}/{sid}/add_column",
                    json={"name": "AGE", "default_value": 0})
    assert r.status_code == 422, r.text


def test_paste_rows_append(client, synth):
    sid = _fresh(synth, "paste")
    tsv = "AGE,LDL\n55,130\n60,140"
    r = client.post(f"{BASE}/{sid}/paste",
                    json={"tsv": tsv, "has_header": True, "mode": "append"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_pasted"] == 2
    assert body["total_rows"] == 202


def test_paste_rows_empty(client, synth):
    sid = _fresh(synth, "paste_empty")
    r = client.post(f"{BASE}/{sid}/paste",
                    json={"tsv": "   ", "has_header": True})
    assert r.status_code == 422, r.text


def test_rename_column(client, synth):
    sid = _fresh(synth, "rename")
    r = client.post(f"{BASE}/{sid}/rename",
                    json={"old_name": "LDL", "new_name": "LDL_C"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["new_name"] == "LDL_C"


def test_rename_column_remaps_active_filter(client, synth):
    sid = _fresh(synth, "rename_filter")
    store.save_filter(sid, [
        {"column": "LDL", "operator": "gt", "value": "100", "join": "AND"},
    ])

    r = client.post(f"{BASE}/{sid}/rename",
                    json={"old_name": "LDL", "new_name": "LDL_C"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["case_filter"]["conditions"] == [
        {"column": "LDL_C", "operator": "gt", "value": "100", "join": "AND"},
    ]
    assert body["case_filter"]["selected"] <= body["case_filter"]["total"]


def test_rename_column_not_found(client, synth):
    sid = _fresh(synth, "rename_miss")
    r = client.post(f"{BASE}/{sid}/rename",
                    json={"old_name": "NOPE", "new_name": "X"})
    assert r.status_code == 404, r.text


def test_duplicate_column(client, synth):
    sid = _fresh(synth, "dup")
    r = client.post(f"{BASE}/{sid}/duplicate_column",
                    json={"column": "AGE"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "AGE_copy"


def test_duplicate_column_not_found(client, synth):
    sid = _fresh(synth, "dup_miss")
    r = client.post(f"{BASE}/{sid}/duplicate_column",
                    json={"column": "NOPE"})
    assert r.status_code == 404, r.text


def test_paste_cells(client, synth):
    sid = _fresh(synth, "paste_cells")
    r = client.post(f"{BASE}/{sid}/paste_cells",
                    json={"start_row": 0, "start_col": "AGE", "tsv": "50\t130\n55\t140"})
    assert r.status_code == 200, r.text
    assert r.json()["pasted"] >= 1


def test_paste_cells_bad_column(client, synth):
    sid = _fresh(synth, "paste_cells_bad")
    r = client.post(f"{BASE}/{sid}/paste_cells",
                    json={"start_row": 0, "start_col": "NOPE", "tsv": "1"})
    assert r.status_code == 400, r.text


def test_paste_cells_explicit_visible_targets(client, synth):
    sid = _fresh(synth, "paste_cells_targets")
    before = store.get(sid).copy()
    r = client.post(f"{BASE}/{sid}/paste_cells", json={
        "start_row": 0,
        "start_col": "AGE",
        "row_indices": [2, 0],
        "target_columns": ["LDL", "AGE"],
        "tsv": "111\t51\n222\t52",
    })
    assert r.status_code == 200, r.text
    after = store.get(sid)
    assert after.at[2, "LDL"] == 111
    assert after.at[2, "AGE"] == 51
    assert after.at[0, "LDL"] == 222
    assert after.at[0, "AGE"] == 52
    assert after.at[1, "LDL"] == before.at[1, "LDL"]


def test_unique_values(client, synth):
    sid = _fresh(synth, "unique")
    r = client.get(f"{BASE}/{sid}/unique/GROUP")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "values" in body
    assert set(body["values"]) <= {"A", "B", "C"}


def test_unique_values_not_found(client, synth):
    sid = _fresh(synth, "unique_miss")
    r = client.get(f"{BASE}/{sid}/unique/NOPE")
    assert r.status_code == 404, r.text


def test_transform_refuses_to_write_over_its_own_source(client):
    """Reported: applying a tertile replaced the variable it was computed from.

    The UI had no name prefix for tertile, quartile or median split, so the
    suggested "New column name" was the source column itself and one click
    destroyed the original values.
    """
    sid = make_session(
        pd.DataFrame({"Hasta no": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}),
        "tcomp_tf_self_overwrite",
    )
    r = client.post(
        f"{BASE}/{sid}/transform",
        json={"source_col": "Hasta no", "transform": "tertile", "new_col": "Hasta no"},
    )
    assert r.status_code == 422, r.text
    assert "source column" in r.json()["detail"]
    # The original values are still there.
    assert store.get(sid)["Hasta no"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_transform_still_writes_to_a_new_column(client):
    sid = make_session(pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), "tcomp_tf_ok")
    r = client.post(
        f"{BASE}/{sid}/transform",
        json={"source_col": "X", "transform": "tertile", "new_col": "Tert_X"},
    )
    assert r.status_code == 200, r.text
    df = store.get(sid)
    assert df["X"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert df["Tert_X"].dropna().min() >= 1
