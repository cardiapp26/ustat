"""Bar heights as a percentage of each group, not a mean of a 0/1 column.

"What fraction of this tertile was malignant" is the question a risk-factor
figure asks. A mean over a 0/1 outcome answers it arithmetically but reports
0.37 where the figure needs 37% — a rescale the caller has to remember, and
label, on their own.
"""
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def sid() -> str:
    # 42 per tertile; 15, 17 and 16 malignant.
    df = pd.DataFrame({
        "SII_T": [1.0] * 42 + [2.0] * 42 + [3.0] * 42,
        "Malign": ([1.0] * 15 + [0.0] * 27) + ([1.0] * 17 + [0.0] * 25) + ([1.0] * 16 + [0.0] * 26),
        "Histology": (["malignant"] * 15 + ["benign"] * 27)
                     + (["malignant"] * 17 + ["benign"] * 25)
                     + (["malignant"] * 16 + ["benign"] * 26),
    })
    return make_session(df, "bar_pct")


def _bar(client, sid, **kw):
    r = client.post("/api/charts/bar", json={"session_id": sid, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_percentage_of_each_group(client, sid):
    out = _bar(client, sid, x="SII_T", y="Malign", y_mode="percentage")
    assert out["y_mode"] == "percentage"
    assert [d["value"] for d in out["data"]] == [35.7, 40.5, 38.1]


def test_the_denominator_travels_with_the_percentage(client, sid):
    """37% of 8 and 37% of 800 are the same bar and not the same finding."""
    out = _bar(client, sid, x="SII_T", y="Malign", y_mode="percentage")
    assert [(d["k"], d["n"]) for d in out["data"]] == [(15, 42), (17, 42), (16, 42)]


def test_a_named_target_value(client, sid):
    """A categorical outcome names the level that counts, rather than needing
    to be recoded to 0/1 first."""
    out = _bar(client, sid, x="SII_T", y="Histology",
               y_mode="percentage", target_value="malignant")
    assert [d["value"] for d in out["data"]] == [35.7, 40.5, 38.1]


def test_the_target_matches_across_float_spellings(client, sid):
    """The outcome column is float64, so its levels stringify as "1.0" while a
    user types "1" — the same mismatch that hid value labels everywhere."""
    out = _bar(client, sid, x="SII_T", y="Malign", y_mode="percentage", target_value="1")
    assert [d["value"] for d in out["data"]] == [35.7, 40.5, 38.1]


def test_mean_mode_is_unchanged(client, sid):
    out = _bar(client, sid, x="SII_T", y="Malign")
    assert out["y_mode"] == "mean"
    assert out["data"][0]["value"] == pytest.approx(15 / 42)


def test_count_mode_is_unchanged(client, sid):
    out = _bar(client, sid, x="SII_T")
    assert out["y_mode"] == "count"
    assert sorted(d["value"] for d in out["data"]) == [42, 42, 42]


def test_group_labels_use_the_value_label_key(client, sid):
    """Float codes arrive as "1", not "1.0", so the client's label lookup
    resolves them."""
    out = _bar(client, sid, x="SII_T", y="Malign", y_mode="percentage")
    assert [d["label"] for d in out["data"]] == ["1", "2", "3"]


def test_an_unknown_mode_is_refused(client, sid):
    r = client.post("/api/charts/bar", json={
        "session_id": sid, "x": "SII_T", "y": "Malign", "y_mode": "median",
    })
    assert r.status_code == 400
    assert "y_mode" in r.json()["detail"]


def test_rows_missing_the_outcome_leave_the_denominator(client):
    """A blank outcome is not a negative one — counting it as such would
    understate every percentage."""
    df = pd.DataFrame({
        "grp": ["A"] * 4 + ["B"] * 4,
        "out": [1.0, 0.0, None, None, 1.0, 1.0, 0.0, 0.0],
    })
    sid = make_session(df, "bar_pct_missing")
    out = _bar(client, sid, x="grp", y="out", y_mode="percentage")
    by = {d["label"]: d for d in out["data"]}
    assert by["A"]["n"] == 2 and by["A"]["value"] == 50.0
    assert by["B"]["n"] == 4 and by["B"]["value"] == 50.0


# ── the grouping column has to reach the client ───────────────────────────────

def test_boxplot_reports_the_grouping_column(client, sid):
    """The client resolves each group's value labels from this name. Without
    it the lookup ran against an empty map, so a labelled histology column
    drew its raw codes on every box, violin, raincloud and strip chart."""
    r = client.post("/api/charts/boxplot", json={
        "session_id": sid, "x": "SII_T", "color": "Histology",
    })
    assert r.status_code == 200, r.text
    assert r.json()["color"] == "Histology"


def test_boxplot_without_a_group_says_so(client, sid):
    r = client.post("/api/charts/boxplot", json={"session_id": sid, "x": "SII_T"})
    assert r.status_code == 200, r.text
    assert r.json()["color"] is None


# ── splitting by a second column ──────────────────────────────────────────────

def test_a_grouping_column_splits_the_bars(client, sid):
    """Reported: the Color / Group selector did nothing on a bar chart. The
    request carried the column and the handler never read it."""
    r = client.post("/api/charts/bar", json={
        "session_id": sid, "x": "SII_T", "y": "Malign",
        "y_mode": "percentage", "color": "Histology",
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["color"] == "Histology"
    assert {s["group"] for s in out["series"]} == {"benign", "malignant"}
    # Every malignant row is malignant, and no benign row is.
    by = {s["group"]: s for s in out["series"]}
    assert all(d["value"] == 100.0 for d in by["malignant"]["data"])
    assert all(d["value"] == 0.0 for d in by["benign"]["data"])


def test_grouping_works_without_a_y_column(client, sid):
    out = client.post("/api/charts/bar", json={
        "session_id": sid, "x": "SII_T", "color": "Histology",
    }).json()
    assert out["y_mode"] == "count"
    by = {s["group"]: [d["value"] for d in s["data"]] for s in out["series"]}
    assert by["malignant"] == [15, 17, 16]


def test_an_unknown_grouping_column_is_refused(client, sid):
    r = client.post("/api/charts/bar", json={
        "session_id": sid, "x": "SII_T", "color": "NOPE",
    })
    assert r.status_code == 400
    assert "NOPE" in r.json()["detail"]


def test_bars_are_ordered_by_level_not_by_frequency(client, sid):
    """value_counts orders by count, so a tertile axis came out 3, 1, 2 — and
    in a grouped chart two series could order their bars differently from
    each other, which makes them uncomparable."""
    out = client.post("/api/charts/bar", json={"session_id": sid, "x": "SII_T"}).json()
    assert [d["label"] for d in out["data"]] == ["1", "2", "3"]

    grouped = client.post("/api/charts/bar", json={
        "session_id": sid, "x": "SII_T", "color": "Histology",
    }).json()
    orders = {tuple(d["label"] for d in s["data"]) for s in grouped["series"]}
    assert orders == {("1", "2", "3")}


def test_ungrouped_bars_keep_the_flat_shape(client, sid):
    """Without a grouping column the response stays a single `data` list, so
    nothing that already consumed it has to learn about `series`."""
    out = client.post("/api/charts/bar", json={"session_id": sid, "x": "SII_T"}).json()
    assert "data" in out and "series" not in out
