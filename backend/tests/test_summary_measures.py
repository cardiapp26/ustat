"""Summary-measures analysis of serial data (Matthews et al. 1990)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp

from conftest import make_session
from services import summary_measures as sm

URL = "/api/summary_measures/compute"
ALL = list(sm.VALID_MEASURES)


def _long(rows):
    return pd.DataFrame(rows, columns=["id", "day", "y", "arm"])


def _loop_auc(t, y):
    total = 0.0
    for i in range(1, len(t)):
        total += (t[i] - t[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return total


def _grouped_frame(seed=1, per_group=(8, 9)):
    rng = np.random.default_rng(seed)
    rows, sid = [], 0
    for gi, n in enumerate(per_group):
        for _ in range(n):
            sid += 1
            base = rng.normal(10 + 2 * gi, 1.5)
            slope = rng.normal(0.05 * (gi + 1), 0.02)
            for d in (0, 14, 30, 90):
                rows.append((f"s{sid}", d, base + slope * d + rng.normal(0, 0.4), f"g{gi}"))
    return _long(rows)


def _post(client, df, sid, **kw):
    make_session(df, sid)
    body = {"session_id": sid, "subject_col": "id", "time_col": "day", "value_col": "y"}
    body.update(kw)
    return client.post(URL, json=body)


# ───────────────────────────── service level ─────────────────────────────

def test_hand_computed_auc_family():
    t = np.array([0.0, 14.0, 30.0, 90.0])
    y = np.array([10.0, 12.0, 9.0, 7.0])
    # 14*(10+12)/2 + 16*(12+9)/2 + 60*(9+7)/2 = 154 + 168 + 480
    assert sm.trapezoid_auc(t, y) == pytest.approx(802.0)
    assert _loop_auc(t, y) == pytest.approx(802.0)
    res = sm.subject_measures(t, y, ALL)["values"]
    assert res["auc"] == pytest.approx(802.0)
    assert res["auc_per_time"] == pytest.approx(802.0 / 90.0)
    assert res["auc_above_baseline"] == pytest.approx(802.0 - 10.0 * 90.0)
    assert res["peak"] == 12.0
    assert res["time_to_peak"] == 14.0
    assert res["mean"] == pytest.approx(9.5)
    assert res["first"] == 10.0 and res["last"] == 7.0
    assert res["change_last_first"] == -3.0


def test_slope_matches_polyfit():
    t = np.array([0.0, 14.0, 30.0, 90.0, 120.0])
    y = np.array([3.1, 4.0, 4.4, 7.9, 9.1])
    res = sm.subject_measures(t, y, ["slope"])["values"]
    assert res["slope"] == pytest.approx(np.polyfit(t, y, 1)[0], rel=1e-10)


def test_peak_tmax_first_occurrence_on_tie():
    res = sm.subject_measures(np.array([0.0, 1.0, 2.0]), np.array([5.0, 9.0, 9.0]),
                              ["peak", "time_to_peak"])["values"]
    assert res["peak"] == 9.0 and res["time_to_peak"] == 1.0


def test_min_points_gates_auc_and_slope_but_not_peak():
    r = sm.subject_measures(np.array([5.0]), np.array([3.0]), ALL)["values"]
    assert r["peak"] == 3.0 and r["time_to_peak"] == 5.0 and r["mean"] == 3.0
    for m in ("auc", "auc_per_time", "auc_above_baseline", "slope", "change_last_first"):
        assert np.isnan(r[m])
    three = sm.subject_measures(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]),
                                ["auc", "slope"], min_points=4)["values"]
    assert np.isnan(three["auc"]) and np.isnan(three["slope"])


def test_duplicate_times_averaged():
    t, y, dup = sm.collapse_duplicate_times(np.array([30.0, 0.0, 14.0, 14.0]),
                                            np.array([9.0, 10.0, 12.0, 14.0]))
    assert dup
    assert t.tolist() == [0.0, 14.0, 30.0]
    assert y.tolist() == [10.0, 13.0, 9.0]


def test_interpolate_fills_interior_gaps_only():
    grid = np.array([0.0, 10.0, 20.0, 30.0])
    t = np.array([0.0, 30.0])
    y = np.array([0.0, 30.0])
    r = sm.subject_measures(t, y, ["auc", "mean", "slope"], missing="interpolate", grid=grid)
    assert r["n_interpolated"] == 2
    assert r["values"]["mean"] == pytest.approx(15.0)       # 0, 10, 20, 30
    assert r["values"]["auc"] == pytest.approx(450.0)       # unchanged by linear interpolation
    assert r["values"]["slope"] == pytest.approx(1.0)
    om = sm.subject_measures(t, y, ["mean"], missing="omit", grid=grid)
    assert om["values"]["mean"] == pytest.approx(15.0) and om["n_interpolated"] == 0
    # No extrapolation beyond the subject's own first and last visit.
    r2 = sm.subject_measures(np.array([10.0, 20.0]), np.array([1.0, 2.0]), ["mean"],
                             missing="interpolate", grid=grid)
    assert r2["n_interpolated"] == 0


# ───────────────────────────── endpoint ─────────────────────────────

def test_endpoint_single_subject_values_and_shape(client):
    rows = []
    for sid, ys in (("a", [10, 12, 9, 7]), ("b", [8, 9, 11, 12])):
        rows += [(sid, d, v, "x") for d, v in zip((0, 14, 30, 90), ys)]
    r = _post(client, _long(rows), "sm_shape", measures=ALL)
    assert r.status_code == 200, r.text
    j = r.json()
    for key in ("subjects", "descriptives", "comparisons", "summary", "warnings",
                "result_text", "methods_text", "r_code", "export_rows", "measures"):
        assert key in j
    a = next(s for s in j["subjects"] if s["subject"] == "a")
    assert a["n_points"] == 4
    assert a["auc"] == pytest.approx(802.0)
    assert a["auc_per_time"] == pytest.approx(802.0 / 90.0)
    assert a["auc_above_baseline"] == pytest.approx(-98.0)
    assert a["time_to_peak"] == 14.0
    b = next(s for s in j["subjects"] if s["subject"] == "b")
    assert b["auc"] == pytest.approx(_loop_auc([0, 14, 30, 90], [8, 9, 11, 12]))
    assert j["comparisons"] == []
    assert "Matthews" in j["methods_text"] and "300:230" in j["methods_text"]
    assert not any(chr(0x2014) in str(v) for v in (j["result_text"], j["methods_text"], j["r_code"]))


def test_endpoint_slope_peak_vs_numpy(client):
    df = _grouped_frame(seed=3)
    r = _post(client, df, "sm_np", measures=["slope", "peak", "time_to_peak", "auc"], group_col="arm")
    assert r.status_code == 200
    by_id = {s["subject"]: s for s in r.json()["subjects"]}
    for sid, sub in df.groupby("id"):
        sub = sub.sort_values("day")
        t, y = sub["day"].to_numpy(float), sub["y"].to_numpy(float)
        row = by_id[sid]
        assert row["slope"] == pytest.approx(np.polyfit(t, y, 1)[0], rel=1e-8)
        assert row["peak"] == pytest.approx(y.max())
        assert row["time_to_peak"] == t[np.argmax(y)]
        assert row["auc"] == pytest.approx(_loop_auc(t, y))


def test_two_group_welch_and_mannwhitney_match_scipy(client):
    df = _grouped_frame(seed=5, per_group=(9, 11))
    r = _post(client, df, "sm_two", measures=["auc_per_time"], group_col="arm", alpha=0.05)
    assert r.status_code == 200
    j = r.json()
    subj = pd.DataFrame(j["subjects"])
    a = subj.loc[subj.group == "g0", "auc_per_time"].to_numpy(float)
    b = subj.loc[subj.group == "g1", "auc_per_time"].to_numpy(float)
    c = j["comparisons"][0]
    ref = sp.ttest_ind(a, b, equal_var=False)
    assert c["test"] == "Welch t-test"
    assert c["p"] == pytest.approx(ref.pvalue, rel=1e-9)
    assert c["t"] == pytest.approx(ref.statistic, rel=1e-9)
    ci = ref.confidence_interval(0.95)
    assert c["mean_difference"] == pytest.approx(a.mean() - b.mean())
    assert c["ci_low"] == pytest.approx(ci.low, rel=1e-9)
    assert c["ci_high"] == pytest.approx(ci.high, rel=1e-9)
    assert c["p_nonparametric"] == pytest.approx(sp.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)
    sp_ = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    assert c["effect_size"]["value"] == pytest.approx((a.mean() - b.mean()) / sp_)
    assert "normality" in c and c["normality"]["p"] is not None
    assert "sensitivity" in j["result_text"] and "Welch" in j["result_text"]
    desc = {d["group"]: d for d in j["descriptives"]}
    assert desc["g0"]["n"] == 9 and desc["g1"]["n"] == 11
    assert desc["g0"]["mean"] == pytest.approx(a.mean())
    assert desc["g1"]["iqr"] == pytest.approx(np.percentile(b, 75) - np.percentile(b, 25))
    assert "t.test(auc_per_time ~ grp" in j["r_code"]


def test_three_group_anova_and_kruskal(client):
    df = _grouped_frame(seed=7, per_group=(6, 7, 8))
    r = _post(client, df, "sm_three", measures=["auc", "slope"], group_col="arm")
    assert r.status_code == 200
    j = r.json()
    subj = pd.DataFrame(j["subjects"])
    for c in j["comparisons"]:
        m = c["measure"]
        arrs = [subj.loc[subj.group == g, m].to_numpy(float) for g in ("g0", "g1", "g2")]
        f = sp.f_oneway(*arrs)
        assert c["test"] == "One-way ANOVA"
        assert c["f"] == pytest.approx(f.statistic, rel=1e-9)
        assert c["p"] == pytest.approx(f.pvalue, rel=1e-9)
        assert c["p_nonparametric"] == pytest.approx(sp.kruskal(*arrs).pvalue, rel=1e-9)
        allv = np.concatenate(arrs)
        ssb = sum(len(a) * (a.mean() - allv.mean()) ** 2 for a in arrs)
        assert c["effect_size"]["value"] == pytest.approx(ssb / np.sum((allv - allv.mean()) ** 2))
        assert (c["df_between"], c["df_within"]) == (2, 18)
    assert "ANOVA" in j["result_text"] and "Kruskal" in j["result_text"]
    assert "aov(" in j["r_code"] and "kruskal.test(" in j["r_code"]


def test_missing_points_trapezoid_over_observed(client):
    rows = [("a", 0, 10.0, "x"), ("a", 14, np.nan, "x"), ("a", 30, 9.0, "x"), ("a", 90, 7.0, "x"),
            ("b", 0, 8.0, "x"), ("b", 14, 9.0, "x"), ("b", 30, 11.0, "x"), ("b", 90, 12.0, "x")]
    r = _post(client, _long(rows), "sm_miss", measures=["auc", "slope", "mean"])
    assert r.status_code == 200
    j = r.json()
    a = next(s for s in j["subjects"] if s["subject"] == "a")
    assert a["n_points"] == 3
    assert a["auc"] == pytest.approx(_loop_auc([0, 30, 90], [10, 9, 7]))
    assert a["slope"] == pytest.approx(np.polyfit([0, 30, 90], [10, 9, 7], 1)[0])
    assert any("missing value" in w for w in j["warnings"])
    # interpolate: AUC unchanged, mean and slope use the filled grid point
    r2 = _post(client, _long(rows), "sm_miss", measures=["auc", "mean"], missing="interpolate")
    a2 = next(s for s in r2.json()["subjects"] if s["subject"] == "a")
    assert a2["auc"] == pytest.approx(a["auc"])
    assert a2["n_interpolated"] == 1
    filled = np.interp(14, [0, 30, 90], [10, 9, 7])
    assert a2["mean"] == pytest.approx(np.mean([10, filled, 9, 7]))


def test_duplicate_times_endpoint_warns_and_averages(client):
    rows = [("a", 0, 10.0, "x"), ("a", 14, 12.0, "x"), ("a", 14, 14.0, "x"), ("a", 30, 9.0, "x"),
            ("b", 0, 8.0, "x"), ("b", 14, 9.0, "x"), ("b", 30, 11.0, "x")]
    j = _post(client, _long(rows), "sm_dup", measures=["auc"]).json()
    a = next(s for s in j["subjects"] if s["subject"] == "a")
    assert a["n_points"] == 3
    assert a["auc"] == pytest.approx(_loop_auc([0, 14, 30], [10, 13, 9]))
    assert any("Duplicate" in w for w in j["warnings"])


def test_too_few_points_listed_in_warnings(client):
    rows = [("a", 0, 10.0, "x"), ("a", 14, 12.0, "x"), ("a", 30, 9.0, "x"),
            ("b", 0, 8.0, "x"), ("b", 14, 9.0, "x"), ("b", 30, 11.0, "x"),
            ("c", 0, 5.0, "x")]
    j = _post(client, _long(rows), "sm_few", measures=["auc", "slope", "peak"]).json()
    c = next(s for s in j["subjects"] if s["subject"] == "c")
    assert c["n_points"] == 1 and c["auc"] is None and c["slope"] is None and c["peak"] == 5.0
    assert any("c (n = 1)" in w for w in j["warnings"])
    d = next(x for x in j["descriptives"] if x["measure"] == "auc")
    assert d["n"] == 2
    j4 = _post(client, _long(rows), "sm_few", measures=["auc"], min_points=3).json()
    assert next(s for s in j4["subjects"] if s["subject"] == "a")["auc"] is not None


def test_validation_errors(client):
    df = _grouped_frame()
    assert _post(client, df, "sm_v", measures=["auc", "bogus"]).status_code == 422
    assert _post(client, df, "sm_v", missing="locf").status_code == 422
    assert _post(client, df, "sm_v", value_col="nope").status_code == 422
    assert _post(client, df, "sm_v", time_col="arm").status_code == 422        # non-numeric time
    assert _post(client, df, "sm_v", value_col="arm").status_code == 422       # non-numeric value
    assert _post(client, df, "sm_v", group_col="nope").status_code == 422
    r = client.post(URL, json={"session_id": "no_such_session_xyz", "subject_col": "id",
                               "time_col": "day", "value_col": "y"})
    assert r.status_code == 404


def test_validation_subject_and_group_counts(client):
    one = _long([("a", 0, 1.0, "x"), ("a", 1, 2.0, "x")])
    assert _post(client, one, "sm_one").status_code == 422
    rows = [("a", 0, 1.0, "g0"), ("a", 1, 2.0, "g0"), ("b", 0, 1.0, "g0"), ("b", 1, 3.0, "g0"),
            ("c", 0, 2.0, "g1"), ("c", 1, 2.5, "g1")]
    r = _post(client, _long(rows), "sm_grp", group_col="arm")
    assert r.status_code == 422 and "at least 2 subjects" in r.json()["detail"]
    mixed = [("a", 0, 1.0, "g0"), ("a", 1, 2.0, "g1"), ("b", 0, 1.0, "g0"), ("b", 1, 3.0, "g0")]
    r = _post(client, _long(mixed), "sm_mix", group_col="arm")
    assert r.status_code == 422 and "more than one group" in r.json()["detail"]


def test_default_measures_and_export_rows(client):
    j = _post(client, _grouped_frame(), "sm_def", group_col="arm").json()
    assert j["measures"] == ["auc_per_time", "peak", "time_to_peak", "slope"]
    assert len(j["comparisons"]) == 4
    flat = [c for row in j["export_rows"] for c in row]
    assert "Welch t-test" in flat
    assert all(len(r) == len(j["export_rows"][0]) or len(r) == 10 for r in j["export_rows"])


def test_constant_measure_does_not_break_response(client):
    # Every subject has the same peak and time to peak: zero variance in a measure.
    rows = []
    for g, sids in (("g0", ("a", "b", "c")), ("g1", ("d", "e", "f"))):
        for s in sids:
            rows += [(s, 0, 1.0, g), (s, 10, 5.0, g), (s, 20, 2.0, g)]
    r = _post(client, _long(rows), "sm_const", measures=["peak", "time_to_peak"], group_col="arm")
    assert r.status_code == 200
    for c in r.json()["comparisons"]:
        assert c["p"] is None
