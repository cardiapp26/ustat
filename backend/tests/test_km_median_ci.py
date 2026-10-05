"""Kaplan-Meier median survival 95% CI (matches lifelines median_survival_times)."""

import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from main import app
from services import store
from services.km_median_ci import km_median_ci

client = TestClient(app)


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    # Group A: events well before the end of follow-up, median reached.
    for t in rng.exponential(10, 60):
        rows.append({"time": float(t), "event": 1, "grp": "A"})
    # Group B: almost everything censored, median not reached.
    for t in rng.uniform(1, 20, 30):
        rows.append({"time": float(t), "event": 0, "grp": "B"})
    for t in (2.0, 3.0):
        rows.append({"time": t, "event": 1, "grp": "B"})
    return pd.DataFrame(rows)


def test_helper_matches_lifelines_directly():
    df = _data()
    sub = df[df.grp == "A"]
    kmf = KaplanMeierFitter().fit(sub["time"], sub["event"])
    ref = median_survival_times(kmf.confidence_interval_)
    lo, hi = km_median_ci(kmf)
    assert lo == pytest.approx(float(ref.iloc[0, 0]))
    assert hi == pytest.approx(float(ref.iloc[0, 1]))
    assert lo < kmf.median_survival_time_ < hi


def test_helper_none_when_not_reached():
    df = _data()
    sub = df[df.grp == "B"]
    kmf = KaplanMeierFitter().fit(sub["time"], sub["event"])
    assert math.isinf(kmf.median_survival_time_)
    lo, hi = km_median_ci(kmf)
    assert hi is None


def test_km_endpoint_median_ci_fields():
    df = _data()
    store.save("km_med_ci", df)
    r = client.post(
        "/api/models/survival/km",
        json={"session_id": "km_med_ci", "duration_col": "time", "event_col": "event", "group_col": "grp"},
    )
    assert r.status_code == 200, r.text
    groups = {g["group"]: g for g in r.json()["groups"]}

    sub = df[df.grp == "A"]
    kmf = KaplanMeierFitter().fit(sub["time"], sub["event"])
    ref = median_survival_times(kmf.confidence_interval_)
    a = groups["A"]
    assert a["median_survival_ci_low"] == pytest.approx(float(ref.iloc[0, 0]))
    assert a["median_survival_ci_high"] == pytest.approx(float(ref.iloc[0, 1]))
    assert a["median_survival_ci_low"] < a["median_survival"] < a["median_survival_ci_high"]

    b = groups["B"]
    assert b["median_survival"] is None
    assert b["median_survival_ci_high"] is None


def test_landmark_summary_has_median_ci():
    rng = np.random.default_rng(2)
    n = 150
    df = pd.DataFrame(
        {
            "time": rng.exponential(10, n),
            "event": rng.integers(0, 2, n),
            "grp": rng.choice(["a", "b"], n),
        }
    )
    store.save("km_med_lm", df)
    r = client.post(
        "/api/survival_advanced/landmark",
        json={
            "session_id": "km_med_lm", "duration_col": "time", "event_col": "event",
            "landmark_time": 1.0, "group_col": "grp", "predictors": [],
        },
    )
    assert r.status_code == 200, r.text
    rows = r.json()["export_rows"]
    assert rows[0][-2:] == ["Median 95% CI low", "Median 95% CI high"]
    assert len(rows) == 3
