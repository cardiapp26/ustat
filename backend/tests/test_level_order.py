"""Category order for order-aware methods, read from one place end to end.

services/level_order.py decides the low-to-high order for ordinal logistic
regression, Jonckheere-Terpstra and Cochran-Armitage: the request, then the
Data Dictionary, then numeric codes, then a known grading vocabulary. The
ordinal model refuses (422) when none applies instead of sorting the labels
alphabetically, which is what it used to do.
"""
import math

import numpy as np
import pandas as pd
import pytest
from conftest import make_session

from services import store
from services.project_file import build_project, parse_project, restore_project

# R 4.x, MASS::polr(factor(health, levels = c("Poor", "Fair", "Good"),
#                          ordered = TRUE) ~ x, Hess = TRUE) on _health() below.
POLR = {"beta": 1.2345114839, "se": 0.1586842564, "zeta": (-0.4754611098, 0.9456256148)}
# The same polr fit with the levels sorted alphabetically (Fair < Good < Poor),
# which is the model uSTAT used to fit for these labels: the effect of x
# comes out with the wrong sign.
POLR_ALPHABETICAL_BETA = -0.5755062505


def _health(labels=("Poor", "Fair", "Good"), seed=20260923, n=240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    lin = 1.1 * x + rng.logistic(0, 1, n)
    code = np.where(lin < -0.6, 0, np.where(lin < 0.9, 1, 2))
    return pd.DataFrame({
        "health": np.array(labels, dtype=object)[code],
        "x": np.round(x, 6),
        "event": (x > 0).astype(int),
    })


UNRECOGNISED = ("Trace", "Moderate", "Heavy")  # alphabetical order is exactly reversed


def _fit(client, sid, **extra):
    return client.post("/api/models/ordinal", json={
        "session_id": sid, "outcome": "health", "predictors": ["x"], **extra,
    })


def _set_dictionary_order(client, sid, column, order):
    r = client.post(f"/api/sessions/{sid}/metadata", json={
        "columns": {column: {"level_order": list(order)}},
    })
    assert r.status_code == 200, r.text


def _beta(body) -> float:
    return next(c for c in body["coefficients"] if c["variable"] == "x")["log_odds"]


def _assert_matches_polr(body, sign=1.0):
    cx = next(c for c in body["coefficients"] if c["variable"] == "x")
    assert cx["log_odds"] == pytest.approx(sign * POLR["beta"], rel=1e-4)
    assert cx["se"] == pytest.approx(POLR["se"], rel=1e-4)
    # statsmodels parameterises the second cut-point as log(zeta2 - zeta1).
    th = [t["coef"] for t in body["thresholds"]]
    if sign > 0:
        assert (th[0], th[0] + math.exp(th[1])) == pytest.approx(POLR["zeta"], rel=1e-4)


# ── Ordinal logistic regression ──────────────────────────────────────────────

def test_recognised_grading_labels_fit_the_polr_model(client):
    """Poor / Fair / Good, the reported case: ordered by meaning, matching R."""
    sid = make_session(_health(), "lvl_poor_fair_good")
    r = _fit(client, sid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["categories_in_rank_order"] == ["Poor", "Fair", "Good"]
    assert body["level_order_source"] == "recognised ordinal labels"
    _assert_matches_polr(body)
    assert _beta(body) > 0 > POLR_ALPHABETICAL_BETA
    assert "Poor < Fair < Good" in body["result_text"]


def test_text_categories_with_no_known_order_are_refused(client):
    sid = make_session(_health(UNRECOGNISED), "lvl_unknown_order")
    r = _fit(client, sid)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "Category order" in detail
    assert "alphabetically" in detail


def test_dictionary_order_is_used(client):
    sid = make_session(_health(UNRECOGNISED), "lvl_dictionary")
    _set_dictionary_order(client, sid, "health", UNRECOGNISED)
    r = _fit(client, sid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["categories_in_rank_order"] == list(UNRECOGNISED)
    assert body["level_order_source"] == "data dictionary"
    _assert_matches_polr(body)  # same codes as Poor/Fair/Good, same model
    assert "as set in the data dictionary" in body["result_text"]


def test_dictionary_order_overrides_numeric_codes(client):
    """Float codes 1.0/2.0/3.0 against the dictionary's "3", "2", "1"."""
    df = _health()
    df["health"] = df["health"].map({"Poor": 1.0, "Fair": 2.0, "Good": 3.0})
    sid = make_session(df, "lvl_numeric_reversed")
    by_code = _fit(client, sid).json()
    assert by_code["level_order_source"] == "numeric value"
    _assert_matches_polr(by_code)

    _set_dictionary_order(client, sid, "health", ["3", "2", "1"])
    reversed_ = _fit(client, sid).json()
    assert reversed_["level_order_source"] == "data dictionary"
    assert reversed_["categories_in_rank_order"] == ["3", "2", "1"]
    # Reversing a proportional-odds outcome negates the slope exactly.
    _assert_matches_polr(reversed_, sign=-1.0)


def test_request_order_overrides_dictionary(client):
    sid = make_session(_health(UNRECOGNISED), "lvl_request_wins")
    _set_dictionary_order(client, sid, "health", UNRECOGNISED[::-1])
    body = _fit(client, sid, level_order=list(UNRECOGNISED)).json()
    assert body["level_order_source"] == "request"
    _assert_matches_polr(body)


def test_stale_dictionary_order_is_refused_not_guessed(client):
    sid = make_session(_health(UNRECOGNISED), "lvl_stale")
    _set_dictionary_order(client, sid, "health", ["Trace", "Moderate"])
    r = _fit(client, sid)
    assert r.status_code == 422, r.text
    assert "Heavy" in r.json()["detail"]


def test_dictionary_level_missing_from_the_data_is_skipped(client):
    """A filter can remove a level; the order of the rest still holds."""
    sid = make_session(_health(UNRECOGNISED), "lvl_extra_level")
    _set_dictionary_order(client, sid, "health", ["None", *UNRECOGNISED])
    body = _fit(client, sid).json()
    assert body["categories_in_rank_order"] == list(UNRECOGNISED)
    _assert_matches_polr(body)


# ── Trend tests read the same order ──────────────────────────────────────────

def test_jonckheere_reads_the_dictionary_order(client):
    sid = make_session(_health(UNRECOGNISED), "lvl_jt")
    _set_dictionary_order(client, sid, "health", UNRECOGNISED)
    r = client.post("/api/stats/jonckheere_terpstra", json={
        "session_id": sid, "column": "x", "group_column": "health",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["level_order"] == list(UNRECOGNISED)
    assert body["level_order_source"] == "data dictionary"
    assert body["warnings"] == []
    assert body["z"] > 0  # x rises with the true order


def test_cochran_armitage_reads_the_dictionary_order(client):
    sid = make_session(_health(UNRECOGNISED), "lvl_ca")
    _set_dictionary_order(client, sid, "health", UNRECOGNISED)
    r = client.post("/api/categorical/cochran_armitage", json={
        "session_id": sid, "ordinal_col": "health", "event_col": "event",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert [row["level"] for row in body["summary"]["levels"]] == list(UNRECOGNISED)
    assert body["level_order_source"] == "data dictionary"
    assert body["summary"]["direction"] == "increasing"


# ── The order survives where metadata goes ───────────────────────────────────

def test_session_columns_carry_the_order(client):
    sid = make_session(_health(UNRECOGNISED), "lvl_columns")
    _set_dictionary_order(client, sid, "health", UNRECOGNISED)
    cols = client.get(f"/api/sessions/{sid}").json()["columns"]
    assert next(c for c in cols if c["name"] == "health")["level_order"] == list(UNRECOGNISED)


def test_project_file_round_trip_keeps_the_order():
    """dictionary.json is written with sorted keys; a list keeps its order."""
    order = ["10", "9", "2"]  # string-sorted this would read 10, 2, 9
    df = pd.DataFrame({"grade": [2, 9, 10] * 4, "x": np.arange(12.0)})
    sid = make_session(df, "lvl_project")
    store.save_metadata(sid, {"grade": {"level_order": order}})
    restored = restore_project(parse_project(build_project(sid)))
    assert store.get_metadata(restored)["grade"]["level_order"] == order
