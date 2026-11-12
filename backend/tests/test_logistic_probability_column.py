from urllib.parse import quote

import numpy as np
import pandas as pd
import statsmodels.api as sm

from conftest import make_session
from services import store


def _session(suffix):
    rng = np.random.default_rng(823)
    x = rng.normal(size=100)
    group = rng.choice(["case", "control"], size=100, p=[0.4, 0.6])
    frame = pd.DataFrame({
        "x": x,
        "event": group,
        "keep": np.arange(100) % 5 != 0,
        "site": np.where(np.arange(100) % 2, "A", "B"),
        "freq": np.arange(100) % 3 + 1,
        "event_predicted_probability": 0.5,
    })
    frame.loc[[7, 13], "x"] = np.nan
    return make_session(frame, f"logit_prob_{suffix}"), frame


def _fit(client, sid, headers=None, **options):
    response = client.post("/api/models/logistic", json={
        "session_id": sid, "outcome": "event", "predictors": ["x"],
        **options,
    }, headers={**(headers or {}), "X-Ustat-Save-Probabilities": "true"})
    assert response.status_code == 200, response.text
    return response.json()


def _save(client, sid, fit, headers=None):
    return client.post("/api/models/logistic/save_probabilities", json={
        "session_id": sid, "fit_token": fit["probability_fit"]["token"],
    }, headers=headers)


def test_save_probabilities_respects_filter_split_weights_and_undo(client):
    sid, original = _session("save")
    store.save_filter(sid, [{"column": "keep", "operator": "eq", "value": "1"}])
    store.save_weight(sid, "freq")
    headers = {"X-Ustat-Split": quote('{"column":"site","level":"A"}')}
    fit = _fit(client, sid, headers)
    assert fit["outcome_event_level"] == "control"
    response = _save(client, sid, fit, headers)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["name"] != "event_predicted_probability"
    saved = store.get(sid)
    assert saved is not None
    pd.testing.assert_frame_equal(saved[original.columns], original)

    rows = original.loc[original["keep"] & original["site"].eq("A") & original["x"].notna()]
    repeated = rows.loc[rows.index.repeat(rows["freq"])]
    model = sm.Logit(repeated["event"].eq("control"), sm.add_constant(repeated[["x"]])).fit(disp=False)
    expected = model.predict(sm.add_constant(rows[["x"]])).reindex(original.index)
    np.testing.assert_allclose(saved[result["name"]], expected, equal_nan=True)
    assert result["n_computed"] == len(rows)
    assert result["preview"][7][result["name"]] is None
    assert result["preview"][7]["site"] == original.loc[7, "site"]
    assert {"name": result["name"], "dtype": "float64", "kind": "numeric", "id_like": False} in result["columns"]

    dca = client.post("/api/decision_curve/dca", json={
        "session_id": sid, "outcome": "event", "probability_col": result["name"],
    }, headers=headers)
    assert dca.status_code == 200, dca.text
    assert dca.json()["mode"] == "precomputed"
    assert dca.json()["curves"]["model"]["net_benefit"]

    assert _save(client, sid, fit, headers).status_code == 409
    pd.testing.assert_frame_equal(store.undo(sid), original)
    pd.testing.assert_frame_equal(store.redo(sid), saved)


def test_save_probabilities_rejects_changed_scope_and_data(client):
    sid, original = _session("stale")
    other, _ = _session("other")
    fit = _fit(client, sid)
    assert _save(client, other, fit).status_code == 409

    store.save_filter(sid, [{"column": "site", "operator": "eq", "value": "A"}])
    assert _save(client, sid, fit).status_code == 409
    store.clear_filter(sid)

    newer = original.copy()
    newer.loc[0, "x"] += 1
    store.save(sid, newer)
    assert _save(client, sid, fit).status_code == 409
    pd.testing.assert_frame_equal(store.get(sid), newer)


def test_save_mice_probabilities_are_from_first_completed_fit(client):
    sid, original = _session("mice")
    fit = _fit(client, sid, imputation="mice")
    assert fit["probability_fit"]["source"] == "first imputed dataset"
    response = _save(client, sid, fit)
    assert response.status_code == 200, response.text
    assert response.json()["n_computed"] == len(original)
    assert original["x"].isna().sum() == store.get(sid)["x"].isna().sum()


def test_a_late_cell_save_does_not_overwrite_the_probability_column(client, monkeypatch):
    sid, original = _session("interleaved")
    fit = _fit(client, sid)
    original_save = store.save

    def save_after_probability_append(session_id, frame, *args, **kwargs):
        store.append_fitted_probabilities(sid, fit["probability_fit"]["token"])
        return original_save(session_id, frame, *args, **kwargs)

    monkeypatch.setattr(store, "save", save_after_probability_append)
    response = client.patch(f"/api/sessions/{sid}/cell", json={
        "row_index": 0, "column": "x", "value": "5",
    })
    assert response.status_code == 409, response.text
    saved = store.get(sid)
    assert saved is not None
    assert "event_predicted_probability_2" in saved.columns
    pd.testing.assert_frame_equal(saved[original.columns], original)
