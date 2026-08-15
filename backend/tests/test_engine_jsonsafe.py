"""Pin `engine.jsonsafe.sanitize`.

`sanitize` replaced two near-identical helpers that used to live in the
routers: `_sanitize` in `routers/stats/inferential.py` (recursed over
dict/list, turned non-finite floats into `None`, but left numpy scalars
alone) and `_safe` in `routers/ml.py` (unwrapped numpy scalars and caught
non-finite floats, but only at the top level -- it never recursed into a
dict or list). Each endpoint got whichever subset its author happened to
copy, so whether a NaN reached the client as `null` or as invalid JSON
depended on which router handled the request.

`sanitize` is meant to be the union: recursive AND numpy-aware, everywhere.
These tests pin that union so a future edit cannot quietly drop either half.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ustat_engine.jsonsafe import sanitize


# ── non-finite floats -> None, at any depth ─────────────────────────────────

@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_becomes_none_at_top_level(value):
    assert sanitize(value) is None


def test_non_finite_floats_become_none_nested_in_dict():
    out = sanitize({"a": float("nan"), "b": {"c": float("inf")}})
    assert out == {"a": None, "b": {"c": None}}


def test_non_finite_floats_become_none_nested_in_list():
    out = sanitize([1.0, float("nan"), [float("-inf"), 2.0]])
    assert out == [1.0, None, [None, 2.0]]


# ── numpy scalars unwrap to the matching Python type ────────────────────────

def test_numpy_int64_unwraps_to_python_int():
    out = sanitize(np.int64(7))
    assert type(out) is int
    assert out == 7


def test_numpy_int32_unwraps_to_python_int():
    out = sanitize(np.int32(7))
    assert type(out) is int
    assert out == 7


def test_numpy_float64_unwraps_to_python_float():
    out = sanitize(np.float64(1.5))
    assert type(out) is float
    assert out == 1.5


def test_numpy_float32_unwraps_to_python_float():
    out = sanitize(np.float32(1.5))
    assert type(out) is float
    assert out == pytest.approx(1.5)


def test_numpy_bool_unwraps_to_python_bool():
    # np.bool_(True) == True, but type(np.bool_(True)) is not bool and it is
    # not JSON-serialisable as a plain Python bool. An equality-only
    # assertion would pass even if sanitize left the numpy type untouched, so
    # the type itself is what has to be checked here.
    out = sanitize(np.bool_(True))
    assert type(out) is bool
    assert out is True

    out_false = sanitize(np.bool_(False))
    assert type(out_false) is bool
    assert out_false is False


def test_numpy_nan_float_becomes_none_after_unwrapping():
    # This has to unwrap the np.float64 to a Python float *and then* hit the
    # non-finite branch -- not get released early because it "looks like"
    # a numpy type the non-finite check doesn't recognise.
    out = sanitize(np.float64("nan"))
    assert out is None


def test_numpy_inf_float_becomes_none_after_unwrapping():
    out = sanitize(np.float64("inf"))
    assert out is None


# ── numpy arrays become plain, sanitized lists ──────────────────────────────

def test_numpy_1d_array_becomes_list():
    out = sanitize(np.array([1, 2, 3], dtype=np.int64))
    assert out == [1, 2, 3]
    assert isinstance(out, list)
    assert all(type(v) is int for v in out)


def test_numpy_array_with_nan_sanitizes_contents():
    out = sanitize(np.array([1.0, np.nan, 3.0]))
    assert out == [1.0, None, 3.0]


def test_numpy_2d_array_becomes_nested_list():
    arr = np.array([[1.0, np.nan], [np.inf, 4.0]])
    out = sanitize(arr)
    assert out == [[1.0, None], [None, 4.0]]
    assert isinstance(out, list)
    assert isinstance(out[0], list)


# ── tuples, sets and frozensets become lists ────────────────────────────────

def test_tuple_becomes_list():
    out = sanitize((1, 2, float("nan")))
    assert out == [1, 2, None]
    assert isinstance(out, list)


def test_set_becomes_list():
    out = sanitize({1, 2, 3})
    assert isinstance(out, list)
    assert sorted(out) == [1, 2, 3]


def test_frozenset_becomes_list():
    out = sanitize(frozenset({1, 2, 3}))
    assert isinstance(out, list)
    assert sorted(out) == [1, 2, 3]


# ── ordinary values pass through untouched ──────────────────────────────────

def test_string_passes_through():
    assert sanitize("hello") == "hello"


def test_none_passes_through():
    assert sanitize(None) is None


def test_int_passes_through():
    out = sanitize(5)
    assert out == 5
    assert type(out) is int


def test_finite_float_passes_through():
    out = sanitize(3.14)
    assert out == 3.14
    assert type(out) is float


def test_bool_stays_bool_not_int():
    # bool is a subclass of int in Python; the int branch (`isinstance(obj,
    # (str, bool, int))`) must return it as-is rather than letting it fall
    # through to something that would coerce it.
    out_true = sanitize(True)
    assert out_true is True
    assert type(out_true) is bool

    out_false = sanitize(False)
    assert out_false is False
    assert type(out_false) is bool


# ── the assertion that actually matters: allow_nan=False acceptance ────────

def test_sanitized_realistic_structure_survives_strict_json_dumps():
    """This is the whole point of the function.

    `json.dumps(..., allow_nan=False)` is what a browser's `JSON.parse`
    effectively enforces -- Python's default `json.dumps` happily emits
    `NaN`/`Infinity` tokens that are not valid JSON at all, so a bug that
    leaked a non-finite float or a numpy scalar past `sanitize` would still
    look "fine" to a lenient encoder while breaking every strict consumer.
    """
    raw = {
        "scalar_nan": float("nan"),
        "scalar_inf": float("inf"),
        "scalar_neg_inf": float("-inf"),
        "np_int": np.int64(42),
        "np_float": np.float32(2.5),
        "np_bool": np.bool_(True),
        "np_array": np.array([1.0, np.nan, np.inf, 4.0]),
        "np_array_2d": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "nested": {
            "list_of_things": [1, float("nan"), np.float64(3.3), (1, 2, 3)],
            "a_set": {1, 2},
        },
        "ordinary": {"str": "ok", "bool": False, "none": None, "float": 1.25},
    }

    # The original, un-sanitized structure is not valid strict JSON: it
    # contains NaN/Infinity tokens and objects json.dumps cannot serialise at
    # all (numpy scalars, numpy arrays, sets) without a custom encoder.
    with pytest.raises((ValueError, TypeError)):
        json.dumps(raw, allow_nan=False)

    cleaned = sanitize(raw)

    # Must round-trip through the strictest possible encoder call.
    encoded = json.dumps(cleaned, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["scalar_nan"] is None
    assert decoded["scalar_inf"] is None
    assert decoded["scalar_neg_inf"] is None
    assert decoded["np_int"] == 42
    assert decoded["np_float"] == pytest.approx(2.5)
    assert decoded["np_bool"] is True
    assert decoded["np_array"] == [1.0, None, None, 4.0]
    assert decoded["np_array_2d"] == [[1, 2], [3, 4]]
    assert decoded["nested"]["list_of_things"] == [1, None, pytest.approx(3.3), [1, 2, 3]]
    assert sorted(decoded["nested"]["a_set"]) == [1, 2]
    assert decoded["ordinary"] == {"str": "ok", "bool": False, "none": None, "float": 1.25}


def test_math_isnan_helper_agrees_with_sanitize_boundary():
    # Sanity check on the boundary itself, independent of sanitize: makes
    # the "non-finite" branch's own definition explicit for readers of this
    # test file rather than assumed.
    assert math.isnan(float("nan"))
    assert math.isinf(float("inf"))
    assert sanitize(float("nan")) is None
    assert sanitize(float("inf")) is None
