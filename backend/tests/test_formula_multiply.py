"""The formula builder has to be able to multiply two columns.

simpleeval replaces `*` with safe_mult, whose guard against `"x" * 10**9`
reads `b * len(a) > MAX_STRING_LENGTH`. For two columns that comparison is a
boolean Series and `and` cannot reduce it, so every product of two columns
failed with "the truth value of a Series is ambiguous" — while a + b, a - b
and a / b all worked, which is why it went unnoticed.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from services import store


@pytest.fixture()
def sid(client):
    store.save("fx", pd.DataFrame({
        "Yas": [60, 45, 72],
        "AST": [40, 80, 25],
        "ALT": [50, 30, 60],
        "Trombosit": [200, 150, 95],
    }))
    return "fx"


def _formula(client, sid, name, formula):
    r = client.post(f"/api/compute/{sid}/formula",
                    json={"new_col": name, "formula": formula})
    assert r.status_code == 200, r.text
    return r.json()


def test_two_columns_can_be_multiplied(client, sid):
    j = _formula(client, sid, "prod", "Yas * AST")
    assert j["preview_values"] == [2400, 3600, 1800]


def test_fib4(client, sid):
    """FIB-4 = (age x AST) / (platelets x sqrt(ALT)). The worked example in
    every reference: age 60, AST 40, ALT 50, platelets 200 -> 1.70."""
    j = _formula(client, sid, "FIB4", "(Yas * AST) / (Trombosit * SQRT(ALT))")
    expected = [
        60 * 40 / (200 * math.sqrt(50)),
        45 * 80 / (150 * math.sqrt(30)),
        72 * 25 / (95 * math.sqrt(60)),
    ]
    for got, want in zip(j["preview_values"], expected):
        assert abs(got - want) < 1e-12
    assert round(j["preview_values"][0], 2) == 1.70


@pytest.mark.parametrize("formula,expected", [
    ("Yas * 2", [120, 90, 144]),
    ("2 * Yas", [120, 90, 144]),
    ("Yas * AST * 2", [4800, 7200, 3600]),
    ("Yas + AST", [100, 125, 97]),
    ("Yas - AST", [20, -35, 47]),
])
def test_the_other_arithmetic_still_works(client, sid, formula, expected):
    assert _formula(client, sid, "t", formula)["preview_values"] == expected


def test_the_repetition_guard_is_still_in_force(client, sid):
    """The guard exists to stop a formula exhausting memory; keeping column
    arithmetic working must not disarm it."""
    r = client.post(f"/api/compute/{sid}/formula",
                    json={"new_col": "boom", "formula": '"x" * 999999999'})
    assert r.status_code != 200
    assert "long" in r.json()["detail"].lower()
