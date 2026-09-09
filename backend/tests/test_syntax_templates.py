"""The syntax view: services/syntax_templates.py + /api/project/syntax.

The Python side of every template must be valid Python (ast.parse); the R
side gets structural checks (and a parse when Rscript is on the machine).
Unknown panels and unknown subtypes return None end to end -- an honest gap,
never a guessed formula.
"""
import ast
import shutil
import subprocess

from fastapi.testclient import TestClient

from main import app
from services.syntax_templates import translate

client = TestClient(app)


def _parses_r(code: str) -> bool:
    if not shutil.which("Rscript"):
        return True  # structural checks stand on machines without R
    proc = subprocess.run(
        ["Rscript", "-e", "invisible(parse(text=commandArgs(TRUE)[1]))", code],
        capture_output=True, text=True, timeout=60,
    )
    return proc.returncode == 0


def test_every_covered_combo_generates_valid_code():
    combos = [
        ("hypothesis", {"test": "ttest_1sample", "col": "age", "mu": 50}),
        ("hypothesis", {"test": "ttest_2sample", "col": "age", "groupCol": "arm"}),
        ("hypothesis", {"test": "anova", "col": "age", "groupCol": "arm"}),
        ("hypothesis", {"test": "mannwhitney", "col": "age", "groupCol": "arm"}),
        ("hypothesis", {"test": "kruskal", "col": "age", "groupCol": "arm"}),
        ("hypothesis", {"test": "chisquare", "col": "smoker", "col2": "arm"}),
        ("hypothesis", {"test": "fisher", "col": "smoker", "col2": "arm"}),
        ("models", {"model": "linear", "outcome": "sbp", "predictors": ["age", "bmi"]}),
        ("models", {"model": "logistic", "outcome": "dm", "predictors": ["age", "bmi"]}),
        ("models", {"model": "poisson", "outcome": "events", "predictors": ["age"]}),
        ("models", {"model": "cox", "durationCol": "t", "eventCol": "died", "predictors": ["age", "arm"]}),
    ]
    for panel, params in combos:
        out = translate(panel, params)
        assert out is not None, (panel, params)
        ast.parse(out["python"])
        assert _parses_r(out["r"]), (panel, params, out["r"])
        # The honesty disclaimer must ride every translation.
        assert "provenance line" in out["python"]
        assert "provenance line" in out["r"]


def test_column_names_land_in_the_code():
    out = translate("models", {"model": "logistic", "outcome": "dm", "predictors": ["age", "bmi"]})
    assert "dm ~ age + bmi" in out["python"]
    assert "dm ~ age + bmi" in out["r"]
    assert "exp(" in out["r"]  # odds ratios, not raw coefficients


def test_unknown_panel_and_subtype_return_none():
    assert translate("roc", {"y": "DM"}) is None
    assert translate("hypothesis", {"test": "mancova", "col": "x"}) is None
    assert translate("models", {"model": "firth"}) is None


def test_endpoint_round_trip_and_honest_nulls():
    r = client.post(
        "/api/project/syntax",
        json={"panel": "hypothesis", "params": {"test": "anova", "col": "age", "groupCol": "arm"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert "anova_lm" in body["python"]
    assert "aov(" in body["r"]

    r2 = client.post("/api/project/syntax", json={"panel": "meta", "params": {}})
    assert r2.json() == {"title": None, "python": None, "r": None}
