"""Validation cards are enforced documentation (qa/validation_cards/).

What CI guarantees here:
- every card parses and carries the full schema (the four data-semantics
  behavior answers included -- a card may say "not applicable", never nothing);
- every reference-artifact pointer resolves to a real key in a real file;
- every endpoint a card names is actually served by the app;
- every reference entry (models_audit, tests_audit, every parity file) is
  claimed by exactly one card, so adding a reference method without writing
  its card is a red build;
- every "covered" edge case names an existing test or fixture file;
- every finding carries an explicit open / fixed / re-verified status.
"""
import json
import pathlib

import yaml

from main import app

REPO = pathlib.Path(__file__).resolve().parents[2]
CARDS_DIR = REPO / "qa" / "validation_cards"

REQUIRED_KEYS = {
    "method", "title", "panel", "endpoints", "covers", "implementation",
    "assumptions", "unsupported", "behavior", "reference", "compared",
    "tolerance", "edge_cases", "findings",
}
BEHAVIOR_KEYS = {"missing_data", "weights", "ties", "category_coding"}
EDGE_STATUSES = {"covered", "open"}
FINDING_STATUSES = {"open", "fixed", "re-verified"}


def _cards():
    files = sorted(CARDS_DIR.glob("*.yaml"))
    assert files, f"no validation cards found in {CARDS_DIR}"
    return [(f, yaml.safe_load(f.read_text())) for f in files]


# Reference inventories that are build products, not sources: qa/models_audit
# regenerates reference.json from the pinned R image on every run and
# .gitignores it, so a plain checkout (CI) does not have it. Cards may point
# into it, and those pointers are fully checked wherever the file exists --
# locally after `qa/r_reference/run.sh`, and in any job that regenerates it.
# What must never happen is the check silently degrading to nothing, so the
# tracked inventories below are required unconditionally.
REGENERATED = {"qa/models_audit/reference.json", "qa/models_audit/endpoints.json"}
TRACKED_INVENTORIES = ("qa/tests_audit/reference.json",)


def _available(rel_path: str) -> bool:
    """Is this reference inventory readable in the current checkout?"""
    path = REPO / rel_path
    if path.exists():
        return True
    assert rel_path in REGENERATED, (
        f"reference inventory {rel_path} is missing and is not a known "
        "regenerated artifact; it should be tracked in git"
    )
    return False


def _resolve(artifact: dict):
    path = REPO / artifact["file"]
    if not _available(artifact["file"]):
        return None
    data = json.loads(path.read_text())
    key = artifact.get("key")
    if key is None:
        return data
    node = data
    for part in str(key).split("."):
        assert isinstance(node, dict) and part in node, (
            f"artifact key {key!r} does not resolve in {artifact['file']}"
        )
        node = node[part]
    return node


def test_cards_conform_to_schema():
    for path, card in _cards():
        missing = REQUIRED_KEYS - set(card)
        assert not missing, f"{path.name}: missing keys {sorted(missing)}"
        behavior = card["behavior"]
        assert BEHAVIOR_KEYS <= set(behavior), (
            f"{path.name}: behavior must answer all of {sorted(BEHAVIOR_KEYS)}"
        )
        for key in BEHAVIOR_KEYS:
            assert isinstance(behavior[key], str) and behavior[key].strip(), (
                f"{path.name}: behavior.{key} must be a non-empty answer "
                "(say 'Not applicable' explicitly, never leave it blank)"
            )
        tol = card["tolerance"]
        assert isinstance(tol.get("rel"), float), f"{path.name}: tolerance.rel must be a float"
        assert tol.get("rationale"), f"{path.name}: tolerance needs its rationale"
        impl = card["implementation"]
        for field in ("library", "entrypoint", "version_source"):
            assert impl.get(field), f"{path.name}: implementation.{field} required"


def test_reference_artifacts_resolve():
    for path, card in _cards():
        artifacts = card["reference"].get("artifacts") or []
        assert artifacts, f"{path.name}: at least one reference artifact required"
        for artifact in artifacts:
            _resolve(artifact)


def test_endpoints_exist():
    served = {route.path for route in app.routes if hasattr(route, "path")}
    for path, card in _cards():
        for endpoint in card["endpoints"]:
            assert endpoint in served, f"{path.name}: endpoint {endpoint} not served by the app"


def test_every_reference_entry_is_claimed_by_exactly_one_card():
    required = set()
    # models_audit is regenerated (see REGENERATED); enforce it whenever the
    # checkout has it, and let a plain checkout enforce the rest.
    if _available("qa/models_audit/reference.json"):
        models = json.loads((REPO / "qa/models_audit/reference.json").read_text())["models"]
        required |= {f"models:{k}" for k in models}
    for rel in TRACKED_INVENTORIES:
        assert _available(rel), f"tracked reference inventory {rel} is missing"
    tests_ref = json.loads((REPO / "qa/tests_audit/reference.json").read_text())
    required |= {f"tests:{k}" for k in tests_ref if k != "meta"}
    parity = sorted((REPO / "qa/parity").glob("*.json"))
    assert parity, "qa/parity holds no fixtures; the inventory check would be vacuous"
    required |= {f"parity:{p.name}" for p in parity}

    claimed: dict = {}
    for path, card in _cards():
        for key in card["covers"]:
            assert key not in claimed, (
                f"{key} claimed by both {claimed[key]} and {path.name}"
            )
            claimed[key] = path.name

    unclaimed = required - set(claimed)
    assert not unclaimed, (
        "reference entries without a validation card (write the card or "
        f"retire the entry): {sorted(unclaimed)}"
    )
    phantom = set(claimed) - required
    if not _available("qa/models_audit/reference.json"):
        # Its inventory is unknown in this checkout, so a models: claim cannot
        # be called phantom here. It is checked wherever the file exists.
        phantom = {key for key in phantom if not key.startswith("models:")}
    assert not phantom, f"cards claim reference entries that do not exist: {sorted(phantom)}"


def test_covered_edge_cases_name_a_real_test():
    for path, card in _cards():
        for edge in card["edge_cases"]:
            assert edge.get("status") in EDGE_STATUSES, (
                f"{path.name}: edge case status must be one of {sorted(EDGE_STATUSES)}"
            )
            if edge["status"] == "covered":
                ref = edge.get("test", "")
                assert ref, f"{path.name}: covered edge case {edge['case']!r} must name its test"
                # Accept "path::test_name" and "path (note)": the file is the
                # part before the first "::" or space.
                file_ref = ref.split("::", 1)[0].split(" ", 1)[0]
                target = REPO / file_ref
                assert target.exists(), (
                    f"{path.name}: covered edge case names missing file {target}"
                )


def test_findings_carry_explicit_status():
    for path, card in _cards():
        for finding in card["findings"]:
            assert finding.get("status") in FINDING_STATUSES, (
                f"{path.name}: finding {finding.get('id')!r} status must be one of "
                f"{sorted(FINDING_STATUSES)}"
            )
            assert finding.get("summary"), f"{path.name}: finding needs a summary"
