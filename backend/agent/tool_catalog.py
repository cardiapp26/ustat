# backend/agent/tool_catalog.py
"""Auto-generated, machine-readable tool catalog for the uSTAT FastAPI backend.

The catalog is built by walking ``app.openapi()`` at import time.  Each entry
contains the HTTP contract (method, path, session location) plus a concise,
model-friendly schema for non-session parameters.  Semantic ``requires`` guards
are attached to the key Phase-1 analysis tools so the runner can reject
mistyped calls before they hit the API.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from main import app


class Tool(TypedDict):
    """One registry entry."""

    name: str
    method: str
    path: str
    session_in: Literal["body", "path"]
    body: Dict[str, str]
    requires: Dict[str, str]
    doc: str


# Explicit short names for the tools the Phase-1 runner and tests expect.
# All other routes derive their name from the URL path.
_NAME_OVERRIDES: Dict[str, str] = {
    "/api/stats/table1": "table1",
    "/api/stats/fisher": "fisher",
    "/api/stats/chisquare": "chisquare",
    "/api/stats/ttest": "ttest",
    "/api/stats/anova": "anova",
    "/api/stats/mannwhitney": "mannwhitney",
    "/api/stats/kruskal": "kruskal",
    "/api/stats/roc": "roc",
    "/api/stats/correlation_matrix": "correlation_matrix",
    "/api/stats/correlation_pair": "correlation_pair",
    "/api/models/survival/km": "km",
    "/api/models/survival/cox": "cox",
    "/api/compute/{session_id}/formula": "formula",
    "/api/stats/{session_id}/descriptive": "descriptive",
}

# Semantic guards for the key Phase-1 tools.  Syntax:
#   "numeric" | "categorical" | "ordinal" | "date" | "text" | "any"
#   "categorical(N)" -> categorical with exactly N levels
_REQUIRES_OVERRIDES: Dict[str, Dict[str, str]] = {
    "table1": {"variables": "any", "group_column": "categorical"},
    "fisher": {"row_column": "categorical(2)", "col_column": "categorical(2)"},
    "chisquare": {"row_column": "categorical", "col_column": "categorical"},
    "ttest": {"column": "numeric", "group_column": "categorical(2)"},
    "anova": {"column": "numeric", "group_column": "categorical"},
    "mannwhitney": {"column": "numeric", "group_column": "categorical(2)"},
    "kruskal": {"column": "numeric", "group_column": "categorical"},
    "roc": {"score_column": "numeric", "outcome_column": "categorical(2)"},
    "correlation_matrix": {"variables": "numeric"},
    "correlation_pair": {"var1": "numeric", "var2": "numeric"},
    "km": {"duration_col": "numeric", "event_col": "categorical(2)"},
    "cox": {"duration_col": "numeric", "event_col": "categorical(2)"},
}


def _resolve_schema(
    schema: Any,
    components: Dict[str, Any],
    _stack: Optional[set] = None,
) -> Any:
    """Inline ``$ref`` pointers against ``#/components/schemas``.

    Cycles are broken by returning the original ``$ref`` string so the catalog
    stays finite and serialisable.
    """
    if _stack is None:
        _stack = set()

    if isinstance(schema, list):
        return [_resolve_schema(item, components, _stack) for item in schema]

    if not isinstance(schema, dict):
        return schema

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in _stack:
            return {"$ref": ref}
        _stack = _stack | {ref}
        name = ref.split("/")[-1]
        target = components.get("schemas", {}).get(name)
        if target is None:
            return {"$ref": ref}
        return _resolve_schema(target, components, _stack)

    return {
        key: _resolve_schema(value, components, _stack) for key, value in schema.items()
    }


def _get_body_schema(
    operation: Dict[str, Any], components: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the resolved JSON-schema for the operation's request body, if any."""
    request_body = operation.get("requestBody", {})
    content = request_body.get("content", {})
    for media_type in ("application/json", "multipart/form-data"):
        if media_type in content:
            schema = content[media_type].get("schema", {})
            return _resolve_schema(schema, components) or {}
    return {}


def _type_hint(schema: Any) -> str:
    """Convert a tiny JSON-schema fragment into a human-readable type hint."""
    if not isinstance(schema, dict):
        return "any"

    any_of = schema.get("anyOf") or schema.get("oneOf")
    if any_of:
        non_null = [
            s for s in any_of if isinstance(s, dict) and s.get("type") != "null"
        ]
        base = _type_hint(non_null[0]) if non_null else "any"
        if any(isinstance(s, dict) and s.get("type") == "null" for s in any_of):
            return f"{base}?"
        return base

    typ = schema.get("type", "any")
    if typ == "array":
        items = schema.get("items", {})
        return f"list[{_type_hint(items)}]"
    if typ == "integer":
        return "int"
    if typ == "number":
        return "float"
    if typ == "boolean":
        return "bool"
    if typ == "string":
        enum = schema.get("enum")
        if enum and len(enum) <= 3:
            return "|".join(repr(v) for v in enum)
        return "str"
    if typ == "object":
        return "dict"
    return str(typ)


def _flatten_body(
    body_schema: Dict[str, Any],
    parameters: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Return a flat ``arg_name -> type_hint`` map for non-session parameters.

    The runner injects ``session_id`` itself, so it is stripped from the body
    schema.  Query parameters from GET endpoints are also included so that the
    catalog documents every argument a caller can supply.
    """
    out: Dict[str, str] = {}

    props = body_schema.get("properties", {})
    required = set(body_schema.get("required", []))
    for name, prop_schema in props.items():
        if name == "session_id":
            continue
        hint = _type_hint(prop_schema)
        if name not in required:
            hint = f"{hint}?"
        out[name] = hint

    for param in parameters:
        if param.get("in") != "query":
            continue
        name = param.get("name", "")
        if name == "session_id" or not name:
            continue
        hint = _type_hint(param.get("schema", {}))
        if not param.get("required", False):
            hint = f"{hint}?"
        out[name] = hint

    return out


def _session_in(
    path: str, body_schema: Dict[str, Any]
) -> Optional[Literal["body", "path"]]:
    """Detect how the endpoint receives its session id."""
    if "{session_id}" in path:
        return "path"
    properties = body_schema.get("properties", {})
    if "session_id" in properties:
        return "body"
    return None


def _derive_name(path: str, method: str, used: set[str]) -> str:
    """Create a stable, identifier-safe tool name from a URL path.

    Rules:
    * Explicit overrides (e.g. ``/api/stats/table1`` -> ``table1``).
    * Path-session routes such as ``/api/compute/{session_id}/formula`` fall
      back to the explicit override if present; otherwise they are prefixed
      with the router segment before ``{session_id}`` (e.g.
      ``compute_rename``) to keep names unique.
    * Everything else uses the final URL segment.
    """
    if path in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[path]

    parts = [p for p in path.split("/") if p]

    # Path-session routes: /api/<router>/{session_id}/.../<op>
    if "{session_id}" in parts:
        idx = parts.index("{session_id}")
        prefix = parts[idx - 1].strip("{}") if idx > 0 else ""
        op_parts = parts[idx + 1 :]
        op = "_".join(p.strip("{}") for p in op_parts if p)
        candidate = f"{prefix}_{op}".strip("_")
    else:
        meaningful = [p.strip("{}") for p in parts if p and not p.startswith("{")]
        if meaningful and meaningful[0] == "api":
            meaningful = meaningful[1:]
        candidate = meaningful[-1] if meaningful else "tool"

    candidate = re.sub(r"[^a-z0-9_]+", "_", candidate.lower()).strip("_")

    if candidate and candidate not in used:
        return candidate

    base = candidate or "tool"
    fallback = f"{base}_{method.lower()}"
    if fallback not in used:
        return fallback

    n = 2
    while f"{fallback}_{n}" in used:
        n += 1
    return f"{fallback}_{n}"


def _doc(operation: Dict[str, Any], method: str, path: str) -> str:
    """Pick a concise description from the OpenAPI operation object."""
    summary = (operation.get("summary") or "").strip()
    description = (operation.get("description") or "").strip()
    if summary and description:
        first_line = description.splitlines()[0]
        return f"{summary}: {first_line}"
    return summary or description or f"{method.upper()} {path}"


def build_catalog() -> List[Tool]:
    """Generate the full catalog from the running FastAPI OpenAPI schema."""
    spec = app.openapi()
    components = spec.get("components", {})

    tools: List[Tool] = []
    used_names: set[str] = set()

    for path, path_item in sorted(spec.get("paths", {}).items()):
        if not path.startswith("/api/"):
            continue
        # The client handles upload/health directly; they are not analysis tools.
        if path in ("/api/health", "/api/upload/"):
            continue

        for method, operation in path_item.items():
            if method == "parameters" or not isinstance(operation, dict):
                continue
            if method.upper() not in {"GET", "POST", "PUT", "DELETE"}:
                continue

            body_schema = _get_body_schema(operation, components)
            session_in = _session_in(path, body_schema)
            if session_in is None:
                # Skip endpoints that do not reference a session id (e.g. /api/sessions/blank).
                continue

            name = _derive_name(path, method, used_names)
            used_names.add(name)

            tool = Tool(
                name=name,
                method=method.upper(),
                path=path,
                session_in=session_in,
                body=_flatten_body(body_schema, operation.get("parameters", [])),
                requires=dict(_REQUIRES_OVERRIDES.get(name, {})),
                doc=_doc(operation, method, path),
            )
            tools.append(tool)

    return tools


def _parse_guard(guard: str):
    """Parse a guard string into (kind, exact_level_count_or_None)."""
    match = re.fullmatch(
        r"(numeric|categorical|ordinal|date|text|any)(?:\((\d+)\))?", guard
    )
    if not match:
        raise ValueError(f"Invalid requires guard: {guard!r}")
    kind = match.group(1)
    levels = int(match.group(2)) if match.group(2) else None
    return kind, levels


def check_requires(
    tool: Tool,
    args: Dict[str, Any],
    kinds: Dict[str, str],
    levels: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """Return an error string if ``args`` violates the tool's semantic guards.

    Parameters
    ----------
    tool:
        The catalog entry to validate against.
    args:
        The non-session arguments the model wants to send.
    kinds:
        Mapping ``column_name -> kind`` from the upload response.
    levels:
        Optional mapping ``column_name -> number_of_unique_levels`` so that
        ``categorical(2)`` guards can be enforced.

    Returns
    -------
    ``None`` when the call passes, otherwise a human-readable error message.
    """
    levels = levels or {}
    for param, guard in tool.get("requires", {}).items():
        value = args.get(param)
        if value is None:
            continue

        columns = value if isinstance(value, list) else [value]
        kind_req, level_req = _parse_guard(guard)

        for column in columns:
            if not isinstance(column, str):
                continue

            actual_kind = kinds.get(column)
            if kind_req != "any" and actual_kind != kind_req:
                return (
                    f"{tool['name']} requires {param} column '{column}' to be {kind_req}, "
                    f"got {actual_kind!r}"
                )

            if kind_req == "categorical" and level_req is not None and column in levels:
                actual_levels = levels[column]
                if actual_levels != level_req:
                    return (
                        f"{tool['name']} requires {param} column '{column}' to have "
                        f"{level_req} levels, got {actual_levels}"
                    )

    return None


def get_tool(name: str) -> Optional[Tool]:
    """Look up a catalog entry by its short name."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None


# Static snapshot of the catalog.  Regenerate with:
#   python scripts/generate_tool_catalog.py --write
# The build_catalog() helper remains available for re-validation.
TOOLS: List[Tool] = [
    {
        'name': 'ancova',
        'method': 'POST',
        'path': '/api/advanced_anova/ancova',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'group_col': 'str',
            'covariates': 'list[str]',
            'alpha': 'float?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Ancova',
    },
    {
        'name': 'mancova',
        'method': 'POST',
        'path': '/api/advanced_anova/mancova',
        'session_in': 'body',
        'body': {
            'outcomes': 'list[str]',
            'group_col': 'str',
            'covariates': 'list[str]?',
            'alpha': 'float?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Mancova: One-way MANCOVA: several continuous outcomes vs a grouping factor while',
    },
    {
        'name': 'two_way_anova',
        'method': 'POST',
        'path': '/api/advanced_anova/two_way_anova',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'factor1': 'str',
            'factor2': 'str',
            'alpha': 'float?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Two Way Anova',
    },
    {
        'name': 'bland_altman',
        'method': 'POST',
        'path': '/api/agreement/bland_altman',
        'session_in': 'body',
        'body': {
            'method1': 'str',
            'method2': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Bland Altman',
    },
    {
        'name': 'concordance',
        'method': 'POST',
        'path': '/api/agreement/concordance',
        'session_in': 'body',
        'body': {
            'method1': 'str',
            'method2': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Concordance',
    },
    {
        'name': 'deming',
        'method': 'POST',
        'path': '/api/agreement/deming',
        'session_in': 'body',
        'body': {
            'method1': 'str',
            'method2': 'str',
            'error_ratio': 'float?',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Deming Regression',
    },
    {
        'name': 'passing_bablok',
        'method': 'POST',
        'path': '/api/agreement/passing_bablok',
        'session_in': 'body',
        'body': {
            'method1': 'str',
            'method2': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Passing Bablok',
    },
    {
        'name': 'bayesian',
        'method': 'POST',
        'path': '/api/bayesian',
        'session_in': 'body',
        'body': {
            'analysis_type': 'str',
            'outcome': 'str',
            'predictor': 'str??',
            'predictors': 'list[str]??',
            'mu': 'float?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Run Bayesian',
    },
    {
        'name': 'binomial',
        'method': 'POST',
        'path': '/api/categorical/binomial',
        'session_in': 'body',
        'body': {
            'column': 'str',
            'expected_proportion': 'float?',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Binomial Test',
    },
    {
        'name': 'cochran_armitage',
        'method': 'POST',
        'path': '/api/categorical/cochran_armitage',
        'session_in': 'body',
        'body': {
            'ordinal_col': 'str',
            'event_col': 'str',
            'scores': 'list[float]??',
            'success_value': 'str??',
            'alpha': 'float?',
            'level_order': 'list[str]??',
        },
        'requires': {},
        'doc': 'Cochran Armitage',
    },
    {
        'name': 'cochran_q',
        'method': 'POST',
        'path': '/api/categorical/cochran_q',
        'session_in': 'body',
        'body': {
            'columns': 'list[str]',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Cochran Q Test',
    },
    {
        'name': 'mantel_haenszel',
        'method': 'POST',
        'path': '/api/categorical/mantel_haenszel',
        'session_in': 'body',
        'body': {
            'row_col': 'str',
            'col_col': 'str',
            'strata_col': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Mantel Haenszel Test',
    },
    {
        'name': 'mcnemar',
        'method': 'POST',
        'path': '/api/categorical/mcnemar',
        'session_in': 'body',
        'body': {
            'col1': 'str',
            'col2': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Mcnemar Test',
    },
    {
        'name': 'one_proportion',
        'method': 'POST',
        'path': '/api/categorical/one_proportion',
        'session_in': 'body',
        'body': {
            'column': 'str',
            'null_proportion': 'float?',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'One Proportion Ztest',
    },
    {
        'name': 'two_proportions',
        'method': 'POST',
        'path': '/api/categorical/two_proportions',
        'session_in': 'body',
        'body': {
            'column': 'str',
            'group_column': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Two Proportions Ztest',
    },
    {
        'name': 'did',
        'method': 'POST',
        'path': '/api/causal/did',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'group_col': 'str',
            'time_col': 'str',
            'covariates': 'list[str]?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Difference In Differences: Canonical 2×2 difference-in-differences for a continuous outcome. The',
    },
    {
        'name': 'iv_2sls',
        'method': 'POST',
        'path': '/api/causal/iv_2sls',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'endogenous': 'str',
            'instruments': 'list[str]',
            'covariates': 'list[str]?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Iv 2Sls: Two-stage least squares IV estimator for a continuous outcome with one',
    },
    {
        'name': 'mediation',
        'method': 'POST',
        'path': '/api/causal/mediation',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'treatment': 'str',
            'mediator': 'str',
            'covariates': 'list[str]?',
            'bootstrap': 'int?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Mediation: Linear causal mediation (Baron-Kenny / Preacher-Hayes) for a continuous',
    },
    {
        'name': 'rdd',
        'method': 'POST',
        'path': '/api/causal/rdd',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'running': 'str',
            'cutoff': 'float',
            'bandwidth': 'float??',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Regression Discontinuity: Sharp regression-discontinuity: estimate the local average treatment',
    },
    {
        'name': 'sem',
        'method': 'POST',
        'path': '/api/causal/sem',
        'session_in': 'body',
        'body': {
            'treatments': 'list[str]?',
            'mediators': 'list[str]?',
            'outcomes': 'list[str]?',
            'covariates': 'list[str]?',
            'serial': 'bool?',
            'bootstrap': 'int?',
            'imputation': 'str?',
            'lavaan_spec': 'str??',
        },
        'requires': {},
        'doc': 'Sem Path: Fit a structural equation / path model. Equivalent to PROCESS Models',
    },
    {
        'name': 'target_trial',
        'method': 'POST',
        'path': '/api/causal/target_trial',
        'session_in': 'body',
        'body': {
            'treatment': 'str',
            'outcome': 'str',
            'confounders': 'list[str]',
            'eligibility': 'list[dict]?',
            'strategies': 'list[str]?',
            'time_zero': 'str?',
            'imputation': 'str?',
            'bootstrap': 'int?',
        },
        'requires': {},
        'doc': 'Target Trial: Target-trial emulation: apply explicit eligibility, estimate the',
    },
    {
        'name': 'bar',
        'method': 'POST',
        'path': '/api/charts/bar',
        'session_in': 'body',
        'body': {
            'x': 'str',
            'y': 'str??',
            'color': 'str??',
            'shape': 'str??',
            'bins': 'int?',
        },
        'requires': {},
        'doc': 'Bar',
    },
    {
        'name': 'boxplot',
        'method': 'POST',
        'path': '/api/charts/boxplot',
        'session_in': 'body',
        'body': {
            'x': 'str',
            'y': 'str??',
            'color': 'str??',
            'shape': 'str??',
            'bins': 'int?',
        },
        'requires': {},
        'doc': 'Boxplot',
    },
    {
        'name': 'histogram',
        'method': 'POST',
        'path': '/api/charts/histogram',
        'session_in': 'body',
        'body': {
            'x': 'str',
            'y': 'str??',
            'color': 'str??',
            'shape': 'str??',
            'bins': 'int?',
        },
        'requires': {},
        'doc': 'Histogram',
    },
    {
        'name': 'km_composite',
        'method': 'POST',
        'path': '/api/charts/km_composite',
        'session_in': 'body',
        'body': {
            'group_col': 'str',
            'endpoints': 'list[dict]',
            'risk_times': 'list[float]??',
            'group_order': 'list[str]??',
            'as_cumulative_incidence': 'bool?',
            'inset': 'bool?',
            'inset_max_pct': 'float??',
            'as_percent': 'bool?',
            'imputation': 'str?',
            'title': 'str??',
        },
        'requires': {},
        'doc': 'Km Composite: NEJM-style composite Kaplan-Meier figure.',
    },
    {
        'name': 'paired_box',
        'method': 'POST',
        'path': '/api/charts/paired_box',
        'session_in': 'body',
        'body': {
            'y': 'str',
            'group': 'str',
            'pair_id': 'str',
        },
        'requires': {},
        'doc': 'Paired Box: Matched-pair box plot: one box per group plus a connector line joining',
    },
    {
        'name': 'scatter',
        'method': 'POST',
        'path': '/api/charts/scatter',
        'session_in': 'body',
        'body': {
            'x': 'str',
            'y': 'str??',
            'color': 'str??',
            'shape': 'str??',
            'bins': 'int?',
        },
        'requires': {},
        'doc': 'Scatter',
    },
    {
        'name': 'score_composite',
        'method': 'POST',
        'path': '/api/charts/score_composite',
        'session_in': 'body',
        'body': {
            'group_col': 'str',
            'scores': 'list[dict]',
            'group_order': 'list[str]??',
            'bins': 'int?',
            'title': 'str??',
            'positive_values': 'list[str]?',
        },
        'requires': {},
        'doc': 'Score Composite: Build a manuscript-style score distribution + component prevalence figure.',
    },
    {
        'name': 'splom',
        'method': 'POST',
        'path': '/api/charts/splom',
        'session_in': 'body',
        'body': {
            'variables': 'list[str]',
            'color': 'str??',
        },
        'requires': {},
        'doc': 'Splom',
    },
    {
        'name': 'subgroup_bar',
        'method': 'POST',
        'path': '/api/charts/subgroup_bar',
        'session_in': 'body',
        'body': {
            'y_col': 'str',
            'subgroup_col': 'str',
            'xaxis_col': 'str',
            'color_col': 'str??',
            'y_mode': 'str?',
            'target_value': 'str??',
            'error_type': 'str?',
        },
        'requires': {},
        'doc': 'Subgroup Bar',
    },
    {
        'name': 'run',
        'method': 'POST',
        'path': '/api/code/run',
        'session_in': 'body',
        'body': {
            'code': 'str',
            'timeout': 'int?',
        },
        'requires': {},
        'doc': 'Code Runner Run',
    },
    {
        'name': 'compute_add_column',
        'method': 'POST',
        'path': '/api/compute/{session_id}/add_column',
        'session_in': 'path',
        'body': {
            'name': 'str',
            'default_value': 'any??',
            'position': 'int?',
        },
        'requires': {},
        'doc': 'Add Column',
    },
    {
        'name': 'compute_add_row',
        'method': 'POST',
        'path': '/api/compute/{session_id}/add_row',
        'session_in': 'path',
        'body': {
            'position': 'int?',
        },
        'requires': {},
        'doc': 'Add Row',
    },
    {
        'name': 'compute_clean_outliers',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clean_outliers',
        'session_in': 'path',
        'body': {
            'columns': 'list[str]',
            'method': 'str?',
            'threshold': 'float?',
        },
        'requires': {},
        'doc': 'Clean Outliers',
    },
    {
        'name': 'compute_clinical_bmi',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/bmi',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Bmi',
    },
    {
        'name': 'compute_clinical_bsa',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/bsa',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Bsa: Body Surface Area = sqrt(height_cm × weight_kg / 3600)',
    },
    {
        'name': 'compute_clinical_chadsva',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/chadsva',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Chadsva: CHA₂DS₂-VA score (2024 ESC guideline update — sex no longer counted).',
    },
    {
        'name': 'compute_clinical_chadsvasc',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/chadsvasc',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Chadsvasc: CHA₂DS₂-VASc score for AF stroke risk.',
    },
    {
        'name': 'compute_clinical_egfr',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/egfr',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Egfr: Race-free CKD-EPI 2021 eGFR formula.',
    },
    {
        'name': 'compute_clinical_grace',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/grace',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Grace: GRACE 2.0 integer risk score for ACS (in-hospital mortality).',
    },
    {
        'name': 'compute_clinical_h2fpef',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/h2fpef',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical H2Fpef: H2FPEF score for HFpEF probability (0-9).',
    },
    {
        'name': 'compute_clinical_hasbled',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/hasbled',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Hasbled: HAS-BLED bleeding risk score (0-9).',
    },
    {
        'name': 'compute_clinical_maggic',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/maggic',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Maggic: MAGGIC Heart Failure Risk Score (Pocock et al. 2013, EHJ).',
    },
    {
        'name': 'compute_clinical_map',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/map',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Map: MAP = (SBP + 2 × DBP) / 3',
    },
    {
        'name': 'compute_clinical_qtc',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/qtc',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Qtc: Corrected QT interval (Bazett): QTc = QT_ms / sqrt(RR_s) = QT / sqrt(60/HR)',
    },
    {
        'name': 'compute_clinical_timi_nstemi',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/timi_nstemi',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Timi Nstemi: TIMI risk score for NSTEMI/UA (0-7). Each criterion = 1 point.',
    },
    {
        'name': 'compute_clinical_timi_stemi',
        'method': 'POST',
        'path': '/api/compute/{session_id}/clinical/timi_stemi',
        'session_in': 'path',
        'body': {
            'column_map': 'dict',
            'female_value': 'str??',
            'new_col': 'str??',
        },
        'requires': {},
        'doc': 'Clinical Timi Stemi: TIMI risk score for STEMI (0-14). Points as per original publication.',
    },
    {
        'name': 'compute_column_col_name',
        'method': 'DELETE',
        'path': '/api/compute/{session_id}/column/{col_name}',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Delete Column',
    },
    {
        'name': 'compute_column_values_col_name',
        'method': 'GET',
        'path': '/api/compute/{session_id}/column_values/{col_name}',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Column Values: Every value in a column, in row order.',
    },
    {
        'name': 'compute_delete_columns',
        'method': 'POST',
        'path': '/api/compute/{session_id}/delete_columns',
        'session_in': 'path',
        'body': {
            'columns': 'list[str]',
        },
        'requires': {},
        'doc': 'Delete Columns: Drop several columns in one atomic mutation (bulk tick-and-delete).',
    },
    {
        'name': 'compute_delete_rows',
        'method': 'POST',
        'path': '/api/compute/{session_id}/delete_rows',
        'session_in': 'path',
        'body': {
            'row_indices': 'list[int]',
        },
        'requires': {},
        'doc': 'Delete Rows',
    },
    {
        'name': 'compute_drop_missing',
        'method': 'POST',
        'path': '/api/compute/{session_id}/drop_missing',
        'session_in': 'path',
        'body': {
            'columns': 'list[str]',
        },
        'requires': {},
        'doc': 'Drop Missing',
    },
    {
        'name': 'compute_duplicate_column',
        'method': 'POST',
        'path': '/api/compute/{session_id}/duplicate_column',
        'session_in': 'path',
        'body': {
            'column': 'str',
        },
        'requires': {},
        'doc': 'Duplicate Column',
    },
    {
        'name': 'compute_fill_blanks',
        'method': 'POST',
        'path': '/api/compute/{session_id}/fill_blanks',
        'session_in': 'path',
        'body': {
            'column': 'str',
            'value': 'str',
            'new_column': 'str??',
        },
        'requires': {},
        'doc': 'Fill Blanks',
    },
    {
        'name': 'compute_find_replace',
        'method': 'POST',
        'path': '/api/compute/{session_id}/find_replace',
        'session_in': 'path',
        'body': {
            'columns': 'list[str]',
            'find_value': 'str',
            'replace_value': 'str',
        },
        'requires': {},
        'doc': 'Find Replace',
    },
    {
        'name': 'formula',
        'method': 'POST',
        'path': '/api/compute/{session_id}/formula',
        'session_in': 'path',
        'body': {
            'formula': 'str',
            'new_col': 'str',
        },
        'requires': {},
        'doc': 'Formula Compute: Evaluate a pandas-safe formula expression and save as a new column.',
    },
    {
        'name': 'compute_missing_diagnostics',
        'method': 'POST',
        'path': '/api/compute/{session_id}/missing_diagnostics',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Missing Diagnostics: Heuristic MCAR-vs-MAR hint (no AI). For each column with missing values,',
    },
    {
        'name': 'compute_parse_dates',
        'method': 'POST',
        'path': '/api/compute/{session_id}/parse_dates',
        'session_in': 'path',
        'body': {
            'column': 'str',
            'order': 'str?',
            'century_threshold': 'int?',
            'preview_only': 'bool?',
        },
        'requires': {},
        'doc': 'Parse Dates: Convert a column of mixed-format date text into real datetime64, in place.',
    },
    {
        'name': 'compute_paste',
        'method': 'POST',
        'path': '/api/compute/{session_id}/paste',
        'session_in': 'path',
        'body': {
            'tsv': 'str',
            'has_header': 'bool?',
            'mode': 'str?',
        },
        'requires': {},
        'doc': 'Paste Rows',
    },
    {
        'name': 'compute_paste_cells',
        'method': 'POST',
        'path': '/api/compute/{session_id}/paste_cells',
        'session_in': 'path',
        'body': {
            'start_row': 'int??',
            'start_col': 'str??',
            'tsv': 'str',
            'row_indices': 'list[int]??',
            'target_columns': 'list[str]??',
        },
        'requires': {},
        'doc': 'Paste Cells: Paste a TSV grid of values starting at a given cell position.',
    },
    {
        'name': 'compute_paste_column',
        'method': 'POST',
        'path': '/api/compute/{session_id}/paste_column',
        'session_in': 'path',
        'body': {
            'name': 'str',
            'values': 'list[str?]',
            'position': 'int?',
        },
        'requires': {},
        'doc': 'Paste Column: Insert a whole column (name + per-row values) — the paste side of',
    },
    {
        'name': 'compute_recode',
        'method': 'POST',
        'path': '/api/compute/{session_id}/recode',
        'session_in': 'path',
        'body': {
            'rules': 'list[dict]',
            'else_val': 'any??',
            'new_col': 'str',
        },
        'requires': {},
        'doc': 'Recode Compute',
    },
    {
        'name': 'compute_rename',
        'method': 'POST',
        'path': '/api/compute/{session_id}/rename',
        'session_in': 'path',
        'body': {
            'old_name': 'str',
            'new_name': 'str',
        },
        'requires': {},
        'doc': 'Rename Column',
    },
    {
        'name': 'compute_replace_values',
        'method': 'POST',
        'path': '/api/compute/{session_id}/replace_values',
        'session_in': 'path',
        'body': {
            'column': 'str',
            'mapping': 'dict',
        },
        'requires': {},
        'doc': 'Replace Values: Replace cell values in ONE column via a value→value map, in place.',
    },
    {
        'name': 'compute_transform',
        'method': 'POST',
        'path': '/api/compute/{session_id}/transform',
        'session_in': 'path',
        'body': {
            'source_col': 'str',
            'transform': 'str',
            'new_col': 'str',
        },
        'requires': {},
        'doc': 'Transform Compute',
    },
    {
        'name': 'compute_unique_col_name',
        'method': 'GET',
        'path': '/api/compute/{session_id}/unique/{col_name}',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Unique Values',
    },
    {
        'name': 'calibration',
        'method': 'POST',
        'path': '/api/decision_curve/calibration',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'n_bins': 'int?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Calibration',
    },
    {
        'name': 'dca',
        'method': 'POST',
        'path': '/api/decision_curve/dca',
        'session_in': 'body',
        'body': {
            'outcome': 'str??',
            'predictors': 'list[str]??',
            'probability_col': 'str??',
            'risk_col': 'str??',
            'duration_col': 'str??',
            'event_col': 'str??',
            'time_horizon': 'float??',
            'threshold_range': 'list[float]?',
            'n_thresholds': 'int?',
            'bootstrap_corrected': 'bool?',
            'n_boot': 'int?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Dca: Phase 13: Full router + frontend integration ready.',
    },
    {
        'name': 'hosmer_lemeshow',
        'method': 'POST',
        'path': '/api/decision_curve/hosmer_lemeshow',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'n_groups': 'int?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Hosmer Lemeshow Endpoint: Standalone Hosmer-Lemeshow goodness-of-fit for a logistic model.',
    },
    {
        'name': 'integrated_extval_dca',
        'method': 'POST',
        'path': '/api/decision_curve/integrated_extval_dca',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'prediction_col': 'str',
            'dev_session_id': 'str??',
            'covariates': 'list[str]??',
            'survival_prob_cols': 'list[str]??',
            'time_points': 'list[float]??',
            'time_horizon': 'float??',
            'threshold_range': 'list[float]?',
            'n_thresholds': 'int?',
            'bootstrap_corrected_dca': 'bool?',
            'n_boot': 'int?',
            'flexible_calibration': 'bool?',
            'prediction_source': 'str?',
            'competing_risk_status_col': 'str??',
            'competing_risk_event_code': 'int?',
            'predicted_cif_col': 'str??',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Integrated Extval Dca: One-call integration: model predictions -> External Validation -> DCA.',
    },
    {
        'name': 'linear_full',
        'method': 'POST',
        'path': '/api/diagnostics/linear_full',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Linear Full Diagnostics',
    },
    {
        'name': 'factor_pca',
        'method': 'POST',
        'path': '/api/factor/factor_pca',
        'session_in': 'body',
        'body': {
            'items': 'list[str]',
            'extraction': 'str?',
            'rotation': 'str?',
            'n_factors': 'int??',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Factor Pca',
    },
    {
        'name': 'external_impute_apply',
        'method': 'POST',
        'path': '/api/missing_data/external_impute_apply',
        'session_in': 'body',
        'body': {
            'target': 'str',
            'reference_target': 'str??',
            'predictors': 'str',
            'predictor_mappings': 'str??',
            'method': 'str?',
            'mechanism': 'str?',
            'max_iter': 'int?',
            'random_state': 'int?',
            'stratify_by': 'str??',
            'file': 'str',
        },
        'requires': {},
        'doc': 'External Impute Apply',
    },
    {
        'name': 'external_impute_preview',
        'method': 'POST',
        'path': '/api/missing_data/external_impute_preview',
        'session_in': 'body',
        'body': {
            'target': 'str',
            'reference_target': 'str??',
            'predictors': 'str',
            'predictor_mappings': 'str??',
            'method': 'str?',
            'mechanism': 'str?',
            'max_iter': 'int?',
            'random_state': 'int?',
            'stratify_by': 'str??',
            'file': 'str',
        },
        'requires': {},
        'doc': 'External Impute Preview',
    },
    {
        'name': 'external_impute_transfer',
        'method': 'POST',
        'path': '/api/missing_data/external_impute_transfer',
        'session_in': 'body',
        'body': {
            'target': 'str',
            'preview_rows': 'list[dict]',
        },
        'requires': {},
        'doc': 'External Impute Transfer',
    },
    {
        'name': 'imputation_compare',
        'method': 'POST',
        'path': '/api/missing_data/imputation_compare',
        'session_in': 'body',
        'body': {
            'columns': 'list[str]',
            'strategies': 'list[str]',
        },
        'requires': {},
        'doc': 'Imputation Compare',
    },
    {
        'name': 'mcar_test',
        'method': 'POST',
        'path': '/api/missing_data/mcar_test',
        'session_in': 'body',
        'body': {
            'columns': 'list[str]??',
        },
        'requires': {},
        'doc': 'Mcar Test',
    },
    {
        'name': 'mnar_sensitivity',
        'method': 'POST',
        'path': '/api/missing_data/mnar_sensitivity',
        'session_in': 'body',
        'body': {
            'columns': 'list[str]',
            'outcome_col': 'str??',
            'predictors': 'list[str]?',
            'model_type': 'str?',
            'delta_values': 'list[float]?',
            'n_imputations': 'int?',
            'max_iter': 'int?',
            'passive_formulas': 'dict?',
            'duration_col': 'str??',
            'event_col': 'str??',
            'selection_predictors': 'list[str]?',
            'auxiliary_candidates': 'list[str]??',
            'run_heckman': 'bool?',
            'run_isni': 'bool?',
            'run_survival_mnar': 'bool?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Mnar Sensitivity',
    },
    {
        'name': 'pattern',
        'method': 'POST',
        'path': '/api/missing_data/pattern',
        'session_in': 'body',
        'body': {
            'columns': 'list[str]??',
        },
        'requires': {},
        'doc': 'Pattern',
    },
    {
        'name': 'feature_importance',
        'method': 'POST',
        'path': '/api/ml/feature_importance',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'task': 'str??',
            'n_estimators': 'int?',
            'max_depth': 'int??',
            'min_samples_leaf': 'int?',
            'learning_rate': 'float?',
            'cv_folds': 'int?',
            'class_weight_balanced': 'bool?',
            'n_permutation_repeats': 'int?',
            'random_state': 'int?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Feature Importance: Permutation importance only (no curves) for a quick screen.',
    },
    {
        'name': 'gradient_boosting',
        'method': 'POST',
        'path': '/api/ml/gradient_boosting',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'task': 'str??',
            'n_estimators': 'int?',
            'max_depth': 'int??',
            'min_samples_leaf': 'int?',
            'learning_rate': 'float?',
            'cv_folds': 'int?',
            'class_weight_balanced': 'bool?',
            'n_permutation_repeats': 'int?',
            'random_state': 'int?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Gradient Boosting',
    },
    {
        'name': 'random_forest',
        'method': 'POST',
        'path': '/api/ml/random_forest',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'task': 'str??',
            'n_estimators': 'int?',
            'max_depth': 'int??',
            'min_samples_leaf': 'int?',
            'learning_rate': 'float?',
            'cv_folds': 'int?',
            'class_weight_balanced': 'bool?',
            'n_permutation_repeats': 'int?',
            'random_state': 'int?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Random Forest',
    },
    {
        'name': 'added_value',
        'method': 'POST',
        'path': '/api/model_compare/added_value',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'base_predictors': 'list[str]',
            'new_predictors': 'list[str]',
            'model_type': 'str?',
            'imputation': 'str?',
            'cv_folds': 'int?',
            'bootstrap': 'int?',
        },
        'requires': {},
        'doc': 'Added Value: Quantify the incremental predictive value of adding new predictor(s) to a',
    },
    {
        'name': 'compare_models',
        'method': 'POST',
        'path': '/api/model_compare/compare_models',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'model_specs': 'list[dict]',
            'model_type': 'str?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Compare Models',
    },
    {
        'name': 'nested_lr_test',
        'method': 'POST',
        'path': '/api/model_compare/nested_lr_test',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors_reduced': 'list[str]',
            'predictors_full': 'list[str]',
            'model_type': 'str?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Nested Lr Test',
    },
    {
        'name': 'cox_diagnostics',
        'method': 'POST',
        'path': '/api/model_diagnostics/cox_diagnostics',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'predictors': 'list[str]',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Cox Diagnostics',
    },
    {
        'name': 'external_validation_logistic',
        'method': 'POST',
        'path': '/api/model_diagnostics/external_validation_logistic',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'prob_column': 'str',
            'dev_auc': 'float??',
            'dev_calibration_slope': 'float??',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'External Validation Logistic: External validation of a binary prediction model: take the predicted',
    },
    {
        'name': 'logistic_diagnostics',
        'method': 'POST',
        'path': '/api/model_diagnostics/logistic_diagnostics',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Logistic Diagnostics',
    },
    {
        'name': 'missing_data_sensitivity',
        'method': 'POST',
        'path': '/api/model_diagnostics/missing_data_sensitivity',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'model_type': 'str?',
            'mechanism': 'str?',
            'missing_rate': 'float?',
            'delta_range': 'list[float]?',
            'n_steps': 'int?',
            'duration_col': 'str??',
            'event_col': 'str??',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Missing Data Sensitivity: Run a delta-adjustment sensitivity analysis to explore how results',
    },
    {
        'name': 'model_validation',
        'method': 'POST',
        'path': '/api/model_diagnostics/model_validation',
        'session_in': 'body',
        'body': {
            'outcome': 'str??',
            'prob_column': 'str??',
            'model_type': 'str?',
            'n_boot': 'int?',
            'include_optimism': 'bool?',
            'imputation': 'str?',
            'duration_col': 'str??',
            'event_col': 'str??',
            'linear_predictor_col': 'str??',
            'predictors': 'list[str]?',
            'cv_folds': 'int?',
        },
        'requires': {},
        'doc': 'Model Validation: Bootstrap performance + optional optimism correction for a prediction model.',
    },
    {
        'name': 'nri_idi',
        'method': 'POST',
        'path': '/api/model_diagnostics/nri_idi',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'prob_old': 'str',
            'prob_new': 'str',
            'cutoff': 'float?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Nri Idi: Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI).',
    },
    {
        'name': 'causal_sensitivity',
        'method': 'POST',
        'path': '/api/models/causal_sensitivity',
        'session_in': 'body',
        'body': {
            'observed_estimate': 'float',
            'ci_low': 'float??',
            'ci_high': 'float??',
            'measure': "'rr'|'or'|'hr'?",
            'rare_outcome': 'bool?',
            'baseline_risk': 'float??',
            'smd': 'float??',
            'confounding_strength': 'float?',
            'prevalence_exposed': 'float?',
            'prevalence_unexposed': 'float?',
            'unmeasured_confounders': 'list[dict]?',
            'treatment_col': 'str??',
            'outcome_col': 'str??',
            'monotone_treatment_response': 'bool?',
            'p_y1_treated': 'float??',
            'p_y1_control': 'float??',
            'p_treated': 'float??',
            'match_id_col': 'str??',
            'rosenbaum_gamma_max': 'float?',
            'rosenbaum_n_gamma': 'int?',
            'negative_control_outcome_col': 'str??',
            'negative_control_covariates': 'list[str]?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Causal Sensitivity: Causal sensitivity suite: E-value, QBA, Manski bounds, Rosenbaum bounds,',
    },
    {
        'name': 'firth_logistic',
        'method': 'POST',
        'path': '/api/models/firth_logistic',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'scale_factors': 'dict??',
            'imputation': 'str??',
            'max_iter': 'int?',
            'tol': 'float?',
            'interactions': 'list[list[str]]??',
        },
        'requires': {},
        'doc': 'Firth Logistic Regression',
    },
    {
        'name': 'gamma',
        'method': 'POST',
        'path': '/api/models/gamma',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'link': 'str?',
            'imputation': 'str??',
            'robust_se': 'bool??',
        },
        'requires': {},
        'doc': 'Gamma Regression',
    },
    {
        'name': 'gee',
        'method': 'POST',
        'path': '/api/models/gee',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'group_col': 'str',
            'family': 'str?',
            'cov_struct': 'str?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Gee Regression',
    },
    {
        'name': 'iptw',
        'method': 'POST',
        'path': '/api/models/iptw',
        'session_in': 'body',
        'body': {
            'treatment_col': 'str',
            'covariates': 'list[str]',
            'estimand': 'str?',
            'stabilize': 'bool?',
            'weight_truncation': 'str?',
            'weight_truncation_lo': 'float?',
            'weight_truncation_hi': 'float?',
            'weight_truncation_max': 'float?',
            'outcome_type': 'str?',
            'outcome_col': 'str??',
            'survival_duration_col': 'str??',
            'survival_event_col': 'str??',
            'se_method': 'str?',
            'imputation': 'str??',
            'score_method': 'str??',
            'random_state': 'int??',
            'trim_common_support': 'bool??',
        },
        'requires': {},
        'doc': 'Iptw Analysis',
    },
    {
        'name': 'linear',
        'method': 'POST',
        'path': '/api/models/linear',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'imputation': 'str??',
            'robust_se': 'bool??',
            'interactions': 'list[list[str]]??',
        },
        'requires': {},
        'doc': 'Linear Regression',
    },
    {
        'name': 'linear_diag',
        'method': 'POST',
        'path': '/api/models/linear_diag',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Linear Diagnostics',
    },
    {
        'name': 'lmm',
        'method': 'POST',
        'path': '/api/models/lmm',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'fixed_effects': 'list[str]',
            'group_col': 'str',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Linear Mixed Model',
    },
    {
        'name': 'logistic',
        'method': 'POST',
        'path': '/api/models/logistic',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'scale_factors': 'dict??',
            'selection': 'str??',
            'imputation': 'str??',
            'robust_se': 'bool??',
            'interactions': 'list[list[str]]??',
            'use_firth': 'bool??',
        },
        'requires': {},
        'doc': 'Logistic Regression',
    },
    {
        'name': 'logistic_table',
        'method': 'POST',
        'path': '/api/models/logistic_table',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'scale_factors': 'dict??',
            'selection': 'str??',
            'imputation': 'str??',
            'robust_se': 'bool??',
            'interactions': 'list[list[str]]??',
            'use_firth': 'bool??',
        },
        'requires': {},
        'doc': 'Logistic Or Table',
    },
    {
        'name': 'melt',
        'method': 'POST',
        'path': '/api/models/melt',
        'session_in': 'body',
        'body': {
            'id_col': 'str',
            'value_cols': 'list[str]',
            'time_var_name': 'str?',
            'value_var_name': 'str?',
            'time_labels': 'list[str]??',
        },
        'requires': {},
        'doc': 'Melt Wide To Long: Reshape wide-format repeated measures into long format and save back to session.',
    },
    {
        'name': 'mnar_sensitivity_post',
        'method': 'POST',
        'path': '/api/models/mnar_sensitivity',
        'session_in': 'body',
        'body': {
            'columns': 'list[str]',
            'outcome_col': 'str??',
            'predictors': 'list[str]?',
            'selection_predictors': 'list[str]?',
            'auxiliary_candidates': 'list[str]?',
            'delta_values': 'list[float]?',
            'n_imputations': 'int?',
            'max_iter': 'int?',
            'passive_formulas': 'dict?',
            'duration_col': 'str??',
            'event_col': 'str??',
            'model_type': 'str?',
            'run_heckman': 'bool?',
            'run_isni': 'bool?',
            'run_survival_mnar': 'bool?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Mnar Sensitivity: Run MNAR missing data sensitivity analysis including pattern mixture models,',
    },
    {
        'name': 'multi_outcome_regression',
        'method': 'POST',
        'path': '/api/models/multi_outcome_regression',
        'session_in': 'body',
        'body': {
            'outcomes': 'list[str]',
            'predictors': 'list[str]',
            'covariates': 'list[str]?',
            'standardize': 'bool??',
            'imputation': 'str??',
            'robust_se': 'bool??',
        },
        'requires': {},
        'doc': 'Multi Outcome Regression',
    },
    {
        'name': 'negbinom',
        'method': 'POST',
        'path': '/api/models/negbinom',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'imputation': 'str??',
            'robust_se': 'bool??',
        },
        'requires': {},
        'doc': 'Negative Binomial Regression',
    },
    {
        'name': 'ordinal',
        'method': 'POST',
        'path': '/api/models/ordinal',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Ordinal Regression: Proportional-odds ordinal logistic regression (statsmodels OrderedModel).',
    },
    {
        'name': 'poisson',
        'method': 'POST',
        'path': '/api/models/poisson',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'imputation': 'str??',
            'robust_se': 'bool??',
        },
        'requires': {},
        'doc': 'Poisson Regression',
    },
    {
        'name': 'polynomial',
        'method': 'POST',
        'path': '/api/models/polynomial',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictor': 'str',
            'degree': 'int?',
            'covariates': 'list[str]?',
            'imputation': 'str??',
            'robust_se': 'bool??',
        },
        'requires': {},
        'doc': 'Polynomial Regression',
    },
    {
        'name': 'psm',
        'method': 'POST',
        'path': '/api/models/psm',
        'session_in': 'body',
        'body': {
            'treatment_col': 'str',
            'covariates': 'list[str]',
            'outcome_col': 'str??',
            'caliper': 'float??',
            'caliper_scale': 'str??',
            'ratio': 'int??',
            'imputation': 'str??',
            'trim_common_support': 'bool??',
            'random_state': 'int??',
            'score_method': 'str??',
            'matching_method': 'str??',
            'exact_match': 'list[str]??',
            'outcome_type': 'str??',
            'survival_duration_col': 'str??',
            'survival_event_col': 'str??',
            'compute_rosenbaum': 'bool??',
            'rosenbaum_gamma_max': 'float??',
        },
        'requires': {},
        'doc': 'Propensity Score Matching: Main PSM endpoint (mounted at /api/models/psm).',
    },
    {
        'name': 'rcs',
        'method': 'POST',
        'path': '/api/models/rcs',
        'session_in': 'body',
        'body': {
            'predictor': 'str',
            'outcome': 'str??',
            'covariates': 'list[str]?',
            'n_knots': 'int?',
            'ref_value': 'float??',
            'model_type': 'str?',
            'imputation': 'str?',
            'duration_col': 'str??',
            'event_col': 'str??',
            'knot_positions': 'list[float]??',
            'interaction_covariates': 'list[str]??',
        },
        'requires': {},
        'doc': 'Rcs Regression',
    },
    {
        'name': 'stepwise',
        'method': 'POST',
        'path': '/api/models/stepwise',
        'session_in': 'body',
        'body': {
            'model_type': 'str?',
            'outcome': 'str',
            'candidates': 'list[str]',
            'direction': 'str?',
            'criterion': 'str?',
            'p_enter': 'float?',
            'p_remove': 'float?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Stepwise Selection',
    },
    {
        'name': 'cox',
        'method': 'POST',
        'path': '/api/models/survival/cox',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'predictors': 'list[str]',
            'imputation': 'str??',
            'interactions': 'list[list[str]]??',
        },
        'requires': {
            'duration_col': 'numeric',
            'event_col': 'categorical(2)',
        },
        'doc': 'Cox Regression',
    },
    {
        'name': 'cox_horizons',
        'method': 'POST',
        'path': '/api/models/survival/cox_horizons',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'predictor': 'str',
            'covariates': 'list[str]??',
            'horizons': 'list[float]',
            'horizon_labels': 'list[str]??',
            'include_full': 'bool?',
            'full_label': 'str?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Cox Horizons',
    },
    {
        'name': 'cox_model_specs',
        'method': 'POST',
        'path': '/api/models/survival/cox_model_specs',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'exposure': 'str',
            'exposure_reference': 'str??',
            'specs': 'list[dict]',
            'include_unadjusted': 'bool?',
        },
        'requires': {},
        'doc': "Cox Model Specs: One exposure's adjusted HR across several model specifications.",
    },
    {
        'name': 'cox_rcs',
        'method': 'POST',
        'path': '/api/models/survival/cox_rcs',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'spline_terms': 'list[dict]',
            'covariates': 'list[str]?',
            'include_interaction': 'bool?',
            'imputation': 'str??',
            'grid_size': 'int?',
        },
        'requires': {},
        'doc': 'Cox Rcs',
    },
    {
        'name': 'cox_tv',
        'method': 'POST',
        'path': '/api/models/survival/cox_tv',
        'session_in': 'body',
        'body': {
            'id_col': 'str',
            'start_col': 'str',
            'stop_col': 'str',
            'event_col': 'str',
            'predictors': 'list[str]',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Cox Time Varying',
    },
    {
        'name': 'cox_uni_multi',
        'method': 'POST',
        'path': '/api/models/survival/cox_uni_multi',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'predictors': 'list[str]',
            'references': 'dict??',
            'parsimonious': 'list[str]??',
        },
        'requires': {},
        'doc': 'Cox Uni Multi: Paired unadjusted (univariable) vs adjusted (multivariable) Cox HRs —',
    },
    {
        'name': 'km',
        'method': 'POST',
        'path': '/api/models/survival/km',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'group_col': 'str??',
            'stratify_col': 'str??',
            'imputation': 'str??',
            'survival_times': 'list[float]??',
            'pairwise': 'bool?',
            'pairwise_correction': 'str?',
            'risk_times': 'list[float]??',
            'include_censors': 'bool?',
        },
        'requires': {
            'duration_col': 'numeric',
            'event_col': 'categorical(2)',
        },
        'doc': 'Kaplan Meier',
    },
    {
        'name': 'gatekeeping',
        'method': 'POST',
        'path': '/api/multiplicity/gatekeeping',
        'session_in': 'body',
        'body': {
            'families': 'list[dict]',
            'method': 'str?',
            'logic': 'str?',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Gatekeeping',
    },
    {
        'name': 'build',
        'method': 'POST',
        'path': '/api/nomogram/build',
        'session_in': 'body',
        'body': {
            'outcome': 'str',
            'predictors': 'list[str]',
            'model_type': 'str?',
            'imputation': 'str?',
        },
        'requires': {},
        'doc': 'Build Nomogram: Build a nomogram from logistic or Cox regression.',
    },
    {
        'name': 'method_appendix',
        'method': 'POST',
        'path': '/api/pub_export/method_appendix',
        'session_in': 'body',
        'body': {
            'title': 'str?',
            'include_data_io': 'bool?',
            'include_software': 'bool?',
        },
        'requires': {},
        'doc': 'Method Appendix Docx: Build a Methods-section DOCX from the session audit log.',
    },
    {
        'name': 'table_docx',
        'method': 'POST',
        'path': '/api/pub_export/table_docx',
        'session_in': 'body',
        'body': {
            'group_column': 'str??',
            'variables': 'list[str]',
            'variable_kinds': 'dict??',
            'selected_stats': 'list[str]??',
        },
        'requires': {},
        'doc': 'Table Docx: Generate Table 1 as a Word document (.docx) download.',
    },
    {
        'name': 'cronbach',
        'method': 'POST',
        'path': '/api/reliability/cronbach',
        'session_in': 'body',
        'body': {
            'items': 'list[str]',
        },
        'requires': {},
        'doc': 'Cronbach',
    },
    {
        'name': 'friedman',
        'method': 'POST',
        'path': '/api/repeated/friedman',
        'session_in': 'body',
        'body': {
            'columns': 'list[str]',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Friedman',
    },
    {
        'name': 'mixed_anova',
        'method': 'POST',
        'path': '/api/repeated/mixed_anova',
        'session_in': 'body',
        'body': {
            'subject_col': 'str',
            'within_col': 'str',
            'between_col': 'str',
            'value_col': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Mixed Anova: Mixed ANOVA via OLS with Type II SS (within × between interaction).',
    },
    {
        'name': 'paired_ttest',
        'method': 'POST',
        'path': '/api/repeated/paired_ttest',
        'session_in': 'body',
        'body': {
            'col1': 'str',
            'col2': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Paired Ttest',
    },
    {
        'name': 'rm_anova',
        'method': 'POST',
        'path': '/api/repeated/rm_anova',
        'session_in': 'body',
        'body': {
            'subject_col': 'str',
            'within_col': 'str',
            'value_col': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Rm Anova',
    },
    {
        'name': 'wilcoxon_signed_rank',
        'method': 'POST',
        'path': '/api/repeated/wilcoxon_signed_rank',
        'session_in': 'body',
        'body': {
            'col1': 'str',
            'col2': 'str',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Wilcoxon Signed Rank',
    },
    {
        'name': 'sessions',
        'method': 'GET',
        'path': '/api/sessions/{session_id}',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Get Session Info: Retrieve session details (filename, rows, columns, preview) for a saved session ID.',
    },
    {
        'name': 'sessions_audit',
        'method': 'GET',
        'path': '/api/sessions/{session_id}/audit',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Get Audit: Return the audit trail for a session.',
    },
    {
        'name': 'sessions_clear_cells',
        'method': 'POST',
        'path': '/api/sessions/{session_id}/clear_cells',
        'session_in': 'path',
        'body': {
            'cells': 'list[any]',
        },
        'requires': {},
        'doc': 'Clear Cells: Clear (set to NaN) multiple cells at once.',
    },
    {
        'name': 'sessions_decimals',
        'method': 'POST',
        'path': '/api/sessions/{session_id}/decimals',
        'session_in': 'path',
        'body': {
            'column': 'str',
            'decimals': 'int??',
        },
        'requires': {},
        'doc': 'Set Column Decimals',
    },
    {
        'name': 'sessions_decimals_get',
        'method': 'GET',
        'path': '/api/sessions/{session_id}/decimals',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Get Column Decimals',
    },
    {
        'name': 'sessions_export',
        'method': 'GET',
        'path': '/api/sessions/{session_id}/export',
        'session_in': 'path',
        'body': {
            'fmt': 'str?',
            'filename': 'str?',
            'col_kinds': 'str?',
        },
        'requires': {},
        'doc': 'Export Dataset',
    },
    {
        'name': 'sessions_export_csv',
        'method': 'GET',
        'path': '/api/sessions/{session_id}/export/csv',
        'session_in': 'path',
        'body': {
            'filename': 'str?',
        },
        'requires': {},
        'doc': 'Export Csv: Export session data as CSV file.',
    },
    {
        'name': 'sessions_export_xlsx',
        'method': 'GET',
        'path': '/api/sessions/{session_id}/export/xlsx',
        'session_in': 'path',
        'body': {
            'filename': 'str?',
        },
        'requires': {},
        'doc': 'Export Xlsx: Export session data as XLSX file.',
    },
    {
        'name': 'sessions_kind',
        'method': 'POST',
        'path': '/api/sessions/{session_id}/kind',
        'session_in': 'path',
        'body': {
            'column': 'str',
            'kind': 'str',
        },
        'requires': {},
        'doc': 'Set Column Kind: Persist a user-driven kind change (data-tab badge / dictionary).',
    },
    {
        'name': 'sessions_metadata',
        'method': 'POST',
        'path': '/api/sessions/{session_id}/metadata',
        'session_in': 'path',
        'body': {
            'columns': 'dict',
        },
        'requires': {},
        'doc': 'Save Metadata: Store column-level metadata for the session.',
    },
    {
        'name': 'sessions_name_suggestions',
        'method': 'GET',
        'path': '/api/sessions/{session_id}/name_suggestions',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': "Name Suggestions: Suggest readable Sentence-case names for the session's columns.",
    },
    {
        'name': 'sessions_redo',
        'method': 'POST',
        'path': '/api/sessions/{session_id}/redo',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Redo Action: Redo the last undone mutation.',
    },
    {
        'name': 'sessions_rename',
        'method': 'POST',
        'path': '/api/sessions/{session_id}/rename',
        'session_in': 'path',
        'body': {
            'filename': 'str',
        },
        'requires': {},
        'doc': 'Rename Session',
    },
    {
        'name': 'sessions_reorder_columns',
        'method': 'POST',
        'path': '/api/sessions/{session_id}/reorder_columns',
        'session_in': 'path',
        'body': {
            'columns': 'list[any]',
        },
        'requires': {},
        'doc': 'Reorder Columns: Reorder DataFrame columns to match frontend drag-and-drop order.',
    },
    {
        'name': 'sessions_row_row_index',
        'method': 'DELETE',
        'path': '/api/sessions/{session_id}/row/{row_index}',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Delete Row: Delete a specific row containing an outlier.',
    },
    {
        'name': 'sessions_save_session',
        'method': 'GET',
        'path': '/api/sessions/{session_id}/save_session',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Save Session: Export the full session as a downloadable JSON file.',
    },
    {
        'name': 'sessions_select_cases',
        'method': 'POST',
        'path': '/api/sessions/{session_id}/select_cases',
        'session_in': 'path',
        'body': {
            'conditions': 'list[any]',
            'apply': 'bool?',
        },
        'requires': {},
        'doc': 'Select Cases',
    },
    {
        'name': 'sessions_select_cases_delete',
        'method': 'DELETE',
        'path': '/api/sessions/{session_id}/select_cases',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Clear Cases',
    },
    {
        'name': 'sessions_undo',
        'method': 'POST',
        'path': '/api/sessions/{session_id}/undo',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Undo Action: Undo the last data mutation (backend DataFrame + return refreshed preview).',
    },
    {
        'name': 'anova',
        'method': 'POST',
        'path': '/api/stats/anova',
        'session_in': 'body',
        'body': {
            'column': 'str',
            'group_column': 'str',
        },
        'requires': {
            'column': 'numeric',
            'group_column': 'categorical',
        },
        'doc': 'Anova',
    },
    {
        'name': 'chisquare',
        'method': 'POST',
        'path': '/api/stats/chisquare',
        'session_in': 'body',
        'body': {
            'row_column': 'str',
            'col_column': 'str',
        },
        'requires': {
            'row_column': 'categorical',
            'col_column': 'categorical',
        },
        'doc': 'Chisquare',
    },
    {
        'name': 'cohens_kappa',
        'method': 'POST',
        'path': '/api/stats/cohens_kappa',
        'session_in': 'body',
        'body': {
            'rater1_col': 'str',
            'rater2_col': 'str',
        },
        'requires': {},
        'doc': 'Cohens Kappa',
    },
    {
        'name': 'correlation_matrix',
        'method': 'POST',
        'path': '/api/stats/correlation_matrix',
        'session_in': 'body',
        'body': {
            'variables': 'list[str]',
            'method': 'str??',
            'imputation': 'str??',
        },
        'requires': {
            'variables': 'numeric',
        },
        'doc': 'Correlation Matrix Post',
    },
    {
        'name': 'correlation_pair',
        'method': 'POST',
        'path': '/api/stats/correlation_pair',
        'session_in': 'body',
        'body': {
            'var1': 'str',
            'var2': 'str',
            'method': 'str??',
            'imputation': 'str??',
        },
        'requires': {
            'var1': 'numeric',
            'var2': 'numeric',
        },
        'doc': 'Correlation Pair',
    },
    {
        'name': 'fisher',
        'method': 'POST',
        'path': '/api/stats/fisher',
        'session_in': 'body',
        'body': {
            'row_column': 'str',
            'col_column': 'str',
        },
        'requires': {
            'row_column': 'categorical(2)',
            'col_column': 'categorical(2)',
        },
        'doc': 'Fisher Exact',
    },
    {
        'name': 'fleiss_kappa',
        'method': 'POST',
        'path': '/api/stats/fleiss_kappa',
        'session_in': 'body',
        'body': {
            'rater_cols': 'list[str]',
        },
        'requires': {},
        'doc': 'Fleiss Kappa Endpoint',
    },
    {
        'name': 'icc',
        'method': 'POST',
        'path': '/api/stats/icc',
        'session_in': 'body',
        'body': {
            'rater1_col': 'str',
            'rater2_col': 'str',
        },
        'requires': {},
        'doc': 'Icc Endpoint',
    },
    {
        'name': 'jonckheere_terpstra',
        'method': 'POST',
        'path': '/api/stats/jonckheere_terpstra',
        'session_in': 'body',
        'body': {
            'column': 'str',
            'group_column': 'str',
            'scores': 'list[float]??',
            'alpha': 'float?',
        },
        'requires': {},
        'doc': 'Jonckheere Terpstra',
    },
    {
        'name': 'kruskal',
        'method': 'POST',
        'path': '/api/stats/kruskal',
        'session_in': 'body',
        'body': {
            'column': 'str',
            'group_column': 'str',
            'posthoc_correction': 'str??',
        },
        'requires': {
            'column': 'numeric',
            'group_column': 'categorical',
        },
        'doc': 'Kruskal',
    },
    {
        'name': 'mannwhitney',
        'method': 'POST',
        'path': '/api/stats/mannwhitney',
        'session_in': 'body',
        'body': {
            'column': 'str',
            'group_column': 'str',
        },
        'requires': {
            'column': 'numeric',
            'group_column': 'categorical(2)',
        },
        'doc': 'Mannwhitney',
    },
    {
        'name': 'noninferiority',
        'method': 'POST',
        'path': '/api/stats/noninferiority',
        'session_in': 'body',
        'body': {
            'outcome_col': 'str',
            'group_col': 'str',
            'test_group': 'str??',
            'ref_group': 'str??',
            'outcome_type': 'str?',
            'effect': 'str?',
            'margin': 'float?',
            'bound': 'str?',
            'alpha': 'float?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Noninferiority',
    },
    {
        'name': 'roc',
        'method': 'POST',
        'path': '/api/stats/roc',
        'session_in': 'body',
        'body': {
            'score_column': 'str',
            'outcome_column': 'str',
            'direction': 'str??',
            'manual_cutoff': 'float??',
            'imputation': 'str??',
            'stratify_by': 'str??',
            'stratify_values': 'list[any]??',
        },
        'requires': {
            'score_column': 'numeric',
            'outcome_column': 'categorical(2)',
        },
        'doc': "Roc Analysis: ROC analysis for a numeric score predicting a binary 0/1 outcome. Use ``direction='lower'`` when higher scores indicate lower event risk; the backend reports ``1 - AUC`` with swapped confidence-interval bounds. Use ``direction='higher'`` when higher scores indicate higher event risk, and ``direction='auto'`` to flip automatically when the naive AUC is < 0.5.",
    },
    {
        'name': 'roc_combined',
        'method': 'POST',
        'path': '/api/stats/roc_combined',
        'session_in': 'body',
        'body': {
            'predictor_columns': 'list[str]',
            'outcome_column': 'str',
            'model_name': 'str??',
        },
        'requires': {},
        'doc': 'Roc Combined',
    },
    {
        'name': 'roc_compare',
        'method': 'POST',
        'path': '/api/stats/roc_compare',
        'session_in': 'body',
        'body': {
            'score_column_1': 'str',
            'score_column_2': 'str',
            'outcome_column': 'str',
            'direction_1': 'str??',
            'direction_2': 'str??',
        },
        'requires': {},
        'doc': 'Roc Compare',
    },
    {
        'name': 'roc_multi_compare',
        'method': 'POST',
        'path': '/api/stats/roc_multi_compare',
        'session_in': 'body',
        'body': {
            'score_columns': 'list[str]',
            'outcome_column': 'str',
            'directions': 'list[str]??',
            'p_adjust': 'str??',
        },
        'requires': {},
        'doc': 'Roc Multi Compare',
    },
    {
        'name': 'table1',
        'method': 'POST',
        'path': '/api/stats/table1',
        'session_in': 'body',
        'body': {
            'group_column': 'str??',
            'variables': 'list[str]',
            'variable_kinds': 'dict??',
            'selected_stats': 'list[str]??',
            'normality_mode': 'str??',
            'column_decimals': 'dict??',
        },
        'requires': {
            'variables': 'any',
            'group_column': 'categorical',
        },
        'doc': 'Table1',
    },
    {
        'name': 'tost',
        'method': 'POST',
        'path': '/api/stats/tost',
        'session_in': 'body',
        'body': {
            'column': 'str',
            'group_column': 'str??',
            'paired_column': 'str??',
            'low': 'float',
            'high': 'float',
            'mu': 'float??',
            'test_type': 'str?',
        },
        'requires': {},
        'doc': 'Tost',
    },
    {
        'name': 'ttest',
        'method': 'POST',
        'path': '/api/stats/ttest',
        'session_in': 'body',
        'body': {
            'column': 'str',
            'group_column': 'str??',
            'mu': 'float??',
            'equal_var': 'bool?',
        },
        'requires': {
            'column': 'numeric',
            'group_column': 'categorical(2)',
        },
        'doc': 'Ttest',
    },
    {
        'name': 'weighted_descriptive',
        'method': 'POST',
        'path': '/api/stats/weighted_descriptive',
        'session_in': 'body',
        'body': {
            'value_cols': 'list[str]',
            'weight_col': 'str',
            'group_col': 'str??',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Weighted Descriptive',
    },
    {
        'name': 'stats_column_summary',
        'method': 'GET',
        'path': '/api/stats/{session_id}/column_summary',
        'session_in': 'path',
        'body': {
            'column': 'str',
            'kind': 'str??',
        },
        'requires': {},
        'doc': 'Column Summary',
    },
    {
        'name': 'stats_correlation',
        'method': 'GET',
        'path': '/api/stats/{session_id}/correlation',
        'session_in': 'path',
        'body': {
            'method': 'str?',
        },
        'requires': {},
        'doc': 'Correlation',
    },
    {
        'name': 'descriptive',
        'method': 'GET',
        'path': '/api/stats/{session_id}/descriptive',
        'session_in': 'path',
        'body': {
            'column': 'str??',
        },
        'requires': {},
        'doc': 'Descriptive',
    },
    {
        'name': 'stats_frequency',
        'method': 'GET',
        'path': '/api/stats/{session_id}/frequency',
        'session_in': 'path',
        'body': {
            'column': 'str??',
        },
        'requires': {},
        'doc': 'Frequency',
    },
    {
        'name': 'stats_missing',
        'method': 'GET',
        'path': '/api/stats/{session_id}/missing',
        'session_in': 'path',
        'body': {
            'columns': 'str?',
        },
        'requires': {},
        'doc': 'Get Missing: Return per-column missing counts and total rows affected for the given',
    },
    {
        'name': 'stats_raw',
        'method': 'GET',
        'path': '/api/stats/{session_id}/raw',
        'session_in': 'path',
        'body': {
            'columns': 'str?',
        },
        'requires': {},
        'doc': 'Get Raw Columns',
    },
    {
        'name': 'stats_refresh',
        'method': 'GET',
        'path': '/api/stats/{session_id}/refresh',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Refresh Session: Return updated session metadata after in-place operations.',
    },
    {
        'name': 'stats_sparklines',
        'method': 'GET',
        'path': '/api/stats/{session_id}/sparklines',
        'session_in': 'path',
        'body': {},
        'requires': {},
        'doc': 'Get Sparklines',
    },
    {
        'name': 'causal_sensitivity_post',
        'method': 'POST',
        'path': '/api/survival_advanced/causal_sensitivity',
        'session_in': 'body',
        'body': {
            'observed_estimate': 'float?',
            'ci_low': 'float??',
            'ci_high': 'float??',
            'measure': 'str?',
            'rare_outcome': 'bool?',
            'baseline_risk': 'float??',
            'smd': 'float??',
            'confounding_strength': 'float?',
            'prevalence_exposed': 'float?',
            'prevalence_unexposed': 'float?',
            'unmeasured_confounders': 'list[dict]?',
            'treatment_col': 'str??',
            'outcome_col': 'str??',
            'monotone_treatment_response': 'bool?',
            'p_y1_treated': 'float??',
            'p_y1_control': 'float??',
            'p_treated': 'float??',
            'match_id_col': 'str??',
            'rosenbaum_gamma_max': 'float?',
            'rosenbaum_n_gamma': 'int?',
            'negative_control_outcome_col': 'str??',
            'negative_control_covariates': 'list[str]?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Causal Sensitivity',
    },
    {
        'name': 'dynamic_prediction',
        'method': 'POST',
        'path': '/api/survival_advanced/dynamic_prediction',
        'session_in': 'body',
        'body': {
            'landmark_time': 'float',
            'current_state': 'int?',
            'id_col': 'str?',
            'from_state_col': 'str?',
            'to_state_col': 'str?',
            'entry_col': 'str?',
            'exit_col': 'str?',
            'event_col': 'str?',
            'predictors': 'list[str]',
            'horizon': 'float?',
            'n_points': 'int?',
            'transition_model_type': 'str??',
            'run_bootstrap': 'bool??',
            'n_bootstrap': 'int??',
            'run_microsimulation': 'bool??',
            'n_simulations': 'int??',
        },
        'requires': {},
        'doc': 'Dynamic Prediction',
    },
    {
        'name': 'external_validation',
        'method': 'POST',
        'path': '/api/survival_advanced/external_validation',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'predicted_lp_col': 'str',
            'time_points': 'list[float]??',
            'survival_probs': 'list[list[float]]??',
            'dev_metrics': 'dict??',
        },
        'requires': {},
        'doc': 'External Validation',
    },
    {
        'name': 'fine_gray',
        'method': 'POST',
        'path': '/api/survival_advanced/fine_gray',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'event_of_interest': 'int?',
            'group_col': 'str??',
            'predictors': 'list[str]??',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Fine Gray',
    },
    {
        'name': 'frailty',
        'method': 'POST',
        'path': '/api/survival_advanced/frailty',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'cluster_col': 'str',
            'predictors': 'list[str]',
            'penalizer': 'float?',
            'frailty_distribution': 'str?',
            'estimation_method': 'str?',
            'nested_cluster_cols': 'list[str]?',
            'correlated_cluster_col': 'str??',
            'baseline_hazard': 'str?',
            'include_diagnostics': 'bool?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Shared Frailty',
    },
    {
        'name': 'interval_censored',
        'method': 'POST',
        'path': '/api/survival_advanced/interval_censored',
        'session_in': 'body',
        'body': {
            'lower_col': 'str',
            'upper_col': 'str',
            'covariates': 'list[str]?',
            'group_col': 'str??',
        },
        'requires': {},
        'doc': 'Interval Censored',
    },
    {
        'name': 'landmark',
        'method': 'POST',
        'path': '/api/survival_advanced/landmark',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'landmark_time': 'float',
            'group_col': 'str??',
            'predictors': 'list[str]??',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Landmark Analysis',
    },
    {
        'name': 'mice',
        'method': 'POST',
        'path': '/api/survival_advanced/mice',
        'session_in': 'body',
        'body': {
            'columns': 'list[str]',
            'n_imputations': 'int?',
            'max_iter': 'int?',
            'random_state': 'int?',
            'mechanism': 'str?',
            'new_columns': 'bool?',
        },
        'requires': {},
        'doc': 'Mice Imputation',
    },
    {
        'name': 'mice_preview',
        'method': 'POST',
        'path': '/api/survival_advanced/mice_preview',
        'session_in': 'body',
        'body': {
            'columns': 'list[str]',
            'max_iter': 'int?',
            'random_state': 'int?',
            'mechanism': 'str?',
        },
        'requires': {},
        'doc': 'Mice Preview',
    },
    {
        'name': 'mice_transfer',
        'method': 'POST',
        'path': '/api/survival_advanced/mice_transfer',
        'session_in': 'body',
        'body': {
            'preview_rows': 'list[dict]',
        },
        'requires': {},
        'doc': 'Mice Transfer',
    },
    {
        'name': 'ml_survival_benchmark',
        'method': 'POST',
        'path': '/api/survival_advanced/ml_survival_benchmark',
        'session_in': 'body',
        'body': {
            'duration_col': 'str?',
            'event_col': 'str?',
            'predictors': 'list[str]??',
            'n_estimators': 'int?',
            'nested_cv': 'bool?',
            'repeated_cv_repeats': 'int?',
            'cv_folds': 'int?',
            'inner_cv_folds': 'int?',
            'hyperparameter_iter': 'int?',
            'include_shap': 'bool?',
            'include_partial_dependence': 'bool?',
            'include_competing_risks_ml': 'bool?',
            'optimization_method': 'str?',
        },
        'requires': {},
        'doc': 'Ml Survival Benchmark',
    },
    {
        'name': 'multistate',
        'method': 'POST',
        'path': '/api/survival_advanced/multistate',
        'session_in': 'body',
        'body': {
            'id_col': 'str?',
            'from_state_col': 'str?',
            'to_state_col': 'str?',
            'entry_col': 'str?',
            'exit_col': 'str?',
            'event_col': 'str?',
            'predictors': 'list[str]',
            'imputation': 'str??',
            'transition_model_type': 'str??',
        },
        'requires': {},
        'doc': 'Multistate',
    },
    {
        'name': 'recurrent_lwyy',
        'method': 'POST',
        'path': '/api/survival_advanced/recurrent_lwyy',
        'session_in': 'body',
        'body': {
            'id_col': 'str',
            'start_col': 'str',
            'stop_col': 'str',
            'event_col': 'str',
            'predictors': 'list[str]',
            'group_col': 'str??',
            'model_type': 'str?',
            'event_order_col': 'str??',
            'time_scale': 'str?',
            'terminal_time_col': 'str??',
            'terminal_event_col': 'str??',
            'include_negative_binomial': 'bool?',
            'include_joint_frailty_spec': 'bool?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Recurrent Lwyy',
    },
    {
        'name': 'rmst',
        'method': 'POST',
        'path': '/api/survival_advanced/rmst',
        'session_in': 'body',
        'body': {
            'duration_col': 'str',
            'event_col': 'str',
            'tau': 'float',
            'group_col': 'str??',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Rmst',
    },
    {
        'name': 'arima',
        'method': 'POST',
        'path': '/api/timeseries/arima',
        'session_in': 'body',
        'body': {
            'value_col': 'str',
            'time_col': 'str??',
            'p': 'int?',
            'd': 'int?',
            'q': 'int?',
            'P': 'int?',
            'D': 'int?',
            'Q': 'int?',
            's': 'int?',
            'auto': 'bool?',
            'forecast_steps': 'int?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Arima',
    },
    {
        'name': 'decompose',
        'method': 'POST',
        'path': '/api/timeseries/decompose',
        'session_in': 'body',
        'body': {
            'value_col': 'str',
            'time_col': 'str??',
            'period': 'int?',
            'method': 'str?',
            'model': 'str?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Decompose',
    },
    {
        'name': 'stationarity',
        'method': 'POST',
        'path': '/api/timeseries/stationarity',
        'session_in': 'body',
        'body': {
            'value_col': 'str',
            'time_col': 'str??',
            'n_lags': 'int?',
            'imputation': 'str??',
        },
        'requires': {},
        'doc': 'Stationarity',
    },
]

__all__ = [
    "Tool",
    "TOOLS",
    "build_catalog",
    "check_requires",
    "get_tool",
]
