# LLM-Driven Backend Automation Plan

**Goal:** Let an LLM agent (Claude/GPT/local model) drive the uSTAT backend end-to-end
with **no browser and no human clicks**: start the API, upload a dataset, and run the
requested statistical analyses by calling HTTP endpoints, then return interpreted results.

This plan adds a thin **agent-facing layer** on top of the existing FastAPI backend. The
backend itself needs *no rewrite* — every analysis is already an HTTP endpoint keyed by a
`session_id`. What is missing is (1) a machine-readable tool catalog, (2) a typed client
the model can call, and (3) an orchestration prompt/loop that maps a natural-language ask
to a sequence of endpoint calls.

---

## 0. What already exists (survey — do not rebuild)

| Concern | Where | Note |
|---|---|---|
| App entrypoint | `backend/main.py` → `app` | FastAPI, `uvicorn main:app --port 8000` |
| Desktop launcher | `backend/desktop_main.py` | embeds uvicorn; reuse for local one-shot |
| Health probe | `GET /api/health` | returns `{status, active_sessions, memory}` |
| Upload | `POST /api/upload/` (multipart `file`) | returns `session_id`, `columns[]`, `preview[]` |
| Session store | `backend/services/store.py` | **in-memory, keyed by `session_id`**, autosaved to disk |
| Analyses | `backend/routers/*.py` | ~30 routers, all take `session_id` + JSON body |
| Column typing | `upload._detect_kind` | numeric / categorical / ordinal / date / text |

**Key architectural fact:** the backend is *stateful per session*. Upload once → get a
`session_id` → pass that id to every subsequent analysis call. The agent never re-sends the
dataframe; it references the session. This is the whole reason automation is cheap.

### Endpoint contract pattern

Two shapes exist and the agent layer must handle both:

- **Path-param sessions** (compute, some stats GETs): `POST /api/compute/{session_id}/formula`
- **Body-param sessions** (most analyses): `POST /api/stats/table1` with `{"session_id": "...", ...}`

Representative bodies (verified against source):

```jsonc
// POST /api/stats/table1   (backend/routers/stats/descriptive.py:690)
{ "session_id": "…", "variables": ["AGE","SEX","DM"], "group_column": "ARM",
  "selected_stats": ["auto"], "normality_mode": "overall" }

// POST /api/stats/fisher   (inferential.py:198 — requires a 2×2)
{ "session_id": "…", "row_column": "STROKE", "col_column": "ARM" }

// POST /api/stats/chisquare, /ttest, /anova, /mannwhitney, /kruskal, /roc … same style
```

---

## 1. Deliverables (new files)

```
backend/agent/
  __init__.py
  tool_catalog.py        # machine-readable registry: name → {method, path, body schema, doc}
  client.py              # UstatClient: typed Python wrapper over httpx (start/upload/call)
  runner.py              # orchestration loop: NL request → tool calls → results
  prompts.py             # system prompt + tool-use instructions for the model
docs/
  AGENT_TOOLS.md         # auto-generated human view of tool_catalog (for review)
scripts/
  agent_demo.py          # end-to-end: boot server → upload sample → run analyses → print
tests/
  test_agent_client.py   # client + catalog contract tests (mock backend)
  test_agent_e2e.py      # real uvicorn subprocess, real upload, assert results
```

No production endpoint changes in Phase 1–3. Optional Phase 4 adds one convenience endpoint.

---

## 2. Bring-up: start the backend programmatically

Two supported modes; the client picks based on config.

### 2a. Managed subprocess (default for a headless agent)

```python
# backend/agent/client.py  (excerpt)
import subprocess, sys, time, httpx

class UstatServer:
    def __init__(self, host="127.0.0.1", port=8000, cwd="backend"):
        self.base = f"http://{host}:{port}"
        self._proc = None; self._cwd = cwd; self._host = host; self._port = port

    def start(self, timeout=30):
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--host", self._host, "--port", str(self._port),
             "--workers", "1", "--log-level", "warning"],
            cwd=self._cwd,
        )
        self._wait_healthy(timeout)

    def _wait_healthy(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.base}/api/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.4)
        raise RuntimeError("uvicorn did not become healthy")

    def stop(self):
        if self._proc: self._proc.terminate(); self._proc.wait(10)
```

Health gate is the existing `GET /api/health`. **Never** proceed to upload before it returns
`status: ok` — a race here is the most common automation failure.

### 2b. Attach to a running server

If a dev server is already up (ports 5173 frontend / 8000 backend), the client just points at
`http://127.0.0.1:8000` and skips `start()`. Do **not** spawn a second uvicorn on the same port.

Env the agent may set before start: `MAX_UPLOAD_BYTES`, `CORS_ALLOWED_ORIGINS`,
`ENABLE_CODE_RUNNER` (leave off unless sandboxed), `USTAT_FRONTEND_DIST` (ignore for API-only).

---

## 3. Load the dataset

```python
def upload(self, path: str) -> dict:
    with open(path, "rb") as fh:
        r = httpx.post(f"{self.base}/api/upload/",
                       files={"file": (os.path.basename(path), fh)}, timeout=120)
    r.raise_for_status()
    return r.json()   # {session_id, filename, rows, columns:[{name,dtype,kind,...}], preview}
```

Accepted formats (from `upload.SUPPORTED`): `.csv .xlsx .xls .sas7bdat .sav .dta`. The response
`columns[]` carries the **inferred `kind`** per column and, for SPSS/Stata, `label` +
`value_labels`. **The agent must read `columns[]` and build a column→kind map before choosing
tests** — e.g. it should not run a t-test on a column whose `kind` is `categorical`.

Persist `session_id` in the runner state; it is the handle for everything downstream.

---

## 4. Tool catalog (the core new artifact)

A declarative registry the model consumes as its tool schema. One entry per analysis:

```python
# backend/agent/tool_catalog.py  (shape)
Tool = TypedDict("Tool", {
    "name": str, "method": str, "path": str,
    "session_in": Literal["body", "path"],
    "body": dict,          # JSON-schema of the non-session params
    "requires": dict,      # semantic guards, e.g. {"group_column": "categorical<=?"}
    "doc": str,
})

TOOLS: list[Tool] = [
  {"name": "table1", "method": "POST", "path": "/api/stats/table1", "session_in": "body",
   "body": {"variables": "list[str]", "group_column": "str?", "selected_stats": "list[str]?",
            "normality_mode": "'overall'|'within_group'?"},
   "requires": {"variables": "any", "group_column": "categorical"},
   "doc": "Publication Table 1: descriptives + auto test selection (chi-sq/Fisher/t/MWU)."},

  {"name": "fisher", "method": "POST", "path": "/api/stats/fisher", "session_in": "body",
   "body": {"row_column": "str", "col_column": "str"},
   "requires": {"row_column": "categorical(2)", "col_column": "categorical(2)"},
   "doc": "Fisher exact test on a 2×2. Use for small expected cell counts."},

  {"name": "chisquare", "method": "POST", "path": "/api/stats/chisquare", "session_in": "body",
   "body": {"row_column": "str", "col_column": "str"}, "requires": {...},
   "doc": "Chi-square test of independence for r×c categorical association."},

  # … ttest, anova, mannwhitney, kruskal, roc, correlation_matrix, km/cox (models router),
  #    missing_data, decision_curve, etc. One row each.
]
```

**Generation, not hand-maintenance:** build `tool_catalog.py` (or validate it in CI) by walking
`app.routes` at import time — FastAPI already knows every path, method, and Pydantic body model.
A generator script reads `app.openapi()` and diffs against the catalog so a new endpoint can't
silently drift out of the agent's reach. `docs/AGENT_TOOLS.md` renders from the same source.

### Semantic guards (`requires`)

The catalog encodes *when* a tool is valid so the runner can reject a bad call before it hits the
API (cheaper, clearer errors than a 400):

- `categorical(2)` → column `kind` categorical **and** exactly 2 levels (Fisher, 2-group t-test)
- `categorical` → any categorical (group_column for ANOVA/Kruskal)
- `numeric` → outcome for t-test/ANOVA/correlation

These map directly onto the `kind` field already returned by upload.

---

## 5. Orchestration loop

```
runner.run(nl_request, dataset_path):
  1. server.start()  (or attach)                     # §2
  2. meta = client.upload(dataset_path)              # §3
     cols = {c["name"]: c["kind"] for c in meta["columns"]}
  3. plan = model.plan(nl_request, cols, TOOLS)      # model emits ordered tool calls
  4. for call in plan:
        validate(call, cols, TOOLS.requires)         # semantic guard, §4
        resp = client.call(call.name, session_id, call.args)
        record(resp)                                  # keep raw JSON
  5. summary = model.interpret(records, nl_request)   # NL answer + tables
  6. server.stop()
  return {summary, records}
```

The model is invoked twice: **plan** (choose tools + args from the column map) and **interpret**
(turn raw endpoint JSON into prose + tables). Everything in between is deterministic Python. This
keeps the model out of the transport and makes runs reproducible/testable by mocking step 3.

### Tool-call transport options (pick one)

- **Native tool use** (Claude/OpenAI function calling): emit `TOOLS` as function schemas; the
  loop executes each `tool_use` block and feeds results back. Preferred.
- **MCP server**: wrap `client.py` as an MCP server exposing `upload` + one tool per analysis so
  any MCP-capable client drives it. Good for reuse across agents; more scaffolding.
- **Plain JSON protocol**: model returns `{"calls":[…]}`; loop parses. Works with any model,
  weakest typing.

---

## 6. Test-selection correctness (why this plan exists)

The manuscript convention is *"Fisher's exact where appropriate"* (small expected cell counts).
The backend has **two** categorical-p code paths and they disagree:

- `stats/descriptive.py::_categorical_p_with_rule` — **correct**: runs `chi2_contingency`,
  inspects `expected`, falls back to Fisher (2×2) or Fisher–Freeman–Halton MC (r×c) when any
  expected cell < 5. This backs Table 1.
- `charts.py::_component_pvalue` — **group-count only**: 2 groups → always Fisher, ≥3 → always
  chi-square, **ignores expected counts**. This backs Score Composite component prevalence.

**Agent-layer requirement:** the runner must document *which* endpoint produced each p-value and
which rule it used, so a reviewer never has to guess (this is exactly the Supplementary-Table-5
confusion). Concretely, `interpret` annotates every p with its `test` string (endpoints already
return `test` / `methods_text`). Optional Phase 4: unify `_component_pvalue` to reuse
`_categorical_p_with_rule` so both paths honor the same "Fisher where appropriate" rule.

---

## 7. Error handling & verification

- **Health race:** gate every run on `/api/health` (§2). Fail loud, don't retry uploads blindly.
- **Non-finite stats:** backend converts NaN/Inf into a `400` with a readable `detail`
  (`main.py:_value_error_handler`). Runner surfaces that `detail` verbatim — do not swallow.
- **Wrong column kind:** caught by `requires` guards before the call; runner returns a
  precondition error naming the column and its actual `kind`.
- **Truncated preview:** upload `preview` is capped at 2000 rows — never treat it as the full
  dataset; always compute on the server via `session_id`.
- **Verification rule (from repo CLAUDE.md):** after building the layer, actually run
  `pytest backend/tests/test_agent_*.py` and the demo script; never report success on a file
  write alone.

---

## 8. Phased execution (≤5 files per phase, verify between)

- **Phase 1 — client + catalog (foundation).** `agent/client.py`, `agent/tool_catalog.py`,
  `agent/__init__.py`, `scripts/agent_demo.py`, `tests/test_agent_client.py`.
  Verify: demo boots uvicorn, uploads a fixture CSV, runs `table1` + `fisher`, prints JSON;
  client tests green against a mocked backend.
- **Phase 2 — catalog generator + guards.** OpenAPI-diff generator, `requires` validation,
  `docs/AGENT_TOOLS.md`. Verify: generator run leaves catalog unchanged (no drift); guard unit
  tests reject mistyped calls.
- **Phase 3 — orchestration + prompts.** `agent/runner.py`, `agent/prompts.py`,
  `tests/test_agent_e2e.py`. Verify: e2e test drives a real subprocess server through a scripted
  plan and asserts numeric results match direct endpoint calls.
- **Phase 4 (optional) — parity + convenience.** Unify `_component_pvalue` with
  `_categorical_p_with_rule` (§6); optionally add `POST /api/agent/run` so an external caller
  triggers the whole loop server-side. Verify: `test_score_composite_chart.py` still green;
  new parity test asserts small-cell r×c uses Fisher–Freeman–Halton.

---

## 9. Minimal end-to-end (what the demo proves)

```python
srv = UstatServer(cwd="backend"); srv.start()
meta = client.upload("qa/fixtures/trial.csv")
sid  = meta["session_id"]
t1   = client.call("table1", sid, {"variables": ["AGE","SEX","STROKE","CKD","PAD"],
                                   "group_column": "ARM", "selected_stats": ["auto"]})
fx   = client.call("fisher", sid, {"row_column": "STROKE", "col_column": "ARM"})
print(t1["rows"], fx["p"], fx["test"])
srv.stop()
```

Success = server boots, dataset loads, Table 1 rows + a Fisher p come back with their `test`
labels, all without touching the browser.
```
