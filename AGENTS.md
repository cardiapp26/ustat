# AGENTS.md

Instructions for AI coding agents (Codex, Claude, etc.) working in this repo.
Human-facing docs live in [README.md](README.md), [MANUAL.md](MANUAL.md), and
[docs/SPEC.md](docs/SPEC.md) — read those for feature/architecture depth. This
file is the short version: how to run things, where things live, what not to
break.

## What this is

uSTAT — a server-hosted clinical biostatistics platform (SPSS/R/Stata
alternative). FastAPI backend does all statistical computation server-side;
React frontend only renders results. No persistence, no accounts — sessions
are in-RAM with a 30 min TTL (`backend/services/store.py`).

## Stack

- Backend: Python 3.11+, FastAPI, pandas/numpy/scipy/statsmodels/lifelines/scikit-learn
- Frontend: React 19 + TypeScript, Vite, Zustand (`frontend/src/store.ts`), Plotly.js
- Desktop: Tauri v2 (`src-tauri/`) wraps the same frontend + a bundled backend binary

## Run locally

```bash
# backend
cd backend && source .venv/bin/activate  # venv already exists at repo root .venv typically
uvicorn main:app --reload --port 8000

# frontend
cd frontend && npm run dev   # Vite on :5173, proxies /api/* to :8000
```

Health check: `curl http://localhost:8000/api/health`.

## Test & lint before calling anything done

```bash
# backend
cd backend && python -m pytest -q

# frontend
cd frontend && npx tsc --noEmit && npm run lint && npm run test
```

CI (`.github/workflows/ci.yml`) runs backend pytest+coverage and frontend
tsc/build/lint on every push — don't skip these locally.

## Code layout

- `backend/routers/*` — thin FastAPI routers, one per domain (`models/`,
  `stats/`, `session.py`, `compute.py`, `charts.py`, …). Routers validate
  input and call services; they should not contain statistical logic.
- `backend/services/*` — pure, focused modules (20+) doing the actual
  computation. No god-files — if a service file is growing past ~500 LOC,
  split it.
- `backend/services/store.py` — the in-memory session store. All per-session
  state (dataframe, filename, column kind/decimals overrides, undo/redo,
  audit log) lives here in parallel dicts keyed by `session_id`.
- `frontend/src/components/*Panel.tsx` — one component per analysis tab.
- `frontend/src/store.ts` — Zustand store, including `panelCache` (each
  panel's persisted variable selections across tab switches) and
  `renameInPanelCache` (keeps that cache in sync with column renames).
- `frontend/src/api.ts` — all backend calls; add new endpoints here rather
  than calling `axios`/`fetch` directly from components.

## Conventions that matter

- **Response contracts are load-bearing.** Panels (e.g. `IPTWPanel.tsx`,
  `PSMPanel.tsx`) destructure specific fields from the backend response with
  no optional-chaining in places — changing a router's response shape
  without updating the panel (or vice versa) causes a silent runtime crash,
  not a type error. When touching `routers/models/*` or `routers/charts.py`,
  grep the matching frontend panel for the fields you're changing.
- **Kind vs dtype.** A column's "kind" (numeric/categorical/date/ordinal) is
  a declared override (`store.get_kind_overrides`), separate from pandas
  dtype. A blank numeric column has `object` dtype until something is
  written to it — code that branches on dtype instead of declared kind will
  silently mishandle blank columns. See `backend/routers/session.py`
  `update_cell` for the pattern.
- **No second implementation of a statistic.** If you're about to compute a
  p-value, effect size, or any inferential statistic in TypeScript, stop. The
  rule is not "statistics run on the server" — it is that a statistic has
  exactly one implementation, in Python, cross-checked against the
  peer-reviewed library it wraps. That one implementation lives in
  `backend/ustat_engine/` and runs in two places: this server, and the
  browser, where the same package is installed as a wheel under Pyodide so
  patient data need never leave the machine. Both runtimes compute a sha256
  over the engine's own sources and refuse to run locally unless they match,
  so the two can never silently answer the same question differently.
  `backend/tests/test_engine_isolation.py` enforces what the engine may
  import; `backend/requirements.txt` explains why numpy/scipy/pandas are
  pinned to Pyodide's versions and must not be bumped on one side alone.
- **Turkish bug reports are common** (primary user base). Repro and fix in
  English-named code/commits as usual; just expect issue text in Turkish.
- **Git hygiene**: several files in this repo tend to carry large unrelated
  pre-existing uncommitted diffs from other in-progress work. Before
  `git add` on a file you edited, run `git diff --stat` on it — if the diff
  is much bigger than your actual change, isolate your hunk instead of
  staging the whole file.
- **Commits**: Conventional Commits format (`fix:`, `feat:`, `chore:`, …).
  Don't bundle unrelated changes into one commit.

## Where NOT to look for instructions

Personal global CLI configs (e.g. a user's `~/.claude/*` files, shell
aliases, token-optimizing proxies) are environment-specific to whoever is
driving the agent — they are never part of this repo and should never be
added to it.
