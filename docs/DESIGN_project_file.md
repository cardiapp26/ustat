# uSTAT Project File (.ustat) Design

Status: Phase 1 backend landed (2026-09-09): container build/parse/restore
in `backend/services/project_file.py`, `/api/project/{sid}/save` and
`/api/project/load` in `backend/routers/project.py`, legacy v1.x JSON
import, round-trip and integrity tests in
`backend/tests/test_project_file.py`. Phase 1 frontend landed the same
day: the header Save menu offers "Project (.ustat)", the upload zone opens
`.ustat` files, and every restore path (upload, Recent Sessions, crash
recovery) goes through `/api/project/load`, which reads legacy JSON and
`.ustat` alike by content. Deferred from Phase 1: autosave still snapshots
the v1.2 JSON, because `.ustat` bytes are not deterministic (zip entry
timestamps, `manifest.modified`) and the autosave dedupe hashes the whole
payload; switching it lands with Phase 2 alongside a content-stable hash
over the parts that matter. Phases 2-4 remain. Implements the P0 "Project
file" row of [ROADMAP_product_maturity.md](ROADMAP_product_maturity.md).

## Goal

One file that holds a research project: data, its preparation history,
the variable dictionary, multiple named analyses with their settings,
their results and plots, and enough provenance to say what produced every
number. Opening the file on another machine reproduces the project;
closing the program loses nothing.

Non-goals for this design:

- Not a replacement for dataset export (CSV/XLSX/SAV keep working as is).
- Not an end-to-end reproducibility guarantee across engine versions. The
  file records the versions that produced each result; it does not promise
  a different engine build will match them. Re-run and compare is the
  supported operation.
- Not multi-user or concurrent editing.

## What exists today (and what it lacks)

- `GET /api/sessions/{sid}/save_session` produces a single JSON (v1.2)
  with `data` (records), `columns`, `col_metadata`, `kind_overrides`,
  `decimals_overrides`, `case_filter`, `audit`, `filename`
  (`backend/routers/session.py`). No analyses, no results, no plots, no
  notes, no versions/hash/seed.
- The browser autosaves that same blob to IndexedDB with light metadata
  (`frontend/src/lib/sessionDb.ts`) and re-uploads it to resume.
- Panel settings and cached results live in the Zustand `panelCache` and
  are not part of the saved file. A panel cache is not a portable project.
- Results now carry a `ResultStamp` (data version, filter, params, engine)
  and provenance (engine + library versions) in the browser
  (`frontend/src/lib/resultStamp.ts`, `lib/engine/provenance.ts`). These
  die with the tab; the project file is where they should live.
- The audit trail is `{action, params, timestamp}` entries: informative,
  but not defined as replayable.

## Container format

A `.ustat` file is a **zip archive with named JSON parts** (jamovi's
`.omv` precedent). Zip rather than one big JSON because parts can be read
selectively (open the dictionary without parsing 100 MB of data), binary
parts (plot images, a future Parquet dataset) fit without base64, and
unknown parts survive re-save byte-for-byte, which is the
forward-compatibility rule.

```
project.ustat
├── manifest.json          identity, versions, hashes, seed
├── data/
│   ├── dataset.json       records, same serializer as save_session v1.2
│   └── originals.json     ingest-preserved raw cells + coercion report
├── dictionary.json        per-column semantics
├── prep/steps.json        ordered, replayable preparation recipe
├── analyses/index.json    the analysis tree: id, name, kind, order
├── analyses/{id}.json     one saved analysis: params + engine choice
├── results/{id}.json      result payload + ResultStamp + provenance
├── plots/{id}.json        Plotly layout/config for that analysis's plots
├── notes.md               researcher notes, free Markdown
└── audit.json             full audit trail (informational)
```

### manifest.json

```json
{
  "schema_version": "2.0.0",
  "kind": "ustat-project",
  "created": "2026-09-09T12:00:00Z",
  "modified": "2026-09-09T14:30:00Z",
  "app_version": "…",
  "engine": {"kind": "python", "detail": "…", "packages": {"scipy": "…"}},
  "seed": 12345,
  "data_hash": "sha256:…",
  "parts": {"data/dataset.json": "sha256:…", "dictionary.json": "sha256:…"}
}
```

- `data_hash` is the sha256 of the canonical serialization of
  `data/dataset.json`. Every result stamp references it; on open, a result
  whose stamp does not match the current hash is marked stale, reusing the
  existing `StaleResultNotice` path.
- `engine`/`packages` come from the runtime identity work
  (`backend/services/runtime_identity.py`, `X-uStat-*` headers): the file
  records what actually produced each result, per result, and the manifest
  records the environment at last save.
- `seed`: the project-level random seed handed to seedable analyses
  (bootstrap, MICE, ML CV splits). Recorded so re-run means re-run.

### dictionary.json

Per column: name, dtype, kind (numeric/categorical/ordinal/datetime/text),
label, unit, value labels, **user-defined missing codes**, **category
order** (for ordinals), **reference group** (for modeling), decimal
places. This is the P0 "data semantics" row made durable: import, save,
open, export must not lose any of these.

### prep/steps.json

The replayable recipe, distinct from the audit log:

```json
[
  {"op": "import", "params": {"filename": "trial.csv", "options": {…}}},
  {"op": "define_missing", "params": {"column": "age", "codes": [999]}},
  {"op": "compute", "params": {"target": "bmi", "formula": "kg/(m*m)"}},
  {"op": "filter", "params": {"conditions": […]}}
]
```

Rules: each step is `{op, params}` with a documented, versioned op
vocabulary; steps are ordered and deterministic; replaying them against
the imported raw data must reproduce `data/dataset.json` (a round-trip
test enforces this). The audit log stays as the informational "what
happened when"; the recipe is the normative "how to rebuild". This is
also the substrate for the P1 "R code / Python code" script export: a
script generator walks the same steps plus the saved analyses.

### analyses/ and results/

A saved analysis is the durable form of what a panel run is today:

```json
{
  "id": "a3f2…", "name": "Model 2 (adjusted)", "kind": "models/logistic",
  "engine": "python", "params": {…}, "created": "…"
}
```

`params` is exactly the request body the panel sent (the same object the
`ResultStamp.paramsKey` is derived from today). `results/{id}.json` holds
the response payload plus the stamp and per-result provenance. The UI
grows a project tree ("Model 1", "Model 2", "Sensitivity analysis"):
create, rename, duplicate, re-run. Re-run posts `params` again and
replaces the result with a fresh stamp.

## Save / load flow

- Backend: `POST /api/project/{sid}/save` streams the zip;
  `POST /api/project/load` accepts it, restores the dataset + dictionary
  into the session store, and returns the analyses/results parts for the
  frontend to hydrate its project tree. The frontend owns panel state;
  the backend owns data state; the file carries both.
- Autosave: `sessionDb.ts` stores the whole `.ustat` blob instead of the
  v1.2 JSON. Same Dexie schema, same LRU/trash policy; `payload` becomes
  binary (Blob) with a `format: "ustat" | "json-v1"` discriminator.
- Desktop (Tauri): register the `.ustat` file association; File > Save
  writes through to the chosen path instead of a browser download.
- Legacy: `load_session` keeps accepting v1.0-1.2 JSON files forever and
  imports them as a project with data + dictionary + audit and zero saved
  analyses. First save writes `.ustat`.

## Versioning and migration

- `schema_version` is semver. Loader policy: same major = load, preserve
  unknown keys and unknown zip parts untouched on re-save; older major =
  run recorded migrations (v1.x JSON import is migration zero); newer
  major = refuse with a clear "made by a newer uSTAT" message rather than
  a partial load.
- Every schema change lands with a migration and a fixture file in
  `backend/tests/fixtures/projects/` so old files stay loadable under CI.

## Phasing

1. **Container + data semantics**: zip writer/reader, manifest,
   dictionary, data + originals, notes, audit; legacy import; round-trip
   tests (import, save, open, export with no loss of meaning).
2. **Saved analyses**: analyses/results/plots parts, project tree UI,
   re-run from params, stamps checked against `data_hash` on open.
3. **Recipe**: formalize prep ops, replay test, script export
   (R/Python) generated from steps + analyses.
4. **Polish**: seeds threaded into every seedable endpoint, Drive/cloud
   sync of `.ustat` blobs, size hardening (streaming zip for >100 MB
   datasets).

## Risks

- **Size**: records-JSON is verbose; the 200 MB IndexedDB cap is real.
  Mitigation: zip deflate (records compress well), later a Parquet part
  (needs pyarrow, currently not a backend dependency).
- **Two owners of truth**: backend session vs frontend project tree can
  drift. Rule: the file is the only durable truth; both sides hydrate
  from it and neither invents state the file cannot carry.
- **Engine drift**: a re-run under newer libraries may move numbers.
  That is surfaced, not hidden: stamp + provenance per result, and the
  validation-card tolerance work decides what counts as a deviation.

## Acceptance (from the roadmap)

- Open the `.ustat` on another machine: same data, dictionary, analyses,
  results, notes; stale marking correct; re-run reproduces results within
  documented tolerances under the same engine versions.
- Kill the app mid-work: autosave restores the project including saved
  analyses, not just the dataset.
