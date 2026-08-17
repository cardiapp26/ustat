/**
 * The statistics engine, running in a worker.
 *
 * A worker rather than the main thread because these calls are not fast. A Cox
 * fit or a bootstrap runs for seconds to minutes; on the main thread that is a
 * frozen tab, no spinner, no cancel. The user would conclude the app had
 * crashed, which for a long analysis is indistinguishable from the truth.
 *
 * Everything is loaded from this origin. Not from a CDN -- the app's CSP
 * forbids it, and a CDN would otherwise be able to observe which statistical
 * packages a given clinician just loaded, which is a weaker version of the
 * problem this whole feature exists to solve.
 */
/// <reference lib="webworker" />
import type { EngineIdentity, WorkerRequest, WorkerResponse } from "./types";

interface PyodideApi {
  loadPackage(names: string[]): Promise<void>;
  runPython(code: string): string;
  pyimport(name: string): { install(url: string): Promise<void> };
  globals: { set(name: string, value: unknown): void };
  toPy(value: unknown): unknown;
}

let pyodide: PyodideApi | null = null;
const loadedPackages = new Set<string>();
let identity: EngineIdentity | null = null;

function post(message: WorkerResponse): void {
  (self as unknown as Worker).postMessage(message);
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const text = await r.text();
  // A dev server answers a missing file with the SPA index at HTTP 200, so a
  // 200 is not by itself evidence the file exists.
  if (/^\s*<(!doctype|html)/i.test(text)) {
    throw new Error(`${url}: not found (served the SPA fallback)`);
  }
  return JSON.parse(text) as T;
}

async function boot(packages: string[], runtimeBase: string, wheelBase: string) {
  if (!pyodide) {
    const mod = (await import(/* @vite-ignore */ `${runtimeBase}pyodide.mjs`)) as {
      loadPyodide(options: { indexURL: string }): Promise<PyodideApi>;
    };
    pyodide = await mod.loadPyodide({ indexURL: runtimeBase });
  }

  const missing = packages.filter((p) => !loadedPackages.has(p));
  if (missing.length) {
    // micropip is always needed: the engine itself arrives as a wheel.
    await pyodide.loadPackage([...missing, "micropip"]);
    missing.forEach((p) => loadedPackages.add(p));
    loadedPackages.add("micropip");
  }

  if (!identity) {
    const manifest = await fetchJson<{ wheel: string; source_fingerprint: string }>(
      `${wheelBase}manifest.json`,
    );
    const micropip = pyodide.pyimport("micropip");
    await micropip.install(`${wheelBase}${manifest.wheel}`);

    identity = JSON.parse(
      pyodide.runPython(`
import json
import ustat_engine
import ustat_engine.stats.power
try:
    import ustat_engine.meta
except ImportError:
    pass
json.dumps(ustat_engine.identity())
`),
    ) as EngineIdentity;

    if (identity.fingerprint !== manifest.source_fingerprint) {
      // The wheel that arrived is not the wheel that was built -- a stale
      // service-worker cache, a half-finished deploy. Refusing here is the
      // point: computing anyway would return numbers from code nobody shipped.
      throw new Error(
        `engine fingerprint ${identity.fingerprint} does not match the manifest ` +
          `(${manifest.source_fingerprint})`,
      );
    }
  }

  return identity;
}

/**
 * Load Pyodide packages the boot plan did not ask for.
 *
 * Frames need pandas, and the load plan is per analysis: `stats.power` asks for
 * numpy/scipy/statsmodels and nothing else, so pandas is not there until a
 * frame arrives. Paying for it at boot would charge every local run for a
 * dependency most of them never touch.
 */
async function ensurePackages(names: string[]): Promise<void> {
  if (!pyodide) throw new Error("engine was not initialised");
  const missing = names.filter((p) => !loadedPackages.has(p));
  if (!missing.length) return;
  await pyodide.loadPackage(missing);
  missing.forEach((p) => loadedPackages.add(p));
}

/**
 * Datasets the worker is holding, by key. Capped at two.
 *
 * Two rather than one so switching between an unfiltered and a filtered view
 * (or between two column-sets of the same dataset) does not re-transfer on
 * every toggle; two rather than many because these are whole patient datasets
 * sitting in a wasm heap that has a hard ceiling, and a cache that grows until
 * it dies takes the tab with it. Eviction is least-recently-USED, not
 * least-recently-pushed: a frame pushed first and read on every analysis is
 * the one worth keeping.
 */
const FRAME_CACHE_SETUP = `
import ustat_engine.frame.envelope as _ustat_envelope

try:
    _ustat_frames
except NameError:
    _ustat_frames = {}


def _ustat_put_frame(key, envelope, cap=2):
    _ustat_frames.pop(key, None)
    _ustat_frames[key] = _ustat_envelope.frame_from_envelope(envelope)
    while len(_ustat_frames) > cap:
        _ustat_frames.pop(next(iter(_ustat_frames)))


def _ustat_take_frame(key):
    if key not in _ustat_frames:
        return None
    # Re-insert so dict order is use order, which is what the eviction reads.
    _ustat_frames[key] = _ustat_frames.pop(key)
    return _ustat_frames[key]
`;

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.cmd === "init") {
      const info = await boot(msg.packages, msg.runtimeBase, msg.wheelBase);
      post({ id: msg.id, ok: true, kind: "init", identity: info });
      return;
    }

    if (msg.cmd === "frame") {
      if (!pyodide) throw new Error("engine was not initialised");
      await ensurePackages(["pandas"]);
      pyodide.globals.set("_ustat_frame_key", msg.frameKey);
      pyodide.globals.set("_ustat_frame_env", pyodide.toPy(msg.envelope));
      pyodide.runPython(`${FRAME_CACHE_SETUP}
_ustat_put_frame(_ustat_frame_key, _ustat_frame_env)
`);
      post({ id: msg.id, ok: true, result: { frameKey: msg.frameKey } });
      return;
    }

    if (msg.cmd === "run") {
      if (!pyodide) throw new Error("engine was not initialised");
      if (msg.frameKey) {
        pyodide.globals.set("_ustat_frame_key", msg.frameKey);
        const held = pyodide.runPython(`${FRAME_CACHE_SETUP}
"yes" if _ustat_take_frame(_ustat_frame_key) is not None else "no"
`);
        if (held !== "yes") {
          // The caller has to push the frame again. Saying so is better than
          // running without it: an analysis handed no dataset would either
          // error obscurely or, worse, answer from a stale one.
          post({
            id: msg.id,
            ok: false,
            reason: "frame-missing",
            detail: `no frame is held under ${msg.frameKey}`,
          });
          return;
        }
      } else {
        pyodide.globals.set("_ustat_frame_key", null);
      }
      pyodide.globals.set("_ustat_analysis", msg.analysisId);
      pyodide.globals.set("_ustat_params", pyodide.toPy(msg.params));
      const raw = pyodide.runPython(`
import json
import ustat_engine
_frame = _ustat_take_frame(_ustat_frame_key) if _ustat_frame_key else None
try:
    _r = ustat_engine.run(_ustat_analysis, frame=_frame, params=_ustat_params)
    _out = json.dumps({"ok": True, "result": ustat_engine.sanitize(_r)}, allow_nan=False)
except ustat_engine.EngineError as exc:
    _out = json.dumps({"ok": False, "error": str(exc),
                       "status_hint": getattr(exc, "status_hint", None)})
_out
`);
      const parsed = JSON.parse(raw) as
        | { ok: true; result: unknown }
        | { ok: false; error: string; status_hint: number | null };

      if (parsed.ok) {
        post({ id: msg.id, ok: true, result: parsed.result });
      } else {
        // An analysis that rejects its input is a real answer, not a reason to
        // ask the server the same question and get the same rejection.
        post({ id: msg.id, ok: false, reason: "engine-error", detail: parsed.error });
      }
    }
  } catch (err) {
    post({
      id: msg.id,
      ok: false,
      reason: pyodide ? "crashed" : "runtime-load-failed",
      detail: err instanceof Error ? err.message : String(err),
    });
  }
};
