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

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.cmd === "init") {
      const info = await boot(msg.packages, msg.runtimeBase, msg.wheelBase);
      post({ id: msg.id, ok: true, kind: "init", identity: info });
      return;
    }

    if (msg.cmd === "run") {
      if (!pyodide) throw new Error("engine was not initialised");
      pyodide.globals.set("_ustat_analysis", msg.analysisId);
      pyodide.globals.set("_ustat_params", pyodide.toPy(msg.params));
      const raw = pyodide.runPython(`
import json
import ustat_engine
try:
    _r = ustat_engine.run(_ustat_analysis, params=_ustat_params)
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
