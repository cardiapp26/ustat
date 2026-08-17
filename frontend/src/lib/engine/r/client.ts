/**
 * Main-thread control of the in-browser R engine.
 *
 * The webR counterpart of `../client.ts`, deliberately shaped so the two can be
 * read side by side: same lifecycle (`ensureWorker` → `acquire` → `new Worker`,
 * `teardown` → `release`), same `registerRuntime` at module scope, same
 * `LocalComputeUnavailable` with a named reason on every way out, same
 * remember-the-failure rule so a boot that cannot succeed is not paid for twice.
 *
 * Two things differ, and both are load-bearing:
 *
 *   1. There is no opt-in flag. The Python engine is opt-in because it changes
 *      what an existing default does; R is chosen explicitly, once, at the
 *      welcome gate, and that choice IS the opt-in.
 *   2. Verification takes two links rather than one. The worker proves the
 *      bundle it fetched matches its manifest; this module proves that manifest
 *      matches GET /api/engine/r/identity, and that the manifest agrees with the
 *      copy this app was built against. Any of the three failing means the
 *      browser and the server would answer differently, so nothing runs here.
 */
import type { LocalRunFailure, LocalUnavailableReason } from "../types";
import type { RBootInfo, RWorkerRequest, RWorkerRequestBody, RWorkerResponse } from "./types";
import { LocalComputeUnavailable } from "../client";
import { R_ALLOW_LIST, R_BUNDLE_PACKAGES, rManifestDisagreement } from "./loadPlan";
import { acquire, registerRuntime, release } from "../arbiter";

const RUNTIME_BASE = "/webr/runtime/";
const BUNDLE_BASE = "/webr/bundle/";
/** No trailing slash: webR appends `/bin/emscripten/contrib/<R version>` itself,
 *  and its own default (`https://repo.r-wasm.org`) is written the same way. */
const REPO_URL = "/webr/repo";

let worker: Worker | null = null;
let nextId = 1;
let bootPromise: Promise<RBootInfo> | null = null;
let bootInfo: RBootInfo | null = null;
/**
 * The webR RELEASE this app vendored, from GET /api/engine/r/identity.
 *
 * Preferred over `RBootInfo.webrVersion` for display, because webR's own
 * `WebR.version` is baked at build time by upstream's release tooling and is
 * wrong: the v0.6.0 tarball's webr.mjs reports "0.5.10-dev". `vendored.json`
 * records the tag of the archive that was actually downloaded, which is the
 * thing a reader would go and look up.
 */
let vendoredWebrVersion: string | null = null;
/** Set once R local compute has been ruled out for this page, so it is not retried. */
let disabledFor: LocalRunFailure | null = null;

const pending = new Map<
  number,
  { resolve: (r: RWorkerResponse) => void; reject: (e: unknown) => void }
>();

function ensureWorker(): Worker {
  if (worker) return worker;
  // Claim the tab before the worker exists, not after: the arbiter has to tear
  // down a resident Pyodide runtime BEFORE webR's wasm heap is allocated, or
  // the two are briefly co-resident, which is the exact failure it exists to
  // prevent.
  acquire("r");
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (event: MessageEvent<RWorkerResponse>) => {
    const entry = pending.get(event.data.id);
    if (!entry) return;
    pending.delete(event.data.id);
    entry.resolve(event.data);
  };
  worker.onerror = (event) => {
    const failure: LocalRunFailure = {
      reason: "crashed",
      detail: event.message || "webR worker error",
    };
    pending.forEach((entry) => entry.reject(new LocalComputeUnavailable(failure)));
    pending.clear();
    teardown();
  };
  return worker;
}

function teardown(): void {
  // Terminating this worker takes webR's nested worker with it -- a worker's
  // dedicated children are terminated with their parent -- so there is no
  // separate webR.close() to chase across the boundary.
  worker?.terminate();
  worker = null;
  bootPromise = null;
  bootInfo = null;
  release("r");
  resetHooks.forEach((hook) => hook());
}

// Registered once, at module scope: the arbiter must be able to tear this
// client down without importing it, which is why this is a registration call
// rather than the arbiter reaching in directly.
registerRuntime("r", { teardown });

function send(message: RWorkerRequestBody): Promise<RWorkerResponse> {
  const id = nextId++;
  const w = ensureWorker();
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    w.postMessage({ ...message, id } as RWorkerRequest);
  });
}

interface RServerIdentity {
  version: string;
  fingerprint: string | null;
  modules: number;
  analyses: string[];
  webr_version: string | null;
  r_version: string | null;
}

/** What R engine sources the server holds, so a local result can be shown to match it. */
async function serverRIdentity(): Promise<RServerIdentity> {
  const r = await fetch("/api/engine/r/identity");
  if (!r.ok) throw new Error(`/api/engine/r/identity: HTTP ${r.status}`);
  return (await r.json()) as RServerIdentity;
}

function unavailable(reason: LocalUnavailableReason, detail: string): LocalComputeUnavailable {
  return new LocalComputeUnavailable({ reason, detail });
}

async function boot(analysisId: string): Promise<RBootInfo> {
  const [response, server] = await Promise.all([
    send({
      cmd: "init",
      analysisId,
      packages: [...R_BUNDLE_PACKAGES],
      runtimeBase: RUNTIME_BASE,
      bundleBase: BUNDLE_BASE,
      repoUrl: REPO_URL,
    }),
    serverRIdentity(),
  ]);

  if (!response.ok) {
    throw new LocalComputeUnavailable({ reason: response.reason, detail: response.detail });
  }
  const info = (response as { boot: RBootInfo }).boot;

  if (!info.sourceFingerprint || info.sourceFingerprint !== server.fingerprint) {
    // The bundle's manifest and this server are not describing the same R
    // sources. Computing here would give an answer the server could not
    // reproduce, and the reader would have no way to tell which they had.
    throw unavailable(
      "fingerprint-mismatch",
      `browser R bundle ${info.sourceFingerprint || "unknown"} does not match ` +
        `server R engine ${server.fingerprint ?? "unknown"}`,
    );
  }

  const disagreement = rManifestDisagreement(info.manifestAnalyses);
  if (disagreement) {
    // The allow-list was decided from a build-time copy of the manifest. This
    // is the check that the decision was made on current information.
    throw unavailable("fingerprint-mismatch", `R manifest disagrees with this build: ${disagreement}`);
  }

  bootInfo = info;
  vendoredWebrVersion = server.webr_version;
  return info;
}

/**
 * Why a local R run of `analysisId` cannot even be attempted, or null.
 *
 * Split out of `runLocalR` for the same reason the Python client splits it: a
 * caller must be able to ask BEFORE fetching a patient dataset to hand to an
 * engine that was never going to run it.
 */
export function localRRunBlockedBecause(analysisId: string): LocalRunFailure | null {
  if (disabledFor) return disabledFor;
  if (!R_ALLOW_LIST.has(analysisId)) {
    return {
      reason: "no-r-implementation",
      detail: `${analysisId} has no R implementation verified in a browser yet`,
    };
  }
  if (typeof Worker === "undefined") {
    return { reason: "no-worker-support", detail: "this browser has no module workers" };
  }
  return null;
}

/**
 * Boot webR for `analysisId` (or reuse the running one), or throw.
 *
 * Separate from `runLocalR` because a frame-based analysis has to hand over its
 * dataset before it can run, and there is nothing to hand it to until this has
 * resolved.
 */
export async function ensureREngineBooted(analysisId: string): Promise<void> {
  const blocked = localRRunBlockedBecause(analysisId);
  if (blocked) throw new LocalComputeUnavailable(blocked);

  if (!bootPromise) bootPromise = boot(analysisId);
  try {
    await bootPromise;
  } catch (err) {
    // Booting failed for a reason that will not fix itself on retry -- a
    // missing package on the mirror, a mismatched bundle. Remember it rather
    // than paying the failure again on every subsequent analysis.
    const failure: LocalRunFailure =
      err instanceof LocalComputeUnavailable
        ? { reason: err.reason, detail: err.message }
        : { reason: "r-runtime-load-failed", detail: String(err) };
    disabledFor = failure;
    teardown();
    throw new LocalComputeUnavailable(failure);
  }
}

/**
 * Hand the worker a dataset to keep under `frameKey`.
 *
 * The worker rebuilds the envelope into a data.frame the moment it arrives, so
 * a push to an unbooted worker cannot work; callers that reach here before
 * `runLocalR` must call `ensureREngineBooted` first. Same contract, same
 * failure, same reason as the Python client's `pushFrame`.
 */
export async function pushFrameR(frameKey: string, envelope: unknown): Promise<void> {
  if (disabledFor) throw new LocalComputeUnavailable(disabledFor);
  if (!bootPromise) {
    throw unavailable("r-runtime-load-failed", "pushFrameR was called before the engine was booted");
  }
  const response = await send({ cmd: "frame", frameKey, envelope });
  if (!response.ok) {
    throw new LocalComputeUnavailable({ reason: response.reason, detail: response.detail });
  }
}

export interface RunLocalROptions {
  /** A frame previously handed over with `pushFrameR`. */
  frameKey?: string;
}

export async function runLocalR<T>(
  analysisId: string,
  params: unknown,
  options: RunLocalROptions = {},
): Promise<T> {
  const blocked = localRRunBlockedBecause(analysisId);
  if (blocked) throw new LocalComputeUnavailable(blocked);

  await ensureREngineBooted(analysisId);

  const response = await send({
    cmd: "run",
    analysisId,
    params,
    frameKey: options.frameKey,
  });
  if (!response.ok) {
    if (response.reason === "r-engine-bug") {
      // Loud on purpose. Every other reason here is a fact about the
      // environment; this one is a defect in backend/ustat_engine_r/ that only
      // shows up in a browser, and the fallback would otherwise hide it behind
      // a correct-looking server answer.
      console.error(`[R engine] ${analysisId} raised an R-level error: ${response.detail}`);
    }
    throw new LocalComputeUnavailable({ reason: response.reason, detail: response.detail });
  }
  return (response as { result: T }).result;
}

/** How to describe the engine that just answered, e.g. "R 4.6.0 · webR 0.6.0". */
export function rEngineDetail(): string | undefined {
  if (!bootInfo) return undefined;
  return `R ${bootInfo.identity.r_version} · webR ${vendoredWebrVersion ?? bootInfo.webrVersion}`;
}

/** What the last boot cost, for the welcome screen's honesty about it. */
export function rBootInfo(): RBootInfo | null {
  return bootInfo;
}

/**
 * Start pulling the R runtime down without booting it.
 *
 * Called the moment the user picks R at the welcome gate, because they are
 * about to spend twenty seconds choosing a file and the runtime is ~22 MB. Only
 * the fetch: no worker, no wasm heap, no arbiter claim -- a user who picks R and
 * then never runs an analysis pays for bandwidth they already spent and nothing
 * else.
 *
 * Failures are swallowed. This is a cache warm-up; if it does not happen the
 * real boot fetches the same URLs and reports properly on its own.
 */
export function prefetchRRuntime(): void {
  if (typeof fetch === "undefined") return;
  const urls = [
    `${RUNTIME_BASE}R.wasm`,
    `${RUNTIME_BASE}R.js`,
    `${BUNDLE_BASE}manifest.json`,
  ];
  for (const url of urls) {
    // Same-origin, no credentials, and explicitly cache-friendly so the
    // service worker's CacheFirst rule for /webr/ is what the real boot hits.
    fetch(url, { credentials: "omit" }).catch(() => {
      /* a warm-up that fails is not an error; the boot will report for itself */
    });
  }
}

/**
 * Things to forget when the webR worker goes away.
 *
 * A registry rather than a direct call into `../frame.ts`: importing that from
 * here would be a cycle, and would drag the axios instance and the zustand
 * store into every consumer of this module.
 */
const resetHooks: Array<() => void> = [];

export function onREngineReset(hook: () => void): void {
  resetHooks.push(hook);
}

/** Test seam: forget any recorded failure and drop the worker. */
export function resetREngine(): void {
  disabledFor = null;
  vendoredWebrVersion = null;
  teardown();
}
