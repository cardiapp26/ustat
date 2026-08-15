/**
 * Main-thread control of the in-browser statistics engine.
 *
 * Local compute is opt-in while it is being rolled out: `isLocalComputeEnabled`
 * is false unless the user has turned it on. Nothing here changes what any
 * existing analysis returns until that flag is set, which is the point -- the
 * feature can be built, shipped and exercised before it is trusted with
 * anybody's default behaviour.
 *
 * Every path out of this module either produces a local result or an explicit,
 * named reason why not. Falling back to the server is fine; falling back
 * without recording why is how a "runs on your device" feature quietly stops
 * doing that.
 */
import type {
  EngineIdentity,
  LocalRunFailure,
  LocalUnavailableReason,
  WorkerRequest,
  WorkerRequestBody,
  WorkerResponse,
} from "./types";
import { LOCAL_ALLOW_LIST, mergePackages } from "./loadPlan";

const RUNTIME_BASE = "/pyodide/runtime/";
const WHEEL_BASE = "/pyodide/wheels/";
const PREFERENCE_KEY = "ustat.localCompute";

export class LocalComputeUnavailable extends Error {
  readonly reason: LocalUnavailableReason;

  constructor(failure: LocalRunFailure) {
    super(failure.detail);
    this.name = "LocalComputeUnavailable";
    this.reason = failure.reason;
  }
}

export function isLocalComputeEnabled(): boolean {
  try {
    return localStorage.getItem(PREFERENCE_KEY) === "on";
  } catch {
    // Private browsing modes can throw on localStorage access. Absent a
    // readable preference the answer is "not enabled".
    return false;
  }
}

export function setLocalComputeEnabled(on: boolean): void {
  try {
    localStorage.setItem(PREFERENCE_KEY, on ? "on" : "off");
  } catch {
    /* nothing to do: the preference simply will not persist */
  }
}

let worker: Worker | null = null;
let nextId = 1;
let bootPromise: Promise<EngineIdentity> | null = null;
let loaded: string[] = [];
/** Set once local compute has been ruled out for this page, so it is not retried. */
let disabledFor: LocalRunFailure | null = null;

const pending = new Map<
  number,
  { resolve: (r: WorkerResponse) => void; reject: (e: unknown) => void }
>();

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
    const entry = pending.get(event.data.id);
    if (!entry) return;
    pending.delete(event.data.id);
    entry.resolve(event.data);
  };
  worker.onerror = (event) => {
    const failure: LocalRunFailure = {
      reason: "crashed",
      detail: event.message || "worker error",
    };
    pending.forEach((entry) => entry.reject(new LocalComputeUnavailable(failure)));
    pending.clear();
    teardown();
  };
  return worker;
}

function teardown(): void {
  worker?.terminate();
  worker = null;
  bootPromise = null;
  loaded = [];
}

function send(message: WorkerRequestBody): Promise<WorkerResponse> {
  const id = nextId++;
  const w = ensureWorker();
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    w.postMessage({ ...message, id } as WorkerRequest);
  });
}

/** What engine the server is running, so a local result can be shown to match it. */
async function serverIdentity(): Promise<EngineIdentity> {
  const r = await fetch("/api/engine/identity");
  if (!r.ok) throw new Error(`/api/engine/identity: HTTP ${r.status}`);
  return (await r.json()) as EngineIdentity;
}

async function boot(analysisId: string): Promise<EngineIdentity> {
  const packages = mergePackages(loaded, analysisId);
  const [response, server] = await Promise.all([
    send({ cmd: "init", packages, runtimeBase: RUNTIME_BASE, wheelBase: WHEEL_BASE }),
    serverIdentity(),
  ]);

  if (!response.ok) {
    throw new LocalComputeUnavailable({ reason: response.reason, detail: response.detail });
  }
  const browser = (response as { identity: EngineIdentity }).identity;

  if (!browser.fingerprint || browser.fingerprint !== server.fingerprint) {
    // The two runtimes are not running the same code. Computing here would
    // give an answer the server could not reproduce, and the reader would
    // have no way to tell which one they were looking at.
    throw new LocalComputeUnavailable({
      reason: "fingerprint-mismatch",
      detail:
        `browser engine ${browser.fingerprint ?? "unknown"} does not match ` +
        `server engine ${server.fingerprint ?? "unknown"}`,
    });
  }

  loaded = packages;
  return browser;
}

/**
 * Run `analysisId` in the browser, or throw LocalComputeUnavailable with the
 * reason. Callers are expected to fall back to the server on that error.
 */
export async function runLocal<T>(analysisId: string, params: unknown): Promise<T> {
  if (disabledFor) throw new LocalComputeUnavailable(disabledFor);

  if (!isLocalComputeEnabled()) {
    throw new LocalComputeUnavailable({
      reason: "disabled-by-user",
      detail: "local compute is off",
    });
  }
  if (!LOCAL_ALLOW_LIST.has(analysisId)) {
    throw new LocalComputeUnavailable({
      reason: "not-allow-listed",
      detail: `${analysisId} has no verified parity fixtures yet`,
    });
  }
  if (typeof Worker === "undefined") {
    throw new LocalComputeUnavailable({
      reason: "no-worker-support",
      detail: "this browser has no module workers",
    });
  }

  if (!bootPromise) bootPromise = boot(analysisId);
  try {
    await bootPromise;
  } catch (err) {
    // Booting failed for a reason that will not fix itself on retry -- a
    // missing wheel, a mismatched engine. Remember it rather than paying the
    // failure again on every subsequent analysis.
    const failure: LocalRunFailure =
      err instanceof LocalComputeUnavailable
        ? { reason: err.reason, detail: err.message }
        : { reason: "runtime-load-failed", detail: String(err) };
    disabledFor = failure;
    teardown();
    throw new LocalComputeUnavailable(failure);
  }

  const response = await send({ cmd: "run", analysisId, params });
  if (!response.ok) {
    throw new LocalComputeUnavailable({ reason: response.reason, detail: response.detail });
  }
  return (response as { result: T }).result;
}

/** Test seam: forget any recorded failure and drop the worker. */
export function resetLocalEngine(): void {
  disabledFor = null;
  teardown();
}
