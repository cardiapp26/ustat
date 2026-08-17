/**
 * The R statistics engine, running in a worker.
 *
 * The webR counterpart of `../worker.ts`, and written to diff against it: same
 * message protocol, same boot-then-frame-then-run shape, same two-entry LRU of
 * resident datasets, same rule that an analysis rejecting its own input is an
 * answer rather than a reason to ask the server.
 *
 * Everything is loaded from this origin -- the runtime, the package mirror and
 * the engine bundle. Not from repo.r-wasm.org, which is webR's default and
 * would let a third party observe which statistical packages a given clinician
 * just loaded; that is a weaker version of the problem this whole feature
 * exists to solve, and the app's CSP forbids it besides.
 *
 * WHY THIS WORKER EXISTS AT ALL, GIVEN webR ALREADY HAS ONE
 * ---------------------------------------------------------
 * webR starts its own worker for R itself. This one sits in front of it and
 * holds the things the main thread must not: the fetched bundle, the frames,
 * and the verification state. Doing that work on the main thread would put a
 * 52 KB parse, a sha256 and every result's JSON.parse on the frame the user is
 * looking at, and would make `frameKey` residency the main thread's problem
 * rather than the engine's.
 */
/// <reference lib="webworker" />
import type { LocalUnavailableReason } from "../types";
import type {
  RBootInfo,
  RBundleManifest,
  REngineIdentity,
  RWorkerRequest,
  RWorkerResponse,
} from "./types";

/**
 * A failure that already knows what to call itself.
 *
 * Without it every throw inside `bootEngine` reaches the catch as an anonymous
 * Error and gets reported as "crashed", which would file "the bundle we were
 * served is not the bundle that was built" -- a refusal with a fix -- under the
 * same name as a wasm trap.
 */
class RefusalError extends Error {
  readonly reason: LocalUnavailableReason;

  constructor(reason: LocalUnavailableReason, message: string) {
    super(message);
    this.name = "RefusalError";
    this.reason = reason;
  }
}

/** The slice of webR's API this worker uses. Typed by hand: the vendored
 *  runtime ships no .d.ts, and depending on the `webr` npm package would mean
 *  two copies of the runtime -- one bundled, one vendored -- that could drift. */
interface RObjectProxy {
  bind(name: string, value: unknown): Promise<void>;
}

interface WebRApi {
  /** webR's own version, e.g. "0.6.0". */
  version: string;
  init(): Promise<void>;
  installPackages(
    names: string[],
    options?: { quiet?: boolean; mount?: boolean; repos?: string | string[] },
  ): Promise<void>;
  evalRString(code: string): Promise<string>;
  evalRVoid(code: string): Promise<void>;
  objs: { globalEnv: RObjectProxy };
  FS: { writeFile(path: string, data: Uint8Array): Promise<void> };
}

interface WebRModule {
  WebR: new (options: {
    baseUrl: string;
    repoUrl: string;
    channelType: number;
    REnv?: Record<string, string>;
  }) => WebRApi;
  ChannelType: { PostMessage: number; SharedArrayBuffer: number; Automatic: number };
}

/** Where the verified bundle is written inside webR's virtual filesystem. */
const BUNDLE_PATH = "/tmp/ustat_engine_r.R";

let webR: WebRApi | null = null;
const installedPackages = new Set<string>();
let boot: RBootInfo | null = null;

function post(message: RWorkerResponse): void {
  (self as unknown as Worker).postMessage(message);
}

/** A dev server answers a missing file with the SPA index at HTTP 200, so a 200
 *  is not by itself evidence the file exists. Same guard as the Pyodide worker. */
function assertNotSpaFallback(url: string, text: string): void {
  if (/^\s*<(!doctype|html)/i.test(text)) {
    throw new Error(`${url}: not found (served the SPA fallback)`);
  }
}

async function fetchText(url: string): Promise<string> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const text = await r.text();
  assertNotSpaFallback(url, text);
  return text;
}

async function fetchJson<T>(url: string): Promise<T> {
  return JSON.parse(await fetchText(url)) as T;
}

function toHex(buffer: ArrayBuffer): string {
  return [...new Uint8Array(buffer)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * The host-side frame cache, in R. Capped at two, evicted least-recently-USED.
 *
 * The exact policy `../worker.ts` uses on the Python side and for the same
 * reasons: two, so toggling between a filtered and an unfiltered view does not
 * re-transfer on every switch; not many, because these are whole patient
 * datasets in a wasm heap with a hard ceiling. Least-recently-used rather than
 * least-recently-pushed, because a frame pushed first and read by every
 * analysis is the one worth keeping.
 *
 * An environment holds the list so the two helpers can rebind it without `<<-`
 * reaching into the global environment the engine itself lives in.
 */
const FRAME_CACHE_SETUP = `
.ustat_host <- new.env(parent = emptyenv())
.ustat_host$frames <- list()

.ustat_put_frame <- function(key, envelope_json, cap = 2L) {
  # Drop first, then re-insert, so the list's order is insertion order and the
  # eviction below always takes the oldest rather than an arbitrary element.
  .ustat_host$frames[[key]] <- NULL
  .ustat_host$frames[[key]] <- ustat_frame_from_json(envelope_json)
  while (length(.ustat_host$frames) > cap) {
    .ustat_host$frames[[1L]] <- NULL
  }
  invisible(TRUE)
}

.ustat_take_frame <- function(key) {
  if (!(key %in% names(.ustat_host$frames))) {
    return(NULL)
  }
  held <- .ustat_host$frames[[key]]
  # Re-insert so list order is USE order, which is what the eviction reads.
  .ustat_host$frames[[key]] <- NULL
  .ustat_host$frames[[key]] <- held
  held
}
`;

/**
 * Install what is missing, then prove it is actually there.
 *
 * `webr::install()` -- which `installPackages` is a thin wrapper over -- reports
 * a failed download by printing and carrying on, so its resolving is not
 * evidence of anything. The `requireNamespace` sweep afterwards is what turns a
 * missing package into a named refusal here instead of an obscure
 * "could not find function" from inside an analysis later.
 */
async function ensurePackages(names: string[]): Promise<void> {
  if (!webR) throw new Error("engine was not initialised");
  const missing = names.filter((p) => !installedPackages.has(p));
  if (!missing.length) return;

  await webR.installPackages(missing, { quiet: true, mount: true });

  const escaped = missing.map((p) => JSON.stringify(p)).join(", ");
  const absent = await webR.evalRString(`
paste(
  Filter(
    function(p) !requireNamespace(p, quietly = TRUE),
    c(${escaped})
  ),
  collapse = ","
)
`);
  if (absent) {
    throw new RefusalError(
      "r-runtime-load-failed",
      `R package(s) ${absent} did not install from the local mirror ` +
        `(webr::install reports failures by printing, not by throwing)`,
    );
  }
  missing.forEach((p) => installedPackages.add(p));
}

/** What the FETCHED manifest says `analysisId` needs. The manifest is the
 *  authority; the build-time copy in ./loadPlan.ts only decides whether to get
 *  this far. An id the manifest does not declare needs nothing extra -- the run
 *  itself will refuse it with a 404 from `ustat_get`. */
function packagesForAnalysis(analysisId: string): string[] {
  const declared = boot?.manifestAnalyses.find((a) => a.id === analysisId);
  return [...(declared?.r_packages ?? [])];
}

async function bootEngine(
  analysisId: string,
  packages: string[],
  runtimeBase: string,
  bundleBase: string,
  repoUrl: string,
): Promise<RBootInfo> {
  const startedAt = Date.now();

  if (!webR) {
    const mod = (await import(/* @vite-ignore */ `${runtimeBase}webr.mjs`)) as unknown as WebRModule;
    webR = new mod.WebR({
      baseUrl: runtimeBase,
      // Our own mirror. webR's default is https://repo.r-wasm.org, which is
      // both a CSP violation here and the exact leak vendoring prevents.
      repoUrl,
      // PostMessage, never SharedArrayBuffer: SAB needs COEP, and COEP breaks
      // the Google Drive sync the welcome screen offers. The cost is that R
      // cannot block on stdin, which nothing here asks it to.
      channelType: mod.ChannelType.PostMessage,
      // webR defaults ALL_PROXY to socks5h://localhost:8580, which is the
      // address of a proxy that does not exist here. Left set, anything in R
      // that consults it (download.file, curl) tries to reach it.
      REnv: { ALL_PROXY: "" },
    });
    await webR.init();
  }

  if (!boot) {
    const manifest = await fetchJson<RBundleManifest>(`${bundleBase}manifest.json`);
    const bundleUrl = `${bundleBase}${manifest.bundle}`;
    const source = await fetchText(bundleUrl);

    // The bundle is pure ASCII by construction (scripts/build_r_bundle.py
    // refuses to write a byte above 0x7f), so encoding the decoded text
    // reproduces the bytes the server hashed.
    const bytes = new TextEncoder().encode(source);
    const sha256 = toHex(await crypto.subtle.digest("SHA-256", bytes));
    if (sha256 !== manifest.bundle_sha256) {
      // The bundle that arrived is not the bundle that was built -- a stale
      // service-worker entry, a half-finished deploy. Refusing here is the
      // point: computing anyway would return numbers from code nobody shipped.
      throw new RefusalError(
        "fingerprint-mismatch",
        `R bundle sha256 ${sha256} does not match the manifest (${manifest.bundle_sha256})`,
      );
    }

    // Named by the manifest, not by this file: `r_packages` is what the R spec
    // declared, and a spec that gains a dependency must not need a matching
    // edit here to keep working.
    const declared = manifest.analyses.find((a) => a.id === analysisId)?.r_packages ?? [];
    await ensurePackages([...new Set([...packages, ...declared])]);

    // Written to the virtual filesystem and sourced rather than eval'd as one
    // string: R then has real source references, so a failure inside the
    // bundle names a line instead of pointing at the whole file.
    await webR.FS.writeFile(BUNDLE_PATH, bytes);
    await webR.evalRVoid(`source(${JSON.stringify(BUNDLE_PATH)}, local = FALSE, echo = FALSE)`);
    await webR.evalRVoid("ustat_init()");
    await webR.evalRVoid(FRAME_CACHE_SETUP);

    const identity = JSON.parse(await webR.evalRString("ustat_identity_json()")) as REngineIdentity;

    boot = {
      identity,
      sourceFingerprint: manifest.source_fingerprint,
      manifestAnalyses: manifest.analyses,
      webrVersion: webR.version,
      bundleBytes: bytes.byteLength,
      bootMs: Date.now() - startedAt,
    };
  }

  return boot;
}

self.onmessage = async (event: MessageEvent<RWorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.cmd === "init") {
      const info = await bootEngine(
        msg.analysisId,
        msg.packages,
        msg.runtimeBase,
        msg.bundleBase,
        msg.repoUrl,
      );
      post({ id: msg.id, ok: true, kind: "init", boot: info });
      return;
    }

    if (msg.cmd === "frame") {
      if (!webR) throw new Error("engine was not initialised");
      // JSON text rather than an R list built field by field: the envelope's
      // parser is `ustat_frame_from_json`, which is the same code path the
      // server-side tests exercise. Rebuilding the list here would be a second
      // implementation of the envelope, in JavaScript, which is exactly what
      // the two-thin-engines rule forbids.
      await webR.objs.globalEnv.bind(".ustat_frame_key", msg.frameKey);
      await webR.objs.globalEnv.bind(".ustat_frame_json", JSON.stringify(msg.envelope));
      await webR.evalRVoid(".ustat_put_frame(.ustat_frame_key, .ustat_frame_json)");
      post({ id: msg.id, ok: true, result: { frameKey: msg.frameKey } });
      return;
    }

    if (msg.cmd === "run") {
      if (!webR) throw new Error("engine was not initialised");
      // A second analysis in the same session may declare a package the first
      // did not. webR keeps what it has installed, so this is a Set check in
      // the common case.
      await ensurePackages(packagesForAnalysis(msg.analysisId));
      await webR.objs.globalEnv.bind(".ustat_frame_key", msg.frameKey ?? "");
      if (msg.frameKey) {
        const held = await webR.evalRString(
          'if (is.null(.ustat_take_frame(.ustat_frame_key))) "no" else "yes"',
        );
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
      }
      await webR.objs.globalEnv.bind(".ustat_analysis", msg.analysisId);
      await webR.objs.globalEnv.bind(".ustat_params_json", JSON.stringify(msg.params ?? {}));
      // Argument order is (analysis_id, params, frame) -- NOT the Python
      // engine's (analysis_id, frame, params). Getting it wrong here would
      // hand a data.frame in where params belong and fail deep inside R.
      const raw = await webR.evalRString(`
ustat_run_json(
  .ustat_analysis,
  .ustat_params_json,
  if (nzchar(.ustat_frame_key)) .ustat_take_frame(.ustat_frame_key) else NULL
)
`);
      const parsed = JSON.parse(raw) as
        | { ok: true; result: unknown }
        | { ok: false; error: { message: string; status_hint: number | null } };

      if (parsed.ok) {
        post({ id: msg.id, ok: true, result: parsed.result });
        return;
      }

      if (parsed.error.status_hint === 500) {
        // 500 from the R engine is not a refusal, it is a bug in the R code:
        // an R-level condition that ustat_run_json could not attribute to the
        // request. The server can still answer the question, so this falls
        // back -- but under its own name, because "R has a bug" and "the user
        // asked for something impossible" must not read the same in the log.
        post({
          id: msg.id,
          ok: false,
          reason: "r-engine-bug",
          detail: parsed.error.message,
        });
        return;
      }

      // 400/404/409/422: an analysis that rejects its input is a real answer,
      // not a reason to ask the server the same question and get the same
      // rejection. Identical to the Python worker's treatment.
      post({ id: msg.id, ok: false, reason: "engine-error", detail: parsed.error.message });
    }
  } catch (err) {
    post({
      id: msg.id,
      ok: false,
      reason:
        err instanceof RefusalError
          ? err.reason
          : webR
            ? "crashed"
            : "r-runtime-load-failed",
      detail: err instanceof Error ? err.message : String(err),
    });
  }
};
