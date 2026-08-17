/**
 * The webR worker protocol, exercised without webR.
 *
 * Booting the real runtime in a test would download 17 MB of R.wasm and take
 * tens of seconds, so the worker is replaced by a stub that records what the
 * client sent it. What is worth pinning here is the part the client owns: that
 * an analysis outside the allow-list never gets as far as allocating anything,
 * that a manifest which does not match the server refuses rather than computes,
 * and that the two kinds of R failure -- a refusal and a bug -- keep their
 * separate names on the way out. What R then computes is the backend suite's
 * job; there is one implementation of it and it is R's.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RWorkerRequest, RWorkerResponse } from "./types";
import { R_MANIFEST_ANALYSES } from "./loadPlan";
import {
  ensureREngineBooted,
  localRRunBlockedBecause,
  pushFrameR,
  rEngineDetail,
  resetREngine,
  runLocalR,
} from "./client";

const FINGERPRINT = "a".repeat(64);

let sent: RWorkerRequest[] = [];
/** Overridable per test: what the stub worker answers a `run` with. */
let runReply: (msg: RWorkerRequest & { cmd: "run" }) => RWorkerResponse = (msg) => ({
  id: msg.id,
  ok: true,
  result: { echoed: msg.analysisId },
});
/** Overridable per test: the manifest fingerprint the worker claims to hold. */
let workerFingerprint = FINGERPRINT;
/** Overridable per test: what /api/engine/r/identity answers. */
let serverFingerprint: string | null = FINGERPRINT;

class StubWorker {
  onmessage: ((e: MessageEvent<RWorkerResponse>) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;

  postMessage(msg: RWorkerRequest): void {
    sent.push(msg);
    const reply: RWorkerResponse =
      msg.cmd === "init"
        ? {
            id: msg.id,
            ok: true,
            kind: "init",
            boot: {
              identity: {
                schema: "ustat.frame/1",
                r_version: "4.6.0",
                analyses: ["stats.ttest"],
                packages: ["moments", "nortest"],
              },
              sourceFingerprint: workerFingerprint,
              manifestAnalyses: R_MANIFEST_ANALYSES.map((a) => ({
                id: a.id,
                needs_frame: a.needsFrame,
                packages: [...a.rPackages],
                r_packages: [...a.rPackages],
              })),
              webrVersion: "0.5.10-dev",
              bundleBytes: 52171,
              bootMs: 1234,
            },
          }
        : msg.cmd === "frame"
          ? { id: msg.id, ok: true, result: { frameKey: msg.frameKey } }
          : runReply(msg);
    // Asynchronous, like the real boundary: the client must not depend on a
    // reply that has already arrived by the time postMessage returns.
    queueMicrotask(() => this.onmessage?.({ data: reply } as MessageEvent<RWorkerResponse>));
  }

  terminate(): void {}
}

beforeEach(() => {
  sent = [];
  runReply = (msg) => ({ id: msg.id, ok: true, result: { echoed: msg.analysisId } });
  workerFingerprint = FINGERPRINT;
  serverFingerprint = FINGERPRINT;
  vi.stubGlobal("Worker", StubWorker);
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => ({
        version: "0.1.0",
        fingerprint: serverFingerprint,
        modules: 7,
        analyses: ["stats.ttest"],
        webr_version: "0.6.0",
        r_version: "4.6",
      }),
    })),
  );
  resetREngine();
});

afterEach(() => {
  resetREngine();
  vi.unstubAllGlobals();
});

function lastOf(cmd: RWorkerRequest["cmd"]): RWorkerRequest | undefined {
  return [...sent].reverse().find((m) => m.cmd === cmd);
}

describe("the allow-list gate", () => {
  it("refuses an analysis with no verified R implementation, by name", () => {
    expect(localRRunBlockedBecause("stats.power")).toMatchObject({
      reason: "no-r-implementation",
    });
  });

  it("lets an allow-listed analysis through", () => {
    expect(localRRunBlockedBecause("stats.ttest")).toBeNull();
  });

  it("does not allocate a worker for an analysis it was never going to run", async () => {
    await expect(runLocalR("stats.power", {})).rejects.toMatchObject({
      name: "LocalComputeUnavailable",
      reason: "no-r-implementation",
    });
    expect(sent).toEqual([]);
  });
});

describe("boot", () => {
  it("asks for jsonlite and names the analysis, leaving its packages to the manifest", async () => {
    await ensureREngineBooted("stats.ttest");
    expect(lastOf("init")).toMatchObject({
      cmd: "init",
      analysisId: "stats.ttest",
      packages: ["jsonlite"],
      runtimeBase: "/webr/runtime/",
      bundleBase: "/webr/bundle/",
      repoUrl: "/webr/repo",
    });
  });

  it("names the running engine once it is up, using the VENDORED webR release", async () => {
    expect(rEngineDetail()).toBeUndefined();
    await ensureREngineBooted("stats.ttest");
    // The stub worker reports webR's own self-description, which upstream's
    // v0.6.0 tarball leaves stamped "0.5.10-dev". The vendoring manifest's tag
    // is the one a reader could go and look up, so it wins.
    expect(rEngineDetail()).toBe("R 4.6.0 · webR 0.6.0");
  });

  it("refuses when the bundle's manifest and the server describe different sources", async () => {
    serverFingerprint = "b".repeat(64);
    await expect(ensureREngineBooted("stats.ttest")).rejects.toMatchObject({
      reason: "fingerprint-mismatch",
    });
  });

  it("does not pay a hopeless boot twice", async () => {
    serverFingerprint = null;
    await expect(ensureREngineBooted("stats.ttest")).rejects.toMatchObject({
      reason: "fingerprint-mismatch",
    });
    const initsAfterFirst = sent.filter((m) => m.cmd === "init").length;
    await expect(ensureREngineBooted("stats.ttest")).rejects.toMatchObject({
      reason: "fingerprint-mismatch",
    });
    expect(sent.filter((m) => m.cmd === "init").length).toBe(initsAfterFirst);
  });
});

describe("frames and runs", () => {
  it("sends an envelope under its key, without running anything", async () => {
    await ensureREngineBooted("stats.ttest");
    sent = [];

    await pushFrameR("sess-1:filterhash:sbp,arm", { schema: "ustat.frame/1" });

    expect(lastOf("frame")).toMatchObject({
      cmd: "frame",
      frameKey: "sess-1:filterhash:sbp,arm",
      envelope: { schema: "ustat.frame/1" },
    });
    expect(sent.some((m) => m.cmd === "run")).toBe(false);
  });

  it("refuses a frame push before the engine is booted rather than losing it silently", async () => {
    await expect(pushFrameR("k", { schema: "ustat.frame/1" })).rejects.toMatchObject({
      reason: "r-runtime-load-failed",
    });
  });

  it("carries the frame key on the run that uses it", async () => {
    await runLocalR("stats.ttest", { column: "sbp" }, { frameKey: "sess-1" });
    expect(lastOf("run")).toMatchObject({ cmd: "run", frameKey: "sess-1" });
  });

  it("reports a missing frame as its own reason, not as a crash", async () => {
    runReply = (msg) => ({
      id: msg.id,
      ok: false,
      reason: "frame-missing",
      detail: "no frame is held under sess-1",
    });
    await expect(
      runLocalR("stats.ttest", {}, { frameKey: "sess-1" }),
    ).rejects.toMatchObject({ reason: "frame-missing" });
  });

  it("keeps a refusal (409) apart from an R-level bug (500)", async () => {
    runReply = (msg) => ({
      id: msg.id,
      ok: false,
      reason: "engine-error",
      detail: "frame does not match the active Select Cases",
    });
    await expect(runLocalR("stats.ttest", {})).rejects.toMatchObject({
      reason: "engine-error",
      message: "frame does not match the active Select Cases",
    });

    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    runReply = (msg) => ({
      id: msg.id,
      ok: false,
      reason: "r-engine-bug",
      detail: "object 'nope' not found",
    });
    await expect(runLocalR("stats.ttest", {})).rejects.toMatchObject({
      reason: "r-engine-bug",
    });
    // An R bug that only appears in a browser has to be visible; the fallback
    // would otherwise hide it behind a correct-looking server answer.
    expect(logged).toHaveBeenCalledWith(expect.stringContaining("object 'nope' not found"));
    logged.mockRestore();
  });
});
