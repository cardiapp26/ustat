/**
 * The worker protocol, exercised without Pyodide.
 *
 * Booting the real runtime in a test would take tens of seconds and download a
 * wheel, so the worker is replaced by a stub that records what the client sent
 * it. That is the part worth pinning here: whether a `run` carries the frame
 * key it was given, and whether a worker that no longer holds the frame reaches
 * the caller as a named reason rather than as a generic crash. What the engine
 * then computes is the backend suite's job — there is only one implementation
 * of it, and it is Python.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { WorkerRequest, WorkerResponse } from "./types";
import { LocalComputeUnavailable, pushFrame, resetLocalEngine, runLocal, setLocalComputeEnabled } from "./client";

const FINGERPRINT = "0".repeat(64);

let sent: WorkerRequest[] = [];
/** Overridable per test: what the stub worker answers a `run` with. */
let runReply: (msg: WorkerRequest & { cmd: "run" }) => WorkerResponse = (msg) => ({
  id: msg.id,
  ok: true,
  result: { echoed: msg.analysisId },
});

class StubWorker {
  onmessage: ((e: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;

  postMessage(msg: WorkerRequest): void {
    sent.push(msg);
    const reply: WorkerResponse =
      msg.cmd === "init"
        ? { id: msg.id, ok: true, kind: "init", identity: { version: "0.1.0", fingerprint: FINGERPRINT, modules: 3 } }
        : msg.cmd === "frame"
          ? { id: msg.id, ok: true, result: { frameKey: msg.frameKey } }
          : runReply(msg);
    // Asynchronous, like the real boundary: the client must not depend on a
    // reply that has already arrived by the time postMessage returns.
    queueMicrotask(() => this.onmessage?.({ data: reply } as MessageEvent<WorkerResponse>));
  }

  terminate(): void {}
}

beforeEach(() => {
  sent = [];
  runReply = (msg) => ({ id: msg.id, ok: true, result: { echoed: msg.analysisId } });
  vi.stubGlobal("Worker", StubWorker);
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => ({ version: "0.1.0", fingerprint: FINGERPRINT, modules: 3 }),
    })),
  );
  setLocalComputeEnabled(true);
  resetLocalEngine();
});

afterEach(() => {
  setLocalComputeEnabled(false);
  resetLocalEngine();
  vi.unstubAllGlobals();
});

function lastOf(cmd: WorkerRequest["cmd"]): WorkerRequest | undefined {
  return [...sent].reverse().find((m) => m.cmd === cmd);
}

describe("the frame channel", () => {
  it("sends an envelope under its key, without running anything", async () => {
    // A frame push needs a booted worker; a run is the cheapest way to get one.
    await runLocal("stats.power", { test: "t" });
    sent = [];

    await pushFrame("sess-1:filterhash:age,arm", { schema: "ustat.frame/1" });

    const frameMsg = lastOf("frame");
    expect(frameMsg).toMatchObject({
      cmd: "frame",
      frameKey: "sess-1:filterhash:age,arm",
      envelope: { schema: "ustat.frame/1" },
    });
    expect(sent.some((m) => m.cmd === "run")).toBe(false);
  });

  it("carries the frame key on the run that uses it", async () => {
    await runLocal("stats.power", { test: "t" }, { frameKey: "sess-1" });
    expect(lastOf("run")).toMatchObject({ cmd: "run", frameKey: "sess-1" });
  });

  it("leaves the frame key off a run that did not ask for one", async () => {
    await runLocal("stats.power", { test: "t" });
    expect((lastOf("run") as { frameKey?: string }).frameKey).toBeUndefined();
  });

  it("reports a missing frame as its own reason, not as a crash", async () => {
    runReply = (msg) => ({
      id: msg.id,
      ok: false,
      reason: "frame-missing",
      detail: "no frame is held under sess-1",
    });

    await expect(runLocal("stats.power", {}, { frameKey: "sess-1" })).rejects.toMatchObject({
      name: "LocalComputeUnavailable",
      reason: "frame-missing",
    });
  });

  it("a missing frame is recoverable: the caller may push and retry", async () => {
    let held = false;
    runReply = (msg) =>
      held
        ? { id: msg.id, ok: true, result: { ok: true } }
        : { id: msg.id, ok: false, reason: "frame-missing", detail: "not held" };

    await expect(runLocal("stats.power", {}, { frameKey: "k" })).rejects.toBeInstanceOf(
      LocalComputeUnavailable,
    );
    // A frame-missing must NOT have disabled local compute for the page the way
    // a boot failure does — the fix is one message away.
    held = true;
    await pushFrame("k", { schema: "ustat.frame/1" });
    await expect(runLocal("stats.power", {}, { frameKey: "k" })).resolves.toEqual({ ok: true });
  });
});
