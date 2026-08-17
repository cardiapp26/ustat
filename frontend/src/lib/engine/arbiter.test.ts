/**
 * The arbiter's own rules, isolated from either real client.
 *
 * Tests are ordered deliberately. The very first one exercises acquiring and
 * switching away from an engine kind that has never called `registerRuntime`
 * -- so it has to run before any other test in this file registers "r" or
 * "python", which is why it comes first rather than being reordered for
 * readability. `resetArbiter()` only forgets who is resident, not who is
 * registered (see the comment on it in arbiter.ts), so registrations made by
 * later tests persist for the rest of the file; that is fine for every test
 * here except this one.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import type { WorkerRequest, WorkerResponse } from "./types";
import { acquire, current, registerRuntime, release, resetArbiter } from "./arbiter";

afterEach(() => {
  resetArbiter();
});

describe("acquire, before anything is registered", () => {
  it("does not throw acquiring, or switching away from, an engine that was never registered", () => {
    // "r" has not been registered by anything yet -- this is the point.
    expect(() => acquire("r")).not.toThrow();
    expect(current()).toBe("r");

    // Switching away from "r" has to tear it down, but there is nothing
    // registered to call -- a client that has not finished loading yet is a
    // normal state, not a bug, and must not throw here either.
    expect(() => acquire("python")).not.toThrow();
    expect(current()).toBe("python");
  });
});

describe("acquire", () => {
  it("acquiring the same engine twice does not tear anything down", () => {
    const teardown = vi.fn();
    registerRuntime("python", { teardown });

    acquire("python");
    acquire("python");

    expect(teardown).not.toHaveBeenCalled();
    expect(current()).toBe("python");
  });

  it("acquiring a different engine tears down the first, and only the first", () => {
    const pythonTeardown = vi.fn();
    const rTeardown = vi.fn();
    registerRuntime("python", { teardown: pythonTeardown });
    registerRuntime("r", { teardown: rTeardown });

    acquire("python");
    acquire("r");

    expect(pythonTeardown).toHaveBeenCalledTimes(1);
    expect(rTeardown).not.toHaveBeenCalled();
    expect(current()).toBe("r");
  });

  it("a teardown hook that throws does not block the incoming engine from acquiring", () => {
    // Log-and-continue, on purpose: a broken outgoing engine must not
    // prevent the incoming one from getting the tab. If it did, "R's
    // teardown threw" would turn into "the user can no longer run Python
    // either" -- strictly worse than the leaked memory this module exists
    // to prevent.
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    registerRuntime("python", {
      teardown: () => {
        throw new Error("boom");
      },
    });
    registerRuntime("r", { teardown: vi.fn() });

    acquire("python");
    expect(() => acquire("r")).not.toThrow();

    expect(current()).toBe("r");
    expect(errorSpy).toHaveBeenCalled();
    errorSpy.mockRestore();
  });
});

describe("current", () => {
  it("is null before anything has acquired", () => {
    expect(current()).toBeNull();
  });

  it("reports whichever engine most recently acquired", () => {
    registerRuntime("python", { teardown: vi.fn() });
    registerRuntime("r", { teardown: vi.fn() });

    acquire("python");
    expect(current()).toBe("python");

    acquire("r");
    expect(current()).toBe("r");
  });
});

describe("release", () => {
  it("clears current() when releasing the resident engine", () => {
    registerRuntime("python", { teardown: vi.fn() });
    acquire("python");

    release("python");

    expect(current()).toBeNull();
  });

  it("is a no-op for an engine that is not current", () => {
    registerRuntime("python", { teardown: vi.fn() });
    registerRuntime("r", { teardown: vi.fn() });
    acquire("python");

    release("r");

    expect(current()).toBe("python");
  });
});

describe("the Python client participates for real", () => {
  const FINGERPRINT = "0".repeat(64);

  /** Every stub instance constructed, in order -- there is only ever one here. */
  const instances: StubWorker[] = [];

  class StubWorker {
    onmessage: ((e: MessageEvent<WorkerResponse>) => void) | null = null;
    onerror: ((e: unknown) => void) | null = null;
    terminated = false;

    constructor() {
      instances.push(this);
    }

    postMessage(msg: WorkerRequest): void {
      const reply: WorkerResponse =
        msg.cmd === "init"
          ? {
              id: msg.id,
              ok: true,
              kind: "init",
              identity: { version: "0.1.0", fingerprint: FINGERPRINT, modules: 3 },
            }
          : { id: msg.id, ok: true, result: {} };
      // Asynchronous, like the real boundary: the client must not depend on
      // a reply that has already arrived by the time postMessage returns.
      queueMicrotask(() => this.onmessage?.({ data: reply } as MessageEvent<WorkerResponse>));
    }

    terminate(): void {
      this.terminated = true;
    }
  }

  it("acquires the arbiter when its worker boots, and is torn down when R takes the tab", async () => {
    vi.stubGlobal("Worker", StubWorker);
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ version: "0.1.0", fingerprint: FINGERPRINT, modules: 3 }),
      })),
    );

    try {
      // Importing client.ts here, rather than at the top of the file, is
      // what makes the earlier "never registered" test valid: client.ts
      // calls registerRuntime("python", ...) at module scope, so importing
      // it eagerly would have registered "python" before this file's first
      // test ever ran.
      const { runLocal, setLocalComputeEnabled, resetLocalEngine } = await import("./client");
      setLocalComputeEnabled(true);
      resetLocalEngine();

      await runLocal("stats.power", { test: "t" });
      const bootedWorker = instances[instances.length - 1];

      expect(current()).toBe("python");
      expect(bootedWorker?.terminated).toBe(false);

      acquire("r");

      expect(bootedWorker?.terminated).toBe(true);
      expect(current()).toBe("r");

      setLocalComputeEnabled(false);
      resetLocalEngine();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
