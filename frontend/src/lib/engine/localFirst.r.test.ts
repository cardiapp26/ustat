/**
 * Routing in an R session: which engine is asked, and what happens when it says no.
 *
 * The property that matters most here is a negative one. An R session's
 * fallback ladder has exactly two rungs -- R, then the server -- and never
 * reaches for local Python, because booting Pyodide makes the arbiter tear webR
 * down and the next R-capable analysis re-downloads ~22 MB of runtime. That is
 * the kind of rule which is obvious while it is being written and invisible six
 * months later, so it is pinned rather than commented.
 *
 * Both engines are mocked. Nothing here boots a runtime.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LocalRunFailure } from "./types";

const runLocalR = vi.fn();
const ensureREngineBooted = vi.fn(async () => {});
let rBlockedBy: LocalRunFailure | null = null;

const runLocal = vi.fn();
const ensureEngineBooted = vi.fn(async () => {});

vi.mock("./r/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./r/client")>();
  return {
    ...actual,
    localRRunBlockedBecause: () => rBlockedBy,
    ensureREngineBooted: (...args: unknown[]) => ensureREngineBooted(...(args as [])),
    runLocalR: (...args: unknown[]) => runLocalR(...args),
    rEngineDetail: () => "R 4.6.0 · webR 0.6.0",
  };
});

vi.mock("./client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./client")>();
  return {
    ...actual,
    localRunBlockedBecause: () => null,
    ensureEngineBooted: (...args: unknown[]) => ensureEngineBooted(...(args as [])),
    runLocal: (...args: unknown[]) => runLocal(...args),
  };
});

const { LocalComputeUnavailable } = await import("./client");
const { localFirst } = await import("./localFirst");
const { useStore } = await import("../../store");

const server = vi.fn(async () => ({ data: { from: "server" }, status: 200 }));

beforeEach(() => {
  rBlockedBy = null;
  runLocalR.mockReset();
  runLocal.mockReset();
  ensureREngineBooted.mockClear();
  ensureEngineBooted.mockClear();
  server.mockClear();
  useStore.setState({ engine: "r", engineNotices: {} });
});

afterEach(() => {
  useStore.setState({ engine: "python", engineNotices: {} });
});

describe("an R session", () => {
  it("answers an allow-listed analysis with R, and says so", async () => {
    runLocalR.mockResolvedValue({ t: -6.28 });

    const res = await localFirst("stats.ttest", { column: "sbp" }, server);

    expect(res).toMatchObject({
      data: { t: -6.28 },
      runtime: "local",
      engine: "r",
      engineDetail: "R 4.6.0 · webR 0.6.0",
    });
    expect(res.fellBackBecause).toBeUndefined();
    expect(server).not.toHaveBeenCalled();
  });

  it("sends an analysis R cannot do to the SERVER, not to local Python", async () => {
    rBlockedBy = { reason: "no-r-implementation", detail: "stats.power has no R implementation" };

    const res = await localFirst("stats.power", { test: "t" }, server);

    expect(res).toMatchObject({
      runtime: "server",
      engine: "python",
      fellBackBecause: "no-r-implementation",
    });
    expect(server).toHaveBeenCalledTimes(1);
    // The rung that must not exist. Booting Pyodide here would evict webR and
    // make the next t-test re-download the R runtime.
    expect(runLocal).not.toHaveBeenCalled();
    expect(ensureEngineBooted).not.toHaveBeenCalled();
  });

  it("falls back to the server when webR itself will not load", async () => {
    rBlockedBy = null;
    runLocalR.mockRejectedValue(
      new LocalComputeUnavailable({ reason: "r-runtime-load-failed", detail: "R.wasm 404" }),
    );

    const res = await localFirst("stats.ttest", {}, server);

    expect(res).toMatchObject({ runtime: "server", engine: "python", fellBackBecause: "r-runtime-load-failed" });
    expect(runLocal).not.toHaveBeenCalled();
  });

  it("falls back -- but visibly -- when R has a bug rather than an objection", async () => {
    runLocalR.mockRejectedValue(
      new LocalComputeUnavailable({ reason: "r-engine-bug", detail: "object 'nope' not found" }),
    );

    const res = await localFirst("stats.ttest", {}, server);

    expect(res).toMatchObject({ runtime: "server", fellBackBecause: "r-engine-bug" });
  });

  it("does not ask the server to repeat a refusal R already made", async () => {
    runLocalR.mockRejectedValue(
      new LocalComputeUnavailable({
        reason: "engine-error",
        detail: "frame does not match the active Select Cases",
      }),
    );

    await expect(localFirst("stats.ttest", {}, server)).rejects.toMatchObject({
      reason: "engine-error",
    });
    expect(server).not.toHaveBeenCalled();
  });

  it("records where every answer came from, for the badge bar to read", async () => {
    runLocalR.mockResolvedValue({ t: 1 });
    await localFirst("stats.ttest", {}, server);
    rBlockedBy = { reason: "no-r-implementation", detail: "no R power" };
    await localFirst("stats.power", {}, server);

    const notices = useStore.getState().engineNotices;
    expect(notices["stats.ttest"]).toMatchObject({ engine: "r" });
    expect(notices["stats.ttest"].fellBackBecause).toBeUndefined();
    expect(notices["stats.power"]).toMatchObject({
      engine: "python",
      fellBackBecause: "no-r-implementation",
    });
  });
});

describe("a Python session", () => {
  beforeEach(() => {
    useStore.setState({ engine: "python", engineNotices: {} });
  });

  it("never reaches for R", async () => {
    runLocal.mockResolvedValue({ result: 42 });

    const res = await localFirst("stats.power", { test: "t" }, server);

    expect(res).toMatchObject({ runtime: "local", engine: "python" });
    expect(runLocalR).not.toHaveBeenCalled();
    expect(ensureREngineBooted).not.toHaveBeenCalled();
  });
});
