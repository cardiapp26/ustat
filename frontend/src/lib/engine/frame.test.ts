/**
 * The frame key, and what it costs to be wrong about it.
 *
 * The key is the only thing standing between "this worker is holding the data
 * you think it is" and a result computed over a dataset the user has since
 * edited or filtered. So the tests here are mostly about it changing when it
 * must: a key that is too stable is not a cache, it is a stale answer.
 *
 * The opposite failure is cheap by comparison — a key that changes when it did
 * not need to costs one extra transfer — so the skip-the-fetch test is here to
 * show the cache works at all, not to defend it.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

// `vi.mock` factories are hoisted above the file's own consts, so the spies
// they close over have to be hoisted with them.
const { get, pushFrame, resetHooks } = vi.hoisted(() => ({
  get: vi.fn(),
  pushFrame: vi.fn(),
  resetHooks: [] as Array<() => void>,
}));

vi.mock("../../api", () => ({
  default: { get: (...args: unknown[]) => get(...args) },
}));

vi.mock("./client", () => ({
  pushFrame: (...args: unknown[]) => pushFrame(...args),
  onLocalEngineReset: (hook: () => void) => resetHooks.push(hook),
}));

const STORE = { dataVersion: 0, caseFilter: null as { conditions: unknown[] } | null };
vi.mock("../../store", () => ({
  useStore: { getState: () => STORE },
}));

const ENVELOPE = {
  schema: "ustat.frame/1",
  filter: { conditions: [], fingerprint: "fp-unfiltered" },
};

import { ensureFrame, filterFingerprintFor, frameKeyFor, resetPushedFrames } from "./frame";

beforeEach(() => {
  get.mockReset();
  get.mockResolvedValue({ data: ENVELOPE });
  pushFrame.mockReset();
  pushFrame.mockResolvedValue(undefined);
  STORE.dataVersion = 0;
  STORE.caseFilter = null;
  resetPushedFrames();
});

const FILTER = [{ column: "age", operator: "gt", value: "50", join: "AND" }];

describe("frameKeyFor", () => {
  it("is stable for the same inputs", () => {
    expect(frameKeyFor("s1", 3, FILTER, ["sbp", "arm"])).toBe(
      frameKeyFor("s1", 3, FILTER, ["sbp", "arm"]),
    );
  });

  it("does not depend on the order the columns were named in", () => {
    // Same transfer, same frame: the engine reads columns by name. Treating
    // [sbp, arm] and [arm, sbp] as two frames would re-send identical bytes.
    expect(frameKeyFor("s1", 3, FILTER, ["arm", "sbp"])).toBe(
      frameKeyFor("s1", 3, FILTER, ["sbp", "arm"]),
    );
  });

  it("does not depend on the key order of a condition object", () => {
    const reordered = [{ join: "AND", value: "50", operator: "gt", column: "age" }];
    expect(frameKeyFor("s1", 3, reordered, ["sbp"])).toBe(
      frameKeyFor("s1", 3, FILTER, ["sbp"]),
    );
  });

  it.each([
    ["session id", frameKeyFor("s2", 3, FILTER, ["sbp", "arm"])],
    ["dataVersion", frameKeyFor("s1", 4, FILTER, ["sbp", "arm"])],
    ["filter conditions", frameKeyFor("s1", 3, [], ["sbp", "arm"])],
    ["a condition's value", frameKeyFor("s1", 3, [{ ...FILTER[0], value: "51" }], ["sbp", "arm"])],
    ["the column set", frameKeyFor("s1", 3, FILTER, ["sbp", "arm", "sex"])],
  ])("changes when the %s changes", (_what, other) => {
    expect(other).not.toBe(frameKeyFor("s1", 3, FILTER, ["sbp", "arm"]));
  });
});

describe("ensureFrame", () => {
  it("fetches only the named columns and hands the envelope to the worker", async () => {
    const key = await ensureFrame("s1", ["sbp", "arm"]);

    expect(get).toHaveBeenCalledWith("/api/sessions/s1/frame", {
      params: { columns: "sbp,arm" },
    });
    expect(pushFrame).toHaveBeenCalledWith(key, ENVELOPE);
    expect(key).toBe(frameKeyFor("s1", 0, [], ["sbp", "arm"]));
  });

  it("keeps the envelope's filter fingerprint so the run can state it", async () => {
    const key = await ensureFrame("s1", ["sbp", "arm"]);
    expect(filterFingerprintFor(key)).toBe("fp-unfiltered");
  });

  it("skips the fetch when that key was already pushed", async () => {
    await ensureFrame("s1", ["sbp", "arm"]);
    await ensureFrame("s1", ["arm", "sbp"]);

    expect(get).toHaveBeenCalledTimes(1);
    expect(pushFrame).toHaveBeenCalledTimes(1);
  });

  it("fetches again once the data underneath has changed", async () => {
    await ensureFrame("s1", ["sbp", "arm"]);
    STORE.dataVersion = 1;
    await ensureFrame("s1", ["sbp", "arm"]);

    expect(get).toHaveBeenCalledTimes(2);
  });

  it("fetches again once Select Cases has changed", async () => {
    await ensureFrame("s1", ["sbp", "arm"]);
    STORE.caseFilter = { conditions: FILTER };
    await ensureFrame("s1", ["sbp", "arm"]);

    expect(get).toHaveBeenCalledTimes(2);
  });

  it("does not remember a key whose push failed", async () => {
    pushFrame.mockRejectedValueOnce(new Error("worker is gone"));
    await expect(ensureFrame("s1", ["sbp"])).rejects.toThrow("worker is gone");

    pushFrame.mockResolvedValue(undefined);
    await ensureFrame("s1", ["sbp"]);
    // Two fetches: the first attempt must not have left the key looking resident.
    expect(get).toHaveBeenCalledTimes(2);
  });

  it("forgets every key when the engine is reset", async () => {
    await ensureFrame("s1", ["sbp"]);
    // A terminated worker holds nothing; a key remembered across that would
    // make the next run come back frame-missing for no visible reason.
    resetHooks.forEach((hook) => hook());
    await ensureFrame("s1", ["sbp"]);

    expect(get).toHaveBeenCalledTimes(2);
  });
});
