import { describe, it, expect } from "vitest";
import {
  describeStale,
  filterKey,
  makeStamp,
  stableStringify,
  staleReasons,
  type ResultStamp,
} from "./resultStamp";
import type { CaseFilter } from "../store";

const base = { dataVersion: 3, caseFilter: null, engine: "python" as const };

function stamp(over: Partial<Parameters<typeof makeStamp>[0]> = {}): ResultStamp {
  return makeStamp({ ...base, params: { a: 1 }, ...over });
}

describe("stableStringify", () => {
  it("keys alike regardless of insertion order", () => {
    expect(stableStringify({ a: 1, b: 2 })).toBe(stableStringify({ b: 2, a: 1 }));
  });

  it("distinguishes values, not just shapes", () => {
    expect(stableStringify({ a: 1 })).not.toBe(stableStringify({ a: 2 }));
  });

  it("keeps array order, which is meaningful for a predictor list", () => {
    expect(stableStringify(["x", "y"])).not.toBe(stableStringify(["y", "x"]));
  });

  it("treats an absent key and an explicit undefined as the same", () => {
    expect(stableStringify({ a: 1, b: undefined })).toBe(stableStringify({ a: 1 }));
  });

  it("survives nesting", () => {
    expect(stableStringify({ o: { b: 1, a: [2, { d: 4, c: 3 }] } }))
      .toBe(stableStringify({ o: { a: [2, { c: 3, d: 4 }], b: 1 } }));
  });
});

describe("filterKey", () => {
  const filter = (over: Partial<CaseFilter> = {}): CaseFilter => ({
    conditions: [{ column: "age", op: ">", value: 60 }] as unknown as CaseFilter["conditions"],
    selected: 40,
    total: 100,
    ...over,
  });

  it("collapses no filter and an empty filter to the same key", () => {
    expect(filterKey(null)).toBe("none");
    expect(filterKey({ conditions: [], selected: 0, total: 0 })).toBe("none");
  });

  it("changes when the conditions change", () => {
    expect(filterKey(filter())).not.toBe(
      filterKey(filter({ conditions: [{ column: "age", op: ">", value: 70 }] as unknown as CaseFilter["conditions"] })),
    );
  });

  it("ignores excludedRows, a view concern that moves with the preview window", () => {
    expect(filterKey(filter({ excludedRows: [1, 2, 3] }))).toBe(filterKey(filter()));
  });
});

describe("staleReasons", () => {
  it("returns nothing when nothing moved", () => {
    const s = stamp();
    expect(staleReasons(s, stamp())).toEqual([]);
  });

  it("returns nothing when there is no stamp to compare against", () => {
    expect(staleReasons(null, stamp())).toEqual([]);
  });

  it("flags a data edit", () => {
    const before = stamp();
    expect(staleReasons(before, stamp({ dataVersion: 4 }))).toEqual(["data"]);
  });

  it("flags a filter change", () => {
    const before = stamp();
    const after = stamp({
      caseFilter: { conditions: [{ column: "age", op: ">", value: 60 }] as unknown as CaseFilter["conditions"], selected: 4, total: 9 },
    });
    expect(staleReasons(before, after)).toEqual(["filter"]);
  });

  it("flags a settings change", () => {
    const before = stamp();
    expect(staleReasons(before, stamp({ params: { a: 2 } }))).toEqual(["params"]);
  });

  it("flags an engine switch", () => {
    const before = stamp();
    expect(staleReasons(before, stamp({ engine: "r" }))).toEqual(["engine"]);
  });

  it("reports every reason that applies, in a fixed order", () => {
    const before = stamp();
    const after = stamp({ dataVersion: 9, params: { a: 2 }, engine: "r" });
    expect(staleReasons(before, after)).toEqual(["data", "params", "engine"]);
  });

  it("ignores data and filter for an analysis that reads no dataset", () => {
    const before = stamp();
    const after = stamp({
      dataVersion: 99,
      caseFilter: { conditions: [{ column: "age", op: ">", value: 1 }] as unknown as CaseFilter["conditions"], selected: 1, total: 2 },
    });
    expect(staleReasons(before, after, { dependsOnData: false })).toEqual([]);
    // The settings still count: they are the only inputs such an analysis has.
    expect(staleReasons(before, stamp({ params: { a: 2 } }), { dependsOnData: false }))
      .toEqual(["params"]);
  });

  it("does not care that the run happened at a different time", () => {
    const before = { ...stamp(), at: 0 };
    expect(staleReasons(before, { ...stamp(), at: 1e12 })).toEqual([]);
  });
});

describe("describeStale", () => {
  it("is empty for a current result", () => {
    expect(describeStale([])).toBe("");
  });

  it("reads as one clause", () => {
    expect(describeStale(["data"])).toBe("the data changed");
    expect(describeStale(["data", "filter"]))
      .toBe("the data changed and the case filter changed");
    expect(describeStale(["data", "filter", "params"]))
      .toBe("the data changed, the case filter changed and the analysis settings changed");
  });
});
