import { describe, expect, it } from "vitest";
import { makeSession } from "../test/testUtils";
import { analysisDefaultColumns } from "./columnDefaults";

describe("analysisDefaultColumns", () => {
  it("uses full-data classification over preview and preserves candidate order", () => {
    const session = makeSession({
      rows: 3000,
      columns: [
        { name: "pid", dtype: "int64", kind: "numeric", id_like: true },
        { name: "visit_number", dtype: "float64", kind: "numeric", id_like: false },
        { name: "label", dtype: "object", kind: "text", role: "id" },
        { name: "age", dtype: "float64", kind: "numeric", id_like: false },
      ],
      preview: Array.from({ length: 10 }, (_, i) => ({ pid: 1, visit_number: i / 2 })),
    });

    const candidates = ["age", "pid", "label", "visit_number"];
    expect(analysisDefaultColumns(session, candidates)).toEqual(["age", "visit_number"]);
    expect(candidates).toEqual(["age", "pid", "label", "visit_number"]);
    expect(analysisDefaultColumns(session, ["pid", "label"])).toEqual([]);
  });

  it("keeps unique integer and date measurements eligible for automatic selection", () => {
    const session = makeSession({
      rows: 12,
      columns: [
        { name: "pid", dtype: "int64", kind: "numeric" },
        { name: "age", dtype: "int64", kind: "numeric", id_like: true },
        { name: "visit_date", dtype: "datetime64[ns]", kind: "date", id_like: true },
      ],
      preview: Array.from({ length: 12 }, (_, i) => ({ pid: i, age: i + 30, visit_date: `2026-01-${i + 10}` })),
    });
    expect(analysisDefaultColumns(session, ["pid", "age", "visit_date"])).toEqual(["age", "visit_date"]);
    const legacy = { ...session, columns: session.columns.map((column) => ({ ...column, id_like: undefined })) };
    expect(analysisDefaultColumns(legacy, ["pid", "age", "visit_date"])).toEqual(["age", "visit_date"]);
  });
});
