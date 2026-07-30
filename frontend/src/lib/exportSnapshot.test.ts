import { describe, expect, it } from "vitest";
import {
  baseName,
  buildDelimited,
  cellToText,
  snapshotColumns,
} from "./exportSnapshot";

const payload = {
  filename: "cohort.xlsx",
  columns: [{ name: "id" }, { name: "age" }, { name: "site" }],
  data: [
    { id: 1, age: 61.5, site: "s1" },
    { id: 2, age: null, site: "s2" },
    { id: 3, age: 44, site: null },
  ],
};

describe("snapshotColumns", () => {
  it("takes the declared column order, not the first row's keys", () => {
    expect(snapshotColumns(payload)).toEqual(["id", "age", "site"]);
  });

  it("falls back to the union of row keys in first-seen order", () => {
    // A row can omit a key entirely when its value is missing, so reading the
    // header off row 0 would drop the column from the file.
    const p = { data: [{ id: 1 }, { id: 2, age: 44 }, { id: 3, site: "s1" }] };
    expect(snapshotColumns(p)).toEqual(["id", "age", "site"]);
  });
});

describe("cellToText", () => {
  it("writes a missing value as an empty field, never as a word", () => {
    // "null" or "NaN" in a CSV stops being missing and becomes a category —
    // the exact failure this codebase has had to fix repeatedly elsewhere.
    expect(cellToText(null)).toBe("");
    expect(cellToText(undefined)).toBe("");
    expect(cellToText(Number.NaN)).toBe("");
    expect(cellToText(Number.POSITIVE_INFINITY)).toBe("");
  });

  it("keeps numbers and booleans verbatim", () => {
    expect(cellToText(0)).toBe("0");
    expect(cellToText(-1.25)).toBe("-1.25");
    expect(cellToText(false)).toBe("false");
  });
});

describe("buildDelimited", () => {
  it("writes a header and one line per row", () => {
    const csv = buildDelimited(payload, ",");
    expect(csv.split("\r\n")).toEqual([
      "id,age,site",
      "1,61.5,s1",
      "2,,s2",
      "3,44,",
    ]);
  });

  it("uses tabs when asked", () => {
    expect(buildDelimited(payload, "\t").split("\r\n")[1]).toBe("1\t61.5\ts1");
  });

  it("quotes fields containing the delimiter, a quote or a newline", () => {
    const p = {
      columns: [{ name: "note" }],
      data: [
        { note: "a,b" },
        { note: 'say "hi"' },
        { note: "line1\nline2" },
      ],
    };
    expect(buildDelimited(p, ",").split("\r\n")).toEqual([
      "note",
      '"a,b"',
      '"say ""hi"""',
      '"line1\nline2"',
    ]);
  });

  it("emits an empty field for a key the row does not carry at all", () => {
    const p = { columns: [{ name: "id" }, { name: "age" }], data: [{ id: 7 }] };
    expect(buildDelimited(p, ",").split("\r\n")[1]).toBe("7,");
  });

  it("writes only a header when there are no rows", () => {
    expect(buildDelimited({ columns: [{ name: "id" }], data: [] }, ",")).toBe("id");
  });
});

describe("baseName", () => {
  it("strips one extension so the download is not doubled", () => {
    expect(baseName({ name: "cohort.xlsx", payload: "" }, payload)).toBe("cohort");
    expect(baseName({ name: "v1.2.final.csv", payload: "" }, payload)).toBe("v1.2.final");
  });

  it("leaves a name that only looks like it has an extension", () => {
    expect(baseName({ name: "study.2026", payload: "" }, payload)).toBe("study.2026");
  });

  it("falls back to the payload filename and then to a default", () => {
    expect(baseName({ name: "", payload: "" }, payload)).toBe("cohort");
    expect(baseName({ name: "", payload: "" }, {})).toBe("dataset");
  });
});
