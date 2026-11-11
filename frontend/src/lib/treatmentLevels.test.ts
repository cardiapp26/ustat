import { describe, expect, it } from "vitest";
import { treatedLevelParam, treatmentCandidates } from "./treatmentLevels";

const preview = [
  { arm: "treat", dm: 1, stage: "I", age: 61, grp: 1 },
  { arm: "control", dm: 0, stage: "II", age: 70 },
  { arm: "treat", dm: 1, stage: "III", age: 55 },
  { arm: "control", dm: 0, stage: "I", age: null, grp: 2 },
];

describe("treatmentCandidates", () => {
  it("keeps 0/1 columns and other two-level columns, with sorted levels for the latter", () => {
    expect(treatmentCandidates(preview, ["arm", "dm", "stage", "age", "grp"])).toEqual({
      arm: ["control", "treat"],
      dm: null,
      // 1/2 coding needs the same explicit choice as text.
      grp: ["1", "2"],
    });
  });
});

describe("treatedLevelParam", () => {
  it("sends the level only for a text treatment with a valid choice", () => {
    expect(treatedLevelParam(["control", "treat"], "treat")).toEqual({ treated_level: "treat" });
    expect(treatedLevelParam(["control", "treat"], "")).toEqual({});
    expect(treatedLevelParam(null, "treat")).toEqual({});
  });
});
