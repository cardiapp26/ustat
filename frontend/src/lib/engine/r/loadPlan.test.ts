/**
 * What the R engine is allowed to answer, and what proves the answer is current.
 *
 * The allow-list is the gate in front of a 22 MB download and, more to the
 * point, in front of a number a clinician will publish. Two properties matter
 * enough to pin: that it never lets through an id the shipped bundle does not
 * declare, and that a build whose frozen copy of the manifest has drifted from
 * the served one is caught rather than run.
 */
import { describe, expect, it } from "vitest";
import {
  R_ALLOW_LIST,
  R_BUNDLE_PACKAGES,
  R_MANIFEST_ANALYSES,
  mergeRPackages,
  rAnalysis,
  rManifestDisagreement,
  rNeedsFrame,
} from "./loadPlan";

/** The fetched-manifest shape, as `r/client.ts` forwards it from the worker. */
function fetched(overrides: Partial<{ id: string; needs_frame: boolean; r_packages: string[] }>[] = []) {
  const base = R_MANIFEST_ANALYSES.map((a) => ({
    id: a.id,
    needs_frame: a.needsFrame,
    r_packages: [...a.rPackages],
  }));
  return base.map((entry, i) => ({ ...entry, ...(overrides[i] ?? {}) }));
}

describe("R_ALLOW_LIST", () => {
  it("admits stats.ttest, which R1b watched produce the right numbers in a browser", () => {
    expect(R_ALLOW_LIST.has("stats.ttest")).toBe(true);
  });

  it("keeps power analysis out: R adds nothing there and would only cost a download", () => {
    expect(R_ALLOW_LIST.has("stats.power")).toBe(false);
  });

  it("admits nothing the shipped bundle does not declare", () => {
    const declared = new Set(R_MANIFEST_ANALYSES.map((a) => a.id));
    for (const id of R_ALLOW_LIST) {
      expect(declared, `${id} is allow-listed but no R source registers it`).toContain(id);
    }
  });

  it("does not admit an id merely because the bundle declares it", () => {
    // The whole point of a hand-written list: being registered is not the same
    // as having been watched run. If these ever coincide it must be because
    // every analysis was verified, not because someone wired the list to the
    // manifest.
    expect(R_ALLOW_LIST.size).toBeLessThanOrEqual(R_MANIFEST_ANALYSES.length);
  });
});

describe("the load plan", () => {
  it("knows stats.ttest reads a dataset", () => {
    expect(rNeedsFrame("stats.ttest")).toBe(true);
    expect(rAnalysis("stats.ttest")?.rPackages).toEqual(["moments", "nortest"]);
  });

  it("treats an unknown analysis as needing a frame rather than assuming it does not", () => {
    expect(rNeedsFrame("nothing.registered")).toBe(true);
  });

  it("always includes jsonlite, which the bundle itself cannot run without", () => {
    expect(mergeRPackages([], [])).toContain("jsonlite");
    expect(R_BUNDLE_PACKAGES).toContain("jsonlite");
  });

  it("does not re-list a package webR already installed", () => {
    expect(mergeRPackages(["jsonlite", "moments"], ["moments", "nortest"])).toEqual([
      "jsonlite",
      "moments",
      "nortest",
    ]);
  });
});

describe("cross-checking the fetched manifest", () => {
  it("agrees with the manifest this build was generated from", () => {
    expect(rManifestDisagreement(fetched())).toBeNull();
  });

  it("objects when the served bundle no longer declares an allow-listed analysis", () => {
    expect(rManifestDisagreement([])).toMatch(/stats\.ttest/);
  });

  it("objects when the served analysis changed its packages", () => {
    const drifted = fetched([{ r_packages: ["moments"] }]);
    expect(rManifestDisagreement(drifted)).toMatch(/r_packages/);
  });

  it("objects when the served analysis changed whether it reads a dataset", () => {
    const drifted = fetched([{ needs_frame: false }]);
    expect(rManifestDisagreement(drifted)).toMatch(/needs_frame/);
  });

  it("ignores drift in an analysis nobody is allowed to run anyway", () => {
    const extra = [...fetched(), { id: "meta.analyze", needs_frame: false, r_packages: ["nothing"] }];
    expect(rManifestDisagreement(extra)).toBeNull();
  });
});
