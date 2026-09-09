import { describe, it, expect, beforeEach } from "vitest";
import {
  describeProvenance,
  forKey,
  fromResponseHeaders,
  latest,
  parsePackages,
  provenanceLines,
  record,
  resetProvenance,
  type Provenance,
} from "./provenance";

const HEADERS = {
  "x-ustat-runtime": "server",
  "x-ustat-engine": "python",
  "x-ustat-engine-version": "0.1.0",
  "x-ustat-engine-fingerprint": "b8a7237331bddcf1",
  "x-ustat-python": "3.11.9",
  "x-ustat-packages": "numpy=2.0.2,scipy=1.14.1,pandas=2.2.3",
};

describe("parsePackages", () => {
  it("reads the compact header form", () => {
    expect(parsePackages("numpy=2.0.2,scipy=1.14.1"))
      .toEqual({ numpy: "2.0.2", scipy: "1.14.1" });
  });

  it("survives an absent or malformed header without inventing versions", () => {
    expect(parsePackages(undefined)).toEqual({});
    expect(parsePackages("")).toEqual({});
    expect(parsePackages("garbage,=1.0,name=")).toEqual({});
  });

  it("keeps a version containing an equals sign intact", () => {
    expect(parsePackages("pkg=1.0=post1")).toEqual({ pkg: "1.0=post1" });
  });
});

describe("fromResponseHeaders", () => {
  it("reads engine, runtime and versions off the response", () => {
    const p = fromResponseHeaders(HEADERS);
    expect(p.runtime).toBe("server");
    expect(p.engine).toBe("python");
    expect(p.languageVersion).toBe("3.11.9");
    expect(p.packages).toEqual({ numpy: "2.0.2", scipy: "1.14.1", pandas: "2.2.3" });
    expect(p.engineVersion).toBe("0.1.0");
  });

  it("does not guess versions an older server did not send", () => {
    const p = fromResponseHeaders({});
    // The engine still defaults -- a bare server IS the Python one -- but a
    // version nobody reported must read as absent, not as a plausible number.
    expect(p.engine).toBe("python");
    expect(p.runtime).toBe("server");
    expect(p.languageVersion).toBeUndefined();
    expect(p.packages).toEqual({});
  });

  it("tolerates a header bag that is not lowercased", () => {
    const p = fromResponseHeaders({ "X-uStat-Engine": "python", "X-uStat-Python": "3.12" });
    expect(p.languageVersion).toBe("3.12");
  });
});

describe("describeProvenance", () => {
  it("names the engine, the version that decides the number, and where it ran", () => {
    expect(describeProvenance(fromResponseHeaders(HEADERS)))
      .toBe("Python 3.11.9 · scipy 1.14.1 · on the server");
  });

  it("says 'in your browser' for a local run", () => {
    const local: Provenance = {
      runtime: "local", engine: "r", languageVersion: "4.5.2",
      packages: { webR: "0.6.0" }, at: 0,
    };
    expect(describeProvenance(local)).toBe("R 4.5.2 · in your browser");
  });

  it("is empty rather than misleading when nothing was recorded", () => {
    expect(describeProvenance(null)).toBe("");
  });
});

describe("provenanceLines", () => {
  it("spells out every fact an export footer needs", () => {
    const lines = provenanceLines(fromResponseHeaders(HEADERS));
    expect(lines).toContain("Engine: Python");
    expect(lines).toContain("Computed: on the server");
    expect(lines).toContain("Runtime version: 3.11.9");
    expect(lines.some((l) => l.startsWith("Packages: ") && l.includes("scipy 1.14.1"))).toBe(true);
    expect(lines.some((l) => l.startsWith("uSTAT engine: 0.1.0"))).toBe(true);
  });

  it("records why the server answered an R session", () => {
    const p = { ...fromResponseHeaders(HEADERS), fellBackBecause: "not-implemented-in-r" };
    expect(provenanceLines(p)).toContain("Server answered because: not-implemented-in-r");
  });
});

describe("the ledger", () => {
  beforeEach(() => resetProvenance());

  it("keeps one entry per request key", () => {
    const a = fromResponseHeaders(HEADERS);
    const b: Provenance = { runtime: "local", engine: "r", at: 1 };
    record("/api/models/cox", a);
    record("stats.ttest", b);
    expect(forKey("/api/models/cox")?.engine).toBe("python");
    expect(forKey("stats.ttest")?.engine).toBe("r");
  });

  it("latest() is the most recent answer, whichever key it came under", () => {
    record("/api/a", fromResponseHeaders(HEADERS));
    record("stats.power", { runtime: "local", engine: "r", at: 2 });
    expect(latest()?.engine).toBe("r");
  });

  it("starts empty, so a result with no run behind it claims nothing", () => {
    expect(latest()).toBeNull();
    expect(forKey("/api/anything")).toBeUndefined();
  });

  it("bounds its own memory", () => {
    for (let i = 0; i < 250; i++) record(`/api/${i}`, { runtime: "server", engine: "python", at: i });
    expect(forKey("/api/0")).toBeUndefined();
    expect(forKey("/api/249")).toBeDefined();
  });
});
