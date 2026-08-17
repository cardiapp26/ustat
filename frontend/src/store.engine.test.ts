/**
 * The engine choice: what it survives, and what resets it.
 *
 * Getting this wrong is not a cosmetic bug. The choice decides which runtime a
 * session downloads and which engine's numbers a clinician is reading, and it
 * is made on a screen the user only sees once. Two failures are pinned here
 * because both would be silent: a mid-session refresh flipping the engine under
 * a running analysis, and a brand-new dataset inheriting the engine a previous
 * one happened to end on.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { useStore, loadSessionEngine, type Session } from "./store";

function session(id: string): Session {
  return { session_id: id, filename: "trial.csv", rows: 300, columns: [], preview: [] };
}

beforeEach(() => {
  sessionStorage.clear();
  useStore.setState({ session: null, engine: "python", engineNotices: {} });
});

describe("choosing an engine", () => {
  it("defaults to Python, so a first visit downloads nothing extra", () => {
    expect(loadSessionEngine()).toBe("python");
  });

  it("persists to sessionStorage, so a mid-session reload keeps it", () => {
    useStore.getState().setEngine("r");
    expect(sessionStorage.getItem("ustat.engine")).toBe("r");
    expect(loadSessionEngine()).toBe("r");
  });

  it("does not persist past the tab: a new visit starts neutral", () => {
    useStore.getState().setEngine("r");
    // sessionStorage is per tab-session; localStorage would have survived here,
    // which is exactly the behaviour this choice must NOT have.
    expect(localStorage.getItem("ustat.engine")).toBeNull();
  });
});

describe("setSession", () => {
  it("keeps the engine when the same session_id is re-hydrated", () => {
    useStore.getState().setEngine("r");
    useStore.getState().setSession(session("abc"));
    expect(useStore.getState().engine).toBe("r");

    // A rename, a dtype flip, a refresh after compute: same id, new payload.
    sessionStorage.setItem("ustat.engine", "python");
    useStore.getState().setSession({ ...session("abc"), filename: "renamed.csv" });
    expect(useStore.getState().engine).toBe("r");
  });

  it("resets the engine to the gate's choice when a NEW session_id appears", () => {
    useStore.getState().setEngine("r");
    useStore.getState().setSession(session("abc"));
    expect(useStore.getState().engine).toBe("r");

    sessionStorage.setItem("ustat.engine", "python");
    useStore.getState().setSession(session("def"));
    expect(useStore.getState().engine).toBe("python");
  });

  it("drops the engine notices with the old dataset", () => {
    useStore.getState().noteEngineRun("stats.ttest", { engine: "r", at: 1 });
    useStore.getState().setSession(session("abc"));
    expect(useStore.getState().engineNotices).toEqual({});
  });
});

describe("engineNotices", () => {
  it("keeps one record per analysis, the most recent winning", () => {
    useStore.getState().noteEngineRun("stats.ttest", { engine: "r", at: 1 });
    useStore.getState().noteEngineRun("stats.ttest", {
      engine: "python",
      fellBackBecause: "r-runtime-load-failed",
      at: 2,
    });
    expect(useStore.getState().engineNotices["stats.ttest"]).toMatchObject({
      engine: "python",
      at: 2,
    });
  });

  it("is cleared when the engine changes, so no notice outlives its engine", () => {
    useStore.getState().noteEngineRun("stats.ttest", { engine: "python", at: 1 });
    useStore.getState().setEngine("r");
    expect(useStore.getState().engineNotices).toEqual({});
  });
});
