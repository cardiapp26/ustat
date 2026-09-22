import { describe, it, expect, beforeEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useStampedResult } from "./useStampedResult";
import { useStore } from "../store";
import { installSession } from "../test/testUtils";

describe("useStampedResult", () => {
  beforeEach(() => installSession());

  it("keeps the result across a remount, the reason the cache exists", () => {
    const first = renderHook(() => useStampedResult<{ auc: number }>("demo", { a: 1 }));
    act(() => first.result.current.setResult({ auc: 0.81 }));
    first.unmount();

    const second = renderHook(() => useStampedResult<{ auc: number }>("demo", { a: 1 }));
    expect(second.result.current.result).toEqual({ auc: 0.81 });
    expect(second.result.current.stale).toBe(false);
  });

  it("treats a cached result with no stamp as out of date, not current", () => {
    // A result cached before stamping existed (or through a path that drops
    // the stamp) says nothing about what it was computed from.
    act(() => useStore.getState().setPanelCache("legacy", { result: { auc: 0.7 } }));
    const view = renderHook(() => useStampedResult<{ auc: number }>("legacy", { a: 1 }));
    expect(view.result.current.result).toEqual({ auc: 0.7 });
    expect(view.result.current.stale).toBe(true);
    expect(view.result.current.staleReasons).toEqual(["data"]);
  });

  it("marks a result from another session out of date even at the same data version", () => {
    // dataVersion restarts at 0 for every session, so it cannot tell two
    // datasets apart on its own.
    const run = renderHook(() => useStampedResult<{ auc: number }>("swap", { a: 1 }));
    act(() => run.result.current.setResult({ auc: 0.81 }));
    run.unmount();
    const s = useStore.getState();
    act(() => useStore.setState({ session: { ...s.session!, session_id: "another" }, dataVersion: s.dataVersion }));
    const again = renderHook(() => useStampedResult<{ auc: number }>("swap", { a: 1 }));
    expect(again.result.current.staleReasons).toEqual(["data"]);
  });

  it("marks a cached result stale once the data changes under it", () => {
    // The reported bug: edit a cell, come back to the panel, and the previous
    // fit is still on screen presented as current.
    const run = renderHook(() => useStampedResult<{ auc: number }>("models", { outcome: "y" }));
    act(() => run.result.current.setResult({ auc: 0.81 }));
    run.unmount();

    act(() => { useStore.getState().bumpDataVersion(); });

    const back = renderHook(() => useStampedResult<{ auc: number }>("models", { outcome: "y" }));
    expect(back.result.current.result).toEqual({ auc: 0.81 });
    expect(back.result.current.stale).toBe(true);
    expect(back.result.current.staleReasons).toEqual(["data"]);
  });

  it("marks it stale when the case filter moves", () => {
    const { result, rerender } = renderHook(() => useStampedResult("demo", { a: 1 }));
    act(() => result.current.setResult({ ok: true }));
    expect(result.current.stale).toBe(false);

    act(() => {
      useStore.getState().setCaseFilter({
        conditions: [{ column: "AGE", op: ">", value: 60 }] as never,
        selected: 1,
        total: 3,
      });
    });
    rerender();
    expect(result.current.staleReasons).toEqual(["filter"]);
  });

  it("marks it stale when the settings change", () => {
    let params: Record<string, unknown> = { outcome: "y", predictors: ["AGE"] };
    const { result, rerender } = renderHook(() => useStampedResult("demo", params));
    act(() => result.current.setResult({ ok: true }));

    params = { outcome: "y", predictors: ["AGE", "LDL"] };
    rerender();
    expect(result.current.staleReasons).toEqual(["params"]);
  });

  it("is not stale merely because the params object was rebuilt", () => {
    let params: Record<string, unknown> = { b: 2, a: 1 };
    const { result, rerender } = renderHook(() => useStampedResult("demo", params));
    act(() => result.current.setResult({ ok: true }));

    params = { a: 1, b: 2 };   // same values, new object, keys the other way
    rerender();
    expect(result.current.stale).toBe(false);
  });

  it("ignores data changes for an analysis that reads no dataset", () => {
    const { result, rerender } = renderHook(
      () => useStampedResult("power", { alpha: "0.05" }, { dependsOnData: false }),
    );
    act(() => result.current.setResult({ n: 64 }));
    act(() => { useStore.getState().bumpDataVersion(); });
    rerender();
    expect(result.current.stale).toBe(false);
  });

  it("clears the staleness by recomputing, not by the data settling", () => {
    const { result, rerender } = renderHook(() => useStampedResult("demo", { a: 1 }));
    act(() => result.current.setResult({ v: 1 }));
    act(() => { useStore.getState().bumpDataVersion(); });
    rerender();
    expect(result.current.stale).toBe(true);

    act(() => result.current.setResult({ v: 2 }));
    rerender();
    expect(result.current.stale).toBe(false);
  });

  it("merges into the panel cache instead of clobbering its sibling keys", () => {
    // usePersistedPanelState keeps each panel's variable selections in the
    // same object. The hand-written version replaced it, so running a model
    // silently reset the panel's own pickers on the next remount.
    act(() => { useStore.getState().setPanelCache("models", { outcome: "LDL", predictors: ["AGE"] }); });
    const { result } = renderHook(() => useStampedResult("models", { a: 1 }));
    act(() => result.current.setResult({ v: 1 }));

    const cache = useStore.getState().panelCache.models as Record<string, unknown>;
    expect(cache.outcome).toBe("LDL");
    expect(cache.predictors).toEqual(["AGE"]);
    expect(cache.result).toEqual({ v: 1 });
    expect(cache.stamp).toBeTruthy();
  });

  it("drops the stamp with the result", () => {
    const { result } = renderHook(() => useStampedResult("demo", { a: 1 }));
    act(() => result.current.setResult({ v: 1 }));
    act(() => result.current.setResult(null));
    expect(result.current.stamp).toBeNull();
    expect(result.current.stale).toBe(false);
  });
});
