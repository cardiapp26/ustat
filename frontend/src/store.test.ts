import { describe, it, expect, beforeEach } from "vitest";
import { http, HttpResponse } from "msw";
import { runColumnStructureMutation, useStore } from "./store";
import { server } from "./test/server";
import { makeSession } from "./test/testUtils";
import { computeFormula } from "./api";

describe("renameInPanelCache", () => {
  beforeEach(() => {
    useStore.setState({ panelCache: {} });
  });

  it("remaps a plain string selection to the new column name", () => {
    useStore.getState().setPanelCache("charts", { x: "LDL", color: "SEX" });
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache.charts).toEqual({ x: "LDL_mgdl", color: "SEX" });
  });

  it("remaps the old name inside array selections without touching other entries", () => {
    useStore.getState().setPanelCache("iptw", { covariates: ["AGE", "LDL", "SEX"] });
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache.iptw).toEqual({ covariates: ["AGE", "LDL_mgdl", "SEX"] });
  });

  it("remaps the same old name across multiple panels in one call", () => {
    useStore.getState().setPanelCache("charts", { x: "LDL" });
    useStore.getState().setPanelCache("table1", { variables: ["LDL", "AGE"] });
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache.charts).toEqual({ x: "LDL_mgdl" });
    expect(useStore.getState().panelCache.table1).toEqual({ variables: ["LDL_mgdl", "AGE"] });
  });

  it("leaves values that don't match the old name untouched", () => {
    useStore.getState().setPanelCache("charts", { x: "AGE", bins: 20 });
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache.charts).toEqual({ x: "AGE", bins: 20 });
  });

  it("is a no-op on an empty panelCache", () => {
    useStore.getState().renameInPanelCache("LDL", "LDL_mgdl");
    expect(useStore.getState().panelCache).toEqual({});
  });
});

describe("shared column structure mutations", () => {
  beforeEach(() => {
    useStore.setState({
      session: makeSession(),
      columnDecimals: { AGE: 1, LDL: 2 },
      panelCache: {
        charts: { x: "AGE", color: "GROUP" },
        table1: { variables: ["AGE", "LDL", "GROUP"] },
      },
      caseFilter: {
        conditions: [
          { column: "AGE", operator: "gt", value: "50", join: "AND" },
          { column: "GROUP", operator: "eq", value: "A", join: "AND" },
        ],
        selected: 2,
        total: 3,
      },
      table1Result: { stale: true },
      dataVersion: 10,
      undoDepth: 0,
      redoDepth: 0,
      columnMutationUndo: [],
      columnMutationRedo: [],
    });
  });

  it("renames columns across session data and every dependent store slice", () => {
    useStore.getState().renameSessionColumn("AGE", "AGE_YEARS");

    const state = useStore.getState();
    expect(state.session?.columns.map((column) => column.name)).toEqual([
      "AGE_YEARS", "LDL", "DM", "GROUP",
    ]);
    expect(state.session?.preview[0]).toMatchObject({ AGE_YEARS: 55 });
    expect(state.session?.preview[0]).not.toHaveProperty("AGE");
    expect(state.columnDecimals).toEqual({ AGE_YEARS: 1, LDL: 2 });
    expect(state.panelCache.charts).toEqual({ x: "AGE_YEARS", color: "GROUP" });
    expect(state.panelCache.table1).toEqual({
      variables: ["AGE_YEARS", "LDL", "GROUP"],
    });
    expect(state.caseFilter?.conditions[0].column).toBe("AGE_YEARS");
    expect(state.table1Result).toBeNull();
    expect(state.dataVersion).toBe(11);
  });

  it("deletes columns across session data and removes stale dependencies", () => {
    useStore.getState().removeSessionColumn("AGE");

    const state = useStore.getState();
    expect(state.session?.columns.map((column) => column.name)).toEqual([
      "LDL", "DM", "GROUP",
    ]);
    expect(state.session?.preview[0]).not.toHaveProperty("AGE");
    expect(state.columnDecimals).toEqual({ LDL: 2 });
    expect(state.panelCache.charts).toEqual({ x: "", color: "GROUP" });
    expect(state.panelCache.table1).toEqual({
      variables: ["LDL", "GROUP"],
    });
    expect(state.caseFilter?.conditions).toEqual([
      { column: "GROUP", operator: "eq", value: "A", join: "AND" },
    ]);
    expect(state.table1Result).toBeNull();
    expect(state.dataVersion).toBe(11);
  });

  it("recursively remaps nested selections and column-keyed maps", () => {
    useStore.setState({
      panelCache: {
        models: {
          glmInteractions: [["AGE", "LDL"], ["LDL", "GROUP"]],
          references: { AGE: "55", GROUP: "A" },
          endpoint: { column: "AGE", settings: { stratifier: "GROUP" } },
          result: {
            coefficients: [{ variable: "AGE", estimate: 0.4 }],
          },
        },
        table1: { kindOverrides: { AGE: "numeric", GROUP: "categorical" } },
      },
    });

    useStore.getState().renameSessionColumn("AGE", "AGE_YEARS");

    expect(useStore.getState().panelCache).toMatchObject({
      models: {
        glmInteractions: [["AGE_YEARS", "LDL"], ["LDL", "GROUP"]],
        references: { AGE_YEARS: "55", GROUP: "A" },
        endpoint: { column: "AGE_YEARS", settings: { stratifier: "GROUP" } },
        result: null,
      },
      table1: {
        kindOverrides: { AGE_YEARS: "numeric", GROUP: "categorical" },
      },
    });
  });

  it("drops invalid nested compound selections on delete", () => {
    useStore.setState({
      panelCache: {
        models: {
          glmInteractions: [["AGE", "LDL"], ["LDL", "GROUP"]],
          references: { AGE: "55", GROUP: "A" },
          endpoint: { column: "AGE" },
        },
      },
    });

    useStore.getState().removeSessionColumn("AGE");

    expect(useStore.getState().panelCache.models).toEqual({
      glmInteractions: [["LDL", "GROUP"]],
      references: { GROUP: "A" },
      endpoint: { column: "" },
    });
  });

  it("uses backend filter counts after a structural mutation", () => {
    const serverFilter = {
      conditions: [
        { column: "GROUP", operator: "eq" as const, value: "A", join: "AND" as const },
      ],
      selected: 17,
      total: 200,
    };

    useStore.getState().removeSessionColumn("AGE", serverFilter);

    expect(useStore.getState().caseFilter).toEqual(serverFilter);
    expect(useStore.getState().session?.case_filter).toEqual(serverFilter);
  });

  it("restores and reapplies dependent state with structural undo and redo", async () => {
    const originalSession = makeSession();
    useStore.getState().renameSessionColumn("AGE", "AGE_YEARS");
    useStore.setState({ undoDepth: 1, redoDepth: 0 });
    const renamedSession = useStore.getState().session!;

    server.use(
      http.post("/api/sessions/test-session/undo", () => HttpResponse.json({
        rows: originalSession.rows,
        columns: originalSession.columns,
        preview: originalSession.preview,
        undo_depth: 0,
        redo_depth: 1,
      })),
      http.post("/api/sessions/test-session/redo", () => HttpResponse.json({
        rows: renamedSession.rows,
        columns: renamedSession.columns,
        preview: renamedSession.preview,
        undo_depth: 1,
        redo_depth: 0,
      })),
    );

    await useStore.getState().undo();
    expect(useStore.getState().session?.columns[0].name).toBe("AGE");
    expect(useStore.getState().columnDecimals).toEqual({ AGE: 1, LDL: 2 });
    expect(useStore.getState().panelCache.charts).toEqual({
      x: "AGE",
      color: "GROUP",
    });
    expect(useStore.getState().caseFilter?.conditions[0].column).toBe("AGE");
    expect(useStore.getState().table1Result).toEqual({ stale: true });

    await useStore.getState().redo();
    expect(useStore.getState().session?.columns[0].name).toBe("AGE_YEARS");
    expect(useStore.getState().columnDecimals).toEqual({
      AGE_YEARS: 1,
      LDL: 2,
    });
    expect(useStore.getState().panelCache.charts).toEqual({
      x: "AGE_YEARS",
      color: "GROUP",
    });
    expect(useStore.getState().caseFilter?.conditions[0].column).toBe("AGE_YEARS");
    expect(useStore.getState().table1Result).toBeNull();
  });

  it("does not consume rename dependencies when undo only reverses a reorder", async () => {
    useStore.getState().renameSessionColumn("AGE", "AGE_YEARS");
    useStore.setState({ undoDepth: 1, redoDepth: 0 });

    server.use(
      http.post("/api/sessions/test-session/reorder_columns", () =>
        HttpResponse.json({ columns: ["LDL", "DM", "GROUP", "AGE_YEARS"] })
      ),
      http.post("/api/sessions/test-session/undo", () => HttpResponse.json({
        rows: 3,
        columns: useStore.getState().session!.columns,
        preview: useStore.getState().session!.preview,
        undo_depth: 1,
        redo_depth: 1,
      })),
    );

    await useStore.getState().reorderColumns(0, 3);
    expect(useStore.getState().undoDepth).toBe(2);

    await useStore.getState().undo();

    expect(useStore.getState().panelCache.charts).toEqual({
      x: "AGE_YEARS",
      color: "GROUP",
    });
    expect(useStore.getState().columnDecimals).toEqual({
      AGE_YEARS: 1,
      LDL: 2,
    });
    expect(useStore.getState().columnMutationUndo).toHaveLength(1);
  });

  it("ignores overlapping reorders so server and local order cannot cross", async () => {
    let calls = 0;
    server.use(
      http.post("/api/sessions/test-session/reorder_columns", async () => {
        calls += 1;
        await new Promise((resolve) => setTimeout(resolve, 40));
        return HttpResponse.json({ columns: ["LDL", "DM", "GROUP", "AGE"] });
      }),
    );

    const first = useStore.getState().reorderColumns(0, 3);
    const second = useStore.getState().reorderColumns(1, 2);
    const outcomes = await Promise.allSettled([first, second]);

    expect(calls).toBe(1);
    expect(outcomes.map((outcome) => outcome.status)).toEqual([
      "fulfilled",
      "rejected",
    ]);
    expect(useStore.getState().session?.columns.map((column) => column.name)).toEqual([
      "LDL", "DM", "GROUP", "AGE",
    ]);
  });

  it("blocks rename or delete work while a reorder is in flight", async () => {
    server.use(
      http.post("/api/sessions/test-session/reorder_columns", async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
        return HttpResponse.json({ columns: ["LDL", "DM", "GROUP", "AGE"] });
      }),
    );
    let overlappingMutationRan = false;

    const reorder = useStore.getState().reorderColumns(0, 3);
    const overlap = runColumnStructureMutation("test-session", async () => {
      overlappingMutationRan = true;
    });
    const [, overlapResult] = await Promise.allSettled([reorder, overlap]);

    expect(overlapResult.status).toBe("rejected");
    expect(overlappingMutationRan).toBe(false);
  });

  it("refreshes authoritative order when a column is created during reorder", async () => {
    server.use(
      http.post("/api/sessions/test-session/reorder_columns", async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
        return HttpResponse.json({
          columns: ["LDL", "DM", "GROUP", "AGE", "NEW_COLUMN"],
        });
      }),
      http.get("/api/stats/test-session/refresh", () => HttpResponse.json({
        rows: 3,
        columns: [
          { name: "LDL", dtype: "float64", kind: "numeric" },
          { name: "DM", dtype: "int64", kind: "numeric" },
          { name: "GROUP", dtype: "object", kind: "categorical" },
          { name: "AGE", dtype: "float64", kind: "numeric" },
          { name: "NEW_COLUMN", dtype: "float64", kind: "numeric" },
        ],
        preview: [],
      })),
    );

    const reorder = useStore.getState().reorderColumns(0, 3);
    useStore.getState().addSessionColumn(
      { name: "NEW_COLUMN", dtype: "float64", kind: "numeric" },
      [1, 2, 3],
    );
    await reorder;

    expect(useStore.getState().session?.columns.map((column) => column.name)).toEqual([
      "LDL", "DM", "GROUP", "AGE", "NEW_COLUMN",
    ]);
  });

  it("does not send undo while a reorder is in flight", async () => {
    let undoCalls = 0;
    server.use(
      http.post("/api/sessions/test-session/reorder_columns", async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
        return HttpResponse.json({ columns: ["LDL", "DM", "GROUP", "AGE"] });
      }),
      http.post("/api/sessions/test-session/undo", () => {
        undoCalls += 1;
        return HttpResponse.json({});
      }),
    );

    const reorder = useStore.getState().reorderColumns(0, 3);
    await useStore.getState().undo();
    await reorder;

    expect(undoCalls).toBe(0);
    expect(useStore.getState().session?.columns.map((column) => column.name)).toEqual([
      "LDL", "DM", "GROUP", "AGE",
    ]);
  });

  it("does not send undo while computed-column creation is in flight", async () => {
    let undoCalls = 0;
    server.use(
      http.post("/api/compute/test-session/formula", async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
        return HttpResponse.json({ name: "BMI" });
      }),
      http.post("/api/sessions/test-session/undo", () => {
        undoCalls += 1;
        return HttpResponse.json({});
      }),
    );

    const compute = computeFormula("test-session", {
      formula: "AGE * 2",
      new_col: "BMI",
    });
    await useStore.getState().undo();
    await compute;

    expect(undoCalls).toBe(0);
  });
});
