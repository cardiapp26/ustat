import { create } from "zustand";
import { runColumnStructureMutation } from "./lib/columnStructureLock";

export { runColumnStructureMutation } from "./lib/columnStructureLock";

export type ColKind = "numeric" | "categorical" | "ordinal" | "text" | "date";

export interface ColMeta {
  name: string;
  dtype: string;
  /** Measurement kind. `ordinal` = ordered categorical (numeric-coded): it is
   *  eligible both where numeric and where categorical columns are, and lets
   *  order-aware tests (Spearman, trend, Jonckheere, ordinal logistic) detect it. */
  kind: ColKind;
  label?: string;
  description?: string;
  units?: string;
  value_labels?: Record<string, string>;
  missing_ranges?: Array<{ lo: string | number | null; hi: string | number | null }>;
  missing_user_values?: Array<string | number | null>;
  measure?: "nominal" | "ordinal" | "scale" | string;
  role?: "outcome" | "predictor" | "covariate" | "id" | "time" | "event" | "";
  /** When true the column is hidden from analysis variable pickers (kept in the
   *  dataset, e.g. NAME / row-id columns). Toggled from the data-tab menu. */
  analysis_excluded?: boolean;
  /** Optional pretty name suggestion (advisory; rename applies it). */
  display_name?: string;
}

/** Columns eligible for analysis (not flagged "exclude from analysis"). */
export const analysisCols = (cols: ColMeta[]): ColMeta[] =>
  cols.filter((c) => !c.analysis_excluded);

/** Ordinal counts as quantitative — usable wherever a numeric column is. */
export const isNumericKind = (k: ColKind): boolean => k === "numeric" || k === "ordinal";

/** Ordinal counts as categorical too — usable wherever a categorical column is. */
export const isCategoricalKind = (k: ColKind): boolean => k === "categorical" || k === "ordinal";

/** Human-readable kind label (SPSS-flavoured). */
export const KIND_LABEL: Record<ColKind, string> = {
  numeric: "Numeric",
  categorical: "Categorical",
  ordinal: "Ordered Categorical",
  text: "Text",
  date: "Date",
};

export interface Session {
  session_id: string;
  filename: string;
  rows: number;
  columns: ColMeta[];
  preview: Record<string, unknown>[];
  case_filter?: CaseFilter | null;
}

export type PaletteName = "indigo" | "clinical" | "nature" | "grayscale" | "warm" | "jama";

export interface PlotTheme {
  palette: PaletteName;
  fontFamily: string;
  fontSize: number;
  lineWidth: number;
  markerSize: number;
  markerOpacity: number;
  plotBg: string;
}

export const DEFAULT_THEME: PlotTheme = {
  palette: "indigo",
  fontFamily: "system-ui, sans-serif",
  fontSize: 11,
  lineWidth: 2,
  markerSize: 6,
  markerOpacity: 0.7,
  plotBg: "#ffffff",
};

export const PALETTES: Record<PaletteName, string[]> = {
  indigo:    ["#6366f1","#f59e0b","#10b981","#ef4444","#8b5cf6","#06b6d4","#84cc16","#f97316"],
  clinical:  ["#1a5276","#2874a6","#5dade2","#27ae60","#d35400","#8e44ad","#c0392b","#2c3e50"],
  nature:    ["#27ae60","#2ecc71","#f39c12","#e67e22","#8e44ad","#3498db","#e74c3c","#1abc9c"],
  grayscale: ["#111827","#374151","#6b7280","#9ca3af","#d1d5db","#4b5563","#1f2937","#374151"],
  warm:      ["#dc2626","#ea580c","#d97706","#ca8a04","#65a30d","#16a34a","#0891b2","#7c3aed"],
  jama:      ["#003087","#7f0000","#003b00","#5e0070","#663300","#004c4c","#004080","#380038"],
};

export type CaseOperator = "eq" | "ne" | "gt" | "lt" | "gte" | "lte" | "contains" | "missing" | "not_missing";

export interface CaseCondition {
  column: string;
  operator: CaseOperator;
  value: string;
  join: "AND" | "OR";
}

export interface CaseFilter {
  conditions: CaseCondition[];
  selected: number;
  total: number;
}

interface ColumnDependentState {
  columnDecimals: Record<string, number>;
  caseFilter: CaseFilter | null;
  panelCache: Record<string, unknown>;
  table1Result: unknown;
}

interface ColumnMutationSnapshot {
  sessionId: string;
  undoDepthAfter: number;
  before: ColumnDependentState;
  after: ColumnDependentState;
}

interface AppState {
  session: Session | null;
  originalSession: Session | null;
  /** The Recent-work row this session was resumed from, when it was.
   *
   *  Autosave has to know which saved row to write back to. It cannot work
   *  that out from the session itself: the server id is minted fresh on every
   *  restore, and the filename inside a duplicate's blob is still the
   *  original's — which sent the copy's edits to the original row and left
   *  the copy frozen at the moment it was duplicated. */
  localSessionId: string | null;
  setLocalSessionId: (id: string | null) => void;
  activeTab: string;
  showGrid: boolean;
  plotTheme: PlotTheme;
  caseFilter: CaseFilter | null;
  setSession: (s: Session) => void;
  setOriginalSession: (s: Session | null) => void;
  /** Rename the active session. Updates the React store immediately and
   *  fires a backend POST /sessions/{sid}/rename so the renamed value is
   *  round-tripped on subsequent save_session calls. Errors swallowed —
   *  the local rename succeeds regardless. */
  renameSession: (name: string) => void;
  setActiveTab: (t: string) => void;
  toggleGrid: () => void;
  clearSession: () => void;
  setPlotTheme: (patch: Partial<PlotTheme>) => void;
  setCaseFilter: (f: CaseFilter | null) => void;
  // Column kind override (data tab kind badge)
  updateColumnKind: (name: string, kind: ColMeta["kind"]) => void;
  // Inline cell editing
  updatePreviewCell: (rowIdx: number, col: string, value: unknown) => void;
  // Computed columns (Compute tab)
  addSessionColumn: (col: ColMeta, previewValues: (number | string | null)[]) => void;
  renameSessionColumn: (
    oldName: string,
    newName: string,
    serverCaseFilter?: CaseFilter | null,
  ) => void;
  removeSessionColumn: (
    name: string,
    serverCaseFilter?: CaseFilter | null,
  ) => void;
  removeSessionColumns: (
    names: string[],
    serverCaseFilter?: CaseFilter | null,
  ) => void;
  // Column reordering (drag & drop)
  reorderColumns: (fromIndex: number, toIndex: number) => Promise<void>;
  // Table 1 persistence across tab switches
  table1Result: unknown;
  setTable1Result: (r: unknown) => void;
  clearTable1: () => void;
  // Generic panel result cache — persists results across tab switches
  panelCache: Record<string, unknown>;
  setPanelCache: (panel: string, data: unknown) => void;
  clearPanelCache: (panel: string) => void;
  // Column rename propagation — every panel's persisted variable selection
  // (usePersistedPanelState) lives in panelCache, keyed by panel id. A rename
  // in the Data tab doesn't touch those cached strings, so a panel with the
  // old name still selected sends it straight to the backend and 404s on
  // "Column not found" the next time the user runs it there.
  renameInPanelCache: (oldName: string, newName: string) => void;
  // Cross-panel forest handoff — one panel (e.g. Cox time-horizon) drops
  // a set of forest rows here, the Forest Builder picks them up on mount
  // and clears it. Shape matches ForestRowInput.
  forestHandoff: Array<{ label: string; est: number | null; ci_low: number | null; ci_high: number | null; p: number | null; extra: string }> | null;
  forestHandoffLayout: { customTitle?: string; customSubtitle?: string; xLabel?: string; leftHeader?: string; rightHeader?: string; returnTab?: string; returnLabel?: string } | null;
  setForestHandoff: (
    rows: Array<{ label: string; est: number | null; ci_low: number | null; ci_high: number | null; p: number | null; extra: string }> | null,
    layout?: { customTitle?: string; customSubtitle?: string; xLabel?: string; leftHeader?: string; rightHeader?: string; returnTab?: string; returnLabel?: string } | null,
  ) => void;
  // Deep-link target for the Visual tab's inner sub-tab ("forest", etc.).
  // Consumed once by VisualChartsCombo then cleared.
  visualSubTab: string | null;
  setVisualSubTab: (sub: string | null) => void;
  // Column decimal formatting
  columnDecimals: Record<string, number>;  // col name → decimal places
  setColumnDecimals: (col: string, decimals: number) => void;
  clearColumnDecimals: (col: string) => void;
  // Hide a column from analysis variable pickers (kept in the dataset).
  setColumnAnalysisExcluded: (name: string, excluded: boolean) => void;
  // Undo / Redo (backend-driven)
  undoDepth: number;
  redoDepth: number;
  columnMutationUndo: ColumnMutationSnapshot[];
  columnMutationRedo: ColumnMutationSnapshot[];
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  /** Monotonic counter bumped on every data mutation (column add/remove, cell
   *  edit, recode, find-replace, date parse, paste, undo/redo). Drives autosave
   *  so in-place edits that don't change row/column counts still get persisted. */
  dataVersion: number;
  bumpDataVersion: () => void;
  deleteRow: (rowIdx: number) => Promise<void>;
  
  // Descriptive tab UI state
  descriptiveTab: "histogram" | "boxplot" | "violin" | "qq";
  setDescriptiveTab: (tab: "histogram" | "boxplot" | "violin" | "qq") => void;

  // Session History for Unified R Replication Code
  sessionHistory: { action: string; params: Record<string, unknown> }[];
  logAction: (action: string, params: Record<string, unknown>) => void;
  clearHistory: () => void;
}

const loadTheme = (): PlotTheme => {
  try { return { ...DEFAULT_THEME, ...JSON.parse(localStorage.getItem("plotTheme") ?? "{}") }; }
  catch { return DEFAULT_THEME; }
};

const REMOVED_COLUMN = Symbol("removed-column");
function isCachedResultKey(key: string): boolean {
  return /(result|results|delong)$/i.test(key);
}

function containsColumnReference(value: unknown, column: string): boolean {
  if (value === column) return true;
  if (Array.isArray(value)) {
    return value.some((item) => containsColumnReference(item, column));
  }
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).some(
      ([key, item]) => key === column || containsColumnReference(item, column),
    );
  }
  return false;
}

function remapPanelCacheValue(
  value: unknown,
  oldName: string,
  newName?: string,
): unknown | typeof REMOVED_COLUMN {
  if (value === oldName) return newName ?? REMOVED_COLUMN;
  if (Array.isArray(value)) {
    const next: unknown[] = [];
    for (const item of value) {
      // Nested tuples/specs form one compound selection. Removing one column
      // invalidates that whole entry, not merely one tuple position.
      if (
        !newName
        && item
        && typeof item === "object"
        && containsColumnReference(item, oldName)
      ) {
        continue;
      }
      const remapped = remapPanelCacheValue(item, oldName, newName);
      if (remapped !== REMOVED_COLUMN) next.push(remapped);
    }
    return next;
  }
  if (value && typeof value === "object") {
    const next: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (isCachedResultKey(key)) {
        next[key] = Array.isArray(item) ? [] : null;
        continue;
      }
      if (!newName && key === oldName) continue;
      const remapped = remapPanelCacheValue(item, oldName, newName);
      const nextKey = key === oldName ? (newName ?? key) : key;
      next[nextKey] = remapped === REMOVED_COLUMN ? "" : remapped;
    }
    return next;
  }
  return value;
}

function updatePanelCacheColumn(
  cache: Record<string, unknown>,
  oldName: string,
  newName?: string,
) {
  return remapPanelCacheValue(cache, oldName, newName) as Record<string, unknown>;
}

function dependentState(
  state: Pick<AppState, "columnDecimals" | "caseFilter" | "panelCache" | "table1Result">,
): ColumnDependentState {
  return {
    columnDecimals: state.columnDecimals,
    caseFilter: state.caseFilter,
    panelCache: state.panelCache,
    table1Result: state.table1Result,
  };
}

export const useStore = create<AppState>((set, get) => ({
  session: null,
  originalSession: null,
  localSessionId: null,
  setLocalSessionId: (id) => set({ localSessionId: id }),
  setOriginalSession: (s) => set({ originalSession: s }),
  renameSession: (rawName: string) => {
    const name = (rawName || "").trim();
    if (!name) return;
    set((state) => {
      if (!state.session) return {};
      const sid = state.session.session_id;
      // Fire-and-forget backend sync so save_session round-trips the
      // new value. Failure is non-fatal — local rename still applied.
      import("./api").then(({ default: api }) => {
        api.post(`/api/sessions/${sid}/rename`, { filename: name }).catch(() => null);
      });
      return { session: { ...state.session, filename: name } };
    });
  },
  activeTab: "data",
  descriptiveTab: "histogram",
  setDescriptiveTab: (t) => set({ descriptiveTab: t }),
  showGrid: localStorage.getItem("showGrid") !== "false",
  plotTheme: loadTheme(),
  table1Result: null,
  caseFilter: null,
  dataVersion: 0,
  sessionHistory: [],
  logAction: (action, params) => set((state) => ({ sessionHistory: [...state.sessionHistory, { action, params }] })),
  clearHistory: () => set({ sessionHistory: [] }),
  setSession: (s) => set((state) => {
    // Preserve UI state (decimal formatting, table1, filters, undo/redo
    // depth) across same-session refreshes (rename, dtype flip, refresh
    // after compute). Only the *initial* load — when session_id flips —
    // resets the per-column formatting to defaults.
    const sameSession = !!(s && state.session && s.session_id === state.session.session_id);
    if (sameSession) {
      return { session: s };
    }
    return {
      session: s,
      // A different session_id means a different dataset is open. Whoever
      // opened it says which saved row it belongs to, if any; until then it
      // belongs to none, so autosave falls back to matching.
      localSessionId: null,
      activeTab: "data",
      table1Result: null,
      caseFilter: s.case_filter ?? null,
      panelCache: {},
      undoDepth: 0,
      redoDepth: 0,
      columnMutationUndo: [],
      columnMutationRedo: [],
      dataVersion: 0,
      columnDecimals: {},
      sessionHistory: [],
    };
  }),
  setActiveTab: (t) => set({ activeTab: t }),
  setCaseFilter: (f) => set({
    caseFilter: f,
    table1Result: null,
    panelCache: {},
  }),
  toggleGrid: () => set((state) => {
    const next = !state.showGrid;
    localStorage.setItem("showGrid", String(next));
    return { showGrid: next };
  }),
  setPlotTheme: (patch) => set((state) => {
    const next = { ...state.plotTheme, ...patch };
    localStorage.setItem("plotTheme", JSON.stringify(next));
    return { plotTheme: next };
  }),
  clearSession: () => set({
    session: null,
    originalSession: null,
    activeTab: "data",
    table1Result: null,
    caseFilter: null,
    panelCache: {},
    undoDepth: 0,
    redoDepth: 0,
    columnMutationUndo: [],
    columnMutationRedo: [],
  }),
  updateColumnKind: (name, kind) => {
    set((state) => {
      if (!state.session) return state;
      // Fire-and-forget persistence to the backend so save_session captures
      // the user's classification override. Import done lazily to avoid a
      // circular import (api.ts → store).
      import("./api").then(({ setColumnKind }) => {
        setColumnKind(state.session!.session_id, name, kind).catch(() => {
          /* network errors here are non-fatal — UI state is the source of
             truth for the current session; the next save attempt re-syncs. */
        });
      });
      return {
        session: {
          ...state.session,
          columns: state.session.columns.map((c) =>
            c.name === name ? { ...c, kind } : c
          ),
        },
      };
    });
  },
  bumpDataVersion: () => set((state) => ({ dataVersion: state.dataVersion + 1 })),
  updatePreviewCell: (rowIdx, col, value) =>
    set((state) => {
      if (!state.session) return state;
      const preview = [...state.session.preview];
      preview[rowIdx] = { ...preview[rowIdx], [col]: value };
      return { session: { ...state.session, preview }, dataVersion: state.dataVersion + 1 };
    }),
  addSessionColumn: (col, previewValues) =>
    set((state) => {
      if (!state.session) return state;
      // Replace existing column IN PLACE (e.g. Convert value must not move the
      // column to the end), or append when it's genuinely new.
      const existingIdx = state.session.columns.findIndex((c) => c.name === col.name);
      const columns = existingIdx >= 0
        ? state.session.columns.map((c, i) => (i === existingIdx ? col : c))
        : [...state.session.columns, col];
      const preview = state.session.preview.map((row, i) => ({
        ...row,
        [col.name]: previewValues[i] ?? null,
      }));
      return { session: { ...state.session, columns, preview }, dataVersion: state.dataVersion + 1 };
    }),
  renameSessionColumn: (oldName, newName, serverCaseFilter) =>
    set((state) => {
      if (!state.session || oldName === newName) return state;
      if (!state.session.columns.some((column) => column.name === oldName)) {
        return state;
      }
      const before = dependentState(state);
      const columns = state.session.columns.map((column) =>
        column.name === oldName ? { ...column, name: newName } : column
      );
      const preview = state.session.preview.map((row) => {
        if (!(oldName in row)) return row;
        const nextRow = { ...row, [newName]: row[oldName] };
        delete nextRow[oldName];
        return nextRow;
      });
      const columnDecimals = { ...state.columnDecimals };
      if (oldName in columnDecimals) {
        columnDecimals[newName] = columnDecimals[oldName];
        delete columnDecimals[oldName];
      }
      const caseFilter = serverCaseFilter === undefined
        ? (state.caseFilter ? {
            ...state.caseFilter,
            conditions: state.caseFilter.conditions.map((condition) =>
              condition.column === oldName
                ? { ...condition, column: newName }
                : condition
            ),
          } : null)
        : serverCaseFilter;
      const after: ColumnDependentState = {
        columnDecimals,
        caseFilter,
        panelCache: updatePanelCacheColumn(state.panelCache, oldName, newName),
        table1Result: null,
      };
      const snapshot: ColumnMutationSnapshot = {
        sessionId: state.session.session_id,
        undoDepthAfter: state.undoDepth + 1,
        before,
        after,
      };
      return {
        session: {
          ...state.session,
          columns,
          preview,
          case_filter: caseFilter,
        },
        ...after,
        columnMutationUndo: [...state.columnMutationUndo, snapshot].slice(-50),
        columnMutationRedo: [],
        dataVersion: state.dataVersion + 1,
      };
    }),
  removeSessionColumn: (name, serverCaseFilter) => {
    get().removeSessionColumns([name], serverCaseFilter);
  },
  removeSessionColumns: (names, serverCaseFilter) =>
    set((state) => {
      if (!state.session) return state;
      const removedNames = new Set(
        names.filter((name) =>
          state.session!.columns.some((column) => column.name === name)
        ),
      );
      if (removedNames.size === 0) return state;
      const before = dependentState(state);
      const columns = state.session.columns.filter(
        (column) => !removedNames.has(column.name),
      );
      const preview = state.session.preview.map((row) => {
        const r = { ...row };
        for (const name of removedNames) delete r[name];
        return r;
      });
      const columnDecimals = { ...state.columnDecimals };
      for (const name of removedNames) delete columnDecimals[name];
      const remainingConditions = state.caseFilter?.conditions.filter(
        (condition) => !removedNames.has(condition.column),
      ) ?? [];
      const caseFilter = serverCaseFilter === undefined
        ? (state.caseFilter
          ? (remainingConditions.length > 0
            ? { ...state.caseFilter, conditions: remainingConditions }
            : null)
          : null)
        : serverCaseFilter;
      const after: ColumnDependentState = {
        columnDecimals,
        caseFilter,
        panelCache: [...removedNames].reduce(
          (cache, name) => updatePanelCacheColumn(cache, name),
          state.panelCache,
        ),
        table1Result: null,
      };
      const snapshot: ColumnMutationSnapshot = {
        sessionId: state.session.session_id,
        undoDepthAfter: state.undoDepth + 1,
        before,
        after,
      };
      return {
        session: {
          ...state.session,
          columns,
          preview,
          case_filter: caseFilter,
        },
        ...after,
        columnMutationUndo: [...state.columnMutationUndo, snapshot].slice(-50),
        columnMutationRedo: [],
        dataVersion: state.dataVersion + 1,
      };
    }),
  reorderColumns: async (fromIndex, toIndex) => {
    const state = get();
    if (!state.session || fromIndex === toIndex) return;
    const sessionId = state.session.session_id;
    const originalNames = state.session.columns.map((column) => column.name);
    const reorderedNames = [...originalNames];
    const [moved] = reorderedNames.splice(fromIndex, 1);
    if (!moved) return;
    reorderedNames.splice(toIndex, 0, moved);
    await runColumnStructureMutation(sessionId, async () => {
      const { default: api } = await import("./api");
      await api.post(`/api/sessions/${sessionId}/reorder_columns`, {
        columns: reorderedNames,
      });

      const afterRequest = get();
      const afterRequestNames =
        afterRequest.session?.session_id === sessionId
          ? afterRequest.session.columns.map((column) => column.name)
          : [];
      const structureChanged =
        afterRequestNames.length !== originalNames.length
        || afterRequestNames.some(
          (name, index) => name !== originalNames[index],
        );
      if (structureChanged) {
        // A column was created/renamed/deleted outside this action while the
        // reorder request was running. Read server's final order instead of
        // silently keeping a divergent optimistic order.
        const refreshed = await api.get(`/api/stats/${sessionId}/refresh`);
        set((current) => {
          if (current.session?.session_id !== sessionId) return current;
          return {
            session: { ...current.session, ...refreshed.data },
            undoDepth: current.undoDepth + 1,
            redoDepth: 0,
            columnMutationRedo: [],
          };
        });
        return;
      }

      set((current) => {
        if (current.session?.session_id !== sessionId) return current;
        const currentNames = current.session.columns.map((column) => column.name);
        if (
          currentNames.length !== originalNames.length
          || currentNames.some((name, index) => name !== originalNames[index])
        ) {
          return current;
        }
        const columnsByName = new Map(
          current.session.columns.map((column) => [column.name, column]),
        );
        const columns = reorderedNames
          .map((name) => columnsByName.get(name))
          .filter((column): column is ColMeta => Boolean(column));
        return {
          session: { ...current.session, columns },
          undoDepth: current.undoDepth + 1,
          redoDepth: 0,
          columnMutationRedo: [],
        };
      });
    });
  },
  setTable1Result: (r) => set({ table1Result: r }),
  clearTable1: () => set({ table1Result: null }),
  panelCache: {},
  setPanelCache: (panel, data) => set((state) => ({ panelCache: { ...state.panelCache, [panel]: data } })),
  clearPanelCache: (panel) => set((state) => {
    const next = { ...state.panelCache };
    delete next[panel];
    return { panelCache: next };
  }),
  renameInPanelCache: (oldName, newName) => set((state) => ({
    panelCache: updatePanelCacheColumn(state.panelCache, oldName, newName),
  })),
  forestHandoff: null,
  forestHandoffLayout: null,
  setForestHandoff: (rows, layout = null) => set({ forestHandoff: rows, forestHandoffLayout: layout }),
  visualSubTab: null,
  setVisualSubTab: (sub) => set({ visualSubTab: sub }),
  // Column decimal formatting
  columnDecimals: {},
  setColumnDecimals: (col, decimals) => set((state) => {
    // Fire-and-forget persistence so save_session captures the formatting.
    // Lazy import avoids the circular api.ts → store cycle.
    if (state.session) {
      const sid = state.session.session_id;
      import("./api").then(({ setColumnDecimalsApi }) => {
        setColumnDecimalsApi(sid, col, decimals).catch(() => {
          /* UI state remains the source of truth for the current session;
             a transient sync error will be re-tried on next change. */
        });
      });
    }
    return { columnDecimals: { ...state.columnDecimals, [col]: decimals } };
  }),
  clearColumnDecimals: (col) => set((state) => {
    // Fire-and-forget clear so save_session reflects the "auto" reset.
    if (state.session) {
      const sid = state.session.session_id;
      import("./api").then(({ setColumnDecimalsApi }) => {
        setColumnDecimalsApi(sid, col, null).catch(() => { /* non-fatal */ });
      });
    }
    const next = { ...state.columnDecimals };
    delete next[col];
    return { columnDecimals: next };
  }),
  setColumnAnalysisExcluded: (name, excluded) => set((state) => {
    if (!state.session) return state;
    const sid = state.session.session_id;
    import("./api").then(({ saveMetadata }) => {
      saveMetadata(sid, { [name]: { analysis_excluded: excluded } }).catch(() => { /* non-fatal */ });
    });
    return {
      session: {
        ...state.session,
        columns: state.session.columns.map((c) =>
          c.name === name ? { ...c, analysis_excluded: excluded } : c),
      },
    };
  }),
  // Undo / Redo — backend-driven (DataFrame snapshots on server)
  undoDepth: 0,
  redoDepth: 0,
  columnMutationUndo: [],
  columnMutationRedo: [],
  undo: async () => {
    const state = useStore.getState();
    if (!state.session) return;
    const sessionId = state.session.session_id;
    try {
      await runColumnStructureMutation(sessionId, async () => {
        const { default: api } = await import("./api");
        const res = await api.post(`/api/sessions/${sessionId}/undo`);
        const d = res.data;
        set((current) => {
          if (current.session?.session_id !== sessionId) return current;
          const snapshotIndex = current.columnMutationUndo.findLastIndex(
            (snapshot) =>
              snapshot.sessionId === sessionId
              && snapshot.undoDepthAfter === current.undoDepth,
          );
          const snapshot = current.columnMutationUndo[snapshotIndex];
          return {
            session: {
              ...current.session,
              rows: d.rows,
              columns: d.columns,
              preview: d.preview,
              case_filter: snapshot?.before.caseFilter ?? current.caseFilter,
            },
            ...(snapshot?.before ?? {}),
            columnMutationUndo: snapshot
              ? current.columnMutationUndo.filter((_, index) => index !== snapshotIndex)
              : current.columnMutationUndo,
            columnMutationRedo: snapshot
              ? [...current.columnMutationRedo, snapshot]
              : current.columnMutationRedo,
            undoDepth: d.undo_depth ?? 0,
            redoDepth: d.redo_depth ?? 0,
            dataVersion: current.dataVersion + 1,
          };
        });
      });
    } catch { /* nothing to undo */ }
  },
  redo: async () => {
    const state = useStore.getState();
    if (!state.session) return;
    const sessionId = state.session.session_id;
    try {
      await runColumnStructureMutation(sessionId, async () => {
        const { default: api } = await import("./api");
        const res = await api.post(`/api/sessions/${sessionId}/redo`);
        const d = res.data;
        set((current) => {
          if (current.session?.session_id !== sessionId) return current;
          const nextUndoDepth = d.undo_depth ?? 0;
          const snapshotIndex = current.columnMutationRedo.findLastIndex(
            (snapshot) =>
              snapshot.sessionId === sessionId
              && snapshot.undoDepthAfter === nextUndoDepth,
          );
          const snapshot = current.columnMutationRedo[snapshotIndex];
          return {
            session: {
              ...current.session,
              rows: d.rows,
              columns: d.columns,
              preview: d.preview,
              case_filter: snapshot?.after.caseFilter ?? current.caseFilter,
            },
            ...(snapshot?.after ?? {}),
            columnMutationUndo: snapshot
              ? [...current.columnMutationUndo, snapshot]
              : current.columnMutationUndo,
            columnMutationRedo: snapshot
              ? current.columnMutationRedo.filter((_, index) => index !== snapshotIndex)
              : current.columnMutationRedo,
            undoDepth: nextUndoDepth,
            redoDepth: d.redo_depth ?? 0,
            dataVersion: current.dataVersion + 1,
          };
        });
      });
    } catch { /* nothing to redo */ }
  },
  deleteRow: async (rowIdx: number) => {
    const state = useStore.getState();
    if (!state.session) return;
    try {
      const { deleteRow } = await import("./api");
      const res = await deleteRow(state.session.session_id, rowIdx);
      const d = res.data;
      // Same refresh payload structure as undo/redo, triggers global re-renders
      set({ 
        session: { ...state.session, rows: d.rows, columns: d.columns, preview: d.preview },
        // Add 1 to undo depth because delete is a destructive action we pushed
        undoDepth: state.undoDepth + 1,
        redoDepth: 0,
        columnMutationRedo: [],
      });
    } catch (e) {
      console.error("Failed to delete row", e);
      throw e;
    }
  },
}));
