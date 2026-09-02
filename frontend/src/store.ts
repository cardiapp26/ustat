import { create } from "zustand";
import { runColumnStructureMutation } from "./lib/columnStructureLock";
import type { EngineKind } from "./lib/engine/types";
import type { LegendPosition, ThemePreset } from "./lib/plotPresets";

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

export type PaletteName = "indigo" | "clinical" | "nature" | "grayscale" | "warm" | "jama" | "custom";

export interface PlotTheme {
  palette: PaletteName;
  /** ggplot2-style frame: grid, axis lines, panel background. The palette
   *  colours the series; this styles everything around them. */
  preset: ThemePreset;
  legendPosition: LegendPosition;
  fontFamily: string;
  fontSize: number;
  lineWidth: number;
  markerSize: number;
  markerOpacity: number;
  plotBg: string;
  /** Colours used when palette is "custom". Order is the trace order. */
  customPalette: string[];
  /**
   * Colour pinned to a named series, keyed by the group label as it is
   * printed. A journal that asks for "treatment in red" is asking for THIS,
   * not for a palette: palette order follows how the groups happen to sort,
   * so adding a third arm silently recolours the first two.
   */
  seriesColors: Record<string, string>;
}

export const DEFAULT_THEME: PlotTheme = {
  palette: "indigo",
  // "minimal" is the frame every chart already had: white panel, faint
  // grid, no axis lines. A saved theme from before presets existed loads
  // into it unchanged.
  preset: "minimal",
  legendPosition: "auto",
  fontFamily: "system-ui, sans-serif",
  fontSize: 11,
  lineWidth: 2,
  markerSize: 6,
  markerOpacity: 0.7,
  plotBg: "#ffffff",
  customPalette: ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c"],
  seriesColors: {},
};

export const PALETTES: Record<PaletteName, string[]> = {
  indigo:    ["#6366f1","#f59e0b","#10b981","#ef4444","#8b5cf6","#06b6d4","#84cc16","#f97316"],
  clinical:  ["#1a5276","#2874a6","#5dade2","#27ae60","#d35400","#8e44ad","#c0392b","#2c3e50"],
  nature:    ["#27ae60","#2ecc71","#f39c12","#e67e22","#8e44ad","#3498db","#e74c3c","#1abc9c"],
  grayscale: ["#111827","#374151","#6b7280","#9ca3af","#d1d5db","#4b5563","#1f2937","#374151"],
  warm:      ["#dc2626","#ea580c","#d97706","#ca8a04","#65a30d","#16a34a","#0891b2","#7c3aed"],
  jama:      ["#003087","#7f0000","#003b00","#5e0070","#663300","#004c4c","#004080","#380038"],
  // Placeholder: the live values come from theme.customPalette, so that a
  // palette the user edits is not frozen into a module constant.
  custom:    ["#4c72b0","#dd8452","#55a868","#c44e52","#8172b3","#937860","#da8bc3","#8c8c8c"],
};

/** The colours actually in force, with the custom list resolved. */
export function paletteOf(theme: PlotTheme): string[] {
  if (theme.palette !== "custom") return PALETTES[theme.palette];
  const colours = (theme.customPalette ?? []).filter(Boolean);
  // An empty custom palette would hand Plotly nothing to cycle and every
  // trace would come out the same default blue.
  return colours.length ? colours : PALETTES.indigo;
}

export type CaseOperator = "eq" | "ne" | "gt" | "lt" | "gte" | "lte" | "contains" | "missing" | "not_missing";

export interface CaseCondition {
  column: string;
  operator: CaseOperator;
  value: string;
  join: "AND" | "OR";
}

/** A session the server had forgotten, put back from its autosaved copy. */
export interface SessionRecoveryNotice {
  /** When the restore happened. */
  restoredAt: number;
  /** When the restored snapshot was taken — everything after it is lost. */
  snapshotAt: number;
  /** The dataset's name, so the notice names the file it is talking about. */
  name: string;
}

export interface CaseFilter {
  conditions: CaseCondition[];
  selected: number;
  total: number;
  /** Positions in the UNFILTERED preview that the filter drops. The grid marks
   *  them by default rather than hiding them, the way SPSS marks a filtered
   *  case; its "Only selected" toggle hides them on request. Either way an
   *  edit addresses the row's position in the unfiltered frame, so the marks
   *  are a view concern and never an addressing one. */
  excludedRows?: number[];
  /** Excluded rows past the preview window, which cannot be marked. */
  excludedBeyondPreview?: number;
}

/**
 * Where the analysis in `analysisId` actually ran, and why not where it was
 * asked to.
 *
 * One record per analysis id rather than a list, because the question a reader
 * has is "what produced the number I am looking at", and only the most recent
 * run of that analysis can answer it. Written by `localFirst` on every routed
 * call and read by exactly one module (`EngineProvenance`) -- the alternative,
 * a `runtime`/`engine` prop threaded through fifty panels, is fifty chances to
 * forget one and show the wrong provenance.
 */
export interface EngineNotice {
  engine: EngineKind;
  /** e.g. "R 4.6.0 · webR 0.6.0". Absent when the engine did not report one. */
  engineDetail?: string;
  /** The named reason a local run did not happen, when one was expected. */
  fellBackBecause?: string;
  at: number;
}

/**
 * The engine chosen at the welcome gate, in sessionStorage.
 *
 * sessionStorage and not localStorage, deliberately. The choice is made once,
 * per piece of work, in front of a screen that explains what R costs; it has to
 * survive a mid-session reload (an in-memory server session is restored, the
 * engine must not silently flip under it), and it must NOT survive into a new
 * visit, where the user would otherwise be paying 22 MB for a decision they
 * made last week and have forgotten.
 */
const ENGINE_KEY = "ustat.engine";
const ENGINE_SOURCE_KEY = "ustat.engine.source";

export function loadSessionEngine(): EngineKind {
  try {
    return sessionStorage.getItem(ENGINE_KEY) === "r" ? "r" : "python";
  } catch {
    // Private browsing modes can throw on sessionStorage access. Absent a
    // readable choice the answer is the default.
    return "python";
  }
}

/**
 * Where the current engine came from.
 *
 * "gate" is the choice made on the welcome screen. "resume" is a recent
 * session being reopened in the engine it was worked in, which happens without
 * passing the gate at all -- so a reader can see R named in the header having
 * last seen Python selected on the welcome screen, and nothing on screen
 * explains the difference. Recording which one it was is what lets the badge
 * say so.
 */
export type EngineSource = "gate" | "resume";

export function loadSessionEngineSource(): EngineSource {
  try {
    return sessionStorage.getItem(ENGINE_SOURCE_KEY) === "resume" ? "resume" : "gate";
  } catch {
    return "gate";
  }
}

function persistSessionEngine(engine: EngineKind, source: EngineSource): void {
  try {
    sessionStorage.setItem(ENGINE_KEY, engine);
    sessionStorage.setItem(ENGINE_SOURCE_KEY, source);
  } catch {
    /* nothing to do: the choice simply will not survive a reload */
  }
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
  /** Set when the backend had forgotten the session and it was restored from
   *  the autosaved snapshot. The restore rolls the app back to that snapshot's
   *  moment, so it is never a silent event: whoever renders this owes the user
   *  an explanation of what the rollback cost. Cleared when they dismiss it. */
  sessionRecovery: SessionRecoveryNotice | null;
  setSessionRecovery: (n: SessionRecoveryNotice | null) => void;
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
  setShowGrid: (on: boolean) => void;
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
  // Append instead of replace. A published forest often combines rows from
  // SEVERAL fits — a continuous exposure and its dichotomised form cannot sit
  // in one model — so a sender that expects to be called repeatedly asks for
  // its rows to be added to what is already there.
  forestHandoffAppend: boolean;
  setForestHandoff: (
    rows: Array<{ label: string; est: number | null; ci_low: number | null; ci_high: number | null; p: number | null; extra: string }> | null,
    layout?: { customTitle?: string; customSubtitle?: string; xLabel?: string; leftHeader?: string; rightHeader?: string; returnTab?: string; returnLabel?: string } | null,
    append?: boolean,
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

  /** Which statistics engine this session runs in the browser. Chosen at the
   *  welcome gate; see `loadSessionEngine` for why it lives in sessionStorage. */
  engine: EngineKind;
  /** Whether `engine` was chosen at the welcome gate or restored with a resumed session. */
  engineSource: EngineSource;
  setEngine: (engine: EngineKind, source?: EngineSource) => void;
  /** Where each analysis last ran, keyed by analysis id. Written by `localFirst`. */
  engineNotices: Record<string, EngineNotice>;
  noteEngineRun: (analysisId: string, notice: EngineNotice) => void;

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
  sessionRecovery: null,
  setSessionRecovery: (n) => set({ sessionRecovery: n }),
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
  engine: loadSessionEngine(),
  engineSource: loadSessionEngineSource(),
  setEngine: (engine, source = "gate") => {
    persistSessionEngine(engine, source);
    // The notices describe runs by the engine that is being left behind; a
    // stale "computed with Python" line under a fresh R session would be a
    // claim about a number that is no longer on screen.
    set({ engine, engineSource: source, engineNotices: {} });
  },
  engineNotices: {},
  noteEngineRun: (analysisId, notice) =>
    set((state) => ({ engineNotices: { ...state.engineNotices, [analysisId]: notice } })),
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
      // A changed row or column count is a structural edit — a dropped column,
      // a new one, deleted rows. Those are precisely what a session recovery
      // cannot reconstruct: the backend holds the data in memory only, so a
      // restart inside the autosave debounce restores a snapshot from before
      // the edit and the work appears to undo itself. Snapshot now instead of
      // waiting out the debounce. Cell edits keep the shape and stay debounced,
      // which is what the debounce is for.
      //
      // Dynamic import: the autosave hook reads this store, so a static one
      // would close the cycle.
      const shapeChanged =
        state.session!.rows !== s.rows ||
        state.session!.columns.length !== s.columns.length;
      if (shapeChanged) {
        void import("./hooks/useAutoSession")
          .then((m) => m.flushAutoSession())
          .catch(() => { /* the debounced save still covers it */ });
      }
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
      // A new dataset gets the engine the gate was left on -- which is what the
      // user just chose, since the gate is the only screen that can set it.
      // Re-hydrating the SAME session_id (rename, dtype flip, refresh after
      // compute) returns early above and never reaches here, so a mid-session
      // refresh cannot flip the engine under a running analysis.
      engine: loadSessionEngine(),
      engineSource: loadSessionEngineSource(),
      engineNotices: {},
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
  setShowGrid: (on) => set(() => {
    localStorage.setItem("showGrid", String(on));
    return { showGrid: on };
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
    // The notices belong to a session that no longer exists. `engine` stays:
    // the user is going back to the gate, where it is the current selection.
    engineNotices: {},
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
  forestHandoffAppend: false,
  setForestHandoff: (rows, layout = null, append = false) =>
    set({ forestHandoff: rows, forestHandoffLayout: layout, forestHandoffAppend: append }),
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
