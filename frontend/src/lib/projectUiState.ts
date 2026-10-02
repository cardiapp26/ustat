/**
 * The frontend half of a project's `ui/state.json` (docs/DESIGN_project_file.md,
 * Phase 2): what gets saved, and how it comes back without lying.
 *
 * What it carries is the panel state the app would otherwise lose on close:
 * every panel's selections and its stamped result (`useStampedResult` keeps
 * both in `panelCache`), plus the active tab.
 *
 * The subtle part is the stamps. A `ResultStamp.dataVersion` is a counter in
 * the session that produced it; the restored session starts a fresh counter
 * at 0. Copied verbatim, every restored result would compare 7 !== 0 and show
 * as "the data changed" -- stale for a reason that is not true, which is the
 * exact failure mode stamps exist to prevent, mirrored. So stamps are rebased
 * on restore:
 *
 *   - a result that was CURRENT at save (stamp.dataVersion === the saved
 *     session's dataVersion) is rebased to the new session's 0: the dataset
 *     restored under it is byte-identical (the manifest hash guards that), so
 *     it is still current;
 *   - a result that was ALREADY STALE at save is rebased to -1, which can
 *     never equal a live counter: it stays stale, and its reasons survive.
 *
 * Filter and params keys need no rebasing: they are content-derived
 * (`stableStringify`), and the restored filter/selections reproduce them.
 */
import { useStore, type SavedAnalysis, type SplitFile } from "../store";
import type { ResultStamp } from "./resultStamp";
import { restoreOutputItems, useOutputDoc, type OutputItem } from "./outputDoc";

export interface ProjectUiState {
  /** The saving session's data counter, so restore can tell fresh from stale. */
  dataVersion: number;
  activeTab?: string;
  panelCache: Record<string, unknown>;
  table1Result?: unknown;
  /** Named analyses. The backend lifts these out of ui/state.json into the
   *  project's analyses/ and results/ parts and merges them back on load. */
  savedAnalyses?: SavedAnalysis[];
  /** The output document (lib/outputDoc), in order. */
  outputItems?: OutputItem[];
  /** Split File: the variable, its levels and the level on view. */
  splitFile?: SplitFile;
}

/** Per-level Split File results are recomputed on demand; saving them would
 *  multiply the size of every snapshot by the number of levels. */
function withoutSplitResults(panelCache: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(panelCache).map(([panel, entry]) => {
    if (typeof entry !== "object" || entry === null || !("splitResults" in entry)) return [panel, entry];
    const { splitResults: _dropped, ...rest } = entry as Record<string, unknown>;
    void _dropped;
    return [panel, rest];
  }));
}

/** Snapshot the live store's panel state for ui/state.json. */
export function collectUiState(): ProjectUiState {
  const s = useStore.getState();
  const out: ProjectUiState = {
    dataVersion: s.dataVersion,
    activeTab: s.activeTab,
    panelCache: withoutSplitResults(s.panelCache),
  };
  if (s.table1Result != null) out.table1Result = s.table1Result;
  if (s.savedAnalyses.length > 0) out.savedAnalyses = s.savedAnalyses;
  const outputItems = useOutputDoc.getState().items;
  if (outputItems.length > 0) out.outputItems = outputItems;
  if (s.splitFile) out.splitFile = s.splitFile;
  return out;
}

function isStamp(value: unknown): value is ResultStamp {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as ResultStamp).dataVersion === "number" &&
    typeof (value as ResultStamp).paramsKey === "string"
  );
}

function rebaseEntry(entry: unknown, savedDataVersion: number): unknown {
  if (typeof entry !== "object" || entry === null) return entry;
  const stamp = (entry as { stamp?: unknown }).stamp;
  if (!isStamp(stamp)) return entry;
  return {
    ...entry,
    stamp: {
      ...stamp,
      dataVersion: stamp.dataVersion === savedDataVersion ? 0 : -1,
      // The project reopens as a new session: a stamp that was current in the
      // saved one is current in this one (the data version decides that).
      sessionId: useStore.getState().session?.session_id ?? null,
    },
  };
}

/**
 * Hydrate the store from a loaded project's ui_state. Call AFTER setSession:
 * setSession clears panelCache for the new session id, and this puts the
 * restored state in its place. Ignores anything that does not look like a
 * ui_state (older files, foreign hands in the JSON).
 */
export function applyUiState(raw: unknown): void {
  if (typeof raw !== "object" || raw === null) return;
  const ui = raw as Partial<ProjectUiState>;
  if (typeof ui.dataVersion !== "number" || typeof ui.panelCache !== "object" || ui.panelCache === null) {
    return;
  }
  const savedDataVersion = ui.dataVersion;
  const panelCache = Object.fromEntries(
    Object.entries(ui.panelCache).map(([panel, entry]) => [
      panel,
      rebaseEntry(entry, savedDataVersion),
    ]),
  );
  // Named analyses: same rebase applies to the stamp inside each snapshot,
  // for the same reason. Entries that do not look like a SavedAnalysis are
  // dropped rather than restored broken.
  const savedAnalyses = Array.isArray(ui.savedAnalyses)
    ? ui.savedAnalyses
        .filter(
          (a): a is SavedAnalysis =>
            typeof a === "object" && a !== null &&
            typeof a.id === "string" && typeof a.name === "string" &&
            typeof a.panel === "string" && typeof a.tab === "string",
        )
        .map((a) => ({ ...a, snapshot: rebaseEntry(a.snapshot, savedDataVersion) }))
    : [];
  // Sanitised again on the way in: a project file can come from anyone.
  useOutputDoc.getState().replaceAll(restoreOutputItems(ui.outputItems));
  const split = ui.splitFile;
  const splitFile = split && typeof split.column === "string" && Array.isArray(split.levels)
    ? {
        column: split.column,
        levels: split.levels.filter((l) => typeof l?.level === "string").map((l) => ({ level: l.level, n: Number(l.n) || 0 })),
        level: typeof split.level === "string" ? split.level : null,
      }
    : null;
  useStore.setState({
    panelCache,
    savedAnalyses,
    splitFile,
    ...(ui.table1Result !== undefined ? { table1Result: ui.table1Result as never } : {}),
    ...(typeof ui.activeTab === "string" && ui.activeTab ? { activeTab: ui.activeTab } : {}),
  });
}
