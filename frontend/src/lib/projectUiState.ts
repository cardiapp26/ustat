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
import { useStore } from "../store";
import type { ResultStamp } from "./resultStamp";

export interface ProjectUiState {
  /** The saving session's data counter, so restore can tell fresh from stale. */
  dataVersion: number;
  activeTab?: string;
  panelCache: Record<string, unknown>;
  table1Result?: unknown;
}

/** Snapshot the live store's panel state for ui/state.json. */
export function collectUiState(): ProjectUiState {
  const s = useStore.getState();
  const out: ProjectUiState = {
    dataVersion: s.dataVersion,
    activeTab: s.activeTab,
    panelCache: s.panelCache,
  };
  if (s.table1Result != null) out.table1Result = s.table1Result;
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
  useStore.setState({
    panelCache,
    ...(ui.table1Result !== undefined ? { table1Result: ui.table1Result as never } : {}),
    ...(typeof ui.activeTab === "string" && ui.activeTab ? { activeTab: ui.activeTab } : {}),
  });
}
