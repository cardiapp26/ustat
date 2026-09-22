/**
 * Whether the result a component sits inside is out of date.
 *
 * Staleness used to reach export only through `ResultExporter`'s `stale` prop,
 * so every other way a number leaves the app stayed open on an out-of-date
 * result: plot downloads, the Plotly modebar camera, Word/HTML tables, the
 * Copy buttons, the Forest Builder hand-off. Threading a prop through each of
 * those, in every panel, is how the gap opened in the first place.
 *
 * Instead a panel wraps its result in `<StaleGuard>` once, and every export
 * control reads `useStaleGuard()`. A control that is not inside a guard sees
 * `{ stale: false }`, which is what it saw before.
 */
import { createContext, useContext } from "react";

export interface StaleGuardState {
  stale: boolean;
  /** Why, for tooltips: "the data changed and the case filter changed". */
  reason?: string;
}

export const StaleGuardContext = createContext<StaleGuardState>({ stale: false });

export function useStaleGuard(): StaleGuardState {
  return useContext(StaleGuardContext);
}

/** Tooltip for an export control disabled by a stale result. */
export function staleExportTitle(reason?: string): string {
  return reason
    ? `Recompute first: this result predates ${reason}`
    : "Recompute first: this result is out of date";
}
