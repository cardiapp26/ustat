/**
 * What a displayed result was computed from, and whether that is still true.
 *
 * A result is not a number, it is a number *plus the conditions it came from*.
 * The panel cache used to keep only the number: edit a cell, come back to the
 * Models tab, and the previous fit was still on screen with nothing to say it
 * predated the edit. Whoever read it would attribute it to the data in front
 * of them, which is exactly the misreading a statistics tool must not enable.
 *
 * So every cached result carries a stamp of the four things that can change
 * underneath it, and a result whose stamp no longer matches the present is
 * marked stale rather than silently reused:
 *
 *   data      the monotonic `dataVersion`, bumped on every cell edit, column
 *             add/remove/recode, paste, undo and redo
 *   filter    the active case filter, keyed by its conditions and counts
 *   params    the analysis settings the run was launched with
 *   engine    which engine computed it, and which build of the app
 *
 * "Marked stale" and not "discarded": the number is still the honest answer to
 * the question that was asked, and throwing it away the moment a filter moves
 * would lose work. It stops being exportable-as-current, which is the part
 * that matters.
 */
import type { EngineKind } from "./engine/types";
import { latest as latestProvenance, type Provenance } from "./engine/provenance";
import type { CaseFilter } from "../store";

declare const __APP_VERSION__: string;

/** The build that computed a result. Absent outside a Vite build (tests). */
export function appVersion(): string {
  return typeof __APP_VERSION__ === "string" ? __APP_VERSION__ : "dev";
}

export interface ResultStamp {
  dataVersion: number;
  filterKey: string;
  paramsKey: string;
  engine: EngineKind;
  engineVersion: string;
  /** Epoch ms of the run, for "computed 4 minutes ago" style copy. */
  at: number;
  /**
   * What actually computed it: engine, runtime and library versions, as
   * reported by whatever produced the answer.
   *
   * Separate from `engine` above, which is the session's *choice*. The two
   * disagree whenever an R session runs an analysis R does not implement yet,
   * and that disagreement is precisely what a reader has to be shown rather
   * than left to discover when the numbers will not reproduce.
   */
  provenance?: Provenance | null;
}

export type StaleReason = "data" | "filter" | "params" | "engine";

/** Sentence fragments, so a banner can join them into one readable clause. */
export const STALE_LABELS: Record<StaleReason, string> = {
  data: "the data changed",
  filter: "the case filter changed",
  params: "the analysis settings changed",
  engine: "the statistics engine changed",
};

/**
 * Deterministic JSON: object keys sorted, so `{a,b}` and `{b,a}` key alike.
 * `JSON.stringify` alone preserves insertion order, which would report a
 * settings change every time a panel happened to rebuild its params object in
 * a different order.
 */
export function stableStringify(value: unknown): string {
  if (value === undefined) return "null";
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj).filter((k) => obj[k] !== undefined).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`).join(",")}}`;
}

/**
 * The filter's identity for staleness purposes.
 *
 * `excludedRows` is deliberately left out: it is a view concern (which preview
 * rows get a strike-through), it moves when the preview window scrolls, and
 * treating it as part of the filter would flag every result stale for a
 * reason no reader would recognise.
 */
export function filterKey(filter: CaseFilter | null | undefined): string {
  if (!filter || !filter.conditions?.length) return "none";
  return stableStringify({
    conditions: filter.conditions,
    selected: filter.selected,
    total: filter.total,
  });
}

export interface StampInputs {
  dataVersion: number;
  caseFilter: CaseFilter | null;
  engine: EngineKind;
  params: unknown;
}

export interface StampInputsWithProvenance extends StampInputs {
  /** Defaults to the most recently recorded run, which is this one. */
  provenance?: Provenance | null;
}

export function makeStamp(inputs: StampInputsWithProvenance): ResultStamp {
  return {
    dataVersion: inputs.dataVersion,
    filterKey: filterKey(inputs.caseFilter),
    paramsKey: stableStringify(inputs.params),
    engine: inputs.engine,
    engineVersion: appVersion(),
    at: Date.now(),
    provenance: inputs.provenance !== undefined ? inputs.provenance : latestProvenance(),
  };
}

export interface StaleOptions {
  /** False for analyses that read no dataset (a-priori power, say). */
  dependsOnData?: boolean;
}

/** Which of the four conditions have moved since `stamp` was taken. */
export function staleReasons(
  stamp: ResultStamp | null | undefined,
  current: ResultStamp,
  opts: StaleOptions = {},
): StaleReason[] {
  if (!stamp) return [];
  const dependsOnData = opts.dependsOnData !== false;
  const out: StaleReason[] = [];
  if (dependsOnData && stamp.dataVersion !== current.dataVersion) out.push("data");
  if (dependsOnData && stamp.filterKey !== current.filterKey) out.push("filter");
  if (stamp.paramsKey !== current.paramsKey) out.push("params");
  if (stamp.engine !== current.engine || stamp.engineVersion !== current.engineVersion) {
    out.push("engine");
  }
  return out;
}

/** "the data changed and the case filter changed" */
export function describeStale(reasons: StaleReason[]): string {
  const parts = reasons.map((r) => STALE_LABELS[r]);
  if (parts.length === 0) return "";
  if (parts.length === 1) return parts[0];
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}
