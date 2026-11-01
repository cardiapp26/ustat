/** Harrell knot-count constants, the knot budget and RCS display helpers. */

/** Knot counts the backend accepts (Harrell RMS Table 2.3). */
export const SUPPORTED_KNOTS = [3, 4, 5, 6, 7] as const;

/** Harrell RMS Table 2.3 default knot percentiles, as the backend places them. */
const HARRELL_PERCENTILES: Record<number, readonly number[]> = {
  3: [10, 50, 90],
  4: [5, 35, 65, 95],
  5: [5, 27.5, 50, 72.5, 95],
  6: [5, 23, 41, 59, 77, 95],
  7: [2.5, 18.33, 34.17, 50, 65.83, 81.67, 97.5],
};

export function harrellPercentiles(k: number): readonly number[] {
  return HARRELL_PERCENTILES[k] ?? [];
}

/** Event count the knot-budget hint is worked out for. */
const EPV_REFERENCE_EVENTS = 17;

export interface KnotBudgetRow {
  k: number;
  params: number;
  epv: number;
  label: "safe" | "borderline" | "overfit risk";
}

/** Parameters and events per parameter for each knot count at the reference event count. */
export function knotBudget(events: number = EPV_REFERENCE_EVENTS): KnotBudgetRow[] {
  return SUPPORTED_KNOTS.map((k) => {
    const params = k - 1;
    const epv = events / params;
    const label = epv >= 8 ? "safe" : epv >= 5 ? "borderline" : "overfit risk";
    return { k, params, epv, label };
  });
}

export const KNOT_BUDGET_EVENTS = EPV_REFERENCE_EVENTS;

/** Time horizon for labels: integers as is, otherwise 4 significant digits. */
export function formatHorizon(t: number): string {
  return Number.isInteger(t) ? String(t) : t.toPrecision(4).replace(/\.?0+$/, "");
}
