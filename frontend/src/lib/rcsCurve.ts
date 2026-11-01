/** Curve-level helpers for RCS plots: distribution layer and CI-excludes-reference regions. */
import type { PlotData } from "./plotTypes";

export interface XDistribution {
  edges: number[];
  counts: number[];
  n_outside: number;
}

/** Bars sit on a hidden secondary axis scaled so they fill the bottom quarter or so. */
const DISTRIBUTION_HEADROOM = 4;

export function distributionTrace(dist: XDistribution): PlotData {
  const centers = dist.counts.map((_, i) => (dist.edges[i] + dist.edges[i + 1]) / 2);
  const widths = dist.counts.map((_, i) => dist.edges[i + 1] - dist.edges[i]);
  return {
    type: "bar", x: centers, y: dist.counts, width: widths, yaxis: "y2",
    marker: { color: "rgba(107,114,128,0.22)", line: { width: 0 } },
    name: "Distribution", showlegend: false,
    hovertemplate: "%{x:.2f}: n = %{y}<extra></extra>",
  } as PlotData;
}

export function distributionAxis(dist: XDistribution): Record<string, unknown> {
  const max = Math.max(1, ...dist.counts);
  return {
    overlaying: "y", side: "right", range: [0, max * DISTRIBUTION_HEADROOM],
    showgrid: false, showticklabels: false, zeroline: false, fixedrange: true,
  };
}

export type RegionSide = "above" | "below";

export interface ExcludedRegion {
  from: number;
  to: number;
  side: RegionSide;
}

function sideAt(lo: number, hi: number, ref: number): RegionSide | null {
  if (lo > ref) return "above";
  if (hi < ref) return "below";
  return null;
}

/** x where the band edge that defines `side` crosses `ref` between points a and b. */
function crossing(x: number[], lo: number[], hi: number[], a: number, b: number, side: RegionSide, ref: number): number {
  const y = side === "above" ? lo : hi;
  const dy = y[b] - y[a];
  if (dy === 0) return (x[a] + x[b]) / 2;
  const t = Math.min(1, Math.max(0, (ref - y[a]) / dy));
  return x[a] + t * (x[b] - x[a]);
}

/**
 * Ranges of x where the 95% band lies wholly above or below the reference
 * (HR/OR = 1, or 0 for a mean difference). Boundaries are interpolated
 * between grid points. These ranges depend on the chosen reference value.
 */
export function excludedRegions(x: number[], lo: number[], hi: number[], ref: number): ExcludedRegion[] {
  const n = Math.min(x.length, lo.length, hi.length);
  const regions: ExcludedRegion[] = [];
  let current: { side: RegionSide; from: number } | null = null;
  let prevSide: RegionSide | null = null;
  for (let i = 0; i < n; i++) {
    const side = sideAt(lo[i], hi[i], ref);
    if (side !== prevSide) {
      if (current && prevSide) {
        regions.push({ side: current.side, from: current.from, to: crossing(x, lo, hi, i - 1, i, prevSide, ref) });
        current = null;
      }
      if (side) {
        current = { side, from: i === 0 ? x[0] : crossing(x, lo, hi, i - 1, i, side, ref) };
      }
    }
    prevSide = side;
  }
  if (current) regions.push({ side: current.side, from: current.from, to: x[n - 1] });
  return regions;
}

const fmt = (v: number): string => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));

/** "38.1 to 45.0 (lower), 70.3 to 85.6 (higher)" */
export function describeRegions(regions: ExcludedRegion[]): string {
  return regions
    .map((r) => `${fmt(r.from)} to ${fmt(r.to)} (${r.side === "above" ? "higher" : "lower"})`)
    .join(", ");
}
