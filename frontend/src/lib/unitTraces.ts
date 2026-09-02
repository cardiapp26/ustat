/**
 * Trace builders shared by the Charts panel and the Summary tab: the icon
 * array and the per-subject waterfall. Both draw from a backend response
 * (/charts/waffle, /charts/waterfall) and take the palette and value labels
 * from the caller, so the two panels cannot drift apart in how a hundred
 * squares or a ranked bar is drawn.
 */
import type { PlotData } from "./plotTypes";
import { labelFor } from "./valueLabels";

export interface WaffleLevel { label: string; count: number; percent: number; cells: number }

export interface WaterfallRow { rank: number; value: number; group?: string | null; label?: string }

/** The icon array: `units` squares in a grid ten wide, filled level by level
 *  from the top-left, one trace per level so the legend names each. Square
 *  markers on an axis-free plot rather than a heatmap, so each unit stays a
 *  countable thing with its own hover. */
export function waffleTraces(
  d: Record<string, unknown>,
  C: string[],
  labels: Record<string, string>,
): PlotData[] {
  const levels = (d.levels as WaffleLevel[]) ?? [];
  const units = Number(d.units) || 100;
  const cols = 10;
  const rows = Math.ceil(units / cols);
  let next = 0;
  return levels.map((lv, i) => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let k = 0; k < lv.cells; k++) {
      const cell = next + k;
      xs.push(cell % cols);
      // Row 0 at the top, as a reader fills a grid.
      ys.push(rows - 1 - Math.floor(cell / cols));
    }
    next += lv.cells;
    const name = labelFor(labels, lv.label, lv.label);
    return {
      type: "scatter",
      mode: "markers",
      x: xs,
      y: ys,
      name,
      marker: { symbol: "square", size: 18, color: C[i % C.length], line: { color: "#ffffff", width: 1.5 } },
      hovertemplate: `${name}<br>${lv.cells} in ${units} · ${lv.count} of ${String(d.n)} (${lv.percent.toFixed(1)}%)<extra></extra>`,
    } as PlotData;
  });
}

/** One bar per subject at its rank, coloured by group when there is one.
 *  Colour is carried per bar rather than per trace so the bars keep their
 *  sorted order on a single category axis; a legend entry per group is added
 *  as an empty trace, since a per-bar colour array has no legend of its own. */
export function waterfallTraces(
  d: Record<string, unknown>,
  C: string[],
  labels: Record<string, string>,
): PlotData[] {
  const rows = (d.rows as WaterfallRow[]) ?? [];
  const groups = [...new Set(rows.map((r) => r.group ?? null))].filter((g): g is string => g !== null);
  const colourOf = (g: string | null | undefined) => (g && groups.length ? C[groups.indexOf(g) % C.length] : C[0]);
  const bars: PlotData = {
    type: "bar",
    x: rows.map((r) => String(r.rank)),
    y: rows.map((r) => r.value),
    marker: { color: rows.map((r) => colourOf(r.group)) },
    text: rows.map((r) => r.label ?? `#${r.rank}`),
    textposition: "none",
    showlegend: false,
    hovertemplate: "%{text}<br>%{y:.1f}<extra></extra>",
  };
  const legend: PlotData[] = groups.map((g, i) => ({
    type: "bar",
    x: [rows[0] ? String(rows[0].rank) : "1"],
    y: [0],
    name: labelFor(labels, g, g),
    marker: { color: C[i % C.length] },
    hoverinfo: "skip",
  } as PlotData));
  return [bars, ...legend];
}

