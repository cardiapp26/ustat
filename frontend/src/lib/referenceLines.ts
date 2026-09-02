/**
 * Reference lines — ggplot2's geom_hline / geom_vline, as Plotly layout
 * shapes with a label annotation.
 *
 * A clinical figure often carries a threshold: LDL 100, BMI 30, a p-value
 * cut-off on a log axis. The line is drawn in data coordinates on one axis
 * and spans the whole plot on the other, so it survives zooming, a log
 * scale, and a category axis on the other side.
 */

export interface RefLine {
  /** Which axis carries the value: "y" draws a horizontal line, "x" a vertical one. */
  axis: "x" | "y";
  value: number;
  label: string;
}

export interface Overlay {
  shapes: Record<string, unknown>[];
  annotations: Record<string, unknown>[];
}

export const REF_LINE_COLOR = "#b91c1c";

/** Parse a user-typed value; blank or non-numeric yields null so the line is skipped. */
export function parseRefValue(raw: string): number | null {
  const t = raw.trim().replace(",", ".");
  if (!t) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/** Shapes and labels for the given lines. Lines without a finite value are dropped. */
export function referenceLineOverlay(lines: readonly RefLine[]): Overlay {
  const shapes: Record<string, unknown>[] = [];
  const annotations: Record<string, unknown>[] = [];
  for (const line of lines) {
    if (!Number.isFinite(line.value)) continue;
    const horizontal = line.axis === "y";
    shapes.push({
      type: "line",
      // Data coordinates on the line's own axis; paper (0-1) across the other,
      // so the line spans the panel whatever the other axis holds.
      xref: horizontal ? "paper" : "x",
      yref: horizontal ? "y" : "paper",
      x0: horizontal ? 0 : line.value,
      x1: horizontal ? 1 : line.value,
      y0: horizontal ? line.value : 0,
      y1: horizontal ? line.value : 1,
      line: { color: REF_LINE_COLOR, width: 1.5, dash: "dash" },
      layer: "above",
    });
    const text = line.label.trim() || String(line.value);
    annotations.push({
      xref: horizontal ? "paper" : "x",
      yref: horizontal ? "y" : "paper",
      x: horizontal ? 1 : line.value,
      y: horizontal ? line.value : 1,
      // Tucked into the top-right end of the line, inside the panel.
      xanchor: horizontal ? "right" : "left",
      yanchor: "bottom",
      xshift: horizontal ? 0 : 4,
      text,
      showarrow: false,
      font: { size: 10, color: REF_LINE_COLOR },
    });
  }
  return { shapes, annotations };
}
