/**
 * Centralized plot style system.
 * All Plotly charts should merge `usePlotLayout()` into their layout prop.
 */
import { useStore, paletteOf } from "./store";

/** Returns a base Plotly layout object that reflects the current global theme. */
export function usePlotLayout(overrides?: Record<string, unknown>): Record<string, unknown> {
  const theme   = useStore((s) => s.plotTheme);
  const showGrid = useStore((s) => s.showGrid);
  const gc = showGrid ? "#e5e7eb" : "transparent";

  return {
    paper_bgcolor: "transparent",
    plot_bgcolor:  theme.plotBg,
    font: { family: theme.fontFamily, color: "#374151", size: theme.fontSize },
    colorway: paletteOf(theme),
    xaxis: { gridcolor: gc, zeroline: false },
    yaxis: { gridcolor: gc, zeroline: false },
    ...overrides,
  };
}

/** Returns the primary color palette array for the current theme. */
export function usePalette(): string[] {
  const theme = useStore((s) => s.plotTheme);
  return paletteOf(theme);
}

/**
 * Repaint whatever traces carry a pinned name.
 *
 * Applied once to the finished trace list rather than threaded through every
 * chart builder: a chart type added later is covered without being told about
 * pinning, and there is one place to read when a colour comes out wrong.
 * Alpha is preserved — a violin's translucent fill stays translucent.
 */
export function applySeriesPins<T extends { name?: unknown }>(
  traces: T[] | null,
  pins: Record<string, string>,
): T[] | null {
  if (!traces || !pins || Object.keys(pins).length === 0) return traces;

  const repaint = (value: unknown, colour: string): unknown => {
    if (typeof value !== "string") return colour;
    // "#rrggbbaa" keeps its alpha; anything else takes the pinned colour whole.
    return /^#[0-9a-f]{8}$/i.test(value) ? colour + value.slice(7) : colour;
  };

  return traces.map((t) => {
    const colour = pins[String((t as { name?: unknown }).name)];
    if (!colour) return t;
    const trace = t as Record<string, unknown>;
    const marker = trace.marker as Record<string, unknown> | undefined;
    const line = trace.line as Record<string, unknown> | undefined;
    const markerLine = marker?.line as Record<string, unknown> | undefined;
    return {
      ...trace,
      ...(marker ? {
        marker: {
          ...marker,
          color: repaint(marker.color, colour),
          ...(markerLine ? { line: { ...markerLine, color: repaint(markerLine.color, colour) } } : {}),
        },
      } : {}),
      ...(line ? { line: { ...line, color: repaint(line.color, colour) } } : {}),
      ...(trace.fillcolor !== undefined ? { fillcolor: repaint(trace.fillcolor, colour) } : {}),
    } as T;
  });
}

/** Returns default marker / line props for the current theme. */
export function useTraceDefaults() {
  const theme = useStore((s) => s.plotTheme);
  return {
    lineWidth: theme.lineWidth,
    markerSize: theme.markerSize,
    markerOpacity: theme.markerOpacity,
  };
}
