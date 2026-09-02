/**
 * Centralized plot style system.
 * All Plotly charts should merge `usePlotLayout()` into their layout prop.
 */
import { useStore, paletteOf, type PlotTheme } from "./store";
import { legendLayout, presetAxis } from "./lib/plotPresets";

/** The base layout for a theme, as a pure function so it can be tested
 *  without a store. `usePlotLayout` reads the store and calls this. */
export function baseLayout(
  theme: PlotTheme,
  showGrid: boolean,
  overrides?: Record<string, unknown>,
): Record<string, unknown> {
  const axis = presetAxis(theme.preset ?? "minimal", showGrid);
  return {
    paper_bgcolor: "transparent",
    plot_bgcolor:  theme.plotBg,
    font: { family: theme.fontFamily, color: "#374151", size: theme.fontSize },
    colorway: paletteOf(theme),
    xaxis: { ...axis },
    yaxis: { ...axis },
    ...legendLayout(theme.legendPosition ?? "auto"),
    ...overrides,
  };
}

/** Returns a base Plotly layout object that reflects the current global theme. */
export function usePlotLayout(overrides?: Record<string, unknown>): Record<string, unknown> {
  const theme   = useStore((s) => s.plotTheme);
  const showGrid = useStore((s) => s.showGrid);
  return baseLayout(theme, showGrid, overrides);
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

/** "#rrggbb" → [h, s, l] in degrees / fractions; null for anything else. */
function hexToHsl(hex: string): [number, number, number] | null {
  const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  const r = ((n >> 16) & 255) / 255, g = ((n >> 8) & 255) / 255, b = (n & 255) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = 0;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) * 60;
  else if (max === g) h = ((b - r) / d + 2) * 60;
  else h = ((r - g) / d + 4) * 60;
  return [h, s, l];
}

function hslToHex(h: number, s: number, l: number): string {
  const f = (k: number) => {
    const a = s * Math.min(l, 1 - l);
    const v = l - a * Math.max(-1, Math.min(Math.min((k + h / 30) % 12 - 3, 9 - (k + h / 30) % 12), 1));
    return Math.round(v * 255).toString(16).padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

/**
 * A lightness ladder of one hue, light to dark, for groups that are ORDERED.
 *
 * Tertile 1 / 2 / 3 or stage I / II / III drawn in three unrelated hues make
 * the reader look up which colour is which; drawn in one hue that darkens
 * with the level, the order is visible without the legend. Built from the
 * palette's own first colour, so the ladder follows the chosen palette.
 * Fewer than two levels get the base colour back, since a ladder of one has
 * nothing to order.
 */
export function ordinalLadder(base: string, n: number): string[] {
  const hsl = hexToHsl(base);
  if (!hsl || n < 2) return [base];
  const [h, s0] = hsl;
  // A neutral base (grey, or a warm grey) keeps its faint saturation; a
  // colour keeps at least a firm one so the light end does not fade to white.
  const s = s0 < 0.1 ? s0 : Math.max(0.45, s0);
  const light = 0.82, dark = 0.22;
  return Array.from({ length: n }, (_, i) => hslToHex(h, s, light - (light - dark) * (i / (n - 1))));
}

const QUIET_GREY = "#b0afa9";

/**
 * Draw one named group in an accent and every other named series in one
 * quiet grey — the figure where the intervention arm is the only thing in
 * colour. Applied after the pins and overriding them: a highlight is the
 * decision to quiet everything else, and a pinned red on a control arm
 * would defeat it. Unnamed traces (bands, rugs, reference lines) are left
 * alone, as is any trace whose colour is an array — a value ramp or a
 * heatmap is data, not a series colour. Alpha suffixes are kept, as in
 * applySeriesPins.
 */
export function applyHighlight<T extends { name?: unknown }>(
  traces: T[] | null,
  group: string,
  accent: string,
): T[] | null {
  if (!traces || !group.trim()) return traces;
  const target = group.trim();
  const named = traces.filter((t) => {
    const marker = (t as { marker?: { color?: unknown } }).marker;
    return typeof t.name === "string" && t.name !== "" && !Array.isArray(marker?.color);
  });
  if (!named.some((t) => String(t.name) === target)) return traces;
  const pins: Record<string, string> = {};
  for (const t of named) {
    const name = String(t.name);
    pins[name] = name === target ? accent : QUIET_GREY;
  }
  return applySeriesPins(traces, pins);
}
