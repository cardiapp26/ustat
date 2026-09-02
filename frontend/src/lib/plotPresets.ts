/**
 * ggplot2-style theme presets and legend placement, as Plotly layout fragments.
 *
 * A palette says which colours the series take; a preset says everything
 * else about the frame — panel background, grid, axis lines, ticks. ggplot2
 * ships these as `theme_bw()`, `theme_classic()` and so on, and a journal
 * that asks for "no grid, axis lines only" is asking for `theme_classic`.
 * Kept as pure data with no store import so the layout hook, the theme bar
 * and the tests all read one definition.
 */

export type ThemePreset = "minimal" | "gray" | "bw" | "classic" | "light" | "dark";

/** Where the legend sits. "auto" leaves Plotly's own placement, which a
 *  panel may already have overridden for its own reasons. */
export type LegendPosition = "auto" | "right" | "bottom" | "inside" | "none";

export interface PresetSpec {
  /** Name as a ggplot2 user knows it. */
  label: string;
  description: string;
  plotBg: string;
  /** Grid colour when the grid is on. */
  gridColor: string;
  /** Whether the preset draws the grid by default; the global grid toggle
   *  still overrides it afterwards. */
  gridDefault: boolean;
  /** "none": no axis lines. "lb": left and bottom only. "box": all four. */
  axisLine: "none" | "lb" | "box";
  lineColor: string;
  ticks: "" | "outside";
}

export const THEME_PRESETS: Record<ThemePreset, PresetSpec> = {
  minimal: {
    label: "theme_minimal",
    description: "White panel, faint grid, no axis lines.",
    plotBg: "#ffffff",
    gridColor: "#e5e7eb",
    gridDefault: true,
    axisLine: "none",
    lineColor: "#9ca3af",
    ticks: "",
  },
  gray: {
    label: "theme_gray",
    description: "ggplot2's default: grey panel with a white grid.",
    plotBg: "#ebebeb",
    gridColor: "#ffffff",
    gridDefault: true,
    axisLine: "none",
    lineColor: "#333333",
    ticks: "outside",
  },
  bw: {
    label: "theme_bw",
    description: "White panel inside a black frame, light grid.",
    plotBg: "#ffffff",
    gridColor: "#e5e7eb",
    gridDefault: true,
    axisLine: "box",
    lineColor: "#333333",
    ticks: "outside",
  },
  classic: {
    label: "theme_classic",
    description: "No grid; left and bottom axis lines only. The usual journal request.",
    plotBg: "#ffffff",
    gridColor: "#e5e7eb",
    gridDefault: false,
    axisLine: "lb",
    lineColor: "#333333",
    ticks: "outside",
  },
  light: {
    label: "theme_light",
    description: "Light grey frame and grid.",
    plotBg: "#ffffff",
    gridColor: "#e5e7eb",
    gridDefault: true,
    axisLine: "box",
    lineColor: "#b0b0b0",
    ticks: "outside",
  },
  dark: {
    label: "theme_dark",
    description: "Dark panel for slides; labels stay in the white margin.",
    plotBg: "#1f2937",
    gridColor: "#4b5563",
    gridDefault: true,
    axisLine: "none",
    lineColor: "#6b7280",
    ticks: "",
  },
};

export const PRESET_ORDER: ThemePreset[] = ["minimal", "gray", "bw", "classic", "light", "dark"];

/** Plotly properties shared by both axes under a preset. Callers spread this
 *  under their own axis settings, so a log scale or a category axis survives. */
export function presetAxis(preset: ThemePreset, showGrid: boolean): Record<string, unknown> {
  const spec = THEME_PRESETS[preset] ?? THEME_PRESETS.minimal;
  return {
    gridcolor: showGrid ? spec.gridColor : "transparent",
    zeroline: false,
    showline: spec.axisLine !== "none",
    linecolor: spec.lineColor,
    linewidth: 1,
    // Plotly's "mirror" draws the opposite side too — that is the box.
    mirror: spec.axisLine === "box",
    ticks: spec.ticks,
    tickcolor: spec.lineColor,
  };
}

export const LEGEND_POSITION_LABELS: Record<LegendPosition, string> = {
  auto: "Default",
  right: "Right of the plot",
  bottom: "Below the plot",
  inside: "Inside, top-right corner",
  none: "Hidden",
};

/** Layout fragment placing the legend. Empty for "auto". */
export function legendLayout(position: LegendPosition): Record<string, unknown> {
  switch (position) {
    case "right":
      return { legend: { orientation: "v", x: 1.02, y: 1, xanchor: "left", yanchor: "top" } };
    case "bottom":
      // Far enough under the plot area to clear the x-axis title; Plotly
      // widens the bottom margin for a legend placed outside the plot.
      return { legend: { orientation: "h", x: 0.5, y: -0.25, xanchor: "center", yanchor: "top" } };
    case "inside":
      return {
        legend: {
          x: 0.98, y: 0.98, xanchor: "right", yanchor: "top",
          bgcolor: "rgba(255,255,255,0.75)", bordercolor: "#e5e7eb", borderwidth: 1,
        },
      };
    case "none":
      return { showlegend: false };
    default:
      return {};
  }
}
