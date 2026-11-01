import type { MutableRefObject } from "react";
import TitledPlot from "./TitledPlot";
import type { PlotCaptureHandle, PlotData } from "../lib/plotTypes";
import { formatHorizon } from "../lib/rcsKnots";

/** Cumulative incidence at a horizon along the spline grid (Austin 2025). */
export interface AbsoluteRiskData {
  horizon: number;
  horizon_is_default: boolean;
  n_at_risk: number;
  n_events_by_horizon: number;
  x_values: number[];
  risk: number[];
  ci_low: number[];
  ci_high: number[];
}

export type RCSView = "hr" | "risk";

interface ViewToggleProps {
  view: RCSView;
  onChange: (view: RCSView) => void;
  horizon: number;
  timeLabel: string;
}

/** Switch between the relative hazard ratio and the absolute risk at t0. */
export function RCSViewToggle({ view, onChange, horizon, timeLabel }: ViewToggleProps) {
  const options: { key: RCSView; label: string }[] = [
    { key: "hr", label: "Hazard ratio" },
    { key: "risk", label: `Absolute risk at ${timeLabel} = ${formatHorizon(horizon)}` },
  ];
  return (
    <div className="flex items-center gap-1 text-[11px]" role="group" aria-label="Curve scale">
      {options.map((o) => (
        <button key={o.key} onClick={() => onChange(o.key)} aria-pressed={view === o.key}
          className={`px-2 py-0.5 rounded border transition-colors ${view === o.key ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-600 hover:bg-gray-50"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

interface PlotProps {
  data: AbsoluteRiskData;
  predictor: string;
  timeLabel: string;
  eventLabel: string;
  knots: number[];
  layout: Record<string, unknown>;
  showGrid: boolean;
  plotRefOut: MutableRefObject<PlotCaptureHandle | null>;
}

const pct = (v: number[]): number[] => v.map((x) => x * 100);

export function AbsoluteRiskPlot({ data, predictor, timeLabel, eventLabel, knots, layout, showGrid, plotRefOut }: PlotProps) {
  const xs = data.x_values;
  const risk = pct(data.risk);
  const knotY = knots.map((k) => {
    const idx = xs.reduce((best, x, i) => (Math.abs(x - k) < Math.abs(xs[best] - k) ? i : best), 0);
    return risk[idx];
  });
  const traces: PlotData[] = [
    {
      type: "scatter", x: [...xs, ...xs.slice().reverse()],
      y: [...pct(data.ci_high), ...pct(data.ci_low).reverse()],
      fill: "toself", fillcolor: "rgba(13,148,136,0.14)", line: { color: "transparent" },
      hoverinfo: "skip", name: "95% CI",
    },
    {
      type: "scatter", mode: "lines", x: xs, y: risk,
      line: { color: "#0d9488", width: 2.5 }, name: "Absolute risk",
      hovertemplate: `${predictor}: %{x:.2f}<br>Risk: %{y:.1f}%<extra></extra>`,
    },
    {
      type: "scatter", mode: "markers", x: knots, y: knotY,
      marker: { color: "#0d9488", size: 7, line: { color: "#fff", width: 1.5 } },
      name: "Knots", showlegend: false,
      hovertemplate: `Knot: %{x:.2f}<br>Risk: %{y:.1f}%<extra></extra>`,
    },
  ];
  const yTitle = `Risk of ${eventLabel} by ${timeLabel} = ${formatHorizon(data.horizon)} (%)`;
  return (
    <TitledPlot
      plotRefOut={plotRefOut}
      storageKey={`rcs-risk:${predictor}:${eventLabel}:t${data.horizon}`}
      defaultTitle=""
      defaultSubtitle={`Cumulative incidence at ${timeLabel} = ${formatHorizon(data.horizon)}${data.horizon_is_default ? " (median follow-up)" : ""} · other covariates at their mean · at risk at horizon: ${data.n_at_risk}, events by horizon: ${data.n_events_by_horizon} · dots = knot locations`}
      defaultXAxis={predictor}
      defaultYAxis={yTitle}
      data={traces}
      layout={{
        ...layout, autosize: true, height: 440,
        xaxis: { gridcolor: "#e5e7eb", showgrid: showGrid, title: { text: predictor }, zeroline: false },
        yaxis: { gridcolor: "#e5e7eb", showgrid: showGrid, title: { text: yTitle }, zeroline: false, rangemode: "tozero", ticksuffix: "%" },
        legend: { font: { size: 11, color: "#374151" }, x: 0.02, y: 0.98, xanchor: "left", yanchor: "top", bgcolor: "rgba(255,255,255,0.85)", bordercolor: "#e5e7eb", borderwidth: 1 },
        margin: { t: 20, r: 20, b: 50, l: 70 },
      }}
      style={{ width: "100%", height: 520 }}
      config={{ responsive: true, displaylogo: false,
        toImageButtonOptions: { format: "png", filename: `RCS_absolute_risk_${predictor}`, width: 1200, height: 600 },
        modeBarButtonsToRemove: ["select2d", "lasso2d"] }}
    />
  );
}
