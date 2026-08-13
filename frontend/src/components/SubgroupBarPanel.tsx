import { useState, useRef, useEffect } from "react";
import Plot from "../PlotComponent";
import { useStore, isCategoricalKind, type Session } from "../store";
import { usePlotLayout, usePalette } from "../plotStyle";
import type { Data, Layout } from "plotly.js";
import { runSubgroupBar, getUniqueValues } from "../api";
import type { PlotData, PlotLayout, PlotCaptureHandle } from "../lib/plotTypes";
import { labelFor } from "../lib/valueLabels";
import PlotExporter from "./PlotExporter";

export default function SubgroupBarPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <SubgroupBarPanelBody session={session} />;
}

function SubgroupBarPanelBody({ session }: { session: Session }) {
  const layout = usePlotLayout();
  const pal = usePalette();
  const showGrid = useStore((s) => s.showGrid);

  const catCols = session.columns.filter((c) => isCategoricalKind(c.kind)).map((c) => c.name);
  const allCols = session.columns.map((c) => c.name);

  // States
  const [yCol, setYCol] = useState(allCols[0] ?? "");
  const [yMode, setYMode] = useState("mean"); // "mean" | "percentage"
  const [targetValue, setTargetValue] = useState("");
  const [subgroupCol, setSubgroupCol] = useState(catCols[0] ?? "");
  const [xaxisCol, setXaxisCol] = useState(catCols[1] ?? catCols[0] ?? "");
  const [colorCol, setColorCol] = useState("");
  const [errorType, setErrorType] = useState("ci"); // "ci" | "se" | "sd" | "none"

  const [uniqueValues, setUniqueValues] = useState<string[]>([]);
  const [plotData, setPlotData] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [chartTitle, setChartTitle] = useState("");

  // States for custom labels
  const [customTitle, setCustomTitle] = useState("");
  const [customYLabel, setCustomYLabel] = useState("");
  const [customXLabel, setCustomXLabel] = useState("");

  // States for legend
  const [showLegend, setShowLegend] = useState<boolean>(true);
  const [customLegendLabels, setCustomLegendLabels] = useState<Record<string, string>>({});

  // States for dimensions
  const [chartWidth, setChartWidth] = useState<number>(800);
  const [chartHeight, setChartHeight] = useState<number>(500);
  const [isAutoWidth, setIsAutoWidth] = useState<boolean>(true);

  // New visual enhancement states
  const [showValueLabels, setShowValueLabels] = useState<boolean>(true);
  const [showSampleSizes, setShowSampleSizes] = useState<boolean>(true);
  const [barWidth, setBarWidth] = useState<number>(0.6);
  const [referenceLine, setReferenceLine] = useState<number | null>(null);
  const [referenceLineLabel, setReferenceLineLabel] = useState<string>("Reference");

  // Sorting option
  const [sortBars, setSortBars] = useState<"none" | "value-desc" | "value-asc">("none");

  // Visual: Alternating background for outer subgroups
  const [showSubgroupBackgrounds, setShowSubgroupBackgrounds] = useState<boolean>(true);

  // Bar pattern/texture for accessibility & print
  const [barPattern, setBarPattern] = useState<"none" | "stripes" | "dots">("none");

  const chartRef = useRef<PlotCaptureHandle | null>(null);
  const plotContainerRef = useRef<HTMLDivElement>(null);

  // Drag the red line on the right edge to resize the chart width (drag left to
  // shrink). Mirrors the resize affordance used in the other analysis panels.
  const startChartWidthResize = (e: React.PointerEvent) => {
    e.preventDefault();
    const startW = plotContainerRef.current?.offsetWidth ?? chartWidth;
    setIsAutoWidth(false);
    setChartWidth(startW);
    const startX = e.clientX;
    const onMove = (ev: PointerEvent) => {
      setChartWidth(Math.max(360, Math.min(1600, startW + (ev.clientX - startX))));
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  };
  const resetChartWidth = () => { setChartWidth(800); setIsAutoWidth(true); };

  // Auto-switch mode or fetch unique values when Y column changes
  useEffect(() => {
    if (!yCol) return;
    const colMeta = session.columns.find((c) => c.name === yCol);
    if (colMeta?.kind === "categorical") {
      setYMode("percentage");
    } else {
      setYMode("mean");
    }
    
    // Fetch unique values for target category selection
    getUniqueValues(session.session_id, yCol)
      .then((res) => {
        const vals = (res.data as unknown[]).map((v) => String(v));
        setUniqueValues(vals);
        setTargetValue(vals[0] ?? "");
      })
      .catch(() => {
        setUniqueValues([]);
        setTargetValue("");
      });
  }, [yCol, session.session_id, session.columns]);

  // Fetch unique values if mode changes to percentage
  useEffect(() => {
    if (yMode === "percentage" && uniqueValues.length === 0 && yCol) {
      getUniqueValues(session.session_id, yCol)
        .then((res) => {
          const vals = (res.data as unknown[]).map((v) => String(v));
          setUniqueValues(vals);
          setTargetValue(vals[0] ?? "");
        })
        .catch(() => {});
    }
  }, [yMode, yCol, session.session_id, uniqueValues.length]);

  // ResizeObserver to sync manual drag-resize with states and exporter
  useEffect(() => {
    const el = plotContainerRef.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          // If the element has inline style width (manually dragged), update width & disable autoWidth
          if (el.style.width) {
            setIsAutoWidth(false);
            setChartWidth(Math.round(width));
          } else {
            // Keep tracking responsive width in state so exporter has correct value
            setChartWidth(Math.round(width));
          }
          setChartHeight(Math.round(height));
        }
      }
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const run = async () => {
    if (!yCol || !subgroupCol || !xaxisCol) {
      setError("Please configure Y-axis, Subgroup, and X-axis variables.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await runSubgroupBar({
        session_id: session.session_id,
        y_col: yCol,
        subgroup_col: subgroupCol,
        xaxis_col: xaxisCol,
        color_col: colorCol || undefined,
        y_mode: yMode,
        target_value: yMode === "percentage" ? targetValue : undefined,
        error_type: errorType,
      });
      setPlotData(res.data);
      
      // Resolve target value label if available
      const yColMeta = session.columns.find((c) => c.name === yCol);
      const yColLabels = yColMeta?.value_labels ?? {};
      const targetValueLabel = labelFor(yColLabels, targetValue, targetValue);

      // Auto-generate a beautiful descriptive title
      const modeText = yMode === "percentage" ? `% Use / Event Rate of ${yCol} (${targetValueLabel})` : `Mean of ${yCol}`;
      const colorText = colorCol ? ` by ${colorCol}` : "";
      const autoTitle = `${modeText} in Subgroups of ${subgroupCol}${colorText}`;
      
      const autoYAxisTitle = yMode === "percentage" ? `% ${yCol} (${targetValueLabel})` : `Mean of ${yCol}`;
      
      setChartTitle(autoTitle);
      setCustomTitle(autoTitle);
      setCustomYLabel(autoYAxisTitle);
      setCustomXLabel(""); // Reset X custom label or let user overwrite
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Error generating subgroup chart");
    } finally {
      setLoading(false);
    }
  };

  // Build Plotly-ready traces
  const traces = plotData ? buildPlotlyTraces(plotData, pal, session, customLegendLabels, {
    showValueLabels,
    showSampleSizes,
    barWidth,
    referenceLine,
    sortBars,
    barPattern,
  }) : null;

  // Layout Configuration
  const yColMeta = session.columns.find((c) => c.name === yCol);
  const yColLabels = yColMeta?.value_labels ?? {};
  const targetValueLabel = labelFor(yColLabels, targetValue, targetValue);
  const yAxisTitle = yMode === "percentage" ? `% ${yCol} (${targetValueLabel})` : `Mean of ${yCol}`;
  
  // Build shapes and annotations for visual enhancements
  const shapes: PlotData[] = [];
  const annotations: PlotData[] = [];

  // Reference line
  if (plotData && referenceLine !== null) {
    shapes.push({
      type: "line",
      x0: 0,
      x1: 1,
      xref: "paper",
      y0: referenceLine,
      y1: referenceLine,
      line: { color: "#dc2626", width: 2, dash: "dash" },
    });
    annotations.push({
      x: 0.98,
      y: referenceLine,
      xref: "paper",
      yref: "y",
      text: referenceLineLabel || "Ref",
      showarrow: false,
      font: { color: "#dc2626", size: 10, weight: "bold" },
      bgcolor: "white",
      bordercolor: "#dc2626",
      borderwidth: 1,
      borderpad: 2,
    });
  }

  // Alternating background colors for outer subgroups (visual hierarchy)
  if (plotData && showSubgroupBackgrounds && plotData.subgroups) {
    const subgroups = plotData.subgroups as unknown[];
    const subgroupCount = subgroups.length;
    subgroups.forEach((_sg, idx) => {
      if (idx % 2 === 0) {
        // Light alternating background
        shapes.push({
          type: "rect",
          xref: "x",
          yref: "paper",
          x0: idx * (1 / subgroupCount) - 0.01,
          x1: (idx + 1) * (1 / subgroupCount) + 0.01,
          y0: 0,
          y1: 1,
          fillcolor: "rgba(243, 244, 246, 0.35)", // very light gray
          line: { width: 0 },
          layer: "below",
        });
      }
    });
  }

  const fullLayout = plotData ? {
    ...layout,
    width: chartWidth - 32,
    height: chartHeight - 32,
    title: { text: customTitle || chartTitle, font: { color: "#374151", size: 13, weight: "bold" } },
    showlegend: showLegend,
    xaxis: {
      ...(layout.xaxis as PlotLayout),
      title: customXLabel ? { text: customXLabel, font: { size: 11 } } : undefined,
      type: "multicategory",
      showgrid: showGrid,
      gridcolor: showGrid ? "#e5e7eb" : "transparent",
      dividercolor: "#9ca3af",
      dividerwidth: 1,
      tickfont: { size: 10 },
    },
    yaxis: {
      ...(layout.yaxis as PlotLayout),
      title: { text: customYLabel || yAxisTitle, font: { size: 11 } },
      showgrid: showGrid,
      gridcolor: showGrid ? "#e5e7eb" : "transparent",
      tickfont: { size: 10 },
    },
    barmode: "group",
    legend: {
      orientation: "h",
      yanchor: "bottom",
      y: 1.02,
      xanchor: "right",
      x: 1,
      font: { size: 10 },
    },
    margin: { t: 70, b: 60, l: 60, r: 20 },
    autosize: true,
    shapes,
    annotations,
  } : null;

  return (
    <div className="flex gap-4 h-full items-start">
      {/* Controls panel */}
      <div className="w-72 flex-shrink-0 space-y-4 overflow-y-auto pr-1" style={{ maxHeight: "calc(100vh - 120px)" }}>
        
        {/* Main Settings Panel */}
        <div className="panel space-y-4">
          <h3 className="text-sm font-semibold text-gray-800 border-b pb-2">Subgroup Bar Settings</h3>
          
          {/* Y Axis Variable */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Y-Axis Variable (Value)</label>
            <select className="select w-full" value={yCol} onChange={(e) => setYCol(e.target.value)}>
              {allCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Y Summary Mode */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Summary Mode</label>
            <div className="flex gap-4 mt-1">
              <label className="flex items-center gap-1.5 cursor-pointer text-xs text-gray-700">
                <input
                  type="radio"
                  name="yMode"
                  value="mean"
                  checked={yMode === "mean"}
                  disabled={session.columns.find((c) => c.name === yCol)?.kind === "categorical"}
                  onChange={() => setYMode("mean")}
                  className="accent-indigo-500"
                />
                Mean
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer text-xs text-gray-700">
                <input
                  type="radio"
                  name="yMode"
                  value="percentage"
                  checked={yMode === "percentage"}
                  onChange={() => setYMode("percentage")}
                  className="accent-indigo-500"
                />
                Percentage (%)
              </label>
            </div>
          </div>

          {/* Target Category Value (Only if Percentage Mode) */}
          {yMode === "percentage" && uniqueValues.length > 0 && (
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">Target Event / Category</label>
              <select className="select w-full text-xs" value={targetValue} onChange={(e) => setTargetValue(e.target.value)}>
                {uniqueValues.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Subgroup Variable (Outer X-Axis) */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Subgroup Variable (Outer X-Axis)</label>
            <select className="select w-full" value={subgroupCol} onChange={(e) => setSubgroupCol(e.target.value)}>
              {catCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <p className="text-[10px] text-gray-400 mt-0.5">Creates separate panels/divisions (e.g. Normotensive, Hypertensive)</p>
          </div>

          {/* X Axis Variable (Inner X-Axis) */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Inner Variable (Inner X-Axis)</label>
            <select className="select w-full" value={xaxisCol} onChange={(e) => setXaxisCol(e.target.value)}>
              {catCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <p className="text-[10px] text-gray-400 mt-0.5">Defines subgroups within each panel (e.g. Year 0 vs Year 6)</p>
          </div>

          {/* Color / Group Variable */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Color / Legend Variable (Bars)</label>
            <select className="select w-full" value={colorCol} onChange={(e) => setColorCol(e.target.value)}>
              <option value="">— None (Single Bar Group) —</option>
              {catCols.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <p className="text-[10px] text-gray-400 mt-0.5">Groups side-by-side bars (e.g. Intervention vs Comparison)</p>
          </div>

          {/* Error Bars Type */}
          <div>
            <label className="text-xs font-medium text-gray-500 block mb-1">Error Bars</label>
            <select className="select w-full text-xs" value={errorType} onChange={(e) => setErrorType(e.target.value)}>
              <option value="ci">95% Confidence Interval (CI)</option>
              <option value="se">Standard Error (SE)</option>
              <option value="sd">Standard Deviation (SD)</option>
              <option value="none">No Error Bars</option>
            </select>
          </div>

          <button className="btn-primary w-full py-2 font-medium" onClick={run} disabled={loading}>
            {loading ? "Generating Chart…" : "Generate Chart"}
          </button>
          
          {error && <p className="text-red-500 text-xs font-medium leading-relaxed">{error}</p>}
        </div>

        {/* Custom Labels & Dimensions Panel */}
        {plotData && (
          <div className="panel space-y-3.5">
            <h3 className="text-sm font-semibold text-gray-800 border-b pb-2">Custom Labels & Dimensions</h3>
            
            {/* Custom Chart Title */}
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">Chart Title</label>
              <input
                type="text"
                className="select w-full text-xs py-1 px-2 border rounded"
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
                placeholder="Title..."
              />
            </div>

            {/* Custom Y Axis Label */}
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">Y-Axis Label</label>
              <input
                type="text"
                className="select w-full text-xs py-1 px-2 border rounded"
                value={customYLabel}
                onChange={(e) => setCustomYLabel(e.target.value)}
                placeholder="Y-axis label..."
              />
            </div>

            {/* Custom X Axis Label */}
            <div>
              <label className="text-xs font-medium text-gray-500 block mb-1">X-Axis Label</label>
              <input
                type="text"
                className="select w-full text-xs py-1 px-2 border rounded"
                value={customXLabel}
                onChange={(e) => setCustomXLabel(e.target.value)}
                placeholder="X-axis label..."
              />
            </div>

            {/* Show/Hide Legend */}
            <div className="flex items-center justify-between pt-2 border-t border-gray-100">
              <span className="text-xs font-medium text-gray-500">Show Legend</span>
              <input
                type="checkbox"
                className="accent-indigo-500 h-3.5 w-3.5 rounded cursor-pointer"
                checked={showLegend}
                onChange={(e) => setShowLegend(e.target.checked)}
              />
            </div>

            {/* Custom Legend Labels */}
            {colorCol && Array.isArray(plotData?.traces) && showLegend && (
              <div className="space-y-2 pt-2 border-t border-gray-100">
                <span className="text-xs font-semibold text-gray-700 block">Custom Legend Labels</span>
                <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
                  {plotData.traces.map((t: { name: unknown }) => {
                    const rawVal = String(t.name);
                    const colorColMeta = session.columns.find((c) => c.name === colorCol);
                    const colorLabels = colorColMeta?.value_labels ?? {};
                    const defaultLabel = labelFor(colorLabels, rawVal, String(rawVal));
                    return (
                      <div key={rawVal} className="flex items-center gap-2">
                        <span className="text-[10px] text-gray-500 min-w-16 truncate" title={rawVal}>{rawVal}:</span>
                        <input
                          type="text"
                          className="select flex-1 text-[11px] py-0.5 px-2 border rounded focus:outline-none focus:border-indigo-400 bg-white"
                          value={customLegendLabels[rawVal] ?? ""}
                          onChange={(e) => {
                            setCustomLegendLabels((prev) => ({
                              ...prev,
                              [rawVal]: e.target.value,
                            }));
                          }}
                          placeholder={defaultLabel}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Dimensions Control */}
            <div className="pt-2 border-t border-gray-100 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-gray-500">Auto Width (Responsive)</span>
                <input
                  type="checkbox"
                  className="accent-indigo-500 h-3.5 w-3.5 rounded cursor-pointer"
                  checked={isAutoWidth}
                  onChange={(e) => {
                    setIsAutoWidth(e.target.checked);
                    if (e.target.checked && plotContainerRef.current) {
                      // Remove inline style width so browser handles it responsively
                      plotContainerRef.current.style.width = "";
                    }
                  }}
                />
              </div>

              {!isAutoWidth && (
                <div>
                  <label className="text-xs font-medium text-gray-500 block mb-1">Chart Width: {chartWidth}px</label>
                  <input
                    type="range"
                    min={400}
                    max={1600}
                    step={20}
                    value={chartWidth}
                    onChange={(e) => {
                      setChartWidth(+e.target.value);
                      if (plotContainerRef.current) {
                        plotContainerRef.current.style.width = `${e.target.value}px`;
                      }
                    }}
                    className="w-full accent-indigo-500 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              )}

              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Chart Height: {chartHeight}px</label>
                <input
                  type="range"
                  min={300}
                  max={1200}
                  step={20}
                  value={chartHeight}
                  onChange={(e) => {
                    setChartHeight(+e.target.value);
                    if (plotContainerRef.current) {
                      plotContainerRef.current.style.height = `${e.target.value}px`;
                    }
                  }}
                  className="w-full accent-indigo-500 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              <p className="text-[10px] text-gray-400 leading-snug">
                💡 Drag the bottom-right corner of the chart panel to resize it manually. The export dimensions will automatically sync!
              </p>

              <div className="text-[10px] text-emerald-600 mt-1">
                ✓ Value labels, reference lines, and bar spacing controls added for publication-ready visuals.
              </div>
            </div>

            {/* Visual Enhancements */}
            <div className="pt-3 border-t border-gray-100 space-y-3">
              <div className="text-xs font-semibold text-gray-600">Visual Polish</div>

              <div className="flex items-center justify-between text-xs">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={showValueLabels} onChange={(e) => setShowValueLabels(e.target.checked)} className="accent-indigo-600" />
                  Show value labels on bars
                </label>
              </div>

              <div className="flex items-center justify-between text-xs">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={showSampleSizes} onChange={(e) => setShowSampleSizes(e.target.checked)} className="accent-indigo-600" />
                  Show N (sample size) under bars
                </label>
              </div>

              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Bar Width / Gap: {barWidth.toFixed(1)}</label>
                <input 
                  type="range" 
                  min={0.3} 
                  max={1.0} 
                  step={0.05} 
                  value={barWidth} 
                  onChange={(e) => setBarWidth(parseFloat(e.target.value))} 
                  className="w-full accent-indigo-500" 
                />
              </div>

              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Reference Line (horizontal)</label>
                <div className="flex gap-2">
                  <input 
                    type="number" 
                    value={referenceLine ?? ""} 
                    onChange={(e) => setReferenceLine(e.target.value ? parseFloat(e.target.value) : null)} 
                    className="select flex-1 text-xs py-1" 
                    placeholder="e.g. 0 or 50"
                  />
                  <input 
                    type="text" 
                    value={referenceLineLabel} 
                    onChange={(e) => setReferenceLineLabel(e.target.value)} 
                    className="select w-28 text-xs py-1" 
                    placeholder="Label"
                  />
                </div>
                {referenceLine !== null && (
                  <button onClick={() => setReferenceLine(null)} className="text-[10px] text-red-500 mt-1 hover:underline">Remove reference line</button>
                )}
              </div>

              {/* Bar Sorting */}
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Sort Bars</label>
                <select 
                  className="select w-full text-xs" 
                  value={sortBars} 
                  onChange={(e) => setSortBars(e.target.value as "none" | "value-desc" | "value-asc")}
                >
                  <option value="none">No sorting (original order)</option>
                  <option value="value-desc">Sort by value (descending)</option>
                  <option value="value-asc">Sort by value (ascending)</option>
                </select>
              </div>

              {/* Alternating Subgroup Backgrounds */}
              <div className="flex items-center justify-between text-xs">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input 
                    type="checkbox" 
                    checked={showSubgroupBackgrounds} 
                    onChange={(e) => setShowSubgroupBackgrounds(e.target.checked)} 
                    className="accent-indigo-600" 
                  />
                  Alternating subgroup backgrounds
                </label>
              </div>

              {/* Bar Patterns (for print / accessibility) */}
              <div>
                <label className="text-xs font-medium text-gray-500 block mb-1">Bar Pattern (Texture)</label>
                <select 
                  className="select w-full text-xs" 
                  value={barPattern} 
                  onChange={(e) => setBarPattern(e.target.value as "none" | "stripes" | "dots")}
                >
                  <option value="none">Solid (default)</option>
                  <option value="stripes">Stripes</option>
                  <option value="dots">Dots</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Informational Guidance tip */}
        <div className="panel bg-indigo-50/50 border-indigo-100 p-3.5 space-y-2">
          <p className="text-[10px] font-bold text-indigo-700 uppercase tracking-wider">Nested Subgroup Chart Guidance</p>
          <p className="text-xs text-gray-600 leading-relaxed">
            This specialized visualization uses a **multicategory hierarchical X-axis** to display summary statistics nested across three distinct grouping variables:
          </p>
          <ul className="text-xs text-gray-600 list-disc list-inside space-y-1">
            <li>**Outer Subgroup**: Splits the chart into separate visual categories.</li>
            <li>**Inner Subgroup**: Shows secondary groupings nested within each main subgroup.</li>
            <li>**Legend Variable**: Renders side-by-side grouped colored bars.</li>
          </ul>
        </div>
      </div>

      {/* Plot area with custom sizing and resizing handle */}
      <div className="flex-1 min-w-0" style={{ maxWidth: "100%" }}>
        {traces ? (
          <div 
            ref={plotContainerRef}
            className="panel relative flex flex-col justify-between bg-white border border-gray-200 shadow-sm"
            style={{ 
              width: isAutoWidth ? "100%" : `${chartWidth}px`, 
              height: `${chartHeight}px`,
              resize: "both",
              overflow: "hidden",
              maxWidth: "100%",
            }}
          >
            <div className="absolute right-4 top-4 z-10">
              <PlotExporter 
                plotRef={chartRef} 
                title={`SubgroupBarChart_${yCol}_by_${subgroupCol}`} 
                defaultWidth={chartWidth}
                defaultHeight={chartHeight}
              />
            </div>
            <div className="flex-1 min-h-0">
              <Plot
                data={traces as unknown as Data[]}
                layout={fullLayout as unknown as Partial<Layout>}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
                config={{ responsive: true, displayModeBar: true, displaylogo: false }}
                onInitialized={(_: object, gd: HTMLElement) => { chartRef.current = gd as unknown as PlotCaptureHandle; }}
                onUpdate={(_: object, gd: HTMLElement) => { chartRef.current = gd as unknown as PlotCaptureHandle; }}
              />
            </div>
            {/* Red vertical resize line on the right — drag left to shrink the chart */}
            <div
              onPointerDown={startChartWidthResize}
              onDoubleClick={resetChartWidth}
              className="absolute top-0 bottom-0 w-[5px] right-0 cursor-col-resize bg-red-500/70 hover:bg-red-600 active:bg-red-700 transition-colors z-20"
              title="Drag the red line to resize the chart width • Double-click to reset"
            />
          </div>
        ) : (
          <div className="panel min-h-[480px] flex flex-col items-center justify-center text-slate-400 p-8 text-center bg-white border border-gray-200 shadow-sm rounded-2xl relative overflow-hidden">
            {/* High-Fidelity SVG Preview Illustration */}
            <div className="w-full max-w-lg mb-6 opacity-85 hover:opacity-100 transition-opacity duration-300 bg-slate-50/50 p-4 rounded-xl border border-slate-100 shadow-inner">
              <svg width="100%" height="200" viewBox="0 0 450 200" fill="none" xmlns="http://www.w3.org/2000/svg" className="mx-auto select-none">
                <defs>
                  <linearGradient id="grad1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#4f46e5" />
                  </linearGradient>
                  <linearGradient id="grad2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#06b6d4" />
                    <stop offset="100%" stopColor="#0891b2" />
                  </linearGradient>
                </defs>

                {/* Grid Lines */}
                <line x1="40" y1="30" x2="410" y2="30" stroke="#f1f5f9" strokeWidth="1" />
                <line x1="40" y1="65" x2="410" y2="65" stroke="#f1f5f9" strokeWidth="1" />
                <line x1="40" y1="100" x2="410" y2="100" stroke="#f1f5f9" strokeWidth="1" />
                <line x1="40" y1="135" x2="410" y2="135" stroke="#f1f5f9" strokeWidth="1" />

                {/* Y-Axis Line */}
                <line x1="40" y1="20" x2="40" y2="170" stroke="#cbd5e1" strokeWidth="1.5" />
                {/* X-Axis Line */}
                <line x1="40" y1="170" x2="410" y2="170" stroke="#cbd5e1" strokeWidth="1.5" />

                {/* Group 1 (Subgroup A) */}
                {/* Category 1 */}
                <rect x="75" y="70" width="14" height="100" rx="2" fill="url(#grad1)" />
                <rect x="91" y="90" width="14" height="80" rx="2" fill="url(#grad2)" />
                <line x1="82" y1="50" x2="82" y2="70" stroke="#475569" strokeWidth="1.2" />
                <line x1="78" y1="50" x2="86" y2="50" stroke="#475569" strokeWidth="1.2" />
                <line x1="98" y1="75" x2="98" y2="90" stroke="#475569" strokeWidth="1.2" />
                <line x1="94" y1="75" x2="102" y2="75" stroke="#475569" strokeWidth="1.2" />

                {/* Category 2 */}
                <rect x="145" y="40" width="14" height="130" rx="2" fill="url(#grad1)" />
                <rect x="161" y="60" width="14" height="110" rx="2" fill="url(#grad2)" />
                <line x1="152" y1="25" x2="152" y2="40" stroke="#475569" strokeWidth="1.2" />
                <line x1="148" y1="25" x2="156" y2="25" stroke="#475569" strokeWidth="1.2" />
                <line x1="168" y1="45" x2="168" y2="60" stroke="#475569" strokeWidth="1.2" />
                <line x1="164" y1="45" x2="172" y2="45" stroke="#475569" strokeWidth="1.2" />

                <text x="125" y="178" fill="#94a3b8" fontSize="8" fontFamily="system-ui" textAnchor="middle">Category A</text>

                {/* Subgroup 1 Bracket */}
                <line x1="68" y1="184" x2="182" y2="184" stroke="#94a3b8" strokeWidth="1" />
                <line x1="68" y1="180" x2="68" y2="184" stroke="#94a3b8" strokeWidth="1" />
                <line x1="182" y1="180" x2="182" y2="184" stroke="#94a3b8" strokeWidth="1" />
                <text x="125" y="194" fill="#64748b" fontSize="8" fontWeight="bold" fontFamily="system-ui" textAnchor="middle">Subgroup 1 (e.g., Male)</text>

                {/* Group 2 (Subgroup B) */}
                {/* Category 1 */}
                <rect x="255" y="85" width="14" height="85" rx="2" fill="url(#grad1)" />
                <rect x="271" y="55" width="14" height="115" rx="2" fill="url(#grad2)" />
                <line x1="262" y1="70" x2="262" y2="85" stroke="#475569" strokeWidth="1.2" />
                <line x1="258" y1="70" x2="266" y2="70" stroke="#475569" strokeWidth="1.2" />
                <line x1="278" y1="40" x2="278" y2="55" stroke="#475569" strokeWidth="1.2" />
                <line x1="274" y1="40" x2="282" y2="40" stroke="#475569" strokeWidth="1.2" />

                {/* Category 2 */}
                <rect x="325" y="115" width="14" height="55" rx="2" fill="url(#grad1)" />
                <rect x="341" y="100" width="14" height="70" rx="2" fill="url(#grad2)" />
                <line x1="332" y1="105" x2="332" y2="115" stroke="#475569" strokeWidth="1.2" />
                <line x1="328" y1="105" x2="336" y2="105" stroke="#475569" strokeWidth="1.2" />
                <line x1="348" y1="88" x2="348" y2="100" stroke="#475569" strokeWidth="1.2" />
                <line x1="344" y1="88" x2="352" y2="88" stroke="#475569" strokeWidth="1.2" />

                <text x="305" y="178" fill="#94a3b8" fontSize="8" fontFamily="system-ui" textAnchor="middle">Category B</text>

                {/* Subgroup 2 Bracket */}
                <line x1="248" y1="184" x2="362" y2="184" stroke="#94a3b8" strokeWidth="1" />
                <line x1="248" y1="180" x2="248" y2="184" stroke="#94a3b8" strokeWidth="1" />
                <line x1="362" y1="180" x2="362" y2="184" stroke="#94a3b8" strokeWidth="1" />
                <text x="305" y="194" fill="#64748b" fontSize="8" fontWeight="bold" fontFamily="system-ui" textAnchor="middle">Subgroup 2 (e.g., Female)</text>

                {/* Legend */}
                <rect x="315" y="10" width="80" height="26" rx="4" fill="white" stroke="#e2e8f0" strokeWidth="1" />
                <rect x="321" y="15" width="6" height="6" rx="1.5" fill="url(#grad1)" />
                <text x="331" y="21" fill="#475569" fontSize="7" fontFamily="system-ui">Level 1</text>
                <rect x="361" y="15" width="6" height="6" rx="1.5" fill="url(#grad2)" />
                <text x="371" y="21" fill="#475569" fontSize="7" fontFamily="system-ui">Level 2</text>
              </svg>
            </div>
            <span className="text-sm font-bold text-slate-800 tracking-tight">Configure and Generate a Nested Subgroup Bar Chart</span>
            <p className="text-xs text-slate-400 max-w-md mt-1.5 leading-relaxed">
              This advanced clinical visualization uses a <strong>multicategory hierarchical X-axis</strong>. Select a continuous Y-axis variable, a primary subgroup (outer categories), and an inner variable to instantly generate publication-ready comparative charts with 95% Confidence Intervals.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/** A raw subgroup-bar trace as returned by the subgroup_bar endpoint. */
interface RawSubgroupTrace {
  name: string;
  x_subgroup: unknown[];
  x_xaxis: unknown[];
  y: number[];
  ns?: unknown;
  error?: number[];
  error_high?: number[];
  error_low?: number[];
}

function buildPlotlyTraces(
  plotData: Record<string, unknown> | null,
  pal: string[],
  session: Session | null,
  customLegendLabels: Record<string, string>,
  options: {
    showValueLabels?: boolean;
    showSampleSizes?: boolean;
    barWidth?: number;
    referenceLine?: number | null;
    sortBars?: "none" | "value-desc" | "value-asc";
    barPattern?: "none" | "stripes" | "dots";
  } = {}
): PlotData[] {
  if (!plotData || !plotData.traces || !session) return [];

  const {
    showValueLabels = true,
    showSampleSizes = true,
    barWidth = 0.6,
    sortBars = "none",
    barPattern = "none"
  } = options;

  const tracesData = [...(plotData.traces as RawSubgroupTrace[])];

  // === 1. Bar Sorting by Value ===
  if (sortBars !== "none") {
    tracesData.sort((a, b) => {
      const sumA = a.y.reduce((sum, v) => sum + (v || 0), 0);
      const sumB = b.y.reduce((sum, v) => sum + (v || 0), 0);
      return sortBars === "value-desc" ? sumB - sumA : sumA - sumB;
    });
  }

  const labelsFor = (colName: unknown): Record<string, string> => {
    const meta = session.columns.find((c) => c.name === colName);
    return meta?.value_labels ?? {};
  };

  const subgroupLabels = labelsFor(plotData.subgroup_col);
  const xaxisLabels = labelsFor(plotData.xaxis_col);
  const colorLabels = labelsFor(plotData.color_col);

  return tracesData.map((t, i) => {
    const mappedSubgroup = t.x_subgroup.map((v) => labelFor(subgroupLabels, v, String(v)));
    const mappedXaxis = t.x_xaxis.map((v) => labelFor(xaxisLabels, v, String(v)));
    const mappedName = customLegendLabels[String(t.name)] || labelFor(colorLabels, t.name, String(t.name));

    const xArray = [mappedSubgroup, mappedXaxis];

    const trace: PlotData = {
      type: "bar",
      name: mappedName,
      x: xArray,
      y: t.y,
      width: barWidth,
      error_y: {
        type: "data",
        // Asymmetric when the backend sends Wilson bounds (percentage CI);
        // symmetric (low == high) for SE/SD and t-CI on means.
        symmetric: false,
        array: t.error_high ?? t.error,
        arrayminus: t.error_low ?? t.error,
        visible: plotData.error_type !== "none",
        color: "#374151",
        thickness: 1.8,
        width: 6,
      },
      marker: {
        color: pal[i % pal.length],
        line: { color: "#1f2937", width: 0.6 },
        pattern: barPattern === "stripes" 
          ? { shape: "/", fgcolor: "rgba(0,0,0,0.25)", size: 6 } 
          : barPattern === "dots" 
            ? { shape: ".", fgcolor: "rgba(0,0,0,0.3)", size: 4 } 
            : undefined,
      },
      customdata: t.ns,
      hovertemplate: 
        `<b>%{x}</b><br>` +
        `Value: %{y:.2f}${plotData.y_mode === 'percentage' ? '%' : ''}<br>` +
        (plotData.error_type !== 'none' ? `Error: %{error_y.array:.2f}<br>` : '') +
        `N: %{customdata}<extra>${mappedName}</extra>`,
    };

    // Value labels on top of bars
    if (showValueLabels) {
      trace.text = t.y.map((v) => v.toFixed(1) + (plotData.y_mode === 'percentage' ? '%' : ''));
      trace.textposition = 'outside';
      trace.textfont = { size: 10, color: '#111827' };
    }

    if (showSampleSizes && t.ns) {
      trace.customdata = t.ns;
    }

    return trace;
  });
}
