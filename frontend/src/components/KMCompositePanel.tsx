import { useMemo, useRef, useState } from "react";
import { runKMComposite } from "../api";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import { describeStale } from "../lib/resultStamp";
import type { PlotCaptureHandle, PlotData, PlotLayout } from "../lib/plotTypes";
import { usePlotLayout, usePalette } from "../plotStyle";
import { useStore, isCategoricalKind, isNumericKind, type ColMeta, type Session } from "../store";
import TitledPlot from "./TitledPlot";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";

interface EndpointRow {
  duration_col: string;
  event_col: string;
  label: string;
}

interface EndpointSummary {
  label: string;
  p_text: string;
  final_by_group: Record<string, number>;
  n_by_group: Record<string, number>;
}

interface KMCompositeResult {
  type: "km_composite";
  group_col: string;
  groups: string[];
  endpoints: EndpointSummary[];
  as_cumulative_incidence: boolean;
  figure: { data: PlotData[]; layout: PlotLayout };
  method_note: string;
}

export default function KMCompositePanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <KMCompositePanelBody session={session} />;
}

function displayName(col: ColMeta | undefined, fallback: string): string {
  return col?.label || col?.display_name || col?.name || fallback;
}

function parseRiskTimes(raw: string): number[] {
  return raw
    .split(",")
    .map((t) => Number(t.trim()))
    .filter((t) => Number.isFinite(t));
}

function KMCompositePanelBody({ session }: { session: Session }) {
  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const plotRef = useRef<PlotCaptureHandle | null>(null);
  const columns = session.columns.filter((col) => !col.analysis_excluded);
  const numericCols = columns.filter((col) => isNumericKind(col.kind)).map((col) => col.name);
  const groupCols = columns.filter((col) => isCategoricalKind(col.kind)).map((col) => col.name);

  const [groupCol, setGroupCol] = usePersistedPanelState("km_composite", "groupCol", groupCols[0] ?? "");
  const [endpoints, setEndpoints] = usePersistedPanelState<EndpointRow[]>("km_composite", "endpoints", [
    { duration_col: numericCols[0] ?? "", event_col: numericCols[1] ?? numericCols[0] ?? "", label: "" },
  ]);
  const [riskTimesRaw, setRiskTimesRaw] = usePersistedPanelState("km_composite", "riskTimes", "0, 3, 6, 9, 12");
  const [cumInc, setCumInc] = usePersistedPanelState("km_composite", "cumInc", true);
  const [inset, setInset] = usePersistedPanelState("km_composite", "inset", true);
  // What the curves and their numbers depend on. Panel labels and the zoom
  // inset only dress the figure, so editing them after a run leaves the
  // estimates current.
  const runParams = {
    groupCol,
    endpoints: endpoints.map((ep) => ({ duration_col: ep.duration_col, event_col: ep.event_col })),
    riskTimes: parseRiskTimes(riskTimesRaw),
    cumInc,
  };
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<KMCompositeResult>("km_composite", runParams);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canRun = Boolean(
    groupCol &&
    endpoints.length >= 1 &&
    endpoints.length <= 4 &&
    endpoints.every((ep) => ep.duration_col && ep.event_col),
  );

  const updateEndpoint = (idx: number, patch: Partial<EndpointRow>) => {
    setEndpoints(endpoints.map((ep, i) => (i === idx ? { ...ep, ...patch } : ep)));
  };
  const addEndpoint = () => {
    if (endpoints.length >= 4) return;
    setEndpoints([...endpoints, { duration_col: numericCols[0] ?? "", event_col: numericCols[0] ?? "", label: "" }]);
  };
  const removeEndpoint = (idx: number) => {
    if (endpoints.length <= 1) return;
    setEndpoints(endpoints.filter((_, i) => i !== idx));
  };

  const run = async () => {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runKMComposite({
        session_id: session.session_id,
        group_col: groupCol,
        risk_times: parseRiskTimes(riskTimesRaw),
        as_cumulative_incidence: cumInc,
        inset,
        title: "Composite Primary End Point and Individual Components",
        endpoints: endpoints.map((ep) => ({
          duration_col: ep.duration_col,
          event_col: ep.event_col,
          label: ep.label.trim() || displayName(columns.find((c) => c.name === ep.event_col), ep.event_col),
        })),
      });
      setResult(res.data as KMCompositeResult);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Error generating KM composite figure");
    } finally {
      setLoading(false);
    }
  };

  const mergedLayout = result
    ? ({ ...baseLayout, ...result.figure.layout, height: result.figure.layout.height ?? 760 } as PlotLayout)
    : null;

  // Backend ships each curve with a fixed default color per arm (keyed via
  // `legendgroup`); re-map to the app's active theme palette so the palette
  // picker affects this chart, mirroring ScoreCompositePanel / SubgroupBar.
  const paletteData = useMemo(() => {
    if (!result) return null;
    return result.figure.data.map((trace) => {
      const legendgroup = (trace as { legendgroup?: string }).legendgroup;
      if (!legendgroup) return trace;
      const idx = result.groups.indexOf(legendgroup);
      const color = pal[(idx < 0 ? 0 : idx) % pal.length];
      const line = (trace as { line?: Record<string, unknown> }).line;
      return { ...trace, line: { ...(line || {}), color } };
    });
  }, [result, pal]);

  const titleText = typeof result?.figure.layout.title === "object"
    ? String((result.figure.layout.title as { text?: unknown }).text ?? "")
    : "Composite Primary End Point and Individual Components";

  return (
    <div className="flex gap-4 h-full">
      <div className="w-80 flex-shrink-0 space-y-4 overflow-y-auto pr-1" style={{ maxHeight: "calc(100vh - 120px)" }}>
        <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
          <h3 className="text-sm font-semibold text-gray-700">Comparison</h3>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Arm / group column</label>
            <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
              {groupCols.map((name) => <option key={name} value={name}>{displayName(columns.find((c) => c.name === name), name)}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">No.-at-risk time points (comma-separated)</label>
            <input className="select w-full" value={riskTimesRaw} onChange={(e) => setRiskTimesRaw(e.target.value)} placeholder="0, 3, 6, 9, 12" />
          </div>
          <div className="flex flex-col gap-1.5 pt-1">
            <label className="flex items-center gap-2 text-xs text-gray-600">
              <input type="checkbox" checked={cumInc} onChange={(e) => setCumInc(e.target.checked)} className="accent-indigo-500" />
              Cumulative incidence (1 − KM), rising from 0
            </label>
            <label className="flex items-center gap-2 text-xs text-gray-600">
              <input type="checkbox" checked={inset} onChange={(e) => setInset(e.target.checked)} className="accent-indigo-500" />
              Magnified zoom inset per panel
            </label>
          </div>
        </div>

        <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Endpoints ({endpoints.length}/4)</h3>
            <button className="text-xs text-indigo-600 hover:text-indigo-800 disabled:text-gray-300"
              onClick={addEndpoint} disabled={endpoints.length >= 4}>+ Add</button>
          </div>
          {endpoints.map((ep, idx) => (
            <div key={idx} className="rounded-lg border border-gray-200 p-2.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-gray-500">Panel {String.fromCharCode(65 + idx)}</span>
                {endpoints.length > 1 && (
                  <button className="text-xs text-red-400 hover:text-red-600" onClick={() => removeEndpoint(idx)}>Remove</button>
                )}
              </div>
              <div>
                <label className="text-[11px] text-gray-400 block mb-0.5">Time / duration column</label>
                <select className="select w-full" value={ep.duration_col} onChange={(e) => updateEndpoint(idx, { duration_col: e.target.value })}>
                  {numericCols.map((name) => <option key={name} value={name}>{displayName(columns.find((c) => c.name === name), name)}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[11px] text-gray-400 block mb-0.5">Event column (0/1)</label>
                <select className="select w-full" value={ep.event_col} onChange={(e) => updateEndpoint(idx, { event_col: e.target.value })}>
                  {numericCols.map((name) => <option key={name} value={name}>{displayName(columns.find((c) => c.name === name), name)}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[11px] text-gray-400 block mb-0.5">Panel label</label>
                <input className="select w-full" value={ep.label} onChange={(e) => updateEndpoint(idx, { label: e.target.value })}
                  placeholder={displayName(columns.find((c) => c.name === ep.event_col), ep.event_col)} />
              </div>
            </div>
          ))}
        </div>

        <button className="btn-primary w-full" onClick={run} disabled={loading || !canRun}>
          {loading ? "Generating Figure..." : "Generate KM Composite"}
        </button>
        {!canRun && (
          <p className="text-xs text-amber-600 leading-relaxed">
            Select an arm column and give every endpoint a time and event column (1–4 endpoints).
          </p>
        )}
        {error && <p className="text-red-500 text-xs">{error}</p>}
      </div>

      <div className="flex-1 panel min-h-0 bg-white border border-gray-200 shadow-sm rounded-2xl p-4 overflow-y-auto">
        {result && mergedLayout ? (
          <div className="space-y-4">
            {stale && (
              <StaleResultNotice reasons={staleWhy} onRecompute={run} busy={loading} what="This KM composite" />
            )}
            <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <TitledPlot
              plotRefOut={plotRef}
              storageKey={`km-composite:${groupCol}:${endpoints.map((e) => e.event_col).join(",")}`}
              data={paletteData ?? result.figure.data}
              layout={mergedLayout}
              config={{ responsive: true, displayModeBar: true, displaylogo: false }}
              defaultTitle={titleText}
              defaultSubtitle={result.method_note}
              defaultXAxis=""
              defaultYAxis=""
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {result.endpoints.map((ep) => (
                <div key={ep.label} className="rounded-lg border border-gray-200 bg-gray-50 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-semibold text-gray-800">{ep.label}</p>
                    <p className="text-xs text-gray-500">{ep.p_text}</p>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {result.groups.map((g) => `${g}: ${ep.final_by_group[g] ?? 0}${result.as_cumulative_incidence ? "%" : ""} (n=${ep.n_by_group[g] ?? 0})`).join(" · ")}
                  </p>
                </div>
              ))}
            </div>
            </StaleGuard>
          </div>
        ) : (
          <div className="h-full min-h-[420px] flex items-center justify-center text-gray-400">
            Configure endpoints and generate a KM composite figure
          </div>
        )}
      </div>
    </div>
  );
}
