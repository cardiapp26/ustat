import { useState, useRef, type ReactNode } from "react";
import { useStore } from "../store";
import { usePlotLayout, usePalette } from "../plotStyle";
import { runMetaAnalyze, runMetaSubgroup, runMetaRegression, runMetaBias } from "../api";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import { describeStale } from "../lib/resultStamp";
import { Tip } from "./Tip";
import TitledPlot from "./TitledPlot";
import ResultExporter from "./ResultExporter";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";
import { fmtP } from "../lib/format";
import type { PlotData, PlotCaptureHandle } from "../lib/plotTypes";

interface MetaStudy {
  label: string;
  effect: number;
  ci_low: number;
  ci_high: number;
  weight_pct: number;
}
interface MetaPoint { moderator: number; effect: number; size: number; label: string }
interface MetaFunnelPoint { effect: number; se: number; label: string }
interface MetaSubgroupRow {
  subgroup: string; k: number; effect: number; ci_low: number; ci_high: number; I2_pct: number;
}
interface MetaPooled { effect: number; ci_low: number; ci_high: number }
interface MetaResult {
  studies?: MetaStudy[];
  random: MetaPooled;
  fixed: MetaPooled;
  measure: string;
  null_line: number;
  I2_pct: number;
  tau2: number;
  Q: number;
  Q_p: number;
  prediction_low?: number | null;
  prediction_high?: number | null;
  export_rows?: string[][];
  points?: MetaPoint[];
  line_x: number[];
  line_y: number[];
  log_scale?: boolean;
  funnel?: MetaFunnelPoint[];
  pooled_effect: number;
  se_max: number;
  subgroups?: MetaSubgroupRow[];
  q_between?: number;
  q_between_p?: number | null;
  slope?: number;
  slope_p?: number;
  slope_ci_low?: number;
  slope_ci_high?: number;
  r2_pct?: number;
  egger_intercept?: number;
  egger_p: number;
  begg_tau?: number;
  begg_p?: number | null;
  trim_fill_missing?: number;
  interpretation?: string;
}

type InputType = "ci" | "se" | "raw";
type Mode = "analyze" | "subgroup" | "regression" | "bias";

interface Row {
  label: string; effect: string; ci_low: string; ci_high: string; se: string;
  e1: string; n1: string; e2: string; n2: string; subgroup: string; moderator: string;
}

const blank = (): Row => ({ label: "", effect: "", ci_low: "", ci_high: "", se: "", e1: "", n1: "", e2: "", n2: "", subgroup: "", moderator: "" });

const SAMPLE: Row[] = [
  { ...blank(), label: "Trial A", effect: "0.75", ci_low: "0.55", ci_high: "1.02", subgroup: "Europe", moderator: "2010" },
  { ...blank(), label: "Trial B", effect: "0.82", ci_low: "0.60", ci_high: "1.12", subgroup: "Europe", moderator: "2013" },
  { ...blank(), label: "Trial C", effect: "1.10", ci_low: "0.80", ci_high: "1.51", subgroup: "US", moderator: "2016" },
  { ...blank(), label: "Trial D", effect: "0.68", ci_low: "0.45", ci_high: "1.03", subgroup: "US", moderator: "2019" },
];

export default function MetaPanel() {
  const showGrid = useStore((s) => s.showGrid);
  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const forestRef = useRef<PlotCaptureHandle | null>(null);
  const bubbleRef = useRef<PlotCaptureHandle | null>(null);
  const funnelRef = useRef<PlotCaptureHandle | null>(null);

  // Persisted with the result. The study table is the whole input here, and
  // `mode` decides how the result is drawn: a restored bias result read under
  // the default "analyze" layout would throw, as it carries no pooled estimate.
  const [measure, setMeasure] = usePersistedPanelState("meta", "measure", "OR");
  const [tau2Method, setTau2Method] = usePersistedPanelState("meta", "tau2Method", "DL");
  const [inputType, setInputType] = usePersistedPanelState<InputType>("meta", "inputType", "ci");
  const [rows, setRows] = usePersistedPanelState<Row[]>("meta", "rows", SAMPLE);
  const [mode, setMode] = usePersistedPanelState<Mode>("meta", "mode", "analyze");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");

  const logScale = ["OR", "RR", "HR"].includes(measure);
  const setCell = (i: number, key: keyof Row, val: string) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, [key]: val } : r)));
  const addRow = () => setRows((rs) => [...rs, blank()]);
  const delRow = (i: number) => setRows((rs) => rs.filter((_, j) => j !== i));

  const parsePaste = () => {
    const lines = pasteText.trim().split(/\r?\n/).filter(Boolean);
    const out: Row[] = lines.map((ln) => {
      const c = ln.split(/\t|,|;/).map((s) => s.trim());
      const r = blank();
      r.label = c[0] ?? "";
      if (inputType === "ci") { r.effect = c[1] ?? ""; r.ci_low = c[2] ?? ""; r.ci_high = c[3] ?? ""; r.subgroup = c[4] ?? ""; r.moderator = c[5] ?? ""; }
      else if (inputType === "se") { r.effect = c[1] ?? ""; r.se = c[2] ?? ""; r.subgroup = c[3] ?? ""; r.moderator = c[4] ?? ""; }
      else { r.e1 = c[1] ?? ""; r.n1 = c[2] ?? ""; r.e2 = c[3] ?? ""; r.n2 = c[4] ?? ""; r.subgroup = c[5] ?? ""; r.moderator = c[6] ?? ""; }
      return r;
    });
    if (out.length) setRows(out);
    setPasteOpen(false); setPasteText("");
  };

  const buildStudies = () => {
    const num = (v: string) => (v.trim() === "" ? undefined : Number(v));
    return rows
      .filter((r) => r.label.trim() !== "")
      .map((r) => {
        const s: Record<string, string | number | undefined> = { label: r.label };
        if (inputType === "ci") { s.effect = num(r.effect); s.ci_low = num(r.ci_low); s.ci_high = num(r.ci_high); }
        else if (inputType === "se") { s.effect = num(r.effect); s.se = num(r.se); }
        else { s.e1 = num(r.e1); s.n1 = num(r.n1); s.e2 = num(r.e2); s.n2 = num(r.n2); }
        if (r.subgroup.trim()) s.subgroup = r.subgroup.trim();
        if (r.moderator.trim()) s.moderator = num(r.moderator);
        return s;
      });
  };

  // Exactly the request body. `mode` picks the endpoint but is left out: it
  // only changes here in run(), together with a new result, and the setter
  // this render hands out would stamp the mode that was on screen before the
  // click, flagging every fresh result stale.
  const runParams = { studies: buildStudies(), measure, tau2Method };
  // Pooling reads only the typed studies, never the dataset, so a cell edit
  // or case filter cannot make it out of date.
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<MetaResult>("meta", runParams, { dependsOnData: false });

  const run = async (m: Mode) => {
    const studies = buildStudies();
    if (studies.length < 2) { setError("Enter at least 2 studies."); return; }
    // After the check, so a refused run cannot switch the layout under the
    // previous mode's result.
    setMode(m);
    setLoading(true); setError(null); setResult(null);
    try {
      const payload = { studies, measure, tau2_method: tau2Method };
      const fn = m === "analyze" ? runMetaAnalyze : m === "subgroup" ? runMetaSubgroup
        : m === "regression" ? runMetaRegression : runMetaBias;
      const res = await fn(payload);
      setResult(res.data);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      const msg = e instanceof Error ? e.message : String(e);
      setError(Array.isArray(detail) ? detail.map((x: { msg?: string }) => x.msg ?? String(x)).join(", ")
        : (typeof detail === "string" ? detail : (msg || "Meta-analysis failed")));
    } finally { setLoading(false); }
  };

  // ── Forest plot (analyze) ──
  const forestPlot = () => {
    if (!result?.studies) return null;
    const studies = result.studies;
    const labels = studies.map((s) => s.label);
    const n = studies.length;
    const yIdx = studies.map((_, i: number) => n - i);
    const data: PlotData[] = [
      {
        type: "scatter", mode: "markers", x: studies.map((s) => s.effect), y: yIdx,
        error_x: { type: "data", symmetric: false,
          array: studies.map((s) => s.ci_high - s.effect),
          arrayminus: studies.map((s) => s.effect - s.ci_low), color: "#6b7280", thickness: 1.2 },
        marker: { color: pal[0], size: studies.map((s) => 6 + 0.4 * s.weight_pct), symbol: "square" },
        text: labels, hovertemplate: "%{text}<br>%{x:.3f}<extra></extra>", name: "Studies",
      },
      // pooled diamond (random)
      {
        type: "scatter", mode: "markers", x: [result.random.effect], y: [0],
        marker: { color: "#dc2626", size: 14, symbol: "diamond" },
        error_x: { type: "data", symmetric: false, array: [result.random.ci_high - result.random.effect],
          arrayminus: [result.random.effect - result.random.ci_low], color: "#dc2626", thickness: 2 },
        name: "Pooled (RE)", hovertemplate: `Pooled ${result.measure} %{x:.3f}<extra></extra>`,
      },
    ];
    return (
      <TitledPlot
        plotRefOut={forestRef}
        storageKey="meta:forest"
        data={data}
        layout={{
          ...baseLayout,
          xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, type: logScale ? "log" : "linear" },
          yaxis: { tickvals: [...yIdx, 0], ticktext: [...labels, "Pooled (RE)"], showgrid: false, automargin: true, range: [-0.8, n + 0.8] },
          shapes: [{ type: "line", x0: result.null_line, x1: result.null_line, yref: "paper", y0: 0, y1: 1, line: { color: "#9ca3af", width: 1, dash: "dash" } }],
          showlegend: false, margin: { t: 36, r: 24, b: 44, l: 110 },
        }}
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        defaultTitle={`Forest plot — pooled ${result.measure}`}
        defaultSubtitle=""
        defaultXAxis={result.measure}
        defaultYAxis="" />
    );
  };

  const bubblePlot = () => {
    if (!result?.points) return null;
    return (
      <TitledPlot
        plotRefOut={bubbleRef}
        storageKey="meta:regression"
        data={[
          { type: "scatter", mode: "markers", x: result.points.map((p) => p.moderator), y: result.points.map((p) => p.effect),
            marker: { color: pal[0], size: result.points.map((p) => p.size), opacity: 0.6 },
            text: result.points.map((p) => p.label), hovertemplate: "%{text}<br>x=%{x}<br>%{y:.3f}<extra></extra>", name: "Studies" },
          { type: "scatter", mode: "lines", x: result.line_x, y: result.line_y, line: { color: "#dc2626", width: 2 }, name: "Fit" },
        ]}
        layout={{
          ...baseLayout,
          xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid },
          yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, type: result.log_scale ? "log" : "linear" },
          showlegend: false, margin: { t: 36, r: 24, b: 44, l: 60 },
        }}
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        defaultTitle={`Meta-regression — ${result.measure} vs moderator`}
        defaultSubtitle=""
        defaultXAxis="Moderator"
        defaultYAxis={result.measure} />
    );
  };

  const funnelPlot = () => {
    if (!result?.funnel) return null;
    const eff = result.funnel.map((f) => f.effect);
    const se = result.funnel.map((f) => f.se);
    return (
      <TitledPlot
        plotRefOut={funnelRef}
        storageKey="meta:funnel"
        data={[
          { type: "scatter", mode: "markers", x: eff, y: se, marker: { color: pal[0], size: 8, opacity: 0.7 },
            text: result.funnel.map((f) => f.label), hovertemplate: "%{text}<br>eff=%{x:.3f}<br>SE=%{y:.3f}<extra></extra>", name: "Studies" },
        ]}
        layout={{
          ...baseLayout,
          xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, type: result.log_scale ? "log" : "linear" },
          yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, autorange: "reversed" as const },
          shapes: [{ type: "line", x0: result.pooled_effect, x1: result.pooled_effect, y0: 0, y1: result.se_max, line: { color: "#9ca3af", width: 1, dash: "dash" } }],
          showlegend: false, margin: { t: 36, r: 24, b: 44, l: 60 },
        }}
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        defaultTitle="Funnel plot"
        defaultSubtitle=""
        defaultXAxis={result.measure}
        defaultYAxis="Standard error" />
    );
  };

  const colsFor = (t: InputType): (keyof Row)[] =>
    t === "ci" ? ["label", "effect", "ci_low", "ci_high", "subgroup", "moderator"]
      : t === "se" ? ["label", "effect", "se", "subgroup", "moderator"]
        : ["label", "e1", "n1", "e2", "n2", "subgroup", "moderator"];
  const COL_LABEL: Record<string, string> = {
    label: "Study", effect: "Effect", ci_low: "CI low", ci_high: "CI high", se: "SE",
    e1: "e₁", n1: "n₁", e2: "e₂", n2: "n₂", subgroup: "Subgroup", moderator: "Moderator",
  };
  const cols = colsFor(inputType);

  return (
    <div className="p-4 space-y-3">
      {/* Top controls */}
      <div className="flex flex-wrap items-center gap-3 panel py-2">
        <label className="flex items-center gap-1.5 text-xs text-gray-600">
          Measure
          <select value={measure} onChange={(e) => setMeasure(e.target.value)} className="text-xs border border-gray-300 rounded px-2 py-1 bg-white">
            {["OR", "RR", "HR", "RD", "MD", "SMD"].map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-xs text-gray-600">
          τ²
          <Tip text="DerSimonian-Laird (closed form) or Paule-Mandel (iterative, recommended)." />
          <select value={tau2Method} onChange={(e) => setTau2Method(e.target.value)} className="text-xs border border-gray-300 rounded px-2 py-1 bg-white">
            <option value="DL">DerSimonian-Laird</option>
            <option value="PM">Paule-Mandel</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-xs text-gray-600">
          Input
          <select value={inputType} onChange={(e) => setInputType(e.target.value as InputType)} className="text-xs border border-gray-300 rounded px-2 py-1 bg-white">
            <option value="ci">Effect + 95% CI</option>
            <option value="se">Effect + SE</option>
            <option value="raw">2×2 counts</option>
          </select>
        </label>
        <div className="flex items-center gap-1.5 ml-auto">
          <button onClick={() => setPasteOpen((v) => !v)} className="text-[11px] px-2 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-50">Paste data</button>
          <button onClick={() => setRows(SAMPLE)} className="text-[11px] px-2 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-50">Sample</button>
          <button onClick={addRow} className="text-[11px] px-2 py-1 rounded border border-indigo-300 text-indigo-600 hover:bg-indigo-50">+ Row</button>
        </div>
      </div>

      {pasteOpen && (
        <div className="panel space-y-2">
          <p className="text-[11px] text-gray-500">
            Paste TSV/CSV — columns in order: {cols.map((c) => COL_LABEL[c]).join(", ")}
          </p>
          <textarea value={pasteText} onChange={(e) => setPasteText(e.target.value)} rows={5}
            className="w-full text-xs font-mono border border-gray-300 rounded p-2 focus:outline-none focus:border-indigo-400"
            placeholder={"Trial A\t0.75\t0.55\t1.02\tEurope\t2010"} />
          <button onClick={parsePaste} className="text-xs px-3 py-1 rounded bg-indigo-600 text-white hover:bg-indigo-700">Load rows</button>
        </div>
      )}

      {/* Study editor */}
      <div className="panel overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
              {cols.map((c) => <th key={c} className="text-left px-2 py-1.5 font-medium">{COL_LABEL[c]}</th>)}
              <th className="px-1 py-1.5"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-gray-100">
                {cols.map((c) => (
                  <td key={c} className="px-1 py-0.5">
                    <input value={r[c]} onChange={(e) => setCell(i, c, e.target.value)}
                      className="w-full text-xs border border-gray-200 rounded px-1.5 py-1 focus:outline-none focus:border-indigo-400"
                      placeholder={COL_LABEL[c]} />
                  </td>
                ))}
                <td className="px-1 py-0.5 text-center">
                  <button onClick={() => delRow(i)} className="text-gray-300 hover:text-red-500">✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Action buttons */}
      <div className="flex flex-wrap gap-2">
        {([["analyze", "Forest + pool"], ["subgroup", "Subgroup"], ["regression", "Meta-regression"], ["bias", "Publication bias"]] as const).map(([m, lbl]) => (
          <button key={m} onClick={() => run(m)} disabled={loading}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors disabled:opacity-50 ${
              mode === m && result ? "bg-indigo-600 text-white" : "border border-indigo-300 text-indigo-600 hover:bg-indigo-50"}`}>
            {loading && mode === m ? "Running…" : lbl}
          </button>
        ))}
      </div>
      {error && <p className="text-xs text-red-500">{error}</p>}

      {/* Results */}
      {result && stale && (
        <StaleResultNotice
          reasons={staleWhy}
          onRecompute={() => run(mode)}
          busy={loading}
          what="This meta-analysis"
        />
      )}
      {result && (
        <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_400px] gap-3 items-start">
          <div>
            {mode === "analyze" && forestPlot()}
            {mode === "subgroup" && forestPlot() /* subgroup also renders pooled forest of all studies if present */}
            {mode === "regression" && bubblePlot()}
            {mode === "bias" && funnelPlot()}
          </div>
          <div className="space-y-3">
            {mode === "analyze" && (
              <div className="panel space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-gray-800">Pooled result</h4>
                  {result.export_rows && (
                    <ResultExporter title={`Meta_${result.measure}`} headers={result.export_rows[0]} rows={result.export_rows.slice(1)} />
                  )}
                </div>
                <div className="grid grid-cols-2 gap-1.5">
                  {[
                    ["Random " + result.measure, `${result.random.effect}`],
                    ["95% CI", `${result.random.ci_low}–${result.random.ci_high}`],
                    ["Fixed " + result.measure, `${result.fixed.effect}`],
                    ["I²", `${result.I2_pct}%`],
                    ["τ²", result.tau2],
                    ["Q (p)", `${result.Q} (${result.Q_p})`],
                  ].map(([k, v]) => (
                    <div key={String(k)} className="bg-gray-50 border border-gray-200 rounded p-1.5 text-center">
                      <p className="text-[9px] text-gray-400">{k}</p>
                      <p className="font-semibold text-gray-800 text-xs font-mono">{v}</p>
                    </div>
                  ))}
                </div>
                {result.prediction_low != null && (
                  <div className="bg-indigo-50 border border-indigo-100 rounded px-2 py-1 text-[11px] text-indigo-800">
                    95% prediction interval: {result.prediction_low}–{result.prediction_high}
                  </div>
                )}
              </div>
            )}

            {mode === "subgroup" && result.subgroups && (
              <div className="panel space-y-2">
                <h4 className="text-sm font-semibold text-gray-800">Subgroups</h4>
                <table className="w-full text-[11px]">
                  <thead className="text-gray-400"><tr><th className="text-left px-1">Group</th><th className="text-right px-1">k</th><th className="text-right px-1">{result.measure}</th><th className="text-right px-1">95% CI</th><th className="text-right px-1">I²</th></tr></thead>
                  <tbody>
                    {result.subgroups.map((s) => (
                      <tr key={s.subgroup} className="border-t border-gray-100">
                        <td className="px-1 py-0.5 font-mono">{s.subgroup}</td>
                        <td className="px-1 py-0.5 text-right">{s.k}</td>
                        <td className="px-1 py-0.5 text-right font-mono text-indigo-700">{s.effect}</td>
                        <td className="px-1 py-0.5 text-right font-mono text-gray-500">{s.ci_low}–{s.ci_high}</td>
                        <td className="px-1 py-0.5 text-right">{s.I2_pct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {result.q_between_p != null && (
                  <div className={`text-[11px] px-2 py-1 rounded ${result.q_between_p < 0.05 ? "bg-indigo-50 text-indigo-700" : "bg-gray-50 text-gray-500"}`}>
                    Between-groups Q = {result.q_between}, p = {result.q_between_p}
                  </div>
                )}
              </div>
            )}

            {mode === "regression" && (
              <div className="panel space-y-2">
                <h4 className="text-sm font-semibold text-gray-800">Meta-regression</h4>
                <div className="grid grid-cols-2 gap-1.5">
                  {([["Slope", result.slope], [<i>p</i>, fmtP(Number(result.slope_p))],
                    ["95% CI", `${result.slope_ci_low}–${result.slope_ci_high}`], ["R²", `${result.r2_pct}%`]] as [ReactNode, ReactNode][]).map(([k, v], i) => (
                    <div key={i} className="bg-gray-50 border border-gray-200 rounded p-1.5 text-center">
                      <p className="text-[9px] text-gray-400">{k}</p>
                      <p className="font-semibold text-gray-800 text-xs font-mono">{v}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {mode === "bias" && (
              <div className="panel space-y-2">
                <h4 className="text-sm font-semibold text-gray-800">Publication bias</h4>
                <div className="space-y-1 text-[11px]">
                  <div className={`flex justify-between px-2 py-1 rounded ${result.egger_p < 0.05 ? "bg-amber-50 text-amber-700" : "bg-emerald-50 text-emerald-700"}`}>
                    <span>Egger intercept</span><span className="font-mono">{result.egger_intercept}, <i>p</i>={fmtP(Number(result.egger_p))}</span>
                  </div>
                  {result.begg_p != null && (
                    <div className="flex justify-between px-2 py-1 rounded bg-gray-50 text-gray-600">
                      <span>Begg τ</span><span className="font-mono">{result.begg_tau}, <i>p</i>={fmtP(Number(result.begg_p))}</span>
                    </div>
                  )}
                  <div className="flex justify-between px-2 py-1 rounded bg-gray-50 text-gray-600">
                    <span>Trim-and-fill missing</span><span className="font-mono">{result.trim_fill_missing}</span>
                  </div>
                </div>
              </div>
            )}

            {result.interpretation && (
              <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3 py-2 text-xs text-indigo-900 leading-relaxed">
                {result.interpretation}
              </div>
            )}
          </div>
        </div>
        </StaleGuard>
      )}
    </div>
  );
}
