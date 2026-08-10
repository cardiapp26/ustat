import { useState, useRef, useMemo } from "react";
import Plot from "../PlotComponent";
import TitledPlot from "./TitledPlot";
import { useStore, paletteOf, isNumericKind } from "../store";
import { runRCS, runCoxRCS } from "../api";
import { Tip, InfoBanner } from "./Tip";
import ResultExporter from "./ResultExporter";
import { MissingGuard, type ImputationStrategy } from "./MissingGuard";
import { fmtP } from "../lib/format";
import type { PlotData, PlotLayout, PlotCaptureHandle } from "../lib/plotTypes";

// Geometry only — background, font and colourway come from the global chart
// theme at render, which this panel used to ignore entirely.
const PLOT_LAYOUT = {
  margin: { t: 30, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: "#e5e7eb" },
  yaxis: { gridcolor: "#e5e7eb" },
};

// ── Types ────────────────────────────────────────────────────────────────────

interface SurfaceData {
  x_col: string; y_col: string;
  x: number[]; y: number[];
  hr: number[][];
  ref: Record<string, number>;
}

interface SplineTermState {
  column: string;
  n_knots: number;
  knot_positions: string;
  ref_value: string;
}

interface CoxRCSCoefficient {
  name: string; coef: number; hr: number; se: number;
  z: number | null; p: number | null; ci_low: number; ci_high: number;
}
interface CoxRCSCurve {
  column: string; x: number[]; hr: number[];
  lower: number[]; upper: number[]; knots: number[]; ref: number;
}
interface CoxRCSResult {
  coefficients: CoxRCSCoefficient[];
  curves_1d?: CoxRCSCurve[];
  surface_2d: SurfaceData | null;
  interaction: null | { lr_stat?: number; df?: number; p?: number; error?: string };
  nonlinearity?: Record<string, { wald: number | null; df: number; p: number | null }>;
  n: number;
  n_events: number;
  concordance?: number;
  aic?: number;
}

interface RCSCovariateSummary {
  name: string;
  effect: number | null;
  coef: number | null;
}
interface RCSCrude {
  x_values: number[];
  or_values: number[];
  ci_low: number[];
  ci_high: number[];
  nonlinearity_p: number | null;
}
interface RCSInteraction {
  lr_stat?: number;
  df?: number;
  p?: number | null;
  covariates?: string[];
  error?: string;
}
/** Univariate RCS result. Field set is server-driven and partly optional. */
interface RCSResult {
  x_values: number[];
  or_values: number[];
  ci_low: number[];
  ci_high: number[];
  x_data: number[];
  predictor: string;
  outcome?: string;
  model_type?: string;
  duration_col?: string;
  event_col?: string;
  knots: number[];
  n_knots: number;
  ref_value: number;
  n: number;
  n_total?: number;
  n_events?: number | null;
  n_excluded?: number | null;
  aic?: number | null;
  nonlinearity_p?: number | null;
  nonlinearity_df?: number | string | null;
  crude?: RCSCrude | null;
  covariates_used?: string[];
  covariates_requested?: string[];
  covariates_summary?: RCSCovariateSummary[];
  interaction?: RCSInteraction | null;
}

// ── Sub-components ───────────────────────────────────────────────────────────

function HRSurfaceCard({ surface }: { surface: SurfaceData }) {
  const [view, setView] = useState<"contour" | "surface3d">("contour");
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="text-xs font-semibold text-gray-700">
          HR over {surface.x_col} × {surface.y_col} (other covariates at mean)
        </div>
        <div className="flex gap-1">
          {(["contour", "surface3d"] as const).map((v) => (
            <button key={v} onClick={() => setView(v)}
              className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${view === v ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
              {v === "contour" ? "2D contour ★" : "3D surface"}
            </button>
          ))}
        </div>
      </div>
      <TitledPlot
        storageKey={`crx-surface:${surface.x_col}:${surface.y_col}:${view}`}
        defaultTitle={`HR surface: ${surface.x_col} × ${surface.y_col}`}
        defaultSubtitle={`Reference: ${surface.x_col} = ${surface.ref[surface.x_col]}, ${surface.y_col} = ${surface.ref[surface.y_col]} · other covariates at their mean`}
        defaultXAxis={surface.x_col}
        defaultYAxis={view === "contour" ? surface.y_col : ""}
        data={[
          view === "contour"
            ? {
                z: surface.hr, x: surface.x, y: surface.y,
                type: "contour", colorscale: "RdBu", reversescale: true,
                contours: { coloring: "heatmap", showlabels: true },
                colorbar: { title: { text: "HR" } },
                hovertemplate: `${surface.x_col}=%{x:.2f}<br>${surface.y_col}=%{y:.2f}<br>HR=%{z:.2f}<extra></extra>`,
              } as PlotData
            : {
                z: surface.hr, x: surface.x, y: surface.y,
                type: "surface", colorscale: "RdBu", reversescale: true,
                colorbar: { title: { text: "HR" } },
                contours: { z: { show: true, usecolormap: true, project: { z: true } } },
                hovertemplate: `${surface.x_col}=%{x:.2f}<br>${surface.y_col}=%{y:.2f}<br>HR=%{z:.2f}<extra></extra>`,
              } as PlotData,
        ]}
        layout={
          view === "contour"
            ? { height: 380, margin: { l: 60, r: 30, t: 10, b: 50 }, xaxis: { title: { text: surface.x_col } }, yaxis: { title: { text: surface.y_col } } }
            : ({ height: 460, margin: { l: 0, r: 0, t: 10, b: 0 }, scene: { xaxis: { title: { text: surface.x_col } }, yaxis: { title: { text: surface.y_col } }, zaxis: { title: { text: "Hazard Ratio" }, type: "log" }, camera: { eye: { x: 1.4, y: -1.4, z: 0.9 } } } } as PlotLayout)
        }
        config={{ displaylogo: false, responsive: true }}
        style={{ width: "100%" }}
      />
      {view === "surface3d" && (
        <div className="text-[10px] text-gray-500">Drag to rotate, scroll to zoom.</div>
      )}
    </div>
  );
}

function CoxRCSResultPanel({ result }: { result: CoxRCSResult }) {
  const coefs = result.coefficients;
  const curves = result.curves_1d ?? [];
  const surface = result.surface_2d;
  const interaction = result.interaction;
  const nonlinearity = result.nonlinearity ?? {};

  return (
    <div className="panel space-y-4">
      <div className="flex flex-wrap items-baseline gap-3 border-b border-gray-100 pb-2">
        <h4 className="font-semibold text-gray-900">Cox proportional hazards (RCS)</h4>
        <span className="text-xs text-gray-500">
          <i>n</i> = {result.n}, events = {result.n_events}, C-index = {result.concordance?.toFixed(3) ?? "—"}
          {result.aic != null && <>, AIC = {result.aic.toFixed(1)}</>}
        </span>
      </div>

      {Object.keys(nonlinearity).length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Nonlinearity (Wald)</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(nonlinearity).map(([col, nl]) => (
              <span key={col} className={`text-xs px-2 py-1 rounded border ${nl.p != null && nl.p < 0.05 ? "bg-indigo-50 text-indigo-700 border-indigo-200" : "bg-gray-50 text-gray-600 border-gray-200"}`}>
                <b>{col}</b>: χ²({nl.df}) = {nl.wald?.toFixed(2) ?? "—"}, <i>p</i> = {fmtP(nl.p)}
              </span>
            ))}
          </div>
        </div>
      )}

      {interaction && (
        <div className={`text-xs p-2 rounded border ${interaction.p != null && interaction.p < 0.05 ? "bg-amber-50 text-amber-900 border-amber-200" : "bg-gray-50 text-gray-700 border-gray-200"}`}>
          <span className="font-semibold uppercase tracking-wider text-[10px] block mb-0.5">RCS × RCS interaction (LR test)</span>
          {interaction.error ? (
            <span className="text-red-600">{interaction.error}</span>
          ) : (
            <>χ²({interaction.df}) = {interaction.lr_stat?.toFixed(2)}, <i>p</i> = {fmtP(interaction.p)}</>
          )}
        </div>
      )}

      {curves.map((c, idx) => (
        <div key={c.column} className="space-y-1">
          <div className="text-xs font-semibold text-gray-700">{c.column} — HR vs reference (other covariates at mean)</div>
          <Plot
            data={[
              { x: c.x, y: c.upper, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
              { x: c.x, y: c.lower, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(99,102,241,0.15)", hoverinfo: "skip", showlegend: false },
              { x: c.x, y: c.hr, type: "scatter", mode: "lines", line: { color: "#4f46e5", width: 2 }, name: "HR", hovertemplate: `${c.column}=%{x:.2f}<br>HR=%{y:.2f}<extra></extra>` },
              { x: c.knots ?? [], y: (c.knots ?? []).map(() => 1), type: "scatter", mode: "markers", marker: { color: "#4f46e5", size: 8, symbol: "circle" }, name: "knots", hoverinfo: "x" },
            ]}
            layout={{
              height: 280, margin: { l: 50, r: 20, t: 10, b: 40 },
              xaxis: { title: { text: c.column } },
              yaxis: { title: { text: "Hazard Ratio" }, type: "log" as const },
              shapes: [{ type: "line", x0: c.ref, x1: c.ref, y0: 0.01, y1: 100, yref: "y", line: { dash: "dot", color: "#9ca3af" } }],
              showlegend: false,
            }}
            config={{ displaylogo: false, responsive: true }}
            style={{ width: "100%" }}
            useResizeHandler
            key={`crx-curve-${idx}`}
          />
          <div className="text-[10px] text-gray-500">Reference = {c.ref}. Knots at: {(c.knots ?? []).join(", ")}.</div>
        </div>
      ))}

      {surface && <HRSurfaceCard surface={surface} />}

      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200 text-gray-600">
              <th className="text-left px-2 py-1">Term</th>
              <th className="text-right px-2 py-1">β</th>
              <th className="text-right px-2 py-1">HR</th>
              <th className="text-right px-2 py-1">95% CI</th>
              <th className="text-right px-2 py-1">SE</th>
              <th className="text-right px-2 py-1">z</th>
              <th className="text-right px-2 py-1"><i>p</i></th>
            </tr>
          </thead>
          <tbody>
            {coefs.map((c) => (
              <tr key={c.name} className="border-b border-gray-100">
                <td className="px-2 py-1 font-mono text-[10px] text-gray-700">{c.name}</td>
                <td className="text-right px-2 py-1">{c.coef.toFixed(3)}</td>
                <td className="text-right px-2 py-1">{c.hr.toFixed(3)}</td>
                <td className="text-right px-2 py-1">{c.ci_low.toFixed(2)} – {c.ci_high.toFixed(2)}</td>
                <td className="text-right px-2 py-1">{c.se.toFixed(3)}</td>
                <td className="text-right px-2 py-1">{c.z?.toFixed(2) ?? "—"}</td>
                <td className={`text-right px-2 py-1 ${c.p != null && c.p < 0.05 ? "font-bold text-indigo-600" : "text-gray-500"}`}>{fmtP(c.p)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SplineTermCard({ label, term, onChange, numCols }: {
  label: string; term: SplineTermState; onChange: (next: SplineTermState) => void; numCols: string[];
}) {
  const patch = (p: Partial<SplineTermState>) => onChange({ ...term, ...p });
  return (
    <div className="border border-gray-200 rounded p-2 space-y-1.5 bg-gray-50">
      <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">{label}</div>
      <div>
        <label className="text-[10px] text-gray-400 block">Column</label>
        <select className="select w-full text-xs py-0.5" value={term.column}
          onChange={(e) => patch({ column: e.target.value })}>
          {numCols.map((c) => <option key={c}>{c}</option>)}
        </select>
      </div>
      <div>
        <label className="text-[10px] text-gray-400 block">Knots</label>
        <div className="flex gap-1">
          {[3, 4, 5].map((k) => (
            <button key={k} onClick={() => patch({ n_knots: k })}
              className={`flex-1 py-0.5 text-[11px] rounded border transition-colors ${term.n_knots === k ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-600 hover:bg-white"}`}>
              {k}{k === 4 ? " ★" : ""}
            </button>
          ))}
        </div>
      </div>
      <div>
        <label className="text-[10px] text-gray-400 block">Custom knot positions (blank = Harrell percentiles)</label>
        <input type="text" placeholder={`e.g. 70, 100, 130, 160 (${term.n_knots} values)`}
          value={term.knot_positions}
          onChange={(e) => patch({ knot_positions: e.target.value })}
          className="select w-full text-[11px] py-0.5" />
      </div>
      <div>
        <label className="text-[10px] text-gray-400 block">Reference value (blank = median)</label>
        <input type="number" placeholder="(median)"
          value={term.ref_value}
          onChange={(e) => patch({ ref_value: e.target.value })}
          className="select w-full text-[11px] py-0.5" />
      </div>
    </div>
  );
}

// ── Main panel ──────────────────────────────────────────────────────────────

export default function RCSPanel() {
  // No non-null assertion on session: async setState after the session is
  // cleared (session delete, test teardown) would otherwise re-render into
  // `session.columns` and crash the whole tab.
  const session = useStore((s) => s.session);
  const showGrid = useStore((s) => s.showGrid);
  const plotTheme = useStore((s) => s.plotTheme);
  const themedLayout: Record<string, unknown> = {
    ...PLOT_LAYOUT,
    paper_bgcolor: "transparent",
    plot_bgcolor: plotTheme.plotBg,
    font: { family: plotTheme.fontFamily, color: "#374151", size: plotTheme.fontSize },
    colorway: paletteOf(plotTheme),
  };

  // Columns flagged "exclude from analysis" (e.g. NAME, row-id) never appear
  // in any picker, including covariates.
  const numCols = useMemo(
    () => (session?.columns ?? []).filter((c) => isNumericKind(c.kind) && !c.analysis_excluded).map((c) => c.name),
    [session?.columns],
  );
  const binaryCols = useMemo(
    () => {
      const cols = session?.columns;
      if (!session?.preview?.length || !cols) return [] as string[];
      return cols
        .filter((c) => {
          if (c.analysis_excluded) return false;
          const vals = session.preview!.map((r) => (r as Record<string, unknown>)[c.name]).filter((v) => v != null);
          const uniq = new Set(vals.map(Number));
          return uniq.size <= 2 && [...uniq].every((v) => v === 0 || v === 1);
        })
        .map((c) => c.name);
    },
    [session?.columns, session?.preview],
  );
  const allCols = useMemo(() => (session?.columns ?? []).filter((c) => !c.analysis_excluded).map((c) => c.name), [session?.columns]);

  // ── Mode: "rcs" (univariate) or "cox_rcs" (multivariable) ─────────────────
  const [mode, setMode] = useState<"rcs" | "cox_rcs">("rcs");

  // ── Shared state ──────────────────────────────────────────────────────────
  const [result, setResult] = useState<RCSResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const rcsPlotRef = useRef<PlotCaptureHandle | null>(null);

  // ── RCS univariate state ──────────────────────────────────────────────────
  const [rcsPredictor, setRcsPredictor] = useState(numCols[0] ?? "");
  const [rcsOutcome, setRcsOutcome] = useState(numCols[1] ?? numCols[0] ?? "");
  const [rcsNKnots, setRcsNKnots] = useState(4);
  const [rcsRefValue, setRcsRefValue] = useState("");
  const [rcsCovariates, setRcsCovariates] = useState<string[]>([]);
  const [rcsInteractionCov, setRcsInteractionCov] = useState<string[]>([]);
  const [rcsLogScale, setRcsLogScale] = useState(true);
  const [rcsShowData, setRcsShowData] = useState(true);
  const [rcsOutcomeType, setRcsOutcomeType] = useState<"logistic" | "linear" | "cox">("cox");
  const [rcsCoxDuration, setRcsCoxDuration] = useState(numCols[0] ?? "");
  const [rcsCoxEvent, setRcsCoxEvent] = useState(binaryCols[0] ?? numCols[1] ?? "");
  const [rcsKnotMode, setRcsKnotMode] = useState<"harrell" | "custom">("harrell");
  const [rcsCustomKnots, setRcsCustomKnots] = useState("");

  // ── Cox-RCS multivariable state ───────────────────────────────────────────
  const [crxDuration, setCrxDuration] = useState(numCols[0] ?? "");
  const [crxEvent, setCrxEvent] = useState(binaryCols[0] ?? numCols[1] ?? "");
  const [crxTerm1, setCrxTerm1] = useState<SplineTermState>({ column: numCols[0] ?? "", n_knots: 4, knot_positions: "", ref_value: "" });
  const [crxTerm2, setCrxTerm2] = useState<SplineTermState>({ column: numCols[1] ?? "", n_knots: 4, knot_positions: "", ref_value: "" });
  const [crxUseTerm2, setCrxUseTerm2] = useState(false);
  const [crxInteraction, setCrxInteraction] = useState(false);
  const [crxCovariates, setCrxCovariates] = useState<string[]>([]);

  const sid = session?.session_id ?? "";

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      let res: Awaited<ReturnType<typeof runRCS>>;
      if (mode === "rcs") {
        const customKnotsArr = rcsKnotMode === "custom"
          ? rcsCustomKnots.split(/[,\s]+/).filter(Boolean).map(Number).filter((n) => !Number.isNaN(n))
          : undefined;
        if (rcsKnotMode === "custom" && customKnotsArr && customKnotsArr.length !== rcsNKnots) {
          throw new Error(`Custom knots: provide exactly ${rcsNKnots} numeric values (got ${customKnotsArr?.length ?? 0}).`);
        }
        const payload: Record<string, unknown> = {
          session_id: sid,
          predictor: rcsPredictor,
          covariates: rcsCovariates,
          n_knots: rcsNKnots,
          ref_value: rcsRefValue !== "" ? parseFloat(rcsRefValue) : undefined,
          model_type: rcsOutcomeType,
          knot_positions: customKnotsArr,
          interaction_covariates: rcsInteractionCov.length > 0
            ? rcsInteractionCov.filter((c) => rcsCovariates.includes(c))
            : undefined,
        };
        if (rcsOutcomeType === "cox") {
          payload.duration_col = rcsCoxDuration;
          payload.event_col = rcsCoxEvent;
        } else {
          payload.outcome = rcsOutcome;
        }
        res = await runRCS(payload);
      } else {
        const buildTerm = (t: SplineTermState) => {
          const kp = t.knot_positions.split(/[,\s]+/).filter(Boolean).map(Number).filter((n) => !Number.isNaN(n));
          if (kp.length > 0 && kp.length !== t.n_knots) {
            throw new Error(`Custom knots for '${t.column}': provide exactly ${t.n_knots} numeric values (got ${kp.length}).`);
          }
          return {
            column: t.column,
            n_knots: t.n_knots,
            knot_positions: kp.length > 0 ? kp : undefined,
            ref_value: t.ref_value !== "" ? parseFloat(t.ref_value) : undefined,
          };
        };
        const terms = [buildTerm(crxTerm1)];
        if (crxUseTerm2) terms.push(buildTerm(crxTerm2));
        res = await runCoxRCS({
          session_id: sid,
          duration_col: crxDuration,
          event_col: crxEvent,
          spline_terms: terms,
          covariates: crxCovariates,
          include_interaction: crxInteraction && terms.length === 2,
          imputation,
        });
      }
      setResult(res.data as RCSResult);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : e instanceof Error ? e.message : "Unknown error",
      );
    } finally { setLoading(false); }
  };

  return (
    <div className="flex gap-4">
      {/* ── Left sidebar ── */}
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">RCS Model</h3>
          {([
            ["rcs", "RCS Dose-Response", "Restricted Cubic Splines — models non-linear (U/J-shaped) relationships between a continuous predictor and a binary, continuous, OR time-to-event outcome. Supports custom knots, covariates, and spline × covariate interaction."],
            ["cox_rcs", "Cox-RCS (multivariable)", "Multivariable Cox proportional hazards with 1 or 2 RCS terms, additive covariates, and an optional RCS × RCS interaction test (LR test). For survival analyses like 'Surv(time,event) ~ rcs(LDL,4) * rcs(AGE,4) + covariates'."],
          ] as const).map(([v, l, desc]) => (
            <label key={v} className="flex items-start gap-2 cursor-pointer group">
              <input type="radio" name="rcsModel" value={v} checked={mode === v}
                onChange={() => { setMode(v); setResult(null); setError(null); }}
                className="accent-indigo-500 mt-0.5" />
              <span className="text-sm text-gray-700 leading-tight">
                {l}
                <Tip text={desc} wide />
              </span>
            </label>
          ))}
        </div>

        <div className="panel space-y-3">
          {mode === "rcs" ? (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Predictor (continuous)</label>
                <select className="select w-full" value={rcsPredictor} onChange={(e) => setRcsPredictor(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Outcome type <Tip text="Logistic: binary 0/1 outcome → OR. Linear: continuous outcome → mean difference. Cox: time-to-event with duration + event columns → HR." wide />
                </label>
                <div className="flex gap-1">
                  {(["logistic", "linear", "cox"] as const).map((t) => (
                    <button key={t} onClick={() => setRcsOutcomeType(t)}
                      className={`flex-1 py-1 text-[11px] rounded border transition-colors ${rcsOutcomeType === t ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-600 hover:bg-gray-50"}`}>
                      {t === "cox" ? "Cox" : t === "logistic" ? "Logistic" : "Linear"}
                    </button>
                  ))}
                </div>
              </div>
              {rcsOutcomeType === "cox" ? (
                <>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Duration column</label>
                    <select className="select w-full" value={rcsCoxDuration} onChange={(e) => setRcsCoxDuration(e.target.value)}>
                      {numCols.map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">
                      Event column (0/1)
                      {binaryCols.length === 0 && <span className="ml-1 text-[10px] text-amber-600">⚠ no binary 0/1 column detected — recode one in the Dictionary</span>}
                    </label>
                    <select className="select w-full" value={rcsCoxEvent} onChange={(e) => setRcsCoxEvent(e.target.value)}>
                      {(binaryCols.length > 0 ? binaryCols : numCols).map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>
                </>
              ) : (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">
                    Outcome {rcsOutcomeType === "logistic" ? "(binary 0/1)" : "(continuous)"}
                    {rcsOutcomeType === "logistic" && binaryCols.length === 0 && (
                      <span className="ml-1 text-[10px] text-amber-600">⚠ no binary 0/1 column</span>
                    )}
                  </label>
                  <select className="select w-full" value={rcsOutcome} onChange={(e) => setRcsOutcome(e.target.value)}>
                    {(rcsOutcomeType === "logistic"
                      ? (binaryCols.length > 0 ? binaryCols : allCols)
                      : rcsOutcomeType === "linear" ? numCols : allCols
                    ).map((c) => <option key={c}>{c}</option>)}
                  </select>
                </div>
              )}
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Knots <Tip text="k knots ⇒ k−1 spline parameters (1 linear + k−2 non-linear). Budget by events-per-variable (EPV ≥ 10 ideal, ≥ 5 borderline). 17 events ⇒ 3 knots (2 params, EPV 8.5, safe); 4 knots (3 params, EPV 5.7, borderline); 5 knots (4 params, EPV 4.25, overfit risk). Harrell percentiles: 3→[10,50,90], 4→[5,35,65,95], 5→[5,27.5,50,72.5,95]." wide />
                </label>
                <div className="flex gap-2">
                  {[3, 4, 5].map((k) => (
                    <button key={k} onClick={() => setRcsNKnots(k)}
                      className={`flex-1 py-1 text-xs rounded border transition-colors ${rcsNKnots === k ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-600 hover:bg-gray-50"}`}>
                      {k}{k === 4 ? " ★" : ""}
                    </button>
                  ))}
                </div>
                {/* Knot-budgeting hint: each row maps a knot count to the
                    parameter count, the EPV at n_events=17 (the sample
                    size that drove this recommendation), and a safety
                    label so the user can pick the right knot count for
                    their actual event count. Selected row is highlighted. */}
                <div className="mt-1.5 rounded border border-indigo-100 bg-indigo-50/40 text-[10px] text-gray-700 leading-tight">
                  <div className="px-2 py-1 border-b border-indigo-100 text-[9px] font-semibold text-indigo-700 uppercase tracking-wider">
                    Knot budgeting (EPV rule, n_events = 17 reference)
                  </div>
                  {[
                    { k: 3, terms: 2, epv: "8.5",  label: "safe",       color: "text-emerald-700" },
                    { k: 4, terms: 3, epv: "5.7",  label: "borderline", color: "text-amber-700" },
                    { k: 5, terms: 4, epv: "4.25", label: "overfit risk", color: "text-rose-700" },
                  ].map((r) => (
                    <div key={r.k}
                      className={`flex items-center justify-between px-2 py-0.5 ${rcsNKnots === r.k ? "bg-indigo-100/60 font-semibold" : ""}`}>
                      <span><b>{r.k}</b> knots ⇒ <b>{r.terms}</b> params · EPV ≈ <span className="font-mono">{r.epv}</span></span>
                      <span className={r.color}>{r.label}</span>
                    </div>
                  ))}
                  <div className="px-2 py-1 border-t border-indigo-100 text-[9px] text-gray-500">
                    Default placement: Harrell&apos;s 10th / 50th / 90th percentiles (3 knots).
                  </div>
                </div>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Knot positions <Tip text="Harrell percentiles are the standard reference method. Custom positions are useful for clinically meaningful cut-points (e.g. LDL 70, 100, 130, 160 mg/dL) — typically reported as a sensitivity analysis." wide />
                </label>
                <div className="flex gap-1 mb-1">
                  {(["harrell", "custom"] as const).map((m) => (
                    <button key={m} onClick={() => setRcsKnotMode(m)}
                      className={`flex-1 py-1 text-[11px] rounded border transition-colors ${rcsKnotMode === m ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-600 hover:bg-gray-50"}`}>
                      {m === "harrell" ? "Harrell percentiles" : "Custom"}
                    </button>
                  ))}
                </div>
                {rcsKnotMode === "custom" && (
                  <input type="text" placeholder={`e.g. 70, 100, 130, 160 (${rcsNKnots} values)`}
                    value={rcsCustomKnots} onChange={(e) => setRcsCustomKnots(e.target.value)}
                    className="select w-full text-xs py-1" />
                )}
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Reference value <Tip text="The effect = 1.0 (or 0 for linear) reference point on the X-axis. Leave blank to use the median." />
                </label>
                <input type="number" placeholder="(median)" value={rcsRefValue}
                  onChange={(e) => setRcsRefValue(e.target.value)}
                  className="select w-full text-xs py-1" />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Covariates (optional)
                  <Tip wide text="Numeric covariates enter as linear adjustment terms. Categorical / text covariates are dummy-coded on the server (drop_first=True) — pick e.g. SEX, DM, HT directly. Tick the small × box on the right of a covariate to ALSO interact it with the spline, i.e. test whether the dose-response shape differs across its levels (e.g. does the LDL curve differ by SEX?). An LR test will appear in the result card." />
                </label>
                <div className="max-h-32 overflow-y-auto space-y-1">
                  {allCols
                    .filter((c) => {
                      // Exclude the predictor and the active outcome columns
                      // only — in Cox mode the (unused) rcsOutcome default must
                      // not hide a valid covariate like AGE.
                      if (c === rcsPredictor) return false;
                      return rcsOutcomeType === "cox"
                        ? c !== rcsCoxDuration && c !== rcsCoxEvent
                        : c !== rcsOutcome;
                    })
                    .map((c) => {
                      const kind = (session?.columns ?? []).find((col) => col.name === c)?.kind ?? "numeric";
                      const isNum = kind === "numeric";
                      const isSelected = rcsCovariates.includes(c);
                      const isInteracting = rcsInteractionCov.includes(c);
                      return (
                        <label key={c} className="flex items-center gap-1.5 text-xs cursor-pointer">
                          <input type="checkbox" checked={isSelected}
                            onChange={() => setRcsCovariates((prev) => {
                              const next = prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c];
                              if (!next.includes(c)) setRcsInteractionCov((p) => p.filter((x) => x !== c));
                              return next;
                            })}
                            className="accent-indigo-500" />
                          <span className="text-gray-700 truncate flex-1">{c}</span>
                          <span className={`text-[9px] px-1 rounded ${isNum ? "bg-blue-50 text-blue-600" : "bg-purple-50 text-purple-600"}`}>
                            {isNum ? "N" : "C"}
                          </span>
                          <label className={`flex items-center gap-0.5 text-[9px] px-1 rounded border cursor-pointer transition-colors
                            ${!isSelected ? "border-gray-200 text-gray-300 cursor-not-allowed"
                              : isInteracting ? "border-amber-300 bg-amber-50 text-amber-700"
                              : "border-gray-300 text-gray-500 hover:border-amber-300 hover:text-amber-600"}`}
                            title={isSelected ? "Add a spline × this covariate interaction (LR test in result)" : "Tick the covariate first to enable interaction"}
                            onClick={(e) => e.stopPropagation()}
                          >
                            <input type="checkbox"
                              disabled={!isSelected}
                              checked={isInteracting}
                              onChange={() => setRcsInteractionCov((prev) => prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c])}
                              className="accent-amber-500 w-2.5 h-2.5" />
                            ×spl
                          </label>
                        </label>
                      );
                    })}
                </div>
              </div>
            </>
          ) : (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Duration column</label>
                <select className="select w-full" value={crxDuration} onChange={(e) => setCrxDuration(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Event column (0/1)
                  {binaryCols.length === 0 && <span className="ml-1 text-[10px] text-amber-600">⚠ no binary 0/1 column detected</span>}
                </label>
                <select className="select w-full" value={crxEvent} onChange={(e) => setCrxEvent(e.target.value)}>
                  {(binaryCols.length > 0 ? binaryCols : numCols).map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <SplineTermCard label="Spline term 1" term={crxTerm1} onChange={setCrxTerm1} numCols={numCols} />
              {crxUseTerm2 ? (
                <>
                  <SplineTermCard label="Spline term 2" term={crxTerm2} onChange={setCrxTerm2} numCols={numCols} />
                  <button
                    onClick={() => { setCrxUseTerm2(false); setCrxInteraction(false); }}
                    className="w-full text-[11px] py-1 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-600 hover:border-red-300 transition-colors"
                  >
                    Remove second spline term
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setCrxUseTerm2(true)}
                  className="w-full text-[11px] py-1 rounded border border-dashed border-indigo-300 text-indigo-600 hover:bg-indigo-50 transition-colors"
                >
                  + Add second spline term (rcs × rcs)
                </button>
              )}
              {crxUseTerm2 && (
                <label className="flex items-start gap-2 cursor-pointer p-2 bg-amber-50 border border-amber-200 rounded">
                  <input type="checkbox" className="accent-amber-600 mt-0.5"
                    checked={crxInteraction} onChange={(e) => setCrxInteraction(e.target.checked)} />
                  <div className="text-xs leading-tight">
                    <span className="font-semibold text-amber-900">Include RCS × RCS interaction</span>
                    <span className="block text-[10px] text-amber-700">Tensor-product of basis columns. Adds an LR test (full vs main-effects-only) and an HR contour plot.</span>
                  </div>
                </label>
              )}
              <div>
                <label className="text-xs text-gray-400 block mb-1">Linear covariates (optional)</label>
                <div className="max-h-32 overflow-y-auto space-y-1">
                  {allCols.filter((c) => c !== crxDuration && c !== crxEvent && c !== crxTerm1.column && (!crxUseTerm2 || c !== crxTerm2.column)).map((c) => (
                    <label key={c} className="flex items-center gap-2 text-xs cursor-pointer">
                      <input type="checkbox" checked={crxCovariates.includes(c)}
                        onChange={() => setCrxCovariates((prev) => prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c])}
                        className="accent-indigo-500" />
                      <span className="text-gray-700 truncate">{c}</span>
                    </label>
                  ))}
                </div>
              </div>
            </>
          )}

          <MissingGuard
            sessionId={sid}
            columns={mode === "rcs"
              ? [rcsPredictor,
                 ...(rcsOutcomeType === "cox" ? [rcsCoxDuration, rcsCoxEvent] : [rcsOutcome]),
                 ...rcsCovariates]
              : [crxDuration, crxEvent,
                 crxTerm1.column,
                 ...(crxUseTerm2 ? [crxTerm2.column] : []),
                 ...crxCovariates]}
            imputation={imputation}
            onImputation={setImputation}
          >
            <button onClick={run} disabled={loading}
              className="w-full py-2 rounded-lg text-sm font-medium bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 transition-colors">
              {loading ? "Running…" : `Run ${mode === "rcs" ? "RCS" : "Cox-RCS"}`}
            </button>
          </MissingGuard>
        </div>
      </div>

      {/* ── Results pane ── */}
      <div className="flex-1 min-w-0 space-y-4 overflow-y-auto">
        {error && (
          <div className="bg-red-50 text-red-700 p-4 rounded-xl text-sm">{error}</div>
        )}

        {result && mode === "cox_rcs" && (
          <CoxRCSResultPanel result={result as unknown as CoxRCSResult} />
        )}

        {result && mode === "rcs" && Array.isArray(result.x_values) && (() => {
          const mt = (result.model_type as string | undefined) ?? "logistic";
          const eff = mt === "cox"
            ? { label: "Hazard Ratio", abbr: "HR", refValue: 1, axisType: "log" as const }
            : mt === "linear"
            ? { label: "Mean difference", abbr: "Δ", refValue: 0, axisType: "linear" as const }
            : { label: "Odds Ratio", abbr: "OR", refValue: 1, axisType: "log" as const };
          const useLogY = rcsLogScale && eff.axisType === "log";
          const outcomeLabel = mt === "cox"
            ? `${result.duration_col ?? ""} (${result.event_col ?? "event"})`
            : (result.outcome ?? "");
          const modelTitle = mt === "cox" ? "Cox-RCS" : mt === "linear" ? "Linear RCS" : "Logistic RCS";
          return (
            <div className="panel space-y-3">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <h4 className="font-semibold text-gray-900">
                  {result.predictor}
                  {outcomeLabel ? <> &amp; <span className="text-indigo-700">{outcomeLabel}</span></> : null}
                  : {modelTitle}
                </h4>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-3 text-xs text-gray-500">
                    <span><i>n</i> = {result.n}{result.n_events != null ? `, events = ${result.n_events}` : ""}</span>
                    {result.aic != null && <span>AIC = {result.aic?.toFixed(1)}</span>}
                  </div>
                  <ResultExporter
                    title={`RCS_${result.predictor}_${outcomeLabel || "result"}`}
                    headers={["x", eff.abbr, "CI_low", "CI_high"]}
                    rows={(result.x_values as number[]).map((x: number, i: number) => [
                      x.toFixed(4),
                      (result.or_values as number[])[i]?.toFixed(4) ?? "",
                      (result.ci_low as number[])[i]?.toFixed(4) ?? "",
                      (result.ci_high as number[])[i]?.toFixed(4) ?? "",
                    ])}
                    plotRef={rcsPlotRef}
                  />
                </div>
              </div>

              <div className="flex flex-wrap gap-1.5 text-[11px]">
                <span className="text-gray-400">{result.n_knots} knots at:</span>
                {((result.knots as number[]) ?? []).map((k: number, i: number) => (
                  <span key={i} className="bg-indigo-50 border border-indigo-100 text-indigo-600 rounded px-1.5 py-0.5">{k}</span>
                ))}
                <span className="text-gray-400 ml-2">reference = <strong>{result.ref_value}</strong> ({eff.abbr} = {eff.refValue.toFixed(1)})</span>
                {result.nonlinearity_p != null && (
                  <span className={`ml-2 rounded px-1.5 py-0.5 border ${result.nonlinearity_p < 0.05 ? "bg-indigo-50 text-indigo-700 border-indigo-200" : "bg-gray-50 text-gray-600 border-gray-200"}`}
                    title={`Joint Wald test on the ${(result.nonlinearity_df ?? "k-2")} non-linear basis columns`}>
                    non-linearity {result.crude ? "adjusted " : ""}<i>p</i> = {fmtP(result.nonlinearity_p)}
                  </span>
                )}
                {result.crude?.nonlinearity_p != null && (
                  <span className="ml-1 rounded px-1.5 py-0.5 border bg-gray-50 text-gray-500 border-gray-200" title="Non-linearity test from the unadjusted model">
                    crude <i>p</i> = {fmtP(result.crude.nonlinearity_p)}
                  </span>
                )}
              </div>

              {Array.isArray(result.covariates_used) && result.covariates_used.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                  <span className="text-gray-400">Adjusted for:</span>
                  {(result.covariates_summary as Array<{ name: string; effect: number | null; coef: number | null }> | undefined ?? []).map((cv) => (
                    <span key={cv.name} className="bg-emerald-50 border border-emerald-100 text-emerald-700 rounded px-1.5 py-0.5"
                      title={cv.coef != null ? `β = ${cv.coef}` : ""}>
                      {cv.name}
                      {cv.effect != null && <> · {eff.abbr} = {cv.effect.toFixed(3)}</>}
                    </span>
                  ))}
                  {result.covariates_requested && result.covariates_requested.length !== result.covariates_used.length && (
                    <span className="text-amber-600 text-[10px] ml-1">
                      ({result.covariates_requested.length - result.covariates_used.length} dropped — categorical encoded as dummies)
                    </span>
                  )}
                </div>
              )}

              {result.n_excluded != null && result.n_excluded > 0 && (
                <p className="text-[10px] text-amber-600">
                  {result.n_excluded} of {result.n_total} rows excluded due to missing values in predictor / outcome / covariates.
                </p>
              )}

              {result.interaction && !result.interaction.error && (
                <div className={`text-xs p-2 rounded border ${result.interaction.p != null && result.interaction.p < 0.05 ? "bg-amber-50 text-amber-900 border-amber-200" : "bg-gray-50 text-gray-700 border-gray-200"}`}>
                  <span className="font-semibold uppercase tracking-wider text-[10px] block mb-0.5">
                    Spline × {Array.isArray(result.interaction.covariates) ? result.interaction.covariates.join(" + ") : "covariate"} interaction (LR test)
                  </span>
                  χ²({result.interaction.df}) = {result.interaction.lr_stat?.toFixed(2)}, <i>p</i> = {fmtP(result.interaction.p)}
                  <span className="block text-[10px] text-gray-500 mt-0.5">
                    {result.interaction.p != null && result.interaction.p < 0.05
                      ? "The dose-response shape differs across levels — consider reporting stratified curves."
                      : "No evidence the dose-response shape differs across covariate levels."}
                  </span>
                </div>
              )}
              {result.interaction?.error && (
                <p className="text-xs text-red-500">Interaction test failed: {result.interaction.error}</p>
              )}

              {(() => {
                // Build dose-response figure data, optionally overlaying the
                // crude (unadjusted) curve when the backend returned one.
                const crude = result.crude as null | {
                  x_values: number[]; or_values: number[];
                  ci_low: number[]; ci_high: number[];
                  nonlinearity_p: number | null;
                };
                const nlp = result.nonlinearity_p as number | null | undefined;
                const nlAdjStr = nlp == null ? null : fmtP(nlp);

                const traces: PlotData[] = [
                  // Adjusted 95% CI band
                  {
                    type: "scatter", x: [...(result.x_values as number[]), ...(result.x_values as number[]).slice().reverse()],
                    y: [...(result.ci_high as number[]), ...(result.ci_low as number[]).slice().reverse()],
                    fill: "toself", fillcolor: crude ? "rgba(220,38,38,0.13)" : "rgba(99,102,241,0.12)",
                    line: { color: "transparent" }, hoverinfo: "skip", name: crude ? "95% CI (adjusted)" : "95% CI",
                    showlegend: true,
                  },
                  // Adjusted central line
                  {
                    type: "scatter", mode: "lines",
                    x: result.x_values as number[], y: result.or_values as number[],
                    line: { color: crude ? "#dc2626" : "#6366f1", width: 2.5 },
                    name: crude
                      ? `Adjusted${result.covariates_used?.length ? ` (${(result.covariates_used as string[]).join(" + ")})` : ""}`
                      : eff.label,
                    hovertemplate: `${result.predictor}: %{x:.2f}<br>${eff.abbr}: %{y:.3f}<extra></extra>`,
                  },
                ];

                if (crude) {
                  traces.push({
                    type: "scatter", mode: "lines",
                    x: crude.x_values, y: crude.or_values,
                    line: { color: "#475569", width: 2, dash: "dash" },
                    name: "Crude (unadjusted)",
                    hovertemplate: `${result.predictor}: %{x:.2f}<br>${eff.abbr} (crude): %{y:.3f}<extra></extra>`,
                  });
                }

                // Knot markers
                traces.push({
                  type: "scatter", mode: "markers",
                  x: (result.knots as number[]) ?? [],
                  y: ((result.knots as number[]) ?? []).map((k: number) => {
                    const xs = result.x_values as number[];
                    const ys = result.or_values as number[];
                    const idx = xs.reduce((best: number, x: number, i: number) =>
                      Math.abs(x - k) < Math.abs(xs[best] - k) ? i : best, 0);
                    return ys[idx];
                  }),
                  marker: { color: crude ? "#dc2626" : "#6366f1", size: 7, line: { color: "#fff", width: 1.5 } },
                  name: "Knots",
                  showlegend: false,
                  hovertemplate: `Knot: %{x:.2f}<br>${eff.abbr}: %{y:.3f}<extra></extra>`,
                });

                if (rcsShowData) {
                  traces.push({
                    type: "scatter", mode: "markers",
                    x: result.x_data as number[],
                    y: Array((result.x_data as number[]).length).fill(useLogY ? Math.exp(-0.35) : (eff.refValue === 0 ? eff.refValue - 0.5 : 0.7)),
                    marker: { color: "#9ca3af", size: 3, opacity: 0.25, symbol: "line-ns-open" },
                    yaxis: "y", showlegend: false, hoverinfo: "skip", name: "Data",
                  });
                }

                // Clean axis label, no redundant "HR = 1.0": e.g.
                // "Hazard ratio (reference: LDL-C 130)".
                const effLabelLc = eff.label.replace(/ Ratio$/, " ratio");
                const yTitle = `${effLabelLc} (reference: ${result.predictor} ${result.ref_value})`;

                // Clamp the x-axis to the spline's evaluated range. The curve
                // is only fit over result.x_values (typically the inner
                // percentile range); the rug ('Show Data Points') uses the raw
                // observations which can extend well past the curve into a
                // sparse tail, so without an explicit range Plotly autoscales
                // to the rug and leaves a large empty strip on the right.
                const xvCurve = (result.x_values as number[]) ?? [];
                const xMinC = xvCurve.length ? Math.min(...xvCurve) : undefined;
                const xMaxC = xvCurve.length ? Math.max(...xvCurve) : undefined;
                const xPad = (xMinC != null && xMaxC != null) ? Math.max((xMaxC - xMinC) * 0.03, 1e-9) : 0;
                const xRange = (xMinC != null && xMaxC != null) ? [xMinC - xPad, xMaxC + xPad] : undefined;

                return (
                  <TitledPlot
                    plotRefOut={rcsPlotRef}
                    storageKey={`rcs2:${result.predictor}:${outcomeLabel}:ref${result.ref_value}:k${result.n_knots}:cov${(result.covariates_used as string[] | undefined)?.length ?? 0}`}
                    // No embedded title — supply the figure legend in the
                    // manuscript. The caption avoids raw column names (uses the
                    // predictor only) and explains the knot dots.
                    defaultTitle=""
                    defaultSubtitle={`Restricted cubic spline (${result.n_knots} knots) · reference ${result.predictor} = ${result.ref_value}${nlAdjStr ? ` · non-linearity ${crude ? "adjusted " : ""}p=${nlAdjStr}` : ""}${result.n_events != null ? ` · n=${result.n}, events=${result.n_events}` : ` · n=${result.n}`}${result.aic != null ? ` · AIC=${(result.aic as number).toFixed(1)}` : ""} · dots = knot locations`}
                    defaultXAxis={result.predictor}
                    defaultYAxis={yTitle}
                    data={traces}
                    layout={{
                      ...themedLayout, autosize: true, height: 440,
                      xaxis: { ...PLOT_LAYOUT.xaxis, showgrid: showGrid, title: { text: result.predictor }, zeroline: false, ...(xRange ? { range: xRange, autorange: false as const } : {}) },
                      yaxis: {
                        ...PLOT_LAYOUT.yaxis, showgrid: showGrid,
                        title: { text: yTitle }, zeroline: false,
                        ...(useLogY ? { type: "log" as const, dtick: 1 } : {}),
                      },
                      shapes: [
                        // Horizontal reference at HR/OR = 1 (or Δ = 0)
                        { type: "line", xref: "paper", yref: "y",
                          x0: 0, x1: 1, y0: eff.refValue, y1: eff.refValue,
                          line: { color: "#9ca3af", width: 1.2, dash: "dash" } },
                        // Vertical reference at x = ref_value
                        { type: "line", xref: "x", yref: "paper",
                          x0: result.ref_value, x1: result.ref_value, y0: 0, y1: 1,
                          line: { color: "#9ca3af", width: 1.2, dash: "dot" } },
                      ],
                      legend: { font: { size: 11, color: "#374151" }, x: 0.02, y: 0.06, xanchor: "left", yanchor: "bottom", bgcolor: "rgba(255,255,255,0.85)", bordercolor: "#e5e7eb", borderwidth: 1 },
                      margin: { t: 20, r: 20, b: 50, l: 70 },
                    }}
                    style={{ width: "100%", height: 520 }}
                    config={{ responsive: true, displaylogo: false,
                      toImageButtonOptions: { format: "png", filename: `RCS_${result.predictor}_${outcomeLabel || "result"}`, width: 1200, height: 600 },
                      modeBarButtonsToRemove: ["select2d", "lasso2d"] }}
                  />
                );
              })()}

              <div className="flex items-center gap-6 pt-1 border-t border-gray-100">
                <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer select-none">
                  <span>Log Scale (Y)</span>
                  <button onClick={() => setRcsLogScale(v => !v)}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${rcsLogScale ? "bg-indigo-600" : "bg-gray-300"}`}>
                    <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${rcsLogScale ? "translate-x-4.5" : "translate-x-0.5"}`} />
                  </button>
                </label>
                <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer select-none">
                  <span>Show Data Points</span>
                  <button onClick={() => setRcsShowData(v => !v)}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${rcsShowData ? "bg-indigo-600" : "bg-gray-300"}`}>
                    <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${rcsShowData ? "translate-x-4.5" : "translate-x-0.5"}`} />
                  </button>
                </label>
              </div>

              <InfoBanner>
                The curve shows the <strong>non-linear dose-response</strong> relationship between <em>{result.predictor}</em>{" "}
                {mt === "cox"
                  ? <>and the <em>hazard</em> of <em>{result.event_col ?? "the event"}</em> (time = <em>{result.duration_col ?? ""}</em>)</>
                  : mt === "linear"
                  ? <>and the mean of <em>{result.outcome}</em></>
                  : <>and the odds of <em>{result.outcome}</em></>}.
                Filled circles mark the {result.n_knots} knot positions. The shaded band is the 95% CI.
                {eff.axisType === "log" && <> <strong>Log scale</strong> is recommended for {eff.abbr}s — it symmetrises the curve and reveals J/U shapes that appear compressed on a linear axis.</>}
              </InfoBanner>
            </div>
          );
        })()}

        {!result && !error && !loading && (
          <div className="panel h-64 flex items-center justify-center text-gray-400">
            Configure and fit an RCS model
          </div>
        )}
      </div>
    </div>
  );
}
