/**
 * ThresholdPanel — two-piecewise regression: where does the effect change?
 *
 * The spline panel next door answers "is this relationship curved?". This one
 * answers the question a paper asks after that: curved where, and how steep on
 * each side. It reports the inflection point with a profile-likelihood
 * interval, the slope below and above it, and the likelihood-ratio test of two
 * lines against one.
 *
 * The caveat travels with the result rather than living in a footnote: the
 * breakpoint is chosen by maximising the same likelihood the test then uses,
 * so that p-value is optimistic. The profile plot is shown for the same
 * reason — a flat ridge means the inflection point is not identified, however
 * confident the table looks.
 */
import { useMemo, useRef, useState } from "react";
import Plot from "../PlotComponent";
import { usePlotLayout, usePalette } from "../plotStyle";
import { analysisCols, isNumericKind, isCategoricalKind, useStore, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runThreshold } from "../api";
import { Tip } from "./Tip";
import ThreeCol from "./ThreeCol";
import ResultExporter from "./ResultExporter";
import { fmtP, pCellTitle } from "../lib/format";
import type { Data, Layout } from "plotly.js";
import type { PlotData, PlotLayout, PlotCaptureHandle } from "../lib/plotTypes";

interface Effect {
  beta?: number | null; se?: number | null; p?: number | null;
  ci_low?: number | null; ci_high?: number | null;
  ratio?: number | null; ratio_ci_low?: number | null; ratio_ci_high?: number | null;
}
export interface ThresholdResult {
  outcome: string; exposure: string; outcome_kind: "continuous" | "binary" | "survival";
  n_used: number; n_dropped: number;
  breakpoint: number; breakpoint_ci: { low?: number; high?: number };
  search_range: { low?: number; high?: number; n_candidates: number };
  effect_below: Effect; effect_above: Effect; effect_difference: Effect; effect_single_line: Effect;
  loglik_single: number; loglik_segmented: number;
  lr_stat: number; lr_p: number;
  effect_label: string;
  profile: { k: number | null; loglik: number | null }[];
  curve: { x: (number | null)[]; y: (number | null)[] };
  verdict: string; result_text: string; warnings: string[]; caveat: string;
}

const KINDS = [
  ["continuous", "Continuous", "Linear regression — the effect is a mean difference per unit."],
  ["binary", "Binary (0 / 1)", "Logistic regression — the effect is an odds ratio per unit."],
  ["survival", "Time-to-event", "Cox regression — the effect is a hazard ratio per unit."],
] as const;

const n3 = (v: number | null | undefined, d = 3) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(d);

export default function ThresholdPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <ThresholdPanelBody session={session} />;
}

function ThresholdPanelBody({ session }: { session: Session }) {
  const themedBase = usePlotLayout();
  const palette = usePalette();
  const cols = useMemo(() => analysisCols(session.columns), [session.columns]);
  const numCols = useMemo(() => cols.filter((c) => isNumericKind(c.kind)).map((c) => c.name), [cols]);
  const catCols = useMemo(() => cols.filter((c) => isCategoricalKind(c.kind)).map((c) => c.name), [cols]);
  const allCols = useMemo(() => cols.map((c) => c.name), [cols]);

  const [kind, setKind] = usePersistedPanelState<"continuous" | "binary" | "survival">("threshold", "kind", "continuous");
  const [outcome, setOutcome] = usePersistedPanelState<string>("threshold", "outcome", "");
  const [exposure, setExposure] = usePersistedPanelState<string>("threshold", "exposure", "");
  const [timeCol, setTimeCol] = usePersistedPanelState<string>("threshold", "time", "");
  const [covariates, setCovariates] = usePersistedPanelState<string[]>("threshold", "covs", []);

  const [result, setResult] = useState<ThresholdResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const curveRef = useRef<PlotCaptureHandle | null>(null);

  const toggleCov = (c: string) =>
    setCovariates((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]));

  const run = async () => {
    if (!outcome || !exposure) { setError("Choose an outcome and a continuous exposure."); return; }
    if (kind === "survival" && !timeCol) { setError("A time-to-event outcome needs a follow-up time column."); return; }
    setLoading(true); setError(null);
    try {
      const res = await runThreshold({
        session_id: session.session_id, outcome, exposure, outcome_kind: kind,
        time_col: kind === "survival" ? timeCol : undefined,
        covariates,
        categorical: covariates.filter((c) => catCols.includes(c)),
      });
      setResult(res.data as ThresholdResult);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(typeof msg === "string" ? msg : e instanceof Error ? e.message : "Analysis failed");
      setResult(null);
    } finally { setLoading(false); }
  };

  const ratioScale = result ? result.outcome_kind !== "continuous" : false;

  const curveTraces: PlotData[] = useMemo(() => {
    if (!result?.curve?.x?.length) return [];
    const y = ratioScale ? (result.curve.y as number[]).map((v) => Math.exp(v)) : (result.curve.y as number[]);
    return [{
      x: result.curve.x as number[], y,
      type: "scatter", mode: "lines", name: "Fitted",
      line: { color: palette[0], width: 2.5 },
      hovertemplate: `${result.exposure} %{x:.3g}<br>${result.effect_label} %{y:.3f}<extra></extra>`,
    }];
  }, [result, palette, ratioScale]);

  const curveLayout: PlotLayout = useMemo(() => {
    const ci = result?.breakpoint_ci ?? {};
    const shapes: Record<string, unknown>[] = [];
    if (result) {
      // The interval is shaded rather than drawn as two lines: a breakpoint is
      // an estimate, and a bare vertical line reads as a known boundary.
      if (ci.low != null && ci.high != null) {
        shapes.push({ type: "rect", xref: "x", yref: "paper", x0: ci.low, x1: ci.high, y0: 0, y1: 1,
          fillcolor: palette[0], opacity: 0.1, line: { width: 0 } });
      }
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: result.breakpoint, x1: result.breakpoint,
        y0: 0, y1: 1, line: { color: palette[1] ?? "#dc2626", width: 1.5, dash: "dash" } });
      if (ratioScale) {
        shapes.push({ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 1, y1: 1,
          line: { color: "#9ca3af", width: 1, dash: "dot" } });
      }
    }
    return {
      ...themedBase, height: 320, margin: { t: 12, r: 16, b: 46, l: 60 }, showlegend: false, shapes,
      xaxis: { title: { text: result?.exposure ?? "" }, zeroline: false },
      yaxis: { title: { text: result?.effect_label ?? "" }, zeroline: false, ...(ratioScale ? { type: "log" } : {}) },
    } as PlotLayout;
  }, [result, themedBase, palette, ratioScale]);

  const profileTraces: PlotData[] = useMemo(() => {
    if (!result?.profile?.length) return [];
    return [{
      x: result.profile.map((p) => p.k as number), y: result.profile.map((p) => p.loglik as number),
      type: "scatter", mode: "lines", name: "Profile log-likelihood",
      line: { color: palette[2] ?? "#0891b2", width: 2 },
      hovertemplate: "breakpoint %{x:.3g}<br>log-likelihood %{y:.2f}<extra></extra>",
    }];
  }, [result, palette]);

  const profileLayout: PlotLayout = useMemo(() => ({
    ...themedBase, height: 200, margin: { t: 12, r: 16, b: 42, l: 60 }, showlegend: false,
    xaxis: { title: { text: `Candidate breakpoint in ${result?.exposure ?? ""}` }, zeroline: false },
    yaxis: { title: { text: "Log-likelihood" }, zeroline: false },
    shapes: result ? [{ type: "line", xref: "x", yref: "paper", x0: result.breakpoint, x1: result.breakpoint,
      y0: 0, y1: 1, line: { color: palette[1] ?? "#dc2626", width: 1.5, dash: "dash" } }] : [],
  } as PlotLayout), [result, themedBase, palette]);

  const cell = (e: Effect) => {
    if (!e || e.beta == null) return "—";
    return ratioScale
      ? `${n3(e.ratio, 2)} (${n3(e.ratio_ci_low, 2)} to ${n3(e.ratio_ci_high, 2)})`
      : `${n3(e.beta)} (${n3(e.ci_low)} to ${n3(e.ci_high)})`;
  };

  const exportRows = result ? [
    ["Below the breakpoint", cell(result.effect_below), fmtP(result.effect_below.p ?? null)],
    ["Above the breakpoint", cell(result.effect_above), fmtP(result.effect_above.p ?? null)],
    ["Difference in slope", cell(result.effect_difference), fmtP(result.effect_difference.p ?? null)],
    ["One straight line (no breakpoint)", cell(result.effect_single_line), fmtP(result.effect_single_line.p ?? null)],
  ] : [];

  return (
    <ThreeCol
      storageKey="ThresholdPanel"
      left={
        <div className="panel space-y-3">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-900">Threshold analysis</h3>
            <Tip wide text="Fits two straight lines joined at an estimated inflection point, and tests them against a single line. Use it when a spline shows the relationship bending and you need to say where it bends and how steep it is on each side. The exposure must be continuous." />
          </div>

          <div className="space-y-1">
            <span className="text-xs font-medium text-gray-500">Outcome type</span>
            {KINDS.map(([id, label, hint]) => (
              <label key={id} className="flex items-start gap-2 cursor-pointer rounded px-1 py-0.5 hover:bg-gray-50">
                <input type="radio" name="thrKind" checked={kind === id} onChange={() => setKind(id)}
                  className="mt-0.5 accent-indigo-500" />
                <span>
                  <span className="text-sm text-gray-700">{label}</span>
                  <span className="block text-[10px] leading-tight text-gray-400">{hint}</span>
                </span>
              </label>
            ))}
          </div>

          <label className="block space-y-1">
            <span className="text-xs font-medium text-gray-500">Outcome</span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>
              {(kind === "continuous" ? numCols : allCols).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          {kind === "survival" && (
            <label className="block space-y-1">
              <span className="text-xs font-medium text-gray-500">Follow-up time</span>
              <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
                value={timeCol} onChange={(e) => setTimeCol(e.target.value)}>
                <option value="">— select —</option>
                {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
          )}

          <label className="block space-y-1">
            <span className="text-xs font-medium text-gray-500 flex items-center gap-1">
              Exposure (continuous)
              <Tip text="The variable searched for a breakpoint. It has to be continuous — a threshold in a categorical variable is just its levels." />
            </span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={exposure} onChange={(e) => setExposure(e.target.value)}>
              <option value="">— select —</option>
              {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <div className="space-y-1">
            <span className="text-xs font-medium text-gray-500">Adjust for</span>
            <div className="max-h-40 overflow-y-auto rounded-lg border border-gray-200 p-1.5 space-y-0.5">
              {allCols.filter((c) => c !== outcome && c !== exposure && c !== timeCol).map((c) => (
                <label key={c} className="flex items-center gap-2 px-1 py-0.5 text-xs text-gray-700 hover:bg-gray-50 rounded cursor-pointer">
                  <input type="checkbox" checked={covariates.includes(c)} onChange={() => toggleCov(c)} />
                  <span className="truncate">{c}</span>
                  {catCols.includes(c) && <span className="ml-auto text-[9px] text-gray-400">cat</span>}
                </label>
              ))}
            </div>
          </div>

          <button className="btn-primary w-full" onClick={run} disabled={loading || !outcome || !exposure}>
            {loading ? "Searching…" : "Find the threshold"}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>
      }
      middle={
        result ? (
          <div className="space-y-3">
            <div className="panel">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-gray-700">Fitted relationship</h4>
                <ResultExporter title={`Threshold_${result.exposure}`} plotRef={curveRef} />
              </div>
              <Plot data={curveTraces as unknown as Data[]} layout={curveLayout as unknown as Partial<Layout>}
                config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                onInitialized={(_: object, gd: HTMLElement) => { curveRef.current = gd as unknown as PlotCaptureHandle; }}
                onUpdate={(_: object, gd: HTMLElement) => { curveRef.current = gd as unknown as PlotCaptureHandle; }}
                style={{ width: "100%" }} useResizeHandler />
              <p className="mt-1 text-[10px] text-gray-400">
                Dashed line: the estimated inflection point. Shaded band: its 95% profile-likelihood
                interval. Covariates are held at their means.
              </p>
            </div>

            <div className="panel">
              <h4 className="mb-2 text-sm font-semibold text-gray-700">
                Profile log-likelihood
                <Tip wide text="How well each candidate breakpoint fits. A sharp peak means the inflection point is well identified; a flat ridge means many breakpoints fit almost equally well and the single number in the table should not be trusted to a decimal place." />
              </h4>
              <Plot data={profileTraces as unknown as Data[]} layout={profileLayout as unknown as Partial<Layout>}
                config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                style={{ width: "100%" }} useResizeHandler />
            </div>
          </div>
        ) : (
          <div className="panel py-12 text-center text-gray-400">
            <p className="mb-2 text-lg">📐</p>
            <p>Pick an outcome and a continuous exposure on the left.</p>
            <p className="mx-auto mt-2 max-w-md text-xs">
              Two straight lines are fitted with a joint at every candidate breakpoint; the one that
              fits best is reported, along with a test of whether it beats a single line at all.
            </p>
          </div>
        )
      }
      right={
        result ? (
          <div className="space-y-3">
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-700">Threshold effect</h4>
                <ResultExporter title="Threshold_effects"
                  headers={["Segment", `${result.effect_label} (95% CI)`, "p"]} rows={exportRows} />
              </div>

              <div className="rounded-lg bg-indigo-50 px-3 py-2">
                <p className="text-[10px] uppercase tracking-wide text-indigo-500">Inflection point</p>
                <p className="font-mono text-lg text-indigo-900">
                  {result.exposure} = {n3(result.breakpoint, 3)}
                </p>
                {result.breakpoint_ci?.low != null && (
                  <p className="text-[11px] text-indigo-700">
                    95% profile interval {n3(result.breakpoint_ci.low, 2)} to {n3(result.breakpoint_ci.high, 2)}
                  </p>
                )}
              </div>

              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400">
                    <th className="py-1 text-left font-medium">Segment</th>
                    <th className="py-1 text-right font-medium">{result.effect_label} (95% CI)</th>
                    <th className="py-1 text-right font-medium"><i>p</i></th>
                  </tr>
                </thead>
                <tbody>
                  {([["Below the point", result.effect_below], ["Above the point", result.effect_above],
                     ["Difference", result.effect_difference],
                     ["One line only", result.effect_single_line]] as [string, Effect][]).map(([label, e], i) => (
                    <tr key={label} className={`border-t border-gray-100 ${i === 3 ? "text-gray-400" : ""}`}>
                      <td className="py-1">{label}</td>
                      <td className="py-1 text-right font-mono">{cell(e)}</td>
                      <td className="py-1 text-right font-mono" title={pCellTitle(e?.p ?? null)}>{fmtP(e?.p ?? null)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-[10px] text-gray-400">
                Effects are per one-unit increase. The last row is the straight-line model the
                breakpoint is being tested against.
              </p>
            </div>

            <div className="panel space-y-2">
              <h4 className="text-sm font-semibold text-gray-700">Two lines or one?</h4>
              <p className="text-xs text-gray-600">
                Likelihood-ratio test {n3(result.lr_stat, 2)} on 1 df,{" "}
                <span title={pCellTitle(result.lr_p)}>{fmtP(result.lr_p)}</span> — {result.verdict}.
              </p>
              <p className="rounded bg-amber-50 px-2 py-1.5 text-[11px] leading-relaxed text-amber-800">
                {result.caveat}
              </p>
              {result.warnings.map((w) => (
                <p key={w} className="rounded bg-amber-50 px-2 py-1.5 text-[11px] leading-relaxed text-amber-800">{w}</p>
              ))}
              <p className="text-[10px] text-gray-400">
                n = {result.n_used}
                {result.n_dropped > 0 && ` · ${result.n_dropped} rows dropped`}
                {` · ${result.search_range.n_candidates} candidate breakpoints searched`}
              </p>
            </div>

            <div className="panel">
              <h4 className="mb-1 text-sm font-semibold text-gray-700">For the manuscript</h4>
              <p className="rounded border border-indigo-100 bg-white px-2 py-1.5 text-[11px] leading-relaxed text-gray-600">
                {result.result_text}
              </p>
            </div>
          </div>
        ) : (
          <div className="panel text-xs text-gray-400">
            The inflection point, the slope on each side and a paste-ready sentence appear here.
          </div>
        )
      }
    />
  );
}
