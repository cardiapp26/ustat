/**
 * SubgroupPanel — the effect inside each stratum, and whether it really differs.
 *
 * The forest plot at the back of a clinical paper. Two things are shown side
 * by side because they answer different questions and are constantly confused:
 * the per-stratum estimates, each from a model fitted inside that stratum, and
 * P for interaction, from one model on the whole sample.
 *
 * Reading significance off the strata is the classic subgroup error — two
 * subgroups can sit either side of the null while the interaction test is
 * nowhere near significant, because a difference in significance is not a
 * significant difference. The panel says so next to the numbers, not in a
 * footnote, and the caveat is carried in the exported result too.
 */
import { Fragment, useMemo, useRef, useState } from "react";
import Plot from "../PlotComponent";
import { usePlotLayout, usePalette } from "../plotStyle";
import { analysisCols, isNumericKind, isCategoricalKind, useStore, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runSubgroup } from "../api";
import { Tip } from "./Tip";
import ResultExporter from "./ResultExporter";
import { fmtP, pCellTitle } from "../lib/format";
import type { Data, Layout } from "plotly.js";
import type { PlotData, PlotLayout, PlotCaptureHandle } from "../lib/plotTypes";

interface Row {
  level: string; n: number; events?: number;
  beta?: number | null; se?: number | null; p?: number | null;
  ci_low?: number | null; ci_high?: number | null;
  ratio?: number | null; ratio_ci_low?: number | null; ratio_ci_high?: number | null;
  note?: string; thin?: boolean;
}
interface Block {
  variable: string; levels: string[]; rows: Row[];
  p_interaction: number | null; interaction_note: string; n_used: number;
}
export interface SubgroupResult {
  outcome: string; exposure: string; outcome_kind: "continuous" | "binary" | "survival";
  effect_label: string; null_value: number;
  overall: Row & { note?: string };
  subgroups: Block[]; warnings: string[]; result_text: string; caveat: string;
}

export default function SubgroupPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <SubgroupPanelBody session={session} />;
}

function SubgroupPanelBody({ session }: { session: Session }) {
  const themedBase = usePlotLayout();
  const palette = usePalette();
  const cols = useMemo(() => analysisCols(session.columns), [session.columns]);
  const numCols = useMemo(() => cols.filter((c) => isNumericKind(c.kind)).map((c) => c.name), [cols]);
  const catCols = useMemo(() => cols.filter((c) => isCategoricalKind(c.kind)).map((c) => c.name), [cols]);
  const allCols = useMemo(() => cols.map((c) => c.name), [cols]);

  const [kind, setKind] = usePersistedPanelState<"continuous" | "binary" | "survival">("subgroup", "kind", "continuous");
  const [outcome, setOutcome] = usePersistedPanelState<string>("subgroup", "outcome", "");
  const [exposure, setExposure] = usePersistedPanelState<string>("subgroup", "exposure", "");
  const [timeCol, setTimeCol] = usePersistedPanelState<string>("subgroup", "time", "");
  const [subgroups, setSubgroups] = usePersistedPanelState<string[]>("subgroup", "subgroups", []);
  const [covariates, setCovariates] = usePersistedPanelState<string[]>("subgroup", "covs", []);

  const [result, setResult] = useState<SubgroupResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const plotRef = useRef<PlotCaptureHandle | null>(null);
  const setForestHandoff = useStore((s) => s.setForestHandoff);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const setVisualSubTab = useStore((s) => s.setVisualSubTab);

  const toggle = (list: string[], set: (v: string[]) => void, c: string) =>
    set(list.includes(c) ? list.filter((x) => x !== c) : [...list, c]);

  const run = async () => {
    if (!outcome || !exposure) { setError("Choose an outcome and an exposure."); return; }
    if (!subgroups.length) { setError("Choose at least one subgroup variable."); return; }
    if (kind === "survival" && !timeCol) { setError("A time-to-event outcome needs a follow-up time column."); return; }
    setLoading(true); setError(null);
    try {
      const res = await runSubgroup({
        session_id: session.session_id, outcome, exposure, subgroups, covariates,
        outcome_kind: kind, time_col: kind === "survival" ? timeCol : undefined,
        categorical: covariates.filter((c) => catCols.includes(c)),
      });
      setResult(res.data as SubgroupResult);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(typeof msg === "string" ? msg : e instanceof Error ? e.message : "Analysis failed");
      setResult(null);
    } finally { setLoading(false); }
  };

  const ratio = result ? result.outcome_kind !== "continuous" : false;
  const est = (r?: Row) => (ratio ? r?.ratio : r?.beta) ?? null;
  const lo = (r?: Row) => (ratio ? r?.ratio_ci_low : r?.ci_low) ?? null;
  const hi = (r?: Row) => (ratio ? r?.ratio_ci_high : r?.ci_high) ?? null;
  const fmt = (r?: Row) => {
    if (!r || est(r) == null) return r?.note ?? "—";
    const d = ratio ? 2 : 3;
    return `${est(r)!.toFixed(d)} (${lo(r)!.toFixed(d)}–${hi(r)!.toFixed(d)})`;
  };

  /** Every plotted row, top to bottom, with its label. */
  const forestRows = useMemo(() => {
    if (!result) return [];
    const out: { label: string; row: Row; header?: boolean }[] = [];
    for (const b of result.subgroups) {
      out.push({ label: b.variable, row: {} as Row, header: true });
      for (const r of b.rows) out.push({ label: `   ${r.level}`, row: r });
    }
    out.push({ label: "Overall", row: result.overall });
    return out;
  }, [result]);

  const traces: PlotData[] = useMemo(() => {
    const plotted = forestRows.filter((r) => !r.header && est(r.row) != null);
    if (!plotted.length) return [];
    const yv = plotted.map((r) => r.label.trim());
    return [{
      x: plotted.map((r) => est(r.row) as number),
      y: yv,
      type: "scatter", mode: "markers", name: "Effect",
      error_x: {
        type: "data", symmetric: false,
        array: plotted.map((r) => (hi(r.row) as number) - (est(r.row) as number)),
        arrayminus: plotted.map((r) => (est(r.row) as number) - (lo(r.row) as number)),
        color: palette[0], thickness: 1.5, width: 4,
      },
      marker: { size: 9, symbol: "square", color: palette[0] },
      hovertemplate: `%{y}<br>${result?.effect_label} %{x:.3f}<extra></extra>`,
    }];
  }, [forestRows, palette, result, ratio]);

  const layout: PlotLayout = useMemo(() => ({
    ...themedBase,
    height: Math.max(240, 34 * forestRows.filter((r) => !r.header).length + 90),
    margin: { t: 12, r: 20, b: 46, l: 130 }, showlegend: false,
    xaxis: {
      title: { text: `${result?.effect_label ?? ""} (95% CI)` }, zeroline: false,
      ...(ratio ? { type: "log" } : {}),
    },
    yaxis: { autorange: "reversed", zeroline: false, automargin: true },
    shapes: result ? [{
      type: "line", xref: "x", yref: "paper",
      x0: result.null_value, x1: result.null_value, y0: 0, y1: 1,
      line: { color: "#9ca3af", width: 1, dash: "dash" },
    }] : [],
  } as PlotLayout), [themedBase, forestRows, result, ratio]);

  const sendToBuilder = () => {
    if (!result) return;
    const rows = forestRows
      .filter((r) => !r.header && est(r.row) != null)
      .map((r) => ({
        label: r.label.trim(), est: est(r.row), ci_low: lo(r.row), ci_high: hi(r.row),
        p: r.row.p ?? null,
        extra: r.row.events != null ? `${r.row.events}/${r.row.n}` : `n=${r.row.n}`,
      }));
    if (!rows.length) { setError("No estimates to send."); return; }
    setForestHandoff(rows, {
      customTitle: `${result.effect_label} for ${result.exposure} by subgroup`,
      customSubtitle: `Outcome: ${result.outcome}`,
      xLabel: `${result.effect_label} (95% CI${ratio ? ", log scale" : ""})`,
      leftHeader: "Subgroup", rightHeader: `${result.effect_label} (95% CI)`,
      returnTab: "models", returnLabel: "← Back to the subgroup analysis",
    }, true);
    setVisualSubTab("forest");
    setActiveTab("visual");
  };

  const exportRows = result ? forestRows.map(({ label, row, header }) => [
    label.trim(), header ? "" : String(row.n ?? ""), header ? "" : fmt(row),
    header ? "" : fmtP(row.p ?? null),
  ]) : [];

  return (
    <div className="flex gap-4">
      <div className="w-64 flex-shrink-0 space-y-3">
        <div className="panel space-y-3">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-900">Subgroup analysis</h3>
            <Tip wide text="Fits the model inside each stratum and, separately, tests whether the effect differs across strata with an exposure × subgroup interaction on the whole sample. The strata answer 'what is the effect here'; only the interaction answers 'is it different here'." />
          </div>

          <label className="block space-y-1">
            <span className="text-xs font-medium text-gray-500">Outcome type</span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
              <option value="continuous">Continuous (mean difference)</option>
              <option value="binary">Binary 0/1 (odds ratio)</option>
              <option value="survival">Time-to-event (hazard ratio)</option>
            </select>
          </label>

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
            <span className="text-xs font-medium text-gray-500">Exposure</span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={exposure} onChange={(e) => setExposure(e.target.value)}>
              <option value="">— select —</option>
              {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <div className="space-y-1">
            <span className="text-xs font-medium text-gray-500">Subgroup variables</span>
            <div className="max-h-32 overflow-y-auto rounded-lg border border-gray-200 p-1.5">
              {allCols.filter((c) => c !== outcome && c !== exposure && c !== timeCol).map((c) => (
                <label key={c} className="flex items-center gap-2 px-1 py-0.5 text-xs text-gray-700 hover:bg-gray-50 rounded cursor-pointer">
                  <input type="checkbox" checked={subgroups.includes(c)}
                    onChange={() => toggle(subgroups, setSubgroups, c)} />
                  <span className="truncate">{c}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="space-y-1">
            <span className="text-xs font-medium text-gray-500">Adjust for (inside every stratum)</span>
            <div className="max-h-28 overflow-y-auto rounded-lg border border-gray-200 p-1.5">
              {allCols.filter((c) => c !== outcome && c !== exposure && c !== timeCol && !subgroups.includes(c)).map((c) => (
                <label key={c} className="flex items-center gap-2 px-1 py-0.5 text-xs text-gray-700 hover:bg-gray-50 rounded cursor-pointer">
                  <input type="checkbox" checked={covariates.includes(c)}
                    onChange={() => toggle(covariates, setCovariates, c)} />
                  <span className="truncate">{c}</span>
                </label>
              ))}
            </div>
          </div>

          <button className="btn-primary w-full" onClick={run} disabled={loading || !outcome || !exposure || !subgroups.length}>
            {loading ? "Fitting…" : "Run subgroup analysis"}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {result ? (
          <>
            <div className="panel">
              <div className="mb-2 flex items-center justify-between gap-2">
                <h4 className="text-sm font-semibold text-gray-700">
                  {result.effect_label} for {result.exposure} — {result.outcome}
                </h4>
                <div className="flex items-center gap-2">
                  <button onClick={sendToBuilder}
                    title="Add these rows to the Forest Builder, where the figure can be titled and styled for a journal."
                    className="flex-shrink-0 whitespace-nowrap rounded-lg border border-indigo-200 bg-indigo-50 px-2.5 py-1 text-xs font-medium text-indigo-700 hover:bg-indigo-100">
                    → Forest Builder
                  </button>
                  <ResultExporter title="Subgroup_analysis"
                    headers={["Subgroup", "n", `${result.effect_label} (95% CI)`, "p"]} rows={exportRows} />
                </div>
              </div>

              <div className="grid gap-3 lg:grid-cols-2">
                <div className="overflow-auto rounded border border-gray-200">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="bg-gray-50 text-gray-500">
                        <th className="px-2 py-1.5 text-left">Subgroup</th>
                        <th className="px-2 py-1.5 text-right">n</th>
                        <th className="px-2 py-1.5 text-right">{result.effect_label} (95% CI)</th>
                        <th className="px-2 py-1.5 text-right"><i>p</i> interaction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.subgroups.map((b) => (
                        <Fragment key={b.variable}>
                          <tr className="border-t border-gray-200 bg-gray-50/60">
                            <th scope="row" className="px-2 py-1.5 text-left font-semibold text-gray-700">{b.variable}</th>
                            <td className="px-2 py-1.5" />
                            <td className="px-2 py-1.5" />
                            <td className="px-2 py-1.5 text-right font-mono" title={b.interaction_note}>
                              {fmtP(b.p_interaction)}
                            </td>
                          </tr>
                          {b.rows.map((r) => (
                            <tr key={`${b.variable}-${r.level}`} className="border-t border-gray-100">
                              <th scope="row" className="px-2 py-1.5 pl-5 text-left font-normal text-gray-600">
                                {r.level}
                                {r.thin && <span className="ml-1 text-[9px] text-amber-600" title="Fewer than 20 observations">thin</span>}
                              </th>
                              <td className="px-2 py-1.5 text-right font-mono text-gray-500">
                                {r.events != null ? `${r.events}/${r.n}` : r.n}
                              </td>
                              <td className="px-2 py-1.5 text-right font-mono">{fmt(r)}</td>
                              <td className="px-2 py-1.5 text-right font-mono text-gray-400" title={pCellTitle(r.p ?? null)}>
                                {fmtP(r.p ?? null)}
                              </td>
                            </tr>
                          ))}
                        </Fragment>
                      ))}
                      <tr className="border-t-2 border-gray-300">
                        <th scope="row" className="px-2 py-1.5 text-left font-semibold text-gray-700">Overall</th>
                        <td className="px-2 py-1.5 text-right font-mono text-gray-500">
                          {result.overall.events != null ? `${result.overall.events}/${result.overall.n}` : result.overall.n}
                        </td>
                        <td className="px-2 py-1.5 text-right font-mono">{fmt(result.overall)}</td>
                        <td className="px-2 py-1.5" />
                      </tr>
                    </tbody>
                  </table>
                  <p className="px-2 py-1 text-[10px] text-gray-400">
                    The right column is the interaction test, on the variable's row. The faint p beside
                    each stratum is that stratum's own — it is not a comparison between strata.
                  </p>
                </div>

                <div>
                  <Plot data={traces as unknown as Data[]} layout={layout as unknown as Partial<Layout>}
                    config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                    onInitialized={(_: object, gd: HTMLElement) => { plotRef.current = gd as unknown as PlotCaptureHandle; }}
                    onUpdate={(_: object, gd: HTMLElement) => { plotRef.current = gd as unknown as PlotCaptureHandle; }}
                    style={{ width: "100%" }} useResizeHandler />
                </div>
              </div>
            </div>

            <div className="panel space-y-2">
              <h4 className="text-sm font-semibold text-gray-700">Reading it</h4>
              <p className="rounded bg-amber-50 px-2 py-1.5 text-[11px] leading-relaxed text-amber-800">
                {result.caveat}
              </p>
              {result.warnings.map((w) => (
                <p key={w} className="rounded bg-amber-50 px-2 py-1.5 text-[11px] leading-relaxed text-amber-800">{w}</p>
              ))}
              <p className="rounded border border-indigo-100 bg-white px-2 py-1.5 text-[11px] leading-relaxed text-gray-600">
                {result.result_text}
              </p>
            </div>
          </>
        ) : (
          <div className="panel py-16 text-center text-gray-400">
            <p className="mb-2 text-lg">🌲</p>
            <p>Pick an exposure and the subgroups to split it by.</p>
            <p className="mx-auto mt-2 max-w-md text-xs">
              Each stratum gets its own model; the test for whether the effect differs between strata
              comes from a single model on everybody.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
