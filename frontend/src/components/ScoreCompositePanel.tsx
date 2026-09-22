import { useMemo, useRef, useState } from "react";
import { runScoreComposite } from "../api";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import { describeStale } from "../lib/resultStamp";
import type { PlotCaptureHandle, PlotData, PlotLayout } from "../lib/plotTypes";
import { usePlotLayout, usePalette } from "../plotStyle";
import { useStore, isCategoricalKind, isNumericKind, type ColMeta, type Session } from "../store";
import TitledPlot from "./TitledPlot";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";

interface ScoreSummary {
  score_col: string;
  label: string;
  p_text: string;
  n_by_group: Record<string, number>;
  components: Array<{ component: string; label: string; p_text: string }>;
}

interface ScoreCompositeResult {
  type: "score_composite";
  group_col: string;
  groups: string[];
  scores: ScoreSummary[];
  figure: { data: PlotData[]; layout: PlotLayout };
  method_note: string;
}

export default function ScoreCompositePanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <ScoreCompositePanelBody session={session} />;
}

function displayName(col: ColMeta | undefined, fallback: string): string {
  return col?.label || col?.display_name || col?.name || fallback;
}

function selectedValues(select: HTMLSelectElement): string[] {
  return Array.from(select.selectedOptions).map((option) => option.value);
}

function ScoreCompositePanelBody({ session }: { session: Session }) {
  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const plotRef = useRef<PlotCaptureHandle | null>(null);
  const columns = session.columns.filter((col) => !col.analysis_excluded);
  const numericCols = columns.filter((col) => isNumericKind(col.kind)).map((col) => col.name);
  const groupCols = columns.filter((col) => isCategoricalKind(col.kind)).map((col) => col.name);

  const [groupCol, setGroupCol] = usePersistedPanelState("score_composite", "groupCol", groupCols[0] ?? "");
  const [scoreA, setScoreA] = usePersistedPanelState("score_composite", "scoreA", numericCols[0] ?? "");
  const [scoreB, setScoreB] = usePersistedPanelState("score_composite", "scoreB", numericCols[1] ?? numericCols[0] ?? "");
  const [labelA, setLabelA] = usePersistedPanelState("score_composite", "labelA", "");
  const [labelB, setLabelB] = usePersistedPanelState("score_composite", "labelB", "");
  const [componentsA, setComponentsA] = usePersistedPanelState<string[]>("score_composite", "componentsA", []);
  const [componentsB, setComponentsB] = usePersistedPanelState<string[]>("score_composite", "componentsB", []);
  const [bins, setBins] = usePersistedPanelState("score_composite", "bins", 8);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const componentOptions = useMemo(
    () => columns
      .map((col) => col.name)
      .filter((name) => name !== groupCol && name !== scoreA && name !== scoreB),
    [columns, groupCol, scoreA, scoreB],
  );
  const activeComponentsA = componentsA.filter((name) => componentOptions.includes(name));
  const activeComponentsB = componentsB.filter((name) => componentOptions.includes(name));

  // The components actually sent, not the raw picks: a pick that has become
  // the group or a score column is dropped from the request. The display
  // labels only name the panels, so retyping one leaves the figure current.
  const runParams = {
    groupCol, bins, scoreA, scoreB,
    componentsA: activeComponentsA, componentsB: activeComponentsB,
  };
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<ScoreCompositeResult>("score_composite", runParams);

  const labelMap = (names: string[]) => Object.fromEntries(names.map((name) => [
    name,
    displayName(columns.find((col) => col.name === name), name),
  ]));

  const canRun = Boolean(groupCol && scoreA && scoreB && scoreA !== scoreB && activeComponentsA.length && activeComponentsB.length);

  const run = async () => {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    try {
      const scoreALabel = labelA.trim() || displayName(columns.find((col) => col.name === scoreA), scoreA);
      const scoreBLabel = labelB.trim() || displayName(columns.find((col) => col.name === scoreB), scoreB);
      const res = await runScoreComposite({
        session_id: session.session_id,
        group_col: groupCol,
        bins,
        title: "Score Distributions and Component Prevalence by Group",
        scores: [
          {
            score_col: scoreA,
            label: scoreALabel,
            components: activeComponentsA,
            component_labels: labelMap(activeComponentsA),
          },
          {
            score_col: scoreB,
            label: scoreBLabel,
            components: activeComponentsB,
            component_labels: labelMap(activeComponentsB),
          },
        ],
      });
      setResult(res.data as ScoreCompositeResult);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Error generating score figure");
    } finally {
      setLoading(false);
    }
  };

  const mergedLayout = result
    ? ({ ...baseLayout, ...result.figure.layout, height: result.figure.layout.height ?? 760 } as PlotLayout)
    : null;
  // Backend ships each trace with a fixed marker color (a sane default for
  // non-browser consumers of the raw API), keyed by group via `legendgroup`.
  // Re-map those colors to the app's active theme palette here so the
  // palette picker in the toolbar actually affects this chart.
  const paletteData = useMemo(() => {
    if (!result) return null;
    return result.figure.data.map((trace) => {
      const legendgroup = (trace as { legendgroup?: string }).legendgroup;
      if (!legendgroup) return trace;
      const idx = result.groups.indexOf(legendgroup);
      const color = pal[(idx < 0 ? 0 : idx) % pal.length];
      const marker = (trace as { marker?: Record<string, unknown> }).marker;
      return { ...trace, marker: { ...(marker || {}), color } };
    });
  }, [result, pal]);
  const titleText = typeof result?.figure.layout.title === "object"
    ? String((result.figure.layout.title as { text?: unknown }).text ?? "")
    : "Score Distributions and Component Prevalence by Group";

  return (
    <div className="flex gap-4 h-full">
      <div className="w-72 flex-shrink-0 space-y-4 overflow-y-auto pr-1" style={{ maxHeight: "calc(100vh - 120px)" }}>
        <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
          <h3 className="text-sm font-semibold text-gray-700">Groups</h3>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Burden / group column</label>
            <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
              {groupCols.map((name) => <option key={name} value={name}>{displayName(columns.find((col) => col.name === name), name)}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Histogram bins</label>
            <input type="range" min={4} max={30} value={bins} onChange={(e) => setBins(Number(e.target.value))} className="w-full accent-indigo-500" />
            <p className="text-[10px] text-gray-400 mt-1">{bins} bins</p>
          </div>
        </div>

        <ScoreBlock
          title="Score A"
          score={scoreA}
          setScore={setScoreA}
          label={labelA}
          setLabel={setLabelA}
          columns={columns}
          numericCols={numericCols.filter((name) => name !== scoreB)}
          componentOptions={componentOptions}
          components={activeComponentsA}
          setComponents={setComponentsA}
        />

        <ScoreBlock
          title="Score B"
          score={scoreB}
          setScore={setScoreB}
          label={labelB}
          setLabel={setLabelB}
          columns={columns}
          numericCols={numericCols.filter((name) => name !== scoreA)}
          componentOptions={componentOptions}
          components={activeComponentsB}
          setComponents={setComponentsB}
        />

        <button className="btn-primary w-full" onClick={run} disabled={loading || !canRun}>
          {loading ? "Generating Figure..." : "Generate Score Figure"}
        </button>
        {!canRun && (
          <p className="text-xs text-amber-600 leading-relaxed">
            Select two different score columns and at least one component for each score.
          </p>
        )}
        {error && <p className="text-red-500 text-xs">{error}</p>}
      </div>

      <div className="flex-1 panel min-h-0 bg-white border border-gray-200 shadow-sm rounded-2xl p-4 overflow-y-auto">
        {result && mergedLayout ? (
          <div className="space-y-4">
            {stale && (
              <StaleResultNotice reasons={staleWhy} onRecompute={run} busy={loading} what="This score figure" />
            )}
            <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <TitledPlot
              plotRefOut={plotRef}
              storageKey={`score-composite:${groupCol}:${scoreA}:${scoreB}`}
              data={paletteData ?? result.figure.data}
              layout={mergedLayout}
              config={{ responsive: true, displayModeBar: true, displaylogo: false }}
              defaultTitle={titleText}
              defaultSubtitle={result.method_note}
              defaultXAxis=""
              defaultYAxis=""
            />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {result.scores.map((score) => (
                <div key={score.score_col} className="rounded-lg border border-gray-200 bg-gray-50 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-semibold text-gray-800">{score.label}</p>
                    <p className="text-xs text-gray-500">{score.p_text}</p>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {result.groups.map((group) => `${group}: n=${score.n_by_group[group] ?? 0}`).join(" · ")}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {score.components.map((component) => (
                      <span key={component.component} className="px-2 py-1 rounded-md bg-white border border-gray-200 text-[11px] text-gray-600">
                        {component.label}: {component.p_text}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            </StaleGuard>
          </div>
        ) : (
          <div className="h-full min-h-[420px] flex items-center justify-center text-gray-400">
            Configure and generate a score composite figure
          </div>
        )}
      </div>
    </div>
  );
}

interface ScoreBlockProps {
  title: string;
  score: string;
  setScore: (value: string) => void;
  label: string;
  setLabel: (value: string) => void;
  columns: ColMeta[];
  numericCols: string[];
  componentOptions: string[];
  components: string[];
  setComponents: (value: string[]) => void;
}

function ScoreBlock({
  title,
  score,
  setScore,
  label,
  setLabel,
  columns,
  numericCols,
  componentOptions,
  components,
  setComponents,
}: ScoreBlockProps) {
  return (
    <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
      <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
      <div>
        <label className="text-xs text-gray-400 block mb-1">Score column</label>
        <select className="select w-full" value={score} onChange={(e) => setScore(e.target.value)}>
          {numericCols.map((name) => <option key={name} value={name}>{displayName(columns.find((col) => col.name === name), name)}</option>)}
        </select>
      </div>
      <div>
        <label className="text-xs text-gray-400 block mb-1">Display label</label>
        <input
          className="select w-full"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder={displayName(columns.find((col) => col.name === score), score)}
        />
      </div>
      <div>
        <label className="text-xs text-gray-400 block mb-1">Components</label>
        <select
          multiple
          className="select w-full min-h-32"
          value={components}
          onChange={(e) => setComponents(selectedValues(e.currentTarget))}
        >
          {componentOptions.map((name) => <option key={name} value={name}>{displayName(columns.find((col) => col.name === name), name)}</option>)}
        </select>
      </div>
    </div>
  );
}
