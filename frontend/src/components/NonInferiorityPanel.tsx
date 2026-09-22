import { useState, useEffect, type ReactNode } from "react";
import { useStore, isNumericKind } from "../store";
import { runNonInferiority, getUniqueValues } from "../api";
import { useStampedResult } from "../hooks/useStampedResult";
import { describeStale } from "../lib/resultStamp";
import { Tip } from "./Tip";
import ResultExporter from "./ResultExporter";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";

interface Assumption {
  name: string;
  detail: string;
  met: boolean;
}

interface NonInferiorityResult {
  non_inferior?: boolean;
  effect?: string;
  estimate?: number | string;
  ci_level?: number | string;
  ci_low?: number | string;
  ci_high?: number | string;
  margin?: number | string;
  bound?: string;
  alpha_one_sided?: number | string;
  p_noninferiority?: number;
  test_group?: string;
  ref_group?: string;
  outcome_type?: string;
  n_test?: number | null;
  n_ref?: number;
  events_test?: number;
  events_ref?: number;
  p_test?: number;
  p_ref?: number;
  mean_test?: number | string;
  mean_ref?: number | string;
  export_rows?: string[][];
  assumptions?: Assumption[];
  interpretation?: string;
}

interface ApiErrorDetail {
  msg?: string;
}

export default function NonInferiorityPanel() {
  const session = useStore((s) => s.session);
  const columns = session?.columns ?? [];
  const sid = session?.session_id ?? "";
  const numCols = columns.filter((c) => isNumericKind(c.kind)).map((c) => c.name);

  const [outcomeType, setOutcomeType] = useState<"binary" | "continuous">("binary");
  const [outcomeCol, setOutcomeCol] = useState("");
  const [groupCol, setGroupCol] = useState("");
  const [levels, setLevels] = useState<string[]>([]);
  const [testGroup, setTestGroup] = useState("");
  const [refGroup, setRefGroup] = useState("");
  const [effect, setEffect] = useState<"RR" | "RD" | "OR">("RR");
  const [margin, setMargin] = useState("1.20");
  const [bound, setBound] = useState<"upper" | "lower">("upper");
  const [alpha, setAlpha] = useState("0.05");
  // The test reads the session's rows (session_id is in the request), so it
  // depends on the data as well as on these settings.
  const runParams = { outcomeType, outcomeCol, groupCol, testGroup, refGroup, effect, margin, bound, alpha };
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<NonInferiorityResult>("noninferiority", runParams);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setResult(null);
    if (!groupCol || !sid) { setLevels([]); return; }
    getUniqueValues(sid, groupCol).then((res) => {
      const vals = (res.data?.values ?? res.data ?? []).map((v: unknown) => String(v)).slice(0, 50);
      setLevels(vals);
      if (vals.length === 2) { setTestGroup(vals[1]); setRefGroup(vals[0]); }
    }).catch(() => setLevels([]));
    // setResult is rebuilt every render (it closes over that render's
    // settings); listing it would refetch the levels and clear the result on
    // each one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groupCol, sid]);

  const ciLevel = (() => { const a = Number(alpha) || 0.05; return ((1 - 2 * a) * 100).toFixed(0); })();

  const run = async () => {
    if (!outcomeCol || !groupCol) { setError("Select outcome and group columns."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await runNonInferiority({
        session_id: sid, outcome_col: outcomeCol, group_col: groupCol,
        test_group: testGroup || undefined, ref_group: refGroup || undefined,
        outcome_type: outcomeType, effect, margin: Number(margin), bound, alpha: Number(alpha),
      });
      setResult(res.data);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      const message = e instanceof Error ? e.message : String(e);
      setError(Array.isArray(detail) ? detail.map((m: ApiErrorDetail) => m.msg ?? String(m)).join(", ")
        : (typeof detail === "string" ? detail : (message || "Failed")));
    } finally { setLoading(false); }
  };

  return (
    <div className="flex gap-4">
      {/* Controls */}
      <div className="w-80 flex-shrink-0 space-y-3">
        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
            Non-Inferiority Test
            <Tip wide text="Regulatory-style margin testing for a two-arm trial (supply the ITT or per-protocol dataset). A one-sided α corresponds to a two-sided (1−2α) CI — α=0.05 ↔ 90% CI, the standard non-inferiority convention. Non-inferiority is concluded from the relevant CI bound vs the prespecified margin: bound 'upper' → non-inferior if the upper CI bound < margin (harmful event, margin>1 for RR/OR); bound 'lower' → non-inferior if the lower CI bound > margin (preserve benefit)." />
          </h3>

          <div className="flex rounded-lg overflow-hidden border border-gray-300">
            {(["binary", "continuous"] as const).map((t) => (
              <button key={t} onClick={() => { setOutcomeType(t); setResult(null); }}
                className={`flex-1 px-2 py-1.5 text-xs font-medium transition-colors ${outcomeType === t ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"}`}>
                {t === "binary" ? "Binary (RR/RD/OR)" : "Continuous (mean)"}
              </button>
            ))}
          </div>

          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Outcome {outcomeType === "binary" ? "(0/1)" : "(numeric)"}</span>
            <select value={outcomeCol} onChange={(e) => { setOutcomeCol(e.target.value); setResult(null); }}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
              <option value="">— select —</option>
              {(outcomeType === "continuous" ? numCols : columns.map((c) => c.name)).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Treatment arm (group)</span>
            <select value={groupCol} onChange={(e) => setGroupCol(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
              <option value="">— select —</option>
              {columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </label>

          {levels.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              <label className="flex flex-col gap-0.5">
                <span className="text-[10px] text-gray-500">Test (new) arm</span>
                <select value={testGroup} onChange={(e) => setTestGroup(e.target.value)}
                  className="text-xs border border-gray-300 rounded px-2 py-1 bg-white">
                  {levels.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
              <label className="flex flex-col gap-0.5">
                <span className="text-[10px] text-gray-500">Reference arm</span>
                <select value={refGroup} onChange={(e) => setRefGroup(e.target.value)}
                  className="text-xs border border-gray-300 rounded px-2 py-1 bg-white">
                  {levels.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            </div>
          )}

          {outcomeType === "binary" && (
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-500 font-medium">Effect measure</span>
              <div className="flex rounded-lg overflow-hidden border border-gray-300">
                {(["RR", "RD", "OR"] as const).map((e) => (
                  <button key={e} onClick={() => setEffect(e)}
                    className={`flex-1 px-2 py-1 text-[11px] ${effect === e ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"}`}>
                    {e}
                  </button>
                ))}
              </div>
            </label>
          )}

          <div className="grid grid-cols-3 gap-2">
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] text-gray-500">Margin</span>
              <input value={margin} onChange={(e) => setMargin(e.target.value)}
                className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] text-gray-500">Bound</span>
              <select value={bound} onChange={(e) => setBound(e.target.value as "upper" | "lower")}
                className="text-xs border border-gray-300 rounded px-2 py-1 bg-white">
                <option value="upper">Upper</option>
                <option value="lower">Lower</option>
              </select>
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] text-gray-500">α (1-sided)</span>
              <input value={alpha} onChange={(e) => setAlpha(e.target.value)}
                className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
            </label>
          </div>
          <p className="text-[10px] text-gray-400">→ two-sided {ciLevel}% CI</p>

          <button onClick={run} disabled={loading}
            className="w-full px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
            {loading ? "Testing…" : "Run non-inferiority test"}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>
      </div>

      {/* Results */}
      <div className="flex-1 min-w-0 space-y-3">
        {!result && !error && (
          <div className="flex items-center justify-center h-64 border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
            Configure the margin test, then run
          </div>
        )}
        {result && stale && (
          <StaleResultNotice
            reasons={staleWhy}
            onRecompute={run}
            busy={loading}
            what="This non-inferiority test"
          />
        )}
        {result && (
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <div className={`panel border-2 ${result.non_inferior ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <div className="flex items-center gap-3">
                <span className="text-2xl">{result.non_inferior ? "✅" : "⚠️"}</span>
                <div>
                  <p className={`font-bold text-sm ${result.non_inferior ? "text-emerald-800" : "text-amber-800"}`}>
                    {result.non_inferior ? "Non-inferiority demonstrated" : "Non-inferiority NOT demonstrated"}
                  </p>
                  <p className="text-xs text-gray-600">
                    {result.effect} = {result.estimate} · {result.ci_level}% CI {result.ci_low}–{result.ci_high} · margin {result.margin}
                  </p>
                </div>
              </div>
            </div>

            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-800">{result.test_group} vs {result.ref_group}</h4>
                {result.export_rows && (
                  <ResultExporter title="NonInferiority" headers={result.export_rows[0]} rows={result.export_rows.slice(1)} />
                )}
              </div>
              <div className="grid grid-cols-2 gap-2">
                {([
                  [result.effect, result.estimate],
                  [`${result.ci_level}% CI`, `${result.ci_low} – ${result.ci_high}`],
                  ["Margin", result.margin],
                  ["Bound tested", result.bound],
                  ["1-sided α", result.alpha_one_sided],
                  [<><i>p</i> (NI)</>, result.p_noninferiority == null ? "—" : result.p_noninferiority < 0.001 ? "<0.001" : result.p_noninferiority],
                ] as [ReactNode, ReactNode][]).map(([k, v], i) => (
                  <div key={i} className="bg-gray-50 border border-gray-200 rounded p-2 text-center">
                    <p className="text-[9px] text-gray-400">{k}</p>
                    <p className="font-semibold text-gray-800 text-xs font-mono">{v}</p>
                  </div>
                ))}
              </div>
              {(result.n_test != null) && (
                <p className="text-[11px] text-gray-500">
                  {result.outcome_type === "binary"
                    ? `Test: ${result.events_test}/${result.n_test} (${result.p_test == null ? "—" : (result.p_test * 100).toFixed(1)}%) · Ref: ${result.events_ref}/${result.n_ref} (${result.p_ref == null ? "—" : (result.p_ref * 100).toFixed(1)}%)`
                    : <>Test: mean {result.mean_test} (<i>n</i>={result.n_test}) · Ref: mean {result.mean_ref} (<i>n</i>={result.n_ref})</>}
                </p>
              )}
            </div>

            {result.assumptions?.map((a: Assumption, i: number) => (
              <div key={i} className={`flex items-start gap-2 text-xs px-3 py-1.5 rounded-lg ${a.met ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                <span>{a.met ? "✓" : "⚠"}</span>
                <span><span className="font-medium">{a.name}</span> — {a.detail}</span>
              </div>
            ))}

            {result.interpretation && (
              <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3 py-2 text-xs text-indigo-900 leading-relaxed">
                {result.interpretation}
              </div>
            )}
          </StaleGuard>
        )}
      </div>
    </div>
  );
}
