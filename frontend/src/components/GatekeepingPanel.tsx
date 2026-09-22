import { useState } from "react";
import { runGatekeeping } from "../api";
import { Tip } from "./Tip";
import ResultExporter from "./ResultExporter";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";
import { fmtP } from "../lib/format";
import { describeStale } from "../lib/resultStamp";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";

interface Hyp { label: string; p: string }
interface Family { name: string; gamma: string; hyps: Hyp[] }

interface ResultHypothesis {
  label?: string;
  p_raw?: number;
  p_adjusted?: number;
  reject?: boolean;
}

interface ResultFamily {
  name?: string;
  gamma?: number;
  n_rejected?: number;
  n?: number;
  hypotheses: ResultHypothesis[];
}

interface GatekeepingResult {
  method?: string;
  logic?: string;
  alpha?: number;
  families: ResultFamily[];
  export_rows?: (string | number | null | undefined)[][];
  interpretation?: string;
}

const SAMPLE: Family[] = [
  { name: "Primary", gamma: "", hyps: [{ label: "All-cause death", p: "0.012" }] },
  { name: "Secondary", gamma: "", hyps: [
    { label: "MI", p: "0.02" }, { label: "Stroke", p: "0.04" }, { label: "HF hospitalisation", p: "0.30" },
  ] },
];

export default function GatekeepingPanel() {
  // Persisted alongside the result: a result restored on remount next to
  // inputs reset to the sample would read as computed from different settings.
  const [families, setFamilies] = usePersistedPanelState<Family[]>("gatekeeping", "families", SAMPLE);
  const [method, setMethod] = usePersistedPanelState<"hochberg" | "holm">("gatekeeping", "method", "hochberg");
  const [logic, setLogic] = usePersistedPanelState<"serial" | "parallel">("gatekeeping", "logic", "serial");
  const [alpha, setAlpha] = usePersistedPanelState<string>("gatekeeping", "alpha", "0.05");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setFam = (i: number, patch: Partial<Family>) =>
    setFamilies((fs) => fs.map((f, j) => (j === i ? { ...f, ...patch } : f)));
  const setHyp = (fi: number, hi: number, patch: Partial<Hyp>) =>
    setFamilies((fs) => fs.map((f, j) => j === fi ? { ...f, hyps: f.hyps.map((h, k) => k === hi ? { ...h, ...patch } : h) } : f));
  const addHyp = (fi: number) => setFamilies((fs) => fs.map((f, j) => j === fi ? { ...f, hyps: [...f.hyps, { label: "", p: "" }] } : f));
  const delHyp = (fi: number, hi: number) => setFamilies((fs) => fs.map((f, j) => j === fi ? { ...f, hyps: f.hyps.filter((_, k) => k !== hi) } : f));
  const addFamily = () => setFamilies((fs) => [...fs, { name: `Family ${fs.length + 1}`, gamma: "", hyps: [{ label: "", p: "" }] }]);
  const delFamily = (fi: number) => setFamilies((fs) => fs.filter((_, j) => j !== fi));

  // The request body is the whole input: gatekeeping adjusts the p-values
  // typed in here and reads no dataset, so a cell edit cannot invalidate it.
  // Stamping the body rather than the raw rows also means an added blank row,
  // which is filtered out before sending, does not mark the result stale.
  const payload = {
    method, logic, alpha: Number(alpha) || 0.05,
    families: families.map((f) => ({
      name: f.name,
      gamma: f.gamma.trim() === "" ? undefined : Number(f.gamma),
      hypotheses: f.hyps.filter((h) => h.label.trim() !== "" && h.p.trim() !== "")
        .map((h) => ({ label: h.label, p: Number(h.p) })),
    })).filter((f) => f.hypotheses.length > 0),
  };
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<GatekeepingResult>("gatekeeping", payload, { dependsOnData: false });

  const run = async () => {
    if (payload.families.length === 0) { setError("Enter at least one family with a hypothesis + p-value."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await runGatekeeping(payload);
      setResult(res.data);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: unknown } }; message?: string };
      const detail = err?.response?.data?.detail;
      setError(Array.isArray(detail) ? detail.map((m: { msg?: string }) => m.msg ?? String(m)).join(", ")
        : (typeof detail === "string" ? detail : (err?.message ?? "Failed")));
    } finally { setLoading(false); }
  };

  return (
    <div className="flex gap-4">
      {/* Controls */}
      <div className="w-[420px] flex-shrink-0 space-y-3">
        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
            Gatekeeping (multiplicity)
            <Tip wide text="Multistage gatekeeping across ordered families of hypotheses (e.g. primary then secondary endpoints) with a truncated Holm or Hochberg test within each family (Dmitrienko, Tamhane & Wiens 2008). Serial = the next family is tested only if every hypothesis in the prior families was rejected; parallel = if at least one in the immediately preceding family was rejected. The truncation γ (0–1) reserves Bonferroni mass to open the gate; leave blank for the default (0.5 non-terminal, 1.0 terminal). Controls family-wise error at α." />
          </h3>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-2 text-[10px] text-amber-800 leading-normal flex items-start gap-1">
            <span>💡</span>
            <span><strong>Illustrative Sample:</strong> The default values below are sample dummy data. Please replace them with your own hypothesis names and raw p-values.</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] text-gray-500">Within-family</span>
              <select value={method} onChange={(e) => setMethod(e.target.value as "hochberg" | "holm")} className="text-xs border border-gray-300 rounded px-2 py-1 bg-white">
                <option value="hochberg">Hochberg</option>
                <option value="holm">Holm</option>
              </select>
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] text-gray-500">Logic</span>
              <select value={logic} onChange={(e) => setLogic(e.target.value as "serial" | "parallel")} className="text-xs border border-gray-300 rounded px-2 py-1 bg-white">
                <option value="serial">Serial</option>
                <option value="parallel">Parallel</option>
              </select>
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="text-[10px] text-gray-500">α (FWER)</span>
              <input value={alpha} onChange={(e) => setAlpha(e.target.value)} className="text-xs border border-gray-300 rounded px-2 py-1" />
            </label>
          </div>
        </div>

        {families.map((f, fi) => (
          <div key={fi} className="panel space-y-1.5">
            <div className="flex items-center gap-2">
              <input value={f.name} onChange={(e) => setFam(fi, { name: e.target.value })}
                className="flex-1 text-xs font-semibold border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" placeholder="Family name" />
              <label className="flex items-center gap-1 text-[10px] text-gray-500">
                γ
                <input value={f.gamma} onChange={(e) => setFam(fi, { gamma: e.target.value })}
                  className="w-12 text-xs border border-gray-200 rounded px-1.5 py-1 text-center focus:outline-none focus:border-indigo-400" placeholder="auto" />
              </label>
              {families.length > 1 && <button onClick={() => delFamily(fi)} className="text-gray-300 hover:text-red-500 text-xs">✕</button>}
            </div>
            {f.hyps.map((h, hi) => (
              <div key={hi} className="flex items-center gap-1.5">
                <input value={h.label} onChange={(e) => setHyp(fi, hi, { label: e.target.value })}
                  className="flex-1 text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" placeholder="Hypothesis label" />
                <input value={h.p} onChange={(e) => setHyp(fi, hi, { p: e.target.value })}
                  className="w-20 text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" placeholder="raw p" />
                <button onClick={() => delHyp(fi, hi)} className="text-gray-300 hover:text-red-500 text-xs">✕</button>
              </div>
            ))}
            <button onClick={() => addHyp(fi)} className="text-[10px] px-2 py-0.5 rounded border border-indigo-200 text-indigo-600 hover:bg-indigo-50">+ Hypothesis</button>
          </div>
        ))}

        <div className="flex gap-2">
          <button onClick={addFamily} className="text-xs px-3 py-1.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50">+ Family</button>
          <button onClick={() => setFamilies(SAMPLE)} className="text-xs px-3 py-1.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50">Sample</button>
          <button onClick={run} disabled={loading}
            className="flex-1 px-4 py-1.5 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
            {loading ? "Running…" : "Run gatekeeping"}
          </button>
        </div>
        {error && <p className="text-xs text-red-500">{error}</p>}
      </div>

      {/* Results */}
      <div className="flex-1 min-w-0 space-y-3">
        {!result && !error && (
          <div className="flex items-center justify-center h-64 border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
            Build ordered families of hypotheses, then run
          </div>
        )}
        {result && stale && (
          <StaleResultNotice
            reasons={staleWhy}
            onRecompute={run}
            busy={loading}
            what="This gatekeeping result"
          />
        )}
        {result && (
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-800">
                  Adjusted p-values <span className="text-gray-400 font-normal">· {result.method} · {result.logic} · α={result.alpha}</span>
                </h4>
                {result.export_rows && (
                  <ResultExporter title="Gatekeeping" headers={result.export_rows[0].map((h) => String(h ?? ""))} rows={result.export_rows.slice(1)} />
                )}
              </div>
              {result.families.map((f: ResultFamily, fi: number) => (
                <div key={fi} className="space-y-1">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="font-semibold text-gray-700">{f.name}</span>
                    <span className="text-gray-400">γ = {f.gamma} · {f.n_rejected}/{f.n} rejected</span>
                  </div>
                  <div className="overflow-auto rounded-lg border border-gray-200">
                    <table className="w-full text-[11px] border-collapse">
                      <thead>
                        <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                          <th className="text-left px-2 py-1 font-medium">Hypothesis</th>
                          <th className="text-right px-2 py-1 font-medium">Raw <i>p</i></th>
                          <th className="text-right px-2 py-1 font-medium">Adjusted <i>p</i></th>
                          <th className="text-center px-2 py-1 font-medium">Decision</th>
                        </tr>
                      </thead>
                      <tbody>
                        {f.hypotheses.map((h: ResultHypothesis, hi: number) => (
                          <tr key={hi} className={`border-b border-gray-100 ${h.reject ? "bg-indigo-50/40" : ""}`}>
                            <td className="px-2 py-1 font-mono text-gray-800">{h.label}</td>
                            <td className="px-2 py-1 font-mono text-right text-gray-500">{h.p_raw}</td>
                            <td className={`px-2 py-1 font-mono text-right ${h.reject ? "text-indigo-700 font-semibold" : "text-gray-600"}`}>
                              {h.p_adjusted == null ? "—" : h.p_adjusted >= 1 ? "1.000" : fmtP(h.p_adjusted)}
                            </td>
                            <td className="px-2 py-1 text-center">
                              <span className={`inline-block text-[10px] font-semibold border rounded-full px-1.5 py-0.5 ${
                                h.reject ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-gray-50 text-gray-500 border-gray-200"}`}>
                                {h.reject ? "Reject H₀" : "Retain"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
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
