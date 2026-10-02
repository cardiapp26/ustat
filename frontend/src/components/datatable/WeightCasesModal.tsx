/** Weight Cases (SPSS WEIGHT BY): pick a column of frequency weights. The
 *  server applies them once, for every analysis (backend services/
 *  case_weights.py). */
import { useState } from "react";
import { X } from "lucide-react";
import { clearWeightCases, setWeightCases } from "../../api";
import { useStore, type CaseWeight, type ColMeta } from "../../store";

interface Props {
  columns: ColMeta[];
  sessionId: string;
  current: CaseWeight | null;
  onClose: () => void;
}

function detailOf(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  return typeof detail === "string" ? detail : "Could not apply the weights.";
}

export default function WeightCasesModal({ columns, sessionId, current, onClose }: Props) {
  const setCaseWeight = useStore((s) => s.setCaseWeight);
  const numeric = columns.filter((c) => c.kind === "numeric" || c.kind === "ordinal");
  const [column, setColumn] = useState(current?.column ?? numeric[0]?.name ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apply = async () => {
    setBusy(true); setError(null);
    try {
      const res = await setWeightCases(sessionId, column);
      setCaseWeight(res.data as CaseWeight);
      onClose();
    } catch (e: unknown) {
      setError(detailOf(e));
    } finally {
      setBusy(false);
    }
  };

  const turnOff = async () => {
    setBusy(true); setError(null);
    try {
      await clearWeightCases(sessionId);
      setCaseWeight(null);
      onClose();
    } catch (e: unknown) {
      setError(detailOf(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/30 flex items-center justify-center" role="dialog" aria-modal="true" aria-label="Weight Cases">
      <div className="bg-white rounded-xl shadow-2xl w-[min(460px,94vw)] p-5 space-y-3">
        <div className="flex items-center">
          <h3 className="font-semibold text-gray-900">Weight Cases</h3>
          <button onClick={onClose} aria-label="Close" className="ml-auto p-1 text-gray-400 hover:text-gray-700"><X size={16} /></button>
        </div>
        <p className="text-xs text-gray-600 leading-relaxed">
          Each row counts as many cases as its weight, in every analysis: use this for
          aggregated data (one row per pattern with a count). Weights must be whole numbers;
          rows with a missing, zero or negative weight are left out, as in SPSS.
        </p>
        <p className="text-[11px] text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1.5 leading-snug">
          Not for sampling or survey weights (fractional, with strata or clusters): those need
          design-based standard errors, and counting rows would give wrong ones.
        </p>
        <label className="block">
          <span className="text-xs text-gray-500">Frequency weight column</span>
          <select value={column} onChange={(e) => setColumn(e.target.value)} className="select w-full mt-1">
            {numeric.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
          </select>
        </label>
        {numeric.length === 0 && <p className="text-xs text-gray-500">No numeric column to weight by.</p>}
        {error && <p className="text-xs text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1.5" role="alert">{error}</p>}
        <div className="flex items-center gap-2 pt-1">
          {current && (
            <button onClick={turnOff} disabled={busy} className="text-xs px-3 py-1.5 rounded border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:opacity-40">
              Turn weighting off
            </button>
          )}
          <button onClick={apply} disabled={busy || !column} className="ml-auto btn-primary text-xs px-3 py-1.5 disabled:opacity-40">
            {busy ? "Applying…" : "Weight by this column"}
          </button>
        </div>
      </div>
    </div>
  );
}
