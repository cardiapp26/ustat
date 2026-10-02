/** Split File (SPSS SPLIT FILE): choose a grouping variable; analyses then run
 *  for each of its levels, picked from the banner on every tab. */
import { useState } from "react";
import { X } from "lucide-react";
import { getSplitLevels } from "../../api";
import { useStore, type ColMeta, type SplitFile } from "../../store";

interface Props {
  columns: ColMeta[];
  sessionId: string;
  current: SplitFile | null;
  onClose: () => void;
}

function detailOf(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  return typeof detail === "string" ? detail : "Could not read the groups of that column.";
}

export default function SplitFileModal({ columns, sessionId, current, onClose }: Props) {
  const setSplitFile = useStore((s) => s.setSplitFile);
  const candidates = columns.filter((c) => c.kind === "categorical" || c.kind === "ordinal");
  const [column, setColumn] = useState(current?.column ?? candidates[0]?.name ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apply = async () => {
    setBusy(true); setError(null);
    try {
      const res = await getSplitLevels(sessionId, column);
      const levels = (res.data as { levels: Array<{ level: string; n: number }> }).levels;
      setSplitFile({ column, levels, level: levels[0]?.level ?? null });
      onClose();
    } catch (e: unknown) {
      setError(detailOf(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/30 flex items-center justify-center" role="dialog" aria-modal="true" aria-label="Split File">
      <div className="bg-white rounded-xl shadow-2xl w-[min(460px,94vw)] p-5 space-y-3">
        <div className="flex items-center">
          <h3 className="font-semibold text-gray-900">Split File</h3>
          <button onClick={onClose} aria-label="Close" className="ml-auto p-1 text-gray-400 hover:text-gray-700"><X size={16} /></button>
        </div>
        <p className="text-xs text-gray-600 leading-relaxed">
          Every analysis runs separately for each group of this variable, as SPSS&apos;s Split File does.
          Pick the group on view from the banner; when an analysis runs, the other groups are computed
          in the background, so switching between them is immediate. Rows with this variable missing
          belong to no group.
        </p>
        <label className="block">
          <span className="text-xs text-gray-500">Split by</span>
          <select value={column} onChange={(e) => setColumn(e.target.value)} className="select w-full mt-1">
            {candidates.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
          </select>
        </label>
        {candidates.length === 0 && <p className="text-xs text-gray-500">No categorical column to split by.</p>}
        {error && <p className="text-xs text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1.5" role="alert">{error}</p>}
        <div className="flex items-center gap-2 pt-1">
          {current && (
            <button onClick={() => { setSplitFile(null); onClose(); }} disabled={busy}
              className="text-xs px-3 py-1.5 rounded border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:opacity-40">
              Turn split off
            </button>
          )}
          <button onClick={apply} disabled={busy || !column} className="ml-auto btn-primary text-xs px-3 py-1.5 disabled:opacity-40">
            {busy ? "Reading groups…" : "Split by this variable"}
          </button>
        </div>
      </div>
    </div>
  );
}
