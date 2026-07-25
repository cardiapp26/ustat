import { useState } from "react";
import { replaceColumnValues } from "../../api";
import { useStore } from "../../store";
import type { ColMeta, Session } from "../../store";

/** Per-column Find & Replace (value map, in place).
 *
 * Lists the column's distinct values; the user types a new value next to each.
 * Applying rewrites the ACTUAL cell data (not just display labels) and, when
 * every value ends up numeric, the column is auto-cast to numeric — so e.g.
 * female→0 / male→1 yields a real 0/1 predictor for analysis. Existing value
 * labels follow the mapping server-side. Value Labels remain a separate tool. */
export function FindReplaceModal({
  colName, columns, preview, session, onClose, onApplied,
}: {
  colName: string;
  columns: ColMeta[];
  preview: Record<string, unknown>[];
  session: Session;
  onClose: () => void;
  onApplied: () => void;
}) {
  const col = columns.find((c) => c.name === colName);
  const uniqueVals = Array.from(
    new Set(preview.map((r) => r[colName]).filter((v) => v !== null && v !== undefined && v !== ""))
  ).map(String).sort((a, b) => {
    const na = Number(a), nb = Number(b);
    return (!isNaN(na) && !isNaN(nb)) ? na - nb : a.localeCompare(b);
  });

  const [draft, setDraft] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Only values the user actually changed (non-empty and different).
  const mapping: Record<string, string> = {};
  for (const v of uniqueVals) {
    const next = (draft[v] ?? "").trim();
    if (next !== "" && next !== v) mapping[v] = next;
  }
  const nChanges = Object.keys(mapping).length;

  const handleApply = async () => {
    if (nChanges === 0) return;
    setLoading(true); setError(null);
    try {
      const res = await replaceColumnValues(session.session_id, { column: colName, mapping });
      const d = res.data;
      const newCol: ColMeta = {
        ...(col ?? { name: colName, dtype: d.dtype, kind: d.kind }),
        name: d.name, dtype: d.dtype, kind: d.kind,
        ...(d.value_labels ? { value_labels: d.value_labels } : {}),
      };
      useStore.getState().addSessionColumn(newCol, d.preview_values);
      onApplied();
      onClose();
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Replace failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-2xl w-[26rem] max-h-[80vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="px-5 py-3.5 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-800">Convert value</h3>
            <p className="text-[11px] text-gray-400 mt-0.5">
              {colName}
              {col?.kind && <span className="ml-1 text-indigo-500">({col.kind})</span>}
            </p>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-lg">✕</button>
        </div>

        <div className="px-5 pt-2 pb-1">
          <p className="text-[11px] text-gray-500 bg-indigo-50 border border-indigo-200 rounded px-2 py-1.5 leading-snug">
            Type a new value next to any you want to change; leave blank to keep.
            This rewrites the real data (no new column). If every value becomes a
            number, the column turns numeric automatically.
          </p>
        </div>

        {/* Value map */}
        <div className="flex-1 overflow-y-auto px-5 py-3 space-y-2">
          {uniqueVals.length === 0 ? (
            <p className="text-xs text-gray-400 text-center py-4">No values found</p>
          ) : (
            uniqueVals.map((val) => (
              <div key={val} className="flex items-center gap-2">
                <span className="max-w-[9rem] truncate text-xs font-mono text-gray-500 bg-gray-100 px-2 py-1 rounded flex-shrink-0" title={val}>
                  {val}
                </span>
                <span className="text-gray-400 text-xs">→</span>
                <input
                  className="flex-1 text-xs border border-gray-300 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-200"
                  placeholder="keep"
                  value={draft[val] ?? ""}
                  onChange={(e) => setDraft((prev) => ({ ...prev, [val]: e.target.value }))}
                />
              </div>
            ))
          )}
        </div>

        {error && <p className="px-5 text-xs text-red-500">{error}</p>}

        {/* Footer */}
        <div className="px-5 py-3 border-t border-gray-200 flex items-center justify-between">
          <button onClick={() => setDraft({})} className="text-xs text-gray-400 hover:text-red-500">Clear</button>
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-gray-400">{nChanges} change{nChanges === 1 ? "" : "s"}</span>
            <button onClick={onClose}
              className="px-3 py-1.5 text-xs text-gray-500 border border-gray-200 rounded-lg hover:bg-gray-50">
              Cancel
            </button>
            <button onClick={handleApply} disabled={loading || nChanges === 0}
              className="px-3 py-1.5 text-xs bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-40">
              {loading ? "Applying…" : "Apply"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
