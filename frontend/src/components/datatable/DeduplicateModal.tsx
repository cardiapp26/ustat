import { useCallback, useEffect, useState } from "react";
import type { ColMeta } from "../../store";
import { deduplicateRows, type DeduplicateResult } from "../../api";

/**
 * Delete duplicate rows, by a key the user picks.
 *
 * The key is the whole decision. Two visits by one patient share an identity
 * number and are two records; a row pasted twice is one. Nothing here guesses
 * which of those a dataset holds — it asks, counts what the answer would
 * delete, and only then offers the button.
 */
export function DeduplicateModal({
  columns,
  sessionId,
  initialKey,
  onDone,
  onClose,
}: {
  columns: ColMeta[];
  sessionId: string;
  /** Column the user right-clicked from, pre-ticked as the likely key. */
  initialKey?: string;
  onDone: (deleted: number) => void;
  onClose: () => void;
}) {
  const [keys, setKeys] = useState<Set<string>>(
    () => new Set(initialKey ? [initialKey] : [])
  );
  const [keep, setKeep] = useState<"first" | "last">("first");
  const [count, setCount] = useState<DeduplicateResult | null>(null);
  const [counting, setCounting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // No ticks means "identical everywhere", which is what the backend reads an
  // empty key list as. Saying so beats an empty panel that looks like nothing
  // is selected yet.
  const keyList = [...keys];
  const wholeRow = keyList.length === 0;

  const toggle = (name: string) =>
    setKeys((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });

  const recount = useCallback(async () => {
    setCounting(true); setError(null);
    try {
      const res = await deduplicateRows(sessionId, keyList, keep, true);
      setCount(res.data);
    } catch {
      setError("Could not count the duplicates");
      setCount(null);
    } finally {
      setCounting(false);
    }
    // keyList is rebuilt every render; the Set it comes from is the real dep.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, keys, keep]);

  // The count has to follow the key, or the number on screen describes a
  // choice the user already moved on from — and that number is the only thing
  // standing between them and a delete.
  useEffect(() => { void recount(); }, [recount]);

  const handleDelete = async () => {
    setBusy(true); setError(null);
    try {
      const res = await deduplicateRows(sessionId, keyList, keep, false);
      onDone(res.data.deleted);
    } catch {
      setError("Delete failed");
      setBusy(false);
    }
  };

  const nDup = count?.duplicate_rows ?? 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg flex flex-col" style={{ maxHeight: "90vh" }}>

        <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-gray-100">
          <div>
            <h2 className="text-base font-semibold text-gray-900">Delete duplicate rows</h2>
            <p className="text-xs text-gray-400 mt-0.5">Rows repeating an earlier row's key</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100">✕</button>
        </div>

        <div className="px-6 py-4 flex flex-col gap-4 overflow-y-auto" style={{ minHeight: 0 }}>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide">
                Rows match when these agree
              </p>
              <div className="flex gap-1.5">
                <button
                  onClick={() => setKeys(new Set())}
                  className="text-[10px] px-2 py-0.5 rounded border border-gray-200 text-gray-500 hover:border-gray-300 hover:text-gray-700"
                >
                  Whole row
                </button>
                {columns[0] && (
                  <button
                    onClick={() => setKeys(new Set([columns[0].name]))}
                    className="text-[10px] px-2 py-0.5 rounded border border-gray-200 text-gray-500 hover:border-gray-300 hover:text-gray-700"
                  >
                    First column
                  </button>
                )}
              </div>
            </div>

            <div className="border border-gray-200 rounded-xl max-h-56 overflow-y-auto divide-y divide-gray-50">
              {columns.map((c) => (
                <label
                  key={c.name}
                  className="flex items-center gap-2 px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 accent-indigo-500"
                    checked={keys.has(c.name)}
                    onChange={() => toggle(c.name)}
                  />
                  <span className="truncate">{c.name}</span>
                  <span className="ml-auto text-[10px] text-gray-300">{c.kind}</span>
                </label>
              ))}
            </div>

            {wholeRow && (
              <p className="text-[11px] text-gray-400 mt-1.5">
                Nothing ticked — only rows identical in every column count as duplicates.
              </p>
            )}
          </div>

          <div>
            <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide mb-1.5">Of each group, keep</p>
            <div className="flex gap-2">
              {(["first", "last"] as const).map((k) => (
                <button
                  key={k}
                  onClick={() => setKeep(k)}
                  className={`text-xs px-3 py-1.5 rounded-lg border transition-colors
                    ${keep === k
                      ? "bg-indigo-50 text-indigo-700 border-indigo-300"
                      : "text-gray-500 border-gray-200 hover:border-gray-300"}`}
                >
                  {k === "first" ? "First occurrence" : "Last occurrence"}
                </button>
              ))}
            </div>
          </div>

          {/* The count is the safety rail: no delete button appears until the
              user has seen how many rows the current key would take. */}
          <div className="rounded-xl bg-gray-50 border border-gray-100 px-3 py-2.5 text-xs text-gray-600">
            {counting ? (
              <span className="text-gray-400">Counting…</span>
            ) : count ? (
              <>
                <span className={nDup > 0 ? "font-semibold text-gray-900" : ""}>
                  {nDup.toLocaleString()} row{nDup === 1 ? "" : "s"} would be deleted
                </span>
                <span className="text-gray-400"> · {count.remaining_rows.toLocaleString()} left</span>
                {count.blank_key_rows > 0 && (
                  <div className="text-[11px] text-gray-400 mt-1">
                    {count.blank_key_rows.toLocaleString()} row
                    {count.blank_key_rows === 1 ? " has" : "s have"} no value in any key column and
                    {count.blank_key_rows === 1 ? " is" : " are"} left alone — a row that says nothing
                    about its identity cannot duplicate another.
                  </div>
                )}
              </>
            ) : (
              <span className="text-gray-400">—</span>
            )}
          </div>

          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>

        <div className="flex items-center justify-between gap-2 px-6 py-3 border-t border-gray-100">
          <span className="text-[11px] text-gray-400">Undo restores the deleted rows</span>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="text-xs px-3 py-1.5 rounded-lg border border-gray-200 text-gray-600 hover:border-gray-300"
            >
              Cancel
            </button>
            <button
              onClick={handleDelete}
              disabled={busy || counting || nDup === 0}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-colors
                ${busy || counting || nDup === 0
                  ? "border-gray-200 text-gray-300 cursor-not-allowed"
                  : "bg-red-500 text-white border-red-500 hover:bg-red-600"}`}
            >
              {busy ? "Deleting…" : `Delete ${nDup.toLocaleString()} row${nDup === 1 ? "" : "s"}`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
