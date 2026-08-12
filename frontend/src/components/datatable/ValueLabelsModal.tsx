import { useCallback, useEffect, useRef, useState } from "react";
import type { Dispatch, PointerEvent as ReactPointerEvent, SetStateAction } from "react";
import { saveMetadata } from "../../api";
import { useStore } from "../../store";
import type { ColMeta, Session } from "../../store";

/** Keep this much of the dialog on screen, so the header stays grabbable. */
const KEEP_VISIBLE = 60;

/** Modal for assigning human-readable labels to a column's distinct values.
 * Extracted from DataTable. */
export function ValueLabelsModal({
  colName, columns, preview, draft, setDraft, session, onClose,
}: {
  colName: string;
  columns: ColMeta[];
  preview: Record<string, unknown>[];
  draft: Record<string, string>;
  setDraft: Dispatch<SetStateAction<Record<string, string>>>;
  session: Session;
  onClose: () => void;
}) {
  const col = columns.find((c) => c.name === colName);
  const uniqueVals = Array.from(
    new Set(preview.map((r) => r[colName]).filter((v) => v !== null && v !== undefined && v !== ""))
  ).map(String).sort((a, b) => {
    const na = Number(a), nb = Number(b);
    return (!isNaN(na) && !isNaN(nb)) ? na - nb : a.localeCompare(b);
  });

  const panelRef = useRef<HTMLDivElement>(null);
  // Null until the dialog is first dragged: it stays centred by the flex
  // parent, so it does not have to be measured before the first paint.
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const dragRef = useRef<{ dx: number; dy: number } | null>(null);

  // A press that STARTS on the backdrop is a dismissal. A press that starts
  // inside the dialog is not, even when it ends on the backdrop — which is
  // what happens whenever a label is selected by dragging across it and the
  // pointer leaves the input. The click event then fires on the nearest common
  // ancestor, the backdrop, and closing on that threw away the user's typing
  // mid-selection.
  const pressedBackdrop = useRef(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const clamp = useCallback((x: number, y: number) => {
    const w = panelRef.current?.getBoundingClientRect().width ?? 384;
    return {
      // Left edge may go past the viewport as long as a strip stays grabbable;
      // the top never goes above 0, or the drag handle is unreachable.
      x: Math.min(Math.max(x, KEEP_VISIBLE - w), window.innerWidth - KEEP_VISIBLE),
      y: Math.min(Math.max(y, 0), window.innerHeight - KEEP_VISIBLE),
    };
  }, []);

  const startDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    // Not from the close button, and not from a text selection in the title.
    if ((e.target as HTMLElement).closest("button")) return;
    const r = panelRef.current?.getBoundingClientRect();
    if (!r) return;
    dragRef.current = { dx: e.clientX - r.left, dy: e.clientY - r.top };
    setPos({ x: r.left, y: r.top });
    e.currentTarget.setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  const onDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    const d = dragRef.current;
    if (!d) return;
    setPos(clamp(e.clientX - d.dx, e.clientY - d.dy));
  };

  const endDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const handleSaveLabels = async () => {
    const updatedCols = session.columns.map((c) =>
      c.name === colName ? { ...c, value_labels: { ...draft } } : c
    );
    useStore.getState().setSession({ ...session, columns: updatedCols });
    try {
      await saveMetadata(session.session_id, { [colName]: { value_labels: draft } });
    } catch { /* ignore */ }
    onClose();
  };

  return (
    <div
      className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center"
      onPointerDown={(e) => { pressedBackdrop.current = e.target === e.currentTarget; }}
      onClick={(e) => { if (e.target === e.currentTarget && pressedBackdrop.current) onClose(); }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="value-labels-title"
        className="bg-white rounded-xl shadow-2xl w-96 max-h-[80vh] flex flex-col"
        style={pos ? { position: "fixed", left: pos.x, top: pos.y, margin: 0 } : undefined}
      >
        {/* Header — also the drag handle */}
        <div
          onPointerDown={startDrag}
          onPointerMove={onDrag}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          className="px-5 py-3.5 border-b border-gray-200 flex items-center justify-between cursor-move select-none touch-none"
          title="Drag to move"
        >
          <div>
            <h3 id="value-labels-title" className="text-sm font-semibold text-gray-800">Value Labels</h3>
            <p className="text-[11px] text-gray-400 mt-0.5">
              {colName}
              {col?.kind && <span className="ml-1 text-indigo-500">({col.kind})</span>}
            </p>
          </div>
          <button onClick={onClose} aria-label="Close"
            className="text-gray-400 hover:text-gray-600 text-lg cursor-pointer">✕</button>
        </div>

        {/* Labels list */}
        <div className="flex-1 overflow-y-auto px-5 py-3 space-y-2">
          {uniqueVals.length === 0 ? (
            <p className="text-xs text-gray-400 text-center py-4">No values found</p>
          ) : (
            uniqueVals.map((val) => (
              <div key={val} className="flex items-center gap-2">
                <span className="w-14 text-xs font-mono text-gray-500 bg-gray-100 px-2 py-1 rounded text-center flex-shrink-0">
                  {val}
                </span>
                <span className="text-gray-400 text-xs">=</span>
                <input
                  className="flex-1 text-xs border border-gray-300 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-200"
                  placeholder={`Label for ${val}`}
                  value={draft[val] ?? ""}
                  onChange={(e) => setDraft((prev) => ({ ...prev, [val]: e.target.value }))}
                />
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-gray-200 flex items-center justify-between">
          <button
            onClick={() => { setDraft({}); }}
            className="text-xs text-gray-400 hover:text-red-500"
          >Clear all</button>
          <div className="flex gap-2">
            <button onClick={onClose}
              className="px-3 py-1.5 text-xs text-gray-500 border border-gray-200 rounded-lg hover:bg-gray-50">
              Cancel
            </button>
            <button onClick={handleSaveLabels}
              className="px-3 py-1.5 text-xs bg-indigo-600 text-white rounded-lg hover:bg-indigo-700">
              Save Labels
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
