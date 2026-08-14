import { useEffect, useMemo, useRef, useState } from "react";
import type { ColMeta } from "../../store";

/** Fill a derived column's blanks by computing them from other columns.
 *
 * The first version put a bare text box in the column context menu, which
 * meant typing column names by hand — unworkable for names like "Notrofil"
 * sitting three columns away, and worse for anything with a space in it. The
 * names are already known, so they belong in a list you click.
 *
 * Deliberately a small dialog rather than a submenu: a submenu closes on the
 * first stray pointer move, and building an expression takes several clicks
 * across a list and a keypad.
 */
export function FormulaFillModal({
  colName, columns, nBlanks, busy, error, onCancel, onSubmit,
}: {
  colName: string;
  columns: ColMeta[];
  nBlanks: number;
  busy: boolean;
  error: string | null;
  onCancel: () => void;
  onSubmit: (formula: string) => void;
}) {
  const [formula, setFormula] = useState("");
  const [search, setSearch] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const pressedBackdrop = useRef(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onCancel(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  // The column being filled is never a useful term in its own formula — every
  // cell it would contribute is one of the blanks being filled.
  const selectable = useMemo(
    () => columns.filter((c) => c.name !== colName),
    [columns, colName],
  );
  const shown = useMemo(() => {
    const q = search.trim().toLowerCase();
    return q ? selectable.filter((c) => c.name.toLowerCase().includes(q)) : selectable;
  }, [selectable, search]);

  /** Insert at the caret, not at the end — the caret is where the user left off. */
  const insert = (text: string) => {
    const el = inputRef.current;
    if (!el) { setFormula((f) => f + text); return; }
    const start = el.selectionStart ?? formula.length;
    const end = el.selectionEnd ?? formula.length;
    setFormula(formula.slice(0, start) + text + formula.slice(end));
    setTimeout(() => {
      el.focus();
      el.setSelectionRange(start + text.length, start + text.length);
    }, 0);
  };

  // A name that is not a plain identifier has to be quoted or the expression
  // parser reads it as several terms and fails on the space.
  const termFor = (name: string) =>
    /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : `\`${name}\``;

  const canRun = formula.trim().length > 0 && !busy;

  return (
    <div
      className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center"
      onPointerDown={(e) => { pressedBackdrop.current = e.target === e.currentTarget; }}
      onClick={(e) => { if (e.target === e.currentTarget && pressedBackdrop.current) onCancel(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="formula-fill-title"
        className="bg-white rounded-xl shadow-2xl w-[420px] max-h-[80vh] flex flex-col"
      >
        <div className="px-5 py-3.5 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h3 id="formula-fill-title" className="text-sm font-semibold text-gray-800">
              Compute blanks
            </h3>
            <p className="text-[11px] text-gray-400 mt-0.5">
              {nBlanks} blank{nBlanks === 1 ? "" : "s"} in <span className="font-mono">{colName}</span>
              <span className="ml-1 text-indigo-500">· recorded values are kept</span>
            </p>
          </div>
          <button onClick={onCancel} aria-label="Close"
            className="text-gray-400 hover:text-gray-600 text-lg cursor-pointer">✕</button>
        </div>

        <div className="px-5 py-3 space-y-2.5 overflow-y-auto">
          <input
            ref={inputRef}
            autoFocus
            className="w-full border border-gray-300 rounded-lg px-2.5 py-1.5 text-sm font-mono focus:border-indigo-500 focus:outline-none"
            placeholder="click a column below, or type"
            value={formula}
            onChange={(e) => setFormula(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && canRun) onSubmit(formula); }}
          />

          <div className="flex flex-wrap gap-1">
            {["+", "-", "*", "/", "(", ")", "**"].map((op) => (
              <button key={op} onClick={() => insert(op)}
                className="px-2 py-0.5 text-xs font-mono rounded border border-gray-300 hover:bg-gray-100 bg-white text-gray-700">
                {op}
              </button>
            ))}
            <button onClick={() => setFormula("")}
              className="px-2 py-0.5 text-[11px] rounded border border-gray-300 hover:bg-gray-100 bg-white text-gray-400 ml-auto">
              Clear
            </button>
          </div>

          <input
            className="w-full border border-gray-200 rounded px-2 py-1 text-xs focus:border-indigo-400 focus:outline-none"
            placeholder="Search columns…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="border border-gray-200 rounded-lg max-h-44 overflow-y-auto divide-y divide-gray-100">
            {shown.length === 0 ? (
              <p className="text-[11px] text-gray-400 px-2.5 py-2">No matching columns.</p>
            ) : shown.map((c) => (
              <button
                key={c.name}
                onClick={() => insert(termFor(c.name))}
                className="w-full text-left px-2.5 py-1 text-xs font-mono text-gray-700 hover:bg-indigo-50 hover:text-indigo-700 flex items-center gap-2"
                title={`Insert ${c.name}`}
              >
                <span className="truncate flex-1">{c.name}</span>
                <span className="text-[9px] text-gray-400 flex-shrink-0">{c.kind}</span>
              </button>
            ))}
          </div>

          {error && (
            <p className="text-[11px] text-red-600 bg-red-50 border border-red-200 rounded px-2 py-1">
              {error}
            </p>
          )}
        </div>

        <div className="px-5 py-3 border-t border-gray-200 flex items-center gap-2">
          <p className="text-[10px] text-gray-400 flex-1">
            Rows whose inputs are missing stay blank.
          </p>
          <button onClick={onCancel}
            className="px-3 py-1.5 text-xs rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-50">
            Cancel
          </button>
          <button
            onClick={() => onSubmit(formula)}
            disabled={!canRun}
            className="px-3 py-1.5 text-xs rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? "Computing…" : "Compute"}
          </button>
        </div>
      </div>
    </div>
  );
}
