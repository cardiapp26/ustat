import { useState, useEffect } from "react";
import type { ColMeta, CaseCondition, CaseOperator } from "../../store";
import { selectCases, clearCases, getUniqueValues } from "../../api";

const OPERATORS: { value: CaseOperator; label: string; noValue?: boolean }[] = [
  { value: "eq",          label: "=" },
  { value: "ne",          label: "≠" },
  { value: "gt",          label: ">" },
  { value: "lt",          label: "<" },
  { value: "gte",         label: "≥" },
  { value: "lte",         label: "≤" },
  { value: "contains",    label: "contains" },
  { value: "missing",     label: "is missing",     noValue: true },
  { value: "not_missing", label: "is not missing", noValue: true },
];

export function SelectCasesModal({
  columns,
  sessionId,
  existing,
  onApply,
  onClear,
  onClose,
}: {
  columns: ColMeta[];
  sessionId: string;
  existing: CaseCondition[];
  onApply: (conditions: CaseCondition[], selected: number, total: number,
            excludedRows: number[], excludedBeyondPreview: number) => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const defaultCond = (): CaseCondition => ({
    column: columns[0]?.name ?? "",
    operator: "eq",
    value: "",
    join: "AND",
  });

  const [conditions, setConditions] = useState<CaseCondition[]>(
    existing.length > 0 ? existing : [defaultCond()]
  );
  const [activeCond, setActiveCond] = useState(0);
  const [colValues, setColValues] = useState<Record<string, string[]>>({});
  const [valuesLoading, setValuesLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<{ selected: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [valSearch, setValSearch] = useState("");

  // Fetch unique values whenever active condition column changes
  const activeCol = conditions[activeCond]?.column ?? "";
  useEffect(() => {
    if (!activeCol || colValues[activeCol] !== undefined) return;
    setValuesLoading(true);
    getUniqueValues(sessionId, activeCol)
      .then((r) => {
        const vals: string[] = (r.data?.values ?? []).map(String);
        setColValues((prev) => ({ ...prev, [activeCol]: vals }));
      })
      .catch(() => setColValues((prev) => ({ ...prev, [activeCol]: [] })))
      .finally(() => setValuesLoading(false));
    // The body GUARDS against re-fetch via `colValues[activeCol] !== undefined`,
    // so we explicitly omit colValues from deps to avoid an infinite loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCol, sessionId]);

  const updateCond = (i: number, patch: Partial<CaseCondition>) => {
    setConditions((prev) => prev.map((c, idx) => idx === i ? { ...c, ...patch } : c));
    setPreview(null);
    setValSearch("");
    // If column changed, clear cached values so they reload
    if (patch.column) {
      setColValues((prev) => {
        const next = { ...prev };
        if (!(patch.column! in next)) return prev;
        return next;
      });
    }
  };

  const addCond = () => {
    setConditions((prev) => {
      const next = [...prev, defaultCond()];
      setActiveCond(next.length - 1);
      return next;
    });
    setValSearch("");
  };

  const removeCond = (i: number) => {
    setConditions((prev) => prev.filter((_, idx) => idx !== i));
    setActiveCond((prev) => Math.max(0, prev > i ? prev - 1 : prev));
  };

  const handlePreview = async () => {
    setBusy(true); setError(null);
    try {
      const res = await selectCases(sessionId, conditions, false);
      setPreview(res.data);
    } catch { setError("Preview failed"); }
    finally { setBusy(false); }
  };

  const handleApply = async () => {
    setBusy(true); setError(null);
    try {
      const res = await selectCases(sessionId, conditions, true);
      onApply(conditions, res.data.selected, res.data.total,
              res.data.excluded_rows ?? [], res.data.excluded_beyond_preview ?? 0);
    } catch { setError("Apply failed"); }
    finally { setBusy(false); }
  };

  const handleClear = async () => {
    setBusy(true);
    try {
      await clearCases(sessionId);
      onClear();
    } finally { setBusy(false); }
  };

  const activeValues = colValues[activeCol] ?? [];
  const filteredValues = valSearch
    ? activeValues.filter((v) => v.toLowerCase().includes(valSearch.toLowerCase()))
    : activeValues;
  const activeOpMeta = OPERATORS.find((o) => o.value === conditions[activeCond]?.operator);
  const showValuePanel = !activeOpMeta?.noValue;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl flex flex-col" style={{ maxHeight: "90vh" }}>

        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-gray-100">
          <div>
            <h2 className="text-base font-semibold text-gray-900">Select Cases</h2>
            <p className="text-xs text-gray-400 mt-0.5">All analyses use only the selected subset</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100">✕</button>
        </div>

        {/* Body — two columns */}
        <div className="flex" style={{ minHeight: 0, overflow: "hidden" }}>

          {/* Left: condition builder */}
          <div className="flex-1 flex flex-col gap-3 px-6 py-4 overflow-y-auto border-r border-gray-100" style={{ minWidth: 0 }}>
            <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide">Conditions</p>

            {conditions.map((cond, i) => {
              const opMeta = OPERATORS.find((o) => o.value === cond.operator);
              const isActive = activeCond === i;
              return (
                <div
                  key={i}
                  onClick={() => { setActiveCond(i); setValSearch(""); }}
                  className={`flex flex-col gap-1.5 rounded-xl p-3 border cursor-pointer transition-colors
                    ${isActive ? "border-violet-300 bg-violet-50" : "border-gray-200 hover:border-gray-300 bg-white"}`}
                >
                  {/* Join label */}
                  {i > 0 && (
                    <select
                      value={cond.join}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => updateCond(i, { join: e.target.value as "AND" | "OR" })}
                      className="text-[10px] w-14 border border-gray-200 rounded px-1.5 py-0.5 bg-white text-gray-600 focus:outline-none"
                    >
                      <option value="AND">AND</option>
                      <option value="OR">OR</option>
                    </select>
                  )}
                  {i === 0 && <span className="text-[10px] text-gray-400 font-medium">WHERE</span>}

                  <div className="flex items-center gap-2">
                    {/* Column */}
                    <select
                      value={cond.column}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => { setActiveCond(i); updateCond(i, { column: e.target.value, value: "" }); }}
                      className="flex-1 text-xs border border-gray-300 rounded-lg px-2 py-1.5 bg-white text-gray-700 focus:outline-none focus:border-violet-400 min-w-0"
                    >
                      {columns.map((c) => (
                        <option key={c.name} value={c.name}>{c.name}</option>
                      ))}
                    </select>

                    {/* Operator */}
                    <select
                      value={cond.operator}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => updateCond(i, { operator: e.target.value as CaseOperator, value: "" })}
                      className="text-xs border border-gray-300 rounded-lg px-2 py-1.5 w-28 bg-white text-gray-700 focus:outline-none focus:border-violet-400 flex-shrink-0"
                    >
                      {OPERATORS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>

                    {/* Remove */}
                    <button
                      onClick={(e) => { e.stopPropagation(); removeCond(i); }}
                      disabled={conditions.length === 1}
                      className="flex-shrink-0 p-1 text-gray-300 hover:text-red-400 disabled:opacity-20 transition-colors"
                    >✕</button>
                  </div>

                  {/* Value display (read-only, click right panel to change) */}
                  {!opMeta?.noValue && (
                    <div className={`text-xs rounded-lg px-3 py-1.5 border min-h-[28px] flex items-center
                      ${cond.value
                        ? "border-violet-300 bg-white text-violet-700 font-medium"
                        : "border-dashed border-gray-300 text-gray-400 italic"}`}
                    >
                      {cond.value || (isActive ? "← click a value on the right" : "no value set")}
                    </div>
                  )}
                </div>
              );
            })}

            <button
              onClick={addCond}
              className="text-xs text-violet-600 hover:text-violet-800 self-start flex items-center gap-1 py-1"
            >
              + Add condition
            </button>
          </div>

          {/* Right: values panel */}
          <div style={{ width: 200, flexShrink: 0, display: "flex", flexDirection: "column", padding: "16px 12px", overflowY: "auto" }}>
            <p style={{ fontSize: 10, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
              Values — <span style={{ color: "#7c3aed", textTransform: "none", fontWeight: 500 }}>{activeCol}</span>
            </p>

            {showValuePanel ? (
              <>
                {/* Search */}
                <input
                  type="text"
                  placeholder="search values…"
                  value={valSearch}
                  onChange={(e) => setValSearch(e.target.value)}
                  style={{ width: "100%", fontSize: 11, border: "1px solid #e5e7eb", borderRadius: 8, padding: "5px 10px", marginBottom: 6, outline: "none", boxSizing: "border-box", color: "#111827" }}
                />

                {/* Value list */}
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {valuesLoading ? (
                    <p style={{ fontSize: 11, color: "#9ca3af", textAlign: "center", padding: "16px 0" }}>Loading…</p>
                  ) : filteredValues.length === 0 ? (
                    <p style={{ fontSize: 11, color: "#d1d5db", textAlign: "center", padding: "16px 0" }}>No values</p>
                  ) : (
                    filteredValues.map((val, vi) => {
                      const isSelected = conditions[activeCond]?.value === val;
                      return (
                        <button
                          key={`${val}-${vi}`}
                          onClick={() => updateCond(activeCond, { value: val })}
                          style={{
                            display: "block",
                            width: "100%",
                            textAlign: "left",
                            fontSize: 12,
                            padding: "8px 10px",
                            borderRadius: 6,
                            border: isSelected ? "1.5px solid #7c3aed" : "1px solid #d1d5db",
                            background: isSelected ? "#7c3aed" : "#ffffff",
                            color: isSelected ? "#ffffff" : "#1f2937",
                            fontWeight: isSelected ? 600 : 500,
                            cursor: "pointer",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            boxSizing: "border-box",
                            transition: "all 0.15s",
                          }}
                          title={val}
                        >
                          {val}
                        </button>
                      );
                    })
                  )}
                </div>

                <p style={{ fontSize: 10, color: "#d1d5db", textAlign: "center", marginTop: 6 }}>
                  {filteredValues.length} value{filteredValues.length !== 1 ? "s" : ""}
                </p>
              </>
            ) : (
              <p style={{ fontSize: 11, color: "#d1d5db", fontStyle: "italic", textAlign: "center", paddingTop: 32 }}>
                No value needed
              </p>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-100 flex flex-col gap-3">
          {/* Preview result */}
          {preview && (
            <div className="bg-violet-50 border border-violet-200 rounded-xl px-4 py-2.5 text-sm text-violet-800 flex items-baseline gap-2">
              <span className="font-bold text-xl">{preview.selected.toLocaleString()}</span>
              <span className="text-violet-500">of {preview.total.toLocaleString()} cases match</span>
              <span className="text-violet-400 text-xs ml-auto">
                {((preview.selected / preview.total) * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {error && <p className="text-xs text-red-500">{error}</p>}

          <div className="flex items-center gap-2">
            <button onClick={handlePreview} disabled={busy}
              className="px-4 py-2 text-sm border border-violet-300 text-violet-700 rounded-xl hover:bg-violet-50 transition-colors disabled:opacity-50">
              Preview
            </button>
            <button onClick={handleApply} disabled={busy}
              className="flex-1 px-4 py-2 text-sm bg-violet-600 text-white rounded-xl hover:bg-violet-700 transition-colors disabled:opacity-50 font-medium">
              {busy ? "Applying…" : "Apply"}
            </button>
            <button onClick={handleClear} disabled={busy}
              className="px-4 py-2 text-sm border border-gray-200 text-gray-500 rounded-xl hover:bg-gray-50 transition-colors disabled:opacity-50">
              Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────
