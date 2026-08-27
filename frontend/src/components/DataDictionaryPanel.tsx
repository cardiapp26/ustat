import { useState, useEffect, Fragment } from "react";
import { useStore, type ColMeta, type Session } from "../store";
import { saveMetadata, getUniqueValues } from "../api";
import DictionaryValueLabelImport, { type ValueLabelImportResult } from "./DictionaryValueLabelImport";

const ROLES = ["", "outcome", "predictor", "covariate", "id", "time", "event"] as const;
const ROLE_COLORS: Record<string, string> = {
  outcome: "bg-red-50 text-red-700",
  predictor: "bg-blue-50 text-blue-700",
  covariate: "bg-purple-50 text-purple-700",
  id: "bg-gray-100 text-gray-500",
  time: "bg-amber-50 text-amber-700",
  event: "bg-green-50 text-green-700",
};

// Auto-detect roles from column names
function autoDetectRole(name: string): string {
  const n = name.toLowerCase();
  if (/^(id|patient|subject|case)/.test(n)) return "id";
  if (/^(time|duration|fupt|follow|days|months|years)/.test(n)) return "time";
  if (/^(event|death|exitus|status|censor|outcome|endpoint)/.test(n)) return "event";
  return "";
}

function formatMissing(m: Partial<ColMeta>): string {
  const ranges = (m.missing_ranges ?? [])
    .filter((r) => r.lo !== null && r.lo !== undefined)
    .map((r) => (r.lo === r.hi ? String(r.lo) : `${String(r.lo)}-${String(r.hi)}`))
    .filter(Boolean);
  const values = (m.missing_user_values ?? [])
    .filter((v) => v !== null && v !== undefined)
    .map((v) => String(v))
    .filter(Boolean);
  return [...ranges, ...values].join(", ");
}

export default function DataDictionaryPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <DataDictionaryPanelBody session={session} />;
}

function DataDictionaryPanelBody({ session }: { session: Session }) {
  const setSession = useStore((s) => s.setSession);

  const [meta, setMeta] = useState<Record<string, Partial<ColMeta>>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  // Inline value-labels editor: which column row is expanded + cached
  // unique-value lists per column. `null` while loading, [] when empty.
  const [openValueLabelsFor, setOpenValueLabelsFor] = useState<string | null>(null);
  const [uniqueValuesByCol, setUniqueValuesByCol] = useState<Record<string, string[] | null>>({});

  // Initialize meta from session columns
  useEffect(() => {
    const m: Record<string, Partial<ColMeta>> = {};
    for (const col of session.columns) {
      m[col.name] = {
        label: col.label ?? "",
        description: col.description ?? "",
        units: col.units ?? "",
        role: col.role ?? (autoDetectRole(col.name) as ColMeta["role"]),
        value_labels: col.value_labels ?? {},
        missing_ranges: col.missing_ranges ?? [],
        missing_user_values: col.missing_user_values ?? [],
        measure: col.measure ?? "",
      };
    }
    setMeta(m);
    // Seed metadata only when the dataset itself changes — not on every column
    // edit (which would clobber the user's in-progress dictionary edits).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.session_id]);

  const update = (colName: string, field: keyof ColMeta, value: string) => {
    setMeta((prev) => ({
      ...prev,
      [colName]: { ...prev[colName], [field]: value } as Partial<ColMeta>,
    }));
    setSaved(false);
  };

  const toggleValueLabels = async (colName: string) => {
    if (openValueLabelsFor === colName) {
      setOpenValueLabelsFor(null);
      return;
    }
    setOpenValueLabelsFor(colName);
    if (uniqueValuesByCol[colName] === undefined) {
      setUniqueValuesByCol((p) => ({ ...p, [colName]: null }));
      try {
        const res = await getUniqueValues(session.session_id, colName);
        const values: string[] = (res.data?.values ?? []).map((v: unknown) => String(v));
        setUniqueValuesByCol((p) => ({ ...p, [colName]: values }));
      } catch {
        setUniqueValuesByCol((p) => ({ ...p, [colName]: [] }));
      }
    }
  };

  const updateValueLabel = (colName: string, rawValue: string, label: string) => {
    setMeta((prev) => {
      const existing = (prev[colName]?.value_labels ?? {}) as Record<string, string>;
      const next = { ...existing };
      if (label.trim() === "") delete next[rawValue];
      else next[rawValue] = label;
      return { ...prev, [colName]: { ...prev[colName], value_labels: next } };
    });
    setSaved(false);
  };

  const clearValueLabels = (colName: string) => {
    setMeta((prev) => ({ ...prev, [colName]: { ...prev[colName], value_labels: {} } }));
    setSaved(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveMetadata(session.session_id, meta);
      // Update local session columns with the metadata
      const updatedCols = session.columns.map((c) => ({
        ...c,
        ...(meta[c.name] ?? {}),
      }));
      setSession({ ...session, columns: updatedCols });
      setSaved(true);
    } catch {
      /* ignore */
    } finally {
      setSaving(false);
    }
  };

  const handleAutoDetect = () => {
    const updated = { ...meta };
    for (const col of session.columns) {
      const detected = autoDetectRole(col.name);
      if (detected && !updated[col.name]?.role) {
        updated[col.name] = { ...updated[col.name], role: detected as ColMeta["role"] };
      }
    }
    setMeta(updated);
    setSaved(false);
  };

  const handleValueLabelImport = async ({ labelsByTarget }: ValueLabelImportResult) => {
    const nextMeta = { ...meta };
    for (const [target, importedLabels] of Object.entries(labelsByTarget)) {
      const existingLabels = (nextMeta[target]?.value_labels ?? {}) as Record<string, string>;
      nextMeta[target] = {
        ...nextMeta[target],
        value_labels: { ...existingLabels, ...importedLabels },
      };
    }
    await saveMetadata(session.session_id, nextMeta);
    setMeta(nextMeta);
    setSession({
      ...session,
      columns: session.columns.map((column) => ({
        ...column,
        ...(nextMeta[column.name] ?? {}),
      })),
    });
    setSaved(true);
  };

  const exportCSV = () => {
    const rows = [["Name", "Label", "Type", "SPSS Measure", "Units", "Role", "Missing", "Description"]];
    for (const col of session.columns) {
      const m = meta[col.name] ?? {};
      rows.push([col.name, m.label ?? "", col.kind, m.measure ?? "", m.units ?? "", m.role ?? "", formatMissing(m), m.description ?? ""]);
    }
    const csv = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\r\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "data_dictionary.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-gray-900">Data Dictionary</h2>
          <p className="text-xs text-gray-400">{session.columns.length} variables \u00B7 {session.rows} observations</p>
        </div>
        <div className="flex gap-2">
          <button onClick={handleAutoDetect} className="text-xs px-3 py-1.5 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50">
            Auto-detect roles
          </button>
          <DictionaryValueLabelImport columns={session.columns} onApply={handleValueLabelImport} />
          <button onClick={exportCSV} className="text-xs px-3 py-1.5 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50">
            Export CSV
          </button>
          <button onClick={handleSave} disabled={saving} className="btn-primary text-xs px-4 py-1.5">
            {saving ? "Saving\u2026" : saved ? "\u2713 Saved" : "Save Metadata"}
          </button>
        </div>
      </div>

      {/* Dictionary table */}
      <div className="overflow-auto rounded-xl border border-gray-200">
        <table className="w-full text-xs">
          <thead className="bg-gray-50 sticky top-0 z-10">
            <tr>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-40">Variable</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-12">Type</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-20">Measure</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-48">Label</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-20">Units</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-28">Role</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-32">Value labels</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium w-28">Missing</th>
              <th className="px-3 py-2 text-left text-gray-500 font-medium">Description</th>
            </tr>
          </thead>
          <tbody>
            {session.columns.map((col) => {
              const m = meta[col.name] ?? {};
              const vLabels = (m.value_labels ?? {}) as Record<string, string>;
              const vLabelCount = Object.values(vLabels).filter((v) => v && String(v).trim() !== "").length;
              const isOpen = openValueLabelsFor === col.name;
              const unique = uniqueValuesByCol[col.name];
              const missing = formatMissing(m);
              return (
                <Fragment key={col.name}>
                <tr className="border-t border-gray-100 hover:bg-gray-50">
                  <td className="px-3 py-1.5 font-mono font-medium text-gray-800">{col.name}</td>
                  <td className="px-3 py-1.5">
                    <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                      col.kind === "numeric" ? "bg-blue-100 text-blue-700" :
                      col.kind === "categorical" ? "bg-orange-100 text-orange-700" :
                      col.kind === "ordinal" ? "bg-teal-100 text-teal-700" :
                      col.kind === "date" ? "bg-purple-100 text-purple-700" :
                      "bg-gray-100 text-gray-500"
                    }`}>{col.kind === "numeric" ? "num" : col.kind === "categorical" ? "cat" : col.kind === "ordinal" ? "ord" : col.kind === "date" ? "date" : "txt"}</span>
                  </td>
                  <td className="px-3 py-1.5 text-gray-500">{m.measure || "-"}</td>
                  <td className="px-1 py-1">
                    <input className="w-full bg-transparent border border-transparent hover:border-gray-200 focus:border-indigo-400 rounded px-2 py-0.5 text-xs focus:outline-none"
                      value={m.label ?? ""} placeholder="Variable label\u2026"
                      onChange={(e) => update(col.name, "label", e.target.value)} />
                  </td>
                  <td className="px-1 py-1">
                    <input className="w-full bg-transparent border border-transparent hover:border-gray-200 focus:border-indigo-400 rounded px-2 py-0.5 text-xs focus:outline-none"
                      value={m.units ?? ""} placeholder="e.g. mmHg"
                      onChange={(e) => update(col.name, "units", e.target.value)} />
                  </td>
                  <td className="px-1 py-1">
                    <select className={`w-full text-xs rounded px-1 py-0.5 border border-transparent hover:border-gray-200 focus:border-indigo-400 focus:outline-none ${ROLE_COLORS[m.role ?? ""] ?? ""}`}
                      value={m.role ?? ""} onChange={(e) => update(col.name, "role", e.target.value)}>
                      {ROLES.map((r) => <option key={r} value={r}>{r || "\u2014"}</option>)}
                    </select>
                  </td>
                  <td className="px-1 py-1">
                    <button
                      onClick={() => toggleValueLabels(col.name)}
                      className={`w-full text-[10px] px-2 py-1 rounded border transition-colors ${
                        isOpen
                          ? "bg-indigo-600 text-white border-indigo-600"
                          : vLabelCount > 0
                            ? "bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100"
                            : "bg-white text-gray-500 border-gray-200 hover:bg-gray-50"
                      }`}
                      title={col.kind === "numeric" ? "Edit value labels \u2014 useful for coded variables like Sex (0/1)" : "Edit value labels"}
                    >
                      {isOpen ? "Close \u25b2" : vLabelCount > 0 ? `${vLabelCount} label${vLabelCount > 1 ? "s" : ""}` : "Edit \u25be"}
                    </button>
                  </td>
                  <td className="px-3 py-1.5 font-mono text-[10px] text-gray-500 max-w-28 truncate" title={missing || ""}>{missing || "-"}</td>
                  <td className="px-1 py-1">
                    <input className="w-full bg-transparent border border-transparent hover:border-gray-200 focus:border-indigo-400 rounded px-2 py-0.5 text-xs focus:outline-none"
                      value={m.description ?? ""} placeholder="Description\u2026"
                      onChange={(e) => update(col.name, "description", e.target.value)} />
                  </td>
                </tr>
                {isOpen && (
                  <tr className="bg-indigo-50/40 border-t border-indigo-100">
                    <td colSpan={9} className="px-4 py-3">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-xs text-gray-700">
                          <span className="font-semibold">Value labels for <span className="font-mono">{col.name}</span></span>
                          <span className="ml-2 text-[10px] text-gray-500">e.g. Sex: 0 = Female, 1 = Male</span>
                        </p>
                        {vLabelCount > 0 && (
                          <button
                            onClick={() => clearValueLabels(col.name)}
                            className="text-[10px] px-2 py-0.5 rounded border border-orange-200 text-orange-600 hover:bg-orange-50"
                          >Clear all</button>
                        )}
                      </div>
                      {unique === null ? (
                        <p className="text-[11px] text-gray-400">Loading unique values\u2026</p>
                      ) : !unique || unique.length === 0 ? (
                        <p className="text-[11px] text-gray-400">No values found.</p>
                      ) : unique.length > 50 ? (
                        <p className="text-[11px] text-amber-600">
                          {unique.length} unique values \u2014 too many to label individually. Value labels are intended for coded categoricals (Sex, Group, etc.).
                        </p>
                      ) : (
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-64 overflow-y-auto">
                          {unique.map((v) => (
                            <label key={v} className="flex items-center gap-2 bg-white border border-gray-200 rounded px-2 py-1.5">
                              <span className="font-mono text-xs text-gray-500 min-w-[3rem] text-right">{v}</span>
                              <span className="text-gray-300">\u2192</span>
                              <input
                                value={vLabels[v] ?? ""}
                                placeholder="label\u2026"
                                onChange={(e) => updateValueLabel(col.name, v, e.target.value)}
                                className="flex-1 text-xs bg-transparent border border-gray-200 rounded px-1.5 py-0.5 focus:outline-none focus:border-indigo-400"
                              />
                            </label>
                          ))}
                        </div>
                      )}
                    </td>
                  </tr>
                )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Role legend */}
      <div className="flex gap-3 text-[10px] text-gray-400">
        {Object.entries(ROLE_COLORS).map(([role, cls]) => (
          <span key={role} className={`px-1.5 py-0.5 rounded ${cls}`}>{role}</span>
        ))}
      </div>
    </div>
  );
}
