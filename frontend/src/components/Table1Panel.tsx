import React, { useState, useEffect, useRef } from "react";
import { useStore, type Session } from "../store";
import api from "../api";
import ResultExporter from "./ResultExporter";
import { fmtP, warningText } from "../lib/format";

// ── Stat definitions ──────────────────────────────────────────────────────────

interface StatDef {
  id: string;
  label: string;
  group: "tendency" | "dispersion" | "percentile" | "counts";
}

const STAT_DEFS: StatDef[] = [
  { id: "auto",       label: "Auto (normality-based)",  group: "tendency" },
  { id: "mean_sd",    label: "Mean ± SD",               group: "tendency" },
  { id: "median_iqr", label: "Median [IQR]",            group: "tendency" },
  { id: "se",         label: "SE of Mean",              group: "dispersion" },
  { id: "ci95",       label: "95% CI",                  group: "dispersion" },
  { id: "variance",   label: "Variance",                group: "dispersion" },
  { id: "min_max",    label: "Min – Max",               group: "dispersion" },
  { id: "p10",        label: "10th Percentile",         group: "percentile" },
  { id: "p25",        label: "25th Percentile",         group: "percentile" },
  { id: "p75",        label: "75th Percentile",         group: "percentile" },
  { id: "p90",        label: "90th Percentile",         group: "percentile" },
  { id: "p95",        label: "95th Percentile",         group: "percentile" },
  { id: "n",          label: "N (non-missing)",         group: "counts" },
  { id: "missing",    label: "Missing count",           group: "counts" },
];

const STAT_GROUPS = [
  { id: "tendency",   label: "Central Tendency" },
  { id: "dispersion", label: "Dispersion" },
  { id: "percentile", label: "Percentiles" },
  { id: "counts",     label: "Counts" },
];

// ── Types ─────────────────────────────────────────────────────────────────────

interface StatRow { label: string; overall: string; group_stats: Record<string, string> }

interface PerGroupNormality {
  p: number | null;
  test: string;
  normal: boolean;
  n: number;
}

interface T1Row {
  variable: string;
  type: "numeric" | "categorical";
  overall_n: number;
  stat_rows?: StatRow[];
  stat_label?: string;
  overall?: string;
  p_value: string | null;
  test: string | null;
  significant?: boolean;
  normal?: boolean;
  normality_test?: string;
  normality_p?: number;
  normality_mode?: "overall" | "within_group";
  per_group_normality?: Record<string, PerGroupNormality>;
  smd?: number | null;
  group_stats: Record<string, string>;
  sub_rows?: { category: string; overall: string; group_stats: Record<string, string> }[];
}

interface T1Result {
  group_column: string | null;
  group_labels: string[];
  group_ns: Record<string, number>;
  total_n: number;
  rows: T1Row[];
  // Mixed shapes on purpose: the category cleaner returns structured dicts,
  // the test rules return prose. warningText() renders either.
  warnings?: unknown[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function pColor(p: string | null) {
  if (!p || p === "N/A") return "text-gray-400";
  if (p === "<0.001") return "text-red-600 font-bold";
  const v = parseFloat(p);
  if (v < 0.05) return "text-amber-600 font-semibold";
  return "text-gray-400";
}

function pStars(p: string | null) {
  if (!p || p === "N/A") return "";
  if (p === "<0.001") return "***";
  const v = parseFloat(p);
  if (v < 0.001) return "***";
  if (v < 0.01) return "**";
  if (v < 0.05) return "*";
  return "ns";
}

// ── Stats selector panel ──────────────────────────────────────────────────────

function StatsSelector({
  selected,
  onChange,
}: {
  selected: Set<string>;
  onChange: (s: Set<string>) => void;
}) {
  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) {
      if (next.size === 1) return;
      next.delete(id);
    } else {
      next.add(id);
    }
    onChange(next);
  };

  return (
    <div className="border-b border-gray-200 pb-1">
      <div className="px-3 pt-2 pb-1 flex items-center justify-between">
        <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
          Statistics shown
        </h3>
        <span className="text-[9px] text-gray-400">numeric vars</span>
      </div>
      {STAT_GROUPS.map((grp) => {
        const defs = STAT_DEFS.filter((d) => d.group === grp.id);
        return (
          <div key={grp.id} className="mb-1">
            <p className="px-3 text-[9px] font-semibold text-gray-400 uppercase tracking-wider mt-1.5">
              {grp.label}
            </p>
            {defs.map((d) => (
              <label
                key={d.id}
                className={`flex items-center gap-2 px-3 py-0.5 cursor-pointer transition-colors
                  ${selected.has(d.id) ? "text-indigo-600" : "text-gray-400 hover:text-gray-600"}`}
              >
                <input
                  type="checkbox"
                  className="accent-indigo-500 flex-shrink-0"
                  checked={selected.has(d.id)}
                  onChange={() => toggle(d.id)}
                />
                <span className="text-xs">{d.label}</span>
              </label>
            ))}
          </div>
        );
      })}
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function Table1Panel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <Table1PanelBody session={session} />;
}

function Table1PanelBody({ session }: { session: Session }) {
  const result = useStore((s) => s.table1Result) as T1Result | null;
  const setResult = useStore((s) => s.setTable1Result);
  const clearTable1 = useStore((s) => s.clearTable1);
  // Per-column decimal overrides set in the Data tab. Backend also has
  // its own session-persisted map (services/store.get_decimals); passing
  // the client snapshot here lets unsaved tweaks preview immediately.
  const columnDecimals = useStore((s) => s.columnDecimals);
  // Persisted form snapshot across tab switches — the per-panel cache
  // already used elsewhere in the store for the same purpose. Holds the
  // user's variable selection, group column, stat picks, kind overrides,
  // and within-group normality flag so leaving the tab and coming back
  // does not nuke ten minutes of clicking.
  const cachedForm = useStore((s) => s.panelCache?.table1Form) as
    | {
        groupCol?: string;
        selected?: string[];
        kindOverrides?: Record<string, "numeric" | "categorical">;
        selectedStats?: string[];
        withinGroupNormality?: boolean;
      }
    | undefined;
  const setPanelCache = useStore((s) => s.setPanelCache);

  // Columns offered as Table 1 variables / group — drop those flagged
  // "exclude from analysis" (e.g. NAME, row-id) in the data tab.
  const pickableCols = session.columns.filter((c) => !c.analysis_excluded);
  const allCols = pickableCols.map((c) => c.name);

  const [groupCol, setGroupCol] = useState<string>(cachedForm?.groupCol ?? "");
  // Hydrate the selected-variables set from cache when available; intersect
  // with the current dataset's columns so a stale cache referencing a
  // dropped column does not crash anything downstream.
  const [selected, setSelected] = useState<Set<string>>(() => {
    if (cachedForm?.selected) {
      const allowed = new Set(allCols);
      return new Set(cachedForm.selected.filter((c) => allowed.has(c)));
    }
    return new Set(allCols);
  });
  const [kindOverrides, setKindOverrides] = useState<Record<string, "numeric" | "categorical">>(
    cachedForm?.kindOverrides ?? {}
  );
  const [selectedStats, setSelectedStats] = useState<Set<string>>(
    new Set(cachedForm?.selectedStats ?? ["auto"])
  );
  const [showStats, setShowStats] = useState(false);
  const [showSMD, setShowSMD] = useState(false);
  const [withinGroupNormality, setWithinGroupNormality] = useState<boolean>(
    cachedForm?.withinGroupNormality ?? false
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  // Mirror form state back into the panel cache on every change so the
  // selection survives a tab switch. Sets are serialised to arrays so
  // the cache stays JSON-safe.
  useEffect(() => {
    setPanelCache("table1Form", {
      groupCol,
      selected: Array.from(selected),
      kindOverrides,
      selectedStats: Array.from(selectedStats),
      withinGroupNormality,
    });
  }, [groupCol, selected, kindOverrides, selectedStats, withinGroupNormality, setPanelCache]);

  // ── Table editing state ──────────────────────────────────────────────────
  const [editRowIdx, setEditRowIdx] = useState<number | null>(null);
  const [editRowVal, setEditRowVal] = useState("");
  const editRowRef = useRef<HTMLInputElement>(null);
  const [editGroupIdx, setEditGroupIdx] = useState<number | null>(null);
  const [editGroupVal, setEditGroupVal] = useState("");
  const editGroupRef = useRef<HTMLInputElement>(null);

  useEffect(() => { editRowRef.current?.select(); }, [editRowIdx]);
  useEffect(() => { editGroupRef.current?.select(); }, [editGroupIdx]);

  const hasGroups = !!(result && result.group_labels.length > 0);

  const commitRowRename = () => {
    if (editRowIdx == null || !result) return;
    const trimmed = editRowVal.trim();
    if (trimmed && trimmed !== result.rows[editRowIdx].variable) {
      const rows = result.rows.map((r, i) =>
        i === editRowIdx ? { ...r, variable: trimmed } : r
      );
      setResult({ ...result, rows });
    }
    setEditRowIdx(null);
  };

  const commitGroupRename = () => {
    if (editGroupIdx == null || !result) return;
    const trimmed = editGroupVal.trim();
    const oldLabel = result.group_labels[editGroupIdx];
    if (trimmed && trimmed !== oldLabel) {
      const group_labels = result.group_labels.map((g, i) =>
        i === editGroupIdx ? trimmed : g
      );
      const group_ns = { ...result.group_ns };
      if (oldLabel in group_ns) {
        group_ns[trimmed] = group_ns[oldLabel];
        delete group_ns[oldLabel];
      }
      // Remap group_stats keys in every row
      const rows = result.rows.map((row) => {
        const group_stats = { ...row.group_stats };
        if (oldLabel in group_stats) {
          group_stats[trimmed] = group_stats[oldLabel];
          delete group_stats[oldLabel];
        }
        const stat_rows = row.stat_rows?.map((sr) => {
          const gs = { ...sr.group_stats };
          if (oldLabel in gs) { gs[trimmed] = gs[oldLabel]; delete gs[oldLabel]; }
          return { ...sr, group_stats: gs };
        });
        const sub_rows = row.sub_rows?.map((sr) => {
          const gs = { ...sr.group_stats };
          if (oldLabel in gs) { gs[trimmed] = gs[oldLabel]; delete gs[oldLabel]; }
          return { ...sr, group_stats: gs };
        });
        return { ...row, group_stats, stat_rows, sub_rows };
      });
      setResult({ ...result, group_labels, group_ns, rows });
    }
    setEditGroupIdx(null);
  };

  const moveRow = (from: number, dir: -1 | 1) => {
    if (!result) return;
    const to = from + dir;
    if (to < 0 || to >= result.rows.length) return;
    const rows = [...result.rows];
    [rows[from], rows[to]] = [rows[to], rows[from]];
    setResult({ ...result, rows });
  };

  useEffect(() => {
    // New dataset → reset the form to defaults (don't carry a stale
    // selection across sessions). The cache useEffect above will then
    // overwrite the stored snapshot with the fresh defaults.
    setSelected(new Set(allCols.filter((c) => c !== groupCol)));
    setKindOverrides({});
    setSelectedStats(new Set(["auto"]));
    setWithinGroupNormality(false);
    clearTable1();
    setError(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.session_id]);

  const toggleKind = (col: string) => {
    setKindOverrides((prev) => {
      const base = session.columns.find((c) => c.name === col)?.kind ?? "numeric";
      const current = prev[col] ?? base;
      const next = current === "numeric" ? "categorical" : "numeric";
      if (next === base) {
        const rest = { ...prev };
        delete rest[col];
        return rest;
      }
      return { ...prev, [col]: next };
    });
    setResult(null);
  };

  const toggle = (col: string) =>
    setSelected((prev) => {
      const s = new Set(prev);
      if (s.has(col)) s.delete(col); else s.add(col);
      return s;
    });

  const selectAll = () => setSelected(new Set(allCols.filter((c) => c !== groupCol)));
  const selectNone = () => setSelected(new Set());

  const handleGroupChange = (col: string) => {
    setGroupCol(col);
    setSelected((prev) => { const s = new Set(prev); s.delete(col); return s; });
    setResult(null);
  };

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    const variable_kinds: Record<string, string> = {};
    Array.from(selected).forEach((col) => {
      const kind = kindOverrides[col] ?? session.columns.find((c) => c.name === col)?.kind;
      if (kind) variable_kinds[col] = kind;
    });
    try {
      const res = await api.post("/api/stats/table1", {
        session_id: session.session_id,
        group_column: groupCol || null,
        variables: Array.from(selected),
        variable_kinds,
        selected_stats: Array.from(selectedStats),
        normality_mode: (groupCol && withinGroupNormality) ? "within_group" : "overall",
        // Pass the client-side decimals map so request-time overrides
        // win over the session-persisted snapshot.
        column_decimals: columnDecimals,
      });

      const rawResult = res.data as T1Result;
      if (rawResult) {
        // 1. Map group labels
        if (rawResult.group_column) {
          const groupColMeta = session.columns.find((c) => c.name === rawResult.group_column);
          const groupLabels = groupColMeta?.value_labels ?? {};
          
          rawResult.group_labels = rawResult.group_labels.map((g) => {
            const mapped = groupLabels[String(g)];
            return mapped !== undefined ? mapped : g;
          });
          
          // Map group_ns keys
          const group_ns: Record<string, number> = {};
          Object.entries(rawResult.group_ns).forEach(([g, n]) => {
            const mapped = groupLabels[String(g)] ?? g;
            group_ns[mapped] = n;
          });
          rawResult.group_ns = group_ns;
        }
        
        // 2. Map row variables categories & group stats
        rawResult.rows = rawResult.rows.map((row) => {
          const colMeta = session.columns.find((c) => c.name === row.variable);
          const vLabels = colMeta?.value_labels ?? {};
          
          // Map group stats keys for numeric and categorical
          if (rawResult.group_column) {
            const groupColMeta = session.columns.find((c) => c.name === rawResult.group_column);
            const groupLabels = groupColMeta?.value_labels ?? {};
            
            const group_stats: Record<string, string> = {};
            Object.entries(row.group_stats).forEach(([g, val]) => {
              const mapped = groupLabels[String(g)] ?? g;
              group_stats[mapped] = val;
            });
            row.group_stats = group_stats;
            
            if (row.stat_rows) {
              row.stat_rows = row.stat_rows.map((sr) => {
                const gs: Record<string, string> = {};
                Object.entries(sr.group_stats).forEach(([g, val]) => {
                  const mapped = groupLabels[String(g)] ?? g;
                  gs[mapped] = val;
                });
                return { ...sr, group_stats: gs };
              });
            }
          }
          
          // Map sub-rows categories
          if (row.type === "categorical" && row.sub_rows) {
            row.sub_rows = row.sub_rows.map((sr) => {
              const mappedCat = vLabels[String(sr.category)] ?? sr.category;
              
              const gs: Record<string, string> = {};
              if (rawResult.group_column) {
                const groupColMeta = session.columns.find((c) => c.name === rawResult.group_column);
                const groupLabels = groupColMeta?.value_labels ?? {};
                
                Object.entries(sr.group_stats).forEach(([g, val]) => {
                  const mapped = groupLabels[String(g)] ?? g;
                  gs[mapped] = val;
                });
              }
              
              return {
                ...sr,
                category: mappedCat,
                group_stats: gs,
              };
            });
          }
          
          return row;
        });
      }

      setResult(rawResult);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      const msg = e instanceof Error ? e.message : String(e);
      setError(detail ?? msg ?? "Error running Table 1");
    } finally { setLoading(false); }
  };

  const buildExportData = () => {
    if (!result) return { headers: [] as string[], rows: [] as string[][] };
    const gl = result.group_labels;
    const headers = ["Variable", "Statistic",
      `Overall (n=${result.total_n})`,
      ...gl.map((g) => `${result.group_column ? result.group_column + "=" : ""}${g} (n=${result.group_ns[g] ?? ""})`),
      "p-value", "Test",
      ...(showSMD ? ["SMD"] : []),
      "Normality test"];
    const rows: string[][] = [];
    result.rows.forEach((row) => {
      if (row.type === "numeric") {
        (row.stat_rows ?? []).forEach((sr, i: number) => {
          rows.push([
            i === 0 ? row.variable : "",
            sr.label, sr.overall,
            ...gl.map((g: string) => sr.group_stats[g] ?? ""),
            i === 0 ? (row.p_value ?? "") : "",
            i === 0 ? (row.test ?? "") : "",
            ...(showSMD ? [i === 0 && row.smd != null ? row.smd.toFixed(3) : ""] : []),
            i === 0 ? `${row.normality_test} (p=${fmtP(row.normality_p)})` : "",
          ]);
        });
      } else {
        rows.push([row.variable, "n (%)", `n=${row.overall_n}`,
          ...gl.map(() => ""), row.p_value ?? "", row.test ?? "",
          ...(showSMD ? [row.smd != null ? row.smd.toFixed(3) : ""] : []), ""]);
        (row.sub_rows ?? []).forEach((sr) => {
          rows.push([`  ${sr.category}`, "", sr.overall,
            ...gl.map((g: string) => sr.group_stats[g] ?? ""), "", "",
            ...(showSMD ? [""] : []), ""]);
        });
      }
    });
    return { headers, rows };
  };

  const { headers: exportHeaders, rows: exportRows } = buildExportData();

  const filteredCols = allCols.filter(
    (c) => c !== groupCol && c.toLowerCase().includes(search.toLowerCase())
  );
  const statsLabel = selectedStats.has("auto")
    ? "Auto"
    : `${selectedStats.size} stat${selectedStats.size > 1 ? "s" : ""}`;

  return (
    <div className="flex gap-0 h-full" style={{ minHeight: 0 }}>

      {/* ── Left sidebar ── */}
      <div className="w-56 flex-shrink-0 flex flex-col border-r border-gray-200 bg-white overflow-hidden">

        {/* Group column */}
        <div className="p-3 border-b border-gray-200 space-y-1.5">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Group by</h3>
          <select
            className="select w-full text-xs"
            value={groupCol}
            onChange={(e) => handleGroupChange(e.target.value)}
          >
            <option value="">— Overall only —</option>
            {pickableCols.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} [{c.kind === "numeric" ? "N" : "C"}]
              </option>
            ))}
          </select>
          {groupCol && (
            <p className="text-[10px] text-gray-400 leading-tight">
              Separate columns per group
            </p>
          )}
        </div>

        {/* Statistics selector (collapsible) */}
        <div className="border-b border-gray-200 flex-shrink-0">
          <button
            className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-50 transition-colors"
            onClick={() => setShowStats((v) => !v)}
          >
            <div className="flex items-center gap-1.5">
              <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                Statistics
              </h3>
              <span className="text-[9px] text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded-full">
                {statsLabel}
              </span>
            </div>
            <span className="text-gray-400 text-xs">{showStats ? "▲" : "▼"}</span>
          </button>
          {showStats && (
            <div className="overflow-y-auto max-h-64">
              <StatsSelector selected={selectedStats} onChange={setSelectedStats} />
            </div>
          )}
        </div>

        {/* Group-dependent options — SMD + within-group normality */}
        {groupCol && (
          <div className="px-3 py-2 border-b border-gray-200 flex-shrink-0 space-y-1.5">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                className="accent-indigo-500"
                checked={showSMD}
                onChange={(e) => setShowSMD(e.target.checked)}
              />
              <span className="text-xs text-gray-600 font-medium">Show SMD</span>
              <span className="text-[9px] text-gray-400">(Standardized Mean Difference)</span>
            </label>
            <label
              className="flex items-start gap-2 cursor-pointer"
              title="Run Shapiro-Wilk / Lilliefors on each group separately. Parametric test (t-test / ANOVA) is used only when every group passes normality (p > 0.05). Matches the actual assumption of parametric tests and is more conservative than the overall normality check."
            >
              <input
                type="checkbox"
                className="accent-indigo-500 mt-0.5"
                checked={withinGroupNormality}
                onChange={(e) => setWithinGroupNormality(e.target.checked)}
              />
              <div className="leading-tight">
                <span className="text-xs text-gray-600 font-medium">Test normality within each group</span>
                <span className="block text-[9px] text-gray-400">Stricter parametric criterion · every group must be normal</span>
              </div>
            </label>
          </div>
        )}

        {/* Variable selector */}
        <div className="px-3 py-2 border-b border-gray-200 flex items-center justify-between flex-shrink-0">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              className="accent-indigo-500 flex-shrink-0"
              ref={(el) => {
                if (!el) return;
                const eligibleCount = allCols.filter((c) => c !== groupCol).length;
                el.indeterminate = selected.size > 0 && selected.size < eligibleCount;
              }}
              checked={(() => {
                const eligibleCount = allCols.filter((c) => c !== groupCol).length;
                return eligibleCount > 0 && selected.size >= eligibleCount;
              })()}
              onChange={(e) => (e.target.checked ? selectAll() : selectNone())}
              title="Toggle all variables"
            />
            <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
              Variables ({selected.size})
            </h3>
          </label>
          <div className="flex gap-2">
            <button className="text-[10px] text-indigo-600 hover:text-indigo-700" onClick={selectAll}>All</button>
            <button className="text-[10px] text-indigo-600 hover:text-indigo-700" onClick={selectNone}>None</button>
          </div>
        </div>
        <div className="px-2 py-1.5 border-b border-gray-200 flex-shrink-0">
          <input className="select w-full text-xs" placeholder="Search variables…"
            value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>

        <div className="overflow-y-auto flex-1">
          {filteredCols.map((col) => {
            const baseKind = session.columns.find((c) => c.name === col)?.kind ?? "numeric";
            const effectiveKind = kindOverrides[col] ?? baseKind;
            const isOverridden = col in kindOverrides;
            const isChecked = selected.has(col);
            return (
              <div key={col}
                className={`flex items-center gap-1.5 px-2 py-1.5 border-b border-gray-100 transition-colors
                  ${isChecked ? "bg-indigo-50" : "hover:bg-gray-50"}`}>
                <input type="checkbox" className="accent-indigo-500 flex-shrink-0"
                  checked={isChecked} onChange={() => toggle(col)} />
                <button
                  onClick={(e) => { e.stopPropagation(); toggleKind(col); }}
                  title={`Currently: ${effectiveKind}. Click to switch to ${effectiveKind === "numeric" ? "categorical" : "numeric"}`}
                  className={`text-[9px] font-bold px-1.5 py-0.5 rounded flex-shrink-0 border transition-colors
                    ${effectiveKind === "numeric"
                      ? "bg-blue-100 text-blue-700 border-blue-300 hover:bg-blue-200"
                      : "bg-purple-100 text-purple-700 border-purple-300 hover:bg-purple-200"}
                    ${isOverridden ? "ring-1 ring-amber-400" : ""}`}>
                  {effectiveKind === "numeric" ? "N" : "C"}
                </button>
                <span className="text-xs text-gray-700 truncate flex-1">{col}</span>
                {isOverridden && (
                  <span className="text-[9px] text-amber-500 flex-shrink-0" title="Type overridden">★</span>
                )}
              </div>
            );
          })}
        </div>

        {/* Actions */}
        <div className="p-3 border-t border-gray-200 space-y-2 flex-shrink-0">
          <button className="btn-primary w-full text-sm py-2" onClick={run}
            disabled={loading || selected.size === 0}>
            {loading ? "Computing…" : "Generate Table"}
          </button>
          {result && (
            <ResultExporter
              title="Table1"
              headers={exportHeaders}
              rows={exportRows}
            />
          )}
          {error && (
            <p className="text-red-500 text-xs bg-red-50 rounded p-2 leading-relaxed">{error}</p>
          )}
        </div>
      </div>

      {/* ── Right: table ── */}
      <div className="flex-1 overflow-auto bg-gray-50">
        {!result && !loading && (
          <div className="h-full flex flex-col items-center justify-center text-gray-400 space-y-3">
            <div className="text-5xl opacity-20">📋</div>
            <p className="text-base font-medium text-gray-500">Table — Baseline Characteristics</p>
            <div className="text-xs text-gray-400 space-y-1 text-center leading-relaxed">
              <p>1. Pick a <span className="text-gray-500">Group by</span> column (e.g. outcome, treatment)</p>
              <p>2. Choose <span className="text-gray-500">Statistics</span> to display</p>
              <p>3. Select variables · Click <span className="text-gray-500">Generate Table</span></p>
            </div>
          </div>
        )}
        {loading && (
          <div className="h-full flex items-center justify-center text-gray-400 animate-pulse">
            Computing statistics…
          </div>
        )}

        {result && (
          <div className="p-4">
            {/* A blank p-value is the one cell that cannot explain itself.
                The reason it is blank — a constant column, a level only one
                arm has, values excluded as missing — arrives here, so show it
                above the table rather than only in the payload. */}
            {(result.warnings?.length ?? 0) > 0 && (
              <div className="mb-3 space-y-1">
                {(result.warnings ?? []).map((w, i) => (
                  <p key={i} className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-800 leading-relaxed">
                    ⚠ {warningText(w)}
                  </p>
                ))}
              </div>
            )}
            <div className="rounded-xl border border-gray-200 overflow-hidden shadow-sm">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-gray-50 border-b border-gray-200">
                    <th className="text-left px-4 py-3 text-gray-700 font-semibold w-44 border-r border-gray-200">
                      Variable
                    </th>
                    <th className="text-center px-3 py-3 text-gray-400 font-normal text-xs w-32 border-r border-gray-200">
                      Statistic
                    </th>
                    <th className="text-center px-4 py-3 text-gray-700 font-semibold border-r border-gray-200">
                      Overall
                      <br /><span className="text-xs font-normal text-gray-400"><i>n</i> = {result.total_n}</span>
                    </th>
                    {result.group_labels.map((g, gi) => (
                      <th key={g} className="text-center px-4 py-3 text-indigo-600 font-semibold border-r border-gray-200">
                        {result.group_column && <span className="text-gray-400 text-xs font-normal">{result.group_column} = </span>}
                        {editGroupIdx === gi ? (
                          <input
                            ref={editGroupRef}
                            value={editGroupVal}
                            onChange={(e) => setEditGroupVal(e.target.value)}
                            onBlur={commitGroupRename}
                            onKeyDown={(e) => { if (e.key === "Enter") commitGroupRename(); if (e.key === "Escape") setEditGroupIdx(null); }}
                            className="w-24 text-center text-sm font-semibold text-indigo-600 border border-indigo-300 rounded px-1 py-0.5 bg-white outline-none"
                          />
                        ) : (
                          <span
                            className="cursor-pointer hover:underline hover:decoration-indigo-300"
                            onDoubleClick={() => { setEditGroupIdx(gi); setEditGroupVal(g); }}
                            title="Double-click to rename"
                          >
                            {g}
                          </span>
                        )}
                        <br /><span className="text-xs font-normal text-gray-400"><i>n</i> = {result.group_ns[g] ?? ""}</span>
                      </th>
                    ))}
                    {hasGroups && (
                      <>
                        <th className="text-center px-3 py-3 text-gray-700 font-semibold w-24"><i>p</i>-value</th>
                        <th className="text-center px-3 py-3 text-gray-400 font-normal text-xs w-28">Test</th>
                      </>
                    )}
                    {hasGroups && showSMD && (
                      <th className="text-center px-3 py-3 text-gray-700 font-semibold w-20">SMD</th>
                    )}
                    <th className="text-center px-2 py-3 text-gray-400 font-normal text-[10px] w-20">
                      Normality
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, ri) =>
                    row.type === "numeric" ? (
                      <React.Fragment key={`num-${ri}`}>
                      {(row.stat_rows ?? []).map((sr, si) => (
                        <tr key={`${ri}-${si}`}
                          className={`border-t transition-colors hover:bg-gray-50
                            ${si === 0 ? "border-gray-200" : "border-gray-100"}
                            ${row.significant && si === 0 ? "bg-amber-50/30" : ""}`}>
                          <td className={`px-4 py-2 border-r border-gray-200
                            ${si === 0 ? "font-medium text-gray-900 group/var" : ""}`}>
                            {si === 0 ? (
                              editRowIdx === ri ? (
                                <input
                                  ref={editRowRef}
                                  value={editRowVal}
                                  onChange={(e) => setEditRowVal(e.target.value)}
                                  onBlur={commitRowRename}
                                  onKeyDown={(e) => { if (e.key === "Enter") commitRowRename(); if (e.key === "Escape") setEditRowIdx(null); }}
                                  className="w-full text-sm font-medium border border-indigo-300 rounded px-1 py-0.5 bg-white outline-none"
                                />
                              ) : (
                                <span className="flex items-center gap-1">
                                  <span className="opacity-0 group-hover/var:opacity-100 flex flex-col -my-1 text-gray-300">
                                    <button onClick={() => moveRow(ri, -1)} disabled={ri === 0}
                                      className="hover:text-indigo-500 disabled:opacity-20 leading-none text-[10px]">▲</button>
                                    <button onClick={() => moveRow(ri, 1)} disabled={ri === result.rows.length - 1}
                                      className="hover:text-indigo-500 disabled:opacity-20 leading-none text-[10px]">▼</button>
                                  </span>
                                  <span
                                    className="cursor-pointer hover:underline hover:decoration-indigo-300"
                                    onDoubleClick={() => { setEditRowIdx(ri); setEditRowVal(row.variable); }}
                                    title="Double-click to rename · arrows to reorder"
                                  >
                                    {row.variable}
                                  </span>
                                </span>
                              )
                            ) : null}
                          </td>
                          <td className="px-3 py-1.5 text-center text-xs text-gray-400 border-r border-gray-200">
                            {sr.label}
                          </td>
                          <td className="px-4 py-1.5 text-center text-gray-700 font-mono text-xs border-r border-gray-200">
                            {sr.overall}
                          </td>
                          {result.group_labels.map((g) => (
                            <td key={g} className="px-4 py-1.5 text-center text-gray-700 font-mono text-xs border-r border-gray-200">
                              {sr.group_stats[g] ?? "—"}
                            </td>
                          ))}
                          {hasGroups && (
                            <>
                              <td className={`px-3 py-1.5 text-center text-xs font-mono ${si === 0 ? pColor(row.p_value) : "text-transparent"}`}>
                                {si === 0 ? (
                                  <>
                                    {row.p_value ?? "—"}
                                    {row.p_value && <span className="ml-0.5 text-[10px] opacity-70">{pStars(row.p_value)}</span>}
                                  </>
                                ) : null}
                              </td>
                              <td className="px-3 py-1.5 text-center text-xs text-gray-400">
                                {si === 0 ? row.test : ""}
                              </td>
                            </>
                          )}
                          {hasGroups && showSMD && (
                            <td className="px-3 py-1.5 text-center text-xs font-mono">
                              {si === 0 && row.smd != null ? (
                                <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                                  row.smd > 0.2 ? "bg-amber-100 text-amber-700" :
                                  row.smd > 0.1 ? "bg-yellow-50 text-yellow-700" :
                                  "bg-green-50 text-green-700"
                                }`}>
                                  {row.smd.toFixed(3)}
                                </span>
                              ) : null}
                            </td>
                          )}
                          <td className="px-2 py-1.5 text-center">
                            {si === 0 && row.normality_test ? (
                              row.normality_mode === "within_group" && row.per_group_normality
                                && Object.keys(row.per_group_normality).length > 0 ? (
                                <div
                                  className={`text-[9px] px-1 py-0.5 rounded font-medium inline-block
                                    ${row.normal
                                      ? "bg-green-100 text-green-700 border border-green-300"
                                      : "bg-orange-100 text-orange-700 border border-orange-300"}`}
                                  title={Object.entries(row.per_group_normality).map(
                                    ([g, n]) => `${g}: ${n.normal ? "Normal" : "Non-normal"} (${n.test}, p=${fmtP(n.p)}, n=${n.n})`
                                  ).join("\n")}
                                >
                                  {row.normal ? "All groups normal" : "≥1 group non-normal"}
                                  <br />
                                  <span className="text-gray-400 font-normal">
                                    per-group · hover for detail
                                  </span>
                                  <div className="mt-0.5 flex flex-wrap gap-0.5 justify-center">
                                    {Object.entries(row.per_group_normality).map(([g, n]) => (
                                      <span
                                        key={g}
                                        className={`px-1 rounded text-[8px] font-mono
                                          ${n.normal ? "bg-green-200 text-green-800" : "bg-orange-200 text-orange-800"}`}
                                        title={`${g}: ${n.test}, p=${fmtP(n.p)}, n=${n.n}`}
                                      >
                                        {g}:{fmtP(n.p)}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              ) : (
                                <div className={`text-[9px] px-1 py-0.5 rounded font-medium inline-block
                                  ${row.normal
                                    ? "bg-green-100 text-green-700 border border-green-300"
                                    : "bg-orange-100 text-orange-700 border border-orange-300"}`}>
                                  {row.normal ? "Normal" : "Non-normal"}
                                  <br />
                                  <span className="text-gray-400 font-normal">
                                    {row.normality_test === "Shapiro-Wilk" ? "S-W" : "K-S"} <i>p</i>={fmtP(row.normality_p)}
                                  </span>
                                </div>
                              )
                            ) : null}
                          </td>
                        </tr>
                      ))}
                      </React.Fragment>
                    ) : (
                      <React.Fragment key={`cat-${ri}`}>
                        <tr key={`${ri}-hdr`} className="border-t-2 border-gray-200 bg-gray-50 group/cat">
                          <td className="px-4 py-2 font-semibold text-indigo-600 border-r border-gray-200">
                            {editRowIdx === ri ? (
                              <input
                                ref={editRowRef}
                                value={editRowVal}
                                onChange={(e) => setEditRowVal(e.target.value)}
                                onBlur={commitRowRename}
                                onKeyDown={(e) => { if (e.key === "Enter") commitRowRename(); if (e.key === "Escape") setEditRowIdx(null); }}
                                className="w-full text-sm font-semibold text-indigo-600 border border-indigo-300 rounded px-1 py-0.5 bg-white outline-none"
                              />
                            ) : (
                              <span className="flex items-center gap-1">
                                <span className="opacity-0 group-hover/cat:opacity-100 flex flex-col -my-1 text-gray-300">
                                  <button onClick={() => moveRow(ri, -1)} disabled={ri === 0}
                                    className="hover:text-indigo-500 disabled:opacity-20 leading-none text-[10px]">▲</button>
                                  <button onClick={() => moveRow(ri, 1)} disabled={ri === result.rows.length - 1}
                                    className="hover:text-indigo-500 disabled:opacity-20 leading-none text-[10px]">▼</button>
                                </span>
                                <span
                                  className="cursor-pointer hover:underline hover:decoration-indigo-300"
                                  onDoubleClick={() => { setEditRowIdx(ri); setEditRowVal(row.variable); }}
                                  title="Double-click to rename · arrows to reorder"
                                >
                                  {row.variable}
                                </span>
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-center text-xs text-gray-400 border-r border-gray-200"><i>n</i> (%)</td>
                          <td className="px-4 py-2 text-center text-xs text-gray-400 border-r border-gray-200"><i>n</i> = {row.overall_n}</td>
                          {result.group_labels.map((g) => {
                            const gn = (row.sub_rows ?? []).reduce((s, sr) => {
                              const m = sr.group_stats[g]?.match(/^(\d+)/);
                              return s + parseInt(m?.[1] ?? "0");
                            }, 0);
                            return (
                              <td key={g} className="px-4 py-2 text-center text-xs text-gray-400 border-r border-gray-200">
                                <i>n</i> = {gn}
                              </td>
                            );
                          })}
                          {hasGroups && (
                            <>
                              <td className={`px-3 py-2 text-center text-xs font-mono ${pColor(row.p_value)}`}>
                                {row.p_value ?? "—"}
                                {row.p_value && <span className="ml-0.5 text-[10px] opacity-70">{pStars(row.p_value)}</span>}
                              </td>
                              <td className="px-3 py-2 text-center text-xs text-gray-400">{row.test}</td>
                            </>
                          )}
                          {hasGroups && showSMD && (
                            <td className="px-3 py-2 text-center text-xs font-mono">
                              {row.smd != null ? (
                                <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                                  row.smd > 0.2 ? "bg-amber-100 text-amber-700" :
                                  row.smd > 0.1 ? "bg-yellow-50 text-yellow-700" :
                                  "bg-green-50 text-green-700"
                                }`}>
                                  {row.smd.toFixed(3)}
                                </span>
                              ) : "—"}
                            </td>
                          )}
                          <td className="px-2 py-2 text-center text-[9px] text-gray-300">—</td>
                        </tr>
                        {(row.sub_rows ?? []).map((sr, si) => (
                          <tr key={`${ri}-${si}`} className="border-t border-gray-100 hover:bg-gray-50 transition-colors">
                            <td className="px-4 py-1.5 pl-8 text-gray-500 text-xs border-r border-gray-200">
                              <span className="text-gray-300 mr-1">›</span>{sr.category}
                            </td>
                            <td className="border-r border-gray-200" />
                            <td className="px-4 py-1.5 text-center text-gray-700 font-mono text-xs border-r border-gray-200">{sr.overall}</td>
                            {result.group_labels.map((g) => (
                              <td key={g} className="px-4 py-1.5 text-center text-gray-700 font-mono text-xs border-r border-gray-200">
                                {sr.group_stats[g] ?? "—"}
                              </td>
                            ))}
                            {hasGroups && <><td /><td /></>}
                            {hasGroups && showSMD && <td />}
                            <td />
                          </tr>
                        ))}
                      </React.Fragment>
                    )
                  )}
                </tbody>
              </table>
            </div>

            {/* Interpretation banner */}
            <div className="mt-3 px-3 py-2 bg-indigo-50 border border-indigo-200 rounded-lg text-[11px] text-indigo-800 space-y-1 leading-relaxed">
              <p className="font-semibold text-indigo-900">How to read this table</p>
              <p>
                Continuous variables are tested for normality to decide how they are summarised and compared:
                <span className="font-medium"> n ≤ 2000 → Shapiro-Wilk · |skewness| ≤ 1.5 at large n → CLT bypass · otherwise → Lilliefors.</span>
              </p>
              <p>Normal → <span className="font-medium">Mean ± SD</span> with t-test/ANOVA · Non-normal → <span className="font-medium">Median [IQR]</span> with Mann-Whitney/Kruskal-Wallis.</p>
              <p>Categorical: Chi-square · Fisher's exact when any expected cell &lt; 5.</p>
              <p className="text-indigo-600">*** <i>p</i>&lt;0.001 · ** <i>p</i>&lt;0.01 · * <i>p</i>&lt;0.05 · ns = not significant</p>
            </div>

            {/* ── Format for Journal ── */}
            <JournalFormatSection result={result} />
          </div>
        )}
      </div>
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════════
// Journal Format Section (AMA style)
// ═════════════════════════════════════════════════════════════════════════════

interface JournalData {
  html?: string;
  validation?: Record<string, string> & { status?: string };
  abbreviations?: Record<string, string>;
  footnotes?: string[];
}

function JournalFormatSection({ result }: { result: T1Result }) {
  const [journalData, setJournalData] = useState<JournalData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [boldP, setBoldP] = useState(true);

  const formatForJournal = async () => {
    setLoading(true); setError(null);
    try {
      const res = await api.post("/api/pub_tables/format", {
        table1_result: result,
        options: { bold_significant_p: boldP, table_number: 1 },
      });
      setJournalData(res.data);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Format failed");
    } finally {
      setLoading(false);
    }
  };

  const exportJournal = async (fmt: "xlsx" | "docx") => {
    if (!journalData) return;
    try {
      const res = await api.post("/api/pub_tables/export", {
        formatted_table: journalData,
        format: fmt,
      }, { responseType: "blob" });
      const blob = new Blob([res.data]);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `Table_1_AMA.${fmt}`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 100);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? (e instanceof Error ? e.message : String(e))
        ?? "Unknown error";
      setError(`${fmt.toUpperCase()} export failed: ${detail}`);
      console.error("Journal export failed:", e);
    }
  };

  return (
    <div className="mt-4 space-y-3">
      <div className="flex items-center gap-3">
        <button onClick={formatForJournal} disabled={loading}
          className="btn-primary text-sm px-4 py-1.5">
          {loading ? "Formatting…" : "Format for Journal (AMA)"}
        </button>
        <label className="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer">
          <input type="checkbox" checked={boldP} onChange={(e) => setBoldP(e.target.checked)}
            className="accent-indigo-500" />
          Bold significant p-values
        </label>
      </div>

      {error && <p className="text-red-500 text-xs">{error}</p>}

      {journalData && (
        <div className="space-y-3">
          {/* Validation status */}
          <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold ${
            journalData.validation?.status === "READY FOR SUBMISSION"
              ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
              : "bg-amber-50 text-amber-700 border border-amber-200"
          }`}>
            {journalData.validation?.status === "READY FOR SUBMISSION" ? "✓" : "⚠"}
            {" "}{journalData.validation?.status}
          </div>

          {/* Validation details */}
          <div className="flex flex-wrap gap-2">
            {Object.entries(journalData.validation || {}).filter(([k]) => k !== "status").map(([k, v]) => (
              <span key={k} className={`text-[10px] px-2 py-0.5 rounded-full border ${
                v === "PASS" ? "bg-emerald-50 text-emerald-600 border-emerald-200" :
                v === "WARN" ? "bg-amber-50 text-amber-600 border-amber-200" :
                "bg-red-50 text-red-600 border-red-200"
              }`}>
                {k.replace(/_/g, " ")}: {v as string}
              </span>
            ))}
          </div>

          {/* Export buttons */}
          <div className="flex items-center gap-2">
            <button onClick={() => exportJournal("docx")}
              className="text-xs px-3 py-1.5 rounded-lg border border-indigo-300 text-indigo-600 hover:bg-indigo-50 transition-colors font-medium">
              Download Word (.docx)
            </button>
            <button onClick={() => exportJournal("xlsx")}
              className="text-xs px-3 py-1.5 rounded-lg border border-emerald-300 text-emerald-600 hover:bg-emerald-50 transition-colors font-medium">
              Download Excel (.xlsx)
            </button>
            <button onClick={() => {
              if (journalData.html) navigator.clipboard.writeText(journalData.html);
            }}
              className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-50 transition-colors">
              Copy HTML
            </button>
          </div>

          {/* Abbreviations */}
          {journalData.abbreviations && Object.keys(journalData.abbreviations).length > 0 && (
            <div className="text-[10px] text-gray-500">
              <span className="font-semibold text-gray-600">Detected abbreviations: </span>
              {Object.entries(journalData.abbreviations).map(([k, v]) => (
                <span key={k} className="mr-2">{k} = {v as string};</span>
              ))}
            </div>
          )}

          {/* Journal-formatted HTML preview */}
          <div className="border border-gray-200 rounded-xl overflow-hidden bg-white p-4">
            <div dangerouslySetInnerHTML={{ __html: journalData.html ?? "" }} />
          </div>

          {/* Footnotes */}
          {(journalData.footnotes?.length ?? 0) > 0 && (
            <div className="text-[10px] text-gray-400 italic space-y-0.5 px-1">
              {(journalData.footnotes ?? []).map((fn: string, i: number) => (
                <p key={i}>{fn}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
