/**
 * ComputePanel — Create New Variable module.
 *
 * Sub-tabs:
 *  1. Formula   — pandas df.eval() expression builder
 *  2. Transform — log / sqrt / square / zscore / …
 *  3. Recode    — IF-THEN rule builder (np.select)
 *  4. Clinical  — preset cardiological calculators (BMI, eGFR, CHA₂DS₂-VASc)
 */

import { useState, useRef, useEffect } from "react";
import { useStore, isNumericKind } from "../store";
import type { ColMeta } from "../store";
import {
  computeFormula,
  computeTransform,
  computeRecode,
  computeClinical,
  deleteColumn,
  getUniqueValues,
} from "../api";
import { Tip } from "./Tip";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";

// ── Types ──────────────────────────────────────────────────────────────────────

interface ComputeResult {
  name: string;
  dtype: string;
  kind: "numeric" | "categorical" | "text" | "date";
  preview_values: (number | string | null)[];
  n_computed: number;
  n_missing: number;
}

interface FormulaTemplate { name: string; formula: string; }

// ── Constants ─────────────────────────────────────────────────────────────────

const TABS = ["Formula", "Transform", "Recode", "Clinical"] as const;
type Tab = (typeof TABS)[number];

const TRANSFORMS = [
  { id: "ln",       label: "Ln (natural log)",      note: "Negative/zero values → NaN" },
  { id: "log10",    label: "Log₁₀",                 note: "Negative/zero values → NaN" },
  { id: "sqrt",     label: "√ Square root",          note: "Negative values → NaN" },
  { id: "square",   label: "x² Square",              note: "" },
  { id: "exp",      label: "eˣ Exponential",         note: "Large values may overflow" },
  { id: "abs",      label: "|x| Absolute value",     note: "" },
  { id: "zscore",   label: "Z-score (standardise)",  note: "Mean=0, SD=1" },
  { id: "tertile",  label: "Tertile (3 groups)",     note: "Equal-sized groups: 1=low, 2=mid, 3=high" },
  { id: "quartile", label: "Quartile (4 groups)",    note: "Equal-sized groups: 1=Q1, 2=Q2, 3=Q3, 4=Q4" },
  { id: "median_split", label: "Median split (2 groups)", note: "Below median=0, Above median=1" },
];

const OPS = ["<", "<=", ">", ">=", "==", "!=", "contains", "!contains"] as const;

const TEMPLATES_KEY = "compute_templates";

// ── Helpers ───────────────────────────────────────────────────────────────────

function loadTemplates(): FormulaTemplate[] {
  try { return JSON.parse(localStorage.getItem(TEMPLATES_KEY) ?? "[]"); }
  catch { return []; }
}
function saveTemplate(t: FormulaTemplate) {
  const ts = loadTemplates().filter((x) => x.name !== t.name);
  localStorage.setItem(TEMPLATES_KEY, JSON.stringify([...ts, t]));
}
function deleteTemplate(name: string) {
  const ts = loadTemplates().filter((x) => x.name !== name);
  localStorage.setItem(TEMPLATES_KEY, JSON.stringify(ts));
}

function SuccessBadge({ result }: { result: ComputeResult }) {
  return (
    <div className="flex items-center gap-2 bg-green-50 border border-green-200 rounded-lg px-3 py-2 text-xs text-green-800">
      <span>✓</span>
      <span>
        <strong>{result.name}</strong> created — {result.n_computed} values computed
        {result.n_missing > 0 && `, ${result.n_missing} missing (NaN)`}
      </span>
    </div>
  );
}

function ErrorBanner({ msg }: { msg: string }) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-700">
      {msg}
    </div>
  );
}

// ── Computed columns sidebar ──────────────────────────────────────────────────

function ComputedColumnsList({
  sessionId,
  computed,
  onDelete,
}: {
  sessionId: string;
  computed: ColMeta[];
  onDelete: (name: string) => void;
}) {
  const [deleting, setDeleting] = useState<string | null>(null);

  const handleDelete = async (name: string) => {
    setDeleting(name);
    try {
      await deleteColumn(sessionId, name);
      onDelete(name);
    } catch {
      // ignore
    } finally {
      setDeleting(null);
    }
  };

  if (!computed.length) return null;

  return (
    <div className="panel space-y-2">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
        Computed this session
      </p>
      {computed.map((c) => (
        <div key={c.name} className="flex items-center justify-between gap-2">
          <span className="text-xs font-mono text-gray-700 truncate">{c.name}</span>
          <div className="flex items-center gap-1 flex-shrink-0">
            <span className="text-[10px] bg-gray-100 text-gray-500 rounded px-1">{c.kind[0].toUpperCase()}</span>
            <button
              onClick={() => handleDelete(c.name)}
              disabled={deleting === c.name}
              className="text-xs text-red-400 hover:text-red-600 px-1 disabled:opacity-40"
              title="Delete this column from session"
            >
              {deleting === c.name ? "…" : "✕"}
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Tab 1: Formula Builder ────────────────────────────────────────────────────

function FormulaTab({
  sessionId,
  columns,
  onResult,
}: {
  sessionId: string;
  columns: ColMeta[];
  onResult: (r: ComputeResult) => void;
}) {
  const [formula, setFormula] = usePersistedPanelState<string>("compute_formula", "formula", "");
  const [newCol, setNewCol] = usePersistedPanelState<string>("compute_formula", "newCol", "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<ComputeResult | null>(null);
  const [templates, setTemplates] = useState<FormulaTemplate[]>(loadTemplates);
  const [tplName, setTplName] = useState("");
  const [showTplInput, setShowTplInput] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const insert = (text: string) => {
    const el = inputRef.current;
    if (!el) { setFormula((f) => f + text); return; }
    const start = el.selectionStart ?? formula.length;
    const end = el.selectionEnd ?? formula.length;
    const next = formula.slice(0, start) + text + formula.slice(end);
    setFormula(next);
    // restore cursor after text
    setTimeout(() => {
      el.focus();
      el.setSelectionRange(start + text.length, start + text.length);
    }, 0);
  };

  const backspace = () => {
    const el = inputRef.current;
    if (!el) { setFormula((f) => f.slice(0, -1)); return; }
    const start = el.selectionStart ?? formula.length;
    const end = el.selectionEnd ?? formula.length;
    if (start !== end) {
      setFormula(formula.slice(0, start) + formula.slice(end));
    } else if (start > 0) {
      setFormula(formula.slice(0, start - 1) + formula.slice(start));
      setTimeout(() => el.setSelectionRange(start - 1, start - 1), 0);
    }
  };

  const run = async () => {
    if (!formula.trim() || !newCol.trim()) return;
    setLoading(true); setError(null); setSuccess(null);
    try {
      const res = await computeFormula(sessionId, { formula, new_col: newCol.trim() });
      setSuccess(res.data);
      onResult(res.data);
    } catch (e: unknown) {
      setError((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Computation failed");
    } finally { setLoading(false); }
  };

  const saveTpl = () => {
    if (!tplName.trim() || !formula.trim()) return;
    const t = { name: tplName.trim(), formula };
    saveTemplate(t);
    setTemplates(loadTemplates());
    setTplName(""); setShowTplInput(false);
  };

  const removeTpl = (name: string) => {
    deleteTemplate(name);
    setTemplates(loadTemplates());
  };

  return (
    <div className="flex gap-4 h-full">
      {/* Variable list */}
      <div className="w-44 flex-shrink-0 space-y-2">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
          Variables
          <Tip text="Click a variable name to insert it into the formula at the cursor position." />
        </p>
        <div className="overflow-y-auto max-h-[calc(100vh-300px)] space-y-0.5">
          {columns.map((c) => (
            <button
              key={c.name}
              onClick={() => insert(c.name)}
              className="block w-full text-left text-xs font-mono px-2 py-1 rounded hover:bg-indigo-50 hover:text-indigo-700 text-gray-700 truncate"
              title={`${c.name} (${c.kind})`}
            >
              {c.name}
            </button>
          ))}
        </div>

        {/* Templates */}
        {templates.length > 0 && (
          <div className="border-t border-gray-200 pt-2 space-y-1">
            <p className="text-[10px] text-gray-400 uppercase tracking-wide">Saved templates</p>
            {templates.map((t) => (
              <div key={t.name} className="flex items-center gap-1">
                <button
                  onClick={() => setFormula(t.formula)}
                  className="flex-1 text-left text-xs text-indigo-600 hover:text-indigo-800 truncate"
                  title={t.formula}
                >
                  {t.name}
                </button>
                <button onClick={() => removeTpl(t.name)} className="text-red-400 hover:text-red-600 text-[10px] px-0.5">✕</button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Formula editor */}
      <div className="flex-1 space-y-3">
        <div className="panel space-y-3">
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-500 w-28 flex-shrink-0">New column name</label>
            <input
              className="select flex-1 text-sm font-mono"
              placeholder="e.g. BMI"
              value={newCol}
              onChange={(e) => setNewCol(e.target.value)}
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs text-gray-500 flex items-center">
              Formula
              <Tip wide text="Use column names exactly as they appear in the list. Operators: + - * / ** (power). Wrap column names containing spaces in backticks: `Column Name`. Missing values propagate automatically." />
            </label>
            <textarea
              ref={inputRef}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono focus:border-indigo-500 focus:outline-none resize-none"
              rows={3}
              placeholder="e.g.  Weight / ((Height / 100) ** 2)"
              value={formula}
              onChange={(e) => setFormula(e.target.value)}
            />
          </div>

          {/* Operator buttons */}
          <div className="flex flex-wrap gap-1">
            {["+", "−", "×", "÷", "(", ")", "**", " "].map((op) => {
              const val = op === "−" ? "-" : op === "×" ? "*" : op === "÷" ? "/" : op === " " ? " " : op;
              return (
                <button
                  key={op}
                  onClick={() => insert(val)}
                  className="px-2.5 py-1 text-sm font-mono rounded border border-gray-300 hover:bg-gray-100 bg-white text-gray-700"
                >
                  {op}
                </button>
              );
            })}
            <button
              onClick={backspace}
              className="px-2.5 py-1 text-sm rounded border border-gray-300 hover:bg-gray-100 bg-white text-gray-500"
              title="Delete last character"
            >⌫</button>
            <button
              onClick={() => setFormula("")}
              className="px-2.5 py-1 text-xs rounded border border-gray-300 hover:bg-gray-100 bg-white text-gray-400"
            >Clear</button>
          </div>

          <div className="flex gap-2 flex-wrap">
            <button
              className="btn-primary"
              onClick={run}
              disabled={loading || !formula.trim() || !newCol.trim()}
            >
              {loading ? "Computing…" : "Apply Formula"}
            </button>

            {/* Save template */}
            {!showTplInput ? (
              <button
                className="px-3 py-1.5 text-xs rounded border border-gray-300 text-gray-500 hover:bg-gray-100"
                onClick={() => setShowTplInput(true)}
                disabled={!formula.trim()}
              >
                Save as template
              </button>
            ) : (
              <div className="flex items-center gap-1">
                <input
                  className="select text-xs w-36"
                  placeholder="Template name"
                  value={tplName}
                  onChange={(e) => setTplName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && saveTpl()}
                />
                <button className="text-xs text-indigo-600 hover:text-indigo-800 px-1" onClick={saveTpl}>Save</button>
                <button className="text-xs text-gray-400 px-1" onClick={() => setShowTplInput(false)}>✕</button>
              </div>
            )}
          </div>

          {error && <ErrorBanner msg={error} />}
          {success && <SuccessBadge result={success} />}
        </div>

        {/* Examples */}
        <div className="panel text-xs space-y-1 text-gray-500">
          <p className="font-semibold text-gray-700">Examples</p>
          {[
            ["BMI", "Weight / ((Height / 100) ** 2)"],
            ["Pulse pressure", "Systolic - Diastolic"],
            ["Log Troponin", "— use the Transform tab for log transforms"],
            ["Creatinine ratio", "Creatinine / Urea"],
          ].map(([label, ex]) => (
            <div key={label} className="flex items-baseline gap-2">
              <span className="font-medium text-gray-600 w-28 flex-shrink-0">{label}:</span>
              <span className="font-mono text-indigo-600">{ex}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Tab 2: Transform ──────────────────────────────────────────────────────────

function TransformTab({
  sessionId,
  numCols,
  allCols,
  onResult,
}: {
  sessionId: string;
  numCols: ColMeta[];
  // Every column, not just the numeric ones: a name clash with a text column
  // overwrites it just as thoroughly.
  allCols: ColMeta[];
  onResult: (r: ComputeResult) => void;
}) {
  const [srcCol, setSrcCol] = usePersistedPanelState<string>("compute_transform", "srcCol", numCols[0]?.name ?? "");
  const [transform, setTransform] = usePersistedPanelState<string>("compute_transform", "transform", "ln");
  const [newCol, setNewCol] = usePersistedPanelState<string>("compute_transform", "newCol", "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<ComputeResult | null>(null);

  const selectedT = TRANSFORMS.find((t) => t.id === transform);

  // Auto-suggest new column name.
  //
  // The prefix map used to have no entry for tertile, quartile or median
  // split, and the fallback was the empty string — so for those three the
  // suggested name WAS the source column, and one click on Apply replaced the
  // variable with its own groups. Every transform now carries a prefix, and
  // the fallback derives one from the transform id rather than from nothing.
  useEffect(() => {
    if (!srcCol) return;
    const prefixMap: Record<string, string> = {
      ln: "Ln_", log10: "Log10_", sqrt: "Sqrt_",
      square: "Sq_", exp: "Exp_", abs: "Abs_", zscore: "Z_",
      tertile: "Tert_", quartile: "Quart_", median_split: "MedSplit_",
    };
    const prefix = prefixMap[transform] ?? `${transform}_`;
    setNewCol(prefix + srcCol);
  }, [srcCol, transform, setNewCol]);

  // Typing a name that already exists is allowed — replacing a column you
  // computed a moment ago is a normal correction — but it has to be visible
  // before the click, not discovered afterwards.
  const clashes = newCol.trim() !== "" && allCols.some((c) => c.name === newCol.trim());
  const isSource = newCol.trim() === srcCol;

  const run = async () => {
    if (!srcCol || !newCol.trim()) return;
    setLoading(true); setError(null); setSuccess(null);
    try {
      const res = await computeTransform(sessionId, { source_col: srcCol, transform, new_col: newCol.trim() });
      setSuccess(res.data);
      onResult(res.data);
    } catch (e: unknown) {
      setError((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Transform failed");
    } finally { setLoading(false); }
  };

  return (
    <div className="max-w-lg space-y-4">
      <div className="panel space-y-4">
        <div className="space-y-1">
          <label className="text-xs text-gray-500" htmlFor="transform-src">Source variable</label>
          <select id="transform-src" className="select w-full" value={srcCol} onChange={(e) => setSrcCol(e.target.value)}>
            {numCols.map((c) => <option key={c.name}>{c.name}</option>)}
          </select>
        </div>

        <div className="space-y-1">
          <label className="text-xs text-gray-500 flex items-center" htmlFor="transform-kind">
            Transformation
            <Tip wide text="Logarithmic transforms are recommended for right-skewed biomarkers (Troponin, CRP, NT-proBNP) before regression analysis. Z-score standardises to mean=0, SD=1 for comparing across different scales." />
          </label>
          <select id="transform-kind" className="select w-full" value={transform} onChange={(e) => setTransform(e.target.value)}>
            {TRANSFORMS.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
          </select>
          {selectedT?.note && (
            <p className="text-[11px] text-amber-600">ℹ {selectedT.note}</p>
          )}
        </div>

        <div className="space-y-1">
          <label className="text-xs text-gray-500" htmlFor="transform-newcol">New column name</label>
          <input
            id="transform-newcol"
            className={`select w-full font-mono ${isSource ? "border-red-400" : clashes ? "border-amber-400" : ""}`}
            value={newCol}
            onChange={(e) => setNewCol(e.target.value)}
          />
          {isSource ? (
            <p className="text-[11px] text-red-600">
              This is the source column — the result would replace the values it
              is computed from. Give it a different name.
            </p>
          ) : clashes ? (
            <p className="text-[11px] text-amber-600">
              ⚠ <span className="font-mono">{newCol.trim()}</span> already exists
              and will be replaced.
            </p>
          ) : null}
        </div>

        <button
          className="btn-primary w-full"
          onClick={run}
          disabled={loading || !srcCol || !newCol.trim() || isSource}
        >
          {loading ? "Applying…" : "Apply Transform"}
        </button>

        {error && <ErrorBanner msg={error} />}
        {success && <SuccessBadge result={success} />}
      </div>
    </div>
  );
}

// ── Tab 3: Recode / Binning ───────────────────────────────────────────────────

interface Condition { col: string; op: string; val: string; }
interface Rule { conditions: Condition[]; result: string; }

function RecodeTab({
  sessionId,
  columns,
  onResult,
}: {
  sessionId: string;
  columns: ColMeta[];
  onResult: (r: ComputeResult) => void;
}) {
  const firstCol = columns[0]?.name ?? "";
  // Persist the rule-builder state across tab switches so navigating to Data
  // (or any other tab) and back doesn't wipe the user's in-progress recode.
  const [newCol, setNewCol] = usePersistedPanelState<string>("compute_recode", "newCol", "NewVar");
  const [elseVal, setElseVal] = usePersistedPanelState<string>("compute_recode", "elseVal", "");
  const [rules, setRules] = usePersistedPanelState<Rule[]>("compute_recode", "rules", [
    { conditions: [{ col: firstCol, op: "<", val: "" }], result: "" },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<ComputeResult | null>(null);

  // Value labels for the new recode column
  const [valueLabels, setValueLabels] = usePersistedPanelState<Record<string, string>>("compute_recode", "valueLabels", {});

  // Cache of unique values per column (fetched on demand)
  const [colValues, setColValues] = useState<Record<string, string[]>>({});
  const fetchingRef = useRef<Set<string>>(new Set());

  const ensureValues = (colName: string) => {
    if (colValues[colName] || fetchingRef.current.has(colName)) return;
    fetchingRef.current.add(colName);
    getUniqueValues(sessionId, colName)
      .then((r) => {
        const vals: string[] = r.data?.values ?? [];
        setColValues((prev) => ({
          ...prev,
          [colName]: vals
            .filter((v) => v !== null && v !== undefined && v !== "")
            .map(String)
            .slice(0, 200), // cap at 200 so datalist stays snappy
        }));
      })
      .catch(() => { /* ignore */ })
      .finally(() => fetchingRef.current.delete(colName));
  };

  // Pre-fetch values for all currently-used columns
  useEffect(() => {
    const used = new Set(rules.flatMap((r) => r.conditions.map((c) => c.col)));
    used.forEach(ensureValues);
  }, [rules]); // eslint-disable-line

  const addRule = () =>
    setRules((r) => [...r, { conditions: [{ col: firstCol, op: "<", val: "" }], result: "" }]);

  const duplicateRule = (i: number) =>
    setRules((r) => {
      const src = r[i];
      // Deep-copy conditions so edits on the duplicate don't mutate the original.
      const clone: Rule = {
        conditions: src.conditions.map((c) => ({ ...c })),
        result: src.result,
      };
      return [...r.slice(0, i + 1), clone, ...r.slice(i + 1)];
    });

  const removeRule = (i: number) => setRules((r) => r.filter((_, idx) => idx !== i));

  const addCond = (ri: number) =>
    setRules((r) => r.map((rule, idx) => idx !== ri ? rule : {
      ...rule,
      conditions: [...rule.conditions, { col: firstCol, op: "<", val: "" }],
    }));

  const removeCond = (ri: number, ci: number) =>
    setRules((r) => r.map((rule, idx) => idx !== ri ? rule : {
      ...rule,
      conditions: rule.conditions.filter((_, i) => i !== ci),
    }));

  const duplicateCond = (ri: number, ci: number) =>
    setRules((r) => r.map((rule, idx) => idx !== ri ? rule : {
      ...rule,
      conditions: [
        ...rule.conditions.slice(0, ci + 1),
        { ...rule.conditions[ci] },  // copy so edits don't mutate the original
        ...rule.conditions.slice(ci + 1),
      ],
    }));

  const updateCond = (ri: number, ci: number, patch: Partial<Condition>) =>
    setRules((r) => r.map((rule, idx) => idx !== ri ? rule : {
      ...rule,
      conditions: rule.conditions.map((c, i) => i !== ci ? c : { ...c, ...patch }),
    }));

  const updateResult = (ri: number, val: string) =>
    setRules((r) => r.map((rule, idx) => idx !== ri ? rule : { ...rule, result: val }));

  // Auto-suggest a short value label from the rule that produces each assigned
  // value (e.g. value "1" from "Age < 51" → "Age<51"). The user can edit or
  // clear it; once touched, we never overwrite their text.
  const userTouchedLabels = useRef<Set<string>>(new Set());

  useEffect(() => {
    const condAbbrev = (c: Condition) => `${c.col}${c.op}${c.val}`.replace(/\s+/g, "");
    const autoFor = (v: string): string => {
      const rule = rules.find((r) => r.result.trim() === v);
      if (rule) {
        const s = rule.conditions.map(condAbbrev).join(" & ");
        return s.length > 28 ? s.slice(0, 27) + "…" : s;
      }
      if (elseVal.trim() === v) return "Other";
      return "";
    };
    const assigned = [
      ...rules.map((r) => r.result.trim()).filter(Boolean),
      ...(elseVal.trim() ? [elseVal.trim()] : []),
    ];
    const uniq = [...new Set(assigned)];
    setValueLabels((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const v of uniq) {
        if (userTouchedLabels.current.has(v)) continue;
        const auto = autoFor(v);
        if (auto && next[v] !== auto) { next[v] = auto; changed = true; }
      }
      return changed ? next : prev;
    });
  }, [rules, elseVal, setValueLabels]);

  const run = async () => {
    setLoading(true); setError(null); setSuccess(null);
    const payload = {
      rules: rules.map((r) => ({
        conditions: r.conditions.map((c) => ({ col: c.col, op: c.op, val: c.val })),
        result: r.result,
      })),
      else_val: elseVal.trim() || null,
      new_col: newCol.trim(),
    };
    try {
      const res = await computeRecode(sessionId, payload);
      setSuccess(res.data);
      onResult(res.data);
      // Guard the common mistake: a recode whose rules matched no rows produces
      // an all-empty column. Warn so the user notices (and can delete it) rather
      // than thinking value-labels created a stray column.
      if ((res.data?.n_computed ?? 0) === 0) {
        setError(`"${newCol.trim()}" came out entirely empty — no rule matched. Right-click → Delete column to remove it.`);
      }
      // Save value labels to metadata if any were defined
      const filledLabels = Object.fromEntries(
        Object.entries(valueLabels).filter(([, v]) => v.trim())
      );
      if (Object.keys(filledLabels).length > 0) {
        import("../api").then(({ saveMetadata }) => {
          saveMetadata(sessionId, { [newCol.trim()]: { value_labels: filledLabels } });
        });
        // Also update local store
        const session = useStore.getState().session;
        if (session) {
          const updatedCols = session.columns.map((c) =>
            c.name === newCol.trim() ? { ...c, value_labels: filledLabels } : c
          );
          useStore.getState().setSession({ ...session, columns: updatedCols });
        }
      }
    } catch (e: unknown) {
      setError((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Recode failed");
    } finally { setLoading(false); }
  };

  return (
    <div className="space-y-4 max-w-2xl">
      <div className="panel space-y-4">
        <div className="flex items-center gap-3">
          <label className="text-xs text-gray-500 flex-shrink-0">New column name</label>
          <input className="select flex-1 font-mono" value={newCol} onChange={(e) => setNewCol(e.target.value)} />
        </div>

        {/* Rules */}
        <div className="space-y-3">
          <p className="text-xs font-semibold text-gray-600 flex items-center">
            Rules
            <Tip wide text="Rules are evaluated in order. The first matching rule wins. If no rule matches and no Else value is set, the result will be NaN (missing)." />
          </p>

          {rules.map((rule, ri) => (
            <div key={ri} className="border border-gray-200 rounded-lg p-3 space-y-2 bg-gray-50">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-gray-500">Rule {ri + 1}</span>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => duplicateRule(ri)}
                    className="text-xs text-indigo-500 hover:text-indigo-700"
                    title="Duplicate this rule (inserted directly below)"
                  >
                    ⧉ Duplicate
                  </button>
                  <button
                    onClick={() => removeRule(ri)}
                    className="text-xs text-red-400 hover:text-red-600"
                  >
                    ✕ Remove
                  </button>
                </div>
              </div>

              {rule.conditions.map((cond, ci) => {
                const listId = `dl-${ri}-${ci}-${cond.col}`;
                const vals = colValues[cond.col] ?? [];
                return (
                <div key={ci} className="flex items-center gap-1.5 flex-wrap">
                  {ci > 0 && <span className="text-xs text-indigo-600 font-semibold w-8 text-center">AND</span>}
                  {ci === 0 && <span className="text-xs text-gray-400 w-8 text-center">IF</span>}

                  <select
                    className="select text-xs flex-1 min-w-[100px]"
                    value={cond.col}
                    onChange={(e) => {
                      updateCond(ri, ci, { col: e.target.value, val: "" });
                      ensureValues(e.target.value);
                    }}
                  >
                    {columns.map((c) => <option key={c.name}>{c.name}</option>)}
                  </select>

                  <select
                    className="select text-xs w-24"
                    value={cond.op}
                    onChange={(e) => updateCond(ri, ci, { op: e.target.value })}
                  >
                    {OPS.map((op) => <option key={op} value={op}>{op}</option>)}
                  </select>

                  {/* Datalist gives dropdown suggestions but still allows free typing */}
                  {vals.length > 0 && <datalist id={listId}>
                    {vals.map((v) => <option key={v} value={v} />)}
                  </datalist>}
                  <input
                    className="select text-xs w-28 font-mono"
                    placeholder="value"
                    value={cond.val}
                    list={vals.length > 0 ? listId : undefined}
                    onChange={(e) => updateCond(ri, ci, { val: e.target.value })}
                  />

                  <button
                    onClick={() => duplicateCond(ri, ci)}
                    title="Duplicate this condition (inserted directly below)"
                    className="text-indigo-300 hover:text-indigo-600 text-xs"
                  >⧉</button>
                  {rule.conditions.length > 1 && (
                    <button onClick={() => removeCond(ri, ci)} className="text-red-300 hover:text-red-500 text-xs">✕</button>
                  )}
                </div>
                );
              })}

              {/* Add AND condition */}
              <button
                onClick={() => addCond(ri)}
                className="text-[11px] text-indigo-500 hover:text-indigo-700"
              >
                + AND condition
              </button>

              {/* Result value */}
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-gray-500 flex-shrink-0">→ Assign value:</span>
                <input
                  className="select text-xs flex-1 font-mono font-semibold"
                  placeholder="e.g. 1 or 'HFrEF'"
                  value={rule.result}
                  onChange={(e) => updateResult(ri, e.target.value)}
                />
              </div>
            </div>
          ))}

          <button
            onClick={addRule}
            className="w-full py-1.5 border border-dashed border-indigo-300 rounded-lg text-xs text-indigo-500 hover:border-indigo-500 hover:bg-indigo-50 transition-colors"
          >
            + Add rule
          </button>
        </div>

        {/* Else value */}
        <div className="flex items-center gap-3 pt-1 border-t border-gray-200">
          <label className="text-xs text-gray-500 flex-shrink-0 flex items-center">
            Else (no rule matches):
            <Tip text="If none of the rules above match a row, this value is used. Leave blank to set as NaN (missing)." />
          </label>
          <input
            className="select flex-1 text-xs font-mono"
            placeholder="Leave blank for NaN (missing)"
            value={elseVal}
            onChange={(e) => setElseVal(e.target.value)}
          />
        </div>

        {/* Value Labels (auto-populated from assigned values) */}
        {(() => {
          const assignedVals = [
            ...rules.map((r) => r.result).filter((v) => v.trim()),
            ...(elseVal.trim() ? [elseVal.trim()] : []),
          ];
          const uniqueVals = [...new Set(assignedVals)].sort((a, b) => {
            const na = Number(a), nb = Number(b);
            return (!isNaN(na) && !isNaN(nb)) ? na - nb : a.localeCompare(b);
          });
          if (uniqueVals.length === 0) return null;
          return (
            <div className="border-t border-gray-200 pt-3 space-y-2">
              <p className="text-xs font-semibold text-gray-600 flex items-center gap-1">
                🔤 Value Labels
                <span className="font-normal text-gray-400">(optional — will appear in charts)</span>
              </p>
              <div className="space-y-1.5">
                {uniqueVals.map((val) => (
                  <div key={val} className="flex items-center gap-2">
                    <span className="w-12 text-xs font-mono text-gray-500 bg-gray-100 px-2 py-1 rounded text-center flex-shrink-0">
                      {val}
                    </span>
                    <span className="text-gray-400 text-xs">=</span>
                    <input
                      className="flex-1 text-xs border border-gray-300 rounded-lg px-2 py-1 focus:outline-none focus:border-indigo-400"
                      placeholder={`Label for ${val}`}
                      value={valueLabels[val] ?? ""}
                      onChange={(e) => {
                        userTouchedLabels.current.add(val);  // stop auto-fill from overwriting
                        setValueLabels((prev) => ({ ...prev, [val]: e.target.value }));
                      }}
                    />
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        <button className="btn-primary w-full" onClick={run} disabled={loading || !newCol.trim()}>
          {loading ? "Applying…" : "Apply Recode"}
        </button>

        {error && <ErrorBanner msg={error} />}
        {success && <SuccessBadge result={success} />}
      </div>
    </div>
  );
}

// ── Tab 4: Clinical Calculators ───────────────────────────────────────────────

interface CalcField {
  key: string;
  label: string;
  required: boolean;
  isSex?: boolean;
  isBinary?: boolean;
  note?: string;       // small italic hint below the field
}

interface CalcDef {
  id: string;
  icon: string;
  title: string;
  subtitle: string;
  fields: CalcField[];
  defaultCol: string;
  info: string;        // plain-English interpretation hint
}

interface CalcGroup {
  label: string;
  icon: string;
  calcs: CalcDef[];
}

const CALC_GROUPS: CalcGroup[] = [
  {
    label: "Basic & Hemodynamic",
    icon: "⚖️",
    calcs: [
      {
        id: "bmi", icon: "⚖️", title: "BMI", subtitle: "Body Mass Index",
        defaultCol: "BMI",
        info: "Underweight <18.5 · Normal 18.5–24.9 · Overweight 25–29.9 · Obese ≥30",
        fields: [
          { key: "weight", label: "Weight (kg)", required: true },
          { key: "height", label: "Height (cm)", required: true },
        ],
      },
      {
        id: "bsa", icon: "📏", title: "BSA", subtitle: "Body Surface Area (Mosteller)",
        defaultCol: "BSA",
        info: "Normal adult ~1.7 m². Used to index echo measurements (LVMI = LVM / BSA).",
        fields: [
          { key: "weight", label: "Weight (kg)", required: true },
          { key: "height", label: "Height (cm)", required: true },
        ],
      },
      {
        id: "map", icon: "🩸", title: "MAP", subtitle: "Mean Arterial Pressure",
        defaultCol: "MAP",
        info: "MAP = (SBP + 2×DBP) / 3. Normal: 70–100 mmHg. <60 mmHg indicates hypoperfusion.",
        fields: [
          { key: "sbp", label: "Systolic BP (mmHg)", required: true },
          { key: "dbp", label: "Diastolic BP (mmHg)", required: true },
        ],
      },
    ],
  },
  {
    label: "Kidney Function",
    icon: "🩺",
    calcs: [
      {
        id: "egfr", icon: "🩺", title: "eGFR", subtitle: "CKD-EPI 2021 (race-free)",
        defaultCol: "eGFR",
        info: "G1 ≥90 · G2 60–89 · G3a 45–59 · G3b 30–44 · G4 15–29 · G5 <15 mL/min/1.73m²",
        fields: [
          { key: "age",        label: "Age (years)",               required: true },
          { key: "sex",        label: "Sex / Gender",              required: true, isSex: true },
          { key: "creatinine", label: "Serum Creatinine (mg/dL)", required: true },
        ],
      },
    ],
  },
  {
    label: "Atrial Fibrillation",
    icon: "💓",
    calcs: [
      {
        id: "chadsva", icon: "🔴", title: "CHA₂DS₂-VA", subtitle: "AF stroke risk (2024 ESC)",
        defaultCol: "CHA2DS2VA",
        info: "Updated 2024 ESC guideline — sex category removed. Score ≥2 → anticoagulation recommended. Max 8.",
        fields: [
          { key: "age",    label: "Age (years)",              required: true },
          { key: "chf",    label: "Heart Failure (0/1)",      required: false, isBinary: true },
          { key: "htn",    label: "Hypertension (0/1)",       required: false, isBinary: true },
          { key: "dm",     label: "Diabetes (0/1)",           required: false, isBinary: true },
          { key: "stroke", label: "Stroke / TIA (0/1)",       required: false, isBinary: true },
          { key: "vasc",   label: "Vascular disease (0/1)",   required: false, isBinary: true },
        ],
      },
      {
        id: "hasbled", icon: "🩺", title: "HAS-BLED", subtitle: "Bleeding risk in AF",
        defaultCol: "HAS_BLED",
        info: "Score ≥3 = high bleeding risk. Use to identify and correct reversible risk factors, not to withhold anticoagulation. Max 9.",
        fields: [
          { key: "htn",        label: "Uncontrolled HTN (SBP>160, 0/1)",  required: false, isBinary: true },
          { key: "renal",      label: "Abnormal renal function (0/1)",     required: false, isBinary: true },
          { key: "liver",      label: "Abnormal liver function (0/1)",     required: false, isBinary: true },
          { key: "stroke",     label: "Stroke history (0/1)",              required: false, isBinary: true },
          { key: "bleeding",   label: "Bleeding history (0/1)",            required: false, isBinary: true },
          { key: "labile_inr", label: "Labile INR (0/1)",                  required: false, isBinary: true },
          { key: "age",        label: "Age (years) — auto-applies >65 rule", required: false,
            note: "If mapped, age > 65 = 1 pt automatically" },
          { key: "drugs",      label: "Antiplatelet / NSAID use (0/1)",    required: false, isBinary: true },
          { key: "alcohol",    label: "Alcohol use (0/1)",                 required: false, isBinary: true },
        ],
      },
    ],
  },
  {
    label: "Acute Coronary Syndrome",
    icon: "🚨",
    calcs: [
      {
        id: "grace", icon: "📊", title: "GRACE 2.0", subtitle: "ACS in-hospital mortality risk",
        defaultCol: "GRACE_Score",
        info: "Score ≤108: Low (<1%) · 109–140: Intermediate (1–3%) · >140: High (>3%) in-hospital mortality.",
        fields: [
          { key: "age",            label: "Age (years)",                        required: true },
          { key: "hr",             label: "Heart rate (bpm)",                   required: true },
          { key: "sbp",            label: "Systolic BP (mmHg)",                 required: true },
          { key: "creatinine",     label: "Serum Creatinine (mg/dL)",           required: true },
          { key: "killip",         label: "Killip class (1–4)",                  required: false,
            note: "Numeric column: 1=no signs, 2=rales, 3=pulm. oedema, 4=cardiogenic shock" },
          { key: "cardiac_arrest", label: "Cardiac arrest on admission (0/1)", required: false, isBinary: true },
          { key: "st_deviation",   label: "ST segment deviation (0/1)",         required: false, isBinary: true },
          { key: "cardiac_markers",label: "Elevated cardiac enzymes (0/1)",    required: false, isBinary: true },
        ],
      },
      {
        id: "timi_nstemi", icon: "⚡", title: "TIMI NSTEMI", subtitle: "UA/NSTEMI risk (0–7)",
        defaultCol: "TIMI_NSTEMI",
        info: "0–2: Low · 3–4: Intermediate · 5–7: High 14-day mortality/MI risk.",
        fields: [
          { key: "age",           label: "Age (years) — auto-applies ≥65 rule", required: false,
            note: "Age ≥ 65 = 1 point. Maps automatically." },
          { key: "risk_factors",  label: "≥3 CAD risk factors (0/1)",     required: false, isBinary: true,
            note: "Family Hx, HTN, DM, smoking, dyslipidaemia" },
          { key: "known_cad",     label: "Known CAD (stenosis ≥50%, 0/1)", required: false, isBinary: true },
          { key: "aspirin",       label: "Aspirin use in last 7 days (0/1)", required: false, isBinary: true },
          { key: "severe_angina", label: "≥2 anginal events in 24h (0/1)", required: false, isBinary: true },
          { key: "st_deviation",  label: "ST deviation ≥0.5 mm (0/1)",     required: false, isBinary: true },
          { key: "markers",       label: "Elevated cardiac markers (0/1)", required: false, isBinary: true },
        ],
      },
      {
        id: "timi_stemi", icon: "🔴", title: "TIMI STEMI", subtitle: "STEMI mortality risk (0–14)",
        defaultCol: "TIMI_STEMI",
        info: "0–2: <2% · 3–4: ~4% · 5–6: ~12% · ≥7: >16% 30-day mortality.",
        fields: [
          { key: "age",             label: "Age (years)",               required: true,
            note: "65–74 = 2 pts, ≥75 = 3 pts" },
          { key: "sbp",             label: "Systolic BP (mmHg)",        required: false,
            note: "SBP < 100 = 3 points" },
          { key: "hr",              label: "Heart rate (bpm)",          required: false,
            note: "HR > 100 = 2 points" },
          { key: "weight",          label: "Weight (kg)",               required: false,
            note: "Weight < 67 kg = 1 point" },
          { key: "killip",          label: "Killip class (1–4)",        required: false,
            note: "Killip II–IV = 2 points" },
          { key: "dm_htn_angina",   label: "DM / HTN / Angina (0/1)",  required: false, isBinary: true },
          { key: "anterior_stemi",  label: "Anterior ST ↑ or LBBB (0/1)", required: false, isBinary: true },
          { key: "late_treatment",  label: "Time to Tx > 4 hours (0/1)", required: false, isBinary: true },
        ],
      },
    ],
  },
  {
    label: "Heart Failure",
    icon: "💙",
    calcs: [
      {
        id: "h2fpef", icon: "💙", title: "H2FPEF", subtitle: "HFpEF probability (0–9)",
        defaultCol: "H2FPEF",
        info: "Score 0–1: Low probability · 2–5: Intermediate (further testing needed) · 6–9: High probability of HFpEF.",
        fields: [
          { key: "bmi",       label: "BMI (kg/m²)",                              required: false,
            note: "BMI > 30 = 2 pts (H²: Heavy)" },
          { key: "htn_meds",  label: "≥2 antihypertensive meds (0/1)",           required: false, isBinary: true },
          { key: "af",        label: "Atrial Fibrillation (0/1)",                required: false, isBinary: true,
            note: "AF = 3 pts (highest weight)" },
          { key: "pulm_htn",  label: "Pulmonary HTN — PASP >35 mmHg (0/1)",     required: false, isBinary: true },
          { key: "age",       label: "Age (years) — auto-applies >60 rule",      required: false,
            note: "Age > 60 = 1 pt" },
          { key: "ee_ratio",  label: "E/e' > 9 on echo (0/1)",                   required: false, isBinary: true },
        ],
      },
      {
        id: "maggic", icon: "📈", title: "MAGGIC", subtitle: "HF survival score (Pocock 2013)",
        defaultCol: "MAGGIC_Score",
        info: "Predicts 1- and 3-year mortality in chronic HF. Higher score = worse prognosis. Creatinine accepted in mg/dL or μmol/L (auto-detected).",
        fields: [
          { key: "age",            label: "Age (years)",                    required: true },
          { key: "sbp",            label: "Systolic BP (mmHg)",             required: true },
          { key: "bmi",            label: "BMI (kg/m²)",                    required: true },
          { key: "creatinine",     label: "Creatinine (mg/dL or μmol/L)", required: true,
            note: "Auto-detected: if values < 20 → mg/dL (×88.4); otherwise μmol/L" },
          { key: "ef",             label: "Ejection Fraction (%)",          required: true },
          { key: "nyha",           label: "NYHA class (1–4)",               required: false,
            note: "Defaults to NYHA II if not mapped" },
          { key: "sex",            label: "Sex / Gender",                   required: false, isSex: true,
            note: "Male = 1 extra point" },
          { key: "diabetes",       label: "Diabetes (0/1)",                 required: false, isBinary: true },
          { key: "copd",           label: "COPD (0/1)",                     required: false, isBinary: true },
          { key: "current_smoker", label: "Current smoker (0/1)",           required: false, isBinary: true },
          { key: "hf_lt18m",       label: "HF diagnosed < 18 months (0/1)", required: false, isBinary: true },
          { key: "bb",             label: "Beta-blocker treatment (0/1)",   required: false, isBinary: true,
            note: "NOT on BB = 3 pts; enter 1=on BB, 0=not on BB" },
          { key: "ace_arb",        label: "ACE inhibitor / ARB (0/1)",      required: false, isBinary: true,
            note: "NOT on ACE/ARB = 1 pt; enter 1=on, 0=not on" },
        ],
      },
    ],
  },
  {
    label: "Electrophysiology",
    icon: "⚡",
    calcs: [
      {
        id: "qtc", icon: "⚡", title: "QTc (Bazett)", subtitle: "Corrected QT interval",
        defaultCol: "QTc_Bazett",
        info: "Normal: ≤450 ms (men), ≤470 ms (women). QTc >500 ms = high risk for torsades de pointes.",
        fields: [
          { key: "qt", label: "QT interval (ms)",    required: true },
          { key: "hr", label: "Heart rate (bpm)",    required: true },
        ],
      },
    ],
  },
];

function ClinicalCalcForm({
  calc,
  sessionId,
  columns,
  onResult,
  onClose,
}: {
  calc: CalcDef;
  sessionId: string;
  columns: ColMeta[];
  onResult: (r: ComputeResult) => void;
  onClose: () => void;
}) {
  const [mapping, setMapping] = useState<Record<string, string>>(() =>
    Object.fromEntries(calc.fields.map((f) => [f.key, f.required ? (columns[0]?.name ?? "") : ""]))
  );
  const [femaleValue, setFemaleValue] = useState<string>("");
  const [uniqueVals, setUniqueVals] = useState<string[]>([]);
  const [newCol, setNewCol] = useState(calc.defaultCol);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<ComputeResult | null>(null);

  const sexField = calc.fields.find((f) => f.isSex);

  // Load unique values for sex column
  useEffect(() => {
    if (!sexField || !mapping[sexField.key] || !sessionId) return;
    getUniqueValues(sessionId, mapping[sexField.key])
      .then((r) => {
        setUniqueVals(r.data.values ?? []);
        setFemaleValue("");
      })
      .catch(() => setUniqueVals([]));
    // Re-fetch only when the sex-field mapping value (a primitive key) changes,
    // not on every `mapping`/`sexField` identity change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapping[sexField?.key ?? ""], sessionId]);

  const run = async () => {
    setLoading(true); setError(null); setSuccess(null);
    const payload = {
      column_map: mapping,
      female_value: femaleValue || undefined,
      new_col: newCol.trim(),
    };
    try {
      const res = await computeClinical(sessionId, calc.id, payload);
      setSuccess(res.data);
      onResult(res.data);
    } catch (e: unknown) {
      setError((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Calculation failed");
    } finally { setLoading(false); }
  };

  return (
    <div className="panel space-y-4 mt-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-gray-900">{calc.icon} {calc.title} — {calc.subtitle}</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-sm">✕</button>
      </div>

      {calc.fields.map((f) => (
        <div key={f.key} className="space-y-0.5">
          <label className="text-xs text-gray-500">
            {f.label}
            {f.required && <span className="text-red-400 ml-0.5">*</span>}
            {f.isBinary && <Tip text="Binary column: 1 = Yes / 0 = No. Leave as '— not available —' if not in your dataset (treated as 0)." />}
          </label>
          <select
            className="select w-full text-sm"
            value={mapping[f.key] ?? ""}
            onChange={(e) => setMapping((m) => ({ ...m, [f.key]: e.target.value }))}
          >
            {!f.required && <option value="">— not available —</option>}
            {columns.map((c) => <option key={c.name}>{c.name}</option>)}
          </select>
          {f.note && <p className="text-[10px] text-amber-600 italic">{f.note}</p>}

          {/* Sex column: show female value picker */}
          {f.isSex && mapping[f.key] && uniqueVals.length > 0 && (
            <div className="flex items-center gap-2 mt-1">
              <label className="text-xs text-gray-500 flex-shrink-0">
                Which value = Female?
                <Tip text="Select the value in your sex column that represents Female. The calculator uses this to apply the correct sex-specific coefficients." />
              </label>
              <select
                className="select flex-1 text-xs"
                value={femaleValue}
                onChange={(e) => setFemaleValue(e.target.value)}
              >
                <option value="">— auto-detect —</option>
                {uniqueVals.map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          )}
        </div>
      ))}

      <div className="flex items-center gap-3">
        <label className="text-xs text-gray-500 flex-shrink-0">Output column name</label>
        <input className="select flex-1 font-mono" value={newCol} onChange={(e) => setNewCol(e.target.value)} />
      </div>

      <button className="btn-primary w-full" onClick={run} disabled={loading || !newCol.trim()}>
        {loading ? "Calculating…" : `Calculate ${calc.title}`}
      </button>

      {error && <ErrorBanner msg={error} />}
      {success && <SuccessBadge result={success} />}

      {/* Interpretation hint */}
      <div className="flex items-start gap-2 bg-blue-50 border border-blue-100 rounded-lg px-3 py-2 text-xs text-blue-700 leading-relaxed">
        <span className="mt-0.5 flex-shrink-0">💡</span>
        <span>{calc.info}</span>
      </div>
    </div>
  );
}

function ClinicalTab({
  sessionId,
  columns,
  onResult,
}: {
  sessionId: string;
  columns: ColMeta[];
  onResult: (r: ComputeResult) => void;
}) {
  const [active, setActive] = useState<string | null>(null);
  const activeCalc = CALC_GROUPS.flatMap((g) => g.calcs).find((c) => c.id === active);

  return (
    <div className="space-y-5 max-w-3xl">
      {CALC_GROUPS.map((group) => (
        <div key={group.label} className="space-y-2">
          {/* Group header */}
          <div className="flex items-center gap-2">
            <span className="text-base">{group.icon}</span>
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              {group.label}
            </span>
            <div className="flex-1 border-t border-gray-200" />
          </div>

          {/* Calculator cards */}
          <div className="grid grid-cols-3 gap-2">
            {group.calcs.map((calc) => (
              <button
                key={calc.id}
                onClick={() => setActive(active === calc.id ? null : calc.id)}
                className={`panel text-left transition-all hover:border-indigo-300 hover:shadow-sm ${
                  active === calc.id ? "border-indigo-400 bg-indigo-50 shadow-sm" : ""
                }`}
              >
                <div className="text-xl mb-1">{calc.icon}</div>
                <div className="font-semibold text-gray-900 text-sm leading-tight">{calc.title}</div>
                <div className="text-[11px] text-gray-400 mt-0.5 leading-snug">{calc.subtitle}</div>
                <div className="mt-2 text-[11px] text-indigo-500 font-medium">
                  {active === calc.id ? "▲ Close" : "▶ Configure →"}
                </div>
              </button>
            ))}
          </div>

          {/* Expanded form — only shown if active calc is in this group */}
          {activeCalc && group.calcs.some((c) => c.id === active) && (
            <ClinicalCalcForm
              calc={activeCalc}
              sessionId={sessionId}
              columns={columns}
              onResult={onResult}
              onClose={() => setActive(null)}
            />
          )}
        </div>
      ))}
    </div>
  );
}



// ── Main ComputePanel ─────────────────────────────────────────────────────────

export default function ComputePanel() {
  const session = useStore((s) => s.session);
  const addSessionColumn = useStore((s) => s.addSessionColumn);
  const removeSessionColumn = useStore((s) => s.removeSessionColumn);

  // Hooks must be called before any early return
  const [tab, setTab] = usePersistedPanelState<Tab>("compute_panel", "tab", "Formula");
  // Track which columns were computed this session (stored in component state)
  const [computedNames, setComputedNames] = useState<string[]>([]);

  if (!session) return null;

  const handleResult = (result: ComputeResult) => {
    const col: ColMeta = { name: result.name, dtype: result.dtype, kind: result.kind };
    addSessionColumn(col, result.preview_values);
    setComputedNames((prev) => prev.includes(result.name) ? prev : [...prev, result.name]);
  };

  const handleDelete = (name: string) => {
    removeSessionColumn(name);
    setComputedNames((prev) => prev.filter((n) => n !== name));
  };

  const allCols = session.columns;
  const numCols = allCols.filter((c) => isNumericKind(c.kind));
  const computedCols = allCols.filter((c) => computedNames.includes(c.name));
  const sid = session.session_id;

  return (
    <div className="flex gap-4 h-full">
      {/* Left sidebar */}
      <div className="w-52 flex-shrink-0 space-y-4">
        {/* Tab switcher */}
        <div className="panel space-y-1 p-2">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`w-full text-left text-sm px-3 py-2 rounded-lg transition-colors ${
                tab === t
                  ? "bg-indigo-600 text-white font-medium"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              {t === "Formula"  && "🧮 "}
              {t === "Transform" && "📐 "}
              {t === "Recode"   && "🔀 "}
              {t === "Clinical" && "🫀 "}
              {t}
            </button>
          ))}
        </div>

        {/* Computed columns list */}
        <ComputedColumnsList
          sessionId={sid}
          computed={computedCols}
          onDelete={handleDelete}
        />
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto">
        {tab === "Formula" && (
          <FormulaTab sessionId={sid} columns={allCols} onResult={handleResult} />
        )}
        {tab === "Transform" && (
          numCols.length === 0
            ? <p className="text-gray-400 text-sm">No numeric columns available for transformation.</p>
            : <TransformTab sessionId={sid} numCols={numCols} allCols={allCols} onResult={handleResult} />
        )}
        {tab === "Recode" && (
          <RecodeTab sessionId={sid} columns={allCols} onResult={handleResult} />
        )}
        {tab === "Clinical" && (
          <ClinicalTab sessionId={sid} columns={allCols} onResult={handleResult} />
        )}
      </div>
    </div>
  );
}
