import { useState, type ReactNode } from "react";
import { runDirectStandardisation, runIndirectStandardisation, runRateRatio } from "../api";
import { Tip } from "./Tip";
import ResultExporter from "./ResultExporter";
import CopyTextButton from "./CopyTextButton";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";
import { fmtP, pCellTitle } from "../lib/format";
import { describeStale } from "../lib/resultStamp";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";

// Rate standardisation and incidence-rate comparison. All three modes take
// inline tables, so no dataset (and no session) is needed.

type Mode = "direct" | "indirect" | "rate_ratio";
type Cell = string | number | null | undefined;

/* ───────────────────────── shared helpers ───────────────────────── */

const INPUT = "text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400";
const SMALL_BTN = "text-[10px] px-2 py-0.5 rounded border border-indigo-200 text-indigo-600 hover:bg-indigo-50";
const TH = "px-2 py-1 font-medium";
const TD = "px-2 py-1 font-mono";

/** Number from a text cell; NaN for blank or non-numeric. */
const toNum = (s: string): number => (s.trim() === "" ? NaN : Number(s));
const isBlank = (s: string): boolean => s.trim() === "";
const isCount = (x: number): boolean => Number.isFinite(x) && x >= 0 && Number.isInteger(x);
const isNonNeg = (x: number): boolean => Number.isFinite(x) && x >= 0;
const isPositive = (x: number): boolean => Number.isFinite(x) && x > 0;

/** Fixed decimals; a dash for a missing or non-finite value. */
function fx(x: number | null | undefined, digits = 2): string {
  return x == null || !Number.isFinite(x) ? "NA" : x.toFixed(digits);
}

/** Significant digits, for raw rates whose magnitude is not known in advance. */
function sig(x: number | null | undefined, digits = 4): string {
  return x == null || !Number.isFinite(x) ? "NA" : String(Number(x.toPrecision(digits)));
}

function ciText(lo: number | null | undefined, hi: number | null | undefined, f: (x: number | null | undefined) => string): string {
  if (lo == null && hi == null) return "NA";
  return `${f(lo)} to ${hi == null ? "∞" : f(hi)}`;
}

/** The 422 `detail` text (a string, or a pydantic list of {msg}), or the transport message. */
function errorMessage(e: unknown): string {
  const err = e as { response?: { data?: { detail?: unknown } }; message?: string };
  const detail = err?.response?.data?.detail;
  if (Array.isArray(detail)) return detail.map((m: { msg?: string }) => m?.msg ?? String(m)).join(", ");
  if (typeof detail === "string") return detail;
  return err?.message ?? "Request failed";
}

/** Split pasted spreadsheet text into trimmed cells (tab, semicolon or comma separated). */
function parsePasted(text: string): string[][] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== "")
    .map((line) => {
      const sep = line.includes("\t") ? "\t" : line.includes(";") ? ";" : ",";
      return line.split(sep).map((c) => c.trim());
    });
}

/** Drop a header row (a first row whose numeric columns are not numbers). */
function dropHeader(rows: string[][], numericCols: number[]): string[][] {
  if (rows.length === 0) return rows;
  const first = rows[0];
  const looksNumeric = numericCols.every((c) => first[c] !== undefined && Number.isFinite(Number(first[c])) && first[c] !== "");
  return looksNumeric ? rows : rows.slice(1);
}

interface Warned { warnings?: string[]; result_text?: string; r_code?: string }

function Segmented<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: readonly (readonly [T, string])[];
}) {
  return (
    <div className="inline-flex gap-1 p-0.5 bg-chip rounded-lg" role="group">
      {options.map(([id, label]) => (
        <button key={id} type="button" aria-pressed={value === id} onClick={() => onChange(id)}
          className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
            value === id ? "bg-surface text-ink-600 shadow-card border border-line" : "text-slate-500 hover:text-slate-700"
          }`}>
          {label}
        </button>
      ))}
    </div>
  );
}

function Field({ label, children, tip }: { label: string; children: ReactNode; tip?: string }) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[10px] text-gray-500">{label}{tip && <Tip wide text={tip} />}</span>
      {children}
    </label>
  );
}

function Problems({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <ul className="text-[11px] text-red-600 space-y-0.5 list-disc pl-4" role="alert">
      {items.slice(0, 6).map((m) => <li key={m}>{m}</li>)}
      {items.length > 6 && <li>and {items.length - 6} more</li>}
    </ul>
  );
}

function PasteBox({ hint, onLoad }: { hint: string; onLoad: (rows: string[][]) => void }) {
  const [text, setText] = useState("");
  return (
    <details className="text-[11px] text-gray-500">
      <summary className="cursor-pointer select-none">Paste from a spreadsheet</summary>
      <div className="mt-1 space-y-1">
        <p className="leading-normal">{hint}</p>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} aria-label="Pasted rows"
          className={`${INPUT} w-full font-mono`} placeholder="Paste tab, semicolon or comma separated rows" />
        <button type="button" className={SMALL_BTN} disabled={text.trim() === ""}
          onClick={() => { onLoad(parsePasted(text)); setText(""); }}>
          Replace rows
        </button>
      </div>
    </details>
  );
}

/** Warnings, interpretation text and R code, shared by the three result views. */
function ResultFooter({ res }: { res: Warned }) {
  return (
    <>
      {res.warnings && res.warnings.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-[11px] text-amber-800 leading-normal space-y-0.5">
          {res.warnings.map((w) => <p key={w}>{w}</p>)}
        </div>
      )}
      {res.result_text && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3 py-2 text-xs text-indigo-900 leading-relaxed">
          <div className="flex items-start justify-between gap-2">
            <p>{res.result_text}</p>
            <CopyTextButton text={res.result_text} label="Copy text" />
          </div>
        </div>
      )}
      {res.r_code && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-gray-500">R code</span>
            <CopyTextButton text={res.r_code} label="Copy R code" />
          </div>
          <pre className="text-[11px] font-mono bg-gray-50 border border-gray-200 rounded-lg p-2 overflow-auto whitespace-pre">{res.r_code}</pre>
        </div>
      )}
    </>
  );
}

function Stat({ label, value, sub, title }: { label: string; value: string; sub?: string; title?: string }) {
  return (
    <div className="border border-gray-200 rounded-lg px-3 py-2 bg-white" title={title}>
      <div className="text-[10px] text-gray-500">{label}</div>
      <div className="text-sm font-semibold text-gray-800 font-mono">{value}</div>
      {sub && <div className="text-[10px] text-gray-500 font-mono">{sub}</div>}
    </div>
  );
}

const RESULT_PLACEHOLDER = (
  <div className="flex items-center justify-center h-64 border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
    Fill in the table on the left, then run
  </div>
);

/** Run state and the request wrapper common to every mode. */
function useRunner() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return { loading, error, setError, setLoading };
}

/* ───────────────────────── 1. direct standardisation ───────────────────────── */

interface DirectRow { label: string; events: string; pt: string; std: string }

interface DirectStratumOut {
  label: string; events: number; person_time: number; standard_population: number;
  rate: number | null; ci_low: number | null; ci_high: number | null;
  weight: number | null; contribution: number | null; contribution_pct: number | null; excluded: boolean;
}

interface DirectResult extends Warned {
  alpha: number; conf_level: number; multiplier: number; n_strata: number;
  total_events: number; total_person_time: number;
  crude_rate: number; crude_ci_low: number; crude_ci_high: number;
  standardised_rate: number; standardised_ci_low: number; standardised_ci_high: number;
  standardised_se: number; ci_method: string;
  strata: DirectStratumOut[];
}

const DIRECT_SAMPLE: DirectRow[] = [
  { label: "0-39", events: "12", pt: "48000", std: "40000" },
  { label: "40-59", events: "55", pt: "36000", std: "35000" },
  { label: "60+", events: "140", pt: "21000", std: "25000" },
];

function validateDirect(rows: DirectRow[], multiplier: string, alpha: string): string[] {
  const out: string[] = [];
  const used = rows.filter((r) => !(isBlank(r.label) && isBlank(r.events) && isBlank(r.pt) && isBlank(r.std)));
  if (used.length === 0) out.push("Enter at least one stratum.");
  used.forEach((r, i) => {
    const name = r.label.trim() || `Row ${i + 1}`;
    const ev = toNum(r.events), pt = toNum(r.pt), sp = toNum(r.std);
    if (!isCount(ev)) out.push(`${name}: events must be a whole number, 0 or more.`);
    if (!isNonNeg(pt)) out.push(`${name}: person-time must be a number, 0 or more.`);
    if (!isNonNeg(sp)) out.push(`${name}: standard population must be a number, 0 or more.`);
    if (isCount(ev) && ev > 0 && isNonNeg(pt) && pt === 0) out.push(`${name}: events above 0 need person-time above 0.`);
  });
  if (used.length > 0 && !used.some((r) => toNum(r.pt) > 0 && toNum(r.std) > 0)) {
    out.push("At least one stratum needs person-time and standard population above 0.");
  }
  if (!isPositive(toNum(multiplier))) out.push("Multiplier must be a number above 0.");
  const a = toNum(alpha);
  if (!(Number.isFinite(a) && a > 0 && a < 1)) out.push("Alpha must be strictly between 0 and 1.");
  return out;
}

function DirectView() {
  const [rows, setRows] = usePersistedPanelState<DirectRow[]>("epidemiology_direct", "rows", DIRECT_SAMPLE);
  const [multiplier, setMultiplier] = usePersistedPanelState<string>("epidemiology_direct", "multiplier", "100000");
  const [alpha, setAlpha] = usePersistedPanelState<string>("epidemiology_direct", "alpha", "0.05");
  const { loading, error, setError, setLoading } = useRunner();

  const setRow = (i: number, patch: Partial<DirectRow>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  const loadPasted = (parsed: string[][]) => {
    const body = dropHeader(parsed, [1, 2, 3]);
    const next: DirectRow[] = body.map((c) =>
      c.length >= 4 ? { label: c[0], events: c[1], pt: c[2], std: c[3] }
        : { label: "", events: c[0] ?? "", pt: c[1] ?? "", std: c[2] ?? "" });
    if (next.length > 0) setRows(next);
  };

  const problems = validateDirect(rows, multiplier, alpha);
  const payload = {
    strata: rows
      .filter((r) => !(isBlank(r.label) && isBlank(r.events) && isBlank(r.pt) && isBlank(r.std)))
      .map((r) => ({
        label: r.label.trim() === "" ? undefined : r.label.trim(),
        events: toNum(r.events), person_time: toNum(r.pt), standard_population: toNum(r.std),
      })),
    multiplier: toNum(multiplier),
    alpha: toNum(alpha),
  };
  const { result, setResult, stale, staleReasons: staleWhy } =
    useStampedResult<DirectResult>("epidemiology_direct", payload, { dependsOnData: false });

  const run = async () => {
    if (problems.length > 0) return;
    setLoading(true); setError(null); setResult(null);
    try { setResult((await runDirectStandardisation(payload)).data); }
    catch (e: unknown) { setError(errorMessage(e)); }
    finally { setLoading(false); }
  };

  const tableRows = (result?.strata ?? []).map((s): Cell[] => [
    s.label, s.events, s.person_time, s.standard_population,
    fx(s.rate), s.excluded ? "excluded" : ciText(s.ci_low, s.ci_high, fx),
    fx(s.weight, 3), fx(s.contribution_pct, 1),
  ]);
  const tableHeaders = ["Stratum", "Events", "Person-time", "Standard pop.", "Rate", "Exact CI", "Weight", "Contribution %"];

  return (
    <div className="flex gap-4">
      <div className="w-[460px] flex-shrink-0 space-y-3">
        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
            Direct standardisation
            <Tip wide text="Directly standardised rate: the weighted sum of stratum rates, with weights proportional to a standard population. The CI is the gamma interval of Fay and Feuer (1997), as in epitools::ageadjust.direct. The crude rate carries an exact Poisson CI." />
          </h3>
          <div className="grid grid-cols-[1fr_4.5rem_5.5rem_5.5rem_1rem] gap-1 text-[10px] text-gray-500">
            <span>Stratum</span><span>Events</span><span>Person-time</span><span>Standard pop.</span><span />
          </div>
          {rows.map((r, i) => (
            <div key={i} className="grid grid-cols-[1fr_4.5rem_5.5rem_5.5rem_1rem] gap-1 items-center">
              <input aria-label={`Stratum ${i + 1} label`} value={r.label} onChange={(e) => setRow(i, { label: e.target.value })} className={INPUT} placeholder="e.g. 40-59" />
              <input aria-label={`Stratum ${i + 1} events`} value={r.events} onChange={(e) => setRow(i, { events: e.target.value })} className={INPUT} inputMode="numeric" />
              <input aria-label={`Stratum ${i + 1} person-time`} value={r.pt} onChange={(e) => setRow(i, { pt: e.target.value })} className={INPUT} inputMode="decimal" />
              <input aria-label={`Stratum ${i + 1} standard population`} value={r.std} onChange={(e) => setRow(i, { std: e.target.value })} className={INPUT} inputMode="decimal" />
              <button type="button" aria-label={`Remove stratum ${i + 1}`} disabled={rows.length <= 1}
                onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}
                className="text-gray-300 hover:text-red-500 text-xs disabled:opacity-30">✕</button>
            </div>
          ))}
          <div className="flex gap-2">
            <button type="button" className={SMALL_BTN} onClick={() => setRows((rs) => [...rs, { label: "", events: "", pt: "", std: "" }])}>+ Stratum</button>
            <button type="button" className={SMALL_BTN} onClick={() => setRows(DIRECT_SAMPLE)}>Sample</button>
          </div>
          <PasteBox hint="Columns: stratum label, events, person-time, standard population (a header row is skipped). Three numeric columns are read as events, person-time, standard population with no label." onLoad={loadPasted} />
        </div>
        <div className="panel grid grid-cols-2 gap-2">
          <Field label="Rate multiplier" tip="Rates are reported per this many person-time units (for example 100000 for a rate per 100,000 person-years).">
            <input aria-label="Rate multiplier" value={multiplier} onChange={(e) => setMultiplier(e.target.value)} className={INPUT} />
          </Field>
          <Field label="Alpha (1 - CI level)">
            <input aria-label="Alpha" value={alpha} onChange={(e) => setAlpha(e.target.value)} className={INPUT} />
          </Field>
        </div>
        <Problems items={problems} />
        <button type="button" onClick={run} disabled={loading || problems.length > 0}
          className="w-full px-4 py-1.5 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
          {loading ? "Running…" : "Run direct standardisation"}
        </button>
        {error && <p className="text-xs text-red-500" role="alert">{error}</p>}
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {!result && !error && RESULT_PLACEHOLDER}
        {result && stale && <StaleResultNotice reasons={staleWhy} onRecompute={run} busy={loading} what="This standardisation result" />}
        {result && (
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-800">
                  Direct standardisation <span className="text-gray-400 font-normal">· per {result.multiplier} person-time · {fx(result.conf_level * 100, 0)}% CI</span>
                </h4>
                <ResultExporter title="Direct standardisation" headers={tableHeaders} rows={tableRows} />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Crude rate" value={fx(result.crude_rate)} sub={`[${fx(result.crude_ci_low)}, ${fx(result.crude_ci_high)}] exact Poisson`} />
                <Stat label="Standardised rate" value={fx(result.standardised_rate)} sub={`[${fx(result.standardised_ci_low)}, ${fx(result.standardised_ci_high)}] Fay-Feuer`} />
              </div>
              <p className="text-[10px] text-gray-500">
                {result.total_events} events over {result.total_person_time} person-time in {result.n_strata} strata; SE of the standardised rate {fx(result.standardised_se, 3)}.
              </p>
              <div className="overflow-auto rounded-lg border border-gray-200">
                <table className="w-full text-[11px] border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                      {tableHeaders.map((h, i) => <th key={h} className={`${TH} ${i === 0 ? "text-left" : "text-right"}`}>{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {tableRows.map((r, i) => (
                      <tr key={i} className={`border-b border-gray-100 ${result.strata[i].excluded ? "text-gray-400" : ""}`}>
                        {r.map((c, j) => <td key={j} className={`${TD} ${j === 0 ? "text-left" : "text-right"}`}>{c}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <ResultFooter res={result} />
          </StaleGuard>
        )}
      </div>
    </div>
  );
}

/* ───────────────────────── 2. indirect standardisation (SMR) ───────────────────────── */

type RefMode = "rate" | "counts";
interface IndirectRow { label: string; obs: string; pt: string; refRate: string; refEvents: string; refPt: string }

interface IndirectStratumOut {
  label: string; observed: number; person_time: number; reference_rate: number; expected: number; ratio: number | null;
}

interface IndirectResult extends Warned {
  alpha: number; conf_level: number; reference_multiplier: number; n_strata: number;
  observed: number; expected: number; smr: number; smr_x100: number;
  byar_ci_low: number; byar_ci_high: number; byar_ci_low_x100: number; byar_ci_high_x100: number;
  exact_ci_low: number; exact_ci_high: number; exact_ci_low_x100: number; exact_ci_high_x100: number;
  p_value: number; p_method: string;
  strata: IndirectStratumOut[];
}

const BLANK_IND: IndirectRow = { label: "", obs: "", pt: "", refRate: "", refEvents: "", refPt: "" };
const INDIRECT_SAMPLE: IndirectRow[] = [
  { label: "0-39", obs: "8", pt: "30000", refRate: "30", refEvents: "", refPt: "" },
  { label: "40-59", obs: "40", pt: "24000", refRate: "130", refEvents: "", refPt: "" },
  { label: "60+", obs: "95", pt: "9000", refRate: "700", refEvents: "", refPt: "" },
];

const indBlank = (r: IndirectRow): boolean =>
  [r.label, r.obs, r.pt, r.refRate, r.refEvents, r.refPt].every(isBlank);

function validateIndirect(rows: IndirectRow[], refMode: RefMode, refMult: string, alpha: string): string[] {
  const out: string[] = [];
  const used = rows.filter((r) => !indBlank(r));
  if (used.length === 0) out.push("Enter at least one stratum.");
  used.forEach((r, i) => {
    const name = r.label.trim() || `Row ${i + 1}`;
    const obs = toNum(r.obs), pt = toNum(r.pt);
    if (!isCount(obs)) out.push(`${name}: observed events must be a whole number, 0 or more.`);
    if (!isNonNeg(pt)) out.push(`${name}: person-time must be a number, 0 or more.`);
    if (isCount(obs) && obs > 0 && isNonNeg(pt) && pt === 0) out.push(`${name}: observed events above 0 need person-time above 0.`);
    if (refMode === "rate") {
      if (!isNonNeg(toNum(r.refRate))) out.push(`${name}: reference rate must be a number, 0 or more.`);
    } else {
      if (!isCount(toNum(r.refEvents))) out.push(`${name}: reference events must be a whole number, 0 or more.`);
      if (!isPositive(toNum(r.refPt))) out.push(`${name}: reference person-time must be a number above 0.`);
    }
  });
  if (refMode === "rate" && !isPositive(toNum(refMult))) out.push("Reference multiplier must be a number above 0.");
  const a = toNum(alpha);
  if (!(Number.isFinite(a) && a > 0 && a < 1)) out.push("Alpha must be strictly between 0 and 1.");
  return out;
}

function IndirectView() {
  const [rows, setRows] = usePersistedPanelState<IndirectRow[]>("epidemiology_indirect", "rows", INDIRECT_SAMPLE);
  const [refMode, setRefMode] = usePersistedPanelState<RefMode>("epidemiology_indirect", "refMode", "rate");
  const [refMult, setRefMult] = usePersistedPanelState<string>("epidemiology_indirect", "refMult", "100000");
  const [alpha, setAlpha] = usePersistedPanelState<string>("epidemiology_indirect", "alpha", "0.05");
  const { loading, error, setError, setLoading } = useRunner();

  const setRow = (i: number, patch: Partial<IndirectRow>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  const loadPasted = (parsed: string[][]) => {
    const body = dropHeader(parsed, refMode === "rate" ? [1, 2, 3] : [1, 2, 3, 4]);
    const next: IndirectRow[] = body.map((c) =>
      refMode === "rate"
        ? { ...BLANK_IND, label: c[0] ?? "", obs: c[1] ?? "", pt: c[2] ?? "", refRate: c[3] ?? "" }
        : { ...BLANK_IND, label: c[0] ?? "", obs: c[1] ?? "", pt: c[2] ?? "", refEvents: c[3] ?? "", refPt: c[4] ?? "" });
    if (next.length > 0) setRows(next);
  };

  const problems = validateIndirect(rows, refMode, refMult, alpha);
  // The reference multiplier only scales a directly entered rate: with
  // reference counts the rate is events / person-time, so it is sent as 1.
  const payload = {
    strata: rows.filter((r) => !indBlank(r)).map((r) => ({
      label: r.label.trim() === "" ? undefined : r.label.trim(),
      observed: toNum(r.obs), person_time: toNum(r.pt),
      ...(refMode === "rate"
        ? { reference_rate: toNum(r.refRate) }
        : { reference_events: toNum(r.refEvents), reference_person_time: toNum(r.refPt) }),
    })),
    reference_multiplier: refMode === "rate" ? toNum(refMult) : 1,
    alpha: toNum(alpha),
  };
  const { result, setResult, stale, staleReasons: staleWhy } =
    useStampedResult<IndirectResult>("epidemiology_indirect", payload, { dependsOnData: false });

  const run = async () => {
    if (problems.length > 0) return;
    setLoading(true); setError(null); setResult(null);
    try { setResult((await runIndirectStandardisation(payload)).data); }
    catch (e: unknown) { setError(errorMessage(e)); }
    finally { setLoading(false); }
  };

  const tableHeaders = ["Stratum", "Observed", "Person-time", "Reference rate", "Expected", "O / E"];
  const tableRows = (result?.strata ?? []).map((s): Cell[] => [
    s.label, s.observed, s.person_time, sig(s.reference_rate), fx(s.expected), fx(s.ratio),
  ]);
  const colHeads = refMode === "rate"
    ? ["Stratum", "Observed", "Person-time", "Ref. rate"]
    : ["Stratum", "Observed", "Person-time", "Ref. events", "Ref. person-time"];
  const gridCols = refMode === "rate"
    ? "grid-cols-[1fr_4rem_5rem_5rem_1rem]"
    : "grid-cols-[1fr_3.5rem_4.5rem_4rem_4.5rem_1rem]";

  return (
    <div className="flex gap-4">
      <div className="w-[480px] flex-shrink-0 space-y-3">
        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
            Indirect standardisation (SMR)
            <Tip wide text="Expected events E = sum of study person-time times the reference rate in each stratum; SMR = observed / expected. Byar's approximation and the exact Poisson CI are both reported, with the exact two-sided Poisson test of SMR = 1." />
          </h3>
          <div className="flex items-center gap-2 text-[10px] text-gray-500">
            <span>Reference rates as</span>
            <Segmented value={refMode} onChange={setRefMode} options={[["rate", "Rate"], ["counts", "Events + person-time"]] as const} />
          </div>
          <div className={`grid ${gridCols} gap-1 text-[10px] text-gray-500`}>
            {colHeads.map((h) => <span key={h}>{h}</span>)}<span />
          </div>
          {rows.map((r, i) => (
            <div key={i} className={`grid ${gridCols} gap-1 items-center`}>
              <input aria-label={`Stratum ${i + 1} label`} value={r.label} onChange={(e) => setRow(i, { label: e.target.value })} className={INPUT} placeholder="e.g. 40-59" />
              <input aria-label={`Stratum ${i + 1} observed`} value={r.obs} onChange={(e) => setRow(i, { obs: e.target.value })} className={INPUT} inputMode="numeric" />
              <input aria-label={`Stratum ${i + 1} person-time`} value={r.pt} onChange={(e) => setRow(i, { pt: e.target.value })} className={INPUT} inputMode="decimal" />
              {refMode === "rate" ? (
                <input aria-label={`Stratum ${i + 1} reference rate`} value={r.refRate} onChange={(e) => setRow(i, { refRate: e.target.value })} className={INPUT} inputMode="decimal" />
              ) : (
                <>
                  <input aria-label={`Stratum ${i + 1} reference events`} value={r.refEvents} onChange={(e) => setRow(i, { refEvents: e.target.value })} className={INPUT} inputMode="numeric" />
                  <input aria-label={`Stratum ${i + 1} reference person-time`} value={r.refPt} onChange={(e) => setRow(i, { refPt: e.target.value })} className={INPUT} inputMode="decimal" />
                </>
              )}
              <button type="button" aria-label={`Remove stratum ${i + 1}`} disabled={rows.length <= 1}
                onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}
                className="text-gray-300 hover:text-red-500 text-xs disabled:opacity-30">✕</button>
            </div>
          ))}
          <div className="flex gap-2">
            <button type="button" className={SMALL_BTN} onClick={() => setRows((rs) => [...rs, { ...BLANK_IND }])}>+ Stratum</button>
            <button type="button" className={SMALL_BTN} onClick={() => { setRows(INDIRECT_SAMPLE); setRefMode("rate"); }}>Sample</button>
          </div>
          <PasteBox
            hint={refMode === "rate"
              ? "Columns: stratum label, observed events, person-time, reference rate (a header row is skipped)."
              : "Columns: stratum label, observed events, person-time, reference events, reference person-time (a header row is skipped)."}
            onLoad={loadPasted} />
        </div>
        <div className="panel space-y-2">
          <div className="grid grid-cols-2 gap-2">
            {refMode === "rate" && (
              <Field label="Reference multiplier">
                <input aria-label="Reference multiplier" value={refMult} onChange={(e) => setRefMult(e.target.value)} className={INPUT} />
              </Field>
            )}
            <Field label="Alpha (1 - CI level)">
              <input aria-label="Alpha" value={alpha} onChange={(e) => setAlpha(e.target.value)} className={INPUT} />
            </Field>
          </div>
          <p className="text-[10px] text-gray-500 leading-normal">
            {refMode === "rate"
              ? "Reference multiplier: the reference rates are per this many person-time units. Use 1 if they are raw rates (events per one person-time unit) and 100000 if they come from a per-100,000 table. Expected events are person-time times rate divided by this multiplier."
              : "Reference rates are derived as reference events divided by reference person-time in each stratum, so no multiplier applies."}
          </p>
        </div>
        <Problems items={problems} />
        <button type="button" onClick={run} disabled={loading || problems.length > 0}
          className="w-full px-4 py-1.5 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
          {loading ? "Running…" : "Run indirect standardisation"}
        </button>
        {error && <p className="text-xs text-red-500" role="alert">{error}</p>}
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {!result && !error && RESULT_PLACEHOLDER}
        {result && stale && <StaleResultNotice reasons={staleWhy} onRecompute={run} busy={loading} what="This SMR result" />}
        {result && (
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-800">
                  Standardised mortality / incidence ratio <span className="text-gray-400 font-normal">· {fx(result.conf_level * 100, 0)}% CI</span>
                </h4>
                <ResultExporter title="SMR by stratum" headers={tableHeaders} rows={tableRows} />
              </div>
              <div className="grid grid-cols-2 xl:grid-cols-4 gap-2">
                <Stat label="Observed (O)" value={String(result.observed)} />
                <Stat label="Expected (E)" value={fx(result.expected)} />
                <Stat label="SMR" value={fx(result.smr)} sub={`x100 = ${fx(result.smr_x100, 1)}`} />
                <Stat label="Exact p (SMR = 1)" value={fmtP(result.p_value)} title={pCellTitle(result.p_value)} />
              </div>
              <div className="overflow-auto rounded-lg border border-gray-200">
                <table className="w-full text-[11px] border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                      <th className={`${TH} text-left`}>Interval</th>
                      <th className={`${TH} text-right`}>SMR CI</th>
                      <th className={`${TH} text-right`}>x100</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100">
                      <td className={`${TD} text-left`}>Byar</td>
                      <td className={`${TD} text-right`}>{fx(result.byar_ci_low)} to {fx(result.byar_ci_high)}</td>
                      <td className={`${TD} text-right`}>{fx(result.byar_ci_low_x100, 1)} to {fx(result.byar_ci_high_x100, 1)}</td>
                    </tr>
                    <tr className="border-b border-gray-100">
                      <td className={`${TD} text-left`}>Exact Poisson</td>
                      <td className={`${TD} text-right`}>{fx(result.exact_ci_low)} to {fx(result.exact_ci_high)}</td>
                      <td className={`${TD} text-right`}>{fx(result.exact_ci_low_x100, 1)} to {fx(result.exact_ci_high_x100, 1)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="overflow-auto rounded-lg border border-gray-200">
                <table className="w-full text-[11px] border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                      {tableHeaders.map((h, i) => <th key={h} className={`${TH} ${i === 0 ? "text-left" : "text-right"}`}>{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {tableRows.map((r, i) => (
                      <tr key={i} className="border-b border-gray-100">
                        {r.map((c, j) => <td key={j} className={`${TD} ${j === 0 ? "text-left" : "text-right"}`}>{c}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-gray-500">Reference rates are shown per {result.reference_multiplier} person-time units. {result.p_method}.</p>
            </div>
            <ResultFooter res={result} />
          </StaleGuard>
        )}
      </div>
    </div>
  );
}

/* ───────────────────────── 3. rate ratio ───────────────────────── */

interface RRGroup { events: number; person_time: number; rate: number; ci_low: number; ci_high: number }

interface RateRatioResult extends Warned {
  alpha: number; conf_level: number;
  group1: RRGroup; group2: RRGroup;
  rate_ratio: number | null; rr_ci_low: number | null; rr_ci_high: number | null; rr_se_log: number | null;
  rr_ci_method: string; rr_exact_ci_low: number | null; rr_exact_ci_high: number | null;
  rate_difference: number; rd_ci_low: number; rd_ci_high: number; rd_se: number;
  p_value: number | null; p_value_midp: number | null; p_value_wald: number | null; p_method: string;
}

interface RRInputs { e1: string; t1: string; e2: string; t2: string }
const RR_SAMPLE: RRInputs = { e1: "30", t1: "12000", e2: "15", t2: "11000" };

function validateRR(v: RRInputs, alpha: string): string[] {
  const out: string[] = [];
  if (!isCount(toNum(v.e1))) out.push("Group 1: events must be a whole number, 0 or more.");
  if (!isPositive(toNum(v.t1))) out.push("Group 1: person-time must be a number above 0.");
  if (!isCount(toNum(v.e2))) out.push("Group 2: events must be a whole number, 0 or more.");
  if (!isPositive(toNum(v.t2))) out.push("Group 2: person-time must be a number above 0.");
  const a = toNum(alpha);
  if (!(Number.isFinite(a) && a > 0 && a < 1)) out.push("Alpha must be strictly between 0 and 1.");
  return out;
}

function RateRatioView() {
  const [v, setV] = usePersistedPanelState<RRInputs>("epidemiology_rr", "inputs", RR_SAMPLE);
  const [alpha, setAlpha] = usePersistedPanelState<string>("epidemiology_rr", "alpha", "0.05");
  const { loading, error, setError, setLoading } = useRunner();

  const problems = validateRR(v, alpha);
  const payload = {
    events1: toNum(v.e1), person_time1: toNum(v.t1),
    events2: toNum(v.e2), person_time2: toNum(v.t2),
    alpha: toNum(alpha),
  };
  const { result, setResult, stale, staleReasons: staleWhy } =
    useStampedResult<RateRatioResult>("epidemiology_rr", payload, { dependsOnData: false });

  const run = async () => {
    if (problems.length > 0) return;
    setLoading(true); setError(null); setResult(null);
    try { setResult((await runRateRatio(payload)).data); }
    catch (e: unknown) { setError(errorMessage(e)); }
    finally { setLoading(false); }
  };

  const level = result ? `${fx(result.conf_level * 100, 0)}%` : "";
  const exportRows: Cell[][] = result ? [
    ["Group 1 rate", sig(result.group1.rate), sig(result.group1.ci_low), sig(result.group1.ci_high)],
    ["Group 2 rate", sig(result.group2.rate), sig(result.group2.ci_low), sig(result.group2.ci_high)],
    ["Rate ratio (log method)", fx(result.rate_ratio, 3), fx(result.rr_ci_low, 3), fx(result.rr_ci_high, 3)],
    ["Rate ratio (exact conditional)", fx(result.rate_ratio, 3), fx(result.rr_exact_ci_low, 3), result.rr_exact_ci_high == null ? "∞" : fx(result.rr_exact_ci_high, 3)],
    ["Rate difference", sig(result.rate_difference), sig(result.rd_ci_low), sig(result.rd_ci_high)],
  ] : [];

  const groupInput = (label: string, ek: "e1" | "e2", tk: "t1" | "t2") => (
    <div className="grid grid-cols-[4.5rem_1fr_1fr] gap-2 items-end">
      <span className="text-xs font-semibold text-gray-700 pb-1.5">{label}</span>
      <Field label="Events">
        <input aria-label={`${label} events`} value={v[ek]} onChange={(e) => setV((p) => ({ ...p, [ek]: e.target.value }))} className={INPUT} inputMode="numeric" />
      </Field>
      <Field label="Person-time">
        <input aria-label={`${label} person-time`} value={v[tk]} onChange={(e) => setV((p) => ({ ...p, [tk]: e.target.value }))} className={INPUT} inputMode="decimal" />
      </Field>
    </div>
  );

  return (
    <div className="flex gap-4">
      <div className="w-[420px] flex-shrink-0 space-y-3">
        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
            Rate ratio
            <Tip wide text="Compares incidence rates of two groups (group 1 relative to group 2). Rates carry exact Poisson CIs; the rate ratio CI uses the log method, with the exact conditional (binomial) CI and mid-p alongside, which also covers a zero count." />
          </h3>
          {groupInput("Group 1", "e1", "t1")}
          {groupInput("Group 2", "e2", "t2")}
          <Field label="Alpha (1 - CI level)">
            <input aria-label="Alpha" value={alpha} onChange={(e) => setAlpha(e.target.value)} className={`${INPUT} w-24`} />
          </Field>
          <button type="button" className={SMALL_BTN} onClick={() => setV(RR_SAMPLE)}>Sample</button>
        </div>
        <Problems items={problems} />
        <button type="button" onClick={run} disabled={loading || problems.length > 0}
          className="w-full px-4 py-1.5 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
          {loading ? "Running…" : "Run rate ratio"}
        </button>
        {error && <p className="text-xs text-red-500" role="alert">{error}</p>}
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {!result && !error && RESULT_PLACEHOLDER}
        {result && stale && <StaleResultNotice reasons={staleWhy} onRecompute={run} busy={loading} what="This rate ratio result" />}
        {result && (
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-800">
                  Rate ratio and rate difference <span className="text-gray-400 font-normal">· {level} CI</span>
                </h4>
                <ResultExporter title="Rate ratio" headers={["Measure", "Estimate", "CI low", "CI high"]} rows={exportRows} />
              </div>
              <div className="grid grid-cols-2 xl:grid-cols-4 gap-2">
                <Stat label="Group 1 rate" value={sig(result.group1.rate)} sub={`[${sig(result.group1.ci_low)}, ${sig(result.group1.ci_high)}]`} />
                <Stat label="Group 2 rate" value={sig(result.group2.rate)} sub={`[${sig(result.group2.ci_low)}, ${sig(result.group2.ci_high)}]`} />
                <Stat label="Rate ratio" value={fx(result.rate_ratio, 3)} sub={result.rr_ci_low != null ? `[${fx(result.rr_ci_low, 3)}, ${fx(result.rr_ci_high, 3)}] log` : "log CI undefined"} />
                <Stat label="Rate difference" value={sig(result.rate_difference)} sub={`[${sig(result.rd_ci_low)}, ${sig(result.rd_ci_high)}]`} />
              </div>
              <div className="overflow-auto rounded-lg border border-gray-200">
                <table className="w-full text-[11px] border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                      <th className={`${TH} text-left`}>Rate ratio interval</th>
                      <th className={`${TH} text-right`}>{level} CI</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100">
                      <td className={`${TD} text-left`}>Log method</td>
                      <td className={`${TD} text-right`}>{result.rr_ci_low == null ? "undefined (zero count)" : ciText(result.rr_ci_low, result.rr_ci_high, (x) => fx(x, 3))}</td>
                    </tr>
                    <tr className="border-b border-gray-100">
                      <td className={`${TD} text-left`}>Exact conditional</td>
                      <td className={`${TD} text-right`}>{result.rr_exact_ci_low == null ? "NA" : ciText(result.rr_exact_ci_low, result.rr_exact_ci_high, (x) => fx(x, 3))}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Stat label="Exact p" value={fmtP(result.p_value)} title={pCellTitle(result.p_value)} />
                <Stat label="Mid-p" value={fmtP(result.p_value_midp)} title={pCellTitle(result.p_value_midp)} />
                <Stat label="Wald p" value={fmtP(result.p_value_wald)} title={pCellTitle(result.p_value_wald)} />
              </div>
              <p className="text-[10px] text-gray-500">{result.p_method}.</p>
            </div>
            <ResultFooter res={result} />
          </StaleGuard>
        )}
      </div>
    </div>
  );
}

/* ───────────────────────── panel ───────────────────────── */

export default function EpidemiologyPanel() {
  const [mode, setMode] = usePersistedPanelState<Mode>("epidemiology", "mode", "direct");
  return (
    <div className="space-y-3">
      <Segmented value={mode} onChange={setMode}
        options={[["direct", "Direct standardisation"], ["indirect", "Indirect standardisation (SMR)"], ["rate_ratio", "Rate ratio"]] as const} />
      {mode === "direct" && <DirectView />}
      {mode === "indirect" && <IndirectView />}
      {mode === "rate_ratio" && <RateRatioView />}
    </div>
  );
}
