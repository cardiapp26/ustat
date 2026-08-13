import type { ReactNode } from "react";
import type { ColMeta } from "../../store";
import { StyledTableExporter } from "../StyledTableExporter";
import type { StyledTableData } from "../../lib/styledTable";
import { fmtPubP } from "../../lib/format";
import { labelFor } from "../../lib/valueLabels";

/** One Cox term's HR statistics for a single model column. */
export interface HRStat {
  hr: number | null;
  hr_ci_low: number | null;
  hr_ci_high: number | null;
  p: number | null;
}

export interface HRRow {
  term: string;
  predictor: string;
  kind: "numeric" | "category";
  category: string | null;
  reference: string | null;
  unadjusted: HRStat | null;
  parsimonious: HRStat | null;
  adjusted: HRStat | null;
}

interface CoxHRTableProps {
  rows: HRRow[];
  columns: ColMeta[];
  n: number;
  nEvents: number;
  nPars: number;
  nEventsPars: number;
  durationCol: string;
  eventCol: string;
}

/** "1.43 (0.69–2.96), p=0.34" — or "—" when the term is absent from a model. */
const fmtCell = (s: HRStat | null): string => {
  if (!s || s.hr == null || !isFinite(s.hr)) return "—";
  const ci =
    s.hr_ci_low != null && s.hr_ci_high != null && isFinite(s.hr_ci_low) && isFinite(s.hr_ci_high)
      ? ` (${s.hr_ci_low.toFixed(2)}–${s.hr_ci_high.toFixed(2)})`
      : "";
  const hasP = s.p != null && isFinite(s.p);
  return `${s.hr.toFixed(2)}${ci}${hasP ? `, ${fmtPubP(s.p)}` : ""}`;
};

/** On-screen-only rendering of `fmtCell`'s output with the "p" italicised.
 *  `fmtCell` itself stays a plain string so the .docx/.html export keeps
 *  working — this just re-splits the trailing ", p=..." for display. */
const renderCell = (txt: string): ReactNode => {
  const m = txt.match(/^(.*, )p(=.+)$/);
  return m ? <>{m[1]}<i>p</i>{m[2]}</> : txt;
};

/** Build a publication row label from column metadata + value labels.
 *  numeric  → "Age (per 1 year)"  (units optional)
 *  category → "LDL-C: <100 vs >130" using value labels when present. */
function rowLabel(row: HRRow, byName: Record<string, ColMeta>): string {
  const col = byName[row.predictor];
  const name = col?.display_name || col?.label || row.predictor;
  if (row.kind === "numeric") {
    return col?.units ? `${name} (per 1 ${col.units})` : name;
  }
  const vl = col?.value_labels ?? {};
  const cat = row.category != null ? labelFor(vl, row.category, row.category) : "";
  const ref = row.reference != null ? labelFor(vl, row.reference, row.reference) : "";
  return `${name}: ${cat} vs ${ref}`;
}

const COLS: Array<{ key: keyof Pick<HRRow, "unadjusted" | "parsimonious" | "adjusted">; head: string }> = [
  { key: "unadjusted", head: "Univariable HR (95% CI), p" },
  { key: "parsimonious", head: "Parsimonious HR (95% CI), p" },
  { key: "adjusted", head: "Fully adjusted HR (95% CI), p" },
];

export default function CoxHRTable({
  rows,
  columns,
  n,
  nEvents,
  nPars,
  nEventsPars,
  durationCol,
  eventCol,
}: CoxHRTableProps) {
  const byName = Object.fromEntries(columns.map((c) => [c.name, c]));

  const caption =
    `Cox proportional-hazards regression for ${eventCol} over ${durationCol}. ` +
    `Fully adjusted model: n=${n}, ${nEvents} events` +
    (nPars > 0 ? ` · Parsimonious model: n=${nPars}, ${nEventsPars} events` : "") +
    `. Univariable = each predictor fitted alone. Parsimonious = the selected subset fitted ` +
    `together. Fully adjusted = all predictors fitted together. HR = hazard ratio; ` +
    `CI = confidence interval. A blank (—) cell means the predictor was not in that model.`;

  const exportData = (): StyledTableData => ({
    title: "Cox proportional-hazards regression (univariable, parsimonious, fully adjusted)",
    caption,
    columns: ["Variable", ...COLS.map((c) => c.head)],
    rows: rows.map((r) => [rowLabel(r, byName), ...COLS.map((c) => fmtCell(r[c.key]))]),
    filename: "cox_hr_table",
  });

  return (
    <div>
      <div className="flex items-center justify-end mb-2">
        <StyledTableExporter data={exportData} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b-2 border-gray-300 text-left text-gray-600">
              <th className="px-2 py-2 font-semibold align-bottom">Variable</th>
              {COLS.map((c) => (
                <th key={c.key} className="px-2 py-2 font-semibold align-bottom">
                  {c.head.endsWith(", p") ? <>{c.head.slice(0, -1)}<i>p</i></> : c.head}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.term} className="border-b border-gray-100">
                <td className="px-2 py-2 text-gray-800">{rowLabel(r, byName)}</td>
                {COLS.map((c) => {
                  const txt = fmtCell(r[c.key]);
                  return (
                    <td
                      key={c.key}
                      className={`px-2 py-2 tabular-nums ${txt === "—" ? "text-gray-300" : "text-gray-700"}`}
                    >
                      {renderCell(txt)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-gray-400 mt-3 leading-snug">
        Cox proportional-hazards regression for <em>{eventCol}</em> over <em>{durationCol}</em>.
        Fully adjusted model: <i>n</i>={n}, {nEvents} events
        {nPars > 0 && <> · Parsimonious model: <i>n</i>={nPars}, {nEventsPars} events</>}.
        Univariable = each predictor fitted alone. Parsimonious = the selected subset fitted
        together. Fully adjusted = all predictors fitted together. Reference categories shown
        in each label. HR = hazard ratio; CI = confidence interval. A blank (—) cell means the
        predictor was not in that model.
      </p>
    </div>
  );
}
