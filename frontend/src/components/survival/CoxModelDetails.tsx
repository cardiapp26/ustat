/** Proportional-hazards test, sample counts and pooling note under the Cox coefficient table. */
import { fmtP } from "../../lib/format";

export interface CoxPHTermRow {
  variable: string;
  test_stat: number | null;
  p: number | null;
}

export interface CoxPHTest {
  global?: { chi2: number | null; df: number | null; p: number | null };
  per_term?: CoxPHTermRow[];
  error?: string;
}

export interface CoxModelDetailsProps {
  phTest?: CoxPHTest | null;
  nAnalyzed?: number;
  nExcluded?: number;
  poolingNote?: string;
  warnings?: unknown[];
}

const PH_ALPHA = 0.05;

function fixed(v: number | null | undefined, digits: number): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "n/a";
}

/**
 * The backend has returned these fields for a while, but the panel only drew
 * the coefficient table: the global Grambsch-Therneau test, the rows lost to
 * missing values and the note that MICE pooled only the coefficients were
 * never on screen.
 */
export function CoxModelDetails({ phTest, nAnalyzed, nExcluded, poolingNote, warnings }: CoxModelDetailsProps) {
  const global = phTest?.global;
  const violated = typeof global?.p === "number" && global.p < PH_ALPHA;
  const textWarnings = (warnings ?? []).filter((w): w is string => typeof w === "string");
  return (
    <div className="space-y-2 text-xs">
      {typeof nAnalyzed === "number" && (
        <p className="text-gray-500">
          {nAnalyzed} rows analysed{typeof nExcluded === "number" && nExcluded > 0 ? `, ${nExcluded} excluded for missing values` : ""}.
        </p>
      )}
      {poolingNote && (
        <p className="rounded border border-indigo-200 bg-indigo-50 px-2 py-1 text-indigo-800">{poolingNote}</p>
      )}
      {phTest?.error && (
        <p className="text-amber-700">Proportional-hazards test could not be computed: {phTest.error}</p>
      )}
      {global && (
        <div className="overflow-auto rounded border border-gray-200">
          <p className="px-2 py-1 font-semibold text-gray-600">
            Proportional-hazards test (scaled Schoenfeld residuals, rank time)
          </p>
          <table className="w-full">
            <thead>
              <tr className="bg-gray-50">
                <th className="px-2 py-1 text-left">Term</th>
                <th className="px-2 py-1 text-right">χ²</th>
                <th className="px-2 py-1 text-right">df</th>
                <th className="px-2 py-1 text-right"><i>p</i></th>
              </tr>
            </thead>
            <tbody>
              {(phTest?.per_term ?? []).map((t) => (
                <tr key={t.variable} className="border-t border-gray-100">
                  <td className="px-2 py-1">{t.variable}</td>
                  <td className="px-2 py-1 text-right font-mono">{fixed(t.test_stat, 3)}</td>
                  <td className="px-2 py-1 text-right font-mono">1</td>
                  <td className="px-2 py-1 text-right font-mono">{fmtP(t.p)}</td>
                </tr>
              ))}
              <tr className="border-t border-gray-200 font-semibold">
                <td className="px-2 py-1">Global</td>
                <td className="px-2 py-1 text-right font-mono">{fixed(global.chi2, 3)}</td>
                <td className="px-2 py-1 text-right font-mono">{global.df ?? "n/a"}</td>
                <td className="px-2 py-1 text-right font-mono">{fmtP(global.p)}</td>
              </tr>
            </tbody>
          </table>
          <p className={`px-2 py-1 ${violated ? "text-amber-700" : "text-gray-500"}`}>
            {violated
              ? "The global test rejects proportional hazards: consider stratifying or a time-varying effect for the flagged terms."
              : "No evidence against proportional hazards at the 5% level."}
          </p>
        </div>
      )}
      {textWarnings.map((w) => (
        <p key={w} className="rounded border border-amber-200 bg-amber-50 px-2 py-1 text-amber-800">{w}</p>
      ))}
    </div>
  );
}
