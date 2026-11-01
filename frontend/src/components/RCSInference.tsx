import { fmtP } from "../lib/format";

export interface KnotComparisonRow {
  n_knots: number;
  aic: number;
  delta_aic: number;
  best: boolean;
  chosen: boolean;
}

export interface RCSTestFields {
  overall_p?: number | null;
  overall_df?: number | null;
  overall_wald?: number | null;
  overall_f?: number | null;
  nonlinearity_p?: number | null;
  nonlinearity_df?: number | string | null;
  nonlinearity_wald?: number | null;
  nonlinearity_f?: number | null;
  test_statistic?: "F" | "chi2" | null;
  tests_include_interaction?: boolean;
  knot_comparison?: KnotComparisonRow[];
}

interface TestChipProps {
  label: string;
  p: number | null | undefined;
  stat: string | null;
  title: string;
}

function TestChip({ label, p, stat, title }: TestChipProps) {
  if (p == null) return null;
  const tone = p < 0.05
    ? "bg-indigo-50 text-indigo-700 border-indigo-200"
    : "bg-gray-50 text-gray-600 border-gray-200";
  return (
    <span className={`rounded px-1.5 py-0.5 border ${tone}`} title={title}>
      {label}{stat ? <>: {stat},</> : null} <i>p</i> = {fmtP(p)}
    </span>
  );
}

function statText(kind: "F" | "chi2" | null | undefined, df: number | string | null | undefined,
  wald: number | null | undefined, f: number | null | undefined): string | null {
  if (df == null) return null;
  if (kind === "F" && f != null) return `F(${df}) = ${f.toFixed(2)}`;
  if (wald != null) return `χ²(${df}) = ${wald.toFixed(2)}`;
  return null;
}

/**
 * Harrell's two tests for a spline term (rms::anova): overall association
 * (all k−1 spline columns) and non-linearity (the k−2 cubic columns).
 */
export function RCSTestChips({ result, adjusted }: { result: RCSTestFields; adjusted: boolean }) {
  const kind = result.test_statistic;
  const prefix = adjusted ? "adjusted " : "";
  const ixNote = result.tests_include_interaction ? " Interaction columns involving the predictor are included." : "";
  return (
    <>
      <TestChip
        label={`${prefix}overall association`}
        p={result.overall_p}
        stat={statText(kind, result.overall_df, result.overall_wald, result.overall_f)}
        title={`Joint test that every spline coefficient is zero (H0: no association).${ixNote}`}
      />
      <TestChip
        label={`${prefix}non-linearity`}
        p={result.nonlinearity_p}
        stat={statText(kind, result.nonlinearity_df, result.nonlinearity_wald, result.nonlinearity_f)}
        title={`Joint test that the non-linear (cubic) coefficients are zero (H0: linear relation).${ixNote}`}
      />
    </>
  );
}

/** AIC for Harrell-default knot counts; lowest AIC is the data-based choice (RMS §2.4.6). */
export function KnotComparisonTable({ rows, hasInteraction }: { rows: KnotComparisonRow[]; hasInteraction: boolean }) {
  if (rows.length === 0) return null;
  return (
    <div className="rounded border border-gray-200 text-[11px]">
      <div className="px-2 py-1 border-b border-gray-200 bg-gray-50 text-[10px] font-semibold text-gray-600 uppercase tracking-wider">
        Knot count by AIC (Harrell default positions)
      </div>
      <table className="w-full">
        <thead>
          <tr className="text-gray-500">
            <th className="text-left px-2 py-0.5 font-medium">Knots</th>
            <th className="text-right px-2 py-0.5 font-medium">AIC</th>
            <th className="text-right px-2 py-0.5 font-medium">ΔAIC</th>
            <th className="px-2 py-0.5" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.n_knots} className={r.chosen ? "bg-indigo-50/60 font-semibold" : ""}>
              <td className="px-2 py-0.5">{r.n_knots}{r.chosen ? " (fitted)" : ""}</td>
              <td className="text-right px-2 py-0.5 font-mono">{r.aic.toFixed(1)}</td>
              <td className="text-right px-2 py-0.5 font-mono">{r.delta_aic.toFixed(1)}</td>
              <td className="px-2 py-0.5 text-emerald-700">{r.best ? "lowest" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="px-2 py-1 border-t border-gray-200 text-[10px] text-gray-500">
        ΔAIC &lt; 2 means the counts fit about equally well; prefer fewer knots then.
        {hasInteraction ? " Compared on the model without interaction terms." : ""}
      </div>
    </div>
  );
}
