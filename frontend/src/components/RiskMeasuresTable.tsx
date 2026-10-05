/**
 * "Clinical effect measures" for a 2x2 comparison: the `risk_measures` object
 * returned by /categorical/two_proportions and by /stats/chisquare on a 2x2.
 *
 * Conventions follow backend/services/risk_measures.py: ARD is exposed minus
 * reference; the NNT is reported when the exposed risk is the lower one and the
 * NNH when it is the higher one; when the ARD interval spans zero the number
 * needed has no single interval, so the backend sends the two-part Altman form
 * as `ci_text` (e.g. "NNH 8 to infinity to NNT 25").
 */

export interface NumberNeeded {
  kind?: "NNT" | "NNH" | null;
  value?: number | null;
  low?: number | null;
  high?: number | null;
  ci_spans_zero?: boolean;
  other_kind?: "NNT" | "NNH" | null;
  other_low?: number | null;
  ci_text?: string | null;
}

export interface RiskMeasures {
  event?: string;
  exposed?: string;
  reference?: string;
  ci_level?: number;
  risk_exposed?: number | null;
  risk_reference?: number | null;
  ard?: number | null;
  ard_ci?: Array<number | null>;
  ard_ci_method?: string;
  arr?: number | null;
  rr?: number | null;
  rr_ci?: Array<number | null>;
  rr_ci_method?: string;
  rrr?: number | null;
  rrr_ci?: Array<number | null>;
  nnt_nnh?: NumberNeeded;
}

function num(v: number | null | undefined, digits = 3): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "n/a";
}

function interval(pair: Array<number | null> | undefined): string {
  const lo = pair?.[0];
  const hi = pair?.[1];
  const ok = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);
  return ok(lo) && ok(hi) ? `[${lo.toFixed(3)}, ${hi.toFixed(3)}]` : "n/a";
}

/** 0.95 -> "95%". */
function levelLabel(level: number | undefined): string {
  const pct = typeof level === "number" && level > 0 && level < 1 ? level * 100 : 95;
  return `${Number(pct.toPrecision(6))}%`;
}

export default function RiskMeasuresTable({ measures }: { measures: RiskMeasures }) {
  const nn = measures.nnt_nnh ?? {};
  const level = levelLabel(measures.ci_level);
  const exposed = measures.exposed ?? "exposed";
  const reference = measures.reference ?? "reference";
  const event = measures.event ?? "event";

  const nnLabel =
    nn.kind === "NNT" ? "NNT (number needed to treat)"
    : nn.kind === "NNH" ? "NNH (number needed to harm)"
    : "NNT / NNH";
  const ardIsZero = measures.ard === 0;
  const nnEstimate =
    typeof nn.value === "number" ? String(nn.value)
    : ardIsZero ? "infinite"
    : "n/a";
  const nnInterval = nn.ci_text ?? "n/a";

  return (
    <div className="space-y-1" data-testid="risk-measures">
      <p className="text-xs font-semibold text-gray-600">Clinical effect measures</p>
      <p className="text-[11px] text-gray-500">
        Event: <span className="font-medium text-gray-700">{event}</span>
        {" · "}Exposed group: <span className="font-medium text-gray-700">{exposed}</span>
        {" · "}Reference group: <span className="font-medium text-gray-700">{reference}</span>
      </p>
      <div className="overflow-auto rounded border border-gray-200">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-2 py-1 text-left">Measure</th>
              <th className="px-2 py-1 text-right">Estimate</th>
              <th className="px-2 py-1 text-right">{level} CI</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-gray-100">
              <td className="px-2 py-1">Risk of event, {exposed}</td>
              <td className="px-2 py-1 text-right font-mono">{num(measures.risk_exposed)}</td>
              <td className="px-2 py-1 text-right font-mono text-gray-400">n/a</td>
            </tr>
            <tr className="border-t border-gray-100">
              <td className="px-2 py-1">Risk of event, {reference}</td>
              <td className="px-2 py-1 text-right font-mono">{num(measures.risk_reference)}</td>
              <td className="px-2 py-1 text-right font-mono text-gray-400">n/a</td>
            </tr>
            <tr className="border-t border-gray-100">
              <td className="px-2 py-1">Absolute risk difference (ARD, exposed minus reference)</td>
              <td className="px-2 py-1 text-right font-mono">{num(measures.ard)}</td>
              <td className="px-2 py-1 text-right font-mono">{interval(measures.ard_ci)}</td>
            </tr>
            <tr className="border-t border-gray-100">
              <td className="px-2 py-1">Risk ratio (RR)</td>
              <td className="px-2 py-1 text-right font-mono">{num(measures.rr)}</td>
              <td className="px-2 py-1 text-right font-mono">{interval(measures.rr_ci)}</td>
            </tr>
            <tr className="border-t border-gray-100">
              <td className="px-2 py-1">Relative risk reduction (RRR = 1 - RR)</td>
              <td className="px-2 py-1 text-right font-mono">{num(measures.rrr)}</td>
              <td className="px-2 py-1 text-right font-mono">{interval(measures.rrr_ci)}</td>
            </tr>
            <tr className="border-t border-gray-100">
              <td className="px-2 py-1">{nnLabel}</td>
              <td className="px-2 py-1 text-right font-mono">{nnEstimate}</td>
              <td className="px-2 py-1 text-right font-mono">{nnInterval}</td>
            </tr>
          </tbody>
        </table>
      </div>
      {nn.kind === "NNT" && (
        <p className="text-[11px] text-gray-500">
          The {exposed} group has the lower risk, so the number needed is reported as an NNT.
        </p>
      )}
      {nn.kind === "NNH" && (
        <p className="text-[11px] text-gray-500">
          The {exposed} group has the higher risk, so the number needed is reported as an NNH.
        </p>
      )}
      {ardIsZero && (
        <p className="text-[11px] text-gray-500">The two risks are equal, so the number needed is infinite.</p>
      )}
      {nn.ci_spans_zero && (
        <p className="text-[11px] text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1">
          The ARD interval includes zero, so the {nn.kind ?? "NNT/NNH"} interval is not a single range: it runs
          through infinity to the {nn.other_kind ?? "other"} side (Altman 1998).
        </p>
      )}
      <p className="text-[10px] text-gray-400">
        ARD CI: {measures.ard_ci_method ?? "Newcombe"}; RR CI: {measures.rr_ci_method ?? "Katz log"}. Number needed is rounded up.
      </p>
    </div>
  );
}
