/**
 * Hodges-Lehmann location estimate with its distribution-free CI, as returned in
 * the `hodges_lehmann` block of /stats/mannwhitney, /stats/wilcoxon_onesample and
 * /repeated/wilcoxon_signed_rank. The numeric fields are null (with a `note`)
 * when the sample is too large for the backend to enumerate the pairs.
 */

export interface HodgesLehmann {
  estimate?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  confidence_level?: number;
  achieved_confidence_level?: number | null;
  method?: string;
  note?: string | null;
}

function isNum(v: number | null | undefined): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function pct(level: number | undefined): string {
  const p = typeof level === "number" && level > 0 && level < 1 ? level * 100 : 95;
  return `${Number(p.toPrecision(6))}%`;
}

export default function HodgesLehmannLine({ hl, label }: { hl: HodgesLehmann; label: string }) {
  const level = pct(hl.confidence_level);
  const value = isNum(hl.estimate)
    ? `${hl.estimate.toFixed(3)} ${isNum(hl.ci_low) && isNum(hl.ci_high) ? `[${hl.ci_low.toFixed(3)}, ${hl.ci_high.toFixed(3)}]` : "[CI n/a]"}`
    : "n/a";
  const achieved = hl.achieved_confidence_level;
  const achievedDiffers =
    isNum(achieved) && typeof hl.confidence_level === "number" && Math.abs(achieved - hl.confidence_level) > 0.005;

  return (
    <div className="rounded-lg bg-indigo-50 px-3 py-1.5 text-xs space-y-0.5" data-testid="hodges-lehmann">
      <div className="flex flex-wrap items-center gap-3">
        <span className="font-semibold text-indigo-800">{label} [{level} CI]</span>
        <span className="font-mono text-indigo-700">{value}</span>
      </div>
      {(hl.method || achievedDiffers) && (
        <p className="text-[10px] text-indigo-500">
          {hl.method}
          {achievedDiffers && isNum(achieved) && `${hl.method ? "; " : ""}achieved confidence level ${(achieved * 100).toFixed(1)}%`}
        </p>
      )}
      {hl.note && <p className="text-[10px] text-gray-500">{hl.note}</p>}
    </div>
  );
}
