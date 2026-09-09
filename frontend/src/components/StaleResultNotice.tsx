import { AlertTriangle, RefreshCw } from "lucide-react";
import { describeStale, type StaleReason } from "../lib/resultStamp";

interface Props {
  reasons: StaleReason[];
  /** When given, renders a button that re-runs the analysis in place. */
  onRecompute?: () => void;
  busy?: boolean;
  /** What is on screen, for the sentence: "This Cox model was computed…". */
  what?: string;
}

/**
 * The line a result gets when the conditions it was computed under have moved.
 *
 * Amber and not red: nothing is broken, and the number was correct for the
 * question it answered. What it no longer is, is an answer about what the user
 * is looking at now -- and the only way to find that out used to be to notice
 * it yourself. Export stays blocked while this is showing (see
 * `ResultExporter`'s `stale` prop), because a downloaded CSV loses every bit of
 * context that would let a reader tell.
 */
export default function StaleResultNotice({ reasons, onRecompute, busy, what = "This result" }: Props) {
  if (reasons.length === 0) return null;
  return (
    <div
      role="status"
      className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-[11px] text-amber-800"
    >
      <AlertTriangle size={14} className="mt-px flex-shrink-0 text-amber-500" />
      <div className="flex-1 leading-snug">
        <b>Out of date.</b> {what} was computed before {describeStale(reasons)}. It is not an
        answer about the data on screen, and export is disabled until it is recomputed.
      </div>
      {onRecompute && (
        <button
          onClick={onRecompute}
          disabled={busy}
          className="flex-shrink-0 inline-flex items-center gap-1 rounded border border-amber-400 bg-white px-2 py-0.5 text-[10px] font-medium text-amber-800 hover:bg-amber-100 disabled:opacity-40"
        >
          <RefreshCw size={10} className={busy ? "animate-spin" : ""} />
          {busy ? "Recomputing…" : "Recompute"}
        </button>
      )}
    </div>
  );
}
