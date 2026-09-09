/**
 * Which engine this SESSION runs, said once, in the header.
 *
 * WHAT THIS CHIP IS NOT. It is not a claim about the result on screen. Only a
 * short allow-list of analyses is routed to a local engine at all; everything
 * else -- every regression, Cox model, ROC and forest -- is a POST to the
 * server's Python. A chip reading "R-based statistics" above a number Python
 * computed is a claim that will not reproduce, so the per-result answer lives
 * with each result (`ResultProvenanceLine`, fed from the run's own reported
 * engine and library versions) and this chip says only what was chosen.
 *
 * WHY ONE MODULE AND NOT FIFTY BADGES. `localFirst` writes every routing
 * decision into one store slice (`engineNotices`) and these two components are
 * the only things that read it -- threading `runtime`/`engine` through ~50
 * analysis panels would be ~50 chances to forget one.
 *
 * Two components rather than one strip, because the two halves earn different
 * amounts of screen. `EngineChip` is always true and always short, so it sits
 * on the header's own row next to the file name -- a permanent full-width band
 * to hold six words is a band the reader stops seeing. `EngineFallbackNotice`
 * appears only in the case that would otherwise mislead -- an R session whose
 * most recent result for the tab on screen came from Python -- and it says so
 * in a full sentence, which needs the width. A reader who chose R and is
 * looking at a Python number has to be told on the same screen as the number,
 * not in a console message they will never open.
 */
import { AlertTriangle } from "lucide-react";
import { useStore } from "../store";
import { PYTHON_FALLBACK_NOTICE } from "../lib/engine/engineDetail";

/**
 * Which analysis ids can produce a result on which tab.
 *
 * Only the routed ones are listed: an analysis that always goes to the server
 * writes no notice, so naming it here would say nothing. Grows as analyses join
 * an allow-list.
 */
const TAB_ANALYSES: Record<string, readonly string[]> = {
  tests: ["stats.ttest"],
  power: ["stats.power"],
};

/**
 * The most recent notice among the analyses the tab on screen can produce.
 *
 * `at` and not insertion order: the store keeps one record per analysis id, so
 * the object's key order is first-seen order, which is not what "most recent"
 * is.
 */
function useRelevantNotice() {
  const activeTab = useStore((s) => s.activeTab);
  const notices = useStore((s) => s.engineNotices);
  return (TAB_ANALYSES[activeTab] ?? [])
    .map((id) => notices[id])
    .filter((n) => n !== undefined)
    .sort((a, b) => b.at - a.at)[0];
}

/** The engine this session runs, as a header chip. Always shown. */
export function EngineChip() {
  const engine = useStore((s) => s.engine);
  const engineSource = useStore((s) => s.engineSource);
  const relevant = useRelevantNotice();

  const isR = engine === "r";
  // A resumed session is opened in the engine its work was done in, which
  // never passes the welcome gate. Saying only "R-based statistics" after the
  // gate showed Python selected reads as a bug; saying where it came from is
  // the difference between an unexplained flip and a restored choice.
  const resumed = engineSource === "resume";

  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[11px] font-semibold rounded-full px-2.5 py-0.5 border flex-shrink-0 ${
        isR
          ? "text-sky-700 border-sky-200 bg-sky-50"
          : "text-ink-600 border-ink-200 bg-ink-50"
      }`}
      title={
        (isR
          ? "This session prefers R, in your browser via webR. Analyses R does not implement yet are computed by the server's Python."
          : "This session runs Python — in your browser when the analysis supports it, otherwise on the server.") +
        (resumed
          ? " It was restored with this saved session, which was worked in that engine, so it did not pass the welcome screen's choice."
          : " Chosen on the welcome screen.") +
        " Each result states the engine and library versions that actually produced it."
      }
    >
      {isR ? "R-based statistics" : "Python-based statistics"}
      {resumed && <span className="font-medium opacity-70">· restored</span>}
      {relevant?.engineDetail && !(isR && relevant.engine === "python") && (
        <span className="hidden lg:inline font-medium opacity-60">· {relevant.engineDetail}</span>
      )}
    </span>
  );
}

/** The amber line an R session gets when Python answered the tab on screen. */
export function EngineFallbackNotice() {
  const engine = useStore((s) => s.engine);
  const relevant = useRelevantNotice();

  if (!(engine === "r" && relevant?.engine === "python")) return null;

  return (
    <div className="flex items-center gap-1.5 px-4 py-1 border-b border-amber-200 bg-amber-50 flex-shrink-0">
      <AlertTriangle size={13} className="flex-shrink-0 text-amber-500" />
      <span className="text-[11px] font-medium text-amber-700" role="status">
        {PYTHON_FALLBACK_NOTICE}
      </span>
    </div>
  );
}
