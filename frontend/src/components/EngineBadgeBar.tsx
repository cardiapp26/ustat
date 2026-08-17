/**
 * One line, above every tab, saying what is computing the numbers.
 *
 * WHY ONE COMPONENT AND NOT FIFTY BADGES. Provenance is a property of the
 * session and of the last run, not of a panel: threading `runtime`/`engine`
 * through ~50 analysis panels would be ~50 chances to forget one, and the panel
 * that got forgotten is the one that quietly claims R computed something Python
 * did. So `localFirst` writes every routing decision into one store slice
 * (`engineNotices`) and this component is the only thing that reads it.
 *
 * The persistent half states the engine the session was opened under. The amber
 * half appears only in the case that would otherwise mislead: an R session whose
 * most recent result for the tab on screen came from Python. A reader who chose
 * R and is looking at a Python number has to be told so on the same screen, not
 * in a console message they will never open.
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

export default function EngineBadgeBar() {
  const engine = useStore((s) => s.engine);
  const activeTab = useStore((s) => s.activeTab);
  const notices = useStore((s) => s.engineNotices);

  // The most recent notice among the analyses this tab can produce. `at` and
  // not insertion order: the store keeps one record per analysis id, so the
  // object's key order is first-seen order, which is not what "most recent" is.
  const relevant = (TAB_ANALYSES[activeTab] ?? [])
    .map((id) => notices[id])
    .filter((n) => n !== undefined)
    .sort((a, b) => b.at - a.at)[0];

  const fellBackToPython = engine === "r" && relevant?.engine === "python";

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-1.5 border-b border-line bg-surface flex-shrink-0">
      <span
        className={`inline-flex items-center gap-1.5 text-[11px] font-semibold rounded-full px-2.5 py-0.5 border ${
          engine === "r"
            ? "text-sky-700 border-sky-200 bg-sky-50"
            : "text-ink-600 border-ink-200 bg-ink-50"
        }`}
        title={
          engine === "r"
            ? "This session runs R in your browser via webR."
            : "This session runs Python — in your browser when the analysis supports it, otherwise on the server."
        }
      >
        {engine === "r" ? "R-based statistics" : "Python-based statistics"}
      </span>

      {relevant?.engineDetail && !fellBackToPython && (
        <span className="text-[11px] text-slate-400">{relevant.engineDetail}</span>
      )}

      {fellBackToPython && (
        <span
          className="inline-flex items-center gap-1.5 text-[11px] font-medium text-amber-700"
          role="status"
        >
          <AlertTriangle size={13} className="flex-shrink-0 text-amber-500" />
          {PYTHON_FALLBACK_NOTICE}
        </span>
      )}
    </div>
  );
}
