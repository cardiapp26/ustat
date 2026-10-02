/** The Split File banner, on every tab: which variable splits the analyses,
 *  and one tab per group (plus the unsplit view) to choose the one on view. */
import { X } from "lucide-react";
import { useStore } from "../store";

export default function SplitFileBar() {
  const split = useStore((s) => s.splitFile);
  const setSplitLevel = useStore((s) => s.setSplitLevel);
  const setSplitFile = useStore((s) => s.setSplitFile);
  if (!split) return null;

  const tab = (level: string | null, label: string, n?: number) => {
    const active = split.level === level;
    return (
      <button
        key={level ?? "__all__"}
        onClick={() => setSplitLevel(level)}
        aria-pressed={active}
        className={`px-2 py-0.5 rounded border text-[11px] transition-colors ${active
          ? "bg-emerald-600 border-emerald-600 text-white font-semibold"
          : "bg-white border-emerald-300 text-emerald-800 hover:bg-emerald-100"}`}
      >
        {label}{n != null && <span className={active ? "text-emerald-100" : "text-emerald-500"}> ({n})</span>}
      </button>
    );
  };

  return (
    <div className="flex items-center gap-2 px-4 py-1 bg-emerald-50 border-t border-emerald-200 text-xs text-emerald-800 flex-wrap" role="status">
      <span className="font-semibold">Split by {split.column}:</span>
      {split.levels.map((l) => tab(l.level, l.level, l.n))}
      {tab(null, "All cases")}
      <span className="text-emerald-500">results show the group selected here</span>
      <button
        onClick={() => setSplitFile(null)}
        className="ml-auto flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-200 hover:bg-emerald-300 text-emerald-900 font-medium transition-colors"
      >
        <X size={10} /> Turn off
      </button>
    </div>
  );
}
