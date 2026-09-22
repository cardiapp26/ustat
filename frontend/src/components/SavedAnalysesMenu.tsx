/**
 * The project tree, header edition: keep the current analysis under a name,
 * list what is kept, restore one back into its panel.
 *
 * Restore puts the saved snapshot (selections + result + stamp) back into the
 * panel and navigates to its tab and sub-tab (lib/panelRegistry); the stamp
 * decides whether the restored number is still current, exactly as it does
 * for a live result. Re-run goes one step further: it sends the request the
 * stamp recorded to the data now open and restores the fresh result, so a
 * kept analysis recomputes without its panel's Run button. An analysis whose
 * panel reshaped the response has no recorded request and can only be
 * restored and recomputed in its panel.
 *
 * The "keep" list offers every panel that currently caches a result, by name.
 */
import { useEffect, useRef, useState } from "react";
import { Bookmark, Check, Code2, Pencil, RefreshCw, Trash2 } from "lucide-react";
import { useStore, type SavedAnalysis } from "../store";
import { panelLabel } from "../lib/panelRegistry";
import SyntaxView from "./SyntaxView";

/** Kept with the request that computed it, so it can be re-run as is. */
function replayable(a: SavedAnalysis): boolean {
  const snapshot = a.snapshot as { stamp?: { request?: unknown } } | null;
  return !!snapshot?.stamp?.request;
}

interface CacheEntryWithResult {
  result?: unknown;
}

function panelsWithResults(cache: Record<string, unknown>): string[] {
  return Object.keys(cache).filter((panel) => {
    const entry = cache[panel] as CacheEntryWithResult | null;
    return entry != null && typeof entry === "object" && entry.result != null;
  });
}

export default function SavedAnalysesMenu() {
  const savedAnalyses = useStore((s) => s.savedAnalyses);
  const panelCache = useStore((s) => s.panelCache);
  const activeTab = useStore((s) => s.activeTab);
  const saveAnalysis = useStore((s) => s.saveAnalysis);
  const renameAnalysis = useStore((s) => s.renameAnalysis);
  const deleteAnalysis = useStore((s) => s.deleteAnalysis);
  const restoreAnalysis = useStore((s) => s.restoreAnalysis);

  const [open, setOpen] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [syntaxFor, setSyntaxFor] = useState<SavedAnalysis | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const keepable = panelsWithResults(panelCache);
  const rerunAnalysis = useStore((s) => s.rerunAnalysis);
  const [rerunning, setRerunning] = useState<string | null>(null);
  const [rerunError, setRerunError] = useState<string | null>(null);

  const rerun = async (a: SavedAnalysis) => {
    setRerunning(a.id);
    setRerunError(null);
    try {
      await rerunAnalysis(a.id);
      setOpen(false);
    } catch (e) {
      const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      setRerunError(typeof detail === "string" ? detail : e instanceof Error ? e.message : "Re-run failed");
    } finally {
      setRerunning(null);
    }
  };

  const commitRename = (id: string) => {
    renameAnalysis(id, renameValue);
    setRenamingId(null);
  };

  return (
    <div className="relative" ref={rootRef}>
      <button
        onClick={() => setOpen((v) => !v)}
        className={`relative p-1.5 rounded-lg transition-colors ${open ? "text-violet-600 bg-violet-50" : "text-gray-400 hover:text-violet-600 hover:bg-violet-50"}`}
        title="Saved analyses"
      >
        <Bookmark size={16} />
        {savedAnalyses.length > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[14px] h-[14px] px-0.5 rounded-full bg-violet-500 text-white text-[9px] leading-[14px] text-center font-semibold">
            {savedAnalyses.length}
          </span>
        )}
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 w-64 bg-white border border-gray-200 rounded-xl shadow-xl z-50 overflow-hidden">
          <p className="px-3 pt-2.5 pb-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
            Saved analyses
          </p>
          {savedAnalyses.length === 0 && (
            <p className="px-3 pb-2 text-xs text-gray-400">Nothing kept yet.</p>
          )}
          {rerunError && (
            <p role="alert" className="px-3 pb-1 text-[10px] text-red-600 leading-snug">{rerunError}</p>
          )}
          {savedAnalyses.map((a) => (
            <div key={a.id} className="flex items-center gap-1 px-3 py-1.5 hover:bg-gray-50 group">
              {renamingId === a.id ? (
                <>
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") commitRename(a.id); if (e.key === "Escape") setRenamingId(null); }}
                    className="flex-1 min-w-0 text-xs border border-violet-300 rounded px-1.5 py-0.5 focus:outline-none"
                  />
                  <button onClick={() => commitRename(a.id)} className="p-1 text-emerald-600" title="Rename">
                    <Check size={12} />
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => { restoreAnalysis(a.id); setOpen(false); }}
                    className="flex-1 min-w-0 text-left"
                    title="Restore into its panel"
                  >
                    <p className="text-xs text-gray-700 font-medium truncate">{a.name}</p>
                    <p className="text-[10px] text-gray-400 truncate">
                      {panelLabel(a.panel)} · {new Date(a.createdAt).toLocaleDateString()}
                    </p>
                  </button>
                  <button
                    onClick={() => void rerun(a)}
                    disabled={!replayable(a) || rerunning !== null}
                    className="p-1 text-gray-300 hover:text-violet-600 opacity-0 group-hover:opacity-100 disabled:hover:text-gray-300 disabled:opacity-30"
                    title={replayable(a)
                      ? "Re-run on the data now open"
                      : "Kept without the request that computed it: restore it and press Recompute in its panel"}
                    aria-label={`Re-run ${a.name}`}
                  >
                    <RefreshCw size={12} className={rerunning === a.id ? "animate-spin" : ""} />
                  </button>
                  <button
                    onClick={() => { setSyntaxFor(a); setOpen(false); }}
                    className="p-1 text-gray-300 hover:text-violet-600 opacity-0 group-hover:opacity-100"
                    title="Show syntax (Python / R)"
                  >
                    <Code2 size={12} />
                  </button>
                  <button
                    onClick={() => { setRenamingId(a.id); setRenameValue(a.name); }}
                    className="p-1 text-gray-300 hover:text-gray-600 opacity-0 group-hover:opacity-100"
                    title="Rename"
                  >
                    <Pencil size={12} />
                  </button>
                  <button
                    onClick={() => deleteAnalysis(a.id)}
                    className="p-1 text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100"
                    title="Delete"
                  >
                    <Trash2 size={12} />
                  </button>
                </>
              )}
            </div>
          ))}
          {keepable.length > 0 && (
            <>
              <div className="border-t border-gray-100" />
              <p className="px-3 pt-2 pb-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
                Keep current result
              </p>
              {keepable.map((panel) => (
                <button
                  key={panel}
                  onClick={() => saveAnalysis(panel, activeTab)}
                  className="w-full px-3 py-1.5 text-left text-xs text-gray-600 hover:bg-violet-50 hover:text-violet-700 transition-colors"
                >
                  {panelLabel(panel)}
                </button>
              ))}
            </>
          )}
        </div>
      )}
      {syntaxFor && <SyntaxView analysis={syntaxFor} onClose={() => setSyntaxFor(null)} />}
    </div>
  );
}
