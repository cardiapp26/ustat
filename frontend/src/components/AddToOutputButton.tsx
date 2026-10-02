/** "+ Output": copy the result area this control sits in into the output
 *  document. Reads the area from the surrounding <StaleGuard>, and refuses
 *  while that result is out of date, like every other way a result leaves
 *  its panel. */
import { useState } from "react";
import { FilePlus2 } from "lucide-react";
import { useStaleGuard, staleExportTitle } from "../lib/staleGuard";
import { captureGuardedArea } from "../lib/outputCapture";
import { useOutputDoc } from "../lib/outputDoc";
import { useStore } from "../store";

interface Props {
  /** Used when the captured area has no heading of its own. */
  fallbackTitle?: string;
}

export default function AddToOutputButton({ fallbackTitle }: Props) {
  const guard = useStaleGuard();
  const add = useOutputDoc((s) => s.add);
  const activeTab = useStore((s) => s.activeTab);
  const [state, setState] = useState<"idle" | "busy" | "done" | "failed">("idle");

  if (!guard.captureId) return null;

  const onClick = async () => {
    if (!guard.captureId) return;
    setState("busy");
    const captured = await captureGuardedArea(guard.captureId);
    if (!captured || !captured.html.trim()) {
      setState("failed");
      return;
    }
    add({ title: captured.title || fallbackTitle || "Result", tab: activeTab ?? "", html: captured.html });
    setState("done");
    window.setTimeout(() => setState("idle"), 1500);
  };

  return (
    <button
      onClick={onClick}
      disabled={guard.stale || state === "busy"}
      title={guard.stale ? staleExportTitle(guard.reason) : "Add this result to the output document"}
      className="px-2 py-0.5 text-[10px] font-medium rounded border border-indigo-200 bg-white text-indigo-600 hover:bg-indigo-50 disabled:opacity-40 transition-colors flex items-center gap-0.5"
    >
      <FilePlus2 size={10} />
      {state === "busy" ? "…" : state === "done" ? "Added" : state === "failed" ? "Not added" : "Output"}
    </button>
  );
}
