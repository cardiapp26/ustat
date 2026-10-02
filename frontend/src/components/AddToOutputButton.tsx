/** "+ Output": copy the result area this control sits in into the output
 *  document. Reads the area from the surrounding <StaleGuard>, and refuses
 *  while that result is out of date, like every other way a result leaves
 *  its panel.
 *
 *  Under Split File, items are titled with the group they show, and "Each
 *  group" walks the groups in turn, adding each one's result once it is
 *  current (the background precompute in useStampedResult provides them). */
import { useState } from "react";
import { FilePlus2 } from "lucide-react";
import { useStaleGuard, staleExportTitle } from "../lib/staleGuard";
import { captureGuardedArea, guardIsCurrent } from "../lib/outputCapture";
import { useOutputDoc } from "../lib/outputDoc";
import { useStore } from "../store";

interface Props {
  /** Used when the captured area has no heading of its own. */
  fallbackTitle?: string;
}

const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

/** Wait until the guarded result is current again after a level switch. */
async function waitUntilCurrent(captureId: string, timeoutMs: number): Promise<boolean> {
  await nextFrame();
  await nextFrame();
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (guardIsCurrent(captureId)) return true;
    await sleep(100);
  }
  return false;
}

const splitLabel = (column: string, level: string | null) =>
  ` [${column} = ${level ?? "all cases"}]`;

export default function AddToOutputButton({ fallbackTitle }: Props) {
  const guard = useStaleGuard();
  const add = useOutputDoc((s) => s.add);
  const activeTab = useStore((s) => s.activeTab);
  const split = useStore((s) => s.splitFile);
  const [state, setState] = useState<"idle" | "busy" | "done" | "failed">("idle");
  const [note, setNote] = useState<string | null>(null);

  if (!guard.captureId) return null;
  const captureId = guard.captureId;

  const captureOne = async (suffix: string): Promise<boolean> => {
    const captured = await captureGuardedArea(captureId);
    if (!captured || !captured.html.trim()) return false;
    add({ title: (captured.title || fallbackTitle || "Result") + suffix, tab: activeTab ?? "", html: captured.html });
    return true;
  };

  const finish = (ok: boolean) => {
    setState(ok ? "done" : "failed");
    window.setTimeout(() => setState("idle"), 1500);
  };

  const onClick = async () => {
    setState("busy"); setNote(null);
    const s = useStore.getState().splitFile;
    finish(await captureOne(s ? splitLabel(s.column, s.level) : ""));
  };

  const onEachGroup = async () => {
    const s = useStore.getState();
    const current = s.splitFile;
    if (!current) return;
    setState("busy"); setNote(null);
    const skipped: string[] = [];
    let added = 0;
    for (const { level } of current.levels) {
      s.setSplitLevel(level);
      if (await waitUntilCurrent(captureId, 20_000) && await captureOne(splitLabel(current.column, level))) {
        added += 1;
      } else {
        skipped.push(level);
      }
    }
    useStore.getState().setSplitLevel(current.level);
    if (skipped.length) {
      setNote(`Not added: ${skipped.join(", ")} (no current result; open that group and Recompute).`);
    }
    finish(added > 0);
  };

  const buttonClass =
    "px-2 py-0.5 text-[10px] font-medium rounded border border-indigo-200 bg-white text-indigo-600 hover:bg-indigo-50 disabled:opacity-40 transition-colors flex items-center gap-0.5";
  const blocked = guard.stale ? staleExportTitle(guard.reason) : undefined;

  return (
    <>
      <button onClick={onClick} disabled={guard.stale || state === "busy"}
        title={blocked ?? "Add this result to the output document"} className={buttonClass}>
        <FilePlus2 size={10} />
        {state === "busy" ? "…" : state === "done" ? "Added" : state === "failed" ? "Not added" : "Output"}
      </button>
      {split && (
        <button onClick={onEachGroup} disabled={guard.stale || state === "busy"}
          title={blocked ?? `Add this result for every group of ${split.column} to the output document`}
          className={buttonClass}>
          Each group
        </button>
      )}
      {note && <span className="text-[10px] text-amber-700" role="status">{note}</span>}
    </>
  );
}
