import { AlertTriangle, X } from "lucide-react";
import { useStore } from "../store";

/**
 * Says what a session recovery cost.
 *
 * Datasets live in the backend's memory, so a restart or a redeploy leaves the
 * open session pointing at nothing. `sessionRecovery` puts the autosaved copy
 * back — but that copy is a snapshot from up to a minute earlier, so the app
 * jumps back in time: a column deleted since then reappears, a column added
 * since then is gone, and the request that hit the dead session usually fails
 * on top of it. Reported exactly that way — an error, a column that vanished,
 * a deleted column that came back — with no way to tell it apart from the app
 * corrupting its own state.
 *
 * The rollback is the best available outcome; being unexplained was the bug.
 */
export default function SessionRecoveredNotice() {
  const notice = useStore((s) => s.sessionRecovery);
  const setNotice = useStore((s) => s.setSessionRecovery);
  if (!notice) return null;

  const at = new Date(notice.snapshotAt);
  const stamp = Number.isFinite(at.getTime())
    ? at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : null;

  return (
    <div className="fixed bottom-4 right-4 z-[60] max-w-sm animate-in fade-in duration-300">
      <div className="bg-amber-50 border border-amber-300 rounded-xl shadow-lg px-3 py-2.5 flex items-start gap-2">
        <AlertTriangle size={14} className="text-amber-600 flex-shrink-0 mt-0.5" />
        <div className="text-[11px] text-amber-900 leading-relaxed">
          <span className="font-semibold">Session restored from the autosave</span>
          {stamp ? <> taken at {stamp}</> : null}.{" "}
          The server had forgotten <span className="font-medium">{notice.name}</span> — a
          restart or an update does that, because the data is never written to disk.
          Anything changed after that snapshot is not in this copy; check the last edits
          you made before carrying on.
        </div>
        <button
          onClick={() => setNotice(null)}
          aria-label="Dismiss"
          className="ml-1 text-amber-600 hover:text-amber-900 flex-shrink-0"
        >
          <X size={13} />
        </button>
      </div>
    </div>
  );
}
