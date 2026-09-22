import { useEffect, useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import { useStore, type Session } from "../store";
import { getMissingCodes, saveMetadata, type MissingCodeSuggestion } from "../api";

/**
 * Values that look like missing codes and are not declared as such.
 *
 * A clinical file writes "not recorded" as 99 or 999, and upload reads those
 * as numbers: a 999 among ages goes into every mean. The server proposes the
 * conventional codes that sit far outside the rest of a column; this band
 * shows them once, above the tabs, with one click to declare each. Declaring
 * keeps the value in the data and makes every analysis read it as missing.
 * Nothing is applied without that click, because 99 can be a real value.
 */
export default function MissingCodeNotice() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <MissingCodeNoticeBody key={session.session_id} session={session} />;
}

type Found = { column: string; s: MissingCodeSuggestion };

function MissingCodeNoticeBody({ session }: { session: Session }) {
  const dataVersion = useStore((s) => s.dataVersion);
  const setSession = useStore((s) => s.setSession);
  const bumpDataVersion = useStore((s) => s.bumpDataVersion);
  const [found, setFound] = useState<Found[]>([]);
  const [dismissed, setDismissed] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getMissingCodes(session.session_id)
      .then((res) => {
        if (!live) return;
        const rows = Object.entries(res.data?.suggestions ?? {}).flatMap(([column, list]) =>
          list.map((s) => ({ column, s })),
        );
        setFound(rows);
      })
      .catch(() => {
        /* advisory only: no band is the right fallback */
      });
    return () => { live = false; };
  }, [session.session_id, dataVersion]);

  if (dismissed || found.length === 0) return null;

  const declare = async ({ column, s }: Found) => {
    const id = `${column}:${s.value}`;
    setBusy(id);
    setError(null);
    try {
      const col = session.columns.find((c) => c.name === column);
      const codes = [...new Set([...(col?.missing_codes ?? []), s.value])];
      await saveMetadata(session.session_id, { [column]: { missing_codes: codes } });
      const latest = useStore.getState().session ?? session;
      setSession({
        ...latest,
        columns: latest.columns.map((c) => (c.name === column ? { ...c, missing_codes: codes } : c)),
      });
      // Every result computed with the code as a number is now out of date.
      bumpDataVersion();
    } catch {
      setError(`Could not declare ${s.value} as missing in ${column}.`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div role="status" className="flex-shrink-0 border-b border-amber-200 bg-amber-50 px-4 py-1.5 text-[11px]">
      <div className="flex items-start gap-1.5">
        <AlertTriangle size={13} className="mt-px flex-shrink-0 text-amber-500" />
        <div className="flex-1 space-y-1">
          <p className="font-medium text-amber-800">
            Possible missing-value codes. They are numbers in every analysis until declared.
          </p>
          <ul className="space-y-0.5">
            {found.map((f) => (
              <li key={`${f.column}:${f.s.value}`} className="flex flex-wrap items-baseline gap-x-2">
                <span className="font-mono font-medium text-gray-800">{f.column}</span>
                <span className="font-mono text-gray-700">{f.s.value} ×{f.s.count}</span>
                <span className="text-gray-500">{f.s.reason}.</span>
                <button
                  onClick={() => void declare(f)}
                  disabled={busy !== null}
                  className="rounded border border-amber-300 bg-white px-1.5 py-px text-[10px] font-medium text-amber-800 hover:bg-amber-100 disabled:opacity-40"
                >
                  {busy === `${f.column}:${f.s.value}` ? "Saving…" : "Treat as missing"}
                </button>
              </li>
            ))}
          </ul>
          {error && <p className="text-red-600">{error}</p>}
          <p className="text-gray-500">Codes stay in the data. Review them any time in Data Dictionary, Missing.</p>
        </div>
        <button
          onClick={() => setDismissed(true)}
          title="Dismiss for this session"
          className="rounded p-0.5 text-gray-400 hover:bg-black/5"
        >
          <X size={12} />
        </button>
      </div>
    </div>
  );
}
