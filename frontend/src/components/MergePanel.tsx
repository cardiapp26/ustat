/**
 * MergePanel — join a second file onto the open dataset.
 *
 * The operation itself is one line of pandas; what this panel is for is the
 * damage that a join does quietly. A key that matches nothing, a lookup file
 * with duplicate keys silently multiplying rows, a column overwritten because
 * both files happened to call it `sbp` — none of those announce themselves in
 * a spreadsheet.
 *
 * So nothing is joined until the user has seen what the join would do. The
 * preview is not a convenience step that can be skipped: it is the only place
 * the row-count arithmetic is shown before it is applied.
 */
import { useMemo, useRef, useState } from "react";
import { useStore, runColumnStructureMutation, type Session } from "../store";
import { uploadFile, mergePreview, mergeApply, getSessionInfo } from "../api";
import { Tip } from "./Tip";

interface Plan {
  rows_left: number; rows_right: number; rows_after: number | null;
  keys_matched: number; left_rows_matched: number; left_rows_unmatched: number;
  left_keys_missing: number; right_keys_missing: number;
  left_duplicate_keys: number; right_duplicate_keys: number; right_keys_unused: number;
  columns_added: string[]; warnings: string[]; result_text?: string;
}

interface Incoming { sessionId: string; filename: string; columns: string[] }

export default function MergePanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <MergePanelBody session={session} />;
}

function MergePanelBody({ session }: { session: Session }) {
  const setSession = useStore((s) => s.setSession);
  const bumpDataVersion = useStore((s) => s.bumpDataVersion);
  const fileRef = useRef<HTMLInputElement>(null);

  const leftCols = useMemo(() => session.columns.map((c) => c.name), [session.columns]);
  const [incoming, setIncoming] = useState<Incoming | null>(null);
  const [leftKey, setLeftKey] = useState("");
  const [rightKey, setRightKey] = useState("");
  const [how, setHow] = useState<"left" | "inner" | "outer">("left");
  const [bring, setBring] = useState<string[]>([]);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [applied, setApplied] = useState<Plan | null>(null);
  const [busy, setBusy] = useState<null | "upload" | "preview" | "apply">(null);
  const [error, setError] = useState<string | null>(null);

  const err = (e: unknown, fallback: string) => {
    const msg = (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
    setError(typeof msg === "string" ? msg : e instanceof Error ? e.message : fallback);
  };

  const pickFile = async (file: File) => {
    setBusy("upload"); setError(null); setPlan(null); setApplied(null);
    try {
      const res = await uploadFile(file);
      const data = res.data as { session_id: string; columns: { name: string }[]; filename?: string };
      const cols = data.columns.map((c) => c.name);
      setIncoming({ sessionId: data.session_id, filename: data.filename ?? file.name, columns: cols });
      // A column with the same name on both sides is the likely key.
      const shared = cols.find((c) => leftCols.includes(c));
      setLeftKey(shared ?? "");
      setRightKey(shared ?? "");
      setBring([]);
    } catch (e) { err(e, "Could not read that file"); }
    finally { setBusy(null); }
  };

  const body = () => ({
    session_id: session.session_id, other_session_id: incoming!.sessionId,
    left_on: [leftKey], right_on: [rightKey], how, columns: bring,
  });

  const preview = async () => {
    setBusy("preview"); setError(null); setApplied(null);
    try { setPlan((await mergePreview(body())).data as Plan); }
    catch (e) { err(e, "Preview failed"); setPlan(null); }
    finally { setBusy(null); }
  };

  const apply = async () => {
    setBusy("apply"); setError(null);
    try {
      const res = await runColumnStructureMutation(session.session_id, () => mergeApply(body()));
      setApplied(res.data as Plan);
      setPlan(null);
      // The join rewrote the sheet — column list and preview both need refetching.
      const info = await getSessionInfo(session.session_id);
      setSession(info.data as Session);
      bumpDataVersion();
    } catch (e) { err(e, "The join failed"); }
    finally { setBusy(null); }
  };

  const ready = !!incoming && !!leftKey && !!rightKey;
  const shown = applied ?? plan;

  const stat = (label: string, value: number | string | null, tone = "") => (
    <div className="rounded bg-gray-50 px-2 py-1.5">
      <p className="text-[10px] text-gray-400">{label}</p>
      <p className={`font-mono text-sm ${tone || "text-gray-700"}`}>{value ?? "—"}</p>
    </div>
  );

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-3">
        <div className="panel space-y-3">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-900">Join another file</h3>
            <Tip wide text="Matches rows from a second file onto the open dataset by a shared identifier. Keys are compared as trimmed text, so 1024 and '1024' are the same participant; case is not folded, because AB12 and ab12 may not be." />
          </div>

          <div>
            <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls,.sav,.dta,.sas7bdat"
              aria-label="File to join"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) void pickFile(f); }}
              className="block w-full text-xs text-gray-500 file:mr-2 file:rounded-lg file:border-0 file:bg-indigo-50 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-indigo-700 hover:file:bg-indigo-100" />
            {incoming && (
              <p className="mt-1 text-[11px] text-gray-500">
                {incoming.filename} — {incoming.columns.length} columns
              </p>
            )}
            {busy === "upload" && <p className="mt-1 text-[11px] text-gray-400">Reading…</p>}
          </div>

          {incoming && (
            <>
              <div className="grid grid-cols-2 gap-2">
                <label className="space-y-1">
                  <span className="text-xs font-medium text-gray-500">Key here</span>
                  <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
                    value={leftKey} onChange={(e) => { setLeftKey(e.target.value); setPlan(null); }}>
                    <option value="">— select —</option>
                    {leftCols.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-medium text-gray-500">Key in the file</span>
                  <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
                    value={rightKey} onChange={(e) => { setRightKey(e.target.value); setPlan(null); }}>
                    <option value="">— select —</option>
                    {incoming.columns.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
              </div>

              <label className="block space-y-1">
                <span className="text-xs font-medium text-gray-500">Rows to keep</span>
                <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
                  value={how} onChange={(e) => { setHow(e.target.value as typeof how); setPlan(null); }}>
                  <option value="left">All rows here (unmatched left empty)</option>
                  <option value="inner">Only rows that matched</option>
                  <option value="outer">All rows from both files</option>
                </select>
              </label>

              <div className="space-y-1">
                <span className="text-xs font-medium text-gray-500">
                  Columns to bring <span className="text-gray-400">(all, if none ticked)</span>
                </span>
                <div className="max-h-32 overflow-y-auto rounded-lg border border-gray-200 p-1.5">
                  {incoming.columns.filter((c) => c !== rightKey).map((c) => (
                    <label key={c} className="flex items-center gap-2 px-1 py-0.5 text-xs text-gray-700 hover:bg-gray-50 rounded cursor-pointer">
                      <input type="checkbox" checked={bring.includes(c)}
                        onChange={() => { setBring((p) => p.includes(c) ? p.filter((x) => x !== c) : [...p, c]); setPlan(null); }} />
                      <span className="truncate">{c}</span>
                      {leftCols.includes(c) && <span className="ml-auto text-[9px] text-amber-600" title="A column of this name already exists here">clash</span>}
                    </label>
                  ))}
                </div>
              </div>

              <button className="btn-primary w-full" onClick={preview} disabled={!ready || busy !== null}>
                {busy === "preview" ? "Checking…" : "Check the join"}
              </button>
            </>
          )}
          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {shown ? (
          <>
            <div className="panel space-y-3">
              <h4 className="text-sm font-semibold text-gray-700">
                {applied ? "Joined" : "What this join would do"}
              </h4>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {stat("Rows here", shown.rows_left)}
                {stat("Rows in the file", shown.rows_right)}
                {stat("Matched", shown.left_rows_matched,
                  shown.left_rows_matched === 0 ? "text-red-600" : "text-green-700")}
                {stat("Rows after", shown.rows_after ?? "depends on duplicates",
                  shown.rows_after != null && shown.rows_after > shown.rows_left ? "text-amber-700" : "")}
                {stat("Unmatched here", shown.left_rows_unmatched)}
                {stat("Unused keys in the file", shown.right_keys_unused)}
                {stat("Blank keys", `${shown.left_keys_missing} / ${shown.right_keys_missing}`)}
                {stat("Duplicate keys in the file", shown.right_duplicate_keys,
                  shown.right_duplicate_keys ? "text-amber-700" : "")}
              </div>

              <div>
                <p className="text-[11px] font-medium text-gray-500">
                  Columns {applied ? "added" : "to add"} ({shown.columns_added.length})
                </p>
                <p className="font-mono text-[11px] text-gray-600">
                  {shown.columns_added.join(", ") || "—"}
                </p>
              </div>

              {shown.warnings.map((w) => (
                <p key={w} className="rounded bg-amber-50 px-2 py-1.5 text-[11px] leading-relaxed text-amber-800">{w}</p>
              ))}

              {!applied && (
                <button className="btn-primary w-full" onClick={apply} disabled={busy !== null}>
                  {busy === "apply" ? "Joining…" : "Apply the join"}
                </button>
              )}
              {applied && (
                <p className="rounded border border-green-200 bg-green-50 px-2 py-1.5 text-[11px] text-green-800">
                  {applied.result_text} The Data tab now shows the joined sheet; Undo reverses it.
                </p>
              )}
            </div>
          </>
        ) : (
          <div className="panel py-16 text-center text-gray-400">
            <p className="mb-2 text-lg">🔗</p>
            <p>Choose a file and the identifier the two share.</p>
            <p className="mx-auto mt-2 max-w-md text-xs">
              Nothing is changed until you have seen how many rows match, how many are left empty,
              and whether duplicate keys would multiply the sheet.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
