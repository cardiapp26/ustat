/** The output document: header button with a count, and the viewer it opens.
 *  Items are added from any result with the "Output" button beside Export. */
import { useState } from "react";
import { ArrowDown, ArrowUp, FileText, Printer, Trash2, X, Download } from "lucide-react";
import { useOutputDoc } from "../lib/outputDoc";
import { downloadOutputHtml, printOutput } from "../lib/outputExport";
import { outputDocxPayload } from "../lib/outputBlocks";
import { exportOutputDocx } from "../api";
import { useStore } from "../store";

export default function OutputViewer() {
  const items = useOutputDoc((s) => s.items);
  const { remove, move, setNote, setTitle, clear } = useOutputDoc.getState();
  const datasetName = useStore((s) => s.session?.filename ?? "");
  const [open, setOpen] = useState(false);
  const [printBlocked, setPrintBlocked] = useState(false);
  const [wordState, setWordState] = useState<{ busy: boolean; error: string | null }>({ busy: false, error: null });

  const downloadWord = async () => {
    setWordState({ busy: true, error: null });
    try {
      const payload = outputDocxPayload(items, datasetName);
      const res = await exportOutputDocx(payload);
      const url = URL.createObjectURL(res.data as Blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${payload.filename}.docx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
      setWordState({ busy: false, error: null });
    } catch (e: unknown) {
      // The error body arrives as a Blob (responseType "blob"); read the detail out of it.
      let message = "Word export failed.";
      const data = (e as { response?: { data?: unknown } })?.response?.data;
      if (data instanceof Blob) {
        try { message = JSON.parse(await data.text()).detail ?? message; } catch { /* keep the generic text */ }
      }
      setWordState({ busy: false, error: message });
    }
  };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="relative p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors"
        title="Output document: results you added, in order, for one export"
        aria-label="Open output document"
      >
        <FileText size={16} />
        {items.length > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[14px] h-[14px] px-0.5 rounded-full bg-indigo-600 text-white text-[9px] leading-[14px] text-center">
            {items.length}
          </span>
        )}
      </button>

      {open && (
        <div className="fixed inset-0 z-50 bg-black/30 flex" role="dialog" aria-modal="true" aria-label="Output document">
          <div className="m-auto w-[min(1000px,96vw)] h-[92vh] bg-white rounded-xl shadow-2xl flex flex-col">
            <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-200">
              <h2 className="font-semibold text-gray-900">Output document</h2>
              <span className="text-xs text-gray-400">{items.length} item{items.length === 1 ? "" : "s"}</span>
              <div className="ml-auto flex items-center gap-2">
                <button disabled={!items.length || wordState.busy} onClick={downloadWord}
                  className="text-xs px-2 py-1 rounded border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:opacity-40 flex items-center gap-1">
                  <Download size={12} /> {wordState.busy ? "Word…" : "Word"}
                </button>
                <button disabled={!items.length} onClick={() => downloadOutputHtml(items, datasetName)}
                  className="text-xs px-2 py-1 rounded border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:opacity-40 flex items-center gap-1">
                  <Download size={12} /> HTML
                </button>
                <button disabled={!items.length} onClick={() => setPrintBlocked(!printOutput(items, datasetName))}
                  className="text-xs px-2 py-1 rounded border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:opacity-40 flex items-center gap-1">
                  <Printer size={12} /> Print / PDF
                </button>
                <button disabled={!items.length}
                  onClick={() => { if (window.confirm("Remove every item from the output document?")) clear(); }}
                  className="text-xs px-2 py-1 rounded border border-red-200 text-red-600 hover:bg-red-50 disabled:opacity-40">
                  Clear
                </button>
                <button onClick={() => setOpen(false)} aria-label="Close" className="p-1 text-gray-400 hover:text-gray-700">
                  <X size={16} />
                </button>
              </div>
            </div>
            {wordState.error && (
              <p className="px-4 py-1.5 text-xs text-red-700 bg-red-50 border-b border-red-200" role="alert">{wordState.error}</p>
            )}
            {printBlocked && (
              <p className="px-4 py-1.5 text-xs text-amber-800 bg-amber-50 border-b border-amber-200">
                The browser blocked the print window. Allow pop-ups for this site, or export HTML and print that.
              </p>
            )}

            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-8 bg-gray-50">
              {items.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-sm text-gray-500 gap-1">
                  <p>No results added yet.</p>
                  <p className="text-xs text-gray-400">
                    Use the <strong>Output</strong> button beside Export on any result to copy it here.
                  </p>
                </div>
              ) : (
                items.map((item, i) => (
                  <section key={item.id} className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start gap-2 mb-1">
                      <input value={item.title} onChange={(e) => setTitle(item.id, e.target.value)}
                        aria-label="Item title"
                        className="flex-1 text-base font-semibold text-gray-900 border-b border-transparent hover:border-gray-200 focus:border-indigo-400 focus:outline-none" />
                      <button onClick={() => move(item.id, -1)} disabled={i === 0} aria-label="Move up"
                        className="p-1 text-gray-400 hover:text-indigo-600 disabled:opacity-30"><ArrowUp size={14} /></button>
                      <button onClick={() => move(item.id, 1)} disabled={i === items.length - 1} aria-label="Move down"
                        className="p-1 text-gray-400 hover:text-indigo-600 disabled:opacity-30"><ArrowDown size={14} /></button>
                      <button onClick={() => remove(item.id)} aria-label="Remove item"
                        className="p-1 text-gray-400 hover:text-red-600"><Trash2 size={14} /></button>
                    </div>
                    <p className="text-[11px] text-gray-400 mb-3">
                      {[item.tab, new Date(item.createdAt).toLocaleString()].filter(Boolean).join(" · ")}
                    </p>
                    {/* Sanitised at capture and again when restored (lib/outputSanitize). */}
                    <div className="output-body overflow-x-auto" dangerouslySetInnerHTML={{ __html: item.html }} />
                    <textarea value={item.note} onChange={(e) => setNote(item.id, e.target.value)}
                      placeholder="Note (included in the export)"
                      className="mt-3 w-full text-sm border border-gray-200 rounded p-2 focus:outline-none focus:border-indigo-400"
                      rows={2} />
                  </section>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
