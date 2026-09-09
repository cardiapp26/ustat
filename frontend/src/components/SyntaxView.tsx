/**
 * jamovi-style syntax view for one saved analysis: the definition as Python
 * and R source, fetched from /api/project/syntax.
 *
 * Two honesty rules, both server-enforced and repeated here because this is
 * the component that shows the code to a person:
 *  - a combination with no template gets "no translation yet" and the raw
 *    params, never a guessed formula;
 *  - every translation carries the disclaimer that uSTAT's implementation
 *    may differ in defaults, with the result's provenance line as authority.
 */
import { useEffect, useState } from "react";
import { X, Copy, Check } from "lucide-react";
import api from "../api";
import type { SavedAnalysis } from "../store";
import { paramsOf } from "../lib/analysisParams";

interface SyntaxResponse {
  title: string | null;
  python: string | null;
  r: string | null;
}

export default function SyntaxView({ analysis, onClose }: { analysis: SavedAnalysis; onClose: () => void }) {
  const [lang, setLang] = useState<"python" | "r">("python");
  const [syntax, setSyntax] = useState<SyntaxResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .post("/api/project/syntax", { panel: analysis.panel, params: paramsOf(analysis) })
      .then((res) => { if (!cancelled) setSyntax(res.data); })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Request failed");
      });
    return () => { cancelled = true; };
  }, [analysis]);

  const code = syntax ? syntax[lang] : null;

  const copy = () => {
    if (!code) return;
    void navigator.clipboard?.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div
        className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl p-5 space-y-3"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-900 text-sm truncate">
            Syntax: {analysis.name}
          </h3>
          <button onClick={onClose} className="p-1 text-gray-400 hover:text-gray-700" title="Close">
            <X size={16} />
          </button>
        </div>
        <div className="flex items-center gap-2">
          {(["python", "r"] as const).map((l) => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                lang === l ? "bg-violet-100 text-violet-700" : "text-gray-500 hover:bg-gray-100"
              }`}
            >
              {l === "python" ? "Python" : "R"}
            </button>
          ))}
          {code && (
            <button
              onClick={copy}
              className="ml-auto flex items-center gap-1 px-2 py-1 rounded-lg text-xs text-gray-500 hover:bg-gray-100"
              title="Copy code"
            >
              {copied ? <Check size={12} className="text-emerald-600" /> : <Copy size={12} />}
              {copied ? "Copied" : "Copy"}
            </button>
          )}
        </div>
        {error && <p className="text-xs text-red-500">{error}</p>}
        {!error && code && (
          <pre className="p-3 bg-gray-50 rounded-lg text-[11px] leading-relaxed overflow-x-auto max-h-96 whitespace-pre">
            {code}
          </pre>
        )}
        {!error && syntax && !code && (
          <div className="text-xs text-gray-500 space-y-2">
            <p>
              No translation for this analysis yet ({analysis.panel}). The
              parameters it ran with, so the definition is still explicit:
            </p>
            <pre className="p-3 bg-gray-50 rounded-lg text-[11px] overflow-x-auto max-h-60 whitespace-pre-wrap">
              {JSON.stringify(paramsOf(analysis), null, 2)}
            </pre>
          </div>
        )}
        {!error && !syntax && <p className="text-xs text-gray-400">Loading…</p>}
      </div>
    </div>
  );
}
