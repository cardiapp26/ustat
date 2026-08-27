import { useRef, useState } from "react";
import type { ColMeta } from "../store";
import {
  parseWideDictionaryRows,
  suggestDictionaryColumnMatches,
  type DictionaryCell,
  type DictionaryColumnMatch,
  type ParsedDictionary,
} from "../lib/dictionaryValueLabels";

interface XlsxWorkbook {
  SheetNames: string[];
  Sheets: Record<string, unknown>;
}

interface XlsxUtils {
  sheet_to_json: (sheet: unknown, options: Record<string, unknown>) => unknown[][];
}

interface XlsxModule {
  read: (data: ArrayBuffer, options: Record<string, unknown>) => XlsxWorkbook;
  utils?: XlsxUtils;
}

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_ROWS = 10_000;
const MAX_COLUMNS = 500;

export interface ValueLabelImportResult {
  labelsByTarget: Record<string, Record<string, string>>;
  sourceColumns: number;
  labelCount: number;
}

export default function DictionaryValueLabelImport({
  columns,
  onApply,
}: {
  columns: ColMeta[];
  onApply: (result: ValueLabelImportResult) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [parsed, setParsed] = useState<ParsedDictionary | null>(null);
  const [matches, setMatches] = useState<DictionaryColumnMatch[]>([]);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [filename, setFilename] = useState("");
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState("");
  const loadVersion = useRef(0);

  const reset = () => {
    loadVersion.current += 1;
    setParsed(null);
    setMatches([]);
    setMapping({});
    setFilename("");
    setLoading(false);
    setError("");
  };

  const close = () => {
    if (applying) return;
    setOpen(false);
    reset();
  };

  const loadFile = async (file: File) => {
    reset();
    const requestVersion = loadVersion.current;
    setFilename(file.name);
    if (file.size > MAX_FILE_BYTES) {
      setError("Dictionary file must be 10 MB or smaller.");
      return;
    }
    setLoading(true);
    try {
      const mod = (await import("xlsx")) as XlsxModule & { default?: XlsxModule };
      const XLSX: XlsxModule | undefined = mod?.utils ? mod : mod?.default;
      if (!XLSX?.utils?.sheet_to_json) throw new Error("XLSX reader is unavailable.");
      const workbook = XLSX.read(await file.arrayBuffer(), { type: "array" });
      const firstSheetName = workbook.SheetNames[0];
      if (!firstSheetName) throw new Error("Dictionary workbook has no sheets.");
      const rows = XLSX.utils.sheet_to_json(workbook.Sheets[firstSheetName], {
        header: 1,
        raw: false,
        defval: "",
        blankrows: false,
      }) as DictionaryCell[][];
      if (rows.length > MAX_ROWS || (rows[0]?.length ?? 0) > MAX_COLUMNS) {
        throw new Error(`Dictionary exceeds ${MAX_ROWS} rows or ${MAX_COLUMNS} columns.`);
      }
      const nextParsed = parseWideDictionaryRows(rows);
      if (nextParsed.columns.length === 0) {
        throw new Error("No value-label columns found. Expected variable names in row 1 and definitions below them.");
      }
      const nextMatches = suggestDictionaryColumnMatches(nextParsed.columns, columns);
      if (requestVersion !== loadVersion.current) return;
      setParsed(nextParsed);
      setMatches(nextMatches);
      setMapping(Object.fromEntries(nextMatches.map((match) => [match.sourceName, match.targetName])));
    } catch (caught) {
      if (requestVersion === loadVersion.current) {
        setError(caught instanceof Error ? caught.message : "Dictionary could not be read.");
      }
    } finally {
      if (requestVersion === loadVersion.current) setLoading(false);
    }
  };

  const selectedTargets = Object.values(mapping).filter(Boolean);
  const seenTargets = new Set<string>();
  const duplicateTargets = new Set<string>();
  for (const target of selectedTargets) {
    if (seenTargets.has(target)) duplicateTargets.add(target);
    seenTargets.add(target);
  }

  const warnings = parsed
    ? [
        ...parsed.warnings,
        ...parsed.columns.flatMap((column) =>
          column.warnings.map((warning) => `${column.sourceName}, ${warning}`),
        ),
      ]
    : [];

  const apply = async () => {
    if (!parsed || duplicateTargets.size > 0) return;
    const labelsByTarget: Record<string, Record<string, string>> = {};
    let sourceColumns = 0;
    let labelCount = 0;
    for (const definition of parsed.columns) {
      const target = mapping[definition.sourceName];
      if (!target) continue;
      labelsByTarget[target] = { ...definition.labels };
      sourceColumns += 1;
      labelCount += Object.keys(definition.labels).length;
    }
    if (sourceColumns === 0) return;

    setApplying(true);
    setError("");
    try {
      await onApply({ labelsByTarget, sourceColumns, labelCount });
      setApplying(false);
      setOpen(false);
      reset();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Value labels could not be saved.");
      setApplying(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="text-xs px-3 py-1.5 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50"
      >
        Import value labels
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" role="presentation">
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="dictionary-import-title"
            className="w-full max-w-3xl max-h-[85vh] overflow-y-auto rounded-xl bg-white shadow-xl border border-gray-200 p-5 space-y-4"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 id="dictionary-import-title" className="text-sm font-semibold text-gray-900">Import value labels</h3>
                <p className="text-xs text-gray-500 mt-1">
                  Row 1 contains variable names. Cells below use label:code or code=label.
                </p>
              </div>
              <button onClick={close} className="text-gray-400 hover:text-gray-700" aria-label="Close import">×</button>
            </div>

            <label className="block rounded-lg border border-dashed border-gray-300 p-3 text-xs text-gray-600 hover:bg-gray-50">
              <span className="font-medium">Dictionary file</span>
              <span className="ml-2 text-gray-400">XLSX, XLS, or CSV, first sheet</span>
              <input
                aria-label="Dictionary file"
                type="file"
                accept=".xlsx,.xls,.csv"
                className="block mt-2 text-xs"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void loadFile(file);
                }}
              />
            </label>

            {loading && <p className="text-xs text-gray-500" role="status">Reading dictionary…</p>}
            {error && <p className="text-xs text-red-600 rounded bg-red-50 px-3 py-2" role="alert">{error}</p>}

            {parsed && (
              <>
                <div className="flex flex-wrap gap-3 text-xs text-gray-600">
                  <span className="font-medium">{filename}</span>
                  <span>{parsed.columns.length} source variables</span>
                  <span>{selectedTargets.length} mapped</span>
                </div>

                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="w-full text-xs">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="text-left px-3 py-2 font-medium text-gray-500">Dictionary column</th>
                        <th className="text-left px-3 py-2 font-medium text-gray-500">Labels</th>
                        <th className="text-left px-3 py-2 font-medium text-gray-500">Dataset variable</th>
                        <th className="text-left px-3 py-2 font-medium text-gray-500">Match</th>
                      </tr>
                    </thead>
                    <tbody>
                      {parsed.columns.map((definition) => {
                        const match = matches.find((item) => item.sourceName === definition.sourceName);
                        const selected = mapping[definition.sourceName] ?? "";
                        const autoSelected = selected && selected === match?.targetName;
                        return (
                          <tr key={definition.sourceName} className="border-t border-gray-100">
                            <td className="px-3 py-2 font-medium text-gray-800">{definition.sourceName}</td>
                            <td className="px-3 py-2 text-gray-500">
                              <div>{Object.keys(definition.labels).length}</div>
                              <div className="max-w-48 truncate text-[10px] text-gray-400" title={Object.entries(definition.labels).map(([code, label]) => `${code} → ${label}`).join("; ")}>
                                {Object.entries(definition.labels).slice(0, 3).map(([code, label]) => `${code} → ${label}`).join("; ")}
                              </div>
                            </td>
                            <td className="px-3 py-2">
                              <select
                                aria-label={`Map ${definition.sourceName}`}
                                value={selected}
                                onChange={(event) => setMapping((current) => ({
                                  ...current,
                                  [definition.sourceName]: event.target.value,
                                }))}
                                className="w-full rounded border border-gray-200 px-2 py-1 bg-white"
                              >
                                <option value="">Skip</option>
                                {columns.map((column) => (
                                  <option key={column.name} value={column.name}>
                                    {column.name}{column.label ? `, ${column.label}` : ""}
                                  </option>
                                ))}
                              </select>
                            </td>
                            <td className="px-3 py-2 text-gray-500">
                              {autoSelected && match
                                ? `${match.reason}, ${Math.round(match.score * 100)}%`
                                : selected ? "manual" : "unmatched"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {duplicateTargets.size > 0 && (
                  <p className="text-xs text-red-600" role="alert">
                    Each dataset variable can receive one dictionary column. Duplicate: {Array.from(duplicateTargets).join(", ")}
                  </p>
                )}

                {warnings.length > 0 && (
                  <details className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
                    <summary className="cursor-pointer font-medium">{warnings.length} parsing warning{warnings.length === 1 ? "" : "s"}</summary>
                    <ul className="list-disc pl-5 mt-2 space-y-1">
                      {warnings.slice(0, 20).map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}
                    </ul>
                  </details>
                )}

                <div className="flex items-center justify-between gap-4">
                  <p className="text-[11px] text-gray-500">
                    Imported labels overwrite the same codes. Other existing labels remain.
                  </p>
                  <div className="flex gap-2">
                    <button onClick={close} disabled={applying} className="text-xs px-3 py-1.5 rounded border border-gray-200">Cancel</button>
                    <button
                      onClick={() => void apply()}
                      disabled={applying || selectedTargets.length === 0 || duplicateTargets.size > 0}
                      className="btn-primary text-xs px-4 py-1.5 disabled:opacity-50"
                    >
                      {applying ? "Saving…" : `Import ${selectedTargets.length} variable${selectedTargets.length === 1 ? "" : "s"}`}
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
