/**
 * Export a saved session snapshot straight from IndexedDB.
 *
 * The header's Save menu exports through the backend, which needs the session
 * to be loaded there. A card on the landing page usually is not: its dataset
 * lives only in the local snapshot. Uploading it just to download it again
 * would put a copy of the data on the server that nothing then removes —
 * there is no endpoint to drop a session — so this builds the file from the
 * stored payload instead. It also means the download works with the backend
 * down, and never disturbs whatever session the user currently has open.
 *
 * SPSS (.sav) is the exception: writing it needs pyreadstat, which is
 * server-side. The snapshot goes to a stateless endpoint that converts it in
 * memory and stores nothing, so no session is left behind; it does need the
 * backend to be reachable.
 */
import { exportSnapshotSav } from "../api";

export type SnapshotFmt = "csv" | "tsv" | "xlsx" | "sav" | "json";

export const SNAPSHOT_FORMATS: { fmt: SnapshotFmt; label: string; ext: string; note?: string }[] = [
  { fmt: "csv", label: "CSV", ext: "csv" },
  { fmt: "tsv", label: "TSV", ext: "tsv" },
  { fmt: "xlsx", label: "Excel", ext: "xlsx" },
  // The one format that leaves the browser, and the card says its snapshot
  // never does, so the menu says so too.
  {
    fmt: "sav", label: "SPSS (.sav)", ext: "sav",
    note: "Converted on the uSTAT server, in memory; nothing is stored there.",
  },
  { fmt: "json", label: "Session (JSON)", ext: "json" },
];

interface SnapshotColumn {
  name: string;
}

interface SavedSessionPayload {
  filename?: string;
  columns?: SnapshotColumn[];
  data?: Record<string, unknown>[];
}

export interface ExportableSnapshot {
  name: string;
  payload: string;
}

/** Column order comes from the payload's own column list, not from the keys
 *  of the first row: a row whose first cell is missing can omit the key
 *  entirely, which would silently drop the column or reorder the file. */
export function snapshotColumns(payload: SavedSessionPayload): string[] {
  const declared = (payload.columns ?? [])
    .map((c) => c?.name)
    .filter((n): n is string => typeof n === "string" && n.length > 0);
  if (declared.length > 0) return declared;

  const seen: string[] = [];
  const known = new Set<string>();
  for (const row of payload.data ?? []) {
    for (const key of Object.keys(row ?? {})) {
      if (!known.has(key)) {
        known.add(key);
        seen.push(key);
      }
    }
  }
  return seen;
}

/** A missing value is an empty field, never the text "null" or "NaN".
 *  Written as a word it stops being missing and becomes a category. */
export function cellToText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function escapeDelimited(text: string, delimiter: string): string {
  if (text.includes('"') || text.includes(delimiter) || /[\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export function buildDelimited(payload: SavedSessionPayload, delimiter: string): string {
  const cols = snapshotColumns(payload);
  const rows = payload.data ?? [];
  const lines: string[] = [
    cols.map((c) => escapeDelimited(c, delimiter)).join(delimiter),
  ];
  for (const row of rows) {
    lines.push(
      cols
        .map((c) => escapeDelimited(cellToText((row ?? {})[c]), delimiter))
        .join(delimiter),
    );
  }
  return lines.join("\r\n");
}

/** Strip one trailing extension so "cohort.xlsx" does not become
 *  "cohort.xlsx.csv". */
export function baseName(record: ExportableSnapshot, payload: SavedSessionPayload): string {
  const raw = (record.name || payload.filename || "dataset").trim();
  const stripped = raw.replace(/\.(csv|tsv|xlsx|xls|sav|json|txt)$/i, "");
  return stripped || "dataset";
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function writeXlsx(
  payload: SavedSessionPayload,
  filename: string,
): Promise<void> {
  // The xlsx package ships both an ESM build with named exports and a CJS one
  // behind `default`; resolve whichever Vite hands over, the same way
  // ResultExporter does.
  interface XlsxUtils {
    aoa_to_sheet: (data: unknown[][]) => unknown;
    book_new: () => unknown;
    book_append_sheet: (wb: unknown, ws: unknown, name: string) => void;
  }
  interface XlsxModule {
    utils?: XlsxUtils;
    write: (wb: unknown, opts: Record<string, unknown>) => ArrayBuffer;
  }
  const mod = (await import("xlsx")) as XlsxModule & { default?: XlsxModule };
  const XLSX: XlsxModule | undefined = mod?.utils ? mod : mod?.default;
  if (!XLSX?.utils?.aoa_to_sheet) {
    throw new Error("xlsx module loaded but utils.aoa_to_sheet is unavailable");
  }
  const cols = snapshotColumns(payload);
  // Cells keep their type here — a number written as text lands in Excel as
  // text and stops being a number.
  const body = (payload.data ?? []).map((row) =>
    cols.map((c) => {
      const v = (row ?? {})[c];
      if (v === null || v === undefined) return "";
      if (typeof v === "number") return Number.isFinite(v) ? v : "";
      if (typeof v === "boolean" || typeof v === "string") return v;
      return cellToText(v);
    }),
  );
  const ws = XLSX.utils.aoa_to_sheet([cols, ...body]);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Data");
  const out = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  triggerDownload(
    new Blob([out], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }),
    filename,
  );
}

/** The server's reason for a failed blob request, which arrives as a Blob. */
async function blobErrorDetail(e: unknown): Promise<string> {
  const err = e as { response?: { data?: unknown }; message?: string };
  const data = err.response?.data;
  if (data instanceof Blob) {
    try {
      const parsed = JSON.parse(await data.text()) as { detail?: unknown };
      if (typeof parsed.detail === "string") return parsed.detail;
    } catch {
      /* not JSON: fall through to the transport message */
    }
  }
  if (!err.response) return "the uSTAT server did not respond";
  return err.message ?? "unknown error";
}

async function writeSav(payload: SavedSessionPayload, filename: string): Promise<void> {
  try {
    const res = await exportSnapshotSav(payload);
    triggerDownload(res.data as Blob, filename);
  } catch (e) {
    throw new Error(`SPSS (.sav) export failed: ${await blobErrorDetail(e)}.`);
  }
}

/** Write the snapshot to disk in the requested format. */
export async function exportSnapshot(
  record: ExportableSnapshot,
  fmt: SnapshotFmt,
): Promise<void> {
  let payload: SavedSessionPayload;
  try {
    payload = JSON.parse(record.payload) as SavedSessionPayload;
  } catch {
    throw new Error("This snapshot could not be read — its saved data is not valid JSON.");
  }

  const base = baseName(record, payload);

  if (fmt === "json") {
    // Already exactly what the backend's save_session returns, so hand it
    // over untouched rather than re-serialising and risking a difference.
    triggerDownload(
      new Blob([record.payload], { type: "application/json" }),
      `${base}.json`,
    );
    return;
  }

  if (fmt === "xlsx") {
    await writeXlsx(payload, `${base}.xlsx`);
    return;
  }

  if (fmt === "sav") {
    await writeSav(payload, `${base}.sav`);
    return;
  }

  const delimiter = fmt === "tsv" ? "\t" : ",";
  // A BOM, matching what the backend's CSV/TSV export writes. Without it
  // Excel reads UTF-8 as latin-1 and Turkish characters arrive mangled.
  const text = "﻿" + buildDelimited(payload, delimiter);
  const mime = fmt === "tsv" ? "text/tab-separated-values" : "text/csv";
  triggerDownload(
    new Blob([text], { type: `${mime};charset=utf-8;` }),
    `${base}.${fmt}`,
  );
}
