/**
 * Shared dataset-export helpers used by the header Save dropdown.
 * Lifted out of DataTable so the same code path is the single source
 * of truth for CSV / XLSX / SPSS / TSV exports and session JSON.
 */
import api from "../api";

export type ExportFmt = "csv" | "tsv" | "xlsx" | "sav";

interface MinimalCol {
  name: string;
  kind: string;
}

interface MinimalSession {
  session_id: string;
  filename?: string;
}

/** Download the full dataset as CSV / TSV / XLSX / SAV. */
export async function exportDataset(
  session: MinimalSession,
  columns: MinimalCol[],
  fmt: ExportFmt,
): Promise<void> {
  const base = (session.filename ?? "data").replace(/\.[^.]+$/, "");
  const colKinds = encodeURIComponent(JSON.stringify(
    Object.fromEntries(columns.map((c) => [c.name, c.kind])),
  ));
  const url = `/api/sessions/${session.session_id}/export?fmt=${fmt}&filename=${encodeURIComponent(base)}&col_kinds=${colKinds}`;
  try {
    const res = await api.get(url, { responseType: "blob" });
    const ct = (res.headers["content-type"] || "").toString();
    if (ct.includes("application/json")) {
      const txt = await (res.data as Blob).text();
      throw new Error(`Server returned JSON instead of ${fmt.toUpperCase()}: ${txt.slice(0, 200)}`);
    }
    const mime = fmt === "xlsx" ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
               : fmt === "sav"  ? "application/x-spss-sav"
               : fmt === "tsv"  ? "text/tab-separated-values"
               : "text/csv";
    const blob = new Blob([res.data], { type: mime });
    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = `${base}.${fmt}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  } catch (e: unknown) {
    let detail = e instanceof Error ? e.message : String(e);
    const blobBody = (e as { response?: { data?: Blob } })?.response?.data;
    if (blobBody instanceof Blob) {
      try {
        const txt = await blobBody.text();
        const parsed = JSON.parse(txt);
        detail = parsed?.detail ?? txt.slice(0, 200);
      } catch {
        /* not JSON, leave detail as-is */
      }
    }
    console.error(`Export as ${fmt} failed:`, e);
    alert(`Export as ${fmt.toUpperCase()} failed: ${detail}`);
  }
}

/** Download the session as a .ustat project file (docs/DESIGN_project_file.md):
 *  data + dictionary + prep recipe + audit + ingest originals, in a zip whose
 *  manifest records engine identity and a dataset hash. */
export async function downloadProjectFile(session: MinimalSession): Promise<void> {
  try {
    // Panel settings and stamped results ride along in ui/state.json, so
    // reopening the file restores the analyses, not just the data.
    const { collectUiState } = await import("./projectUiState");
    const res = await api.post(
      `/api/project/${session.session_id}/save`,
      { ui_state: collectUiState() },
      { responseType: "blob" },
    );
    const blob = new Blob([res.data], { type: "application/zip" });
    const url = URL.createObjectURL(blob);
    const base = (session.filename ?? "project").replace(/\.[^.]+$/, "");
    const a = document.createElement("a");
    a.href = url;
    a.download = `${base}.ustat`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e: unknown) {
    console.error("Save project failed:", e);
    alert(`Save project failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

/** Download the replay script (.py): import + recorded prep steps + export,
 *  plus the saved analyses as definitions. Generated server-side from the
 *  prep recipe; see backend/services/script_export.py. */
export async function downloadAnalysisScript(session: MinimalSession, lang: "python" | "r" = "python"): Promise<void> {
  try {
    const { collectUiState } = await import("./projectUiState");
    const res = await api.post(
      `/api/project/${session.session_id}/script?lang=${lang}`,
      { ui_state: collectUiState() },
      { responseType: "blob" },
    );
    const blob = new Blob([res.data], { type: lang === "python" ? "text/x-python" : "text/x-r" });
    const url = URL.createObjectURL(blob);
    const base = (session.filename ?? "project").replace(/\.[^.]+$/, "");
    const a = document.createElement("a");
    a.href = url;
    a.download = `${base}_replay.${lang === "python" ? "py" : "R"}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e: unknown) {
    console.error("Script export failed:", e);
    alert(`Script export failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

/** Download the session JSON (data + labels + filters + audit). */
export async function downloadSessionJson(session: MinimalSession): Promise<void> {
  try {
    const res = await api.get(`/api/sessions/${session.session_id}/save_session`, { responseType: "blob" });
    const blob = new Blob([res.data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const base = (session.filename ?? "session").replace(/\.[^.]+$/, "");
    const a = document.createElement("a");
    a.href = url;
    a.download = `${base}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e: unknown) {
    console.error("Save session failed:", e);
    alert(`Save session failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}
