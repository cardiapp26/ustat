/** Treatment-column candidates for PSM / IPTW: 0/1 columns and other two-level columns (text or 1/2 coded). */

type PreviewRow = Record<string, unknown>;

/**
 * Column name to its two levels as text (sorted), or `null` for a numeric 0/1
 * column. Columns that are neither are left out.
 */
export function treatmentCandidates(preview: PreviewRow[], cols: string[]): Record<string, string[] | null> {
  const out: Record<string, string[] | null> = {};
  for (const col of cols) {
    const vals = new Set(preview.map((r) => r[col]).filter((v) => v != null && v !== ""));
    if (vals.size !== 2) continue;
    const list = [...vals];
    out[col] = list.every((v) => v === 0 || v === 1) ? null : list.map(String).sort();
  }
  return out;
}

/** `treated_level` for the request: only for a text treatment with a chosen level. */
export function treatedLevelParam(levels: string[] | null | undefined, chosen: string): { treated_level?: string } {
  return levels && levels.includes(chosen) ? { treated_level: chosen } : {};
}
