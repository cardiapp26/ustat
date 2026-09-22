/** Helpers for user-declared missing codes (metadata `missing_codes`). The
 *  server applies them; see backend/services/missing_codes.py. */
import { stableStringify } from "./resultStamp";
import type { ColMeta } from "../store";

/** "99, 999; 8" -> ["99", "999", "8"]. Commas, semicolons or spaces. */
export function parseMissingCodes(text: string): string[] {
  const seen = new Set<string>();
  return text
    .split(/[,;\s]+/)
    .map((t) => t.trim())
    .filter((t) => t !== "" && !seen.has(t) && (seen.add(t), true));
}

export function formatMissingCodes(codes: string[] | undefined): string {
  return (codes ?? []).join(", ");
}

/** Metadata that changes the numbers an analysis sees, as opposed to labels
 *  and descriptions. Saving a change to any of it makes earlier results
 *  out of date, so the caller bumps the data version. */
export function changesAnalysisData(
  before: Partial<ColMeta> | undefined,
  after: Partial<ColMeta> | undefined,
): boolean {
  const key = (m: Partial<ColMeta> | undefined) =>
    stableStringify({ codes: m?.missing_codes ?? [], order: m?.level_order ?? [] });
  return key(before) !== key(after);
}
