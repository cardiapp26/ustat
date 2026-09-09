/** The params a saved analysis ran with, recovered from its stamp.
 *  paramsKey is stableStringify output, which is valid JSON by construction. */
import type { SavedAnalysis } from "../store";

export function paramsOf(analysis: SavedAnalysis): Record<string, unknown> {
  const snapshot = analysis.snapshot as { stamp?: { paramsKey?: string } } | null;
  const key = snapshot?.stamp?.paramsKey;
  if (typeof key !== "string") return {};
  try {
    const parsed = JSON.parse(key);
    return typeof parsed === "object" && parsed !== null ? parsed : {};
  } catch {
    return {};
  }
}
