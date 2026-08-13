/** Value-label lookup that tolerates the two spellings of a whole number.
 *
 * Value labels are keyed by whatever the Value Labels dialog saw, and that
 * dialog reads the grid preview — JSON numbers, so a code of zero becomes the
 * key "0". Most analysis endpoints stringify the same code from a float64
 * pandas column, where it becomes "0.0". The two never matched, so a fully
 * labelled column still displayed its raw codes everywhere the labels came
 * from the server: the distribution donut, Table 1's category rows, the
 * survival group names.
 *
 * Fixing one endpoint would have left the rest, and normalising every
 * endpoint's output would change strings that other code keys off. Making the
 * lookup itself tolerant is the change that cannot miss a caller: an exact hit
 * still wins, so nothing that already worked can start behaving differently.
 */

export type ValueLabels = Record<string, string>;

const WHOLE_FLOAT = /^(-?\d+)\.0+$/;
const INTEGER = /^-?\d+$/;

/** Every spelling of ``raw`` worth trying against a label map, best first. */
function candidates(raw: unknown): string[] {
  const s = String(raw).trim();
  if (!s) return [];
  const keys = [s];
  const whole = WHOLE_FLOAT.exec(s);
  if (whole) {
    keys.push(whole[1]);
  } else if (INTEGER.test(s)) {
    keys.push(`${s}.0`);
  } else {
    // "1.70" and "1.7" are the same code written by two different
    // stringifiers. Number() collapses them; it is only consulted when the
    // literal forms above have already missed.
    const n = Number(s);
    if (Number.isFinite(n) && String(n) !== s) keys.push(String(n));
  }
  return keys;
}

/**
 * The label for ``raw``, or ``fallback`` (default: ``raw`` as a string).
 *
 * Null, undefined and empty values are returned as the fallback untouched —
 * a missing cell has no code to label.
 */
export function labelFor(
  labels: ValueLabels | undefined | null,
  raw: unknown,
  fallback?: string,
): string {
  const shown = fallback ?? (raw == null ? "" : String(raw));
  if (!labels || raw == null || raw === "") return shown;
  for (const key of candidates(raw)) {
    const hit = labels[key];
    if (hit != null && hit !== "") return hit;
  }
  return shown;
}

/** True when ``raw`` has a non-empty label. */
export function hasLabel(labels: ValueLabels | undefined | null, raw: unknown): boolean {
  if (!labels || raw == null || raw === "") return false;
  return candidates(raw).some((k) => {
    const hit = labels[k];
    return hit != null && hit !== "";
  });
}
