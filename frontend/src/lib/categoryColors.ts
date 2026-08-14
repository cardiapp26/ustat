/** Enough distinct colours for every level of a categorical variable.
 *
 * The palettes hold eight colours. A histology column with eleven levels was
 * drawn by cycling them, so the largest slice and one of the smallest came out
 * in the same purple and the legend gave no way to tell which was which. A
 * colour that repeats inside one chart is not a colour: it makes two
 * categories look like one.
 *
 * Past the end of the palette the hues are reused at a different lightness, so
 * the repeat is visible as a lighter or darker variant rather than an exact
 * duplicate. Slices that far down the list are small by construction — the
 * list is ordered by count — so the variant only ever has to be distinguished
 * from its own base colour, not read on its own.
 */

/** Mix ``hex`` toward white (``amount`` > 0) or black (``amount`` < 0). */
function shade(hex: string, amount: number): string {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex.trim());
  if (!m) return hex;
  const channel = (pair: string) => {
    const v = parseInt(pair, 16);
    const target = amount > 0 ? 255 : 0;
    const mixed = Math.round(v + (target - v) * Math.abs(amount));
    return Math.max(0, Math.min(255, mixed)).toString(16).padStart(2, "0");
  };
  return `#${channel(m[1])}${channel(m[2])}${channel(m[3])}`;
}

/** ``count`` colours, extending ``palette`` by lightness when it runs out. */
export function categoryColors(palette: string[], count: number): string[] {
  const base = palette.filter(Boolean);
  if (base.length === 0) return [];
  // Each pass past the first lightens, then darkens, by a widening step.
  const steps = [0, 0.45, -0.3, 0.68, -0.5];
  const out: string[] = [];
  for (let i = 0; i < count; i++) {
    const pass = Math.floor(i / base.length);
    const step = steps[Math.min(pass, steps.length - 1)];
    out.push(step === 0 ? base[i % base.length] : shade(base[i % base.length], step));
  }
  return out;
}
