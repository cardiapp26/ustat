/**
 * Centralised numeric formatters used everywhere a result is rendered.
 *
 * The p-value formatter is the canonical one — every panel should call
 * `fmtP` for the *display* string and `fmtPFull` (or the underlying
 * number) for the tooltip / clipboard / export. This keeps formatting
 * identical across panels and exports, and lets the user always see
 * the full-precision value on hover.
 */

/**
 * Canonical p-value display (journal-standard, matches the backend `format_p`).
 * Single source of truth: every panel and export should call this (or
 * `fmtPubP`) and never hand-roll `.toFixed` on a p-value.
 *   • null / NaN   → "—"
 *   • p < 0.001    → "<0.001"   (never "0.000")
 *   • otherwise    → exact 3 decimals (0.035 → "0.035", 0.043 → "0.043")
 */
export function fmtP(p: number | null | undefined): string {
  if (p == null) return "—";
  const n = typeof p === "number" ? p : parseFloat(String(p));
  if (!Number.isFinite(n)) return "—";
  if (n < 0.001) return "<0.001";
  return n.toFixed(3);
}

/**
 * Prefixed variant for publication tables / figure labels: "p<0.001" / "p=0.035".
 * Same precision as `fmtP`.
 */
export function fmtPubP(p: number | null | undefined): string {
  if (p == null) return "—";
  const n = typeof p === "number" ? p : parseFloat(String(p));
  if (!Number.isFinite(n)) return "—";
  if (n < 0.001) return "p<0.001";
  return `p=${n.toFixed(3)}`;
}

/**
 * Full-precision p-value for tooltips / hover / exports. Falls back to
 * the same 3-decimal output when the input is already coarse.
 */
export function fmtPFull(p: number | null | undefined): string {
  if (p == null) return "—";
  const n = typeof p === "number" ? p : parseFloat(String(p));
  if (!Number.isFinite(n)) return "—";
  // For very small p we show scientific notation so significand is not lost.
  if (n < 1e-4 && n > 0) return n.toExponential(3);
  return n.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

/**
 * Standard tooltip wrapper. Used by every p-value cell so the user
 * sees the full precision on hover.
 */
export function pCellTitle(p: number | null | undefined): string {
  if (p == null) return "p = —";
  return `p = ${fmtPFull(p)}`;
}

/**
 * Plotly-only variant of `fmtPubP` with the leading "p" wrapped in an
 * `<i>` tag — Plotly's `text` / `hovertemplate` render a small HTML
 * subset (unlike JSX, which needs an actual <i> element). Journal
 * convention italicises the p in "p < 0.05" / "p = 0.035".
 */
export function fmtPubPHtml(p: number | null | undefined): string {
  const s = fmtPubP(p);
  return s === "—" ? s : s.replace(/^p/, "<i>p</i>");
}

/**
 * Reduce a backend warning to text that React can render.
 *
 * Warnings are not one shape. Some endpoints emit plain strings, others emit
 * a record — `{type, message, …}` from the chart endpoints, or
 * `{variable, rare_levels, kept_levels, note}` from the category-health
 * helpers. A panel that assumed strings and rendered the value directly
 * crashed the whole tab with React error #31 ("objects are not valid as a
 * React child"), taking the results with it.
 *
 * Everything goes through here so a warning can never cost the user their
 * output. An unrecognised shape degrades to something readable rather than
 * throwing.
 */
export function warningText(w: unknown): string {
  if (w == null) return "";
  if (typeof w === "string") return w;
  if (typeof w === "number" || typeof w === "boolean") return String(w);
  if (Array.isArray(w)) return w.map(warningText).filter(Boolean).join(" · ");
  if (typeof w === "object") {
    const rec = w as Record<string, unknown>;
    // The prose fields, in the order a reader would want them.
    for (const key of ["message", "note", "detail", "text", "warning"]) {
      const v = rec[key];
      if (typeof v === "string" && v.trim()) return v.trim();
    }
    // No prose: build something legible instead of dumping [object Object].
    const parts = Object.entries(rec)
      .filter(([, v]) => v != null && typeof v !== "object")
      .map(([k, v]) => `${k}: ${String(v)}`);
    if (parts.length) return parts.join(", ");
    try {
      return JSON.stringify(w);
    } catch {
      return "Unrecognised warning";
    }
  }
  return String(w);
}
