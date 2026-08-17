/**
 * Thumbnail glyphs for the Chart Type list. The shapes themselves live in
 * ./chartGlyphs so that this file can export only the component (fast refresh
 * requires it) while the glyph table stays importable by the coverage test.
 */
import { GLYPHS } from "./chartGlyphs";

export default function ChartTypeIcon({ type, className = "" }: { type: string; className?: string }) {
  const glyph = GLYPHS[type];
  if (!glyph) return null;
  return (
    <svg
      viewBox="0 0 24 16"
      className={className}
      // Decorative: the label beside it already names the chart, and having a
      // screen reader announce it twice is noise, not information.
      aria-hidden="true"
      focusable="false"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.35}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {glyph}
    </svg>
  );
}
