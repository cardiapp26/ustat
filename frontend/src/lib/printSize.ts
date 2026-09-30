/**
 * Print sizing for figure export: journal column widths, the Plotly scale
 * that hits them, and the DPI stamp that tells Word and journal portals how
 * big the image is meant to be.
 *
 * Two widths are in play and they are deliberately separate. The LAYOUT width
 * (CSS px) is what Plotly lays the chart out at: margins, label wrapping and
 * text size are all fixed relative to it. The PRINT width (mm) is how wide
 * the figure is on the page. Output pixels = print inches x DPI, so the scale
 * handed to Plotly is that pixel count over the layout width. A narrower
 * layout at the same print width means larger printed text, which is the knob
 * a user turns when a journal says "no text under 6 pt".
 */

export type PrintSizeId = "custom" | "col1" | "col15" | "col2";

export interface PrintSize {
  label: string;
  /** Print width in mm; null for free pixel sizing. */
  widthMm: number | null;
}

/** Nature / Springer column widths; Elsevier (90 / 140 / 190 mm), Cell and
 *  JAMA sit within a few mm of these. */
export const PRINT_SIZES: Record<PrintSizeId, PrintSize> = {
  custom: { label: "Custom (px)", widthMm: null },
  col1: { label: "1 column · 89 mm", widthMm: 89 },
  col15: { label: "1.5 column · 120 mm", widthMm: 120 },
  col2: { label: "2 columns · 183 mm", widthMm: 183 },
};

export const PRINT_SIZE_ORDER: PrintSizeId[] = ["custom", "col1", "col15", "col2"];

const MM_PER_INCH = 25.4;
const PT_PER_INCH = 72;

/** Printed size most journals accept as the smallest legible figure text. */
export const MIN_PRINT_PT = 6;
/** Text size the suggested layout width aims for. */
const TARGET_PRINT_PT = 7;

/** Output width in pixels for a print width at a DPI. */
export function printPixels(widthMm: number, dpi: number): number {
  return Math.round((widthMm / MM_PER_INCH) * dpi);
}

/**
 * Plotly `scale` for an export. Free sizing keeps the old meaning (1 layout px
 * = 1 pt, so scale = dpi / 72); a print width scales the layout to exactly
 * the pixel count that width needs at this DPI.
 */
export function exportScale(layoutWidth: number, dpi: number, widthMm: number | null): number {
  if (widthMm === null || layoutWidth <= 0) return dpi / PT_PER_INCH;
  return printPixels(widthMm, dpi) / layoutWidth;
}

/** Printed size, in pt, of text drawn at `fontPx` in a layout this wide. */
export function printedPt(fontPx: number, layoutWidth: number, widthMm: number | null): number {
  if (widthMm === null || layoutWidth <= 0) return fontPx;
  return fontPx * ((widthMm / MM_PER_INCH) * PT_PER_INCH) / layoutWidth;
}

/** Layout width at which `fontPx` text prints at about 7 pt. */
export function suggestedLayoutWidth(fontPx: number, widthMm: number): number {
  return Math.round(fontPx * ((widthMm / MM_PER_INCH) * PT_PER_INCH) / TARGET_PRINT_PT);
}

/** Print height in mm that keeps the layout's aspect ratio. */
export function printHeightMm(widthMm: number, layoutWidth: number, layoutHeight: number): number {
  return layoutWidth > 0 ? (widthMm * layoutHeight) / layoutWidth : 0;
}

// ── PNG pHYs ───────────────────────────────────────────────────────────────

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
/** 8-byte signature + IHDR chunk (4 length + 4 type + 13 data + 4 CRC). */
const IHDR_END = 8 + 25;

/**
 * Stamp a PNG with its resolution (a pHYs chunk right after IHDR).
 *
 * Plotly's PNGs carry no resolution, so Word places a 1051 px image as if it
 * were 96 dpi, 11 inches wide, and journal portals report "72 dpi" and bounce
 * it. With pHYs the same pixels arrive as 3.5 inches at 300 dpi. Any pHYs
 * already present is dropped so the stamp is not duplicated. Bytes that are
 * not a PNG come back unchanged.
 */
export function withPngDpi(png: Uint8Array, dpi: number): Uint8Array {
  if (png.length < IHDR_END || PNG_SIGNATURE.some((b, i) => png[i] !== b)) return png;

  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const kept: Uint8Array[] = [];
  for (let off = IHDR_END; off + 12 <= png.length; ) {
    const len = view.getUint32(off);
    const end = off + 12 + len;
    const type = String.fromCharCode(...png.subarray(off + 4, off + 8));
    if (type !== "pHYs") kept.push(png.subarray(off, end));
    off = end;
  }

  const ppm = Math.round(dpi / 0.0254);
  const chunk = new Uint8Array(21);
  const cv = new DataView(chunk.buffer);
  cv.setUint32(0, 9);
  chunk.set([0x70, 0x48, 0x59, 0x73], 4); // "pHYs"
  cv.setUint32(8, ppm);
  cv.setUint32(12, ppm);
  chunk[16] = 1; // unit: metre
  cv.setUint32(17, crc32(chunk.subarray(4, 17)));

  const total = IHDR_END + chunk.length + kept.reduce((n, c) => n + c.length, 0);
  const out = new Uint8Array(total);
  out.set(png.subarray(0, IHDR_END), 0);
  out.set(chunk, IHDR_END);
  let pos = IHDR_END + chunk.length;
  for (const c of kept) {
    out.set(c, pos);
    pos += c.length;
  }
  return out;
}
