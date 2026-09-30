/**
 * Minimal baseline TIFF (Rev 6.0) encoder.
 *
 * Why this exists: journals frequently ask for TIFF figures, but
 * Plotly's `downloadImage` only emits PNG / JPEG / WebP / SVG. We
 * rasterise the chart to PNG via Plotly, paint it onto a canvas to
 * obtain the RGB pixel buffer, then write an uncompressed baseline
 * RGB TIFF strip in pure JavaScript.
 *
 * Output:
 *   - Little-endian (II)
 *   - 8 bits per sample, 3 samples per pixel (RGB)
 *   - Compression = none (1)
 *   - PhotometricInterpretation = RGB (2)
 *   - Single uncompressed strip
 *   - Resolution stored as XResolution / YResolution / ResolutionUnit=2 (inches)
 *
 * Transparent areas are composited onto a white background — TIFF
 * baseline has no alpha channel.
 */

interface TiffOptions {
  width: number;
  height: number;
  rgba: Uint8ClampedArray; // length = w * h * 4 (RGBA, row-major)
  dpi: number;
}

const TAGS = {
  ImageWidth: 256,
  ImageLength: 257,
  BitsPerSample: 258,
  Compression: 259,
  PhotometricInterpretation: 262,
  StripOffsets: 273,
  Orientation: 274,
  SamplesPerPixel: 277,
  RowsPerStrip: 278,
  StripByteCounts: 279,
  XResolution: 282,
  YResolution: 283,
  ResolutionUnit: 296,
} as const;

const TYPE_SHORT = 3;
const TYPE_LONG = 4;
const TYPE_RATIONAL = 5;

export function encodeTiff(opts: TiffOptions): Uint8Array {
  const { width, height, rgba, dpi } = opts;

  // Composite RGBA → RGB onto white background. Most matplotlib /
  // Plotly exports are opaque, but transparent corners (e.g. legend
  // boxes set to "rgba(...)") would otherwise read black.
  const pixelBytes = width * height * 3;
  const stripData = new Uint8Array(pixelBytes);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    const a = rgba[i + 3] / 255;
    const inv = 1 - a;
    stripData[j]     = Math.round(rgba[i]     * a + 255 * inv);
    stripData[j + 1] = Math.round(rgba[i + 1] * a + 255 * inv);
    stripData[j + 2] = Math.round(rgba[i + 2] * a + 255 * inv);
  }

  // Lay out the file:
  //   [0..8)             header
  //   [8..)              IFD (2 + N*12 + 4 bytes) + external data (BitsPerSample, XRes, YRes) + strip
  const numEntries = 13;
  const ifdStart = 8;
  const ifdSize = 2 + numEntries * 12 + 4;          // 2 for count, 4 for next-IFD pointer
  const bpsOffset = ifdStart + ifdSize;             // 3 SHORTs (6 bytes)
  const xResOffset = bpsOffset + 6;                 // 2 LONGs (8 bytes)
  const yResOffset = xResOffset + 8;                // 2 LONGs (8 bytes)
  const stripOffset = yResOffset + 8;
  const totalSize = stripOffset + stripData.length;

  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  // ── Header ────────────────────────────────────────────────────────────
  view.setUint16(0, 0x4949, true);   // II  (little-endian)
  view.setUint16(2, 42, true);        // magic
  view.setUint32(4, ifdStart, true);  // pointer to IFD

  // ── IFD ───────────────────────────────────────────────────────────────
  view.setUint16(ifdStart, numEntries, true);
  let p = ifdStart + 2;
  const writeEntry = (tag: number, type: number, count: number, valueOrOffset: number, inlineByte = false) => {
    view.setUint16(p, tag, true);
    view.setUint16(p + 2, type, true);
    view.setUint32(p + 4, count, true);
    // For SHORT inline values, Plotly's convention is to right-pad — TIFF
    // spec: small values are stored left-justified in the 4-byte field.
    if (inlineByte && type === TYPE_SHORT) {
      view.setUint16(p + 8, valueOrOffset, true);
      view.setUint16(p + 10, 0, true);
    } else {
      view.setUint32(p + 8, valueOrOffset, true);
    }
    p += 12;
  };

  writeEntry(TAGS.ImageWidth,                 TYPE_LONG,     1, width);
  writeEntry(TAGS.ImageLength,                TYPE_LONG,     1, height);
  writeEntry(TAGS.BitsPerSample,              TYPE_SHORT,    3, bpsOffset);
  writeEntry(TAGS.Compression,                TYPE_SHORT,    1, 1, true);
  writeEntry(TAGS.PhotometricInterpretation,  TYPE_SHORT,    1, 2, true);
  writeEntry(TAGS.StripOffsets,               TYPE_LONG,     1, stripOffset);
  writeEntry(TAGS.Orientation,                TYPE_SHORT,    1, 1, true);
  writeEntry(TAGS.SamplesPerPixel,            TYPE_SHORT,    1, 3, true);
  writeEntry(TAGS.RowsPerStrip,               TYPE_LONG,     1, height);
  writeEntry(TAGS.StripByteCounts,            TYPE_LONG,     1, pixelBytes);
  writeEntry(TAGS.XResolution,                TYPE_RATIONAL, 1, xResOffset);
  writeEntry(TAGS.YResolution,                TYPE_RATIONAL, 1, yResOffset);
  writeEntry(TAGS.ResolutionUnit,             TYPE_SHORT,    1, 2, true); // inches

  // Next IFD = 0 (single image)
  view.setUint32(p, 0, true);

  // ── External tag data ─────────────────────────────────────────────────
  // BitsPerSample = [8, 8, 8]
  view.setUint16(bpsOffset,     8, true);
  view.setUint16(bpsOffset + 2, 8, true);
  view.setUint16(bpsOffset + 4, 8, true);

  // XResolution / YResolution as RATIONAL (two LONGs: numerator, denominator)
  view.setUint32(xResOffset,     dpi, true);
  view.setUint32(xResOffset + 4, 1,   true);
  view.setUint32(yResOffset,     dpi, true);
  view.setUint32(yResOffset + 4, 1,   true);

  // ── Strip data ────────────────────────────────────────────────────────
  u8.set(stripData, stripOffset);

  return u8;
}

/**
 * Convenience: rasterise the given Plotly graph div to a TIFF blob at
 * the requested pixel size and DPI. The browser handles PNG decode +
 * canvas paint; we just convert the RGBA buffer to a baseline TIFF.
 */
export async function plotlyToTiffBlob(
  graphDiv: HTMLElement,
  opts: {
    width: number; height: number; dpi: number; filename?: string;
    /** Plotly scale; defaults to dpi / 72. A print-width export passes the
     *  scale that lands the layout on the column's exact pixel count. */
    scale?: number;
  },
): Promise<Blob> {
  // Prefer the Plotly instance react-plotly.js already attached to the
  // gd (saves a dynamic import + side-steps the ESM tree-shake bug);
  // fall back to plotly.js/dist/plotly only when _Plotly isn't present.
  interface PlotlyToImage {
    toImage: (
      gd: HTMLElement,
      opts: { format: string; width: number; height: number; scale: number; setBackground: string },
    ) => Promise<string>;
  }
  let Plotly = (graphDiv as HTMLElement & { _Plotly?: PlotlyToImage })._Plotly;
  if (!Plotly?.toImage) {
    Plotly = (await import("plotly.js/dist/plotly")).default as unknown as PlotlyToImage;
  }
  const scale = opts.scale ?? opts.dpi / 72;
  const dataUrl: string = await Plotly.toImage(graphDiv, {
    format: "png",
    width: opts.width,
    height: opts.height,
    // Runtime supports `scale` even if the published d.ts is missing it.
    scale,
    setBackground: "opaque",
  });

  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error("PNG decode failed"));
    el.src = dataUrl;
  });

  const pxW = img.naturalWidth;
  const pxH = img.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = pxW;
  canvas.height = pxH;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d context unavailable");
  // Paint onto white background — TIFF baseline has no alpha channel.
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, pxW, pxH);
  ctx.drawImage(img, 0, 0);
  const imgData = ctx.getImageData(0, 0, pxW, pxH);

  const tiff = encodeTiff({
    width: pxW,
    height: pxH,
    rgba: imgData.data,
    dpi: opts.dpi,
  });

  // Copy into a fresh ArrayBuffer to satisfy DOM's BlobPart typing
  // (Uint8Array<ArrayBufferLike> is not assignable to BlobPart in TS 5.7+).
  const ab = new ArrayBuffer(tiff.byteLength);
  new Uint8Array(ab).set(tiff);
  return new Blob([ab], { type: "image/tiff" });
}

/**
 * Trigger a browser download for a Blob.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 100);
}
