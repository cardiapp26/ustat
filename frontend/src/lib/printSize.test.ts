import { describe, expect, it } from 'vitest'
import {
  PRINT_SIZES, exportScale, printHeightMm, printPixels, printedPt, suggestedLayoutWidth, withPngDpi,
} from './printSize'

describe('print sizing', () => {
  it('turns a column width and DPI into pixels', () => {
    // 89 mm = 3.504 in; at 300 dpi that is 1051 px.
    expect(printPixels(89, 300)).toBe(1051)
    expect(printPixels(183, 600)).toBe(4323)
  })

  it('keeps the old dpi / 72 scale for free sizing', () => {
    expect(exportScale(1200, 300, null)).toBeCloseTo(300 / 72)
  })

  it('scales any layout width to exactly the column in pixels', () => {
    for (const layout of [300, 450, 900]) {
      expect(Math.round(layout * exportScale(layout, 300, 89))).toBe(1051)
    }
  })

  it('reports printed text size, larger for a narrower layout', () => {
    const wide = printedPt(11, 900, 89)
    const narrow = printedPt(11, 400, 89)
    expect(narrow).toBeGreaterThan(wide)
    // 11 px in a 252 px layout printed 89 mm wide: 1 px = 1 pt.
    expect(printedPt(11, 252.28, 89)).toBeCloseTo(11, 1)
    expect(printedPt(11, 1200, null)).toBe(11)
  })

  it('suggests a layout width that prints the base text at about 7 pt', () => {
    const w = suggestedLayoutWidth(11, PRINT_SIZES.col1.widthMm!)
    expect(printedPt(11, w, 89)).toBeCloseTo(7, 1)
  })

  it('keeps the aspect ratio in the print height', () => {
    expect(printHeightMm(89, 400, 300)).toBeCloseTo(66.75)
  })
})

describe('withPngDpi', () => {
  // 1x1 RGB PNG: signature, IHDR, IEND (CRCs from zlib).
  const png = Uint8Array.from([
    137, 80, 78, 71, 13, 10, 26, 10,
    0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0, 144, 119, 83, 222,
    0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
  ])
  // pHYs for 300 dpi (11811 px/m, unit metre), CRC from zlib.
  const phys300 = [0, 0, 0, 9, 112, 72, 89, 115, 0, 0, 46, 35, 0, 0, 46, 35, 1, 120, 165, 63, 118]

  it('inserts a pHYs chunk right after IHDR', () => {
    const out = withPngDpi(png, 300)
    expect(Array.from(out.subarray(0, 33))).toEqual(Array.from(png.subarray(0, 33)))
    expect(Array.from(out.subarray(33, 54))).toEqual(phys300)
    expect(Array.from(out.subarray(54))).toEqual(Array.from(png.subarray(33)))
  })

  it('replaces an existing pHYs instead of adding a second', () => {
    const twice = withPngDpi(withPngDpi(png, 72), 300)
    expect(twice.length).toBe(png.length + 21)
    expect(Array.from(twice.subarray(33, 54))).toEqual(phys300)
  })

  it('leaves non-PNG bytes alone', () => {
    const jpeg = Uint8Array.from([0xff, 0xd8, 0xff, 0xe0])
    expect(withPngDpi(jpeg, 300)).toBe(jpeg)
  })
})
