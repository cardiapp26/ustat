import { describe, expect, it } from 'vitest'
import { applySeriesPins } from './plotStyle'
import { DEFAULT_THEME, PALETTES, paletteOf } from './store'

describe('paletteOf', () => {
  it('returns the named palette', () => {
    expect(paletteOf({ ...DEFAULT_THEME, palette: 'jama' })).toEqual(PALETTES.jama)
  })

  it('returns the live custom list, not the module constant', () => {
    const mine = ['#000000', '#ffffff']
    expect(paletteOf({ ...DEFAULT_THEME, palette: 'custom', customPalette: mine })).toEqual(mine)
  })

  it('falls back when the custom list is empty', () => {
    // An empty colourway hands Plotly nothing to cycle and every trace comes
    // out the same default blue.
    expect(paletteOf({ ...DEFAULT_THEME, palette: 'custom', customPalette: [] })).toEqual(PALETTES.indigo)
  })
})

describe('applySeriesPins', () => {
  const pins = { Female: '#c0392b' }

  it('repaints the trace whose name is pinned and leaves the others', () => {
    const [a, b] = applySeriesPins(
      [
        { name: 'Female', marker: { color: '#6366f1', size: 4 } },
        { name: 'Male', marker: { color: '#f59e0b', size: 4 } },
      ],
      pins,
    )!
    expect((a.marker as { color: string }).color).toBe('#c0392b')
    expect((a.marker as { size: number }).size).toBe(4)
    expect((b.marker as { color: string }).color).toBe('#f59e0b')
  })

  it('keeps a translucent fill translucent', () => {
    // A violin's fill is the palette colour plus an alpha suffix; repainting
    // it whole would turn a wash into a solid block.
    const [t] = applySeriesPins([{ name: 'Female', fillcolor: '#6366f155' }], pins)!
    expect(t.fillcolor).toBe('#c0392b55')
  })

  it('repaints line and marker-outline colours too', () => {
    const [t] = applySeriesPins(
      [{ name: 'Female', line: { color: '#6366f1', width: 1 }, marker: { line: { color: '#6366f1' } } }],
      pins,
    )!
    expect((t.line as { color: string }).color).toBe('#c0392b')
    expect((t.marker as { line: { color: string } }).line.color).toBe('#c0392b')
    expect((t.line as { width: number }).width).toBe(1)
  })

  it('matches the label exactly, as printed in the legend', () => {
    const [t] = applySeriesPins([{ name: 'female', marker: { color: '#6366f1' } }], pins)!
    expect((t.marker as { color: string }).color).toBe('#6366f1')
  })

  it('is a no-op with no pins, and survives a null trace list', () => {
    const traces = [{ name: 'Female', marker: { color: '#6366f1' } }]
    expect(applySeriesPins(traces, {})).toBe(traces)
    expect(applySeriesPins(null, pins)).toBeNull()
  })
})
