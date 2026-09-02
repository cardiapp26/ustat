import { describe, expect, it } from 'vitest'
import { applySeriesPins, applyHighlight, baseLayout, ordinalLadder } from './plotStyle'
import { DEFAULT_THEME, PALETTES, paletteOf, type PlotTheme } from './store'
import { PRESET_ORDER, THEME_PRESETS, legendLayout } from './lib/plotPresets'

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

describe('baseLayout presets', () => {
  const at = (preset: PlotTheme['preset'], showGrid = true) =>
    baseLayout({ ...DEFAULT_THEME, preset }, showGrid)

  it('keeps the frame every chart already had under the default preset', () => {
    // A theme saved before presets existed loads into "minimal" and must
    // draw exactly as before: white panel, faint grid, no axis lines.
    const l = at('minimal')
    expect(l.plot_bgcolor).toBe('#ffffff')
    expect(l.xaxis).toMatchObject({ gridcolor: '#e5e7eb', showline: false, zeroline: false })
    expect(l.yaxis).toEqual(l.xaxis)
    expect(l.legend).toBeUndefined()
    expect(l.showlegend).toBeUndefined()
  })

  it('theme_classic draws left and bottom axis lines and no box', () => {
    const l = at('classic')
    expect(l.xaxis).toMatchObject({ showline: true, mirror: false, ticks: 'outside' })
  })

  it('theme_bw boxes the panel on all four sides', () => {
    expect(at('bw').xaxis).toMatchObject({ showline: true, mirror: true })
  })

  it('the grid toggle still wins over the preset grid colour', () => {
    expect((at('gray', false).xaxis as { gridcolor: string }).gridcolor).toBe('transparent')
    expect((at('gray', true).xaxis as { gridcolor: string }).gridcolor).toBe('#ffffff')
  })

  it('every preset carries the panel colour the theme bar applies with it', () => {
    for (const p of PRESET_ORDER) {
      expect(THEME_PRESETS[p].plotBg).toMatch(/^#[0-9a-f]{6}$/i)
    }
  })

  it('caller overrides win over the preset', () => {
    const l = baseLayout({ ...DEFAULT_THEME, preset: 'bw' }, true, { plot_bgcolor: '#000000' })
    expect(l.plot_bgcolor).toBe('#000000')
  })
})

describe('legend position', () => {
  it('is left alone on auto', () => {
    expect(legendLayout('auto')).toEqual({})
  })

  it('hides the legend on none and places it otherwise', () => {
    expect(legendLayout('none')).toEqual({ showlegend: false })
    expect(legendLayout('bottom').legend).toMatchObject({ orientation: 'h', yanchor: 'top' })
    expect(legendLayout('inside').legend).toMatchObject({ xanchor: 'right', yanchor: 'top' })
    expect(legendLayout('right').legend).toMatchObject({ orientation: 'v' })
  })

  it('reaches the base layout', () => {
    const l = baseLayout({ ...DEFAULT_THEME, legendPosition: 'none' }, true)
    expect(l.showlegend).toBe(false)
  })
})

describe('ordinalLadder', () => {
  it('runs light to dark in one hue, one step per level', () => {
    const ladder = ordinalLadder('#334eac', 4)
    expect(ladder).toHaveLength(4)
    // Lightness falls monotonically: each step carries more ink than the last.
    const lum = (hex: string) => parseInt(hex.slice(1, 3), 16) + parseInt(hex.slice(3, 5), 16) + parseInt(hex.slice(5, 7), 16)
    for (let i = 1; i < ladder.length; i++) expect(lum(ladder[i])).toBeLessThan(lum(ladder[i - 1]))
    // All valid hex, none white or black.
    for (const c of ladder) expect(c).toMatch(/^#[0-9a-f]{6}$/)
    expect(ladder[0]).not.toBe('#ffffff')
    expect(ladder[3]).not.toBe('#000000')
  })

  it('keeps a grey base grey', () => {
    const ladder = ordinalLadder('#8f8e88', 3)
    for (const c of ladder) {
      const [r, g, b] = [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16))
      // A warm grey keeps its faint warmth, but never turns into a colour.
      expect(Math.max(r, g, b) - Math.min(r, g, b)).toBeLessThanOrEqual(8)
    }
  })

  it('hands the base back for a single level or a colour it cannot read', () => {
    expect(ordinalLadder('#334eac', 1)).toEqual(['#334eac'])
    expect(ordinalLadder('rgba(1,2,3,0.5)', 4)).toEqual(['rgba(1,2,3,0.5)'])
  })
})

describe('applyHighlight', () => {
  const traces = [
    { name: 'Control', marker: { color: '#6366f1' } },
    { name: 'Treatment', marker: { color: '#f59e0b' }, fillcolor: '#f59e0b55' },
    { marker: { color: '#10b981' } },
    { name: 'By value', marker: { color: [1, 2, 3] } },
  ]

  it('colours the named group and greys every other named series', () => {
    const out = applyHighlight(traces, 'Treatment', '#f5572f')!
    expect((out[0].marker as { color: string }).color).toBe('#b0afa9')
    expect((out[1].marker as { color: string }).color).toBe('#f5572f')
    expect(out[1].fillcolor).toBe('#f5572f55')
  })

  it('leaves unnamed traces and array colours alone', () => {
    const out = applyHighlight(traces, 'Treatment', '#f5572f')!
    expect((out[2].marker as { color: string }).color).toBe('#10b981')
    expect((out[3].marker as { color: number[] }).color).toEqual([1, 2, 3])
  })

  it('does nothing when the group is not on the chart, or blank', () => {
    expect(applyHighlight(traces, 'Placebo', '#f5572f')).toBe(traces)
    expect(applyHighlight(traces, '  ', '#f5572f')).toBe(traces)
  })
})

describe('grayscale palette', () => {
  it('is a ladder with no repeated grey', () => {
    expect(new Set(PALETTES.grayscale).size).toBe(PALETTES.grayscale.length)
  })
})
