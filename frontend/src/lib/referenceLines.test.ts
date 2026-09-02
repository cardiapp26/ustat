import { describe, expect, it } from 'vitest'
import { parseRefValue, referenceLineOverlay } from './referenceLines'

describe('parseRefValue', () => {
  it('reads a number, tolerating a decimal comma', () => {
    expect(parseRefValue('100')).toBe(100)
    expect(parseRefValue(' 2,5 ')).toBe(2.5)
  })

  it('returns null for blank or non-numeric text', () => {
    expect(parseRefValue('')).toBeNull()
    expect(parseRefValue('abc')).toBeNull()
  })
})

describe('referenceLineOverlay', () => {
  it('draws a horizontal line in y data coordinates across the full width', () => {
    const { shapes, annotations } = referenceLineOverlay([{ axis: 'y', value: 100, label: 'LDL target' }])
    expect(shapes).toHaveLength(1)
    expect(shapes[0]).toMatchObject({ xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 100, y1: 100 })
    expect(annotations[0]).toMatchObject({ text: 'LDL target', y: 100, yref: 'y' })
  })

  it('draws a vertical line in x data coordinates across the full height', () => {
    const { shapes } = referenceLineOverlay([{ axis: 'x', value: 30, label: '' }])
    expect(shapes[0]).toMatchObject({ xref: 'x', yref: 'paper', x0: 30, x1: 30, y0: 0, y1: 1 })
  })

  it('labels an unlabelled line with its value', () => {
    const { annotations } = referenceLineOverlay([{ axis: 'x', value: 30, label: '  ' }])
    expect(annotations[0].text).toBe('30')
  })

  it('skips a line without a finite value and keeps the rest', () => {
    const { shapes } = referenceLineOverlay([
      { axis: 'y', value: Number.NaN, label: 'bad' },
      { axis: 'y', value: 1, label: 'ok' },
    ])
    expect(shapes).toHaveLength(1)
    expect(shapes[0]).toMatchObject({ y0: 1 })
  })

  it('is empty with no lines', () => {
    expect(referenceLineOverlay([])).toEqual({ shapes: [], annotations: [] })
  })
})
