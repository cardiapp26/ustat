import { describe, expect, it } from 'vitest'
import { fmtP, fmtPFull, fmtPubP, fmtPubPHtml, pCellTitle, warningText } from './format'

describe('fmtP', () => {
  it('returns em-dash for null/undefined', () => {
    expect(fmtP(null)).toBe('—')
    expect(fmtP(undefined)).toBe('—')
  })

  it('returns em-dash for NaN', () => {
    expect(fmtP(NaN)).toBe('—')
  })

  it('shows "<0.001" below the reporting floor, never "0.000"', () => {
    expect(fmtP(0.0009)).toBe('<0.001')
    expect(fmtP(0)).toBe('<0.001')
  })

  it('formats to 3 decimals otherwise', () => {
    expect(fmtP(0.035)).toBe('0.035')
    expect(fmtP(0.0431)).toBe('0.043')
    expect(fmtP(1)).toBe('1.000')
  })

  it('is the boundary-inclusive at exactly 0.001', () => {
    expect(fmtP(0.001)).toBe('0.001')
  })
})

describe('fmtPubP', () => {
  it('returns em-dash for null', () => {
    expect(fmtPubP(null)).toBe('—')
  })

  it('prefixes with p< below the floor', () => {
    expect(fmtPubP(0.0001)).toBe('p<0.001')
  })

  it('prefixes with p= otherwise', () => {
    expect(fmtPubP(0.035)).toBe('p=0.035')
  })
})

describe('fmtPFull', () => {
  it('returns em-dash for null', () => {
    expect(fmtPFull(null)).toBe('—')
  })

  it('uses scientific notation below 1e-4', () => {
    expect(fmtPFull(0.00001234)).toBe('1.234e-5')
  })

  it('trims trailing zeros for coarser values', () => {
    expect(fmtPFull(0.035)).toBe('0.035')
    expect(fmtPFull(0.5)).toBe('0.5')
  })
})

describe('pCellTitle', () => {
  it('renders a "p = —" placeholder for null', () => {
    expect(pCellTitle(null)).toBe('p = —')
  })

  it('renders full-precision value', () => {
    expect(pCellTitle(0.035)).toBe('p = 0.035')
  })
})

describe('fmtPubPHtml', () => {
  it('passes through the em-dash unmodified', () => {
    expect(fmtPubPHtml(null)).toBe('—')
  })

  it('wraps the leading "p" in <i> for Plotly text', () => {
    expect(fmtPubPHtml(0.035)).toBe('<i>p</i>=0.035')
    expect(fmtPubPHtml(0.0001)).toBe('<i>p</i><0.001')
  })
})

describe('warningText', () => {
  it('passes a plain string through', () => {
    expect(warningText('Levene p = 0.03')).toBe('Levene p = 0.03')
  })

  it('renders the category-health object that crashed the Tests tab', () => {
    // React error #31: this exact shape reached a panel that rendered it
    // directly and took the whole tab down with it.
    const w = {
      variable: 'cp',
      rare_levels: [{ level: 'x', n: 1 }],
      kept_levels: [{ level: 'A', n: 200 }],
      note: "'cp' has 1 category(ies) with <10 rows.",
    }
    expect(warningText(w)).toBe("'cp' has 1 category(ies) with <10 rows.")
  })

  it('prefers message over note', () => {
    expect(warningText({ message: 'first', note: 'second' })).toBe('first')
  })

  it('summarises an object that carries no prose', () => {
    expect(warningText({ n_dropped: 3, type: 'log_axis' })).toBe('n_dropped: 3, type: log_axis')
  })

  it('joins a list of warnings', () => {
    expect(warningText(['a', { note: 'b' }])).toBe('a · b')
  })

  it('never throws on odd input', () => {
    expect(warningText(null)).toBe('')
    expect(warningText(undefined)).toBe('')
    expect(warningText(42)).toBe('42')
    expect(typeof warningText({ nested: { deep: 1 } })).toBe('string')
  })
})
