import { describe, expect, it } from 'vitest'
import { changesAnalysisData, formatMissingCodes, parseMissingCodes } from './missingCodes'

describe('parseMissingCodes', () => {
  it('splits on commas, semicolons and spaces and drops repeats', () => {
    expect(parseMissingCodes('99, 999;8  99')).toEqual(['99', '999', '8'])
    expect(parseMissingCodes(' ')).toEqual([])
  })

  it('keeps text codes as typed', () => {
    expect(parseMissingCodes('UNK, -99')).toEqual(['UNK', '-99'])
  })
})

describe('formatMissingCodes', () => {
  it('joins for the input box', () => {
    expect(formatMissingCodes(['99', '999'])).toBe('99, 999')
    expect(formatMissingCodes(undefined)).toBe('')
  })
})

describe('changesAnalysisData', () => {
  it('is true for missing codes and category order, false for labels', () => {
    expect(changesAnalysisData({ missing_codes: [] }, { missing_codes: ['99'] })).toBe(true)
    expect(changesAnalysisData({ level_order: ['a', 'b'] }, { level_order: [] })).toBe(true)
    expect(changesAnalysisData({ label: 'Age' }, { label: 'Age (years)' })).toBe(false)
    expect(changesAnalysisData({}, { missing_codes: [], level_order: [] })).toBe(false)
  })
})
