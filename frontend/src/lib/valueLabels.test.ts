import { describe, expect, it } from 'vitest'
import { hasLabel, labelFor } from './valueLabels'

// Keys as the Value Labels dialog writes them: it reads the grid preview,
// which holds JSON numbers, so a code of zero is stored as "0".
const HISTO = {
  '0': 'Benign',
  '1': 'Papiller tiroid kanseri',
  '1.7': 'Papiller varyant',
  '8': 'Diğer',
}

describe('labelFor', () => {
  it('resolves the float spelling an analysis endpoint sends', () => {
    // The reported bug: /column_summary stringifies a float64 code as "0.0".
    expect(labelFor(HISTO, '0.0')).toBe('Benign')
    expect(labelFor(HISTO, '1.0')).toBe('Papiller tiroid kanseri')
    expect(labelFor(HISTO, '8.0')).toBe('Diğer')
  })

  it('resolves the integer spelling the grid sends', () => {
    expect(labelFor(HISTO, 0)).toBe('Benign')
    expect(labelFor(HISTO, '8')).toBe('Diğer')
  })

  it('keeps a fractional code distinct from its neighbours', () => {
    // 1.7 and 1.8 are separate categories in the reported dataset. Collapsing
    // them to 1 or to each other would merge two histologies.
    expect(labelFor(HISTO, '1.7')).toBe('Papiller varyant')
    expect(labelFor(HISTO, '1.8')).toBe('1.8')
    expect(labelFor(HISTO, '1.70')).toBe('Papiller varyant')
  })

  it('falls back to the value when nothing is labelled', () => {
    expect(labelFor(HISTO, '2.0')).toBe('2.0')
    expect(labelFor(undefined, '2.0')).toBe('2.0')
    expect(labelFor({}, 'other')).toBe('other')
  })

  it('honours an explicit fallback', () => {
    expect(labelFor(HISTO, '2.0', 'Unlabelled')).toBe('Unlabelled')
  })

  it('never resolves a blank label', () => {
    // The dialog writes an empty string for a value the user skipped. Showing
    // that would replace a visible code with nothing at all.
    expect(labelFor({ '0': '' }, '0.0')).toBe('0.0')
    expect(hasLabel({ '0': '' }, 0)).toBe(false)
  })

  it('leaves missing values alone', () => {
    expect(labelFor(HISTO, null)).toBe('')
    expect(labelFor(HISTO, undefined, 'Missing')).toBe('Missing')
    expect(labelFor(HISTO, '')).toBe('')
  })

  it('does not treat text as a number', () => {
    expect(labelFor({ Benign: 'İyi huylu' }, 'Benign')).toBe('İyi huylu')
    expect(labelFor({ '0': 'zero' }, 'O')).toBe('O')
  })

  it('prefers an exact hit over a normalised one', () => {
    // Both spellings labelled is a legacy session written by the two editors
    // that disagreed. Whatever the caller actually asked for wins.
    const mixed = { '0': 'from the grid', '0.0': 'from the dictionary' }
    expect(labelFor(mixed, '0.0')).toBe('from the dictionary')
    expect(labelFor(mixed, '0')).toBe('from the grid')
  })

  it('reports whether a value carries a label', () => {
    expect(hasLabel(HISTO, '0.0')).toBe(true)
    expect(hasLabel(HISTO, '2.0')).toBe(false)
    expect(hasLabel(undefined, '0')).toBe(false)
  })
})
