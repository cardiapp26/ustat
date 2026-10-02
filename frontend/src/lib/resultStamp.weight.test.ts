import { describe, expect, it } from 'vitest'
import { filterKey, makeStamp, staleReasons } from './resultStamp'

const filter = { conditions: [{ column: 'arm', operator: 'eq', value: 'A', join: 'AND' }], selected: 3, total: 6 }

describe('filterKey with Weight Cases', () => {
  it('is unchanged when no weight is set, so existing stamps stay current', () => {
    expect(filterKey(null)).toBe('none')
    expect(filterKey(filter as never, null)).toBe(filterKey(filter as never))
  })

  it('differs once a weight is set, and between weight columns', () => {
    const none = filterKey(null)
    const a = filterKey(null, 'count')
    const b = filterKey(null, 'n')
    expect(new Set([none, a, b]).size).toBe(3)
  })

  it('makes a result computed unweighted read as out of date after weighting', () => {
    const base = { dataVersion: 1, caseFilter: null, engine: 'python' as const, params: { x: 1 }, sessionId: 's' }
    const before = makeStamp(base)
    const after = makeStamp({ ...base, caseWeight: 'count' })
    expect(staleReasons(before, after).length).toBeGreaterThan(0)
  })
})
