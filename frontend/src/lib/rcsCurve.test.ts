import { describe, expect, it } from 'vitest'
import { describeRegions, distributionAxis, distributionTrace, excludedRegions } from './rcsCurve'

describe('excludedRegions', () => {
  const x = [0, 1, 2, 3, 4, 5, 6]

  it('finds a region above the reference and interpolates its edges', () => {
    const lo = [0.5, 0.8, 1.2, 1.4, 1.2, 0.8, 0.5]
    const hi = lo.map((v) => v + 1)
    const regions = excludedRegions(x, lo, hi, 1)
    expect(regions).toHaveLength(1)
    expect(regions[0].side).toBe('above')
    expect(regions[0].from).toBeCloseTo(1.5)
    expect(regions[0].to).toBeCloseTo(4.5)
  })

  it('finds regions below on both ends and keeps open ends at the grid limits', () => {
    const hi = [0.6, 0.9, 1.5, 2, 1.5, 0.9, 0.6]
    const lo = hi.map((v) => v - 1)
    const regions = excludedRegions(x, lo, hi, 1)
    expect(regions.map((r) => r.side)).toEqual(['below', 'below'])
    expect(regions[0].from).toBe(0)
    expect(regions[0].to).toBeCloseTo(1 + 0.1 / 0.6)
    expect(regions[1].to).toBe(6)
  })

  it('returns nothing when the band always contains the reference', () => {
    expect(excludedRegions(x, x.map(() => 0.5), x.map(() => 1.5), 1)).toEqual([])
  })

  it('works around zero for a mean difference', () => {
    const lo = [-3, -2, -1, 0.5, 1]
    const hi = [-1, -0.5, 1, 2, 3]
    const regions = excludedRegions([0, 1, 2, 3, 4], lo, hi, 0)
    expect(regions.map((r) => r.side)).toEqual(['below', 'above'])
  })

  it('describes regions in reading order', () => {
    expect(describeRegions([
      { from: 38.12, to: 45, side: 'below' },
      { from: 70.33, to: 150.4, side: 'above' },
    ])).toBe('38.1 to 45.0 (lower), 70.3 to 150 (higher)')
  })
})

describe('distribution layer', () => {
  const dist = { edges: [0, 1, 2, 3], counts: [4, 10, 6], n_outside: 2 }

  it('draws one bar per bin at the bin centre on the secondary axis', () => {
    const t = distributionTrace(dist) as unknown as { x: number[]; y: number[]; width: number[]; yaxis: string }
    expect(t.x).toEqual([0.5, 1.5, 2.5])
    expect(t.y).toEqual([4, 10, 6])
    expect(t.width).toEqual([1, 1, 1])
    expect(t.yaxis).toBe('y2')
  })

  it('scales the hidden axis so bars stay in the lower part of the plot', () => {
    const axis = distributionAxis(dist) as { range: number[]; overlaying: string; showticklabels: boolean }
    expect(axis.range).toEqual([0, 40])
    expect(axis.overlaying).toBe('y')
    expect(axis.showticklabels).toBe(false)
  })
})
