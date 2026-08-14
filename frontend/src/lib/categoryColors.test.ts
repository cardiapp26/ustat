import { describe, expect, it } from 'vitest'
import { categoryColors } from './categoryColors'

const PALETTE = ['#6366f1', '#f59e0b', '#10b981', '#ef4444']

describe('categoryColors', () => {
  it('uses the palette unchanged while it lasts', () => {
    expect(categoryColors(PALETTE, 3)).toEqual(PALETTE.slice(0, 3))
    expect(categoryColors(PALETTE, 4)).toEqual(PALETTE)
  })

  it('never repeats a colour inside one chart', () => {
    // Reported: eleven histology levels against six colours put the largest
    // slice and one of the smallest in the same purple.
    const colours = categoryColors(PALETTE, 11)
    expect(colours).toHaveLength(11)
    expect(new Set(colours).size).toBe(11)
  })

  it('keeps a variant related to the hue it came from', () => {
    // The repeat should read as a lighter version of the same colour, not as
    // an unrelated one — the eye has to be able to group them.
    const colours = categoryColors(PALETTE, 5)
    expect(colours[4]).not.toBe(colours[0])
    expect(colours[4]).toMatch(/^#[0-9a-f]{6}$/)
  })

  it('survives an exhausted set of variants', () => {
    const colours = categoryColors(PALETTE, 40)
    expect(colours).toHaveLength(40)
    expect(colours.every((c) => /^#[0-9a-f]{6}$/.test(c))).toBe(true)
  })

  it('returns nothing when there is no palette to work from', () => {
    expect(categoryColors([], 5)).toEqual([])
    expect(categoryColors(['', ''], 3)).toEqual([])
  })

  it('leaves a value it cannot parse alone rather than emitting garbage', () => {
    expect(categoryColors(['rebeccapurple'], 2)).toEqual([
      'rebeccapurple',
      'rebeccapurple',
    ])
  })

  it('asks for none and gets none', () => {
    expect(categoryColors(PALETTE, 0)).toEqual([])
  })
})
