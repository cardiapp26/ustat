import { describe, expect, it } from 'vitest'
import { arrangeLevels, mergeLevelOrder, swap } from './levelOrder'

describe('arrangeLevels', () => {
  it('keeps the data order when nothing is saved', () => {
    expect(arrangeLevels(['Heavy', 'Moderate', 'Trace'], undefined)).toEqual({
      levels: ['Heavy', 'Moderate', 'Trace'],
      unplaced: [],
    })
  })

  it('follows the saved order and appends levels it does not place', () => {
    expect(arrangeLevels(['Heavy', 'Moderate', 'Trace', 'New'], ['Trace', 'Moderate', 'Heavy'])).toEqual({
      levels: ['Trace', 'Moderate', 'Heavy', 'New'],
      unplaced: ['New'],
    })
  })
})

describe('mergeLevelOrder', () => {
  it('keeps the slot of a level the filtered data lacks', () => {
    // "None" is filtered out of the data; moving Heavy above Moderate must
    // not drop it or push it to the end.
    expect(mergeLevelOrder(['None', 'Trace', 'Moderate', 'Heavy'], ['Trace', 'Heavy', 'Moderate']))
      .toEqual(['None', 'Trace', 'Heavy', 'Moderate'])
  })

  it('appends newly placed levels after the saved ones', () => {
    expect(mergeLevelOrder(['A', 'B'], ['B', 'A', 'C'])).toEqual(['B', 'A', 'C'])
  })

  it('returns the present order when nothing was saved', () => {
    expect(mergeLevelOrder(undefined, ['x', 'y'])).toEqual(['x', 'y'])
  })
})

describe('swap', () => {
  it('ignores a move past either end', () => {
    expect(swap(['a', 'b'], 0, -1)).toEqual(['a', 'b'])
    expect(swap(['a', 'b'], 1, 2)).toEqual(['a', 'b'])
    expect(swap(['a', 'b'], 0, 1)).toEqual(['b', 'a'])
  })
})
