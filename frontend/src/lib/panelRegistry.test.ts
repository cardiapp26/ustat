import { describe, expect, it } from 'vitest'
import { locatePanel, panelLabel } from './panelRegistry'

// Every panel's source, to find the cache keys results are stamped under.
const sources = import.meta.glob('../components/**/*.tsx', { query: '?raw', import: 'default', eager: true }) as Record<string, string>

function stampedKeys(): string[] {
  const keys = new Set<string>()
  const re = /useStampedResult(?:<[^>]*>)?\(\s*["'`]([a-z0-9_]+)["'`]/g
  for (const [path, src] of Object.entries(sources)) {
    if (path.includes('.test.')) continue
    for (const m of src.matchAll(re)) keys.add(m[1])
  }
  return [...keys]
}

describe('panelRegistry', () => {
  it('places a sub-result under its panel, sub-tab included', () => {
    expect(locatePanel('survival_km')).toEqual({
      label: 'Survival: km', tab: 'models', combo: { key: 'combo_models', sub: 'survival' },
    })
    expect(locatePanel('subgroup_bar')?.tab).toBe('visual')
    expect(locatePanel('subgroup')?.combo?.sub).toBe('subgroup')
    expect(panelLabel('nonexistent')).toBe('nonexistent')
  })

  it('knows where every stamped result lives', () => {
    const keys = stampedKeys()
    expect(keys.length).toBeGreaterThan(40)
    const unknown = keys.filter((k) => locatePanel(k) === null)
    expect(unknown).toEqual([])
  })
})
