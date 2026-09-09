import { beforeEach, describe, expect, it } from 'vitest'
import { collectUiState, applyUiState } from './projectUiState'
import { useStore } from '../store'
import { installSession } from '../test/testUtils'

function stamp(dataVersion: number, extra: Record<string, unknown> = {}) {
  return {
    dataVersion,
    filterKey: 'none',
    paramsKey: '{}',
    engine: 'python',
    engineVersion: 'dev',
    at: 123,
    ...extra,
  }
}

describe('collectUiState', () => {
  beforeEach(() => installSession())

  it('captures dataVersion, activeTab and panelCache', () => {
    useStore.setState({
      dataVersion: 5,
      activeTab: 'models',
      panelCache: { models: { result: { or: 1.2 }, stamp: stamp(5) } },
    })
    const ui = collectUiState()
    expect(ui.dataVersion).toBe(5)
    expect(ui.activeTab).toBe('models')
    expect((ui.panelCache.models as { result: unknown }).result).toEqual({ or: 1.2 })
    expect(ui.table1Result).toBeUndefined()
  })
})

describe('applyUiState', () => {
  beforeEach(() => installSession())

  it('rebases fresh stamps to the new session counter (0) and keeps stale ones stale (-1)', () => {
    applyUiState({
      dataVersion: 7,
      panelCache: {
        fresh: { result: { p: 0.03 }, stamp: stamp(7) },
        stale: { result: { p: 0.9 }, stamp: stamp(4) },
        noStamp: { selections: { y: 'AGE' } },
      },
    })
    const cache = useStore.getState().panelCache as Record<string, { stamp?: { dataVersion: number } }>
    expect(cache.fresh.stamp?.dataVersion).toBe(0)
    expect(cache.stale.stamp?.dataVersion).toBe(-1)
    expect(cache.noStamp).toEqual({ selections: { y: 'AGE' } })
  })

  it('restores the active tab when present', () => {
    applyUiState({ dataVersion: 1, panelCache: {}, activeTab: 'roc' })
    expect(useStore.getState().activeTab).toBe('roc')
  })

  it('ignores payloads that do not look like a ui_state', () => {
    useStore.setState({ panelCache: { keep: { a: 1 } }, activeTab: 'data' })
    applyUiState(null)
    applyUiState('garbage')
    applyUiState({ panelCache: {} }) // missing dataVersion
    applyUiState({ dataVersion: 2 }) // missing panelCache
    expect(useStore.getState().panelCache).toEqual({ keep: { a: 1 } })
    expect(useStore.getState().activeTab).toBe('data')
  })

  it('round-trips saved analyses, rebasing the stamp inside each snapshot', () => {
    useStore.setState({
      dataVersion: 7,
      savedAnalyses: [
        { id: 'a1', name: 'Model 1', panel: 'models', tab: 'models', createdAt: 1, snapshot: { result: { or: 2 }, stamp: stamp(7) } },
        { id: 'a2', name: 'Old fit', panel: 'models', tab: 'models', createdAt: 2, snapshot: { result: { or: 3 }, stamp: stamp(4) } },
      ],
    })
    const ui = collectUiState()
    expect(ui.savedAnalyses).toHaveLength(2)

    useStore.setState({ savedAnalyses: [], dataVersion: 0 })
    applyUiState(JSON.parse(JSON.stringify(ui)))
    const restored = useStore.getState().savedAnalyses
    expect(restored).toHaveLength(2)
    expect((restored[0].snapshot as { stamp: { dataVersion: number } }).stamp.dataVersion).toBe(0)
    expect((restored[1].snapshot as { stamp: { dataVersion: number } }).stamp.dataVersion).toBe(-1)
  })

  it('drops malformed saved-analysis entries instead of restoring them broken', () => {
    applyUiState({
      dataVersion: 1,
      panelCache: {},
      savedAnalyses: [{ id: 'ok', name: 'n', panel: 'p', tab: 't', createdAt: 1, snapshot: null }, { junk: true }, null],
    })
    const restored = useStore.getState().savedAnalyses
    expect(restored).toHaveLength(1)
    expect(restored[0].id).toBe('ok')
  })

  it('leaves preserved filter and params keys untouched so those comparisons still work', () => {
    applyUiState({
      dataVersion: 2,
      panelCache: { p: { stamp: stamp(2, { filterKey: 'f-key', paramsKey: 'p-key' }) } },
    })
    const cache = useStore.getState().panelCache as Record<string, { stamp: { filterKey: string; paramsKey: string } }>
    expect(cache.p.stamp.filterKey).toBe('f-key')
    expect(cache.p.stamp.paramsKey).toBe('p-key')
  })
})
