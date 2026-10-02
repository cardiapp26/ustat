import { afterEach, describe, expect, it } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useStampedResult } from './useStampedResult'
import { analysisScope, useStore } from '../store'
import { makeStamp } from '../lib/resultStamp'
import { installSession, clearSession } from '../test/testUtils'

afterEach(() => { useStore.setState({ splitFile: null }); clearSession() })

const params = { column: 'sbp' }
const levels = [{ level: 'F', n: 4 }, { level: 'M', n: 5 }]

function stampAt(level: string | null) {
  const s = useStore.getState()
  return makeStamp({
    dataVersion: s.dataVersion, caseFilter: s.caseFilter, engine: s.engine, params,
    sessionId: s.session?.session_id ?? null,
    scope: analysisScope({ caseWeight: null, splitFile: { column: 'sex', levels, level } }),
  })
}

describe('useStampedResult under Split File', () => {
  it("shows the cached result of the level switched to", () => {
    installSession()
    useStore.setState({ splitFile: { column: 'sex', levels, level: 'F' } })
    const F = { n: 4 }, M = { n: 5 }
    useStore.getState().setPanelCache('p', {
      result: F, stamp: stampAt('F'),
      splitResults: { F: { result: F, stamp: stampAt('F') }, M: { result: M, stamp: stampAt('M') } },
    })
    const { result } = renderHook(() => useStampedResult<{ n: number }>('p', params))
    expect(result.current.result).toBe(F)
    expect(result.current.stale).toBe(false)

    act(() => useStore.getState().setSplitLevel('M'))
    expect(result.current.result).toBe(M)
    expect(result.current.stale).toBe(false)
  })

  it('stays out of date for a level with no current result', () => {
    installSession()
    useStore.setState({ splitFile: { column: 'sex', levels, level: 'F' } })
    const F = { n: 4 }
    useStore.getState().setPanelCache('p', {
      result: F, stamp: stampAt('F'), splitResults: { F: { result: F, stamp: stampAt('F') } },
    })
    const { result } = renderHook(() => useStampedResult<{ n: number }>('p', params))
    act(() => useStore.getState().setSplitLevel('M'))
    expect(result.current.result).toBe(F)
    expect(result.current.stale).toBe(true)
  })
})
