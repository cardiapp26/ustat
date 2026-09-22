import { act, renderHook } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import { useStore } from '../store'
import api from '../api'
import { useStampedResult } from '../hooks/useStampedResult'
import { resetRequestLog } from './requestLog'

beforeEach(() => { resetRequestLog(); installSession() })
afterEach(() => { clearSession(); useStore.setState({ panelCache: {}, savedAnalyses: [] }) })

describe('a stamped result records the request that returned it', () => {
  it('captures method, URL and body with the session as {sid}', async () => {
    server.use(http.post('/api/survival_advanced/fine_gray', () => HttpResponse.json({ n: 10 })))
    const view = renderHook(() => useStampedResult<{ n: number }>('survival_fg', { a: 1 }))
    const res = await api.post('/api/survival_advanced/fine_gray', { session_id: 'test-session', duration_col: 't' })
    act(() => view.result.current.setResult(res.data))
    expect(view.result.current.stamp?.request).toEqual({
      method: 'POST', url: '/api/survival_advanced/fine_gray', body: { session_id: '{sid}', duration_col: 't' },
    })
  })

  it('records nothing for a result the panel reshaped', async () => {
    server.use(http.post('/api/survival_advanced/fine_gray', () => HttpResponse.json({ n: 10 })))
    const view = renderHook(() => useStampedResult<{ n: number; extra: boolean }>('survival_fg', { a: 1 }))
    const res = await api.post('/api/survival_advanced/fine_gray', { session_id: 'test-session' })
    act(() => view.result.current.setResult({ ...res.data, extra: true }))
    expect(view.result.current.stamp?.request).toBeNull()
  })
})

describe('a result is stamped with the data it was sent against', () => {
  it('reads as out of date when the data changed while the request was in flight', async () => {
    let release!: () => void
    const gate = new Promise<void>((r) => { release = r })
    server.use(http.post('/api/models/linear', async () => { await gate; return HttpResponse.json({ r2: 0.4 }) }))
    const view = renderHook(() => useStampedResult<{ r2: number }>('models', { a: 1 }))
    const pending = api.post('/api/models/linear', { session_id: 'test-session' })
    act(() => useStore.getState().bumpDataVersion())  // an edit while the fit runs
    release()
    const res = await pending
    act(() => view.result.current.setResult(res.data))
    expect(view.result.current.staleReasons).toEqual(['data'])
  })
})

describe('a request is recorded against the session it was sent to', () => {
  it('writes that session as {sid} even if another dataset opened while it ran', async () => {
    let release!: () => void
    const gate = new Promise<void>((r) => { release = r })
    server.use(http.post('/api/models/linear', async () => { await gate; return HttpResponse.json({ r2: 0.4 }) }))
    const view = renderHook(() => useStampedResult<{ r2: number }>('models', { a: 1 }))
    const pending = api.post('/api/models/linear', { session_id: 'test-session' })
    const s = useStore.getState()
    act(() => useStore.setState({ session: { ...s.session!, session_id: 'another' } }))
    release()
    const res = await pending
    act(() => view.result.current.setResult(res.data))
    expect(view.result.current.stamp?.request?.body).toEqual({ session_id: '{sid}' })
  })
})

describe('rerunAnalysis', () => {
  const kept = (request: unknown) => ({
    id: 'a1', name: 'Fine-Gray', panel: 'survival_fg', tab: 'tests', createdAt: 1,
    snapshot: {
      durationCol: 't',
      result: { n: 10 },
      stamp: { dataVersion: 0, filterKey: 'none', paramsKey: '{"a":1}', engine: 'python', engineVersion: 'dev', at: 1, request },
    },
  })

  it('re-runs the recorded request on the open data and restores the fresh result where it lives', async () => {
    let sent: unknown = null
    server.use(http.post('/api/survival_advanced/fine_gray', async ({ request }) => {
      sent = await request.json()
      return HttpResponse.json({ n: 12 })
    }))
    useStore.setState({
      savedAnalyses: [kept({ method: 'POST', url: '/api/survival_advanced/fine_gray', body: { session_id: '{sid}', duration_col: 't' } })],
      dataVersion: 5,
    })
    const epoch = useStore.getState().restoreEpoch

    await act(() => useStore.getState().rerunAnalysis('a1'))

    expect(sent).toEqual({ session_id: 'test-session', duration_col: 't' })
    const s = useStore.getState()
    const entry = s.panelCache.survival_fg as { result: unknown; stamp: { dataVersion: number; paramsKey: string; sessionId: string }; durationCol: string }
    expect(entry.result).toEqual({ n: 12 })
    expect(entry.durationCol).toBe('t')
    expect(entry.stamp.dataVersion).toBe(5)
    expect(entry.stamp.paramsKey).toBe('{"a":1}')
    expect(entry.stamp.sessionId).toBe('test-session')
    // The kept analysis itself now holds the fresh result.
    expect((s.savedAnalyses[0].snapshot as { result: unknown }).result).toEqual({ n: 12 })
    // Its own tab and sub-tab, not the tab it was kept from.
    expect(s.activeTab).toBe('models')
    expect((s.panelCache.combo_models as { sub: string }).sub).toBe('survival')
    expect(s.restoreEpoch).toBe(epoch + 1)
  })

  it('stamps a re-run with the data it was sent against, not the data when it landed', async () => {
    let release!: () => void
    const gate = new Promise<void>((r) => { release = r })
    server.use(http.post('/api/survival_advanced/fine_gray', async () => { await gate; return HttpResponse.json({ n: 12 }) }))
    useStore.setState({
      savedAnalyses: [kept({ method: 'POST', url: '/api/survival_advanced/fine_gray', body: { session_id: '{sid}' } })],
      dataVersion: 5,
    })
    const pending = useStore.getState().rerunAnalysis('a1')
    await new Promise((r) => setTimeout(r, 0))
    act(() => useStore.getState().bumpDataVersion())  // an edit while it runs
    release()
    await act(() => pending)
    expect((useStore.getState().panelCache.survival_fg as { stamp: { dataVersion: number } }).stamp.dataVersion).toBe(5)
  })

  it('refuses, saying what to do, for an analysis kept without its request', async () => {
    useStore.setState({ savedAnalyses: [kept(null)] })
    await expect(useStore.getState().rerunAnalysis('a1')).rejects.toThrow(/press Recompute/)
  })
})
