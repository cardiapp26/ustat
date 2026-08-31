import { renderHook, act } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { useStore } from '../store'
import { makeSession } from '../test/testUtils'
import { flushAutoSession, useAutoSession } from './useAutoSession'

const saved: { name: string; payload: string }[] = []

vi.mock('../lib/sessionDb', () => ({
  upsertRecentSession: vi.fn(async (input: { name: string; payload: string }) => {
    saved.push({ name: input.name, payload: input.payload })
    return { id: 'row-1' }
  }),
  notifySessionsChanged: vi.fn(),
}))
vi.mock('../lib/cloudSync', () => ({ cloudSync: { markDirty: vi.fn() } }))

/** The payload changes every call, so the dedupe hash never short-circuits. */
let counter = 0
beforeEach(() => {
  saved.length = 0
  counter = 0
  vi.useFakeTimers()
  server.use(
    http.get(/\/api\/sessions\/.*\/save_session/, () =>
      HttpResponse.json({ blob: `snapshot-${++counter}` }),
    ),
  )
  useStore.setState({ session: makeSession(), activeTab: 'data', dataVersion: 0 })
})
afterEach(() => {
  vi.useRealTimers()
  useStore.setState({ session: null })
})

/** Let pending promises settle between timer advances. */
async function tick(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

describe('useAutoSession', () => {
  it('saves once the debounce elapses', async () => {
    renderHook(() => useAutoSession())
    await tick(5_100)
    expect(saved.length).toBe(1)
  })

  it('still saves someone who never stops working', async () => {
    // Reported: a file worked on for a long stretch was missing from Recent
    // work, or stale, after the backend restarted. Every tracked change re-ran
    // the effect, which cleared the 5 s debounce AND the 60 s fallback and
    // started both again — so a user touching something every few seconds,
    // which is what using the app looks like, was never snapshotted at all.
    renderHook(() => useAutoSession())
    for (let i = 0; i < 30; i++) {
      await tick(3_000)                       // never long enough for the debounce
      act(() => { useStore.setState({ dataVersion: i + 1 }) })
    }
    expect(saved.length).toBeGreaterThan(0)
  })

  it('defers a save by at most the ceiling, however busy the user is', async () => {
    renderHook(() => useAutoSession())
    // Change something every second for 25 s: the debounce alone would never
    // fire, so anything saved here came from the ceiling.
    for (let i = 0; i < 25; i++) {
      await tick(1_000)
      act(() => { useStore.setState({ dataVersion: i + 1 }) })
    }
    expect(saved.length).toBeGreaterThan(0)
  })

  it('keeps the periodic fallback running across changes', async () => {
    renderHook(() => useAutoSession())
    await tick(5_100)
    const afterFirst = saved.length
    // Switching tabs used to restart the 60 s interval every time.
    for (let i = 0; i < 10; i++) {
      act(() => { useStore.setState({ activeTab: i % 2 ? 'summary' : 'data' }) })
      await tick(9_000)
    }
    expect(saved.length).toBeGreaterThan(afterFirst)
  })

  it('writes the session name with the snapshot', async () => {
    useStore.setState({ session: makeSession({ filename: 'tiroid.xlsx' }) })
    renderHook(() => useAutoSession())
    await tick(5_100)
    expect(saved.at(-1)?.name).toBe('tiroid.xlsx')
  })

  it('snapshots on demand instead of waiting out the debounce', async () => {
    // Structural edits — a dropped column, deleted rows — are what a recovery
    // cannot reconstruct. If the backend dies inside the debounce window, the
    // restored snapshot silently predates them.
    renderHook(() => useAutoSession())
    act(() => { useStore.setState({ dataVersion: 1 }) })
    await act(async () => { await flushAutoSession() })
    expect(saved.length).toBe(1)
  })

  it('takes a fresh snapshot even while one is already in flight', async () => {
    // The in-flight save was started BEFORE the mutation, so returning early
    // would leave exactly the state this is meant to protect unsaved.
    renderHook(() => useAutoSession())
    const first = tick(5_100)
    await act(async () => { await flushAutoSession() })
    await first
    expect(saved.length).toBeGreaterThanOrEqual(2)
  })

  it('is inert once the session is gone', async () => {
    useStore.setState({ session: null })
    renderHook(() => useAutoSession())
    await act(async () => { await flushAutoSession() })
    expect(saved.length).toBe(0)
  })

  it('does nothing without a session', async () => {
    useStore.setState({ session: null })
    renderHook(() => useAutoSession())
    await tick(70_000)
    expect(saved.length).toBe(0)
  })
})
