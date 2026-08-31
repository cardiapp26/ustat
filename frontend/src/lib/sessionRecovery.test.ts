import axios from 'axios'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import { useStore } from '../store'
import { makeSession } from '../test/testUtils'
import { installSessionRecovery, resetSessionRecovery } from './sessionRecovery'

const rows: Record<string, { id: string; name: string; payload: string; savedAt: number; serverSessionId?: string }> = {}

vi.mock('./sessionDb', () => ({
  getRecentSessionByServerId: vi.fn(async (sid: string) =>
    Object.values(rows).find((r) => r.serverSessionId === sid)),
  getRecentSession: vi.fn(async (id: string) => rows[id]),
}))

let loads = 0

function client() {
  const api = axios.create({ baseURL: '' })
  installSessionRecovery(api)
  return api
}

beforeEach(() => {
  loads = 0
  resetSessionRecovery()
  for (const k of Object.keys(rows)) delete rows[k]
  rows['row-1'] = { id: 'row-1', name: 'tiroid.xlsx', payload: '{"df":1}', savedAt: 1, serverSessionId: 'dead-1' }
  useStore.setState({ session: makeSession({ session_id: 'dead-1' }), localSessionId: null, sessionRecovery: null })
  server.use(
    http.post('/api/sessions/load_session', () => {
      loads += 1
      return HttpResponse.json({ session_id: 'fresh-1', filename: 'tiroid.xlsx', columns: [], preview: [], rows: 0 })
    }),
    http.get('/api/stats/dead-1/descriptive', () =>
      HttpResponse.json({ detail: 'Session not found' }, { status: 404 })),
    http.get('/api/stats/fresh-1/descriptive', () => HttpResponse.json({ ok: true })),
  )
})
afterEach(() => {
  useStore.setState({ session: null, localSessionId: null, sessionRecovery: null })
})

describe('session recovery', () => {
  it('re-uploads the saved copy and retries when the server has forgotten the session', async () => {
    // Datasets live in the backend's memory and are never written to disk, so a
    // restart leaves every open id pointing at nothing while the browser still
    // holds the whole dataset. That used to surface as "Session not found".
    const res = await client().get('/api/stats/dead-1/descriptive')
    expect(res.data).toEqual({ ok: true })
    expect(loads).toBe(1)
    expect(useStore.getState().session?.session_id).toBe('fresh-1')
  })

  it('records what was restored, so the swap is not silent', async () => {
    // The restore rolls the app back to the snapshot's moment. Anything done
    // since — a deleted column, a new one, a block of rows — is gone, and
    // without a notice the user is left thinking the app undid their work at
    // random. Reported: a dropped column reappeared and a new one vanished
    // after a redeploy.
    await client().get('/api/stats/dead-1/descriptive')
    const notice = useStore.getState().sessionRecovery
    expect(notice).toMatchObject({ name: 'tiroid.xlsx', snapshotAt: 1 })
    expect(notice?.restoredAt).toBeGreaterThan(0)
  })

  it('leaves no notice when there is nothing to restore from', async () => {
    for (const k of Object.keys(rows)) delete rows[k]
    await expect(client().get('/api/stats/dead-1/descriptive')).rejects.toThrow()
    expect(useStore.getState().sessionRecovery).toBeNull()
  })

  it('follows a request that still names the dead id after the recovery', async () => {
    // Reported: adding a column after a redeploy "failed" and the column did
    // not appear — but the server had it. The POST was recovered and retried,
    // while the refresh that follows it carried the id captured before the
    // recovery. That one is no longer the open session, so it fell through as
    // a plain 404 and the grid kept showing the pre-restore state.
    const api = client()
    await api.get('/api/stats/dead-1/descriptive')     // performs the recovery
    expect(loads).toBe(1)

    const res = await api.get('/api/stats/dead-1/descriptive')
    expect(res.data).toEqual({ ok: true })
    // The snapshot is already back on the server; uploading it again would
    // fork the session in two.
    expect(loads).toBe(1)
  })

  it('pins the row it was restored from, so autosave writes back to it', async () => {
    await client().get('/api/stats/dead-1/descriptive')
    expect(useStore.getState().localSessionId).toBe('row-1')
  })

  it('recovers once for a burst of simultaneous 404s', async () => {
    // A grid refresh fires several requests at once; each recovering on its own
    // would upload a pile of duplicate sessions and race over the final id.
    const api = client()
    const all = await Promise.all([
      api.get('/api/stats/dead-1/descriptive'),
      api.get('/api/stats/dead-1/descriptive'),
      api.get('/api/stats/dead-1/descriptive'),
    ])
    expect(all.every((r) => r.data.ok)).toBe(true)
    expect(loads).toBe(1)
  })

  it('leaves a 404 for some other session alone', async () => {
    // A stale panel or a deleted matched-cohort session is a genuine 404.
    server.use(http.get('/api/stats/other-9/descriptive', () =>
      HttpResponse.json({ detail: 'Session not found' }, { status: 404 })))
    await expect(client().get('/api/stats/other-9/descriptive')).rejects.toThrow()
    expect(loads).toBe(0)
  })

  it('gives up rather than looping when the restore also 404s', async () => {
    server.use(http.get('/api/stats/fresh-1/descriptive', () =>
      HttpResponse.json({ detail: 'Session not found' }, { status: 404 })))
    await expect(client().get('/api/stats/dead-1/descriptive')).rejects.toThrow()
    expect(loads).toBe(1)          // retried once, not forever
  })

  it('gives up when this browser has no copy to restore from', async () => {
    delete rows['row-1']
    await expect(client().get('/api/stats/dead-1/descriptive')).rejects.toThrow()
    expect(loads).toBe(0)
  })

  it('does not try the same dead session twice', async () => {
    delete rows['row-1']
    const api = client()
    await expect(api.get('/api/stats/dead-1/descriptive')).rejects.toThrow()
    await expect(api.get('/api/stats/dead-1/descriptive')).rejects.toThrow()
    expect(loads).toBe(0)
  })

  it('passes a non-404 through untouched', async () => {
    server.use(http.get('/api/stats/dead-1/descriptive', () =>
      HttpResponse.json({ detail: 'boom' }, { status: 500 })))
    await expect(client().get('/api/stats/dead-1/descriptive')).rejects.toThrow()
    expect(loads).toBe(0)
  })
})
