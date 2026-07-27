// A real IndexedDB, only for this file — the rest of the suite has no need
// for one and should not inherit the global shim.
import 'fake-indexeddb/auto'

import { beforeEach, describe, expect, it } from 'vitest'
import {
  clearAllRecentSessions,
  duplicateRecentSession,
  emptyTrash,
  getRecentSession,
  listRecentSessions,
  upsertRecentSession,
} from './sessionDb'

async function wipe() {
  await clearAllRecentSessions()
  await emptyTrash()
}

/** A session blob of a given shape. `seed` changes the content without
 *  changing its length, so two rows can share a fingerprint yet differ. */
function payloadOf(seed: string, cells = 40): string {
  return JSON.stringify({ data: Array.from({ length: cells }, () => seed) })
}

async function add(name: string, payload: string, serverSessionId = name) {
  return upsertRecentSession({
    serverSessionId,
    name,
    payload,
    nRows: 10,
    nCols: 4,
    source: 'auto',
  })
}

beforeEach(wipe)

describe('duplicateRecentSession', () => {
  it('leaves the original in place', async () => {
    const src = await add('patients.csv', payloadOf('a'))
    await duplicateRecentSession(src.id)

    const names = (await listRecentSessions()).map((r) => r.name).sort()
    expect(names).toEqual(['patients.csv', 'patients.csv (copy)'])
  })

  it('survives the dedupe that runs on every list', async () => {
    // The copy is byte-identical to its source, which is exactly what the
    // dedupe is built to remove. Listing twice is what used to eat it.
    const src = await add('patients.csv', payloadOf('a'))
    await duplicateRecentSession(src.id)
    await listRecentSessions()
    await listRecentSessions()

    expect((await listRecentSessions()).map((r) => r.name).sort()).toEqual([
      'patients.csv',
      'patients.csv (copy)',
    ])
  })

  it('gives the copy its own id and cuts it loose from the server session', async () => {
    const src = await add('patients.csv', payloadOf('a'), 'srv-123')
    const copy = await duplicateRecentSession(src.id)

    expect(copy!.id).not.toBe(src.id)
    // Sharing the server id would let edits in one leak into the other's
    // autosave, since that is the key the autosave upserts on.
    expect(copy!.serverSessionId).toBeUndefined()
    expect(copy!.userCopy).toBe(true)
  })

  it('copies the payload verbatim', async () => {
    const src = await add('patients.csv', payloadOf('a'))
    const copy = await duplicateRecentSession(src.id)

    const from = await getRecentSession(src.id)
    const to = await getRecentSession(copy!.id)
    expect(to!.payload).toBe(from!.payload)
  })

  it('numbers repeated copies instead of colliding', async () => {
    const src = await add('patients.csv', payloadOf('a'))
    await duplicateRecentSession(src.id)
    await duplicateRecentSession(src.id)
    await duplicateRecentSession(src.id)

    // Sort both sides the same way: lexicographically "(copy 2)" precedes
    // "(copy)", because a space sorts before a closing parenthesis.
    const names = (await listRecentSessions()).map((r) => r.name).sort()
    expect(names).toEqual([
      'patients.csv',
      'patients.csv (copy)',
      'patients.csv (copy 2)',
      'patients.csv (copy 3)',
    ].sort())
  })

  it('returns undefined for an id that is not there', async () => {
    expect(await duplicateRecentSession('nope')).toBeUndefined()
  })
})

describe('dedupe on list', () => {
  it('collapses two rows with the same name', async () => {
    await add('same.csv', payloadOf('a'), 'srv-1')
    await add('same.csv', payloadOf('b'), 'srv-2')

    const rows = await listRecentSessions()
    expect(rows).toHaveLength(1)
  })

  it('keeps two different files that happen to share a fingerprint', async () => {
    // Same rows x cols and the same payload length, different content. The
    // old rows x cols x bytes rule deleted one of these outright.
    const a = payloadOf('a')
    const b = payloadOf('b')
    expect(a.length).toBe(b.length)
    expect(a).not.toBe(b)

    await add('trial-a.csv', a, 'srv-a')
    await add('trial-b.csv', b, 'srv-b')

    const names = (await listRecentSessions()).map((r) => r.name).sort()
    expect(names).toEqual(['trial-a.csv', 'trial-b.csv'])
  })

  it('still collapses the same file saved under two names', async () => {
    const shared = payloadOf('a')
    await add('data.csv', shared, 'srv-1')
    await add('data (1).csv', shared, 'srv-2')

    const rows = await listRecentSessions()
    expect(rows).toHaveLength(1)
    // The newer row wins.
    expect(rows[0].name).toBe('data (1).csv')
  })
})
