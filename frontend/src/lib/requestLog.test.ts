import { beforeEach, describe, expect, it } from 'vitest'
import { forSession, recordResponse, requestFor, resetRequestLog } from './requestLog'

const SID = '0b9f2c4e-1234-4abc-9def-0123456789ab'

beforeEach(() => resetRequestLog())

describe('requestLog', () => {
  it('finds the request that returned exactly this object, with the session as {sid}', () => {
    const data = { coefficients: [] }
    recordResponse({ method: 'post', url: '/api/models/logistic', data: JSON.stringify({ session_id: SID, outcome: 'DM' }) }, data)
    expect(requestFor(data, SID)).toEqual({
      method: 'POST', url: '/api/models/logistic', body: { session_id: '{sid}', outcome: 'DM' },
    })
  })

  it('records GET query parameters and a session in the path', () => {
    const data = { AGE: {} }
    recordResponse({ method: 'get', url: `/api/stats/${SID}/descriptive`, params: { column: 'AGE' } }, data)
    expect(requestFor(data, SID)).toEqual({ method: 'GET', url: '/api/stats/{sid}/descriptive?column=AGE', body: null })
  })

  it('does not attribute a reshaped result to the request it came from', () => {
    const data = { auc: 0.8 }
    recordResponse({ method: 'post', url: '/api/roc/analyze', data: '{}' }, data)
    expect(requestFor({ ...data }, SID)).toBeNull()
  })

  it('ignores uploads and non-API calls', () => {
    const a = {}
    const b = {}
    recordResponse({ method: 'post', url: '/api/upload/', data: new FormData() }, a)
    recordResponse({ method: 'get', url: '/assets/x.json' }, b)
    expect(requestFor(a, SID)).toBeNull()
    expect(requestFor(b, SID)).toBeNull()
  })

  it('addresses a recorded request to another session', () => {
    expect(forSession({ method: 'GET', url: '/api/stats/{sid}/descriptive?column=AGE', body: null }, 'new')).toEqual({
      method: 'GET', url: '/api/stats/new/descriptive?column=AGE', body: null,
    })
    expect(forSession({ method: 'POST', url: '/api/models/logistic', body: { session_id: '{sid}', y: 1 } }, 'new').body)
      .toEqual({ session_id: 'new', y: 1 })
  })
})
