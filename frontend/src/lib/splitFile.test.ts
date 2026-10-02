import { describe, expect, it, vi } from 'vitest'
import { AxiosHeaders, type InternalAxiosRequestConfig } from 'axios'
import { SPLIT_HEADER, setSplitProvider, splitHeaderValue, withSplitHeader } from './splitHeader'
import { ALL_LEVELS_KEY, precomputeLevels, repeatable } from './splitPrecompute'
import { analysisScope } from '../store'

const cfg = (url: string, headers: Record<string, string> = {}): InternalAxiosRequestConfig =>
  ({ url, headers: new AxiosHeaders(headers) }) as InternalAxiosRequestConfig

describe('split header', () => {
  it('states the level on view, percent-encoded so any column name survives', () => {
    setSplitProvider(() => ({ column: 'cinsiyet', level: 'Kadın' }))
    const out = withSplitHeader(cfg('/api/stats/ttest'))
    const raw = out.headers.get(SPLIT_HEADER) as string
    expect(JSON.parse(decodeURIComponent(raw))).toEqual({ column: 'cinsiyet', level: 'Kadın' })
    expect(/^[\x20-\x7e]+$/.test(raw)).toBe(true)
  })

  it('adds nothing for the unsplit view or outside /api/', () => {
    setSplitProvider(() => ({ column: 'sex', level: null }))
    expect(withSplitHeader(cfg('/api/stats/ttest')).headers.has(SPLIT_HEADER)).toBe(false)
    setSplitProvider(() => ({ column: 'sex', level: 'F' }))
    expect(withSplitHeader(cfg('/assets/x.js')).headers.has(SPLIT_HEADER)).toBe(false)
  })

  it('leaves a header the caller set itself (the precompute names its own level)', () => {
    setSplitProvider(() => ({ column: 'sex', level: 'F' }))
    const out = withSplitHeader(cfg('/api/stats/ttest', { [SPLIT_HEADER]: splitHeaderValue('sex', 'M') }))
    expect(JSON.parse(decodeURIComponent(out.headers.get(SPLIT_HEADER) as string)).level).toBe('M')
    setSplitProvider(() => null)
  })
})

describe('analysisScope', () => {
  it('is null with neither weights nor a split level, so old stamps keep their key', () => {
    expect(analysisScope({ caseWeight: null, splitFile: null })).toBeNull()
    expect(analysisScope({ caseWeight: null, splitFile: { column: 'sex', levels: [], level: null } })).toBeNull()
  })
  it('names the weight and the level on view', () => {
    expect(analysisScope({ caseWeight: { column: 'n' }, splitFile: { column: 'sex', levels: [], level: 'F' } }))
      .toBe('weight:n|split:sex=F')
  })
})

describe('precomputeLevels', () => {
  const request = { method: 'post', url: '/api/stats/ttest', body: { session_id: '{sid}', column: 'sbp' } }

  it('asks once per level with that level, and for the unsplit view with an empty header', async () => {
    const api = { request: vi.fn(async (c: { headers: Record<string, string> }) => ({ data: { h: c.headers[SPLIT_HEADER] } })) }
    const seen: Record<string, unknown> = {}
    await precomputeLevels({
      api: api as never, request, sessionId: 's1', column: 'sex', targets: ['M', null],
      stampFor: () => ({ dataVersion: 1, caseFilter: null, engine: 'python', params: {}, sessionId: 's1' }),
      cancelled: () => false,
      onLevel: (key, v) => { seen[key] = v.result },
    })
    expect(api.request).toHaveBeenCalledTimes(2)
    expect(api.request.mock.calls[0][0]).toMatchObject({ url: '/api/stats/ttest', data: { session_id: 's1', column: 'sbp' } })
    expect(seen.M).toEqual({ h: splitHeaderValue('sex', 'M') })
    expect(seen[ALL_LEVELS_KEY]).toEqual({ h: '' })
  })

  it('stops when a newer run supersedes it', async () => {
    const api = { request: vi.fn(async () => ({ data: {} })) }
    let calls = 0
    await precomputeLevels({
      api: api as never, request, sessionId: 's1', column: 'sex', targets: ['F', 'M', null],
      stampFor: () => ({ dataVersion: 1, caseFilter: null, engine: 'python', params: {}, sessionId: 's1' }),
      cancelled: () => ++calls > 1,
      onLevel: () => {},
    })
    expect(api.request).toHaveBeenCalledTimes(1)
  })

  it('never repeats endpoints with side effects or heavy ones', () => {
    expect(repeatable({ method: 'post', url: '/api/models/psm', body: null })).toBe(false)
    expect(repeatable({ method: 'post', url: '/api/ml/train', body: null })).toBe(false)
    expect(repeatable({ method: 'post', url: '/api/compute/{sid}/formula', body: null })).toBe(false)
    expect(repeatable({ method: 'post', url: '/api/stats/ttest', body: null })).toBe(true)
    expect(repeatable(null)).toBe(false)
  })
})
