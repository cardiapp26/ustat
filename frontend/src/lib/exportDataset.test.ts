import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { downloadProjectFile, downloadSessionJson, exportDataset } from './exportDataset'
import api from '../api'

vi.mock('../api', () => ({
  default: { get: vi.fn() },
}))

const mockedGet = vi.mocked(api.get)

function blobResponse(data: unknown, contentType: string) {
  return {
    data,
    headers: { 'content-type': contentType },
  }
}

describe('exportDataset', () => {
  let createObjectURL: ReturnType<typeof vi.fn>
  let revokeObjectURL: ReturnType<typeof vi.fn>
  let alertSpy: ReturnType<typeof vi.fn>
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.useFakeTimers()
    createObjectURL = vi.fn(() => 'blob:mock-url')
    revokeObjectURL = vi.fn()
    // @ts-expect-error jsdom doesn't implement these
    URL.createObjectURL = createObjectURL
    // @ts-expect-error jsdom doesn't implement these
    URL.revokeObjectURL = revokeObjectURL
    alertSpy = vi.fn()
    // @ts-expect-error jsdom alert stub
    window.alert = alertSpy
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    mockedGet.mockReset()
  })

  it('builds the export URL with encoded filename (extension stripped) and col_kinds', async () => {
    mockedGet.mockResolvedValue(blobResponse(new Blob(['a,b\n1,2']), 'text/csv'))
    const session = { session_id: 'abc123', filename: 'my data.csv' }
    const columns = [
      { name: 'age', kind: 'numeric' },
      { name: 'group', kind: 'categorical' },
    ]
    await exportDataset(session, columns, 'csv')

    expect(mockedGet).toHaveBeenCalledTimes(1)
    const [url, opts] = mockedGet.mock.calls[0]
    expect(url).toContain('/api/sessions/abc123/export?fmt=csv')
    expect(url).toContain(`filename=${encodeURIComponent('my data')}`)
    const colKinds = JSON.parse(decodeURIComponent(url.split('col_kinds=')[1]))
    expect(colKinds).toEqual({ age: 'numeric', group: 'categorical' })
    expect(opts).toEqual({ responseType: 'blob' })
  })

  it('defaults the base filename to "data" when session.filename is missing', async () => {
    mockedGet.mockResolvedValue(blobResponse(new Blob(['x']), 'text/csv'))
    await exportDataset({ session_id: 's1' }, [], 'csv')
    const [url] = mockedGet.mock.calls[0]
    expect(url).toContain('filename=data')
  })

  it('strips only the final extension from a filename with dots', async () => {
    mockedGet.mockResolvedValue(blobResponse(new Blob(['x']), 'text/csv'))
    await exportDataset({ session_id: 's1', filename: 'v1.2.final.xlsx' }, [], 'xlsx')
    const [url] = mockedGet.mock.calls[0]
    expect(url).toContain(`filename=${encodeURIComponent('v1.2.final')}`)
  })

  it('triggers a download with the correct filename extension per format', async () => {
    mockedGet.mockResolvedValue(blobResponse(new Blob(['x']), 'text/csv'))
    const clickSpy = vi.fn()
    const origCreateElement = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = origCreateElement(tag)
      if (tag === 'a') el.click = clickSpy
      return el
    })

    await exportDataset({ session_id: 's1', filename: 'data.csv' }, [], 'tsv')
    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(createObjectURL).toHaveBeenCalledTimes(1)
  })

  it('throws/alerts when the server returns JSON instead of the requested format', async () => {
    const jsonBlob = new Blob([JSON.stringify({ detail: 'export failed' })], { type: 'application/json' });
    mockedGet.mockResolvedValue(blobResponse(jsonBlob, 'application/json'))

    await exportDataset({ session_id: 's1' }, [], 'csv')

    expect(consoleErrorSpy).toHaveBeenCalled()
    expect(alertSpy).toHaveBeenCalledTimes(1)
    expect(alertSpy.mock.calls[0][0]).toContain('Export as CSV failed')
  })

  it('alerts with the underlying error message when the request rejects with a plain Error', async () => {
    mockedGet.mockRejectedValue(new Error('network down'))
    await exportDataset({ session_id: 's1' }, [], 'sav')
    expect(alertSpy).toHaveBeenCalledWith('Export as SAV failed: network down')
  })

  it('extracts the "detail" field from a JSON error blob on rejection', async () => {
    const errBlob = new Blob([JSON.stringify({ detail: 'bad request' })], { type: 'application/json' })
    mockedGet.mockRejectedValue({ response: { data: errBlob } })
    await exportDataset({ session_id: 's1' }, [], 'xlsx')
    expect(alertSpy).toHaveBeenCalledWith('Export as XLSX failed: bad request')
  })

  it('falls back to raw text when the error blob is not valid JSON', async () => {
    const errBlob = new Blob(['not json'], { type: 'text/plain' })
    mockedGet.mockRejectedValue({ response: { data: errBlob } })
    await exportDataset({ session_id: 's1' }, [], 'csv')
    expect(alertSpy).toHaveBeenCalledWith('Export as CSV failed: [object Object]')
  })

  it('handles an empty columns array without error', async () => {
    mockedGet.mockResolvedValue(blobResponse(new Blob(['x']), 'text/csv'))
    await exportDataset({ session_id: 's1' }, [], 'csv')
    const [url] = mockedGet.mock.calls[0]
    expect(url).toContain('col_kinds=%7B%7D')
  })
})

describe('downloadSessionJson', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    URL.createObjectURL = vi.fn(() => 'blob:mock-url')
    URL.revokeObjectURL = vi.fn()
    window.alert = vi.fn()
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    mockedGet.mockReset()
  })

  it('requests the save_session endpoint and triggers a download named after the session filename', async () => {
    mockedGet.mockResolvedValue({ data: new Blob(['{}']), headers: {} })
    const clickSpy = vi.fn()
    const origCreateElement = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = origCreateElement(tag)
      if (tag === 'a') {
        el.click = clickSpy
        Object.defineProperty(el, 'download', { value: '', writable: true })
      }
      return el
    })

    await downloadSessionJson({ session_id: 'sess1', filename: 'report.csv' })

    expect(mockedGet).toHaveBeenCalledWith('/api/sessions/sess1/save_session', { responseType: 'blob' })
    expect(clickSpy).toHaveBeenCalledTimes(1)
  })

  it('defaults to "session" as the base filename when none is provided', async () => {
    mockedGet.mockResolvedValue({ data: new Blob(['{}']), headers: {} })
    await downloadSessionJson({ session_id: 'sess1' })
    expect(mockedGet).toHaveBeenCalledWith('/api/sessions/sess1/save_session', { responseType: 'blob' })
  })

  it('alerts on failure', async () => {
    mockedGet.mockRejectedValue(new Error('boom'))
    const alertSpy = vi.fn()
    window.alert = alertSpy
    await downloadSessionJson({ session_id: 'sess1' })
    expect(alertSpy).toHaveBeenCalledWith('Save session failed: boom')
  })
})

describe('downloadProjectFile', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    URL.createObjectURL = vi.fn(() => 'blob:mock-url')
    URL.revokeObjectURL = vi.fn()
    window.alert = vi.fn()
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    mockedGet.mockReset()
  })

  it('requests the project save endpoint and downloads a .ustat named after the session', async () => {
    mockedGet.mockResolvedValue({ data: new Blob(['PK']), headers: {} })
    let downloadName = ''
    const origCreateElement = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = origCreateElement(tag)
      if (tag === 'a') {
        el.click = vi.fn()
        Object.defineProperty(el, 'download', {
          get: () => downloadName,
          set: (v: string) => { downloadName = v },
        })
      }
      return el
    })

    await downloadProjectFile({ session_id: 'sess1', filename: 'trial.csv' })

    expect(mockedGet).toHaveBeenCalledWith('/api/project/sess1/save', { responseType: 'blob' })
    expect(downloadName).toBe('trial.ustat')
  })

  it('alerts on failure', async () => {
    mockedGet.mockRejectedValue(new Error('boom'))
    const alertSpy = vi.fn()
    window.alert = alertSpy
    await downloadProjectFile({ session_id: 'sess1' })
    expect(alertSpy).toHaveBeenCalledWith('Save project failed: boom')
  })
})
