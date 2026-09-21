/**
 * A file opened from Finder/Explorer "Open With" on the installed PWA reaches
 * the welcome screen through window.launchQueue and must upload exactly like
 * a file chosen with "Browse file".
 */
import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { uploadFile } from '../api'
import { useStore } from '../store'
import { clearSession, makeSession } from '../test/testUtils'
import UploadZone from './UploadZone'

// Mocked at the api boundary rather than with msw: jsdom's File loses its
// name when it crosses into Node's FormData, and the name is what matters.
vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  uploadFile: vi.fn(),
}))

vi.mock('./RecentSessionsPanel', () => ({ default: () => null }))
vi.mock('../lib/cloudSync', () => ({
  cloudSync: {
    getStatus: vi.fn(() => ({ status: 'signedOut', signedIn: false })),
    subscribe: vi.fn(() => () => {}),
  },
}))

type Consumer = (params: { files: readonly FileSystemHandle[] }) => void

let consumer: Consumer | null = null

function fileHandle(getFile: () => Promise<File>): FileSystemHandle {
  return { kind: 'file', name: 'launched', getFile } as unknown as FileSystemHandle
}

beforeEach(() => {
  clearSession()
  consumer = null
  Object.defineProperty(window, 'launchQueue', {
    configurable: true,
    value: { setConsumer: (c: Consumer) => { consumer = c } },
  })
})

afterEach(() => {
  delete (window as { launchQueue?: unknown }).launchQueue
})

describe('UploadZone file launch', () => {
  it('uploads the file the app was launched with and opens its session', async () => {
    vi.mocked(uploadFile).mockImplementation(async (file: File) =>
      ({ data: makeSession({ session_id: 'launched', filename: file.name }) }) as Awaited<ReturnType<typeof uploadFile>>,
    )
    render(<UploadZone />)
    expect(consumer).not.toBeNull()

    const sav = new File(['x'], 'cohort.sav')
    consumer!({ files: [fileHandle(() => Promise.resolve(sav))] })

    await waitFor(() => expect(useStore.getState().session?.session_id).toBe('launched'))
    expect(uploadFile).toHaveBeenCalledExactlyOnceWith(sav)
  })

  it('says so when the launched file cannot be read', async () => {
    vi.mocked(uploadFile).mockClear()
    render(<UploadZone />)
    consumer!({ files: [fileHandle(() => Promise.reject(new DOMException('gone', 'NotFoundError')))] })

    expect(await screen.findByText('Could not read the file the app was opened with.')).toBeInTheDocument()
    expect(useStore.getState().session).toBeNull()
    expect(uploadFile).not.toHaveBeenCalled()
  })
})
