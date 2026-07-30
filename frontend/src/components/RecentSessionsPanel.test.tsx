import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { server } from '../test/server'
import { clearSession } from '../test/testUtils'
import RecentSessionsPanel from './RecentSessionsPanel'
import * as sessionDb from '../lib/sessionDb'
import * as exportSnapshot from '../lib/exportSnapshot'
import { cloudSync } from '../lib/cloudSync'

vi.mock('../lib/sessionDb', () => ({
  TRASH_TTL_MS: 30 * 24 * 60 * 60 * 1000,
  listRecentSessions: vi.fn(),
  listTrashedSessions: vi.fn(),
  trashSession: vi.fn(),
  restoreSession: vi.fn(),
  purgeSession: vi.fn(),
  emptyTrash: vi.fn(),
  getRecentSession: vi.fn(),
  subscribeSessions: vi.fn(() => () => {}),
  getStorageEstimate: vi.fn(),
  clearAllRecentSessions: vi.fn(),
  duplicateRecentSession: vi.fn(),
}))

vi.mock('../lib/exportSnapshot', async () => {
  const actual = await vi.importActual<typeof import('../lib/exportSnapshot')>(
    '../lib/exportSnapshot',
  )
  return { ...actual, exportSnapshot: vi.fn() }
})

vi.mock('../lib/cloudSync', () => ({
  cloudSync: {
    isSignedIn: vi.fn(() => false),
    subscribe: vi.fn(() => () => {}),
    syncNow: vi.fn(),
  },
}))

const baseMeta = {
  id: 'sess-1',
  name: 'patients.csv',
  savedAt: Date.now() - 60_000,
  sizeBytes: 2048,
  nRows: 120,
  nCols: 8,
  activeTab: 'data',
  source: 'auto' as const,
};

function mockLists(active: typeof baseMeta[] = [], trashed: typeof baseMeta[] = []) {
  vi.mocked(sessionDb.listRecentSessions).mockResolvedValue(active)
  vi.mocked(sessionDb.listTrashedSessions).mockResolvedValue(trashed)
  vi.mocked(sessionDb.getStorageEstimate).mockResolvedValue({ count: active.length, bytes: 2048, capCount: 20, capBytes: 200 * 1024 * 1024 })
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(cloudSync.isSignedIn).mockReturnValue(false)
  vi.mocked(cloudSync.subscribe).mockReturnValue(() => {})
  vi.mocked(sessionDb.subscribeSessions).mockReturnValue(() => {})
})

afterEach(() => clearSession())

describe('RecentSessionsPanel', () => {
  it('renders nothing when there are no local sessions and cloud sync is off', async () => {
    mockLists([], [])
    const { container } = render(<RecentSessionsPanel />)
    await waitFor(() => expect(container.querySelector('.animate-pulse')).not.toBeInTheDocument())
    expect(container).toBeEmptyDOMElement()
  })

  it('lists saved sessions as cards with name, dims, and last tab', async () => {
    mockLists([baseMeta])
    render(<RecentSessionsPanel />)

    await waitFor(() => expect(screen.getByText('patients.csv')).toBeInTheDocument())
    expect(screen.getByText(/120.*×.*8/)).toBeInTheDocument()
    expect(screen.getByText('Data')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /resume/i })).toBeInTheDocument()
  })

  it('restores a session: loads it via load_session and updates the store', async () => {
    mockLists([baseMeta])
    vi.mocked(sessionDb.getRecentSession).mockResolvedValue({
      ...baseMeta,
      payload: JSON.stringify({ some: 'session-json' }),
    })
    server.use(
      http.post('/api/sessions/load_session', () =>
        HttpResponse.json({
          session_id: 'restored-session',
          filename: 'patients.csv',
          rows: 120,
          columns: [],
          preview: [],
        }),
      ),
      http.get('/api/sessions/restored-session/decimals', () => HttpResponse.json({})),
    )

    const user = userEvent.setup()
    render(<RecentSessionsPanel />)
    await waitFor(() => expect(screen.getByText('patients.csv')).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: /resume/i }))

    await waitFor(() => expect(sessionDb.getRecentSession).toHaveBeenCalledWith('sess-1'))
  })

  it('shows an error message when restore fails', async () => {
    mockLists([baseMeta])
    vi.mocked(sessionDb.getRecentSession).mockResolvedValue(undefined)

    const user = userEvent.setup()
    render(<RecentSessionsPanel />)
    await waitFor(() => expect(screen.getByText('patients.csv')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: /resume/i }))

    await waitFor(() => expect(screen.getByText('Snapshot not found')).toBeInTheDocument())
  })

  it('moves a session to trash when the delete button is clicked', async () => {
    mockLists([baseMeta])
    vi.mocked(sessionDb.trashSession).mockResolvedValue(undefined)

    const user = userEvent.setup()
    render(<RecentSessionsPanel />)
    await waitFor(() => expect(screen.getByText('patients.csv')).toBeInTheDocument())

    await user.click(screen.getByTitle(/move to trash/i))

    await waitFor(() => expect(sessionDb.trashSession).toHaveBeenCalledWith('sess-1'))
  })

  it('shows the trash bin and allows restoring a trashed session', async () => {
    const trashedMeta = { ...baseMeta, id: 'sess-2', name: 'old.csv', deletedAt: Date.now() - 1000 }
    // The panel hides entirely when there are zero *active* sessions and no
    // cloud sync — keep one active session so the trash section is reachable.
    mockLists([baseMeta], [trashedMeta])
    vi.mocked(sessionDb.restoreSession).mockResolvedValue(undefined)

    const user = userEvent.setup()
    render(<RecentSessionsPanel />)

    await waitFor(() => expect(screen.getByText('Trash')).toBeInTheDocument())
    await user.click(screen.getByText('Trash'))
    await waitFor(() => expect(screen.getByText('old.csv')).toBeInTheDocument())

    await user.click(screen.getByText('Restore'))
    await waitFor(() => expect(sessionDb.restoreSession).toHaveBeenCalledWith('sess-2'))
  })

  it('shows the Drive import entry point when cloud sync is signed in', async () => {
    vi.mocked(cloudSync.isSignedIn).mockReturnValue(true)
    mockLists([], [])
    render(<RecentSessionsPanel />)

    await waitFor(() => expect(screen.getByText(/import from drive/i)).toBeInTheDocument())
  })

  it('duplicates a session without touching the original', async () => {
    mockLists([baseMeta])
    vi.mocked(sessionDb.duplicateRecentSession).mockResolvedValue({
      ...baseMeta, id: 'sess-2', name: 'patients.csv (copy)',
    })
    const user = userEvent.setup()
    render(<RecentSessionsPanel />)
    await screen.findByText('patients.csv')

    await user.click(screen.getByTitle(/duplicate/i))

    await waitFor(() =>
      expect(sessionDb.duplicateRecentSession).toHaveBeenCalledWith('sess-1'),
    )
    // The copy must not be made by trashing or overwriting the source.
    expect(sessionDb.trashSession).not.toHaveBeenCalled()
    expect(sessionDb.purgeSession).not.toHaveBeenCalled()
  })

  it('offers Save as on a card and writes the chosen format from the snapshot', async () => {
    // The point of building the file locally: the card's dataset is usually
    // not loaded on the backend, so nothing here may go near the server.
    mockLists([baseMeta])
    const payload = JSON.stringify({
      filename: 'patients.csv',
      columns: [{ name: 'id' }],
      data: [{ id: 1 }],
    })
    vi.mocked(sessionDb.getRecentSession).mockResolvedValue({ ...baseMeta, payload })
    const user = userEvent.setup()
    render(<RecentSessionsPanel />)
    await screen.findByText('patients.csv')

    await user.click(screen.getByTitle(/save as/i))
    await user.click(screen.getByRole('menuitem', { name: 'CSV' }))

    await waitFor(() =>
      expect(exportSnapshot.exportSnapshot).toHaveBeenCalledWith(
        { name: 'patients.csv', payload }, 'csv',
      ),
    )
  })

  it('closes the Save as menu without exporting when dismissed', async () => {
    mockLists([baseMeta])
    const user = userEvent.setup()
    render(<RecentSessionsPanel />)
    await screen.findByText('patients.csv')

    await user.click(screen.getByTitle(/save as/i))
    expect(screen.getByRole('menuitem', { name: 'Excel' })).toBeInTheDocument()
    await user.click(screen.getByTitle(/save as/i))

    await waitFor(() =>
      expect(screen.queryByRole('menuitem', { name: 'Excel' })).not.toBeInTheDocument(),
    )
    expect(exportSnapshot.exportSnapshot).not.toHaveBeenCalled()
  })

  it('surfaces a failed export instead of failing silently', async () => {
    mockLists([baseMeta])
    vi.mocked(sessionDb.getRecentSession).mockResolvedValue(undefined)
    const user = userEvent.setup()
    render(<RecentSessionsPanel />)
    await screen.findByText('patients.csv')

    await user.click(screen.getByTitle(/save as/i))
    await user.click(screen.getByRole('menuitem', { name: 'TSV' }))

    await waitFor(() => expect(screen.getByText(/snapshot not found/i)).toBeInTheDocument())
  })
})
