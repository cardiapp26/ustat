/**
 * The welcome screen links to the desktop installers, except inside the
 * desktop app itself, where it would only offer what is already running.
 */
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearSession } from '../test/testUtils'

vi.mock('./RecentSessionsPanel', () => ({ default: () => null }))
vi.mock('../lib/cloudSync', () => ({
  cloudSync: {
    getStatus: vi.fn(() => ({ status: 'signedOut', signedIn: false })),
    subscribe: vi.fn(() => () => {}),
  },
}))

// UploadZone decides at import time, so each case imports a fresh copy.
async function renderFresh() {
  vi.resetModules()
  const { default: UploadZone } = await import('./UploadZone')
  clearSession()
  render(<UploadZone />)
}

afterEach(() => {
  delete (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__
})

describe('UploadZone desktop download link', () => {
  it('points the web app at the latest release', async () => {
    await renderFresh()
    const link = screen.getByRole('link', { name: /download desktop app/i })
    expect(link).toHaveAttribute('href', 'https://github.com/cardiapp26/ustat/releases/latest')
    expect(link).toHaveAttribute('target', '_blank')
  })

  it('is not shown inside the desktop app', async () => {
    Object.defineProperty(window, '__TAURI_INTERNALS__', { configurable: true, value: {} })
    await renderFresh()
    expect(screen.queryByRole('link', { name: /download desktop app/i })).toBeNull()
  })
})
