import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import { useStore } from '../store'
import SessionRecoveredNotice from './SessionRecoveredNotice'

afterEach(() => useStore.setState({ sessionRecovery: null }))

describe('SessionRecoveredNotice', () => {
  it('shows nothing until a recovery happens', () => {
    useStore.setState({ sessionRecovery: null })
    const { container } = render(<SessionRecoveredNotice />)
    expect(container).toBeEmptyDOMElement()
  })

  it('names the file and the moment the restored copy was taken', () => {
    // Without the time the user cannot tell which of their edits survived —
    // that is the whole question a rollback leaves them with.
    const at = new Date(2026, 7, 31, 14, 5).getTime()
    useStore.setState({
      sessionRecovery: { restoredAt: at + 60_000, snapshotAt: at, name: 'tiroid.xlsx' },
    })
    render(<SessionRecoveredNotice />)
    expect(screen.getByText(/tiroid\.xlsx/)).toBeInTheDocument()
    expect(screen.getByText(/Session restored from the autosave/)).toBeInTheDocument()
    expect(screen.getByText(new RegExp(new Date(at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })))).toBeInTheDocument()
  })

  it('goes away when dismissed', async () => {
    useStore.setState({
      sessionRecovery: { restoredAt: Date.now(), snapshotAt: Date.now(), name: 'x.csv' },
    })
    const user = userEvent.setup()
    render(<SessionRecoveredNotice />)
    await user.click(screen.getByRole('button', { name: /dismiss/i }))
    expect(useStore.getState().sessionRecovery).toBeNull()
    expect(screen.queryByText(/Session restored/)).not.toBeInTheDocument()
  })
})
