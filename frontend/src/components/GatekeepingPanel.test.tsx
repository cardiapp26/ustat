import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import { useStore } from '../store'
import GatekeepingPanel from './GatekeepingPanel'

afterEach(() => clearSession())

describe('GatekeepingPanel', () => {
  it('renders without an active session (no session dependency)', () => {
    clearSession()
    render(<GatekeepingPanel />)
    expect(screen.getByText('Gatekeeping (multiplicity)')).toBeInTheDocument()
  })

  it('shows sample families with hypotheses pre-filled', () => {
    installSession()
    render(<GatekeepingPanel />)
    expect(screen.getByDisplayValue('Primary')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Secondary')).toBeInTheDocument()
    expect(screen.getByDisplayValue('All-cause death')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /run gatekeeping/i })).toBeEnabled()
  })

  it('runs gatekeeping with sample data and renders adjusted p-values', async () => {
    installSession()
    server.use(
      http.post('/api/multiplicity/gatekeeping', () =>
        HttpResponse.json({
          method: 'hochberg',
          logic: 'serial',
          alpha: 0.05,
          families: [
            {
              name: 'Primary',
              gamma: 1,
              n_rejected: 1,
              n: 1,
              hypotheses: [{ label: 'All-cause death', p_raw: 0.012, p_adjusted: 0.012, reject: true }],
            },
            {
              name: 'Secondary',
              gamma: 0.5,
              n_rejected: 2,
              n: 3,
              hypotheses: [
                { label: 'MI', p_raw: 0.02, p_adjusted: 0.04, reject: true },
                { label: 'Stroke', p_raw: 0.04, p_adjusted: 0.06, reject: false },
                { label: 'HF hospitalisation', p_raw: 0.3, p_adjusted: 0.3, reject: false },
              ],
            },
          ],
          interpretation: 'Primary endpoint significant, gate opens to secondary family.',
          export_rows: [['Family', 'Label', 'p_raw', 'p_adj'], ['Primary', 'All-cause death', '0.012', '0.012']],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<GatekeepingPanel />)
    await user.click(screen.getByRole('button', { name: /run gatekeeping/i }))

    await waitFor(() => expect(screen.getByText(/Adjusted p-values/)).toBeInTheDocument())
    expect(screen.getByText('Primary endpoint significant, gate opens to secondary family.')).toBeInTheDocument()
    expect(screen.getAllByText('Reject H₀').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Retain').length).toBeGreaterThan(0)
  })

  it('shows the error message from the backend on failure', async () => {
    installSession()
    server.use(
      http.post('/api/multiplicity/gatekeeping', () =>
        HttpResponse.json({ detail: 'Invalid gamma value' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<GatekeepingPanel />)
    await user.click(screen.getByRole('button', { name: /run gatekeeping/i }))

    await waitFor(() => expect(screen.getByText('Invalid gamma value')).toBeInTheDocument())
  })

  it('ignores a data change but closes export once a typed p-value changes', async () => {
    // Gatekeeping adjusts p-values typed into the panel and reads no dataset:
    // a cell edit must not flag it, while editing one of its own inputs must.
    installSession()
    server.use(
      http.post('/api/multiplicity/gatekeeping', () =>
        HttpResponse.json({
          method: 'hochberg',
          logic: 'serial',
          alpha: 0.05,
          families: [
            {
              name: 'Primary',
              gamma: 1,
              n_rejected: 1,
              n: 1,
              hypotheses: [{ label: 'All-cause death', p_raw: 0.012, p_adjusted: 0.012, reject: true }],
            },
          ],
          export_rows: [['Family', 'Label', 'p_raw', 'p_adj'], ['Primary', 'All-cause death', '0.012', '0.012']],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<GatekeepingPanel />)
    await user.click(screen.getByRole('button', { name: /run gatekeeping/i }))
    const csv = await screen.findByRole('button', { name: 'CSV' })
    expect(csv).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeEnabled()

    const p = screen.getByDisplayValue('0.012')
    await user.clear(p)
    await user.type(p, '0.03')

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeDisabled()
  })
})
