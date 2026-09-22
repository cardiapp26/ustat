import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { useStore } from '../store'
import { clearSession, installSession } from '../test/testUtils'
import NonInferiorityPanel from './NonInferiorityPanel'

afterEach(() => clearSession())

describe('NonInferiorityPanel', () => {
  it('renders the empty state without an active session', () => {
    clearSession()
    render(<NonInferiorityPanel />)
    expect(screen.getByText('Non-Inferiority Test')).toBeInTheDocument()
    expect(screen.getByText(/Configure the margin test, then run/i)).toBeInTheDocument()
  })

  it('shows binary/continuous mode toggle and defaults to binary', () => {
    installSession()
    render(<NonInferiorityPanel />)
    expect(screen.getByRole('button', { name: /Binary \(RR\/RD\/OR\)/i })).toHaveClass('bg-indigo-600')
  })

  it('disables Run until outcome and group columns are chosen (shows validation error)', async () => {
    installSession()
    const user = userEvent.setup()
    render(<NonInferiorityPanel />)
    await user.click(screen.getByRole('button', { name: /run non-inferiority test/i }))
    await waitFor(() => expect(screen.getByText('Select outcome and group columns.')).toBeInTheDocument())
  })

  it('loads group levels and runs the non-inferiority test, rendering the result', async () => {
    installSession()
    server.use(
      http.get('/api/compute/test-session/unique/GROUP', () =>
        HttpResponse.json({ values: ['A', 'B'] }),
      ),
      http.post('/api/stats/noninferiority', () =>
        HttpResponse.json({
          non_inferior: true,
          effect: 'RR',
          estimate: 1.05,
          ci_level: 90,
          ci_low: 0.9,
          ci_high: 1.15,
          margin: 1.2,
          bound: 'upper',
          alpha_one_sided: 0.05,
          p_noninferiority: 0.01,
          test_group: 'B',
          ref_group: 'A',
          outcome_type: 'binary',
          n_test: 10,
          n_ref: 12,
          events_test: 3,
          events_ref: 4,
          p_test: 0.3,
          p_ref: 0.33,
          export_rows: [['k', 'v'], ['RR', '1.05']],
          assumptions: [{ name: 'Sample size', detail: 'Adequate power', met: true }],
          interpretation: 'Non-inferiority margin of 1.2 was met.',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<NonInferiorityPanel />)

    const selects = screen.getAllByRole('combobox')
    // outcome select is first, group select is second
    await user.selectOptions(selects[0], 'DM')
    await user.selectOptions(selects[1], 'GROUP')

    await waitFor(() => expect(screen.getByText('Test (new) arm')).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: /run non-inferiority test/i }))

    await waitFor(() => expect(screen.getByText('Non-inferiority demonstrated')).toBeInTheDocument())
    expect(screen.getByText('Non-inferiority margin of 1.2 was met.')).toBeInTheDocument()
    expect(screen.getByText('B vs A')).toBeInTheDocument()
  })

  it('shows the error message from the backend on failure', async () => {
    installSession()
    server.use(
      http.get('/api/compute/test-session/unique/GROUP', () =>
        HttpResponse.json({ values: ['A', 'B'] }),
      ),
      http.post('/api/stats/noninferiority', () =>
        HttpResponse.json({ detail: 'Insufficient events for margin test' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<NonInferiorityPanel />)
    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'DM')
    await user.selectOptions(selects[1], 'GROUP')
    await waitFor(() => expect(screen.getByText('Test (new) arm')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: /run non-inferiority test/i }))

    await waitFor(() => expect(screen.getByText('Insufficient events for margin test')).toBeInTheDocument())
  })
})

describe('NonInferiorityPanel staleness', () => {
  it('closes the export of a margin test once the data changes under it', async () => {
    installSession()
    server.use(
      http.get('/api/compute/test-session/unique/GROUP', () => HttpResponse.json({ values: ['A', 'B'] })),
      http.post('/api/stats/noninferiority', () =>
        HttpResponse.json({
          non_inferior: true, effect: 'RR', estimate: 1.05, ci_level: 90, ci_low: 0.9, ci_high: 1.15,
          margin: 1.2, bound: 'upper', alpha_one_sided: 0.05, p_noninferiority: 0.01,
          test_group: 'B', ref_group: 'A', outcome_type: 'binary',
          export_rows: [['k', 'v'], ['RR', '1.05']],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<NonInferiorityPanel />)
    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'DM')
    await user.selectOptions(selects[1], 'GROUP')
    await waitFor(() => expect(screen.getByText('Test (new) arm')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: /run non-inferiority test/i }))

    const csv = await screen.findByRole('button', { name: 'CSV' })
    expect(csv).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeDisabled()
  })
})
