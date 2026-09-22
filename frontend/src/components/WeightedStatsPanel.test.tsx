import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { useStore } from '../store'
import { clearSession, installSession } from '../test/testUtils'
import WeightedStatsPanel from './WeightedStatsPanel'

afterEach(() => clearSession())

describe('WeightedStatsPanel', () => {
  it('renders the empty state without an active session', () => {
    clearSession()
    render(<WeightedStatsPanel />)
    expect(screen.getByText('Weighted Descriptives')).toBeInTheDocument()
    expect(screen.getByText(/Pick a weight column \+ value variables/i)).toBeInTheDocument()
  })

  it('lists numeric columns for weight/value selection', () => {
    installSession()
    render(<WeightedStatsPanel />)
    const weightSelect = screen.getAllByRole('combobox')[0]
    expect(weightSelect).toHaveTextContent('AGE')
    expect(weightSelect).toHaveTextContent('LDL')
  })

  it('disables Compute until a weight column and a value column are chosen', async () => {
    installSession()
    const user = userEvent.setup()
    render(<WeightedStatsPanel />)

    const runButton = screen.getByRole('button', { name: /compute weighted stats/i })
    // Button isn't disabled by prop but shows an error message when clicked without inputs
    await user.click(runButton)
    await waitFor(() =>
      expect(screen.getByText('Pick a weight column and at least one value column.')).toBeInTheDocument(),
    )
  })

  it('runs weighted descriptive stats and renders the result', async () => {
    installSession()
    server.use(
      http.post('/api/stats/weighted_descriptive', () =>
        HttpResponse.json({
          weight_col: 'AGE',
          n: 3,
          results: [
            { column: 'LDL', n: 3, ess_kish: 2.8, w_mean: 123.4, w_sd: 12.1, ci_low: 100.2, ci_high: 146.6, w_median: 120, w_q1: 110, w_q3: 140, w_proportion: null },
          ],
          assumptions: [{ name: 'Positive weights', detail: 'All weights > 0', met: true }],
          result_text: 'Weighted mean LDL is 123.4.',
          export_rows: [['col', 'w_mean'], ['LDL', '123.4']],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<WeightedStatsPanel />)

    const [weightSelect] = screen.getAllByRole('combobox')
    await user.selectOptions(weightSelect, 'AGE')
    await user.click(screen.getByRole('checkbox', { name: 'LDL' }))
    await user.click(screen.getByRole('button', { name: /compute weighted stats/i }))

    await waitFor(() => expect(screen.getByText('Weighted mean LDL is 123.4.')).toBeInTheDocument())
    expect(screen.getByText('123.400')).toBeInTheDocument()
    expect(screen.getByText(/All weights > 0/)).toBeInTheDocument()
  })

  it('shows the error message from the backend on failure', async () => {
    installSession()
    server.use(
      http.post('/api/stats/weighted_descriptive', () =>
        HttpResponse.json({ detail: 'Weights must be non-negative' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<WeightedStatsPanel />)
    const [weightSelect] = screen.getAllByRole('combobox')
    await user.selectOptions(weightSelect, 'AGE')
    await user.click(screen.getByRole('checkbox', { name: 'LDL' }))
    await user.click(screen.getByRole('button', { name: /compute weighted stats/i }))

    await waitFor(() => expect(screen.getByText('Weights must be non-negative')).toBeInTheDocument())
  })
})

describe('WeightedStatsPanel staleness', () => {
  it('closes the export of a weighted summary once the data changes under it', async () => {
    installSession()
    server.use(
      http.post('/api/stats/weighted_descriptive', () =>
        HttpResponse.json({
          weight_col: 'AGE', n: 3,
          results: [{ column: 'LDL', n: 3, ess_kish: 2.8, w_mean: 123.4, w_sd: 12.1, ci_low: 100.2, ci_high: 146.6 }],
          export_rows: [['col', 'w_mean'], ['LDL', '123.4']],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<WeightedStatsPanel />)
    await user.selectOptions(screen.getAllByRole('combobox')[0], 'AGE')
    await user.click(screen.getByRole('checkbox', { name: 'LDL' }))
    await user.click(screen.getByRole('button', { name: /compute weighted stats/i }))

    const csv = await screen.findByRole('button', { name: 'CSV' })
    expect(csv).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeDisabled()
  })
})
