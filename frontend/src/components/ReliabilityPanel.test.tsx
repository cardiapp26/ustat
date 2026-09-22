import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import { useStore } from '../store'
import ReliabilityPanel from './ReliabilityPanel'

afterEach(() => clearSession())

describe('ReliabilityPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<ReliabilityPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('lists numeric columns as selectable scale items, excludes categoricals', () => {
    installSession()
    render(<ReliabilityPanel />)
    const listbox = screen.getByRole('listbox')
    expect(within(listbox).getByText('AGE')).toBeInTheDocument()
    expect(within(listbox).getByText('LDL')).toBeInTheDocument()
    expect(within(listbox).queryByText('GROUP')).not.toBeInTheDocument()
  })

  it('disables Compute until at least 2 items are selected', () => {
    installSession()
    render(<ReliabilityPanel />)
    expect(screen.getByRole('button', { name: /compute reliability/i })).toBeDisabled()
  })

  it('runs Cronbach alpha and renders the result on success', async () => {
    installSession()
    server.use(
      http.post('/api/reliability/cronbach', () =>
        HttpResponse.json({
          alpha: 0.842,
          omega: 0.86,
          interpretation: 'Good internal consistency',
          k: 2,
          n: 3,
          scale_summary: { mean: 1.5, sd: 0.4, min: 1, max: 2, skewness: 0.1 },
          item_stats: [
            { item: 'AGE', mean: 55, sd: 5, item_total_r: 0.6, alpha_if_deleted: 0.8 },
            { item: 'LDL', mean: 120, sd: 10, item_total_r: 0.5, alpha_if_deleted: 0.9 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ReliabilityPanel />)

    const listbox = screen.getByRole('listbox')
    await user.selectOptions(listbox, ['AGE', 'LDL'])

    const runButton = screen.getByRole('button', { name: /compute reliability/i })
    expect(runButton).toBeEnabled()
    await user.click(runButton)

    await waitFor(() => expect(screen.getByText(/α = 0\.842/)).toBeInTheDocument())
    expect(screen.getByText(/ω = 0\.860/)).toBeInTheDocument()
    expect(screen.getByText('Good internal consistency')).toBeInTheDocument()
    expect(screen.getByText((_, el) => el?.textContent === '2 items, n = 3')).toBeInTheDocument()
    // Item analysis table renders both items
    expect(screen.getByText('Item Analysis')).toBeInTheDocument()
    const rows = screen.getAllByRole('row')
    expect(rows.length).toBeGreaterThanOrEqual(3) // header + 2 items
    // Alpha-if-deleted > alpha flags a "drop?" suggestion for LDL (0.9 > 0.842)
    expect(screen.getByText(/drop\?/)).toBeInTheDocument()
  })

  it('closes the export of an alpha once the data changes under it', async () => {
    installSession()
    server.use(
      http.post('/api/reliability/cronbach', () =>
        HttpResponse.json({
          alpha: 0.842, k: 2, n: 3, interpretation: 'Good internal consistency',
          export_rows: [['Statistic', 'Value'], ['Cronbach alpha', '0.842']],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ReliabilityPanel />)
    await user.selectOptions(screen.getByRole('listbox'), ['AGE', 'LDL'])
    await user.click(screen.getByRole('button', { name: /compute reliability/i }))

    const csv = await screen.findByRole('button', { name: 'CSV' })
    expect(csv).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'XLSX' })).toBeDisabled()
  })

  it('shows the error message from the backend on failure', async () => {
    installSession()
    server.use(
      http.post('/api/reliability/cronbach', () =>
        HttpResponse.json({ detail: 'Items must be numeric' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<ReliabilityPanel />)
    await user.selectOptions(screen.getByRole('listbox'), ['AGE', 'LDL'])
    await user.click(screen.getByRole('button', { name: /compute reliability/i }))

    await waitFor(() => expect(screen.getByText('Items must be numeric')).toBeInTheDocument())
  })
})
