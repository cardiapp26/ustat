import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import RCSPanel from './RCSPanel'

afterEach(() => clearSession())

const noMissing = () =>
  HttpResponse.json({ total_rows: 4, rows_affected: 0, pct_affected: 0, per_column: {} })

beforeEach(() => {
  server.use(http.get('/api/stats/:sid/missing', () => noMissing()))
})

/**
 * MissingGuard fires its own async GET on mount and swaps in a fresh
 * run-button once it settles — give it a tick before interacting further.
 */
async function settle(): Promise<void> {
  await new Promise(resolve => setTimeout(resolve, 50))
}

const rcsSession = () =>
  makeSession({
    columns: [
      { name: 'AGE', dtype: 'float64', kind: 'numeric' },
      { name: 'LDL', dtype: 'float64', kind: 'numeric' },
      { name: 'TIME', dtype: 'float64', kind: 'numeric' },
      { name: 'EVENT', dtype: 'int64', kind: 'numeric' },
      { name: 'DM', dtype: 'int64', kind: 'numeric' },
    ],
    preview: [
      { AGE: 55, LDL: 120, TIME: 10, EVENT: 1, DM: 0 },
      { AGE: 62, LDL: 140, TIME: 20, EVENT: 0, DM: 1 },
      { AGE: 48, LDL: 110, TIME: 15, EVENT: 1, DM: 0 },
      { AGE: 70, LDL: 160, TIME: 5,  EVENT: 1, DM: 1 },
    ],
  })

describe('RCSPanel', () => {
  it('renders the RCS mode by default and disables nothing without a run', () => {
    installSession(rcsSession())
    render(<RCSPanel />)
    expect(screen.getByText('RCS Dose-Response')).toBeInTheDocument()
    expect(screen.getByText('Configure and fit an RCS model')).toBeInTheDocument()
  })

  it('Univariate RCS (Cox outcome): runs and renders the dose-response curve', async () => {
    installSession(rcsSession())
    server.use(
      http.post('/api/models/rcs', () =>
        HttpResponse.json({
          x_values: [100, 120, 140, 160],
          or_values: [0.9, 1.0, 1.3, 1.8],
          ci_low: [0.7, 0.8, 1.0, 1.2],
          ci_high: [1.1, 1.2, 1.6, 2.4],
          x_data: [120, 140, 110, 160],
          predictor: 'LDL',
          model_type: 'cox',
          duration_col: 'TIME',
          event_col: 'EVENT',
          knots: [100, 130, 160],
          n_knots: 3,
          ref_value: 130,
          n: 4,
          n_total: 4,
          n_events: 3,
          aic: 40.2,
          nonlinearity_p: 0.03,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()

    const runBtn = await screen.findByRole('button', { name: 'Run RCS' })
    await user.click(runBtn)

    await waitFor(() => expect(document.querySelector('h4')?.textContent).toContain('Cox-RCS'))
    expect(screen.getByText(/events = 3/)).toBeInTheDocument()
    expect(screen.getAllByTestId('plotly-mock').length).toBeGreaterThan(0)
    // non-linearity badge
    expect(screen.getByText(/non-linearity/)).toBeInTheDocument()
  })

  it('Univariate RCS: shows backend error message on failure', async () => {
    installSession(rcsSession())
    server.use(
      http.post('/api/models/rcs', () =>
        HttpResponse.json({ detail: 'Not enough events for 4 knots' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()

    const runBtn = await screen.findByRole('button', { name: 'Run RCS' })
    await user.click(runBtn)

    await waitFor(() =>
      expect(screen.getByText('Not enough events for 4 knots')).toBeInTheDocument(),
    )
  })

  it('Cox-RCS (multivariable) mode: runs and renders coefficient table', async () => {
    installSession(rcsSession())
    server.use(
      http.post('/api/models/survival/cox_rcs', () =>
        HttpResponse.json({
          coefficients: [
            { name: 'LDL', coef: 0.02, hr: 1.02, se: 0.01, z: 2.0, p: 0.04, ci_low: 1.0, ci_high: 1.05 },
            { name: 'DM', coef: 0.5, hr: 1.65, se: 0.2, z: 2.5, p: 0.01, ci_low: 1.1, ci_high: 2.4 },
          ],
          curves_1d: [
            { column: 'LDL', x: [100, 130, 160], hr: [0.9, 1.0, 1.3], lower: [0.7, 0.8, 1.0], upper: [1.1, 1.2, 1.6], knots: [100, 130, 160], ref: 130 },
          ],
          surface_2d: null,
          interaction: null,
          nonlinearity: { LDL: { wald: 5.2, df: 2, p: 0.07 } },
          n: 4,
          n_events: 3,
          concordance: 0.71,
          aic: 38.5,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()

    await user.click(screen.getByRole('radio', { name: /Cox-RCS \(multivariable\)/ }))
    await settle()

    const runBtn = await screen.findByRole('button', { name: 'Run Cox-RCS' })
    await user.click(runBtn)

    await waitFor(() =>
      expect(screen.getByText('Cox proportional hazards (RCS)')).toBeInTheDocument(),
    )
    expect(screen.getByText(/C-index = 0.710/)).toBeInTheDocument()
    const rows = screen.getAllByRole('row')
    expect(rows.length).toBeGreaterThanOrEqual(2) // header + 2 coefficient rows
    const table = screen.getByRole('table')
    expect(within(table).getByText('DM')).toBeInTheDocument()
  })

  it('Cox-RCS (multivariable) mode: shows backend error message on failure', async () => {
    installSession(rcsSession())
    server.use(
      http.post('/api/models/survival/cox_rcs', () =>
        HttpResponse.json({ detail: 'Model failed to converge' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()

    await user.click(screen.getByRole('radio', { name: /Cox-RCS \(multivariable\)/ }))
    await settle()

    const runBtn = await screen.findByRole('button', { name: 'Run Cox-RCS' })
    await user.click(runBtn)

    await waitFor(() => expect(screen.getByText('Model failed to converge')).toBeInTheDocument())
  })

  it('Univariate RCS: closes the table and figure exports once the data changes under the curve', async () => {
    installSession(rcsSession())
    server.use(
      http.post('/api/models/rcs', () =>
        HttpResponse.json({
          x_values: [100, 130, 160], or_values: [0.9, 1.0, 1.4],
          ci_low: [0.7, 1.0, 1.1], ci_high: [1.1, 1.0, 1.8], x_data: [120, 140, 110, 160],
          predictor: 'LDL', model_type: 'cox', duration_col: 'TIME', event_col: 'EVENT',
          knots: [100, 130, 160], n_knots: 3, ref_value: 130, n: 4, n_events: 3,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()
    await user.click(await screen.findByRole('button', { name: 'Run RCS' }))

    const csv = await screen.findByRole('button', { name: 'CSV' })
    // The figure's own exporter (TitledPlot), beside the table exporter.
    const figure = [screen.getByTitle('Export chart'), screen.getByTitle('Copy chart to clipboard as PNG')]
    expect(csv).toBeEnabled()
    for (const b of figure) expect(b).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'PNG 300dpi' })).toBeDisabled()
    for (const b of figure) expect(b).toBeDisabled()
  })

  it('Cox-RCS: drops the modebar camera from the HR curves once the data changes', async () => {
    installSession(rcsSession())
    server.use(
      http.post('/api/models/survival/cox_rcs', () =>
        HttpResponse.json({
          coefficients: [{ name: 'LDL', coef: 0.02, hr: 1.02, se: 0.01, z: 2.0, p: 0.04, ci_low: 1.0, ci_high: 1.05 }],
          curves_1d: [{ column: 'LDL', x: [100, 130, 160], hr: [0.9, 1.0, 1.3], lower: [0.7, 0.8, 1.0], upper: [1.1, 1.2, 1.6], knots: [100, 130, 160], ref: 130 }],
          surface_2d: null, interaction: null, n: 4, n_events: 3, concordance: 0.71,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()
    await user.click(screen.getByRole('radio', { name: /Cox-RCS \(multivariable\)/ }))
    await settle()
    await user.click(await screen.findByRole('button', { name: 'Run Cox-RCS' }))
    await screen.findByText('Cox proportional hazards (RCS)')

    const removed = () => {
      const config = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-config') ?? '{}')
      return (config.modeBarButtonsToRemove ?? []) as string[]
    }
    expect(removed()).not.toContain('toImage')

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(removed()).toContain('toImage')
  })
})
