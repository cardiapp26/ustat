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
 * MissingGuard fires its own async GET on mount; let it settle so the
 * "Checking missing data…" line is gone before interacting further.
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

  it('keeps identifiers available for manual selection without choosing them by default', () => {
    const session = rcsSession()
    session.columns.unshift({ name: 'pid', dtype: 'int64', kind: 'numeric', role: 'id' })
    installSession(session)
    render(<RCSPanel />)

    for (const label of ['Predictor (continuous)', 'Duration column']) {
      const picker = within(screen.getByText(label).parentElement!).getByRole('combobox')
      expect(picker).toHaveValue('AGE')
      expect(within(picker).getByRole('option', { name: 'pid' })).toBeInTheDocument()
    }
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
          knots: [0.010001, 0.010002, 0.01456],
          n_knots: 3,
          ref_value: 0.01123456,
          nadir: { x: 0.01123456, at_boundary: false },
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
    expect(screen.getByText('0.010001')).toBeInTheDocument()
    expect(screen.getByText('0.010002')).toBeInTheDocument()
    expect(screen.getByText('0.014560')).toBeInTheDocument()
    expect(screen.getByText('nadir = 0.011235')).toBeInTheDocument()
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

describe('RCSPanel absolute risk (Cox)', () => {
  const coxResult = {
    x_values: [40, 60, 80],
    or_values: [0.8, 1.0, 1.6],
    ci_low: [0.6, 1.0, 1.2],
    ci_high: [1.1, 1.0, 2.1],
    x_data: [55, 62, 48, 70],
    predictor: 'AGE',
    model_type: 'cox',
    duration_col: 'TIME',
    event_col: 'EVENT',
    knots: [45, 60, 75],
    n_knots: 3,
    ref_value: 60,
    n: 4,
    n_events: 3,
    nonlinearity_p: 0.2,
    overall_p: 0.01,
    absolute_risk: {
      horizon: 12,
      horizon_is_default: false,
      n_at_risk: 15,
      n_events_by_horizon: 2,
      x_values: [40, 60, 80],
      risk: [0.2, 0.3, 0.5],
      ci_low: [0.1, 0.2, 0.4],
      ci_high: [0.3, 0.4, 0.6],
    },
  }

  it('sends the horizon and switches the curve to absolute risk', async () => {
    installSession(rcsSession())
    let sent: Record<string, unknown> | null = null
    server.use(
      http.post('/api/models/rcs', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json(coxResult)
      }),
    )
    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()

    await user.click(screen.getByRole('button', { name: 'Cox' }))
    await user.type(screen.getByLabelText(/Absolute-risk horizon/), '12')
    await user.click(await screen.findByRole('button', { name: 'Run RCS' }))

    await waitFor(() => expect(sent).not.toBeNull())
    expect(sent!.risk_horizon).toBe(12)

    const riskBtn = await screen.findByRole('button', { name: /Absolute risk at TIME = 12/ })
    expect(screen.getByRole('button', { name: 'Hazard ratio' })).toHaveAttribute('aria-pressed', 'true')
    await user.click(riskBtn)
    expect(riskBtn).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText(/survfit.coxph/)).toBeInTheDocument()
    expect(screen.getByText(/Only 15 patients remain at risk/)).toBeInTheDocument()
    expect(screen.queryByText('Log Scale (Y)')).not.toBeInTheDocument()
  })

  it('rejects a non-positive horizon before calling the server', async () => {
    installSession(rcsSession())
    let called = false
    server.use(http.post('/api/models/rcs', () => { called = true; return HttpResponse.json(coxResult) }))
    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()

    await user.click(screen.getByRole('button', { name: 'Cox' }))
    await user.type(screen.getByLabelText(/Absolute-risk horizon/), '-3')
    await user.click(await screen.findByRole('button', { name: 'Run RCS' }))

    await waitFor(() => expect(screen.getByText(/Risk horizon must be a positive number/)).toBeInTheDocument())
    expect(called).toBe(false)
  })
})

describe('RCSPanel reference and curve layers', () => {
  const result = {
    x_values: [40, 50, 60, 70],
    or_values: [1.8, 1.0, 1.1, 2.2],
    ci_low: [1.2, 1.0, 0.8, 1.4],
    ci_high: [2.6, 1.0, 1.5, 3.4],
    x_data: [45, 55, 65],
    predictor: 'AGE',
    outcome: 'DM',
    model_type: 'logistic',
    knots: [45, 55, 65],
    n_knots: 3,
    ref_value: 50,
    ref_strategy_used: 'nadir',
    nadir: { x: 50, at_boundary: false },
    x_distribution: { edges: [40, 55, 70], counts: [2, 2], n_outside: 0 },
    n: 4,
    nonlinearity_p: 0.01,
  }

  it('sends the nadir strategy and shows the nadir, CI regions and distribution toggle', async () => {
    installSession(rcsSession())
    let sent: Record<string, unknown> | null = null
    server.use(
      http.post('/api/models/rcs', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json(result)
      }),
    )
    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()

    await user.click(screen.getByRole('button', { name: 'Logistic' }))
    await user.click(screen.getByRole('button', { name: 'Nadir (lowest risk)' }))
    await user.click(await screen.findByRole('button', { name: 'Run RCS' }))

    await waitFor(() => expect(sent).not.toBeNull())
    expect(sent!.ref_strategy).toBe('nadir')
    expect(sent!.outcome).toBe('EVENT')
    expect(await screen.findByText('nadir = 50.00')).toBeInTheDocument()
    expect(screen.getByText(/95% CI excludes OR = 1 for AGE/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Show distribution' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('flags a curve without an interior nadir', async () => {
    installSession(rcsSession())
    server.use(http.post('/api/models/rcs', () =>
      HttpResponse.json({ ...result, nadir: { x: 40, at_boundary: true }, ref_strategy_used: 'median' })))
    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()
    await user.click(await screen.findByRole('button', { name: 'Run RCS' }))
    expect(await screen.findByText('no interior nadir (lowest at 40.00)')).toBeInTheDocument()
  })
})

describe('RCSPanel count outcomes (Poisson / quasi-Poisson)', () => {
  const countResult = {
    x_values: [40, 50, 60, 70],
    or_values: [0.8, 1.0, 1.2, 1.6],
    ci_low: [0.6, 1.0, 1.0, 1.2],
    ci_high: [1.1, 1.0, 1.4, 2.1],
    x_data: [45, 55, 65],
    predictor: 'AGE',
    outcome: 'LDL',
    knots: [45, 55, 65],
    n_knots: 3,
    ref_value: 50,
    n: 4,
    total_count: 530,
    nonlinearity_p: 0.2,
  }

  it('sends a Poisson fit with exposure and labels the curve as IRR with an overdispersion warning', async () => {
    installSession(rcsSession())
    let sent: Record<string, unknown> | null = null
    server.use(
      http.post('/api/models/rcs', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          ...countResult, model_type: 'poisson', exposure_col: 'TIME', dispersion: 2.18,
          interaction: { covariates: ['DM'], test: 'chi2', lr_stat: 3.62, df: 3, p: 0.305 },
        })
      }),
    )
    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()

    await user.click(screen.getByRole('button', { name: 'Poisson' }))
    expect(screen.getByText(/Outcome \(count, whole numbers/)).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText(/Exposure \(optional\)/), 'TIME')
    await user.click(await screen.findByRole('button', { name: 'Run RCS' }))

    await waitFor(() => expect(sent).not.toBeNull())
    expect(sent!.model_type).toBe('poisson')
    expect(sent!.exposure_col).toBe('TIME')
    expect(await screen.findByText(/: Poisson RCS/)).toBeInTheDocument()
    expect(screen.getByText(/total count = 530/)).toBeInTheDocument()
    expect(screen.getByText(/overdispersed, the Poisson CI is too narrow; refit as quasi-Poisson/)).toBeInTheDocument()
    expect(screen.getByText(/95% CI excludes IRR = 1 for AGE/)).toBeInTheDocument()
    expect(screen.getByText(/rate of/)).toBeInTheDocument()
    expect(screen.getByText(/χ²\(3\) = 3\.62/)).toBeInTheDocument()
  })

  it('shows F tests, the scaled-dispersion note and no exposure for quasi-Poisson', async () => {
    installSession(rcsSession())
    let sent: Record<string, unknown> | null = null
    server.use(
      http.post('/api/models/rcs', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          ...countResult, model_type: 'quasipoisson', dispersion: 2.18, aic: null, knot_comparison: [],
          test_statistic: 'F', overall_f: 6.71, overall_df: 3, overall_p: 0.0002,
          interaction: { covariates: ['DM'], test: 'F', f: 0.55, df: 3, df_resid: 290, p: 0.649 },
        })
      }),
    )
    const user = userEvent.setup()
    render(<RCSPanel />)
    await settle()

    await user.click(screen.getByRole('button', { name: 'Quasi-Poisson' }))
    await user.click(await screen.findByRole('button', { name: 'Run RCS' }))

    await waitFor(() => expect(sent).not.toBeNull())
    expect(sent!.model_type).toBe('quasipoisson')
    expect(sent!.exposure_col).toBeUndefined()
    expect(await screen.findByText(/: Quasi-Poisson RCS/)).toBeInTheDocument()
    expect(screen.getByText(/standard errors scaled by/)).toBeInTheDocument()
    expect(screen.getByText(/F\(3, 290\) = 0\.55/)).toBeInTheDocument()
    expect(screen.getByText(/interaction \(F test\)/)).toBeInTheDocument()
    expect(screen.getByText(/quasi-Poisson has no likelihood/)).toBeInTheDocument()
    expect(screen.getByText(/expected count of/)).toBeInTheDocument()
  })
})
