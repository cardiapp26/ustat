import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import VisualModelPanel from './VisualModelPanel'

afterEach(() => clearSession())

const noMissing = () =>
  HttpResponse.json({ total_rows: 3, rows_affected: 0, pct_affected: 0, per_column: {} })

beforeEach(() => {
  // MissingGuard fires a GET on every render for the current column selection.
  server.use(http.get('/api/stats/:sid/missing', () => noMissing()))
})

const fourColSession = () =>
  makeSession({
    columns: [
      { name: 'Y', dtype: 'float64', kind: 'numeric' },
      { name: 'X', dtype: 'float64', kind: 'numeric' },
      { name: 'Z', dtype: 'float64', kind: 'numeric' },
      { name: 'PatientID', dtype: 'int64', kind: 'numeric' },
    ],
    preview: [
      { Y: 1, X: 2, Z: 3, PatientID: 1 },
      { Y: 2, X: 3, Z: 4, PatientID: 2 },
      { Y: 3, X: 4, Z: 5, PatientID: 3 },
    ],
  })

function selectAfterLabel(labelText: string): HTMLSelectElement {
  const label = screen.getByText(labelText)
  const wrapper = label.parentElement as HTMLElement
  return within(wrapper).getByRole('combobox') as HTMLSelectElement
}

/**
 * MissingGuard fires its own async GET on mount/column-change and re-renders
 * a fresh run-button instance once that settles. Clicking immediately after
 * render (or after selecting a tab/checkbox) risks landing on an
 * already-detached button. Give it a tick to settle before interacting.
 */
async function settle(): Promise<void> {
  await new Promise(resolve => setTimeout(resolve, 50))
}

describe('VisualModelPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<VisualModelPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('Polynomial tab: fits a polynomial model and renders coefficients + curve', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/models/polynomial', () =>
        HttpResponse.json({
          model: 'Polynomial Regression (degree 2)',
          n: 3,
          degree: 2,
          predictor: 'X',
          outcome: 'Y',
          r_squared: 0.95,
          adj_r_squared: 0.9,
          aic: 12.3,
          bic: 14.1,
          residual_se: 0.2,
          coefficients: [
            { variable: 'X', estimate: 0.5, se: 0.1, t: 5, p: 0.01 },
            { variable: 'X^2', estimate: 0.02, se: 0.01, t: 2, p: 0.04 },
          ],
          scatter: { x: [2, 3, 4], y: [1, 2, 3] },
          curve: { x: [2, 3, 4], y: [1, 2, 3], ci_low: [0.8, 1.8, 2.8], ci_high: [1.2, 2.2, 3.2] },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await settle()

    await user.selectOptions(selectAfterLabel('Outcome (continuous)'), 'Y')
    await user.selectOptions(selectAfterLabel('Predictor'), 'X')
    await settle()

    await user.click(await screen.findByRole('button', { name: 'Fit Polynomial' }))

    await waitFor(() =>
      expect(screen.getByText('Polynomial Regression (degree 2)')).toBeInTheDocument(),
    )
    expect(screen.getByText('X^2')).toBeInTheDocument()
    expect(screen.getAllByTestId('plotly-mock').length).toBeGreaterThan(0)
  })

  it('Polynomial tab: shows backend error message on failure', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/models/polynomial', () =>
        HttpResponse.json({ detail: 'Predictor has zero variance' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await settle()
    await user.click(await screen.findByRole('button', { name: 'Fit Polynomial' }))

    await waitFor(() =>
      expect(screen.getByText('Predictor has zero variance')).toBeInTheDocument(),
    )
  })

  it('LMM tab: fits a mixed model with grouping variable and renders ICC', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/models/lmm', () =>
        HttpResponse.json({
          model: 'Linear Mixed Model',
          n: 3,
          n_groups: 3,
          group: 'PatientID',
          icc: 0.12,
          aic: 20.1,
          bic: 22.4,
          random_effect_variance: 0.5,
          residual_variance: 1.1,
          coefficients: [{ variable: 'X', estimate: 0.4, se: 0.1, t: 4, p: 0.02 }],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await settle()
    await user.click(screen.getByRole('button', { name: 'Linear Mixed Model' }))

    // Fixed effects checkbox for X (PatientID auto-selected as group, blocked as FE)
    const fixedEffectsLabel = screen.getByText('Fixed effects (predictors)')
    const feGroup = fixedEffectsLabel.parentElement as HTMLElement
    const xLabel = within(feGroup).getByText('X').closest('label') as HTMLLabelElement
    await user.click(within(xLabel).getByRole('checkbox'))
    await settle()

    const runBtn = await screen.findByRole('button', { name: 'Fit LMM' })
    expect(runBtn).toBeEnabled()
    await user.click(runBtn)

    await waitFor(() => expect(screen.getAllByText('Linear Mixed Model').length).toBeGreaterThan(0))
    expect(screen.getByText('0.1200')).toBeInTheDocument()
  })

  it('GLM tab: fits a Gamma model and renders coefficients', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/models/gamma', () =>
        HttpResponse.json({
          model: 'Gamma GLM (log link)',
          n: 3,
          aic: 15.2,
          bic: 17.0,
          deviance: 2.1,
          scale: 0.3,
          coefficients: [{ variable: 'X', estimate: 0.3, se: 0.1, z: 3, p: 0.03 }],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await settle()
    await user.click(screen.getByRole('button', { name: 'GLM (Gamma / Neg. Binom.)' }))

    const predictorsLabel = screen.getByText('Predictors')
    const predGroup = predictorsLabel.parentElement as HTMLElement
    const xLabel = within(predGroup).getByText('X').closest('label') as HTMLLabelElement
    await user.click(within(xLabel).getByRole('checkbox'))
    await settle()

    const runBtn = await screen.findByRole('button', { name: 'Fit GLM' })
    expect(runBtn).toBeEnabled()
    await user.click(runBtn)

    await waitFor(() => expect(screen.getByText('Gamma GLM (log link)')).toBeInTheDocument())
    expect(screen.getByText(/multiplicative change/)).toBeInTheDocument()
  })

  it('Diagnostic Plots tab: runs diagnostics and renders four plots', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/models/linear_diag', () =>
        HttpResponse.json({
          n: 3,
          r_squared: 0.8,
          residual_se: 0.3,
          residuals_fitted: { x: [1, 2, 3], y: [0.1, -0.1, 0.05] },
          qq: { theoretical: [-1, 0, 1], sample: [-0.9, 0.05, 1.1], line_x: [-1, 1], line_y: [-1, 1] },
          scale_location: { x: [1, 2, 3], y: [0.3, 0.4, 0.2] },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await settle()
    await user.click(screen.getByRole('button', { name: 'Diagnostic Plots' }))

    const predictorsLabel = screen.getByText('Predictors')
    const predGroup = predictorsLabel.parentElement as HTMLElement
    const xLabel = within(predGroup).getByText('X').closest('label') as HTMLLabelElement
    await user.click(within(xLabel).getByRole('checkbox'))
    await settle()

    await user.click(await screen.findByRole('button', { name: 'Run Diagnostics' }))

    await waitFor(() => expect(screen.getByText('Model summary')).toBeInTheDocument())
    const summary = screen.getByText('Model summary').parentElement as HTMLElement
    expect(summary.textContent).toMatch(/R²\s*=\s*0\.800/)
    const plots = screen.getAllByTestId('plotly-mock')
    expect(plots.length).toBe(4)
  })

  it('Diagnostic Plots tab: shows backend error message on failure', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/models/linear_diag', () =>
        HttpResponse.json({ detail: 'Model failed to converge' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await settle()
    await user.click(screen.getByRole('button', { name: 'Diagnostic Plots' }))

    const predictorsLabel = screen.getByText('Predictors')
    const predGroup = predictorsLabel.parentElement as HTMLElement
    const xLabel = within(predGroup).getByText('X').closest('label') as HTMLLabelElement
    await user.click(within(xLabel).getByRole('checkbox'))
    await settle()

    await user.click(await screen.findByRole('button', { name: 'Run Diagnostics' }))

    await waitFor(() => expect(screen.getByText('Model failed to converge')).toBeInTheDocument())
  })

  it('Polynomial tab: closes the curve export once the data changes under the fit', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/models/polynomial', () =>
        HttpResponse.json({
          model: 'Polynomial Regression (degree 2)',
          n: 3, degree: 2, predictor: 'X', outcome: 'Y',
          r_squared: 0.95, adj_r_squared: 0.9, aic: 12.3, bic: 14.1, residual_se: 0.2,
          coefficients: [{ variable: 'X', estimate: 0.5, se: 0.1, t: 5, p: 0.01 }],
          scatter: { x: [2, 3, 4], y: [1, 2, 3] },
          curve: { x: [2, 3, 4], y: [1, 2, 3], ci_low: [0.8, 1.8, 2.8], ci_high: [1.2, 2.2, 3.2] },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await settle()
    await user.click(await screen.findByRole('button', { name: 'Fit Polynomial' }))
    await screen.findByText('Polynomial Regression (degree 2)')

    const exports = () => [screen.getByRole('button', { name: '⧉' }), screen.getByRole('button', { name: '↓' })]
    for (const b of exports()) expect(b).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    for (const b of exports()) expect(b).toBeDisabled()
  })

  it('Diagnostic Plots tab: closes all four plot exports once the data changes', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/models/linear_diag', () =>
        HttpResponse.json({
          n: 3, r_squared: 0.8, residual_se: 0.3,
          residuals_fitted: { x: [1, 2, 3], y: [0.1, -0.1, 0.05] },
          qq: { theoretical: [-1, 0, 1], sample: [-0.9, 0.05, 1.1], line_x: [-1, 1], line_y: [-1, 1] },
          scale_location: { x: [1, 2, 3], y: [0.3, 0.4, 0.2] },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await settle()
    await user.click(screen.getByRole('button', { name: 'Diagnostic Plots' }))
    const predGroup = screen.getByText('Predictors').parentElement as HTMLElement
    await user.click(within(within(predGroup).getByText('X').closest('label') as HTMLElement).getByRole('checkbox'))
    await settle()
    await user.click(await screen.findByRole('button', { name: 'Run Diagnostics' }))
    await screen.findByText('Model summary')

    expect(screen.getAllByRole('button', { name: '↓' })).toHaveLength(4)
    for (const b of screen.getAllByRole('button', { name: '↓' })) expect(b).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    for (const b of screen.getAllByRole('button', { name: '↓' })) expect(b).toBeDisabled()
    for (const b of screen.getAllByRole('button', { name: '⧉' })) expect(b).toBeDisabled()
  })

  it('LMM tab: reshaping to long format marks the fit on the wide data out of date', async () => {
    // A same-session setSession does not bump the data version, so without
    // the panel bumping it the old fit would still read as current.
    installSession(makeSession({
      columns: [
        { name: 'Y', dtype: 'float64', kind: 'numeric' },
        { name: 'INHOSPITALEF', dtype: 'float64', kind: 'numeric' },
        { name: 'CONTROLEF', dtype: 'float64', kind: 'numeric' },
        { name: 'PatientID', dtype: 'int64', kind: 'numeric' },
      ],
      preview: [
        { Y: 1, INHOSPITALEF: 50, CONTROLEF: 55, PatientID: 1 },
        { Y: 2, INHOSPITALEF: 45, CONTROLEF: 60, PatientID: 2 },
        { Y: 3, INHOSPITALEF: 40, CONTROLEF: 52, PatientID: 3 },
      ],
    }))
    server.use(
      http.post('/api/models/lmm', () =>
        HttpResponse.json({
          model: 'Linear Mixed Model', n: 3, n_groups: 3, group: 'PatientID', icc: 0.12,
          coefficients: [{ variable: 'INHOSPITALEF', estimate: 0.4, se: 0.1, t: 4, p: 0.02 }],
        }),
      ),
      http.post('/api/models/melt', () => HttpResponse.json({ ok: true })),
      http.get('/api/stats/:sid/refresh', () => HttpResponse.json({})),
    )

    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await settle()
    await user.click(screen.getByRole('button', { name: 'Linear Mixed Model' }))
    const feGroup = screen.getByText('Fixed effects (predictors)').parentElement as HTMLElement
    await user.click(within(within(feGroup).getByText('INHOSPITALEF').closest('label') as HTMLElement).getByRole('checkbox'))
    await settle()
    await user.click(await screen.findByRole('button', { name: 'Fit LMM' }))
    await screen.findByText('0.1200')
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()

    await user.click(screen.getByText(/Repeated measures detected/))
    await user.click(screen.getByRole('radio'))
    await user.click(screen.getByRole('button', { name: /Melt → Long Format/ }))

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByText(/the data changed/)).toBeInTheDocument()
  })
})
