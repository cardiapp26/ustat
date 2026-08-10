import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import ModelsPanel from './ModelsPanel'

afterEach(() => clearSession())

// Silence sparklines / missing-data checks that fire on mount for every
// column combination this panel touches (MissingGuard + useModelData).
function stubBackgroundEndpoints() {
  server.use(
    http.get(/\/api\/stats\/.*\/sparklines/, () => HttpResponse.json({})),
    http.get(/\/api\/stats\/.*\/missing/, () =>
      HttpResponse.json({ total_rows: 3, rows_affected: 0, pct_affected: 0, per_column: {} }),
    ),
  )
}

const regressionSession = () =>
  makeSession({
    columns: [
      { name: 'AGE', dtype: 'float64', kind: 'numeric' },
      { name: 'LDL', dtype: 'float64', kind: 'numeric' },
      { name: 'BP', dtype: 'float64', kind: 'numeric' },
      { name: 'DEATH', dtype: 'int64', kind: 'numeric' },
      { name: 'TIME', dtype: 'float64', kind: 'numeric' },
    ],
    preview: [
      { AGE: 55, LDL: 120, BP: 130, DEATH: 0, TIME: 10 },
      { AGE: 62, LDL: 140, BP: 140, DEATH: 1, TIME: 5 },
      { AGE: 48, LDL: 110, BP: 120, DEATH: 0, TIME: 12 },
    ],
  })

function selectAfterLabel(labelText: string | RegExp): HTMLSelectElement {
  const label = screen.getByText(labelText, { selector: 'label' })
  const wrapper = label.parentElement as HTMLElement
  return within(wrapper).getByRole('combobox') as HTMLSelectElement
}

/** Find the predictor checkbox for `name` — disambiguates from <option> text
 *  in the outcome/group selects by requiring an ancestor <label> with a
 *  checkbox input (not a <select>). */
function checkPredictor(name: string) {
  const candidates = screen.getAllByText(name)
  for (const el of candidates) {
    const label = el.closest('label')
    if (!label) continue
    const checkbox = within(label).queryByRole('checkbox')
    if (checkbox) return checkbox as HTMLInputElement
  }
  throw new Error(`No predictor checkbox found for ${name}`)
}

describe('ModelsPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<ModelsPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('Linear regression: fits a model and renders coefficients on success', async () => {
    stubBackgroundEndpoints()
    installSession(regressionSession())
    server.use(
      http.post('/api/models/linear', () =>
        HttpResponse.json({
          model: 'Linear Regression',
          n: 3,
          outcome: 'BP',
          r_squared: 0.85,
          adj_r_squared: 0.7,
          f_stat: 12.3,
          coefficients: [
            { variable: 'const', estimate: 10, se: 1, t: 10, p: 0.001, ci_low: 8, ci_high: 12 },
            { variable: 'AGE', estimate: 0.9, se: 0.2, t: 4.5, p: 0.01, ci_low: 0.5, ci_high: 1.3 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ModelsPanel />)

    await user.selectOptions(selectAfterLabel('Outcome'), 'BP')
    await user.click(checkPredictor('AGE'))

    const runBtn = screen.getByRole('button', { name: 'Fit Model' })
    expect(runBtn).toBeEnabled()
    await user.click(runBtn)

    await waitFor(() =>
      expect(screen.getByText('Linear Regression', { selector: 'h4' })).toBeInTheDocument(),
    )
    expect(screen.getByText('0.8500')).toBeInTheDocument()
  })

  it('Linear regression: shows the backend error message on failure', async () => {
    stubBackgroundEndpoints()
    installSession(regressionSession())
    server.use(
      http.post('/api/models/linear', () =>
        HttpResponse.json({ detail: 'Outcome has zero variance' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<ModelsPanel />)
    await user.selectOptions(selectAfterLabel('Outcome'), 'BP')
    await user.click(checkPredictor('AGE'))
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    await waitFor(() => expect(screen.getByText('Outcome has zero variance')).toBeInTheDocument())
  })

  it('Logistic regression: fits a model and renders odds ratios on success', async () => {
    stubBackgroundEndpoints()
    installSession(regressionSession())
    server.use(
      http.post('/api/models/logistic', () =>
        HttpResponse.json({
          model: 'Logistic Regression',
          n: 3,
          outcome: 'DEATH',
          pseudo_r2: 0.3,
          coefficients: [
            { variable: 'const', log_odds: -1, se: 0.5, z: -2, p: 0.04, odds_ratio: 0.37, or_ci_low: 0.1, or_ci_high: 0.9 },
            { variable: 'AGE', log_odds: 0.05, se: 0.02, z: 2.5, p: 0.02, odds_ratio: 1.05, or_ci_low: 1.01, or_ci_high: 1.1 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ModelsPanel />)
    await user.click(screen.getByText('Logistic Regression', { selector: 'span' }))
    await user.selectOptions(selectAfterLabel(/^Outcome/), 'DEATH')
    await user.click(checkPredictor('AGE'))
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    await waitFor(() =>
      expect(screen.getByText('Logistic Regression', { selector: 'h4' })).toBeInTheDocument(),
    )
    expect(screen.getByText('1.050')).toBeInTheDocument()
  })

  it('Firth logistic: shows the backend error message on failure', async () => {
    stubBackgroundEndpoints()
    installSession(regressionSession())
    server.use(
      http.post('/api/models/firth_logistic', () =>
        HttpResponse.json({ detail: 'Perfect separation could not be resolved' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<ModelsPanel />)
    await user.click(screen.getByText('Firth Logistic (penalized)'))
    await user.selectOptions(selectAfterLabel(/^Outcome/), 'DEATH')
    await user.click(checkPredictor('AGE'))
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    await waitFor(() =>
      expect(screen.getByText('Perfect separation could not be resolved')).toBeInTheDocument(),
    )
  })

  it('OR Table: runs univariate + multivariate logistic table and renders both columns', async () => {
    stubBackgroundEndpoints()
    installSession(regressionSession())
    server.use(
      http.post('/api/models/logistic_table', () =>
        HttpResponse.json({
          model: 'OR Table',
          outcome: 'DEATH',
          selection_method: 'Univariate p < 0.10',
          table: [
            { variable: 'AGE', uni_or: 1.05, uni_ci_low: 1.0, uni_ci_high: 1.1, uni_p: 0.03, multi_or: 1.08, multi_ci_low: 1.01, multi_ci_high: 1.15, multi_p: 0.02 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ModelsPanel />)
    const orRadioEl = document.querySelector('input[name="model"][value="ortable"]') as HTMLElement
    await user.click(orRadioEl)
    await user.selectOptions(selectAfterLabel(/^Outcome/), 'DEATH')
    await user.click(checkPredictor('AGE'))
    const btn = screen.getByRole('button', { name: 'Fit Model' })
    await user.click(btn)

    await waitFor(() => expect(screen.getByText('DEATH', { selector: 'span' })).toBeInTheDocument())
    expect(screen.getByText('1.05 (1.00–1.10)')).toBeInTheDocument()
    expect(screen.getByText('1.08 (1.01–1.15)')).toBeInTheDocument()
  })

  it('OR Table: shows the backend error message on failure', async () => {
    stubBackgroundEndpoints()
    installSession(regressionSession())
    server.use(
      http.post('/api/models/logistic_table', () =>
        HttpResponse.json({ detail: 'Need at least one predictor' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<ModelsPanel />)
    await user.click(document.querySelector('input[name="model"][value="ortable"]') as HTMLElement)
    await user.selectOptions(selectAfterLabel(/^Outcome/), 'DEATH')
    await user.click(checkPredictor('AGE'))
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    await waitFor(() => expect(screen.getByText('Need at least one predictor')).toBeInTheDocument())
  })

  it('sends a Firth fit to the Forest Builder as appendable rows', async () => {
    // A published forest often spans two fits — a continuous exposure and its
    // dichotomised form cannot share a model — so the hand-off appends rather
    // than replacing, and the user does not retype the estimates.
    stubBackgroundEndpoints()
    installSession(regressionSession())
    server.use(
      http.post('/api/models/firth_logistic', () =>
        HttpResponse.json({
          model: 'Firth Penalized Logistic Regression',
          outcome: 'DEATH',
          coefficients: [
            { variable: 'const', p: 0.5, odds_ratio: 0.01, or_ci_low: 0.001, or_ci_high: 0.1 },
            { variable: 'LAR (per 0.1 units)', p: 0.0004, odds_ratio: 1.19, or_ci_low: 1.09, or_ci_high: 1.29 },
            { variable: 'AGE', p: 0.014, odds_ratio: 1.07, or_ci_low: 1.01, or_ci_high: 1.14 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ModelsPanel />)
    await user.click(screen.getByText('Firth Logistic (penalized)'))
    await user.selectOptions(selectAfterLabel(/^Outcome/), 'DEATH')
    await user.click(checkPredictor('AGE'))
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    await waitFor(() => expect(screen.getByRole('button', { name: '→ Forest Builder' })).toBeInTheDocument())
    await user.click(screen.getAllByRole('button', { name: '→ Forest Builder' })[0])

    const s = useStore.getState()
    expect(s.forestHandoffAppend).toBe(true)
    // The intercept is not a row in a forest plot.
    expect(s.forestHandoff).toEqual([
      { label: 'LAR (per 0.1 units)', est: 1.19, ci_low: 1.09, ci_high: 1.29, p: 0.0004, extra: '' },
      { label: 'AGE', est: 1.07, ci_low: 1.01, ci_high: 1.14, p: 0.014, extra: '' },
    ])
    expect(s.forestHandoffLayout).toMatchObject({
      customTitle: 'Firth penalized logistic regression',
      rightHeader: 'OR (95% CI)',
      returnTab: 'models',
    })
    // and it takes the user there
    expect(s.activeTab).toBe('visual')
    expect(s.visualSubTab).toBe('forest')
  })

  it('sends the adjusted estimate from an OR table, flagging any row that only has a crude one', async () => {
    stubBackgroundEndpoints()
    installSession(regressionSession())
    server.use(
      http.post('/api/models/logistic_table', () =>
        HttpResponse.json({
          model: 'OR Table',
          outcome: 'DEATH',
          table: [
            { variable: 'AGE', uni_or: 1.05, uni_ci_low: 1.0, uni_ci_high: 1.1, uni_p: 0.03, multi_or: 1.08, multi_ci_low: 1.01, multi_ci_high: 1.15, multi_p: 0.02 },
            { variable: 'LDL', uni_or: 2.0, uni_ci_low: 1.2, uni_ci_high: 3.4, uni_p: 0.008 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ModelsPanel />)
    await user.click(document.querySelector('input[name="model"][value="ortable"]') as HTMLElement)
    await user.selectOptions(selectAfterLabel(/^Outcome/), 'DEATH')
    await user.click(checkPredictor('AGE'))
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    await waitFor(() => expect(screen.getByRole('button', { name: '→ Forest Builder' })).toBeInTheDocument())
    await user.click(screen.getAllByRole('button', { name: '→ Forest Builder' })[0])

    const rows = useStore.getState().forestHandoff
    expect(rows?.[0]).toMatchObject({ label: 'AGE', est: 1.08, p: 0.02, extra: '' })
    // A term the multivariable model dropped still carries its crude estimate,
    // labelled as such rather than passed off as adjusted.
    expect(rows?.[1]).toMatchObject({ label: 'LDL', est: 2.0, extra: 'unadjusted' })
  })
})
