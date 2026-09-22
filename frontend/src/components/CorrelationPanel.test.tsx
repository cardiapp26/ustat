import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import CorrelationPanel from './CorrelationPanel'

afterEach(() => clearSession())

const numericSession = () =>
  makeSession({
    columns: [
      { name: 'AGE', dtype: 'float64', kind: 'numeric' },
      { name: 'LDL', dtype: 'float64', kind: 'numeric' },
      { name: 'BP', dtype: 'float64', kind: 'numeric' },
      { name: 'GROUP', dtype: 'object', kind: 'categorical' },
    ],
    preview: [
      { AGE: 55, LDL: 120, BP: 130, GROUP: 'A' },
      { AGE: 62, LDL: 140, BP: 140, GROUP: 'B' },
      { AGE: 48, LDL: 110, BP: 120, GROUP: 'A' },
    ],
  })

const pairResponse = {
  r: 0.812,
  p: 0.02,
  n: 3,
  ci_low: 0.1,
  ci_high: 0.98,
  method: 'pearson',
  label: 'r',
  normality_test: 'Shapiro-Wilk',
  normality: {
    AGE: { p: 0.5, statistic: 0.9, normal: true, skewness: 0.1, test: 'Shapiro-Wilk', bypass: null },
    LDL: { p: 0.6, statistic: 0.95, normal: true, skewness: 0.2, test: 'Shapiro-Wilk', bypass: null },
  },
  scatter: { x: [55, 62, 48], y: [120, 140, 110] },
  regression_line: { x: [48, 62], y: [110, 140] },
  ci_band: { x: [48, 62], y_upper: [115, 145], y_lower: [105, 135] },
  result_text: 'A significant positive correlation was found.',
}

const matrixResponse = {
  variables: ['AGE', 'LDL'],
  matrix: { AGE: { AGE: 1, LDL: 0.8 }, LDL: { AGE: 0.8, LDL: 1 } },
  p_matrix: { AGE: { AGE: null, LDL: 0.01 }, LDL: { AGE: 0.01, LDL: null } },
  multicollinearity_warnings: [],
}

const iccResponse = {
  icc: 0.87,
  ci_low: 0.6,
  ci_high: 0.95,
  f_stat: 15.2,
  f_p: 0.001,
  n: 3,
  interpretation: 'Good',
  bland_altman: { means: [125, 130, 115], diffs: [2, -1, 3], mean_diff: 1.3, loa_upper: 5, loa_lower: -3 },
}

function checkVar(name: string) {
  const candidates = screen.getAllByText(name)
  for (const el of candidates) {
    const label = el.closest('label')
    if (!label) continue
    const checkbox = within(label).queryByRole('checkbox')
    if (checkbox) return checkbox as HTMLInputElement
  }
  throw new Error(`No checkbox found for ${name}`)
}

describe('CorrelationPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<CorrelationPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('Pairwise tab: computes correlation for selected variables and renders result', async () => {
    installSession(numericSession())
    server.use(
      http.post('/api/stats/correlation_pair', () =>
        HttpResponse.json({
          r: 0.812,
          p: 0.02,
          n: 3,
          ci_low: 0.1,
          ci_high: 0.98,
          method: 'pearson',
          label: 'r',
          normality_test: 'Shapiro-Wilk',
          normality: {
            AGE: { p: 0.5, statistic: 0.9, normal: true, skewness: 0.1, test: 'Shapiro-Wilk', bypass: null },
            LDL: { p: 0.6, statistic: 0.95, normal: true, skewness: 0.2, test: 'Shapiro-Wilk', bypass: null },
          },
          scatter: { x: [55, 62, 48], y: [120, 140, 110] },
          regression_line: { x: [48, 62], y: [110, 140] },
          ci_band: { x: [48, 62], y_upper: [115, 145], y_lower: [105, 135] },
          result_text: 'A significant positive correlation was found.',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CorrelationPanel />)

    // Default selection is first 4 numeric cols (AGE, LDL, BP here) — clear then pick 2.
    await user.click(screen.getByRole('button', { name: 'None' }))
    await user.click(checkVar('AGE'))
    await user.click(checkVar('LDL'))

    const runBtn = screen.getByRole('button', { name: /Compute Pair/ })
    expect(runBtn).toBeEnabled()
    await user.click(runBtn)

    await waitFor(() => expect(screen.getByText('0.8120')).toBeInTheDocument())
    expect(screen.getByText('A significant positive correlation was found.')).toBeInTheDocument()
  })

  it('Pairwise tab: shows an inline error when fewer than 2 variables are selected', async () => {
    installSession(numericSession())
    const user = userEvent.setup()
    render(<CorrelationPanel />)

    await user.click(screen.getByRole('button', { name: 'None' }))
    await user.click(checkVar('AGE'))

    // Compute button is disabled below 2 selections — assert the disabled state
    // instead of trying to click through it.
    expect(screen.getByRole('button', { name: /Compute Pair/ })).toBeDisabled()
  })

  it('Matrix tab: computes a correlation matrix and renders multicollinearity warnings', async () => {
    installSession(numericSession())
    server.use(
      http.post('/api/stats/correlation_matrix', () =>
        HttpResponse.json({
          variables: ['AGE', 'LDL', 'BP'],
          matrix: {
            AGE: { AGE: 1, LDL: 0.8, BP: 0.3 },
            LDL: { AGE: 0.8, LDL: 1, BP: 0.2 },
            BP: { AGE: 0.3, LDL: 0.2, BP: 1 },
          },
          p_matrix: {
            AGE: { AGE: null, LDL: 0.01, BP: 0.4 },
            LDL: { AGE: 0.01, LDL: null, BP: 0.5 },
            BP: { AGE: 0.4, LDL: 0.5, BP: null },
          },
          multicollinearity_warnings: [{ var1: 'AGE', var2: 'LDL', r: 0.8 }],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CorrelationPanel />)
    await user.click(screen.getByRole('button', { name: 'Matrix' }))

    await user.click(screen.getByRole('button', { name: 'Compute Matrix' }))

    await waitFor(() =>
      expect(screen.getByText('⚠ High Collinearity Detected (|r| ≥ 0.70)')).toBeInTheDocument(),
    )
    expect(screen.getByText('AGE ↔ LDL')).toBeInTheDocument()
    expect(screen.getByText('r = 0.800')).toBeInTheDocument()
  })

  it('ICC tab: computes ICC and renders the Bland-Altman summary', async () => {
    installSession(numericSession())
    server.use(
      http.post('/api/stats/icc', () =>
        HttpResponse.json({
          icc: 0.87,
          ci_low: 0.6,
          ci_high: 0.95,
          f_stat: 15.2,
          f_p: 0.001,
          n: 3,
          interpretation: 'Good',
          bland_altman: { means: [125, 130, 115], diffs: [2, -1, 3], mean_diff: 1.3, loa_upper: 5, loa_lower: -3 },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CorrelationPanel />)
    await user.click(screen.getByRole('button', { name: 'ICC' }))
    await user.click(screen.getByRole('button', { name: 'Compute' }))

    await waitFor(() => expect(screen.getByText('0.870')).toBeInTheDocument())
    expect(screen.getByText('Good', { selector: 'span' })).toBeInTheDocument()
  })

  it("Cohen's kappa tab: computes kappa and renders the confusion matrix summary", async () => {
    installSession(numericSession())
    server.use(
      http.post('/api/stats/cohens_kappa', () =>
        HttpResponse.json({
          kappa: 0.65,
          ci_low: 0.3,
          ci_high: 0.9,
          se: 0.15,
          n: 3,
          interpretation: 'Substantial',
          confusion_matrix: [[2, 0], [1, 0]],
          labels: ['A', 'B'],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CorrelationPanel />)
    await user.click(screen.getByRole('button', { name: "Cohen's κ" }))
    await user.click(screen.getByRole('button', { name: 'Compute' }))

    await waitFor(() => expect(screen.getByText('0.650')).toBeInTheDocument())
    expect(screen.getByText('Substantial', { selector: 'span' })).toBeInTheDocument()
  })

  it('closes every export of the pairwise correlations once the data changes under them', async () => {
    // The Copy button wrote the paragraph straight to the clipboard, and the
    // table and the scatter plot stayed exportable after an edit.
    installSession(numericSession())
    server.use(http.post('/api/stats/correlation_pair', () => HttpResponse.json(pairResponse)))

    const user = userEvent.setup()
    render(<CorrelationPanel />)
    await user.click(screen.getByRole('button', { name: 'None' }))
    await user.click(checkVar('AGE'))
    await user.click(checkVar('LDL'))
    await user.click(screen.getByRole('button', { name: /Compute Pair/ }))
    await screen.findByText('A significant positive correlation was found.')

    const exports = () => [
      screen.getByRole('button', { name: 'Copy' }),
      screen.getByRole('button', { name: 'CSV' }),
      ...screen.getAllByRole('button', { name: '↓' }),
    ]
    for (const b of exports()) expect(b).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    for (const b of exports()) expect(b).toBeDisabled()
  })

  it('closes the hand-rolled matrix CSV once the settings change under it', async () => {
    installSession(numericSession())
    server.use(http.post('/api/stats/correlation_matrix', () => HttpResponse.json(matrixResponse)))

    const user = userEvent.setup()
    render(<CorrelationPanel />)
    await user.click(screen.getByRole('button', { name: 'Matrix' }))
    await user.click(screen.getByRole('button', { name: 'Compute Matrix' }))
    const csv = await screen.findByRole('button', { name: '↓ Export CSV' })
    expect(csv).toBeEnabled()

    await user.click(screen.getByRole('radio', { name: 'spearman' }))

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByText(/the analysis settings changed/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '↓ Export CSV' })).toBeDisabled()
  })

  it('closes the ICC CSV and its Bland-Altman plot once the data changes, and recomputes', async () => {
    installSession(numericSession())
    let calls = 0
    server.use(http.post('/api/stats/icc', () => {
      calls += 1
      return HttpResponse.json(iccResponse)
    }))

    const user = userEvent.setup()
    render(<CorrelationPanel />)
    await user.click(screen.getByRole('button', { name: 'ICC' }))
    await user.click(screen.getByRole('button', { name: 'Compute' }))
    await screen.findByText('0.870')

    const exports = () => [
      screen.getByRole('button', { name: '↓ CSV' }),
      ...screen.getAllByRole('button', { name: '↓' }),
    ]
    for (const b of exports()) expect(b).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    for (const b of exports()) expect(b).toBeDisabled()

    await user.click(screen.getByRole('button', { name: 'Recompute' }))
    await waitFor(() => expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument())
    expect(calls).toBe(2)
    for (const b of exports()) expect(b).toBeEnabled()
  })

  it('ICC tab: shows the backend error message on failure', async () => {
    installSession(numericSession())
    server.use(
      http.post('/api/stats/icc', () =>
        HttpResponse.json({ detail: 'Columns must be numeric' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<CorrelationPanel />)
    await user.click(screen.getByRole('button', { name: 'ICC' }))
    await user.click(screen.getByRole('button', { name: 'Compute' }))

    await waitFor(() => expect(screen.getByText('Columns must be numeric')).toBeInTheDocument())
  })
})
