import { act, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import SurvivalAdvancedPanel from './SurvivalAdvancedPanel'

afterEach(() => clearSession())

const survivalSession = () =>
  makeSession({
    columns: [
      { name: 'TIME', dtype: 'float64', kind: 'numeric' },
      { name: 'EVENT', dtype: 'int64', kind: 'numeric' },
      { name: 'GROUP', dtype: 'object', kind: 'categorical' },
      { name: 'AGE', dtype: 'float64', kind: 'numeric' },
    ],
    preview: [
      { TIME: 100, EVENT: 1, GROUP: 'A', AGE: 55 },
      { TIME: 200, EVENT: 0, GROUP: 'B', AGE: 62 },
      { TIME: 150, EVENT: 1, GROUP: 'A', AGE: 48 },
      { TIME: 300, EVENT: 0, GROUP: 'B', AGE: 70 },
    ],
  })

function selectAfterLabel(labelText: string): HTMLSelectElement {
  const label = screen.getByText(labelText)
  return within(label.parentElement as HTMLElement).getByRole('combobox') as HTMLSelectElement
}

async function selectMethod(user: ReturnType<typeof userEvent.setup>, title: string) {
  const label = screen.getByText(title)
  await user.click(within(label.closest('label') as HTMLElement).getByRole('radio'))
}

/** The Plotly modebar config of every mocked chart on screen. */
const modebarRemovals = () =>
  screen.getAllByTestId('plotly-mock').map(
    (el) => (JSON.parse(el.dataset.config ?? '{}').modeBarButtonsToRemove ?? []) as string[],
  )

const fineGrayResponse = {
  regression_result: {
    model: 'Fine-Gray subdistribution hazard',
    n: 4, n_events_of_interest: 2, n_competing: 0, n_censored: 2, concordance: 0.65,
    coefficients: [{ variable: 'AGE', shr: 1.2, shr_low: 0.9, shr_high: 1.6, p: 0.2 }],
    method_note: 'Aalen-Johansen based subdistribution hazard model.',
  },
  result_text: 'Fine-Gray competing-risks analysis of 4 subjects.',
  export_rows: [['Variable', 'sHR'], ['AGE', 1.2]],
  test: 'Fine-Gray',
  plot: { data: [{ x: [0, 100], y: [0, 0.4], type: 'scatter' }], layout: { title: { text: 'CIF' } } },
}

async function runFineGray(user: ReturnType<typeof userEvent.setup>) {
  await selectMethod(user, 'Fine-Gray')
  await user.selectOptions(selectAfterLabel('Duration'), 'TIME')
  await user.selectOptions(selectAfterLabel('Event (0=censor, 1,2..=events)'), 'EVENT')
  await user.click(screen.getByRole('button', { name: 'Run Fine-Gray' }))
  await screen.findByText('sHR Regression (Fine-Gray)')
}

describe('SurvivalAdvancedPanel: out-of-date results', () => {
  it('closes every export of a Fine-Gray analysis once the data changes under it', async () => {
    installSession(survivalSession())
    server.use(http.post('/api/survival_advanced/fine_gray', () => HttpResponse.json(fineGrayResponse)))

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await runFineGray(user)

    const exports = () => [
      screen.getByRole('button', { name: 'CSV' }),
      screen.getByRole('button', { name: 'XLSX' }),
      screen.getByRole('button', { name: '↓' }),
    ]
    for (const b of exports()) expect(b).toBeEnabled()
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    for (const b of exports()) expect(b).toBeDisabled()
    for (const removed of modebarRemovals()) expect(removed).toContain('toImage')
  })

  it('keeps a Fine-Gray result, current, across a remount', async () => {
    // The panel's clear-on-input-change effects used to run on mount too,
    // which would have thrown the cached result away on every tab switch.
    installSession(survivalSession())
    server.use(http.post('/api/survival_advanced/fine_gray', () => HttpResponse.json(fineGrayResponse)))

    const user = userEvent.setup()
    const first = render(<SurvivalAdvancedPanel />)
    await runFineGray(user)
    first.unmount()

    render(<SurvivalAdvancedPanel />)
    expect(screen.getByText('sHR Regression (Fine-Gray)')).toBeInTheDocument()
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeEnabled()
  })

  it('closes the Kaplan-Meier figure export once the data changes under it', async () => {
    installSession(survivalSession())
    // at_risk / censors present, so the lazy extras fetch has nothing to add.
    server.use(
      http.post('/api/models/survival/km', () =>
        HttpResponse.json({
          groups: [
            { group: 'A', n: 2, events: 2, median_survival: 125, curve: [{ time: 0, survival: 1 }, { time: 150, survival: 0 }], at_risk: [2, 0], censors: [] },
            { group: 'B', n: 2, events: 0, median_survival: null, curve: [{ time: 0, survival: 1 }, { time: 300, survival: 1 }], at_risk: [2, 1], censors: [] },
          ],
          risk_times: [0, 150],
          logrank: { p: 0.032, chi2: 4.6 },
          n_total: 4,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await user.selectOptions(selectAfterLabel('Duration (time)'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    await user.selectOptions(selectAfterLabel('Group (optional)'), 'GROUP')
    await user.click(screen.getByRole('button', { name: 'Run Kaplan-Meier' }))
    await screen.findByText('Log-rank test (overall)')

    expect(screen.getByRole('button', { name: '↓' })).toBeEnabled()
    for (const removed of modebarRemovals()) expect(removed).not.toContain('toImage')

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '↓' })).toBeDisabled()
    for (const removed of modebarRemovals()) expect(removed).toContain('toImage')
  })

  it('refuses to send out-of-date time-horizon HRs to the Forest Builder', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/models/survival/cox_horizons', () =>
        HttpResponse.json({
          predictor: 'AGE',
          covariates: [],
          forest_rows: [
            { label: '1 year', est: 1.2, ci_low: 0.9, ci_high: 1.6, p: 0.2, extra: '(3 events)' },
            { label: 'Full follow-up', est: 1.1, ci_low: 0.95, ci_high: 1.3, p: 0.18, extra: '(5 events)' },
          ],
          interpretation: 'The HR is similar across windows.',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'Time-horizon HR')
    await user.selectOptions(selectAfterLabel('Duration / Time'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    await user.selectOptions(selectAfterLabel('Predictor (HR tracked)'), 'AGE')
    await user.click(screen.getByRole('button', { name: 'Run horizons' }))
    await screen.findByText('The HR is similar across windows.')

    const send = () => screen.getByRole('button', { name: '→ Send to Forest Builder' })
    expect(send()).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(send()).toBeDisabled()
    expect(send()).toHaveAttribute('title', 'Recompute first: this result predates the data changed')
    await user.click(send())
    expect(useStore.getState().activeTab).not.toBe('visual')
  })
})
