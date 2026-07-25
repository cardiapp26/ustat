import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import SurvivalAdvancedPanel from './SurvivalAdvancedPanel'

afterEach(() => clearSession())

/** Session with survival-shaped columns: numeric duration, binary event,
 *  categorical group, plus a numeric predictor for Cox/Fine-Gray/RMST. */
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

/** Find the <select> that immediately follows a given field label text. */
function selectAfterLabel(labelText: string): HTMLSelectElement {
  const label = screen.getByText(labelText)
  const wrapper = label.parentElement as HTMLElement
  return within(wrapper).getByRole('combobox') as HTMLSelectElement
}

/** Switch the left-nav method radio to the given method title. */
async function selectMethod(user: ReturnType<typeof userEvent.setup>, title: string) {
  const label = screen.getByText(title)
  const radio = within(label.closest('label') as HTMLElement).getByRole('radio')
  await user.click(radio)
}

describe('SurvivalAdvancedPanel', () => {
  it('shows an upload prompt without an active session (no crash)', () => {
    clearSession()
    render(<SurvivalAdvancedPanel />)
    expect(screen.getByText('Upload data first.')).toBeInTheDocument()
  })

  it('KM tab (default): runs Kaplan-Meier by group and renders the log-rank result', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/models/survival/km', () =>
        HttpResponse.json({
          groups: [
            { group: 'A', n: 2, events: 2, median_survival: 125, curve: [{ time: 0, survival: 1 }, { time: 150, survival: 0 }] },
            { group: 'B', n: 2, events: 0, median_survival: null, curve: [{ time: 0, survival: 1 }, { time: 300, survival: 1 }] },
          ],
          logrank: { p: 0.032, chi2: 4.6 },
          n_total: 4,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)

    // KM is the default active method.
    expect(screen.getByText('Kaplan-Meier Survival')).toBeInTheDocument()

    await user.selectOptions(selectAfterLabel('Duration (time)'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    await user.selectOptions(selectAfterLabel('Group (optional)'), 'GROUP')

    const runBtn = screen.getByRole('button', { name: 'Run Kaplan-Meier' })
    await user.click(runBtn)

    await waitFor(() => expect(screen.getByText('Log-rank test (overall)')).toBeInTheDocument())
    const footer = screen.getByText('Log-rank test (overall)').parentElement as HTMLElement
    expect(within(footer).getByText('p')).toBeInTheDocument()
    expect(within(footer).getByText(/0\.032/)).toBeInTheDocument()
    expect(within(footer).getByText(/Significant difference/)).toBeInTheDocument()
    // p is italicized per the manuscript-style formatting pass.
    expect(within(footer).getByText('p').tagName).toBe('I')
  })

  it('KM tab: shows the backend error message on failure', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/models/survival/km', () =>
        HttpResponse.json({ detail: 'Event column must be binary 0/1' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)

    await user.selectOptions(selectAfterLabel('Duration (time)'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    await user.click(screen.getByRole('button', { name: 'Run Kaplan-Meier' }))

    await waitFor(() => expect(screen.getByText('Event column must be binary 0/1')).toBeInTheDocument())
  })

  it('Cox tab: runs Cox PH regression and renders the coefficient table', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/models/survival/cox', () =>
        HttpResponse.json({
          n: 4,
          concordance: 0.71,
          log_likelihood: -12.34,
          coefficients: [
            { variable: 'AGE', log_hr: 0.05, se: 0.02, hr: 1.051, hr_ci_low: 1.01, hr_ci_high: 1.09, p: 0.012 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'Cox PH')

    await user.selectOptions(selectAfterLabel('Duration (time)'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    // Tick the AGE predictor checkbox: it's the checkbox row whose sibling
    // span shows the column kind badge ("numeric"), unique to the predictor list.
    const ageLabel = screen.getAllByText('AGE').find((el) => el.nextElementSibling?.textContent === 'numeric')
      ?.closest('label') as HTMLElement
    await user.click(within(ageLabel).getByRole('checkbox'))

    await user.click(screen.getByRole('button', { name: 'Run Cox Regression' }))

    await waitFor(() => expect(screen.getByText('1.0510')).toBeInTheDocument())
    expect(screen.getByText('0.71', { exact: false }) ?? screen.getByText(/0\.7100/)).toBeTruthy()
    const rows = screen.getAllByRole('row')
    expect(rows.length).toBeGreaterThanOrEqual(2) // header rows + data row
  })

  it('Fine-Gray tab: runs competing-risks regression and renders sHR coefficients', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/survival_advanced/fine_gray', () =>
        HttpResponse.json({
          regression_result: {
            model: 'Fine-Gray subdistribution hazard',
            n: 4,
            n_events_of_interest: 2,
            n_competing: 0,
            n_censored: 2,
            concordance: 0.65,
            coefficients: [
              { variable: 'AGE', shr: 1.2, shr_low: 0.9, shr_high: 1.6, p: 0.2 },
            ],
            method_note: 'Aalen-Johansen based subdistribution hazard model.',
          },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'Fine-Gray')

    await user.selectOptions(selectAfterLabel('Duration'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0=censor, 1,2..=events)'), 'EVENT')

    await user.click(screen.getByRole('button', { name: 'Run Fine-Gray' }))

    await waitFor(() => expect(screen.getByText('sHR Regression (Fine-Gray)')).toBeInTheDocument())
    const regressionCard = screen.getByText('sHR Regression (Fine-Gray)')
      .closest('div.border-indigo-200') as HTMLElement
    expect(within(regressionCard).getByText('AGE')).toBeInTheDocument()
    expect(within(regressionCard).getByText('1.20')).toBeInTheDocument()
    expect(screen.getByText('Aalen-Johansen based subdistribution hazard model.')).toBeInTheDocument()
  })

  it('RMST tab: runs restricted mean survival time and renders group + contrast tables', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/survival_advanced/rmst', () =>
        HttpResponse.json({
          rmst_by_group: {
            A: { n: 2, rmst: '120.5', ci_low: '90.0', ci_high: '150.0' },
            B: { n: 2, rmst: '180.2', ci_low: '150.0', ci_high: '210.0' },
          },
          contrasts: [
            { group_a: 'A', group_b: 'B', delta_rmst: '-59.7', p: 0.04 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'RMST')

    await user.selectOptions(selectAfterLabel('Duration'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    const tauInput = screen.getByPlaceholderText('e.g. 1825')
    await user.type(tauInput, '365')
    await user.selectOptions(selectAfterLabel('Group (optional)'), 'GROUP')

    await user.click(screen.getByRole('button', { name: 'Run RMST' }))

    await waitFor(() => expect(screen.getByText('120.5')).toBeInTheDocument())
    expect(screen.getByText('180.2')).toBeInTheDocument()
    expect(screen.getByText('-59.7')).toBeInTheDocument()
  })

  it('E-value tab: computes E-value (session installed for panel access) and renders point/CI cards', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/survival_advanced/evalue', () =>
        HttpResponse.json({
          evalue_point: 2.62,
          evalue_ci: 1.5,
          interpretation: 'An unmeasured confounder would need to be associated with both exposure and outcome by a risk ratio of at least 2.62-fold, above and beyond the measured confounders, to fully explain away the observed association.',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'E-value')

    const estInput = screen.getByPlaceholderText('e.g. 2.5')
    const loInput = screen.getByPlaceholderText('e.g. 1.2')
    const hiInput = screen.getByPlaceholderText('e.g. 5.1')
    await user.type(estInput, '2.5')
    await user.type(loInput, '1.2')
    await user.type(hiInput, '5.1')

    await user.click(screen.getByRole('button', { name: 'Calculate E-value' }))

    await waitFor(() => expect(screen.getByText('2.62')).toBeInTheDocument())
    expect(screen.getByText('1.5')).toBeInTheDocument()
    expect(screen.getByText(/would need to be associated/)).toBeInTheDocument()
  })

  it('Shared Frailty tab: runs the clustered Cox model and renders θ + the coefficient table', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/survival_advanced/frailty', () =>
        HttpResponse.json({
          n_subjects: 4,
          n_clusters: 2,
          n_events: 2,
          theta: 0.4271,
          theta_se: null,
          frailty_distribution: 'gamma',
          estimation_method: 'penalized',
          coefficients: [
            { variable: 'AGE', estimate: 0.048, hr: 1.0492, se: 0.021, p: 0.0223 },
          ],
          cluster_frailties: { A: 1.12, B: 0.89 },
          nested_frailties: {},
          variance_components: {},
          correlated_frailty: null,
          diagnostics: { available: false, reason: 'Diagnostics not requested.' },
          parametric_baseline: {},
          frailty_variance_test: {
            lrt_statistic: 3.1,
            ordinary_chi_square_p: 0.0783,
            chi_bar_square_p: 0.0391,
            mixture: '0.5*chi-square(df=0) + 0.5*chi-square(df=1)',
            interpretation: 'Boundary test for theta=0.',
          },
          concordance: 0.6821,
          log_likelihood: -10.5,
          assumptions: [],
          warnings: ['Only 2 clusters — frailty variance estimate (θ) may be unstable.'],
          result_text: 'Shared gamma frailty Cox model on 4 observations across 2 clusters.',
          model: 'shared_gamma_frailty_cox',
          method_note: 'Gamma shared frailty estimated via penalized Cox.',
          plot: {
            data: [{ x: [0.89, 1.12], type: 'histogram', name: 'Posterior frailties' }],
            layout: { title: 'Estimated Cluster Frailties (Gamma model)' },
          },
          diagnostic_plots: {},
          test: 'Shared Gamma Frailty Cox Model',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'Shared Frailty')

    await user.selectOptions(selectAfterLabel('Duration (time)'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    await user.selectOptions(selectAfterLabel('Cluster (centre / family)'), 'GROUP')
    // Tick the AGE predictor checkbox in the predictor MultiSelect.
    const ageLabel = screen.getAllByText('AGE').find((el) => el.nextElementSibling?.textContent === 'N')
      ?.closest('label') as HTMLElement
    await user.click(within(ageLabel).getByRole('checkbox'))

    await user.click(screen.getByRole('button', { name: 'Run Shared Frailty' }))

    await waitFor(() => expect(screen.getByText('0.4271')).toBeInTheDocument())
    expect(screen.getByText('1.0492')).toBeInTheDocument()
    expect(screen.getByText('0.6821')).toBeInTheDocument()
    expect(screen.getByText(/Shared gamma frailty Cox model on 4 observations/)).toBeInTheDocument()
    expect(screen.getByText(/Only 2 clusters/)).toBeInTheDocument()
  })

  it('Shared Frailty tab: shows the backend error message on failure', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/survival_advanced/frailty', () =>
        HttpResponse.json({ detail: 'Cluster column must have at least 2 levels' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'Shared Frailty')

    await user.selectOptions(selectAfterLabel('Duration (time)'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    await user.selectOptions(selectAfterLabel('Cluster (centre / family)'), 'GROUP')
    const ageLabel = screen.getAllByText('AGE').find((el) => el.nextElementSibling?.textContent === 'N')
      ?.closest('label') as HTMLElement
    await user.click(within(ageLabel).getByRole('checkbox'))

    await user.click(screen.getByRole('button', { name: 'Run Shared Frailty' }))

    await waitFor(() =>
      expect(screen.getByText('Cluster column must have at least 2 levels')).toBeInTheDocument(),
    )
  })

  it('E-value tab: shows validation error when fields are missing (no request sent)', async () => {
    installSession(survivalSession())
    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'E-value')

    await user.click(screen.getByRole('button', { name: 'Calculate E-value' }))

    await waitFor(() =>
      expect(screen.getByText('Enter estimate and confidence interval')).toBeInTheDocument(),
    )
  })

  it('Multi-state tab: fits transition models and renders each transition', async () => {
    installSession(survivalSession())
    let sent: Record<string, unknown> | null = null
    server.use(
      http.post('/api/survival_advanced/multistate', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          transitions_estimated: ['0->1', '1->2'],
          results: {
            '0->1': {
              n_transitions: 85, n_events: 85, concordance: 0.6021, aic: 712.4,
              coefficients: [{ variable: 'AGE', coef: 0.02199, hr: 1.0222, se: 0.01593, p: 0.16737 }],
            },
            '1->2': {
              n_transitions: 40, n_events: 24, concordance: 0.5533, aic: 210.1,
              coefficients: [{ variable: 'AGE', coef: -0.011, hr: 0.9891, se: 0.02, p: 0.5821 }],
            },
          },
          markov_assumption_tests: {
            '0->1': { status: 'Skipped', reason: 'No variation in entry time.' },
            '1->2': { status: 'Tested', p_value: 0.21135, markov_assumption_violated: false },
          },
          model_type: 'cox',
          test: 'Multi-state model',
          result_text: 'Multi-state analysis on 265 transition records.',
        })
      }),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'Multi-state')

    // The long-format requirement must be stated, not assumed.
    expect(screen.getByText(/Needs long-format transition data/i)).toBeInTheDocument()

    await user.selectOptions(selectAfterLabel('Subject id'), 'GROUP')
    await user.selectOptions(selectAfterLabel('From state'), 'EVENT')
    await user.selectOptions(selectAfterLabel('To state'), 'GROUP')
    await user.selectOptions(selectAfterLabel('Entry time'), 'TIME')
    await user.selectOptions(selectAfterLabel('Exit time'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (1 = transition occurred)'), 'EVENT')
    const ageLabel = screen.getAllByText('AGE').find((el) => el.nextElementSibling?.textContent === 'N')
      ?.closest('label') as HTMLElement
    await user.click(within(ageLabel).getByRole('checkbox'))

    await user.click(screen.getByRole('button', { name: 'Fit transition models' }))

    await waitFor(() => expect(screen.getByText(/Transitions estimated: 0->1, 1->2/)).toBeInTheDocument())
    expect(screen.getByText('1.0222')).toBeInTheDocument()
    expect(screen.getByText('0.9891')).toBeInTheDocument()
    // Markov results: one skipped with a reason, one tested and not violated.
    expect(screen.getByText(/No variation in entry time/)).toBeInTheDocument()
    expect(screen.getByText(/not violated/)).toBeInTheDocument()
    expect(sent!.predictors).toEqual(['AGE'])
  })

  it('Multi-state tab: landmark prediction plots state occupancy against the time axis', async () => {
    installSession(survivalSession())
    let sent: Record<string, unknown> | null = null
    server.use(
      http.post('/api/survival_advanced/dynamic_prediction', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          landmark_time: 2,
          current_state: 0,
          n_at_risk: 104,
          times: [2, 4.5, 7],
          state_probabilities: {
            state_0: [1, 0.62, 0.15867],
            state_1: [0, 0.28, 0.45951],
            state_2: [0, 0.1, 0.38182],
          },
          elos: { '0': 2.2995, '1': 1.7567, '2': 0.9438 },
          bootstrap: {},
          microsimulation: {},
          model_type: 'cox',
          test: 'Dynamic prediction',
          result_text: 'Dynamic prediction from landmark 2.',
        })
      }),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'Multi-state')

    await user.selectOptions(selectAfterLabel('Subject id'), 'GROUP')
    await user.selectOptions(selectAfterLabel('From state'), 'EVENT')
    await user.selectOptions(selectAfterLabel('To state'), 'GROUP')
    await user.selectOptions(selectAfterLabel('Entry time'), 'TIME')
    await user.selectOptions(selectAfterLabel('Exit time'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (1 = transition occurred)'), 'EVENT')
    const ageLabel = screen.getAllByText('AGE').find((el) => el.nextElementSibling?.textContent === 'N')
      ?.closest('label') as HTMLElement
    await user.click(within(ageLabel).getByRole('checkbox'))

    await user.clear(screen.getByPlaceholderText('e.g. 2'))
    await user.type(screen.getByPlaceholderText('e.g. 2'), '2')
    await user.click(screen.getByRole('button', { name: 'Predict from landmark' }))

    await waitFor(() => expect(screen.getByText(/104 at risk in state 0/)).toBeInTheDocument())
    expect(screen.getByText('2.2995')).toBeInTheDocument()
    expect(sent!.landmark_time).toBe(2)
  })

  it('Multi-state tab: shows the backend error message on failure', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/survival_advanced/multistate', () =>
        HttpResponse.json({ detail: 'from_state_col not found in dataframe' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await selectMethod(user, 'Multi-state')

    await user.selectOptions(selectAfterLabel('Subject id'), 'GROUP')
    await user.selectOptions(selectAfterLabel('From state'), 'EVENT')
    await user.selectOptions(selectAfterLabel('To state'), 'GROUP')
    await user.selectOptions(selectAfterLabel('Entry time'), 'TIME')
    await user.selectOptions(selectAfterLabel('Exit time'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (1 = transition occurred)'), 'EVENT')
    const ageLabel = screen.getAllByText('AGE').find((el) => el.nextElementSibling?.textContent === 'N')
      ?.closest('label') as HTMLElement
    await user.click(within(ageLabel).getByRole('checkbox'))

    await user.click(screen.getByRole('button', { name: 'Fit transition models' }))

    await waitFor(() =>
      expect(screen.getByText('from_state_col not found in dataframe')).toBeInTheDocument(),
    )
  })
})
