import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import CausalPanel from './CausalPanel'

afterEach(() => clearSession())

const fourColSession = () =>
  makeSession({
    columns: [
      { name: 'Y', dtype: 'float64', kind: 'numeric' },
      { name: 'X', dtype: 'float64', kind: 'numeric' },
      { name: 'Z', dtype: 'float64', kind: 'numeric' },
      { name: 'M', dtype: 'float64', kind: 'numeric' },
    ],
    preview: [
      { Y: 1, X: 2, Z: 3, M: 4 },
      { Y: 2, X: 3, Z: 4, M: 5 },
      { Y: 3, X: 4, Z: 5, M: 6 },
    ],
  })

/** Find the <select> that immediately follows a given field label text. */
function selectAfterLabel(labelText: string): HTMLSelectElement {
  const label = screen.getByText(labelText)
  const wrapper = label.parentElement as HTMLElement
  return within(wrapper).getByRole('combobox') as HTMLSelectElement
}

/** Find a MultiPick checkbox for `colName` within the group titled `groupLabel`. */
function checkboxInGroup(groupLabel: string, colName: string): HTMLInputElement {
  const groupTitle = screen.getByText(groupLabel)
  const group = groupTitle.parentElement as HTMLElement
  const span = within(group).getByText(colName)
  const label = span.closest('label') as HTMLLabelElement
  return within(label).getByRole('checkbox') as HTMLInputElement
}

describe('CausalPanel', () => {
  it('renders the IV tab by default even without an active session (no crash)', () => {
    clearSession()
    render(<CausalPanel />)
    expect(screen.getAllByText('Instrumental Variable (2SLS)').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: 'Run 2SLS' })).toBeDisabled()
  })

  it('IV tab: runs 2SLS and renders the effect estimates on success', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/causal/iv_2sls', () =>
        HttpResponse.json({
          result_text: 'The IV estimate is significant and differs from OLS.',
          n: 3,
          iv_estimate: { estimate: 0.842, ci_low: 0.2, ci_high: 1.5, p: 0.01 },
          ols_estimate: { estimate: 0.5, p: 0.02 },
          first_stage: { f_stat: 25.4, weak_instruments: false },
          wu_hausman: { p: 0.03, endogenous: true },
          sargan: null,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CausalPanel />)

    await user.selectOptions(selectAfterLabel('Outcome (continuous)'), 'Y')
    await user.selectOptions(selectAfterLabel('Endogenous exposure'), 'X')
    await user.click(checkboxInGroup('Instrument(s)', 'Z'))

    const runBtn = screen.getByRole('button', { name: 'Run 2SLS' })
    expect(runBtn).toBeEnabled()
    await user.click(runBtn)

    await waitFor(() => expect(screen.getByText('0.8420')).toBeInTheDocument())
    expect(screen.getByText('25.4')).toBeInTheDocument()
    expect(screen.getByText('adequate (≥10)')).toBeInTheDocument()
  })

  it('Mediation tab: runs and renders ACME/ADE decomposition; shows backend error on failure', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/causal/mediation', () =>
        HttpResponse.json({ detail: 'Mediator has no variance after conditioning' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<CausalPanel />)
    await user.click(screen.getByRole('button', { name: 'Mediation (X→M→Y)' }))

    await user.selectOptions(selectAfterLabel('Outcome Y (continuous)'), 'Y')
    await user.selectOptions(selectAfterLabel('Treatment / exposure X'), 'X')
    await user.selectOptions(selectAfterLabel('Mediator M (continuous)'), 'M')

    const runBtn = screen.getByRole('button', { name: 'Run mediation' })
    expect(runBtn).toBeEnabled()
    await user.click(runBtn)

    await waitFor(() =>
      expect(screen.getByText('Mediator has no variance after conditioning')).toBeInTheDocument(),
    )
  })

  it('DAG Backdoor tab: analyses a DAG (no session required) and renders adjustment sets', async () => {
    clearSession()
    server.use(
      http.post('/api/causal/dag_adjustment', () =>
        HttpResponse.json({
          result_text: 'Adjust for Z to close the backdoor path.',
          adjustment_set: ['Z'],
          do_not_adjust: ['C'],
          roles: { Z: 'confounder', M: 'mediator', C: 'collider' },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CausalPanel />)
    await user.click(screen.getByRole('button', { name: 'DAG Backdoor' }))

    await user.click(screen.getByRole('button', { name: 'Analyse DAG' }))

    await waitFor(() => expect(screen.getByText('Adjust for (minimal set)')).toBeInTheDocument())
    const nodeRoles = screen.getByText('Node roles').parentElement as HTMLElement
    expect(within(nodeRoles).getByText(/confounder/)).toBeInTheDocument()
    expect(within(nodeRoles).getByText(/collider/)).toBeInTheDocument()
  })

  it('DAG Backdoor tab: shows an error message when the DAG endpoint fails', async () => {
    clearSession()
    server.use(
      http.post('/api/causal/dag_adjustment', () =>
        HttpResponse.json({ detail: 'Graph contains a cycle' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<CausalPanel />)
    await user.click(screen.getByRole('button', { name: 'DAG Backdoor' }))
    await user.click(screen.getByRole('button', { name: 'Analyse DAG' }))

    await waitFor(() => expect(screen.getByText('Graph contains a cycle')).toBeInTheDocument())
  })

  describe('Unmeasured Confounding tab', () => {
    /** Find the number <input> that immediately follows a given field label text. */
    function numberInputAfterLabel(labelText: string): HTMLInputElement {
      const label = screen.getByText(labelText)
      const wrapper = label.parentElement as HTMLElement
      return within(wrapper).getByRole('spinbutton') as HTMLInputElement
    }

    /** Open the tab and fill in the required observed estimate. */
    async function openTab(user: ReturnType<typeof userEvent.setup>, estimate = '1.85') {
      render(<CausalPanel />)
      await user.click(screen.getByRole('button', { name: 'Unmeasured Confounding' }))
      await user.type(numberInputAfterLabel('Observed estimate (> 0)'), estimate)
    }

    const baseResponse = {
      test: 'Causal Sensitivity Analysis (E-value + QBA + Partial Identification)',
      e_value: {
        measure: 'rr',
        e_value_point_estimate: 3.111,
        e_value_ci: 1.777,
        interpretation: 'An unmeasured confounder would need a risk ratio of at least 3.11.',
        baseline_risk_used: null,
      },
      e_value_smd: { available: false, reason: 'No SMD supplied.' },
      quantitative_bias_analysis: {
        observed_estimate: 1.85,
        assumed_confounder_risk_ratio: 2.0,
        prevalence_exposed: 0.5,
        prevalence_unexposed: 0.5,
        bias_factor: 1.234,
        bias_corrected_estimate: 1.499,
        bias_direction: 'away from the null',
        interpretation: 'Corrected estimate is 1.499 under the assumed confounder.',
      },
      multi_confounder_sensitivity: {
        available: false,
        reason: 'No unmeasured_confounders array supplied.',
      },
      manski_bounds: {
        available: true,
        assumptions: 'Manski no-assumptions bounds',
        n: 120,
        ey1_bounds: [0.21, 0.71],
        ey0_bounds: [0.13, 0.63],
        ate_bounds: [-0.42, 0.58],
        identified_sign: 'not identified',
        interpretation: 'The ATE interval crosses 0.',
      },
      rosenbaum_bounds: {
        applicable: true,
        b: 20,
        c: 8,
        discordant_pairs: 28,
        p_unbiased: 0.012,
        critical_gamma: 1.42,
        alpha: 0.05,
        gamma_max: 3.0,
        n_pairs_used: 60,
        n_pairs_skipped: 2,
      },
      negative_control_analysis: {
        available: true,
        model: 'logistic',
        n: 118,
        negative_control_outcome: 'DM',
        treatment_effect: 1.031,
        coefficient: 0.0305,
        se: 0.2,
        p: 0.88,
        flag_residual_bias: false,
        interpretation: 'No clear negative-control signal detected.',
      },
      warnings: [],
      assumptions: [
        {
          name: 'No unmeasured confounding',
          met: false,
          detail: 'Sensitivity methods quantify how violations could change inference.',
        },
      ],
      result_text: 'Causal sensitivity suite: E-value 3.111 for observed RR=1.85.',
      export_rows: [['Metric', 'Value']],
      r_code: 'library(EValue)',
    }

    it('runs the suite from a bare effect estimate and renders E-value, QBA and data-driven blocks', async () => {
      installSession()
      let received: Record<string, unknown> = {}
      server.use(
        http.post('/api/models/causal_sensitivity', async ({ request }) => {
          received = (await request.json()) as Record<string, unknown>
          return HttpResponse.json(baseResponse)
        }),
      )

      const user = userEvent.setup()
      await openTab(user)
      await user.selectOptions(selectAfterLabel('Effect measure'), 'or')
      await user.type(numberInputAfterLabel('95% CI lower'), '1.2')
      await user.type(numberInputAfterLabel('95% CI upper'), '2.9')

      const runBtn = screen.getByRole('button', { name: 'Run sensitivity suite' })
      expect(runBtn).toBeEnabled()
      await user.click(runBtn)

      await waitFor(() => expect(screen.getByText('3.11')).toBeInTheDocument())
      expect(received).toMatchObject({
        observed_estimate: 1.85,
        measure: 'or',
        rare_outcome: false,
        ci_low: 1.2,
        ci_high: 2.9,
      })
      // No treatment/outcome columns picked → session fields must not be sent.
      expect(received.session_id).toBeUndefined()

      expect(screen.getByText('1.78')).toBeInTheDocument() // E-value CI limit
      expect(screen.getByText('1.234')).toBeInTheDocument() // QBA bias factor
      expect(screen.getByText('1.499')).toBeInTheDocument() // QBA corrected estimate
      expect(
        screen.getByText('An unmeasured confounder would need a risk ratio of at least 3.11.'),
      ).toBeInTheDocument()
      expect(screen.getByText('-0.42 to 0.58')).toBeInTheDocument() // Manski ATE bounds
      expect(screen.getByText('1.42')).toBeInTheDocument() // Rosenbaum critical gamma
      expect(screen.getByText('No clear negative-control signal detected.')).toBeInTheDocument()
    })

    it('sends the session columns and renders the reason for blocks the backend could not compute', async () => {
      installSession()
      let received: Record<string, unknown> = {}
      server.use(
        http.post('/api/models/causal_sensitivity', async ({ request }) => {
          received = (await request.json()) as Record<string, unknown>
          return HttpResponse.json({
            ...baseResponse,
            manski_bounds: { available: false, reason: 'Outcome must be binary 0/1.' },
            rosenbaum_bounds: { applicable: false, reason: 'No clean 1:1 matched pairs available.' },
            negative_control_analysis: { available: false, reason: 'Need at least 20 complete rows.' },
            warnings: ['Low E-value (<2); result is sensitive to weak unmeasured confounding.'],
          })
        }),
      )

      const user = userEvent.setup()
      await openTab(user)
      await user.click(screen.getByRole('button', { name: /Advanced: SMD & data-driven bounds/ }))
      await user.selectOptions(selectAfterLabel('Treatment (binary 0/1)'), 'DM')
      await user.selectOptions(selectAfterLabel('Outcome (binary 0/1)'), 'GROUP')
      await user.click(screen.getByRole('button', { name: 'Run sensitivity suite' }))

      await waitFor(() =>
        expect(screen.getByText(/Outcome must be binary 0\/1\./)).toBeInTheDocument(),
      )
      expect(received).toMatchObject({
        session_id: 'test-session',
        treatment_col: 'DM',
        outcome_col: 'GROUP',
      })
      expect(screen.getByText(/No clean 1:1 matched pairs available\./)).toBeInTheDocument()
      expect(screen.getByText(/Need at least 20 complete rows\./)).toBeInTheDocument()
      expect(screen.getByText(/No unmeasured_confounders array supplied\./)).toBeInTheDocument()
      expect(screen.getByText(/No SMD supplied\./)).toBeInTheDocument()
      // The placeholder blocks are labelled, not blank.
      expect(screen.getByText('Manski partial-identification bounds')).toBeInTheDocument()
      expect(
        screen.getByText(/Low E-value \(<2\); result is sensitive to weak unmeasured confounding\./),
      ).toBeInTheDocument()
    })

    it('shows the backend error detail when the endpoint fails', async () => {
      clearSession()
      server.use(
        http.post('/api/models/causal_sensitivity', () =>
          HttpResponse.json({ detail: 'ci_low must be < ci_high' }, { status: 400 }),
        ),
      )

      const user = userEvent.setup()
      await openTab(user)
      await user.click(screen.getByRole('button', { name: 'Run sensitivity suite' }))

      await waitFor(() => expect(screen.getByText('ci_low must be < ci_high')).toBeInTheDocument())
    })
  })
})
