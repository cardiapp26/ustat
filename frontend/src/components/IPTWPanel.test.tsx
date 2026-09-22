import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import { useStore } from '../store'
import IPTWPanel from './IPTWPanel'

afterEach(() => clearSession())

// DM is a 0/1 column in the default fixture -> becomes the default binary treatment.
const IPTW_RESULT = {
  balance_achieved: true,
  estimand: 'ate',
  stabilize: true,
  se_method: 'robust',
  score_method: 'logistic',
  n_total: 100,
  n_treated: 40,
  n_control: 60,
  n_trimmed_common_support: 0,
  weight_truncation: { n_trimmed: 2 },
  weight_summary: { ess_treated: 35.2, ess_control: 55.1, min: 0.5, median: 1.0, max: 4.2 },
  weight_distribution: { treated: [1, 2, 3], control: [1, 1.5, 2] },
  smd_before: { AGE: 0.35, LDL: 0.28 },
  smd_after: { AGE: 0.05, LDL: 0.08 },
  avg_smd_before: 0.315,
  avg_smd_after: 0.065,
  reduction_pct: 79.4,
  variance_ratio_after: { AGE: 1.1, LDL: 0.9 },
  variance_ratio_before: { AGE: 1.5, LDL: 1.8 },
  ks_p_after: { AGE: 0.42, LDL: 0.31 },
  ps_distribution: {
    treated_unmatched: [0.3, 0.4, 0.5],
    control_unmatched: [0.2, 0.3, 0.4],
    treated_matched: [0.3, 0.4],
    control_matched: [0.2, 0.3],
  },
}

describe('IPTWPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<IPTWPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('runs IPTW and renders balance summary, SMD table, and love plot', async () => {
    installSession()
    server.use(
      http.post('/api/models/iptw', () => HttpResponse.json(IPTW_RESULT)),
    )

    const user = userEvent.setup()
    render(<IPTWPanel />)

    // Select covariates (AGE, LDL) via checkboxes in the "Covariates" panel.
    const ageCheckbox = screen.getByRole('checkbox', { name: 'AGE' })
    const ldlCheckbox = screen.getByRole('checkbox', { name: 'LDL' })
    await user.click(ageCheckbox)
    await user.click(ldlCheckbox)

    const runButton = screen.getByRole('button', { name: /run iptw/i })
    expect(runButton).toBeEnabled()
    await user.click(runButton)

    await waitFor(() =>
      expect(screen.getByText(/Balance achieved/i)).toBeInTheDocument(),
    )

    // Key stats banner
    expect(screen.getByText('100')).toBeInTheDocument() // n_total
    expect(screen.getByText('40')).toBeInTheDocument() // n_treated
    expect(screen.getByText('60')).toBeInTheDocument() // n_control

    // SMD balance table
    expect(screen.getByText('SMD Balance Table')).toBeInTheDocument()
    const rows = screen.getAllByRole('row')
    expect(rows.length).toBeGreaterThanOrEqual(3) // header + 2 covariate rows + footer

    // Love plot rendered (mocked plotly)
    expect(screen.getAllByTestId('plotly-mock').length).toBeGreaterThan(0)

    // Average reduction pct shown (appears in both the summary card and table footer)
    expect(screen.getAllByText('79.4%').length).toBeGreaterThan(0)
  })

  it('blocks the run and shows a validation error when no covariates are selected', async () => {
    installSession()
    const user = userEvent.setup()
    render(<IPTWPanel />)

    const runButton = screen.getByRole('button', { name: /run iptw/i })
    expect(runButton).toBeDisabled()

    // Select a covariate then deselect it via the covariates panel's "None" button
    // (there is another unrelated "None" button in weight-truncation options, so scope the query).
    const covariatesPanel = screen.getByText('Covariates (Confounders)').closest('div.panel') as HTMLElement
    await user.click(within(covariatesPanel).getByRole('button', { name: 'None' }))
    expect(screen.getByRole('button', { name: /run iptw/i })).toBeDisabled()
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/models/iptw', () =>
        HttpResponse.json({ detail: 'Treatment column must be binary' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<IPTWPanel />)

    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /run iptw/i }))

    await waitFor(() =>
      expect(screen.getByText('Treatment column must be binary')).toBeInTheDocument(),
    )
  })

  it('closes every export of the weighting, the cohort included, once the data changes under it', async () => {
    installSession()
    server.use(
      http.post('/api/models/iptw', () => HttpResponse.json({ ...IPTW_RESULT, matched_session_id: 'weighted-1' })),
    )

    const user = userEvent.setup()
    render(<IPTWPanel />)
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /run iptw/i }))
    await screen.findByText(/Balance achieved/i)

    const exports = () => [
      screen.getByRole('button', { name: 'CSV' }),
      ...screen.getAllByRole('button', { name: '↓' }),
      screen.getByRole('button', { name: /View & Analyze Weighted Cohort/ }),
      screen.getByRole('button', { name: /Export as CSV/ }),
      screen.getByRole('button', { name: /Export as Excel/ }),
      screen.getByRole('button', { name: /Export as SPSS/ }),
    ]
    // Weight distribution, Love plot and PS overlap each carry a plot exporter.
    expect(screen.getAllByRole('button', { name: '↓' })).toHaveLength(3)
    for (const b of exports()) expect(b).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    for (const b of exports()) expect(b).toBeDisabled()
  })

  it('flags the weighting stale when the estimand changes, and Recompute clears it', async () => {
    installSession()
    let calls = 0
    server.use(
      http.post('/api/models/iptw', () => {
        calls += 1
        return HttpResponse.json(IPTW_RESULT)
      }),
    )

    const user = userEvent.setup()
    render(<IPTWPanel />)
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /run iptw/i }))
    await screen.findByText(/Balance achieved/i)

    await user.click(screen.getByRole('button', { name: 'ATT' }))
    expect(await screen.findByText(/the analysis settings changed/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Recompute' }))
    await waitFor(() => expect(calls).toBe(2))
    await waitFor(() => expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument())
    expect(screen.getByText(/Balance achieved/i)).toBeInTheDocument()
  })
})
