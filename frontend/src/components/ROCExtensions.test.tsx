import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import ROCPanel from './ROCPanel'

afterEach(() => clearSession())

const rocSession = () =>
  makeSession({
    columns: [
      { name: 'SCORE1', dtype: 'float64', kind: 'numeric' },
      { name: 'OUTCOME', dtype: 'int64', kind: 'numeric' },
    ],
    preview: [
      { SCORE1: 1.2, OUTCOME: 0 },
      { SCORE1: 2.5, OUTCOME: 1 },
      { SCORE1: 3.1, OUTCOME: 1 },
    ],
  })

function mockNoMissing() {
  server.use(
    http.get('/api/stats/test-session/missing', () =>
      HttpResponse.json({ total_rows: 3, rows_affected: 0, pct_affected: 0, per_column: {} }),
    ),
  )
}

const curve = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.2, tpr: 0.6 },
  { fpr: 1, tpr: 1 },
]

const withCis = {
  cutoff: 2.1, sensitivity: 0.8, specificity: 0.75, ppv: 0.7, npv: 0.82,
  accuracy: 0.77, lr_pos: 3.2, lr_neg: 0.27, youden_j: 0.55,
  tp: 8, tn: 15, fp: 5, fn: 2,
  sensitivity_ci: [0.55, 0.93],
  specificity_ci: [0.53, 0.89],
  ppv_ci: [0.45, 0.87],
  npv_ci: [0.6, 0.93],
  accuracy_ci: [0.6, 0.88],
  lr_pos_ci: [1.5, 6.8],
  lr_neg_ci: [0.08, 0.9],
}

const baseResponse = {
  auc: 0.82, auc_p: 0.012, auc_se: 0.09, auc_z: 2.5, ci_lower: 0.65, ci_upper: 0.95, curve,
  optimal: withCis,
  result_text: 'SCORE1 predicted OUTCOME (AUC = 0.82).',
  n: 3, n_positive: 2, n_negative: 1,
}

async function run(user: ReturnType<typeof userEvent.setup>) {
  await waitFor(() => expect(screen.getByRole('button', { name: 'Run ROC' })).toBeEnabled())
  await user.click(screen.getByRole('button', { name: 'Run ROC' }))
  await screen.findByRole('button', { name: 'CSV' })
}

describe('ROCPanel confidence intervals and post-test probability', () => {
  it('shows 95% CIs next to the optimal-cutoff metrics', async () => {
    installSession(rocSession())
    mockNoMissing()
    server.use(http.post('/api/stats/roc', () => HttpResponse.json(baseResponse)))
    const user = userEvent.setup()
    render(<ROCPanel />)
    await run(user)

    expect(screen.getByText('80.0% [55.0%, 93.0%]')).toBeInTheDocument()
    expect(screen.getByText('75.0% [53.0%, 89.0%]')).toBeInTheDocument()
    expect(screen.getByText('70.0% [45.0%, 87.0%]')).toBeInTheDocument()
    expect(screen.getByText('82.0% [60.0%, 93.0%]')).toBeInTheDocument()
    expect(screen.getByText('77.0% [60.0%, 88.0%]')).toBeInTheDocument()
    expect(screen.getByText('3.20 [1.50, 6.80]')).toBeInTheDocument()
    expect(screen.getByText('0.27 [0.08, 0.90]')).toBeInTheDocument()
  })

  it('falls back to the bare value when a CI is null', async () => {
    installSession(rocSession())
    mockNoMissing()
    server.use(
      http.post('/api/stats/roc', () =>
        HttpResponse.json({
          ...baseResponse,
          optimal: { ...withCis, sensitivity_ci: null, lr_pos_ci: null },
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ROCPanel />)
    await run(user)

    expect(screen.getByText('80.0%')).toBeInTheDocument()
    expect(screen.getByText('3.20')).toBeInTheDocument()
    expect(screen.getByText('75.0% [53.0%, 89.0%]')).toBeInTheDocument()
  })

  it('shows CIs for the manual cutoff when it is selected', async () => {
    installSession(rocSession())
    mockNoMissing()
    server.use(
      http.post('/api/stats/roc', () =>
        HttpResponse.json({
          ...baseResponse,
          manual: {
            ...withCis, cutoff: 3, sensitivity: 0.5, sensitivity_ci: [0.2, 0.8],
          },
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ROCPanel />)
    await user.click(screen.getByRole('checkbox', { name: 'Manual cutoff' }))
    await user.type(screen.getByPlaceholderText('e.g. 42.5'), '3')
    await run(user)

    expect(screen.getByText('At manual cutoff')).toBeInTheDocument()
    expect(screen.getByText('50.0% [20.0%, 80.0%]')).toBeInTheDocument()
  })

  it('sends prevalence and renders the Fagan post-test card', async () => {
    installSession(rocSession())
    mockNoMissing()
    let sent: Record<string, unknown> | undefined
    server.use(
      http.post('/api/stats/roc', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          ...baseResponse,
          post_test: { pretest: 0.2, post_positive: 0.4444, post_negative: 0.0675 },
        })
      }),
    )
    const user = userEvent.setup()
    render(<ROCPanel />)
    await user.type(screen.getByLabelText(/Disease prevalence/), '0.2')
    await run(user)

    expect(sent?.prevalence).toBe(0.2)
    expect(screen.getByText('Post-test probability (Fagan)')).toBeInTheDocument()
    expect(screen.getByTestId('roc-post-test')).toHaveTextContent(
      'Pre-test 20.0%, after a positive test 44.4%, after a negative test 6.8%',
    )
  })

  it('omits prevalence and the card when none is given', async () => {
    installSession(rocSession())
    mockNoMissing()
    let sent: Record<string, unknown> | undefined
    server.use(
      http.post('/api/stats/roc', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({ ...baseResponse, post_test: null })
      }),
    )
    const user = userEvent.setup()
    render(<ROCPanel />)
    await run(user)

    expect(sent).toBeDefined()
    expect('prevalence' in (sent ?? {})).toBe(false)
    expect(screen.queryByText('Post-test probability (Fagan)')).not.toBeInTheDocument()
  })

  it.each(['0', '1', '1.5', '-0.2', '20'])(
    'blocks the run and shows an inline error for prevalence %s',
    async (bad) => {
      installSession(rocSession())
      mockNoMissing()
      const user = userEvent.setup()
      render(<ROCPanel />)
      await user.type(screen.getByLabelText(/Disease prevalence/), bad)

      expect(screen.getByRole('alert')).toHaveTextContent(/between 0 and 1/)
      expect(screen.getByRole('button', { name: 'Run ROC' })).toBeDisabled()
    },
  )

  it('re-enables the run once the prevalence is valid again', async () => {
    installSession(rocSession())
    mockNoMissing()
    const user = userEvent.setup()
    render(<ROCPanel />)
    const box = screen.getByLabelText(/Disease prevalence/)
    await user.type(box, '2')
    expect(screen.getByRole('button', { name: 'Run ROC' })).toBeDisabled()
    await user.clear(box)
    await user.type(box, '0.3')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run ROC' })).toBeEnabled())
  })
})
