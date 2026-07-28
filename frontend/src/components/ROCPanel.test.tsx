import { render, screen, waitFor, within } from '@testing-library/react'
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
      { name: 'SCORE2', dtype: 'float64', kind: 'numeric' },
      { name: 'SCORE3', dtype: 'float64', kind: 'numeric' },
      { name: 'OUTCOME', dtype: 'int64', kind: 'numeric' },
    ],
    preview: [
      { SCORE1: 1.2, SCORE2: 3.4, SCORE3: 0.5, OUTCOME: 0 },
      { SCORE1: 2.5, SCORE2: 1.1, SCORE3: 1.5, OUTCOME: 1 },
      { SCORE1: 3.1, SCORE2: 4.4, SCORE3: 2.5, OUTCOME: 1 },
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

/** Find a variable checkbox by its exact column label text. When the
 *  Predictors (Multi-curve) and Combined Model checklists are both
 *  visible they render the same column names twice, so `within` scopes
 *  the lookup to whichever list is passed in. */
function checkboxFor(labelText: string, scope: HTMLElement = document.body): HTMLInputElement {
  const span = within(scope).getByText(labelText)
  const label = span.closest('label') as HTMLLabelElement
  return within(label).getByRole('checkbox') as HTMLInputElement
}

describe('ROCPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<ROCPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('Single curve tab: runs ROC and renders AUC + metrics on success', async () => {
    installSession(rocSession())
    mockNoMissing()
    server.use(
      http.post('/api/stats/roc', () =>
        HttpResponse.json({
          auc: 0.82,
          auc_p: 0.012,
          auc_se: 0.09,
          auc_z: 2.5,
          ci_lower: 0.65,
          ci_upper: 0.95,
          curve,
          optimal: {
            cutoff: 2.1, sensitivity: 0.8, specificity: 0.75, ppv: 0.7, npv: 0.82,
            accuracy: 0.77, lr_pos: 3.2, lr_neg: 0.27, youden_j: 0.55,
            tp: 8, tn: 15, fp: 5, fn: 2,
          },
          result_text: 'SCORE1 significantly predicted OUTCOME (AUC = 0.82).',
          n: 3, n_positive: 2, n_negative: 1,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ROCPanel />)

    // Score defaults to first numeric column (SCORE1); outcome defaults to
    // the detected binary column (OUTCOME). Just click Run.
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run ROC' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: 'Run ROC' }))

    await waitFor(() => {
      expect(screen.getByText('0.82')).toBeInTheDocument()
    })
    expect(screen.getByText('Good')).toBeInTheDocument()
    expect(screen.getByText(/SCORE1 significantly predicted OUTCOME/)).toBeInTheDocument()
    expect(screen.getByText('Cutoff')).toBeInTheDocument()
    expect(screen.getByText('80.0%')).toBeInTheDocument() // sensitivity
  })

  it('Single curve tab: shows an error message on failure', async () => {
    installSession(rocSession())
    mockNoMissing()
    server.use(
      http.post('/api/stats/roc', () =>
        HttpResponse.json({ detail: 'Score column has no variance' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<ROCPanel />)

    await screen.findByRole('button', { name: 'Run ROC' })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run ROC' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: 'Run ROC' }))

    await waitFor(() => expect(screen.getByText('Score column has no variance')).toBeInTheDocument())
  })

  it('Single curve tab: visibly reports when auto direction flips', async () => {
    installSession(rocSession())
    mockNoMissing()
    server.use(
      http.post('/api/stats/roc', () =>
        HttpResponse.json({
          auc: 0.79,
          ci_lower: 0.66,
          ci_upper: 0.9,
          curve,
          direction_requested: 'auto',
          direction_used: 'lower',
          direction_flipped: true,
          n: 3,
          n_positive: 2,
          n_negative: 1,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ROCPanel />)

    await waitFor(() => expect(screen.getByRole('button', { name: 'Run ROC' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: 'Run ROC' }))

    expect(await screen.findByRole('status')).toHaveTextContent(
      'Auto direction flipped to lower = event: low values of SCORE1 predict the event',
    )
  })

  it('Compare (DeLong) tab: runs comparison and renders the result', async () => {
    installSession(rocSession())
    mockNoMissing()
    server.use(
      http.post('/api/stats/roc_compare', () =>
        HttpResponse.json({
          score_1: 'SCORE1', score_2: 'SCORE2',
          auc_1: 0.82, auc_2: 0.65,
          ci_1_low: 0.65, ci_1_high: 0.95,
          ci_2_low: 0.5, ci_2_high: 0.8,
          ci_diff_low: 0.01, ci_diff_high: 0.33,
          difference: 0.17, z: 2.1, p: 0.036, n: 3,
          significant: true,
          interpretation: 'SCORE1 significantly outperformed SCORE2 (DeLong p = 0.036).',
          curve_1: curve, curve_2: curve,
          direction_1_requested: 'lower',
          direction_1_flipped: true,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ROCPanel />)

    // Expand the AUC Comparison (DeLong) section
    await user.click(screen.getByRole('button', { name: /AUC Comparison \(DeLong\)/ }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run DeLong Test' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: 'Run DeLong Test' }))

    await waitFor(() =>
      expect(screen.getByText('✓ Significant difference (p < 0.05)')).toBeInTheDocument(),
    )
    expect(screen.getByText(/SCORE1 significantly outperformed SCORE2/)).toBeInTheDocument()
    expect(screen.getByText('0.820')).toBeInTheDocument()
    expect(screen.queryByText(/auto-flipped direction/i)).not.toBeInTheDocument()
  })

  it('Multi-compare tab: runs multiple ROCs and the pairwise DeLong matrix', async () => {
    installSession(rocSession())
    mockNoMissing()
    server.use(
      http.post('/api/stats/roc', async ({ request }) => {
        const body = await request.json() as { score_column: string }
        const aucByCol: Record<string, number> = { SCORE1: 0.82, SCORE2: 0.65, SCORE3: 0.7 };
        return HttpResponse.json({
          auc: aucByCol[body.score_column] ?? 0.5,
          curve,
          ci_lower: 0.5, ci_upper: 0.9,
        })
      }),
      http.post('/api/stats/roc_multi_compare', () =>
        HttpResponse.json({
          pairs: [
            { a: 'SCORE1', b: 'SCORE2', delta_auc: 0.17, ci_low: 0.01, ci_high: 0.33, p_raw: 0.03, p_adj: 0.06, significant: false },
          ],
          scores: ['SCORE1', 'SCORE2'],
          n: 3, n_pairs: 1, p_adjust: 'holm',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ROCPanel />)

    await user.click(screen.getByRole('button', { name: 'Multi-curve' }))
    await user.click(checkboxFor('SCORE1'))
    await user.click(checkboxFor('SCORE2'))

    await waitFor(() => expect(screen.getByRole('button', { name: /Run 2 ROCs/ })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /Run 2 ROCs/ }))

    await waitFor(() => expect(screen.getByText('AUC Summary')).toBeInTheDocument())
    expect(screen.getByText('0.82')).toBeInTheDocument()
    expect(screen.getByText('0.65')).toBeInTheDocument()

    // Pairwise DeLong matrix rendered
    await waitFor(() => expect(screen.getByText('Pairwise DeLong')).toBeInTheDocument())
  })

  it('Combined-model tab: fits combined model and renders the AUC', async () => {
    installSession(rocSession())
    mockNoMissing()
    server.use(
      http.post('/api/stats/roc_combined', () =>
        HttpResponse.json({ auc: 0.88, curve }),
      ),
    )

    const user = userEvent.setup()
    render(<ROCPanel />)

    await user.click(screen.getByRole('button', { name: 'Multi-curve' }))
    // Expand Combined Model section
    await user.click(screen.getByRole('button', { name: /Combined Model/ }))
    const variablesHeading = screen.getByText('Variables')
    const combinedPanel = variablesHeading.closest('div')!.parentElement as HTMLElement
    await user.click(checkboxFor('SCORE1', combinedPanel))
    await user.click(checkboxFor('SCORE2', combinedPanel))

    await waitFor(() => expect(screen.getByRole('button', { name: 'Run Combined Model' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: 'Run Combined Model' }))

    await waitFor(() => expect(screen.getByText('0.88')).toBeInTheDocument())
    expect(screen.getAllByText('Combined Model').length).toBeGreaterThan(0)
  })
})
