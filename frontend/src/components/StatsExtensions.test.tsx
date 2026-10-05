import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import HypothesisPanel from './HypothesisPanel'
import CorrelationPanel from './CorrelationPanel'

afterEach(() => clearSession())

it('sends matched numeric columns to the paired sign test', async () => {
  installSession()
  let sent: Record<string, unknown> | undefined
  server.use(http.post('/api/stats/sign_test', async ({ request }) => {
    sent = await request.json() as Record<string, unknown>
    return HttpResponse.json({ test: 'Paired sign result', positive: 2, negative: 1, ties_excluded: 0, p: 1 })
  }))
  const user = userEvent.setup()
  render(<HypothesisPanel />)
  await user.click(screen.getByRole('radio', { name: 'Paired sign test' }))
  await user.click(screen.getByRole('button', { name: /run test/i }))
  await screen.findByRole('heading', { name: 'Paired sign result' })
  expect(sent?.column).toBe('AGE')
  expect(sent?.comparison_column).toBe('LDL')
})

it('routes one-sample Wilcoxon through its endpoint', async () => {
  installSession()
  server.use(http.post('/api/stats/wilcoxon_onesample', () => HttpResponse.json({ test: 'Wilcoxon result', p: 0.2, W: 8 })))
  const user = userEvent.setup()
  render(<HypothesisPanel />)
  await user.click(screen.getByRole('radio', { name: 'One-sample Wilcoxon' }))
  await user.click(screen.getByRole('button', { name: /run test/i }))
  await screen.findByRole('heading', { name: 'Wilcoxon result' })
})

it('sends Bonferroni choice and renders the ANOVA decomposition', async () => {
  installSession()
  let sent: Record<string, unknown> | undefined
  server.use(http.post('/api/stats/anova', async ({ request }) => {
    sent = await request.json() as Record<string, unknown>
    return HttpResponse.json({ test: 'ANOVA result', p: 0.02, anova_table: [
      { source: 'Between groups', ss: 10, df: 2, ms: 5 },
      { source: 'Within groups', ss: 12, df: 6, ms: 2 },
      { source: 'Total', ss: 22, df: 8, ms: null },
    ] })
  }))
  const user = userEvent.setup()
  render(<HypothesisPanel />)
  await user.click(screen.getByRole('radio', { name: 'One-way ANOVA' }))
  const option = screen.getByRole('option', { name: /Bonferroni \(pairwise/ })
  await user.selectOptions(option.closest('select')!, 'bonferroni')
  await user.click(screen.getByRole('button', { name: /run test/i }))
  await screen.findByText('ANOVA variance table')
  expect(screen.getByRole('cell', { name: 'Between groups' })).toBeInTheDocument()
  expect(sent?.posthoc).toBe('bonferroni')
})

it('weighted kappa handles unavailable uncertainty without a render crash', async () => {
  installSession()
  let sent: Record<string, unknown> | undefined
  server.use(http.post('/api/stats/cohens_kappa', async ({ request }) => {
    sent = await request.json() as Record<string, unknown>
    return HttpResponse.json({ kappa: 0.8, weights: 'linear', ci_low: null, ci_high: null, se: null,
      n: 10, interpretation: 'Substantial', labels: ['A', 'B'], confusion_matrix: [[4, 1], [1, 4]],
      uncertainty_note: 'SE, CI and p for weighted kappa are not estimated.' })
  }))
  const user = userEvent.setup()
  render(<CorrelationPanel />)
  await user.click(screen.getByRole('button', { name: "Cohen's κ" }))
  const option = screen.getByRole('option', { name: /Linear/ })
  await user.selectOptions(option.closest('select')!, 'linear')
  await user.type(screen.getByLabelText('Ordered kappa levels'), 'A,B')
  await user.click(screen.getByRole('button', { name: 'Compute' }))
  await screen.findByText('0.800')
  expect(screen.getByText(/SE, CI and p for weighted kappa/)).toBeInTheDocument()
  expect(sent?.weights).toBe('linear')
  expect(sent?.level_order).toEqual(['A', 'B'])
})

it('ICC reports unavailable interval for perfect raters without a render crash', async () => {
  installSession()
  server.use(http.post('/api/stats/icc', () => HttpResponse.json({ icc: 1, ci_low: null, ci_high: null,
    f_stat: null, f_p: 0, n: 4, k: 2, interpretation: 'Excellent',
    interval_note: 'Zero residual variance; interval unavailable.',
    bland_altman: { means: [1, 2, 3, 4], diffs: [0, 0, 0, 0], mean_diff: 0, loa_upper: 0, loa_lower: 0 },
  })))
  const user = userEvent.setup()
  render(<CorrelationPanel />)
  await user.click(screen.getByRole('button', { name: 'ICC' }))
  await user.click(screen.getByRole('button', { name: 'Compute' }))
  await screen.findByText('1.000')
  expect(screen.getByText('No finite estimate')).toBeInTheDocument()
  expect(screen.getByText('Not estimated')).toBeInTheDocument()
})
