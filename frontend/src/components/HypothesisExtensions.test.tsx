import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import HypothesisPanel from './HypothesisPanel'
import RepeatedMeasuresPanel from './RepeatedMeasuresPanel'

afterEach(() => clearSession())

const hl = {
  estimate: 1.5,
  ci_low: 0.4,
  ci_high: 2.6,
  confidence_level: 0.95,
  achieved_confidence_level: 0.953,
  method: 'exact (order statistics of the pairwise differences)',
  note: 'Hodges-Lehmann shift with a distribution-free 95% CI; reproduces R wilcox.test(..., conf.int = TRUE).',
}

const riskMeasures = {
  event: 'Yes',
  exposed: 'Drug',
  reference: 'Placebo',
  ci_level: 0.95,
  risk_exposed: 0.2,
  risk_reference: 0.4,
  ard: -0.2,
  ard_ci: [-0.38, -0.01],
  ard_ci_method: 'Newcombe (unpooled score)',
  arr: 0.2,
  rr: 0.5,
  rr_ci: [0.27, 0.93],
  rr_ci_method: 'Katz log',
  rrr: 0.5,
  rrr_ci: [0.07, 0.73],
  nnt_nnh: {
    kind: 'NNT', value: 5, low: 3, high: 100, ci_spans_zero: false,
    other_kind: null, other_low: null, ci_text: 'NNT 3 to 100',
  },
}

describe('HypothesisPanel: chi-square on a 2x2 table', () => {
  it('shows the clinical effect measures table with the exposed group, reference group and event', async () => {
    installSession()
    server.use(
      http.post('/api/stats/chisquare', () =>
        HttpResponse.json({
          test: 'Chi-square test of independence',
          chi2: 4.5, df: 1, p: 0.034, significant: true,
          relative_risk: 0.5, relative_risk_ci: [0.27, 0.93],
          risk_event: 'Yes', risk_exposed: 'Drug', risk_reference: 'Placebo',
          risk_measures: riskMeasures,
          interpretation: 'Significant association.',
        }),
      ),
    )
    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Chi-square test of independence' })

    expect(screen.getByText('Clinical effect measures')).toBeInTheDocument()
    const table = screen.getByTestId('risk-measures')
    expect(within(table).getByText('Drug')).toBeInTheDocument()
    expect(within(table).getByText('Placebo')).toBeInTheDocument()
    expect(within(table).getByText('Yes')).toBeInTheDocument()
    expect(within(table).getByText('-0.200')).toBeInTheDocument()
    expect(within(table).getByText('[-0.380, -0.010]')).toBeInTheDocument()
    expect(within(table).getByText('[0.270, 0.930]')).toBeInTheDocument()
    expect(within(table).getByText('NNT (number needed to treat)')).toBeInTheDocument()
    expect(within(table).getByText('NNT 3 to 100')).toBeInTheDocument()
    // The object-valued field never leaks into the generic stat grid.
    expect(screen.queryByText('risk_measures')).not.toBeInTheDocument()
    expect(screen.queryByText('[object Object]')).not.toBeInTheDocument()
  })

  it('renders the two-part NNH interval through infinity when the ARD CI spans zero', async () => {
    installSession()
    server.use(
      http.post('/api/stats/chisquare', () =>
        HttpResponse.json({
          test: 'Chi-square test of independence', chi2: 0.8, df: 1, p: 0.37,
          risk_measures: {
            ...riskMeasures,
            risk_exposed: 0.3, risk_reference: 0.2, ard: 0.1, ard_ci: [-0.05, 0.25],
            rr: 1.5, rr_ci: [0.6, 3.7], rrr: -0.5, rrr_ci: [-2.7, 0.4],
            nnt_nnh: {
              kind: 'NNH', value: 10, low: 4, high: null, ci_spans_zero: true,
              other_kind: 'NNT', other_low: 20, ci_text: 'NNH 4 to infinity to NNT 20',
            },
          },
        }),
      ),
    )
    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))

    const table = await screen.findByTestId('risk-measures')
    expect(within(table).getByText('NNH (number needed to harm)')).toBeInTheDocument()
    expect(within(table).getByText('NNH 4 to infinity to NNT 20')).toBeInTheDocument()
    expect(within(table).getByText(/ARD interval includes zero/)).toBeInTheDocument()
  })

  it('omits the table for results without risk measures (r x c tables)', async () => {
    installSession()
    server.use(
      http.post('/api/stats/chisquare', () =>
        HttpResponse.json({ test: 'Chi-square test of independence', chi2: 4.5, df: 3, p: 0.2 }),
      ),
    )
    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Chi-square test of independence' })
    expect(screen.queryByTestId('risk-measures')).not.toBeInTheDocument()
  })
})

describe('Hodges-Lehmann estimates', () => {
  it('shows the Hodges-Lehmann median difference with its CI in the Mann-Whitney result', async () => {
    installSession()
    server.use(
      http.post('/api/stats/mannwhitney', () =>
        HttpResponse.json({
          test: 'Mann-Whitney U', U: 4, p: 0.04, significant: true,
          interpretation: 'Significant difference.',
          hodges_lehmann: hl,
        }),
      ),
    )
    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('radio', { name: /mann-whitney u/i }))
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Mann-Whitney U' })

    const line = screen.getByTestId('hodges-lehmann')
    expect(within(line).getByText('Hodges-Lehmann median difference [95% CI]')).toBeInTheDocument()
    expect(within(line).getByText('1.500 [0.400, 2.600]')).toBeInTheDocument()
    expect(within(line).getByText(/exact \(order statistics of the pairwise differences\)/)).toBeInTheDocument()
    expect(within(line).getByText(/reproduces R wilcox.test/)).toBeInTheDocument()
    expect(screen.queryByText('[object Object]')).not.toBeInTheDocument()
  })

  it('reports an unavailable estimate with its note for very large samples', async () => {
    installSession()
    server.use(
      http.post('/api/stats/mannwhitney', () =>
        HttpResponse.json({
          test: 'Mann-Whitney U', U: 4, p: 0.04,
          hodges_lehmann: {
            estimate: null, ci_low: null, ci_high: null, confidence_level: 0.95,
            achieved_confidence_level: null, method: 'not computed',
            note: 'Hodges-Lehmann shift not computed: 5,000 x 5,000 = 25,000,000 pairwise differences exceed the 4,000,000-pair memory cap.',
          },
        }),
      ),
    )
    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('radio', { name: /mann-whitney u/i }))
    await user.click(screen.getByRole('button', { name: /run test/i }))

    const line = await screen.findByTestId('hodges-lehmann')
    expect(within(line).getByText('n/a')).toBeInTheDocument()
    expect(within(line).getByText(/exceed the 4,000,000-pair memory cap/)).toBeInTheDocument()
  })

  it('labels the one-sample Wilcoxon estimate a pseudomedian and notes a lower achieved level', async () => {
    installSession()
    server.use(
      http.post('/api/stats/wilcoxon_onesample', () =>
        HttpResponse.json({
          test: 'One-sample Wilcoxon signed-rank test', W: 8, p: 0.2,
          hodges_lehmann: { ...hl, achieved_confidence_level: 0.875, note: null },
        }),
      ),
    )
    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('radio', { name: 'One-sample Wilcoxon' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))

    const line = await screen.findByTestId('hodges-lehmann')
    expect(within(line).getByText('Hodges-Lehmann pseudomedian [95% CI]')).toBeInTheDocument()
    expect(within(line).getByText('1.500 [0.400, 2.600]')).toBeInTheDocument()
    expect(within(line).getByText(/achieved confidence level 87.5%/)).toBeInTheDocument()
  })

  it('shows the Hodges-Lehmann median difference in the paired Wilcoxon result', async () => {
    installSession()
    server.use(
      http.post('/api/repeated/wilcoxon_signed_rank', () =>
        HttpResponse.json({
          test: 'Wilcoxon signed-rank test', W: 3, p: 0.03, significant: true,
          interpretation: 'Significant difference.',
          hodges_lehmann: { ...hl, estimate: -2, ci_low: -3.5, ci_high: -0.5 },
        }),
      ),
    )
    const user = userEvent.setup()
    render(<RepeatedMeasuresPanel />)
    await user.click(screen.getByRole('radio', { name: 'Wilcoxon signed-rank' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Wilcoxon signed-rank test' })

    const line = screen.getByTestId('hodges-lehmann')
    expect(within(line).getByText('Hodges-Lehmann median difference [95% CI]')).toBeInTheDocument()
    expect(within(line).getByText('-2.000 [-3.500, -0.500]')).toBeInTheDocument()
    expect(screen.queryByText('[object Object]')).not.toBeInTheDocument()
  })
})
