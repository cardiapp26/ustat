import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import CategoricalTestsPanel from './CategoricalTestsPanel'

afterEach(() => clearSession())

const riskMeasures = {
  event: '1',
  exposed: 'Treated',
  reference: 'Control',
  ci_level: 0.95,
  risk_exposed: 0.1,
  risk_reference: 0.3,
  ard: -0.2,
  ard_ci: [-0.35, -0.04],
  ard_ci_method: 'Newcombe (unpooled score)',
  arr: 0.2,
  rr: 0.333,
  rr_ci: [0.15, 0.74],
  rr_ci_method: 'Katz log',
  rrr: 0.667,
  rrr_ci: [0.26, 0.85],
  nnt_nnh: {
    kind: 'NNT', value: 5, low: 3, high: 25, ci_spans_zero: false,
    other_kind: null, other_low: null, ci_text: 'NNT 3 to 25',
  },
}

function twoPropResult(rm: Record<string, unknown>) {
  return {
    test: 'Two-sample proportion z-test',
    z: -2.5, p: 0.012, significant: true, diff_prop: -0.2,
    interpretation: 'Significant difference between proportions.',
    risk_measures: rm,
  }
}

async function runTwoProportions(body: Record<string, unknown>) {
  installSession()
  server.use(http.post('/api/categorical/two_proportions', () => HttpResponse.json(body)))
  const user = userEvent.setup()
  render(<CategoricalTestsPanel />)
  await user.click(screen.getByRole('radio', { name: 'Two proportions z-test' }))
  await user.click(screen.getByRole('button', { name: /run test/i }))
  await screen.findByRole('heading', { name: 'Two-sample proportion z-test' })
}

describe('two proportions: clinical effect measures', () => {
  it('shows the risk table with groups, event, ARD, RR, RRR and an NNT', async () => {
    await runTwoProportions(twoPropResult(riskMeasures))

    expect(screen.getByText('Clinical effect measures')).toBeInTheDocument()
    const table = screen.getByTestId('risk-measures')
    expect(within(table).getByText('Treated')).toBeInTheDocument()
    expect(within(table).getByText('Control')).toBeInTheDocument()
    expect(within(table).getAllByText('1').length).toBeGreaterThan(0)
    expect(within(table).getByText('-0.200')).toBeInTheDocument()
    expect(within(table).getByText('[-0.350, -0.040]')).toBeInTheDocument()
    expect(within(table).getByText('0.333')).toBeInTheDocument()
    expect(within(table).getByText('[0.150, 0.740]')).toBeInTheDocument()
    expect(within(table).getByText('0.667')).toBeInTheDocument()
    expect(within(table).getByText('NNT (number needed to treat)')).toBeInTheDocument()
    expect(within(table).getByText('NNT 3 to 25')).toBeInTheDocument()
    expect(within(table).queryByText(/NNH/)).not.toBeInTheDocument()
    expect(within(table).getByText(/lower risk, so the number needed is reported as an NNT/)).toBeInTheDocument()
  })

  it('labels an NNH and renders the two-part interval when the ARD CI spans zero', async () => {
    await runTwoProportions(twoPropResult({
      ...riskMeasures,
      ard: 0.1, ard_ci: [-0.04, 0.24], rr: 1.5, rr_ci: [0.9, 2.5], rrr: -0.5, rrr_ci: [-1.5, 0.1],
      nnt_nnh: {
        kind: 'NNH', value: 10, low: 8, high: null, ci_spans_zero: true,
        other_kind: 'NNT', other_low: 25, ci_text: 'NNH 5 to infinity to NNT 25',
      },
    }))

    const table = screen.getByTestId('risk-measures')
    expect(within(table).getByText('NNH (number needed to harm)')).toBeInTheDocument()
    expect(within(table).getByText('NNH 5 to infinity to NNT 25')).toBeInTheDocument()
    expect(within(table).getByText(/ARD interval includes zero/)).toBeInTheDocument()
    expect(within(table).getByText(/higher risk, so the number needed is reported as an NNH/)).toBeInTheDocument()
  })

  it('shows infinite for equal risks and n/a for undefined estimates', async () => {
    await runTwoProportions(twoPropResult({
      ...riskMeasures,
      risk_exposed: 0.2, risk_reference: 0.2, ard: 0, ard_ci: [-0.1, 0.1],
      rr: null, rr_ci: [null, null], rrr: null, rrr_ci: [null, null],
      nnt_nnh: {
        kind: null, value: null, low: null, high: null, ci_spans_zero: false,
        other_kind: null, other_low: null, ci_text: 'Number needed is infinite (no risk difference)',
      },
    }))

    const table = screen.getByTestId('risk-measures')
    expect(within(table).getByText('NNT / NNH')).toBeInTheDocument()
    expect(within(table).getByText('infinite')).toBeInTheDocument()
    expect(within(table).getAllByText('n/a').length).toBeGreaterThanOrEqual(3)
    expect(within(table).getByText(/number needed is infinite\./)).toBeInTheDocument()
  })
})

describe('Mantel-Haenszel: homogeneity of odds ratios', () => {
  async function runMantelHaenszel(body: Record<string, unknown>) {
    installSession()
    server.use(http.post('/api/categorical/mantel_haenszel', () => HttpResponse.json(body)))
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Mantel-Haenszel' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Cochran-Mantel-Haenszel test' })
  }

  const base = {
    test: 'Cochran-Mantel-Haenszel test', statistic: 6.1, p: 0.013, significant: true,
    interpretation: 'Significant common association.',
  }

  it('shows the Breslow-Day row and an amber caution when the odds ratios are heterogeneous', async () => {
    await runMantelHaenszel({
      ...base,
      homogeneity_test: {
        name: 'Breslow-Day test (Tarone adjusted)', statistic: 9.2, df: 2, p: 0.01,
        adjusted: true, homogeneous: false,
      },
      homogeneity_note: null,
      warnings: ['The odds ratios differ across strata of SITE (Breslow-Day test, Tarone adjusted, p = 0.010).'],
    })

    expect(screen.getByText('Homogeneity of odds ratios (Breslow-Day, Tarone)')).toBeInTheDocument()
    expect(screen.getByText(/chi-square\(2\) = 9\.200/)).toHaveTextContent('chi-square(2) = 9.200, p = 0.010')
    expect(screen.getByText('heterogeneous')).toBeInTheDocument()
    const caution = screen.getByRole('alert')
    expect(caution).toHaveTextContent(/single pooled OR may be misleading/)
    expect(caution).toHaveClass('bg-amber-50')
    // The backend warning still shows with the other warnings.
    expect(screen.getByText(/The odds ratios differ across strata of SITE/)).toBeInTheDocument()
  })

  it('shows homogeneous without a caution', async () => {
    await runMantelHaenszel({
      ...base,
      homogeneity_test: {
        name: 'Breslow-Day test (Tarone adjusted)', statistic: 0.4, df: 2, p: 0.82,
        adjusted: true, homogeneous: true,
      },
      homogeneity_note: null,
    })

    expect(screen.getByText('homogeneous')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.queryByText(/may be misleading/)).not.toBeInTheDocument()
  })

  it('shows the note when the homogeneity test could not be computed', async () => {
    await runMantelHaenszel({
      ...base,
      homogeneity_test: null,
      homogeneity_note: 'The Breslow-Day homogeneity test is undefined for these strata.',
    })

    expect(screen.getByText('not computed')).toBeInTheDocument()
    expect(screen.getByText('The Breslow-Day homogeneity test is undefined for these strata.')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

describe('McNemar: paired difference', () => {
  it('shows the Newcombe paired proportion difference and the discordant OR interval', async () => {
    installSession()
    server.use(
      http.post('/api/categorical/mcnemar', () =>
        HttpResponse.json({
          test: "McNemar's test", statistic: 1.2, p: 0.27, significant: false,
          interpretation: 'No significant change.',
          effect_sizes: [{
            name: 'odds_ratio_discordant', value: 2.5, ci_low: 0.9, ci_high: 7.8, magnitude: 'small',
          }],
          paired_difference: {
            estimate: 0.1, ci_low: -0.02, ci_high: 0.22,
            method: 'Newcombe (1998) method 10', confidence_level: 0.95,
            positive_level: '1', phi: 0.4,
            note: "Difference = proportion '1' in AGE minus proportion '1' in LDL = (b - c) / n.",
          },
        }),
      ),
    )
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'McNemar test' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: "McNemar's test" })

    expect(screen.getByText('Paired proportion difference (Newcombe 95% CI)')).toBeInTheDocument()
    expect(screen.getByText('0.100 [-0.020, 0.220]')).toBeInTheDocument()
    expect(screen.getByText(/Difference = proportion '1' in AGE minus/)).toBeInTheDocument()
    expect(screen.getByText('95% CI: [0.900, 7.800]')).toBeInTheDocument()
  })
})

describe('chi-square goodness of fit (one sample)', () => {
  const frequency = {
    GROUP: {
      n: 16, missing: 0,
      categories: [
        { value: 'B', count: 3, pct: 18.8 },
        { value: 'A', count: 9, pct: 56.3 },
        { value: 'D', count: 1, pct: 6.3 },
        { value: 'C', count: 3, pct: 18.8 },
      ],
    },
  }

  const gofResult = {
    test: 'Chi-square goodness-of-fit test',
    chi2: 10.5, df: 3, p: 0.0147, significant: true, n: 16, k: 4,
    categories: [
      { category: 'A', observed: 9, expected_count: 9, expected_proportion: 0.5625, observed_proportion: 0.5625, pearson_residual: 0, adjusted_residual: 0 },
      { category: 'B', observed: 3, expected_count: 3, expected_proportion: 0.1875, observed_proportion: 0.1875, pearson_residual: 0, adjusted_residual: 0 },
      { category: 'C', observed: 3, expected_count: 3, expected_proportion: 0.1875, observed_proportion: 0.1875, pearson_residual: 0, adjusted_residual: 0 },
      { category: 'D', observed: 1, expected_count: 1.25, expected_proportion: 0.0625, observed_proportion: 0.0625, pearson_residual: -0.22, adjusted_residual: 2.31 },
    ],
    proportions_normalised: true,
    exact_multinomial: { p: 0.0231, n_outcomes: 969, note: null },
    effect_sizes: [{ name: 'cohens_w', value: 0.81, ci_low: null, ci_high: null, magnitude: 'large' }],
    assumptions: [{ name: 'Expected counts >= 5', met: false, detail: '3 of 4 expected counts (75%) are below 5.' }],
    warnings: ['3 of 4 expected counts are below 5 (minimum 1.25), so the chi-square approximation may be unreliable.'],
    interpretation: 'Significant departure from the expected distribution.',
    result_text: 'A chi-square goodness-of-fit test compared the distribution of GROUP.',
    methods_text: 'The distribution of GROUP across 4 categories was compared with the stated expected proportions.',
    r_code: 'chisq.test(x = c(9, 3, 3, 1), p = c(9, 3, 3, 1), rescale.p = TRUE)',
    export_rows: [
      ['Category', 'Observed', 'Expected'],
      ['A', 9, 9],
      ['Statistic', 'Value'],
      ['Chi-square', 10.5],
    ],
  }

  function setup() {
    installSession()
    const bodies: Record<string, unknown>[] = []
    server.use(
      http.get('/api/stats/test-session/frequency', () => HttpResponse.json(frequency)),
      http.post('/api/categorical/chisquare_gof', async ({ request }) => {
        bodies.push((await request.json()) as Record<string, unknown>)
        return HttpResponse.json(gofResult)
      }),
    )
    return bodies
  }

  it('lists the observed categories (numeric/alphabetical order) and sends no proportions for equal shares', async () => {
    const bodies = setup()
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square goodness of fit (one sample)' }))

    const equalButton = await screen.findByRole('button', { name: 'Equal proportions' })
    expect(equalButton).toHaveAttribute('aria-pressed', 'true')
    // Categories load from the frequency endpoint; 4 categories -> 25.0% each.
    await waitFor(() => expect(screen.getAllByText('25.0%')).toHaveLength(4))
    const rows = screen.getAllByRole('row').slice(1).map((r) => r.textContent ?? '')
    expect(rows.map((r) => r[0])).toEqual(['A', 'B', 'C', 'D'])

    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Chi-square goodness-of-fit test' })
    expect(bodies).toHaveLength(1)
    expect(bodies[0]).toMatchObject({ session_id: 'test-session', column: 'GROUP', alpha: 0.05 })
    expect(bodies[0]).not.toHaveProperty('expected_proportions')
  })

  it('applies a 9:3:3:1 ratio and sends the weights as expected_proportions', async () => {
    const bodies = setup()
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square goodness of fit (one sample)' }))
    await user.click(await screen.findByRole('button', { name: 'Custom weights' }))

    // Defaults are 1 each (equal), so the test can run straight away.
    expect(screen.getByLabelText('Expected weight for A')).toHaveValue('1')
    await user.type(screen.getByLabelText('Ratio in the order listed'), '9:3:3:1')
    await user.click(screen.getByRole('button', { name: 'Apply ratio' }))
    expect(screen.getByLabelText('Expected weight for A')).toHaveValue('9')
    expect(screen.getByLabelText('Expected weight for B')).toHaveValue('3')
    expect(screen.getByLabelText('Expected weight for C')).toHaveValue('3')
    expect(screen.getByLabelText('Expected weight for D')).toHaveValue('1')

    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Chi-square goodness-of-fit test' })
    expect(bodies).toHaveLength(1)
    expect(bodies[0].expected_proportions).toEqual({ A: 9, B: 3, C: 3, D: 1 })
    expect(bodies[0].column).toBe('GROUP')
  })

  it('accepts a typed fraction and percentage per category, and blocks invalid weights', async () => {
    const bodies = setup()
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square goodness of fit (one sample)' }))
    await user.click(await screen.findByRole('button', { name: 'Custom weights' }))

    const a = screen.getByLabelText('Expected weight for A')
    await user.clear(a)
    await user.type(a, '1/2')
    const b = screen.getByLabelText('Expected weight for B')
    await user.clear(b)
    await user.type(b, '25%')

    // An invalid entry disables Run and flags the field.
    const c = screen.getByLabelText('Expected weight for C')
    await user.clear(c)
    await user.type(c, 'abc')
    expect(c).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByRole('button', { name: /run test/i })).toBeDisabled()
    await user.clear(c)
    await user.type(c, '0.125')
    expect(screen.getByRole('button', { name: /run test/i })).toBeEnabled()

    await user.click(screen.getByRole('button', { name: /run test/i }))
    await waitFor(() => expect(bodies).toHaveLength(1))
    expect(bodies[0].expected_proportions).toEqual({ A: 0.5, B: 25, C: 0.125, D: 1 })
  })

  it('rejects a ratio with the wrong number of parts', async () => {
    setup()
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square goodness of fit (one sample)' }))
    await user.click(await screen.findByRole('button', { name: 'Custom weights' }))
    await user.type(screen.getByLabelText('Ratio in the order listed'), '1:1:1')
    await user.click(screen.getByRole('button', { name: 'Apply ratio' }))
    expect(screen.getByText('The ratio has 3 part(s) but the column has 4 categories.')).toBeInTheDocument()
    expect(screen.getByLabelText('Expected weight for A')).toHaveValue('1')
  })

  it('renders the results table, exact multinomial p, small-count warning and methods text', async () => {
    setup()
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square goodness of fit (one sample)' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Chi-square goodness-of-fit test' })

    expect(screen.getByText('Significant')).toBeInTheDocument()
    expect(screen.getByText('Observed and expected counts')).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Adjusted residual' })).toBeInTheDocument()
    expect(screen.getByText('2.31')).toBeInTheDocument()
    expect(screen.getByText('1.25')).toBeInTheDocument()
    expect(screen.getByText(/Exact multinomial/)).toBeInTheDocument()
    expect(screen.getByText('0.023')).toBeInTheDocument()
    expect(screen.getByText(/969 outcomes enumerated/)).toBeInTheDocument()
    expect(screen.getByText(/expected counts are below 5 \(minimum 1.25\)/)).toBeInTheDocument()
    expect(screen.getByText('cohens w')).toBeInTheDocument()
    expect(screen.getByText('Methods text')).toBeInTheDocument()
    expect(screen.getByText('R code')).toBeInTheDocument()
    // Exported through the shared exporter, like the repeated-measures panel.
    expect(screen.getByRole('button', { name: /csv/i })).toBeInTheDocument()
  })

  it('marks the result out of date when a weight changes after the run', async () => {
    setup()
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square goodness of fit (one sample)' }))
    await user.click(await screen.findByRole('button', { name: 'Custom weights' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Chi-square goodness-of-fit test' })
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()

    const a = screen.getByLabelText('Expected weight for A')
    await user.clear(a)
    await user.type(a, '2')
    expect(await screen.findByText(/the analysis settings changed/)).toBeInTheDocument()
  })

  it('shows the backend error when the proportions do not cover the categories', async () => {
    installSession()
    server.use(
      http.get('/api/stats/test-session/frequency', () => HttpResponse.json(frequency)),
      http.post('/api/categorical/chisquare_gof', () =>
        HttpResponse.json({ detail: "expected_proportions does not cover every observed category of 'GROUP'. Missing: 'D'" }, { status: 422 }),
      ),
    )
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Chi-square goodness of fit (one sample)' }))
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await waitFor(() => expect(screen.getByText(/does not cover every observed category/)).toBeInTheDocument())
  })
})
