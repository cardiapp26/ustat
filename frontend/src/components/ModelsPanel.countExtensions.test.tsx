import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import ModelsPanel from './ModelsPanel'
import VisualModelPanel from './VisualModelPanel'

afterEach(() => clearSession())

function stubBackgroundEndpoints() {
  server.use(
    http.get(/\/api\/stats\/.*\/sparklines/, () => HttpResponse.json({})),
    http.get(/\/api\/stats\/.*\/missing/, () =>
      HttpResponse.json({ total_rows: 3, rows_affected: 0, pct_affected: 0, per_column: {} }),
    ),
  )
}

const countSession = () =>
  makeSession({
    columns: [
      { name: 'EVENTS', dtype: 'int64', kind: 'numeric' },
      { name: 'AGE', dtype: 'float64', kind: 'numeric' },
      { name: 'LDL', dtype: 'float64', kind: 'numeric' },
      { name: 'FUTIME', dtype: 'float64', kind: 'numeric' },
    ],
    preview: [
      { EVENTS: 0, AGE: 55, LDL: 120, FUTIME: 2.5 },
      { EVENTS: 2, AGE: 62, LDL: 140, FUTIME: 4 },
      { EVENTS: 0, AGE: 48, LDL: 110, FUTIME: 1.2 },
    ],
  })

/** A predictor checkbox (the first match, i.e. the count-model list). */
function predictorBox(name: string): HTMLInputElement {
  for (const el of screen.getAllByText(name)) {
    const label = el.closest('label')
    const box = label ? within(label).queryByRole('checkbox') : null
    if (box) return box as HTMLInputElement
  }
  throw new Error(`No predictor checkbox for ${name}`)
}

const poissonCoefs = [
  { variable: 'const', log_irr: -1, irr: 0.37, se: 0.2, z: -5, p: 0.001, ci_low: -1.4, ci_high: -0.6, irr_ci_low: 0.25, irr_ci_high: 0.55 },
  { variable: 'AGE', log_irr: 0.05, irr: 1.05, se: 0.01, z: 5, p: 0.001, ci_low: 0.03, ci_high: 0.07, irr_ci_low: 1.03, irr_ci_high: 1.07 },
]

describe('ModelsPanel count extensions', () => {
  it('Poisson: sends exposure_col only when an exposure is chosen, and notes the rate model', async () => {
    stubBackgroundEndpoints()
    installSession(countSession())
    const bodies: Array<Record<string, unknown>> = []
    server.use(
      http.post('/api/models/poisson', async ({ request }) => {
        const body = await request.json() as Record<string, unknown>
        bodies.push(body)
        return HttpResponse.json({
          model: 'Poisson Regression', outcome: 'EVENTS', n: 3, aic: 10, bic: 12,
          exposure_col: body.exposure_col ?? null, rate_model: Boolean(body.exposure_col),
          dispersion: 0.9, overdispersed: false, dispersion_note: null,
          coefficients: poissonCoefs, result_text: 'Poisson done.',
        })
      }),
    )
    const user = userEvent.setup()
    render(<ModelsPanel />)

    await user.click(screen.getByRole('radio', { name: /^Poisson Regression/ }))
    await user.click(predictorBox('AGE'))

    // Default is None: the request must not carry exposure_col at all.
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))
    await screen.findByText('Poisson Regression', { selector: 'h4' })
    expect(bodies[0]).not.toHaveProperty('exposure_col')
    expect(screen.queryByText(/Rate model: exposure/)).not.toBeInTheDocument()

    const exposure = screen.getByRole('combobox', { name: /Exposure \/ follow-up time/ })
    // The outcome is not offered as an exposure.
    expect(within(exposure).queryByRole('option', { name: 'EVENTS' })).toBeNull()
    await user.selectOptions(exposure, 'FUTIME')
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    await waitFor(() => expect(bodies).toHaveLength(2))
    expect(bodies[1].exposure_col).toBe('FUTIME')
    expect(bodies[1].predictors).toEqual(['AGE'])
    expect(await screen.findByText('Rate model: exposure = FUTIME')).toBeInTheDocument()
    expect(screen.getByText('Dispersion (χ²/df)')).toBeInTheDocument()
  })

  it('Poisson: shows an amber overdispersion warning only when the fit is overdispersed', async () => {
    stubBackgroundEndpoints()
    installSession(countSession())
    let overdispersed = true
    server.use(
      http.post('/api/models/poisson', () =>
        HttpResponse.json({
          model: 'Poisson Regression', outcome: 'EVENTS', n: 3, aic: 10, bic: 12,
          exposure_col: null, rate_model: false,
          dispersion: overdispersed ? 3.4 : 1.0, overdispersed,
          dispersion_note: overdispersed
            ? 'Pearson chi2 / df = 3.40 (> 1.5) indicates overdispersion: consider negative binomial regression or robust (sandwich) standard errors.'
            : null,
          coefficients: poissonCoefs,
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ModelsPanel />)
    await user.click(screen.getByRole('radio', { name: /^Poisson Regression/ }))
    await user.click(predictorBox('AGE'))
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/Overdispersion warning/)
    expect(alert).toHaveTextContent(/negative binomial regression or robust/)
    expect(alert.className).toMatch(/amber/)

    overdispersed = false
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))
    await waitFor(() => expect(screen.getByText('1.000')).toBeInTheDocument())
    expect(screen.queryByText(/Overdispersion warning/)).not.toBeInTheDocument()
  })

  it('Zero-inflated Poisson: posts to /zip with inflation predictors and renders both tables, zeros check and Vuong verdict', async () => {
    stubBackgroundEndpoints()
    installSession(countSession())
    let body: Record<string, unknown> | undefined
    server.use(
      http.post('/api/models/zip', async ({ request }) => {
        body = await request.json() as Record<string, unknown>
        return HttpResponse.json({
          model: 'Zero-inflated Poisson', outcome: 'EVENTS', n: 120, n_excluded: 0, imputation: 'listwise',
          aic: 410.5, bic: 428.2, loglik: -200.2, converged: true,
          exposure_col: 'FUTIME', rate_model: true, inflation_predictors: ['LDL'],
          n_zeros: 70, observed_zero_fraction: 0.5833,
          expected_zeros: 69.4, expected_zero_fraction: 0.5783, expected_zeros_standard: 51.2,
          standard_model_aic: 455.1, standard_model_bic: 463.4,
          alpha: null, alpha_se: null, theta: null,
          vuong: {
            statistic: { raw: 4.1, aic: 3.8, bic: 3.2 },
            p: { raw: 0.00004, aic: 0.00014, bic: 0.0014 },
            favours: { raw: 'zero_inflated', aic: 'zero_inflated', bic: 'zero_inflated' },
            n: 120, standard_model: 'Poisson', preferred: 'zero_inflated',
            preferred_label: 'Zero-inflated Poisson', note: 'Vuong note.',
          },
          warnings: ['Example warning from the server.'],
          count_coefficients: [
            { variable: 'const', log_irr: 0.4, irr: 1.49, se: 0.1, z: 4, p: 0.0001, ci_low: 0.2, ci_high: 0.6, irr_ci_low: 1.22, irr_ci_high: 1.82 },
            { variable: 'AGE', log_irr: 0.03, irr: 1.03, se: 0.01, z: 3, p: 0.003, ci_low: 0.01, ci_high: 0.05, irr_ci_low: 1.01, irr_ci_high: 1.05 },
          ],
          inflation_coefficients: [
            { variable: 'const', logit: -0.7, or: 0.5, se: 0.3, z: -2.3, p: 0.02, ci_low: -1.3, ci_high: -0.1, or_ci_low: 0.27, or_ci_high: 0.9 },
            { variable: 'LDL', logit: 0.02, or: 1.02, se: 0.008, z: 2.5, p: 0.012, ci_low: 0.004, ci_high: 0.036, or_ci_low: 1.004, or_ci_high: 1.037 },
          ],
          result_text: 'Zero-inflated Poisson regression was performed to model EVENTS.',
        })
      }),
    )
    const user = userEvent.setup()
    render(<ModelsPanel />)

    await user.click(screen.getByRole('radio', { name: /^Zero-Inflated Poisson/ }))
    await user.click(predictorBox('AGE'))
    await user.selectOptions(screen.getByRole('combobox', { name: /Exposure \/ follow-up time/ }), 'FUTIME')
    expect(screen.getByText(/Zero-inflation predictors \(optional, default intercept only\)/)).toBeInTheDocument()
    await user.click(screen.getByRole('checkbox', { name: 'Zero-inflation predictor LDL' }))
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    await screen.findByText('Count model (incidence rate ratios)')
    expect(body).toMatchObject({
      outcome: 'EVENTS', predictors: ['AGE'], inflation_predictors: ['LDL'], exposure_col: 'FUTIME',
    })

    // Two coefficient tables.
    expect(screen.getByText('Zero-inflation model (odds of a structural zero)')).toBeInTheDocument()
    expect(screen.getByText('1.490')).toBeInTheDocument()   // count const IRR
    expect(screen.getByText('1.020')).toBeInTheDocument()   // inflation LDL OR

    // Zeros check.
    expect(screen.getByText('Zeros check')).toBeInTheDocument()
    expect(screen.getByText(/^70 \(58\.3%\)/)).toBeInTheDocument()
    expect(screen.getByText('51.2')).toBeInTheDocument()

    // Vuong verdict in plain words.
    const verdict = screen.getAllByRole('status').find((el) => /Preferred:/.test(el.textContent ?? ''))
    expect(verdict).toBeDefined()
    expect(verdict).toHaveTextContent('Preferred: Zero-inflated Poisson.')
    expect(verdict).toHaveTextContent(/fits significantly better than the standard Poisson model/)
    expect(screen.getByText('Rate model: exposure = FUTIME')).toBeInTheDocument()
    expect(screen.getByText('Example warning from the server.')).toBeInTheDocument()
  })

  it('Zero-inflated negative binomial: intercept-only omits inflation_predictors; "neither" verdict advises the simpler model; errors surface', async () => {
    stubBackgroundEndpoints()
    installSession(countSession())
    const bodies: Array<Record<string, unknown>> = []
    let fail = false
    server.use(
      http.post('/api/models/zinb', async ({ request }) => {
        bodies.push(await request.json() as Record<string, unknown>)
        if (fail) return HttpResponse.json({ detail: 'needs at least one zero count' }, { status: 422 })
        return HttpResponse.json({
          model: 'Zero-inflated negative binomial', outcome: 'EVENTS', n: 50, aic: 100, bic: 110, loglik: -45,
          converged: true, inflation_predictors: [], n_zeros: 20, observed_zero_fraction: 0.4,
          expected_zeros: 20.3, expected_zero_fraction: 0.406, expected_zeros_standard: 19.8,
          alpha: 0.45, alpha_se: 0.1, theta: 2.22,
          vuong: {
            statistic: { raw: 0.5, aic: 0.2, bic: -0.4 }, p: { raw: 0.6, aic: 0.84, bic: 0.7 },
            favours: { raw: 'neither', aic: 'neither', bic: 'neither' },
            standard_model: 'Negative binomial', preferred: 'neither', preferred_label: 'Neither (not distinguishable)',
          },
          count_coefficients: [{ variable: 'AGE', log_irr: 0.1, irr: 1.1, se: 0.05, z: 2, p: 0.04, ci_low: 0, ci_high: 0.2, irr_ci_low: 1.0, irr_ci_high: 1.22 }],
          inflation_coefficients: [{ variable: 'const', logit: -1, or: 0.37, se: 0.4, z: -2.5, p: 0.01, ci_low: -1.8, ci_high: -0.2, or_ci_low: 0.17, or_ci_high: 0.82 }],
        })
      }),
    )
    const user = userEvent.setup()
    render(<ModelsPanel />)
    await user.click(screen.getByRole('radio', { name: /^Zero-Inflated Negative Binomial/ }))
    await user.click(predictorBox('AGE'))
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))

    await screen.findByText('Count model (incidence rate ratios)')
    expect(bodies[0]).not.toHaveProperty('inflation_predictors')
    expect(bodies[0]).not.toHaveProperty('exposure_col')
    expect(screen.getByText(/alpha \(dispersion\)/)).toBeInTheDocument()
    const verdict = screen.getAllByRole('status').find((el) => /Preferred:/.test(el.textContent ?? ''))
    expect(verdict).toHaveTextContent(/cannot tell/)
    expect(verdict).toHaveTextContent(/prefer the simpler standard negative binomial model/)

    fail = true
    await user.click(screen.getByRole('button', { name: 'Fit Model' }))
    await waitFor(() => expect(screen.getByText('needs at least one zero count')).toBeInTheDocument())
  })

  it('Negative binomial (Visual Models, GLM tab): sends exposure_col and shows rate model note, alpha and theta', async () => {
    stubBackgroundEndpoints()
    installSession(countSession())
    let body: Record<string, unknown> | undefined
    server.use(
      http.get('/api/stats/:sid/missing', () =>
        HttpResponse.json({ total_rows: 3, rows_affected: 0, pct_affected: 0, per_column: {} })),
      http.post('/api/models/negbinom', async ({ request }) => {
        body = await request.json() as Record<string, unknown>
        return HttpResponse.json({
          model: 'Negative Binomial Regression', n: 3, aic: 20, bic: 22,
          exposure_col: 'FUTIME', rate_model: true, alpha: 0.3456, alpha_se: 0.0789, theta: 2.8935,
          dispersion_note: 'alpha is the negative-binomial dispersion estimated by maximum likelihood.',
          coefficients: poissonCoefs,
        })
      }),
    )
    const user = userEvent.setup()
    render(<VisualModelPanel />)
    await user.click(screen.getByRole('button', { name: 'GLM (Gamma / Neg. Binom.)' }))
    await user.click(screen.getByRole('button', { name: 'Neg. Binom.' }))
    await user.selectOptions(screen.getByRole('combobox', { name: /Exposure \/ follow-up time/ }), 'FUTIME')
    const label = screen.getAllByText('AGE').map((el) => el.closest('label')).find((l) => l && within(l).queryByRole('checkbox'))
    await user.click(within(label as HTMLElement).getByRole('checkbox'))
    await user.click(await screen.findByRole('button', { name: 'Fit GLM' }))

    await screen.findByText('Rate model: exposure = FUTIME')
    expect(body?.exposure_col).toBe('FUTIME')
    expect(body?.predictors).toEqual(['AGE'])
    expect(screen.getByText(/alpha = 0\.3456/)).toBeInTheDocument()
    expect(screen.getByText(/theta = 2\.8935/)).toBeInTheDocument()
  })
})
