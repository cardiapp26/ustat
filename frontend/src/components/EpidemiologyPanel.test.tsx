import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { beforeEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession } from '../test/testUtils'
import { useStore } from '../store'
import EpidemiologyPanel from './EpidemiologyPanel'

// The panel takes inline tables and reads no dataset, so no session is installed.
beforeEach(() => {
  clearSession()
  useStore.setState({ panelCache: {} })
})

const DIRECT_RESPONSE = {
  test: 'Direct standardisation',
  alpha: 0.05,
  conf_level: 0.95,
  multiplier: 100000,
  n_strata: 3,
  total_events: 207,
  total_person_time: 105000,
  crude_rate: 197.14,
  crude_ci_low: 171.3,
  crude_ci_high: 225.9,
  standardised_rate: 321.55,
  standardised_ci_low: 281.2,
  standardised_ci_high: 366.1,
  standardised_se: 21.4,
  ci_method: 'Fay & Feuer (1997) gamma interval',
  strata: [
    { label: '0-39', events: 12, person_time: 48000, standard_population: 40000, rate: 25, ci_low: 12.9, ci_high: 43.7, weight: 0.4, contribution: 10, contribution_pct: 3.1, excluded: false },
    { label: '40-59', events: 55, person_time: 36000, standard_population: 35000, rate: 152.78, ci_low: 115.1, ci_high: 198.9, weight: 0.35, contribution: 53.47, contribution_pct: 16.6, excluded: false },
    { label: '60+', events: 140, person_time: 21000, standard_population: 25000, rate: 666.67, ci_low: 560.9, ci_high: 786.4, weight: 0.25, contribution: 166.67, contribution_pct: 51.8, excluded: false },
  ],
  warnings: ['Stratum X has zero person-time; it is excluded.'],
  result_text: 'Rates were directly standardised across 3 strata.',
  r_code: 'library(epitools)\nres <- ageadjust.direct(count = count, pop = pop, stdpop = stdpop)',
}

describe('EpidemiologyPanel', () => {
  it('renders without a session and shows the three modes', () => {
    render(<EpidemiologyPanel />)
    expect(screen.getByRole('button', { name: 'Direct standardisation' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /indirect standardisation/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Rate ratio' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /run direct standardisation/i })).toBeEnabled()
  })

  it('runs direct standardisation and renders the standardised rate, CI, table and R code', async () => {
    let body: Record<string, unknown> | null = null
    server.use(
      http.post('/api/epidemiology/direct_standardisation', async ({ request }) => {
        body = (await request.json()) as Record<string, unknown>
        return HttpResponse.json(DIRECT_RESPONSE)
      }),
    )
    const user = userEvent.setup()
    render(<EpidemiologyPanel />)
    await user.click(screen.getByRole('button', { name: /run direct standardisation/i }))

    expect(await screen.findByText('321.55')).toBeInTheDocument()
    expect(screen.getByText(/\[281\.20, 366\.10\] Fay-Feuer/)).toBeInTheDocument()
    expect(screen.getByText('197.14')).toBeInTheDocument()
    expect(screen.getByText('40-59')).toBeInTheDocument()
    expect(screen.getByText(/Stratum X has zero person-time/)).toBeInTheDocument()
    expect(screen.getByText('Rates were directly standardised across 3 strata.')).toBeInTheDocument()
    expect(screen.getByText(/res <- ageadjust\.direct/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy R code' })).toBeInTheDocument()

    expect(body).toMatchObject({ multiplier: 100000, alpha: 0.05 })
    const strata = (body as unknown as { strata: Record<string, unknown>[] }).strata
    expect(strata).toHaveLength(3)
    expect(strata[0]).toMatchObject({ label: '0-39', events: 12, person_time: 48000, standard_population: 40000 })
  })

  it('loads pasted spreadsheet rows into the direct table', async () => {
    const user = userEvent.setup()
    render(<EpidemiologyPanel />)
    await user.click(screen.getByText('Paste from a spreadsheet'))
    await user.click(screen.getByLabelText('Pasted rows'))
    await user.paste('Age\tEvents\tPT\tStd\nYoung\t3\t1000\t500\nOld\t9\t800\t300')
    await user.click(screen.getByRole('button', { name: 'Replace rows' }))
    expect(screen.getByDisplayValue('Young')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Old')).toBeInTheDocument()
    expect(screen.queryByDisplayValue('0-39')).not.toBeInTheDocument()
    expect(screen.getAllByLabelText(/standard population$/)).toHaveLength(2)
  })

  it('disables Run and explains why when the table is invalid', async () => {
    const user = userEvent.setup()
    render(<EpidemiologyPanel />)
    const events = screen.getByLabelText('Stratum 1 events')
    await user.clear(events)
    await user.type(events, '2.5')
    expect(screen.getByRole('button', { name: /run direct standardisation/i })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent(/events must be a whole number/i)

    await user.clear(events)
    await user.type(events, '4')
    expect(screen.getByRole('button', { name: /run direct standardisation/i })).toBeEnabled()

    await user.clear(screen.getByLabelText('Rate multiplier'))
    await user.type(screen.getByLabelText('Rate multiplier'), '0')
    expect(screen.getByRole('button', { name: /run direct standardisation/i })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent(/multiplier must be a number above 0/i)
  })

  it('runs the SMR, sends reference_multiplier and renders SMR, both CIs and the p-value', async () => {
    let body: Record<string, unknown> | null = null
    server.use(
      http.post('/api/epidemiology/indirect_standardisation', async ({ request }) => {
        body = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          test: 'Indirect standardisation (SMR)',
          alpha: 0.05,
          conf_level: 0.95,
          reference_multiplier: 100000,
          n_strata: 3,
          observed: 143,
          expected: 107.4,
          smr: 1.3315,
          smr_x100: 133.15,
          byar_ci_low: 1.1234,
          byar_ci_high: 1.5678,
          byar_ci_low_x100: 112.34,
          byar_ci_high_x100: 156.78,
          exact_ci_low: 1.1888,
          exact_ci_high: 1.6012,
          exact_ci_low_x100: 118.88,
          exact_ci_high_x100: 160.12,
          p_value: 0.00042,
          p_method: 'Exact two-sided Poisson',
          strata: [
            { label: '0-39', observed: 8, person_time: 30000, reference_rate: 30, expected: 9, ratio: 0.8889 },
            { label: '40-59', observed: 40, person_time: 24000, reference_rate: 130, expected: 31.2, ratio: 1.2821 },
            { label: '60+', observed: 95, person_time: 9000, reference_rate: 700, expected: 63, ratio: 1.5079 },
          ],
          warnings: [],
          result_text: 'Indirect standardisation gave SMR = 1.33.',
          r_code: 'library(epitools)\npois.exact(x = observed, pt = expected)',
        })
      }),
    )
    const user = userEvent.setup()
    render(<EpidemiologyPanel />)
    await user.click(screen.getByRole('button', { name: /indirect standardisation/i }))
    expect(screen.getByText(/reference rates are per this many person-time units/i)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /run indirect standardisation/i }))

    expect(await screen.findByText('1.33')).toBeInTheDocument()
    expect(screen.getByText('x100 = 133.2')).toBeInTheDocument()
    expect(screen.getByText('1.12 to 1.57')).toBeInTheDocument()
    expect(screen.getByText('1.19 to 1.60')).toBeInTheDocument()
    expect(screen.getByText('<0.001')).toBeInTheDocument()
    expect(screen.getByText('Indirect standardisation gave SMR = 1.33.')).toBeInTheDocument()

    expect(body).toMatchObject({ reference_multiplier: 100000, alpha: 0.05 })
    const strata = (body as unknown as { strata: Record<string, unknown>[] }).strata
    expect(strata[1]).toMatchObject({ label: '40-59', observed: 40, person_time: 24000, reference_rate: 130 })
  })

  it('sends reference events and person-time (multiplier 1) in counts mode', async () => {
    let body: Record<string, unknown> | null = null
    server.use(
      http.post('/api/epidemiology/indirect_standardisation', async ({ request }) => {
        body = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({ conf_level: 0.95, reference_multiplier: 1, observed: 1, expected: 1, smr: 1, smr_x100: 100,
          byar_ci_low: 0, byar_ci_high: 2, byar_ci_low_x100: 0, byar_ci_high_x100: 200,
          exact_ci_low: 0, exact_ci_high: 2, exact_ci_low_x100: 0, exact_ci_high_x100: 200,
          p_value: 1, p_method: 'x', strata: [], warnings: [], result_text: 'ok', r_code: '' })
      }),
    )
    const user = userEvent.setup()
    render(<EpidemiologyPanel />)
    await user.click(screen.getByRole('button', { name: /indirect standardisation/i }))
    await user.click(screen.getByRole('button', { name: 'Events + person-time' }))
    // Reference counts are blank, so Run is disabled until they are filled in.
    expect(screen.getByRole('button', { name: /run indirect standardisation/i })).toBeDisabled()
    for (let i = 1; i <= 3; i++) {
      await user.type(screen.getByLabelText(`Stratum ${i} reference events`), '10')
      await user.type(screen.getByLabelText(`Stratum ${i} reference person-time`), '1000')
    }
    await user.click(screen.getByRole('button', { name: /run indirect standardisation/i }))
    await waitFor(() => expect(body).not.toBeNull())
    expect(body).toMatchObject({ reference_multiplier: 1 })
    const strata = (body as unknown as { strata: Record<string, unknown>[] }).strata
    expect(strata[0]).toMatchObject({ reference_events: 10, reference_person_time: 1000 })
    expect(strata[0]).not.toHaveProperty('reference_rate')
  })

  it('runs the rate ratio and shows the exact CI when the log CI is undefined', async () => {
    server.use(
      http.post('/api/epidemiology/rate_ratio', () =>
        HttpResponse.json({
          test: 'Rate ratio and rate difference',
          alpha: 0.05,
          conf_level: 0.95,
          group1: { events: 5, person_time: 1000, rate: 0.005, ci_low: 0.0016, ci_high: 0.0117 },
          group2: { events: 0, person_time: 1200, rate: 0, ci_low: 0, ci_high: 0.0031 },
          rate_ratio: null,
          rr_ci_low: null,
          rr_ci_high: null,
          rr_se_log: null,
          rr_ci_method: 'Log method',
          rr_exact_ci_low: 1.234,
          rr_exact_ci_high: null,
          rate_difference: 0.005,
          rd_ci_low: 0.0006,
          rd_ci_high: 0.0094,
          rd_se: 0.002,
          p_value: 0.0156,
          p_value_midp: 0.0078,
          p_value_wald: null,
          p_method: 'Conditional exact',
          warnings: ['A group has zero events: the log-method CI is undefined.'],
          result_text: 'Incidence rates were 0.005 in group 1.',
          r_code: 'poisson.test(c(5, 0), T = c(1000, 1200))',
        }),
      ),
    )
    const user = userEvent.setup()
    render(<EpidemiologyPanel />)
    await user.click(screen.getByRole('button', { name: 'Rate ratio' }))
    await user.click(screen.getByRole('button', { name: /run rate ratio/i }))

    expect(await screen.findByText('undefined (zero count)')).toBeInTheDocument()
    expect(screen.getByText('1.234 to ∞')).toBeInTheDocument()
    expect(screen.getByText('0.016')).toBeInTheDocument()
    expect(screen.getByText('0.008')).toBeInTheDocument()
    expect(screen.getByText(/zero events/)).toBeInTheDocument()
    expect(screen.getByText('Incidence rates were 0.005 in group 1.')).toBeInTheDocument()
  })

  it('disables Run for a rate ratio group with zero person-time', async () => {
    const user = userEvent.setup()
    render(<EpidemiologyPanel />)
    await user.click(screen.getByRole('button', { name: 'Rate ratio' }))
    const pt = screen.getByLabelText('Group 2 person-time')
    await user.clear(pt)
    await user.type(pt, '0')
    expect(screen.getByRole('button', { name: /run rate ratio/i })).toBeDisabled()
    expect(within(screen.getByRole('alert')).getByText(/group 2: person-time must be a number above 0/i)).toBeInTheDocument()
  })

  it('surfaces a backend 422 detail as an error', async () => {
    server.use(
      http.post('/api/epidemiology/rate_ratio', () =>
        HttpResponse.json({ detail: "Group 1: 'events1' must be a whole number (got 2.5)." }, { status: 422 }),
      ),
    )
    const user = userEvent.setup()
    render(<EpidemiologyPanel />)
    await user.click(screen.getByRole('button', { name: 'Rate ratio' }))
    await user.click(screen.getByRole('button', { name: /run rate ratio/i }))
    expect(await screen.findByText(/must be a whole number \(got 2\.5\)/)).toBeInTheDocument()
  })
})
