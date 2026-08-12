import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import ThresholdPanel from './ThresholdPanel'

afterEach(() => {
  clearSession()
  localStorage.clear()
})

const session = () =>
  makeSession({
    columns: [
      { name: 'x', dtype: 'float64', kind: 'numeric' },
      { name: 'y', dtype: 'float64', kind: 'numeric' },
      { name: 'age', dtype: 'float64', kind: 'numeric' },
      { name: 'site', dtype: 'object', kind: 'categorical' },
    ],
    preview: [{ x: 1, y: 2, age: 60, site: 'A' }],
  })

const result = (over: Record<string, unknown> = {}) => ({
  outcome: 'y', exposure: 'x', outcome_kind: 'continuous',
  n_used: 397, n_dropped: 3,
  breakpoint: 40.53, breakpoint_ci: { low: 37.74, high: 42.72 },
  search_range: { low: 10, high: 90, n_candidates: 380 },
  effect_below: { beta: 0.2617, se: 0.0137, p: 1e-40, ci_low: 0.2348, ci_high: 0.2886 },
  effect_above: { beta: -0.0894, se: 0.008, p: 1e-20, ci_low: -0.1051, ci_high: -0.0738 },
  effect_difference: { beta: -0.3511, se: 0.0193, p: 6.33e-54, ci_low: -0.3891, ci_high: -0.3131 },
  effect_single_line: { beta: 0.12, se: 0.02, p: 0.001, ci_low: 0.08, ci_high: 0.16 },
  loglik_single: -1067.62, loglik_segmented: -947.03,
  lr_stat: 241.16, lr_p: 2.19e-54,
  effect_label: 'Mean difference',
  profile: [{ k: 20, loglik: -1000 }, { k: 40.53, loglik: -947.03 }, { k: 60, loglik: -990 }],
  curve: { x: [0, 40.53, 100], y: [0, 12, 7] },
  verdict: 'A two-segment model fits better than a straight line',
  result_text: 'A two-piecewise model identified an inflection point of x at 40.5.',
  warnings: ['3 of 400 rows (0.8%) were dropped for missing values in the model variables.'],
  caveat: 'The breakpoint is chosen by maximising the same likelihood the test then uses, so the likelihood-ratio p-value is optimistic.',
  ...over,
})

function mockRun(body: Record<string, unknown> = result()) {
  const seen: { request?: Record<string, unknown> } = {}
  server.use(
    http.post('/api/threshold/analyze', async ({ request }) => {
      seen.request = (await request.json()) as Record<string, unknown>
      return HttpResponse.json(body)
    }),
  )
  return seen
}

async function runWith(user: ReturnType<typeof userEvent.setup>, opts: { exposure?: string } = {}) {
  await user.selectOptions(screen.getByRole('combobox', { name: /^outcome$/i }), 'y')
  await user.selectOptions(screen.getByRole('combobox', { name: /exposure/i }), opts.exposure ?? 'x')
  await user.click(screen.getByRole('button', { name: /find the threshold/i }))
}

describe('ThresholdPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<ThresholdPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('offers only numeric columns as the exposure', () => {
    installSession(session())
    render(<ThresholdPanel />)
    const exposure = screen.getByRole('combobox', { name: /exposure/i })
    expect(within(exposure).getByRole('option', { name: 'x' })).toBeInTheDocument()
    expect(within(exposure).queryByRole('option', { name: 'site' })).not.toBeInTheDocument()
  })

  it('cannot run until an outcome and an exposure are chosen', () => {
    installSession(session())
    render(<ThresholdPanel />)
    expect(screen.getByRole('button', { name: /find the threshold/i })).toBeDisabled()
  })

  it('reports the breakpoint with its interval and both slopes', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<ThresholdPanel />)
    await runWith(user)

    await waitFor(() => expect(screen.getByText(/x = 40\.530/)).toBeInTheDocument())
    expect(screen.getByText(/37\.74 to 42\.72/)).toBeInTheDocument()
    expect(screen.getByText('0.262 (0.235 to 0.289)')).toBeInTheDocument()
    expect(screen.getByText('-0.089 (-0.105 to -0.074)')).toBeInTheDocument()
  })

  it('shows the straight-line model the breakpoint is tested against', async () => {
    // Without it the reader cannot see what the two segments bought.
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<ThresholdPanel />)
    await runWith(user)

    await waitFor(() => expect(screen.getByText('One line only')).toBeInTheDocument())
    expect(screen.getByText('0.120 (0.080 to 0.160)')).toBeInTheDocument()
  })

  it('always shows the caveat about the p-value, not only the p-value', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<ThresholdPanel />)
    await runWith(user)

    await waitFor(() => expect(screen.getByText(/optimistic/)).toBeInTheDocument())
    expect(screen.getByText(/dropped for missing values/)).toBeInTheDocument()
  })

  it('sends the outcome type and marks categorical covariates as such', async () => {
    installSession(session())
    const seen = mockRun()
    const user = userEvent.setup()
    render(<ThresholdPanel />)
    await user.click(screen.getByRole('checkbox', { name: /site/ }))
    await runWith(user)

    await waitFor(() => expect(seen.request).toBeTruthy())
    expect(seen.request).toMatchObject({
      outcome: 'y', exposure: 'x', outcome_kind: 'continuous',
      covariates: ['site'], categorical: ['site'],
    })
  })

  it('asks for a time column before running a survival model', async () => {
    installSession(session())
    const user = userEvent.setup()
    render(<ThresholdPanel />)
    await user.click(screen.getByRole('radio', { name: /time-to-event/i }))
    await user.selectOptions(screen.getByRole('combobox', { name: /^outcome$/i }), 'y')
    await user.selectOptions(screen.getByRole('combobox', { name: /exposure/i }), 'x')
    await user.click(screen.getByRole('button', { name: /find the threshold/i }))

    expect(await screen.findByText(/needs a follow-up time column/i)).toBeInTheDocument()
  })

  it('draws the fitted curve and the profile likelihood', async () => {
    // The profile is what shows whether the breakpoint is identified at all,
    // so it is not optional decoration.
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<ThresholdPanel />)
    await runWith(user)

    await waitFor(() => expect(screen.getAllByTestId('plotly-mock')).toHaveLength(2))
    const profile = JSON.parse(
      screen.getAllByTestId('plotly-mock')[1].getAttribute('data-plotly') ?? '[]')
    expect(profile[0].x).toEqual([20, 40.53, 60])
  })

  it('shows an odds ratio on a log axis for a binary outcome', async () => {
    installSession(session())
    mockRun(result({ outcome_kind: 'binary', effect_label: 'Odds ratio' }))
    const user = userEvent.setup()
    render(<ThresholdPanel />)
    await user.click(screen.getByRole('radio', { name: /binary/i }))
    await runWith(user)

    await waitFor(() => expect(screen.getAllByTestId('plotly-mock').length).toBeGreaterThan(0))
    const layout = JSON.parse(
      screen.getAllByTestId('plotly-mock')[0].getAttribute('data-layout') ?? '{}')
    expect(layout.yaxis.type).toBe('log')
  })

  it('shows the backend error rather than an empty panel', async () => {
    installSession(session())
    server.use(
      http.post('/api/threshold/analyze', () =>
        HttpResponse.json({ detail: 'Only 12 complete rows — too few' }, { status: 400 }),
      ),
    )
    const user = userEvent.setup()
    render(<ThresholdPanel />)
    await runWith(user)

    await waitFor(() => expect(screen.getByText(/too few/)).toBeInTheDocument())
  })
})
