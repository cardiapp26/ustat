import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { useStore } from '../store'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import IntervalCensoredPanel from './IntervalCensoredPanel'

afterEach(() => clearSession())

describe('IntervalCensoredPanel', () => {
  it('renders the form controls given a session', () => {
    installSession()
    const session = makeSession()
    render(<IntervalCensoredPanel session={session} />)
    expect(screen.getByText('Interval-censored survival')).toBeInTheDocument()
    expect(screen.getByText('Lower bound (L)')).toBeInTheDocument()
    expect(screen.getByText('Upper bound (R)')).toBeInTheDocument()
  })

  it('shows a validation error when lower and upper bound are the same column', async () => {
    const session = makeSession()
    const user = userEvent.setup()
    render(<IntervalCensoredPanel session={session} />)

    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'AGE')
    await user.selectOptions(selects[1], 'AGE')
    await user.click(screen.getByRole('button', { name: /run analysis/i }))

    await waitFor(() =>
      expect(screen.getByText('Lower and upper bound must be different columns.')).toBeInTheDocument(),
    )
  })

  it('runs the interval-censored analysis and renders the result', async () => {
    const session = makeSession()
    server.use(
      http.post('/api/survival_advanced/interval_censored', () =>
        HttpResponse.json({
          n: 3,
          n_exact: 0,
          n_interval_censored: 2,
          n_right_censored: 1,
          median_survival_time: 42,
          npmle_curve: [
            { time: 0, survival: 1, lower: 1, upper: 1 },
            { time: 42, survival: 0.5, lower: 0.3, upper: 0.7 },
          ],
          groups: null,
          regression: {
            shape: 1.2,
            aic: 55.5,
            coefficients: [
              { variable: 'DM', time_ratio: 0.8, tr_ci_low: 0.6, tr_ci_high: 1.1, hazard_ratio: 1.3, hr_ci_low: 0.9, hr_ci_high: 1.8, p: 0.04 },
            ],
          },
          result_text: 'Turnbull NPMLE estimated with median survival of 42.',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<IntervalCensoredPanel session={session} />)

    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'AGE')
    await user.selectOptions(selects[1], 'LDL')
    await user.click(screen.getByRole('button', { name: /run analysis/i }))

    await waitFor(() => expect(screen.getByText('Turnbull NPMLE estimated with median survival of 42.')).toBeInTheDocument())
    expect(screen.getByText('42')).toBeInTheDocument()
    expect(screen.getByText('Weibull regression')).toBeInTheDocument()
    const table = screen.getByRole('table')
    expect(table).toHaveTextContent('DM')
  })

  it('shows the error message from the backend on failure', async () => {
    const session = makeSession()
    server.use(
      http.post('/api/survival_advanced/interval_censored', () =>
        HttpResponse.json({ detail: 'Upper bound must be >= lower bound' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<IntervalCensoredPanel session={session} />)

    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'AGE')
    await user.selectOptions(selects[1], 'LDL')
    await user.click(screen.getByRole('button', { name: /run analysis/i }))

    await waitFor(() => expect(screen.getByText('Upper bound must be >= lower bound')).toBeInTheDocument())
  })
})

describe('IntervalCensoredPanel staleness', () => {
  it('closes the curve export once the data changes under it', async () => {
    installSession()
    server.use(
      http.post('/api/survival_advanced/interval_censored', () =>
        HttpResponse.json({
          n: 3, n_exact: 0, n_interval_censored: 2, n_right_censored: 1,
          median_survival_time: 42,
          npmle_curve: [
            { time: 0, survival: 1, lower: 1, upper: 1 },
            { time: 42, survival: 0.5, lower: 0.3, upper: 0.7 },
          ],
          groups: null,
          regression: null,
          result_text: 'Turnbull NPMLE estimated.',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<IntervalCensoredPanel session={makeSession()} />)
    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'AGE')
    await user.selectOptions(selects[1], 'LDL')
    await user.click(screen.getByRole('button', { name: /run analysis/i }))

    const png = await screen.findByRole('button', { name: 'PNG 300dpi' })
    expect(png).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'PNG 300dpi' })).toBeDisabled()
  })
})
