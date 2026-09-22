import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { beforeEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { useStore } from '../store'
import MetaPanel from './MetaPanel'

// MetaPanel is session-independent — studies are entered manually in a table,
// so no store session installation is needed.

// The study table and the result persist in panelCache across remounts, so
// each test starts from an empty cache rather than the last test's studies.
beforeEach(() => useStore.setState({ panelCache: {} }))

describe('MetaPanel', () => {
  it('renders with sample studies preloaded and four mode buttons', () => {
    render(<MetaPanel />)
    expect(screen.getByRole('button', { name: /forest \+ pool/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^subgroup$/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /meta-regression/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /publication bias/i })).toBeInTheDocument()
    expect(screen.getByDisplayValue('Trial A')).toBeInTheDocument()
  })

  it('runs Analyze (forest + pool) and renders pooled result and forest plot', async () => {
    server.use(
      http.post('/api/meta/analyze', () =>
        HttpResponse.json({
          studies: [
            { label: 'Trial A', effect: 0.75, ci_low: 0.55, ci_high: 1.02, weight_pct: 40 },
            { label: 'Trial B', effect: 0.82, ci_low: 0.6, ci_high: 1.12, weight_pct: 60 },
          ],
          random: { effect: 0.78, ci_low: 0.6, ci_high: 1.0 },
          fixed: { effect: 0.79, ci_low: 0.62, ci_high: 0.99 },
          measure: 'OR',
          null_line: 1,
          I2_pct: 12.5,
          tau2: 0.02,
          Q: 3.1,
          Q_p: 0.45,
          interpretation: 'No significant heterogeneity',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<MetaPanel />)
    await user.click(screen.getByRole('button', { name: /forest \+ pool/i }))

    await waitFor(() => expect(screen.getByText('No significant heterogeneity')).toBeInTheDocument())
    expect(screen.getByText('Pooled result')).toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  })

  it('runs Subgroup analysis and renders the subgroup table', async () => {
    server.use(
      http.post('/api/meta/subgroup', () =>
        HttpResponse.json({
          studies: [
            { label: 'Trial A', effect: 0.75, ci_low: 0.55, ci_high: 1.02, weight_pct: 40 },
          ],
          random: { effect: 0.78, ci_low: 0.6, ci_high: 1.0 },
          fixed: { effect: 0.79, ci_low: 0.62, ci_high: 0.99 },
          measure: 'OR',
          null_line: 1,
          I2_pct: 10,
          tau2: 0.01,
          Q: 1.2,
          Q_p: 0.6,
          subgroups: [
            { subgroup: 'Europe', k: 2, effect: 0.77, ci_low: 0.5, ci_high: 1.1, I2_pct: 5 },
            { subgroup: 'US', k: 2, effect: 0.9, ci_low: 0.6, ci_high: 1.3, I2_pct: 20 },
          ],
          q_between: 0.5,
          q_between_p: 0.48,
          egger_p: 0.2,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<MetaPanel />)
    await user.click(screen.getByRole('button', { name: /^subgroup$/i }))

    await waitFor(() => expect(screen.getByText('Subgroups')).toBeInTheDocument())
    expect(screen.getByText('Europe')).toBeInTheDocument()
    expect(screen.getByText('US')).toBeInTheDocument()
  })

  it('runs Meta-regression and renders slope stats', async () => {
    server.use(
      http.post('/api/meta/regression', () =>
        HttpResponse.json({
          random: { effect: 0.78, ci_low: 0.6, ci_high: 1.0 },
          fixed: { effect: 0.79, ci_low: 0.62, ci_high: 0.99 },
          measure: 'OR',
          null_line: 1,
          I2_pct: 10,
          tau2: 0.01,
          Q: 1.2,
          Q_p: 0.6,
          points: [
            { moderator: 2010, effect: 0.75, size: 10, label: 'Trial A' },
            { moderator: 2013, effect: 0.82, size: 12, label: 'Trial B' },
          ],
          line_x: [2010, 2013],
          line_y: [0.75, 0.82],
          slope: 0.02,
          slope_p: 0.3,
          slope_ci_low: -0.01,
          slope_ci_high: 0.05,
          r2_pct: 15,
          egger_p: 0.2,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<MetaPanel />)
    await user.click(screen.getByRole('button', { name: /meta-regression/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(screen.getAllByText(/meta-regression/i).length).toBeGreaterThan(0)
  })

  it('runs Publication bias and renders Egger/Begg stats plus funnel plot', async () => {
    server.use(
      http.post('/api/meta/bias', () =>
        HttpResponse.json({
          random: { effect: 0.78, ci_low: 0.6, ci_high: 1.0 },
          fixed: { effect: 0.79, ci_low: 0.62, ci_high: 0.99 },
          measure: 'OR',
          null_line: 1,
          I2_pct: 10,
          tau2: 0.01,
          Q: 1.2,
          Q_p: 0.6,
          funnel: [
            { effect: 0.75, se: 0.1, label: 'Trial A' },
            { effect: 0.82, se: 0.15, label: 'Trial B' },
          ],
          pooled_effect: 0.78,
          se_max: 0.2,
          egger_intercept: 0.3,
          egger_p: 0.6,
          begg_tau: 0.1,
          begg_p: 0.7,
          trim_fill_missing: 0,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<MetaPanel />)
    await user.click(screen.getByRole('button', { name: /publication bias/i }))

    await waitFor(() => expect(screen.getByText('Egger intercept')).toBeInTheDocument())
    expect(screen.getAllByText(/publication bias/i).length).toBeGreaterThan(0)
    expect(screen.getByText('Begg τ')).toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  })

  it('shows a validation error when fewer than 2 studies are present', async () => {
    const user = userEvent.setup()
    render(<MetaPanel />)
    // Clear the label of all but one sample row so buildStudies() filters
    // the blanked rows out, leaving fewer than 2 valid studies.
    const labelInputs = screen.getAllByPlaceholderText('Study')
    for (let i = 1; i < labelInputs.length; i++) {
      await user.clear(labelInputs[i])
    }
    await user.click(screen.getByRole('button', { name: /forest \+ pool/i }))
    await waitFor(() => expect(screen.getByText('Enter at least 2 studies.')).toBeInTheDocument())
  })

  it('shows the backend error message on failure', async () => {
    server.use(
      http.post('/api/meta/analyze', () =>
        HttpResponse.json({ detail: 'Effect sizes must be positive for OR' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<MetaPanel />)
    await user.click(screen.getByRole('button', { name: /forest \+ pool/i }))

    await waitFor(() => expect(screen.getByText('Effect sizes must be positive for OR')).toBeInTheDocument())
  })
})

describe('MetaPanel staleness', () => {
  it('ignores dataset edits but closes the export once a study changes', async () => {
    server.use(
      http.post('/api/meta/analyze', () =>
        HttpResponse.json({
          studies: [
            { label: 'Trial A', effect: 0.75, ci_low: 0.55, ci_high: 1.02, weight_pct: 50 },
            { label: 'Trial B', effect: 0.82, ci_low: 0.6, ci_high: 1.12, weight_pct: 50 },
          ],
          random: { effect: 0.78, ci_low: 0.6, ci_high: 1.0 },
          fixed: { effect: 0.79, ci_low: 0.62, ci_high: 0.99 },
          measure: 'OR', null_line: 1, I2_pct: 12.5, tau2: 0.02, Q: 3.1, Q_p: 0.45,
          export_rows: [['Study', 'OR'], ['Trial A', '0.75']],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<MetaPanel />)
    await user.click(screen.getByRole('button', { name: /forest \+ pool/i }))

    const csv = await screen.findByRole('button', { name: 'CSV' })
    expect(csv).toBeEnabled()

    // The pool is computed from the typed studies alone: an edit to the
    // loaded dataset says nothing about it.
    act(() => useStore.getState().bumpDataVersion())
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeEnabled()

    await user.type(screen.getByDisplayValue('0.75'), '1')

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeDisabled()
  })
})
