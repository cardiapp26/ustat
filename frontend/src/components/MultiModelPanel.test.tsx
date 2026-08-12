import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import MultiModelPanel from './MultiModelPanel'

afterEach(() => {
  clearSession()
  localStorage.clear()
})

const session = () =>
  makeSession({
    columns: [
      { name: 'sbp', dtype: 'float64', kind: 'numeric' },
      { name: 'crp_q', dtype: 'int64', kind: 'numeric' },
      { name: 'age', dtype: 'float64', kind: 'numeric' },
      { name: 'smoke', dtype: 'object', kind: 'categorical' },
    ],
    preview: [{ sbp: 130, crp_q: 1, age: 60, smoke: 'never' }],
  })

const eff = (level: string, beta: number, p: number) => ({
  level, reference: false, beta, se: 1.5, p,
  ci_low: beta - 3, ci_high: beta + 3,
})

const result = (over: Record<string, unknown> = {}) => ({
  outcome: 'sbp', exposure: 'crp_q', outcome_kind: 'continuous',
  effect_label: 'Mean difference', exposure_categorical: true,
  levels: ['1', '2', '3', '4'], trend_basis: 'level value',
  n_used: 495, n_dropped: 5,
  models: [
    { label: 'Crude', covariates: [],
      effects: [{ level: '1', reference: true }, eff('2', 0.5, 0.7), eff('3', 1.2, 0.4), eff('4', 2.549, 0.137)],
      trend: { level: 'trend', p: 0.1293 } },
    { label: 'Model 2', covariates: ['age', 'smoke'],
      effects: [{ level: '1', reference: true }, eff('2', 0.9, 0.6), eff('3', 1.8, 0.3), eff('4', 3.097, 0.0389)],
      trend: { level: 'trend', p: 0.0440 } },
  ],
  warnings: ['5 of 500 rows (1.0%) were dropped. Every model is fitted on the same complete cases, so the estimate moves across the row because of adjustment rather than because the sample changed.'],
  result_text: 'In the fully adjusted model (Model 2), the mean difference for crp_q was 4: 3.097.',
  ...over,
})

function mockRun(body: Record<string, unknown> = result()) {
  const seen: { request?: Record<string, unknown> } = {}
  server.use(
    http.post('/api/multimodel/analyze', async ({ request }) => {
      seen.request = (await request.json()) as Record<string, unknown>
      return HttpResponse.json(body)
    }),
  )
  return seen
}

async function build(user: ReturnType<typeof userEvent.setup>) {
  await user.selectOptions(screen.getByRole('combobox', { name: /^outcome$/i }), 'sbp')
  await user.selectOptions(screen.getByRole('combobox', { name: /exposure/i }), 'crp_q')
  await user.click(screen.getByRole('button', { name: /build the table/i }))
}

describe('MultiModelPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<MultiModelPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('starts with a crude model and two adjusted slots', () => {
    installSession(session())
    render(<MultiModelPanel />)
    expect(screen.getByDisplayValue('Crude')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Model 1')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Model 2')).toBeInTheDocument()
  })

  it('renders one column per model and one row per level', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<MultiModelPanel />)
    await build(user)

    await waitFor(() => expect(screen.getByRole('columnheader', { name: 'Crude' })).toBeInTheDocument())
    expect(screen.getByRole('columnheader', { name: 'Model 2' })).toBeInTheDocument()
    const rows = screen.getAllByRole('row')
    // header + 4 levels + P for trend
    expect(rows).toHaveLength(6)
    expect(screen.getAllByText('1.00 (reference)')).toHaveLength(2)
  })

  it('shows the estimate moving across the columns', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<MultiModelPanel />)
    await build(user)

    // Scoped to the table: the fully adjusted estimate also appears in the
    // manuscript sentence below it.
    await waitFor(() => expect(screen.getByRole('table')).toBeInTheDocument())
    const table = within(screen.getByRole('table'))
    expect(table.getByText(/2\.549/)).toBeInTheDocument()
    expect(table.getByText(/3\.097/)).toBeInTheDocument()
  })

  it('reports P for trend in every column', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<MultiModelPanel />)
    await build(user)

    await waitFor(() => expect(screen.getByRole('table')).toBeInTheDocument())
    const trendRow = within(screen.getByRole('table'))
      .getByRole('rowheader', { name: /P for trend/ }).closest('tr') as HTMLElement
    expect(within(trendRow).getByText('0.129')).toBeInTheDocument()
    expect(within(trendRow).getByText('0.044')).toBeInTheDocument()
  })

  it('says the models share their rows, which is what makes the table readable', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<MultiModelPanel />)
    await build(user)

    await waitFor(() => expect(screen.getAllByText(/same complete cases/).length).toBeGreaterThan(0))
    expect(screen.getByText(/n = 495 in every column/)).toBeInTheDocument()
  })

  it('sends each model with its own adjustment set', async () => {
    installSession(session())
    const seen = mockRun()
    const user = userEvent.setup()
    render(<MultiModelPanel />)

    // Model 1 adjusts for age; Model 2 adds smoke.
    const model1 = screen.getByDisplayValue('Model 1').closest('div')?.parentElement as HTMLElement
    await user.click(within(model1).getByRole('checkbox', { name: 'age' }))
    const model2 = screen.getByDisplayValue('Model 2').closest('div')?.parentElement as HTMLElement
    await user.click(within(model2).getByRole('checkbox', { name: 'age' }))
    await user.click(within(model2).getByRole('checkbox', { name: 'smoke' }))
    await build(user)

    await waitFor(() => expect(seen.request).toBeTruthy())
    expect(seen.request?.models).toEqual([
      { label: 'Crude', covariates: [] },
      { label: 'Model 1', covariates: ['age'] },
      { label: 'Model 2', covariates: ['age', 'smoke'] },
    ])
    // Only the categorical ones are flagged as such.
    expect(seen.request?.categorical).toEqual(['smoke'])
  })

  it('adds and removes models, keeping at least one', async () => {
    installSession(session())
    const user = userEvent.setup()
    render(<MultiModelPanel />)

    await user.click(screen.getByRole('button', { name: '+ Add' }))
    expect(screen.getByDisplayValue('Model 3')).toBeInTheDocument()

    for (const label of ['Model 3', 'Model 2', 'Model 1']) {
      await user.click(screen.getByRole('button', { name: `Remove ${label}` }))
    }
    expect(screen.getByRole('button', { name: 'Remove Crude' })).toBeDisabled()
  })

  it('shows odds ratios for a binary outcome', async () => {
    installSession(session())
    mockRun(result({
      outcome_kind: 'binary', effect_label: 'Odds ratio',
      models: [{ label: 'Crude', covariates: [],
        effects: [{ level: '1', reference: true },
                  { level: '4', reference: false, beta: 1.88, se: 0.77, p: 0.015,
                    ratio: 6.53, ratio_ci_low: 1.43, ratio_ci_high: 29.7 }] }],
      levels: ['1', '4'],
    }))
    const user = userEvent.setup()
    render(<MultiModelPanel />)
    await user.selectOptions(screen.getByRole('combobox', { name: /outcome type/i }), 'binary')
    await build(user)

    await waitFor(() => expect(screen.getByText(/6\.53 \(1\.43–29\.70\)/)).toBeInTheDocument())
  })

  it('shows the backend error rather than an empty table', async () => {
    installSession(session())
    server.use(
      http.post('/api/multimodel/analyze', () =>
        HttpResponse.json({ detail: 'The exposure cannot also be an adjustment variable' }, { status: 400 }),
      ),
    )
    const user = userEvent.setup()
    render(<MultiModelPanel />)
    await build(user)

    await waitFor(() => expect(screen.getByText(/cannot also be/)).toBeInTheDocument())
  })
})
