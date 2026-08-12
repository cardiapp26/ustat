import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import SubgroupPanel from './SubgroupPanel'

afterEach(() => {
  clearSession()
  localStorage.clear()
  useStore.setState({ forestHandoff: null, forestHandoffAppend: false, panelCache: {} })
})

const session = () =>
  makeSession({
    columns: [
      { name: 'sbp', dtype: 'float64', kind: 'numeric' },
      { name: 'treat', dtype: 'float64', kind: 'numeric' },
      { name: 'age', dtype: 'float64', kind: 'numeric' },
      { name: 'sex', dtype: 'object', kind: 'categorical' },
    ],
    preview: [{ sbp: 140, treat: 1, age: 60, sex: 'F' }],
  })

const row = (level: string, beta: number, p: number, n = 296) => ({
  level, n, beta, se: 1.1, p, ci_low: beta - 2, ci_high: beta + 2,
})

const result = (over: Record<string, unknown> = {}) => ({
  outcome: 'sbp', exposure: 'treat', outcome_kind: 'continuous',
  effect_label: 'Mean difference', null_value: 0,
  overall: { level: 'Overall', n: 597, beta: -2.6, se: 0.8, p: 0.001, ci_low: -4.2, ci_high: -1.0 },
  subgroups: [{
    variable: 'sex', levels: ['F', 'M'],
    rows: [row('F', -5.759, 0.00001), row('M', 0.505, 0.65, 300)],
    p_interaction: 0.000137, interaction_note: 'likelihood-ratio test on 1 df', n_used: 596,
  }],
  warnings: [],
  result_text: 'Overall the mean difference for treat was -2.600.',
  caveat: 'Significance within a stratum is not evidence that the effect differs between strata.',
  ...over,
})

function mockRun(body: Record<string, unknown> = result()) {
  const seen: { request?: Record<string, unknown> } = {}
  server.use(
    http.post('/api/subgroup/analyze', async ({ request }) => {
      seen.request = (await request.json()) as Record<string, unknown>
      return HttpResponse.json(body)
    }),
  )
  return seen
}

async function run(user: ReturnType<typeof userEvent.setup>) {
  await user.selectOptions(screen.getByRole('combobox', { name: /^outcome$/i }), 'sbp')
  await user.selectOptions(screen.getByRole('combobox', { name: /exposure/i }), 'treat')
  const box = screen.getByText('Subgroup variables').closest('div') as HTMLElement
  await user.click(within(box).getByRole('checkbox', { name: 'sex' }))
  await user.click(screen.getByRole('button', { name: /run subgroup analysis/i }))
}

describe('SubgroupPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<SubgroupPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('will not run without a subgroup variable', () => {
    installSession(session())
    render(<SubgroupPanel />)
    expect(screen.getByRole('button', { name: /run subgroup analysis/i })).toBeDisabled()
  })

  it('puts the interaction p on the variable row, not on the strata', async () => {
    // The whole point of the layout: the stratum p-values answer "is there an
    // effect here", and only the variable row answers "is it different here".
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<SubgroupPanel />)
    await run(user)

    await waitFor(() => expect(screen.getByRole('table')).toBeInTheDocument())
    const table = within(screen.getByRole('table'))
    const varRow = table.getByRole('rowheader', { name: 'sex' }).closest('tr') as HTMLElement
    expect(within(varRow).getByText('<0.001')).toBeInTheDocument()
    const fRow = table.getByRole('rowheader', { name: /^F/ }).closest('tr') as HTMLElement
    expect(within(fRow).getByText('-5.759 (-7.759–-3.759)')).toBeInTheDocument()
  })

  it('shows the overall effect as the reference row', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<SubgroupPanel />)
    await run(user)

    await waitFor(() => expect(screen.getByRole('rowheader', { name: 'Overall' })).toBeInTheDocument())
    const overall = screen.getByRole('rowheader', { name: 'Overall' }).closest('tr') as HTMLElement
    expect(within(overall).getByText('597')).toBeInTheDocument()
  })

  it('always shows the caveat about reading significance off the strata', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<SubgroupPanel />)
    await run(user)

    await waitFor(() =>
      expect(screen.getByText(/not evidence that the effect differs between strata/)).toBeInTheDocument())
  })

  it('flags a stratum too thin to estimate anything from', async () => {
    installSession(session())
    mockRun(result({
      subgroups: [{
        variable: 'sex', levels: ['F', 'M'],
        rows: [row('F', -5.7, 0.01), { ...row('M', 1.2, 0.9, 12), thin: true }],
        p_interaction: 0.4, interaction_note: 'likelihood-ratio test on 1 df', n_used: 300,
      }],
      warnings: ["'sex': M has fewer than 20 observations, so its estimate is fitted to very little."],
    }))
    const user = userEvent.setup()
    render(<SubgroupPanel />)
    await run(user)

    await waitFor(() => expect(screen.getByText('thin')).toBeInTheDocument())
    expect(screen.getByText(/fitted to very little/)).toBeInTheDocument()
  })

  it('draws a forest with a reference line at the null', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<SubgroupPanel />)
    await run(user)

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const layout = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-layout') ?? '{}')
    expect(layout.shapes[0].x0).toBe(0)
    const traces = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-plotly') ?? '[]')
    expect(traces[0].y).toEqual(['F', 'M', 'Overall'])
  })

  it('uses a log axis and a null of 1 for a ratio outcome', async () => {
    installSession(session())
    mockRun(result({
      outcome_kind: 'binary', effect_label: 'Odds ratio', null_value: 1,
      overall: { level: 'Overall', n: 599, beta: -0.5, ratio: 0.61, ratio_ci_low: 0.4, ratio_ci_high: 0.93, p: 0.02 },
      subgroups: [{
        variable: 'sex', levels: ['F'],
        rows: [{ level: 'F', n: 297, beta: -0.55, ratio: 0.58, ratio_ci_low: 0.33, ratio_ci_high: 1.01, p: 0.05 }],
        p_interaction: 0.8, interaction_note: 'likelihood-ratio test on 1 df', n_used: 598,
      }],
    }))
    const user = userEvent.setup()
    render(<SubgroupPanel />)
    await run(user)

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const layout = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-layout') ?? '{}')
    expect(layout.xaxis.type).toBe('log')
    expect(layout.shapes[0].x0).toBe(1)
  })

  it('hands the rows to the Forest Builder, appending', async () => {
    installSession(session())
    mockRun()
    const user = userEvent.setup()
    render(<SubgroupPanel />)
    await run(user)

    await waitFor(() => expect(screen.getByRole('button', { name: '→ Forest Builder' })).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: '→ Forest Builder' }))

    const s = useStore.getState()
    expect(s.forestHandoffAppend).toBe(true)
    expect(s.forestHandoff?.map((r) => r.label)).toEqual(['F', 'M', 'Overall'])
    expect(s.forestHandoff?.[0].est).toBeCloseTo(-5.759, 10)
    expect(s.forestHandoff?.[0].ci_low).toBeCloseTo(-7.759, 10)
    expect(s.forestHandoff?.[0].ci_high).toBeCloseTo(-3.759, 10)
    expect(s.activeTab).toBe('visual')
  })

  it('shows the backend error rather than an empty table', async () => {
    installSession(session())
    server.use(
      http.post('/api/subgroup/analyze', () =>
        HttpResponse.json({ detail: 'The exposure cannot also be a subgroup variable' }, { status: 400 }),
      ),
    )
    const user = userEvent.setup()
    render(<SubgroupPanel />)
    await run(user)

    await waitFor(() => expect(screen.getByText(/cannot also be a subgroup/)).toBeInTheDocument())
  })
})
