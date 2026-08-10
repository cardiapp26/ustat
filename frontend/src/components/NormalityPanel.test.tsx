import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import NormalityPanel from './NormalityPanel'

afterEach(() => {
  clearSession()
  localStorage.clear()
})

const block = (over: Record<string, unknown> = {}) => ({
  label: 'All (pooled)',
  n: 58, n_missing: 2, n_total: 60, constant: false,
  mean: 12.9, sd: 3.4, median: 12.7, q1: 10.4, q3: 15.6, min: 6.1, max: 20.2,
  shape: { n: 58, skewness: 0.12, skew_se: 0.31, skew_z: 0.38, kurtosis: -0.4, kurt_se: 0.62, kurt_z: -0.65 },
  shape_flag: false,
  tests: [
    { id: 'shapiro', name: 'Shapiro-Wilk', stat: 0.978, p: 0.64, applicable: true, note: '' },
    { id: 'anderson', name: 'Anderson-Darling', stat: 0.271, p: 0.656, applicable: true, note: '' },
    { id: 'jarque_bera', name: 'Jarque-Bera', stat: null, p: null, applicable: false, note: 'asymptotic, needs n ≥ 30' },
  ],
  primary: { id: 'shapiro', name: 'Shapiro-Wilk', stat: 0.978, p: 0.64, applicable: true, note: '' },
  verdict: { code: 'normal', label: 'Consistent with normal', reason: 'The test does not reject normality.', notes: [] },
  qq: { theoretical: [-1.5, 0, 1.5], sample: [8, 12.7, 18], line: { slope: 3.4, intercept: 12.9 } },
  histogram: { bin_edges: [6, 10, 14, 18], counts: [10, 25, 23], curve_x: [6, 12, 18], curve_y: [4, 26, 5] },
  sentence: 'QT did not depart from a normal distribution (Shapiro-Wilk W = 0.978, p=0.640, n = 58).',
  ...over,
})

const payload = (over: Record<string, unknown> = {}) => ({
  alpha: 0.05,
  group_column: null,
  group_levels: [],
  variables: [{ variable: 'AGE', overall: block(), groups: [] }],
  warnings: [],
  guidance: 'Assess normality from the test, the shape statistics and the Q-Q plot together.',
  ...over,
})

function mockRun(body: Record<string, unknown> = payload()) {
  const seen: { request?: Record<string, unknown> } = {}
  server.use(
    http.post('/api/stats/normality', async ({ request }) => {
      seen.request = (await request.json()) as Record<string, unknown>
      return HttpResponse.json(body)
    }),
  )
  return seen
}

describe('NormalityPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<NormalityPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('offers numeric columns to test and categorical columns to group by', () => {
    installSession()
    render(<NormalityPanel />)
    expect(screen.getByRole('checkbox', { name: 'AGE' })).toBeInTheDocument()
    expect(screen.queryByRole('checkbox', { name: 'GROUP' })).not.toBeInTheDocument()
    const groupSelect = screen.getByRole('combobox', { name: /group by/i })
    expect(within(groupSelect).getByRole('option', { name: 'GROUP' })).toBeInTheDocument()
    expect(within(groupSelect).queryByRole('option', { name: 'AGE' })).not.toBeInTheDocument()
  })

  it('cannot run with nothing selected', () => {
    installSession()
    render(<NormalityPanel />)
    expect(screen.getByRole('button', { name: /assess normality/i })).toBeDisabled()
  })

  it('renders the primary test, the shape statistics and the verdict', async () => {
    installSession()
    mockRun()
    const user = userEvent.setup()
    render(<NormalityPanel />)

    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /assess normality/i }))

    await waitFor(() => expect(screen.getAllByText('Shapiro-Wilk').length).toBeGreaterThan(0))
    expect(screen.getAllByText('0.978').length).toBeGreaterThan(0)
    expect(screen.getAllByText('0.640').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Consistent with normal').length).toBeGreaterThan(0)
    // Skewness with its z-score in brackets — the z is what the cutoffs apply to.
    expect(screen.getByText((_, el) => el?.textContent === '0.12 (0.4)')).toBeInTheDocument()
  })

  it('sends the grouping column and lists one row per group beside the pooled row', async () => {
    installSession()
    const seen = mockRun(payload({
      group_column: 'GROUP',
      group_levels: ['A', 'B'],
      variables: [{
        variable: 'AGE',
        overall: block({ verdict: { code: 'non_normal', label: 'Clear departure from normal', reason: 'x', notes: [] } }),
        groups: [
          block({ label: 'A', n: 29 }),
          block({ label: 'B', n: 29 }),
        ],
        group_summary: 'All groups consistent with normal',
      }],
    }))

    const user = userEvent.setup()
    render(<NormalityPanel />)
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.selectOptions(screen.getByRole('combobox', { name: /group by/i }), 'GROUP')
    await user.click(screen.getByRole('button', { name: /assess normality/i }))

    await waitFor(() => expect(seen.request).toBeTruthy())
    expect(seen.request).toMatchObject({ variables: ['AGE'], group_column: 'GROUP' })
    // The pooled sample is shown next to the groups, not instead of them: a
    // mixture of two separated groups can fail normality on that account alone.
    // One column carries both — the variable name on its pooled row, the group
    // label indented under it.
    const main = screen.getAllByRole('table')[0]
    const labels = within(main)
      .getAllByRole('row')
      .slice(1)
      .map((r) => (r as HTMLTableRowElement).cells[0].textContent)
    expect(labels).toEqual(['AGE', 'A', 'B'])
    expect(screen.getAllByTitle('AGE — All (pooled)').length).toBe(1)
    expect(screen.getAllByTitle('AGE — group A').length).toBe(1)
    expect(screen.getAllByText('Clear departure from normal').length).toBeGreaterThan(0)
  })

  it('shows a test that could not run with the reason instead of a blank p', async () => {
    installSession()
    mockRun()
    const user = userEvent.setup()
    render(<NormalityPanel />)
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /assess normality/i }))

    await waitFor(() => expect(screen.getByText('Jarque-Bera')).toBeInTheDocument())
    expect(screen.getByText(/asymptotic, needs n/)).toBeInTheDocument()
  })

  it('surfaces the small-sample and large-sample caveats the backend attaches', async () => {
    installSession()
    mockRun(payload({
      variables: [{
        variable: 'AGE',
        overall: block({
          n: 12,
          verdict: {
            code: 'normal', label: 'Consistent with normal', reason: 'ok',
            notes: ['n < 20: the test has little power here.'],
          },
        }),
        groups: [],
      }],
    }))
    const user = userEvent.setup()
    render(<NormalityPanel />)
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /assess normality/i }))

    await waitFor(() => expect(screen.getByText(/little power/)).toBeInTheDocument())
  })

  it('renders a paste-ready sentence for every row', async () => {
    installSession()
    mockRun()
    const user = userEvent.setup()
    render(<NormalityPanel />)
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /assess normality/i }))

    await waitFor(() =>
      expect(screen.getByText(/did not depart from a normal distribution/)).toBeInTheDocument())
  })

  it('draws the Q-Q reference line from the returned slope and intercept', async () => {
    installSession()
    mockRun()
    const user = userEvent.setup()
    render(<NormalityPanel />)
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /assess normality/i }))

    await waitFor(() => expect(screen.getAllByTestId('plotly-mock').length).toBe(2))
    const traces = JSON.parse(screen.getAllByTestId('plotly-mock')[0].getAttribute('data-plotly') ?? '[]')
    const line = traces.find((t: { name?: string }) => t.name === 'Normal reference')
    expect(line.x).toEqual([-1.5, 1.5])
    expect(line.y[0]).toBeCloseTo(3.4 * -1.5 + 12.9)
    expect(line.y[1]).toBeCloseTo(3.4 * 1.5 + 12.9)
  })

  it('shows the backend error rather than an empty result', async () => {
    installSession()
    server.use(
      http.post('/api/stats/normality', () =>
        HttpResponse.json({ detail: 'No valid variables selected' }, { status: 400 }),
      ),
    )
    const user = userEvent.setup()
    render(<NormalityPanel />)
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /assess normality/i }))

    await waitFor(() => expect(screen.getByText('No valid variables selected')).toBeInTheDocument())
  })
})
