import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import ChartsPanel from './ChartsPanel'

afterEach(() => clearSession())

describe('ChartsPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<ChartsPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows chart type radio buttons and defaults to histogram', () => {
    installSession()
    render(<ChartsPanel />)
    expect(screen.getByRole('radio', { name: /histogram/i })).toBeChecked()
    expect(screen.getByRole('radio', { name: /scatter/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /boxplot/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /violin/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /^bar$/i })).toBeInTheDocument()
  })

  it('runs histogram generation and renders the plot on success', async () => {
    installSession()
    server.use(
      http.post('/api/charts/histogram', () =>
        HttpResponse.json({
          type: 'histogram',
          x: 'AGE',
          bins: [
            { x0: 40, x1: 50, count: 1 },
            { x0: 50, x1: 60, count: 2 },
          ],
          kde: [
            { x: 45, y: 0.01 },
            { x: 55, y: 0.02 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)

    const runButton = screen.getByRole('button', { name: /generate chart/i })
    await user.click(runButton)

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    // Custom labels panel appears once plot data is present
    expect(screen.getByText('Custom Labels')).toBeInTheDocument()
  })

  it('runs a scatter chart with x/y selection', async () => {
    installSession()
    server.use(
      http.post('/api/charts/scatter', () =>
        HttpResponse.json({
          type: 'scatter',
          x: 'AGE',
          y: 'LDL',
          points: [
            { AGE: 55, LDL: 120 },
            { AGE: 62, LDL: 140 },
          ],
          regression: { line_x: [55, 62], line_y: [120, 140], r2: 0.8 },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)

    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
  })

  it('sends the agreement options with a scatter request', async () => {
    installSession()
    let sent: Record<string, unknown> = {}
    server.use(
      http.post('/api/charts/scatter', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          type: 'scatter', x: 'AGE', y: 'LDL',
          points: [{ AGE: 55, LDL: 120 }, { AGE: 62, LDL: 140 }],
          regression: { line_x: [55, 62], line_y: [120, 140], r2: 0.8, space: 'log10-log10' },
          log_x: true, log_y: true,
          identity: { line_x: [55, 140], line_y: [55, 140], n_below: 2, n_above: 0 },
          warnings: [],
        })
      }),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    await user.click(screen.getByRole('checkbox', { name: /log x axis/i }))
    await user.click(screen.getByRole('checkbox', { name: /log y axis/i }))
    await user.click(screen.getByRole('checkbox', { name: /y = x reference line/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(sent.log_x).toBe(true)
    expect(sent.log_y).toBe(true)
    expect(sent.identity_line).toBe(true)
  })

  it('surfaces the warning when a log axis drops points', async () => {
    installSession()
    server.use(
      http.post('/api/charts/scatter', () =>
        HttpResponse.json({
          type: 'scatter', x: 'AGE', y: 'LDL',
          points: [{ AGE: 55, LDL: 120 }, { AGE: 62, LDL: 140 }],
          regression: { line_x: [55, 62], line_y: [120, 140], r2: 0.8 },
          log_x: true, log_y: false, identity: {},
          warnings: [{ type: 'log_axis_nonpositive', n_dropped: 3, message: '3 of 5 points had a zero or negative value on AGE and cannot be placed on a log axis. They are omitted.' }],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByText(/3 of 5 points/)).toBeInTheDocument())
  })

  it('renders a dumbbell chart and reports the largest gap', async () => {
    installSession()
    server.use(
      http.post('/api/charts/dumbbell', () =>
        HttpResponse.json({
          type: 'dumbbell', category: 'variable', start: 'implied', end: 'printed', group: null, sort: 'gap',
          rows: [
            { category: 'Uric acid', start: 0.4, end: 1.31, gap: 0.91, group: null },
            { category: 'Age', start: 0.36, end: 0.39, gap: 0.03, group: null },
          ],
          summary: {
            n: 2, mean_gap: 0.47, median_abs_gap: 0.47, max_abs_gap: 0.91,
            largest_gap_category: 'Uric acid', n_end_above_start: 2, n_end_below_start: 0,
          },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /dumbbell/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(screen.getByText(/largest 0.910 at/)).toBeInTheDocument()
    expect(screen.getByText('Uric acid')).toBeInTheDocument()
  })

  it('refuses a dumbbell whose two value columns are the same', async () => {
    installSession()
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /dumbbell/i }))

    const selects = screen.getAllByRole('combobox')
    const openSel = selects.find((s) => s.previousElementSibling?.textContent?.match(/open marker/i))
    const filledSel = selects.find((s) => s.previousElementSibling?.textContent?.match(/filled marker/i))
    expect(openSel && filledSel).toBeTruthy()
    await user.selectOptions(filledSel!, (openSel as HTMLSelectElement).value)
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    expect(await screen.findByText(/two value columns must differ/i)).toBeInTheDocument()
  })

  const boxplotResponse = {
    type: 'boxplot',
    x: 'AGE',
    color: 'GROUP',
    groups: [
      { group: 'M', values: [10, 12, 14, 16], row_indices: [0, 1, 2, 3] },
      { group: 'F', values: [20, 22, 24, 26], row_indices: [4, 5, 6, 7] },
    ],
  }

  const compareResponse = {
    type: 'compare_means', y: 'AGE', group: 'GROUP',
    levels: ['M', 'F'], n_per_group: { M: 4, F: 4 },
    test: 'Welch t-test', test_selected_by: 'auto: all groups passed Shapiro-Wilk',
    p_adjust: 'holm', p_shown_is_adjusted: true,
    comparisons: [{
      group1: 'M', group2: 'F', p: 0.0004, p_adj: 0.0004, p_shown: 0.0004,
      stars: '***', label: '***', x1: 0, x2: 1, span: 1, level: 0,
    }],
    omnibus: {},
  }

  async function drawRaincloud() {
    installSession()
    server.use(http.post('/api/charts/boxplot', () => HttpResponse.json(boxplotResponse)))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /raincloud/i }))
    const colorSelect = screen.getAllByRole('combobox').find(
      (s) => s.previousElementSibling?.textContent?.match(/color \/ group/i),
    )
    await user.selectOptions(colorSelect!, 'GROUP')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    return { user, traces: JSON.parse(screen.getByTestId('plotly-mock').dataset.plotly!) }
  }

  it('draws a raincloud as a half violin, a box and every raw point', async () => {
    const { traces } = await drawRaincloud()
    expect(traces).toHaveLength(2)
    for (const t of traces) {
      expect(t.type).toBe('violin')
      expect(t.side).toBe('positive')      // density on one side only
      expect(t.points).toBe('all')          // the cloud is the whole point
      expect(t.pointpos).toBeLessThan(0)    // scattered on the other side
      expect(t.box.visible).toBe(true)      // box on the centre line
    }
  })

  it('clips the raincloud density to the observed range', async () => {
    const { traces } = await drawRaincloud()
    // Without this the kernel runs past the smallest observation, which on a
    // strictly positive measure draws density below zero.
    expect(traces.every((t: { spanmode: string }) => t.spanmode === 'hard')).toBe(true)
  })

  it('hides the point toggle for a raincloud, which is made of points', async () => {
    const { user } = await drawRaincloud()
    expect(screen.queryByRole('checkbox', { name: /show every point/i })).not.toBeInTheDocument()
    await user.click(screen.getByRole('radio', { name: /boxplot/i }))
    expect(screen.getByRole('checkbox', { name: /show every point/i })).toBeInTheDocument()
  })

  it('draws significance brackets and states the test and adjustment', async () => {
    installSession()
    server.use(
      http.post('/api/charts/boxplot', () => HttpResponse.json(boxplotResponse)),
      http.post('/api/charts/compare_means', () => HttpResponse.json(compareResponse)),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /boxplot/i }))
    await user.click(screen.getByRole('checkbox', { name: /significance brackets/i }))

    const colorSelect = screen.getAllByRole('combobox').find(
      (s) => s.previousElementSibling?.textContent?.match(/color \/ group/i),
    )
    await user.selectOptions(colorSelect!, 'GROUP')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    // The test name also appears as a dropdown option, so match the legend line.
    expect(screen.getByText(/auto: all groups passed Shapiro-Wilk/)).toBeInTheDocument()
    expect(screen.getByText(/adjusted for 1 comparisons \(holm\)/)).toBeInTheDocument()
    expect(screen.getByText(/\*\*\*\* ≤ 0.0001/)).toBeInTheDocument()
  })

  it('says so when the shown p-values are unadjusted', async () => {
    installSession()
    server.use(
      http.post('/api/charts/boxplot', () => HttpResponse.json(boxplotResponse)),
      http.post('/api/charts/compare_means', () =>
        HttpResponse.json({ ...compareResponse, p_adjust: 'none', p_shown_is_adjusted: false }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /boxplot/i }))
    await user.click(screen.getByRole('checkbox', { name: /significance brackets/i }))
    const colorSelect = screen.getAllByRole('combobox').find(
      (s) => s.previousElementSibling?.textContent?.match(/color \/ group/i),
    )
    await user.selectOptions(colorSelect!, 'GROUP')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByText(/unadjusted across 1 comparisons/)).toBeInTheDocument())
  })

  it('keeps the chart when only the comparison call fails', async () => {
    installSession()
    server.use(
      http.post('/api/charts/boxplot', () => HttpResponse.json(boxplotResponse)),
      http.post('/api/charts/compare_means', () =>
        HttpResponse.json({ detail: 'Need at least two levels' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /boxplot/i }))
    await user.click(screen.getByRole('checkbox', { name: /significance brackets/i }))
    const colorSelect = screen.getAllByRole('combobox').find(
      (s) => s.previousElementSibling?.textContent?.match(/color \/ group/i),
    )
    await user.selectOptions(colorSelect!, 'GROUP')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(screen.getByText(/Chart drawn, but the comparisons failed/)).toBeInTheDocument()
  })

  it('renders an error plot and names the whisker', async () => {
    installSession()
    server.use(
      http.post('/api/charts/errorplot', () =>
        HttpResponse.json({
          type: 'errorplot', y: 'AGE', group: null, centre: 'mean', spread: 'ci',
          ci_level: 0.95, spread_label: 'mean with 95% CI',
          rows: [{ group: 'All', n: 8, centre: 18, lower: 14, upper: 22 }],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /error plot/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
  })

  it('renders an ECDF and reports the KS statistic', async () => {
    installSession()
    server.use(
      http.post('/api/charts/ecdf', () =>
        HttpResponse.json({
          type: 'ecdf', x: 'AGE', group: 'GROUP',
          curves: [
            { group: 'M', n: 2, x: [10, 12], y: [0.5, 1] },
            { group: 'F', n: 2, x: [20, 22], y: [0.5, 1] },
          ],
          ks: { test: 'Two-sample Kolmogorov-Smirnov', statistic: 1.0, p: 0.033, note: 'D is the largest vertical distance between the two curves.' },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /ecdf/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
  })

  it('renders a pie chart', async () => {
    installSession()
    server.use(
      http.post('/api/charts/pie', () =>
        HttpResponse.json({
          type: 'pie', category: 'GROUP', value: null, measure: 'count', total: 8,
          slices: [{ label: 'M', value: 5, percent: 62.5 }, { label: 'F', value: 3, percent: 37.5 }],
          n_folded_into_other: 0,
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /pie|donut/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
  })

  it('renders a balloon plot and reports the chi-square test', async () => {
    installSession()
    server.use(
      http.post('/api/charts/balloon', () =>
        HttpResponse.json({
          type: 'balloon', row: 'GROUP', col: 'OUTCOME',
          rows: ['M', 'F'], cols: ['Alive', 'Dead'],
          cells: [
            { row: 'M', col: 'Alive', count: 5, expected: 4.0, residual: 0.5 },
            { row: 'M', col: 'Dead', count: 1, expected: 2.0, residual: -0.7 },
            { row: 'F', col: 'Alive', count: 3, expected: 4.0, residual: -0.5 },
            { row: 'F', col: 'Dead', count: 3, expected: 2.0, residual: 0.7 },
          ],
          n: 12, chi2: 1.5, df: 1, p: 0.2207, warnings: [],
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /balloon/i }))
    const colSelect = screen.getAllByRole('combobox').find(
      (s) => s.previousElementSibling?.textContent?.match(/^columns$/i),
    )
    await user.selectOptions(colSelect!, 'GROUP')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(screen.getByText(/χ² = 1.50, df = 1/)).toBeInTheDocument()
  })

  it('refuses a balloon plot with the same variable on both axes', async () => {
    installSession()
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /balloon/i }))
    const selects = screen.getAllByRole('combobox')
    const rowSel = selects.find((s) => s.previousElementSibling?.textContent?.match(/^rows$/i))
    const colSel = selects.find((s) => s.previousElementSibling?.textContent?.match(/^columns$/i))
    await user.selectOptions(colSel!, (rowSel as HTMLSelectElement).value)
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    expect(await screen.findByText(/must differ/i)).toBeInTheDocument()
  })

  it('renders a facet grid and surfaces the truncation warning', async () => {
    installSession()
    server.use(
      http.post('/api/charts/facet', () =>
        HttpResponse.json({
          type: 'facet', kind: 'boxplot', x: 'AGE', y: null, facet: 'GROUP', color: null,
          panels: [
            { panel: 'A', n: 4, groups: [{ group: 'All', values: [1, 2, 3, 4] }] },
            { panel: 'B', n: 4, groups: [{ group: 'All', values: [5, 6, 7, 8] }] },
          ],
          shared_range: { x: [1, 8] },
          warnings: [{ type: 'panels_truncated', n_dropped: 3, message: "'GROUP' has 3 more levels than the 2-panel limit; those panels are not drawn." }],
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /facet grid/i }))
    const facetSelect = screen.getAllByRole('combobox').find(
      (s) => s.previousElementSibling?.textContent?.match(/split into panels/i),
    )
    await user.selectOptions(facetSelect!, 'GROUP')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(screen.getByText(/3 more levels than the 2-panel limit/)).toBeInTheDocument()
  })

  it('prints the summary table with missing counts', async () => {
    installSession()
    server.use(
      http.post('/api/charts/boxplot', () => HttpResponse.json(boxplotResponse)),
      http.post('/api/charts/summary_stats', () =>
        HttpResponse.json({
          type: 'summary_stats', y: 'AGE', group: null,
          rows: [{ group: 'All', n: 7, n_missing: 2, mean: 18, sd: 6, median: 17, q1: 13, q3: 23, iqr: 10, min: 10, max: 26 }],
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /boxplot/i }))
    await user.click(screen.getByRole('checkbox', { name: /summary table below/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByText('Summary')).toBeInTheDocument())
    expect(screen.getByText(/\+2 missing/)).toBeInTheDocument()
    expect(screen.getByText(/18.00 ± 6.00/)).toBeInTheDocument()
  })

  it('sends the ellipse, marginal and shape options with a scatter', async () => {
    installSession()
    let sent: Record<string, unknown> = {}
    server.use(
      http.post('/api/charts/scatter', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          type: 'scatter', x: 'AGE', y: 'LDL',
          points: [{ AGE: 55, LDL: 120, GROUP: 'a' }, { AGE: 62, LDL: 140, GROUP: 'b' }],
          regression: { line_x: [55, 62], line_y: [120, 140], r2: 0.8 },
          log_x: false, log_y: false, identity: {}, shape: 'GROUP',
          ellipses: [{ group: 'All', n: 2, x: [1, 2, 3], y: [1, 2, 3] }],
          ellipse_level: 0.95, marginal: {}, warnings: [],
        })
      }),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    await user.click(screen.getByRole('checkbox', { name: /confidence ellipse/i }))
    await user.click(screen.getByRole('checkbox', { name: /marginal histograms/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(sent.ellipse).toBe(true)
    expect(sent.marginal).toBe(true)
  })

  it('shows the error message from the backend on failure', async () => {
    installSession()
    server.use(
      http.post('/api/charts/histogram', () =>
        HttpResponse.json({ detail: 'Column not found' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByText('Column not found')).toBeInTheDocument())
  })
})
