import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
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

  it('prints the count on every non-empty bar, not only on hover', async () => {
    // A histogram is read for "how many are in this category", and a printed
    // figure has no hover at all.
    installSession()
    server.use(
      http.post('/api/charts/histogram', () =>
        HttpResponse.json({
          type: 'histogram', x: 'AGE',
          bins: [
            { x0: 0, x1: 1, count: 386 },
            { x0: 1, x1: 2, count: 0 },
            { x0: 2, x1: 3, count: 61 },
          ],
          kde: [{ x: 0.5, y: 0.4 }, { x: 2.5, y: 0.1 }],
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const traces = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-plotly') ?? '[]')
    const bar = traces.find((t: { type?: string }) => t.type === 'bar')
    // The empty bin prints nothing: a row of zeroes along the axis is noise.
    expect(bar.text).toEqual(['386', '', '61'])
    // Above the bar, and not clipped off the top by the tallest one.
    expect(bar.textposition).toBe('outside')
    expect(bar.cliponaxis).toBe(false)
  })

  it('drops the bar labels once the bins are too dense to read', async () => {
    installSession()
    server.use(
      http.post('/api/charts/histogram', () =>
        HttpResponse.json({
          type: 'histogram', x: 'AGE',
          bins: Array.from({ length: 40 }, (_, i) => ({ x0: i, x1: i + 1, count: i + 1 })),
          kde: [{ x: 1, y: 0.1 }],
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const traces = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-plotly') ?? '[]')
    expect(traces.find((t: { type?: string }) => t.type === 'bar').text).toBeUndefined()
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

  async function drawStrip(opts: { horizontal?: boolean; log?: boolean } = {}) {
    installSession()
    server.use(http.post('/api/charts/boxplot', () => HttpResponse.json(boxplotResponse)))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /^strip \(points/i }))
    const colorSelect = screen.getAllByRole('combobox').find(
      (s) => s.previousElementSibling?.textContent?.match(/color \/ group/i),
    )
    await user.selectOptions(colorSelect!, 'GROUP')
    if (opts.horizontal) await user.click(screen.getByRole('checkbox', { name: /horizontal/i }))
    if (opts.log) await user.click(screen.getByRole('checkbox', { name: /log scale/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const el = screen.getByTestId('plotly-mock')
    return { user, traces: JSON.parse(el.dataset.plotly!), layout: JSON.parse(el.dataset.layout!) }
  }

  it('splits the bars when a Color / Group column is chosen', async () => {
    // Reported: choosing Color / Group changed nothing. The request carried
    // the column and the handler never read it, so the control was inert and
    // silent — which reads as a setting that WAS applied.
    installSession()
    server.use(http.post('/api/charts/bar', () => HttpResponse.json({
      type: 'bar', x: 'TERTILE', y: '% Malign', y_mode: 'percentage', color: 'GROUP',
      series: [
        { group: 'M', data: [{ label: '1', value: 40, n: 10, k: 4 }] },
        { group: 'F', data: [{ label: '1', value: 60, n: 10, k: 6 }] },
      ],
    })))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /^bar$/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())

    const traces = JSON.parse(screen.getByTestId('plotly-mock').dataset.plotly!)
    expect(traces).toHaveLength(2)
    expect(traces.map((t: { name: string }) => t.name)).toEqual(['M', 'F'])
    expect(traces[0].y).toEqual([40])
    expect(traces[1].y).toEqual([60])
    // Each split keeps its own denominator.
    expect(traces[1].customdata).toEqual([[6, 10]])
  })

  it('offers Color / Group only on charts that read it', async () => {
    // A pie renders identically whatever is chosen here; showing the control
    // implies it did something.
    installSession()
    const user = userEvent.setup()
    render(<ChartsPanel />)
    const colourSelect = () => screen.queryAllByRole('combobox').find(
      (s) => s.previousElementSibling?.textContent?.match(/color \/ group/i),
    )
    await user.click(screen.getByRole('radio', { name: /boxplot/i }))
    expect(colourSelect()).toBeTruthy()
    for (const inert of [/^pie$/i, /balloon/i, /sankey/i, /stacked bar/i]) {
      await user.click(screen.getByRole('radio', { name: inert }))
      expect(colourSelect()).toBeUndefined()
    }
  })

  it('draws a bar chart as the percentage of each group, labelled', async () => {
    // "What fraction of this tertile was malignant" is the question a
    // risk-factor figure asks. Reading it off a gridline is a guess.
    installSession()
    server.use(http.post('/api/charts/bar', () => HttpResponse.json({
      type: 'bar', x: 'TERTILE', y: '% Malign', y_mode: 'percentage',
      data: [
        { label: '1', value: 35.7, n: 42, k: 15 },
        { label: '2', value: 40.5, n: 42, k: 17 },
      ],
    })))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /^bar$/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())

    const [trace] = JSON.parse(screen.getByTestId('plotly-mock').dataset.plotly!)
    expect(trace.y).toEqual([35.7, 40.5])
    expect(trace.text).toEqual(['36%', '41%'])
    expect(trace.textposition).toBe('outside')
    // n and k travel with the percentage: 36% of 8 is not 36% of 800.
    expect(trace.customdata).toEqual([[15, 42], [17, 42]])
  })

  it('leaves a mean bar chart unlabelled by percent', async () => {
    installSession()
    server.use(http.post('/api/charts/bar', () => HttpResponse.json({
      type: 'bar', x: 'GROUP', y: 'AGE', y_mode: 'mean',
      data: [{ label: 'M', value: 55 }, { label: 'F', value: 62 }],
    })))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /^bar$/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())

    const [trace] = JSON.parse(screen.getByTestId('plotly-mock').dataset.plotly!)
    expect(trace.text).toEqual(['55', '62'])
    expect(trace.customdata).toBeUndefined()
  })

  it('draws a strip chart as points plus a median rule and no box', async () => {
    const { traces } = await drawStrip()
    // One transparent box per group carrying the points, then a median trace.
    const pointTraces = traces.filter((t: { type: string }) => t.type === 'box')
    expect(pointTraces).toHaveLength(2)
    for (const t of pointTraces) {
      expect(t.boxpoints).toBe('all')
      // A visible box or median line would make this a box plot.
      expect(t.fillcolor).toBe('rgba(0,0,0,0)')
      expect(t.line.color).toBe('rgba(0,0,0,0)')
    }
    const median = traces.find((t: { name: string }) => t.name === 'Median')
    expect(median.mode).toBe('markers')
    // 12, 14 -> 13 and 22, 24 -> 23
    expect(median.y).toEqual([13, 23])
  })

  it('puts the groups down the side when asked', async () => {
    const { traces } = await drawStrip({ horizontal: true })
    const pointTraces = traces.filter((t: { type: string }) => t.type === 'box')
    // Values on x, groups implied by the trace name — the transpose of the
    // default. A long histology name will not fit under a tick.
    expect(pointTraces[0].x).toEqual([10, 12, 14, 16])
    expect(pointTraces[0].y).toBeUndefined()
    const median = traces.find((t: { name: string }) => t.name === 'Median')
    expect(median.x).toEqual([13, 23])
    expect(median.y).toEqual(['M', 'F'])
  })

  it('logs the value axis and leaves the category axis alone', async () => {
    const { layout } = await drawStrip({ log: true })
    expect(layout.yaxis.type).toBe('log')
    // The category axis must never be logged — it carries names, not numbers.
    expect(layout.xaxis.type).toBe('category')
  })

  it('moves the log scale with the orientation', async () => {
    // Which axis carries the values is the orientation's business; logging the
    // wrong one would either do nothing or corrupt the category order.
    const { layout } = await drawStrip({ horizontal: true, log: true })
    expect(layout.xaxis.type).toBe('log')
    expect(layout.yaxis.type).toBe('category')
  })

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

  it('marks each group mean with a diamond when asked', async () => {
    installSession()
    server.use(http.post('/api/charts/boxplot', () => HttpResponse.json(boxplotResponse)))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /boxplot/i }))
    await user.click(screen.getByRole('checkbox', { name: /mark the mean/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const traces = JSON.parse(screen.getByTestId('plotly-mock').dataset.plotly!)
    const mean = traces.find((t: { name?: string }) => t.name === 'Mean')
    expect(mean.marker.symbol).toBe('diamond')
    // The box draws the median (13 and 23); the diamond has to be the MEAN.
    expect(mean.y).toEqual([13, 23])
  })

  it('has no mean marker unless it is asked for', async () => {
    installSession()
    server.use(http.post('/api/charts/boxplot', () => HttpResponse.json(boxplotResponse)))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /boxplot/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const traces = JSON.parse(screen.getByTestId('plotly-mock').dataset.plotly!)
    expect(traces.some((t: { name?: string }) => t.name === 'Mean')).toBe(false)
  })

  const scatterWithBand = {
    type: 'scatter', x: 'AGE', y: 'BMI',
    points: [{ AGE: 40, BMI: 22 }, { AGE: 50, BMI: 25 }, { AGE: 60, BMI: 27 }],
    regression: {
      slope: 0.25, intercept: 12, r: 0.98, r2: 0.96, p: 0.0004, se: 0.05, n: 3,
      line_x: [40, 60], line_y: [22, 27],
      band: { x: [40, 50, 60], lo: [21, 24.5, 26], hi: [23, 25.5, 28], level: 0.95 },
      spearman: { rho: 1, p: 0.0001 },
    },
  }

  async function drawScatterWithBand() {
    installSession()
    server.use(http.post('/api/charts/scatter', () => HttpResponse.json(scatterWithBand)))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    return JSON.parse(screen.getByTestId('plotly-mock').dataset.plotly!)
  }

  it('fills the confidence band under the fitted line', async () => {
    const traces = await drawScatterWithBand()
    const filled = traces.find((t: { fill?: string }) => t.fill === 'tonexty')
    expect(filled.y).toEqual([23, 25.5, 28])
    // The invisible lower edge must come first, or the fill has nothing to
    // reach down to, and both must precede the markers so the band sits under.
    expect(traces[0].y).toEqual([21, 24.5, 26])
    expect(traces.indexOf(filled)).toBeLessThan(
      traces.findIndex((t: { mode?: string }) => t.mode === 'markers'))
  })

  it('reports Pearson and Spearman under the scatter', async () => {
    await drawScatterWithBand()
    expect(screen.getByText(/Pearson r = 0\.980/)).toBeInTheDocument()
    expect(screen.getByText(/Spearman ρ = 1\.000/)).toBeInTheDocument()
    expect(screen.getByText(/n = 3/)).toBeInTheDocument()
  })

  it('gives one panel per variable, each on its own scale', async () => {
    installSession()
    server.use(http.post('/api/charts/facet', async ({ request }) => {
      const body = await request.json() as { variables?: string[]; facet?: string }
      expect(body.variables).toEqual(['AGE', 'LDL'])
      expect(body.facet).toBeUndefined()
      return HttpResponse.json({
        type: 'facet', kind: 'boxplot', facet_by: 'variable', color: 'GROUP',
        panels: [
          { panel: 'AGE', n: 4, range: [40, 60], groups: [{ group: 'M', values: [40, 60] }] },
          { panel: 'LDL', n: 4, range: [22, 27], groups: [{ group: 'M', values: [22, 27] }] },
        ],
        shared_range: {}, warnings: [],
      })
    }))

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /facet grid/i }))
    const mode = screen.getAllByRole('combobox').find(
      (el) => el.previousElementSibling?.textContent?.match(/one panel per/i))
    await user.selectOptions(mode!, 'variable')
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('checkbox', { name: 'LDL' }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const layout = JSON.parse(screen.getByTestId('plotly-mock').dataset.layout!)
    // No shared range: milliseconds and a unitless index do not share an axis.
    expect(layout.yaxis.range).toBeUndefined()
    expect(layout.yaxis2.range).toBeUndefined()
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

  it('resolves a pie slice code to its value label', async () => {
    // Reported: a histology column labelled 0 = Benign, 1 = Papiller ...
    // still drew 0.0, 1.0, 7.0, 8.0 in the pie, because the backend
    // stringifies a float64 code as "0.0" while the label dialog writes "0".
    // Every other chart in the panel already resolved through this path;
    // the pie was the one built straight from the raw slice label.
    installSession(makeSession({
      columns: [
        { name: 'GROUP', dtype: 'object', kind: 'categorical', value_labels: { '0': 'Benign', '1': 'Malign' } },
      ],
    }))
    server.use(
      http.post('/api/charts/pie', () =>
        HttpResponse.json({
          type: 'pie', category: 'GROUP', value: null, measure: 'count', total: 8,
          slices: [{ label: '0', value: 5, percent: 62.5 }, { label: '1', value: 3, percent: 37.5 }],
          n_folded_into_other: 0,
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /pie|donut/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    const mock = await screen.findByTestId('plotly-mock')
    const data = JSON.parse(mock.getAttribute('data-plotly') ?? '[]')
    expect(data[0].labels).toEqual(['Benign', 'Malign'])
  })

  it('gives every pie slice a distinct colour past the palette length', async () => {
    // A donut with more categories than the six-colour palette used to repeat
    // colours, so the largest slice and a small one looked identical.
    installSession()
    const labels = Array.from({ length: 9 }, (_, i) => `L${i}`)
    server.use(
      http.post('/api/charts/pie', () =>
        HttpResponse.json({
          type: 'pie', category: 'GROUP', value: null, measure: 'count', total: 90,
          slices: labels.map((l) => ({ label: l, value: 10, percent: 100 / 9 })),
          n_folded_into_other: 0,
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /pie|donut/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    const mock = await screen.findByTestId('plotly-mock')
    const data = JSON.parse(mock.getAttribute('data-plotly') ?? '[]')
    const colors = data[0].marker.colors as string[]
    expect(colors).toHaveLength(9)
    expect(new Set(colors).size).toBe(9)
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

describe('ChartsPanel scatter trend lines', () => {
  const points = [{ AGE: 55, LDL: 120, SEX: 'F' }, { AGE: 62, LDL: 140, SEX: 'M' }, { AGE: 70, LDL: 150, SEX: 'F' }]

  it('draws one LOESS curve per group in place of the overall line, and no band', async () => {
    installSession()
    server.use(
      http.post('/api/charts/scatter', () =>
        HttpResponse.json({
          type: 'scatter', x: 'AGE', y: 'LDL', color: 'SEX', fit: 'loess', fit_per_group: true,
          points,
          regression: { method: 'loess', line_x: [55, 62, 70], line_y: [120, 140, 150], r2: 0.9, r: 0.95, p: 0.01, n: 3, span: 0.75, band: {} },
          regressions: [
            { group: 'F', n: 2, method: 'loess', line_x: [55, 70], line_y: [120, 150], r: 1, r2: 1, span: 0.75, band: {} },
            { group: 'M', n: 1, method: 'loess', line_x: [], line_y: [], r: null, r2: null, note: 'LOESS needs at least 4 points spread along x' },
          ],
        }),
      ),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())

    const traces = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-plotly') ?? '[]') as Array<{ name?: string; fill?: string }>
    const names = traces.map((t) => t.name ?? '')
    expect(names.some((n) => n.startsWith('F · LOESS'))).toBe(true)
    // The group that could not be fitted gets no line, and no overall line stands in for it.
    expect(names.some((n) => n.startsWith('M · LOESS'))).toBe(false)
    expect(names.filter((n) => /LOESS|Fit \(/.test(n))).toHaveLength(1)
    expect(traces.some((t) => t.fill === 'tonexty')).toBe(false)
    expect(screen.getByText(/carries no band/)).toBeInTheDocument()
    expect(screen.getByText(/M: LOESS needs at least 4 points/)).toBeInTheDocument()
  })

  it('sends the chosen method and span with the request', async () => {
    installSession()
    let sent: Record<string, unknown> = {}
    server.use(
      http.post('/api/charts/scatter', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          type: 'scatter', x: 'AGE', y: 'LDL', fit: 'none', points,
          regression: { method: 'none', line_x: [], line_y: [], r: 0.5, r2: 0.25, p: 0.3, n: 3 },
          regressions: [],
        })
      }),
    )
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    await user.selectOptions(screen.getByRole("combobox", { name: /trend line method/i }), 'none')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(sent.fit).toBe('none')
    expect(sent.loess_span).toBe(0.75)
    const traces = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-plotly') ?? '[]') as Array<{ name?: string }>
    expect(traces.filter((t) => /Fit \(|LOESS/.test(t.name ?? ''))).toHaveLength(0)
  })
})

describe('ChartsPanel grouped histogram', () => {
  const grouped = {
    type: 'histogram', x: 'AGE', color: 'SEX',
    bins: [{ x0: 0, x1: 10, count: 6 }, { x0: 10, x1: 20, count: 4 }],
    edges: [0, 10, 20], bin_width: 10,
    kde: [{ x: 5, y: 0.06 }, { x: 15, y: 0.04 }],
    groups: [
      { group: 'F', n: 4, counts: [3, 1], kde: [{ x: 5, y: 0.075 }, { x: 15, y: 0.025 }], values: [1, 2, 3, 12] },
      { group: 'M', n: 6, counts: [3, 3], kde: [{ x: 5, y: 0.05 }, { x: 15, y: 0.05 }], values: [4, 5, 6, 13, 14, 15] },
    ],
  }

  async function draw() {
    installSession()
    server.use(http.post('/api/charts/histogram', () => HttpResponse.json(grouped)))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    return user
  }

  it('draws one bar series per group on the shared edges, without printed counts', async () => {
    const user = await draw()
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const traces = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-plotly') ?? '[]') as Array<Record<string, unknown>>
    const bars = traces.filter((t) => t.type === 'bar')
    expect(bars.map((b) => b.name)).toEqual(['F', 'M'])
    expect(bars[0].x).toEqual([5, 15])
    expect(bars[0].y).toEqual([3, 1])
    expect(bars[0].text).toBeUndefined()
    // No rug until asked for.
    expect(traces.some((t) => String(t.name).endsWith('rug'))).toBe(false)
  })

  it('rescales counts to percent per group and adds a rug on request', async () => {
    const user = await draw()
    await user.selectOptions(screen.getByRole('combobox', { name: /histogram y axis/i }), 'percent')
    await user.click(screen.getByRole('checkbox', { name: /rug/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const traces = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-plotly') ?? '[]') as Array<Record<string, unknown>>
    const bars = traces.filter((t) => t.type === 'bar')
    // F: 3 of 4 and 1 of 4; M: 3 of 6 twice — each group over its own n.
    expect(bars[0].y).toEqual([75, 25])
    expect(bars[1].y).toEqual([50, 50])
    const rugs = traces.filter((t) => String(t.name).endsWith('rug'))
    expect(rugs).toHaveLength(2)
    expect(rugs[0].x).toEqual([1, 2, 3, 12])
  })

  it('draws only the density curves when asked, one per group, filled', async () => {
    const user = await draw()
    await user.selectOptions(screen.getByRole('combobox', { name: /histogram display/i }), 'density')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    const traces = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-plotly') ?? '[]') as Array<Record<string, unknown>>
    expect(traces.some((t) => t.type === 'bar')).toBe(false)
    const curves = traces.filter((t) => t.mode === 'lines')
    expect(curves.map((c) => c.name)).toEqual(['F KDE', 'M KDE'])
    expect(curves[0].fill).toBe('tozeroy')
  })
})

describe('ChartsPanel facet grid and continuous colour', () => {
  const facetBody = {
    type: 'facet', kind: 'boxplot', x: 'AGE', y: null, facet: 'GROUP', color: null,
    panels: [
      { panel: 'A', n: 4, groups: [{ group: 'All', values: [1, 2, 3, 4] }], range: [1, 4] },
      { panel: 'B', n: 4, groups: [{ group: 'All', values: [5, 6, 7, 8] }], range: [5, 8] },
    ],
    shared_range: {}, scales: 'free', ncol: 1, warnings: [],
  }

  it('leaves a freed axis to autoscale and lays the panels out in one column', async () => {
    installSession()
    let sent: Record<string, unknown> = {}
    server.use(http.post('/api/charts/facet', async ({ request }) => {
      sent = (await request.json()) as Record<string, unknown>
      return HttpResponse.json(facetBody)
    }))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /facet grid/i }))
    const facetSelect = screen.getAllByRole('combobox').find(
      (s) => s.previousElementSibling?.textContent?.match(/split into panels/i))
    await user.selectOptions(facetSelect!, 'GROUP')
    await user.selectOptions(screen.getByRole('combobox', { name: /panel scales/i }), 'free')
    await user.selectOptions(screen.getByRole('combobox', { name: /panel columns/i }), '1')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())

    expect(sent.scales).toBe('free')
    expect(sent.ncol).toBe(1)
    const layout = JSON.parse(screen.getByTestId('plotly-mock').dataset.layout!)
    // Freed: no range is imposed, so each panel autoscales with its padding.
    expect(layout.yaxis.range).toBeUndefined()
    expect(layout.yaxis2.range).toBeUndefined()
    // One column: the two panels sit on top of each other, not side by side.
    expect(layout.xaxis.domain).toEqual(layout.xaxis2.domain)
    expect(layout.yaxis.domain[0]).toBeGreaterThan(layout.yaxis2.domain[1])
  })

  it('colours scatter points by a numeric column with a colour bar', async () => {
    installSession()
    let sent: Record<string, unknown> = {}
    server.use(http.post('/api/charts/scatter', async ({ request }) => {
      sent = (await request.json()) as Record<string, unknown>
      return HttpResponse.json({
        type: 'scatter', x: 'AGE', y: 'LDL', gradient: 'DM', gradient_range: [0, 1],
        points: [{ AGE: 55, LDL: 120, DM: 0 }, { AGE: 62, LDL: 140, DM: 1 }],
        regression: { method: 'lm', line_x: [55, 62], line_y: [120, 140], r2: 0.8, r: 0.9, p: 0.1, n: 2 },
        regressions: [],
      })
    }))
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    await user.selectOptions(screen.getByRole('combobox', { name: /colour by value/i }), 'DM')
    await user.selectOptions(screen.getByRole('combobox', { name: /colour ramp/i }), 'Cividis')
    await user.click(screen.getByRole('button', { name: /generate chart/i }))
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())

    expect(sent.gradient).toBe('DM')
    const traces = JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-plotly') ?? '[]') as Array<Record<string, never>>
    const markers = traces.find((t) => t.mode === 'markers')!
    const marker = markers.marker as unknown as Record<string, unknown>
    expect(marker.color).toEqual([0, 1])
    expect(marker.colorscale).toBe('Cividis')
    expect(marker.showscale).toBe(true)
    expect(marker.cmin).toBe(0)
    expect(marker.cmax).toBe(1)
  })

  it('hides the colour ramp once a group column owns the colours', async () => {
    installSession()
    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    expect(screen.getByRole('combobox', { name: /colour by value/i })).toBeInTheDocument()
    const groupSelect = screen.getAllByRole('combobox').find(
      (el) => el.previousElementSibling?.textContent?.match(/^color \/ group$/i))
    await user.selectOptions(groupSelect!, 'GROUP')
    expect(screen.queryByRole('combobox', { name: /colour by value/i })).not.toBeInTheDocument()
  })
})
