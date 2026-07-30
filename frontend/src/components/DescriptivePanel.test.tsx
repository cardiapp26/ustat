import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse, delay } from 'msw'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import DescriptivePanel from './DescriptivePanel'

afterEach(() => {
  clearSession()
  vi.restoreAllMocks()
  localStorage.removeItem('uStat.descriptiveColumnListW')
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
})

const numericSummary = {
  type: 'numeric',
  histogram: [
    { bin_start: 40, bin_end: 50, count: 1 },
    { bin_start: 50, bin_end: 60, count: 1 },
    { bin_start: 60, bin_end: 70, count: 1 },
  ],
  raw_values: [55, 62, 48],
  outliers: [],
  normality_deviants: [],
  qq: [
    { x: -1, y: 48 },
    { x: 0, y: 55 },
    { x: 1, y: 62 },
  ],
  n: 3,
  missing: 0,
  display_decimals: 2,
  mean: 55,
  std: 7.02,
  median: 55,
  min: 48,
  max: 62,
  q1: 51.5,
  q3: 58.5,
  iqr: 7,
  whisker_low: 48,
  whisker_high: 62,
  skewness: 0.05,
  kurtosis: -1.2,
  normal: true,
  normality_label: 'Normal',
  normality_test: 'Shapiro-Wilk',
  normality_p: 0.842,
}

const categoricalSummary = {
  type: 'categorical',
  histogram: [],
  qq: [],
  categories: [
    { value: 'A', count: 2, pct: 66.7 },
    { value: 'B', count: 1, pct: 33.3 },
  ],
  n: 3,
  n_categories: 2,
  missing: 0,
}

function mockCommonEndpoints() {
  server.use(
    http.get('/api/stats/test-session/sparklines', () => HttpResponse.json({})),
    http.get('/api/stats/test-session/descriptive', () => HttpResponse.json({})),
  )
}

describe('DescriptivePanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<DescriptivePanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('resizes the column list in both directions and persists the width', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
    )

    const view = render(<DescriptivePanel />)

    let divider = await screen.findByRole('separator', { name: 'Resize column list' })
    expect(divider).toHaveAttribute('aria-valuenow', '320')

    fireEvent.pointerDown(divider, {
      button: 0, clientX: 320, isPrimary: true, pointerId: 1,
    })
    fireEvent.pointerMove(document, { clientX: 420, pointerId: 1 })
    fireEvent.pointerUp(document, { pointerId: 1 })
    expect(divider).toHaveAttribute('aria-valuenow', '420')
    expect(localStorage.getItem('uStat.descriptiveColumnListW')).toBe('420')

    view.unmount()
    render(<DescriptivePanel />)
    divider = await screen.findByRole('separator', { name: 'Resize column list' })
    expect(divider).toHaveAttribute('aria-valuenow', '420')

    fireEvent.pointerDown(divider, {
      button: 0, clientX: 420, isPrimary: true, pointerId: 2,
    })
    fireEvent.pointerMove(document, { clientX: 270, pointerId: 2 })
    fireEvent.pointerUp(document, { pointerId: 2 })
    expect(divider).toHaveAttribute('aria-valuenow', '270')
    expect(localStorage.getItem('uStat.descriptiveColumnListW')).toBe('270')

    fireEvent.doubleClick(divider)
    expect(divider).toHaveAttribute('aria-valuenow', '320')
    expect(localStorage.getItem('uStat.descriptiveColumnListW')).toBe('320')
  })

  it('cleans up drag state on pointer cancellation and unmount', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
    )

    document.body.style.cursor = 'crosshair'
    document.body.style.userSelect = 'text'
    const view = render(<DescriptivePanel />)
    const divider = await screen.findByRole('separator', { name: 'Resize column list' })

    fireEvent.pointerDown(divider, {
      button: 0, clientX: 320, isPrimary: true, pointerId: 3,
    })
    fireEvent.pointerMove(document, { clientX: 400, pointerId: 3 })
    expect(document.body.style.cursor).toBe('col-resize')
    expect(document.body.style.userSelect).toBe('none')

    fireEvent.pointerCancel(document, { pointerId: 3 })
    expect(document.body.style.cursor).toBe('crosshair')
    expect(document.body.style.userSelect).toBe('text')
    expect(divider).toHaveAttribute('aria-valuenow', '400')

    fireEvent.pointerMove(document, { clientX: 450, pointerId: 3 })
    expect(divider).toHaveAttribute('aria-valuenow', '400')

    fireEvent.pointerDown(divider, {
      button: 0, clientX: 400, isPrimary: true, pointerId: 4,
    })
    expect(document.body.style.cursor).toBe('col-resize')
    view.unmount()
    expect(document.body.style.cursor).toBe('crosshair')
    expect(document.body.style.userSelect).toBe('text')
  })

  it('ignores right clicks, non-primary pointers, and unrelated pointer IDs', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
    )

    render(<DescriptivePanel />)
    const divider = await screen.findByRole('separator', { name: 'Resize column list' })

    fireEvent.pointerDown(divider, {
      button: 2, clientX: 320, isPrimary: true, pointerId: 5,
    })
    fireEvent.pointerMove(document, { clientX: 420, pointerId: 5 })
    expect(divider).toHaveAttribute('aria-valuenow', '320')

    fireEvent.pointerDown(divider, {
      button: 0, clientX: 320, isPrimary: false, pointerId: 6,
    })
    fireEvent.pointerMove(document, { clientX: 420, pointerId: 6 })
    expect(divider).toHaveAttribute('aria-valuenow', '320')

    fireEvent.pointerDown(divider, {
      button: 0, clientX: 320, isPrimary: true, pointerId: 7,
    })
    fireEvent.pointerMove(document, { clientX: 500, pointerId: 8 })
    fireEvent.pointerUp(document, { pointerId: 8 })
    expect(divider).toHaveAttribute('aria-valuenow', '320')

    fireEvent.pointerMove(document, { clientX: 400, pointerId: 7 })
    expect(divider).toHaveAttribute('aria-valuenow', '400')
    fireEvent.pointerUp(document, { pointerId: 7 })
  })

  it('supports keyboard resizing and reserves space for the result pane', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
    )

    render(<DescriptivePanel />)
    const divider = await screen.findByRole('separator', { name: 'Resize column list' })
    expect(divider).toHaveAttribute('tabindex', '0')

    fireEvent.keyDown(divider, { key: 'ArrowRight' })
    expect(divider).toHaveAttribute('aria-valuenow', '336')
    fireEvent.keyDown(divider, { key: 'ArrowLeft' })
    expect(divider).toHaveAttribute('aria-valuenow', '320')
    fireEvent.keyDown(divider, { key: 'Home' })
    expect(divider).toHaveAttribute('aria-valuenow', '224')

    const originalInnerWidth = window.innerWidth
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 900 })
    fireEvent.resize(window)
    // 900 - 560 = 340, but the divider keeps at least 120 px of travel above
    // its minimum, so the reservation gives way to 224 + 120.
    expect(divider).toHaveAttribute('aria-valuemax', '344')
    fireEvent.keyDown(divider, { key: 'End' })
    expect(divider).toHaveAttribute('aria-valuenow', '344')
    fireEvent.keyDown(divider, { key: 'Enter' })
    expect(divider).toHaveAttribute('aria-valuenow', '320')
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      value: originalInnerWidth,
    })
  })

  it('leaves the divider movable on a window too narrow to reserve the result pane', async () => {
    // Below a 784 px window the reservation used to collapse the maximum onto
    // the minimum, so aria-valuemin and aria-valuemax were both 224 and the
    // divider could not move — at exactly the width where the column names
    // are too clipped to read.
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
    )
    const originalInnerWidth = window.innerWidth
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 744 })

    render(<DescriptivePanel />)
    const divider = await screen.findByRole('separator', { name: 'Resize column list' })
    fireEvent.resize(window)

    expect(divider.getAttribute('aria-valuemax')).not.toBe(divider.getAttribute('aria-valuemin'))
    fireEvent.keyDown(divider, { key: 'End' })
    expect(divider).toHaveAttribute('aria-valuenow', '344')

    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      value: originalInnerWidth,
    })
  })

  it('falls back to default width when localStorage reads are blocked', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
    )
    vi.spyOn(Storage.prototype, 'getItem').mockImplementationOnce(() => {
      throw new DOMException('Blocked', 'SecurityError')
    })

    render(<DescriptivePanel />)

    const divider = await screen.findByRole('separator', { name: 'Resize column list' })
    expect(divider).toHaveAttribute('aria-valuenow', '320')
  })

  it('renames, reorders, and deletes columns through the shared Data session', async () => {
    installSession()
    mockCommonEndpoints()
    let renameBody: Record<string, unknown> | null = null
    let reorderBody: Record<string, unknown> | null = null
    let deletedColumn: string | null = null
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
      http.post('/api/compute/test-session/rename', async ({ request }) => {
        renameBody = await request.json() as Record<string, unknown>
        return HttpResponse.json({ ok: true })
      }),
      http.post('/api/sessions/test-session/reorder_columns', async ({ request }) => {
        reorderBody = await request.json() as Record<string, unknown>
        return HttpResponse.json({ ok: true })
      }),
      http.delete('/api/compute/test-session/column/:column', ({ params }) => {
        deletedColumn = String(params.column)
        return HttpResponse.json({ ok: true })
      }),
    )
    vi.spyOn(window, 'confirm').mockReturnValue(true)

    const user = userEvent.setup()
    render(<DescriptivePanel />)
    await screen.findByTestId('summary-column-AGE')

    await user.click(screen.getByRole('button', { name: 'Rename AGE' }))
    const renameInput = screen.getByRole('textbox', { name: 'Rename AGE' })
    await user.clear(renameInput)
    await user.type(renameInput, 'AGE_YEARS{Enter}')

    await waitFor(() => {
      expect(useStore.getState().session?.columns[0].name).toBe('AGE_YEARS')
    })
    expect(renameBody).toEqual({ old_name: 'AGE', new_name: 'AGE_YEARS' })
    expect(useStore.getState().session?.preview[0]).toMatchObject({ AGE_YEARS: 55 })
    expect(useStore.getState().session?.preview[0]).not.toHaveProperty('AGE')
    expect(screen.getByTestId('summary-column-AGE_YEARS')).toBeInTheDocument()

    const source = screen.getByTestId('summary-column-AGE_YEARS')
    const target = screen.getByTestId('summary-column-GROUP')
    fireEvent.dragStart(source, { dataTransfer: { effectAllowed: 'move' } })
    fireEvent.dragOver(target)
    fireEvent.drop(target)

    await waitFor(() => {
      expect(useStore.getState().session?.columns.map((column) => column.name)).toEqual([
        'LDL', 'DM', 'GROUP', 'AGE_YEARS',
      ])
    })
    await waitFor(() => {
      expect(reorderBody).toEqual({
        columns: ['LDL', 'DM', 'GROUP', 'AGE_YEARS'],
      })
    })

    await user.click(screen.getByRole('button', { name: 'Delete LDL' }))
    await waitFor(() => expect(deletedColumn).toBe('LDL'))
    expect(window.confirm).toHaveBeenCalledWith(
      'Delete column "LDL"? You can undo this from the Data tab.',
    )
    expect(useStore.getState().session?.columns.map((column) => column.name)).toEqual([
      'DM', 'GROUP', 'AGE_YEARS',
    ])
    expect(useStore.getState().session?.preview[0]).not.toHaveProperty('LDL')
  })

  it('keeps the column when Summary deletion is cancelled', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
    )
    vi.spyOn(window, 'confirm').mockReturnValue(false)

    const user = userEvent.setup()
    render(<DescriptivePanel />)
    await screen.findByTestId('summary-column-LDL')
    await user.click(screen.getByRole('button', { name: 'Delete LDL' }))

    expect(useStore.getState().session?.columns.some((column) => column.name === 'LDL')).toBe(true)
  })

  it('does not apply a delayed rename to a newly opened session', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
      http.get('/api/stats/other-session/sparklines', () => HttpResponse.json({})),
      http.get('/api/stats/other-session/descriptive', () => HttpResponse.json({})),
      http.get('/api/stats/other-session/column_summary', () => HttpResponse.json(numericSummary)),
      http.post('/api/compute/test-session/rename', async () => {
        await delay(80)
        return HttpResponse.json({
          new_name: 'AGE_YEARS',
          case_filter: null,
        })
      }),
    )

    const user = userEvent.setup()
    render(<DescriptivePanel />)
    await screen.findByTestId('summary-column-AGE')
    await user.click(screen.getByRole('button', { name: 'Rename AGE' }))
    const input = screen.getByRole('textbox', { name: 'Rename AGE' })
    await user.clear(input)
    await user.type(input, 'AGE_YEARS{Enter}')

    useStore.getState().setSession(makeSession({ session_id: 'other-session' }))

    await waitFor(() => {
      expect(useStore.getState().session?.session_id).toBe('other-session')
      expect(useStore.getState().session?.columns[0].name).toBe('AGE')
    })
    await delay(100)
    expect(useStore.getState().session?.columns[0].name).toBe('AGE')
  })

  it('does not apply a delayed delete to a newly opened session', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
      http.get('/api/stats/other-session/sparklines', () => HttpResponse.json({})),
      http.get('/api/stats/other-session/descriptive', () => HttpResponse.json({})),
      http.get('/api/stats/other-session/column_summary', () => HttpResponse.json(numericSummary)),
      http.delete('/api/compute/test-session/column/:column', async () => {
        await delay(80)
        return HttpResponse.json({ deleted: 'LDL', case_filter: null })
      }),
    )
    vi.spyOn(window, 'confirm').mockReturnValue(true)

    const user = userEvent.setup()
    render(<DescriptivePanel />)
    await screen.findByTestId('summary-column-LDL')
    await user.click(screen.getByRole('button', { name: 'Delete LDL' }))

    useStore.getState().setSession(makeSession({ session_id: 'other-session' }))

    await delay(100)
    expect(useStore.getState().session?.session_id).toBe('other-session')
    expect(useStore.getState().session?.columns.some((column) => column.name === 'LDL')).toBe(true)
  })

  it('keeps a newer Summary selection when rename completes later', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', ({ request }) => {
        const column = new URL(request.url).searchParams.get('column')
        return HttpResponse.json(column === 'GROUP' ? categoricalSummary : numericSummary)
      }),
      http.post('/api/compute/test-session/rename', async () => {
        await delay(80)
        return HttpResponse.json({
          new_name: 'AGE_YEARS',
          case_filter: null,
        })
      }),
    )

    const user = userEvent.setup()
    render(<DescriptivePanel />)
    await screen.findByTestId('summary-column-AGE')
    await user.click(screen.getByRole('button', { name: 'Rename AGE' }))
    const input = screen.getByRole('textbox', { name: 'Rename AGE' })
    await user.clear(input)
    await user.type(input, 'AGE_YEARS{Enter}')
    await user.click(screen.getByTestId('summary-column-GROUP'))

    await waitFor(() => {
      expect(screen.getByText((_, element) =>
        element?.textContent === 'Categorical · n=3',
      )).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(useStore.getState().session?.columns[0].name).toBe('AGE_YEARS')
    })
    expect(screen.getByText((_, element) =>
      element?.textContent === 'Categorical · n=3',
    )).toBeInTheDocument()
  })

  it('updates mounted Scatter selectors after Summary rename', async () => {
    installSession()
    useStore.setState({
      panelCache: {
        descriptive: { view: 'scatter' },
        descriptive_numeric: {
          xCol: 'AGE',
          yCol: 'LDL',
          color: 'GROUP',
          shape: '',
        },
      },
    })
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
      http.post('/api/charts/scatter', () => HttpResponse.json({
        points: [],
        regression: {
          r: null,
          r2: null,
          p: null,
          slope: null,
          intercept: null,
          line_x: [],
          line_y: [],
        },
      })),
      http.post('/api/compute/test-session/rename', () => HttpResponse.json({
        new_name: 'AGE_YEARS',
        case_filter: null,
      })),
    )

    const user = userEvent.setup()
    render(<DescriptivePanel />)
    await screen.findByDisplayValue('AGE')
    await user.click(screen.getByRole('button', { name: 'Rename AGE' }))
    const input = screen.getByRole('textbox', { name: 'Rename AGE' })
    await user.clear(input)
    await user.type(input, 'AGE_YEARS{Enter}')

    await waitFor(() => {
      expect(screen.getByDisplayValue('AGE_YEARS')).toBeInTheDocument()
    })
    expect(useStore.getState().panelCache.descriptive_numeric).toMatchObject({
      xCol: 'AGE_YEARS',
      yCol: 'LDL',
      color: 'GROUP',
    })
  })

  it('refetches Scatter after undo restores a deleted selector', async () => {
    const original = makeSession()
    installSession(original)
    useStore.setState({
      panelCache: {
        descriptive: { view: 'scatter' },
        descriptive_numeric: {
          xCol: 'AGE',
          yCol: 'LDL',
          color: 'GROUP',
          shape: '',
        },
      },
    })
    mockCommonEndpoints()
    let scatterCalls = 0
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
      http.post('/api/charts/scatter', () => {
        scatterCalls += 1
        return HttpResponse.json({
          points: [],
          regression: {
            r: null,
            r2: null,
            p: null,
            slope: null,
            intercept: null,
            line_x: [],
            line_y: [],
          },
        })
      }),
      http.delete('/api/compute/test-session/column/:column', () =>
        HttpResponse.json({ deleted: 'LDL', case_filter: null })
      ),
      http.post('/api/sessions/test-session/undo', () => HttpResponse.json({
        rows: original.rows,
        columns: original.columns,
        preview: original.preview,
        case_filter: null,
        undo_depth: 0,
        redo_depth: 1,
      })),
    )
    vi.spyOn(window, 'confirm').mockReturnValue(true)

    const user = userEvent.setup()
    render(<DescriptivePanel />)
    await waitFor(() => expect(scatterCalls).toBe(1))
    expect(screen.getByDisplayValue('LDL')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Delete LDL' }))
    await waitFor(() => {
      expect(useStore.getState().session?.columns.some((column) => column.name === 'LDL')).toBe(false)
      expect(screen.queryByDisplayValue('LDL')).not.toBeInTheDocument()
    })

    await useStore.getState().undo()

    await waitFor(() => expect(scatterCalls).toBe(2))
    expect(screen.getByDisplayValue('LDL')).toBeInTheDocument()
  })

  it('numeric column: loads and displays summary stats, normality test, and n', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', ({ request }) => {
        const url = new URL(request.url)
        expect(url.searchParams.get('column')).toBe('AGE')
        return HttpResponse.json(numericSummary)
      }),
    )

    render(<DescriptivePanel />)

    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    // header row
    await waitFor(() =>
      expect(screen.getByText((_, el) => el?.textContent === 'Continuous · n=3')).toBeInTheDocument(),
    )
    expect(screen.getByText((_, el) => el?.textContent === '· Normal (p=0.842)')).toBeInTheDocument()

    // normality badge box
    expect(screen.getByText('Normal')).toBeInTheDocument()
    expect(screen.getByText((_, el) => el?.textContent === '(Shapiro-Wilk p = 0.842)')).toBeInTheDocument()

    // stats strip
    expect(screen.getByText('Mean')).toBeInTheDocument()
    expect(screen.getAllByText('55.00').length).toBeGreaterThan(0)

    // default chart tab is histogram → plotly mock present
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  })

  it('categorical column: loads and displays frequency table info', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', ({ request }) => {
        const url = new URL(request.url)
        const column = url.searchParams.get('column')
        if (column === 'GROUP') return HttpResponse.json(categoricalSummary)
        return HttpResponse.json(numericSummary)
      }),
    )

    const user = userEvent.setup()
    render(<DescriptivePanel />)

    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    await user.click(screen.getByText('GROUP'))

    await waitFor(() =>
      expect(screen.getByText((_, el) => el?.textContent === 'Categorical · n=3')).toBeInTheDocument(),
    )
    expect(screen.getByText('2 categories')).toBeInTheDocument()
    expect(
      screen.getAllByText(
        (_, el) => el?.textContent === '2 categories, n = 3. Report as n (%). Most frequent: A (66.7%).',
      ).length,
    ).toBeGreaterThan(0)
  })

  it('switches between chart tabs (Histogram -> Box Plot -> Q-Q Plot)', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
    )

    const user = userEvent.setup()
    render(<DescriptivePanel />)

    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())

    // Box Plot tab
    await user.click(screen.getByRole('button', { name: 'Box Plot' }))
    await waitFor(() => {
      const plots = screen.getAllByTestId('plotly-mock')
      const boxPlot = plots.find((p) => (p.getAttribute('data-plotly') ?? '').includes('"type":"box"'))
      expect(boxPlot).toBeTruthy()
    })

    // Q-Q Plot tab
    await user.click(screen.getByRole('button', { name: 'Q-Q Plot' }))
    await waitFor(() => {
      const plots = screen.getAllByTestId('plotly-mock')
      const qqPlot = plots.find((p) => (p.getAttribute('data-plotly') ?? '').includes('Reference'))
      expect(qqPlot).toBeTruthy()
    })
  })

  it('shows an error-safe empty state when the summary request fails', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () =>
        HttpResponse.json({ detail: 'Column not found' }, { status: 500 }),
      ),
    )

    render(<DescriptivePanel />)

    // Loading first
    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    // Summary never resolves successfully -> falls back to "select a column" prompt,
    // no crash and no stale data rendered.
    await waitFor(() =>
      expect(screen.getByText('Select a column to view distribution')).toBeInTheDocument(),
    )
  })

  it('a slow stale request cannot overwrite a faster newer selection (race guard)', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', async ({ request }) => {
        const url = new URL(request.url)
        const column = url.searchParams.get('column')
        if (column === 'AGE') {
          // AGE was clicked FIRST but resolves LAST.
          await delay(50)
          return HttpResponse.json(numericSummary)
        }
        // GROUP was clicked SECOND but resolves FIRST.
        return HttpResponse.json(categoricalSummary)
      }),
    )

    const user = userEvent.setup()
    render(<DescriptivePanel />)

    // AGE auto-loads on mount (slow), then immediately switch to GROUP (fast).
    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    await user.click(screen.getByText('GROUP'))

    // GROUP's fast response should render...
    await waitFor(() =>
      expect(screen.getByText((_, el) => el?.textContent === 'Categorical · n=3')).toBeInTheDocument(),
    )
    // ...and AGE's slow response, arriving afterward, must NOT clobber it.
    await new Promise((r) => setTimeout(r, 80))
    expect(screen.getByText((_, el) => el?.textContent === 'Categorical · n=3')).toBeInTheDocument()
    expect(screen.queryByText((_, el) => el?.textContent === 'Continuous · n=3')).not.toBeInTheDocument()
  })
})
