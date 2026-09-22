import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import SubgroupBarPanel from './SubgroupBarPanel'

afterEach(() => {
  clearSession()
  vi.unstubAllGlobals()
})

/** Find the <select> that immediately follows a given field label text. */
function selectAfterLabel(labelText: string): HTMLSelectElement {
  const label = screen.getByText(labelText)
  const wrapper = label.parentElement as HTMLElement
  return within(wrapper).getByRole('combobox') as HTMLSelectElement
}

/** Find the radio <input> associated with visible option text within a group. */
function radioByText(text: string): HTMLInputElement {
  const label = screen.getByText(text).closest('label') as HTMLLabelElement
  return within(label).getByRole('radio') as HTMLInputElement
}

const twoGroupSession = () =>
  makeSession({
    columns: [
      { name: 'AGE', dtype: 'float64', kind: 'numeric' },
      { name: 'LDL', dtype: 'float64', kind: 'numeric' },
      { name: 'GROUP', dtype: 'object', kind: 'categorical' },
      { name: 'SEX', dtype: 'object', kind: 'categorical' },
    ],
    preview: [
      { AGE: 55, LDL: 120, GROUP: 'A', SEX: 'M' },
      { AGE: 62, LDL: 140, GROUP: 'B', SEX: 'F' },
      { AGE: 48, LDL: 110, GROUP: 'A', SEX: 'M' },
    ],
  })

/** A generated percentage chart for the categorical Y column GROUP. */
const percentageChart = () => ({
  subgroup_col: 'GROUP',
  xaxis_col: 'SEX',
  color_col: null,
  y_mode: 'percentage',
  error_type: 'ci',
  subgroups: ['A', 'B'],
  traces: [
    {
      name: 'GROUP',
      x_subgroup: ['A', 'B'],
      x_xaxis: ['M', 'F'],
      y: [40, 60],
      ns: [10, 12],
      error_high: [5, 6],
      error_low: [5, 6],
    },
  ],
})

describe('SubgroupBarPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<SubgroupBarPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows the configure placeholder before generating a chart', async () => {
    installSession(twoGroupSession())
    server.use(
      http.get('/api/compute/:sessionId/unique/:col', () => HttpResponse.json([])),
    )
    render(<SubgroupBarPanel />)
    await waitFor(() =>
      expect(screen.getByText('Configure and Generate a Nested Subgroup Bar Chart')).toBeInTheDocument(),
    )
  })

  it('mean mode: generates chart with error bars and value labels for a numeric Y variable', async () => {
    installSession(twoGroupSession())
    server.use(
      http.get('/api/compute/:sessionId/unique/:col', () => HttpResponse.json([])),
      http.post('/api/charts/subgroup_bar', () =>
        HttpResponse.json({
          subgroup_col: 'GROUP',
          xaxis_col: 'SEX',
          color_col: null,
          y_mode: 'mean',
          error_type: 'ci',
          subgroups: ['A', 'B'],
          traces: [
            {
              name: 'AGE',
              x_subgroup: ['A', 'A', 'B'],
              x_xaxis: ['M', 'F', 'F'],
              y: [55, 60, 62],
              ns: [10, 8, 12],
              error_high: [2, 3, 2.5],
              error_low: [2, 3, 2.5],
            },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SubgroupBarPanel />)

    // Y-axis default is AGE (numeric), mean mode selected automatically.
    await waitFor(() => expect(radioByText('Mean')).toBeChecked())

    await user.click(screen.getByRole('button', { name: /Generate Chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(screen.getByText('Custom Labels & Dimensions')).toBeInTheDocument()
  })

  it('percentage mode: fetches target categories and shows an error on failed generation', async () => {
    installSession(twoGroupSession())
    server.use(
      http.get('/api/compute/:sessionId/unique/:col', () => HttpResponse.json(['A', 'B'])),
      http.post('/api/charts/subgroup_bar', () =>
        HttpResponse.json({ detail: 'Not enough data per group to compute a rate' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<SubgroupBarPanel />)

    // Switch Y-axis to the categorical GROUP column -> auto switches to percentage mode.
    await user.selectOptions(selectAfterLabel('Y-Axis Variable (Value)'), 'GROUP')
    await waitFor(() => expect(radioByText('Percentage (%)')).toBeChecked())

    // Target category select should be populated from the mocked unique values.
    await waitFor(() => expect(screen.getByText('Target Event / Category')).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: /Generate Chart/i }))

    await waitFor(() =>
      expect(screen.getByText('Not enough data per group to compute a rate')).toBeInTheDocument(),
    )
  })

  it('closes the chart exports and the modebar camera once the data changes under it', async () => {
    installSession(twoGroupSession())
    server.use(
      http.get('/api/compute/:sessionId/unique/:col', () => HttpResponse.json([])),
      http.post('/api/charts/subgroup_bar', () => HttpResponse.json({ ...percentageChart(), y_mode: 'mean' })),
    )
    const user = userEvent.setup()
    render(<SubgroupBarPanel />)
    await waitFor(() => expect(radioByText('Mean')).toBeChecked())
    await user.click(screen.getByRole('button', { name: /Generate Chart/i }))
    await screen.findByTestId('plotly-mock')

    const exports = () => [screen.getByRole('button', { name: '⧉' }), screen.getByRole('button', { name: '↓' })]
    const removed = () =>
      JSON.parse(screen.getByTestId('plotly-mock').getAttribute('data-config') ?? '{}').modeBarButtonsToRemove ?? []
    for (const b of exports()) expect(b).toBeEnabled()
    expect(removed()).not.toContain('toImage')

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    for (const b of exports()) expect(b).toBeDisabled()
    expect(removed()).toContain('toImage')
  })

  it('brings the chart back current, with its settings, after a remount', async () => {
    // The chart is cached across a tab switch; settings that were not would
    // bring it back stale, with the default Y column and target on screen.
    // A restored chart is on screen at mount, so the panel's size observer
    // attaches straight away; jsdom has none.
    vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
    installSession(twoGroupSession())
    server.use(
      http.get('/api/compute/:sessionId/unique/:col', () => HttpResponse.json(['A', 'B'])),
      http.post('/api/charts/subgroup_bar', () => HttpResponse.json(percentageChart())),
    )
    const user = userEvent.setup()
    const first = render(<SubgroupBarPanel />)
    await user.selectOptions(selectAfterLabel('Y-Axis Variable (Value)'), 'GROUP')
    await waitFor(() => expect(screen.getByText('Target Event / Category')).toBeInTheDocument())
    await user.selectOptions(selectAfterLabel('Target Event / Category'), 'B')
    await user.click(screen.getByRole('button', { name: /Generate Chart/i }))
    await screen.findByTestId('plotly-mock')
    first.unmount()

    render(<SubgroupBarPanel />)
    await screen.findByTestId('plotly-mock')
    await waitFor(() => expect(selectAfterLabel('Target Event / Category').value).toBe('B'))
    expect(selectAfterLabel('Y-Axis Variable (Value)').value).toBe('GROUP')
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '↓' })).toBeEnabled()
  })
})
