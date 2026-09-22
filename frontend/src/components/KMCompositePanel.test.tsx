import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import KMCompositePanel from './KMCompositePanel'

afterEach(() => clearSession())

const trialSession = () =>
  makeSession({
    columns: [
      { name: 'arm', dtype: 'object', kind: 'categorical' },
      { name: 'dur_primary', dtype: 'float64', kind: 'numeric', label: 'Time to primary' },
      { name: 'ev_primary', dtype: 'int64', kind: 'numeric', label: 'Primary event' },
      { name: 'dur_death', dtype: 'float64', kind: 'numeric' },
      { name: 'ev_death', dtype: 'int64', kind: 'numeric' },
    ],
    preview: [
      { arm: 'FFRangio', dur_primary: 5, ev_primary: 0, dur_death: 5, ev_death: 0 },
      { arm: 'Pressure wire', dur_primary: 8, ev_primary: 1, dur_death: 8, ev_death: 0 },
    ],
  })

const kmResponse = {
  type: 'km_composite',
  group_col: 'arm',
  groups: ['FFRangio', 'Pressure wire'],
  as_cumulative_incidence: true,
  endpoints: [
    {
      label: 'Primary End Point',
      p_text: 'p = 0.434',
      final_by_group: { FFRangio: 6.9, 'Pressure wire': 7.1 },
      n_by_group: { FFRangio: 212, 'Pressure wire': 188 },
    },
  ],
  figure: {
    data: [
      { type: 'scatter', mode: 'lines', x: [0, 12], y: [0, 6.9], legendgroup: 'FFRangio', line: { color: '#000' } },
      { type: 'scatter', mode: 'lines', x: [0, 12], y: [0, 7.1], legendgroup: 'Pressure wire', line: { color: '#000' } },
    ],
    layout: { title: { text: 'Composite Primary End Point and Individual Components' }, height: 760 },
  },
  method_note: 'Cumulative incidence (1 - Kaplan-Meier survival) by group.',
}

describe('KMCompositePanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<KMCompositePanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('generates a KM composite figure and shows per-endpoint summary', async () => {
    installSession(trialSession())
    server.use(
      http.post('/api/charts/km_composite', () => HttpResponse.json(kmResponse)),
    )

    const user = userEvent.setup()
    render(<KMCompositePanel />)

    await user.click(screen.getByRole('button', { name: /generate km composite/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(screen.getByText('Primary End Point')).toBeInTheDocument()
    expect(screen.getByText('p = 0.434')).toBeInTheDocument()
    // Final cumulative-incidence % + n rendered in the summary row.
    expect(screen.getByText(/FFRangio: 6.9% \(n=212\)/)).toBeInTheDocument()
  })

  it('surfaces a backend error detail', async () => {
    installSession(trialSession())
    server.use(
      http.post('/api/charts/km_composite', () =>
        HttpResponse.json({ detail: "Event column 'ev_primary' must be binary 0/1." }, { status: 422 }),
      ),
    )

    const user = userEvent.setup()
    render(<KMCompositePanel />)
    await user.click(screen.getByRole('button', { name: /generate km composite/i }))

    await waitFor(() => expect(screen.getByText(/must be binary 0\/1/)).toBeInTheDocument())
  })

  it('closes every export of the figure once the data changes under it', async () => {
    installSession(trialSession())
    server.use(
      http.post('/api/charts/km_composite', () => HttpResponse.json(kmResponse)),
    )

    const user = userEvent.setup()
    render(<KMCompositePanel />)
    await user.click(screen.getByRole('button', { name: /generate km composite/i }))
    await screen.findByTestId('plotly-mock')

    // TitledPlot never offers the Plotly camera, so its own exporter is the
    // figure's whole export surface.
    const exports = () => [
      screen.getByRole('button', { name: '↓' }),
      screen.getByRole('button', { name: '⧉' }),
    ]
    for (const b of exports()) expect(b).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/This KM composite was computed before the data changed/)).toBeInTheDocument()
    for (const b of exports()) expect(b).toBeDisabled()
  })
})
