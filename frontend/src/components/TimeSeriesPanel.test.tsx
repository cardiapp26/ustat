import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { useStore } from '../store'
import { clearSession, installSession } from '../test/testUtils'
import TimeSeriesPanel from './TimeSeriesPanel'

afterEach(() => clearSession())

describe('TimeSeriesPanel', () => {
  it('renders the ARIMA mode selector even without a session (guards against null session)', () => {
    clearSession()
    render(<TimeSeriesPanel />)
    expect(screen.getByRole('button', { name: /^arima$/i })).toBeInTheDocument()
  })

  it('shows the three mode tabs and defaults to ARIMA', () => {
    installSession()
    render(<TimeSeriesPanel />)
    expect(screen.getByRole('button', { name: /^arima$/i })).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /decompose/i }).length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /^stationarity$/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /fit arima/i })).toBeInTheDocument()
  })

  it('runs ARIMA and renders fit stats + coefficients table', async () => {
    installSession()
    server.use(
      http.post('/api/timeseries/arima', () =>
        HttpResponse.json({
          value_col: 'AGE',
          order: [1, 1, 1],
          seasonal_order: [0, 0, 0, 0],
          aic: 123.4,
          bic: 130.1,
          ljung_box_p: 0.45,
          n: 3,
          fitted: [
            { x: 1, observed: 55, fitted: 54 },
            { x: 2, observed: 62, fitted: 60 },
          ],
          forecast: [
            { x: 3, forecast: 58, ci_low: 50, ci_high: 66 },
          ],
          coefficients: [
            { term: 'ar.L1', estimate: 0.5, se: 0.1, p: 0.01 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<TimeSeriesPanel />)

    const valueSelect = screen.getByRole('combobox', { name: /value \(series\)/i })
    await user.selectOptions(valueSelect, 'AGE')

    await user.click(screen.getByRole('button', { name: /fit arima/i }))

    await waitFor(() => expect(screen.getByText('AIC')).toBeInTheDocument())
    expect(screen.getByText('123.4')).toBeInTheDocument()
    expect(screen.getByText('Coefficients')).toBeInTheDocument()
    expect(screen.getByText('ar.L1')).toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  })

  it('runs decomposition and shows trend/seasonal strength', async () => {
    installSession()
    server.use(
      http.post('/api/timeseries/decompose', () =>
        HttpResponse.json({
          value_col: 'AGE',
          method: 'stl',
          period: 12,
          x: [1, 2, 3],
          observed: [1, 2, 3],
          trend: [1, 2, 3],
          seasonal: [0, 0, 0],
          resid: [0, 0, 0],
          strength_trend: 0.87,
          strength_seasonal: 0.12,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<TimeSeriesPanel />)
    const decomposeButtons = screen.getAllByRole('button', { name: /decompose/i })
    await user.click(decomposeButtons[0])
    const runButtons = screen.getAllByRole('button', { name: /decompose/i })
    await user.click(runButtons[runButtons.length - 1])

    await waitFor(() => expect(screen.getByText('0.87')).toBeInTheDocument())
    expect(screen.getByText('0.12')).toBeInTheDocument()
  })

  it('runs stationarity test and shows ADF/KPSS results', async () => {
    installSession()
    server.use(
      http.post('/api/timeseries/stationarity', () =>
        HttpResponse.json({
          adf_p: 0.02,
          adf_stationary: true,
          kpss_p: 0.08,
          kpss_stationary: true,
          acf: [{ lag: 1, value: 0.4, ci_low: -0.2, ci_high: 0.2 }],
          pacf: [{ lag: 1, value: 0.4, ci_low: -0.2, ci_high: 0.2 }],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<TimeSeriesPanel />)
    await user.click(screen.getByRole('button', { name: /^stationarity$/i }))
    await user.click(screen.getByRole('button', { name: /test stationarity/i }))

    await waitFor(() => expect(screen.getByText(/ADF \(H₀: unit root\)/)).toBeInTheDocument())
    expect(screen.getByText(/KPSS \(H₀: stationary\)/)).toBeInTheDocument()
  })

  it('shows a validation error when no value column is selected', async () => {
    installSession({
      session_id: 'test-session',
      filename: 'test.csv',
      rows: 3,
      columns: [{ name: 'GROUP', dtype: 'object', kind: 'categorical' }],
      preview: [{ GROUP: 'A' }],
    })

    const user = userEvent.setup()
    render(<TimeSeriesPanel />)
    await user.click(screen.getByRole('button', { name: /fit arima/i }))

    await waitFor(() => expect(screen.getByText('Select a value column.')).toBeInTheDocument())
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/timeseries/arima', () =>
        HttpResponse.json({ detail: 'Series too short for ARIMA' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<TimeSeriesPanel />)
    await user.click(screen.getByRole('button', { name: /fit arima/i }))

    await waitFor(() => expect(screen.getByText('Series too short for ARIMA')).toBeInTheDocument())
  })
})

describe('TimeSeriesPanel staleness', () => {
  const arima = {
    value_col: 'AGE', order: [1, 1, 1], seasonal_order: [0, 0, 0, 0],
    aic: 123.4, bic: 130.1, ljung_box_p: 0.45, n: 3,
    fitted: [{ x: 1, observed: 55, fitted: 54 }, { x: 2, observed: 62, fitted: 60 }],
    forecast: [{ x: 3, forecast: 58, ci_low: 50, ci_high: 66 }],
    coefficients: [{ term: 'ar.L1', estimate: 0.5, se: 0.1, p: 0.01 }],
  }

  it('closes the coefficient export of a fit once the data changes under it', async () => {
    installSession()
    server.use(http.post('/api/timeseries/arima', () => HttpResponse.json(arima)))

    const user = userEvent.setup()
    render(<TimeSeriesPanel />)
    await user.selectOptions(screen.getByRole('combobox', { name: /value \(series\)/i }), 'AGE')
    await user.click(screen.getByRole('button', { name: /fit arima/i }))

    const csv = await screen.findByRole('button', { name: 'CSV' })
    expect(csv).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeDisabled()
  })

  it('brings a decomposition back after a remount in its own layout, still current', async () => {
    // The result outlives the panel, so the mode that says how to draw it has
    // to as well: read under the default ARIMA layout it would throw.
    installSession()
    server.use(
      http.post('/api/timeseries/decompose', () =>
        HttpResponse.json({
          value_col: 'AGE', method: 'stl', period: 12,
          x: [1, 2, 3], observed: [1, 2, 3], trend: [1, 2, 3], seasonal: [0, 0, 0], resid: [0, 0, 0],
          strength_trend: 0.87, strength_seasonal: 0.12,
        }),
      ),
    )

    const user = userEvent.setup()
    const { unmount } = render(<TimeSeriesPanel />)
    await user.click(screen.getAllByRole('button', { name: /decompose/i })[0])
    const runButtons = screen.getAllByRole('button', { name: /decompose/i })
    await user.click(runButtons[runButtons.length - 1])
    await screen.findByText('0.87')
    unmount()

    render(<TimeSeriesPanel />)
    expect(screen.getByText('0.87')).toBeInTheDocument()
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()
  })
})
