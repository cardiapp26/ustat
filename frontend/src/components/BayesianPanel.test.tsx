import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import { useStore } from '../store'
import BayesianPanel from './BayesianPanel'

afterEach(() => clearSession())

describe('BayesianPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<BayesianPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('defaults to one-sample t-test and shows the mu input', () => {
    installSession()
    render(<BayesianPanel />)
    expect(screen.getByDisplayValue('Bayesian One-Sample t-test')).toBeInTheDocument()
    expect(screen.getByText('Test Value (mu)')).toBeInTheDocument()
  })

  it('runs one-sample Bayesian t-test and renders BF10/BF01', async () => {
    installSession()
    server.use(
      http.post('/api/bayesian', () =>
        HttpResponse.json({
          analysis: 'Bayesian One-Sample t-test',
          n: 3,
          bf10: 12.5,
          bf01: 0.08,
          interpretation: 'Strong evidence for H1',
          statistic_label: 't',
          statistic_value: 3.1,
          df: 2,
          effect_size_label: 'd',
          effect_size_value: 1.4,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.click(screen.getByRole('button', { name: /compute bayes factor/i }))

    await waitFor(() => expect(screen.getByText('12.5000')).toBeInTheDocument())
    expect(screen.getByText('0.0800')).toBeInTheDocument()
    expect(screen.getByText(/Strong evidence for H1/)).toBeInTheDocument()
  })

  it('shows the grouping-variable selector for independent t-test', async () => {
    installSession()
    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.selectOptions(screen.getByDisplayValue('Bayesian One-Sample t-test'), 'ttest_ind')
    expect(screen.getByText('Grouping Variable')).toBeInTheDocument()
  })

  it('disables Compute for regression until predictors are selected', async () => {
    installSession()
    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.selectOptions(screen.getByDisplayValue('Bayesian One-Sample t-test'), 'regression')
    expect(screen.getByRole('button', { name: /compute bayes factor/i })).toBeDisabled()

    const multi = screen.getByText('Predictors (numeric)').closest('div')!.querySelector('select')!
    await user.selectOptions(multi, ['LDL'])
    expect(screen.getByRole('button', { name: /compute bayes factor/i })).toBeEnabled()
  })

  it('closes the plot export once the data changes under the Bayes factor', async () => {
    installSession()
    server.use(
      http.post('/api/bayesian', () =>
        HttpResponse.json({
          analysis: 'Bayesian One-Sample t-test',
          n: 3, bf10: 12.5, bf01: 0.08,
          interpretation: 'Strong evidence for H1',
          statistic_label: 't', statistic_value: 3.1, df: 2,
          effect_size_label: 'd', effect_size_value: 1.4,
          plot_coords: [
            { x: -1, prior: 0.1, posterior: 0.05 },
            { x: 0, prior: 0.3, posterior: 0.1 },
            { x: 1, prior: 0.1, posterior: 0.4 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.click(screen.getByRole('button', { name: /compute bayes factor/i }))
    await screen.findByText('12.5000')

    const exports = () => [
      screen.getByRole('button', { name: '⧉' }),
      screen.getByRole('button', { name: '↓' }),
    ]
    for (const b of exports()) expect(b).toBeEnabled()
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    for (const b of exports()) expect(b).toBeDisabled()
  })

  it('dates the result when the test value it was computed against changes', async () => {
    installSession()
    server.use(
      http.post('/api/bayesian', () =>
        HttpResponse.json({
          analysis: 'Bayesian One-Sample t-test',
          n: 3, bf10: 12.5, bf01: 0.08,
          interpretation: 'Strong evidence for H1',
          statistic_label: 't', statistic_value: 3.1, df: 2,
          effect_size_label: 'd', effect_size_value: 1.4,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.click(screen.getByRole('button', { name: /compute bayes factor/i }))
    await screen.findByText('12.5000')

    await user.type(screen.getByRole('spinbutton'), '5')

    expect(await screen.findByText(/analysis settings changed/)).toBeInTheDocument()
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/bayesian', () =>
        HttpResponse.json({ detail: 'Invalid outcome column' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.click(screen.getByRole('button', { name: /compute bayes factor/i }))

    await waitFor(() => expect(screen.getByText('Invalid outcome column')).toBeInTheDocument())
  })
})
