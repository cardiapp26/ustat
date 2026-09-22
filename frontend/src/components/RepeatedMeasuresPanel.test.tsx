import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { useStore } from '../store'
import { clearSession, installSession } from '../test/testUtils'
import RepeatedMeasuresPanel from './RepeatedMeasuresPanel'

afterEach(() => clearSession())

describe('RepeatedMeasuresPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<RepeatedMeasuresPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('defaults to Paired t-test and shows measurement selectors', () => {
    installSession()
    render(<RepeatedMeasuresPanel />)
    expect(screen.getByRole('radio', { name: /paired t-test/i })).toBeChecked()
    expect(screen.getByText('Measurement 1')).toBeInTheDocument()
    expect(screen.getByText('Measurement 2')).toBeInTheDocument()
  })

  it('runs Paired t-test and renders the result card', async () => {
    installSession()
    server.use(
      http.post('/api/repeated/paired_ttest', () =>
        HttpResponse.json({
          test: 'Paired t-test',
          interpretation: 'Significant mean difference',
          significant: true,
          t: 3.21,
          df: 2,
          p: 0.041,
          effect_sizes: [{ name: 'cohens_d_z', value: 1.85, magnitude: 'large' }],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<RepeatedMeasuresPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByText('Significant mean difference')).toBeInTheDocument())
    const heading = screen.getByRole('heading', { name: 'Paired t-test' })
    expect(heading).toBeInTheDocument()
    expect(screen.getByText('Significant')).toBeInTheDocument()
    expect(screen.getByText('cohens d z')).toBeInTheDocument()
  })

  it('disables Run Test until 3+ conditions selected for Friedman', async () => {
    installSession()
    const user = userEvent.setup()
    render(<RepeatedMeasuresPanel />)
    await user.click(screen.getByRole('radio', { name: /friedman/i }))

    expect(screen.getByRole('button', { name: /run test/i })).toBeDisabled()

    const listbox = screen.getByRole('listbox')
    await user.selectOptions(listbox, ['AGE', 'LDL', 'DM'])
    expect(screen.getByRole('button', { name: /run test/i })).toBeEnabled()
  })

  it('runs Friedman test and renders results', async () => {
    installSession()
    server.use(
      http.post('/api/repeated/friedman', () =>
        HttpResponse.json({
          test: 'Friedman test',
          interpretation: 'At least one condition differs',
          significant: true,
          chi2: 6.5,
          p: 0.039,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<RepeatedMeasuresPanel />)
    await user.click(screen.getByRole('radio', { name: /friedman/i }))
    await user.selectOptions(screen.getByRole('listbox'), ['AGE', 'LDL', 'DM'])
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByText('At least one condition differs')).toBeInTheDocument())
    expect(screen.getByRole('heading', { name: 'Friedman test' })).toBeInTheDocument()
  })

  it('shows subject/within/between selectors for Mixed ANOVA', async () => {
    installSession()
    const user = userEvent.setup()
    render(<RepeatedMeasuresPanel />)
    await user.click(screen.getByRole('radio', { name: /mixed anova/i }))

    expect(screen.getByText('Subject ID column')).toBeInTheDocument()
    expect(screen.getByText('Within-subjects factor')).toBeInTheDocument()
    expect(screen.getByText('Between-subjects factor')).toBeInTheDocument()
    expect(screen.getByText('Outcome (numeric)')).toBeInTheDocument()
  })

  it('runs Mixed ANOVA and renders the effects table', async () => {
    installSession()
    server.use(
      http.post('/api/repeated/mixed_anova', () =>
        HttpResponse.json({
          test: 'Mixed ANOVA',
          interpretation: 'Significant interaction',
          effects: [
            { term: 'time', F: 5.2, df_num: 1, df_den: 8, p: 0.03, effect_size: { value: 0.2 }, significant: true },
            { term: 'group:time', F: 1.1, df_num: 1, df_den: 8, p: 0.4, effect_size: { value: 0.05 }, significant: false },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<RepeatedMeasuresPanel />)
    await user.click(screen.getByRole('radio', { name: /mixed anova/i }))
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByText('ANOVA Effects')).toBeInTheDocument())
    const table = screen.getByRole('table')
    expect(within(table).getByText('time')).toBeInTheDocument()
    expect(within(table).getByText('group:time')).toBeInTheDocument()
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/repeated/paired_ttest', () =>
        HttpResponse.json({ detail: 'Columns must be numeric' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<RepeatedMeasuresPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByText('Columns must be numeric')).toBeInTheDocument())
  })
})

describe('RepeatedMeasuresPanel staleness', () => {
  it('closes the export of a test once the data changes under it', async () => {
    installSession()
    server.use(
      http.post('/api/repeated/paired_ttest', () =>
        HttpResponse.json({ test: 'Paired t-test', interpretation: 'Significant mean difference', t: 3.21, df: 2, p: 0.041 }),
      ),
    )

    const user = userEvent.setup()
    render(<RepeatedMeasuresPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))

    const csv = await screen.findByRole('button', { name: 'CSV' })
    expect(csv).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeDisabled()
  })
})
