import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import { useStore } from '../store'
import CategoricalTestsPanel from './CategoricalTestsPanel'

afterEach(() => clearSession())

describe('CategoricalTestsPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<CategoricalTestsPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('defaults to the binomial test with a null-proportion input', () => {
    installSession()
    render(<CategoricalTestsPanel />)
    expect(screen.getByRole('radio', { name: 'Binomial test' })).toBeChecked()
    expect(screen.getByText('Expected proportion')).toBeInTheDocument()
  })

  it('switches to two-proportions and shows the group column selector', async () => {
    installSession()
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Two proportions z-test' }))
    expect(screen.getByText('Group column')).toBeInTheDocument()
  })

  it('runs the binomial test and renders the result card', async () => {
    installSession()
    server.use(
      http.post('/api/categorical/binomial', () =>
        HttpResponse.json({
          test: 'Binomial test',
          interpretation: 'Observed proportion differs from expected.',
          significant: true,
          successes: 2,
          n: 3,
          observed_p: 0.667,
          expected_p: 0.5,
          p_value: 0.03,
          result_text: 'The observed proportion of 0.667 differs significantly from 0.5.',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Binomial test' })).toBeInTheDocument())
    expect(screen.getByText('Significant')).toBeInTheDocument()
    expect(screen.getByText('Observed proportion differs from expected.')).toBeInTheDocument()
    expect(screen.getByText('The observed proportion of 0.667 differs significantly from 0.5.')).toBeInTheDocument()
  })

  it('shows the error message from the backend on failure', async () => {
    installSession()
    server.use(
      http.post('/api/categorical/binomial', () =>
        HttpResponse.json({ detail: 'Column is not binary' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByText('Column is not binary')).toBeInTheDocument())
  })

  it('surfaces backend warnings above the numbers', async () => {
    // An assumed level ordering can invert the trend direction, so the warning
    // must be visible — it used to be dropped from the result card entirely.
    installSession()
    server.use(
      http.post('/api/categorical/cochran_armitage', () =>
        HttpResponse.json({
          test: 'Cochran-Armitage trend test',
          z: -4.0249,
          p: 0.000057,
          significant: true,
          level_order_source: 'alphabetical (assumed)',
          warnings: [
            "'dose' has non-numeric levels, so they were ordered alphabetically: ['High', 'Low', 'Medium'].",
          ],
          interpretation: 'Significant linear trend (direction: decreasing).',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByLabelText('Cochran-Armitage trend'))
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() =>
      expect(screen.getByText(/ordered alphabetically/i)).toBeInTheDocument(),
    )
  })

  it('sends level_order only when the user supplies one', async () => {
    installSession()
    const bodies: Record<string, unknown>[] = []
    server.use(
      http.post('/api/categorical/cochran_armitage', async ({ request }) => {
        bodies.push((await request.json()) as Record<string, unknown>)
        return HttpResponse.json({ test: 'Cochran-Armitage trend test', z: 1, p: 0.3 })
      }),
    )

    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByLabelText('Cochran-Armitage trend'))

    await user.click(screen.getByRole('button', { name: /run test/i }))
    await waitFor(() => expect(bodies).toHaveLength(1))
    expect(bodies[0]).not.toHaveProperty('level_order')

    await user.type(screen.getByPlaceholderText(/Low, Medium, High/i), 'Low, Medium, High')
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await waitFor(() => expect(bodies).toHaveLength(2))
    expect(bodies[1].level_order).toEqual(['Low', 'Medium', 'High'])
  })

  it('marks a test out of date once the data changes, and Recompute reruns it', async () => {
    installSession()
    let calls = 0
    server.use(
      http.post('/api/categorical/binomial', () => {
        calls += 1
        return HttpResponse.json({ test: 'Binomial test', significant: true, n: 3, p_value: 0.03 })
      }),
    )

    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: 'Binomial test' })
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByText(/This Binomial test was computed before the data changed/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Recompute' }))
    await waitFor(() => expect(calls).toBe(2))
    await waitFor(() => expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument())
  })

  it('flags a test stale for an input it read, not for one it ignores', async () => {
    // Cochran's Q reads only its column list; the "Binary column" picker stays
    // on screen for it but never reaches the request.
    installSession()
    server.use(
      http.post('/api/categorical/cochran_q', () =>
        HttpResponse.json({ test: "Cochran's Q", significant: false, q: 1.2, p: 0.55 }),
      ),
    )

    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: "Cochran's Q" }))
    await user.selectOptions(screen.getByRole('listbox'), ['AGE', 'LDL', 'DM'])
    await user.click(screen.getByRole('button', { name: /run test/i }))
    await screen.findByRole('heading', { name: "Cochran's Q" })

    await user.selectOptions(screen.getByRole('combobox'), 'LDL')
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()

    await user.deselectOptions(screen.getByRole('listbox'), 'DM')
    expect(await screen.findByText(/the analysis settings changed/)).toBeInTheDocument()
  })
})
