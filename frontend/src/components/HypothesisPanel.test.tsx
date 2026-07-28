import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import HypothesisPanel from './HypothesisPanel'

afterEach(() => clearSession())

describe('HypothesisPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<HypothesisPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('defaults to the one-sample t-test and runs it on success', async () => {
    installSession()
    server.use(
      http.post('/api/stats/ttest', () =>
        HttpResponse.json({
          test: 'One-sample t-test',
          interpretation: 'Significant difference from test value.',
          significant: true,
          statistic: 2.5,
          p: 0.03,
          df: 2,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<HypothesisPanel />)

    expect(screen.getByRole('radio', { name: /one-sample t-test/i })).toBeChecked()

    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'One-sample t-test' })).toBeInTheDocument())
    expect(screen.getByText('Significant difference from test value.')).toBeInTheDocument()
    expect(screen.getByText('Significant')).toBeInTheDocument()
  })

  it('switches to Mann-Whitney U and runs it with a group column', async () => {
    installSession()
    server.use(
      http.post('/api/stats/mannwhitney', () =>
        HttpResponse.json({
          test: 'Mann-Whitney U',
          interpretation: 'No significant difference.',
          significant: false,
          statistic: 1.2,
          p: 0.6,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<HypothesisPanel />)

    await user.click(screen.getByRole('radio', { name: /mann-whitney u/i }))
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Mann-Whitney U' })).toBeInTheDocument())
    expect(screen.getByText('No significant difference.')).toBeInTheDocument()
    expect(screen.getByText('Not significant')).toBeInTheDocument()
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/stats/ttest', () =>
        HttpResponse.json({ detail: 'Column contains no valid numeric data' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() =>
      expect(screen.getByText('Column contains no valid numeric data')).toBeInTheDocument(),
    )
  })

  it('renders an object warning instead of crashing the tab', async () => {
    // Reported from production: /api/stats/chisquare mixes plain strings with
    // rare-level dicts in one `warnings` list. Rendering the dict directly
    // threw React error #31 and the error boundary swallowed the whole Tests
    // tab, results included.
    installSession()
    server.use(
      http.post('/api/stats/ttest', () =>
        HttpResponse.json({
          test: 'One-sample t-test',
          interpretation: 'Significant.',
          significant: true,
          statistic: 2.5,
          p: 0.03,
          warnings: [
            {
              variable: 'cp',
              rare_levels: [{ level: 'rare_one', n: 2 }],
              kept_levels: [{ level: 'typical', n: 120 }],
              note: "'cp' has 1 category(ies) with <10 rows.",
            },
            'Some expected cell counts < 5. Consider Fisher\'s exact test instead.',
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'One-sample t-test' })).toBeInTheDocument(),
    )
    expect(screen.getByText(/has 1 category\(ies\) with <10 rows/)).toBeInTheDocument()
    expect(screen.getByText(/expected cell counts < 5/)).toBeInTheDocument()
    // The result itself must still be on screen.
    expect(screen.getByText('Significant.')).toBeInTheDocument()
  })
})
