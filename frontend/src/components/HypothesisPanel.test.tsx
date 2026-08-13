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

  it('switching to a categorical test re-points the column selection', async () => {
    // The primary selector draws from the numeric list for a t-test and the
    // categorical list for chi-square. The selection used to carry over, so
    // AGE stayed in `col` while the dropdown showed only GROUP: the value
    // being sent appeared nowhere on screen.
    installSession()
    let sent: Record<string, unknown> | null = null
    server.use(
      http.post('/api/stats/chisquare', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          test: 'Chi-square', interpretation: 'ok', significant: false, p: 0.5,
        })
      }),
    )

    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('radio', { name: /chi-square/i }))
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(sent).not.toBeNull())
    // GROUP is the only categorical column in the fixture.
    expect(sent!.row_column).toBe('GROUP')
    expect(sent!.row_column).not.toBe('AGE')
  })

  it("shows Cramer's V with its confidence interval and names the exact test", async () => {
    // A bare effect size cannot be read as evidence of a small effect rather
    // than of a small study, which is exactly what it is quoted for beside a
    // non-significant p. The backend now returns the interval, and the panel
    // has to put it on screen next to the test that actually produced the p.
    installSession()
    server.use(
      http.post('/api/stats/chisquare', () =>
        HttpResponse.json({
          test: 'Fisher-Freeman-Halton (MC, 5000 resamples)',
          exact_test: 'Fisher-Freeman-Halton (MC, 5000 resamples)',
          interpretation: 'No significant association.',
          significant: false,
          p: 0.772,
          p_chisquare: 0.681,
          chi2: 1.86,
          dof: 4,
          effect_sizes: [
            { name: 'cramers_v', value: 0.086, ci_low: 0.0, ci_high: 0.155, magnitude: 'small' },
          ],
          warnings: [
            'Some expected cell counts are below 5, so the reported p comes from Fisher-Freeman-Halton (MC, 5000 resamples) rather than from the chi-square.',
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('radio', { name: /chi-square/i }))
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() =>
      expect(
        screen.getByRole('heading', { name: 'Fisher-Freeman-Halton (MC, 5000 resamples)' }),
      ).toBeInTheDocument(),
    )
    expect(screen.getByText('95% CI: [0.000, 0.155]')).toBeInTheDocument()
    expect(screen.getByText(/expected cell counts are below 5/)).toBeInTheDocument()
  })

  it('two-way ANOVA lets you choose the first factor', async () => {
    // factor1 is sent as groupCol, but "two_way" was missing from the list
    // that renders the Group column selector, so the field was invisible.
    installSession()
    const user = userEvent.setup()
    render(<HypothesisPanel />)
    await user.click(screen.getByRole('radio', { name: /two-way anova/i }))
    expect(screen.getByText('Group column')).toBeInTheDocument()
  })
})
