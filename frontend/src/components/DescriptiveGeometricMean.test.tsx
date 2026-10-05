import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import DescriptivePanel from './DescriptivePanel'

afterEach(() => clearSession())

const baseNumeric = {
  type: 'numeric',
  histogram: [{ bin_start: 40, bin_end: 50, count: 1 }],
  raw_values: [55, 62, 48],
  outliers: [],
  normality_deviants: [],
  qq: [{ x: 0, y: 55 }],
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

function mockSummary(summary: Record<string, unknown>) {
  server.use(
    http.get('/api/stats/test-session/sparklines', () => HttpResponse.json({})),
    http.get('/api/stats/test-session/descriptive', () => HttpResponse.json({})),
    http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(summary)),
  )
}

async function openAge() {
  const user = userEvent.setup()
  render(<DescriptivePanel />)
  await user.click(await screen.findByTestId('summary-column-AGE'))
}

describe('DescriptivePanel geometric mean', () => {
  it('shows geometric mean with its 95% CI and the geometric SD', async () => {
    installSession()
    mockSummary({
      ...baseNumeric,
      geometric_mean: 54.62,
      geometric_sd: 1.14,
      geometric_mean_ci_lower: 40.1,
      geometric_mean_ci_upper: 74.4,
      geometric_mean_note: null,
    })
    await openAge()

    await screen.findByText('Geometric mean (95% CI)')
    expect(screen.getByText('54.62 [40.10, 74.40]')).toBeInTheDocument()
    expect(screen.getByText('Geometric SD')).toBeInTheDocument()
    expect(screen.getByText('1.14')).toBeInTheDocument()
  })

  it('explains a missing geometric mean for a non-negative column that contains zero', async () => {
    installSession()
    const note = 'Not defined: contains zero or negative values (geometric mean needs strictly positive data).'
    mockSummary({
      ...baseNumeric,
      min: 0,
      geometric_mean: null,
      geometric_sd: null,
      geometric_mean_ci_lower: null,
      geometric_mean_ci_upper: null,
      geometric_mean_note: note,
    })
    await openAge()

    const hint = await screen.findByText('Not defined (zero values)')
    expect(hint.closest('span[title]')).toHaveAttribute('title', note)
    expect(screen.queryByText('Geometric SD')).not.toBeInTheDocument()
  })

  it('stays quiet for a column with negative values (geometric mean simply not applicable)', async () => {
    installSession()
    mockSummary({
      ...baseNumeric,
      min: -3,
      geometric_mean: null,
      geometric_sd: null,
      geometric_mean_ci_lower: null,
      geometric_mean_ci_upper: null,
      geometric_mean_note: 'Not defined: contains zero or negative values (geometric mean needs strictly positive data).',
    })
    await openAge()

    await waitFor(() => expect(screen.getByText('Harmonic mean')).toBeInTheDocument())
    expect(screen.queryByText('Geometric mean (95% CI)')).not.toBeInTheDocument()
    expect(screen.queryByText(/Not defined \(zero values\)/)).not.toBeInTheDocument()
  })
})
