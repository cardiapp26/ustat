import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { KnotComparisonTable, RCSTestChips } from './RCSInference'
import { SUPPORTED_KNOTS, harrellPercentiles, knotBudget } from '../lib/rcsKnots'

describe('RCSTestChips', () => {
  it('shows overall association and non-linearity with chi-square statistics', () => {
    render(
      <RCSTestChips
        adjusted={false}
        result={{
          test_statistic: 'chi2',
          overall_p: 0.0414, overall_df: 3, overall_wald: 8.234,
          nonlinearity_p: 0.8999, nonlinearity_df: 2, nonlinearity_wald: 0.211,
        }}
      />,
    )
    expect(screen.getByText(/overall association: χ²\(3\) = 8\.23/)).toBeInTheDocument()
    expect(screen.getByText(/non-linearity: χ²\(2\) = 0\.21/)).toBeInTheDocument()
  })

  it('uses the F statistic for linear models and labels adjusted tests', () => {
    render(
      <RCSTestChips
        adjusted
        result={{
          test_statistic: 'F',
          overall_p: 1e-15, overall_df: 3, overall_wald: 80.6, overall_f: 26.85,
          nonlinearity_p: 0.0256, nonlinearity_df: 2, nonlinearity_wald: 7.42, nonlinearity_f: 3.71,
        }}
      />,
    )
    expect(screen.getByText(/adjusted overall association: F\(3\) = 26\.85/)).toBeInTheDocument()
    expect(screen.getByText(/adjusted non-linearity: F\(2\) = 3\.71/)).toBeInTheDocument()
  })

  it('renders nothing for tests the server did not return', () => {
    const { container } = render(<RCSTestChips adjusted={false} result={{}} />)
    expect(container.textContent).toBe('')
  })
})

describe('KnotComparisonTable', () => {
  const rows = [
    { n_knots: 3, aic: 347.376, delta_aic: 1.517, best: false, chosen: false },
    { n_knots: 4, aic: 349.164, delta_aic: 3.304, best: false, chosen: true },
    { n_knots: 5, aic: 345.860, delta_aic: 0, best: true, chosen: false },
  ]

  it('lists every knot count and marks the fitted and lowest-AIC rows', () => {
    render(<KnotComparisonTable rows={rows} hasInteraction={false} />)
    const body = screen.getAllByRole('row').slice(1)
    expect(body).toHaveLength(3)
    expect(within(body[1]).getByText('4 (fitted)')).toBeInTheDocument()
    expect(within(body[2]).getByText('lowest')).toBeInTheDocument()
    expect(within(body[0]).getByText('347.4')).toBeInTheDocument()
    expect(screen.queryByText(/without interaction/)).not.toBeInTheDocument()
  })

  it('notes that interactions are left out of the comparison', () => {
    render(<KnotComparisonTable rows={rows} hasInteraction />)
    expect(screen.getByText(/without interaction terms/)).toBeInTheDocument()
  })

  it('renders nothing without rows (custom knots)', () => {
    const { container } = render(<KnotComparisonTable rows={[]} hasInteraction={false} />)
    expect(container.textContent).toBe('')
  })
})

describe('knotBudget', () => {
  it('covers 3 to 7 knots with k - 1 parameters', () => {
    expect([...SUPPORTED_KNOTS]).toEqual([3, 4, 5, 6, 7])
    const rows = knotBudget(17)
    expect(rows.map(r => r.params)).toEqual([2, 3, 4, 5, 6])
    expect(rows.map(r => r.label)).toEqual(['safe', 'borderline', 'overfit risk', 'overfit risk', 'overfit risk'])
    expect(rows[0].epv).toBeCloseTo(8.5)
  })

  it('lists one Harrell percentile per knot', () => {
    for (const k of SUPPORTED_KNOTS) expect(harrellPercentiles(k)).toHaveLength(k)
    expect(harrellPercentiles(6)).toEqual([5, 23, 41, 59, 77, 95])
  })
})
