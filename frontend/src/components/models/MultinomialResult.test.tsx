import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import MultinomialResult, { type MultinomialResultData } from './MultinomialResult'

const coef = (variable: string, log_rrr: number, p: number) => ({
  variable, log_rrr, se: 0.1, z: log_rrr / 0.1, p,
  rrr: Math.exp(log_rrr), rrr_ci_low: Math.exp(log_rrr - 0.196), rrr_ci_high: Math.exp(log_rrr + 0.196),
})

const RESULT: MultinomialResultData = {
  model: 'Multinomial Logistic Regression',
  outcome: 'af_type',
  n: 120,
  categories: ['1', '2', '3'],
  reference: '1',
  category_counts: { '1': 50, '2': 40, '3': 30 },
  equations: [
    { category: '2', vs: '1', n: 40, coefficients: [coef('Intercept', -0.2, 0.4), coef('age', 0.05, 0.003)] },
    { category: '3', vs: '1', n: 30, coefficients: [coef('Intercept', -0.5, 0.1), coef('age', 0.01, 0.6)] },
  ],
  lr_tests: [{ variable: 'age', lr_chi2: 9.87, df: 2, p: 0.0072 }],
  model_lr_chi2: 9.87, model_lr_df: 2, model_lr_p: 0.0072,
  pseudo_r2: 0.038, aic: 251.2, bic: 267.9,
  classification_table: { labels: ['1', '2', '3'], table: [[30, 15, 5], [12, 25, 3], [10, 8, 12]], accuracy: 0.558 },
  warnings: ['Few cases per estimated parameter'],
  result_text: 'A multinomial logistic regression of af_type ...',
}

const LABELS = { '1': 'Paroxysmal', '2': 'Persistent', '3': 'Permanent' }

describe('MultinomialResult', () => {
  it('names each equation by its category labels against the reference', () => {
    render(<MultinomialResult result={RESULT} valueLabels={LABELS} />)
    expect(screen.getByText('Persistent vs Paroxysmal')).toBeTruthy()
    expect(screen.getByText('Permanent vs Paroxysmal')).toBeTruthy()
    expect(screen.getByText('Paroxysmal', { selector: 'strong' })).toBeTruthy()
  })

  it('shows the per-predictor likelihood-ratio test and the model fit', () => {
    render(<MultinomialResult result={RESULT} valueLabels={LABELS} />)
    expect(screen.getByText('Likelihood-ratio tests')).toBeTruthy()
    expect(screen.getAllByText('9.87').length).toBeGreaterThan(0)
    expect(screen.getByText('55.8% correctly classified')).toBeTruthy()
  })

  it('does not print an RRR for the intercept', () => {
    render(<MultinomialResult result={RESULT} />)
    const interceptRows = screen.getAllByText('Intercept').map((el) => el.closest('tr')!)
    for (const row of interceptRows) {
      const cells = Array.from(row.querySelectorAll('td')).map((td) => td.textContent)
      expect(cells[3]).toBe('—')
    }
  })

  it('surfaces server warnings', () => {
    render(<MultinomialResult result={RESULT} />)
    expect(screen.getByRole('status').textContent).toContain('Few cases per estimated parameter')
  })
})
