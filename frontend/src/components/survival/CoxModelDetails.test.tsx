import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CoxModelDetails } from './CoxModelDetails'

describe('CoxModelDetails', () => {
  it('shows the per-term and global PH test and flags a violation', () => {
    render(
      <CoxModelDetails
        phTest={{
          global: { chi2: 9.84, df: 2, p: 0.0073 },
          per_term: [
            { variable: 'age', test_stat: 0.42, p: 0.52 },
            { variable: 'arm_treat', test_stat: 8.9, p: 0.0029 },
          ],
        }}
        nAnalyzed={85}
        nExcluded={15}
      />,
    )
    expect(screen.getByText('arm_treat')).toBeInTheDocument()
    expect(screen.getByText('Global')).toBeInTheDocument()
    expect(screen.getByText('9.840')).toBeInTheDocument()
    expect(screen.getByText(/rejects proportional hazards/)).toBeInTheDocument()
    expect(screen.getByText('85 rows analysed, 15 excluded for missing values.')).toBeInTheDocument()
  })

  it('shows the pooling note and a test failure message', () => {
    render(<CoxModelDetails phTest={{ error: 'singular matrix' }} poolingNote="Pooled over 5 imputed datasets." />)
    expect(screen.getByText('Pooled over 5 imputed datasets.')).toBeInTheDocument()
    expect(screen.getByText(/could not be computed: singular matrix/)).toBeInTheDocument()
    expect(screen.queryByText('Global')).not.toBeInTheDocument()
  })
})
