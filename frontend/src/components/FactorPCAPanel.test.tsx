import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { useStore } from '../store'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import FactorPCAPanel from './FactorPCAPanel'

afterEach(() => clearSession())

const baseResult = {
  factors: ['PC1', 'PC2'],
  rotation_method: 'varimax',
  n_factors: 2,
  loadings: [
    { variable: 'AGE', PC1: 0.8, PC2: 0.1, h2: 0.65, u2: 0.35 },
    { variable: 'LDL', PC1: 0.2, PC2: 0.75, h2: 0.6, u2: 0.4 },
    { variable: 'DM', PC1: 0.6, PC2: 0.3, h2: 0.45, u2: 0.55 },
  ],
  scree_coords: [
    { component: 1, eigenvalue: 2.1 },
    { component: 2, eigenvalue: 1.2 },
    { component: 3, eigenvalue: 0.5 },
  ],
  biplot: [
    { variable: 'AGE', x: 0.8, y: 0.1 },
    { variable: 'LDL', x: 0.2, y: 0.75 },
    { variable: 'DM', x: 0.6, y: 0.3 },
  ],
  variance_explained: [
    { component: 1, eigenvalue: 2.1, pct_variance: 42.0, cum_variance: 42.0 },
    { component: 2, eigenvalue: 1.2, pct_variance: 24.0, cum_variance: 66.0 },
    { component: 3, eigenvalue: 0.5, pct_variance: 10.0, cum_variance: 76.0 },
  ],
  export_rows: [['Variable', 'PC1', 'PC2']],
  r_code: 'principal(data, nfactors=2)',
  suitability: {
    bartlett_chi2: 45.2,
    bartlett_df: 3,
    bartlett_p: 0.001,
    overall_kmo: 0.72,
    kmo_rating: 'Middling',
  },
}

describe('FactorPCAPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<FactorPCAPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('requires at least 3 numeric variables before running', async () => {
    installSession(
      makeSession({
        columns: [
          { name: 'AGE', dtype: 'float64', kind: 'numeric' },
          { name: 'LDL', dtype: 'float64', kind: 'numeric' },
          { name: 'DM', dtype: 'int64', kind: 'numeric' },
          { name: 'GROUP', dtype: 'object', kind: 'categorical' },
        ],
      }),
    )
    const user = userEvent.setup()
    render(<FactorPCAPanel />)

    await user.selectOptions(screen.getByRole('listbox'), ['AGE', 'LDL'])
    await user.click(screen.getByRole('button', { name: /run factor analysis/i }))

    await waitFor(() =>
      expect(screen.getByText('Please select at least 3 numeric variables.')).toBeInTheDocument(),
    )
  })

  it('runs PCA extraction and renders suitability, loadings, scree, and biplot tabs', async () => {
    installSession()
    server.use(
      http.post('/api/factor/factor_pca', () => HttpResponse.json(baseResult)),
    )

    const user = userEvent.setup()
    render(<FactorPCAPanel />)

    await user.selectOptions(screen.getByRole('listbox'), ['AGE', 'LDL', 'DM'])
    await user.click(screen.getByRole('button', { name: /run factor analysis/i }))

    await waitFor(() => expect(screen.getByText("Kaiser-Meyer-Olkin (KMO)")).toBeInTheDocument())
    expect(screen.getByText('0.720')).toBeInTheDocument()
    expect(screen.getByText(/Rating: Middling/)).toBeInTheDocument()
    expect(screen.getByText(/Significant: Variables are sufficiently correlated/)).toBeInTheDocument()

    // Loadings Matrix tab
    await user.click(screen.getByRole('button', { name: 'Loadings Matrix' }))
    expect(screen.getByText(/Factor Loadings Matrix \(varimax\)/)).toBeInTheDocument()
    const rows = screen.getAllByRole('row')
    expect(rows.length).toBeGreaterThanOrEqual(4) // header + 3 variables

    // Scree Plot tab
    await user.click(screen.getByRole('button', { name: 'Scree Plot' }))
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()

    // Biplot tab
    await user.click(screen.getByRole('button', { name: 'Loadings Plot (2D)' }))
    expect(screen.getAllByTestId('plotly-mock').length).toBeGreaterThan(0)
  })

  it('runs EFA extraction with manual factor count and shows an error on failure', async () => {
    installSession()
    server.use(
      http.post('/api/factor/factor_pca', () =>
        HttpResponse.json({ detail: 'Insufficient sample size for factor extraction' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<FactorPCAPanel />)

    await user.selectOptions(screen.getByRole('listbox'), ['AGE', 'LDL', 'DM'])
    await user.selectOptions(screen.getByDisplayValue('PCA (Principal Components)'), 'efa')
    await user.click(screen.getByText('Manual'))

    await user.click(screen.getByRole('button', { name: /run factor analysis/i }))

    await waitFor(() =>
      expect(screen.getByText('Insufficient sample size for factor extraction')).toBeInTheDocument(),
    )
  })
})

describe('FactorPCAPanel staleness', () => {
  it('closes the export of a fit once the data changes under it', async () => {
    installSession()
    server.use(http.post('/api/factor/factor_pca', () => HttpResponse.json(baseResult)))

    const user = userEvent.setup()
    render(<FactorPCAPanel />)
    await user.selectOptions(screen.getByRole('listbox'), ['AGE', 'LDL', 'DM'])
    await user.click(screen.getByRole('button', { name: /run factor analysis/i }))

    const csv = await screen.findByRole('button', { name: 'CSV' })
    expect(csv).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'CSV' })).toBeDisabled()
  })
})
