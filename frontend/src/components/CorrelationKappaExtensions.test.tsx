import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import CorrelationPanel from './CorrelationPanel'

afterEach(() => clearSession())

const base = {
  n: 40,
  interpretation: 'Substantial',
  labels: ['A', 'B'],
  confusion_matrix: [[18, 2], [3, 17]],
}

async function openKappa(user: ReturnType<typeof userEvent.setup>) {
  render(<CorrelationPanel />)
  await user.click(screen.getByRole('button', { name: "Cohen's κ" }))
}

describe('Cohen kappa uncertainty', () => {
  it('weighted kappa shows SE, 95% CI, z and p, and the uncertainty note', async () => {
    installSession()
    let sent: Record<string, unknown> | undefined
    server.use(
      http.post('/api/stats/cohens_kappa', async ({ request }) => {
        sent = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({
          ...base,
          kappa: 0.8123,
          weights: 'quadratic',
          se: 0.0712,
          ci_low: 0.6728,
          ci_high: 0.9518,
          se_null: 0.1034,
          z: 7.857,
          p: 0.0000000001,
          po: 0.9123,
          pe: 0.5321,
          uncertainty_note: 'Fleiss-Cohen-Everitt variance with quadratic agreement weights.',
        })
      }),
    )
    const user = userEvent.setup()
    await openKappa(user)
    await user.selectOptions(screen.getByRole('option', { name: /Quadratic/ }).closest('select')!, 'quadratic')
    await user.type(screen.getByLabelText('Ordered kappa levels'), 'A,B')
    await user.click(screen.getByRole('button', { name: 'Compute' }))

    await screen.findByText('0.812')
    expect(sent?.weights).toBe('quadratic')
    expect(screen.getByText('[0.673, 0.952]')).toBeInTheDocument()
    expect(screen.getByText('0.0712')).toBeInTheDocument()
    expect(screen.getByText('7.857')).toBeInTheDocument()
    expect(screen.getByText('<0.001')).toBeInTheDocument()
    expect(screen.getByText('0.912 / 0.532')).toBeInTheDocument()
    expect(screen.getByText(/Fleiss-Cohen-Everitt variance/)).toBeInTheDocument()
    expect(screen.queryByText('Not estimated')).not.toBeInTheDocument()
  })

  it('unweighted kappa display keeps SE and CI but adds no z/p rows', async () => {
    installSession()
    server.use(
      http.post('/api/stats/cohens_kappa', () =>
        HttpResponse.json({
          ...base,
          kappa: 0.65,
          weights: 'none',
          se: 0.15,
          ci_low: 0.3,
          ci_high: 0.9,
          se_null: 0.1,
          z: 6.5,
          p: 0.001,
          po: 0.9,
          pe: 0.5,
          uncertainty_note: null,
        }),
      ),
    )
    const user = userEvent.setup()
    await openKappa(user)
    await user.click(screen.getByRole('button', { name: 'Compute' }))

    await screen.findByText('0.650')
    expect(screen.getByText('[0.300, 0.900]')).toBeInTheDocument()
    expect(screen.getByText('0.1500')).toBeInTheDocument()
    expect(screen.queryByText(/Test of κ = 0/)).not.toBeInTheDocument()
    expect(screen.queryByText('Observed / expected agreement:')).not.toBeInTheDocument()
  })
})
