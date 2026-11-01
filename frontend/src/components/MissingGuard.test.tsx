import { render, screen, waitFor } from '@testing-library/react'
import { http, HttpResponse, delay } from 'msw'
import { describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { MissingGuard } from './MissingGuard'

const missingUrl = '/api/stats/s1/missing'

function renderGuard() {
  return render(
    <MissingGuard sessionId="s1" columns={['A']} imputation="listwise" onImputation={() => {}}>
      <button>Run</button>
    </MissingGuard>,
  )
}

describe('MissingGuard', () => {
  it('keeps the same run button node when the check finishes clean', async () => {
    server.use(http.get(missingUrl, async () => {
      await delay(20)
      return HttpResponse.json({ total_rows: 3, rows_affected: 0, pct_affected: 0, per_column: {} })
    }))
    renderGuard()
    expect(await screen.findByText('Checking missing data…')).toBeInTheDocument()
    const before = screen.getByRole('button', { name: 'Run' })
    await waitFor(() => expect(screen.queryByText('Checking missing data…')).not.toBeInTheDocument())
    // A remount would leave `before` detached, and a click on it would be lost.
    expect(screen.getByRole('button', { name: 'Run' })).toBe(before)
    expect(before.isConnected).toBe(true)
  })

  it('keeps the same run button node when missing values are found', async () => {
    server.use(http.get(missingUrl, async () => {
      await delay(20)
      return HttpResponse.json({
        total_rows: 10, rows_affected: 3, pct_affected: 30, per_column: { A: { count: 3, pct: 30 } },
      })
    }))
    renderGuard()
    const before = await screen.findByRole('button', { name: 'Run' })
    expect(await screen.findByText(/3 of 10 rows have missing values/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Run' })).toBe(before)
    expect(screen.getByRole('radio', { name: /Listwise deletion/ })).toBeChecked()
  })
})
