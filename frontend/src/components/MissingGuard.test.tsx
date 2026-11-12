import { act, render, screen, waitFor } from '@testing-library/react'
import { http, HttpResponse, delay } from 'msw'
import { describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { MissingGuard } from './MissingGuard'

const missingUrl = '/api/stats/s1/missing'

function guard(columns = ['A']) {
  return (
    <MissingGuard sessionId="s1" columns={columns} imputation="listwise" onImputation={() => {}}>
      <button>Run</button>
    </MissingGuard>
  )
}

function renderGuard() {
  return render(guard())
}

describe('MissingGuard', () => {
  it('ignores an older check that finishes after the selection changed', async () => {
    let finishOldRequest!: () => void
    const oldRequest = new Promise<void>((resolve) => { finishOldRequest = resolve })
    let oldRequestStarted = false
    server.use(http.get(missingUrl, async ({ request }) => {
      if (new URL(request.url).searchParams.get('columns') === 'A') {
        oldRequestStarted = true
        await oldRequest
        return HttpResponse.json({ total_rows: 10, rows_affected: 2, pct_affected: 20, per_column: {} })
      }
      return HttpResponse.json({ total_rows: 10, rows_affected: 1, pct_affected: 10, per_column: {} })
    }))
    const { rerender } = renderGuard()
    await waitFor(() => expect(oldRequestStarted).toBe(true))
    rerender(guard(['B']))
    expect(await screen.findByText(/1 of 10 rows have missing values/)).toBeInTheDocument()
    await act(async () => {
      finishOldRequest()
      await delay(30)
    })
    expect(screen.getByText(/1 of 10 rows have missing values/)).toBeInTheDocument()
  })
  it('shows flagged values separately without offering to impute them', async () => {
    server.use(http.get(missingUrl, () => HttpResponse.json({
      total_rows: 10, rows_affected: 2, pct_affected: 20, raw_rows_affected: 0,
      per_column: { A: { count: 2, pct: 20, raw_count: 0, n_implausible: 2 } },
    })))
    renderGuard()
    expect(await screen.findByText(/2 flagged values/)).toBeInTheDocument()
    expect(screen.queryByText(/rows have missing values/)).toBeNull()
    expect(screen.queryByRole('radio', { name: /Listwise deletion/ })).toBeNull()
  })

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
