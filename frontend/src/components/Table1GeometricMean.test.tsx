import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import Table1Panel from './Table1Panel'

afterEach(() => clearSession())

const GM_WARNING = "'LDL': geometric mean not shown, the variable contains zero or negative values."

const T1_GM_RESULT = {
  group_column: null,
  group_labels: [],
  group_ns: {},
  total_n: 3,
  rows: [
    {
      variable: 'AGE',
      type: 'numeric',
      overall_n: 3,
      stat_rows: [
        { label: 'Geometric mean (GSD)', overall: '54.62 (1.14)', group_stats: {} },
        { label: 'Geometric mean [95% CI]', overall: '54.62 [40.10–74.40]', group_stats: {} },
      ],
      p_value: null,
      test: null,
      group_stats: {},
    },
    {
      variable: 'LDL',
      type: 'numeric',
      overall_n: 3,
      stat_rows: [
        { label: 'Geometric mean (GSD)', overall: '-', group_stats: {} },
      ],
      p_value: null,
      test: null,
      group_stats: {},
    },
  ],
  warnings: [GM_WARNING],
}

describe('Table1Panel geometric mean options', () => {
  it('lists the geometric mean options, sends them and shows the backend warning', async () => {
    installSession()
    let capturedBody: Record<string, unknown> | null = null
    server.use(
      http.post('/api/stats/table1', async ({ request }) => {
        capturedBody = (await request.json()) as Record<string, unknown>
        return HttpResponse.json(T1_GM_RESULT)
      }),
    )

    const user = userEvent.setup()
    render(<Table1Panel />)

    await user.click(screen.getByRole('button', { name: /statistics/i }))
    await user.click(screen.getByRole('checkbox', { name: 'Geometric mean (GSD)' }))
    await user.click(screen.getByRole('checkbox', { name: 'Geometric mean [95% CI]' }))
    await user.click(screen.getByRole('checkbox', { name: /auto \(normality-based\)/i }))

    await user.click(screen.getByRole('button', { name: /generate table/i }))

    await waitFor(() => expect(screen.getAllByText('AGE').length).toBeGreaterThan(0))

    expect(capturedBody).not.toBeNull()
    expect(capturedBody!.selected_stats).toEqual(
      expect.arrayContaining(['geometric_mean', 'gm_ci']),
    )
    expect(capturedBody!.selected_stats).not.toContain('auto')

    expect(screen.getByText('54.62 (1.14)')).toBeInTheDocument()
    expect(screen.getByText('54.62 [40.10–74.40]')).toBeInTheDocument()
    expect(screen.getByText(new RegExp(GM_WARNING.replace(/[[\]()'.]/g, '\\$&')))).toBeInTheDocument()
  })
})
