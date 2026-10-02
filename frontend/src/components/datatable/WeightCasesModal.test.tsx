import { afterEach, describe, expect, it } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../../test/server'
import { clearSession, installSession } from '../../test/testUtils'
import { useStore } from '../../store'
import WeightCasesModal from './WeightCasesModal'

afterEach(() => clearSession())

const columns = [
  { name: 'sbp', dtype: 'float64', kind: 'numeric' as const },
  { name: 'count', dtype: 'int64', kind: 'numeric' as const },
  { name: 'arm', dtype: 'object', kind: 'categorical' as const },
]

describe('WeightCasesModal', () => {
  it('applies the weight column and records it in the store', async () => {
    installSession()
    server.use(http.post('/api/sessions/:sid/weight_cases', () =>
      HttpResponse.json({ column: 'count', n_rows: 6, n_excluded: 0, sum_weights: 13 })))
    const user = userEvent.setup()
    render(<WeightCasesModal columns={columns} sessionId="s1" current={null} onClose={() => {}} />)
    expect(screen.queryByRole('option', { name: 'arm' })).toBeNull()
    await user.selectOptions(screen.getByRole('combobox'), 'count')
    await user.click(screen.getByRole('button', { name: /weight by this column/i }))
    await waitFor(() => expect(useStore.getState().caseWeight?.sum_weights).toBe(13))
  })

  it("shows the server's reason when the weights are refused", async () => {
    installSession()
    server.use(http.post('/api/sessions/:sid/weight_cases', () =>
      HttpResponse.json({ detail: "'count' holds fractional weights." }, { status: 422 })))
    const user = userEvent.setup()
    render(<WeightCasesModal columns={columns} sessionId="s1" current={null} onClose={() => {}} />)
    await user.click(screen.getByRole('button', { name: /weight by this column/i }))
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('fractional'))
    expect(useStore.getState().caseWeight).toBeNull()
  })
})

describe('store: the weight follows its column', () => {
  it('renames with the column and clears when the column is dropped', () => {
    installSession()
    const { session } = useStore.getState()
    useStore.setState({
      session: { ...session!, columns: [...session!.columns, { name: 'count', dtype: 'int64', kind: 'numeric' }] },
      caseWeight: { column: 'count', sum_weights: 13 },
    })
    useStore.getState().renameSessionColumn('count', 'n')
    expect(useStore.getState().caseWeight?.column).toBe('n')
    useStore.getState().removeSessionColumns(['n'])
    expect(useStore.getState().caseWeight).toBeNull()
  })
})
