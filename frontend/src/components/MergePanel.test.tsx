import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import MergePanel from './MergePanel'

afterEach(() => {
  clearSession()
  localStorage.clear()
})

const session = () =>
  makeSession({
    columns: [
      { name: 'pid', dtype: 'int64', kind: 'numeric' },
      { name: 'age', dtype: 'float64', kind: 'numeric' },
      { name: 'sbp', dtype: 'float64', kind: 'numeric' },
    ],
    preview: [{ pid: 1001, age: 61, sbp: 130 }],
  })

const plan = (over: Record<string, unknown> = {}) => ({
  rows_left: 5, rows_right: 4, rows_after: 5,
  keys_matched: 3, left_rows_matched: 3, left_rows_unmatched: 2,
  left_keys_missing: 0, right_keys_missing: 0,
  left_duplicate_keys: 0, right_duplicate_keys: 0, right_keys_unused: 1,
  columns_added: ['crp', 'sbp_2'],
  warnings: ['2 rows in the open dataset found no match and will have the new columns left empty.'],
  ...over,
})

function mockUpload(columns = ['pid', 'crp', 'sbp']) {
  server.use(
    http.post('/api/upload/', () =>
      HttpResponse.json({ session_id: 'other-1', filename: 'labs.csv',
                          columns: columns.map((name) => ({ name, dtype: 'object', kind: 'text' })) }),
    ),
  )
}

function mockPreview(body: Record<string, unknown> = plan()) {
  const seen: { request?: Record<string, unknown> } = {}
  server.use(
    http.post('/api/merge/preview', async ({ request }) => {
      seen.request = (await request.json()) as Record<string, unknown>
      return HttpResponse.json(body)
    }),
  )
  return seen
}

async function attach(user: ReturnType<typeof userEvent.setup>) {
  mockUpload()
  const input = screen.getByLabelText(/file to join/i)
  await user.upload(input, new File(['pid,crp\n1,2\n'], 'labs.csv', { type: 'text/csv' }))
  await waitFor(() => expect(screen.getByText(/labs\.csv/)).toBeInTheDocument())
}

describe('MergePanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<MergePanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('guesses the key from a column name the two files share', async () => {
    installSession(session())
    const user = userEvent.setup()
    render(<MergePanel />)
    await attach(user)

    expect((screen.getByRole('combobox', { name: /key here/i }) as HTMLSelectElement).value).toBe('pid')
    expect((screen.getByRole('combobox', { name: /key in the file/i }) as HTMLSelectElement).value).toBe('pid')
  })

  it('marks a column that would clash with one already here', async () => {
    installSession(session())
    const user = userEvent.setup()
    render(<MergePanel />)
    await attach(user)

    const sbp = screen.getByRole('checkbox', { name: /sbp/ }).closest('label') as HTMLElement
    expect(within(sbp).getByText('clash')).toBeInTheDocument()
  })

  it('shows the row arithmetic before anything is joined', async () => {
    installSession(session())
    mockPreview()
    const user = userEvent.setup()
    render(<MergePanel />)
    await attach(user)
    await user.click(screen.getByRole('button', { name: /check the join/i }))

    await waitFor(() => expect(screen.getByText('What this join would do')).toBeInTheDocument())
    expect(screen.getByText(/found no match/)).toBeInTheDocument()
    expect(screen.getByText('crp, sbp_2')).toBeInTheDocument()
    // Nothing is applied until the second button is pressed.
    expect(screen.getByRole('button', { name: /apply the join/i })).toBeInTheDocument()
  })

  it('warns loudly when duplicate keys would multiply the sheet', async () => {
    installSession(session())
    mockPreview(plan({
      rows_after: null, right_duplicate_keys: 2,
      warnings: ['The incoming file has 2 rows sharing a key that another row already uses. Each match will be repeated once per duplicate, so the result can have more rows than the dataset you started with.'],
    }))
    const user = userEvent.setup()
    render(<MergePanel />)
    await attach(user)
    await user.click(screen.getByRole('button', { name: /check the join/i }))

    await waitFor(() => expect(screen.getByText(/more rows than the dataset you started with/)).toBeInTheDocument())
    expect(screen.getByText('depends on duplicates')).toBeInTheDocument()
  })

  it('sends the chosen keys, join type and columns', async () => {
    installSession(session())
    const seen = mockPreview()
    const user = userEvent.setup()
    render(<MergePanel />)
    await attach(user)
    await user.selectOptions(screen.getByRole('combobox', { name: /rows to keep/i }), 'outer')
    await user.click(screen.getByRole('checkbox', { name: /crp/ }))
    await user.click(screen.getByRole('button', { name: /check the join/i }))

    await waitFor(() => expect(seen.request).toBeTruthy())
    expect(seen.request).toMatchObject({
      other_session_id: 'other-1', left_on: ['pid'], right_on: ['pid'],
      how: 'outer', columns: ['crp'],
    })
  })

  it('applies the join and refreshes the sheet', async () => {
    installSession(session())
    mockPreview()
    server.use(
      http.post('/api/merge/apply', () =>
        HttpResponse.json(plan({ result_text: 'Joined 2 column(s) onto the dataset by pid: 3 of 5 rows matched.' })),
      ),
      http.get('/api/sessions/test-session', () =>
        HttpResponse.json({ ...session(), columns: [...session().columns, { name: 'crp', dtype: 'float64', kind: 'numeric' }] }),
      ),
    )
    const user = userEvent.setup()
    render(<MergePanel />)
    await attach(user)
    await user.click(screen.getByRole('button', { name: /check the join/i }))
    await waitFor(() => expect(screen.getByRole('button', { name: /apply the join/i })).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: /apply the join/i }))

    await waitFor(() => expect(screen.getByText('Joined')).toBeInTheDocument())
    expect(screen.getByText(/3 of 5 rows matched/)).toBeInTheDocument()
    // The apply button is gone: there is nothing left to confirm.
    expect(screen.queryByRole('button', { name: /apply the join/i })).not.toBeInTheDocument()
  })

  it('changing a key clears a stale preview', async () => {
    // Otherwise the numbers on screen describe a join the user has moved on from.
    installSession(session())
    mockPreview()
    const user = userEvent.setup()
    render(<MergePanel />)
    await attach(user)
    await user.click(screen.getByRole('button', { name: /check the join/i }))
    await waitFor(() => expect(screen.getByText('What this join would do')).toBeInTheDocument())

    await user.selectOptions(screen.getByRole('combobox', { name: /key here/i }), 'age')
    expect(screen.queryByText('What this join would do')).not.toBeInTheDocument()
  })

  it('shows the backend refusal', async () => {
    installSession(session())
    server.use(
      http.post('/api/merge/preview', () =>
        HttpResponse.json({ detail: 'The incoming file has no columns to add besides the key' }, { status: 400 }),
      ),
    )
    const user = userEvent.setup()
    render(<MergePanel />)
    await attach(user)
    await user.click(screen.getByRole('button', { name: /check the join/i }))

    await waitFor(() => expect(screen.getByText(/no columns to add/)).toBeInTheDocument())
  })
})
