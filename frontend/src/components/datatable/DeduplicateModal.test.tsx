import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { describe, expect, it, vi } from 'vitest'
import type { ColMeta } from '../../store'
import { server } from '../../test/server'
import { DeduplicateModal } from './DeduplicateModal'

const columns: ColMeta[] = [
  { name: 'file_no', dtype: 'int64', kind: 'numeric' },
  { name: 'name', dtype: 'object', kind: 'categorical' },
  { name: 'visit', dtype: 'object', kind: 'categorical' },
]

/** Records every request and answers with a count that depends on the key,
 *  the way the backend does: a narrower key finds more duplicates. */
function stubCounts(counts: Record<string, number> = {}, blankKeyRows = 0) {
  const seen: Array<{ key_columns: string[]; keep: string; dry_run: boolean }> = []
  server.use(
    http.post('/api/compute/test-session/deduplicate', async ({ request }) => {
      const body = (await request.json()) as (typeof seen)[number]
      seen.push(body)
      const n = counts[body.key_columns.join(',')] ?? 0
      return HttpResponse.json({
        deleted: body.dry_run ? 0 : n,
        duplicate_rows: n,
        remaining_rows: 100 - n,
        blank_key_rows: blankKeyRows,
        key_columns: body.key_columns,
        keep: body.keep,
      })
    }),
  )
  return seen
}

function open(props: Partial<Parameters<typeof DeduplicateModal>[0]> = {}) {
  const onDone = vi.fn()
  const onClose = vi.fn()
  render(
    <DeduplicateModal
      columns={columns}
      sessionId="test-session"
      onDone={onDone}
      onClose={onClose}
      {...props}
    />,
  )
  return { onDone, onClose }
}

describe('DeduplicateModal', () => {
  it('counts whole-row duplicates before anything is ticked', async () => {
    const seen = stubCounts({ '': 4 })
    open()

    await waitFor(() => expect(screen.getByText(/4 rows would be deleted/)).toBeInTheDocument())
    // An empty key list is the backend's "identical in every column".
    expect(seen[0]).toMatchObject({ key_columns: [], keep: 'first', dry_run: true })
    expect(seen.every((s) => s.dry_run)).toBe(true)
    expect(screen.getByText(/only rows identical in every column/i)).toBeInTheDocument()
  })

  it('re-counts when the key changes, so the number always describes the current key', async () => {
    // The count is the only thing standing between the user and a delete; a
    // stale one would describe a key they already moved off.
    const seen = stubCounts({ '': 1, file_no: 12 })
    const user = userEvent.setup()
    open()
    await waitFor(() => expect(screen.getByText(/1 row would be deleted/)).toBeInTheDocument())

    await user.click(screen.getByRole('checkbox', { name: /file_no/ }))

    await waitFor(() => expect(screen.getByText(/12 rows would be deleted/)).toBeInTheDocument())
    expect(seen.at(-1)).toMatchObject({ key_columns: ['file_no'], dry_run: true })
  })

  it('deletes with the key on screen and reports the count back', async () => {
    const seen = stubCounts({ 'file_no,name': 7 })
    const user = userEvent.setup()
    const { onDone } = open()

    await user.click(screen.getByRole('checkbox', { name: /file_no/ }))
    await user.click(screen.getByRole('checkbox', { name: /^name/ }))
    await waitFor(() => expect(screen.getByText(/7 rows would be deleted/)).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: /Delete 7 rows/ }))

    await waitFor(() => expect(onDone).toHaveBeenCalledWith(7))
    expect(seen.at(-1)).toMatchObject({ key_columns: ['file_no', 'name'], keep: 'first', dry_run: false })
  })

  it('carries the keep choice into both the count and the delete', async () => {
    const seen = stubCounts({ file_no: 3 })
    const user = userEvent.setup()
    open()
    await user.click(screen.getByRole('checkbox', { name: /file_no/ }))
    await waitFor(() => expect(screen.getByText(/3 rows would be deleted/)).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: /Last occurrence/ }))

    await waitFor(() => expect(seen.at(-1)).toMatchObject({ keep: 'last', dry_run: true }))
    await user.click(screen.getByRole('button', { name: /Delete 3 rows/ }))
    await waitFor(() => expect(seen.at(-1)).toMatchObject({ keep: 'last', dry_run: false }))
  })

  it('offers no delete when the key finds nothing', async () => {
    stubCounts({ '': 0 })
    open()
    await waitFor(() => expect(screen.getByText(/0 rows would be deleted/)).toBeInTheDocument())
    expect(screen.getByRole('button', { name: /Delete 0 rows/ })).toBeDisabled()
  })

  it('says which rows were spared for having no key at all', async () => {
    // Without this the user cannot tell an exemption from a miscount: pandas
    // treats NaN == NaN as equal, so those rows would otherwise collapse.
    stubCounts({ file_no: 2 }, 5)
    const user = userEvent.setup()
    open()
    await user.click(screen.getByRole('checkbox', { name: /file_no/ }))
    await waitFor(() =>
      expect(screen.getByText(/5 rows have no value in any key column/)).toBeInTheDocument(),
    )
  })

  it('pre-ticks the column it was opened from', async () => {
    const seen = stubCounts({ visit: 2 })
    open({ initialKey: 'visit' })
    await waitFor(() => expect(seen[0]).toMatchObject({ key_columns: ['visit'] }))
    expect(screen.getByRole('checkbox', { name: /visit/ })).toBeChecked()
  })

  it('keeps the dataset when counting fails', async () => {
    server.use(
      http.post('/api/compute/test-session/deduplicate', () =>
        HttpResponse.json({ detail: 'nope' }, { status: 500 }),
      ),
    )
    open()
    await waitFor(() =>
      expect(screen.getByText(/Could not count the duplicates/)).toBeInTheDocument(),
    )
    expect(screen.getByRole('button', { name: /Delete 0 rows/ })).toBeDisabled()
  })
})
