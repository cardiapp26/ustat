import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import type { ColMeta, Session } from '../store'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import DataTable from './DataTable'
import { useStore } from '../store'

afterEach(() => clearSession())

/** A sheet of `n` rows over `cols` numeric columns. */
function bigSession(n: number, cols = 6): Session {
  const columns: ColMeta[] = Array.from({ length: cols }, (_, j) => ({
    name: `c${j}`,
    dtype: 'float64',
    kind: 'numeric',
  }))
  const preview = Array.from({ length: n }, (_, i) =>
    Object.fromEntries(columns.map((c, j) => [c.name, i * 100 + j])),
  )
  return makeSession({ columns, preview, rows: n })
}

const bodyRows = () => document.querySelectorAll('tbody tr[class*="group"]').length

describe('DataTable row virtualisation', () => {
  it('re-asks for the excluded rows when the data underneath changes', async () => {
    // Excluded rows are POSITIONS, so a deletion shifts every one below it and
    // stale marks would shade the wrong records. The filter itself stays
    // correct — its conditions are evaluated on values — but the shading has
    // to be restated.
    installSession(bigSession(10))
    let calls = 0
    server.use(
      http.post('/api/sessions/test-session/select_cases', () => {
        calls += 1
        return HttpResponse.json({
          selected: 8, total: 10, applied: true,
          excluded_rows: [1, 2], excluded_beyond_preview: 0,
        })
      }),
    )
    useStore.setState({
      caseFilter: {
        conditions: [{ column: 'c0', operator: 'eq', value: '0', join: 'AND' }],
        selected: 7, total: 10, excludedRows: [3, 5, 9], excludedBeyondPreview: 0,
      },
    })
    render(<DataTable />)

    await waitFor(() => expect(calls).toBeGreaterThan(0))
    await waitFor(() =>
      expect(useStore.getState().caseFilter?.excludedRows).toEqual([1, 2]),
    )
  })

  it('does not ask when no filter is active', async () => {
    installSession(bigSession(10))
    let calls = 0
    server.use(
      http.post('/api/sessions/test-session/select_cases', () => {
        calls += 1
        return HttpResponse.json({ selected: 10, total: 10, applied: true })
      }),
    )
    render(<DataTable />)
    await new Promise((r) => setTimeout(r, 50))
    expect(calls).toBe(0)
  })

  it('bands the row the cursor is in', async () => {
    // On a wide sheet the cell being edited scrolls out of sight long before
    // its row does; without a band nothing ties the values under the cursor
    // to the row they belong to.
    installSession(bigSession(12, 8))
    const user = userEvent.setup()
    render(<DataTable />)
    const rows = () => [...document.querySelectorAll('tbody tr[class*="group"]')]
    expect(rows().filter((r) => r.className.includes('bg-indigo-50'))).toHaveLength(0)

    const cells = rows()[4].querySelectorAll('td')
    await user.click(cells[2])

    const banded = rows().filter((r) => r.className.includes('bg-indigo-50'))
    expect(banded).toHaveLength(1)
    expect(rows().indexOf(banded[0])).toBe(4)
  })

  it('marks rows Select Cases excluded rather than hiding them by default', () => {
    // Hiding is opt-in ("Only selected"), not the default: a sheet that
    // silently drops rows disagrees with the file the user loaded, for a
    // filter they may have set an hour ago.
    installSession(bigSession(10))
    // The grid restates the excluded positions whenever the data could have
    // moved under them, so the handler has to exist even when the test is
    // about the marking rather than the refresh. It answers with the same
    // rows so nothing shifts.
    server.use(
      http.post('/api/sessions/test-session/select_cases', () =>
        HttpResponse.json({
          selected: 7, total: 10, applied: true,
          excluded_rows: [3, 5, 9], excluded_beyond_preview: 0,
        }),
      ),
    )
    useStore.setState({
      caseFilter: {
        conditions: [{ column: 'c0', operator: 'eq', value: '0', join: 'AND' }],
        selected: 7, total: 10,
        excludedRows: [3, 5, 9],
        excludedBeyondPreview: 0,
      },
    })
    render(<DataTable />)

    const rows = [...document.querySelectorAll('tbody tr[class*="group"]')]
    expect(rows).toHaveLength(10)
    const faded = rows.filter((r) => r.className.includes('opacity-40'))
    expect(faded).toHaveLength(3)
    // The row number carries the strike, the way SPSS marks a filtered case.
    expect(document.querySelectorAll('.line-through')).toHaveLength(3)
  })

  it('sinks Select Cases excluded rows to the bottom of a sort', async () => {
    // A sort that treats excluded rows as ordinary ones scatters the cases the
    // filter just set aside back through the sheet, and the selection stops
    // reading as a selection. They sink instead, in their original order.
    installSession(bigSession(6))
    server.use(
      http.post('/api/sessions/test-session/select_cases', () =>
        HttpResponse.json({
          selected: 4, total: 6, applied: true,
          excluded_rows: [1, 2], excluded_beyond_preview: 0,
        }),
      ),
    )
    useStore.setState({
      caseFilter: {
        conditions: [{ column: 'c0', operator: 'eq', value: '0', join: 'AND' }],
        selected: 4, total: 6,
        excludedRows: [1, 2],
        excludedBeyondPreview: 0,
      },
    })
    const user = userEvent.setup()
    render(<DataTable />)

    // c0 already ascends, so the sink is the only thing the sort can change.
    const firstCol = () =>
      [...document.querySelectorAll('tbody tr[class*="group"]')].map(
        (r) => r.querySelectorAll('td')[1].textContent,
      )
    expect(firstCol()).toEqual(['0', '100', '200', '300', '400', '500'])

    await user.click(screen.getAllByTitle('Sort ascending')[0])

    await waitFor(() =>
      expect(firstCol()).toEqual(['0', '300', '400', '500', '100', '200']),
    )
  })

  it('nests a later sort inside the ties of the earlier one', async () => {
    // The first column sorted stays the primary key. Sorting a second column
    // orders the rows within the ties of the first, so the earlier sort is
    // never disturbed by a later one.
    installSession(makeSession({
      columns: [
        { name: 'a', dtype: 'int64', kind: 'numeric' },
        { name: 'b', dtype: 'int64', kind: 'numeric' },
      ],
      preview: [
        { a: 1, b: 1 }, { a: 2, b: 1 }, { a: 3, b: 2 }, { a: 1, b: 2 }, { a: 2, b: 2 },
      ],
      rows: 5,
    }))
    const user = userEvent.setup()
    render(<DataTable />)
    const col = (i: number) => () =>
      [...document.querySelectorAll('tbody tr[class*="group"]')].map(
        (r) => r.querySelectorAll('td')[i].textContent,
      )
    const colA = col(1), colB = col(2)
    const desc = () => screen.getAllByTitle('Sort descending')
    const asc = () => screen.getAllByTitle('Sort ascending')

    // a ▼: ties keep file order
    await user.click(desc()[0])
    await waitFor(() => expect(colA()).toEqual(['3', '2', '2', '1', '1']))
    expect(colB()).toEqual(['2', '1', '2', '1', '2'])

    // then b ▼: a is untouched, b descends inside each a group
    await user.click(desc()[1])
    await waitFor(() => expect(colB()).toEqual(['2', '2', '1', '2', '1']))
    expect(colA()).toEqual(['3', '2', '2', '1', '1'])
    expect(screen.getByText(/Sort: a ▼/)).toHaveTextContent('then b ▼')
    expect(desc()[1]).toHaveAttribute('aria-pressed', 'true')

    // b ▲ flips b in place; a still leads
    await user.click(asc()[1])
    await waitFor(() => expect(colB()).toEqual(['2', '1', '2', '1', '2']))
    expect(colA()).toEqual(['3', '2', '2', '1', '1'])

    // pressing the active direction again drops b; a ▼ alone remains
    await user.click(asc()[1])
    await waitFor(() => expect(asc()[1]).toHaveAttribute('aria-pressed', 'false'))
    expect(colA()).toEqual(['3', '2', '2', '1', '1'])
    expect(screen.getByText(/Sort: a ▼/)).not.toHaveTextContent('then')
  })

  /** A 6-row sheet with rows 1 and 2 excluded by an active Select Cases. */
  function excludedSession() {
    installSession(bigSession(6))
    server.use(
      http.post('/api/sessions/test-session/select_cases', () =>
        HttpResponse.json({
          selected: 4, total: 6, applied: true,
          excluded_rows: [1, 2], excluded_beyond_preview: 0,
        }),
      ),
    )
    useStore.setState({
      caseFilter: {
        conditions: [{ column: 'c0', operator: 'eq', value: '0', join: 'AND' }],
        selected: 4, total: 6,
        excludedRows: [1, 2],
        excludedBeyondPreview: 0,
      },
    })
  }

  const firstColValues = () =>
    [...document.querySelectorAll('tbody tr[class*="group"]')].map(
      (r) => r.querySelectorAll('td')[1].textContent,
    )

  it('hides the excluded rows once Only selected is on', async () => {
    excludedSession()
    const user = userEvent.setup()
    render(<DataTable />)
    expect(firstColValues()).toHaveLength(6)

    await user.click(screen.getByRole('button', { name: /only selected/i }))

    await waitFor(() =>
      expect(firstColValues()).toEqual(['0', '300', '400', '500']),
    )
    // Nothing left to strike through: the excluded rows are gone, not faded.
    expect(document.querySelectorAll('.line-through')).toHaveLength(0)

    await user.click(screen.getByRole('button', { name: /only selected/i }))
    await waitFor(() => expect(firstColValues()).toHaveLength(6))
  })

  it('still addresses hidden-away rows by their unfiltered position', async () => {
    // The reason the grid did not hide rows before: cell edits carry a row
    // index into the UNFILTERED frame. If hiding renumbered them, every edit
    // below the first hidden row would land on the wrong record.
    excludedSession()
    let patched: { row_index: number; column: string; value: unknown } | null = null
    server.use(
      http.patch('/api/sessions/test-session/cell', async ({ request }) => {
        patched = (await request.json()) as typeof patched
        return HttpResponse.json({ value: 999 })
      }),
    )
    const user = userEvent.setup()
    render(<DataTable />)
    await user.click(screen.getByRole('button', { name: /only selected/i }))
    await waitFor(() => expect(firstColValues()).toHaveLength(4))

    // Second visible row is original row 3 — rows 1 and 2 are hidden.
    const rows = [...document.querySelectorAll('tbody tr[class*="group"]')]
    await user.dblClick(rows[1].querySelectorAll('td')[1])
    await user.keyboard('999{Enter}')

    await waitFor(() => expect(patched).not.toBeNull())
    expect(patched!.row_index).toBe(3)
    expect(patched!.column).toBe('c0')
  })

  it('opens the duplicate-row cleanup from the row menu and refreshes after it', async () => {
    // The menu is where a user notices duplicates — they are looking at two
    // identical rows when they right-click.
    installSession(bigSession(6))
    let refreshed = 0
    server.use(
      http.post('/api/compute/test-session/deduplicate', async ({ request }) => {
        const body = (await request.json()) as { dry_run: boolean }
        return HttpResponse.json({
          deleted: body.dry_run ? 0 : 2,
          duplicate_rows: 2, remaining_rows: 4, blank_key_rows: 0,
          key_columns: [], keep: 'first',
        })
      }),
      http.get('/api/stats/test-session/refresh', () => {
        refreshed += 1
        return HttpResponse.json({ rows: 4, columns: [], preview: [] })
      }),
    )
    const user = userEvent.setup()
    render(<DataTable />)

    const gutter = document.querySelectorAll('tbody tr[class*="group"]')[0].querySelectorAll('td')[0]
    fireEvent.contextMenu(gutter)
    await user.click(screen.getByText(/Delete duplicates/))

    await waitFor(() =>
      expect(screen.getByText(/2 rows would be deleted/)).toBeInTheDocument(),
    )
    await user.click(screen.getByRole('button', { name: /Delete 2 rows/ }))

    // Ticks and selections are stored as row positions, so dropping rows has
    // to be followed by a reload rather than a re-render of stale positions.
    await waitFor(() => expect(refreshed).toBe(1))
    expect(screen.queryByText(/would be deleted/)).not.toBeInTheDocument()
  })

  it('leaves every row alone when no filter is set', () => {
    installSession(bigSession(10))
    render(<DataTable />)
    expect(document.querySelectorAll('.line-through')).toHaveLength(0)
    expect(document.querySelectorAll('[class*="opacity-40"]')).toHaveLength(0)
  })

  it('renders every row for a small sheet', () => {
    installSession(bigSession(40))
    render(<DataTable />)
    expect(bodyRows()).toBe(40)
  })

  it('renders only a window for a large sheet', () => {
    // 1000 x 125 was ~126,000 cells and ~6s to become usable; only the visible
    // slice (plus overscan) should reach the DOM.
    installSession(bigSession(1000))
    render(<DataTable />)
    const rendered = bodyRows()
    expect(rendered).toBeGreaterThan(0)
    expect(rendered).toBeLessThan(120)
  })

  it('keeps the scroll height correct with spacer rows', () => {
    installSession(bigSession(1000))
    render(<DataTable />)
    const spacers = [...document.querySelectorAll('tbody tr[aria-hidden="true"] td')]
    expect(spacers.length).toBeGreaterThan(0)
    // Spacers stand in for the rows that were not rendered: 33px each.
    const padded = spacers.reduce((sum, td) => {
      const h = parseFloat((td as HTMLElement).style.height || '0')
      return sum + h
    }, 0)
    expect(padded / 33 + bodyRows()).toBeGreaterThanOrEqual(1000)
  })

  it('numbers the visible rows by their absolute position', () => {
    installSession(bigSession(1000))
    render(<DataTable />)
    // The first rendered row is row 1 at the top of an unscrolled grid.
    expect(screen.getByTitle(/Original row #1\b/)).toBeInTheDocument()
  })

  it('keeps every row exactly one line tall, whatever the cell holds', () => {
    // Reported: scrolling a 560-row sheet juddered and snapped back to the
    // same rows. The sheet had a free-text notes column; its cells wrapped, so
    // rows were 45, 125, even 157px against the 33px the window arithmetic
    // assumes. padTop + rendered + padBottom then stopped matching the real
    // scroll height, the container resized under its own scrollbar as rows
    // swapped in and out (34 distinct scrollHeights over 40 scroll steps), and
    // the browser clamped scrollTop back.
    //
    // jsdom does no layout, so this pins the cause rather than the pixels: the
    // row carries a fixed height and the cells cannot wrap.
    const columns: ColMeta[] = [
      { name: 'id', dtype: 'int64', kind: 'numeric' },
      { name: 'Notlar', dtype: 'object', kind: 'text' },
    ]
    const long = 'Hasta postoperatif donemde sorunsuz izlendi, komplikasyon gelismedi, '
      + 'poliklinik takibine alindi ve iki hafta sonraya randevu verildi'
    const preview = Array.from({ length: 400 }, (_, i) => ({ id: i, Notlar: long }))
    installSession(makeSession({ columns, preview, rows: 400 }))
    render(<DataTable />)

    const rows = [...document.querySelectorAll('tbody tr[class*="group"]')] as HTMLElement[]
    expect(rows.length).toBeGreaterThan(0)
    expect(rows.every((r) => r.style.height === '33px')).toBe(true)

    const cells = [...rows[0].querySelectorAll('td')] as HTMLElement[]
    expect(cells.every((td) => td.className.includes('whitespace-nowrap'))).toBe(true)
  })

  it('clips a long value but keeps it readable on hover', () => {
    const columns: ColMeta[] = [{ name: 'Notlar', dtype: 'object', kind: 'text' }]
    const long = 'Tiroidektomi sonrasi hipokalsemi gelisti, kalsiyum replasmani baslandi'
    installSession(makeSession({ columns, preview: [{ Notlar: long }], rows: 1 }))
    render(<DataTable />)

    const span = screen.getByTitle(long)
    expect(span.className).toContain('truncate')
  })

  it('shows the empty state when every row is filtered out', () => {
    installSession(makeSession({ columns: [], preview: [], rows: 0 }))
    render(<DataTable />)
    expect(bodyRows()).toBe(0)
  })
})

describe('DataTable column-name tooltip', () => {
  /** jsdom reports 0 for both widths, so truncation has to be simulated. */
  function withClippedText(clipped: boolean) {
    const proto = window.HTMLElement.prototype
    const scroll = Object.getOwnPropertyDescriptor(proto, 'scrollWidth')
    const client = Object.getOwnPropertyDescriptor(proto, 'clientWidth')
    Object.defineProperty(proto, 'scrollWidth', { configurable: true, get: () => (clipped ? 300 : 100) })
    Object.defineProperty(proto, 'clientWidth', { configurable: true, get: () => 100 })
    return () => {
      if (scroll) Object.defineProperty(proto, 'scrollWidth', scroll)
      if (client) Object.defineProperty(proto, 'clientWidth', client)
    }
  }

  const longName = 'Cardiac death-follow up (in-hospital and 30 day)'

  function sessionWithLongName(): Session {
    const columns: ColMeta[] = [{ name: longName, dtype: 'int64', kind: 'categorical' }]
    return makeSession({ columns, preview: [{ [longName]: 1 }], rows: 1 })
  }

  it('shows the full name when the header text is clipped', async () => {
    const restore = withClippedText(true)
    try {
      installSession(sessionWithLongName())
      const { default: userEvent } = await import('@testing-library/user-event')
      const user = userEvent.setup()
      render(<DataTable />)

      expect(screen.queryByRole('tooltip')).toBeNull()
      // Grab the header span first: once the tooltip opens it carries the
      // same text, so a fresh query would match both.
      const header = screen.getByText(longName)
      await user.hover(header)
      expect(screen.getByRole('tooltip')).toHaveTextContent(longName)

      await user.unhover(header)
      expect(screen.queryByRole('tooltip')).toBeNull()
    } finally {
      restore()
    }
  })

  it('stays quiet when the name already fits', async () => {
    const restore = withClippedText(false)
    try {
      installSession(sessionWithLongName())
      const { default: userEvent } = await import('@testing-library/user-event')
      const user = userEvent.setup()
      render(<DataTable />)

      await user.hover(screen.getByText(longName))
      expect(screen.queryByRole('tooltip')).toBeNull()
    } finally {
      restore()
    }
  })
})

describe('DataTable filter visibility', () => {
  /** Three rows, one of which has a missing cell. */
  function sessionWithMissing(): Session {
    const columns: ColMeta[] = [
      { name: 'id', dtype: 'int64', kind: 'numeric' },
      { name: 'ldl', dtype: 'float64', kind: 'numeric' },
    ]
    const preview = [
      { id: 1, ldl: 3.1 },
      { id: 2, ldl: null },
      { id: 3, ldl: 2.4 },
    ]
    return makeSession({ columns, preview, rows: 3 })
  }

  it('says the rows are filtered and clears it in one click', async () => {
    // A column's missing badge turns on the global missing-only filter and
    // adds a filter on that column, so one click on a small badge in a header
    // could drop the row count with nothing next to it saying why.
    installSession(sessionWithMissing())
    const user = userEvent.setup()
    render(<DataTable />)
    expect(bodyRows()).toBe(3)
    expect(screen.queryByRole('button', { name: /filtered/i })).not.toBeInTheDocument()

    await user.click(screen.getByTitle(/click to show only those rows/i))
    expect(bodyRows()).toBe(1)

    const clear = screen.getByRole('button', { name: /filtered/i })
    await user.click(clear)
    expect(bodyRows()).toBe(3)
    expect(screen.queryByRole('button', { name: /filtered/i })).not.toBeInTheDocument()
  })
})

describe('DataTable cell selection actions', () => {
  function sheet(): Session {
    const columns: ColMeta[] = [
      { name: 'id', dtype: 'int64', kind: 'numeric' },
      { name: 'grp', dtype: 'object', kind: 'categorical' },
    ]
    const preview = [
      { id: 1, grp: 'a' },
      { id: 2, grp: '' },
      { id: 3, grp: 'a' },
    ]
    return makeSession({ columns, preview, rows: 3 })
  }

  /** Right-click the first body cell after selecting it. */
  async function openCellMenu(user: ReturnType<typeof userEvent.setup>) {
    const cell = document.querySelectorAll('tbody tr td')[1] as HTMLElement
    await user.click(cell)
    fireEvent.contextMenu(cell)
  }

  it('offers Convert value and Fill blanks on a cell selection', async () => {
    installSession(sheet())
    const user = userEvent.setup()
    render(<DataTable />)

    await openCellMenu(user)
    expect(await screen.findByText(/Convert value/)).toBeInTheDocument()
    expect(screen.getByText(/Fill blanks with/)).toBeInTheDocument()
  })

  it('says what Convert will do, and changes it when a From value is typed', async () => {
    installSession(sheet())
    const user = userEvent.setup()
    render(<DataTable />)

    await openCellMenu(user)
    await user.click(await screen.findByText(/Convert value/))

    // With From empty the whole selection is written; that is a different
    // action from a recode and the dialog has to say which one it is.
    expect(screen.getByText(/Every selected cell is set to this value/)).toBeInTheDocument()
    await user.type(screen.getByLabelText('Convert from'), 'a')
    expect(screen.getByText(/Only selected cells holding that value are changed/)).toBeInTheDocument()
  })

  it('Fill blanks says it leaves everything else alone', async () => {
    installSession(sheet())
    const user = userEvent.setup()
    render(<DataTable />)

    await openCellMenu(user)
    await user.click(await screen.findByText(/Fill blanks with/))

    expect(screen.getByLabelText('Fill value')).toBeInTheDocument()
    expect(screen.queryByLabelText('Convert from')).not.toBeInTheDocument()
    expect(screen.getByText(/Only cells that are currently empty are written/)).toBeInTheDocument()
  })

  it('closes the dialog on Cancel without writing anything', async () => {
    installSession(sheet())
    const user = userEvent.setup()
    render(<DataTable />)

    await openCellMenu(user)
    await user.click(await screen.findByText(/Fill blanks with/))
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() =>
      expect(screen.queryByLabelText('Fill value')).not.toBeInTheDocument(),
    )
  })
})

describe('DataTable column missing-value actions', () => {
  it('offers fill methods when the full-data server count finds missing sentinels', async () => {
    installSession(makeSession({
      columns: [{ name: 'CREATININ', dtype: 'float64', kind: 'numeric' }],
      preview: [
        { CREATININ: 0.62 },
        { CREATININ: 999 },
        { CREATININ: 1.17 },
      ],
      rows: 3,
    }))
    server.use(
      http.get('/api/stats/test-session/column_badges', () =>
        HttpResponse.json({
          n_rows: 3,
          columns: {
            CREATININ: { n_missing: 1, pct_missing: 33.3, min: 0.62, max: 1.17, n_valid: 2 },
          },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<DataTable />)

    await screen.findByTitle(/1 missing value/)
    fireEvent.contextMenu(screen.getByText('CREATININ').closest('th')!)

    const fillGroup = await screen.findByRole('button', { name: /Fill 1 blanks/i })
    await user.hover(fillGroup)
    expect(await screen.findByRole('button', { name: '📊 Mean' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '📊 Median' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /MICE \(multiple imputation\)/i })).toBeInTheDocument()
  })
})
