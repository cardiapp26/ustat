import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import DataDictionaryPanel from './DataDictionaryPanel'

vi.mock('xlsx', () => ({
  read: () => ({ SheetNames: ['Dictionary'], Sheets: { Dictionary: {} } }),
  utils: {
    sheet_to_json: () => [
      ['Group', 'Diabetes Mellitus'],
      ['Control:0', 'No:0'],
      ['Patient:1', 'Yes:1'],
    ],
  },
}))

afterEach(() => clearSession())

describe('DataDictionaryPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<DataDictionaryPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('lists all session columns as dictionary rows', () => {
    installSession()
    render(<DataDictionaryPanel />)
    expect(screen.getByText('AGE')).toBeInTheDocument()
    expect(screen.getByText('LDL')).toBeInTheDocument()
    expect(screen.getByText('GROUP')).toBeInTheDocument()
    expect(
      screen.getByText(
        (_, el) => el?.tagName === 'P' && /4 variables/.test(el?.textContent ?? '') && /3 observations/.test(el?.textContent ?? ''),
      ),
    ).toBeInTheDocument()
  })

  it('lets the user edit a label and role for a column', async () => {
    installSession()
    const user = userEvent.setup()
    render(<DataDictionaryPanel />)

    const row = screen.getByText('AGE').closest('tr')!
    const labelInput = within(row).getByPlaceholderText(/Variable label/)
    await user.type(labelInput, 'Age in years')
    expect(labelInput).toHaveValue('Age in years')

    const roleSelect = within(row).getByRole('combobox')
    await user.selectOptions(roleSelect, 'covariate')
    expect(roleSelect).toHaveValue('covariate')
  })

  it('shows imported SPSS dictionary metadata', () => {
    installSession(makeSession({
      columns: [
        {
          name: 'Grup',
          dtype: 'float64',
          kind: 'categorical',
          label: 'Patient control group',
          value_labels: { '0': 'Hasta', '1': 'Kontrol', '9': 'Cevapsiz' },
          missing_ranges: [{ lo: 9, hi: 9 }],
          measure: 'nominal',
        },
      ],
      preview: [{ Grup: 0 }, { Grup: 1 }, { Grup: null }],
    }))

    render(<DataDictionaryPanel />)
    const row = screen.getByText('Grup').closest('tr')!
    expect(within(row).getByDisplayValue('Patient control group')).toBeInTheDocument()
    expect(within(row).getByText('nominal')).toBeInTheDocument()
    expect(within(row).getByText('9')).toBeInTheDocument()
    expect(within(row).getByRole('button', { name: /3 labels/i })).toBeInTheDocument()
  })

  it('saves metadata and shows the saved confirmation on success', async () => {
    installSession()
    server.use(
      http.post('/api/sessions/test-session/metadata', () => HttpResponse.json({ ok: true })),
    )

    const user = userEvent.setup()
    render(<DataDictionaryPanel />)

    await user.click(screen.getByRole('button', { name: /save metadata/i }))

    await waitFor(() => expect(screen.getByRole('button', { name: /saved/i })).toBeInTheDocument())
  })

  it('opens the value-labels editor and loads unique values for a column', async () => {
    installSession()
    server.use(
      http.get('/api/compute/test-session/unique/GROUP', () =>
        HttpResponse.json({ values: ['A', 'B'] }),
      ),
    )

    const user = userEvent.setup()
    render(<DataDictionaryPanel />)

    const row = screen.getByText('GROUP').closest('tr')!
    await user.click(within(row).getByRole('button', { name: /edit/i }))

    await waitFor(() => expect(screen.getByText(/Value labels for/)).toBeInTheDocument())
    // One label input per value, and the same values in the order editor.
    expect(screen.getAllByPlaceholderText('label…')).toHaveLength(2)
    const editor = screen.getByRole('region', { name: 'Category order for GROUP' })
    expect(within(editor).getByText('A')).toBeInTheDocument()
    expect(within(editor).getByText('B')).toBeInTheDocument()
  })

  it('sets a category order low to high and saves it as level_order', async () => {
    installSession()
    let savedBody: { columns: Record<string, { level_order?: string[] }> } | null = null
    server.use(
      // The unique-values endpoint returns levels sorted alphabetically.
      http.get('/api/compute/test-session/unique/GROUP', () =>
        HttpResponse.json({ values: ['Heavy', 'Moderate', 'Trace'] }),
      ),
      http.post('/api/sessions/test-session/metadata', async ({ request }) => {
        savedBody = (await request.json()) as typeof savedBody
        return HttpResponse.json({ status: 'ok' })
      }),
    )

    const user = userEvent.setup()
    render(<DataDictionaryPanel />)
    const row = screen.getByText('GROUP').closest('tr')!
    await user.click(within(row).getByRole('button', { name: /edit/i }))

    const editor = await screen.findByRole('region', { name: 'Category order for GROUP' })
    await user.click(within(editor).getByRole('button', { name: 'Move Trace up' }))
    await user.click(within(editor).getByRole('button', { name: 'Move Trace up' }))
    await user.click(within(editor).getByRole('button', { name: 'Move Heavy down' }))
    const items = within(editor).getAllByRole('listitem').map((li) => li.textContent)
    expect(items.map((t) => t?.replace(/[↑↓]/g, ''))).toEqual(['1.Trace', '2.Moderate', '3.Heavy'])

    await user.click(screen.getByRole('button', { name: /save metadata/i }))
    await waitFor(() => expect(savedBody).not.toBeNull())
    expect(savedBody!.columns.GROUP.level_order).toEqual(['Trace', 'Moderate', 'Heavy'])
  })

  it('clears a saved category order by sending an empty list', async () => {
    installSession(makeSession({
      columns: [{ name: 'GROUP', dtype: 'object', kind: 'categorical', level_order: ['Trace', 'Moderate', 'Heavy'] }],
      preview: [{ GROUP: 'Trace' }],
    }))
    let savedBody: { columns: Record<string, { level_order?: string[] }> } | null = null
    server.use(
      http.get('/api/compute/test-session/unique/GROUP', () =>
        HttpResponse.json({ values: ['Heavy', 'Moderate', 'Trace'] }),
      ),
      http.post('/api/sessions/test-session/metadata', async ({ request }) => {
        savedBody = (await request.json()) as typeof savedBody
        return HttpResponse.json({ status: 'ok' })
      }),
    )

    const user = userEvent.setup()
    render(<DataDictionaryPanel />)
    const row = screen.getByText('GROUP').closest('tr')!
    await user.click(within(row).getByRole('button', { name: /ordered/i }))
    const editor = await screen.findByRole('region', { name: 'Category order for GROUP' })
    await user.click(within(editor).getByRole('button', { name: 'Clear order' }))
    await user.click(screen.getByRole('button', { name: /save metadata/i }))
    await waitFor(() => expect(savedBody).not.toBeNull())
    expect(savedBody!.columns.GROUP.level_order).toEqual([])
  })

  it('imports a wide value-label dictionary with automatic and label-based matching', async () => {
    installSession(makeSession({
      columns: [
        { name: 'GROUP', dtype: 'int64', kind: 'categorical', value_labels: { '0': 'Old', '9': 'Unknown' } },
        { name: 'DM', dtype: 'int64', kind: 'categorical', label: 'Diabetes Mellitus' },
      ],
      preview: [{ GROUP: 0, DM: 0 }, { GROUP: 1, DM: 1 }],
    }))
    let savedBody: unknown = null
    server.use(
      http.post('/api/sessions/test-session/metadata', async ({ request }) => {
        savedBody = await request.json()
        return HttpResponse.json({ status: 'ok' })
      }),
    )

    const user = userEvent.setup()
    render(<DataDictionaryPanel />)
    await user.click(screen.getByRole('button', { name: /import value labels/i }))
    await user.upload(
      screen.getByLabelText('Dictionary file'),
      new File(['mock workbook'], 'labels.xlsx', {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      }),
    )

    await waitFor(() => expect(screen.getByLabelText('Map Group')).toHaveValue('GROUP'))
    expect(screen.getByLabelText('Map Diabetes Mellitus')).toHaveValue('DM')
    await user.click(screen.getByRole('button', { name: /import 2 variables/i }))

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    expect(savedBody).toMatchObject({
      columns: {
        GROUP: { value_labels: { '0': 'Control', '1': 'Patient', '9': 'Unknown' } },
        DM: { value_labels: { '0': 'No', '1': 'Yes' } },
      },
    })
    expect(screen.getByRole('button', { name: /saved/i })).toBeInTheDocument()
  })
})
