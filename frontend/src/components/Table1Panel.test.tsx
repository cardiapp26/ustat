import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import Table1Panel from './Table1Panel'

afterEach(() => clearSession())

const T1_RESULT_GROUPED = {
  group_column: 'GROUP',
  group_labels: ['A', 'B'],
  group_ns: { A: 2, B: 1 },
  total_n: 3,
  rows: [
    {
      variable: 'AGE',
      type: 'numeric',
      overall_n: 3,
      stat_rows: [
        {
          label: 'Mean ± SD',
          overall: '55.0 ± 5.7',
          group_stats: { A: '51.5 ± 4.9', B: '62.0' },
        },
      ],
      p_value: '0.042',
      test: "Student's t-test",
      significant: true,
      normal: true,
      normality_test: 'Shapiro-Wilk',
      normality_p: 0.71,
      smd: 0.35,
      group_stats: { A: '51.5 ± 4.9', B: '62.0' },
    },
    {
      variable: 'GROUP',
      type: 'categorical',
      overall_n: 3,
      p_value: '1.000',
      test: 'Chi-square',
      group_stats: {},
      sub_rows: [
        { category: 'A', overall: '2 (66.7%)', group_stats: { A: '2 (100%)', B: '0 (0%)' } },
        { category: 'B', overall: '1 (33.3%)', group_stats: { A: '0 (0%)', B: '1 (100%)' } },
      ],
    },
  ],
}

describe('Table1Panel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<Table1Panel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('generates the table with default (Auto) stat selection and renders p-values, SMD-less by default', async () => {
    installSession()
    let capturedBody: Record<string, unknown> | null = null
    server.use(
      http.post('/api/stats/table1', async ({ request }) => {
        capturedBody = (await request.json()) as Record<string, unknown>
        return HttpResponse.json(T1_RESULT_GROUPED)
      }),
    )

    const user = userEvent.setup()
    render(<Table1Panel />)

    // Group by GROUP
    await user.selectOptions(screen.getByRole('combobox'), 'GROUP')

    await user.click(screen.getByRole('button', { name: /generate table/i }))

    await waitFor(() => expect(screen.getAllByText('AGE').length).toBeGreaterThan(0))

    // p-value and test rendered
    expect(screen.getByText('0.042')).toBeInTheDocument()
    expect(screen.getByText("Student's t-test")).toBeInTheDocument()

    // "n" and "p" are italicized in the header per app convention
    const pHeader = screen.getByRole('columnheader', { name: /p-value/i })
    expect(pHeader.querySelector('i')?.textContent).toBe('p')
    const overallHeader = screen.getByRole('columnheader', { name: /overall/i })
    expect(overallHeader.querySelector('i')?.textContent).toBe('n')

    // default request used the "auto" stat
    expect(capturedBody).not.toBeNull()
    expect(capturedBody!.selected_stats).toEqual(['auto'])

    // SMD column not shown by default (checkbox unchecked)
    expect(screen.queryByText('SMD')).not.toBeInTheDocument()
  })

  it('supports a custom stat selection (Median [IQR] + 95% CI) and shows SMD when enabled', async () => {
    installSession()
    let capturedBody: Record<string, unknown> | null = null
    server.use(
      http.post('/api/stats/table1', async ({ request }) => {
        capturedBody = (await request.json()) as Record<string, unknown>
        return HttpResponse.json(T1_RESULT_GROUPED)
      }),
    )

    const user = userEvent.setup()
    render(<Table1Panel />)

    await user.selectOptions(screen.getByRole('combobox'), 'GROUP')

    // Open statistics selector and pick custom stats
    await user.click(screen.getByRole('button', { name: /statistics/i }))
    await user.click(screen.getByRole('checkbox', { name: /median \[iqr\]/i }))
    await user.click(screen.getByRole('checkbox', { name: /95% ci/i }))
    // Deselect "Auto" (still leaves 2 selected, so it's allowed to uncheck)
    await user.click(screen.getByRole('checkbox', { name: /auto \(normality-based\)/i }))

    // Enable SMD display
    await user.click(screen.getByRole('checkbox', { name: /show smd/i }))

    await user.click(screen.getByRole('button', { name: /generate table/i }))

    await waitFor(() => expect(screen.getAllByText('AGE').length).toBeGreaterThan(0))

    expect(capturedBody).not.toBeNull()
    expect(capturedBody!.selected_stats).toEqual(
      expect.arrayContaining(['median_iqr', 'ci95']),
    )
    expect(capturedBody!.selected_stats).not.toEqual(expect.arrayContaining(['auto']))

    // SMD column header + value rendered
    expect(screen.getByText('SMD')).toBeInTheDocument()
    expect(screen.getByText('0.350')).toBeInTheDocument()
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/stats/table1', () =>
        HttpResponse.json({ detail: 'No variables selected for analysis' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<Table1Panel />)

    await user.click(screen.getByRole('button', { name: /generate table/i }))

    await waitFor(() =>
      expect(screen.getByText('No variables selected for analysis')).toBeInTheDocument(),
    )
  })

  it('shows why a p-value is blank instead of leaving an unexplained dash', async () => {
    // A blank p is the one cell that cannot explain itself: a constant
    // column, a level only one arm has, or rows excluded as missing all
    // produce it. The backend says which; the panel used to drop the message.
    installSession()
    server.use(
      http.post('/api/stats/table1', () =>
        HttpResponse.json({
          ...T1_RESULT_GROUPED,
          warnings: [
            "'SITE': no p-value \u2014 only one category \u2014 nothing to compare.",
            {
              variable: 'STATUS',
              dropped_levels: [{ level: 'unknown', n: 4 }],
              note: "'STATUS': 4 row(s) hold a value that reads as missing.",
            },
          ],
          rows: [
            {
              variable: 'SITE',
              type: 'categorical',
              overall_n: 3,
              p_value: null,
              test: null,
              group_stats: {},
              sub_rows: [
                {
                  category: 'central',
                  overall: '3 (100.0%)',
                  group_stats: { A: '2 (100.0%)', B: '1 (100.0%)' },
                },
              ],
            },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<Table1Panel />)
    await user.selectOptions(screen.getByRole('combobox'), 'GROUP')
    await user.click(screen.getByRole('button', { name: /generate table/i }))

    await waitFor(() =>
      expect(screen.getByText(/only one category/)).toBeInTheDocument(),
    )
    // The structured warning must render as prose, not "[object Object]".
    expect(screen.getByText(/reads as missing/)).toBeInTheDocument()
    expect(screen.queryByText(/\[object Object\]/)).not.toBeInTheDocument()
  })
})
