import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import ForestBuilderPanel from './ForestBuilderPanel'

afterEach(() => clearSession())

describe('ForestBuilderPanel', () => {
  it('renders the empty-state illustration with no rows entered', () => {
    clearSession()
    render(<ForestBuilderPanel />)
    expect(screen.getByText('Interactive Forest Plot Builder')).toBeInTheDocument()
    expect(screen.queryByTestId('plotly-mock')).not.toBeInTheDocument()
  })

  it('manually entering a valid row renders the forest plot', async () => {
    clearSession()
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    const rowsTable = screen.getByText('Rows').closest('div') as HTMLElement
    const table = within(rowsTable.parentElement as HTMLElement).getByRole('table')
    const firstRow = within(table).getAllByRole('row')[1] // header + first data row

    await user.type(within(firstRow).getByPlaceholderText('Label'), 'Model A')
    const numberInputs = within(firstRow).getAllByRole('spinbutton')
    // Order: Est, CI low, CI high, p
    await user.type(numberInputs[0], '2.03')
    await user.type(numberInputs[1], '1.02')
    await user.type(numberInputs[2], '4.03')
    await user.type(numberInputs[3], '0.04')

    expect(screen.queryByText('Interactive Forest Plot Builder')).not.toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
    expect(screen.getByText('1 of 1 valid')).toBeInTheDocument()
  })

  it('loading a preset populates the rows table and renders the plot', async () => {
    clearSession()
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    await user.click(screen.getByRole('button', { name: 'Sensitivity — model specifications' }))

    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
    expect(screen.getByText('6 of 6 valid')).toBeInTheDocument()

    // "Clear all" resets back to the empty state
    await user.click(screen.getByRole('button', { name: '✕ Clear all' }))
    expect(screen.getByText('Interactive Forest Plot Builder')).toBeInTheDocument()
    expect(screen.getByText('0 of 1 valid')).toBeInTheDocument()
  })

  it('adding, reordering, and deleting rows updates the valid-row count', async () => {
    clearSession()
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    await user.click(screen.getByRole('button', { name: 'Multiple endpoints / time horizons' }))
    expect(screen.getByText('5 of 5 valid')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '+ Add row' }))
    expect(screen.getByText('5 of 6 valid')).toBeInTheDocument()

    // Delete the newly added (invalid, last) row via its own row's delete button
    const rowsTable = screen.getByText('Rows').closest('div') as HTMLElement
    const table = within(rowsTable.parentElement as HTMLElement).getByRole('table')
    const dataRows = within(table).getAllByRole('row').slice(1) // skip header
    const lastRow = dataRows[dataRows.length - 1]
    await user.click(within(lastRow).getByTitle('Delete'))
    expect(screen.getByText('5 of 5 valid')).toBeInTheDocument()

    // Move the first row down — row count / validity unaffected, but the
    // reorder handler should not throw and the table stays at 5 rows.
    const rowsAfterDelete = within(table).getAllByRole('row').slice(1)
    await user.click(within(rowsAfterDelete[0]).getByTitle('Move down'))
    expect(screen.getByText('5 of 5 valid')).toBeInTheDocument()
  })

  // Timeout raised from the 5s default: this test types two CSV lines through
  // userEvent one keystroke at a time (~85 keystrokes), which takes ~1.7s
  // standalone but exceeds 5s when the whole suite's parallel environments are
  // spinning up on the same machine. The assertions are unchanged — this is
  // the machine's budget, not the product's latency.
  it('bulk paste (CSV) parses rows and skips a non-numeric header line', async () => {
    clearSession()
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    await user.click(screen.getByRole('button', { name: '📋 Paste rows' }))
    const textarea = screen.getByPlaceholderText(/Unadjusted, 2.03, 1.02, 4.03, 0.04/)
    await user.type(
      textarea,
      'label,est,ci_low,ci_high,p{Enter}Unadjusted,2.03,1.02,4.03,0.04{Enter}Adjusted,1.27,0.61,2.63,0.52',
    )
    await user.click(screen.getByRole('button', { name: 'Apply' }))

    expect(screen.getByText('2 of 2 valid')).toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  }, 15000)

  it('Load from Active Dataset: with a session, auto-maps columns and loads preview rows on click', async () => {
    installSession(
      makeSession({
        columns: [
          { name: 'study', dtype: 'object', kind: 'categorical' },
          { name: 'hr', dtype: 'float64', kind: 'numeric' },
          { name: 'ci_low', dtype: 'float64', kind: 'numeric' },
          { name: 'ci_high', dtype: 'float64', kind: 'numeric' },
        ],
        preview: [
          { study: 'Model A', hr: 1.5, ci_low: 1.1, ci_high: 2.0 },
          { study: 'Model B', hr: 0.9, ci_low: 0.6, ci_high: 1.3 },
        ],
      }),
    )
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    expect(screen.getByText('test.csv')).toBeInTheDocument()
    const loadBtn = screen.getByRole('button', { name: /Load Dataset Rows/ })
    expect(loadBtn).toBeEnabled()
    await user.click(loadBtn)

    expect(screen.getByText('2 of 2 valid')).toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  })

  describe('cross-panel handoff', () => {
    const row = (label: string, est: number) => ({
      label, est, ci_low: est - 0.2, ci_high: est + 0.5, p: 0.01, extra: '',
    })

    // The sheet now lives in panelCache so it survives a tab switch, which
    // also means it survives into the next test unless cleared.
    beforeEach(() => {
      clearSession()
      useStore.setState({ panelCache: {}, forestHandoff: null, forestHandoffAppend: false })
    })

    it('a replacing handoff loads the rows and the sender\'s layout', () => {
      useStore.getState().setForestHandoff([row('Model A', 1.5), row('Model B', 0.9)], {
        rightHeader: 'HR (95% CI)', returnTab: 'models', returnLabel: '← Back to Cox model',
      })
      render(<ForestBuilderPanel />)

      expect(screen.getByText('2 of 2 valid')).toBeInTheDocument()
      expect(screen.getByRole('button', { name: '← Back to Cox model' })).toBeInTheDocument()
      // The buffer is cleared so a later visit does not re-inject stale rows.
      expect(useStore.getState().forestHandoff).toBeNull()
    })

    it('an appending handoff adds to what is already there', async () => {
      // The whole point: a continuous exposure and its dichotomised form come
      // from two fits and belong in one figure.
      clearSession()
      const user = userEvent.setup()
      const { rerender } = render(<ForestBuilderPanel />)
      useStore.getState().setForestHandoff([row('LAR (per 0.1 unit)', 1.19)], { rightHeader: 'OR (95% CI)' })
      rerender(<ForestBuilderPanel />)
      await waitFor(() => expect(screen.getByText('1 of 1 valid')).toBeInTheDocument())

      useStore.getState().setForestHandoff([row('High LAR (>0.555)', 4.27)], { rightHeader: 'OR (95% CI)' }, true)
      rerender(<ForestBuilderPanel />)

      await waitFor(() => expect(screen.getByText('2 of 2 valid')).toBeInTheDocument())
      const labels = screen.getAllByPlaceholderText('Label').map((i) => (i as HTMLInputElement).value)
      // Blank starter row dropped, not kept as a gap between the two fits.
      expect(labels).toEqual(['LAR (per 0.1 unit)', 'High LAR (>0.555)'])
      await user.click(screen.getByRole('button', { name: /Clear/ }))
    })

    it('accumulates across a tab switch, which is what unmounts the panel', async () => {
      // Regression: the rows lived in component state, so "← Back to the
      // model" and a second hand-off produced a figure with only the second
      // fit in it — silently losing the first.
      const first = render(<ForestBuilderPanel />)
      useStore.getState().setForestHandoff([row('LAR (per 0.1 unit)', 1.19)], { rightHeader: 'OR (95% CI)' })
      first.rerender(<ForestBuilderPanel />)
      await waitFor(() => expect(screen.getByText('1 of 1 valid')).toBeInTheDocument())
      first.unmount()

      useStore.getState().setForestHandoff([row('High LAR (>0.555)', 4.27)], { rightHeader: 'OR (95% CI)' }, true)
      render(<ForestBuilderPanel />)
      await waitFor(() => expect(screen.getByText('2 of 2 valid')).toBeInTheDocument())
      const labels = screen.getAllByPlaceholderText('Label').map((i) => (i as HTMLInputElement).value)
      expect(labels).toEqual(['LAR (per 0.1 unit)', 'High LAR (>0.555)'])
    })

    it('an appending handoff into an empty sheet still takes the layout', async () => {
      const { rerender } = render(<ForestBuilderPanel />)
      useStore.getState().setForestHandoff([row('AGE', 1.07)], {
        rightHeader: 'OR (95% CI)', returnTab: 'models', returnLabel: '← Back to the model',
      }, true)
      rerender(<ForestBuilderPanel />)

      await waitFor(() => expect(screen.getByText('1 of 1 valid')).toBeInTheDocument())
      expect(screen.getByRole('button', { name: '← Back to the model' })).toBeInTheDocument()
    })
  })
})
