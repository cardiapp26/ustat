import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import { useStore } from '../store'
import AddedValuePanel from './AddedValuePanel'

const ADDED_VALUE_RESULT = {
  added_value: true,
  result_text: 'Adding LDL improves discrimination and reclassification.',
  n: 3,
  n_excluded: 0,
  prediction_basis: 'in-sample',
  discrimination: { auc_base: 0.7, auc_full: 0.82, delta_auc: 0.12, delong_p: 0.03, significant: true },
  reclassification: { idi: 0.05, idi_ci: [0.01, 0.09], nri: 0.2, nri_ci: [0.05, 0.35], nri_events: 0.15, nri_nonevents: 0.05 },
  fit: { lr_stat: 5.2, lr_p: 0.02, delta_aic: -3.1, nagelkerke_base: 0.2, nagelkerke_full: 0.35 },
  calibration: { base: { calibration_slope: 1, brier: 0.2 }, full: { calibration_slope: 0.95, brier: 0.18 }, preserved: true },
}

async function configureAndRun(user: ReturnType<typeof userEvent.setup>) {
  await user.selectOptions(screen.getByRole('combobox'), 'DM')
  const baseGroup = screen.getByText('Base model predictors (established)').parentElement!
  const newGroup = screen.getByText('New predictor(s) to evaluate').parentElement!
  await user.click(within(baseGroup).getByRole('checkbox', { name: 'AGE' }))
  await user.click(within(newGroup).getByRole('checkbox', { name: 'LDL' }))
  await user.click(screen.getByRole('button', { name: /assess added value/i }))
  await screen.findByText(ADDED_VALUE_RESULT.result_text)
}

afterEach(() => clearSession())

describe('AddedValuePanel', () => {
  it('renders a placeholder when there is no session (no outcome available)', () => {
    clearSession()
    render(<AddedValuePanel />)
    expect(
      screen.getByText(/Pick an outcome, base predictors, and a new predictor/i),
    ).toBeInTheDocument()
  })

  it('lists columns as outcome + predictor checkboxes', () => {
    installSession()
    render(<AddedValuePanel />)
    expect(screen.getAllByText('AGE').length).toBeGreaterThan(0)
    expect(screen.getAllByText('LDL').length).toBeGreaterThan(0)
  })

  it('disables the run button until outcome, base and new predictors are chosen', async () => {
    installSession()
    const user = userEvent.setup()
    render(<AddedValuePanel />)

    const runButton = screen.getByRole('button', { name: /assess added value/i })
    expect(runButton).toBeDisabled()

    const outcomeSelect = screen.getByRole('combobox')
    await user.selectOptions(outcomeSelect, 'DM')
    expect(runButton).toBeDisabled()

    const baseGroup = screen.getByText('Base model predictors (established)').parentElement!
    const newGroup = screen.getByText('New predictor(s) to evaluate').parentElement!

    // Check AGE as base predictor
    await user.click(within(baseGroup).getByRole('checkbox', { name: 'AGE' }))
    expect(runButton).toBeDisabled()

    // Check LDL as new predictor
    await user.click(within(newGroup).getByRole('checkbox', { name: 'LDL' }))
    expect(runButton).toBeEnabled()
  })

  it('runs added-value analysis and renders the result', async () => {
    installSession()
    server.use(
      http.post('/api/model_compare/added_value', () =>
        HttpResponse.json({
          added_value: true,
          result_text: 'Adding LDL improves discrimination and reclassification.',
          n: 3,
          n_excluded: 0,
          prediction_basis: 'in-sample',
          discrimination: { auc_base: 0.7, auc_full: 0.82, delta_auc: 0.12, delong_p: 0.03, significant: true },
          reclassification: { idi: 0.05, idi_ci: [0.01, 0.09], nri: 0.2, nri_ci: [0.05, 0.35], nri_events: 0.15, nri_nonevents: 0.05 },
          fit: { lr_stat: 5.2, lr_p: 0.02, delta_aic: -3.1, nagelkerke_base: 0.2, nagelkerke_full: 0.35 },
          calibration: { base: { calibration_slope: 1, brier: 0.2 }, full: { calibration_slope: 0.95, brier: 0.18 }, preserved: true },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<AddedValuePanel />)

    await user.selectOptions(screen.getByRole('combobox'), 'DM')
    const baseGroup = screen.getByText('Base model predictors (established)').parentElement!
    const newGroup = screen.getByText('New predictor(s) to evaluate').parentElement!
    await user.click(within(baseGroup).getByRole('checkbox', { name: 'AGE' }))
    await user.click(within(newGroup).getByRole('checkbox', { name: 'LDL' }))

    const runButton = screen.getByRole('button', { name: /assess added value/i })
    expect(runButton).toBeEnabled()
    await user.click(runButton)

    await waitFor(() =>
      expect(screen.getByText('Adding LDL improves discrimination and reclassification.')).toBeInTheDocument(),
    )
    expect(screen.getByText('0.700')).toBeInTheDocument()
    expect(screen.getByText('0.820')).toBeInTheDocument()
    expect(screen.getByText('+0.120')).toBeInTheDocument()
  })

  it('shows the error message from the backend on failure', async () => {
    installSession()
    server.use(
      http.post('/api/model_compare/added_value', () =>
        HttpResponse.json({ detail: 'Model failed to converge' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<AddedValuePanel />)
    await user.selectOptions(screen.getByRole('combobox'), 'DM')
    const baseGroup = screen.getByText('Base model predictors (established)').parentElement!
    const newGroup = screen.getByText('New predictor(s) to evaluate').parentElement!
    await user.click(within(baseGroup).getByRole('checkbox', { name: 'AGE' }))
    await user.click(within(newGroup).getByRole('checkbox', { name: 'LDL' }))
    await user.click(screen.getByRole('button', { name: /assess added value/i }))

    await waitFor(() => expect(screen.getByText('Model failed to converge')).toBeInTheDocument())
  })

  it('marks the comparison out of date once the data changes, and Recompute refits it', async () => {
    // The panel shows no export control of its own (its ResultExporter has no
    // table or plot to carry), so the notice is what stands between a reader
    // and a comparison fitted on data that is no longer on screen.
    installSession()
    let calls = 0
    server.use(
      http.post('/api/model_compare/added_value', () => {
        calls += 1
        return HttpResponse.json(ADDED_VALUE_RESULT)
      }),
    )

    const user = userEvent.setup()
    render(<AddedValuePanel />)
    await configureAndRun(user)
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/Out of date\./)).toBeInTheDocument()
    expect(screen.getByText(/the data changed/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Recompute' }))
    await waitFor(() => expect(calls).toBe(2))
    await waitFor(() => expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument())
  })

  it('keeps the comparison current across a tab switch', async () => {
    // The result is cached with its stamp; the selections it was fitted under
    // have to come back with it, or it would read as stale on every return.
    installSession()
    server.use(http.post('/api/model_compare/added_value', () => HttpResponse.json(ADDED_VALUE_RESULT)))

    const user = userEvent.setup()
    const { unmount } = render(<AddedValuePanel />)
    await configureAndRun(user)
    unmount()

    render(<AddedValuePanel />)
    expect(screen.getByText(ADDED_VALUE_RESULT.result_text)).toBeInTheDocument()
    expect(screen.queryByText(/Out of date\./)).not.toBeInTheDocument()
  })
})
