import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import DecisionCurvePanel from './DecisionCurvePanel'

afterEach(() => clearSession())

const baseResult = {
  curves: {
    thresholds: [0.1, 0.2, 0.3],
    model: { thresholds: [0.1, 0.2, 0.3], net_benefit: [0.3, 0.25, 0.1] },
    treat_all: { net_benefit: [0.2, 0.1, 0.0] },
    treat_none: { net_benefit: [0, 0, 0] },
  },
  summary: {
    max_net_benefit: 0.3,
    max_net_benefit_threshold: 0.1,
    harm_threshold: 0.35,
    positive_nb_range: [0.05, 0.3],
    interventions_avoided_per_100_at_max: 12.5,
  },
  assumptions: ['Outcomes are correctly coded 0/1.'],
  warnings: ['Small sample size may reduce reliability.'],
  result_text: 'The model provides net benefit over both treat-all and treat-none.',
  prevalence: 0.3,
  mode: 'survival',
}

describe('DecisionCurvePanel', () => {
  it('renders survival mode inputs by default with an active session', () => {
    installSession()
    render(<DecisionCurvePanel />)
    expect(screen.getByText('Decision Curve Analysis')).toBeInTheDocument()
    expect(screen.getByText('Duration / Time column')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /run decision curve analysis/i })).toBeDisabled()
  })

  it('switches to binary outcome mode and shows the relevant inputs', async () => {
    installSession()
    const user = userEvent.setup()
    render(<DecisionCurvePanel />)
    await user.click(screen.getByRole('button', { name: /binary outcome/i }))
    expect(screen.getByText('Probability / Risk column')).toBeInTheDocument()
    expect(screen.getByText('Binary Outcome (0/1)')).toBeInTheDocument()
  })

  it('enables Run only once duration/event/risk are all selected in survival mode', async () => {
    installSession()
    const user = userEvent.setup()
    render(<DecisionCurvePanel />)

    const runButton = screen.getByRole('button', { name: /run decision curve analysis/i })
    expect(runButton).toBeDisabled()

    const selects = screen.getAllByRole('combobox')
    // Order in DOM: duration, event, risk, (no select for time horizon — it's a number input)
    await user.selectOptions(selects[0], 'AGE')
    expect(runButton).toBeDisabled()
    await user.selectOptions(selects[1], 'GROUP')
    expect(runButton).toBeDisabled()
    await user.selectOptions(selects[2], 'LDL')
    expect(runButton).toBeEnabled()
  })

  it('runs DCA in survival mode and renders the net benefit summary', async () => {
    installSession()
    server.use(http.post('/api/decision_curve/dca', () => HttpResponse.json(baseResult)))

    const user = userEvent.setup()
    render(<DecisionCurvePanel />)

    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'AGE')
    await user.selectOptions(selects[1], 'GROUP')
    await user.selectOptions(selects[2], 'LDL')

    await user.click(screen.getByRole('button', { name: /run decision curve analysis/i }))

    await waitFor(() => expect(screen.getByText('Clinical Utility Summary')).toBeInTheDocument())
    expect(screen.getByText('0.3000')).toBeInTheDocument()
    expect(screen.getByText('12.5')).toBeInTheDocument()
    expect(
      screen.getByText('The model provides net benefit over both treat-all and treat-none.'),
    ).toBeInTheDocument()
    expect(screen.getByText('Outcomes are correctly coded 0/1.')).toBeInTheDocument()
    expect(screen.getByText(/small sample size may reduce reliability/i)).toBeInTheDocument()
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/decision_curve/dca', () =>
        HttpResponse.json({ detail: 'Risk column must be numeric' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<DecisionCurvePanel />)

    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'AGE')
    await user.selectOptions(selects[1], 'GROUP')
    await user.selectOptions(selects[2], 'LDL')
    await user.click(screen.getByRole('button', { name: /run decision curve analysis/i }))

    await waitFor(() => expect(screen.getByText('Risk column must be numeric')).toBeInTheDocument())
  })

  it('runs the integrated external validation + DCA pipeline and renders both blocks', async () => {
    installSession()
    server.use(
      http.post('/api/decision_curve/integrated_extval_dca', () =>
        HttpResponse.json({
          test: 'Integrated External Validation + Decision Curve Analysis',
          prediction_source: 'precomputed',
          n_validation: 42,
          external_validation: {
            n_validation: 42,
            validation_c_index: 0.71,
            validation_calibration_slope: 0.95,
            validation_calibration_intercept: 0.02,
          },
          decision_curve: {
            curves: {
              thresholds: [0.1, 0.2, 0.3],
              model_net_benefit: [0.3, 0.25, 0.1],
              treat_all_net_benefit: [0.2, 0.1, 0.0],
              treat_none_net_benefit: [0, 0, 0],
            },
            summary: {
              max_net_benefit: 0.3,
              max_net_benefit_threshold: 0.1,
              interventions_avoided_per_100_at_max: 12.5,
            },
            n: 42,
            bootstrap_corrected_dca: {
              available: true,
              n_boot: 200,
              summary: { max_corrected_net_benefit: 0.22, threshold_at_max_corrected: 0.1 },
            },
          },
          transportability: null,
          result_text: 'Integrated validation/DCA pipeline for precomputed predictions on n=42 validation observations.',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<DecisionCurvePanel />)

    await user.click(screen.getByRole('button', { name: /ext-val \+ dca/i }))

    const runButton = screen.getByRole('button', { name: /run external validation \+ dca/i })
    expect(runButton).toBeDisabled()

    // Order in DOM: duration, event, prediction
    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'AGE')
    expect(runButton).toBeDisabled()
    await user.selectOptions(selects[1], 'GROUP')
    expect(runButton).toBeDisabled()
    await user.selectOptions(selects[2], 'LDL')
    expect(runButton).toBeEnabled()

    await user.click(runButton)

    await waitFor(() => expect(screen.getByText('External Validation')).toBeInTheDocument())
    expect(screen.getByText('n = 42')).toBeInTheDocument()
    expect(screen.getByText('0.710')).toBeInTheDocument()
    expect(screen.getByText('0.950')).toBeInTheDocument()
    expect(screen.getByText('Bootstrap-Corrected DCA')).toBeInTheDocument()
    expect(screen.getByText('0.2200')).toBeInTheDocument()
    expect(screen.getByText('Clinical Utility Summary')).toBeInTheDocument()
    expect(
      screen.getByText(/integrated validation\/dca pipeline for precomputed predictions/i),
    ).toBeInTheDocument()
  })

  it('shows the integrated-mode backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/decision_curve/integrated_extval_dca', () =>
        HttpResponse.json({ detail: 'Need at least 20 complete validation observations.' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<DecisionCurvePanel />)

    await user.click(screen.getByRole('button', { name: /ext-val \+ dca/i }))
    const selects = screen.getAllByRole('combobox')
    await user.selectOptions(selects[0], 'AGE')
    await user.selectOptions(selects[1], 'GROUP')
    await user.selectOptions(selects[2], 'LDL')
    await user.click(screen.getByRole('button', { name: /run external validation \+ dca/i }))

    await waitFor(() =>
      expect(screen.getByText('Need at least 20 complete validation observations.')).toBeInTheDocument(),
    )
  })
})
