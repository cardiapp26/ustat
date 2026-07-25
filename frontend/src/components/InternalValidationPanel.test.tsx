import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import InternalValidationPanel from './InternalValidationPanel'

afterEach(() => clearSession())

describe('InternalValidationPanel', () => {
  it('renders the tab bar even without an active session', () => {
    clearSession()
    render(<InternalValidationPanel />)
    expect(screen.getByRole('button', { name: /internal \(bootstrap \+ cv\)/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /external \(logistic\)/i })).toBeInTheDocument()
  })

  it('defaults to the Internal tab with Logistic model type', () => {
    installSession()
    render(<InternalValidationPanel />)
    expect(screen.getByRole('button', { name: 'Logistic' })).toBeInTheDocument()
    expect(screen.getByText('Outcome (binary 0/1)')).toBeInTheDocument()
  })

  it('disables Run until outcome and predictors are chosen', () => {
    installSession()
    render(<InternalValidationPanel />)
    expect(screen.getByRole('button', { name: /run internal validation/i })).toBeDisabled()
  })

  it('runs internal validation (logistic) and renders discrimination tiles', async () => {
    installSession()
    server.use(
      http.post('/api/model_diagnostics/model_validation', () =>
        HttpResponse.json({
          interpretation: 'Modest overfitting detected',
          n: 100, n_predictors: 2, n_boot: 200,
          apparent: { auc: 0.82, calibration_slope: 1.0, brier: 0.15 },
          optimism: { auc: 0.05 },
          corrected: { auc: 0.77, calibration_slope: 0.9 },
          cv: { auc: 0.76, calibration_slope: 0.88, brier: 0.16, folds: 5 },
          overfit_gap: 0.05,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)

    const outcomeSelect = screen.getByText('Outcome (binary 0/1)').closest('div')!.querySelector('select')!
    await user.selectOptions(outcomeSelect, 'DM')
    const predictorCheckbox = screen.getByRole('checkbox', { name: 'AGE' })
    await user.click(predictorCheckbox)

    await user.click(screen.getByRole('button', { name: /run internal validation/i }))

    await waitFor(() => expect(screen.getByText('Modest overfitting detected')).toBeInTheDocument())
    expect(screen.getByText('0.820')).toBeInTheDocument()
    expect(screen.getByText('0.770')).toBeInTheDocument()
  })

  it('switches to the External tab and runs external validation', async () => {
    installSession()
    server.use(
      http.post('/api/model_diagnostics/external_validation_logistic', () =>
        HttpResponse.json({
          result_text: 'Calibration acceptable in validation cohort',
          n: 50,
          discrimination: { auc: 0.79, auc_ci: [0.7, 0.88], se: 0.05 },
          calibration: {
            slope: 0.95, intercept: 0.02, oe_ratio: 1.01,
            hosmer_lemeshow: { chi2: 5.1, df: 8, p: 0.75 },
            brier: 0.14, acceptable: true,
          },
          calibration_plot: [{ pred: 0.2, obs: 0.22, n: 20 }],
          dev_vs_val: { auc_drop: 0.03, slope_shift: -0.05 },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)
    await user.click(screen.getByRole('button', { name: /external \(logistic\)/i }))

    const outcomeSelect = screen.getByText('Outcome (binary 0/1)').closest('div')!.querySelector('select')!
    await user.selectOptions(outcomeSelect, 'DM')
    const probSelect = screen.getByText('Predicted probability column (0–1)').closest('div')!.querySelector('select')!
    await user.selectOptions(probSelect, 'AGE')

    await user.click(screen.getByRole('button', { name: /run external validation/i }))

    await waitFor(() => expect(screen.getByText('Calibration acceptable in validation cohort')).toBeInTheDocument())
    expect(screen.getByText('0.790')).toBeInTheDocument()
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/model_diagnostics/model_validation', () =>
        HttpResponse.json({ detail: 'Too few events for bootstrap' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)
    const outcomeSelect = screen.getByText('Outcome (binary 0/1)').closest('div')!.querySelector('select')!
    await user.selectOptions(outcomeSelect, 'DM')
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /run internal validation/i }))

    await waitFor(() => expect(screen.getByText('Too few events for bootstrap')).toBeInTheDocument())
  })

  it('runs NRI / IDI on the Reclassification tab', async () => {
    installSession()
    server.use(
      http.post('/api/model_diagnostics/nri_idi', () =>
        // Mirrors a real 400-row run captured against the live endpoint.
        HttpResponse.json({
          n: 400,
          cutoff_used: 0.5,
          nri: {
            estimate: 0.0862, ci_low: 0.0156, ci_high: 0.1615,
            contribution_events: 0.0525, contribution_non_events: 0.0337,
          },
          idi: { estimate: 0.0517, ci_low: 0.0288, ci_high: 0.0761 },
          reclassification_counts: {
            up_in_events: 21, down_in_events: 0,
            up_in_non_events: 18, down_in_non_events: 0,
          },
          test: 'NRI + IDI (with bootstrap CI)',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)
    await user.click(screen.getByRole('button', { name: /reclassification \(nri \/ idi\)/i }))

    const outcomeSelect = screen.getByText('Outcome (binary 0/1)').closest('div')!.querySelector('select')!
    await user.selectOptions(outcomeSelect, 'DM')
    const oldSelect = screen.getByText('Old model probability (0–1)').closest('div')!.querySelector('select')!
    await user.selectOptions(oldSelect, 'AGE')
    const newSelect = screen.getByText('New model probability (0–1)').closest('div')!.querySelector('select')!
    await user.selectOptions(newSelect, 'LDL')

    await user.click(screen.getByRole('button', { name: /run nri \/ idi/i }))

    await waitFor(() => expect(screen.getByText('+0.086')).toBeInTheDocument())
    expect(screen.getByText('+0.0517')).toBeInTheDocument()
    expect(screen.getByText('21')).toBeInTheDocument()
    expect(screen.getByText('18')).toBeInTheDocument()
    // Zero counts are legitimate results, not "missing" — they must render as 0.
    expect(screen.getByText('Down in events').nextElementSibling).toHaveTextContent('0')
    expect(screen.getByText('Down in non-events').nextElementSibling).toHaveTextContent('0')
    expect(screen.getByText(/NRI \+ IDI \(with bootstrap CI\)/)).toBeInTheDocument()
  })

  it('runs external survival validation with only the always-present fields', async () => {
    installSession()
    server.use(
      http.post('/api/survival_advanced/external_validation', () =>
        // Mirrors a real 300-subject run with no time_points: every optional
        // block (tdAUC, IBS, IPTW, performance_vs_dev, …) is simply absent.
        HttpResponse.json({
          n_validation: 300,
          validation_c_index: 0.6073,
          validation_calibration_slope: 1.4896,
          validation_calibration_intercept: -1.218,
          note: 'Use survival_probs (n_samples x n_times) for accurate IBS and tdAUC.',
          test: 'External Validation & Calibration (Phase 9 - Enhanced)',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)
    await user.click(screen.getByRole('button', { name: /external \(survival\)/i }))

    const durationSelect = screen.getByText('Duration / time').closest('div')!.querySelector('select')!
    await user.selectOptions(durationSelect, 'AGE')
    const eventSelect = screen.getByText('Event (1 = event, 0 = censored)').closest('div')!.querySelector('select')!
    await user.selectOptions(eventSelect, 'DM')
    const lpSelect = screen.getByText('Predicted linear predictor / risk score').closest('div')!.querySelector('select')!
    await user.selectOptions(lpSelect, 'LDL')

    await user.click(screen.getByRole('button', { name: /run survival validation/i }))

    await waitFor(() => expect(screen.getByText('0.607')).toBeInTheDocument())
    expect(screen.getByText('1.49')).toBeInTheDocument()
    expect(screen.getByText('-1.22')).toBeInTheDocument()
    // Optional blocks must be omitted entirely rather than rendered as blanks.
    expect(screen.queryByText('Time-dependent AUC')).not.toBeInTheDocument()
    expect(screen.queryByText('Dev → validation')).not.toBeInTheDocument()
    expect(screen.queryByText('Integrated Brier')).not.toBeInTheDocument()
  })

  it('renders the optional survival blocks when the backend returns them', async () => {
    installSession()
    server.use(
      http.post('/api/survival_advanced/external_validation', () =>
        HttpResponse.json({
          n_validation: 180,
          validation_c_index: 0.712,
          validation_calibration_slope: 0.93,
          validation_calibration_intercept: 0.04,
          note: 'Use survival_probs (n_samples x n_times) for accurate IBS and tdAUC.',
          integrated_brier_score: { ibs: 0.1742, n_time_points: 2 },
          time_dependent_auc: [{ time: 12, auc: 0.688, n_at_risk: 150, n_events: 20 }],
          performance_vs_dev: { c_index_drop: 0.028, calibration_slope_shift: -0.07 },
          test: 'External Validation & Calibration (Phase 9 - Enhanced)',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)
    await user.click(screen.getByRole('button', { name: /external \(survival\)/i }))

    const durationSelect = screen.getByText('Duration / time').closest('div')!.querySelector('select')!
    await user.selectOptions(durationSelect, 'AGE')
    const eventSelect = screen.getByText('Event (1 = event, 0 = censored)').closest('div')!.querySelector('select')!
    await user.selectOptions(eventSelect, 'DM')
    const lpSelect = screen.getByText('Predicted linear predictor / risk score').closest('div')!.querySelector('select')!
    await user.selectOptions(lpSelect, 'LDL')

    await user.click(screen.getByRole('button', { name: /run survival validation/i }))

    await waitFor(() => expect(screen.getByText('0.712')).toBeInTheDocument())
    expect(screen.getByText('0.93')).toBeInTheDocument()
    expect(screen.getByText('AUC at t = 12.00')).toBeInTheDocument()
    expect(screen.getByText('0.688')).toBeInTheDocument()
    expect(screen.getByText('0.174')).toBeInTheDocument()
    expect(screen.getByText('Dev → validation')).toBeInTheDocument()
  })

  it('renders the service-level {error} shape from external survival validation', async () => {
    installSession()
    server.use(
      http.post('/api/survival_advanced/external_validation', () =>
        HttpResponse.json({
          error: 'Too few complete observations in validation data',
          test: 'External Validation & Calibration (Phase 9 - Enhanced)',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)
    await user.click(screen.getByRole('button', { name: /external \(survival\)/i }))

    const durationSelect = screen.getByText('Duration / time').closest('div')!.querySelector('select')!
    await user.selectOptions(durationSelect, 'AGE')
    const eventSelect = screen.getByText('Event (1 = event, 0 = censored)').closest('div')!.querySelector('select')!
    await user.selectOptions(eventSelect, 'DM')
    const lpSelect = screen.getByText('Predicted linear predictor / risk score').closest('div')!.querySelector('select')!
    await user.selectOptions(lpSelect, 'LDL')

    await user.click(screen.getByRole('button', { name: /run survival validation/i }))

    await waitFor(() =>
      expect(screen.getByText('Too few complete observations in validation data')).toBeInTheDocument())
    expect(screen.queryByText('Discrimination & calibration')).not.toBeInTheDocument()
  })

  it('surfaces the 400 detail from the NRI / IDI endpoint', async () => {
    installSession()
    server.use(
      http.post('/api/model_diagnostics/nri_idi', () =>
        HttpResponse.json(
          { detail: 'At least 100 observations recommended for stable NRI/IDI estimates.' },
          { status: 400 },
        ),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)
    await user.click(screen.getByRole('button', { name: /reclassification \(nri \/ idi\)/i }))

    const outcomeSelect = screen.getByText('Outcome (binary 0/1)').closest('div')!.querySelector('select')!
    await user.selectOptions(outcomeSelect, 'DM')
    const oldSelect = screen.getByText('Old model probability (0–1)').closest('div')!.querySelector('select')!
    await user.selectOptions(oldSelect, 'AGE')
    const newSelect = screen.getByText('New model probability (0–1)').closest('div')!.querySelector('select')!
    await user.selectOptions(newSelect, 'LDL')

    await user.click(screen.getByRole('button', { name: /run nri \/ idi/i }))

    await waitFor(() =>
      expect(screen.getByText('At least 100 observations recommended for stable NRI/IDI estimates.'))
        .toBeInTheDocument())
  })
})
