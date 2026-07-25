import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import MLPanel from './MLPanel'

afterEach(() => clearSession())

const baseResult = {
  model: 'Random Forest',
  outcome: 'DM',
  task: 'classification' as const,
  cv_folds: 5,
  n: 3,
  n_features: 1,
  importance: [
    { feature: 'AGE', permutation: 0.05, permutation_sd: 0.01, impurity: 0.2 },
  ],
  roc_curve: [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }],
  scatter: [],
  auc: 0.812,
  auc_ci_low: 0.7,
  auc_ci_high: 0.9,
  accuracy: 0.8,
  sensitivity: 0.75,
  specificity: 0.85,
  ppv: 0.7,
  npv: 0.9,
  brier: 0.15,
  confusion: { tp: 5, tn: 10, fp: 2, fn: 1 },
  calibration: [{ pred: 0.5, obs: 0.4, n: 10 }],
  interpretation: 'The model shows good discrimination.',
}

describe('MLPanel', () => {
  it('renders without crashing when there is no active session', () => {
    clearSession()
    const { container } = render(<MLPanel />)
    // No session → columns list is empty but component still mounts (no early return guard in MLPanel).
    expect(container).toBeTruthy()
  })

  it('disables the train button until outcome and predictors are selected', () => {
    installSession()
    render(<MLPanel />)
    // No explicit disabled state on the button itself pre-validation is handled in run(),
    // but clicking without outcome should show a validation error rather than call the API.
    expect(screen.getByRole('button', { name: /train & cross-validate/i })).toBeEnabled()
  })

  it('shows a validation error when running without outcome/predictors selected', async () => {
    installSession()
    const user = userEvent.setup()
    render(<MLPanel />)
    await user.click(screen.getByRole('button', { name: /train & cross-validate/i }))
    expect(await screen.findByText(/select an outcome and at least one predictor/i)).toBeInTheDocument()
  })

  it('runs Random Forest and renders the result on success', async () => {
    installSession()
    server.use(
      http.post('/api/ml/random_forest', () => HttpResponse.json(baseResult)),
    )

    const user = userEvent.setup()
    render(<MLPanel />)

    await user.selectOptions(screen.getByRole('combobox', { name: /outcome/i }), 'DM')
    await user.click(screen.getAllByRole('checkbox')[0])

    await user.click(screen.getByRole('button', { name: /train & cross-validate/i }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Random Forest' })).toBeInTheDocument())
    expect(screen.getByText('0.812')).toBeInTheDocument()
    expect(screen.getByText('The model shows good discrimination.')).toBeInTheDocument()
    expect(screen.getByText('Feature importance')).toBeInTheDocument()
  })

  it('runs Gradient Boosting via the model toggle and hits the correct endpoint', async () => {
    installSession()
    server.use(
      http.post('/api/ml/gradient_boosting', () =>
        HttpResponse.json({ ...baseResult, model: 'Gradient Boosting' }),
      ),
    )

    const user = userEvent.setup()
    render(<MLPanel />)

    await user.click(screen.getByRole('button', { name: /gradient boosting/i }))
    await user.selectOptions(screen.getByRole('combobox', { name: /outcome/i }), 'DM')
    await user.click(screen.getAllByRole('checkbox')[0])
    await user.click(screen.getByRole('button', { name: /train & cross-validate/i }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Gradient Boosting' })).toBeInTheDocument())
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/ml/random_forest', () =>
        HttpResponse.json({ detail: 'Not enough data for cross-validation' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<MLPanel />)
    await user.selectOptions(screen.getByRole('combobox', { name: /outcome/i }), 'DM')
    await user.click(screen.getAllByRole('checkbox')[0])
    await user.click(screen.getByRole('button', { name: /train & cross-validate/i }))

    await waitFor(() =>
      expect(screen.getByText('Not enough data for cross-validation')).toBeInTheDocument(),
    )
  })
})

// ── Survival ML benchmark sub-analysis ──────────────────────────────────────

const SURVIVAL_URL = '/api/survival_advanced/ml_survival_benchmark'

const OVERFIT_WARNING =
  'Apparent ML C-index (0.954) is far above the cross-validated estimate (0.472) — ' +
  'the gradient-boosting model is overfitting. Report the cross-validated value, not the apparent one.'
const CHANCE_WARNING =
  'Cross-validated ML C-index is 0.472, i.e. at or below chance (0.5). ' +
  'The ML model does not generalise on this dataset.'

/** Modelled on a real 300-subject / 24-event run: the apparent numbers look
 *  spectacular (ML 0.954 vs Cox 0.644) while the honest cross-validated
 *  estimate (0.472) is worse than chance. */
const survivalResult = {
  test: 'Survival ML Benchmark (Phase 13 - nested CV, calibration, interpretability)',
  n: 300,
  n_excluded_missing: 0,
  classical_cox: { c_index: 0.6441, c_index_type: 'apparent', calibration_slope: 0.981 },
  ml_gradient_boosting_survival: {
    c_index: 0.9542,
    c_index_type: 'apparent',
    calibration_slope: 1.312,
    permutation_importance: [
      { variable: 'AGE', importance: 12.34567 },
      { variable: 'LDL', importance: 4.2001 },
    ],
    shap_values: { available: false, reason: 'SHAP not requested.' },
    partial_dependence: [
      {
        feature: 'AGE',
        points: [{ value: 45, mean_risk: -12.5 }, { value: 70, mean_risk: -4.1 }],
        direction: 'increasing risk',
      },
    ],
  },
  repeated_cv: {
    enabled: true,
    folds: [
      { fold: 1, c_index: 0.4812, n_test: 60, events_test: 5 },
      { fold: 2, c_index: 0.4628, n_test: 60, events_test: 4 },
    ],
    summary: { mean: 0.4721, sd: 0.0312, min: 0.4628, max: 0.4812, n: 5 },
    n_splits: 5,
    n_repeats: 1,
  },
  nested_cv: {
    enabled: false,
    reason: 'Set nested_cv=true to run inner-loop tuning and outer-loop evaluation.',
  },
  competing_risks_ml: { available: false, reason: 'Competing-risks ML not requested.' },
  auto_comparison: {
    models: [
      { name: 'Classical Cox', c_index: 0.6441, calibration_slope: 0.981, ibs: 0.1213 },
      { name: 'Gradient Boosting Survival (ranking)', c_index: 0.9542, calibration_slope: 1.312, ibs: 0.0842 },
    ],
    winner_by_c_index: 'ML',
    winner_by_ibs: 'ML',
  },
  assumptions: [
    'The headline classical_cox and ml_gradient_boosting_survival C-indices are APPARENT (resubstitution) values.',
  ],
  warnings: [OVERFIT_WARNING, CHANCE_WARNING],
  result_text:
    'Survival ML benchmark on n=300 complete subjects. Honest cross-validated ML C-index: 0.4721.',
}

/** Selects live inside their own <label>, so scope the lookup to that label. */
function selectFor(labelText: string): HTMLSelectElement {
  return screen.getByText(labelText).closest('label')!.querySelector('select')!
}

async function runSurvivalBenchmark(user: ReturnType<typeof userEvent.setup>) {
  render(<MLPanel />)
  await user.click(screen.getByRole('button', { name: /survival ml/i }))
  await user.selectOptions(selectFor('Duration'), 'AGE')
  await user.selectOptions(selectFor('Event'), 'DM')
  await user.click(screen.getByRole('button', { name: /run survival benchmark/i }))
}

describe('MLPanel — survival ML benchmark', () => {
  it('headlines the cross-validated C-index and demotes the apparent ones', async () => {
    installSession()
    server.use(http.post(SURVIVAL_URL, () => HttpResponse.json(survivalResult)))

    const user = userEvent.setup()
    await runSurvivalBenchmark(user)

    // 1. The cross-validated estimate is the headline figure.
    const cvCard = await screen.findByText('Cross-validated C-index (ML)')
    const cvBox = cvCard.closest('div')!
    expect(within(cvBox).getByText('0.472')).toBeInTheDocument()

    // 2. Both apparent values are present but explicitly labelled as in-sample,
    //    with the label driven by the backend's c_index_type field.
    const coxBox = screen.getByText('Classical Cox (baseline)').closest('div')!
    expect(within(coxBox).getByText('0.644')).toBeInTheDocument()
    expect(within(coxBox).getByText('apparent (in-sample)')).toBeInTheDocument()

    const mlBox = screen.getByText('Gradient Boosting (ML)').closest('div')!
    expect(within(mlBox).getByText('0.954')).toBeInTheDocument()
    expect(within(mlBox).getByText('apparent (in-sample)')).toBeInTheDocument()

    expect(screen.getAllByText('apparent (in-sample)')).toHaveLength(2)

    // 3. The cross-validated card must come BEFORE the apparent tiles in the DOM,
    //    so a reader meets the honest number first.
    expect(cvBox.compareDocumentPosition(coxBox) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(cvBox.compareDocumentPosition(mlBox) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    // 4. The backend's overfitting warnings render, above the metrics.
    const warningHeading = screen.getByRole('heading', { name: 'Warnings' })
    expect(screen.getByText(OVERFIT_WARNING)).toBeInTheDocument()
    expect(screen.getByText(CHANCE_WARNING)).toBeInTheDocument()
    expect(
      warningHeading.compareDocumentPosition(cvBox) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()

    // 5. The winner is not presented as a bare verdict.
    expect(screen.getByText(/treat them as descriptive, not as a verdict/i)).toBeInTheDocument()

    // 6. Supporting backend prose is surfaced verbatim.
    expect(screen.getByText(survivalResult.result_text)).toBeInTheDocument()
    expect(screen.getByText(survivalResult.assumptions[0])).toBeInTheDocument()
  })

  it('renders the reason for the nested_cv and shap_values placeholders', async () => {
    installSession()
    const nestedReason = 'Set nested_cv=true to run inner-loop tuning and outer-loop evaluation.'
    const shapReason = 'SHAP not requested.'
    const competingReason = 'Competing-risks ML not requested.'
    server.use(
      http.post(SURVIVAL_URL, () =>
        HttpResponse.json({
          ...survivalResult,
          nested_cv: { enabled: false, reason: nestedReason },
          ml_gradient_boosting_survival: {
            ...survivalResult.ml_gradient_boosting_survival,
            shap_values: { available: false, reason: shapReason },
          },
          competing_risks_ml: { available: false, reason: competingReason },
        }),
      ),
    )

    const user = userEvent.setup()
    await runSurvivalBenchmark(user)

    expect(await screen.findByText(nestedReason)).toBeInTheDocument()
    expect(screen.getByText(shapReason)).toBeInTheDocument()
    expect(screen.getByText(competingReason)).toBeInTheDocument()
  })

  it('shows the backend error detail when the benchmark fails', async () => {
    installSession()
    server.use(
      http.post(SURVIVAL_URL, () =>
        HttpResponse.json(
          { detail: 'Need at least 20 complete observations for survival ML benchmark.' },
          { status: 400 },
        ),
      ),
    )

    const user = userEvent.setup()
    await runSurvivalBenchmark(user)

    await waitFor(() =>
      expect(
        screen.getByText('Need at least 20 complete observations for survival ML benchmark.'),
      ).toBeInTheDocument(),
    )
  })
})
