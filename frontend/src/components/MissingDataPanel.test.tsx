import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import type { ColMeta } from '../store'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import MissingDataPanel from './MissingDataPanel'

afterEach(() => clearSession())

// MSW + jsdom cannot round-trip multipart File parts: the XHR interceptor
// re-serializes jsdom FormData through undici, which emits the file as an
// empty "blob" part, and undici's own multipart parser then throws inside
// request.formData(). String fields survive intact, so the reference-impute
// tests assert against the raw multipart text instead (jsdom-only quirk —
// real browsers post real File bytes and the backend path is unaffected).
async function readMultipartFields(request: Request): Promise<Record<string, string>> {
  const raw = await request.text()
  const fields: Record<string, string> = {}
  const re = /name="([^"]+)"(?:; filename="[^"]*")?[^\r\n]*\r\n(?:Content-Type: [^\r\n]*\r\n)?\r\n([\s\S]*?)(?=\r\n------|$)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(raw)) !== null) {
    fields[m[1]] = m[2]
  }
  return fields
}

// Session with actual missing values so the Overview table renders rows
// (installSession()'s default preview has no nulls/blanks).
const columnsWithMissing: ColMeta[] = [
  { name: 'AGE', dtype: 'float64', kind: 'numeric' },
  { name: 'LDL', dtype: 'float64', kind: 'numeric' },
  { name: 'GROUP', dtype: 'object', kind: 'categorical' },
]

function installMissingSession() {
  installSession(
    makeSession({
      columns: columnsWithMissing,
      preview: [
        { AGE: 55, LDL: null, GROUP: 'A' },
        { AGE: null, LDL: 140, GROUP: 'B' },
        { AGE: 48, LDL: 110, GROUP: '' },
      ],
    }),
  )
}

describe('MissingDataPanel', () => {
  it('shows an upload prompt without an active session', () => {
    clearSession()
    render(<MissingDataPanel />)
    expect(screen.getByText(/upload data first/i)).toBeInTheDocument()
  })

  it('shows the all-clear message when no columns have missing values', () => {
    installSession()
    render(<MissingDataPanel />)
    expect(screen.getByText(/no missing values detected/i)).toBeInTheDocument()
  })

  it('lists columns with missing data in the Overview sub-tab', () => {
    installMissingSession()
    render(<MissingDataPanel />)
    expect(screen.getByRole('tab', { name: /missing data overview/i })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    const table = screen.getAllByRole('table')[0]
    expect(within(table).getByText('AGE')).toBeInTheDocument()
    expect(within(table).getByText('LDL')).toBeInTheDocument()
    expect(within(table).getByText('GROUP')).toBeInTheDocument()
  })

  it('runs Little\'s MCAR test + missingness diagnostics after selecting columns', async () => {
    installMissingSession()
    server.use(
      http.post('/api/compute/test-session/missing_diagnostics', () =>
        HttpResponse.json({
          columns: [
            { name: 'AGE', n_missing: 1, pct: 33.3, kind: 'numeric', is_numeric: true, depends_on: ['LDL'], likely: 'MAR' },
            { name: 'LDL', n_missing: 1, pct: 33.3, kind: 'numeric', is_numeric: true, depends_on: [], likely: 'MCAR' },
          ],
          overall_hint: 'Some dependence detected.',
          recommendation: 'Use MICE.',
          any_mar: true,
        }),
      ),
      http.post('/api/missing_data/mcar_test', () =>
        HttpResponse.json({ statistic: 4.21, df: 2, p: 0.12, significant: false }),
      ),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)

    const table = screen.getAllByRole('table')[0]
    const ageRow = within(table).getByText('AGE').closest('tr')!
    const ldlRow = within(table).getByText('LDL').closest('tr')!
    await user.click(within(ageRow).getByRole('checkbox'))
    await user.click(within(ldlRow).getByRole('checkbox'))

    await user.click(screen.getByRole('button', { name: /analyze missingness/i }))

    await waitFor(() => expect(screen.getByText(/little's mcar test/i)).toBeInTheDocument())
    expect(
      screen.getAllByText(
        (_, el) => el?.tagName === 'DIV' && (el?.textContent ?? '').includes('χ²=4.21, df=2, p=0.120.'),
      ).length,
    ).toBeGreaterThan(0)
    expect(screen.getByText(/Some dependence detected\./)).toBeInTheDocument()
    expect(screen.getByText(/Use MICE\./)).toBeInTheDocument()
    expect(screen.getByText(/missingness related to LDL/)).toBeInTheDocument()
  })

  it('shows a diagnostics error from the backend', async () => {
    installMissingSession()
    server.use(
      http.post('/api/compute/test-session/missing_diagnostics', () =>
        HttpResponse.json({ detail: 'Diagnostics failed' }, { status: 500 }),
      ),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)
    await user.click(screen.getAllByRole('checkbox')[0])
    await user.click(screen.getByRole('button', { name: /analyze missingness/i }))

    await waitFor(() => expect(screen.getByText('Diagnostics failed')).toBeInTheDocument())
  })

  it('previews reference dataset imputation', async () => {
    installMissingSession()
    server.use(
      http.post('/api/missing_data/external_impute_reference_columns', () =>
        HttpResponse.json({
          n_rows: 2,
          columns: [
            { name: 'age', dtype: 'int64', kind: 'numeric', n_missing: 0 },
            { name: 'ldl', dtype: 'int64', kind: 'numeric', n_missing: 0 },
            { name: 'REFERENCE_ONLY', dtype: 'object', kind: 'categorical', n_missing: 0 },
          ],
        }),
      ),
      http.post('/api/missing_data/external_impute_preview', async ({ request }) => {
        const fd = await readMultipartFields(request)
        expect(fd.target).toBe('LDL')
        expect(fd.reference_target).toBe('ldl')
        expect(fd.predictors).toBe(JSON.stringify(['age']))
        expect(fd.predictor_mappings).toBe(JSON.stringify({ age: 'AGE' }))
        // File bytes don't survive the jsdom↔undici round-trip; presence is what matters.
        expect(Object.keys(fd)).toContain('file')
        return HttpResponse.json({
          target: 'LDL',
          reference_target: 'ldl',
          predictors: ['AGE'],
          reference_predictors: ['age'],
          method: 'PMM',
          mechanism: 'unknown',
          n_missing_target: 1,
          n_imputed: 1,
          reference_rows: 2,
          reference_complete_rows: 2,
          preview_rows: [{ row_index: 0, imputed_value: 128, predictors_missing: 0 }],
          result_text: "1 missing value(s) in 'LDL' were imputed using 1 predictor(s).",
        })
      }),
      http.post('/api/missing_data/external_impute_transfer', async ({ request }) => {
        const body = await request.json() as {
          session_id: string;
          target: string;
          preview_rows: Array<{ row_index: number; imputed_value: unknown }>;
        }
        expect(body.session_id).toBe('test-session')
        expect(body.target).toBe('LDL')
        expect(body.preview_rows).toEqual([{ row_index: 0, imputed_value: 128 }])
        return HttpResponse.json({
          target: 'LDL',
          n_imputed: 1,
          applied: true,
          result_text: "1 previewed value(s) were transferred into 'LDL'.",
        })
      }),
      http.get('/api/stats/test-session/refresh', () =>
        HttpResponse.json({
          columns: columnsWithMissing,
          preview: [
            { AGE: 55, LDL: 128, GROUP: 'A' },
            { AGE: null, LDL: 140, GROUP: 'B' },
            { AGE: 48, LDL: 110, GROUP: '' },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)

    await user.click(screen.getByRole('tab', { name: /reference imputation/i }))
    await user.selectOptions(screen.getByLabelText(/current missing target/i), 'LDL')
    await user.upload(
      screen.getByLabelText(/reference dataset/i),
      new File(['age,ldl\n55,128\n61,140\n'], 'reference.csv', { type: 'text/csv' }),
    )
    await waitFor(() => expect(screen.getByLabelText(/reference target match/i)).toHaveValue('ldl'))
    await waitFor(() => expect(screen.getAllByText('REFERENCE_ONLY').length).toBeGreaterThan(0))
    expect(screen.getByDisplayValue('AGE')).toBeInTheDocument()
    const agePredictor = screen.getByLabelText('age')
    await user.click(agePredictor)
    await user.click(screen.getByRole('button', { name: /preview target estimates/i }))

    await waitFor(() => expect(screen.getByText(/1 missing value/)).toBeInTheDocument())
    expect(screen.getByText('128')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /transfer data/i })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: /transfer data/i }))
    await waitFor(() => expect(screen.getByText(/1 value\(s\) transferred into LDL/)).toBeInTheDocument())
  })

  it('sends stratify_by when a stratification column is selected', async () => {
    const columnsWithStratum: ColMeta[] = [
      { name: 'AGE', dtype: 'float64', kind: 'numeric' },
      { name: 'LDL', dtype: 'float64', kind: 'numeric' },
      { name: 'DM', dtype: 'int64', kind: 'numeric' },
    ]
    installSession(
      makeSession({
        columns: columnsWithStratum,
        preview: [
          { AGE: 55, LDL: null, DM: 0 },
          { AGE: 61, LDL: 140, DM: 1 },
        ],
      }),
    )

    let stratifyBySent: string | null = null
    server.use(
      http.post('/api/missing_data/external_impute_reference_columns', () =>
        HttpResponse.json({
          n_rows: 2,
          columns: [
            { name: 'age', dtype: 'int64', kind: 'numeric', n_missing: 0 },
            { name: 'ldl', dtype: 'int64', kind: 'numeric', n_missing: 0 },
            { name: 'dm', dtype: 'int64', kind: 'numeric', n_missing: 0 },
          ],
        }),
      ),
      http.post('/api/missing_data/external_impute_preview', async ({ request }) => {
        const fd = await readMultipartFields(request)
        stratifyBySent = fd.stratify_by ?? null
        return HttpResponse.json({
          target: 'LDL',
          reference_target: 'ldl',
          predictors: ['AGE'],
          method: 'PMM',
          mechanism: 'unknown',
          n_missing_target: 1,
          n_imputed: 1,
          reference_rows: 2,
          reference_complete_rows: 2,
          preview_rows: [{ row_index: 0, imputed_value: 128, predictors_missing: 0, stratum: '0' }],
          result_text: "1 missing value(s) in 'LDL' were imputed using 1 predictor(s).",
        })
      }),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)

    await user.click(screen.getByRole('tab', { name: /reference imputation/i }))
    await user.selectOptions(screen.getByLabelText(/current missing target/i), 'LDL')
    await user.upload(
      screen.getByLabelText(/reference dataset/i),
      new File(['age,ldl,dm\n55,128,0\n61,140,1\n'], 'reference.csv', { type: 'text/csv' }),
    )
    await waitFor(() => expect(screen.getByLabelText(/reference target match/i)).toHaveValue('ldl'))

    const agePredictor = screen.getByLabelText('age')
    await user.click(agePredictor)

    await user.selectOptions(screen.getByLabelText(/stratify by/i), 'DM')
    await user.click(screen.getByRole('button', { name: /preview target estimates/i }))

    await waitFor(() => expect(screen.getByText('128')).toBeInTheDocument())
    const previewTable = screen.getAllByRole('table').find((t) =>
      within(t).queryByText(/estimated value/i),
    )!
    const row = within(previewTable).getByText('128').closest('tr')!
    expect(within(row).getAllByText('0').length).toBeGreaterThanOrEqual(2)
    expect(stratifyBySent).toBe('DM')
  })

  it('previews PMM imputation and transfers values into original columns', async () => {
    installMissingSession()
    server.use(
      http.post('/api/survival_advanced/mice_preview', async ({ request }) => {
        const body = (await request.json()) as {
          session_id: string
          columns: string[]
          max_iter: number
          random_state: number
          mechanism: string
        }
        expect(body.session_id).toBe('test-session')
        expect(body.columns).toEqual(expect.arrayContaining(['AGE', 'LDL']))
        expect(body.mechanism).toBe('unknown')
        return HttpResponse.json({
          preview_rows: [
            { row_index: 1, column: 'AGE', imputed_value: 58 },
            { row_index: 0, column: 'LDL', imputed_value: 125 },
          ],
          columns: [
            { column: 'AGE', method: 'PMM', n_imputed: 1, mean_imputed: 58, min_imputed: 58, max_imputed: 58 },
            { column: 'LDL', method: 'PMM', n_imputed: 1, mean_imputed: 125, min_imputed: 125, max_imputed: 125 },
          ],
          total_imputed: 2,
          result_text: 'Preview: 2 missing values will be imputed.',
          methods_text: 'PMM preview.',
          export_rows: [['Column', 'Method', 'N Imputed', 'Mean / Mode', 'Min', 'Max'], ['AGE', 'PMM', 1, 58, 58, 58]],
          preview_only: true,
        })
      }),
      http.post('/api/survival_advanced/mice_transfer', async ({ request }) => {
        const body = (await request.json()) as {
          session_id: string
          preview_rows: Array<{ row_index: number; column: string; imputed_value: unknown }>
        }
        expect(body.session_id).toBe('test-session')
        expect(body.preview_rows).toEqual([
          { row_index: 1, column: 'AGE', imputed_value: 58 },
          { row_index: 0, column: 'LDL', imputed_value: 125 },
        ])
        return HttpResponse.json({
          n_imputed: { AGE: 1, LDL: 1 },
          total_imputed: 2,
          columns: ['AGE', 'LDL'],
          result_text: 'Transferred 2 previewed value(s) into original columns: AGE, LDL.',
        })
      }),
      http.get('/api/stats/test-session/refresh', () =>
        HttpResponse.json({
          columns: columnsWithMissing,
          preview: [
            { AGE: 55, LDL: 125, GROUP: 'A' },
            { AGE: 58, LDL: 140, GROUP: 'B' },
            { AGE: 48, LDL: 110, GROUP: '' },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)

    const table = screen.getAllByRole('table')[0]
    const ageRow = within(table).getByText('AGE').closest('tr')!
    const ldlRow = within(table).getByText('LDL').closest('tr')!
    await user.click(within(ageRow).getByRole('checkbox'))
    await user.click(within(ldlRow).getByRole('checkbox'))

    await user.click(screen.getByRole('button', { name: /preview pmm/i }))
    await waitFor(() => expect(screen.getByText(/preview:/i)).toBeInTheDocument())
    expect(screen.getByText('125')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /transfer to original columns/i })).toBeEnabled()

    await user.click(screen.getByRole('button', { name: /transfer to original columns/i }))
    await waitFor(() =>
      expect(screen.getByText(/2 value\(s\) transferred into original columns/i)).toBeInTheDocument(),
    )
  })

  it('runs MNAR sensitivity and renders each sub-analysis', async () => {
    installMissingSession()
    let sentBody: { session_id: string; columns: string[]; delta_values: number[]; model_type: string } | null = null
    server.use(
      http.post('/api/models/mnar_sensitivity', async ({ request }) => {
        sentBody = (await request.json()) as typeof sentBody
        return HttpResponse.json({
          test: 'MNAR Missing Data Sensitivity Analysis',
          n: 3,
          columns: ['AGE', 'LDL'],
          pattern_mixture_model: {
            method: 'pattern_mixture_delta_adjustment',
            n_imputations: 5,
            delta_values: [-1, 0, 1],
            scenarios: [
              { delta: -1, pooled_means: { AGE: 51.1111, LDL: 121.2222 } },
              { delta: 0, pooled_means: { AGE: 52.3333, LDL: 122.4444 } },
              { delta: 1, pooled_means: { AGE: 53.5555, LDL: 123.6666 } },
            ],
            interpretation: 'Delta shifts apply only to originally missing cells.',
          },
          model_delta_sensitivity: {
            model_type: 'logistic',
            results: [{ delta: 0, log_odds: 0.9191, odds_ratio: 2.507, se: 0.1717 }],
          },
          heckman_selection_model: {
            available: true,
            n_total: 3,
            n_observed_outcome: 2,
            selection_rate: 0.6667,
            inverse_mills_ratio_p: 0.0321,
            selection_bias_signal: true,
            outcome_coefficients: [{ variable: 'inverse_mills_ratio', estimate: 1.2345, se: 0.5432, p: 0.0321 }],
          },
          isni: { available: true, indices: [{ variable: 'AGE', isni: 0.7654, high_sensitivity: true }] },
          mice_convergence_diagnostics: {
            variables: { AGE: { r_hat_proxy: 1.0808, converged: true } },
            warning: 'R-hat is approximated from independent chains.',
          },
          imputation_model_diagnostics: {
            checks: [{ variable: 'AGE', available: true, observed_mean: 51.5, imputed_mean: 52.75, ks_p: 0.4321 }],
          },
          congeniality_assessment: {
            congenial: true,
            analysis_variables_missing_from_imputation: [],
            passive_variables: [],
            recommendation: 'Imputation model covers the listed analysis variables.',
          },
          passive_imputation: { formulas: {}, preview: {} },
          survival_specific_imputation: { enabled: false, auxiliary_variables: [] },
          auxiliary_variable_guidance: {
            recommended_auxiliary_variables: [
              { target: 'AGE', candidate: 'LDL', missingness_corr_abs: 0.11, value_corr_abs: 0.22, priority_score: 0.9876 },
            ],
            method_note: 'Prioritizes variables associated with missingness.',
          },
          survival_mnar_sensitivity: {
            available: false,
            reason: 'Survival MNAR not requested or duration/event/predictors missing.',
          },
          warnings: ['Potential MICE convergence concern for: LDL.'],
          assumptions: [
            { name: 'MNAR scenario analysis', met: true, detail: 'Delta values encode unverifiable assumptions.' },
          ],
          result_text: 'MNAR sensitivity analysis ran for 2 variable(s) across 3 delta scenario(s).',
          r_code: 'library(mice)',
        })
      }),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)

    await user.click(screen.getByRole('tab', { name: /mnar sensitivity/i }))
    const varList = screen.getByRole('group', { name: /variables to analyse/i })
    await user.click(within(varList).getByLabelText(/^AGE/))
    await user.click(within(varList).getByLabelText(/^LDL/))
    await user.click(screen.getByRole('button', { name: /run mnar sensitivity/i }))

    await waitFor(() =>
      expect(screen.getByText(/MNAR sensitivity analysis ran for 2 variable\(s\)/)).toBeInTheDocument(),
    )

    expect(sentBody).toEqual({
      session_id: 'test-session',
      columns: ['AGE', 'LDL'],
      delta_values: [-1, 0, 1],
      model_type: 'logistic',
    })

    // Pattern-mixture scenarios
    expect(screen.getByText('53.5555')).toBeInTheDocument()
    expect(screen.getByText('121.2222')).toBeInTheDocument()
    // Model-based delta sensitivity
    expect(screen.getByText('0.9191')).toBeInTheDocument()
    // Heckman
    expect(screen.getByText('1.2345')).toBeInTheDocument()
    // ISNI
    expect(screen.getByText('0.7654')).toBeInTheDocument()
    // MICE convergence
    expect(screen.getByText('1.0808')).toBeInTheDocument()
    // Imputation model diagnostics
    expect(screen.getByText(/observed mean 51\.5 → imputed mean 52\.75/)).toBeInTheDocument()
    // Congeniality
    expect(screen.getByText(/Imputation model covers the listed analysis variables\./)).toBeInTheDocument()
    // Auxiliary guidance
    expect(screen.getByText('0.9876')).toBeInTheDocument()
    // Warnings + assumptions + R code
    expect(screen.getByText('Potential MICE convergence concern for: LDL.')).toBeInTheDocument()
    expect(screen.getByText(/Delta values encode unverifiable assumptions\./)).toBeInTheDocument()
    expect(screen.getByText('library(mice)')).toBeInTheDocument()
  })

  it('shows the reason for every MNAR sub-analysis the backend could not compute', async () => {
    installMissingSession()
    server.use(
      http.post('/api/models/mnar_sensitivity', () =>
        HttpResponse.json({
          test: 'MNAR Missing Data Sensitivity Analysis',
          n: 3,
          columns: ['AGE'],
          pattern_mixture_model: {
            scenarios: [{ delta: 0, pooled_means: { AGE: 52.3333 } }],
          },
          model_delta_sensitivity: null,
          heckman_selection_model: {
            available: false,
            reason: 'Selection equation failed: Singular matrix in probit stage.',
          },
          isni: { available: false, reason: 'ISNI not requested or outcome/predictors missing.' },
          mice_convergence_diagnostics: { variables: {}, warning: 'R-hat is approximated.' },
          imputation_model_diagnostics: {
            checks: [{ variable: 'AGE', available: false, reason: 'Not enough observed/imputed values.' }],
          },
          congeniality_assessment: { congenial: true, recommendation: 'Covered.' },
          passive_imputation: { formulas: {}, preview: {} },
          survival_specific_imputation: { enabled: false, auxiliary_variables: [] },
          auxiliary_variable_guidance: { recommended_auxiliary_variables: [] },
          survival_mnar_sensitivity: { available: false, reason: 'Need at least 20 complete rows.' },
          warnings: [],
          assumptions: [],
          result_text: 'MNAR sensitivity analysis ran for 1 variable(s).',
          r_code: 'library(mice)',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)

    await user.click(screen.getByRole('tab', { name: /mnar sensitivity/i }))
    const varList = screen.getByRole('group', { name: /variables to analyse/i })
    await user.click(within(varList).getByLabelText(/^AGE/))
    await user.click(screen.getByRole('button', { name: /run mnar sensitivity/i }))

    await waitFor(() =>
      expect(screen.getByText(/MNAR sensitivity analysis ran for 1 variable\(s\)/)).toBeInTheDocument(),
    )

    // Both explicitly unavailable blocks surface their own reason.
    expect(screen.getByText(/Selection equation failed: Singular matrix in probit stage\./)).toBeInTheDocument()
    expect(screen.getByText(/ISNI not requested or outcome\/predictors missing\./)).toBeInTheDocument()
    expect(screen.getByText(/Need at least 20 complete rows\./)).toBeInTheDocument()
    // Per-variable posterior predictive checks carry their own reason too.
    expect(screen.getByText(/Not enough observed\/imputed values\./)).toBeInTheDocument()
    // A null block falls back to a readable explanation instead of an empty tile.
    expect(screen.getByText(/requires an outcome model with predictors/i)).toBeInTheDocument()
    // Blocks that did compute still render.
    expect(screen.getByText('52.3333')).toBeInTheDocument()
  })

  it('shows an MNAR sensitivity error from the backend', async () => {
    installMissingSession()
    server.use(
      http.post('/api/models/mnar_sensitivity', () =>
        HttpResponse.json({ detail: 'Select at least one variable with missing data.' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)

    await user.click(screen.getByRole('tab', { name: /mnar sensitivity/i }))
    const varList = screen.getByRole('group', { name: /variables to analyse/i })
    await user.click(within(varList).getByLabelText(/^AGE/))
    await user.click(screen.getByRole('button', { name: /run mnar sensitivity/i }))

    await waitFor(() =>
      expect(screen.getByText('Select at least one variable with missing data.')).toBeInTheDocument(),
    )
  })

  it('switches to the Data Cleaning sub-tab', async () => {
    installMissingSession()
    const user = userEvent.setup()
    render(<MissingDataPanel />)

    const cleaningTab = screen.getByRole('tab', { name: /data cleaning/i })
    await user.click(cleaningTab)
    expect(cleaningTab).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: /missing data overview/i })).toHaveAttribute(
      'aria-selected',
      'false',
    )
  })

  it('sends the outcome model only when both outcome and predictors are chosen', async () => {
    // Without an outcome model the backend skips model_delta_sensitivity,
    // Heckman and ISNI, which would leave the Model type selector inert.
    installMissingSession()
    const bodies: Record<string, unknown>[] = []
    server.use(
      http.post('/api/models/mnar_sensitivity', async ({ request }) => {
        bodies.push((await request.json()) as Record<string, unknown>)
        return HttpResponse.json({
          test: 'MNAR Missing Data Sensitivity Analysis',
          n: 3,
          columns: ['AGE'],
          result_text: 'MNAR sensitivity analysis ran for 1 variable(s).',
        })
      }),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)
    await user.click(screen.getByRole('tab', { name: /mnar sensitivity/i }))

    const varList = screen.getByRole('group', { name: /variables to analyse/i })
    await user.click(within(varList).getByLabelText(/^AGE/))

    // 1. Outcome picked but no predictors -> the outcome model must be omitted.
    const outcomeSelect = screen
      .getByText(/outcome \(optional\)/i)
      .closest('label')!
      .querySelector('select')!
    await user.selectOptions(outcomeSelect, 'GROUP')
    await user.click(screen.getByRole('button', { name: /run mnar sensitivity/i }))
    await waitFor(() => expect(bodies).toHaveLength(1))
    expect(bodies[0]).not.toHaveProperty('outcome_col')
    expect(bodies[0]).not.toHaveProperty('predictors')

    // 2. Add a predictor -> now both travel with the request.
    const predList = screen.getByRole('group', { name: /outcome-model predictors/i })
    await user.click(within(predList).getByLabelText(/^AGE/))
    await user.click(screen.getByRole('button', { name: /run mnar sensitivity/i }))
    await waitFor(() => expect(bodies).toHaveLength(2))
    expect(bodies[1].outcome_col).toBe('GROUP')
    expect(bodies[1].predictors).toEqual(['AGE'])
  })

  it('warns when an outcome is chosen but no predictor is', async () => {
    installMissingSession()
    const user = userEvent.setup()
    render(<MissingDataPanel />)
    await user.click(screen.getByRole('tab', { name: /mnar sensitivity/i }))

    expect(screen.queryByText(/or the outcome model is ignored/i)).not.toBeInTheDocument()
    const outcomeSelect = screen
      .getByText(/outcome \(optional\)/i)
      .closest('label')!
      .querySelector('select')!
    await user.selectOptions(outcomeSelect, 'GROUP')
    expect(screen.getByText(/or the outcome model is ignored/i)).toBeInTheDocument()
  })
})
