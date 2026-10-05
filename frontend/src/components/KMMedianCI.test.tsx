import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import SurvivalAdvancedPanel from './SurvivalAdvancedPanel'

afterEach(() => clearSession())

const survivalSession = () =>
  makeSession({
    columns: [
      { name: 'TIME', dtype: 'float64', kind: 'numeric' },
      { name: 'EVENT', dtype: 'int64', kind: 'numeric' },
      { name: 'GROUP', dtype: 'object', kind: 'categorical' },
    ],
    preview: [
      { TIME: 100, EVENT: 1, GROUP: 'A' },
      { TIME: 200, EVENT: 0, GROUP: 'B' },
    ],
  })

function selectAfterLabel(labelText: string): HTMLSelectElement {
  const label = screen.getByText(labelText)
  return within(label.parentElement as HTMLElement).getByRole('combobox') as HTMLSelectElement
}

describe('Kaplan-Meier median survival 95% CI', () => {
  it('shows [low, high] per group and NR for undefined bounds or a missing median', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/models/survival/km', () =>
        HttpResponse.json({
          groups: [
            {
              group: 'A', n: 60, events: 40, median_survival: 125,
              median_survival_ci_low: 98, median_survival_ci_high: null,
              curve: [{ time: 0, survival: 1 }, { time: 150, survival: 0.4 }],
            },
            {
              group: 'B', n: 60, events: 5, median_survival: null,
              median_survival_ci_low: null, median_survival_ci_high: null,
              curve: [{ time: 0, survival: 1 }, { time: 300, survival: 0.9 }],
            },
          ],
          logrank: { p: 0.032, chi2: 4.6 },
          n_total: 120,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await user.selectOptions(selectAfterLabel('Duration (time)'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    await user.selectOptions(selectAfterLabel('Group (optional)'), 'GROUP')
    await user.click(screen.getByRole('button', { name: /run kaplan-meier/i }))

    expect(await screen.findByRole('columnheader', { name: /median 95% ci/i })).toBeInTheDocument()
    // Bound reached for the lower side only: upper bound prints as NR.
    expect(screen.getByRole('cell', { name: '[98, NR]' })).toBeInTheDocument()
    // Median not reached: both bounds NR.
    expect(screen.getByRole('cell', { name: '[NR, NR]' })).toBeInTheDocument()
    // The point-estimate median column is unchanged.
    expect(screen.getByRole('cell', { name: '125' })).toBeInTheDocument()
  })

  it('falls back to a dash when an older response carries no CI fields', async () => {
    installSession(survivalSession())
    server.use(
      http.post('/api/models/survival/km', () =>
        HttpResponse.json({
          groups: [
            { group: 'A', n: 2, events: 2, median_survival: 125, curve: [{ time: 0, survival: 1 }, { time: 150, survival: 0 }] },
          ],
          n_total: 2,
        }),
      ),
    )
    const user = userEvent.setup()
    render(<SurvivalAdvancedPanel />)
    await user.selectOptions(selectAfterLabel('Duration (time)'), 'TIME')
    await user.selectOptions(selectAfterLabel('Event (0/1)'), 'EVENT')
    await user.click(screen.getByRole('button', { name: /run kaplan-meier/i }))
    expect(await screen.findByRole('columnheader', { name: /median 95% ci/i })).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: '-' })).toBeInTheDocument()
  })
})
