import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import ScoreCompositePanel from './ScoreCompositePanel'

afterEach(() => clearSession())

const scoreSession = () =>
  makeSession({
    columns: [
      { name: 'burden', dtype: 'object', kind: 'categorical' },
      { name: 'cha2ds2_va', dtype: 'int64', kind: 'numeric', label: 'CHA2DS2-VA' },
      { name: 'atria', dtype: 'int64', kind: 'numeric', label: 'ATRIA' },
      { name: 'htn', dtype: 'int64', kind: 'numeric', label: 'HTN (H)' },
      { name: 'dm', dtype: 'int64', kind: 'numeric', label: 'Diabetes (D)' },
      { name: 'ckd', dtype: 'int64', kind: 'numeric', label: 'CKD' },
      { name: 'stroke', dtype: 'int64', kind: 'numeric', label: 'Stroke' },
    ],
    preview: [
      { burden: 'LTB', cha2ds2_va: 2, atria: 1, htn: 1, dm: 0, ckd: 0, stroke: 0 },
      { burden: 'HTB', cha2ds2_va: 4, atria: 6, htn: 1, dm: 1, ckd: 1, stroke: 0 },
    ],
  })

const scoreResponse = {
  type: 'score_composite',
  group_col: 'burden',
  groups: ['LTB', 'HTB'],
  scores: [
    {
      score_col: 'cha2ds2_va',
      label: 'CHA2DS2-VA',
      p_text: 'p = 0.558',
      n_by_group: { LTB: 10, HTB: 8 },
      components: [{ component: 'htn', label: 'HTN (H)', p_text: 'p = 1.000' }],
    },
    {
      score_col: 'atria',
      label: 'ATRIA',
      p_text: 'p = 0.352',
      n_by_group: { LTB: 10, HTB: 8 },
      components: [{ component: 'ckd', label: 'CKD', p_text: 'p = 0.500' }],
    },
  ],
  figure: {
    data: [{ type: 'bar', x: [0, 1], y: [70, 80] }],
    layout: { title: { text: 'Score Distributions and Component Prevalence by Group' }, height: 760 },
  },
  method_note: 'Score comparisons use Mann-Whitney U.',
}

describe('ScoreCompositePanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<ScoreCompositePanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('generates a score-composite figure', async () => {
    installSession(scoreSession())
    server.use(http.post('/api/charts/score_composite', () => HttpResponse.json(scoreResponse)))

    const user = userEvent.setup()
    const { container } = render(<ScoreCompositePanel />)

    const selects = Array.from(container.querySelectorAll('select')) as HTMLSelectElement[]
    await user.selectOptions(selects[2], ['htn', 'dm'])
    await user.selectOptions(selects[4], ['ckd', 'stroke'])
    await user.click(screen.getByRole('button', { name: /generate score figure/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    expect(screen.getAllByText('CHA2DS2-VA').length).toBeGreaterThan(0)
    expect(screen.getAllByText('ATRIA').length).toBeGreaterThan(0)
    expect(screen.getByText('p = 0.558')).toBeInTheDocument()
  })

  it('closes every export of the figure once the data changes under it', async () => {
    installSession(scoreSession())
    server.use(http.post('/api/charts/score_composite', () => HttpResponse.json(scoreResponse)))

    const user = userEvent.setup()
    const { container } = render(<ScoreCompositePanel />)
    const selects = Array.from(container.querySelectorAll('select')) as HTMLSelectElement[]
    await user.selectOptions(selects[2], ['htn', 'dm'])
    await user.selectOptions(selects[4], ['ckd', 'stroke'])
    await user.click(screen.getByRole('button', { name: /generate score figure/i }))
    await screen.findByTestId('plotly-mock')

    // TitledPlot never offers the Plotly camera, so its own exporter is the
    // figure's whole export surface.
    const exports = () => [
      screen.getByRole('button', { name: '↓' }),
      screen.getByRole('button', { name: '⧉' }),
    ]
    for (const b of exports()) expect(b).toBeEnabled()

    act(() => useStore.getState().bumpDataVersion())

    expect(await screen.findByText(/This score figure was computed before the data changed/)).toBeInTheDocument()
    for (const b of exports()) expect(b).toBeDisabled()
  })
})
