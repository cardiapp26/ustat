import { describe, expect, it } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { server } from '../test/server'
import SyntaxView from './SyntaxView'
import { paramsOf } from '../lib/analysisParams'
import type { SavedAnalysis } from '../store'

function analysis(over: Partial<SavedAnalysis> = {}): SavedAnalysis {
  return {
    id: 'a1',
    name: 'Model 1',
    panel: 'models',
    tab: 'models',
    createdAt: 1,
    snapshot: { stamp: { paramsKey: '{"model":"logistic","outcome":"dm","predictors":["age"]}' } },
    ...over,
  }
}

describe('paramsOf', () => {
  it('recovers the run params from the stamp paramsKey', () => {
    expect(paramsOf(analysis())).toEqual({ model: 'logistic', outcome: 'dm', predictors: ['age'] })
  })

  it('returns {} for a missing or unparsable key', () => {
    expect(paramsOf(analysis({ snapshot: null }))).toEqual({})
    expect(paramsOf(analysis({ snapshot: { stamp: { paramsKey: 'not json' } } }))).toEqual({})
  })
})

describe('SyntaxView', () => {
  it('fetches the translation and switches between Python and R', async () => {
    let sent: unknown = null
    server.use(
      http.post('/api/project/syntax', async ({ request }) => {
        sent = await request.json()
        return HttpResponse.json({
          title: 'Logistic regression',
          python: 'import statsmodels\n# python code',
          r: 'glm(dm ~ age)\n# r code',
        })
      }),
    )
    render(<SyntaxView analysis={analysis()} onClose={() => {}} />)
    await waitFor(() => expect(screen.getByText(/python code/)).toBeInTheDocument())
    expect(sent).toEqual({
      panel: 'models',
      params: { model: 'logistic', outcome: 'dm', predictors: ['age'] },
    })
    fireEvent.click(screen.getByText('R'))
    expect(screen.getByText(/r code/)).toBeInTheDocument()
  })

  it('shows the honest no-translation state with the raw params', async () => {
    server.use(
      http.post('/api/project/syntax', () =>
        HttpResponse.json({ title: null, python: null, r: null })),
    )
    render(<SyntaxView analysis={analysis({ panel: 'meta' })} onClose={() => {}} />)
    await waitFor(() =>
      expect(screen.getByText(/No translation for this analysis yet/)).toBeInTheDocument())
    expect(screen.getByText(/"outcome": "dm"/)).toBeInTheDocument()
  })
})
