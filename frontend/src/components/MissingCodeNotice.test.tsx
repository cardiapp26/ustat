import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import { useStore } from '../store'
import MissingCodeNotice from './MissingCodeNotice'

afterEach(() => clearSession())

const SUGGESTION = { value: '999', count: 2, reason: '999 is far above every other value (largest 70)' }

describe('MissingCodeNotice', () => {
  it('stays hidden when nothing looks like a code', async () => {
    installSession()
    server.use(http.get('/api/sessions/test-session/missing_codes', () =>
      HttpResponse.json({ counts: {}, suggestions: {} })))
    const { container } = render(<MissingCodeNotice />)
    await waitFor(() => expect(container).toBeEmptyDOMElement())
  })

  it('declares a proposed code with one click and marks earlier results out of date', async () => {
    installSession()
    let saved: unknown = null
    server.use(
      http.get('/api/sessions/test-session/missing_codes', () =>
        HttpResponse.json({ counts: {}, suggestions: { AGE: [SUGGESTION] } })),
      http.post('/api/sessions/test-session/metadata', async ({ request }) => {
        saved = await request.json()
        return HttpResponse.json({ status: 'ok' })
      }),
    )
    const before = useStore.getState().dataVersion
    render(<MissingCodeNotice />)

    expect(await screen.findByText(/Possible missing-value codes/)).toBeInTheDocument()
    expect(screen.getByText('999 ×2')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Treat as missing' }))

    await waitFor(() => expect(saved).toEqual({ columns: { AGE: { missing_codes: ['999'] } } }))
    await waitFor(() => expect(useStore.getState().dataVersion).toBe(before + 1))
    const age = useStore.getState().session!.columns.find((c) => c.name === 'AGE')
    expect(age?.missing_codes).toEqual(['999'])
  })
})
