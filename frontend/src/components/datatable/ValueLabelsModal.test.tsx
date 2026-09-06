import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { describe, expect, it, vi } from 'vitest'
import type { ColMeta, Session } from '../../store'
import { server } from '../../test/server'
import { makeSession } from '../../test/testUtils'
import { ValueLabelsModal } from './ValueLabelsModal'

const columns: ColMeta[] = [{ name: 'bethesda', dtype: 'int64', kind: 'categorical' }]
const preview = [{ bethesda: 0 }, { bethesda: 1 }, { bethesda: 2 }, { bethesda: 5 }]

function setup(over: Partial<Parameters<typeof ValueLabelsModal>[0]> = {}) {
  const onClose = vi.fn()
  const setDraft = vi.fn()
  const onApplied = vi.fn()
  const session: Session = makeSession({ columns, preview, rows: preview.length })
  render(
    <ValueLabelsModal
      colName="bethesda" columns={columns} preview={preview}
      draft={{ 0: 'yok' }} setDraft={setDraft} session={session} onClose={onClose}
      onApplied={onApplied}
      {...over}
    />,
  )
  const dialog = screen.getByRole('dialog')
  const backdrop = dialog.parentElement as HTMLElement
  return { onClose, setDraft, onApplied, dialog, backdrop }
}

/** The two calls a swap makes: the rewrite, then the reread of the column. */
function stubSwap(detail?: string) {
  const posted: Array<{ column: string; labels: Record<string, string> }> = []
  server.use(
    http.post('/api/sessions/test-session/swap_value_labels', async ({ request }) => {
      posted.push((await request.json()) as (typeof posted)[number])
      if (detail) return HttpResponse.json({ detail }, { status: 422 })
      return HttpResponse.json({ changed: 1, value_labels: { yok: '0' } })
    }),
    http.get('/api/stats/test-session/refresh', () =>
      HttpResponse.json({
        rows: 4,
        columns: [{ name: 'bethesda', dtype: 'object', kind: 'categorical', value_labels: { yok: '0' } }],
        preview: [{ bethesda: 'yok' }, { bethesda: 1 }, { bethesda: 2 }, { bethesda: 5 }],
      }),
    ),
  )
  return posted
}

const swapButton = () => screen.getByRole('button', { name: /swap|rewrite/i })

describe('ValueLabelsModal', () => {
  it('lists each distinct value in numeric order with its label', () => {
    setup()
    expect(screen.getByDisplayValue('yok')).toBeInTheDocument()
    const codes = [...screen.getByRole('dialog').querySelectorAll('span.font-mono')]
      .map((s) => s.textContent)
    expect(codes).toEqual(['0', '1', '2', '5'])
  })

  it('closes on a click that starts and ends on the backdrop', () => {
    const { onClose, backdrop } = setup()
    fireEvent.pointerDown(backdrop)
    fireEvent.click(backdrop)
    expect(onClose).toHaveBeenCalled()
  })

  it('stays open when a drag inside it merely ends on the backdrop', () => {
    // Reported: selecting a label by dragging across the input closed the
    // dialog and threw the typing away. The click fires on the nearest common
    // ancestor of press and release — the backdrop — so closing on that alone
    // is wrong. What matters is where the press started.
    const { onClose, dialog, backdrop } = setup()
    const input = screen.getByDisplayValue('yok')
    fireEvent.pointerDown(input)
    fireEvent.pointerUp(backdrop)
    fireEvent.click(backdrop)          // bubbles from the common ancestor
    expect(onClose).not.toHaveBeenCalled()
    expect(dialog).toBeInTheDocument()
  })

  it('does not close when the dialog itself is clicked', async () => {
    const { onClose } = setup()
    await userEvent.click(screen.getByDisplayValue('yok'))
    expect(onClose).not.toHaveBeenCalled()
  })

  it('moves when the header is dragged, and only then becomes positioned', () => {
    const { dialog } = setup()
    expect(dialog.style.position).toBe('')      // centred by the flex parent
    const header = screen.getByTitle('Drag to move')
    header.setPointerCapture = vi.fn()
    header.releasePointerCapture = vi.fn()

    fireEvent.pointerDown(header, { clientX: 200, clientY: 100, pointerId: 1 })
    fireEvent.pointerMove(header, { clientX: 320, clientY: 260, pointerId: 1 })
    fireEvent.pointerUp(header, { pointerId: 1 })

    expect(dialog.style.position).toBe('fixed')
    // jsdom reports a zero rect, so the offset is the raw pointer delta.
    expect(parseFloat(dialog.style.top)).toBeGreaterThan(0)
  })

  it('never lets the header be dragged above the top of the window', () => {
    // Dragged off the top, the handle is unreachable and the dialog is stuck.
    const { dialog } = setup()
    const header = screen.getByTitle('Drag to move')
    header.setPointerCapture = vi.fn()
    header.releasePointerCapture = vi.fn()

    fireEvent.pointerDown(header, { clientX: 100, clientY: 100, pointerId: 1 })
    fireEvent.pointerMove(header, { clientX: 100, clientY: -400, pointerId: 1 })
    expect(parseFloat(dialog.style.top)).toBeGreaterThanOrEqual(0)
  })

  it('does not start a drag from the close button', () => {
    const { onClose, dialog } = setup()
    const close = screen.getByRole('button', { name: 'Close' })
    fireEvent.pointerDown(close, { clientX: 300, clientY: 100, pointerId: 1 })
    expect(dialog.style.position).toBe('')
    fireEvent.click(close)
    expect(onClose).toHaveBeenCalled()
  })

  it('will not swap a column where nothing has been labelled yet', () => {
    setup({ draft: {} })
    expect(swapButton()).toBeDisabled()
  })

  it('asks once before rewriting, naming what the swap will produce', async () => {
    setup()
    const posted = stubSwap()
    await userEvent.click(swapButton())
    // Armed, not fired: the data is not touched by the first click.
    expect(posted).toHaveLength(0)
    expect(screen.getByText(/yok = 0/)).toBeInTheDocument()
  })

  it('puts the labels in the cells and the cells in the labels', async () => {
    const { setDraft, onApplied } = setup()
    const posted = stubSwap()
    await userEvent.click(swapButton())
    await userEvent.click(swapButton())

    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toEqual({ column: 'bethesda', labels: { 0: 'yok' } })
    // The dialog now reads the other way round: the label is the code.
    await waitFor(() => expect(setDraft).toHaveBeenCalledWith({ yok: '0' }))
    expect(onApplied).toHaveBeenCalled()
  })

  it('disarms the confirmation when a label is edited after it was armed', async () => {
    setup()
    stubSwap()
    await userEvent.click(swapButton())
    expect(screen.getByText(/yok = 0/)).toBeInTheDocument()
    fireEvent.change(screen.getByDisplayValue('yok'), { target: { value: 'yok2' } })
    expect(screen.queryByText(/yok = 0/)).not.toBeInTheDocument()
  })

  it("shows the server's reason when a swap would merge two values", async () => {
    const { setDraft } = setup()
    stubSwap('Swapping would merge two distinct values into one')
    await userEvent.click(swapButton())
    await userEvent.click(swapButton())
    expect(await screen.findByText(/merge two distinct values/)).toBeInTheDocument()
    expect(setDraft).not.toHaveBeenCalled()
  })

  it('closes on Escape', () => {
    const { onClose } = setup()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })
})
