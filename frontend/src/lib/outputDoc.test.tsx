import { afterEach, describe, expect, it } from 'vitest'
import { render } from '@testing-library/react'
import StaleGuard from '../components/StaleGuard'
import { useStaleGuard } from './staleGuard'
import { sanitizeOutputHtml } from './outputSanitize'
import { captureGuardedArea } from './outputCapture'
import { restoreOutputItems, useOutputDoc } from './outputDoc'
import { applyUiState, collectUiState } from './projectUiState'
import { installSession, clearSession } from '../test/testUtils'

afterEach(() => { useOutputDoc.getState().clear(); clearSession() })

describe('sanitizeOutputHtml', () => {
  it.each([
    ['<img src="x" onerror="alert(1)">', /onerror|<img/],
    ['<img src="https://evil.example/t.png">', /<img/],
    ['<a href="javascript:alert(1)">x</a>', /javascript|href/],
    ['<script>alert(1)</script><p>ok</p>', /script|alert/],
    ['<svg onload="alert(1)"><circle/></svg>', /svg|onload/],
    ['<iframe src="https://evil.example"></iframe>', /iframe/],
    ['<div style="background:url(javascript:alert(1))">x</div>', /style|javascript/],
    ['<p onclick="steal()">x</p>', /onclick/],
  ])('removes %s', (input, forbidden) => {
    expect(sanitizeOutputHtml(input)).not.toMatch(forbidden)
  })

  it('keeps tables, text, classes and embedded PNG figures', () => {
    const png = 'data:image/png;base64,iVBORw0KGgo='
    const out = sanitizeOutputHtml(
      `<h4 class="font-semibold">Model</h4><table><tr><th colspan="2">B</th></tr><tr><td>1.2</td><td>x</td></tr></table><img src="${png}" alt="Figure">`,
    )
    expect(out).toContain('<h4 class="font-semibold">Model</h4>')
    expect(out).toContain('<th colspan="2">B</th>')
    const img = new DOMParser().parseFromString(out, 'text/html').querySelector('img')!
    expect(img.getAttribute('src')).toBe(png)
    expect(img.getAttribute('alt')).toBe('Figure')
  })

  it('escapes text instead of interpreting it', () => {
    const out = sanitizeOutputHtml('<p>&lt;script&gt;x&lt;/script&gt;</p>')
    expect(out).toBe('<p>&lt;script&gt;x&lt;/script&gt;</p>')
  })
})

function Probe({ onId }: { onId: (id: string | undefined) => void }) {
  onId(useStaleGuard().captureId)
  return null
}

describe('captureGuardedArea', () => {
  it('copies exactly the guarded result, without its toolbar', async () => {
    let id: string | undefined
    render(
      <div>
        <p>outside before</p>
        <StaleGuard stale={false}>
          <h4>Logistic Regression</h4>
          <div data-output-skip=""><button>CSV</button><span>Export</span></div>
          <table><tbody><tr><td>age</td><td>1.05</td></tr></tbody></table>
          <Probe onId={(v) => { id = v }} />
        </StaleGuard>
        <p>outside after</p>
      </div>,
    )
    expect(id).toBeTruthy()
    const got = await captureGuardedArea(id!)
    expect(got?.title).toBe('Logistic Regression')
    expect(got?.html).toContain('<td>1.05</td>')
    expect(got?.html).not.toContain('Export')
    expect(got?.html).not.toContain('outside')
  })

  it('drops input groups guarded together with the result', async () => {
    let id: string | undefined
    render(
      <StaleGuard stale={false}>
        <div>
          <div><label>Duration<select><option>fu</option></select></label></div>
          <div><span>Predictors</span><label><input type="checkbox" />age</label></div>
        </div>
        <div><h4>CIF</h4><table><tbody><tr><td>0.59</td></tr></tbody></table>
          <div><label>Width<input defaultValue="auto" /></label></div></div>
        <Probe onId={(v) => { id = v }} />
      </StaleGuard>,
    )
    const got = await captureGuardedArea(id!)
    expect(got?.html).toContain('<td>0.59</td>')
    for (const word of ['Duration', 'Predictors', 'age', 'Width']) expect(got?.html).not.toContain(word)
  })

  it('nested guards share the outermost area', () => {
    const ids: Array<string | undefined> = []
    render(
      <StaleGuard stale={false}>
        <Probe onId={(v) => ids.push(v)} />
        <StaleGuard stale={false}><Probe onId={(v) => ids.push(v)} /></StaleGuard>
      </StaleGuard>,
    )
    expect(ids[0]).toBeTruthy()
    expect(ids[1]).toBe(ids[0])
  })
})

describe('output document store', () => {
  it('adds, moves, annotates and removes items in order', () => {
    const doc = useOutputDoc.getState()
    const a = doc.add({ title: 'A', tab: 'models', html: '<p>a</p>' })
    const b = doc.add({ title: 'B', tab: 'tests', html: '<p>b</p>' })
    doc.move(b.id, -1)
    expect(useOutputDoc.getState().items.map((i) => i.title)).toEqual(['B', 'A'])
    doc.setNote(a.id, 'see Table 2')
    doc.remove(b.id)
    expect(useOutputDoc.getState().items).toMatchObject([{ title: 'A', note: 'see Table 2' }])
  })

  it('re-sanitises items restored from a file', () => {
    const items = restoreOutputItems([
      { id: '1', title: 'T', tab: 'x', createdAt: 1, note: '', html: '<img src=x onerror=alert(1)><p>ok</p>' },
      { nonsense: true },
    ])
    expect(items).toHaveLength(1)
    expect(items[0].html).toBe('<p>ok</p>')
  })

  it('travels with the project UI state', () => {
    installSession()
    useOutputDoc.getState().add({ title: 'Kept', tab: 'models', html: '<p>kept</p>' })
    const saved = JSON.parse(JSON.stringify(collectUiState()))
    useOutputDoc.getState().clear()
    applyUiState(saved)
    expect(useOutputDoc.getState().items.map((i) => i.title)).toEqual(['Kept'])
  })
})
