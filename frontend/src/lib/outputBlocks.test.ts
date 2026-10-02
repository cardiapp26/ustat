import { describe, expect, it } from 'vitest'
import { htmlToBlocks } from './outputBlocks'

describe('htmlToBlocks', () => {
  it('turns a captured result into headings, cards, paragraphs, tables and figures in order', () => {
    const png = 'data:image/png;base64,iVBORw0KGgo='
    const blocks = htmlToBlocks(`
      <div><h4>Multinomial Logistic Regression</h4>
        <div class="grid"><div><p>N</p><p>300</p></div><div><p>AIC</p><p>624.03</p></div></div>
        <p>Reference category: <strong>persistent</strong>.</p>
        <table><thead><tr><th>Predictor</th><th>p</th></tr></thead>
          <tbody><tr><td>age</td><td>&lt;0.001</td></tr></tbody></table>
        <img src="${png}" alt="Figure">
      </div>`)
    expect(blocks).toEqual([
      { type: 'heading', text: 'Multinomial Logistic Regression', level: 4 },
      { type: 'paragraph', text: 'N: 300' },
      { type: 'paragraph', text: 'AIC: 624.03' },
      { type: 'paragraph', text: 'Reference category: persistent.' },
      { type: 'table', rows: [['Predictor', 'p'], ['age', '<0.001']], header_rows: 1 },
      { type: 'image', image: png },
    ])
  })

  it('keeps inline text that sits between block children', () => {
    expect(htmlToBlocks('<div>Before <table><tr><td>x</td></tr></table> after</div>')).toEqual([
      { type: 'paragraph', text: 'Before' },
      { type: 'table', rows: [['x']], header_rows: 0 },
      { type: 'paragraph', text: 'after' },
    ])
  })

  it('counts an all-<th> first row as a header without a thead', () => {
    const [t] = htmlToBlocks('<table><tr><th>a</th></tr><tr><td>1</td></tr></table>')
    expect(t).toMatchObject({ type: 'table', header_rows: 1 })
  })
})
