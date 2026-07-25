import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { ColMeta, Session } from '../store'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import DataTable from './DataTable'

afterEach(() => clearSession())

/** A sheet of `n` rows over `cols` numeric columns. */
function bigSession(n: number, cols = 6): Session {
  const columns: ColMeta[] = Array.from({ length: cols }, (_, j) => ({
    name: `c${j}`,
    dtype: 'float64',
    kind: 'numeric',
  }))
  const preview = Array.from({ length: n }, (_, i) =>
    Object.fromEntries(columns.map((c, j) => [c.name, i * 100 + j])),
  )
  return makeSession({ columns, preview, rows: n })
}

const bodyRows = () => document.querySelectorAll('tbody tr[class*="group"]').length

describe('DataTable row virtualisation', () => {
  it('renders every row for a small sheet', () => {
    installSession(bigSession(40))
    render(<DataTable />)
    expect(bodyRows()).toBe(40)
  })

  it('renders only a window for a large sheet', () => {
    // 1000 x 125 was ~126,000 cells and ~6s to become usable; only the visible
    // slice (plus overscan) should reach the DOM.
    installSession(bigSession(1000))
    render(<DataTable />)
    const rendered = bodyRows()
    expect(rendered).toBeGreaterThan(0)
    expect(rendered).toBeLessThan(120)
  })

  it('keeps the scroll height correct with spacer rows', () => {
    installSession(bigSession(1000))
    render(<DataTable />)
    const spacers = [...document.querySelectorAll('tbody tr[aria-hidden="true"] td')]
    expect(spacers.length).toBeGreaterThan(0)
    // Spacers stand in for the rows that were not rendered: 33px each.
    const padded = spacers.reduce((sum, td) => {
      const h = parseFloat((td as HTMLElement).style.height || '0')
      return sum + h
    }, 0)
    expect(padded / 33 + bodyRows()).toBeGreaterThanOrEqual(1000)
  })

  it('numbers the visible rows by their absolute position', () => {
    installSession(bigSession(1000))
    render(<DataTable />)
    // The first rendered row is row 1 at the top of an unscrolled grid.
    expect(screen.getByTitle(/Original row #1\b/)).toBeInTheDocument()
  })

  it('shows the empty state when every row is filtered out', () => {
    installSession(makeSession({ columns: [], preview: [], rows: 0 }))
    render(<DataTable />)
    expect(bodyRows()).toBe(0)
  })
})
