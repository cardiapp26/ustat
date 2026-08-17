import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ChartTypeIcon from './ChartTypeIcon'
import { ICON_TYPES } from './chartGlyphs'
import { CHART_TYPES } from '../../lib/chartTypes'

describe('ChartTypeIcon', () => {
  it('has a glyph for every chart type the picker offers', () => {
    // The failure this guards against is silent: a new chart type renders with
    // a blank space where every other row has a thumbnail, which reads as a
    // rendering bug rather than a missing asset.
    expect([...CHART_TYPES].filter((t) => !ICON_TYPES.includes(t))).toEqual([])
  })

  it('draws no glyph the picker cannot reach', () => {
    expect(ICON_TYPES.filter((t) => !(CHART_TYPES as readonly string[]).includes(t))).toEqual([])
  })

  it('renders an svg that inherits the surrounding colour', () => {
    // `currentColor` is what lets one class on the wrapper tint the glyph with
    // the label when a row is selected.
    const { container } = render(<ChartTypeIcon type="boxplot" className="w-6 h-4" />)
    const svg = container.querySelector('svg')
    expect(svg).toBeInTheDocument()
    expect(svg).toHaveAttribute('stroke', 'currentColor')
    expect(svg).toHaveClass('w-6', 'h-4')
  })

  it('is hidden from screen readers, since the label already names the chart', () => {
    const { container } = render(<ChartTypeIcon type="violin" />)
    expect(container.querySelector('svg')).toHaveAttribute('aria-hidden', 'true')
  })

  it('renders nothing for an unknown type instead of an empty box', () => {
    const { container } = render(<ChartTypeIcon type="not_a_chart" />)
    expect(container).toBeEmptyDOMElement()
  })
})
