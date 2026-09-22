import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createRef } from 'react'
import StaleGuard from './StaleGuard'
import CopyTextButton from './CopyTextButton'
import ResultExporter from './ResultExporter'
import PlotExporter from './PlotExporter'
import { StyledTableExporter } from './StyledTableExporter'
import { ForestBuilderButton } from './models/widgets'
import Plot from '../PlotComponent'
import type { PlotRef } from '../lib/plotTypes'

const TABLE = () => ({ title: 'T', columns: ['a'], rows: [['1']] })

/** Every way a result can leave the app, rendered inside one guard. */
function Exports({ onForest = () => {} }: { onForest?: () => void }) {
  return (
    <>
      <CopyTextButton text="paragraph" />
      <ResultExporter title="Cox" headers={['h']} rows={[[1]]} />
      <StyledTableExporter data={TABLE} />
      <PlotExporter plotRef={createRef() as PlotRef} />
      <ForestBuilderButton onClick={onForest} />
      <Plot data={[]} layout={{}} config={{ modeBarButtonsToRemove: ['lasso2d'] }} />
    </>
  )
}

const EXPORT_BUTTONS = ['Copy', 'CSV', 'XLSX', 'Copy table', 'Word', 'HTML', '⧉', '↓', '→ Forest Builder']

describe('StaleGuard', () => {
  it('leaves every export open for a current result', () => {
    render(<StaleGuard stale={false}><Exports /></StaleGuard>)
    for (const name of EXPORT_BUTTONS) expect(screen.getByRole('button', { name })).toBeEnabled()
    const config = JSON.parse(screen.getByTestId('plotly-mock').dataset.config ?? '{}')
    expect(config.modeBarButtonsToRemove).toEqual(['lasso2d'])
  })

  it('closes every export path of an out-of-date result', () => {
    render(<StaleGuard stale reason="the data changed"><Exports /></StaleGuard>)
    for (const name of EXPORT_BUTTONS) {
      const btn = screen.getByRole('button', { name })
      expect(btn, name).toBeDisabled()
      expect(btn, name).toHaveAttribute('title', 'Recompute first: this result predates the data changed')
    }
    // The Plotly modebar camera is an export too.
    const config = JSON.parse(screen.getByTestId('plotly-mock').dataset.config ?? '{}')
    expect(config.modeBarButtonsToRemove).toEqual(['lasso2d', 'toImage'])
  })

  it('does not hand a stale fit to the Forest Builder even if clicked', async () => {
    const onForest = vi.fn()
    render(<StaleGuard stale><Exports onForest={onForest} /></StaleGuard>)
    await userEvent.click(screen.getByRole('button', { name: '→ Forest Builder' }))
    expect(onForest).not.toHaveBeenCalled()
  })

  it('makes an inner result stale when the outer one is', () => {
    render(
      <StaleGuard stale reason="the case filter changed">
        <StaleGuard stale={false}><CopyTextButton text="x" /></StaleGuard>
      </StaleGuard>,
    )
    expect(screen.getByRole('button', { name: 'Copy' }))
      .toHaveAttribute('title', 'Recompute first: this result predates the case filter changed')
  })

  it('leaves controls outside any guard as they were', () => {
    render(<CopyTextButton text="x" />)
    expect(screen.getByRole('button', { name: 'Copy' })).toBeEnabled()
  })
})
