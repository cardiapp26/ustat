import { beforeEach, describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import SavedAnalysesMenu from './SavedAnalysesMenu'
import { useStore } from '../store'
import { installSession } from '../test/testUtils'

function openMenu() {
  fireEvent.click(screen.getByTitle('Saved analyses'))
}

describe('SavedAnalysesMenu', () => {
  beforeEach(() => {
    installSession()
    useStore.setState({ savedAnalyses: [], panelCache: {}, activeTab: 'models' })
  })

  it('offers to keep panels that cache a result, and keeps one on click', () => {
    useStore.setState({
      panelCache: {
        models: { result: { or: 2.1 }, stamp: null },
        roc: { selections: { y: 'DM' } }, // no result: not keepable
      },
    })
    render(<SavedAnalysesMenu />)
    openMenu()
    expect(screen.getByText('models')).toBeInTheDocument()
    expect(screen.queryByText('roc')).not.toBeInTheDocument()

    fireEvent.click(screen.getByText('models'))
    const kept = useStore.getState().savedAnalyses
    expect(kept).toHaveLength(1)
    expect(kept[0].name).toBe('Analysis 1')
    expect(kept[0].panel).toBe('models')
    expect(kept[0].tab).toBe('models')
  })

  it('keeps a deep copy: later cache changes do not touch the kept snapshot', () => {
    useStore.setState({ panelCache: { models: { result: { or: 2.1 } } } })
    render(<SavedAnalysesMenu />)
    openMenu()
    fireEvent.click(screen.getByText('models'))
    useStore.getState().setPanelCache('models', { result: { or: 9.9 } })
    const kept = useStore.getState().savedAnalyses[0]
    expect((kept.snapshot as { result: { or: number } }).result.or).toBe(2.1)
  })

  it('restores a kept analysis into its panel and navigates to its tab', () => {
    useStore.setState({ panelCache: { models: { result: { or: 2.1 } } } })
    render(<SavedAnalysesMenu />)
    openMenu()
    fireEvent.click(screen.getByText('models'))
    // Simulate the panel cache moving on and the user elsewhere. The menu
    // stays open after a keep, so no re-toggle here.
    useStore.setState({ panelCache: {}, activeTab: 'data' })

    fireEvent.click(screen.getByTitle('Restore into its panel'))
    const state = useStore.getState()
    expect(state.activeTab).toBe('models')
    expect((state.panelCache.models as { result: { or: number } }).result.or).toBe(2.1)
  })

  it('renames and deletes a kept analysis', () => {
    useStore.setState({ panelCache: { models: { result: { or: 1 } } } })
    render(<SavedAnalysesMenu />)
    openMenu()
    fireEvent.click(screen.getByText('models'))

    fireEvent.click(screen.getByTitle('Rename'))
    const input = screen.getByDisplayValue('Analysis 1')
    fireEvent.change(input, { target: { value: 'Adjusted model' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(useStore.getState().savedAnalyses[0].name).toBe('Adjusted model')

    fireEvent.click(screen.getByTitle('Delete'))
    expect(useStore.getState().savedAnalyses).toHaveLength(0)
  })
})
