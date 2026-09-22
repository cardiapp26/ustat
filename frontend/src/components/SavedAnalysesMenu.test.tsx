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
    expect(screen.getByText('Regression model')).toBeInTheDocument()
    expect(screen.queryByText('ROC')).not.toBeInTheDocument()

    fireEvent.click(screen.getByText('Regression model'))
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
    fireEvent.click(screen.getByText('Regression model'))
    useStore.getState().setPanelCache('models', { result: { or: 9.9 } })
    const kept = useStore.getState().savedAnalyses[0]
    expect((kept.snapshot as { result: { or: number } }).result.or).toBe(2.1)
  })

  it('restores a kept analysis into its panel and navigates to its tab', () => {
    useStore.setState({ panelCache: { models: { result: { or: 2.1 } } } })
    render(<SavedAnalysesMenu />)
    openMenu()
    fireEvent.click(screen.getByText('Regression model'))
    // Simulate the panel cache moving on and the user elsewhere. The menu
    // stays open after a keep, so no re-toggle here.
    useStore.setState({ panelCache: {}, activeTab: 'data' })

    fireEvent.click(screen.getByTitle('Restore into its panel'))
    const state = useStore.getState()
    expect(state.activeTab).toBe('models')
    expect((state.panelCache.models as { result: { or: number } }).result.or).toBe(2.1)
  })

  it('keeps an analysis under its own tab even when kept from another', () => {
    // The Kaplan-Meier fit lives in Models > Survival; keeping it from the
    // Tests tab used to restore it to Tests.
    useStore.setState({ panelCache: { survival_km: { result: { n: 1 } } }, activeTab: 'tests' })
    render(<SavedAnalysesMenu />)
    openMenu()
    fireEvent.click(screen.getByText('Survival: km'))
    expect(useStore.getState().savedAnalyses[0].tab).toBe('models')
    useStore.setState({ activeTab: 'data' })
    fireEvent.click(screen.getByTitle('Restore into its panel'))
    expect(useStore.getState().activeTab).toBe('models')
    expect((useStore.getState().panelCache.combo_models as { sub: string }).sub).toBe('survival')
  })

  it('offers re-run only for an analysis kept with its request', () => {
    useStore.setState({
      savedAnalyses: [
        { id: 'a', name: 'With request', panel: 'models', tab: 'models', createdAt: 1,
          snapshot: { result: {}, stamp: { request: { method: 'POST', url: '/api/models/linear', body: {} } } } },
        { id: 'b', name: 'Without', panel: 'models', tab: 'models', createdAt: 1, snapshot: { result: {}, stamp: {} } },
      ],
    })
    render(<SavedAnalysesMenu />)
    openMenu()
    expect(screen.getByRole('button', { name: 'Re-run With request' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Re-run Without' })).toBeDisabled()
  })

  it('renames and deletes a kept analysis', () => {
    useStore.setState({ panelCache: { models: { result: { or: 1 } } } })
    render(<SavedAnalysesMenu />)
    openMenu()
    fireEvent.click(screen.getByText('Regression model'))

    fireEvent.click(screen.getByTitle('Rename'))
    const input = screen.getByDisplayValue('Analysis 1')
    fireEvent.change(input, { target: { value: 'Adjusted model' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(useStore.getState().savedAnalyses[0].name).toBe('Adjusted model')

    fireEvent.click(screen.getByTitle('Delete'))
    expect(useStore.getState().savedAnalyses).toHaveLength(0)
  })
})
