import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useStore } from '../../store'
import GrayTestCard from './GrayTestCard'

const runGrayTest = vi.fn()
vi.mock('../../api', async (orig) => ({
  ...(await orig<typeof import('../../api')>()),
  runGrayTest: (...args: unknown[]) => runGrayTest(...args),
}))
vi.mock('../../lib/engine/r/client', () => ({ prefetchRRuntime: vi.fn() }))

const RESULT = {
  test: "Gray's test", statistic: 5.0366, df: 2, p: 0.0806, event_of_interest: 1,
  hypothesis: 'Equal cumulative incidence of event 1 across arm groups',
  groups: [
    { group: 'control', n: 44, event_of_interest: 23, competing_events: 12, censored: 9 },
    { group: 'drugA', n: 59, event_of_interest: 14, competing_events: 26, censored: 19 },
  ],
  by_cause: [{ event: 1, statistic: 5.0366, p: 0.0806, df: 2 }, { event: 2, statistic: 6.8454, p: 0.0326, df: 2 }],
  n: 103, n_excluded: 0, engine: 'R cmprsk::cuminc (rho = 0)', r_code: 'library(cmprsk)',
}

const props = { sessionId: 's1', durationCol: 'fu', eventCol: 'status', groupCol: 'arm', eventOfInterest: 1 }

beforeEach(() => { runGrayTest.mockReset(); runGrayTest.mockResolvedValue({ data: RESULT }) })
afterEach(() => useStore.getState().setEngine('python'))

describe('GrayTestCard', () => {
  it('in a Python session offers to move to R instead of running', () => {
    useStore.getState().setEngine('python')
    render(<GrayTestCard {...props} />)
    expect(screen.getByRole('button', { name: /switch session to r and run/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /run gray's test/i })).toBeNull()
    expect(runGrayTest).not.toHaveBeenCalled()
  })

  it('switching moves the session to R and runs the test there', async () => {
    useStore.getState().setEngine('python')
    const user = userEvent.setup()
    render(<GrayTestCard {...props} />)
    await user.click(screen.getByRole('button', { name: /switch session to r and run/i }))
    expect(useStore.getState().engine).toBe('r')
    await waitFor(() => expect(screen.getByText('5.037')).toBeInTheDocument())
    expect(runGrayTest).toHaveBeenCalledWith(expect.objectContaining({
      duration_col: 'fu', event_col: 'status', group_col: 'arm', event_of_interest: 1,
    }))
    expect(screen.getByText(/event 2: χ²\(2\) = 6.85/)).toBeInTheDocument()
  })

  it('in an R session runs directly', async () => {
    useStore.getState().setEngine('r')
    const user = userEvent.setup()
    render(<GrayTestCard {...props} />)
    await user.click(screen.getByRole('button', { name: /run gray's test/i }))
    await waitFor(() => expect(screen.getByText('control')).toBeInTheDocument())
  })

  it('shows why when the engine cannot run it', async () => {
    useStore.getState().setEngine('r')
    runGrayTest.mockRejectedValue(new Error("Gray's test runs in the R engine (cmprsk::cuminc) and could not run here."))
    const user = userEvent.setup()
    render(<GrayTestCard {...props} />)
    await user.click(screen.getByRole('button', { name: /run gray's test/i }))
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('R engine'))
  })

  it('asks for a group column first', () => {
    render(<GrayTestCard {...props} groupCol="" />)
    expect(screen.getByText(/choose a group column/i)).toBeInTheDocument()
  })
})
