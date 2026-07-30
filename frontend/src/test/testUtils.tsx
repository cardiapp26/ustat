import type { ColMeta, Session } from '../store'
import { useStore } from '../store'

/** Build a minimal but valid Session for panel tests. */
export function makeSession(overrides: Partial<Session> = {}): Session {
  const columns: ColMeta[] = overrides.columns ?? [
    { name: 'AGE', dtype: 'float64', kind: 'numeric' },
    { name: 'LDL', dtype: 'float64', kind: 'numeric' },
    { name: 'DM', dtype: 'int64', kind: 'numeric' },
    { name: 'GROUP', dtype: 'object', kind: 'categorical' },
  ]
  return {
    session_id: 'test-session',
    filename: 'test.csv',
    rows: 3,
    columns,
    preview: [
      { AGE: 55, LDL: 120, DM: 0, GROUP: 'A' },
      { AGE: 62, LDL: 140, DM: 1, GROUP: 'B' },
      { AGE: 48, LDL: 110, DM: 0, GROUP: 'A' },
    ],
    ...overrides,
  }
}

/** Reset the Zustand store to a clean slate with the given session installed.
 *  The store is a module-level singleton, so every panel test must call this
 *  before rendering to avoid leaking state between tests. */
export function installSession(session: Session = makeSession()): void {
  useStore.setState({
    session,
    originalSession: session,
    activeTab: 'data',
    table1Result: null,
    caseFilter: null,
    panelCache: {},
    undoDepth: 0,
    redoDepth: 0,
    columnMutationUndo: [],
    columnMutationRedo: [],
    dataVersion: 0,
    columnDecimals: {},
    sessionHistory: [],
  })
}

export function clearSession(): void {
  useStore.setState({ session: null, originalSession: null })
}
