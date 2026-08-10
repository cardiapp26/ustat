import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import React from 'react'
import { afterAll, afterEach, beforeAll, vi } from 'vitest'
import { server } from './server'

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => {
  server.resetHandlers()
  cleanup()
})
afterAll(() => server.close())

// Plotly renders to canvas/WebGL, which jsdom doesn't implement — panel
// tests assert on data/table content, not the chart pixels, so a lightweight
// stub is enough to let react-plotly.js mount without crashing.
vi.mock('react-plotly.js', () => ({
  default: (props: Record<string, unknown>) => {
    // The layout carries axis ranges, subplot domains and the bracket
    // annotations — assertions about what the reader will SEE need it as much
    // as they need the traces.
    return React.createElement('div', {
      'data-testid': 'plotly-mock',
      'data-plotly': JSON.stringify(props.data ?? []),
      'data-layout': JSON.stringify(props.layout ?? {}),
    })
  },
}))
