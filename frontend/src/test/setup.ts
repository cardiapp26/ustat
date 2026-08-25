import '@testing-library/jest-dom/vitest'
import { cleanup, configure } from '@testing-library/react'
import React from 'react'
import { afterAll, afterEach, beforeAll, vi } from 'vitest'
import { server } from './server'

// Testing Library keeps its own budget for waitFor and findBy*, and vitest's
// testTimeout does not cover it: the default is one second, so on a loaded
// machine a test can fail while the test itself still has nineteen seconds
// left. That is what CI hit -- ROCPanel's findByRole('status') gave up at
// 1055 ms on a two-core runner, green on every local run and green in
// isolation there too. Five seconds is the same kind of number as the
// testTimeout above: the machine's budget, not the product's, and still short
// enough that a query which will never match reports where it was waiting
// instead of running out the whole test.
configure({ asyncUtilTimeout: 5000 })

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

// jsdom implements no layout, so Element.scrollIntoView does not exist. The
// grid calls it after every keyboard move to keep the focused cell in view;
// without a stub the call lands in a requestAnimationFrame callback and
// surfaces as an unhandled TypeError that no test can catch.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function scrollIntoView() {}
}
