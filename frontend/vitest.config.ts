import { mergeConfig } from 'vite'
import { defineConfig as defineVitestConfig } from 'vitest/config'
import viteConfig from './vite.config'

export default mergeConfig(
  viteConfig,
  defineVitestConfig({
    test: {
      environment: 'jsdom',
      // Raised from vitest's 5 s default. Each jsdom environment costs about a
      // second to stand up and 70 test files run in parallel across 8 cores, so
      // a test that takes 1.7 s alone can sit past 5 s waiting for the machine.
      // The failures that produced were not reproducible -- a different three
      // or four files each run, all green in isolation -- which is the worst
      // kind of red: it trains people to re-run rather than to read. This is
      // the machine's budget, not the product's latency, and no assertion
      // changes by raising it.
      // 40/40: must stay ABOVE the 30 s asyncUtilTimeout in src/test/setup.ts,
      // or a stuck findBy* would hit this ceiling first and report a bare
      // test timeout instead of which query was waiting.
      testTimeout: 40000,
      hookTimeout: 40000,
      globals: true,
      setupFiles: ['./src/test/setup.ts'],
      css: false,
      exclude: ['node_modules', 'dist', 'src-tauri'],
    },
  }),
)
