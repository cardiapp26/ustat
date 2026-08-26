/**
 * The boot recovery in index.html, enforced by reading the file.
 *
 * The bug it clears cannot be reproduced in jsdom: it needs a service worker
 * serving a precached index.html whose hashed bundle the browser has since
 * evicted, which is a browser-storage state, not a DOM one. So this checks the
 * two things that make the recovery work at all and that a well-meaning edit
 * would quietly remove:
 *
 *   1. the SEO shell still carries `id="boot-shell"` — the script's only
 *      signal that React never mounted, since createRoot() clears that subtree
 *      on its first render;
 *   2. the script is a CLASSIC script, guarded against reload loops and
 *      against running while offline.
 *
 * This is a textual check over the file, not a parse, and it proves only that
 * the pieces are present in the right shape — not that they run correctly.
 */
/// <reference types="node" />
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

// The jsdom environment resolves import.meta.url to an http URL served by
// Vite, so the path comes from cwd (the `frontend` directory) instead.
const html = readFileSync(join(process.cwd(), 'index.html'), 'utf8')

describe('index.html boot recovery', () => {
  it('marks the SEO shell with the id the recovery script looks for', () => {
    expect(html).toMatch(/<div id="boot-shell"/)
    // Both halves of the signal: the marker in the DOM and the lookup for it.
    expect(html).toContain("document.getElementById('boot-shell')")
  })

  it('runs the recovery as a classic script', () => {
    // A module script is deferred and shares the failure it is meant to catch:
    // when module loading is what broke, a module recovery never runs.
    const scripts = [...html.matchAll(/<script([^>]*)>/g)].map((m) => m[1])
    const recovery = scripts.find((attrs) => !attrs.includes('src='))
    expect(recovery).toBeDefined()
    expect(recovery).not.toContain('type="module"')
  })

  it('bounds itself to one attempt per tab and stays out of the way offline', () => {
    // Without the flag a genuinely broken deploy becomes a reload loop, which
    // is worse than the blank page it replaces.
    expect(html).toContain("var HEALED = 'ustat:boot-healed'")
    expect(html).toContain('sessionStorage.getItem(HEALED)')
    expect(html).toContain('sessionStorage.setItem(HEALED,')
    // Offline the precache is the only copy of the app; clearing it strands
    // the visitor rather than rescuing them.
    expect(html).toContain('navigator.onLine === false')
  })

  it('clears the worker and Cache Storage but never localStorage', () => {
    expect(html).toContain('navigator.serviceWorker.getRegistrations()')
    expect(html).toContain('caches.keys()')
    expect(html).toContain('location.reload()')
    // The session id and saved preferences live in localStorage and had
    // nothing to do with this failure.
    const recovery = html.slice(html.indexOf("var HEALED = 'ustat:boot-healed'"))
    expect(recovery).not.toContain('localStorage')
  })
})
