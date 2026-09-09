/**
 * Which engine produced the number on screen, said by whatever produced it.
 *
 * THE THING THIS REPLACES. The header bar read the session's engine choice and
 * printed "R-based statistics". Two analyses are actually routed to a local
 * engine (`stats.ttest`, `stats.power`); every regression, Cox model, ROC and
 * random forest is a plain POST to the server's Python. So a reader who chose
 * R was told R had computed things R never touched, and would fail to
 * reproduce them in R for a reason nothing on screen explained.
 *
 * Provenance is therefore recorded at the boundary where the answer arrives --
 * the axios interceptor for server answers, `localFirst` for local ones -- and
 * carried by each result rather than by the session. Versions come from the
 * server's own `X-uStat-*` headers, because "computed with Python" is not
 * reproducible and "scipy 1.14.1" is.
 */
import type { EngineKind, Runtime } from "./types";

export interface Provenance {
  /** Where it ran: this browser, or the server. */
  runtime: Runtime;
  /** Which engine: R (webR) or Python (Pyodide here, CPython there). */
  engine: EngineKind;
  /** Language runtime version, e.g. "3.12" or "4.5.2". */
  languageVersion?: string;
  /** The versions that decide the number, e.g. `{ scipy: "1.14.1" }`. */
  packages?: Record<string, string>;
  /** Our own engine build, so a mismatch between the two copies is visible. */
  engineVersion?: string;
  engineFingerprint?: string;
  /** Set when a local run was possible in principle but the server answered. */
  fellBackBecause?: string;
  at: number;
}

/** Header names, matching backend/services/runtime_identity.py. */
const H = {
  runtime: "x-ustat-runtime",
  engine: "x-ustat-engine",
  engineVersion: "x-ustat-engine-version",
  fingerprint: "x-ustat-engine-fingerprint",
  python: "x-ustat-python",
  packages: "x-ustat-packages",
} as const;

/** `numpy=2.0.2,scipy=1.14.1` → `{ numpy: "2.0.2", scipy: "1.14.1" }`. */
export function parsePackages(header: string | undefined): Record<string, string> {
  if (!header) return {};
  const out: Record<string, string> = {};
  for (const pair of header.split(",")) {
    const eq = pair.indexOf("=");
    if (eq <= 0) continue;
    const name = pair.slice(0, eq).trim();
    const value = pair.slice(eq + 1).trim();
    if (name && value) out[name] = value;
  }
  return out;
}

type HeaderBag = Record<string, unknown> | undefined;

/**
 * Case-insensitive header lookup.
 *
 * HTTP header names are case-insensitive and the casing here is `X-uStat-`,
 * which no capitalisation rule reproduces. axios lowercases its keys, a raw
 * fetch `Headers` and hand-written test doubles do not, so the bag is folded
 * once rather than guessed at spelling by spelling.
 */
function foldKeys(headers: HeaderBag): Record<string, string> {
  const out: Record<string, string> = {};
  if (!headers) return out;
  for (const [key, value] of Object.entries(headers)) {
    if (typeof value === "string" && value) out[key.toLowerCase()] = value;
  }
  return out;
}

/**
 * Read a server answer's provenance off its own headers.
 *
 * Returns a record even when the headers are missing: an older server, or a
 * proxy that stripped them, must not make provenance silently disappear. The
 * absent versions are then absent rather than guessed, which is the honest
 * shape -- a reader can tell "the server did not say" from "scipy 1.14.1".
 */
export function fromResponseHeaders(headers: HeaderBag): Provenance {
  const h = foldKeys(headers);
  return {
    runtime: (h[H.runtime] as Runtime) ?? "server",
    engine: (h[H.engine] as EngineKind) ?? "python",
    languageVersion: h[H.python],
    packages: parsePackages(h[H.packages]),
    engineVersion: h[H.engineVersion],
    engineFingerprint: h[H.fingerprint],
    at: Date.now(),
  };
}

/** "Python 3.12 · scipy 1.14.1 · in your browser" */
export function describeProvenance(p: Provenance | null | undefined): string {
  if (!p) return "";
  const language = p.engine === "r" ? "R" : "Python";
  const bits = [p.languageVersion ? `${language} ${p.languageVersion}` : language];
  // scipy for Python and the R version itself are what a methods section
  // quotes; the rest belongs in the full record, not in a one-line label.
  const headline = p.engine === "r" ? undefined : p.packages?.scipy;
  if (headline) bits.push(`scipy ${headline}`);
  bits.push(p.runtime === "local" ? "in your browser" : "on the server");
  return bits.join(" · ");
}

/** Every version the run reported, for an export footer or a methods table. */
export function provenanceLines(p: Provenance | null | undefined): string[] {
  if (!p) return [];
  const lines = [
    `Engine: ${p.engine === "r" ? "R" : "Python"}`,
    `Computed: ${p.runtime === "local" ? "in the browser" : "on the server"}`,
  ];
  if (p.languageVersion) lines.push(`Runtime version: ${p.languageVersion}`);
  const packages = Object.entries(p.packages ?? {});
  if (packages.length) {
    lines.push(`Packages: ${packages.map(([n, v]) => `${n} ${v}`).join(", ")}`);
  }
  if (p.engineVersion) {
    lines.push(
      `uSTAT engine: ${p.engineVersion}` +
      (p.engineFingerprint ? ` (${p.engineFingerprint})` : ""),
    );
  }
  if (p.fellBackBecause) lines.push(`Server answered because: ${p.fellBackBecause}`);
  return lines;
}

/**
 * The provenance of the most recent answer, per request path.
 *
 * A ledger and not a single slot: two panels can be waiting at once, and
 * attributing one panel's result to the other's engine is the same class of
 * error this module exists to remove. Keyed by path because that is what both
 * recorders know; `latest()` is the fallback for a caller that does not.
 */
const ledger = new Map<string, Provenance>();
let mostRecent: Provenance | null = null;
const MAX_ENTRIES = 200;

export function record(key: string, provenance: Provenance): void {
  if (ledger.size >= MAX_ENTRIES) {
    const oldest = ledger.keys().next();
    if (!oldest.done) ledger.delete(oldest.value);
  }
  ledger.set(key, provenance);
  mostRecent = provenance;
}

export function forKey(key: string): Provenance | undefined {
  return ledger.get(key);
}

export function latest(): Provenance | null {
  return mostRecent;
}

/** Test seam: the ledger is module state and would otherwise leak across tests. */
export function resetProvenance(): void {
  ledger.clear();
  mostRecent = null;
}
