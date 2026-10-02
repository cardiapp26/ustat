/**
 * Split File on the wire: every /api/ request states the level on view in an
 * `X-Ustat-Split` header (backend services/split_scope), so each analysis
 * runs on that level's rows without any endpoint knowing about splits.
 *
 * The store registers what the active split is, the same way it registers
 * the send context for lib/requestLog: api.ts cannot import the store
 * without closing an import cycle.
 */
import type { InternalAxiosRequestConfig } from "axios";

export const SPLIT_HEADER = "X-Ustat-Split";

export interface SplitHeaderScope {
  column: string;
  level: string | null;
}

let provider: () => SplitHeaderScope | null = () => null;

export function setSplitProvider(fn: () => SplitHeaderScope | null): void {
  provider = fn;
}

/** Header values are Latin-1; column names and levels are not. */
export function splitHeaderValue(column: string, level: string): string {
  return encodeURIComponent(JSON.stringify({ column, level }));
}

/** Add the active split level, unless the request already names one (the
 *  background precompute of other levels sets its own). */
export function withSplitHeader(config: InternalAxiosRequestConfig): InternalAxiosRequestConfig {
  if (typeof config.url !== "string" || !config.url.startsWith("/api/")) return config;
  if (config.headers?.has?.(SPLIT_HEADER)) return config;
  const scope = provider();
  if (!scope || scope.level == null) return config;
  config.headers.set(SPLIT_HEADER, splitHeaderValue(scope.column, scope.level));
  return config;
}
