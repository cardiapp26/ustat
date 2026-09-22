/**
 * Which request produced a result, recorded so a saved analysis can be re-run.
 *
 * A saved analysis used to keep only the panel's own field names, so neither
 * the replay script nor a re-run could say which endpoint computed it. The
 * axios interceptor records every recent API response here, and a panel's
 * `setResult(r)` looks up the entry whose `data` IS `r` (same object). That
 * identity is the whole correlation: it proves this request returned exactly
 * the object on screen. A panel that reshapes the response before showing it
 * gets no match, and its analysis is simply not replayable, rather than
 * replayed with a request that computed something else.
 */

export interface RecordedRequest {
  method: string;
  /** Path and query, the session as the literal "{sid}". */
  url: string;
  /** JSON body with a top-level session_id replaced by "{sid}", or null. */
  body: Record<string, unknown> | null;
}

/** What the data looked like when a request was SENT. A result stamped with
 *  the state at response time would call a fit current that was computed on
 *  the data before an edit made while it was in flight. */
export interface SendContext {
  dataVersion: number;
  caseFilter: unknown;
  sessionId: string | null;
}

interface Entry {
  method: string;
  url: string;
  body: Record<string, unknown> | null;
  data: unknown;
  sent: SendContext | null;
}

// Supplied by the store (which this module must not import: the api module
// imports this one, and the store imports the api).
let sendContext: (() => SendContext) | null = null;
const SENT = Symbol("ustat.sent");

export function setSendContextProvider(provider: (() => SendContext) | null): void {
  sendContext = provider;
}

/** Called by the axios request interceptor: note the state at send time. */
export function annotateRequest<T extends object>(config: T): T {
  if (sendContext) (config as Record<symbol, unknown>)[SENT] = sendContext();
  return config;
}

const RECENT = 32;
let entries: Entry[] = [];

interface RequestConfigLike {
  method?: string;
  url?: string;
  params?: Record<string, unknown>;
  data?: unknown;
}

function jsonBody(data: unknown): Record<string, unknown> | null | undefined {
  if (data == null || data === "") return null;
  if (typeof data === "string") {
    try {
      const parsed: unknown = JSON.parse(data);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? parsed as Record<string, unknown>
        : undefined;
    } catch {
      return undefined;
    }
  }
  if (typeof data === "object" && Object.getPrototypeOf(data) === Object.prototype) {
    return data as Record<string, unknown>;
  }
  return undefined; // FormData, Blob: an upload, not a replayable call
}

function withQuery(url: string, params: Record<string, unknown> | undefined): string {
  if (!params) return url;
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) q.append(k, String(v));
  }
  const s = q.toString();
  return s ? `${url}${url.includes("?") ? "&" : "?"}${s}` : url;
}

/** Called by the axios response interceptor for every /api/ answer. */
export function recordResponse(config: RequestConfigLike | undefined, data: unknown): void {
  const url = config?.url;
  if (!config || typeof url !== "string" || !url.startsWith("/api/")) return;
  const body = jsonBody(config.data);
  if (body === undefined) return;
  const sent = ((config as Record<symbol, unknown>)[SENT] as SendContext | undefined) ?? null;
  entries = [
    { method: (config.method ?? "get").toUpperCase(), url: withQuery(url, config.params), body, data, sent },
    ...entries,
  ].slice(0, RECENT);
}

/** The state the request that returned exactly `data` was sent under. */
export function sentContextFor(data: unknown): SendContext | null {
  if (data == null || typeof data !== "object") return null;
  return entries.find((e) => e.data === data)?.sent ?? null;
}

/** The request whose response is exactly `data`, with `sessionId` written as
 *  "{sid}"; null when no recent request returned this object. */
export function requestFor(data: unknown, sessionId: string | null | undefined): RecordedRequest | null {
  if (data == null || typeof data !== "object") return null;
  const hit = entries.find((e) => e.data === data);
  if (!hit) return null;
  const sid = sessionId ?? "";
  const url = sid ? hit.url.split(sid).join("{sid}") : hit.url;
  let body = hit.body;
  if (body && sid && body.session_id === sid) body = { ...body, session_id: "{sid}" };
  return { method: hit.method, url, body };
}

/** The recorded request as sent to the current session. */
export function forSession(request: RecordedRequest, sessionId: string): { method: string; url: string; body: Record<string, unknown> | null } {
  const url = request.url.split("{sid}").join(sessionId);
  const body = request.body && request.body.session_id === "{sid}"
    ? { ...request.body, session_id: sessionId }
    : request.body;
  return { method: request.method, url, body };
}

/** Test seam: module state would otherwise leak across tests. */
export function resetRequestLog(): void {
  entries = [];
}
