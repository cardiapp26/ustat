/**
 * UpdatePrompt — auto-detects a newer build of uSTAT and offers a one-click reload/update.
 *
 * How it works (browser and installed PWA):
 *   - Registers a service worker that pre-caches every asset in the dist/ folder.
 *   - When a new build is deployed, SW fires `needRefresh`.
 *   - Falls back to polling a static `/version.json` if SW is unavailable.
 *
 * The desktop app renders nothing here. Its frontend ships inside the app, so
 * a new version arrives only with a new app, and the app's own updater
 * (src-tauri/src/updater.rs) checks for that and asks with a native dialog.
 * It cannot be done from this page: the desktop window shows the page served
 * by the local backend, a remote origin to Tauri, so plugin calls from here
 * are refused.
 */

import { useEffect, useState } from "react";
import { useRegisterSW } from "virtual:pwa-register/react";
import { RefreshCw, X, AlertCircle, CheckCircle2 } from "lucide-react";

// Build-time version stamp. Injected by vite.config.ts via `define`.
declare const __APP_VERSION__: string;
declare const __BUILD_TIME__: string;

const POLL_INTERVAL_MS = 60_000; // 1 minute
const FALLBACK_POLL_MS = 5 * 60_000; // 5 minutes

export default function UpdatePrompt() {
  const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

  // ── Primary path: service-worker update detection (Browser only) ────
  const {
    needRefresh: [needRefresh, setNeedRefresh],
    offlineReady: [offlineReady, setOfflineReady],
    updateServiceWorker,
  } = useRegisterSW({
    onRegisteredSW(_swUrl, registration) {
      if (isTauri || !registration) return;
      setInterval(() => {
        registration.update().catch(() => null);
      }, POLL_INTERVAL_MS);
    },
    onRegisterError(err) {
      if (!isTauri) console.warn("[UpdatePrompt] SW register failed:", err);
    },
  });

  // ── Fallback path: poll /version.json when SW is unavailable (Browser only) ──
  const [fallbackStale, setFallbackStale] = useState(false);
  useEffect(() => {
    if (isTauri) return;
    if ("serviceWorker" in navigator) return;
    const check = async () => {
      try {
        const res = await fetch("/version.json", { cache: "no-store" });
        if (!res.ok) return;
        const { version, build } = (await res.json()) as { version?: string; build?: string };
        const remoteKey = build ?? version ?? "";
        const localKey = (typeof __BUILD_TIME__ === "string" ? __BUILD_TIME__ : __APP_VERSION__) ?? "";
        if (remoteKey && localKey && remoteKey !== localKey) {
          setFallbackStale(true);
        }
      } catch {
        /* network blip */
      }
    };
    void check();
    const t = setInterval(check, FALLBACK_POLL_MS);
    return () => clearInterval(t);
  }, [isTauri]);

  // ── Browser reload ──────────────────────────────────────────────────
  const [reloading, setReloading] = useState(false);

  /** Navigate away from the cached shell. The query parameter makes the
   *  navigation request miss any precached match for the current URL. */
  const hardReload = () => {
    const url = new URL(location.href);
    url.searchParams.set("_v", Date.now().toString(36));
    location.replace(url.toString());
  };

  const handleBrowserUpdate = async () => {
    setReloading(true);
    if (fallbackStale) {
      hardReload();
      return;
    }
    try {
      await updateServiceWorker(true);
    } catch {
      /* fall through to the forced reload below */
    }
    // updateServiceWorker() is supposed to activate the waiting worker and
    // reload the page itself. When the worker is stuck in "waiting" — or the
    // promise resolves without the controller ever changing — nothing happens
    // and the button looks dead, which is what users hit. If we are still
    // here shortly after, drop the workers and reload by hand.
    window.setTimeout(async () => {
      try {
        const regs = (await navigator.serviceWorker?.getRegistrations?.()) ?? [];
        await Promise.all(regs.map((r) => r.unregister().catch(() => false)));
      } catch {
        /* not fatal — the reload below still picks up new assets */
      }
      hardReload();
    }, 1500);
  };

  const showRefresh = !isTauri && (needRefresh || fallbackStale);
  const showOffline = !isTauri && offlineReady && !needRefresh;

  if (!showRefresh && !showOffline) return null;

  // ── UI ──────────────────────────────────────────────────────────────
  if (showRefresh) {
    return (
      <div className="fixed bottom-4 right-4 z-[60] max-w-sm animate-in slide-in-from-bottom-2 fade-in duration-300">
        <div className="bg-white border-2 border-indigo-500 rounded-2xl shadow-2xl shadow-indigo-200/50 overflow-hidden">
          <div className="bg-indigo-600 text-white px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle size={16} />
              <span className="text-xs font-bold tracking-tight">New version ready</span>
            </div>
            <button
              onClick={() => {
                setNeedRefresh(false);
                setFallbackStale(false);
              }}
              disabled={reloading}
              className="text-indigo-200 hover:text-white hover:bg-indigo-700 rounded p-0.5 transition-colors disabled:opacity-50"
              title="Dismiss"
            >
              <X size={14} />
            </button>
          </div>
          <div className="p-4">
            <p className="text-xs text-gray-700 leading-relaxed">
              uSTAT has been updated. This tab may still be running the old version; reload to pick up the new features and fixes.
            </p>
            <p className="text-[10px] text-gray-400 font-mono mt-1">
              Mevcut: v{__APP_VERSION__} · Build {__BUILD_TIME__}
            </p>
            <div className="mt-3 flex items-center gap-2">
              <button
                onClick={() => void handleBrowserUpdate()}
                disabled={reloading}
                className="flex-1 flex items-center justify-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold px-3 py-2 rounded-lg shadow-sm transition-colors disabled:bg-indigo-400"
              >
                <RefreshCw size={14} className={reloading ? "animate-spin" : ""} />
                {reloading ? "Reloading…" : "Reload to update"}
              </button>
              {!reloading && (
                <button
                  onClick={() => {
                    setNeedRefresh(false);
                    setFallbackStale(false);
                  }}
                  className="text-xs text-gray-500 hover:text-gray-700 px-2 py-2"
                >
                  Sonra
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-[60] max-w-sm animate-in fade-in duration-300">
      <div className="bg-emerald-50 border border-emerald-200 rounded-xl shadow-md px-3 py-2 flex items-center gap-2">
        <CheckCircle2 size={14} className="text-emerald-600 flex-shrink-0" />
        <span className="text-[11px] text-emerald-800">Ready for offline use.</span>
        <button
          onClick={() => setOfflineReady(false)}
          className="ml-1 text-emerald-600 hover:text-emerald-900"
        >
          <X size={12} />
        </button>
      </div>
    </div>
  );
}
