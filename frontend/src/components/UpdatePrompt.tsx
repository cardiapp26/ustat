/**
 * UpdatePrompt — auto-detects a newer build of uSTAT and offers a one-click reload/update.
 *
 * How it works:
 *   1. Detects if running inside Tauri (desktop). If so, uses Tauri Auto-Updater.
 *   2. Otherwise (browser PWA):
 *      - Registers a service worker that pre-caches every asset in the dist/ folder.
 *      - When a new build is deployed, SW fires `needRefresh`.
 *      - Falls back to polling a static `/version.json` if SW is unavailable.
 */

import { useEffect, useState } from "react";
import { useRegisterSW } from "virtual:pwa-register/react";
import { RefreshCw, X, AlertCircle, CheckCircle2, Download } from "lucide-react";

// Build-time version stamp. Injected by vite.config.ts via `define`.
declare const __APP_VERSION__: string;
declare const __BUILD_TIME__: string;

const POLL_INTERVAL_MS = 60_000; // 1 minute
const FALLBACK_POLL_MS = 5 * 60_000; // 5 minutes

interface TauriUpdate {
  version?: string;
  downloadAndInstall: () => Promise<void>;
}

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

  // ── Tauri Auto-Updater Path (Desktop only) ──────────────────────────
  const [tauriUpdate, setTauriUpdate] = useState<TauriUpdate | null>(null);
  const [tauriUpdating, setTauriUpdating] = useState(false);
  const [tauriStatus, setTauriStatus] = useState("");

  useEffect(() => {
    if (!isTauri) return;

    const checkTauriUpdate = async () => {
      try {
        const { check } = await import("@tauri-apps/plugin-updater");
        const update = await check();
        if (update) {
          setTauriUpdate(update);
        }
      } catch (err) {
        console.error("[UpdatePrompt] Tauri updater failed:", err);
      }
    };

    void checkTauriUpdate();
    // Poll for updates every 15 minutes in desktop mode
    const t = setInterval(checkTauriUpdate, 15 * 60_000);
    return () => clearInterval(t);
  }, [isTauri]);

  const handleTauriUpdate = async () => {
    if (!tauriUpdate) return;
    try {
      setTauriUpdating(true);
      setTauriStatus("Downloading update...");
      
      // Download and install the update bundle (.dmg/.msi/.deb etc)
      await tauriUpdate.downloadAndInstall();
      
      setTauriStatus("Installing and restarting...");
      
      // Relaunch the application
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (err) {
      console.error("[UpdatePrompt] Failed to install Tauri update:", err);
      setTauriStatus("Update failed. Please restart and try again.");
      setTauriUpdating(false);
    }
  };

  const showRefresh = (!isTauri && (needRefresh || fallbackStale)) || (isTauri && !!tauriUpdate);
  const showOffline = !isTauri && offlineReady && !needRefresh;

  if (!showRefresh && !showOffline) return null;

  // ── UI ──────────────────────────────────────────────────────────────
  if (showRefresh) {
    const newVersion = isTauri && tauriUpdate ? tauriUpdate.version : "";

    return (
      <div className="fixed bottom-4 right-4 z-[60] max-w-sm animate-in slide-in-from-bottom-2 fade-in duration-300">
        <div className="bg-white border-2 border-indigo-500 rounded-2xl shadow-2xl shadow-indigo-200/50 overflow-hidden">
          <div className="bg-indigo-600 text-white px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle size={16} />
              <span className="text-xs font-bold tracking-tight">
                {isTauri ? "New desktop version ready" : "New version ready"}
              </span>
            </div>
            <button
              onClick={() => {
                if (isTauri) {
                  setTauriUpdate(null);
                } else {
                  setNeedRefresh(false);
                  setFallbackStale(false);
                }
              }}
              disabled={tauriUpdating}
              className="text-indigo-200 hover:text-white hover:bg-indigo-700 rounded p-0.5 transition-colors disabled:opacity-50"
              title="Dismiss"
            >
              <X size={14} />
            </button>
          </div>
          <div className="p-4">
            <p className="text-xs text-gray-700 leading-relaxed">
              {isTauri
                ? `An update for uSTAT Desktop (v${newVersion}) is available. Download and install it now?`
                : "uSTAT has been updated. This tab may still be running the old version — reload to pick up the new features and fixes."}
            </p>
            <p className="text-[10px] text-gray-400 font-mono mt-1">
              Mevcut: v{__APP_VERSION__} {!isTauri && `· Build ${__BUILD_TIME__}`}
            </p>
            {tauriStatus && (
              <p className="text-xs text-indigo-600 font-semibold mt-2 animate-pulse">
                {tauriStatus}
              </p>
            )}
            <div className="mt-3 flex items-center gap-2">
              <button
                onClick={() => {
                  if (isTauri) {
                    void handleTauriUpdate();
                  } else {
                    if (fallbackStale) {
                      const url = new URL(location.href);
                      url.searchParams.set("_v", Date.now().toString(36));
                      location.replace(url.toString());
                      return;
                    }
                    void updateServiceWorker(true);
                  }
                }}
                disabled={tauriUpdating}
                className="flex-1 flex items-center justify-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold px-3 py-2 rounded-lg shadow-sm transition-colors disabled:bg-indigo-400"
              >
                {tauriUpdating ? (
                  <RefreshCw size={14} className="animate-spin" />
                ) : isTauri ? (
                  <Download size={14} />
                ) : (
                  <RefreshCw size={14} />
                )}
                {tauriUpdating ? "Updating..." : isTauri ? "Download & Install" : "Reload to update"}
              </button>
              {!tauriUpdating && (
                <button
                  onClick={() => {
                    if (isTauri) {
                      setTauriUpdate(null);
                    } else {
                      setNeedRefresh(false);
                      setFallbackStale(false);
                    }
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
