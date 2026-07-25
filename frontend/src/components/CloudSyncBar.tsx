/**
 * CloudSyncBar — Google Drive sync status indicator for the uSTAT header.
 *
 * Sits next to the auto-save pill. States:
 *   - Not signed in: "Connect Google Drive" button (opens the GIS popup /
 *     redirect flow).
 *   - Signed in: avatar + email + status dot (syncing = amber pulse,
 *     ok = emerald, error = red). Clicking the dot triggers a manual
 *     syncNow; the kebab menu offers "Oturumu Kapat".
 *   - setupNeeded (client id missing): button stays clickable and shows
 *     an explanatory alert, mirroring notepad's behaviour.
 *
 * Re-renders live via cloudSync.subscribe().
 */

import { useEffect, useState } from "react";
import { Cloud, LogOut, RefreshCw } from "lucide-react";
import {
  cloudSync,
  type CloudStatusInfo,
} from "../lib/cloudSync";
import { CLOUD_CONFIG } from "../lib/cloudConfig";

export default function CloudSyncBar() {
  const [info, setInfo] = useState<CloudStatusInfo>(cloudSync.getStatus());
  const [menuOpen, setMenuOpen] = useState(false);

  // Live status updates from the sync motor.
  useEffect(() => {
    const unsub = cloudSync.subscribe(setInfo);
    return unsub;
  }, []);

  // Close the kebab menu on outside click.
  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-cloud-menu]")) setMenuOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [menuOpen]);

  const onSignIn = () => {
    if (info.status === "setupNeeded") {
      window.alert(
        "Google Drive sync needs an OAuth Client ID.\n\n" +
          "Setup (done by the app owner):\n" +
          "1. console.cloud.google.com → new project\n" +
          "2. Enable the Google Drive API\n" +
          "3. OAuth consent screen + scope: drive.appdata\n" +
          "4. Credentials → OAuth client ID (Web)\n" +
          "   - JS origins: https://ustat.drtr.uk, http://localhost:5173, http://127.0.0.1\n" +
          "   - Redirect URIs: https://ustat.drtr.uk/ (trailing slash)\n" +
          "5. Paste the Client ID into frontend/src/lib/cloudConfig.ts\n" +
          "6. Reload\n\n" +
          "Details are at the top of cloudConfig.ts.",
      );
      return;
    }
    void cloudSync.signIn().catch((e) => console.warn("[cloud] signIn", e));
  };

  const onSignOut = () => {
    if (window.confirm("Disconnect Google Drive? Your locally saved sessions are kept.")) {
      setMenuOpen(false);
      void cloudSync.signOut();
    }
  };

  const onManualSync = () => {
    setMenuOpen(false);
    void cloudSync.syncNow(true).catch((e) =>
      console.warn("[cloud] manual sync", e),
    );
  };

  // Not signed in → connect button.
  if (!info.signedIn) {
    return (
      <button
        onClick={onSignIn}
        title={
          info.status === "setupNeeded"
            ? "Google OAuth Client ID eksik (cloudConfig.ts)"
            : "Back your sessions up to your own Google Drive and carry them across devices"
        }
        className={`flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-semibold transition-colors ${
          info.status === "setupNeeded"
            ? "text-amber-600 bg-amber-50 hover:bg-amber-100 border border-amber-200"
            : "text-indigo-600 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200"
        }`}
      >
        <Cloud size={13} />
        <span className="hidden sm:inline">Connect Drive</span>
      </button>
    );
  }

  // Signed in → avatar + status dot + kebab.
  const dotColor =
    info.status === "syncing"
      ? "bg-amber-400 animate-pulse"
      : info.status === "ok"
        ? "bg-emerald-500"
        : info.status === "error"
          ? "bg-red-500"
          : "bg-gray-300";

  const statusLabel =
    info.status === "syncing"
      ? "Senkronize ediliyor…"
      : info.status === "ok"
        ? "Drive ile senkronize"
        : info.status === "error"
          ? `Sync error${info.message ? ": " + info.message : ""}`
          : "Idle";

  const lastSyncStr = info.lastSync
    ? new Date(info.lastSync).toLocaleString()
    : "-";

  return (
    <div className="relative flex items-center" data-cloud-menu>
      <button
        onClick={onManualSync}
        title={`${statusLabel}\nSon senkronizasyon: ${lastSyncStr}`}
        className="flex items-center gap-1.5 px-2 py-1 rounded-lg hover:bg-gray-100 transition-colors"
      >
        {info.user?.picture ? (
          <img
            src={info.user.picture}
            alt=""
            className="w-5 h-5 rounded-full"
            referrerPolicy="no-referrer"
          />
        ) : (
          <Cloud size={14} className="text-indigo-500" />
        )}
        <span className="hidden md:inline text-[11px] text-gray-600 max-w-[120px] truncate">
          {info.user?.email || info.user?.name || "Google Drive"}
        </span>
        <span className={`inline-block w-1.5 h-1.5 rounded-full ${dotColor}`} />
      </button>

      <button
        onClick={() => setMenuOpen((v) => !v)}
        title="Drive options"
        className={`p-1 rounded-lg transition-colors ${menuOpen ? "text-gray-700 bg-gray-100" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}
      >
        <RefreshCw size={12} />
      </button>

      {menuOpen && (
        <div className="absolute right-0 top-full mt-1 w-52 bg-white border border-gray-200 rounded-xl shadow-xl z-50 overflow-hidden">
          <div className="px-3 pt-2.5 pb-1 border-b border-gray-100">
            <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
              Google Drive
            </p>
            <p className="text-xs text-gray-700 truncate" title={info.user?.email}>
              {info.user?.email || info.user?.name || "Connected"}
            </p>
          </div>
          <button
            onClick={onManualSync}
            className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-gray-700 hover:bg-gray-50 transition-colors"
          >
            <RefreshCw size={13} />
            Sync now
          </button>
          <button
            onClick={onSignOut}
            className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-red-600 hover:bg-red-50 transition-colors border-t border-gray-100"
          >
            <LogOut size={13} />
            Oturumu Kapat
          </button>
        </div>
      )}
    </div>
  );
}

// Keep CLOUD_CONFIG referenced so tree-shaking doesn't drop it if a future
// build inlines the config check elsewhere. (No-op at runtime.)
void CLOUD_CONFIG;
