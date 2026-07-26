import { useCallback, useEffect, useState } from "react";
import { Upload, Info, Zap, BarChart2, ShieldAlert, ListChecks, Sparkles, NotebookPen, FileText, HeartPulse, Workflow, Layers, HelpCircle, Newspaper, Cloud, CloudDownload, LogOut, RefreshCw } from "lucide-react";
import { createBlankSession, uploadFile } from "../api";
import api from "../api";
import { useStore } from "../store";
import AboutModal from "./AboutModal";
import HelpModal from "./HelpModal";
import RecentSessionsPanel from "./RecentSessionsPanel";
import PowerPanel from "./PowerPanel";
import RefreshAppButton from "./RefreshAppButton";
import { cloudSync, type CloudStatusInfo } from "../lib/cloudSync";

export default function UploadZone() {
  const setSession = useStore((s) => s.setSession);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAbout, setShowAbout] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [mode, setMode] = useState<"home" | "power">("home");
  // Google Drive cloud-sync state — drives the connection strip on the
  // welcome screen. Live-updates via the sync motor's subscription.
  const [cloud, setCloud] = useState<CloudStatusInfo>(cloudSync.getStatus());
  const [cloudBusy, setCloudBusy] = useState(false);
  useEffect(() => {
    const unsub = cloudSync.subscribe(setCloud);
    return unsub;
  }, []);

  const onDriveConnect = () => {
    if (cloud.status === "setupNeeded") {
      window.alert(
        "Google Drive sync needs an OAuth Client ID.\n\n" +
          "Setup details are at the top of frontend/src/lib/cloudConfig.ts.",
      );
      return;
    }
    void cloudSync.signIn().catch((e) => console.warn("[cloud] signIn", e));
  };
  const onDriveSync = async () => {
    setCloudBusy(true);
    try {
      await cloudSync.syncNow(true);
    } catch (e) {
      console.warn("[cloud] sync", e);
    } finally {
      setCloudBusy(false);
    }
  };
  const onDriveDisconnect = () => {
    if (window.confirm("Disconnect Google Drive? Your locally saved sessions are kept.")) {
      void cloudSync.signOut();
    }
  };

  // Entering Power Analysis pushes a browser history entry so the back button
  // (and the in-app "Statistical Analysis" button) returns to the uSTAT home
  // screen instead of leaving the app entirely.
  useEffect(() => {
    // Any back/forward navigation returns to the home screen. (We deliberately
    // don't restore power mode from history state: a reload can leave a stale
    // power entry as the current history position, which would then wrongly
    // re-open power on back. Home-on-popstate is the robust choice and keeps
    // the user inside the app.)
    const onPop = () => setMode("home");
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const enterPower = () => {
    window.history.pushState({ ustatMode: "power" }, "");
    setMode("power");
  };
  // Consume the pushed entry so history stays balanced; popstate flips to home.
  const exitPower = () => window.history.back();

  const handle = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      // Session JSON files → load_session endpoint
      const isSessionJson = file.name.endsWith(".json");
      if (isSessionJson) {
        const form = new FormData();
        form.append("file", file);
        const res = await api.post("/api/sessions/load_session", form);
        setSession(res.data);
        // Restored sessions carry server-side decimal overrides — pull them
        // into the store so the table renders with the user's formatting.
        // The setSession call above reset columnDecimals because session_id
        // flipped; this re-hydrates it from the backend snapshot.
        try {
          const { getColumnDecimalsApi } = await import("../api");
          const dres = await getColumnDecimalsApi(res.data.session_id);
          if (dres.data && Object.keys(dres.data).length > 0) {
            const { useStore: store } = await import("../store");
            store.setState({ columnDecimals: dres.data });
          }
        } catch { /* non-fatal — fall back to defaults */ }
      } else {
        const res = await uploadFile(file);
        setSession(res.data);
      }
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string }; status?: number }; message?: string };
      const detail = err.response?.data?.detail;
      const status = err.response?.status;
      const msg = detail
        ? `${detail}`
        : err.message?.includes("Network")
        ? "Cannot connect to backend (localhost:8000). Is it running?"
        : `Upload failed (${status ?? err.message ?? "unknown error"})`;
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [setSession]);

  const startBlankWorkspace = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await createBlankSession();
      setSession(res.data);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string }; status?: number }; message?: string };
      const detail = err.response?.data?.detail;
      const status = err.response?.status;
      const msg = detail
        ? `${detail}`
        : err.message?.includes("Network")
        ? "Cannot connect to backend (localhost:8000). Is it running?"
        : `Could not start blank workspace (${status ?? err.message ?? "unknown error"})`;
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [setSession]);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) handle(e.dataTransfer.files[0]);
  };

  // ── Power Analysis Mode ──
  if (mode === "power") {
    return (
      <div className="flex flex-col h-screen bg-page">
        {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
        {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-3 bg-surface border-b border-line flex-shrink-0">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={exitPower}
              title="Back to the welcome screen"
              className="flex items-center gap-3 rounded-lg -m-1 p-1 hover:bg-ink-50 focus:outline-none focus:ring-2 focus:ring-ink-200 transition-colors"
            >
              <img src="/logo.png" alt="uSTAT" className="w-8 h-8 object-contain" />
              <span className="font-serif text-lg font-semibold text-slate-950 tracking-tight">uSTAT</span>
            </button>
            <span className="text-xs text-slate-300">·</span>
            <span className="text-xs text-gold-600 font-medium">Power Analysis</span>
          </div>
          <div className="flex items-center gap-2">
            <RefreshAppButton />
            <button onClick={() => setShowHelp(true)}
              title="Open Help & Analysis Guide"
              className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-ink-500 border border-line rounded-lg px-3 py-1.5 hover:border-ink-200 transition-colors">
              <HelpCircle size={14} />
              Help &amp; Guide
            </button>
            <button onClick={exitPower}
              className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-ink-500 border border-line rounded-lg px-3 py-1.5 hover:border-ink-200 transition-colors">
              <BarChart2 size={14} />
              Statistical Analysis
            </button>
          </div>
        </header>
        {/* Power Panel */}
        <main className="flex-1 overflow-y-auto p-4">
          <PowerPanel />
        </main>
        <footer className="text-center py-2 border-t border-line bg-surface">
          <p className="text-xs text-slate-500 tracking-wide">&copy; 2026 Dr. Yusuf Ho&#x15F;o&#x11F;lu. All rights reserved.</p>
        </footer>
      </div>
    );
  }

  // ── Home / Upload Mode ──
  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-8 p-8 bg-page">
      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
      <div className="flex flex-col items-center gap-3">
        <img
          src="/logo.png"
          alt="uSTAT logo"
          className="w-24 h-24 object-contain"
          style={{ filter: "drop-shadow(0 9px 22px rgba(31,72,128,0.18))" }}
        />
        <div className="text-center">
          <h1 className="font-serif text-5xl font-medium text-slate-950 leading-none tracking-tight">uSTAT</h1>
          <p className="text-sm text-slate-400 leading-none mt-3 tracking-wide">Statistical Analysis Platform</p>
        </div>
        <div
          className="update-app-banner flex items-center gap-3 rounded-lg border border-ink-200 bg-ink-50 px-3.5 py-2 shadow-card"
          role="note"
        >
          <RefreshAppButton
            variant="inline"
            label="Update app"
            className="!text-sm !font-semibold !text-ink-600 hover:!text-ink-700"
          />
          <p className="border-l border-ink-200 pl-3 text-xs font-medium text-ink-500">
            If something isn&apos;t working, update your app.
          </p>
        </div>
      </div>

      {/* Mode selector — symmetric tiles */}
      <div className="grid grid-cols-2 gap-4 w-full max-w-2xl">
        {/* Statistical Analysis = drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`flex flex-col items-center justify-center gap-2 px-5 py-6 rounded-card border-[1.5px] border-dashed transition-all min-h-[170px]
            ${dragging
              ? "border-ink-500 bg-ink-150"
              : "border-ink-200 bg-ink-50 hover:border-ink-500 hover:bg-ink-150"}`}
        >
          <div className="flex items-center gap-2 text-ink-600">
            <BarChart2 size={22} />
            <span className="text-[17px] font-semibold text-slate-800">Statistical Analysis</span>
          </div>
          <div className="flex flex-col items-center gap-1.5">
            <Upload size={22} className="text-ink-500" />
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={startBlankWorkspace}
                className="text-xs font-semibold text-ink-600 border border-ink-200 bg-white rounded-full px-3.5 py-1 hover:border-ink-400 hover:bg-ink-50 transition-colors"
              >
                New file
              </button>
              <button
                type="button"
                onClick={() => document.getElementById("file-input")?.click()}
                className="text-xs font-semibold text-ink-600 border border-ink-200 bg-white rounded-full px-3.5 py-1 hover:border-ink-400 hover:bg-ink-50 transition-colors"
              >
                Browse file
              </button>
            </div>
            <p className="text-xs text-ink-500/80">or drop your data file here</p>
            <p className="text-[11px] text-ink-300 mt-1.5 text-center px-2 tracking-wide">CSV · Excel · SPSS · SAS · Stata · Session JSON</p>
          </div>
          <input
            id="file-input"
            type="file"
            className="hidden"
            accept=".csv,.xlsx,.xls,.sas7bdat,.sav,.dta,.json"
            onChange={(e) => e.target.files?.[0] && handle(e.target.files[0])}
          />
        </div>

        {/* Power Analysis — separate, equal size */}
        <button
          onClick={enterPower}
          className="flex flex-col items-center justify-center gap-2 px-5 py-6 rounded-card border-[1.5px] border-line bg-surface text-slate-400 hover:border-gold-400 hover:bg-gold-50 hover:text-gold-600 hover:-translate-y-0.5 transition-all min-h-[170px]"
        >
          <div className="flex items-center gap-2">
            <Zap size={22} className="text-gold-600" />
            <span className="text-[17px] font-semibold text-slate-800">Power Analysis</span>
          </div>
          <span className="text-xs text-slate-400">Sample size &amp; power — no data needed</span>
          <span className="mt-1.5 text-xs font-semibold text-gold-600 border border-gold-200 bg-gold-100 rounded-full px-3.5 py-1">Open calculator</span>
        </button>
      </div>

      {/* Recent sessions (auto-saved in IndexedDB) — only renders when
          the user has at least one saved snapshot. Lets them resume
          exactly where they left off without re-uploading the dataset. */}
      <RecentSessionsPanel />

      {/* Google Drive cloud-sync strip — always visible on the welcome
          screen so a brand-new device (no local snapshots) can still
          connect and pull sessions from Drive. Collapses to a single
          "connected" row once signed in. */}
      <div className="w-full max-w-2xl rounded-xl border border-sky-200 bg-sky-50/60 px-4 py-3 flex items-center gap-3 flex-wrap">
        <Cloud size={18} className="text-sky-500 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          {cloud.signedIn ? (
            <>
              <p className="text-xs font-semibold text-sky-800 truncate">
                Google Drive connected
                {cloud.user?.email && (
                  <span className="font-normal text-sky-600"> · {cloud.user.email}</span>
                )}
              </p>
              <p className="text-[10px] text-sky-500">
                {cloud.status === "syncing"
                  ? "Senkronize ediliyor…"
                  : cloud.status === "error"
                    ? `Error: ${cloud.message || "sync failed"}`
                    : cloud.lastSync
                      ? `Son senkronizasyon: ${new Date(cloud.lastSync).toLocaleString()}`
                      : "Not synced yet"}
              </p>
            </>
          ) : (
            <>
              <p className="text-xs font-semibold text-sky-800">
                Carry your work across devices &amp; back it up with Google Drive
              </p>
              <p className="text-[10px] text-sky-500">
                Sessions are backed up to your own private Drive folder — they never pass through our server.
              </p>
            </>
          )}
        </div>
        {cloud.signedIn ? (
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <button
              onClick={onDriveSync}
              disabled={cloudBusy}
              className="inline-flex items-center gap-1 text-[11px] font-semibold text-sky-700 bg-white border border-sky-200 hover:bg-sky-100 rounded-lg px-2.5 py-1.5 transition-colors disabled:opacity-50"
            >
              <RefreshCw size={12} className={cloudBusy ? "animate-spin" : ""} />
              {cloudBusy ? "…" : "Senkronize et"}
            </button>
            <button
              onClick={onDriveDisconnect}
              title="Disconnect"
              className="inline-flex items-center text-[11px] font-semibold text-sky-600 hover:text-red-600 bg-white border border-sky-200 hover:border-red-200 hover:bg-red-50 rounded-lg px-2 py-1.5 transition-colors"
            >
              <LogOut size={12} />
            </button>
          </div>
        ) : (
          <button
            onClick={onDriveConnect}
            className={`inline-flex items-center gap-1.5 text-[11px] font-semibold rounded-lg px-3 py-1.5 transition-colors flex-shrink-0 ${
              cloud.status === "setupNeeded"
                ? "text-amber-700 bg-amber-50 border border-amber-300 hover:bg-amber-100"
                : "text-white bg-sky-600 hover:bg-sky-700"
            }`}
          >
            <CloudDownload size={13} />
            Connect Drive
          </button>
        )}
      </div>

      {/* Quick facts — privacy, scope, cost */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 w-full max-w-2xl text-xs">
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
          <ShieldAlert size={14} className="text-amber-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-gray-700">Privacy</p>
            <p className="text-gray-500 leading-snug">Files held in memory only — never written to disk. Cleared 30 min after you stop using the app.</p>
          </div>
        </div>
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
          <ListChecks size={14} className="text-indigo-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-gray-700">Scope</p>
            <p className="text-gray-500 leading-snug">t-tests, ANOVA, regression, non-parametric, survival, power &amp; more.</p>
          </div>
        </div>
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white">
          <Sparkles size={14} className="text-emerald-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-gray-700">Cost</p>
            <p className="text-gray-500 leading-snug">Free to use. No account, no paywall.</p>
          </div>
        </div>
      </div>

      {/* ── Other drtr.uk apps ─────────────────────────────────────────── */}
      <div className="w-full max-w-2xl">
        <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider text-center mb-2">
          More tools by Dr. Yusuf Ho&#x15F;o&#x11F;lu
        </p>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
          {[
            { url: "https://not.drtr.uk",    Icon: NotebookPen, name: "Notepad",    desc: "Local-only text editor" },
            { url: "https://pdf.drtr.uk/",   Icon: FileText,    name: "PDF",        desc: "Annotate &amp; sign in browser" },
            { url: "https://ecgcal.drtr.uk/",Icon: HeartPulse,  name: "ECG Caliper",desc: "Digital ECG wave analyzer" },
            { url: "https://neodw.drtr.uk/", Icon: Layers,      name: "NeoDW",      desc: "DICOM workstation (any modality)" },
            { url: "https://flow.drtr.uk",   Icon: Workflow,    name: "AcademicFlow", desc: "Journal flowchart designer" },
            { url: "https://arted.drtr.uk/", Icon: Newspaper, name: "Articleditor", desc: "Academic article editor" },
          ].map(({ url, Icon, name, desc }) => (
            <a
              key={url}
              href={url}
              target="_blank"
              rel="noreferrer"
              className="flex flex-col items-center gap-1 px-2 py-2 rounded-lg border border-gray-200 bg-white hover:border-indigo-300 hover:bg-indigo-50 transition-colors text-center group"
            >
              <Icon size={18} className="text-indigo-500 group-hover:text-indigo-600" />
              <span className="text-xs font-semibold text-gray-700 group-hover:text-indigo-700">{name}</span>
              <span
                className="text-[9px] text-gray-400 leading-snug"
                dangerouslySetInnerHTML={{ __html: desc }}
              />
            </a>
          ))}
        </div>
      </div>

      {loading && <p className="text-indigo-600 animate-pulse">Opening and parsing your data…</p>}
      {error && <p className="text-red-500 text-sm">{error}</p>}

      <div className="flex items-center gap-4 flex-wrap justify-center">
        <button
          onClick={() => setShowHelp(true)}
          className="flex items-center gap-1.5 text-indigo-600 hover:text-indigo-800 text-xs font-semibold transition-colors border border-indigo-200 hover:border-indigo-400 bg-indigo-50/60 hover:bg-indigo-100 rounded-full px-3 py-1.5"
          title="7-section walkthrough: Quick Start, Tests, Regression, Causal, Prediction & Validation, EFA/Bayes/Meta, R Hub"
        >
          <HelpCircle size={14} />
          Help &amp; Analysis Guide
        </button>
        <button
          onClick={() => setShowAbout(true)}
          className="flex items-center gap-1.5 text-gray-400 hover:text-indigo-600 text-xs transition-colors"
        >
          <Info size={14} />
          About uSTAT — packages &amp; methods
        </button>
        <RefreshAppButton variant="inline" />
      </div>

      <p className="text-xs text-slate-500 tracking-wide mt-2">&copy; 2026 Dr. Yusuf Ho&#x15F;o&#x11F;lu. All rights reserved.</p>
      <div className="flex flex-wrap items-center gap-2 text-[10px] text-gray-400 mt-1">
        <a href="/privacy" target="_blank" rel="noreferrer" className="hover:text-indigo-500">Privacy</a>
        <span className="text-gray-200">·</span>
        <a href="/terms" target="_blank" rel="noreferrer" className="hover:text-indigo-500">Terms</a>
        <span className="text-gray-200">·</span>
        <a href="/security" target="_blank" rel="noreferrer" className="hover:text-indigo-500">Security</a>
        <span className="text-gray-200">·</span>
        <a href="https://github.com/afstudy20-gif/ustat" target="_blank" rel="noreferrer" className="hover:text-indigo-500">Source</a>
      </div>

    </div>
  );
}
