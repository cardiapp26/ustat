import { AlertCircle, RefreshCw, X } from "lucide-react";

interface UploadStatusProps {
  /** True while a file (or a blank workspace) is being opened. */
  loading: boolean;
  /** Name of the file being opened; null for a blank workspace. */
  fileName: string | null;
  error: string | null;
  onDismissError: () => void;
}

/**
 * Progress and failure for opening a dataset, shown where they cannot be
 * missed. Both used to be a line of text at the foot of the welcome screen,
 * below the fold: a 214-column workbook takes over ten seconds to parse, and a
 * file opened from Finder's "Open With" arrives with nothing on screen to
 * suggest it is being read, so users gave up on uploads that were working.
 */
export default function UploadStatus({ loading, fileName, error, onDismissError }: UploadStatusProps) {
  return (
    <>
      {loading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-page/80 backdrop-blur-sm">
          <div
            role="status"
            aria-live="polite"
            className="flex items-center gap-3 rounded-card border border-ink-200 bg-surface px-5 py-4 shadow-card max-w-md"
          >
            <RefreshCw size={20} className="text-ink-500 animate-spin flex-shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-semibold text-slate-800 truncate">
                {fileName ? `Opening ${fileName}` : "Starting a blank workspace"}
              </p>
              <p className="text-xs text-slate-500">Large files can take a few seconds.</p>
            </div>
          </div>
        </div>
      )}
      {error && (
        <div
          role="alert"
          className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 shadow-card max-w-lg w-[calc(100%-2rem)]"
        >
          <AlertCircle size={16} className="text-red-500 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-700 flex-1">{error}</p>
          <button
            type="button"
            onClick={onDismissError}
            aria-label="Dismiss"
            className="text-red-400 hover:text-red-600 flex-shrink-0"
          >
            <X size={14} />
          </button>
        </div>
      )}
    </>
  );
}
