/**
 * Files the operating system hands to the installed PWA.
 *
 * The manifest's `file_handlers` (vite.config.ts) registers uSTAT with the OS
 * for .sav, .xlsx, .xls and .csv, so Finder's "Open With" and Explorer's
 * "Open with" menus offer it once the app is installed from Chrome or Edge.
 * A file opened that way arrives here, through `window.launchQueue`, as a
 * FileSystemFileHandle rather than through the upload input.
 *
 * `launchQueue` exists only in Chromium browsers, and only carries files when
 * the page was started by a file launch. Everywhere else (Safari, Firefox, a
 * plain browser tab, the Tauri shell) this is a no-op.
 */

interface LaunchParams {
  readonly files: readonly FileSystemHandle[];
}

interface LaunchQueue {
  setConsumer(consumer: (params: LaunchParams) => void): void;
}

type WindowWithLaunchQueue = Window & { launchQueue?: LaunchQueue };

function isFileHandle(handle: FileSystemHandle): handle is FileSystemFileHandle {
  return handle.kind === "file";
}

/**
 * Deliver the file the app was launched with to `onFile`.
 *
 * A session holds one dataset, so only the first file of a launch is opened.
 * The manifest asks for one window per file (`launch_type:
 * "multiple-clients"`), which means a multi-file selection normally never
 * reaches here as one launch; the guard covers browsers that ignore it.
 *
 * Launch params are queued by the browser until a consumer is set, so calling
 * this from a component's effect does not lose a launch that happened before
 * React mounted. Setting a new consumer replaces the old one and does not
 * re-deliver params already consumed.
 */
export function consumeLaunchedFiles(
  onFile: (file: File) => void,
  onError: (error: unknown) => void,
): void {
  const queue = (window as WindowWithLaunchQueue).launchQueue;
  if (!queue) return;
  queue.setConsumer((params) => {
    const handle = params.files.find(isFileHandle);
    if (!handle) return;
    handle.getFile().then(onFile, onError);
  });
}
