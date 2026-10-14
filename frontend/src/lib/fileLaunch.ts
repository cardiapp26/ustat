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

interface LaunchListener {
  readonly onFile: (file: File) => void;
  readonly onError: (error: unknown) => void;
}

let watchedQueue: LaunchQueue | null = null;
let listener: LaunchListener | null = null;
let heldHandle: FileSystemFileHandle | null = null;
let launchedWithFile = false;

function deliver(handle: FileSystemFileHandle, to: LaunchListener): void {
  handle.getFile().then(to.onFile, to.onError);
}

/**
 * Start receiving file launches. Called from main.tsx before React renders.
 *
 * The browser queues launch params until a consumer is set and hands them
 * over as soon as one is, so taking them here, ahead of every component
 * effect, is what lets `wasLaunchedWithFile` answer truthfully at startup. A
 * launch that arrives before the welcome screen is listening is held for it.
 *
 * A session holds one dataset, so only the first file of a launch is opened.
 * The manifest asks for one window per file (`launch_type:
 * "multiple-clients"`), which means a multi-file selection normally never
 * reaches here as one launch; the guard covers browsers that ignore it.
 */
export function watchFileLaunches(): void {
  const queue = (window as WindowWithLaunchQueue).launchQueue;
  if (!queue || queue === watchedQueue) return;
  watchedQueue = queue;
  queue.setConsumer((params) => {
    const handle = params.files.find(isFileHandle);
    if (!handle) return;
    launchedWithFile = true;
    if (listener) deliver(handle, listener);
    else heldHandle = handle;
  });
}

/**
 * Whether the OS started this page with a file.
 *
 * Startup code that would navigate away (the silent Drive re-auth redirect in
 * cloudSync.init) must check this first: leaving the page unloads the
 * document the launch belongs to, and the file is not delivered again.
 */
export function wasLaunchedWithFile(): boolean {
  return launchedWithFile;
}

/**
 * Deliver the file the app was launched with to `onFile`, now if it already
 * arrived, otherwise when it does. Returns a function that stops delivery.
 */
export function consumeLaunchedFiles(
  onFile: (file: File) => void,
  onError: (error: unknown) => void,
): () => void {
  watchFileLaunches();
  const mine: LaunchListener = { onFile, onError };
  listener = mine;
  if (heldHandle) {
    const handle = heldHandle;
    heldHandle = null;
    deliver(handle, mine);
  }
  return () => {
    if (listener === mine) listener = null;
  };
}
