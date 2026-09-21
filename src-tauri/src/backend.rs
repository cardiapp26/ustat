// The bundled Python backend: start it, wait until it serves, stop it.
//
// Every failure ends in a dialog that names the cause and the log file. The
// first release did none of this: a backend that died at startup left a
// window on "Starting statistical engine..." forever, and a missing backend
// binary exited before the user saw anything at all.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

/// A cold first start imports the whole scipy stack, and on Windows each of
/// its DLLs may be scanned by antivirus on first load; a healthy backend has
/// been seen to take well over the half minute the first release allowed.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
/// How long the backend gets to shut down after SIGTERM before it is killed.
const STOP_GRACE: Duration = Duration::from_secs(3);
/// How much of the log a failure dialog quotes: a Python traceback's last
/// frames and the exception, not the whole file.
const LOG_TAIL_BYTES: u64 = 1500;

pub(crate) struct BackendProcess(Mutex<Option<Child>>);

/// Start the backend and, once it serves, point the main window at it.
pub(crate) fn start(app: &AppHandle, port: u16) {
    let log_path = log_path(app);
    match spawn(app, port, log_path.as_deref()) {
        Ok(child) => {
            app.manage(BackendProcess(Mutex::new(Some(child))));
            let app = app.clone();
            std::thread::spawn(move || navigate_when_ready(&app, port, log_path.as_deref()));
        }
        Err(e) => fail(
            app,
            &format!("uSTAT's statistics engine could not be started.\n\n{e}"),
            None,
        ),
    }
}

/// Stop the backend, if it is still running. Safe to call twice: the child is
/// taken out of the state, so a second call finds nothing to stop.
pub(crate) fn stop(app: &AppHandle) {
    let Some(state) = app.try_state::<BackendProcess>() else {
        return;
    };
    let child = state.0.lock().ok().and_then(|mut guard| guard.take());
    if let Some(child) = child {
        stop_child(child);
    }
}

fn binary_path(app: &AppHandle) -> Result<PathBuf, String> {
    let name = if cfg!(windows) {
        "ustat-backend.exe"
    } else {
        "ustat-backend"
    };
    let dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("No resource directory: {e}"))?;
    Ok(dir.join("binaries").join(name))
}

/// Where the backend's output goes, recreated on each launch. None when the
/// log directory cannot be made; the app still runs, it just has no log.
fn log_path(app: &AppHandle) -> Option<PathBuf> {
    let dir = app.path().app_log_dir().ok()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("backend.log"))
}

fn spawn(app: &AppHandle, port: u16, log: Option<&Path>) -> Result<Child, String> {
    let binary = binary_path(app)?;
    let (stdout, stderr) = match log.and_then(|p| File::create(p).ok()) {
        Some(file) => {
            let copy = file.try_clone().map_err(|e| e.to_string())?;
            (Stdio::from(file), Stdio::from(copy))
        }
        // Never Stdio::piped(): nothing reads it, and a full pipe buffer
        // blocks the backend on its next write.
        None => (Stdio::null(), Stdio::null()),
    };
    let mut command = Command::new(&binary);
    command
        .args(["--port", &port.to_string()])
        .env("USTAT_NO_BROWSER", "1")
        .env("USTAT_DESKTOP_MODE", "1")
        .stdout(stdout)
        .stderr(stderr);
    #[cfg(windows)]
    {
        // The backend is a console program; without this Windows opens a
        // console window beside the app for as long as it runs.
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
        .spawn()
        .map_err(|e| format!("{}: {e}", binary.display()))
}

fn navigate_when_ready(app: &AppHandle, port: u16, log: Option<&Path>) {
    let address = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let start = Instant::now();
    while start.elapsed() < STARTUP_TIMEOUT {
        if let Some(code) = exited(app) {
            let reason = format!("uSTAT's statistics engine stopped while starting ({code}).");
            return fail(app, &reason, log);
        }
        if std::net::TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok() {
            if let (Some(window), Ok(url)) = (
                app.get_webview_window("main"),
                format!("http://127.0.0.1:{port}").parse(),
            ) {
                let _ = window.navigate(url);
            }
            return;
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    let reason = format!(
        "uSTAT's statistics engine did not start within {} seconds.",
        STARTUP_TIMEOUT.as_secs()
    );
    fail(app, &reason, log);
}

/// The backend's exit status if it has already exited.
fn exited(app: &AppHandle) -> Option<String> {
    let state = app.try_state::<BackendProcess>()?;
    let mut guard = state.0.lock().ok()?;
    let status = guard.as_mut()?.try_wait().ok()??;
    guard.take();
    Some(status.to_string())
}

/// Tell the user what went wrong, quoting the end of the log, then quit.
fn fail(app: &AppHandle, reason: &str, log: Option<&Path>) {
    stop(app);
    let mut message = reason.to_owned();
    if let Some(path) = log {
        let tail = log_tail(path);
        if !tail.trim().is_empty() {
            message.push_str(&format!("\n\nEnd of the log:\n{}", tail.trim_end()));
        }
        message.push_str(&format!("\n\nFull log: {}", path.display()));
    }
    let app = app.clone();
    app.dialog()
        .message(message)
        .title("uSTAT could not start")
        .kind(MessageDialogKind::Error)
        .show(move |_| app.exit(1));
}

fn log_tail(path: &Path) -> String {
    let Ok(mut file) = File::open(path) else {
        return String::new();
    };
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = file.seek(SeekFrom::Start(len.saturating_sub(LOG_TAIL_BYTES)));
    let mut bytes = Vec::new();
    let _ = file.read_to_end(&mut bytes);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Ask the backend to exit, kill it if it has not within the grace period,
/// and reap it either way so it does not linger as a zombie.
fn stop_child(mut child: Child) {
    #[cfg(unix)]
    {
        // SAFETY: kill(2) takes no pointers; the pid is our own child, which
        // has not been reaped yet, so it cannot have been reused.
        unsafe {
            libc::kill(child.id() as i32, libc::SIGTERM);
        }
    }
    #[cfg(windows)]
    {
        let _ = child.kill();
    }
    let deadline = Instant::now() + STOP_GRACE;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = child.kill();
    let _ = child.wait();
}
