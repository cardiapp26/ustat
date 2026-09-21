// uSTAT Desktop — Tauri v2 library entry point
//
// Architecture:
// 1. On launch, spawn the bundled Python backend (FastAPI/uvicorn) as a subprocess
// 2. Poll until the backend is ready via TCP connection
// 3. Navigate the webview to http://127.0.0.1:<port>
// 4. On close, kill the backend process
// 5. In release builds, check GitHub Releases for signed updates (updater.rs)

mod updater;

use std::process::Child;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager};

/// How long the backend gets to shut down after SIGTERM before it is killed.
const BACKEND_STOP_GRACE: Duration = Duration::from_secs(3);

struct BackendProcess(std::sync::Mutex<Option<Child>>);

/// Find a free TCP port starting from `preferred`.
fn find_free_port(preferred: u16) -> u16 {
    for port in preferred..preferred + 50 {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind to any port")
        .local_addr()
        .unwrap()
        .port()
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
    let deadline = Instant::now() + BACKEND_STOP_GRACE;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Stop the bundled backend, if it is still running. Safe to call twice: the
/// child is taken out of the state, so a second call finds nothing to stop.
pub(crate) fn stop_backend(app: &AppHandle) {
    let Some(state) = app.try_state::<BackendProcess>() else {
        return;
    };
    let child = state.0.lock().ok().and_then(|mut guard| guard.take());
    if let Some(child) = child {
        stop_child(child);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = find_free_port(18731);

    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
        .setup(move |app| {
            let handle = app.handle().clone();

            // Resolve the bundled backend binary path
            let resource_dir = handle
                .path()
                .resource_dir()
                .expect("cannot resolve resource dir");

            let backend_binary = if cfg!(target_os = "windows") {
                resource_dir.join("binaries").join("ustat-backend.exe")
            } else {
                resource_dir.join("binaries").join("ustat-backend")
            };

            // Spawn the backend process
            let port_str = port.to_string();
            let child = std::process::Command::new(&backend_binary)
                .args(["--port", &port_str])
                .env("USTAT_NO_BROWSER", "1")
                .env("USTAT_DESKTOP_MODE", "1")
                // Nothing reads these. Piped, the backend would block on its
                // first write past the OS pipe buffer and hang the app.
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap_or_else(|e| {
                    eprintln!(
                        "Failed to start uSTAT backend at {:?}: {}",
                        backend_binary, e
                    );
                    std::process::exit(1);
                });

            app.manage(BackendProcess(std::sync::Mutex::new(Some(child))));

            // A debug build has the placeholder version 0.1.0 and would be
            // offered every release; only shipped builds check.
            if !cfg!(debug_assertions) {
                updater::spawn_update_checks(handle.clone());
            }

            // Spawn a thread to wait for the backend, then navigate
            std::thread::spawn(move || {
                let url = format!("http://127.0.0.1:{}", port);
                let max_wait = std::time::Duration::from_secs(30);
                let start = std::time::Instant::now();

                loop {
                    if start.elapsed() > max_wait {
                        eprintln!("uSTAT backend did not start within 30s");
                        break;
                    }
                    if std::net::TcpStream::connect_timeout(
                        &format!("127.0.0.1:{}", port).parse().unwrap(),
                        std::time::Duration::from_millis(200),
                    )
                    .is_ok()
                    {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        if let Some(window) = handle.get_webview_window("main") {
                            let _ = window.navigate(url.parse().unwrap());
                        }
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(300));
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                stop_backend(window.app_handle());
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running uSTAT desktop");
}
