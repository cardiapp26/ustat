// uSTAT Desktop, Tauri v2 library entry point
//
// Architecture:
// 1. On launch, spawn the bundled Python backend (FastAPI/uvicorn) as a subprocess
// 2. Poll until the backend is ready via TCP connection
// 3. Navigate the webview to http://127.0.0.1:<port>
// 4. On close, kill the backend process
// 5. In release builds, check GitHub Releases for signed updates (updater.rs)
//
// Steps 1-4 live in backend.rs.

mod backend;
mod updater;

use tauri::Manager;

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
            backend::start(&handle, port);

            // A debug build has the placeholder version 0.1.0 and would be
            // offered every release; only shipped builds check.
            if !cfg!(debug_assertions) {
                updater::spawn_update_checks(handle);
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                backend::stop(window.app_handle());
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running uSTAT desktop");
}
