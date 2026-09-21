// Desktop auto-update.
//
// The check runs in Rust rather than in the web frontend on purpose: the
// window shows the page served by the local backend at http://127.0.0.1:<port>,
// which Tauri treats as a remote origin, so plugin calls from that page are
// refused unless a capability is opened to it. Opening one would hand any
// script on that origin the updater, and everything else granted there. A
// native dialog driven from here needs no IPC at all.
//
// Flow: check the GitHub release feed at startup and every CHECK_INTERVAL,
// ask the user, download, stop the backend, install, restart. A version the
// user declined is not offered again until the app restarts.

use std::time::Duration;

use tauri::AppHandle;
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::{Update, UpdaterExt};

const CHECK_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

/// Start the background update loop on its own thread. The dialogs block, and
/// blocking dialogs must stay off both the main thread and the async runtime.
pub fn spawn_update_checks(app: AppHandle) {
    std::thread::spawn(move || {
        let mut declined: Option<String> = None;
        loop {
            if let Err(e) = tauri::async_runtime::block_on(check_once(&app, &mut declined)) {
                eprintln!("uSTAT update check failed: {e}");
            }
            std::thread::sleep(CHECK_INTERVAL);
        }
    });
}

async fn check_once(
    app: &AppHandle,
    declined: &mut Option<String>,
) -> tauri_plugin_updater::Result<()> {
    let Some(update) = app.updater()?.check().await? else {
        return Ok(());
    };
    if declined.as_deref() == Some(update.version.as_str()) {
        return Ok(());
    }
    if !ask_to_install(app, &update) {
        *declined = Some(update.version.clone());
        return Ok(());
    }

    let bytes = match update.download(|_, _| {}, || {}).await {
        Ok(bytes) => bytes,
        Err(e) => {
            report_failure(app, &format!("The update could not be downloaded.\n\n{e}"));
            return Ok(());
        }
    };

    // The backend's files are part of what gets replaced, and on Windows the
    // installer cannot overwrite an executable that is still running. The
    // installer also exits this process there without running window
    // handlers, which would leave the backend orphaned.
    crate::backend::stop(app);
    if let Err(e) = update.install(bytes) {
        // The backend is already gone, so restarting is the only way back to
        // a working app; the current version is still installed.
        report_failure(app, &format!("The update could not be installed.\n\n{e}"));
    }
    app.restart();
}

fn ask_to_install(app: &AppHandle, update: &Update) -> bool {
    app.dialog()
        .message(format!(
            "uSTAT {} is available (you have {}).\n\n\
             Installing restarts uSTAT. An analysis that is open now will be \
             closed, so save your project first if you need it.",
            update.version, update.current_version,
        ))
        .title("uSTAT update")
        .kind(MessageDialogKind::Info)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Install and restart".into(),
            "Later".into(),
        ))
        .blocking_show()
}

fn report_failure(app: &AppHandle, message: &str) {
    app.dialog()
        .message(message)
        .title("uSTAT update")
        .kind(MessageDialogKind::Error)
        .blocking_show();
}
