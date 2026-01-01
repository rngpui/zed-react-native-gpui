//! React Native Chat Panel for Zed
//!
//! This crate provides a demo panel that embeds a React Native-based chat UI
//! within Zed using the react-native-gpui library. It demonstrates:
//!
//! - Embedding RNGPUI within an existing GPUI application
//! - Bridging Zed's LLM infrastructure to React Native via TurboModules
//! - Creating panels that render React Native surfaces
//!
//! ## Usage
//!
//! Initialize in Zed's main:
//! ```rust,ignore
//! #[cfg(feature = "rn-chat-demo")]
//! rn_chat_panel::init(cx);
//! ```

mod generated;
mod chat_pane;
mod acp_thread_projection;
mod native_components;
mod panel;
mod runtime;
mod zed_host_command;

pub use zed_host_command::{
    send, send_with_reply_blocking, set_agent_signal_emitters, set_llm_signal_emitters,
    set_theme_signal_emitter, CommandError, FontSettingsData, ThemeColorsData, ThemeData,
    WorkspaceInfo, ZedHostCommand,
};

// Force link the FFI crate to provide symbols required by the C++ runtime
extern crate rngpui_ffi;

use collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

// FFI declarations for turbomodule registration callback
unsafe extern "C" {
    /// Set a callback for host apps to register additional TurboModules.
    /// Must be called before runtime initialization.
    fn gpui_set_extra_turbomodules_callback(
        callback: Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
    );
}

// C++ shim function that registers Zed's turbomodules (ZedTheme, ZedLLM, ZedWorkspace)
unsafe extern "C" {
    fn zed_register_turbomodules(registry: *mut c_void, invoker: *mut c_void);
}

use gpui::App;
use rngpui::font_resolver::{set_default_font_family, set_family_aliases};
use rngpui::library_mode::{BundleSource, LibraryConfig, LibraryModeHandle};
use settings::Settings;
use theme::ThemeSettings;

pub use panel::RNChatPanel;

/// Global handle to keep the RN runtime alive.
static LIBRARY_HANDLE: OnceLock<LibraryModeHandle> = OnceLock::new();

/// Initialize the RN Chat Panel subsystem.
///
/// Must be called once during Zed startup, after GPUI is ready but before
/// any windows are created that might use the RN panel.
///
/// # Panics
///
/// Panics if called more than once or if RN runtime initialization fails.
pub fn init(cx: &mut App) {
    log::info!("rn_chat_panel: Initializing RNGPUI in library mode");

    // Configure Zed's fonts as defaults for React Native
    configure_fonts(cx);

    // Register extensions before calling `rngpui::init()`.
    // `rngpui::init()` initializes the extension registry; registering after
    // that point is ignored.
    rngpui_ext_rnsvg::init();
    native_components::init();
    rngpui::init();

    // Initialize signal emitters (connects craby signals to the command bus)
    ::zedmodules::init_signal_emitters();

    // Initialize the Zed host command bus
    zed_host_command::init(cx);

    // Register Zed's turbomodules (ZedTheme, ZedLLM, ZedWorkspace) BEFORE runtime init.
    // This callback will be invoked during TurboModule registry creation.
    unsafe {
        gpui_set_extra_turbomodules_callback(Some(zed_register_turbomodules));
    }
    log::info!("rn_chat_panel: Registered Zed turbomodules callback");

    let config = LibraryConfig {
        bundle_source: get_bundle_source(),
        module_name: Some("RNChatPanel".to_string()),
    };

    let handle = rngpui::library_mode::initialize_library_mode(config, cx)
        .expect("Failed to initialize RNGPUI library mode");

    LIBRARY_HANDLE
        .set(handle)
        .expect("RN runtime already initialized");

    // Register the panel with workspace
    panel::init(cx);

    log::info!("rn_chat_panel: Initialization complete");
}

/// Configure React Native to use Zed's font settings.
fn configure_fonts(cx: &App) {
    let settings = ThemeSettings::get_global(cx);

    // Set Zed's UI font as the default for React Native
    let ui_font_family = settings.ui_font.family.to_string();
    log::info!(
        "rn_chat_panel: Setting default font family to '{}'",
        ui_font_family
    );
    set_default_font_family(&ui_font_family);

    // Map generic font families to Zed's fonts
    let buffer_font_family = settings.buffer_font.family.to_string();
    let aliases: HashMap<String, String> = [
        ("sans-serif".to_string(), ui_font_family.clone()),
        ("monospace".to_string(), buffer_font_family.clone()),
        ("ui-sans-serif".to_string(), ui_font_family),
        ("ui-monospace".to_string(), buffer_font_family),
    ]
    .into_iter()
    .collect();
    log::info!("rn_chat_panel: Setting font family aliases: {:?}", aliases);
    set_family_aliases(aliases);
}

/// Get the bundle source based on build configuration.
fn get_bundle_source() -> BundleSource {
    #[cfg(feature = "dev-mode")]
    {
        let url = std::env::var("RN_METRO_URL").unwrap_or_else(|_| {
            "http://localhost:8082/index.bundle?platform=gpui&dev=true".to_string()
        });
        log::info!("rn_chat_panel: Using Metro bundle source: {}", url);
        BundleSource::Metro { url }
    }

    #[cfg(not(feature = "dev-mode"))]
    {
        // For now, default to Metro in non-dev mode too
        // TODO: Implement embedded bundle support
        let url = std::env::var("RN_METRO_URL").unwrap_or_else(|_| {
            "http://localhost:8082/index.bundle?platform=gpui&dev=true".to_string()
        });
        log::info!("rn_chat_panel: Using Metro bundle source: {}", url);
        BundleSource::Metro { url }
    }
}

/// Check if the RN runtime is initialized.
pub fn is_initialized() -> bool {
    LIBRARY_HANDLE.get().is_some()
}
