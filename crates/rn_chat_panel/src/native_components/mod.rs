//! Native GPUI components for rn_chat_panel.
//!
//! These components provide native rendering of diff and terminal views
//! inside React Native's ToolCallCard. Props are minimal (just IDs and indices)
//! with content looked up on the native side from the tool call content store.

mod native_diff_view;
mod native_terminal_view;

use gpui_host::extensions::{register_component, register_extension, Extension};

pub use native_diff_view::NativeDiffView;
pub use native_terminal_view::NativeTerminalView;

/// Extension registration for rn_chat_panel native components.
pub struct RnChatPanelExtension;

impl Extension for RnChatPanelExtension {
    fn name(&self) -> &'static str {
        "rn-chat-panel"
    }

    fn register_components(&self) {
        register_component::<NativeDiffView>();
        register_component::<NativeTerminalView>();
    }
}

/// Initialize the rn_chat_panel native components.
///
/// Call this before `gpui_host::app_bootstrap::run_with_supervisor()` in the
/// app's main function to register the components with the RNGPUI runtime.
pub fn init() {
    register_extension(Box::new(RnChatPanelExtension));
}
