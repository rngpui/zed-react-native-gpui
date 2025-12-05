#[rustfmt::skip]
pub(crate) mod ffi;
pub(crate) mod generated;

pub(crate) mod zed_icons_impl;
pub(crate) mod zed_llm_impl;
pub(crate) mod zed_markdown_impl;
pub(crate) mod zed_theme_impl;
pub(crate) mod zed_workspace_impl;

use ffi::bridging::*;
use generated::*;

/// Initialize signal emitters for rn_chat_panel_types.
/// This should be called during crate initialization.
pub fn init_signal_emitters() {
    // Set up theme signal emitter
    rn_chat_panel_types::set_theme_signal_emitter(Box::new(|module_id, theme_data| {
        emit_theme_changed_signal(module_id, theme_data);
    }));

    // Set up workspace signal emitter
    rn_chat_panel_types::set_workspace_signal_emitter(Box::new(|module_id, workspace_info| {
        emit_workspace_changed_signal(module_id, workspace_info);
    }));

    // Set up LLM signal emitters
    rn_chat_panel_types::set_llm_signal_emitters(
        Box::new(|module_id, request_id, chunk| {
            emit_llm_chunk_signal(module_id, request_id, chunk);
        }),
        Box::new(|module_id, request_id| {
            emit_llm_done_signal(module_id, request_id);
        }),
        Box::new(|module_id, request_id, error| {
            emit_llm_error_signal(module_id, request_id, error);
        }),
    );

    // Set up ACP thread history signal emitter
    rn_chat_panel_types::set_acp_threads_signal_emitter(Box::new(|module_id, threads_json| {
        emit_acp_threads_changed_signal(module_id, threads_json);
    }));

    // Set up tool call signal emitter
    rn_chat_panel_types::set_tool_call_signal_emitter(Box::new(|module_id, event| {
        emit_tool_call_event_signal(module_id, event);
    }));

    // Set up thread snapshot signal emitter
    rn_chat_panel_types::set_thread_snapshot_emitter(Box::new(|module_id, snapshot| {
        emit_thread_snapshot_signal(module_id, snapshot);
    }));

    // Set up agent connection signal emitters
    rn_chat_panel_types::set_agent_signal_emitters(
        Box::new(|module_id, agent_type, session_id| {
            emit_agent_connected_signal(module_id, agent_type, session_id);
        }),
        Box::new(|module_id, agent_type, session_id| {
            emit_agent_disconnected_signal(module_id, agent_type, session_id);
        }),
    );
}

fn emit_theme_changed_signal(module_id: usize, theme_data: rn_chat_panel_types::ThemeData) {
    let manager = get_signal_manager();
    let craby_theme = ThemeData {
        name: theme_data.name,
        appearance: theme_data.appearance,
        colors: ThemeColors {
            background: theme_data.colors.background,
            surface_background: theme_data.colors.surface_background,
            elevated_surface_background: theme_data.colors.elevated_surface_background,
            panel_background: theme_data.colors.panel_background,
            element_background: theme_data.colors.element_background,
            element_hover: theme_data.colors.element_hover,
            element_active: theme_data.colors.element_active,
            element_selected: theme_data.colors.element_selected,
            element_disabled: theme_data.colors.element_disabled,
            ghost_element_background: theme_data.colors.ghost_element_background,
            ghost_element_hover: theme_data.colors.ghost_element_hover,
            ghost_element_active: theme_data.colors.ghost_element_active,
            ghost_element_selected: theme_data.colors.ghost_element_selected,
            text: theme_data.colors.text,
            text_muted: theme_data.colors.text_muted,
            text_placeholder: theme_data.colors.text_placeholder,
            text_disabled: theme_data.colors.text_disabled,
            text_accent: theme_data.colors.text_accent,
            icon: theme_data.colors.icon,
            icon_muted: theme_data.colors.icon_muted,
            icon_disabled: theme_data.colors.icon_disabled,
            icon_accent: theme_data.colors.icon_accent,
            border: theme_data.colors.border,
            border_variant: theme_data.colors.border_variant,
            border_focused: theme_data.colors.border_focused,
            border_selected: theme_data.colors.border_selected,
            border_disabled: theme_data.colors.border_disabled,
            tab_bar_background: theme_data.colors.tab_bar_background,
            tab_active_background: theme_data.colors.tab_active_background,
            tab_inactive_background: theme_data.colors.tab_inactive_background,
            status_bar_background: theme_data.colors.status_bar_background,
            title_bar_background: theme_data.colors.title_bar_background,
            toolbar_background: theme_data.colors.toolbar_background,
            scrollbar_thumb_background: theme_data.colors.scrollbar_thumb_background,
        },
        fonts: FontSettings {
            ui_font_family: theme_data.fonts.ui_font_family,
            ui_font_size: theme_data.fonts.ui_font_size as f64,
            buffer_font_family: theme_data.fonts.buffer_font_family,
            buffer_font_size: theme_data.fonts.buffer_font_size as f64,
        },
    };

    let signal = Box::new(ZedThemeSignal::ThemeChanged(ThemeChangedEvent {
        theme: craby_theme,
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "themeChanged", signal_ptr);
    }
}

fn emit_workspace_changed_signal(module_id: usize, workspace_info: rn_chat_panel_types::WorkspaceInfo) {
    let manager = get_signal_manager();
    let craby_workspace = WorkspaceInfo {
        project_name: match workspace_info.project_name {
            Some(val) => NullableString { null: false, val },
            None => NullableString { null: true, val: String::new() },
        },
        root_path: match workspace_info.root_path {
            Some(val) => NullableString { null: false, val },
            None => NullableString { null: true, val: String::new() },
        },
        current_file_path: match workspace_info.current_file_path {
            Some(val) => NullableString { null: false, val },
            None => NullableString { null: true, val: String::new() },
        },
    };

    let signal = Box::new(ZedWorkspaceSignal::WorkspaceChanged(WorkspaceChangedEvent {
        workspace: craby_workspace,
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "workspaceChanged", signal_ptr);
    }
}

fn emit_llm_chunk_signal(module_id: usize, request_id: u64, chunk: String) {
    let manager = get_signal_manager();
    let signal = Box::new(ZedLLMSignal::LlmChunk(LLMChunkEvent {
        request_id: request_id as f64,
        chunk,
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "llmChunk", signal_ptr);
    }
}

fn emit_llm_done_signal(module_id: usize, request_id: u64) {
    let manager = get_signal_manager();
    let signal = Box::new(ZedLLMSignal::LlmDone(LLMDoneEvent {
        request_id: request_id as f64,
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "llmDone", signal_ptr);
    }
}

fn emit_llm_error_signal(module_id: usize, request_id: u64, error: String) {
    let manager = get_signal_manager();
    let signal = Box::new(ZedLLMSignal::LlmError(LLMErrorEvent {
        request_id: request_id as f64,
        error,
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "llmError", signal_ptr);
    }
}

fn emit_acp_threads_changed_signal(module_id: usize, threads_json: String) {
    let manager = get_signal_manager();
    let signal = Box::new(ZedLLMSignal::AcpThreadsChanged(AcpThreadsChangedEvent {
        threads: threads_json,
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "acpThreadsChanged", signal_ptr);
    }
}

fn emit_tool_call_event_signal(
    module_id: usize,
    event: rn_chat_panel_types::ToolCallEvent,
) {
    let manager = get_signal_manager();

    // Convert the event to JSON for the JS side to parse
    let event_json = serde_json::to_string(&event).unwrap_or_default();

    let signal = Box::new(ZedLLMSignal::ToolCallEvent(ToolCallEventEvent {
        event: event_json,
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "toolCallEvent", signal_ptr);
    }
}

fn emit_thread_snapshot_signal(
    module_id: usize,
    snapshot: rn_chat_panel_types::ThreadSnapshot,
) {
    let manager = get_signal_manager();
    let event_json = serde_json::to_string(&snapshot).unwrap_or_default();

    let signal = Box::new(ZedLLMSignal::ThreadSnapshot(ThreadSnapshotPayload {
        snapshot: event_json,
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "threadSnapshot", signal_ptr);
    }
}

fn emit_agent_connected_signal(
    module_id: usize,
    agent_type: rn_chat_panel_types::AgentServerType,
    session_id: String,
) {
    let manager = get_signal_manager();
    let signal = Box::new(ZedLLMSignal::AgentConnected(AgentConnectedEvent {
        agent_type: agent_type.to_string(),
        session_id,
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "agentConnected", signal_ptr);
    }
}

fn emit_agent_disconnected_signal(
    module_id: usize,
    agent_type: rn_chat_panel_types::AgentServerType,
    _session_id: String,
) {
    let manager = get_signal_manager();
    let signal = Box::new(ZedLLMSignal::AgentDisconnected(AgentDisconnectedEvent {
        agent_type: agent_type.to_string(),
        reason: "disconnected".to_string(),
    }));
    let signal_ptr = Box::into_raw(signal) as usize;
    unsafe {
        manager.emit(module_id, "agentDisconnected", signal_ptr);
    }
}
