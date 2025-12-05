//! Zed-specific host commands for React Native TurboModules.
//!
//! This module handles command execution on the GPUI app thread.
//! The command bus infrastructure is provided by rn_chat_panel_types.

use acp_thread::{AcpThread, AgentThreadEntry, Diff, Terminal, ToolCall, ToolCallStatus as AcpToolCallStatus};
use agent::{DbThreadMetadata, ThreadsDatabase};
use agent_client_protocol as acp;
use agent_servers::{AgentServer, AgentServerDelegate, ClaudeCode, Codex, Gemini};
use agent_settings::{AgentSettings, builtin_profiles};
use anyhow::anyhow;
use async_channel::{Receiver, Sender, bounded};
use futures::{FutureExt, StreamExt};
use gpui::{App, AsyncApp, Entity, Hsla, WeakEntity, Window};
use language_model::{
    LanguageModelCompletionEvent, LanguageModelRegistry, LanguageModelRequest,
    LanguageModelRequestMessage, MessageContent, Role,
};
use project::Project;
use settings::{Settings, update_settings_file};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use icons::IconName;
use strum::IntoEnumIterator;
use theme::{ActiveTheme, Appearance, GlobalTheme, ThemeSettings};
use workspace::{
    Workspace,
    dock::{DockPosition, Panel as _, UtilityPane as _, UtilityPanePosition},
    utility_pane::utility_slot_for_dock_position,
};

// Re-export types from rn_chat_panel_types for convenience
#[allow(unused_imports)]
pub use rn_chat_panel_types::{
    agent_listeners, emit_agent_connected, emit_agent_disconnected, emit_theme_changed,
    emit_tool_call_event, llm_chunk_emitter,
    llm_done_emitter, llm_error_emitter, llm_requests, send, send_with_reply_blocking,
    set_agent_signal_emitters, set_llm_signal_emitters, set_theme_signal_emitter,
    theme_listeners, thread_snapshot_listeners, tool_call_listeners, workspace_listeners,
    AgentServerInfo, AgentServerType, CommandError, FontSettingsData, LanguageModelInfo,
    LanguageModelProviderInfo, LlmRequestState, PermissionOption, RnCodeHighlight, RnListItem,
    RnListItemType, RnMarkdownElement, RnMarkdownInline, RnAgentModeInfo, RnAgentModelInfo,
    RnParsedMarkdown, RnTextStyle, RnTokenUsage,
    ThemeColorsData, ThemeData, ToolCallEvent, ToolCallInfo, ToolCallStatus, ToolKind,
    WorkspaceInfo, ZedHostCommand,
};

use crate::runtime::{AgentKey, RnChatRuntime};

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Initialize the Zed host command bus.
///
/// Must be called once during rn_chat_panel initialization, after GPUI is ready.
pub fn init(cx: &mut App) {
    // Initialize the command bus and get the receiver
    let Some(receiver) = rn_chat_panel_types::init_bus() else {
        return; // Already initialized
    };

    // Start the command processing loop on the GPUI thread
    cx.spawn(async move |cx: &mut AsyncApp| {
        run_loop(receiver, cx).await;
    })
    .detach();

    // Observe theme changes and emit signals to registered listeners
    cx.observe_global::<GlobalTheme>(|cx| {
        let theme_data = build_theme_data(cx);
        emit_theme_changed(theme_data);
    })
    .detach();
}

async fn run_loop(receiver: Receiver<ZedHostCommand>, cx: &mut AsyncApp) {
    while let Ok(cmd) = receiver.recv().await {
        let result = cx.update(|app| handle_command(cmd, app));

        if let Err(err) = result {
            log::error!("zed_host_command: failed to handle command: {err}");
        }
    }
}

fn resolve_runtime(cx: &App) -> Option<Entity<RnChatRuntime>> {
    let workspace_window = cx.active_window()?.downcast::<Workspace>()?;
    let workspace = workspace_window.read(cx).ok()?;
    let panel = workspace.panel::<crate::RNChatPanel>(cx)?;
    Some(panel.read(cx).runtime().clone())
}

// ============================================================================
// Command Handlers
// ============================================================================

fn handle_command(cmd: ZedHostCommand, cx: &mut App) {
    match cmd {
        ZedHostCommand::GetTheme { reply } => {
            let data = build_theme_data(cx);
            let _ = reply.send_blocking(data);
        }

        ZedHostCommand::GetColor { name, reply } => {
            let color = get_color_by_name(&name, cx);
            let _ = reply.send_blocking(color);
        }

        ZedHostCommand::GetWorkspaceInfo { reply } => {
            let info = build_workspace_info(cx);
            let _ = reply.send_blocking(info);
        }

        ZedHostCommand::IsLlmAvailable { reply } => {
            let registry = LanguageModelRegistry::read_global(cx);
            let available = registry.default_model().is_some();
            let _ = reply.send_blocking(available);
        }

        ZedHostCommand::SendLlmMessage {
            module_id,
            message,
            reply,
        } => {
            let request_id = handle_send_llm_message(module_id, message, cx);
            let _ = reply.send_blocking(request_id);
        }

        ZedHostCommand::CancelLlmRequest { request_id } => {
            handle_cancel_llm_request(request_id);
        }

        // Extended LLM commands
        ZedHostCommand::GetProviders { reply } => {
            let providers = get_providers(cx);
            let _ = reply.send_blocking(providers);
        }

        ZedHostCommand::GetModels { reply } => {
            let models = get_models(cx);
            let _ = reply.send_blocking(models);
        }

        ZedHostCommand::GetDefaultModelId { reply } => {
            let registry = LanguageModelRegistry::read_global(cx);
            let model_id = registry.default_model().map(|m| m.model.id().0.to_string());
            let _ = reply.send_blocking(model_id);
        }

        ZedHostCommand::SendLlmMessageToModel {
            module_id,
            model_id,
            message,
            reply,
        } => {
            let request_id = handle_send_llm_message_to_model(module_id, model_id, message, cx);
            let _ = reply.send_blocking(request_id);
        }

        // Agent server commands
        ZedHostCommand::GetAgentServers { reply } => {
            let servers = get_agent_servers(cx);
            let _ = reply.send_blocking(servers);
        }

        ZedHostCommand::ConnectAgent { agent_type, reply } => {
            let success = handle_connect_agent(agent_type, cx);
            let _ = reply.send_blocking(success);
        }

        ZedHostCommand::DisconnectAgent { agent_type } => {
            handle_disconnect_agent(agent_type, cx);
        }

        ZedHostCommand::SendAgentMessage {
            module_id,
            agent_type,
            message,
            reply,
        } => {
            let request_id = handle_send_agent_message(module_id, agent_type, message, cx);
            let _ = reply.send_blocking(request_id);
        }

        ZedHostCommand::EnsureAgentThread { agent_type, reply } => {
            let ok = handle_ensure_agent_thread(agent_type, cx);
            let _ = reply.send_blocking(ok);
        }
        ZedHostCommand::StartAgentThread { agent_type, reply } => {
            let ok = handle_start_agent_thread(agent_type, cx);
            let _ = reply.send_blocking(ok);
        }

        ZedHostCommand::GetAgentModels { agent_type, reply } => {
            handle_get_agent_models(agent_type, reply, cx);
        }

        ZedHostCommand::SetAgentModel {
            agent_type,
            model_id,
            reply,
        } => {
            handle_set_agent_model(agent_type, model_id, reply, cx);
        }

        ZedHostCommand::GetAgentModes { agent_type, reply } => {
            let modes = get_agent_modes(agent_type, cx);
            let _ = reply.send_blocking(modes);
        }

        ZedHostCommand::SetAgentMode {
            agent_type,
            mode_id,
            reply,
        } => {
            let ok = handle_set_agent_mode(agent_type, mode_id, cx);
            let _ = reply.send_blocking(ok);
        }

        ZedHostCommand::RegisterThemeListener { module_id } => {
            theme_listeners().lock().unwrap().insert(module_id);
        }

        ZedHostCommand::UnregisterThemeListener { module_id } => {
            theme_listeners().lock().unwrap().remove(&module_id);
            log::debug!(
                "[ZedHostCommand] Unregistered theme listener: {}",
                module_id
            );
        }

        ZedHostCommand::RegisterWorkspaceListener { module_id } => {
            workspace_listeners().lock().unwrap().insert(module_id);
            log::debug!(
                "[ZedHostCommand] Registered workspace listener: {}",
                module_id
            );
        }

        ZedHostCommand::UnregisterWorkspaceListener { module_id } => {
            workspace_listeners().lock().unwrap().remove(&module_id);
            log::debug!(
                "[ZedHostCommand] Unregistered workspace listener: {}",
                module_id
            );
        }

        ZedHostCommand::RegisterAgentListener { module_id } => {
            agent_listeners().lock().unwrap().insert(module_id);
            log::debug!(
                "[ZedHostCommand] Registered agent listener: {}",
                module_id
            );
        }

        ZedHostCommand::UnregisterAgentListener { module_id } => {
            agent_listeners().lock().unwrap().remove(&module_id);
            log::debug!(
                "[ZedHostCommand] Unregistered agent listener: {}",
                module_id
            );
        }

        // ACP thread history commands
        ZedHostCommand::ListAcpThreads { reply } => {
            handle_list_acp_threads(reply, cx);
        }

        ZedHostCommand::SearchAcpThreads { query, reply } => {
            handle_search_acp_threads(query, reply, cx);
        }

        ZedHostCommand::OpenAcpThread { thread_id, reply } => {
            let ok = handle_open_acp_thread(thread_id, cx);
            let _ = reply.send_blocking(ok);
        }

        ZedHostCommand::RegisterAcpThreadsListener { module_id } => {
            rn_chat_panel_types::acp_threads_listeners()
                .lock()
                .unwrap()
                .insert(module_id);
            log::debug!(
                "[ZedHostCommand] Registered ACP threads listener: {}",
                module_id
            );
        }

        ZedHostCommand::UnregisterAcpThreadsListener { module_id } => {
            rn_chat_panel_types::acp_threads_listeners()
                .lock()
                .unwrap()
                .remove(&module_id);
            log::debug!(
                "[ZedHostCommand] Unregistered ACP threads listener: {}",
                module_id
            );
        }

        // Tool call commands
        ZedHostCommand::RegisterToolCallListener { module_id } => {
            tool_call_listeners().lock().unwrap().insert(module_id);
            log::debug!(
                "[ZedHostCommand] Registered tool call listener: {}",
                module_id
            );
        }

        ZedHostCommand::UnregisterToolCallListener { module_id } => {
            tool_call_listeners().lock().unwrap().remove(&module_id);
            log::debug!(
                "[ZedHostCommand] Unregistered tool call listener: {}",
                module_id
            );
        }

        ZedHostCommand::RegisterThreadSnapshotListener { module_id } => {
            thread_snapshot_listeners().lock().unwrap().insert(module_id);
            log::debug!(
                "[ZedHostCommand] Registered thread snapshot listener: {}",
                module_id
            );
        }

        ZedHostCommand::UnregisterThreadSnapshotListener { module_id } => {
            thread_snapshot_listeners().lock().unwrap().remove(&module_id);
            log::debug!(
                "[ZedHostCommand] Unregistered thread snapshot listener: {}",
                module_id
            );
        }

        ZedHostCommand::RespondToToolAuthorization {
            tool_call_id,
            permission_option_id,
            reply,
        } => {
            let success =
                handle_tool_authorization_response(&tool_call_id, &permission_option_id, cx);
            let _ = reply.send_blocking(success);
        }

        ZedHostCommand::CancelToolCall { tool_call_id, reply } => {
            let success = handle_cancel_tool_call(&tool_call_id, cx);
            let _ = reply.send_blocking(success);
        }

        // Markdown parsing
        ZedHostCommand::ParseMarkdown { markdown, reply } => {
            let parsed = parse_markdown_to_rn(&markdown);
            let _ = reply.send_blocking(parsed);
        }

        // UI commands
        ZedHostCommand::SetChatPaneVisible { visible } => {
            set_chat_pane_visible(visible, cx);
        }

        ZedHostCommand::MinimizeChatPane => {
            minimize_chat_pane(cx);
        }

        // Icon commands
        ZedHostCommand::GetIconSvg { name, reply } => {
            let svg = get_icon_svg(&name, cx);
            reply.send_blocking(svg).ok();
        }

        ZedHostCommand::ListIcons { reply } => {
            let icons = list_icons();
            reply.send_blocking(icons).ok();
        }
    }
}

// ============================================================================
// Theme Helpers
// ============================================================================

fn set_chat_pane_visible(visible: bool, cx: &mut App) {
    let Some(workspace_window) = cx.active_window().and_then(|w| w.downcast::<Workspace>()) else {
        return;
    };

    let _ = workspace_window.update(cx, |workspace, window, cx| {
        let Some(panel) = workspace.panel::<crate::RNChatPanel>(cx) else {
            return;
        };

        let panel_id = panel.entity_id();
        let dock_position = panel.read(cx).position(window, cx);
        let slot = utility_slot_for_dock_position(dock_position);

        let pane_position = match dock_position {
            DockPosition::Left => UtilityPanePosition::Right,
            DockPosition::Right => UtilityPanePosition::Left,
            DockPosition::Bottom => UtilityPanePosition::Right,
        };

        if visible {
            let pane = panel.update(cx, |panel, cx| panel.ensure_chat_pane(pane_position, cx));
            pane.update(cx, |pane, cx| pane.set_expanded(true, cx));
            workspace.register_utility_pane(slot, panel_id, pane, cx);
        } else {
            if let Some(pane) = panel.read(cx).existing_chat_pane() {
                pane.update(cx, |pane, cx| pane.set_expanded(false, cx));
            }

            workspace.clear_utility_pane_if_provider(slot, panel_id, cx);
        }
    });
}

fn minimize_chat_pane(cx: &mut App) {
    use workspace::dock::MinimizePane;

    let Some(workspace_window) = cx.active_window().and_then(|w| w.downcast::<Workspace>()) else {
        return;
    };

    let _ = workspace_window.update(cx, |workspace, _window, cx| {
        let Some(panel) = workspace.panel::<crate::RNChatPanel>(cx) else {
            return;
        };

        if let Some(pane) = panel.read(cx).existing_chat_pane() {
            pane.update(cx, |pane, cx| {
                cx.emit(MinimizePane);
                pane.set_expanded(false, cx);
            });
        }
    });
}

// ============================================================================
// Icon Helpers
// ============================================================================

fn get_icon_svg(name: &str, cx: &App) -> String {
    let icon_name: Result<IconName, _> = name.parse();
    let Ok(icon_name) = icon_name else {
        log::warn!("get_icon_svg: invalid icon name: {name}");
        return String::new();
    };

    let path = icon_name.path();
    let assets = cx.asset_source();

    match assets.load(&path) {
        Ok(Some(cow)) => {
            let svg = String::from_utf8_lossy(&cow).into_owned();
            let svg = ensure_svg_view_box(svg);
            log::debug!("get_icon_svg: loaded {name} ({} bytes)", svg.len());
            svg
        }
        Ok(None) => {
            log::warn!("get_icon_svg: asset not found: {path}");
            String::new()
        }
        Err(e) => {
            log::error!("get_icon_svg: failed to load {path}: {e}");
            String::new()
        }
    }
}

fn ensure_svg_view_box(svg: String) -> String {
    let start = match svg.find("<svg") {
        Some(i) => i,
        None => return svg,
    };

    let tag_end = match svg[start..].find('>') {
        Some(rel) => start + rel,
        None => return svg,
    };

    let tag = &svg[start..=tag_end];
    if tag.to_ascii_lowercase().contains("viewbox=") {
        return svg;
    }

    let Some(width) = find_xml_attr(tag, "width").and_then(parse_svg_number) else {
        return svg;
    };
    let Some(height) = find_xml_attr(tag, "height").and_then(parse_svg_number) else {
        return svg;
    };
    if width <= 0.0 || height <= 0.0 {
        return svg;
    }

    let width = format_svg_number(width);
    let height = format_svg_number(height);
    let view_box = format!(" viewBox=\"0 0 {width} {height}\"");

    let insert_at = start + "<svg".len();
    let mut out = String::with_capacity(svg.len() + view_box.len());
    out.push_str(&svg[..insert_at]);
    out.push_str(&view_box);
    out.push_str(&svg[insert_at..]);
    out
}

fn find_xml_attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("{name}=");
    let idx = tag.find(&needle)?;
    let mut i = idx + needle.len();

    while tag.as_bytes().get(i).is_some_and(|b| b.is_ascii_whitespace()) {
        i += 1;
    }

    let quote = *tag.as_bytes().get(i)?;
    if quote != b'"' && quote != b'\'' {
        return None;
    }
    i += 1;

    let end_rel = tag[i..].find(char::from(quote))?;
    Some(&tag[i..i + end_rel])
}

fn parse_svg_number(value: &str) -> Option<f32> {
    let trimmed = value.trim();
    let number: String = trimmed
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    if number.is_empty() {
        return None;
    }
    number.parse::<f32>().ok()
}

fn format_svg_number(value: f32) -> String {
    if (value.fract()).abs() < f32::EPSILON {
        format!("{}", value as i32)
    } else {
        let mut s = value.to_string();
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
    }
}

fn list_icons() -> Vec<String> {
    IconName::iter().map(|i| {
        let s: &'static str = i.into();
        s.to_string()
    }).collect()
}

fn build_theme_data(cx: &App) -> ThemeData {
    let theme = cx.theme();
    let colors = theme.colors();
    let settings = ThemeSettings::get_global(cx);

    ThemeData {
        name: theme.name.to_string(),
        appearance: match theme.appearance {
            Appearance::Light => "light".to_string(),
            Appearance::Dark => "dark".to_string(),
        },
        colors: ThemeColorsData {
            // Backgrounds
            background: hsla_to_hex(colors.background),
            surface_background: hsla_to_hex(colors.surface_background),
            elevated_surface_background: hsla_to_hex(colors.elevated_surface_background),
            panel_background: hsla_to_hex(colors.panel_background),

            // Element states
            element_background: hsla_to_hex(colors.element_background),
            element_hover: hsla_to_hex(colors.element_hover),
            element_active: hsla_to_hex(colors.element_active),
            element_selected: hsla_to_hex(colors.element_selected),
            element_disabled: hsla_to_hex(colors.element_disabled),

            // Ghost element states
            ghost_element_background: hsla_to_hex(colors.ghost_element_background),
            ghost_element_hover: hsla_to_hex(colors.ghost_element_hover),
            ghost_element_active: hsla_to_hex(colors.ghost_element_active),
            ghost_element_selected: hsla_to_hex(colors.ghost_element_selected),

            // Text
            text: hsla_to_hex(colors.text),
            text_muted: hsla_to_hex(colors.text_muted),
            text_placeholder: hsla_to_hex(colors.text_placeholder),
            text_disabled: hsla_to_hex(colors.text_disabled),
            text_accent: hsla_to_hex(colors.text_accent),

            // Icons
            icon: hsla_to_hex(colors.icon),
            icon_muted: hsla_to_hex(colors.icon_muted),
            icon_disabled: hsla_to_hex(colors.icon_disabled),
            icon_accent: hsla_to_hex(colors.icon_accent),

            // Borders
            border: hsla_to_hex(colors.border),
            border_variant: hsla_to_hex(colors.border_variant),
            border_focused: hsla_to_hex(colors.border_focused),
            border_selected: hsla_to_hex(colors.border_selected),
            border_disabled: hsla_to_hex(colors.border_disabled),

            // UI elements
            tab_bar_background: hsla_to_hex(colors.tab_bar_background),
            tab_active_background: hsla_to_hex(colors.tab_active_background),
            tab_inactive_background: hsla_to_hex(colors.tab_inactive_background),
            status_bar_background: hsla_to_hex(colors.status_bar_background),
            title_bar_background: hsla_to_hex(colors.title_bar_background),
            toolbar_background: hsla_to_hex(colors.toolbar_background),
            scrollbar_thumb_background: hsla_to_hex(colors.scrollbar_thumb_background),
        },
        fonts: FontSettingsData {
            ui_font_family: settings.ui_font.family.to_string(),
            ui_font_size: settings.agent_ui_font_size(cx).into(),
            buffer_font_family: settings.buffer_font.family.to_string(),
            buffer_font_size: settings.agent_buffer_font_size(cx).into(),
        },
    }
}

fn get_color_by_name(name: &str, cx: &App) -> Option<String> {
    let colors = cx.theme().colors();

    let color = match name {
        "background" => colors.background,
        "surfaceBackground" | "surface_background" => colors.surface_background,
        "elevatedSurfaceBackground" | "elevated_surface_background" => {
            colors.elevated_surface_background
        }
        "panelBackground" | "panel_background" => colors.panel_background,
        "elementBackground" | "element_background" => colors.element_background,
        "elementHover" | "element_hover" => colors.element_hover,
        "elementActive" | "element_active" => colors.element_active,
        "elementSelected" | "element_selected" => colors.element_selected,
        "elementDisabled" | "element_disabled" => colors.element_disabled,
        "ghostElementBackground" | "ghost_element_background" => colors.ghost_element_background,
        "ghostElementHover" | "ghost_element_hover" => colors.ghost_element_hover,
        "ghostElementActive" | "ghost_element_active" => colors.ghost_element_active,
        "ghostElementSelected" | "ghost_element_selected" => colors.ghost_element_selected,
        "text" => colors.text,
        "textMuted" | "text_muted" => colors.text_muted,
        "textPlaceholder" | "text_placeholder" => colors.text_placeholder,
        "textDisabled" | "text_disabled" => colors.text_disabled,
        "textAccent" | "text_accent" => colors.text_accent,
        "icon" => colors.icon,
        "iconMuted" | "icon_muted" => colors.icon_muted,
        "iconDisabled" | "icon_disabled" => colors.icon_disabled,
        "iconAccent" | "icon_accent" => colors.icon_accent,
        "border" => colors.border,
        "borderVariant" | "border_variant" => colors.border_variant,
        "borderFocused" | "border_focused" => colors.border_focused,
        "borderSelected" | "border_selected" => colors.border_selected,
        "borderDisabled" | "border_disabled" => colors.border_disabled,
        _ => return None,
    };

    Some(hsla_to_hex(color))
}

/// Convert HSLA color to hex string (#RRGGBBAA).
fn hsla_to_hex(color: Hsla) -> String {
    let rgba = color.to_rgb();
    format!(
        "#{:02x}{:02x}{:02x}{:02x}",
        (rgba.r * 255.0) as u8,
        (rgba.g * 255.0) as u8,
        (rgba.b * 255.0) as u8,
        (rgba.a * 255.0) as u8
    )
}

// ============================================================================
// Workspace Helpers
// ============================================================================

fn build_workspace_info(cx: &App) -> WorkspaceInfo {
    let Some(workspace_window) = cx.active_window().and_then(|window| window.downcast::<Workspace>())
    else {
        return WorkspaceInfo::default();
    };

    let workspace = match workspace_window.read(cx) {
        Ok(workspace) => workspace,
        Err(_) => return WorkspaceInfo::default(),
    };

    let project_entity = workspace.project().clone();
    let project = project_entity.read(cx);

    let project_name = project.worktrees(cx).next().map(|wt| {
        let name = wt.read(cx).root_name();
        name.as_unix_str().to_string()
    });

    let root_path = project.worktrees(cx).next().and_then(|wt| {
        wt.read(cx).abs_path().to_str().map(|s| s.to_string())
    });

    let current_file_path = workspace.active_item(cx).and_then(|item| {
        item.project_path(cx).map(|pp| pp.path.as_unix_str().to_string())
    });

    WorkspaceInfo {
        project_name,
        root_path,
        current_file_path,
    }
}

// ============================================================================
// LLM Helpers
// ============================================================================

fn handle_send_llm_message(module_id: usize, message: String, cx: &mut App) -> u64 {
    let request_id = REQUEST_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);

    log::info!(
        "[ZedHostCommand] SendLlmMessage request_id={} module_id={} message={}",
        request_id,
        module_id,
        &message[..message.len().min(100)]
    );

    // Get the default model from the registry
    let registry = LanguageModelRegistry::read_global(cx);
    let Some(configured_model) = registry.default_model() else {
        // No model configured - emit error signal
        if let Some(error_emitter) = llm_error_emitter() {
            error_emitter(module_id, request_id, "No language model configured".to_string());
        }
        return request_id;
    };

    let model = configured_model.model.clone();

    // Create cancellation channel
    let (cancel_tx, cancel_rx) = bounded::<()>(1);

    // Store request state for cancellation
    llm_requests().lock().unwrap().insert(
        request_id,
        LlmRequestState {
            module_id,
            cancel_tx,
        },
    );

    // Build the LLM request
    let request = LanguageModelRequest {
        messages: vec![LanguageModelRequestMessage {
            role: Role::User,
            content: vec![MessageContent::Text(message)],
            cache: false,
            reasoning_details: None,
        }],
        ..Default::default()
    };

    // Spawn async task to handle streaming
    cx.spawn(async move |cx| {
        let stream_result = model.stream_completion(request, cx).await;

        match stream_result {
            Ok(mut events) => {
                loop {
                    futures::select_biased! {
                        _ = cancel_rx.recv().fuse() => {
                            log::info!("[ZedHostCommand] LLM request {} cancelled", request_id);
                            break;
                        }
                        event = events.next().fuse() => {
                            match event {
                                Some(Ok(LanguageModelCompletionEvent::Text(text))) => {
                                    if let Some(chunk_emitter) = llm_chunk_emitter() {
                                        chunk_emitter(module_id, request_id, text);
                                    }
                                }
                                Some(Ok(LanguageModelCompletionEvent::Stop(_))) => {
                                    if let Some(done_emitter) = llm_done_emitter() {
                                        done_emitter(module_id, request_id);
                                    }
                                    break;
                                }
                                Some(Ok(_)) => {
                                    // Ignore other event types
                                }
                                Some(Err(err)) => {
                                    log::error!("[ZedHostCommand] LLM streaming error: {}", err);
                                    if let Some(error_emitter) = llm_error_emitter() {
                                        error_emitter(module_id, request_id, err.to_string());
                                    }
                                    break;
                                }
                                None => {
                                    if let Some(done_emitter) = llm_done_emitter() {
                                        done_emitter(module_id, request_id);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            Err(err) => {
                log::error!("[ZedHostCommand] Failed to start LLM stream: {}", err);
                if let Some(error_emitter) = llm_error_emitter() {
                    error_emitter(module_id, request_id, err.to_string());
                }
            }
        }

        // Clean up request state
        llm_requests().lock().unwrap().remove(&request_id);
    })
    .detach();

    request_id
}

fn handle_cancel_llm_request(request_id: u64) {
    log::info!("[ZedHostCommand] CancelLlmRequest request_id={}", request_id);

    if let Some(state) = llm_requests().lock().unwrap().remove(&request_id) {
        let _ = state.cancel_tx.send_blocking(());
    }
}

// ============================================================================
// Extended LLM Helpers
// ============================================================================

fn get_providers(cx: &App) -> Vec<LanguageModelProviderInfo> {
    let registry = LanguageModelRegistry::read_global(cx);
    registry
        .providers()
        .iter()
        .map(|provider| LanguageModelProviderInfo {
            id: provider.id().0.to_string(),
            name: provider.name().0.to_string(),
            is_authenticated: provider.is_authenticated(cx),
        })
        .collect()
}

fn get_models(cx: &App) -> Vec<LanguageModelInfo> {
    let registry = LanguageModelRegistry::read_global(cx);
    registry
        .available_models(cx)
        .map(|model| LanguageModelInfo {
            id: model.id().0.to_string(),
            name: model.name().0.to_string(),
            provider_id: model.provider_id().0.to_string(),
            provider_name: model.provider_name().0.to_string(),
            supports_images: model.supports_images(),
            supports_tools: model.supports_tools(),
        })
        .collect()
}

fn handle_send_llm_message_to_model(
    module_id: usize,
    model_id: String,
    message: String,
    cx: &mut App,
) -> u64 {
    let request_id = REQUEST_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);

    log::info!(
        "[ZedHostCommand] SendLlmMessageToModel request_id={} model_id={} message={}",
        request_id,
        model_id,
        &message[..message.len().min(100)]
    );

    // Find the model by ID
    let registry = LanguageModelRegistry::read_global(cx);
    let model = registry
        .available_models(cx)
        .find(|m| m.id().0.to_string() == model_id);

    let Some(model) = model else {
        if let Some(error_emitter) = llm_error_emitter() {
            error_emitter(
                module_id,
                request_id,
                format!("Model '{}' not found or not authenticated", model_id),
            );
        }
        return request_id;
    };

    // Create cancellation channel
    let (cancel_tx, cancel_rx) = bounded::<()>(1);

    // Store request state for cancellation
    llm_requests().lock().unwrap().insert(
        request_id,
        LlmRequestState {
            module_id,
            cancel_tx,
        },
    );

    // Build the LLM request
    let request = LanguageModelRequest {
        messages: vec![LanguageModelRequestMessage {
            role: Role::User,
            content: vec![MessageContent::Text(message)],
            cache: false,
            reasoning_details: None,
        }],
        ..Default::default()
    };

    // Spawn async task to handle streaming
    cx.spawn(async move |cx| {
        let stream_result = model.stream_completion(request, cx).await;

        match stream_result {
            Ok(mut events) => {
                loop {
                    futures::select_biased! {
                        _ = cancel_rx.recv().fuse() => {
                            log::info!("[ZedHostCommand] LLM request {} cancelled", request_id);
                            break;
                        }
                        event = events.next().fuse() => {
                            match event {
                                Some(Ok(LanguageModelCompletionEvent::Text(text))) => {
                                    if let Some(chunk_emitter) = llm_chunk_emitter() {
                                        chunk_emitter(module_id, request_id, text);
                                    }
                                }
                                Some(Ok(LanguageModelCompletionEvent::Stop(_))) => {
                                    if let Some(done_emitter) = llm_done_emitter() {
                                        done_emitter(module_id, request_id);
                                    }
                                    break;
                                }
                                Some(Ok(_)) => {
                                    // Ignore other event types
                                }
                                Some(Err(err)) => {
                                    log::error!("[ZedHostCommand] LLM streaming error: {}", err);
                                    if let Some(error_emitter) = llm_error_emitter() {
                                        error_emitter(module_id, request_id, err.to_string());
                                    }
                                    break;
                                }
                                None => {
                                    if let Some(done_emitter) = llm_done_emitter() {
                                        done_emitter(module_id, request_id);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            Err(err) => {
                log::error!("[ZedHostCommand] Failed to start LLM stream: {}", err);
                if let Some(error_emitter) = llm_error_emitter() {
                    error_emitter(module_id, request_id, err.to_string());
                }
            }
        }

        // Clean up request state
        llm_requests().lock().unwrap().remove(&request_id);
    })
    .detach();

    request_id
}

// ============================================================================
// Agent Server Helpers
// ============================================================================

fn get_agent_servers(cx: &App) -> Vec<AgentServerInfo> {
    let runtime = resolve_runtime(cx);
    let is_connected = |agent_type: AgentServerType| {
        runtime
            .as_ref()
            .is_some_and(|runtime| runtime.read(cx).is_connected(&AgentKey::from(agent_type)))
    };

    vec![
        AgentServerInfo {
            agent_type: AgentServerType::Codex,
            name: "Codex".to_string(),
            is_connected: is_connected(AgentServerType::Codex),
            custom_name: None,
        },
        AgentServerInfo {
            agent_type: AgentServerType::Claude,
            name: "Claude Code".to_string(),
            is_connected: is_connected(AgentServerType::Claude),
            custom_name: None,
        },
        AgentServerInfo {
            agent_type: AgentServerType::Gemini,
            name: "Gemini CLI".to_string(),
            is_connected: is_connected(AgentServerType::Gemini),
            custom_name: None,
        },
        AgentServerInfo {
            agent_type: AgentServerType::Custom,
            name: "Zed Agent".to_string(),
            is_connected: is_connected(AgentServerType::Custom),
            custom_name: None,
        },
    ]
}

fn server_for_agent_type(agent_type: &AgentServerType) -> Option<Rc<dyn AgentServer>> {
    match agent_type {
        AgentServerType::Codex => Some(Rc::new(Codex) as Rc<dyn AgentServer>),
        AgentServerType::Claude => Some(Rc::new(ClaudeCode) as Rc<dyn AgentServer>),
        AgentServerType::Gemini => Some(Rc::new(Gemini) as Rc<dyn AgentServer>),
        AgentServerType::Custom => None,
    }
}

fn handle_connect_agent(agent_type: AgentServerType, cx: &mut App) -> bool {
    log::info!("[ZedHostCommand] ConnectAgent {:?}", agent_type);

    let Some(runtime) = resolve_runtime(cx) else {
        log::warn!("[ZedHostCommand] No runtime available for agent connection");
        return false;
    };

    let agent_key = AgentKey::from(agent_type.clone());
    if runtime.read(cx).is_connected(&agent_key) {
        log::info!("[ZedHostCommand] Agent {:?} already connected", agent_type);
        return true;
    }

    match agent_type {
        AgentServerType::Custom => {
            // "Custom" is our internal native agent using Zed's LLM infrastructure
            let registry = LanguageModelRegistry::read_global(cx);
            if registry.default_model().is_none() {
                log::warn!("[ZedHostCommand] No language model configured for native agent");
                return false;
            }

            let Some(workspace_window) =
                cx.active_window().and_then(|window| window.downcast::<Workspace>())
            else {
                return false;
            };

            let result = workspace_window
                .update(cx, |workspace, window, cx| {
                    let Some(panel) = workspace.panel::<crate::RNChatPanel>(cx) else {
                        return Err(anyhow!("RNChatPanel not available"));
                    };
                    panel.update(cx, |panel, cx| {
                        panel.ensure_native_agent(window, cx);
                    });
                    Ok(())
                })
                .and_then(|r| r);

            result.is_ok()
        }
        _ => {
            let Some(project) = runtime.read(cx).project() else {
                return false;
            };

            let Some(server) = server_for_agent_type(&agent_type) else {
                return false;
            };

            let store = project.read(cx).agent_server_store().clone();
            let root_dir = project
                .read(cx)
                .worktrees(cx)
                .next()
                .map(|wt| wt.read(cx).abs_path().to_path_buf())
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from("/"));

            let delegate = AgentServerDelegate::new(store, project.clone(), None, None);
            let connect_task = server.connect(Some(root_dir.as_path()), delegate, cx);
            let runtime = runtime.clone();
            let agent_key = AgentKey::from(agent_type.clone());

            cx.spawn(async move |cx| {
                let (conn, _login) = match connect_task.await {
                    Ok(ok) => ok,
                    Err(err) => {
                        log::error!(
                            "[ZedHostCommand] Failed to connect agent {:?}: {err}",
                            agent_type
                        );
                        return;
                    }
                };

                let _ = cx.update(|app| {
                    runtime.update(app, |runtime, cx| {
                        runtime.set_connection(agent_key, conn, cx);
                    });
                });
            })
            .detach();

            true
        }
    }
}

fn handle_disconnect_agent(agent_type: AgentServerType, cx: &mut App) {
    log::info!("[ZedHostCommand] DisconnectAgent {:?}", agent_type);
    if let Some(runtime) = resolve_runtime(cx) {
        runtime.update(cx, |runtime, cx| {
            runtime.remove_connection(&AgentKey::from(agent_type), cx);
        });
    }
}

fn handle_send_agent_message(
    module_id: usize,
    agent_type: AgentServerType,
    message: String,
    cx: &mut App,
) -> u64 {
    let request_id = REQUEST_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);

    log::info!(
        "[ZedHostCommand] SendAgentMessage request_id={} agent={:?} message={}",
        request_id,
        agent_type,
        &message[..message.len().min(100)]
    );

    match agent_type {
        AgentServerType::Custom => {
            if !handle_connect_agent(AgentServerType::Custom, cx) {
                if let Some(error_emitter) = llm_error_emitter() {
                    error_emitter(
                        module_id,
                        request_id,
                        "No language model configured for Zed Agent".to_string(),
                    );
                }
                return request_id;
            }

            let Some(workspace_window) =
                cx.active_window().and_then(|window| window.downcast::<Workspace>())
            else {
                log::warn!("[ZedHostCommand] No active workspace window");
                if let Some(error_emitter) = llm_error_emitter() {
                    error_emitter(
                        module_id,
                        request_id,
                        "No active workspace window".to_string(),
                    );
                }
                return request_id;
            };

            let result = workspace_window
                .update(cx, |workspace, window, cx| {
                    let Some(panel) = workspace.panel::<crate::RNChatPanel>(cx) else {
                        return Err(anyhow!("RNChatPanel not available"));
                    };

                    panel.update(cx, |panel, cx| {
                        panel.ensure_native_agent(window, cx);
                        panel.send_message(message.clone(), cx)
                    })
                })
                .and_then(|result| result);

            if let Err(err) = result {
                log::error!("[ZedHostCommand] Failed to send agent message: {err}");
                if let Some(error_emitter) = llm_error_emitter() {
                    error_emitter(
                        module_id,
                        request_id,
                        format!("Failed to send agent message: {err}"),
                    );
                }
            }

            request_id
        }
        _ => {
            let Some(runtime) = resolve_runtime(cx) else {
                if let Some(error_emitter) = llm_error_emitter() {
                    error_emitter(
                        module_id,
                        request_id,
                        "Failed to resolve runtime".to_string(),
                    );
                }
                return request_id;
            };

            let agent_key = AgentKey::from(agent_type.clone());
            if let Some(thread) = runtime.read(cx).active_thread(&agent_key) {
                let send_task =
                    thread.update(cx, |thread, cx| thread.send(vec![message.into()], cx));
                cx.background_executor()
                    .spawn(async move {
                        if let Err(err) = send_task.await {
                            log::error!("rn_chat_panel: Failed to send message: {err}");
                        }
                    })
                    .detach();
                return request_id;
            }

            let Some(project) = runtime.read(cx).project() else {
                if let Some(error_emitter) = llm_error_emitter() {
                    error_emitter(
                        module_id,
                        request_id,
                        "Failed to resolve project".to_string(),
                    );
                }
                return request_id;
            };

            let store = project.read(cx).agent_server_store().clone();
            let root_dir = project
                .read(cx)
                .worktrees(cx)
                .next()
                .map(|wt| wt.read(cx).abs_path().to_path_buf())
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from("/"));

            let maybe_connection = runtime.read(cx).connection(&agent_key);
            let connect_task = if maybe_connection.is_none() {
                let Some(server) = server_for_agent_type(&agent_type) else {
                    if let Some(error_emitter) = llm_error_emitter() {
                        error_emitter(
                            module_id,
                            request_id,
                            format!("Unsupported agent type: {agent_type:?}"),
                        );
                    }
                    return request_id;
                };

                let delegate = AgentServerDelegate::new(store, project.clone(), None, None);
                Some(server.connect(Some(root_dir.as_path()), delegate, cx))
            } else {
                None
            };

            let runtime = runtime.clone();
            let project = project.clone();
            let message = message.clone();

            cx.spawn(async move |cx| {
                let connection = match maybe_connection {
                    Some(conn) => conn,
                    None => {
                        let Some(connect_task) = connect_task else {
                            return;
                        };
                        let (conn, _login) = match connect_task.await {
                            Ok(ok) => ok,
                            Err(err) => {
                                log::error!(
                                    "[ZedHostCommand] Failed to connect agent {:?}: {err}",
                                    agent_type
                                );
                                if let Some(error_emitter) = llm_error_emitter() {
                                    error_emitter(module_id, request_id, err.to_string());
                                }
                                return;
                            }
                        };

                        let _ = cx.update(|app| {
                            runtime.update(app, |runtime, cx| {
                                runtime.set_connection(agent_key.clone(), conn.clone(), cx);
                            });
                        });

                        conn
                    }
                };

                let thread_task = match cx.update(|app| {
                    connection.clone().new_thread(project.clone(), root_dir.as_path(), app)
                }) {
                    Ok(task) => task,
                    Err(err) => {
                        if let Some(error_emitter) = llm_error_emitter() {
                            error_emitter(module_id, request_id, err.to_string());
                        }
                        return;
                    }
                };
                let thread: Entity<AcpThread> = match thread_task.await {
                    Ok(thread) => thread,
                    Err(err) => {
                        if let Some(error_emitter) = llm_error_emitter() {
                            error_emitter(module_id, request_id, err.to_string());
                        }
                        return;
                    }
                };

                let _ = cx.update(|app| {
                    runtime.update(app, |runtime, cx| {
                        runtime.set_active_thread(agent_key.clone(), thread.clone(), cx);
                    })
                });

                let send_task = match cx.update(|app| {
                    thread.update(app, |thread, cx| thread.send(vec![message.into()], cx))
                }) {
                    Ok(task) => task,
                    Err(err) => {
                        if let Some(error_emitter) = llm_error_emitter() {
                            error_emitter(module_id, request_id, err.to_string());
                        }
                        return;
                    }
                };
                if let Err(err) = send_task.await {
                    log::error!("rn_chat_panel: Failed to send message: {err}");
                }
            })
            .detach();

            request_id
        }
    }
}

// ============================================================================
// Agent Configuration Helpers (Models / Modes)
// ============================================================================

fn handle_ensure_agent_thread(agent_type: AgentServerType, cx: &mut App) -> bool {
    match agent_type {
        AgentServerType::Custom => {
            if !handle_connect_agent(AgentServerType::Custom, cx) {
                return false;
            }

            let Some(workspace_window) =
                cx.active_window().and_then(|window| window.downcast::<Workspace>())
            else {
                return false;
            };

            let result = workspace_window
                .update(cx, |workspace, window, cx| {
                    let Some(panel) = workspace.panel::<crate::RNChatPanel>(cx) else {
                        return Err(anyhow!("RNChatPanel not available"));
                    };

                    panel.update(cx, |panel, cx| {
                        panel.ensure_native_agent(window, cx);

                        if panel
                            .runtime()
                            .read(cx)
                            .active_thread(&AgentKey::from(AgentServerType::Custom))
                            .is_some()
                        {
                            return Ok(());
                        }

                        if panel.has_native_agent() {
                            panel.start_new_thread(cx)?;
                        }

                        Ok(())
                    })
                })
                .and_then(|result| result);

            result.is_ok()
        }
        _ => handle_connect_agent(agent_type, cx),
    }
}

fn handle_start_agent_thread(agent_type: AgentServerType, cx: &mut App) -> bool {
    match agent_type {
        AgentServerType::Custom => {
            if !handle_connect_agent(AgentServerType::Custom, cx) {
                return false;
            }

            let Some(workspace_window) =
                cx.active_window().and_then(|window| window.downcast::<Workspace>())
            else {
                return false;
            };

            let result = workspace_window
                .update(cx, |workspace, window, cx| {
                    let Some(panel) = workspace.panel::<crate::RNChatPanel>(cx) else {
                        return Err(anyhow!("RNChatPanel not available"));
                    };

                    panel.update(cx, |panel, cx| {
                        panel.ensure_native_agent(window, cx);
                        panel.start_new_thread(cx)
                    })
                })
                .and_then(|result| result);

            result.is_ok()
        }
        _ => {
            let Some(runtime) = resolve_runtime(cx) else {
                return false;
            };

            let Some(project) = runtime.read(cx).project() else {
                return false;
            };

            let Some(server) = server_for_agent_type(&agent_type) else {
                return false;
            };

            let store = project.read(cx).agent_server_store().clone();
            let root_dir = project
                .read(cx)
                .worktrees(cx)
                .next()
                .map(|wt| wt.read(cx).abs_path().to_path_buf())
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from("/"));

            let agent_key = AgentKey::from(agent_type.clone());
            let maybe_connection = runtime.read(cx).connection(&agent_key);
            let connect_task = if maybe_connection.is_none() {
                let delegate = AgentServerDelegate::new(store, project.clone(), None, None);
                Some(server.connect(Some(root_dir.as_path()), delegate, cx))
            } else {
                None
            };

            let runtime = runtime.clone();
            let project = project.clone();

            cx.spawn(async move |cx| {
                let connection = match maybe_connection {
                    Some(conn) => conn,
                    None => {
                        let (conn, _login) = match connect_task {
                            Some(task) => match task.await {
                                Ok(ok) => ok,
                                Err(err) => {
                                    log::error!(
                                        "[ZedHostCommand] Failed to connect agent {:?}: {err}",
                                        agent_type
                                    );
                                    return;
                                }
                            },
                            None => return,
                        };

                        let _ = cx.update(|app| {
                            runtime.update(app, |runtime, cx| {
                                runtime.set_connection(agent_key.clone(), conn.clone(), cx);
                            })
                        });

                        conn
                    }
                };

                let thread_task = match cx.update(|app| {
                    connection
                        .clone()
                        .new_thread(project.clone(), root_dir.as_path(), app)
                }) {
                    Ok(task) => task,
                    Err(err) => {
                        log::error!(
                            "[ZedHostCommand] Failed to create thread for {:?}: {err}",
                            agent_type
                        );
                        return;
                    }
                };

                let thread = match thread_task.await {
                    Ok(thread) => thread,
                    Err(err) => {
                        log::error!(
                            "[ZedHostCommand] Failed to start thread for {:?}: {err}",
                            agent_type
                        );
                        return;
                    }
                };

                let _ = cx.update(|app| {
                    runtime.update(app, |runtime, cx| {
                        runtime.set_active_thread(agent_key.clone(), thread, cx);
                    })
                });
            })
            .detach();

            true
        }
    }
}

fn handle_get_agent_models(agent_type: AgentServerType, reply: Sender<Vec<RnAgentModelInfo>>, cx: &mut App) {
    match agent_type {
        AgentServerType::Custom => {
            let models = get_custom_agent_models(cx);
            let _ = reply.send_blocking(models);
        }
        _ => {
            let Some(runtime) = resolve_runtime(cx) else {
                let _ = reply.send_blocking(Vec::new());
                return;
            };

            let agent_key = AgentKey::from(agent_type.clone());
            let Some(connection) = runtime.read(cx).connection(&agent_key) else {
                let _ = reply.send_blocking(Vec::new());
                return;
            };

            let Some(project) = runtime.read(cx).project() else {
                let _ = reply.send_blocking(Vec::new());
                return;
            };

            let root_dir = project
                .read(cx)
                .worktrees(cx)
                .next()
                .map(|wt| wt.read(cx).abs_path().to_path_buf())
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from("/"));

            let existing_thread = runtime.read(cx).active_thread(&agent_key);
            let runtime = runtime.clone();

            cx.spawn(async move |cx| {
                let thread = match existing_thread {
                    Some(thread) => thread,
                    None => {
                        let thread_task = match cx.update(|app| {
                            connection
                                .clone()
                                .new_thread(project.clone(), root_dir.as_path(), app)
                        }) {
                            Ok(task) => task,
                            Err(_) => {
                                let _ = reply.send_blocking(Vec::new());
                                return;
                            }
                        };

                        let thread: Entity<AcpThread> = match thread_task.await {
                            Ok(thread) => thread,
                            Err(_) => {
                                let _ = reply.send_blocking(Vec::new());
                                return;
                            }
                        };

                        let _ = cx.update(|app| {
                            runtime.update(app, |runtime, cx| {
                                runtime.set_active_thread(agent_key.clone(), thread.clone(), cx);
                            });
                        });

                        thread
                    }
                };

                let session_id = match cx.update(|app| thread.read(app).session_id().clone()) {
                    Ok(id) => id,
                    Err(_) => {
                        let _ = reply.send_blocking(Vec::new());
                        return;
                    }
                };

                let Some(selector) = connection.model_selector(&session_id) else {
                    let _ = reply.send_blocking(Vec::new());
                    return;
                };

                let (models_task, selected_task) = match cx.update(|app| {
                    (
                        selector.list_models(app),
                        selector.selected_model(app),
                    )
                }) {
                    Ok(tasks) => tasks,
                    Err(_) => {
                        let _ = reply.send_blocking(Vec::new());
                        return;
                    }
                };

                let (models, selected_model) = futures::join!(models_task, selected_task);
                let Ok(selected_model) = selected_model else {
                    let _ = reply.send_blocking(Vec::new());
                    return;
                };

                let Ok(models) = models else {
                    let _ = reply.send_blocking(Vec::new());
                    return;
                };

                let selected_id = selected_model.id.0.to_string();

                let mut out: Vec<RnAgentModelInfo> = Vec::new();
                match models {
                    acp_thread::AgentModelList::Flat(list) => {
                        out.extend(list.into_iter().map(|m| RnAgentModelInfo {
                            id: m.id.0.to_string(),
                            name: m.name.to_string(),
                            description: m.description.map(|d| d.to_string()),
                            selected: m.id.0.to_string() == selected_id,
                        }));
                    }
                    acp_thread::AgentModelList::Grouped(groups) => {
                        for (_group, list) in groups {
                            out.extend(list.into_iter().map(|m| RnAgentModelInfo {
                                id: m.id.0.to_string(),
                                name: m.name.to_string(),
                                description: m.description.map(|d| d.to_string()),
                                selected: m.id.0.to_string() == selected_id,
                            }));
                        }
                    }
                }

                let _ = reply.send_blocking(out);
            })
            .detach();
        }
    }
}

fn handle_set_agent_model(
    agent_type: AgentServerType,
    model_id: String,
    reply: Sender<bool>,
    cx: &mut App,
) {
    match agent_type {
        AgentServerType::Custom => {
            let Some((provider, model)) = model_id.split_once('/') else {
                let _ = reply.send_blocking(false);
                return;
            };

            let Some(workspace_window) =
                cx.active_window().and_then(|window| window.downcast::<Workspace>())
            else {
                let _ = reply.send_blocking(false);
                return;
            };

            let provider = provider.to_string();
            let model = model.to_string();

            let ok = workspace_window
                .update(cx, |workspace, _window, cx| {
                    let fs = workspace.app_state().fs.clone();
                    update_settings_file(fs, cx, move |settings, _cx| {
                        settings
                            .agent
                            .get_or_insert_default()
                            .set_model(settings::LanguageModelSelection {
                                provider: provider.clone().into(),
                                model: model.clone(),
                            });
                    });
                    Ok(())
                })
                .and_then(|r| r)
                .is_ok();

            let _ = reply.send_blocking(ok);
        }
        _ => {
            let Some(runtime) = resolve_runtime(cx) else {
                let _ = reply.send_blocking(false);
                return;
            };

            let agent_key = AgentKey::from(agent_type.clone());
            let Some(connection) = runtime.read(cx).connection(&agent_key) else {
                let _ = reply.send_blocking(false);
                return;
            };

            let Some(thread) = runtime.read(cx).active_thread(&agent_key) else {
                let _ = reply.send_blocking(false);
                return;
            };

            let session_id = thread.read(cx).session_id().clone();

            let Some(selector) = connection.model_selector(&session_id) else {
                let _ = reply.send_blocking(false);
                return;
            };

            let request_model_id = model_id.clone();
            cx.spawn(async move |cx| {
                let set_task = match cx.update(|app| {
                    selector.select_model(acp::ModelId::new(request_model_id.clone()), app)
                }) {
                    Ok(task) => task,
                    Err(_) => return,
                };

                if let Err(err) = set_task.await {
                    log::error!("rn_chat_panel: Failed to select model: {err}");
                }
            })
            .detach();

            let _ = reply.send_blocking(true);
        }
    }
}

fn get_custom_agent_models(cx: &App) -> Vec<RnAgentModelInfo> {
    let selected_id = {
        let settings = AgentSettings::get_global(cx);
        let selection = settings
            .profiles
            .get(&settings.default_profile)
            .and_then(|profile| profile.default_model.clone())
            .or_else(|| settings.default_model.clone());

        selection.map(|sel| format!("{}/{}", sel.provider.0, sel.model))
    };

    let registry = LanguageModelRegistry::read_global(cx);
    registry
        .available_models(cx)
        .map(|model| {
            let id = format!("{}/{}", model.provider_id().0, model.id().0);
            RnAgentModelInfo {
                id: id.clone(),
                name: model.name().0.to_string(),
                description: Some(model.provider_name().0.to_string()),
                selected: selected_id.as_ref().is_some_and(|selected| selected == &id),
            }
        })
        .collect()
}

fn get_agent_modes(agent_type: AgentServerType, cx: &mut App) -> Vec<RnAgentModeInfo> {
    match agent_type {
        AgentServerType::Custom => {
            let settings = AgentSettings::get_global(cx);
            let profile_id = settings.default_profile.as_str();
            let always_allow = settings.always_allow_tool_actions;

            let selected = if profile_id == builtin_profiles::ASK {
                "read_only"
            } else if always_allow {
                "agent_full_access"
            } else {
                "agent"
            };

            vec![
                RnAgentModeInfo {
                    id: "read_only".to_string(),
                    name: "Read Only".to_string(),
                    description: Some("No tools or edits".to_string()),
                    selected: selected == "read_only",
                },
                RnAgentModeInfo {
                    id: "agent".to_string(),
                    name: "Agent".to_string(),
                    description: Some("Ask before tools or edits".to_string()),
                    selected: selected == "agent",
                },
                RnAgentModeInfo {
                    id: "agent_full_access".to_string(),
                    name: "Agent (Full Access)".to_string(),
                    description: Some("Auto-allow tools and edits".to_string()),
                    selected: selected == "agent_full_access",
                },
            ]
        }
        _ => {
            let Some(runtime) = resolve_runtime(cx) else {
                return Vec::new();
            };

            let agent_key = AgentKey::from(agent_type.clone());
            let Some(connection) = runtime.read(cx).connection(&agent_key) else {
                return Vec::new();
            };

            let Some(thread) = runtime.read(cx).active_thread(&agent_key) else {
                return Vec::new();
            };

            let session_id = thread.read(cx).session_id().clone();
            let Some(session_modes) = connection.session_modes(&session_id, cx) else {
                return Vec::new();
            };

            let current = session_modes.current_mode();
            session_modes
                .all_modes()
                .into_iter()
                .map(|mode| RnAgentModeInfo {
                    id: mode.id.0.to_string(),
                    name: mode.name.to_string(),
                    description: mode.description.map(|d| d.to_string()),
                    selected: mode.id == current,
                })
                .collect()
        }
    }
}

fn handle_set_agent_mode(agent_type: AgentServerType, mode_id: String, cx: &mut App) -> bool {
    match agent_type {
        AgentServerType::Custom => {
            let Some(workspace_window) =
                cx.active_window().and_then(|window| window.downcast::<Workspace>())
            else {
                return false;
            };

            let (profile_id, always_allow) = match mode_id.as_str() {
                "read_only" => (builtin_profiles::ASK, false),
                "agent_full_access" => (builtin_profiles::WRITE, true),
                "agent" => (builtin_profiles::WRITE, false),
                _ => return false,
            };

            workspace_window
                .update(cx, |workspace, _window, cx| {
                    let fs = workspace.app_state().fs.clone();
                    let profile_id = profile_id.to_string();
                    update_settings_file(fs, cx, move |settings, _cx| {
                        settings
                            .agent
                            .get_or_insert_default()
                            .set_profile(profile_id.clone().into());
                        settings
                            .agent
                            .get_or_insert_default()
                            .set_always_allow_tool_actions(always_allow);
                    });
                    Ok(())
                })
                .and_then(|r| r)
                .is_ok()
        }
        _ => {
            let Some(runtime) = resolve_runtime(cx) else {
                return false;
            };

            let agent_key = AgentKey::from(agent_type.clone());
            let Some(connection) = runtime.read(cx).connection(&agent_key) else {
                return false;
            };

            let Some(thread) = runtime.read(cx).active_thread(&agent_key) else {
                return false;
            };

            let session_id = thread.read(cx).session_id().clone();
            let Some(session_modes) = connection.session_modes(&session_id, cx) else {
                return false;
            };

            let task = session_modes.set_mode(acp::SessionModeId::new(mode_id), cx);
            cx.spawn(async move |_cx| {
                if let Err(err) = task.await {
                    log::error!("rn_chat_panel: Failed to set agent mode: {err}");
                }
            })
            .detach();

            true
        }
    }
}



// ============================================================================
// ACP Thread History Helpers
// ============================================================================

fn db_metadata_to_acp(meta: DbThreadMetadata) -> rn_chat_panel_types::AcpThreadMetadata {
    rn_chat_panel_types::AcpThreadMetadata {
        id: meta.id.0.to_string(),
        title: meta.title.to_string(),
        updated_at: meta.updated_at.to_rfc3339(),
    }
}

fn handle_list_acp_threads(
    reply: Sender<Vec<rn_chat_panel_types::AcpThreadMetadata>>,
    cx: &mut App,
) {
    if let Some(runtime) = resolve_runtime(cx) {
        if let Some(threads) = runtime.read(cx).acp_threads(cx) {
            let _ = reply.send_blocking(threads);
            return;
        }
    }

    let db_future = ThreadsDatabase::connect(cx);
    cx.spawn(async move |_cx| {
        let result = match db_future.await {
            Ok(db) => match db.list_threads().await {
                Ok(threads) => threads.into_iter().map(db_metadata_to_acp).collect(),
                Err(e) => {
                    log::error!("[ZedHostCommand] Failed to list ACP threads: {}", e);
                    Vec::new()
                }
            },
            Err(e) => {
                log::error!("[ZedHostCommand] Failed to connect to ThreadsDatabase: {}", e);
                Vec::new()
            }
        };
        let _ = reply.send_blocking(result);
    })
    .detach();
}

fn handle_search_acp_threads(
    query: String,
    reply: Sender<Vec<rn_chat_panel_types::AcpThreadMetadata>>,
    cx: &mut App,
) {
    if let Some(runtime) = resolve_runtime(cx) {
        if let Some(threads) = runtime.read(cx).acp_threads(cx) {
            let query_lower = query.to_lowercase();
            let result = threads
                .into_iter()
                .filter(|t| t.title.to_lowercase().contains(&query_lower))
                .collect();
            let _ = reply.send_blocking(result);
            return;
        }
    }

    let db_future = ThreadsDatabase::connect(cx);
    let query_lower = query.to_lowercase();
    cx.spawn(async move |_cx| {
        let result = match db_future.await {
            Ok(db) => match db.list_threads().await {
                Ok(threads) => threads
                    .into_iter()
                    .filter(|t| t.title.to_lowercase().contains(&query_lower))
                    .map(db_metadata_to_acp)
                    .collect(),
                Err(e) => {
                    log::error!("[ZedHostCommand] Failed to search ACP threads: {}", e);
                    Vec::new()
                }
            },
            Err(e) => {
                log::error!("[ZedHostCommand] Failed to connect to ThreadsDatabase: {}", e);
                Vec::new()
            }
        };
        let _ = reply.send_blocking(result);
    })
    .detach();
}

fn handle_open_acp_thread(thread_id: String, cx: &mut App) -> bool {
    let session_id = acp::SessionId::new(thread_id);
    let Some(workspace_window) =
        cx.active_window().and_then(|window| window.downcast::<Workspace>())
    else {
        return false;
    };

    let result = workspace_window
        .update(cx, |workspace, window, cx| {
            let Some(panel) = workspace.panel::<crate::RNChatPanel>(cx) else {
                return Err(anyhow!("RNChatPanel not available"));
            };

            panel.update(cx, |panel, cx| {
                panel.ensure_native_agent(window, cx);
                panel.load_thread_by_id(session_id.clone(), cx)
            })
        })
        .and_then(|result| result);

    if result.is_ok() {
        set_chat_pane_visible(true, cx);
    }

    result.is_ok()
}

// ============================================================================
// Tool Call Helpers
// ============================================================================

/// Handle an authorization response from React Native.
/// This calls authorize_tool_call on the AcpThread to complete the authorization.
fn handle_tool_authorization_response(
    tool_call_id: &str,
    permission_option_id: &str,
    cx: &mut App,
) -> bool {
    log::info!(
        "[ZedHostCommand] Tool authorization response: tool_call_id={} option={}",
        tool_call_id,
        permission_option_id
    );

    let Some(runtime) = resolve_runtime(cx) else {
        log::warn!("[ZedHostCommand] No runtime available");
        return false;
    };

    let Some(thread) = runtime.read(cx).thread_for_tool_call(tool_call_id) else {
        log::warn!("[ZedHostCommand] Tool call not found: {}", tool_call_id);
        return false;
    };

    let tool_call_id = acp::ToolCallId::new(tool_call_id.to_string());
    let option_id = acp::PermissionOptionId::new(permission_option_id.to_string());
    let tool_call_id_lookup = tool_call_id.clone();
    let option_id_lookup = option_id.clone();

    thread.update(cx, |thread, cx| {
        let option_kind = thread
            .tool_call(&tool_call_id_lookup)
            .and_then(|(_, call)| match &call.status {
                AcpToolCallStatus::WaitingForConfirmation { options, .. } => {
                    options
                        .iter()
                        .find(|opt| opt.option_id == option_id_lookup)
                        .map(|opt| opt.kind)
                }
                _ => None,
            })
            .unwrap_or(acp::PermissionOptionKind::AllowOnce);

        thread.authorize_tool_call(tool_call_id, option_id, option_kind, cx);
    });

    true
}

fn handle_cancel_tool_call(tool_call_id: &str, _cx: &mut App) -> bool {
    log::info!(
        "[ZedHostCommand] Cancel tool call: tool_call_id={}",
        tool_call_id
    );
    false
}

// ============================================================================
// Tool Call Helpers (for native diff/terminal views)
// ============================================================================

fn find_tool_call<'a>(thread: &'a AcpThread, tool_call_id: &str) -> Option<&'a ToolCall> {
    thread.entries().iter().find_map(|entry| {
        if let AgentThreadEntry::ToolCall(call) = entry {
            if call.id.to_string() == tool_call_id {
                return Some(call);
            }
        }
        None
    })
}

fn with_tool_call<R>(
    runtime: &Entity<RnChatRuntime>,
    tool_call_id: &str,
    cx: &App,
    f: impl FnOnce(&AcpThread, &ToolCall) -> R,
) -> Option<R> {
    let thread = runtime.read(cx).thread_for_tool_call(tool_call_id)?;
    let thread_ref = thread.read(cx);
    let tool_call = find_tool_call(thread_ref, tool_call_id)?;
    Some(f(thread_ref, tool_call))
}

fn with_tool_call_opt<T>(
    runtime: &Entity<RnChatRuntime>,
    tool_call_id: &str,
    cx: &App,
    f: impl FnOnce(&AcpThread, &ToolCall) -> Option<T>,
) -> Option<T> {
    with_tool_call(runtime, tool_call_id, cx, f).flatten()
}

pub fn tool_call_content_counts(tool_call_id: &str, cx: &App) -> Option<(u32, u32)> {
    let runtime = resolve_runtime(cx)?;
    with_tool_call(&runtime, tool_call_id, cx, |_thread, tool_call| {
        let diff_count = tool_call.diffs().count() as u32;
        let terminal_count = tool_call.terminals().count() as u32;
        (diff_count, terminal_count)
    })
}

pub fn get_diff_for_tool_call(
    tool_call_id: &str,
    diff_index: usize,
    cx: &App,
) -> Option<Entity<Diff>> {
    let runtime = resolve_runtime(cx)?;
    with_tool_call_opt(&runtime, tool_call_id, cx, |_thread, tool_call| {
        tool_call.diffs().nth(diff_index).cloned()
    })
}

pub fn get_terminal_for_tool_call(
    tool_call_id: &str,
    terminal_index: usize,
    cx: &App,
) -> Option<Entity<Terminal>> {
    let runtime = resolve_runtime(cx)?;
    with_tool_call_opt(&runtime, tool_call_id, cx, |_thread, tool_call| {
        tool_call.terminals().nth(terminal_index).cloned()
    })
}

/// Get the workspace for native components.
#[allow(dead_code)]
pub fn get_workspace(window: &Window) -> Option<WeakEntity<Workspace>> {
    window
        .root::<Workspace>()
        .and_then(|root| root)
        .map(|workspace| workspace.downgrade())
}

/// Get the project for native components.
#[allow(dead_code)]
pub fn get_project(window: &Window, cx: &App) -> Option<Entity<Project>> {
    window
        .root::<Workspace>()
        .and_then(|root| root)
        .map(|workspace| workspace.read(cx).project().clone())
}


// ============================================================================
// Markdown Parsing
// ============================================================================

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

fn parse_markdown_to_rn(markdown: &str) -> RnParsedMarkdown {
    let options = Options::all();
    let parser = Parser::new_ext(markdown, options);
    let events: Vec<Event> = parser.collect();

    let mut converter = MarkdownConverter::new();
    converter.convert(events);

    RnParsedMarkdown {
        elements: converter.elements,
    }
}

struct MarkdownConverter {
    elements: Vec<RnMarkdownElement>,
    inline_stack: Vec<RnMarkdownInline>,
    current_style: RnTextStyle,
    list_stack: Vec<ListState>,
    in_code_block: bool,
    code_block_language: Option<String>,
    code_block_content: String,
}

struct ListState {
    ordered: bool,
    start: Option<u64>,
    current_number: u64,
    items: Vec<RnListItem>,
    current_item_content: Vec<RnMarkdownElement>,
}

impl MarkdownConverter {
    fn new() -> Self {
        Self {
            elements: Vec::new(),
            inline_stack: Vec::new(),
            current_style: RnTextStyle::default(),
            list_stack: Vec::new(),
            in_code_block: false,
            code_block_language: None,
            code_block_content: String::new(),
        }
    }

    fn convert(&mut self, events: Vec<Event>) {
        for event in events {
            self.process_event(event);
        }
    }

    fn process_event(&mut self, event: Event) {
        match event {
            Event::Start(tag) => self.start_tag(tag),
            Event::End(tag) => self.end_tag(tag),
            Event::Text(text) => self.text(text.to_string()),
            Event::Code(code) => self.inline_code(code.to_string()),
            Event::SoftBreak | Event::HardBreak => self.line_break(),
            Event::Rule => self.horizontal_rule(),
            _ => {}
        }
    }

    fn start_tag(&mut self, tag: Tag) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { level: _, .. } => {
                self.inline_stack.clear();
            }
            Tag::BlockQuote(_) => {}
            Tag::CodeBlock(kind) => {
                self.in_code_block = true;
                self.code_block_language = match kind {
                    pulldown_cmark::CodeBlockKind::Fenced(lang) => {
                        let lang = lang.to_string();
                        if lang.is_empty() {
                            None
                        } else {
                            Some(lang)
                        }
                    }
                    pulldown_cmark::CodeBlockKind::Indented => None,
                };
                self.code_block_content.clear();
            }
            Tag::List(start) => {
                self.list_stack.push(ListState {
                    ordered: start.is_some(),
                    start,
                    current_number: start.unwrap_or(1),
                    items: Vec::new(),
                    current_item_content: Vec::new(),
                });
            }
            Tag::Item => {
                if let Some(list) = self.list_stack.last_mut() {
                    list.current_item_content.clear();
                }
            }
            Tag::Emphasis => {
                self.current_style.italic = Some(true);
            }
            Tag::Strong => {
                self.current_style.bold = Some(true);
            }
            Tag::Strikethrough => {
                self.current_style.strikethrough = Some(true);
            }
            Tag::Link { dest_url, .. } => {
                self.inline_stack.push(RnMarkdownInline::Link {
                    text: String::new(),
                    url: dest_url.to_string(),
                    style: None,
                });
            }
            Tag::Image { dest_url, title, .. } => {
                self.inline_stack.push(RnMarkdownInline::Image {
                    alt: title.to_string(),
                    url: dest_url.to_string(),
                });
            }
            Tag::Table(_) | Tag::TableHead | Tag::TableRow | Tag::TableCell => {
                // Table support - simplified for now
            }
            _ => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                if !self.inline_stack.is_empty() {
                    let content = std::mem::take(&mut self.inline_stack);
                    if let Some(list) = self.list_stack.last_mut() {
                        list.current_item_content
                            .push(RnMarkdownElement::Paragraph { content });
                    } else {
                        self.elements.push(RnMarkdownElement::Paragraph { content });
                    }
                }
            }
            TagEnd::Heading(level) => {
                let content = std::mem::take(&mut self.inline_stack);
                self.elements.push(RnMarkdownElement::Heading {
                    level: level as u8,
                    content,
                });
            }
            TagEnd::BlockQuote(_) => {
                // For simplicity, we'll handle block quotes as they close
            }
            TagEnd::CodeBlock => {
                self.in_code_block = false;
                let code = std::mem::take(&mut self.code_block_content);
                let language = self.code_block_language.take();
                self.elements.push(RnMarkdownElement::CodeBlock {
                    language,
                    code,
                    highlights: Vec::new(), // No syntax highlighting yet
                });
            }
            TagEnd::List(_) => {
                if let Some(list) = self.list_stack.pop() {
                    let element = RnMarkdownElement::List {
                        ordered: list.ordered,
                        start: list.start,
                        items: list.items,
                    };
                    if let Some(parent_list) = self.list_stack.last_mut() {
                        parent_list.current_item_content.push(element);
                    } else {
                        self.elements.push(element);
                    }
                }
            }
            TagEnd::Item => {
                if let Some(list) = self.list_stack.last_mut() {
                    let content = std::mem::take(&mut list.current_item_content);
                    let item_type = if list.ordered {
                        let num = list.current_number;
                        list.current_number += 1;
                        RnListItemType::Ordered { number: num }
                    } else {
                        RnListItemType::Unordered
                    };
                    list.items.push(RnListItem { item_type, content });
                }
            }
            TagEnd::Emphasis => {
                self.current_style.italic = None;
            }
            TagEnd::Strong => {
                self.current_style.bold = None;
            }
            TagEnd::Strikethrough => {
                self.current_style.strikethrough = None;
            }
            TagEnd::Link => {
                // Link text was accumulated, now we pop it
                if let Some(RnMarkdownInline::Link { text, url, .. }) = self.inline_stack.pop() {
                    self.inline_stack.push(RnMarkdownInline::Link {
                        text,
                        url,
                        style: if self.current_style.is_empty() {
                            None
                        } else {
                            Some(self.current_style.clone())
                        },
                    });
                }
            }
            TagEnd::Image => {
                // Image was already added
            }
            _ => {}
        }
    }

    fn text(&mut self, text: String) {
        if self.in_code_block {
            self.code_block_content.push_str(&text);
            return;
        }

        // Check if we're inside a link
        if let Some(RnMarkdownInline::Link {
            text: link_text, ..
        }) = self.inline_stack.last_mut()
        {
            link_text.push_str(&text);
            return;
        }

        let style = if self.current_style.is_empty() {
            None
        } else {
            Some(self.current_style.clone())
        };

        self.inline_stack
            .push(RnMarkdownInline::Text { text, style });
    }

    fn inline_code(&mut self, code: String) {
        self.inline_stack.push(RnMarkdownInline::Code { code });
    }

    fn line_break(&mut self) {
        self.inline_stack.push(RnMarkdownInline::Text {
            text: "\n".to_string(),
            style: None,
        });
    }

    fn horizontal_rule(&mut self) {
        self.elements.push(RnMarkdownElement::HorizontalRule);
    }
}
