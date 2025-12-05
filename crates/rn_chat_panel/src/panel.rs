//! RNChatPanel - A React Native-based chat panel for Zed
//!
//! This panel embeds a React Native surface that renders a chat UI
//! connected to Zed's LLM infrastructure.

use std::rc::Rc;
use std::sync::Arc;
use std::path::PathBuf;

use acp_thread::{AcpThread, AgentConnection, ToolCallStatus};
use agent::{HistoryStore, NativeAgent, NativeAgentConnection, Templates};
use agent_client_protocol as acp;
use anyhow::{anyhow, Result};
use assistant_slash_command::SlashCommandWorkingSet;
use assistant_text_thread::TextThreadStore;
use fs::Fs;
use gpui::{
    Action, App, AsyncWindowContext, Context, Entity, EventEmitter, FocusHandle, Focusable,
    IntoElement, ParentElement, Pixels, Render, Styled, Task, WeakEntity, Window, actions,
    deferred, div, px,
};
use gpui_host::{SurfaceId, SurfaceSlot};
use gpui_host_core::EntityProjection;
use project::Project;
use prompt_store::{PromptBuilder, PromptStore};
use rn_chat_panel_types::AgentServerType;
use ui::{IconName, SpinnerLabel, prelude::*};
use uuid::Uuid;
use workspace::{
    Workspace,
    dock::{DockPosition, Panel, PanelEvent, UtilityPane, UtilityPanePosition},
    utility_pane::utility_slot_for_dock_position,
};
use crate::acp_thread_projection::thread_snapshot_from_acp_thread;
use crate::runtime::{AgentKey, RnChatRuntime, RnChatRuntimeEvent};

const RN_CHAT_PANEL_KEY: &str = "RNChatPanel";

actions!(
    rn_chat_panel,
    [
        /// Toggles focus on the RN Chat panel.
        ToggleFocus
    ]
);

/// Initialize the RN Chat Panel actions and workspace integration.
pub fn init(cx: &mut App) {
    cx.observe_new(
        |workspace: &mut Workspace, _window, _: &mut Context<Workspace>| {
            workspace.register_action(|workspace, _: &ToggleFocus, window, cx| {
                workspace.toggle_panel_focus::<RNChatPanel>(window, cx);
            });
        },
    )
    .detach();
}

/// A panel that renders a React Native chat interface.
pub struct RNChatPanel {
    /// The React Native surface ID for this panel's content.
    surface_id: Option<SurfaceId>,

    /// Session ID for scoping cross-surface state (thread selection, pane visibility, etc).
    session_id: String,

    /// Focus handle for the panel.
    focus_handle: FocusHandle,

    /// Current dock position.
    position: DockPosition,

    /// Panel width (when docked left/right).
    width: Option<Pixels>,

    /// Panel height (when docked bottom).
    height: Option<Pixels>,

    /// Whether the panel is currently active.
    active: bool,

    /// Whether the surface has been created.
    surface_created: bool,

    /// Weak reference to the workspace.
    workspace: WeakEntity<Workspace>,

    /// Weak reference to the project.
    project: WeakEntity<Project>,

    /// Filesystem access.
    fs: Arc<dyn Fs>,

    /// History store for agent threads.
    history_store: Option<Entity<HistoryStore>>,

    /// Text thread store for thread persistence.
    text_thread_store: Option<Entity<TextThreadStore>>,

    /// Prompt store for custom prompts.
    prompt_store: Option<Entity<PromptStore>>,

    /// Shared runtime state for this workspace's project.
    runtime: Entity<RnChatRuntime>,

    /// Native agent entity for tool-enabled messaging.
    native_agent: Option<Entity<NativeAgent>>,

    /// Projection for the active thread.
    thread_projection: Option<EntityProjection<AcpThread>>,

    /// Task for agent initialization.
    _agent_init_task: Option<Task<()>>,

    /// Utility pane for thread/history view (created lazily).
    chat_pane: Option<Entity<crate::chat_pane::RNChatPane>>,

    /// Whether the utility pane was expanded when the main panel was deactivated.
    /// Used to restore the utility pane state when the main panel is reactivated.
    utility_pane_was_expanded: bool,
}

impl RNChatPanel {
    /// Create a new RN Chat Panel.
    pub fn new(
        workspace: &Workspace,
        text_thread_store: Option<Entity<TextThreadStore>>,
        prompt_store: Option<Entity<PromptStore>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();
        let session_id = Uuid::new_v4().to_string();
        let weak_workspace = workspace.weak_handle();
        let project = workspace.project().clone();
        let weak_project = project.downgrade();
        let fs = workspace.app_state().fs.clone();
        let runtime = RnChatRuntime::get_or_create(&project, cx);

        cx.subscribe(&runtime, Self::handle_runtime_event).detach();
        runtime.update(cx, |runtime, cx| {
            runtime.ensure_active_thread_sync(cx);
        });

        // Eagerly prepare the window for RN surfaces. This allocates a frame handle
        // and registers the window waker before any surfaces are created, avoiding
        // first-frame edge cases where animations may not start correctly.
        if crate::is_initialized() {
            gpui_host::library_mode::prepare_window(window, cx);
        }

        // Create history store from text_thread_store
        let history_store = text_thread_store
            .as_ref()
            .map(|store| cx.new(|cx| HistoryStore::new(store.clone(), cx)));
        if let Some(history_store) = history_store.clone() {
            runtime.update(cx, |runtime, cx| {
                runtime.set_history_store(history_store, cx);
            });
        }

        Self {
            surface_id: None,
            session_id,
            focus_handle,
            position: DockPosition::Left,
            width: Some(px(400.)),
            height: Some(px(300.)),
            active: false,
            surface_created: false,
            workspace: weak_workspace,
            project: weak_project,
            fs,
            history_store,
            text_thread_store,
            prompt_store,
            runtime,
            native_agent: None,
            thread_projection: None,
            _agent_init_task: None,
            chat_pane: None,
            utility_pane_was_expanded: false,
        }
    }

    pub(crate) fn ensure_chat_pane(
        &mut self,
        pane_position: UtilityPanePosition,
        cx: &mut Context<Self>,
    ) -> Entity<crate::chat_pane::RNChatPane> {
        let session_id = self.session_id.clone();
        let workspace = self.workspace.clone();

        if let Some(pane) = &self.chat_pane {
            pane.update(cx, |pane, cx| pane.set_position(pane_position, cx));
            return pane.clone();
        }

        let pane = cx.new(|cx| {
            crate::chat_pane::RNChatPane::new(
                workspace,
                session_id,
                pane_position,
                cx,
            )
        });
        self.chat_pane = Some(pane.clone());
        pane
    }

    pub(crate) fn existing_chat_pane(&self) -> Option<Entity<crate::chat_pane::RNChatPane>> {
        self.chat_pane.clone()
    }

    /// Load the panel asynchronously.
    /// This follows the pattern used by other Zed panels.
    pub fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> gpui::Task<anyhow::Result<Entity<Self>>> {
        let prompt_store = cx.update(|_window, cx| PromptStore::global(cx));
        cx.spawn(async move |cx| {
            let (project, prompt_builder) = workspace.update(cx, |workspace, cx| {
                let project = workspace.project().clone();
                let fs = workspace.app_state().fs.clone();
                let prompt_builder = PromptBuilder::load(fs, false, cx);
                (project, prompt_builder)
            })?;

            let slash_commands = Arc::new(SlashCommandWorkingSet::default());
            let text_thread_store = workspace
                .update(cx, |_, cx| {
                    TextThreadStore::new(project, prompt_builder, slash_commands, cx)
                })?
                .await?;

            let prompt_store = match prompt_store {
                Ok(prompt_store) => prompt_store.await.ok(),
                Err(_) => None,
            };

            workspace.update_in(cx, |workspace, window, cx| {
                cx.new(|cx| Self::new(workspace, Some(text_thread_store), prompt_store, window, cx))
            })
        })
    }

    /// Ensure the React Native surface exists.
    fn ensure_surface(&mut self, cx: &mut Context<Self>) {
        if self.surface_created {
            return;
        }

        if !crate::is_initialized() {
            log::warn!("rn_chat_panel: Cannot create surface - RN runtime not initialized");
            return;
        }

        log::info!("rn_chat_panel: Creating React Native surface");

        let initial_props = serde_json::json!({
            "sessionId": self.session_id.clone(),
            "role": "history",
            "paneSide": match self.position {
                DockPosition::Left => Some("left"),
                DockPosition::Right => Some("right"),
                _ => None,
            },
        });
        let initial_props_json = match serde_json::to_string(&initial_props) {
            Ok(json) => json,
            Err(err) => {
                log::error!("rn_chat_panel: Failed to serialize initial props: {}", err);
                return;
            }
        };

        match gpui_host::library_mode::create_surface_for_panel_with_initial_props_json(
            "RNChatHistoryPanel",
            &initial_props_json,
            cx,
        ) {
            Ok(surface_id) => {
                log::info!("rn_chat_panel: Surface created with ID {:?}", surface_id);
                self.surface_id = Some(surface_id);
                self.surface_created = true;
                cx.notify();
            }
            Err(e) => {
                log::error!("rn_chat_panel: Failed to create surface: {}", e);
            }
        }
    }

    /// Initialize the NativeAgent for tool-enabled messaging.
    /// This should be called when the Custom agent type is connected.
    pub fn ensure_native_agent(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(agent) = self.native_agent.clone() {
            let agent_key = AgentKey::from(AgentServerType::Custom);
            if !self.runtime.read(cx).is_connected(&agent_key) {
                let connection = Rc::new(NativeAgentConnection(agent)) as Rc<dyn AgentConnection>;
                self.runtime.update(cx, |runtime, cx| {
                    runtime.set_connection(agent_key, connection.clone(), cx);
                });
            }
            return;
        }

        let Some(history_store) = self.history_store.clone() else {
            log::warn!("rn_chat_panel: Cannot create NativeAgent - no history store");
            return;
        };

        let Some(project) = self.project.upgrade() else {
            log::warn!("rn_chat_panel: Cannot create NativeAgent - project dropped");
            return;
        };

        let fs = self.fs.clone();
        let prompt_store = self.prompt_store.clone();
        let templates = Templates::new();

        log::info!("rn_chat_panel: Creating NativeAgent for tool support");

        let task = cx.spawn_in(window, async move |this, cx| {
            let agent_result = NativeAgent::new(
                project.clone(),
                history_store,
                templates,
                prompt_store,
                fs,
                cx,
            )
            .await;

            match agent_result {
                Ok(agent) => {
                    let connection = NativeAgentConnection(agent.clone());

                    // Store agent and register connection in runtime
                    this.update(cx, |panel, cx| {
                        panel.native_agent = Some(agent.clone());
                        let connection = Rc::new(connection) as Rc<dyn AgentConnection>;
                        panel.runtime.update(cx, |runtime, cx| {
                            runtime.set_connection(
                                AgentKey::from(AgentServerType::Custom),
                                connection.clone(),
                                cx,
                            );
                        });

                        log::info!("rn_chat_panel: NativeAgent created successfully");
                        cx.notify();
                    })
                    .ok();
                }
                Err(e) => {
                    log::error!("rn_chat_panel: Failed to create NativeAgent: {}", e);
                }
            }
        });

        self._agent_init_task = Some(task);
    }

    pub fn start_new_thread(&mut self, cx: &mut Context<Self>) -> Result<()> {
        let agent_key = AgentKey::from(AgentServerType::Custom);
        let Some(connection) = self.runtime.read(cx).connection(&agent_key) else {
            return Err(anyhow!("No agent connection"));
        };
        let Some(project) = self.project.upgrade() else {
            return Err(anyhow!("No project"));
        };

        let root_dir = project
            .read(cx)
            .worktrees(cx)
            .next()
            .map(|wt| wt.read(cx).abs_path().to_path_buf())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("/"));

        let task = connection.clone().new_thread(project.clone(), root_dir.as_path(), cx);

        cx.spawn(async move |this, cx| {
            let thread: Entity<AcpThread> = match task.await {
                Ok(thread) => thread,
                Err(err) => {
                    log::error!("rn_chat_panel: Failed to start thread: {err}");
                    return;
                }
            };

            this.update(cx, |panel, cx| {
                panel.runtime.update(cx, |runtime, cx| {
                    runtime.set_active_thread(agent_key.clone(), thread, cx);
                });
            })
            .ok();
        })
        .detach();

        Ok(())
    }

    pub fn load_thread_by_id(
        &mut self,
        session_id: acp::SessionId,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let Some(native_agent) = self.native_agent.clone() else {
            return Err(anyhow!("Native agent not initialized"));
        };

        let runtime = self.runtime.clone();
        let agent_key = AgentKey::from(AgentServerType::Custom);
        let open_task = native_agent.update(cx, |agent, cx| agent.open_thread(session_id.clone(), cx));

        cx.spawn(async move |_, cx| {
            let acp_thread = match open_task.await {
                Ok(thread) => thread,
                Err(err) => {
                    log::error!("rn_chat_panel: Failed to open ACP thread: {err}");
                    return;
                }
            };

            let _ = cx.update(|app| {
                runtime.update(app, |runtime, cx| {
                    runtime.set_active_thread(agent_key, acp_thread, cx);
                });
            });
        })
        .detach();

        Ok(())
    }

    pub fn send_message(&mut self, message: String, cx: &mut Context<Self>) -> Result<()> {
        let agent_key = AgentKey::from(AgentServerType::Custom);
        if let Some(thread) = self.runtime.read(cx).active_thread(&agent_key) {
            let send_task = thread.update(cx, |thread, cx| thread.send(vec![message.into()], cx));
            cx.background_executor()
                .spawn(async move {
                    if let Err(err) = send_task.await {
                        log::error!("rn_chat_panel: Failed to send message: {err}");
                    }
                })
                .detach();
            return Ok(());
        }

        let Some(connection) = self.runtime.read(cx).connection(&agent_key) else {
            return Err(anyhow!("No agent connection"));
        };
        let Some(project) = self.project.upgrade() else {
            return Err(anyhow!("No project"));
        };

        let root_dir = project
            .read(cx)
            .worktrees(cx)
            .next()
            .map(|wt| wt.read(cx).abs_path().to_path_buf())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("/"));

        let task = connection.clone().new_thread(project.clone(), root_dir.as_path(), cx);

        cx.spawn(async move |this, cx| {
            let thread: Entity<AcpThread> = match task.await {
                Ok(thread) => thread,
                Err(err) => {
                    log::error!("rn_chat_panel: Failed to start thread: {err}");
                    return;
                }
            };

            this.update(cx, |panel, cx| {
                panel.runtime.update(cx, |runtime, cx| {
                    runtime.set_active_thread(agent_key.clone(), thread.clone(), cx);
                });
                let send_task =
                    thread.update(cx, |thread, cx| thread.send(vec![message.into()], cx));
                cx.background_executor()
                    .spawn(async move {
                        if let Err(err) = send_task.await {
                            log::error!("rn_chat_panel: Failed to send message: {err}");
                        }
                    })
                    .detach();
            })
            .ok();
        })
        .detach();

        Ok(())
    }

    pub fn authorize_tool(
        &mut self,
        tool_call_id: &str,
        option_id: &str,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let thread = self
            .runtime
            .read(cx)
            .thread_for_tool_call(tool_call_id)
            .ok_or_else(|| anyhow!("No active thread"))?;
        let tool_call_id = acp::ToolCallId::new(tool_call_id.to_string());
        let option_id = acp::PermissionOptionId::new(option_id.to_string());
        let tool_call_id_lookup = tool_call_id.clone();
        let option_id_lookup = option_id.clone();

        thread.update(cx, |thread, cx| {
            let option_kind = thread
                .tool_call(&tool_call_id_lookup)
                .and_then(|(_, call)| match &call.status {
                    ToolCallStatus::WaitingForConfirmation { options, .. } => {
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

        Ok(())
    }

    /// Check if the native agent is initialized.
    pub fn has_native_agent(&self) -> bool {
        self.native_agent.is_some()
    }

    pub(crate) fn runtime(&self) -> &Entity<RnChatRuntime> {
        &self.runtime
    }

    fn attach_thread_projection(&mut self, agent_key: &AgentKey, cx: &mut Context<Self>) {
        let Some(thread) = self.runtime.read(cx).active_thread(agent_key) else {
            self.thread_projection = None;
            return;
        };

        let module_id = self.module_id();
        let projection = EntityProjection::new_with(
            thread,
            thread_snapshot_from_acp_thread,
            move |snapshot| {
                rn_chat_panel_types::emit_thread_snapshot(module_id, snapshot);
            },
            cx,
        );
        self.thread_projection = Some(projection);
    }

    fn handle_runtime_event(
        &mut self,
        _runtime: Entity<RnChatRuntime>,
        event: &RnChatRuntimeEvent,
        cx: &mut Context<Self>,
    ) {
        match event {
            RnChatRuntimeEvent::ConnectionChanged { agent_key, connected } => {
                if *connected {
                    rn_chat_panel_types::emit_agent_connected(
                        agent_key.agent_type.clone(),
                        self.session_id.clone(),
                    );
                } else {
                    rn_chat_panel_types::emit_agent_disconnected(
                        agent_key.agent_type.clone(),
                        self.session_id.clone(),
                    );
                }
            }
            RnChatRuntimeEvent::ActiveThreadChanged { agent_key } => {
                self.attach_thread_projection(agent_key, cx);
            }
            RnChatRuntimeEvent::ToolCallsUpdated { .. } => {}
        }
    }

    fn module_id(&self) -> usize {
        self.surface_id.map(usize::from).unwrap_or(0)
    }
}

impl EventEmitter<PanelEvent> for RNChatPanel {}

impl Focusable for RNChatPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for RNChatPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let bundle_loaded = gpui_host::library_mode::is_bundle_loaded();
        let bundle_url = std::env::var("RN_METRO_URL").unwrap_or_else(|_| {
            "http://localhost:8082/index.bundle?platform=gpui&dev=true".to_string()
        });

        let content = match self.surface_id {
            Some(surface_id) => {
                let loading_overlay = v_flex()
                    .absolute()
                    .inset_0()
                    .gap_1()
                    .items_center()
                    .justify_center()
                    .bg(cx.theme().colors().panel_background.opacity(0.92))
                    .child(
                        h_flex()
                            .gap_1()
                            .items_center()
                            .child(SpinnerLabel::new().color(Color::Accent))
                            .child(Label::new("Waiting for Metro bundle…")),
                    )
                    .child(
                        Label::new(bundle_url.clone())
                            .size(LabelSize::Small)
                            .color(Color::Muted)
                            .truncate(),
                    )
                    .child(
                        Label::new("Start it with: pnpm start --port 8082")
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    )
                    .child(
                        Label::new("Or set RN_METRO_URL to a different bundle URL")
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                    );

                div()
                    .size_full()
                    .relative()
                    .child(div().size_full().child(SurfaceSlot::new(surface_id)))
                    .when(!bundle_loaded, |this| {
                        this.child(
                            deferred(loading_overlay).with_priority(usize::MAX - 50_000),
                        )
                    })
            }
            None => {
                // Show loading state
                div()
                    .size_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(cx.theme().colors().surface_background)
                    .child(
                        div()
                            .text_color(cx.theme().colors().text_muted)
                            .child(
                                h_flex()
                                    .gap_1()
                                    .items_center()
                                    .child(SpinnerLabel::new().color(Color::Accent))
                                    .child("Loading React Native surface..."),
                            )
                    )
            }
        };

        div()
            .key_context("RNChatPanel")
            .track_focus(&self.focus_handle)
            .size_full()
            .bg(cx.theme().colors().panel_background)
            .child(content)
    }
}

impl Panel for RNChatPanel {
    fn persistent_name() -> &'static str {
        "RNChatPanel"
    }

    fn panel_key() -> &'static str {
        RN_CHAT_PANEL_KEY
    }

    fn position(&self, _window: &Window, _cx: &App) -> DockPosition {
        self.position
    }

    fn position_is_valid(&self, _position: DockPosition) -> bool {
        // Allow all positions
        true
    }

    fn set_position(
        &mut self,
        position: DockPosition,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.position = position;
        cx.notify();
    }

    fn size(&self, window: &Window, cx: &App) -> Pixels {
        match self.position(window, cx) {
            DockPosition::Left | DockPosition::Right => self.width.unwrap_or(px(400.)),
            DockPosition::Bottom => self.height.unwrap_or(px(300.)),
        }
    }

    fn set_size(&mut self, size: Option<Pixels>, window: &mut Window, cx: &mut Context<Self>) {
        match self.position(window, cx) {
            DockPosition::Left | DockPosition::Right => self.width = size,
            DockPosition::Bottom => self.height = size,
        }
        cx.notify();
    }

    fn icon(&self, _window: &Window, _cx: &App) -> Option<IconName> {
        Some(IconName::Ai)
    }

    fn icon_tooltip(&self, _window: &Window, _cx: &App) -> Option<&'static str> {
        Some("RN Chat Panel")
    }

    fn toggle_action(&self) -> Box<dyn Action> {
        Box::new(ToggleFocus)
    }

    fn activation_priority(&self) -> u32 {
        // Use 7 (available slot between collab_panel=6 and notification_panel=8)
        7
    }

    fn set_active(&mut self, active: bool, window: &mut Window, cx: &mut Context<Self>) {
        let was_active = self.active;
        self.active = active;

        if active && !was_active {
            // Create surface when panel becomes active
            self.ensure_surface(cx);

            // Initialize the NativeAgent for tool support
            self.ensure_native_agent(window, cx);

            // Restore utility pane if it was expanded before (deferred to avoid double-borrow)
            if self.utility_pane_was_expanded {
                let workspace = self.workspace.clone();
                let dock_position = self.position;
                let pane_position = match dock_position {
                    DockPosition::Left => UtilityPanePosition::Right,
                    DockPosition::Right => UtilityPanePosition::Left,
                    DockPosition::Bottom => UtilityPanePosition::Right,
                };
                let pane = self.ensure_chat_pane(pane_position, cx);
                let pane_clone = pane.clone();
                pane.update(cx, |pane, cx| pane.set_expanded(true, cx));
                let panel_id = cx.entity_id();

                cx.defer(move |cx| {
                    if let Some(workspace) = workspace.upgrade() {
                        let slot = utility_slot_for_dock_position(dock_position);
                        workspace.update(cx, |workspace, cx| {
                            workspace.register_utility_pane(slot, panel_id, pane_clone, cx);
                        });
                    }
                });
            }
        } else if !active && was_active {
            // Unregister surface from compositor when panel becomes inactive
            if let Some(surface_id) = self.surface_id {
                gpui_host::window::unregister_surface_from_compositor(surface_id, cx);
            }

            // Close utility pane and remember its state
            if let Some(pane) = self.chat_pane.as_ref() {
                let was_expanded = pane.read(cx).expanded(cx);
                self.utility_pane_was_expanded = was_expanded;

                if was_expanded {
                    pane.update(cx, |pane, cx| pane.set_expanded(false, cx));

                    // Defer workspace update to avoid double-borrow panic
                    let workspace = self.workspace.clone();
                    let dock_position = self.position;
                    let panel_id = cx.entity_id();
                    cx.defer(move |cx| {
                        if let Some(workspace) = workspace.upgrade() {
                            let slot = utility_slot_for_dock_position(dock_position);
                            workspace.update(cx, |workspace, cx| {
                                workspace.clear_utility_pane_if_provider(slot, panel_id, cx);
                            });
                        }
                    });
                }
            }
        }

        cx.notify();
    }

    fn starts_open(&self, _window: &Window, _cx: &App) -> bool {
        false
    }

    fn needs_overflow_visible(&self) -> bool {
        true
    }
}

impl Drop for RNChatPanel {
    fn drop(&mut self) {
        if let Some(surface_id) = self.surface_id.take() {
            log::info!("rn_chat_panel: Panel dropped, surface {:?} will be cleaned up", surface_id);
            // Note: Surface cleanup happens when the RN runtime shuts down
            // or could be done explicitly if we had access to cx here
        }
    }
}
