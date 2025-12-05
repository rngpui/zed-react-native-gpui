use gpui::{
    App, Context, EventEmitter, FocusHandle, Focusable, InteractiveElement, Pixels, Render, Styled,
    Window, px,
};
use gpui_host::{SurfaceId, SurfaceSlot};
use ui::{ActiveTheme, IconName, IntoElement, ParentElement, div};
use workspace::{
    Workspace,
    dock::{ClosePane, MinimizePane, UtilityPane, UtilityPanePosition},
};

const DEFAULT_CHAT_PANE_WIDTH: Pixels = px(360.0);

pub struct RNChatPane {
    focus_handle: FocusHandle,
    _workspace: gpui::WeakEntity<Workspace>,
    session_id: String,
    surface_id: Option<SurfaceId>,
    surface_created: bool,
    expanded: bool,
    width: Option<Pixels>,
    position_left: bool,
}

impl EventEmitter<MinimizePane> for RNChatPane {}
impl EventEmitter<ClosePane> for RNChatPane {}

impl RNChatPane {
    pub fn new(
        workspace: gpui::WeakEntity<Workspace>,
        session_id: String,
        position: UtilityPanePosition,
        cx: &mut Context<Self>,
    ) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            _workspace: workspace,
            session_id,
            surface_id: None,
            surface_created: false,
            expanded: false,
            width: None,
            position_left: matches!(position, UtilityPanePosition::Left),
        }
    }

    pub fn set_position(&mut self, position: UtilityPanePosition, cx: &mut Context<Self>) {
        let new_left = matches!(position, UtilityPanePosition::Left);
        if self.position_left != new_left {
            self.position_left = new_left;

            if let Some(surface_id) = self.surface_id.take() {
                gpui_host::window::unregister_surface_from_compositor(surface_id, cx);
                gpui_host::library_mode::destroy_surface(surface_id, cx);
            }

            self.surface_created = false;
        }
        cx.notify();
    }

    fn ensure_surface(&mut self, cx: &mut Context<Self>) {
        if self.surface_created {
            return;
        }

        if !crate::is_initialized() {
            log::warn!("rn_chat_panel: Cannot create history surface - RN runtime not initialized");
            return;
        }

        let initial_props = serde_json::json!({
            "sessionId": self.session_id.clone(),
            "role": "main",
            "paneSide": if self.position_left { Some("left") } else { Some("right") },
        });
        let initial_props_json = match serde_json::to_string(&initial_props) {
            Ok(json) => json,
            Err(err) => {
                log::error!("rn_chat_panel: Failed to serialize history initial props: {}", err);
                return;
            }
        };

        match gpui_host::library_mode::create_surface_for_panel_with_initial_props_json(
            "RNChatPanel",
            &initial_props_json,
            cx,
        ) {
            Ok(surface_id) => {
                self.surface_id = Some(surface_id);
                self.surface_created = true;
                cx.notify();
            }
            Err(e) => {
                log::error!("rn_chat_panel: Failed to create history surface: {}", e);
            }
        }
    }

}

impl Focusable for RNChatPane {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl UtilityPane for RNChatPane {
    fn position(&self, _window: &Window, _cx: &App) -> UtilityPanePosition {
        if self.position_left { UtilityPanePosition::Left } else { UtilityPanePosition::Right }
    }

    fn toggle_icon(&self, _cx: &App) -> IconName {
        IconName::Thread
    }

    fn expanded(&self, _cx: &App) -> bool {
        self.expanded
    }

    fn set_expanded(&mut self, expanded: bool, cx: &mut Context<Self>) {
        self.expanded = expanded;
        if !expanded {
            if let Some(surface_id) = self.surface_id {
                gpui_host::window::unregister_surface_from_compositor(surface_id, cx);
            }
        }
        cx.notify();
    }

    fn width(&self, _cx: &App) -> Pixels {
        self.width.unwrap_or(DEFAULT_CHAT_PANE_WIDTH)
    }

    fn set_width(&mut self, width: Option<Pixels>, cx: &mut Context<Self>) {
        self.width = width;
        cx.notify();
    }
}

impl Render for RNChatPane {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.ensure_surface(cx);

        let content = match self.surface_id {
            Some(surface_id) => div()
                .size_full()
                .relative()
                .child(div().size_full().child(SurfaceSlot::new(surface_id))),
            None => div().size_full().child("Loading history…"),
        };
        let content = div().h_full().flex_grow().child(content);

        let separator = || {
            div()
                .w(px(1.0))
                .h_full()
                .bg(cx.theme().colors().border)
                .flex_none()
        };

        let mut content = Some(content);
        let content_with_separator = if self.position_left {
            let content = content.take().expect("content");
            div()
                .size_full()
                .flex()
                .flex_row()
                .child(content)
                .child(separator())
        } else {
            let content = content.take().expect("content");
            div()
                .size_full()
                .flex()
                .flex_row()
                .child(separator())
                .child(content)
        };

        div()
            .key_context("RNChatPane")
            .track_focus(&self.focus_handle)
            .size_full()
            .child(content_with_separator)
    }
}
