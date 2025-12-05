//! NativeTerminalView - Renders a terminal view for a tool call.
//!
//! This component displays the terminal output associated with a tool call
//! (typically from an execute command). The terminal entity is looked up
//! using the toolCallId and terminalIndex props.
//!
//! Phase 1 (current): Render-only terminal view.
//! Phase 2 (future): Stop button, expand/collapse behavior.

use gpui::{AnyElement, AppContext, Entity, IntoElement, ParentElement, Styled, div};
use gpui_host::{
    IslandContext, NativeComponent, NativePaintContext, NoEvents,
};
use terminal_view::TerminalView;
use theme::ActiveTheme;

use crate::generated::NativeTerminalViewProps;
use crate::zed_host_command::{
    get_project, get_terminal_for_tool_call, get_workspace, tool_call_content_counts,
};

/// State for the NativeTerminalView component.
pub struct NativeTerminalViewState {
    /// The TerminalView entity, created when terminal is available.
    terminal_view: Option<Entity<TerminalView>>,
    /// The tool_call_id we last created the view for.
    cached_tool_call_id: String,
    /// The terminal index we last created the view for.
    cached_terminal_index: i32,
}

impl Default for NativeTerminalViewState {
    fn default() -> Self {
        Self {
            terminal_view: None,
            cached_tool_call_id: String::new(),
            cached_terminal_index: -1,
        }
    }
}

#[derive(Default)]
pub struct NativeTerminalView;

impl NativeComponent for NativeTerminalView {
    const NAME: &'static str = "NativeTerminalView";
    type Props = NativeTerminalViewProps;
    type Events = NoEvents;
    type State = NativeTerminalViewState;

    fn paint(
        &self,
        _state: &Self::State,
        _props: &Self::Props,
        _ctx: &mut NativePaintContext<'_, '_>,
    ) {
        // Painting handled by element()
    }

    fn element(
        &self,
        state: &Self::State,
        props: &Self::Props,
        ctx: IslandContext<'_>,
    ) -> Option<AnyElement> {
        // Try to render with the cached terminal view if available and valid
        if let Some(terminal_view) = &state.terminal_view {
            if state.cached_tool_call_id == props.tool_call_id
                && state.cached_terminal_index == props.terminal_index
            {
                let view_element = terminal_view.clone().into_any_element();
                return Some(
                    div()
                        .size_full()
                        .overflow_hidden()
                        .child(view_element)
                        .into_any_element(),
                );
            }
        }

        // No valid terminal view yet - render a placeholder
        let counts = tool_call_content_counts(&props.tool_call_id, ctx.cx);
        let status = if counts.is_none() {
            "Loading terminal..."
        } else if counts.is_some_and(|(_, terminal_count)| terminal_count == 0) {
            "No terminal available"
        } else {
            "Preparing terminal view..."
        };

        Some(
            div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_color(ctx.cx.theme().colors().text_muted)
                .text_sm()
                .child(status)
                .into_any_element(),
        )
    }

    fn update(
        &self,
        state: &mut Self::State,
        _old_props: Option<&Self::Props>,
        props: &Self::Props,
        ctx: &mut gpui_host::UpdateContext<'_>,
    ) {
        // Check if props changed - if so, we need to recreate everything
        let props_changed = state.cached_tool_call_id != props.tool_call_id
            || state.cached_terminal_index != props.terminal_index;

        if props_changed {
            state.terminal_view = None;
        }

        self.try_create_terminal_view(state, props, ctx.window, ctx.cx);
    }

    fn frame(
        &self,
        state: &mut Self::State,
        props: &Self::Props,
        ctx: &mut gpui_host::NativeFrameContext<'_, '_>,
    ) {
        self.try_create_terminal_view(state, props, ctx.window, ctx.cx);
    }
}

impl NativeTerminalView {
    fn try_create_terminal_view(
        &self,
        state: &mut NativeTerminalViewState,
        props: &NativeTerminalViewProps,
        window: &mut gpui::Window,
        cx: &mut gpui::App,
    ) {
        // If we already have a terminal view, nothing to do
        if state.terminal_view.is_some() {
            return;
        }

        // Look up the terminal content
        let terminal_index = props.terminal_index.max(0) as usize;
        let Some(acp_terminal) =
            get_terminal_for_tool_call(&props.tool_call_id, terminal_index, cx)
        else {
            return;
        };

        // Get workspace and project for the terminal view
        let Some(workspace) = get_workspace(window) else {
            log::warn!("NativeTerminalView: No workspace available");
            return;
        };

        let Some(project) = get_project(window, cx) else {
            log::warn!("NativeTerminalView: No project available");
            return;
        };

        // Get the inner terminal::Terminal entity
        let terminal = acp_terminal.read(cx).inner().clone();

        // Create a TerminalView in embedded mode (constrained height when unfocused)
        let terminal_view = cx.new(|cx| {
            let mut view = TerminalView::new(
                terminal,
                workspace,
                None, // workspace_id
                project.downgrade(),
                window,
                cx,
            );
            view.set_embedded_mode(Some(10), cx);
            view
        });

        state.terminal_view = Some(terminal_view);
        state.cached_tool_call_id = props.tool_call_id.clone();
        state.cached_terminal_index = props.terminal_index;
    }
}
