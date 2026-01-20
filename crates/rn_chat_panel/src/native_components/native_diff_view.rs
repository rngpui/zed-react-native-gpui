//! NativeDiffView - Renders a diff editor for a tool call.
//!
//! This component displays the file diff associated with a tool call.
//! The diff content is looked up using the toolCallId and diffIndex props.
//!
//! Phase 1 (current): Render-only, read-only diff view.
//! Phase 2 (future): Accept/reject hunks with GPUI actions.

use editor::Editor;
use gpui::{AnyElement, AppContext, Entity, IntoElement, ParentElement, Styled, div};
use rngpui::{
    IslandContext, NativeComponent, NativePaintContext, NoEvents,
};
use theme::ActiveTheme;

use crate::generated::NativeDiffViewProps;
use crate::zed_host_command::{get_diff_for_tool_call, get_project, tool_call_content_counts};

/// State for the NativeDiffView component.
pub struct NativeDiffViewState {
    /// The Editor entity, created when diff is available.
    editor: Option<Entity<Editor>>,
    /// The tool_call_id we last created the editor for.
    cached_tool_call_id: String,
    /// The diff index we last created the editor for.
    cached_diff_index: i32,
}

impl Default for NativeDiffViewState {
    fn default() -> Self {
        Self {
            editor: None,
            cached_tool_call_id: String::new(),
            cached_diff_index: -1,
        }
    }
}

#[derive(Default)]
pub struct NativeDiffView;

impl NativeComponent for NativeDiffView {
    const NAME: &'static str = "NativeDiffView";
    type Props = NativeDiffViewProps;
    type Events = NoEvents;
    type State = NativeDiffViewState;

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
        // Try to render with the cached editor if available and valid
        if let Some(editor) = &state.editor {
            if state.cached_tool_call_id == props.tool_call_id
                && state.cached_diff_index == props.diff_index
            {
                let editor_element = editor.clone().into_any_element();
                return Some(
                    div()
                        .size_full()
                        .overflow_hidden()
                        .child(editor_element)
                        .into_any_element(),
                );
            }
        }

        // No valid editor yet - render a placeholder
        let counts = tool_call_content_counts(&props.tool_call_id, ctx.cx);
        let status = if counts.is_none() {
            "Loading diff..."
        } else if counts.is_some_and(|(diff_count, _)| diff_count == 0) {
            "No diff available"
        } else {
            "Preparing diff view..."
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
        ctx: &mut rngpui::UpdateContext<'_>,
    ) {
        // Check if props changed - if so, we need to recreate everything
        let props_changed = state.cached_tool_call_id != props.tool_call_id
            || state.cached_diff_index != props.diff_index;

        if props_changed {
            state.editor = None;
        }

        self.try_create_editor(state, props, ctx.window, ctx.cx);
    }

    fn frame(
        &self,
        state: &mut Self::State,
        props: &Self::Props,
        ctx: &mut rngpui::NativeFrameContext<'_, '_>,
    ) {
        self.try_create_editor(state, props, ctx.window, ctx.cx);
    }
}

impl NativeDiffView {
    fn try_create_editor(
        &self,
        state: &mut NativeDiffViewState,
        props: &NativeDiffViewProps,
        window: &mut gpui::Window,
        cx: &mut gpui::App,
    ) {
        // If we already have an editor, nothing to do
        if state.editor.is_some() {
            return;
        }

        // Look up the diff content
        let diff_index = props.diff_index.max(0) as usize;
        let Some(diff_entity) = get_diff_for_tool_call(&props.tool_call_id, diff_index, cx) else {
            return;
        };

        // Check if the diff has a revealed range (content is ready)
        let has_content = diff_entity.read(cx).has_revealed_range(cx);
        if !has_content {
            return;
        }

        // Create a read-only editor from the diff's multibuffer
        let multibuffer = diff_entity.read(cx).multibuffer().clone();
        let project = get_project(window, cx);

        let editor = cx.new(|cx| {
            let mut editor = Editor::for_multibuffer(multibuffer, project, window, cx);
            editor.set_read_only(true);
            editor.set_show_gutter(true, cx);
            editor
        });

        state.editor = Some(editor);
        state.cached_tool_call_id = props.tool_call_id.clone();
        state.cached_diff_index = props.diff_index;
    }
}
