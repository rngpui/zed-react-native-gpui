use crate::{
    ActiveTooltip, AnyView, App, AvailableSpace, Bounds, DispatchPhase, Element,
    ElementId, GlobalElementId, HighlightStyle, Hitbox, HitboxBehavior, InspectorElementId,
    IntoElement, LayoutId, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point,
    SharedString, Size, TextOverflow, TextRun, TextStyle, TooltipId, TruncateFrom, UpdateResult,
    WhiteSpace, Window, WrappedLine, WrappedLineLayout, register_tooltip_mouse_handlers,
    set_tooltip_on_window,
};
use anyhow::Context as _;
use itertools::Itertools;
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    hash::{Hash, Hasher},
    mem,
    ops::Range,
    rc::Rc,
    sync::Arc,
};
use util::ResultExt;
use collections::FxHasher;

impl Element for &'static str {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _window: &mut Window,
        _cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        unreachable!("&'static str uses retained node path")
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _text_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        unreachable!("&'static str uses retained node path")
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _text_layout: &mut Self::RequestLayoutState,
        _: &mut (),
        _window: &mut Window,
        _cx: &mut App,
    ) {
        unreachable!("&'static str uses retained node path")
    }

    fn into_any(self) -> crate::AnyElement {
        crate::AnyElement::new(self)
    }

    fn create_render_node(&mut self) -> Option<Box<dyn crate::RenderNode>> {
        Some(Box::new(TextNode::new(SharedString::from(*self), None)))
    }

    fn update_render_node(
        &mut self,
        node: &mut dyn crate::RenderNode,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Option<UpdateResult> {
        if let Some(text_node) = node.as_any_mut().downcast_mut::<TextNode>() {
            let text_changed = text_node.text != *self;
            let runs_changed = text_node.runs.is_some();
            if text_changed || runs_changed {
                text_node.update_from(SharedString::from(*self), None);
                Some(UpdateResult::LAYOUT_CHANGED)
            } else {
                Some(UpdateResult::UNCHANGED)
            }
        } else {
            None
        }
    }
}

impl IntoElement for &'static str {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl IntoElement for String {
    type Element = SharedString;

    fn into_element(self) -> Self::Element {
        self.into()
    }
}

impl Element for SharedString {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _window: &mut Window,
        _cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        unreachable!("SharedString uses retained node path")
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _text_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        unreachable!("SharedString uses retained node path")
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _text_layout: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        unreachable!("SharedString uses retained node path")
    }

    fn into_any(self) -> crate::AnyElement {
        crate::AnyElement::new(self)
    }

    fn create_render_node(&mut self) -> Option<Box<dyn crate::RenderNode>> {
        Some(Box::new(TextNode::new(self.clone(), None)))
    }

    fn update_render_node(
        &mut self,
        node: &mut dyn crate::RenderNode,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Option<UpdateResult> {
        if let Some(text_node) = node.as_any_mut().downcast_mut::<TextNode>() {
            let text_changed = text_node.text != *self;
            let runs_changed = text_node.runs.is_some();
            if text_changed || runs_changed {
                text_node.update_from(self.clone(), None);
                Some(UpdateResult::LAYOUT_CHANGED)
            } else {
                Some(UpdateResult::UNCHANGED)
            }
        } else {
            None
        }
    }
}

impl IntoElement for SharedString {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// Renders text with runs of different styles.
///
/// Callers are responsible for setting the correct style for each run.
/// For text with a uniform style, you can usually avoid calling this constructor
/// and just pass text directly.
pub struct StyledText {
    text: SharedString,
    runs: Option<Vec<TextRun>>,
    delayed_highlights: Option<Vec<(Range<usize>, HighlightStyle)>>,
    layout: TextLayout,
}

impl StyledText {
    /// Construct a new styled text element from the given string.
    pub fn new(text: impl Into<SharedString>) -> Self {
        StyledText {
            text: text.into(),
            runs: None,
            delayed_highlights: None,
            layout: TextLayout::default(),
        }
    }

    /// Get the layout for this element. This can be used to map indices to pixels and vice versa.
    pub fn layout(&self) -> &TextLayout {
        &self.layout
    }

    /// Set the styling attributes for the given text, as well as
    /// as any ranges of text that have had their style customized.
    pub fn with_default_highlights(
        mut self,
        default_style: &TextStyle,
        highlights: impl IntoIterator<Item = (Range<usize>, HighlightStyle)>,
    ) -> Self {
        debug_assert!(
            self.delayed_highlights.is_none(),
            "Can't use `with_default_highlights` and `with_highlights`"
        );
        let runs = Self::compute_runs(&self.text, default_style, highlights);
        self.with_runs(runs)
    }

    /// Set the styling attributes for the given text, as well as
    /// as any ranges of text that have had their style customized.
    pub fn with_highlights(
        mut self,
        highlights: impl IntoIterator<Item = (Range<usize>, HighlightStyle)>,
    ) -> Self {
        debug_assert!(
            self.runs.is_none(),
            "Can't use `with_highlights` and `with_default_highlights`"
        );
        self.delayed_highlights = Some(
            highlights
                .into_iter()
                .inspect(|(run, _)| {
                    debug_assert!(self.text.is_char_boundary(run.start));
                    debug_assert!(self.text.is_char_boundary(run.end));
                })
                .collect::<Vec<_>>(),
        );
        self
    }

    fn compute_runs(
        text: &str,
        default_style: &TextStyle,
        highlights: impl IntoIterator<Item = (Range<usize>, HighlightStyle)>,
    ) -> Vec<TextRun> {
        let mut runs = Vec::new();
        let mut ix = 0;
        for (range, highlight) in highlights {
            if ix < range.start {
                debug_assert!(text.is_char_boundary(range.start));
                runs.push(default_style.clone().to_run(range.start - ix));
            }
            debug_assert!(text.is_char_boundary(range.end));
            runs.push(
                default_style
                    .clone()
                    .highlight(highlight)
                    .to_run(range.len()),
            );
            ix = range.end;
        }
        if ix < text.len() {
            runs.push(default_style.to_run(text.len() - ix));
        }
        runs
    }

    /// Set the text runs for this piece of text.
    pub fn with_runs(mut self, runs: Vec<TextRun>) -> Self {
        let mut text = &**self.text;
        for run in &runs {
            text = text.get(run.len..).expect("invalid text run");
        }
        assert!(text.is_empty(), "invalid text run");
        self.runs = Some(runs);
        self
    }
}

impl Element for StyledText {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let runs = self.runs.take().or_else(|| {
            self.delayed_highlights.take().map(|delayed_highlights| {
                Self::compute_runs(&self.text, &window.text_style(), delayed_highlights)
            })
        });

        let layout_id = self.layout.layout(self.text.clone(), runs, window, cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        self.layout.prepaint(bounds, &self.text)
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.layout.paint(&self.text, window, cx)
    }
}

impl IntoElement for StyledText {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// The Layout for TextElement. This can be used to map indices to pixels and vice versa.
#[derive(Default, Clone)]
pub struct TextLayout(Rc<RefCell<Option<TextLayoutInner>>>);

struct TextLayoutInner {
    len: usize,
    lines: SmallVec<[WrappedLine; 1]>,
    line_height: Pixels,
    wrap_width: Option<Pixels>,
    size: Option<Size<Pixels>>,
    bounds: Option<Bounds<Pixels>>,
}

impl TextLayout {
    fn layout(
        &self,
        text: SharedString,
        runs: Option<Vec<TextRun>>,
        window: &mut Window,
        _: &mut App,
    ) -> LayoutId {
        let text_style = window.text_style();
        let font_size = text_style.font_size.to_pixels(window.rem_size());
        let line_height = text_style
            .line_height
            .to_pixels(font_size.into(), window.rem_size());

        let runs = if let Some(runs) = runs {
            runs
        } else {
            vec![text_style.to_run(text.len())]
        };

        let content_hash = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            font_size.0.to_bits().hash(&mut hasher);
            line_height.0.to_bits().hash(&mut hasher);
            hasher.finish()
        };

        window.request_measured_layout_cached(Default::default(), content_hash, {
            let element_state = self.clone();

            move |known_dimensions, available_space, window, cx| {
                let wrap_width = if text_style.white_space == WhiteSpace::Normal {
                    known_dimensions.width.or(match available_space.width {
                        crate::AvailableSpace::Definite(x) => Some(x),
                        _ => None,
                    })
                } else {
                    None
                };

                let (truncate_width, truncation_affix, truncate_from) =
                    if let Some(text_overflow) = text_style.text_overflow.clone() {
                        let width = known_dimensions.width.or(match available_space.width {
                            crate::AvailableSpace::Definite(x) => match text_style.line_clamp {
                                Some(max_lines) => Some(x * max_lines),
                                None => Some(x),
                            },
                            _ => None,
                        });

                        match text_overflow {
                            TextOverflow::Truncate(s) => (width, s, TruncateFrom::End),
                            TextOverflow::TruncateStart(s) => (width, s, TruncateFrom::Start),
                        }
                    } else {
                        (None, "".into(), TruncateFrom::End)
                    };

                // Only use cached layout if:
                // 1. We have a cached size
                // 2. wrap_width matches (or both are None)
                // 3. truncate_width is None (if truncate_width is Some, we need to re-layout
                //    because the previous layout may have been computed without truncation)
                if let Some(text_layout) = element_state.0.borrow().as_ref()
                    && let Some(size) = text_layout.size
                    && (wrap_width.is_none() || wrap_width == text_layout.wrap_width)
                    && truncate_width.is_none()
                {
                    return size;
                }

                let mut line_wrapper = cx.text_system().line_wrapper(text_style.font(), font_size);
                let (text, runs) = if let Some(truncate_width) = truncate_width {
                    line_wrapper.truncate_line(
                        text.clone(),
                        truncate_width,
                        &truncation_affix,
                        &runs,
                        truncate_from,
                    )
                } else {
                    (text.clone(), Cow::Borrowed(&*runs))
                };
                let len = text.len();

                let Some(lines) = window
                    .text_system()
                    .shape_text(
                        text,
                        font_size,
                        &runs,
                        wrap_width,            // Wrap if we know the width.
                        text_style.line_clamp, // Limit the number of lines if line_clamp is set.
                    )
                    .log_err()
                else {
                    element_state.0.borrow_mut().replace(TextLayoutInner {
                        lines: Default::default(),
                        len: 0,
                        line_height,
                        wrap_width,
                        size: Some(Size::default()),
                        bounds: None,
                    });
                    return Size::default();
                };

                let mut size: Size<Pixels> = Size::default();
                for line in &lines {
                    let line_size = line.size(line_height);
                    size.height += line_size.height;
                    size.width = size.width.max(line_size.width).ceil();
                }

                element_state.0.borrow_mut().replace(TextLayoutInner {
                    lines,
                    len,
                    line_height,
                    wrap_width,
                    size: Some(size),
                    bounds: None,
                });

                size
            }
        })
    }

    fn prepaint(&self, bounds: Bounds<Pixels>, text: &str) {
        let mut element_state = self.0.borrow_mut();
        let element_state = element_state
            .as_mut()
            .with_context(|| format!("measurement has not been performed on {text}"))
            .unwrap();
        element_state.bounds = Some(bounds);
    }

    fn paint(&self, text: &str, window: &mut Window, cx: &mut App) {
        let element_state = self.0.borrow();
        let element_state = element_state
            .as_ref()
            .with_context(|| format!("measurement has not been performed on {text}"))
            .unwrap();
        let bounds = element_state
            .bounds
            .with_context(|| format!("prepaint has not been performed on {text}"))
            .unwrap();

        let line_height = element_state.line_height;
        let mut line_origin = bounds.origin;
        let text_style = window.text_style();
        for line in &element_state.lines {
            line.paint_background(
                line_origin,
                line_height,
                text_style.text_align,
                Some(bounds),
                window,
                cx,
            )
            .log_err();
            line.paint(
                line_origin,
                line_height,
                text_style.text_align,
                Some(bounds),
                window,
                cx,
            )
            .log_err();
            line_origin.y += line.size(line_height).height;
        }
    }

    /// Get the byte index into the input of the pixel position.
    pub fn index_for_position(&self, mut position: Point<Pixels>) -> Result<usize, usize> {
        let element_state = self.0.borrow();
        let element_state = element_state
            .as_ref()
            .expect("measurement has not been performed");
        let bounds = element_state
            .bounds
            .expect("prepaint has not been performed");

        if position.y < bounds.top() {
            return Err(0);
        }

        let line_height = element_state.line_height;
        let mut line_origin = bounds.origin;
        let mut line_start_ix = 0;
        for line in &element_state.lines {
            let line_bottom = line_origin.y + line.size(line_height).height;
            if position.y > line_bottom {
                line_origin.y = line_bottom;
                line_start_ix += line.len() + 1;
            } else {
                let position_within_line = position - line_origin;
                match line.index_for_position(position_within_line, line_height) {
                    Ok(index_within_line) => return Ok(line_start_ix + index_within_line),
                    Err(index_within_line) => return Err(line_start_ix + index_within_line),
                }
            }
        }

        Err(line_start_ix.saturating_sub(1))
    }

    /// Get the pixel position for the given byte index.
    pub fn position_for_index(&self, index: usize) -> Option<Point<Pixels>> {
        let element_state = self.0.borrow();
        let element_state = element_state
            .as_ref()
            .expect("measurement has not been performed");
        let bounds = element_state
            .bounds
            .expect("prepaint has not been performed");
        let line_height = element_state.line_height;

        let mut line_origin = bounds.origin;
        let mut line_start_ix = 0;

        for line in &element_state.lines {
            let line_end_ix = line_start_ix + line.len();
            if index < line_start_ix {
                break;
            } else if index > line_end_ix {
                line_origin.y += line.size(line_height).height;
                line_start_ix = line_end_ix + 1;
                continue;
            } else {
                let ix_within_line = index - line_start_ix;
                return Some(line_origin + line.position_for_index(ix_within_line, line_height)?);
            }
        }

        None
    }

    /// Retrieve the layout for the line containing the given byte index.
    pub fn line_layout_for_index(&self, index: usize) -> Option<Arc<WrappedLineLayout>> {
        let element_state = self.0.borrow();
        let element_state = element_state
            .as_ref()
            .expect("measurement has not been performed");
        let bounds = element_state
            .bounds
            .expect("prepaint has not been performed");
        let line_height = element_state.line_height;

        let mut line_origin = bounds.origin;
        let mut line_start_ix = 0;

        for line in &element_state.lines {
            let line_end_ix = line_start_ix + line.len();
            if index < line_start_ix {
                break;
            } else if index > line_end_ix {
                line_origin.y += line.size(line_height).height;
                line_start_ix = line_end_ix + 1;
                continue;
            } else {
                return Some(line.layout.clone());
            }
        }

        None
    }

    /// The bounds of this layout.
    pub fn bounds(&self) -> Bounds<Pixels> {
        self.0.borrow().as_ref().unwrap().bounds.unwrap()
    }

    /// The line height for this layout.
    pub fn line_height(&self) -> Pixels {
        self.0.borrow().as_ref().unwrap().line_height
    }

    /// The UTF-8 length of the underlying text.
    pub fn len(&self) -> usize {
        self.0.borrow().as_ref().unwrap().len
    }

    /// The text for this layout.
    pub fn text(&self) -> String {
        self.0
            .borrow()
            .as_ref()
            .unwrap()
            .lines
            .iter()
            .map(|s| &s.text)
            .join("\n")
    }

    /// The text for this layout (with soft-wraps as newlines)
    pub fn wrapped_text(&self) -> String {
        let mut accumulator = String::new();

        for wrapped in self.0.borrow().as_ref().unwrap().lines.iter() {
            let mut seen = 0;
            for boundary in wrapped.layout.wrap_boundaries.iter() {
                let index = wrapped.layout.unwrapped_layout.runs[boundary.run_ix].glyphs
                    [boundary.glyph_ix]
                    .index;

                accumulator.push_str(&wrapped.text[seen..index]);
                accumulator.push('\n');
                seen = index;
            }
            accumulator.push_str(&wrapped.text[seen..]);
            accumulator.push('\n');
        }
        // Remove trailing newline
        accumulator.pop();
        accumulator
    }
}

/// A text element that can be interacted with.
pub struct InteractiveText {
    element_id: ElementId,
    text: StyledText,
    click_listener:
        Option<Box<dyn Fn(&[Range<usize>], InteractiveTextClickEvent, &mut Window, &mut App)>>,
    hover_listener: Option<Box<dyn Fn(Option<usize>, MouseMoveEvent, &mut Window, &mut App)>>,
    tooltip_builder: Option<Rc<dyn Fn(usize, &mut Window, &mut App) -> Option<AnyView>>>,
    tooltip_id: Option<TooltipId>,
    clickable_ranges: Vec<Range<usize>>,
}

struct InteractiveTextClickEvent {
    mouse_down_index: usize,
    mouse_up_index: usize,
}

#[doc(hidden)]
#[derive(Default)]
pub struct InteractiveTextState {
    mouse_down_index: Rc<Cell<Option<usize>>>,
    hovered_index: Rc<Cell<Option<usize>>>,
    active_tooltip: Rc<RefCell<Option<ActiveTooltip>>>,
}

/// InteractiveTest is a wrapper around StyledText that adds mouse interactions.
impl InteractiveText {
    /// Creates a new InteractiveText from the given text.
    pub fn new(id: impl Into<ElementId>, text: StyledText) -> Self {
        Self {
            element_id: id.into(),
            text,
            click_listener: None,
            hover_listener: None,
            tooltip_builder: None,
            tooltip_id: None,
            clickable_ranges: Vec::new(),
        }
    }

    /// on_click is called when the user clicks on one of the given ranges, passing the index of
    /// the clicked range.
    pub fn on_click(
        mut self,
        ranges: Vec<Range<usize>>,
        listener: impl Fn(usize, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.click_listener = Some(Box::new(move |ranges, event, window, cx| {
            for (range_ix, range) in ranges.iter().enumerate() {
                if range.contains(&event.mouse_down_index) && range.contains(&event.mouse_up_index)
                {
                    listener(range_ix, window, cx);
                }
            }
        }));
        self.clickable_ranges = ranges;
        self
    }

    /// on_hover is called when the mouse moves over a character within the text, passing the
    /// index of the hovered character, or None if the mouse leaves the text.
    pub fn on_hover(
        mut self,
        listener: impl Fn(Option<usize>, MouseMoveEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.hover_listener = Some(Box::new(listener));
        self
    }

    /// tooltip lets you specify a tooltip for a given character index in the string.
    pub fn tooltip(
        mut self,
        builder: impl Fn(usize, &mut Window, &mut App) -> Option<AnyView> + 'static,
    ) -> Self {
        self.tooltip_builder = Some(Rc::new(builder));
        self
    }
}

impl Element for InteractiveText {
    type RequestLayoutState = ();
    type PrepaintState = Hitbox;

    fn id(&self) -> Option<ElementId> {
        Some(self.element_id.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        self.text
            .request_layout(global_id, inspector_id, window, cx)
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        state: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Hitbox {
        let global_id = *global_id.unwrap();
        window.with_element_state::<InteractiveTextState, _>(
            &global_id,
            |interactive_state, window| {
                let mut interactive_state = interactive_state;

                if let Some(interactive_state) = interactive_state.as_mut() {
                    if self.tooltip_builder.is_some() {
                        self.tooltip_id =
                            set_tooltip_on_window(&interactive_state.active_tooltip, window);
                    } else {
                        // If there is no longer a tooltip builder, remove the active tooltip.
                        interactive_state.active_tooltip.take();
                    }
                }

                self.text
                    .prepaint(Some(&global_id), inspector_id, bounds, state, window, cx);
                let hitbox =
                    window.insert_hitbox_with_fiber(bounds, HitboxBehavior::Normal, global_id);
                (hitbox, interactive_state.unwrap_or_default())
            },
        )
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        hitbox: &mut Hitbox,
        window: &mut Window,
        cx: &mut App,
    ) {
        let global_id = *global_id.unwrap();
        let text_layout = self.text.layout().clone();
        window.with_element_state::<InteractiveTextState, _>(
            &global_id,
            |interactive_state, window| {
                let mut interactive_state = interactive_state.unwrap_or_default();
                if let Some(click_listener) = self.click_listener.take() {
                    let mouse_position = window.mouse_position();
                    if let Ok(ix) = text_layout.index_for_position(mouse_position)
                        && self
                            .clickable_ranges
                            .iter()
                            .any(|range| range.contains(&ix))
                    {
                        window.set_cursor_style(crate::CursorStyle::PointingHand, hitbox)
                    }

                    let text_layout = text_layout.clone();
                    let mouse_down = interactive_state.mouse_down_index.clone();
                    if let Some(mouse_down_index) = mouse_down.get() {
                        let hitbox = hitbox.clone();
                        let clickable_ranges = mem::take(&mut self.clickable_ranges);
                        window.on_mouse_event(
                            move |event: &MouseUpEvent, phase, window: &mut Window, cx| {
                                if phase == DispatchPhase::Bubble && hitbox.is_hovered(window) {
                                    if let Ok(mouse_up_index) =
                                        text_layout.index_for_position(event.position)
                                    {
                                        click_listener(
                                            &clickable_ranges,
                                            InteractiveTextClickEvent {
                                                mouse_down_index,
                                                mouse_up_index,
                                            },
                                            window,
                                            cx,
                                        )
                                    }

                                    mouse_down.take();
                                    window.refresh();
                                }
                            },
                        );
                    } else {
                        let hitbox = hitbox.clone();
                        window.on_mouse_event(move |event: &MouseDownEvent, phase, window, _| {
                            if phase == DispatchPhase::Bubble
                                && hitbox.is_hovered(window)
                                && let Ok(mouse_down_index) =
                                    text_layout.index_for_position(event.position)
                            {
                                mouse_down.set(Some(mouse_down_index));
                                window.refresh();
                            }
                        });
                    }
                }

                window.on_mouse_event({
                    let mut hover_listener = self.hover_listener.take();
                    let hitbox = hitbox.clone();
                    let text_layout = text_layout.clone();
                    let hovered_index = interactive_state.hovered_index.clone();
                    move |event: &MouseMoveEvent, phase, window, cx| {
                        if phase == DispatchPhase::Bubble && hitbox.is_hovered(window) {
                            let current = hovered_index.get();
                            let updated = text_layout.index_for_position(event.position).ok();
                            if current != updated {
                                hovered_index.set(updated);
                                if let Some(hover_listener) = hover_listener.as_ref() {
                                    hover_listener(updated, event.clone(), window, cx);
                                }
                                window.invalidate_fiber_paint(global_id);
                            }
                        }
                    }
                });

                if let Some(tooltip_builder) = self.tooltip_builder.clone() {
                    let active_tooltip = interactive_state.active_tooltip.clone();
                    let build_tooltip = Rc::new({
                        let tooltip_is_hoverable = false;
                        let text_layout = text_layout.clone();
                        move |window: &mut Window, cx: &mut App| {
                            text_layout
                                .index_for_position(window.mouse_position())
                                .ok()
                                .and_then(|position| tooltip_builder(position, window, cx))
                                .map(|view| (view, tooltip_is_hoverable))
                        }
                    });

                    // Use bounds instead of testing hitbox since this is called during prepaint.
                    let check_is_hovered_during_prepaint = Rc::new({
                        let source_bounds = hitbox.bounds;
                        let text_layout = text_layout.clone();
                        let pending_mouse_down = interactive_state.mouse_down_index.clone();
                        move |window: &Window| {
                            text_layout
                                .index_for_position(window.mouse_position())
                                .is_ok()
                                && source_bounds.contains(&window.mouse_position())
                                && pending_mouse_down.get().is_none()
                        }
                    });

                    let check_is_hovered = Rc::new({
                        let hitbox = hitbox.clone();
                        let text_layout = text_layout.clone();
                        let pending_mouse_down = interactive_state.mouse_down_index.clone();
                        move |window: &Window| {
                            text_layout
                                .index_for_position(window.mouse_position())
                                .is_ok()
                                && hitbox.is_hovered(window)
                                && pending_mouse_down.get().is_none()
                        }
                    });

                    register_tooltip_mouse_handlers(
                        &active_tooltip,
                        self.tooltip_id,
                        build_tooltip,
                        check_is_hovered,
                        check_is_hovered_during_prepaint,
                        window,
                    );
                }

                self.text.paint(
                    Some(&global_id),
                    inspector_id,
                    bounds,
                    &mut (),
                    &mut (),
                    window,
                    cx,
                );

                ((), interactive_state)
            },
        );
    }
}

impl IntoElement for InteractiveText {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// Retained render node for text elements.
///
/// This node owns all text-specific data including the text content,
/// optional style runs, and cached layout (shaped lines).
pub(crate) struct TextNode {
    /// The text content.
    text: SharedString,
    /// Optional styled runs.
    runs: Option<Vec<TextRun>>,
    /// Resolved text style captured during layout_begin.
    /// This is the inherited text style from parent elements.
    resolved_text_style: Option<TextStyle>,
    /// Cached intrinsic sizing (min/max content) keyed by `SizingInput`.
    intrinsic: Option<(crate::SizingInput, crate::IntrinsicSize)>,
    /// Cached layout data (populated during measure).
    layout: Option<TextNodeLayout>,
    /// Bounds set during prepaint.
    bounds: Option<Bounds<Pixels>>,
}

/// Cached layout data for TextNode.
struct TextNodeLayout {
    input: crate::SizingInput,
    lines: SmallVec<[WrappedLine; 1]>,
    line_height: Pixels,
    wrap_width: Option<Pixels>,
    truncate_width: Option<Pixels>,
    size: Size<Pixels>,
    text_style: TextStyle,
}

impl TextNode {
    /// Create a new TextNode with the given text.
    pub fn new(text: SharedString, runs: Option<Vec<TextRun>>) -> Self {
        Self {
            text,
            runs,
            resolved_text_style: None,
            intrinsic: None,
            layout: None,
            bounds: None,
        }
    }

    /// Update this node from a descriptor.
    pub fn update_from(&mut self, text: SharedString, runs: Option<Vec<TextRun>>) {
        // Clear cached layout since text content may have changed
        self.layout = None;
        self.intrinsic = None;
        self.text = text;
        self.runs = runs;
    }

    fn sizing_input(
        &self,
        text_style: &TextStyle,
        font_size: Pixels,
        line_height: Pixels,
        runs: &[TextRun],
    ) -> crate::SizingInput {
        let mut content_hasher = FxHasher::default();
        self.text.hash(&mut content_hasher);
        runs.len().hash(&mut content_hasher);
        for run in runs {
            run.len.hash(&mut content_hasher);
            run.font.hash(&mut content_hasher);
        }
        let content_hash = content_hasher.finish();

        let mut style_hasher = FxHasher::default();
        text_style.font().hash(&mut style_hasher);
        font_size.0.to_bits().hash(&mut style_hasher);
        line_height.0.to_bits().hash(&mut style_hasher);
        text_style.line_clamp.hash(&mut style_hasher);
        match text_style.white_space {
            WhiteSpace::Normal => 0u8.hash(&mut style_hasher),
            WhiteSpace::Nowrap => 1u8.hash(&mut style_hasher),
        }
        match &text_style.text_overflow {
            None => 0u8.hash(&mut style_hasher),
            Some(TextOverflow::Truncate(affix)) => {
                1u8.hash(&mut style_hasher);
                affix.hash(&mut style_hasher);
            }
            Some(TextOverflow::TruncateStart(affix)) => {
                2u8.hash(&mut style_hasher);
                affix.hash(&mut style_hasher);
            }
        }
        match text_style.text_align {
            crate::TextAlign::Left => 0u8.hash(&mut style_hasher),
            crate::TextAlign::Center => 1u8.hash(&mut style_hasher),
            crate::TextAlign::Right => 2u8.hash(&mut style_hasher),
        }
        let style_hash = style_hasher.finish();

        crate::SizingInput::new(content_hash, style_hash)
    }

    fn shape_and_store_layout(
        &mut self,
        input: crate::SizingInput,
        text_style: TextStyle,
        wrap_width: Option<Pixels>,
        truncate_width: Option<Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Size<Pixels> {
        let font_size = text_style.font_size.to_pixels(window.rem_size());
        let line_height = text_style
            .line_height
            .to_pixels(font_size.into(), window.rem_size());

        let runs = self
            .runs
            .clone()
            .unwrap_or_else(|| vec![text_style.to_run(self.text.len())]);

        let (truncate_width, truncation_affix, truncate_from) =
            if let Some(text_overflow) = text_style.text_overflow.clone()
                && truncate_width.is_some()
            {
                match text_overflow {
                    TextOverflow::Truncate(s) => (truncate_width, s, TruncateFrom::End),
                    TextOverflow::TruncateStart(s) => (truncate_width, s, TruncateFrom::Start),
                }
            } else {
                (None, "".into(), TruncateFrom::End)
            };

        // Only use cached layout if:
        // - input matches (content/style),
        // - wrap_width matches, and
        // - truncation state matches.
        if let Some(layout) = &self.layout
            && layout.input == input
            && layout.wrap_width == wrap_width
            && layout.truncate_width == truncate_width
        {
            return layout.size;
        }

        let mut line_wrapper = cx.text_system().line_wrapper(text_style.font(), font_size);
        let (text, runs) = if let Some(truncate_width) = truncate_width {
            line_wrapper.truncate_line(
                self.text.clone(),
                truncate_width,
                &truncation_affix,
                &runs,
                truncate_from,
            )
        } else {
            (self.text.clone(), Cow::Borrowed(&*runs))
        };

        let Some(lines) = window
            .text_system()
            .shape_text(text, font_size, &runs, wrap_width, text_style.line_clamp)
            .log_err()
        else {
            let size = Size::default();
            self.layout = Some(TextNodeLayout {
                input,
                lines: Default::default(),
                line_height,
                wrap_width,
                truncate_width,
                size,
                text_style,
            });
            return size;
        };

        let mut size: Size<Pixels> = Size::default();
        for line in &lines {
            let line_size = line.size(line_height);
            size.height += line_size.height;
            size.width = size.width.max(line_size.width).ceil();
        }

        self.layout = Some(TextNodeLayout {
            input,
            lines,
            line_height,
            wrap_width,
            truncate_width,
            size,
            text_style,
        });

        size
    }
}

impl crate::RenderNode for TextNode {
    fn needs_child_bounds(&self) -> bool {
        false
    }

    fn taffy_style(&self, _rem_size: crate::Pixels, _scale_factor: f32) -> taffy::style::Style {
        taffy::style::Style::default()
    }

    fn layout_begin(&mut self, ctx: &mut crate::LayoutCtx) -> crate::LayoutFrame {
        // Capture the inherited text style from the window stack.
        // This is set up by parent Divs during their layout_begin.
        self.resolved_text_style = Some(ctx.window.text_style());
        crate::LayoutFrame {
            handled: true,
            ..Default::default()
        }
    }

    fn compute_intrinsic_size(
        &mut self,
        ctx: &mut crate::SizingCtx,
    ) -> crate::IntrinsicSizeResult {
        let text_style = ctx.window.text_style();
        self.resolved_text_style = Some(text_style.clone());

        let font_size = text_style.font_size.to_pixels(ctx.window.rem_size());
        let line_height = text_style
            .line_height
            .to_pixels(font_size.into(), ctx.window.rem_size());

        let runs = self
            .runs
            .clone()
            .unwrap_or_else(|| vec![text_style.to_run(self.text.len())]);

        let input = self.sizing_input(&text_style, font_size, line_height, &runs);
        if let Some((cached_input, cached_size)) = &self.intrinsic
            && *cached_input == input
        {
            return crate::IntrinsicSizeResult {
                size: cached_size.clone(),
                input,
            };
        }

        // Unwrapped (max-content) size; we currently treat min-content as equivalent to preserve
        // existing behavior. Height-for-width is handled in `resolve_size_query`.
        let size = self.shape_and_store_layout(
            input.clone(),
            text_style.clone(),
            None,
            None,
            ctx.window,
            ctx.cx,
        );

        let intrinsic = crate::IntrinsicSize {
            min_content: size,
            max_content: size,
        };
        self.intrinsic = Some((input.clone(), intrinsic.clone()));

        crate::IntrinsicSizeResult {
            size: intrinsic,
            input,
        }
    }

    fn resolve_size_query(
        &mut self,
        query: crate::SizeQuery,
        cached: &crate::IntrinsicSize,
        ctx: &mut crate::SizingCtx,
    ) -> Size<Pixels> {
        let Some(mut text_style) = self.resolved_text_style.clone().or_else(|| {
            // Best-effort fallback: this can happen when layout is invoked without a preceding
            // intrinsic sizing pass (e.g. legacy in-frame layouts).
            Some(ctx.window.text_style())
        }) else {
            return cached.max_content;
        };

        // For queries that provide a definite width, compute wrapped height for that width.
        match query {
            crate::SizeQuery::MinContent => cached.min_content,
            crate::SizeQuery::MaxContent => cached.max_content,
            crate::SizeQuery::ForHeight(height) => Size {
                width: cached.max_content.width,
                height,
            },
            crate::SizeQuery::ForWidth(width) => {
                let wrap_width = if text_style.white_space == WhiteSpace::Normal {
                    Some(width)
                } else {
                    None
                };

                let truncate_width = if text_style.text_overflow.is_some() {
                    match text_style.line_clamp {
                        Some(max_lines) => Some(width * max_lines),
                        None => Some(width),
                    }
                } else {
                    None
                };

                let font_size = text_style.font_size.to_pixels(ctx.window.rem_size());
                let line_height = text_style
                    .line_height
                    .to_pixels(font_size.into(), ctx.window.rem_size());
                let runs = self
                    .runs
                    .clone()
                    .unwrap_or_else(|| vec![text_style.to_run(self.text.len())]);
                let input = self.sizing_input(&text_style, font_size, line_height, &runs);

                self.shape_and_store_layout(
                    input,
                    text_style,
                    wrap_width,
                    truncate_width,
                    ctx.window,
                    ctx.cx,
                )
            }
            crate::SizeQuery::Definite(size) => {
                let wrap_width = if text_style.white_space == WhiteSpace::Normal {
                    Some(size.width)
                } else {
                    None
                };

                let truncate_width = if text_style.text_overflow.is_some() {
                    match text_style.line_clamp {
                        Some(max_lines) => Some(size.width * max_lines),
                        None => Some(size.width),
                    }
                } else {
                    None
                };

                let font_size = text_style.font_size.to_pixels(ctx.window.rem_size());
                let line_height = text_style
                    .line_height
                    .to_pixels(font_size.into(), ctx.window.rem_size());
                let runs = self
                    .runs
                    .clone()
                    .unwrap_or_else(|| vec![text_style.to_run(self.text.len())]);
                let input = self.sizing_input(&text_style, font_size, line_height, &runs);

                let measured = self.shape_and_store_layout(
                    input,
                    text_style,
                    wrap_width,
                    truncate_width,
                    ctx.window,
                    ctx.cx,
                );

                Size {
                    width: measured.width.min(size.width),
                    height: measured.height.min(size.height),
                }
            }
        }
    }

    fn measure(
        &mut self,
        known: Size<Option<Pixels>>,
        available: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Size<Pixels>> {
        let text_style = self
            .resolved_text_style
            .clone()
            .unwrap_or_else(|| window.text_style());

        let wrap_width = if text_style.white_space == WhiteSpace::Normal {
            known.width.or(match available.width {
                AvailableSpace::Definite(x) => Some(x),
                _ => None,
            })
        } else {
            None
        };

        let truncate_width = if text_style.text_overflow.is_some() {
            known.width.or(match available.width {
                AvailableSpace::Definite(x) => match text_style.line_clamp {
                    Some(max_lines) => Some(x * max_lines),
                    None => Some(x),
                },
                _ => None,
            })
        } else {
            None
        };

        let font_size = text_style.font_size.to_pixels(window.rem_size());
        let line_height = text_style
            .line_height
            .to_pixels(font_size.into(), window.rem_size());
        let runs = self
            .runs
            .clone()
            .unwrap_or_else(|| vec![text_style.to_run(self.text.len())]);
        let input = self.sizing_input(&text_style, font_size, line_height, &runs);

        Some(self.shape_and_store_layout(
            input,
            text_style,
            wrap_width,
            truncate_width,
            window,
            cx,
        ))
    }

    fn prepaint_begin(&mut self, ctx: &mut crate::PrepaintCtx) -> crate::PrepaintFrame {
        self.bounds = Some(ctx.bounds);

        crate::PrepaintFrame {
            handled: true,
            skip_children: true,
            ..Default::default()
        }
    }

    fn prepaint_end(&mut self, _ctx: &mut crate::PrepaintCtx, _frame: crate::PrepaintFrame) {
        // Nothing to pop for text
    }

    fn paint_begin(&mut self, ctx: &mut crate::PaintCtx) -> crate::PaintFrame {
        if let Some(ref layout) = self.layout {
            // Use paint-time bounds directly. Text depends on up-to-date bounds for correct positioning
            // (e.g. after window resizes). Prepaint may be replayed, so relying solely on cached
            // prepaint bounds can lead to stale coordinates.
            let bounds = ctx.bounds;
            self.bounds = Some(bounds);

            let mut line_origin = bounds.origin;
            for line in &layout.lines {
                line.paint_background(
                    line_origin,
                    layout.line_height,
                    layout.text_style.text_align,
                    Some(bounds),
                    ctx.window,
                    ctx.cx,
                )
                .log_err();
                line.paint(
                    line_origin,
                    layout.line_height,
                    layout.text_style.text_align,
                    Some(bounds),
                    ctx.window,
                    ctx.cx,
                )
                .log_err();
                line_origin.y += line.size(layout.line_height).height;
            }
        }

        crate::PaintFrame {
            handled: true,
            skip_children: true,
            ..Default::default()
        }
    }

    fn paint_end(&mut self, _ctx: &mut crate::PaintCtx, _frame: crate::PaintFrame) {
        // Nothing to pop for text
    }

    // Uses default downcasting implementations.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{self as gpui, div, px, point, size, Context, Render, RenderNode, TestAppContext};

    struct RootView;

    impl Render for RootView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
        }
    }

    #[gpui::test]
    fn test_text_node_update_render_node_preserves_layout_when_unchanged(cx: &mut TestAppContext) {
        let (_view, cx) = cx.add_window_view(|_, _| RootView);

        cx.update(|window, app| {
            let known = Size {
                width: None,
                height: None,
            };
            let available = Size {
                width: AvailableSpace::Definite(px(500.)),
                height: AvailableSpace::Definite(px(500.)),
            };

            // &'static str updates should preserve cached layout when unchanged.
            let mut node = TextNode::new(SharedString::from("hello"), None);
            node.measure(known, available, window, app);
            assert!(node.layout.is_some());
            let mut element: &'static str = "hello";
            match element.update_render_node(&mut node, window, app) {
                Some(update) => assert!(!update.any_change()),
                None => panic!("expected &'static str to update an existing TextNode"),
            }
            assert!(node.layout.is_some());

            element = "world";
            match element.update_render_node(&mut node, window, app) {
                Some(update) => assert!(update.layout_changed && update.paint_changed),
                None => panic!("expected &'static str to update an existing TextNode"),
            }
            assert!(node.layout.is_none());

            // SharedString updates should preserve cached layout when unchanged.
            let mut node = TextNode::new(SharedString::from("hello"), None);
            node.measure(known, available, window, app);
            assert!(node.layout.is_some());
            let mut element = SharedString::from("hello");
            match element.update_render_node(&mut node, window, app) {
                Some(update) => assert!(!update.any_change()),
                None => panic!("expected SharedString to update an existing TextNode"),
            }
            assert!(node.layout.is_some());

            element = SharedString::from("world");
            match element.update_render_node(&mut node, window, app) {
                Some(update) => assert!(update.layout_changed && update.paint_changed),
                None => panic!("expected SharedString to update an existing TextNode"),
            }
            assert!(node.layout.is_none());
        });
    }

    #[gpui::test]
    fn test_text_node_paint_begin_uses_paint_bounds(cx: &mut TestAppContext) {
        let (_view, cx) = cx.add_window_view(|_, _| RootView);

        cx.update(|window, app| {
            let known = Size {
                width: None,
                height: None,
            };
            let available = Size {
                width: AvailableSpace::Definite(px(500.)),
                height: AvailableSpace::Definite(px(500.)),
            };

            let mut node = TextNode::new(SharedString::from("hello"), None);
            node.measure(known, available, window, app);

            let old_bounds = Bounds::new(point(px(1.), px(2.)), size(px(3.), px(4.)));
            let new_bounds = Bounds::new(point(px(10.), px(20.)), size(px(30.), px(40.)));
            node.bounds = Some(old_bounds);

            window.invalidator.set_phase(crate::window::DrawPhase::Paint);
            window.next_frame.scene.begin_frame();

            let fiber_id = window.fiber.tree.create_placeholder_fiber();
            let mut paint_ctx = crate::PaintCtx {
                fiber_id,
                bounds: new_bounds,
                child_bounds: Vec::new(),
                inspector_id: None,
                window,
                cx: app,
            };
            node.paint_begin(&mut paint_ctx);

            assert_eq!(
                node.bounds,
                Some(new_bounds),
                "TextNode must use paint-time bounds to avoid stale positioning when prepaint is replayed"
            );
        });
    }
}
