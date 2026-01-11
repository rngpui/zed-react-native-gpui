//! Elements are the workhorses of GPUI. They are responsible for laying out and painting all of
//! the contents of a window. Elements form a tree and are laid out according to the web layout
//! standards as implemented by [taffy](https://github.com/DioxusLabs/taffy). Most of the time,
//! you won't need to interact with this module or these APIs directly. Elements provide their
//! own APIs and GPUI, or other element implementation, uses the APIs in this module to convert
//! that element tree into the pixels you see on the screen.
//!
//! # Element Basics
//!
//! Elements are constructed by calling [`Render::render()`] on the root view of the window,
//! which recursively constructs the element tree from the current state of the application,.
//! These elements are then laid out by Taffy, and painted to the screen according to their own
//! implementation of [`Element::paint()`]. Before the start of the next frame, the entire element
//! tree and any callbacks they have registered with GPUI are dropped and the process repeats.
//!
//! But some state is too simple and voluminous to store in every view that needs it, e.g.
//! whether a hover has been started or not. For this, GPUI provides the [`Element::PrepaintState`], associated type.
//!
//! # Implementing your own elements
//!
//! Elements are intended to be the low level, imperative API to GPUI. They are responsible for upholding,
//! or breaking, GPUI's features as they deem necessary. As an example, most GPUI elements are expected
//! to stay in the bounds that their parent element gives them. But with [`Window::with_content_mask`],
//! you can ignore this restriction and paint anywhere inside of the window's bounds. This is useful for overlays
//! and popups and anything else that shows up 'on top' of other elements.
//! With great power, comes great responsibility.
//!
//! However, most of the time, you won't need to implement your own elements. GPUI provides a number of
//! elements that should cover most common use cases out of the box and it's recommended that you use those
//! to construct `components`, using the [`RenderOnce`] trait and the `#[derive(IntoElement)]` macro. Only implement
//! elements when you need to take manual control of the layout and painting process, such as when using
//! your own custom layout algorithm or rendering a code editor.

use crate::{
    AnyView, App, AvailableSpace, Bounds, Context, ElementId, EntityId, FocusHandle,
    InspectorElementId, LayoutId, Pixels, Point, Size, Style, StyleRefinement, UpdateResult, VKey,
    Window, util::FluentBuilder,
};
use std::{
    any::{Any, type_name},
    mem, panic,
};
use taffy::tree::NodeId;

/// Implemented by types that participate in laying out and painting the contents of a window.
/// Elements form a tree and are laid out according to web-based layout rules, as implemented by Taffy.
/// You can create custom elements by implementing this trait, see the module-level documentation
/// for more details.
pub trait Element: 'static + IntoElement {
    /// The type of state returned from [`Element::request_layout`]. A mutable reference to this state is subsequently
    /// provided to [`Element::prepaint`] and [`Element::paint`].
    type RequestLayoutState: 'static;

    /// The type of state returned from [`Element::prepaint`]. A mutable reference to this state is subsequently
    /// provided to [`Element::paint`].
    type PrepaintState: 'static;

    /// If this element has a unique identifier, return it here. This is used to track elements across frames, and
    /// will cause a GlobalElementId to be passed to the request_layout, prepaint, and paint methods.
    ///
    /// The global id can in turn be used to access state that's connected to an element with the same id across
    /// frames. This id must be unique among children of the first containing element with an id.
    fn id(&self) -> Option<ElementId>;

    /// Source location where this element was constructed, used to disambiguate elements in the
    /// inspector and navigate to their source code.
    fn source_location(&self) -> Option<&'static panic::Location<'static>>;

    /// Before an element can be painted, we need to know where it's going to be and how big it is.
    /// Use this method to request a layout from Taffy and initialize the element's state.
    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState);

    /// After laying out an element, we need to commit its bounds to the current frame for hitbox
    /// purposes. The state argument is the same state that was returned from [`Element::request_layout()`].
    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState;

    /// Once layout has been completed, this method will be called to paint the element to the screen.
    /// The state argument is the same state that was returned from [`Element::request_layout()`].
    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    );

    /// Convert this element into a dynamically-typed [`AnyElement`].
    fn into_any(self) -> AnyElement {
        AnyElement::new(self)
    }

    /// Returns the reconciliation key for this element.
    /// Elements with the same key are considered the "same" element across frames.
    /// Default returns `VKey::None`.
    fn fiber_key(&self) -> VKey {
        VKey::None
    }

    /// Returns a slice of this element's children for fiber tree construction.
    /// Default returns an empty slice.
    fn fiber_children(&self) -> &[AnyElement] {
        &[]
    }

    /// Returns a mutable slice of this element's children.
    /// Default returns an empty slice.
    fn fiber_children_mut(&mut self) -> &mut [AnyElement] {
        &mut []
    }

    /// Returns this element's cached style, if any.
    fn cached_style(&self) -> Option<&StyleRefinement> {
        None
    }

    /// Returns a dynamic view handle if this element represents a view.
    fn as_any_view(&self) -> Option<AnyView> {
        None
    }

    /// Creates a new render node for this element.
    ///
    /// Return `Some(node)` if this element supports retained rendering.
    /// Return `None` for legacy/opaque elements that don't support caching.
    ///
    /// Retained nodes persist in the fiber tree and own element-specific state
    /// (interactivity, scroll position, caches, etc.). They enable incremental
    /// rendering by implementing the scope-based prepaint/paint lifecycle.
    ///
    /// Default returns `None`, meaning the element is treated as opaque
    /// (always dirty, no caching).
    fn create_render_node(&mut self) -> Option<Box<dyn crate::RenderNode>> {
        None
    }

    /// Updates an existing render node with this element's current data.
    ///
    /// Called during reconciliation when a matching node already exists.
    /// Returns `Some(UpdateResult)` if the update succeeded (node type matched),
    /// `None` if the node type didn't match (caller should create a new node).
    ///
    /// The `UpdateResult` indicates what changed, which the fiber system uses
    /// to set appropriate dirty flags:
    /// - `layout_changed`: layout needs to be recomputed
    /// - `paint_changed`: prepaint and paint need to re-run
    ///
    /// IMPORTANT: Implementations must only consume element data (via `take_*`
    /// methods) when the downcast succeeds. If the downcast fails, the element
    /// data must remain intact so `create_render_node` can be called.
    ///
    /// Default implementation returns `None` (no render node support).
    fn update_render_node(
        &mut self,
        _node: &mut dyn crate::RenderNode,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Option<UpdateResult> {
        None
    }

    /// Expand this element into its inner element, if applicable.
    ///
    /// This is used to make wrapper elements like `Component<C>` transparent
    /// to the fiber system. When an element is expanded, its inner element
    /// is used directly for reconciliation instead of the wrapper.
    ///
    /// Returns `Some(inner_element)` if this element should be expanded,
    /// `None` if it should be processed as-is.
    ///
    /// Default returns `None` (no expansion).
    fn try_expand(&mut self, _window: &mut Window, _cx: &mut App) -> Option<AnyElement> {
        None
    }

    /// Returns true if this element requires fiber-backed layout.
    ///
    /// Elements that have render nodes (Div, Svg, etc.) require the fiber path
    /// and will panic if their legacy `request_layout` is called. This method
    /// allows callers to check before deciding which layout path to use.
    ///
    /// Default returns `false` (legacy layout is supported).
    fn requires_fiber_layout(&self) -> bool {
        false
    }
}

/// Implemented by any type that can be converted into an element.
pub trait IntoElement: Sized {
    /// The specific type of element into which the implementing type is converted.
    /// Useful for converting other types into elements automatically, like Strings
    type Element: Element;

    /// Convert self into a type that implements [`Element`].
    fn into_element(self) -> Self::Element;

    /// Convert self into a dynamically-typed [`AnyElement`].
    fn into_any_element(self) -> AnyElement {
        self.into_element().into_any()
    }
}

impl<T: IntoElement> FluentBuilder for T {}

/// An object that can be drawn to the screen. This is the trait that distinguishes "views" from
/// other entities. Views are `Entity`'s which `impl Render` and drawn to the screen.
pub trait Render: 'static + Sized {
    /// Render this view into an element tree.
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement;
}

impl Render for Empty {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        Empty
    }
}

/// You can derive [`IntoElement`] on any type that implements this trait.
/// It is used to construct reusable `components` out of plain data. Think of
/// components as a recipe for a certain pattern of elements. RenderOnce allows
/// you to invoke this pattern, without breaking the fluent builder pattern of
/// the element APIs.
pub trait RenderOnce: 'static {
    /// Render this component into an element tree. Note that this method
    /// takes ownership of self, as compared to [`Render::render()`] method
    /// which takes a mutable reference.
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement;
}

/// This is a helper trait to provide a uniform interface for constructing elements that
/// can accept any number of any kind of child elements
pub trait ParentElement {
    /// Extend this element's children with the given child elements.
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>);

    /// Add a single child element to this element.
    fn child(mut self, child: impl IntoElement) -> Self
    where
        Self: Sized,
    {
        self.extend(std::iter::once(child.into_any_element()));
        self
    }

    /// Add multiple child elements to this element.
    fn children(mut self, children: impl IntoIterator<Item = impl IntoElement>) -> Self
    where
        Self: Sized,
    {
        self.extend(children.into_iter().map(|child| child.into_any_element()));
        self
    }
}

/// An element for rendering components. An implementation detail of the [`IntoElement`] derive macro
/// for [`RenderOnce`]
#[doc(hidden)]
pub struct Component<C: RenderOnce> {
    component: Option<C>,
    #[cfg(debug_assertions)]
    source: &'static core::panic::Location<'static>,
}

impl<C: RenderOnce> Component<C> {
    /// Create a new component from the given RenderOnce type.
    #[track_caller]
    pub fn new(component: C) -> Self {
        Component {
            component: Some(component),
            #[cfg(debug_assertions)]
            source: core::panic::Location::caller(),
        }
    }
}

impl<C: RenderOnce> Element for Component<C> {
    type RequestLayoutState = AnyElement;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        #[cfg(debug_assertions)]
        return Some(self.source);

        #[cfg(not(debug_assertions))]
        return None;
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        window.with_global_id(ElementId::Name(type_name::<C>().into()), |_, window| {
            let mut element = self
                .component
                .take()
                .unwrap()
                .render(window, cx)
                .into_any_element();

            let layout_id = if element.requires_fiber_layout()
                && window.fiber.legacy_layout_parent.is_some()
            {
                window.layout_element_in_legacy_context(&mut element, cx)
            } else {
                element.request_layout(window, cx)
            };
            (layout_id, element)
        })
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        element: &mut AnyElement,
        window: &mut Window,
        cx: &mut App,
    ) {
        if element.requires_fiber_layout() {
            return;
        }
        window.with_global_id(ElementId::Name(type_name::<C>().into()), |_, window| {
            element.prepaint(window, cx);
        })
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        element: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        if element.requires_fiber_layout() {
            return;
        }
        window.with_global_id(ElementId::Name(type_name::<C>().into()), |_, window| {
            element.paint(window, cx);
        })
    }

    fn try_expand(&mut self, window: &mut Window, cx: &mut App) -> Option<AnyElement> {
        self.component
            .take()
            .map(|c| c.render(window, cx).into_any_element())
    }
}

impl<C: RenderOnce> IntoElement for Component<C> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// A globally unique identifier for an element, used to track state across frames.
pub type GlobalElementId = NodeId;

pub(crate) trait ElementObject {
    fn inner_element(&mut self) -> &mut dyn Any;

    fn request_layout(&mut self, window: &mut Window, cx: &mut App) -> LayoutId;

    fn prepaint(&mut self, window: &mut Window, cx: &mut App);

    fn paint(&mut self, window: &mut Window, cx: &mut App);

    fn layout_as_root(
        &mut self,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Size<Pixels>;

    fn reset(&mut self);

    fn fiber_key(&self) -> VKey;
    fn fiber_children(&self) -> &[AnyElement];
    fn fiber_children_mut(&mut self) -> &mut [AnyElement];
    fn cached_style(&self) -> Option<&StyleRefinement>;
    fn as_any_view(&self) -> Option<AnyView>;
    fn create_render_node(&mut self) -> Option<Box<dyn crate::RenderNode>>;
    fn update_render_node(
        &mut self,
        node: &mut dyn crate::RenderNode,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<UpdateResult>;
    fn try_expand(&mut self, window: &mut Window, cx: &mut App) -> Option<AnyElement>;
    fn requires_fiber_layout(&self) -> bool;
}

/// A wrapper around an implementer of [`Element`] that allows it to be drawn in a window.
pub struct Drawable<E: Element> {
    /// The drawn element.
    pub element: E,
    phase: ElementDrawPhase<E::RequestLayoutState, E::PrepaintState>,
}

#[derive(Default)]
enum ElementDrawPhase<RequestLayoutState, PrepaintState> {
    #[default]
    Start,
    RequestLayout {
        layout_id: LayoutId,
        global_id: Option<GlobalElementId>,
        inspector_id: Option<InspectorElementId>,
        request_layout: RequestLayoutState,
    },
    LayoutComputed {
        layout_id: LayoutId,
        global_id: Option<GlobalElementId>,
        inspector_id: Option<InspectorElementId>,
        available_space: Size<AvailableSpace>,
        request_layout: RequestLayoutState,
    },
    Prepaint {
        global_id: Option<GlobalElementId>,
        inspector_id: Option<InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: RequestLayoutState,
        prepaint: PrepaintState,
    },
    Painted,
}

/// A wrapper around an implementer of [`Element`] that allows it to be drawn in a window.
impl<E: Element> Drawable<E> {
    pub(crate) fn new(element: E) -> Self {
        Drawable {
            element,
            phase: ElementDrawPhase::Start,
        }
    }

    fn request_layout(&mut self, window: &mut Window, cx: &mut App) -> LayoutId {
        log::debug!("[DRAWABLE] request_layout called");
        match mem::take(&mut self.phase) {
            ElementDrawPhase::Start => {
                let global_id = window
                    .current_fiber_id()
                    .unwrap_or_else(|| window.fiber.tree.create_placeholder_fiber());
                log::debug!(
                    "[DRAWABLE] Created/retrieved global_id={:?} for element",
                    global_id
                );

                let inspector_id;
                #[cfg(any(feature = "inspector", debug_assertions))]
                {
                    inspector_id = self.element.source_location().map(|source| {
                        let path = crate::InspectorElementPath {
                            global_id,
                            source_location: source,
                        };
                        window.build_inspector_element_id(path)
                    });
                }
                #[cfg(not(any(feature = "inspector", debug_assertions)))]
                {
                    inspector_id = None;
                }

                let (layout_id, request_layout) =
                    window.with_element_id_stack(&global_id, |window| {
                        self.element.request_layout(
                            Some(&global_id),
                            inspector_id.as_ref(),
                            window,
                            cx,
                        )
                    });

                self.phase = ElementDrawPhase::RequestLayout {
                    layout_id,
                    global_id: Some(global_id),
                    inspector_id,
                    request_layout,
                };
                log::debug!(
                    "[DRAWABLE] Transitioned to RequestLayout phase, layout_id={:?}",
                    layout_id
                );
                layout_id
            }
            _ => {
                log::error!("[DRAWABLE] PANIC: request_layout called more than once!");
                panic!("must call request_layout only once")
            }
        }
    }

    pub(crate) fn prepaint(&mut self, window: &mut Window, cx: &mut App) {
        let current_phase = format!("{:?}", mem::discriminant(&self.phase));
        log::debug!(
            "[DRAWABLE] prepaint called, current_phase={}",
            current_phase
        );
        match mem::take(&mut self.phase) {
            ElementDrawPhase::RequestLayout {
                layout_id,
                global_id,
                inspector_id,
                mut request_layout,
            }
            | ElementDrawPhase::LayoutComputed {
                layout_id,
                global_id,
                inspector_id,
                mut request_layout,
                ..
            } => {
                log::debug!(
                    "[DRAWABLE] Transitioning to Prepaint phase, global_id={:?}, layout_id={:?}",
                    global_id,
                    layout_id
                );
                let bounds = window.layout_bounds(layout_id);
                let prepaint = if let Some(global_id) = global_id {
                    window.with_element_id_stack(&global_id, |window| {
                        self.element.prepaint(
                            Some(&global_id),
                            inspector_id.as_ref(),
                            bounds,
                            &mut request_layout,
                            window,
                            cx,
                        )
                    })
                } else {
                    self.element.prepaint(
                        None,
                        inspector_id.as_ref(),
                        bounds,
                        &mut request_layout,
                        window,
                        cx,
                    )
                };

                self.phase = ElementDrawPhase::Prepaint {
                    global_id,
                    inspector_id,
                    bounds,
                    request_layout,
                    prepaint,
                };
                log::debug!(
                    "[DRAWABLE] Successfully transitioned to Prepaint phase, global_id={:?}",
                    global_id
                );
            }
            phase => {
                log::error!(
                    "[DRAWABLE] PANIC: prepaint called in invalid state! current_phase={:?}",
                    mem::discriminant(&phase)
                );
                panic!("must call request_layout before prepaint")
            }
        }
    }

    pub(crate) fn paint(
        &mut self,
        window: &mut Window,
        cx: &mut App,
    ) -> (E::RequestLayoutState, E::PrepaintState) {
        log::debug!("[DRAWABLE] paint called");
        match mem::take(&mut self.phase) {
            ElementDrawPhase::Prepaint {
                global_id,
                inspector_id,
                bounds,
                mut request_layout,
                mut prepaint,
                ..
            } => {
                log::debug!("[DRAWABLE] Painting element, global_id={:?}", global_id);
                if let Some(global_id) = global_id {
                    window.with_element_id_stack(&global_id, |window| {
                        self.element.paint(
                            Some(&global_id),
                            inspector_id.as_ref(),
                            bounds,
                            &mut request_layout,
                            &mut prepaint,
                            window,
                            cx,
                        );
                    });
                } else {
                    self.element.paint(
                        None,
                        inspector_id.as_ref(),
                        bounds,
                        &mut request_layout,
                        &mut prepaint,
                        window,
                        cx,
                    );
                }

                self.phase = ElementDrawPhase::Painted;
                log::debug!("[DRAWABLE] Transitioned to Painted phase");
                (request_layout, prepaint)
            }
            phase => {
                log::error!(
                    "[DRAWABLE] PANIC: paint called in invalid state! current_phase={:?}",
                    mem::discriminant(&phase)
                );
                panic!("must call prepaint before paint")
            }
        }
    }

    pub(crate) fn layout_as_root(
        &mut self,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Size<Pixels> {
        if matches!(&self.phase, ElementDrawPhase::Start) {
            self.request_layout(window, cx);
        }

        let layout_id = match mem::take(&mut self.phase) {
            ElementDrawPhase::RequestLayout {
                layout_id,
                global_id,
                inspector_id,
                request_layout,
            } => {
                window.compute_layout(layout_id, available_space, cx);
                self.phase = ElementDrawPhase::LayoutComputed {
                    layout_id,
                    global_id,
                    inspector_id,
                    available_space,
                    request_layout,
                };
                layout_id
            }
            ElementDrawPhase::LayoutComputed {
                layout_id,
                global_id,
                inspector_id,
                available_space: prev_available_space,
                request_layout,
            } => {
                if available_space != prev_available_space {
                    window.compute_layout(layout_id, available_space, cx);
                }
                self.phase = ElementDrawPhase::LayoutComputed {
                    layout_id,
                    global_id,
                    inspector_id,
                    available_space,
                    request_layout,
                };
                layout_id
            }
            _ => panic!("cannot measure after painting"),
        };

        window.layout_bounds(layout_id).size
    }
}

impl<E> ElementObject for Drawable<E>
where
    E: Element,
    E::RequestLayoutState: 'static,
{
    fn inner_element(&mut self) -> &mut dyn Any {
        &mut self.element
    }

    fn request_layout(&mut self, window: &mut Window, cx: &mut App) -> LayoutId {
        Drawable::request_layout(self, window, cx)
    }

    fn prepaint(&mut self, window: &mut Window, cx: &mut App) {
        Drawable::prepaint(self, window, cx);
    }

    fn paint(&mut self, window: &mut Window, cx: &mut App) {
        Drawable::paint(self, window, cx);
    }

    fn layout_as_root(
        &mut self,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Size<Pixels> {
        Drawable::layout_as_root(self, available_space, window, cx)
    }

    fn reset(&mut self) {
        self.phase = ElementDrawPhase::Start;
    }

    fn fiber_key(&self) -> VKey {
        let key = self.element.fiber_key();
        if key != VKey::None {
            return key;
        }

        // Preserve legacy `Element::id()` semantics by treating ids as reconciliation keys.
        //
        // Many existing GPUI elements (including `Div`) set an id via the public `.id(...)`
        // builder API but do not override `fiber_key`. In the retained fiber architecture,
        // failing to key by id would cause those elements to be reconciled positionally,
        // leading to unnecessary churn and incorrect retention behavior across structural changes.
        self.element
            .id()
            .map(VKey::Element)
            .unwrap_or(VKey::None)
    }

    fn fiber_children(&self) -> &[AnyElement] {
        self.element.fiber_children()
    }

    fn fiber_children_mut(&mut self) -> &mut [AnyElement] {
        self.element.fiber_children_mut()
    }

    fn cached_style(&self) -> Option<&StyleRefinement> {
        self.element.cached_style()
    }

    fn as_any_view(&self) -> Option<AnyView> {
        self.element.as_any_view()
    }

    fn create_render_node(&mut self) -> Option<Box<dyn crate::RenderNode>> {
        self.element.create_render_node()
    }

    fn update_render_node(
        &mut self,
        node: &mut dyn crate::RenderNode,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<UpdateResult> {
        self.element.update_render_node(node, window, cx)
    }

    fn try_expand(&mut self, window: &mut Window, cx: &mut App) -> Option<AnyElement> {
        self.element.try_expand(window, cx)
    }

    fn requires_fiber_layout(&self) -> bool {
        self.element.requires_fiber_layout()
    }
}

#[allow(missing_docs)]
#[derive(Clone)]
pub(crate) struct ViewData {
    pub view: AnyView,
    pub has_cached_child: bool,
}

impl ViewData {
    pub fn new(view: AnyView) -> Self {
        Self {
            view,
            has_cached_child: false,
        }
    }

    pub fn entity_id(&self) -> EntityId {
        self.view.entity_id()
    }
}

pub(crate) struct LegacyElement {
    pub(crate) element: Option<Box<dyn ElementObject>>,
    pub(crate) element_id: Option<ElementId>,
    pub(crate) type_name: &'static str,
}

impl Clone for LegacyElement {
    fn clone(&self) -> Self {
        LegacyElement {
            element: None,
            element_id: self.element_id.clone(),
            type_name: self.type_name,
        }
    }
}

/// Modifiers that can be applied to any element descriptor.
///
/// These are metadata that affect how the element is rendered without
/// creating additional nodes in the retained fiber tree. They are
/// lowered into fiber flags during reconciliation.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct ElementModifiers {
    /// If set, this element's subtree is painted in a deferred pass
    /// (after all non-deferred content). Higher priority = painted later (on top).
    pub deferred_priority: Option<usize>,
}

/// A dynamically typed element that can be used to store any element type.
pub struct AnyElement {
    pub(crate) inner: Box<dyn ElementObject>,
    pub(crate) modifiers: ElementModifiers,
}

impl Default for AnyElement {
    fn default() -> Self {
        AnyElement::new(Empty)
    }
}

impl AnyElement {
    pub(crate) fn new<E>(element: E) -> Self
    where
        E: 'static + Element,
        E::RequestLayoutState: Any,
    {
        AnyElement {
            inner: Box::new(Drawable::new(element)),
            modifiers: ElementModifiers::default(),
        }
    }

    /// Attempt to downcast a reference to the boxed element to a specific type.
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.inner.inner_element().downcast_mut::<T>()
    }

    pub(crate) fn children(&self) -> &[AnyElement] {
        self.inner.fiber_children()
    }

    pub(crate) fn key(&self) -> VKey {
        self.inner.fiber_key()
    }

    pub(crate) fn style(&self) -> Option<&StyleRefinement> {
        self.cached_style()
    }

    /// Returns the modifiers applied to this element.
    pub(crate) fn modifiers(&self) -> ElementModifiers {
        self.modifiers
    }

    pub(crate) fn child_count(&self) -> usize {
        self.children().len()
    }

    /// Returns the total number of elements in this subtree, including self.
    pub fn count(&self) -> usize {
        let mut count = 1;
        for child in self.children() {
            count += child.count();
        }
        count
    }

    /// Resets the element to its initial drawing phase.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub(crate) fn cached_style(&self) -> Option<&StyleRefinement> {
        self.inner.cached_style()
    }

    pub(crate) fn as_any_view(&self) -> Option<AnyView> {
        self.inner.as_any_view()
    }

    pub(crate) fn create_render_node(&mut self) -> Option<Box<dyn crate::RenderNode>> {
        self.inner.create_render_node()
    }

    pub(crate) fn update_render_node(
        &mut self,
        node: &mut dyn crate::RenderNode,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<UpdateResult> {
        self.inner.update_render_node(node, window, cx)
    }

    /// Expands this element if it's a wrapper type like Component.
    ///
    /// Returns the inner element if expansion occurred, None if the element
    /// should be processed as-is.
    pub(crate) fn try_expand(&mut self, window: &mut Window, cx: &mut App) -> Option<AnyElement> {
        self.inner.try_expand(window, cx)
    }

    /// Expands wrapper elements (like Component) in place, propagating modifiers.
    ///
    /// This recursively expands this element and all its children so that
    /// Components are transparent before reconciliation. This ensures the real
    /// children/keys participate in normal reconciliation with proper identity.
    pub(crate) fn expand_wrappers(&mut self, window: &mut Window, cx: &mut App) {
        // First expand this element if it's a wrapper
        while let Some(mut expanded) = self.try_expand(window, cx) {
            // Propagate modifiers from the wrapper to the expanded element.
            // If the wrapper had modifiers (e.g., deferred_priority), they should
            // be inherited by the rendered element (unless already set).
            if self.modifiers.deferred_priority.is_some()
                && expanded.modifiers.deferred_priority.is_none()
            {
                expanded.modifiers.deferred_priority = self.modifiers.deferred_priority;
            }
            *self = expanded;
        }

        // Then recursively expand all children
        for child in self.children_mut() {
            child.expand_wrappers(window, cx);
        }
    }

    /// Returns true if this element requires fiber-backed layout.
    pub(crate) fn requires_fiber_layout(&self) -> bool {
        self.inner.requires_fiber_layout()
    }

    pub(crate) fn children_mut(&mut self) -> &mut [AnyElement] {
        self.inner.fiber_children_mut()
    }

    /// Request the layout ID of the element stored in this `AnyElement`.
    /// Used for laying out child elements in a parent element.
    pub fn request_layout(&mut self, window: &mut Window, cx: &mut App) -> LayoutId {
        self.inner.request_layout(window, cx)
    }

    /// Prepares the element to be painted by storing its bounds, giving it a chance to draw hitboxes and
    /// request autoscroll before the final paint pass is confirmed.
    pub fn prepaint(&mut self, window: &mut Window, cx: &mut App) -> Option<FocusHandle> {
        self.inner.prepaint(window, cx);
        None
    }

    /// Paints the element stored in this `AnyElement`.
    pub fn paint(&mut self, window: &mut Window, cx: &mut App) {
        self.inner.paint(window, cx);
    }

    /// Paints this element at the given absolute origin.
    ///
    /// This mirrors `prepaint_at` for out-of-tree element trees that need to
    /// paint at a specific window origin (e.g. external scenegraph layers).
    pub fn paint_at(&mut self, origin: Point<Pixels>, window: &mut Window, cx: &mut App) {
        crate::window::context::PaintCx::new(window)
            .with_absolute_element_offset(origin, |window| self.paint(window, cx))
    }

    /// Performs layout for this element within the given available space and returns its size.
    pub fn layout_as_root(
        &mut self,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Size<Pixels> {
        self.inner.layout_as_root(available_space, window, cx)
    }

    /// Prepaints this element at the given absolute origin.
    /// If any element in the subtree beneath this element is focused, its FocusHandle is returned.
    pub fn prepaint_at(
        &mut self,
        origin: Point<Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<FocusHandle> {
        crate::window::context::PrepaintCx::new(window)
            .with_absolute_element_offset(origin, |window| self.prepaint(window, cx))
    }

    /// Performs layout on this element in the available space, then prepaints it at the given absolute origin.
    /// If any element in the subtree beneath this element is focused, its FocusHandle is returned.
    pub fn prepaint_as_root(
        &mut self,
        origin: Point<Pixels>,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<FocusHandle> {
        self.layout_as_root(available_space, window, cx);
        crate::window::context::PrepaintCx::new(window)
            .with_absolute_element_offset(origin, |window| self.prepaint(window, cx))
    }
}

impl Element for AnyElement {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let layout_id = self.request_layout(window, cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.paint(window, cx);
    }
}

impl IntoElement for AnyElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }

    fn into_any_element(self) -> AnyElement {
        self
    }
}

/// The empty element, which renders nothing.
pub struct Empty;

impl IntoElement for Empty {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for Empty {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        (
            window.request_layout(
                Style {
                    display: crate::Display::None,
                    ..Default::default()
                },
                None,
                cx,
            ),
            (),
        )
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _state: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }
}

/// Helper trait for computing element hashes with common patterns.
/// Reduces boilerplate when implementing fiber hash methods for elements.
pub trait ElementHashing {
    /// Hash an interactivity's base, hover, focus, and active styles for layout.
    /// This is used by elements that have interactive styling like Div and Svg.
    fn hash_interactivity_layout(interactivity: &crate::Interactivity) -> u64 {
        use collections::FxHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = FxHasher::default();
        interactivity.base_style.layout_hash().hash(&mut hasher);
        if let Some(ref s) = interactivity.hover_style {
            s.layout_hash().hash(&mut hasher);
        }
        if let Some(ref s) = interactivity.focus_style {
            s.layout_hash().hash(&mut hasher);
        }
        if let Some(ref s) = interactivity.in_focus_style {
            s.layout_hash().hash(&mut hasher);
        }
        if let Some(ref s) = interactivity.focus_visible_style {
            s.layout_hash().hash(&mut hasher);
        }
        if let Some(ref s) = interactivity.active_style {
            s.layout_hash().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Hash an interactivity's base, hover, focus, and active styles for painting.
    /// Includes element_id since it may affect paint behavior (e.g., focus rings).
    fn hash_interactivity_paint(interactivity: &crate::Interactivity) -> u64 {
        use collections::FxHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = FxHasher::default();
        interactivity.element_id.hash(&mut hasher);
        interactivity.base_style.paint_hash().hash(&mut hasher);
        if let Some(ref s) = interactivity.hover_style {
            s.paint_hash().hash(&mut hasher);
        }
        if let Some(ref s) = interactivity.focus_style {
            s.paint_hash().hash(&mut hasher);
        }
        if let Some(ref s) = interactivity.in_focus_style {
            s.paint_hash().hash(&mut hasher);
        }
        if let Some(ref s) = interactivity.focus_visible_style {
            s.paint_hash().hash(&mut hasher);
        }
        if let Some(ref s) = interactivity.active_style {
            s.paint_hash().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Convenience function to hash multiple values into a single u64.
    /// Useful for implementing content_hash, layout_hash, and paint_hash.
    fn hash_values<F>(f: F) -> u64
    where
        F: FnOnce(&mut collections::FxHasher),
    {
        use collections::FxHasher;
        use std::hash::Hasher;
        let mut hasher = FxHasher::default();
        f(&mut hasher);
        hasher.finish()
    }
}
