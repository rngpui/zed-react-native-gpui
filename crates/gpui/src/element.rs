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
    App, AvailableSpace, Bounds, CachePolicy, Context, DispatchNodeId,
    ElementId, FocusHandle, InspectorElementId, LayoutId, Pixels, Point, Size,
    Style, Window,
    retained::RetainedElementId,
    util::FluentBuilder,
};
use derive_more::{Deref, DerefMut};
use std::{
    any::{Any, TypeId, type_name},
    fmt::{self, Debug, Display},
    hash::Hash,
    mem, panic,
    sync::Arc,
};

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
    ///
    /// ## Per-Element Caching (Phase 20)
    ///
    /// The framework automatically handles per-element caching. When an element's
    /// `paint_inputs_hash()` matches the previous frame, its own primitives are
    /// copied from cache and new primitive insertions are automatically skipped.
    /// Children still paint normally with their own cache checks.
    ///
    /// To enable caching for your element, implement `paint_inputs_hash()` to return
    /// a non-zero hash of the element's paint inputs (style, content, etc.).
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

    /// Hash of the element's inputs for retained element caching.
    /// Returns 0 to disable caching (always repaint).
    fn element_hash(&self) -> u64 {
        0
    }

    /// Hash of the element's style inputs for retained tree reconciliation.
    /// Defaults to `element_hash` for backward compatibility.
    fn style_hash(&self) -> u64 {
        self.element_hash()
    }

    /// Optionally provide a hash of the content that affects this element's rendering.
    /// Returning `None` disables primitive caching for this element.
    fn content_hash(
        &self,
        _id: Option<&GlobalElementId>,
        _bounds: Bounds<Pixels>,
        _window: &Window,
        _cx: &App,
    ) -> Option<u64> {
        None
    }

    /// Compute a hash of the element's paint inputs (state that affects paint output).
    ///
    /// This is used by the implicit element identity system (Phase 20) to memoize paint
    /// outputs based on element position + input hash. If the same element at the same
    /// position has the same input hash as the previous frame, its paint output can be reused.
    ///
    /// ## What to include in the hash:
    /// - Style properties (background, border, padding, etc.)
    /// - Text content
    /// - Any state that affects what primitives are generated
    ///
    /// ## What NOT to include:
    /// - Bounds (handled separately by the memoization system)
    /// - Transient state that doesn't affect rendering
    /// - Children (they have their own implicit IDs and hashes)
    ///
    /// Returns 0 by default, which means the element will always be repainted.
    /// Implement this method to enable paint memoization for your element.
    fn paint_inputs_hash(&self) -> u64 {
        0
    }

    /// Configure whether primitive caching is allowed for this element.
    fn cache_policy(&self) -> CachePolicy {
        CachePolicy::Default
    }

    /// Returns true if this element has interactive styles (hover, active) that
    /// change based on user interaction. Used to determine if parent elements
    /// can safely cache this element to texture.
    ///
    /// Elements with interactive styles should not be cached to texture because
    /// the cached texture would become stale when the style changes.
    fn has_interactive_styles(&self) -> bool {
        false
    }

    /// Reset this element's children for reuse in a new frame.
    /// Called recursively when an element is being reused from the retained tree.
    /// Override this if your element has children that need resetting.
    fn reset_children_for_reuse(&mut self) {}

    /// Convert this element into a dynamically-typed [`AnyElement`].
    fn into_any(self) -> AnyElement {
        AnyElement::new(self)
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
        self.extend(std::iter::once(child.into_element().into_any()));
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

            let layout_id = element.request_layout(window, cx);
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
        window.with_global_id(ElementId::Name(type_name::<C>().into()), |_, window| {
            element.paint(window, cx);
        })
    }
}

impl<C: RenderOnce> IntoElement for Component<C> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// A globally unique identifier for an element, used to track state across frames.
#[derive(Deref, DerefMut, Clone, Default, Debug, Eq, PartialEq, Hash)]
pub struct GlobalElementId(pub(crate) Arc<[ElementId]>);

impl GlobalElementId {
    /// Create a GlobalElementId for the root layer.
    /// This uses a special "__root__" name that won't conflict with user-provided IDs.
    pub fn root() -> Self {
        Self(Arc::from([ElementId::Name("__root__".into())]))
    }
}

impl Display for GlobalElementId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, element_id) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ".")?;
            }
            write!(f, "{}", element_id)?;
        }
        Ok(())
    }
}

trait ElementObject {
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

    fn has_interactive_styles(&self) -> bool;

    fn reset_for_reuse(&mut self);

    fn reset_children_for_reuse(&mut self);

    /// Get the TypeId of the concrete element type.
    /// Used for reconciliation to match elements by type.
    fn element_type_id(&self) -> TypeId;

    /// Get the style hash of the element (excluding children count).
    /// Used for reconciliation to detect style changes.
    fn style_hash(&self) -> u64;

    /// Get this element's local key (if any) for reconciliation.
    fn element_key(&self) -> Option<ElementId>;

    /// Get the retained ID assigned during reconciliation.
    fn retained_id(&self) -> Option<RetainedElementId>;

    /// Set the retained ID assigned during reconciliation.
    fn set_retained_id(&mut self, id: RetainedElementId);
}

/// A wrapper around an implementer of [`Element`] that allows it to be drawn in a window.
pub struct Drawable<E: Element> {
    /// The drawn element.
    pub element: E,
    phase: ElementDrawPhase<E::RequestLayoutState, E::PrepaintState>,
    retained_id: Option<RetainedElementId>,
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
    Cached {
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
        layout_id: LayoutId,
        node_id: DispatchNodeId,
        global_id: Option<GlobalElementId>,
        inspector_id: Option<InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: RequestLayoutState,
        prepaint: PrepaintState,
    },
    CachedPrepaint {
        layout_id: LayoutId,
        global_id: Option<GlobalElementId>,
        inspector_id: Option<InspectorElementId>,
    },
    Painted,
}

/// A wrapper around an implementer of [`Element`] that allows it to be drawn in a window.
impl<E: Element> Drawable<E> {
    pub(crate) fn new(element: E) -> Self {
        Drawable {
            element,
            phase: ElementDrawPhase::Start,
            retained_id: None,
        }
    }

    fn request_layout(&mut self, window: &mut Window, cx: &mut App) -> LayoutId {
        fn run_request_layout<E: Element>(
            this: &mut Drawable<E>,
            window: &mut Window,
            cx: &mut App,
        ) -> LayoutId {
            match mem::take(&mut this.phase) {
                ElementDrawPhase::Start => {
                    let global_id = this.element.id().map(|element_id| {
                        window.element_id_stack.push(element_id);
                        GlobalElementId(Arc::from(&*window.element_id_stack))
                    });

                    let inspector_id;
                    #[cfg(any(feature = "inspector", debug_assertions))]
                    {
                        inspector_id = this.element.source_location().map(|source| {
                            let path = crate::InspectorElementPath {
                                global_id: GlobalElementId(Arc::from(&*window.element_id_stack)),
                                source_location: source,
                            };
                            window.build_inspector_element_id(path)
                        });
                    }
                    #[cfg(not(any(feature = "inspector", debug_assertions)))]
                    {
                        inspector_id = None;
                    }

                    let (layout_id, request_layout) = this.element.request_layout(
                        global_id.as_ref(),
                        inspector_id.as_ref(),
                        window,
                        cx,
                    );

                    if global_id.is_some() {
                        window.element_id_stack.pop();
                    }

                    if window.is_layout_cached(layout_id) {
                        this.phase = ElementDrawPhase::Cached {
                            layout_id,
                            global_id,
                            inspector_id,
                            request_layout,
                        };
                    } else {
                        this.phase = ElementDrawPhase::RequestLayout {
                            layout_id,
                            global_id,
                            inspector_id,
                            request_layout,
                        };
                    }
                    layout_id
                }
                _ => panic!("must call request_layout only once"),
            }
        }

        let mut retained_context = None;
        if let Some(context) = window.retained_context_mut() {
            let is_root = context.node_stack.is_empty();
            let parent_id = context.node_stack.last().copied();
            let child_index = if is_root {
                0
            } else if let Some(counter) = context.child_indices.last_mut() {
                let idx = *counter;
                *counter += 1;
                idx
            } else {
                0
            };
            retained_context = Some((
                context.reconciling,
                is_root,
                parent_id,
                child_index,
                context.root_id,
                context.force_layout_stack.last().copied().unwrap_or(false),
                context.force_prepaint_stack.last().copied().unwrap_or(false),
            ));
        }

        if let Some((
            reconciling,
            is_root,
            parent_id,
            child_index,
            root_id,
            parent_force_layout,
            parent_force_prepaint,
        )) = retained_context
        {
            let element_type = TypeId::of::<E>();
            let element_key = self.element.id();
            let content_hash = self.element.style_hash();
            let mut retained_id = self.retained_id;

            if reconciling {
                if is_root {
                    retained_id = Some(root_id);
                    self.set_retained_id(root_id);
                } else if let Some(parent_id) = parent_id {
                    let child_id = window.retained_tree.reconcile_child(
                        parent_id,
                        child_index,
                        element_type,
                        element_key.clone(),
                        content_hash,
                    );
                    retained_id = Some(child_id);
                    self.set_retained_id(child_id);
                }
            } else if retained_id.is_none() && is_root {
                retained_id = Some(root_id);
                self.set_retained_id(root_id);
            }

            if let Some(retained_id) = retained_id {
                let (needs_layout, needs_prepaint, subtree_dirty, layout_id) = window
                    .retained_tree
                    .get(retained_id)
                    .map(|retained| {
                        (
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::NEEDS_LAYOUT),
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::NEEDS_PREPAINT),
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::SUBTREE_DIRTY),
                            retained.layout_id,
                        )
                    })
                    .unwrap_or((true, true, true, None));

                let force_layout =
                    parent_force_layout || parent_force_prepaint || needs_layout;
                let force_prepaint =
                    parent_force_prepaint || needs_prepaint || force_layout;

                if !reconciling && !force_layout && !subtree_dirty {
                    if let Some(layout_id) = layout_id {
                        window.mark_layout_subtree_accessed(layout_id);
                        return layout_id;
                    }
                }

                if force_layout && !matches!(self.phase, ElementDrawPhase::Start) {
                    self.reset_for_reuse();
                }

                if let Some(context) = window.retained_context_mut() {
                    context.node_stack.push(retained_id);
                    context.child_indices.push(0);
                    context.force_layout_stack.push(force_layout);
                    context.force_prepaint_stack.push(force_prepaint);
                }

                let layout_id = run_request_layout(self, window, cx);

                if let Some(retained) = window.retained_tree.get_mut(retained_id) {
                    retained.layout_id = Some(layout_id);
                    retained.clear_layout_dirty();
                }

                let mut child_count = None;
                if let Some(context) = window.retained_context_mut() {
                    let count = context.child_indices.pop().unwrap_or(0);
                    context.node_stack.pop();
                    context.force_layout_stack.pop();
                    context.force_prepaint_stack.pop();
                    child_count = Some(count);
                }
                if reconciling {
                    if let Some(count) = child_count {
                        window.retained_tree.truncate_children(retained_id, count);
                    }
                }

                return layout_id;
            }
        }

        run_request_layout(self, window, cx)
    }

    pub(crate) fn prepaint(&mut self, window: &mut Window, cx: &mut App) {
        fn run_prepaint<E: Element>(
            this: &mut Drawable<E>,
            window: &mut Window,
            cx: &mut App,
        ) {
            match mem::take(&mut this.phase) {
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
                    if let Some(element_id) = this.element.id() {
                        window.element_id_stack.push(element_id);
                        debug_assert_eq!(&*global_id.as_ref().unwrap().0, &*window.element_id_stack);
                    }

                    // Compute stable identity during prepaint so hitboxes can derive stable keys.
                    let computed_id = window.compute_element_id::<E>(global_id.as_ref());
                    window.begin_element_prepaint_identity(computed_id);

                    let bounds = window.layout_bounds(layout_id);
                    let node_id = window
                        .next_frame
                        .dispatch_tree
                        .push_node_with_id(computed_id);
                    let prepaint_start = window.prepaint_index();
                    let prepaint = this.element.prepaint(
                        global_id.as_ref(),
                        inspector_id.as_ref(),
                        bounds,
                        &mut request_layout,
                        window,
                        cx,
                    );
                    let prepaint_end = window.prepaint_index();
                    window.set_cached_prepaint_range(layout_id, prepaint_start..prepaint_end);
                    window.next_frame.dispatch_tree.pop_node();

                    window.end_element_prepaint_identity();

                    if global_id.is_some() {
                        window.element_id_stack.pop();
                    }

                    this.phase = ElementDrawPhase::Prepaint {
                        layout_id,
                        node_id,
                        global_id,
                        inspector_id,
                        bounds,
                        request_layout,
                        prepaint,
                    };
                }
                ElementDrawPhase::Cached {
                    layout_id,
                    global_id,
                    inspector_id,
                    mut request_layout,
                } => {
                    if let Some(element_id) = this.element.id() {
                        window.element_id_stack.push(element_id);
                        debug_assert_eq!(&*global_id.as_ref().unwrap().0, &*window.element_id_stack);
                    }

                    let computed_id = window.compute_element_id::<E>(global_id.as_ref());
                    window.begin_element_prepaint_identity(computed_id);

                    let cached_prepaint = window.cached_prepaint_range(layout_id);
                    let cached_paint = window.cached_paint_range(layout_id);

                    if let (Some(prepaint_range), Some(_paint_range)) =
                        (cached_prepaint, cached_paint)
                    {
                        let prepaint_start = window.prepaint_index();
                        window.reuse_prepaint(prepaint_range);
                        let prepaint_end = window.prepaint_index();
                        window.set_cached_prepaint_range(layout_id, prepaint_start..prepaint_end);

                        window.end_element_prepaint_identity();

                        if global_id.is_some() {
                            window.element_id_stack.pop();
                        }

                        this.phase = ElementDrawPhase::CachedPrepaint {
                            layout_id,
                            global_id,
                            inspector_id,
                        };
                    } else {
                        let bounds = window.layout_bounds(layout_id);
                        let node_id = window
                            .next_frame
                            .dispatch_tree
                            .push_node_with_id(computed_id);
                        let prepaint_start = window.prepaint_index();
                        let prepaint = this.element.prepaint(
                            global_id.as_ref(),
                            inspector_id.as_ref(),
                            bounds,
                            &mut request_layout,
                            window,
                            cx,
                        );
                        let prepaint_end = window.prepaint_index();
                        window.set_cached_prepaint_range(layout_id, prepaint_start..prepaint_end);
                        window.next_frame.dispatch_tree.pop_node();

                        window.end_element_prepaint_identity();

                        if global_id.is_some() {
                            window.element_id_stack.pop();
                        }

                        this.phase = ElementDrawPhase::Prepaint {
                            layout_id,
                            node_id,
                            global_id,
                            inspector_id,
                            bounds,
                            request_layout,
                            prepaint,
                        };
                    }
                }
                _ => panic!("must call request_layout before prepaint"),
            }
        }

        let mut retained_context = None;
        if let Some(context) = window.retained_context_mut() {
            retained_context = Some((
                context.reconciling,
                context.force_prepaint_stack.last().copied().unwrap_or(false),
            ));
        }

        if let Some((reconciling, parent_force_prepaint)) = retained_context {
            if let Some(retained_id) = self.retained_id {
                let (needs_layout, needs_prepaint, subtree_dirty, layout_id) = window
                    .retained_tree
                    .get(retained_id)
                    .map(|retained| {
                        (
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::NEEDS_LAYOUT),
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::NEEDS_PREPAINT),
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::SUBTREE_DIRTY),
                            retained.layout_id,
                        )
                    })
                    .unwrap_or((true, true, true, None));

                let force_prepaint =
                    parent_force_prepaint || needs_prepaint || needs_layout;

                if !reconciling && !force_prepaint && !subtree_dirty {
                    if let Some(layout_id) = layout_id {
                        if let Some(prepaint_range) = window.cached_prepaint_range(layout_id) {
                            let prepaint_start = window.prepaint_index();
                            window.reuse_prepaint(prepaint_range);
                            let prepaint_end = window.prepaint_index();
                            window.set_cached_prepaint_range(
                                layout_id,
                                prepaint_start..prepaint_end,
                            );
                            if let Some(retained) = window.retained_tree.get_mut(retained_id) {
                                retained.clear_prepaint_dirty();
                            }
                            return;
                        }
                    }
                }

                if let Some(context) = window.retained_context_mut() {
                    context.force_prepaint_stack.push(force_prepaint);
                }

                run_prepaint(self, window, cx);

                if let Some(retained) = window.retained_tree.get_mut(retained_id) {
                    retained.clear_prepaint_dirty();
                }

                if let Some(context) = window.retained_context_mut() {
                    context.force_prepaint_stack.pop();
                }

                return;
            }
        }

        run_prepaint(self, window, cx);
    }

    pub(crate) fn paint(&mut self, window: &mut Window, cx: &mut App) {
        fn run_paint<E: Element>(
            this: &mut Drawable<E>,
            window: &mut Window,
            cx: &mut App,
        ) {
            match mem::take(&mut this.phase) {
                ElementDrawPhase::Prepaint {
                    layout_id,
                    node_id,
                    global_id,
                    inspector_id,
                    bounds,
                    mut request_layout,
                    mut prepaint,
                    ..
                } => {
                    if let Some(element_id) = this.element.id() {
                        window.element_id_stack.push(element_id);
                        debug_assert_eq!(&*global_id.as_ref().unwrap().0, &*window.element_id_stack);
                    }

                    // Phase 20: Per-element caching
                    // Compute element ID and maintain path stack for ALL elements.
                    // This ensures consistent child indices for cache key stability.
                    let computed_id = window.compute_element_id::<E>(global_id.as_ref());
                    window
                        .next_frame
                        .dispatch_tree
                        .clear_handlers_for(computed_id);

                    // Only do caching overhead if the element supports it (non-zero hash)
                    let input_hash = this
                        .element
                        .content_hash(global_id.as_ref(), bounds, window, cx)
                        .unwrap_or(0);

                    window.next_frame.dispatch_tree.set_active_node(node_id);

                    let paint_start = window.paint_index();
                    if input_hash != 0 {
                        // Element supports caching - do full per-element tracking
                        use crate::display_list::PaintCacheResult;
                        let cache_result = window.begin_element_paint(computed_id, input_hash);

                        match cache_result {
                            PaintCacheResult::SubtreeSkip => {
                                // Entire subtree cached AND no event handlers - skip paint() entirely.
                                // Items already copied in begin_element_paint.
                                window.finalize_element_subtree_skip();
                            }
                            PaintCacheResult::Hit | PaintCacheResult::Miss => {
                                // Either cache miss (repaint) or hit (own items cached but paint children).
                                // Caching happens at primitive insertion level.
                                this.element.paint(
                                    global_id.as_ref(),
                                    inspector_id.as_ref(),
                                    bounds,
                                    &mut request_layout,
                                    &mut prepaint,
                                    window,
                                    cx,
                                );
                                window.finalize_element_paint(computed_id);
                            }
                        }
                    } else {
                        // No caching - paint directly, but still maintain path stack
                        this.element.paint(
                            global_id.as_ref(),
                            inspector_id.as_ref(),
                            bounds,
                            &mut request_layout,
                            &mut prepaint,
                            window,
                            cx,
                        );
                        // Pop path stack to balance the push in compute_element_id
                        window.pop_element_path();
                    }

                    let paint_end = window.paint_index();
                    window.set_cached_paint_range(layout_id, paint_start..paint_end);
                    window.clear_layout_dirty_flags(layout_id);

                    if global_id.is_some() {
                        window.element_id_stack.pop();
                    }

                    this.phase = ElementDrawPhase::Painted;
                }
                ElementDrawPhase::CachedPrepaint {
                    layout_id,
                    global_id,
                    inspector_id: _,
                } => {
                    if let Some(element_id) = this.element.id() {
                        window.element_id_stack.push(element_id);
                        debug_assert_eq!(&*global_id.as_ref().unwrap().0, &*window.element_id_stack);
                    }

                    let computed_id = window.compute_element_id::<E>(global_id.as_ref());
                    if let Some(global_id) = global_id.as_ref() {
                        window.reuse_scene_display_list(global_id);
                    }
                    window.reuse_display_list_subtree(computed_id);
                    let paint_start = window.paint_index();
                    if let Some(paint_range) = window.cached_paint_range(layout_id) {
                        window.reuse_paint(paint_range);
                    }
                    let paint_end = window.paint_index();
                    window.set_cached_paint_range(layout_id, paint_start..paint_end);
                    window.pop_element_path();
                    window.clear_layout_dirty_flags(layout_id);

                    if global_id.is_some() {
                        window.element_id_stack.pop();
                    }

                    this.phase = ElementDrawPhase::Painted;
                }
                _ => panic!("must call prepaint before paint"),
            }
        }

        let mut retained_context = None;
        if let Some(context) = window.retained_context_mut() {
            retained_context = Some((
                context.reconciling,
                context.force_prepaint_stack.last().copied().unwrap_or(false),
            ));
        }

        if let Some((reconciling, parent_force_prepaint)) = retained_context {
            if let Some(retained_id) = self.retained_id {
                let (needs_layout, needs_prepaint, needs_paint, subtree_dirty, layout_id) = window
                    .retained_tree
                    .get(retained_id)
                    .map(|retained| {
                        (
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::NEEDS_LAYOUT),
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::NEEDS_PREPAINT),
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::NEEDS_PAINT),
                            retained
                                .dirty
                                .contains(crate::retained::DirtyFlags::SUBTREE_DIRTY),
                            retained.layout_id,
                        )
                    })
                    .unwrap_or((true, true, true, true, None));

                let force_paint = parent_force_prepaint || needs_prepaint || needs_layout;

                if !reconciling && !force_paint && !needs_paint && !subtree_dirty {
                    if let Some(layout_id) = layout_id {
                        if let Some(paint_range) = window.cached_paint_range(layout_id) {
                            let paint_start = window.paint_index();
                            window.reuse_paint(paint_range);
                            let paint_end = window.paint_index();
                            window.set_cached_paint_range(layout_id, paint_start..paint_end);
                            window.clear_layout_dirty_flags(layout_id);
                            if let Some(retained) = window.retained_tree.get_mut(retained_id) {
                                retained.clear_paint_dirty();
                            }
                            window.retained_tree.clear_subtree_dirty_if_clean(retained_id);
                            return;
                        }
                    }
                }

                if let Some(context) = window.retained_context_mut() {
                    context.force_prepaint_stack.push(force_paint);
                }

                run_paint(self, window, cx);

                if let Some(retained) = window.retained_tree.get_mut(retained_id) {
                    retained.clear_paint_dirty();
                }
                window.retained_tree.clear_subtree_dirty_if_clean(retained_id);

                if let Some(context) = window.retained_context_mut() {
                    context.force_prepaint_stack.pop();
                }

                return;
            }
        }

        run_paint(self, window, cx);
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
            ElementDrawPhase::Cached {
                layout_id,
                global_id,
                inspector_id,
                request_layout,
            } => {
                window.compute_layout(layout_id, available_space, cx);
                self.phase = ElementDrawPhase::Cached {
                    layout_id,
                    global_id,
                    inspector_id,
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

    fn has_interactive_styles(&self) -> bool {
        self.element.has_interactive_styles()
    }

    fn reset_for_reuse(&mut self) {
        self.phase = ElementDrawPhase::Start;
        self.reset_children_for_reuse();
    }

    fn reset_children_for_reuse(&mut self) {
        self.element.reset_children_for_reuse();
    }

    fn element_type_id(&self) -> TypeId {
        TypeId::of::<E>()
    }

    fn style_hash(&self) -> u64 {
        self.element.style_hash()
    }

    fn element_key(&self) -> Option<ElementId> {
        self.element.id()
    }

    fn retained_id(&self) -> Option<RetainedElementId> {
        self.retained_id
    }

    fn set_retained_id(&mut self, id: RetainedElementId) {
        self.retained_id = Some(id);
    }
}

/// A dynamically typed element that can be used to store any element type.
pub struct AnyElement(Box<dyn ElementObject>);

impl AnyElement {
    pub(crate) fn new<E>(element: E) -> Self
    where
        E: 'static + Element,
        E::RequestLayoutState: Any,
    {
        AnyElement(Box::new(Drawable::new(element)))
    }

    /// Attempt to downcast a reference to the boxed element to a specific type.
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.0.inner_element().downcast_mut::<T>()
    }

    /// Request the layout ID of the element stored in this `AnyElement`.
    /// Used for laying out child elements in a parent element.
    pub fn request_layout(&mut self, window: &mut Window, cx: &mut App) -> LayoutId {
        self.0.request_layout(window, cx)
    }

    /// Prepares the element to be painted by storing its bounds, giving it a chance to draw hitboxes and
    /// request autoscroll before the final paint pass is confirmed.
    pub fn prepaint(&mut self, window: &mut Window, cx: &mut App) -> Option<FocusHandle> {
        let focus_assigned = window.next_frame.focus.is_some();

        self.0.prepaint(window, cx);

        if !focus_assigned && let Some(focus_id) = window.next_frame.focus {
            return FocusHandle::for_id(focus_id, &cx.focus_handles);
        }

        None
    }

    /// Paints the element stored in this `AnyElement`.
    pub fn paint(&mut self, window: &mut Window, cx: &mut App) {
        self.0.paint(window, cx);
    }

    /// Paints this element at the given absolute origin.
    ///
    /// This mirrors `prepaint_at` for out-of-tree element trees that need to
    /// paint at a specific window origin (e.g. external scenegraph layers).
    pub fn paint_at(&mut self, origin: Point<Pixels>, window: &mut Window, cx: &mut App) {
        window.with_absolute_element_offset(origin, |window| self.paint(window, cx))
    }

    /// Performs layout for this element within the given available space and returns its size.
    pub fn layout_as_root(
        &mut self,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Size<Pixels> {
        self.0.layout_as_root(available_space, window, cx)
    }

    /// Prepaints this element at the given absolute origin.
    /// If any element in the subtree beneath this element is focused, its FocusHandle is returned.
    pub fn prepaint_at(
        &mut self,
        origin: Point<Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<FocusHandle> {
        window.with_absolute_element_offset(origin, |window| self.prepaint(window, cx))
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
        window.with_absolute_element_offset(origin, |window| self.prepaint(window, cx))
    }

    /// Returns true if this element has interactive styles (hover, active) that
    /// change based on user interaction.
    pub fn has_interactive_styles(&self) -> bool {
        self.0.has_interactive_styles()
    }

    /// Reset the element's phase to Start so it can be reused in a new frame.
    /// Called when reusing retained elements from the previous frame.
    pub fn reset_for_reuse(&mut self) {
        self.0.reset_for_reuse();
    }

    /// Get the TypeId of the concrete element type.
    /// Used for reconciliation to match elements by type.
    pub fn element_type_id(&self) -> TypeId {
        self.0.element_type_id()
    }

    /// Get the style hash of the element (excluding children count).
    /// Used for reconciliation to detect style changes.
    pub fn style_hash(&self) -> u64 {
        self.0.style_hash()
    }

    /// Get the element key (if any) for reconciliation.
    pub fn element_key(&self) -> Option<ElementId> {
        self.0.element_key()
    }

    /// Get the retained element ID assigned during reconciliation.
    pub(crate) fn retained_id(&self) -> Option<RetainedElementId> {
        self.0.retained_id()
    }

    /// Set the retained element ID assigned during reconciliation.
    pub(crate) fn set_retained_id(&mut self, id: RetainedElementId) {
        self.0.set_retained_id(id);
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
        _: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let layout_id = self.request_layout(window, cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
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
        _: Option<&GlobalElementId>,
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
