use crate::{
    AnyElement, AnyEntity, AnyWeakEntity, App, Bounds, ContentMask, Context, Element, ElementId,
    Entity, EntityId, GlobalElementId, InspectorElementId, IntoElement, LayoutId, PaintIndex,
    Pixels, PrepaintStateIndex, Render, Style, StyleRefinement, TextStyle, WeakEntity,
    retained::RetainedElementId,
};
use crate::{Empty, Window};
use anyhow::Result;
use collections::FxHashSet;
use refineable::Refineable;
use std::mem;
use std::rc::Rc;
use std::{any::TypeId, fmt, ops::Range};

struct AnyViewState {
    prepaint_range: Range<PrepaintStateIndex>,
    paint_range: Range<PaintIndex>,
    cache_key: ViewCacheKey,
    accessed_entities: FxHashSet<EntityId>,
}

#[derive(Default)]
struct ViewCacheKey {
    bounds: Bounds<Pixels>,
    content_mask: ContentMask<Pixels>,
    text_style: TextStyle,
}

/// State indicating whether we need to call render() or can use retained element.
pub struct EntityLayoutState {
    root_id: RetainedElementId,
}

impl<V: Render> Element for Entity<V> {
    type RequestLayoutState = EntityLayoutState;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::View(self.entity_id()))
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let entity_id = self.entity_id();
        let is_dirty = window.is_view_dirty(entity_id);
        let has_retained = window.has_retained_view_root(entity_id);

        // Check if we need to render (before borrowing retained view)
        let needs_render = is_dirty || !has_retained;
        let mut reconciling = false;

        eprintln!(
            "[VIEW request_layout] entity={:?} is_dirty={} has_retained={} needs_render={}",
            entity_id, is_dirty, has_retained, needs_render
        );

        if needs_render {
            let mut element = self.update(cx, |view, cx| view.render(window, cx).into_any_element());
            let root_id = window.retained_tree.reconcile_root(
                entity_id,
                element.element_type_id(),
                element.element_key(),
                element.style_hash(),
            );
            element.set_retained_id(root_id);

            eprintln!(
                "[VIEW request_layout] RENDERED view {:?}, root_id={:?}, element_type={:?}",
                entity_id, root_id, element.element_type_id()
            );

            let retained = window.get_or_create_retained_view::<V>(entity_id);
            retained.root = Some(element);
            retained.root_id = Some(root_id);

            window.record_view_rendered();
            reconciling = true;
        } else {
            eprintln!("[VIEW request_layout] SKIPPED render for view {:?}", entity_id);
            window.record_view_skipped();
        }

        // Get root_id before borrowing retained for layout
        let root_id = window
            .get_retained_view(entity_id)
            .and_then(|r| r.root_id)
            .expect("retained view root missing");

        // Take the root element to avoid borrow conflicts
        let mut root = window.take_retained_root(entity_id).expect("retained root missing");

        let layout_id = window.with_retained_context(entity_id, root_id, reconciling, |window| {
            window.with_rendered_view(entity_id, |window| root.request_layout(window, cx))
        });

        eprintln!(
            "[VIEW request_layout] entity={:?} layout_id={:?} reconciling={}",
            entity_id, layout_id, reconciling
        );

        // Return the root element
        window.return_retained_root(entity_id, root);
        window.register_view_layout(entity_id, layout_id);

        (layout_id, EntityLayoutState { root_id })
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        state: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let entity_id = self.entity_id();
        eprintln!("[VIEW prepaint] entity={:?} root_id={:?}", entity_id, state.root_id);
        window.set_view_id(entity_id);

        // Take the root element to avoid borrow conflicts
        let mut root = window
            .take_retained_root(entity_id)
            .expect("retained root missing in prepaint");

        window.with_retained_context(entity_id, state.root_id, false, |window| {
            window.with_rendered_view(entity_id, |window| root.prepaint(window, cx));
        });

        // Return the root element
        window.return_retained_root(entity_id, root);
        eprintln!("[VIEW prepaint] entity={:?} DONE", entity_id);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        state: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let entity_id = self.entity_id();
        eprintln!("[VIEW paint] entity={:?} root_id={:?}", entity_id, state.root_id);

        // Take the root element to avoid borrow conflicts
        let mut root = window
            .take_retained_root(entity_id)
            .expect("retained root missing in paint");

        window.with_retained_context(entity_id, state.root_id, false, |window| {
            window.with_rendered_view(entity_id, |window| root.paint(window, cx));
        });

        // Return the root element
        window.return_retained_root(entity_id, root);
        eprintln!("[VIEW paint] entity={:?} DONE", entity_id);
    }
}

/// A dynamically-typed handle to a view, which can be downcast to a [Entity] for a specific type.
#[derive(Clone, Debug)]
pub struct AnyView {
    entity: AnyEntity,
    render: fn(&AnyView, &mut Window, &mut App) -> AnyElement,
    cached_style: Option<Rc<StyleRefinement>>,
}

impl<V: Render> From<Entity<V>> for AnyView {
    fn from(value: Entity<V>) -> Self {
        AnyView {
            entity: value.into_any(),
            render: any_view::render::<V>,
            cached_style: None,
        }
    }
}

impl AnyView {
    /// Indicate that this view should be cached when using it as an element.
    /// When using this method, the view's previous layout and paint will be recycled from the previous frame if [Context::notify] has not been called since it was rendered.
    /// The one exception is when [Window::refresh] is called, in which case caching is ignored.
    pub fn cached(mut self, style: StyleRefinement) -> Self {
        self.cached_style = Some(style.into());
        self
    }

    /// Convert this to a weak handle.
    pub fn downgrade(&self) -> AnyWeakView {
        AnyWeakView {
            entity: self.entity.downgrade(),
            render: self.render,
        }
    }

    /// Convert this to a [Entity] of a specific type.
    /// If this handle does not contain a view of the specified type, returns itself in an `Err` variant.
    pub fn downcast<T: 'static>(self) -> Result<Entity<T>, Self> {
        match self.entity.downcast() {
            Ok(entity) => Ok(entity),
            Err(entity) => Err(Self {
                entity,
                render: self.render,
                cached_style: self.cached_style,
            }),
        }
    }

    /// Gets the [TypeId] of the underlying view.
    pub fn entity_type(&self) -> TypeId {
        self.entity.entity_type
    }

    /// Gets the entity id of this handle.
    pub fn entity_id(&self) -> EntityId {
        self.entity.entity_id()
    }
}

impl PartialEq for AnyView {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl Eq for AnyView {}

impl Element for AnyView {
    type RequestLayoutState = Option<AnyElement>;
    type PrepaintState = Option<AnyElement>;

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::View(self.entity_id()))
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let entity_id = self.entity_id();
        let is_dirty = window.is_view_dirty(entity_id);
        let has_retained = window.has_retained_view_root(entity_id);
        let needs_render = is_dirty || !has_retained;

        eprintln!(
            "[ANYVIEW request_layout] entity={:?} is_dirty={} has_retained={} needs_render={}",
            entity_id, is_dirty, has_retained, needs_render
        );

        // Disable caching when inspecting so that mouse_hit_test has all hitboxes.
        let caching_disabled = window.is_inspector_picking(cx);
        let mut reconciling = false;

        if needs_render && !caching_disabled {
            // Render and reconcile with retained tree
            let mut element = (self.render)(self, window, cx);
            let root_id = window.retained_tree.reconcile_root(
                entity_id,
                element.element_type_id(),
                element.element_key(),
                element.style_hash(),
            );
            element.set_retained_id(root_id);

            eprintln!(
                "[ANYVIEW request_layout] RENDERED view {:?}, root_id={:?}",
                entity_id, root_id
            );

            let retained = window.get_or_create_retained_view_untyped(entity_id);
            retained.root = Some(element);
            retained.root_id = Some(root_id);

            window.record_view_rendered();
            reconciling = true;
        } else if !needs_render {
            eprintln!("[ANYVIEW request_layout] SKIPPED render for view {:?}", entity_id);
            window.record_view_skipped();
        }

        // Get root_id - either from fresh render or existing retained view
        let root_id = window
            .get_retained_view(entity_id)
            .and_then(|r| r.root_id);

        let result = if let Some(root_id) = root_id {
            // Take the root element to avoid borrow conflicts
            let mut root = window.take_retained_root(entity_id).expect("retained root missing");

            let layout_id = window.with_retained_context(entity_id, root_id, reconciling, |window| {
                window.with_rendered_view(entity_id, |window| root.request_layout(window, cx))
            });

            eprintln!(
                "[ANYVIEW request_layout] entity={:?} layout_id={:?} reconciling={}",
                entity_id, layout_id, reconciling
            );

            // Return the root element
            window.return_retained_root(entity_id, root);
            (layout_id, None) // Element stored in retained view
        } else {
            // Fallback path: no retained view (e.g., caching disabled)
            window.with_rendered_view(entity_id, |window| {
                // Explicit cached_style path (user-provided style)
                if let Some(style) = self.cached_style.as_ref() {
                    if !caching_disabled {
                        let mut root_style = Style::default();
                        root_style.refine(style);
                        let layout_id = window.request_layout(root_style, None, cx);
                        return (layout_id, None);
                    }
                }

                // Default path: render the element without retained tree
                let mut element = (self.render)(self, window, cx);
                let layout_id = element.request_layout(window, cx);
                (layout_id, Some(element))
            })
        };

        window.register_view_layout(entity_id, result.0);
        result
    }

    fn prepaint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _element: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<AnyElement> {
        let entity_id = self.entity_id();
        window.set_view_id(entity_id);

        eprintln!("[ANYVIEW prepaint] entity={:?} bounds={:?}", entity_id, bounds);

        // Get the root_id from retained view
        let root_id = window
            .get_retained_view(entity_id)
            .and_then(|r| r.root_id);

        if let Some(root_id) = root_id {
            // Take the root element to avoid borrow conflicts
            let mut root = window
                .take_retained_root(entity_id)
                .expect("retained root missing in prepaint");

            window.with_retained_context(entity_id, root_id, false, |window| {
                window.with_rendered_view(entity_id, |window| root.prepaint(window, cx));
            });

            // Update cached size for automatic view caching
            window.view_cache_sizes.insert(entity_id, bounds.size);

            // Return the root element
            window.return_retained_root(entity_id, root);
            eprintln!("[ANYVIEW prepaint] entity={:?} DONE with retained", entity_id);
            None // Element stored in retained view
        } else {
            // Fallback: no retained view
            eprintln!("[ANYVIEW prepaint] entity={:?} FALLBACK no retained view", entity_id);
            None
        }
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _element: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let entity_id = self.entity_id();
        eprintln!("[ANYVIEW paint] entity={:?}", entity_id);

        // Get the root_id from retained view
        let root_id = window
            .get_retained_view(entity_id)
            .and_then(|r| r.root_id);

        if let Some(root_id) = root_id {
            // Take the root element to avoid borrow conflicts
            let mut root = window
                .take_retained_root(entity_id)
                .expect("retained root missing in paint");

            window.with_retained_context(entity_id, root_id, false, |window| {
                window.with_rendered_view(entity_id, |window| root.paint(window, cx));
            });

            // Return the root element
            window.return_retained_root(entity_id, root);
            eprintln!("[ANYVIEW paint] entity={:?} DONE with retained", entity_id);
        } else {
            // Fallback: no retained view
            eprintln!("[ANYVIEW paint] entity={:?} FALLBACK no retained view", entity_id);
        }
    }
}

impl<V: 'static + Render> IntoElement for Entity<V> {
    type Element = Entity<V>;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl IntoElement for AnyView {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// A weak, dynamically-typed view handle that does not prevent the view from being released.
pub struct AnyWeakView {
    entity: AnyWeakEntity,
    render: fn(&AnyView, &mut Window, &mut App) -> AnyElement,
}

impl AnyWeakView {
    /// Convert to a strongly-typed handle if the referenced view has not yet been released.
    pub fn upgrade(&self) -> Option<AnyView> {
        let entity = self.entity.upgrade()?;
        Some(AnyView {
            entity,
            render: self.render,
            cached_style: None,
        })
    }
}

impl<V: 'static + Render> From<WeakEntity<V>> for AnyWeakView {
    fn from(view: WeakEntity<V>) -> Self {
        AnyWeakView {
            entity: view.into(),
            render: any_view::render::<V>,
        }
    }
}

impl PartialEq for AnyWeakView {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl std::fmt::Debug for AnyWeakView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyWeakView")
            .field("entity_id", &self.entity.entity_id)
            .finish_non_exhaustive()
    }
}

mod any_view {
    use crate::{AnyElement, AnyView, App, IntoElement, Render, Window};

    pub(crate) fn render<V: 'static + Render>(
        view: &AnyView,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyElement {
        let view = view.clone().downcast::<V>().unwrap();
        view.update(cx, |view, cx| view.render(window, cx).into_any_element())
    }
}

/// A view that renders nothing
pub struct EmptyView;

impl Render for EmptyView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        Empty
    }
}
