use crate::{
    AbsoluteLength, AnyElement, App, Bounds, DefiniteLength, Edges, Length, Pixels, Point,
    Size, Style, Window, display_list::ComputedElementId, point, size,
};
use bitflags::bitflags;
use collections::{FxHashMap, FxHashSet};
use stacksafe::{StackSafe, stacksafe};
use std::{fmt::Debug, ops::Range};
use taffy::{
    TaffyTree, TraversePartialTree as _,
    geometry::{Point as TaffyPoint, Rect as TaffyRect, Size as TaffySize},
    prelude::min_content,
    style::AvailableSpace as TaffyAvailableSpace,
    tree::NodeId,
};

bitflags! {
    /// Flags indicating what work needs to be done for a node.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct DirtyFlags: u8 {
        /// The node's view needs to call render() again.
        const NEEDS_RENDER  = 0b0001;
        /// The node needs layout recomputation.
        const NEEDS_LAYOUT  = 0b0010;
        /// The node needs to be repainted.
        const NEEDS_PAINT   = 0b0100;
        /// At least one descendant is dirty (for tree traversal optimization).
        const SUBTREE_DIRTY = 0b1000;
    }
}

type NodeMeasureFn = StackSafe<
    Box<
        dyn FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> Size<Pixels>,
    >,
>;

/// Cached prepaint state for an element (hitboxes, focus handles, etc.).
/// Placeholder for Phase 6 implementation.
#[derive(Default)]
pub struct CachedPrepaint {
    // TODO: Add prepaint cached state
}

/// Cached paint output for an element (display items).
/// Placeholder for Phase 6 implementation.
#[derive(Default)]
pub struct CachedPaint {
    // TODO: Add paint cached state (display items, bounds)
}

struct NodeContext {
    /// Measure function for leaf nodes with intrinsic sizing.
    measure: Option<NodeMeasureFn>,

    /// Generation when this node was last accessed.
    last_access: u64,

    /// The element stored in this node for retention across frames.
    element: Option<AnyElement>,

    /// Dirty flags indicating what work needs to be done.
    dirty_flags: DirtyFlags,

    /// Cached prepaint state (hitboxes, etc.).
    cached_prepaint: Option<CachedPrepaint>,

    /// Cached paint output (display items).
    cached_paint: Option<CachedPaint>,
}

/// Statistics for layout cache performance.
#[derive(Default, Clone, Copy, Debug)]
pub struct LayoutCacheStats {
    /// Number of cache hits (reused nodes)
    pub hits: u64,
    /// Number of cache misses (new nodes created)
    pub misses: u64,
    /// Number of stale nodes pruned
    pub pruned: u64,
}

pub struct TaffyLayoutEngine {
    taffy: TaffyTree<NodeContext>,
    absolute_layout_bounds: FxHashMap<LayoutId, Bounds<Pixels>>,
    computed_layouts: FxHashSet<LayoutId>,
    layout_bounds_scratch_space: Vec<LayoutId>,
    /// Maps element IDs to their Taffy node IDs for reuse across frames
    element_to_node: FxHashMap<ComputedElementId, NodeId>,
    /// Current frame generation
    generation: u64,
    /// Statistics for cache performance
    stats: LayoutCacheStats,
}

const EXPECT_MESSAGE: &str = "we should avoid taffy layout errors by construction if possible";

impl TaffyLayoutEngine {
    pub fn new() -> Self {
        let mut taffy = TaffyTree::new();
        taffy.enable_rounding();
        TaffyLayoutEngine {
            taffy,
            absolute_layout_bounds: FxHashMap::default(),
            computed_layouts: FxHashSet::default(),
            layout_bounds_scratch_space: Vec::new(),
            element_to_node: FxHashMap::default(),
            generation: 0,
            stats: LayoutCacheStats::default(),
        }
    }

    /// Called at the start of each frame to prepare for layout
    pub fn begin_frame(&mut self) {
        self.generation += 1;
        self.absolute_layout_bounds.clear();
        self.computed_layouts.clear();
    }

    /// Called at the end of each frame to prune stale nodes
    pub fn end_frame(&mut self) {
        // Remove nodes that weren't accessed this frame
        let current_generation = self.generation;
        let taffy = &mut self.taffy;
        let stats = &mut self.stats;

        self.element_to_node.retain(|_element_id, &mut node_id| {
            if let Some(context) = taffy.get_node_context(node_id) {
                if context.last_access < current_generation {
                    // Node wasn't accessed this frame - remove it
                    let _ = taffy.remove(node_id);
                    stats.pruned += 1;
                    return false;
                }
            }
            true
        });
    }

    /// Returns the stats and resets the counters.
    pub fn take_stats(&mut self) -> LayoutCacheStats {
        std::mem::take(&mut self.stats)
    }

    /// Returns the current stats without resetting them.
    #[allow(dead_code)]
    pub fn peek_stats(&self) -> LayoutCacheStats {
        self.stats
    }

    /// Returns a copy of the current stats (Phase 5: Instrumentation).
    pub fn stats(&self) -> LayoutCacheStats {
        self.stats
    }

    /// Request layout for an element with a known identity.
    /// Reuses the existing Taffy node if available.
    pub fn request_layout_with_id(
        &mut self,
        element_id: ComputedElementId,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        children: &[LayoutId],
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);

        if let Some(&node_id) = self.element_to_node.get(&element_id) {
            // Reuse existing node
            if let Some(context) = self.taffy.get_node_context_mut(node_id) {
                context.last_access = self.generation;
                context.measure = None; // Clear any old measure function
            }

            // Update style
            self.taffy.set_style(node_id, taffy_style).expect(EXPECT_MESSAGE);

            // Update children
            self.taffy
                .set_children(node_id, LayoutId::to_taffy_slice(children))
                .expect(EXPECT_MESSAGE);

            self.stats.hits += 1;
            node_id.into()
        } else {
            // Create new node
            let node_id = if children.is_empty() {
                self.taffy
                    .new_leaf_with_context(
                        taffy_style,
                        NodeContext {
                            measure: None,
                            last_access: self.generation,
                            element: None,
                            dirty_flags: DirtyFlags::empty(),
                            cached_prepaint: None,
                            cached_paint: None,
                        },
                    )
                    .expect(EXPECT_MESSAGE)
            } else {
                self.taffy
                    .new_with_children(taffy_style, LayoutId::to_taffy_slice(children))
                    .expect(EXPECT_MESSAGE)
            };

            // For nodes created with new_with_children, set context manually
            if !children.is_empty() {
                self.taffy
                    .set_node_context(
                        node_id,
                        Some(NodeContext {
                            measure: None,
                            last_access: self.generation,
                            element: None,
                            dirty_flags: DirtyFlags::empty(),
                            cached_prepaint: None,
                            cached_paint: None,
                        }),
                    )
                    .expect(EXPECT_MESSAGE);
            }

            self.element_to_node.insert(element_id, node_id);
            self.stats.misses += 1;
            node_id.into()
        }
    }

    /// Request layout for an element with a known identity and a measure function.
    /// Reuses the existing Taffy node if available (but always updates the measure function).
    pub fn request_measured_layout_with_id(
        &mut self,
        element_id: ComputedElementId,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        measure: impl FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> Size<Pixels>
            + 'static,
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);
        // Cast to dyn FnMut to match the NodeMeasureFn type
        let boxed: Box<
            dyn FnMut(
                Size<Option<Pixels>>,
                Size<AvailableSpace>,
                &mut Window,
                &mut App,
            ) -> Size<Pixels>,
        > = Box::new(measure);
        let measure_fn: NodeMeasureFn = StackSafe::new(boxed);

        if let Some(&node_id) = self.element_to_node.get(&element_id) {
            // Reuse existing node - update style and measure function
            self.taffy.set_style(node_id, taffy_style).expect(EXPECT_MESSAGE);

            // Update context with new measure function
            self.taffy
                .set_node_context(
                    node_id,
                    Some(NodeContext {
                        measure: Some(measure_fn),
                        last_access: self.generation,
                        element: None,
                        dirty_flags: DirtyFlags::empty(),
                        cached_prepaint: None,
                        cached_paint: None,
                    }),
                )
                .expect(EXPECT_MESSAGE);

            // Mark dirty since measure function changed
            self.taffy.mark_dirty(node_id).expect(EXPECT_MESSAGE);

            self.stats.hits += 1;
            node_id.into()
        } else {
            // Create new node
            let node_id = self
                .taffy
                .new_leaf_with_context(
                    taffy_style,
                    NodeContext {
                        measure: Some(measure_fn),
                        last_access: self.generation,
                        element: None,
                        dirty_flags: DirtyFlags::empty(),
                        cached_prepaint: None,
                        cached_paint: None,
                    },
                )
                .expect(EXPECT_MESSAGE);

            self.element_to_node.insert(element_id, node_id);
            self.stats.misses += 1;
            node_id.into()
        }
    }

    pub fn request_layout(
        &mut self,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        children: &[LayoutId],
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);
        let context = NodeContext {
            measure: None,
            last_access: self.generation,
            element: None,
            dirty_flags: DirtyFlags::empty(),
            cached_prepaint: None,
            cached_paint: None,
        };

        if children.is_empty() {
            self.taffy
                .new_leaf_with_context(taffy_style, context)
                .expect(EXPECT_MESSAGE)
                .into()
        } else {
            let node_id = self.taffy
                // This is safe because LayoutId is repr(transparent) to taffy::tree::NodeId.
                .new_with_children(taffy_style, LayoutId::to_taffy_slice(children))
                .expect(EXPECT_MESSAGE);
            self.taffy.set_node_context(node_id, Some(context)).expect(EXPECT_MESSAGE);
            node_id.into()
        }
    }

    pub fn request_measured_layout(
        &mut self,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        measure: impl FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> Size<Pixels>
        + 'static,
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);

        self.taffy
            .new_leaf_with_context(
                taffy_style,
                NodeContext {
                    measure: Some(StackSafe::new(Box::new(measure))),
                    last_access: self.generation,
                    element: None,
                    dirty_flags: DirtyFlags::empty(),
                    cached_prepaint: None,
                    cached_paint: None,
                },
            )
            .expect(EXPECT_MESSAGE)
            .into()
    }

    // Used to understand performance
    #[allow(dead_code)]
    fn count_all_children(&self, parent: LayoutId) -> anyhow::Result<u32> {
        let mut count = 0;

        for child in self.taffy.children(parent.0)? {
            // Count this child.
            count += 1;

            // Count all of this child's children.
            count += self.count_all_children(LayoutId(child))?
        }

        Ok(count)
    }

    // Used to understand performance
    #[allow(dead_code)]
    fn max_depth(&self, depth: u32, parent: LayoutId) -> anyhow::Result<u32> {
        println!(
            "{parent:?} at depth {depth} has {} children",
            self.taffy.child_count(parent.0)
        );

        let mut max_child_depth = 0;

        for child in self.taffy.children(parent.0)? {
            max_child_depth = std::cmp::max(max_child_depth, self.max_depth(0, LayoutId(child))?);
        }

        Ok(depth + 1 + max_child_depth)
    }

    // Used to understand performance
    #[allow(dead_code)]
    fn get_edges(&self, parent: LayoutId) -> anyhow::Result<Vec<(LayoutId, LayoutId)>> {
        let mut edges = Vec::new();

        for child in self.taffy.children(parent.0)? {
            edges.push((parent, LayoutId(child)));

            edges.extend(self.get_edges(LayoutId(child))?);
        }

        Ok(edges)
    }

    #[stacksafe]
    pub fn compute_layout(
        &mut self,
        id: LayoutId,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) {
        // Leaving this here until we have a better instrumentation approach.
        // println!("Laying out {} children", self.count_all_children(id)?);
        // println!("Max layout depth: {}", self.max_depth(0, id)?);

        // Output the edges (branches) of the tree in Mermaid format for visualization.
        // println!("Edges:");
        // for (a, b) in self.get_edges(id)? {
        //     println!("N{} --> N{}", u64::from(a), u64::from(b));
        // }
        //

        if !self.computed_layouts.insert(id) {
            let mut stack = &mut self.layout_bounds_scratch_space;
            stack.push(id);
            while let Some(id) = stack.pop() {
                self.absolute_layout_bounds.remove(&id);
                stack.extend(
                    self.taffy
                        .children(id.into())
                        .expect(EXPECT_MESSAGE)
                        .into_iter()
                        .map(LayoutId::from),
                );
            }
        }

        let scale_factor = window.scale_factor();

        let transform = |v: AvailableSpace| match v {
            AvailableSpace::Definite(pixels) => {
                AvailableSpace::Definite(Pixels(pixels.0 * scale_factor))
            }
            AvailableSpace::MinContent => AvailableSpace::MinContent,
            AvailableSpace::MaxContent => AvailableSpace::MaxContent,
        };
        let available_space = size(
            transform(available_space.width),
            transform(available_space.height),
        );

        self.taffy
            .compute_layout_with_measure(
                id.into(),
                available_space.into(),
                |known_dimensions, available_space, _id, node_context, _style| {
                    let Some(node_context) = node_context else {
                        return taffy::geometry::Size::default();
                    };

                    let Some(ref mut measure) = node_context.measure else {
                        return taffy::geometry::Size::default();
                    };

                    let known_dimensions = Size {
                        width: known_dimensions.width.map(|e| Pixels(e / scale_factor)),
                        height: known_dimensions.height.map(|e| Pixels(e / scale_factor)),
                    };

                    let available_space: Size<AvailableSpace> = available_space.into();
                    let untransform = |ev: AvailableSpace| match ev {
                        AvailableSpace::Definite(pixels) => {
                            AvailableSpace::Definite(Pixels(pixels.0 / scale_factor))
                        }
                        AvailableSpace::MinContent => AvailableSpace::MinContent,
                        AvailableSpace::MaxContent => AvailableSpace::MaxContent,
                    };
                    let available_space = size(
                        untransform(available_space.width),
                        untransform(available_space.height),
                    );

                    let a: Size<Pixels> = measure(known_dimensions, available_space, window, cx);
                    size(a.width.0 * scale_factor, a.height.0 * scale_factor).into()
                },
            )
            .expect(EXPECT_MESSAGE);
    }

    pub fn layout_bounds(&mut self, id: LayoutId, scale_factor: f32) -> Bounds<Pixels> {
        if let Some(layout) = self.absolute_layout_bounds.get(&id).cloned() {
            return layout;
        }

        let layout = self.taffy.layout(id.into()).expect(EXPECT_MESSAGE);
        let mut bounds = Bounds {
            origin: point(
                Pixels(layout.location.x / scale_factor),
                Pixels(layout.location.y / scale_factor),
            ),
            size: size(
                Pixels(layout.size.width / scale_factor),
                Pixels(layout.size.height / scale_factor),
            ),
        };

        if let Some(parent_id) = self.taffy.parent(id.0) {
            let parent_bounds = self.layout_bounds(parent_id.into(), scale_factor);
            bounds.origin += parent_bounds.origin;
        }
        self.absolute_layout_bounds.insert(id, bounds);

        bounds
    }

    // ============================================================
    // Element Storage Methods
    // ============================================================

    /// Store an element in the node's context.
    pub fn store_element(&mut self, id: LayoutId, element: AnyElement) {
        if let Some(context) = self.taffy.get_node_context_mut(id.into()) {
            context.element = Some(element);
        }
    }

    /// Get a reference to the element stored in the node's context.
    pub fn get_element(&self, id: LayoutId) -> Option<&AnyElement> {
        self.taffy
            .get_node_context(id.into())
            .and_then(|ctx| ctx.element.as_ref())
    }

    /// Take the element from the node's context, leaving None in its place.
    pub fn take_element(&mut self, id: LayoutId) -> Option<AnyElement> {
        self.taffy
            .get_node_context_mut(id.into())
            .and_then(|ctx| ctx.element.take())
    }

    // ============================================================
    // Dirty Flag Methods
    // ============================================================

    /// Get the dirty flags for a node.
    pub fn get_dirty_flags(&self, id: LayoutId) -> DirtyFlags {
        self.taffy
            .get_node_context(id.into())
            .map(|ctx| ctx.dirty_flags)
            .unwrap_or(DirtyFlags::empty())
    }

    /// Set the dirty flags for a node (replaces existing flags).
    pub fn set_dirty_flags(&mut self, id: LayoutId, flags: DirtyFlags) {
        if let Some(context) = self.taffy.get_node_context_mut(id.into()) {
            context.dirty_flags = flags;
        }
    }

    /// Mark a node with dirty flags and propagate SUBTREE_DIRTY up to ancestors.
    pub fn mark_dirty_with_flags(&mut self, id: LayoutId, flags: DirtyFlags) {
        // Set flags on this node
        if let Some(context) = self.taffy.get_node_context_mut(id.into()) {
            context.dirty_flags |= flags;
        }

        // Propagate SUBTREE_DIRTY up to ancestors
        let mut current = self.taffy.parent(id.into());
        while let Some(parent_id) = current {
            if let Some(context) = self.taffy.get_node_context_mut(parent_id) {
                if context.dirty_flags.contains(DirtyFlags::SUBTREE_DIRTY) {
                    break; // Already marked, ancestors must be too
                }
                context.dirty_flags |= DirtyFlags::SUBTREE_DIRTY;
            }
            current = self.taffy.parent(parent_id);
        }
    }

    /// Clear all dirty flags for a node.
    pub fn clear_dirty_flags(&mut self, id: LayoutId) {
        if let Some(context) = self.taffy.get_node_context_mut(id.into()) {
            context.dirty_flags = DirtyFlags::empty();
        }
    }

    // ============================================================
    // Cached Paint Methods
    // ============================================================

    /// Get a reference to the cached prepaint for a node.
    pub fn get_cached_prepaint(&self, id: LayoutId) -> Option<&CachedPrepaint> {
        self.taffy
            .get_node_context(id.into())
            .and_then(|ctx| ctx.cached_prepaint.as_ref())
    }

    /// Set the cached prepaint for a node.
    pub fn set_cached_prepaint(&mut self, id: LayoutId, cached: CachedPrepaint) {
        if let Some(context) = self.taffy.get_node_context_mut(id.into()) {
            context.cached_prepaint = Some(cached);
        }
    }

    /// Get a reference to the cached paint for a node.
    pub fn get_cached_paint(&self, id: LayoutId) -> Option<&CachedPaint> {
        self.taffy
            .get_node_context(id.into())
            .and_then(|ctx| ctx.cached_paint.as_ref())
    }

    /// Set the cached paint for a node.
    pub fn set_cached_paint(&mut self, id: LayoutId, cached: CachedPaint) {
        if let Some(context) = self.taffy.get_node_context_mut(id.into()) {
            context.cached_paint = Some(cached);
        }
    }

    // ============================================================
    // Tree Traversal Methods
    // ============================================================

    /// Get the parent of a node, if it has one.
    pub fn parent(&self, id: LayoutId) -> Option<LayoutId> {
        self.taffy.parent(id.into()).map(LayoutId::from)
    }

    /// Get the children of a node.
    pub fn children(&self, id: LayoutId) -> Vec<LayoutId> {
        self.taffy
            .children(id.into())
            .map(|children| children.into_iter().map(LayoutId::from).collect())
            .unwrap_or_default()
    }
}

/// A unique identifier for a layout node, generated when requesting a layout from Taffy
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(transparent)]
pub struct LayoutId(NodeId);

impl LayoutId {
    fn to_taffy_slice(node_ids: &[Self]) -> &[taffy::NodeId] {
        // SAFETY: LayoutId is repr(transparent) to taffy::tree::NodeId.
        unsafe { std::mem::transmute::<&[LayoutId], &[taffy::NodeId]>(node_ids) }
    }
}

impl std::hash::Hash for LayoutId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        u64::from(self.0).hash(state);
    }
}

impl From<NodeId> for LayoutId {
    fn from(node_id: NodeId) -> Self {
        Self(node_id)
    }
}

impl From<LayoutId> for NodeId {
    fn from(layout_id: LayoutId) -> NodeId {
        layout_id.0
    }
}

trait ToTaffy<Output> {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> Output;
}

impl ToTaffy<taffy::style::Style> for Style {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Style {
        use taffy::style_helpers::{fr, length, minmax, repeat};

        fn to_grid_line(
            placement: &Range<crate::GridPlacement>,
        ) -> taffy::Line<taffy::GridPlacement> {
            taffy::Line {
                start: placement.start.into(),
                end: placement.end.into(),
            }
        }

        fn to_grid_repeat<T: taffy::style::CheapCloneStr>(
            unit: &Option<u16>,
        ) -> Vec<taffy::GridTemplateComponent<T>> {
            // grid-template-columns: repeat(<number>, minmax(0, 1fr));
            unit.map(|count| vec![repeat(count, vec![minmax(length(0.0), fr(1.0))])])
                .unwrap_or_default()
        }

        fn to_grid_repeat_min_content<T: taffy::style::CheapCloneStr>(
            unit: &Option<u16>,
        ) -> Vec<taffy::GridTemplateComponent<T>> {
            // grid-template-columns: repeat(<number>, minmax(min-content, 1fr));
            unit.map(|count| vec![repeat(count, vec![minmax(min_content(), fr(1.0))])])
                .unwrap_or_default()
        }

        taffy::style::Style {
            display: self.display.into(),
            overflow: self.overflow.into(),
            scrollbar_width: self.scrollbar_width.to_taffy(rem_size, scale_factor),
            position: self.position.into(),
            inset: self.inset.to_taffy(rem_size, scale_factor),
            size: self.size.to_taffy(rem_size, scale_factor),
            min_size: self.min_size.to_taffy(rem_size, scale_factor),
            max_size: self.max_size.to_taffy(rem_size, scale_factor),
            aspect_ratio: self.aspect_ratio,
            margin: self.margin.to_taffy(rem_size, scale_factor),
            padding: self.padding.to_taffy(rem_size, scale_factor),
            border: self.border_widths.to_taffy(rem_size, scale_factor),
            align_items: self.align_items.map(|x| x.into()),
            align_self: self.align_self.map(|x| x.into()),
            align_content: self.align_content.map(|x| x.into()),
            justify_content: self.justify_content.map(|x| x.into()),
            gap: self.gap.to_taffy(rem_size, scale_factor),
            flex_direction: self.flex_direction.into(),
            flex_wrap: self.flex_wrap.into(),
            flex_basis: self.flex_basis.to_taffy(rem_size, scale_factor),
            flex_grow: self.flex_grow,
            flex_shrink: self.flex_shrink,
            grid_template_rows: to_grid_repeat(&self.grid_rows),
            grid_template_columns: if self.grid_cols_min_content.is_some() {
                to_grid_repeat_min_content(&self.grid_cols_min_content)
            } else {
                to_grid_repeat(&self.grid_cols)
            },
            grid_row: self
                .grid_location
                .as_ref()
                .map(|location| to_grid_line(&location.row))
                .unwrap_or_default(),
            grid_column: self
                .grid_location
                .as_ref()
                .map(|location| to_grid_line(&location.column))
                .unwrap_or_default(),
            ..Default::default()
        }
    }
}

impl ToTaffy<f32> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> f32 {
        match self {
            AbsoluteLength::Pixels(pixels) => {
                let pixels: f32 = pixels.into();
                pixels * scale_factor
            }
            AbsoluteLength::Rems(rems) => {
                let pixels: f32 = (*rems * rem_size).into();
                pixels * scale_factor
            }
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentageAuto> for Length {
    fn to_taffy(
        &self,
        rem_size: Pixels,
        scale_factor: f32,
    ) -> taffy::prelude::LengthPercentageAuto {
        match self {
            Length::Definite(length) => length.to_taffy(rem_size, scale_factor),
            Length::Auto => taffy::prelude::LengthPercentageAuto::auto(),
        }
    }
}

impl ToTaffy<taffy::style::Dimension> for Length {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::prelude::Dimension {
        match self {
            Length::Definite(length) => length.to_taffy(rem_size, scale_factor),
            Length::Auto => taffy::prelude::Dimension::auto(),
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentage> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentage {
        match self {
            DefiniteLength::Absolute(length) => match length {
                AbsoluteLength::Pixels(pixels) => {
                    let pixels: f32 = pixels.into();
                    taffy::style::LengthPercentage::length(pixels * scale_factor)
                }
                AbsoluteLength::Rems(rems) => {
                    let pixels: f32 = (*rems * rem_size).into();
                    taffy::style::LengthPercentage::length(pixels * scale_factor)
                }
            },
            DefiniteLength::Fraction(fraction) => {
                taffy::style::LengthPercentage::percent(*fraction)
            }
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentageAuto> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentageAuto {
        match self {
            DefiniteLength::Absolute(length) => match length {
                AbsoluteLength::Pixels(pixels) => {
                    let pixels: f32 = pixels.into();
                    taffy::style::LengthPercentageAuto::length(pixels * scale_factor)
                }
                AbsoluteLength::Rems(rems) => {
                    let pixels: f32 = (*rems * rem_size).into();
                    taffy::style::LengthPercentageAuto::length(pixels * scale_factor)
                }
            },
            DefiniteLength::Fraction(fraction) => {
                taffy::style::LengthPercentageAuto::percent(*fraction)
            }
        }
    }
}

impl ToTaffy<taffy::style::Dimension> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Dimension {
        match self {
            DefiniteLength::Absolute(length) => match length {
                AbsoluteLength::Pixels(pixels) => {
                    let pixels: f32 = pixels.into();
                    taffy::style::Dimension::length(pixels * scale_factor)
                }
                AbsoluteLength::Rems(rems) => {
                    taffy::style::Dimension::length((*rems * rem_size * scale_factor).into())
                }
            },
            DefiniteLength::Fraction(fraction) => taffy::style::Dimension::percent(*fraction),
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentage> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentage {
        match self {
            AbsoluteLength::Pixels(pixels) => {
                let pixels: f32 = pixels.into();
                taffy::style::LengthPercentage::length(pixels * scale_factor)
            }
            AbsoluteLength::Rems(rems) => {
                let pixels: f32 = (*rems * rem_size).into();
                taffy::style::LengthPercentage::length(pixels * scale_factor)
            }
        }
    }
}

impl<T, T2> From<TaffyPoint<T>> for Point<T2>
where
    T: Into<T2>,
    T2: Clone + Debug + Default + PartialEq,
{
    fn from(point: TaffyPoint<T>) -> Point<T2> {
        Point {
            x: point.x.into(),
            y: point.y.into(),
        }
    }
}

impl<T, T2> From<Point<T>> for TaffyPoint<T2>
where
    T: Into<T2> + Clone + Debug + Default + PartialEq,
{
    fn from(val: Point<T>) -> Self {
        TaffyPoint {
            x: val.x.into(),
            y: val.y.into(),
        }
    }
}

impl<T, U> ToTaffy<TaffySize<U>> for Size<T>
where
    T: ToTaffy<U> + Clone + Debug + Default + PartialEq,
{
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> TaffySize<U> {
        TaffySize {
            width: self.width.to_taffy(rem_size, scale_factor),
            height: self.height.to_taffy(rem_size, scale_factor),
        }
    }
}

impl<T, U> ToTaffy<TaffyRect<U>> for Edges<T>
where
    T: ToTaffy<U> + Clone + Debug + Default + PartialEq,
{
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> TaffyRect<U> {
        TaffyRect {
            top: self.top.to_taffy(rem_size, scale_factor),
            right: self.right.to_taffy(rem_size, scale_factor),
            bottom: self.bottom.to_taffy(rem_size, scale_factor),
            left: self.left.to_taffy(rem_size, scale_factor),
        }
    }
}

impl<T, U> From<TaffySize<T>> for Size<U>
where
    T: Into<U>,
    U: Clone + Debug + Default + PartialEq,
{
    fn from(taffy_size: TaffySize<T>) -> Self {
        Size {
            width: taffy_size.width.into(),
            height: taffy_size.height.into(),
        }
    }
}

impl<T, U> From<Size<T>> for TaffySize<U>
where
    T: Into<U> + Clone + Debug + Default + PartialEq,
{
    fn from(size: Size<T>) -> Self {
        TaffySize {
            width: size.width.into(),
            height: size.height.into(),
        }
    }
}

/// The space available for an element to be laid out in
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum AvailableSpace {
    /// The amount of space available is the specified number of pixels
    Definite(Pixels),
    /// The amount of space available is indefinite and the node should be laid out under a min-content constraint
    #[default]
    MinContent,
    /// The amount of space available is indefinite and the node should be laid out under a max-content constraint
    MaxContent,
}

impl AvailableSpace {
    /// Returns a `Size` with both width and height set to `AvailableSpace::MinContent`.
    ///
    /// This function is useful when you want to create a `Size` with the minimum content constraints
    /// for both dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use gpui::AvailableSpace;
    /// let min_content_size = AvailableSpace::min_size();
    /// assert_eq!(min_content_size.width, AvailableSpace::MinContent);
    /// assert_eq!(min_content_size.height, AvailableSpace::MinContent);
    /// ```
    pub const fn min_size() -> Size<Self> {
        Size {
            width: Self::MinContent,
            height: Self::MinContent,
        }
    }
}

impl From<AvailableSpace> for TaffyAvailableSpace {
    fn from(space: AvailableSpace) -> TaffyAvailableSpace {
        match space {
            AvailableSpace::Definite(Pixels(value)) => TaffyAvailableSpace::Definite(value),
            AvailableSpace::MinContent => TaffyAvailableSpace::MinContent,
            AvailableSpace::MaxContent => TaffyAvailableSpace::MaxContent,
        }
    }
}

impl From<TaffyAvailableSpace> for AvailableSpace {
    fn from(space: TaffyAvailableSpace) -> AvailableSpace {
        match space {
            TaffyAvailableSpace::Definite(value) => AvailableSpace::Definite(Pixels(value)),
            TaffyAvailableSpace::MinContent => AvailableSpace::MinContent,
            TaffyAvailableSpace::MaxContent => AvailableSpace::MaxContent,
        }
    }
}

impl From<Pixels> for AvailableSpace {
    fn from(pixels: Pixels) -> Self {
        AvailableSpace::Definite(pixels)
    }
}

impl From<Size<Pixels>> for Size<AvailableSpace> {
    fn from(size: Size<Pixels>) -> Self {
        Size {
            width: AvailableSpace::Definite(size.width),
            height: AvailableSpace::Definite(size.height),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirty_flags_operations() {
        let mut flags = DirtyFlags::empty();
        assert!(flags.is_empty());

        flags |= DirtyFlags::NEEDS_RENDER;
        assert!(flags.contains(DirtyFlags::NEEDS_RENDER));
        assert!(!flags.contains(DirtyFlags::NEEDS_LAYOUT));

        flags |= DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT;
        assert!(flags.contains(DirtyFlags::NEEDS_RENDER));
        assert!(flags.contains(DirtyFlags::NEEDS_LAYOUT));
        assert!(flags.contains(DirtyFlags::NEEDS_PAINT));
        assert!(!flags.contains(DirtyFlags::SUBTREE_DIRTY));

        let combined = DirtyFlags::NEEDS_RENDER | DirtyFlags::SUBTREE_DIRTY;
        assert!(combined.intersects(DirtyFlags::NEEDS_RENDER));
        assert!(combined.intersects(DirtyFlags::SUBTREE_DIRTY));
        assert!(!combined.intersects(DirtyFlags::NEEDS_LAYOUT));
    }

    #[test]
    fn test_dirty_flags_default() {
        let flags = DirtyFlags::default();
        assert!(flags.is_empty());
    }

    #[test]
    fn test_layout_engine_dirty_flag_methods() {
        let mut engine = TaffyLayoutEngine::new();
        let style = Style::default();
        let rem_size = Pixels(16.0);
        let scale_factor = 1.0;

        let layout_id = engine.request_layout(style.clone(), rem_size, scale_factor, &[]);

        // Initially flags should be empty
        assert!(engine.get_dirty_flags(layout_id).is_empty());

        // Set flags
        engine.set_dirty_flags(layout_id, DirtyFlags::NEEDS_PAINT);
        assert!(engine.get_dirty_flags(layout_id).contains(DirtyFlags::NEEDS_PAINT));

        // Clear flags
        engine.clear_dirty_flags(layout_id);
        assert!(engine.get_dirty_flags(layout_id).is_empty());
    }

    #[test]
    fn test_dirty_flag_propagation() {
        let mut engine = TaffyLayoutEngine::new();
        let style = Style::default();
        let rem_size = Pixels(16.0);
        let scale_factor = 1.0;

        // Create a tree: grandparent -> parent -> child
        let grandparent = engine.request_layout(style.clone(), rem_size, scale_factor, &[]);
        let parent = engine.request_layout(style.clone(), rem_size, scale_factor, &[]);
        let child = engine.request_layout(style.clone(), rem_size, scale_factor, &[]);

        // Re-create with proper parent-child relationships
        engine.taffy.set_children(grandparent.into(), &[parent.into()]).unwrap();
        engine.taffy.set_children(parent.into(), &[child.into()]).unwrap();

        // Mark the child dirty
        engine.mark_dirty_with_flags(child, DirtyFlags::NEEDS_PAINT);

        // Child should have NEEDS_PAINT
        assert!(engine.get_dirty_flags(child).contains(DirtyFlags::NEEDS_PAINT));

        // Parent and grandparent should have SUBTREE_DIRTY
        assert!(engine.get_dirty_flags(parent).contains(DirtyFlags::SUBTREE_DIRTY));
        assert!(engine.get_dirty_flags(grandparent).contains(DirtyFlags::SUBTREE_DIRTY));
    }

    #[test]
    fn test_tree_traversal() {
        let mut engine = TaffyLayoutEngine::new();
        let style = Style::default();
        let rem_size = Pixels(16.0);
        let scale_factor = 1.0;

        // Create a tree: parent -> [child1, child2]
        let child1 = engine.request_layout(style.clone(), rem_size, scale_factor, &[]);
        let child2 = engine.request_layout(style.clone(), rem_size, scale_factor, &[]);
        let parent = engine.request_layout(style.clone(), rem_size, scale_factor, &[child1, child2]);

        // Test parent() method
        assert_eq!(engine.parent(child1), Some(parent));
        assert_eq!(engine.parent(child2), Some(parent));
        assert_eq!(engine.parent(parent), None);

        // Test children() method
        let children = engine.children(parent);
        assert_eq!(children.len(), 2);
        assert!(children.contains(&child1));
        assert!(children.contains(&child2));
    }
}
