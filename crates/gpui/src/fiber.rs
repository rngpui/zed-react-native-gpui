use crate::{
    AnyDescriptor, App, AvailableSpace, Bounds, EntityId, Hitbox, Pixels, SharedString, Size,
    StyleRefinement, Window,
};
use collections::FxHashMap;
use slotmap::{KeyData, new_key_type, SlotMap};
use taffy::{
    Cache, CacheTree, Layout, LayoutInput, LayoutOutput, LayoutPartialTree, NodeId, RoundTree,
    TraversePartialTree, TraverseTree,
    compute_cached_layout, compute_hidden_layout, compute_leaf_layout,
    LayoutBlockContainer, compute_block_layout,
    LayoutFlexboxContainer, compute_flexbox_layout,
    LayoutGridContainer, compute_grid_layout,
};
use taffy::style::{AvailableSpace as TaffyAvailableSpace, Display, Style as TaffyStyle};
use taffy::tree::RunMode;

/// A function that measures the size of a leaf node during layout.
/// Called with known dimensions (if any), available space, and the window/app context.
pub type MeasureFunc =
    Box<dyn FnMut(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels>>;

new_key_type! {
    /// A unique identifier for a fiber in the fiber tree
    pub struct FiberId;

    /// A unique identifier for a cached paint list
    pub struct PaintListId;

    /// A unique identifier for cached prepaint state
    pub struct PrepaintStateId;
}

impl FiberId {
    /// Convert this FiberId to a u64 for interop with Taffy's NodeId.
    pub fn as_u64(self) -> u64 {
        self.0.as_ffi()
    }
}

impl From<u64> for FiberId {
    fn from(value: u64) -> Self {
        FiberId(KeyData::from_ffi(value))
    }
}

impl From<NodeId> for FiberId {
    fn from(node_id: NodeId) -> Self {
        FiberId::from(u64::from(node_id))
    }
}

impl From<FiberId> for NodeId {
    fn from(fiber_id: FiberId) -> Self {
        NodeId::from(fiber_id.as_u64())
    }
}

/// Dirty flags for tracking what needs to be recomputed
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirtyFlags(u8);

impl DirtyFlags {
    /// No dirty flags set
    pub const NONE: DirtyFlags = DirtyFlags(0);
    /// Node needs layout recomputation
    pub const NEEDS_LAYOUT: DirtyFlags = DirtyFlags(0b0001);
    /// Node needs paint recomputation
    pub const NEEDS_PAINT: DirtyFlags = DirtyFlags(0b0010);
    /// Bounds changed due to ancestor layout updates
    pub const BOUNDS_CHANGED: DirtyFlags = DirtyFlags(0b0100);
    /// A descendant has dirty flags set
    pub const HAS_DIRTY_DESCENDANT: DirtyFlags = DirtyFlags(0b1000);
    /// Structure changed (children added/removed/reordered)
    pub const STRUCTURE_CHANGED: DirtyFlags = DirtyFlags(0b0001_0000);
    /// Style changed (layout-affecting or paint-only)
    pub const STYLE_CHANGED: DirtyFlags = DirtyFlags(0b0010_0000);
    /// Backwards-compatible alias for HAS_DIRTY_DESCENDANT
    pub const SUBTREE_DIRTY: DirtyFlags = DirtyFlags(0b1000);
    /// All flags set (for initial/full render)
    pub const ALL: DirtyFlags = DirtyFlags(0b0011_1111);

    /// Check if any flags are set
    pub fn any(self) -> bool {
        self.0 != 0
    }

    /// Check if specific flags are set
    pub fn contains(self, other: DirtyFlags) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Check if any of the specified flags are set
    pub fn intersects(self, other: DirtyFlags) -> bool {
        (self.0 & other.0) != 0
    }

    /// Add flags
    pub fn insert(&mut self, other: DirtyFlags) {
        self.0 |= other.0;
    }

    /// Remove flags
    pub fn remove(&mut self, other: DirtyFlags) {
        self.0 &= !other.0;
    }

    /// Clear all flags
    pub fn clear(&mut self) {
        self.0 = 0;
    }
}

impl std::ops::BitOr for DirtyFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        DirtyFlags(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for DirtyFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// A fiber node - the persistent unit of the UI tree.
/// Fibers cache expensive computed results across frames.
pub struct Fiber {
    /// Unique identifier for this fiber
    pub id: FiberId,

    /// Hash of the descriptor that created this fiber.
    /// Used to detect changes between frames.
    pub descriptor_hash: u64,

    /// Hash of layout-affecting descriptor properties.
    pub layout_hash: u64,

    /// Hash of paint-affecting descriptor properties.
    pub paint_hash: u64,

    /// Cached child count for structure-change detection.
    pub child_count: usize,

    /// Cached style snapshot for this fiber, if applicable.
    pub cached_style: Option<StyleRefinement>,

    /// Cached text content for this fiber, if applicable.
    pub cached_text: Option<SharedString>,

    /// Cached hitbox from the last prepaint, if any.
    pub cached_hitbox: Option<Hitbox>,

    /// Cached layout bounds from the last frame
    pub bounds: Option<Bounds<Pixels>>,

    /// Cached prepaint state for this fiber, if any.
    pub(crate) prepaint_state: Option<PrepaintStateId>,

    /// Cached paint list for this fiber, if any.
    pub(crate) paint_list: Option<PaintListId>,

    /// Parent fiber, if any
    pub parent: Option<FiberId>,

    /// First child fiber (linked list for children)
    pub first_child: Option<FiberId>,

    /// Next sibling fiber (linked list for children)
    pub next_sibling: Option<FiberId>,

    /// Dirty tracking flags
    pub dirty: DirtyFlags,

    /// Taffy cache for constraint combinations
    pub taffy_cache: Cache,

    /// Taffy style for this node
    pub taffy_style: TaffyStyle,

    /// Unrounded layout from Taffy
    pub unrounded_layout: Layout,

    /// Final rounded layout from Taffy
    pub final_layout: Layout,

    /// Optional measure function for leaf nodes
    pub measure_func: Option<MeasureFunc>,

    /// Cached hash for measured content
    pub measure_hash: Option<u64>,
}

impl Fiber {
    /// Create a new fiber with the given ID and descriptor hash
    pub fn new(
        id: FiberId,
        descriptor_hash: u64,
        layout_hash: u64,
        paint_hash: u64,
        child_count: usize,
    ) -> Self {
        Self {
            id,
            descriptor_hash,
            layout_hash,
            paint_hash,
            child_count,
            cached_style: None,
            cached_text: None,
            cached_hitbox: None,
            bounds: None,
            prepaint_state: None,
            paint_list: None,
            parent: None,
            first_child: None,
            next_sibling: None,
            dirty: DirtyFlags::ALL,
            taffy_cache: Cache::new(),
            taffy_style: TaffyStyle::default(),
            unrounded_layout: Layout::new(),
            final_layout: Layout::new(),
            measure_func: None,
            measure_hash: None,
        }
    }

    /// Check if this fiber or any descendant is dirty
    pub fn needs_work(&self) -> bool {
        self.dirty.any()
    }
}

/// The fiber tree - stores all fibers and manages their relationships
pub struct FiberTree {
    /// Storage for all fibers
    pub fibers: SlotMap<FiberId, Fiber>,

    /// The root fiber of the tree
    pub root: Option<FiberId>,

    /// Maps view entity IDs to their fiber roots
    pub view_roots: FxHashMap<EntityId, FiberId>,

    /// Layout context for measured nodes (set during layout computation)
    layout_context: Option<LayoutContext>,
}

struct LayoutContext {
    window: *mut Window,
    app: *mut App,
    scale_factor: f32,
}

impl Default for FiberTree {
    fn default() -> Self {
        Self::new()
    }
}

impl FiberTree {
    /// Create a new empty fiber tree
    pub fn new() -> Self {
        Self {
            fibers: SlotMap::with_key(),
            root: None,
            view_roots: FxHashMap::default(),
            layout_context: None,
        }
    }

    /// Create a new fiber and return its ID
    pub fn create_fiber_for(&mut self, descriptor: &AnyDescriptor) -> FiberId {
        let descriptor_hash = descriptor.content_hash();
        let layout_hash = descriptor.layout_hash();
        let paint_hash = descriptor.paint_hash();
        let child_count = descriptor.child_count();
        self.fibers.insert_with_key(|id| {
            Fiber::new(id, descriptor_hash, layout_hash, paint_hash, child_count)
        })
    }

    /// Get a fiber by ID
    pub fn get(&self, id: FiberId) -> Option<&Fiber> {
        self.fibers.get(id)
    }

    /// Get a mutable fiber by ID
    pub fn get_mut(&mut self, id: FiberId) -> Option<&mut Fiber> {
        self.fibers.get_mut(id)
    }

    /// Remove a fiber and all its descendants
    pub fn remove(&mut self, id: FiberId) {
        if let Some(parent_id) = self.fibers.get(id).and_then(|fiber| fiber.parent) {
            self.unlink_child(parent_id, id);
        }
        let mut to_remove = vec![id];

        while let Some(fiber_id) = to_remove.pop() {
            if let Some(fiber) = self.fibers.remove(fiber_id) {
                let mut child = fiber.first_child;
                while let Some(child_id) = child {
                    to_remove.push(child_id);
                    child = self.fibers.get(child_id).and_then(|f| f.next_sibling);
                }
            }
        }
    }

    /// Set the parent-child relationship between fibers
    pub fn set_parent(&mut self, child_id: FiberId, parent_id: FiberId) {
        if let Some(child) = self.fibers.get_mut(child_id) {
            child.parent = Some(parent_id);
        }
    }

    /// Add a child to a parent fiber
    pub fn add_child(&mut self, parent_id: FiberId, child_id: FiberId) {
        self.set_parent(child_id, parent_id);

        if let Some(parent) = self.fibers.get_mut(parent_id) {
            if parent.first_child.is_none() {
                parent.first_child = Some(child_id);
            } else {
                let mut current = parent.first_child;
                while let Some(id) = current {
                    let fiber = &self.fibers[id];
                    if fiber.next_sibling.is_none() {
                        break;
                    }
                    current = fiber.next_sibling;
                }

                if let Some(last_child_id) = current {
                    if let Some(last_child) = self.fibers.get_mut(last_child_id) {
                        last_child.next_sibling = Some(child_id);
                    }
                }
            }
        }

        self.invalidate_layout_cache_upwards(parent_id);
    }

    /// Mark a fiber as dirty and propagate SUBTREE_DIRTY to ancestors
    pub fn mark_dirty(&mut self, fiber_id: FiberId, flags: DirtyFlags) {
        if let Some(fiber) = self.fibers.get_mut(fiber_id) {
            fiber.dirty |= flags;
        }

        self.propagate_has_dirty_descendant(fiber_id);
        if flags.intersects(DirtyFlags::NEEDS_LAYOUT) {
            self.propagate_needs_layout(fiber_id);
            self.invalidate_layout_cache_upwards(fiber_id);
        }
    }

    /// Propagate SUBTREE_DIRTY flag up to ancestors
    pub fn propagate_has_dirty_descendant(&mut self, fiber_id: FiberId) {
        let mut current = self.fibers.get(fiber_id).and_then(|f| f.parent);

        while let Some(parent_id) = current {
            if let Some(parent) = self.fibers.get_mut(parent_id) {
                if parent.dirty.contains(DirtyFlags::HAS_DIRTY_DESCENDANT) {
                    break;
                }
                parent.dirty.insert(DirtyFlags::HAS_DIRTY_DESCENDANT);
                current = parent.parent;
            } else {
                break;
            }
        }
    }

    /// Propagate NEEDS_LAYOUT up to ancestors.
    pub fn propagate_needs_layout(&mut self, fiber_id: FiberId) {
        let mut current = self.fibers.get(fiber_id).and_then(|f| f.parent);

        while let Some(parent_id) = current {
            let Some(parent) = self.fibers.get_mut(parent_id) else {
                break;
            };
            if parent.dirty.contains(DirtyFlags::NEEDS_LAYOUT) {
                break;
            }
            parent.dirty.insert(DirtyFlags::NEEDS_LAYOUT);
            current = parent.parent;
        }
    }

    /// Backwards-compatible alias for propagate_has_dirty_descendant.
    pub fn propagate_subtree_dirty(&mut self, fiber_id: FiberId) {
        self.propagate_has_dirty_descendant(fiber_id);
    }

    /// Propagate BOUNDS_CHANGED flag down to descendants.
    pub fn propagate_bounds_changed(&mut self, fiber_id: FiberId) {
        let mut stack: Vec<FiberId> = self.children(fiber_id).collect();
        while let Some(current_id) = stack.pop() {
            if let Some(fiber) = self.fibers.get_mut(current_id) {
                fiber.dirty.insert(DirtyFlags::BOUNDS_CHANGED);
                let mut child = fiber.first_child;
                while let Some(child_id) = child {
                    stack.push(child_id);
                    child = self.fibers.get(child_id).and_then(|f| f.next_sibling);
                }
            }
        }
    }

    /// Clear dirty flags for a fiber (call after processing)
    pub fn clear_dirty(&mut self, fiber_id: FiberId) {
        if let Some(fiber) = self.fibers.get_mut(fiber_id) {
            fiber.dirty.clear();
        }
    }

    /// Iterate over children of a fiber
    pub fn children(&self, parent_id: FiberId) -> FiberChildIter<'_> {
        let first_child = self
            .fibers
            .get(parent_id)
            .and_then(|f| f.first_child);
        FiberChildIter {
            tree: self,
            current: first_child,
        }
    }

    /// Returns true if any ancestor of the given fiber is layout-dirty.
    pub fn has_layout_dirty_ancestor(&self, fiber_id: FiberId) -> bool {
        let mut current = self.fibers.get(fiber_id).and_then(|f| f.parent);
        while let Some(parent_id) = current {
            let Some(parent) = self.fibers.get(parent_id) else {
                break;
            };
            if parent.dirty.contains(DirtyFlags::NEEDS_LAYOUT) {
                return true;
            }
            current = parent.parent;
        }
        false
    }

    /// Returns true if any ancestor of the given fiber has any dirty flag set.
    pub fn has_any_dirty_ancestor(&self, fiber_id: FiberId) -> bool {
        let mut current = self.fibers.get(fiber_id).and_then(|f| f.parent);
        while let Some(parent_id) = current {
            let Some(parent) = self.fibers.get(parent_id) else {
                break;
            };
            if parent.dirty.any() {
                return true;
            }
            current = parent.parent;
        }
        false
    }

    /// Get the root fiber for a view entity
    pub fn get_view_root(&self, entity_id: EntityId) -> Option<FiberId> {
        self.view_roots.get(&entity_id).copied()
    }

    /// Set the root fiber for a view entity
    pub fn set_view_root(&mut self, entity_id: EntityId, fiber_id: FiberId) {
        self.view_roots.insert(entity_id, fiber_id);
    }

    /// Clear the entire tree
    pub fn clear(&mut self) {
        self.fibers.clear();
        self.root = None;
        self.view_roots.clear();
    }

    /// Clear dirty flags on all fibers.
    pub fn clear_all_dirty(&mut self) {
        for fiber in self.fibers.values_mut() {
            fiber.dirty.clear();
        }
    }

    pub(crate) fn set_layout_context(
        &mut self,
        window: *mut Window,
        app: *mut App,
        scale_factor: f32,
    ) {
        self.layout_context = Some(LayoutContext {
            window,
            app,
            scale_factor,
        });
    }

    pub(crate) fn clear_layout_context(&mut self) {
        self.layout_context = None;
    }

    /// Invalidate the layout cache for a fiber and all its ancestors.
    /// Called when a fiber's style changes to ensure layout is recomputed.
    pub fn invalidate_layout_cache_upwards(&mut self, fiber_id: FiberId) {
        let mut current = Some(fiber_id);
        while let Some(id) = current {
            let Some(fiber) = self.fibers.get_mut(id) else {
                break;
            };
            fiber.taffy_cache.clear();
            current = fiber.parent;
        }
    }

    /// Reconcile a descriptor against a fiber, returning whether it changed.
    /// If the descriptor's content hash differs from the fiber's stored hash,
    /// the fiber is marked as needing layout and its hash is updated.
    pub fn reconcile(
        &mut self,
        fiber_id: FiberId,
        descriptor: &AnyDescriptor,
    ) -> bool {
        let new_descriptor_hash = descriptor.content_hash();
        let new_layout_hash = descriptor.layout_hash();
        let new_paint_hash = descriptor.paint_hash();
        let new_child_count = descriptor.child_count();

        let mut changed = false;
        let mut dirty_flags = DirtyFlags::NONE;
        if let Some(fiber) = self.fibers.get_mut(fiber_id) {

            if fiber.child_count != new_child_count {
                dirty_flags |= DirtyFlags::STRUCTURE_CHANGED;
                dirty_flags |= DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT;
            }

            if fiber.layout_hash != new_layout_hash {
                dirty_flags |= DirtyFlags::STYLE_CHANGED;
                dirty_flags |= DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT;
            } else if fiber.paint_hash != new_paint_hash {
                dirty_flags |= DirtyFlags::STYLE_CHANGED;
                dirty_flags |= DirtyFlags::NEEDS_PAINT;
            }

            if fiber.descriptor_hash != new_descriptor_hash && dirty_flags == DirtyFlags::NONE {
                dirty_flags |= DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT;
            }

            if dirty_flags != DirtyFlags::NONE {
                fiber.descriptor_hash = new_descriptor_hash;
                fiber.layout_hash = new_layout_hash;
                fiber.paint_hash = new_paint_hash;
                fiber.child_count = new_child_count;
                Self::update_cached_props(fiber, descriptor);
                changed = true;
            } else {
                match descriptor {
                    AnyDescriptor::Div(_) if fiber.cached_style.is_none() => {
                        Self::update_cached_props(fiber, descriptor);
                    }
                    AnyDescriptor::Text(_) if fiber.cached_text.is_none() => {
                        Self::update_cached_props(fiber, descriptor);
                    }
                    _ => {}
                }
            }
        }
        if dirty_flags != DirtyFlags::NONE {
            self.mark_dirty(fiber_id, dirty_flags);
        }

        // Reconcile children by position
        let children = descriptor.children();
        self.reconcile_children(fiber_id, children);

        if changed {
            let child_ids: Vec<FiberId> = self.children(fiber_id).collect();
            for child_id in child_ids {
                if let Some(child) = self.fibers.get_mut(child_id) {
                    child.dirty.insert(DirtyFlags::BOUNDS_CHANGED);
                }
            }
        }

        changed
    }

    /// Reconcile children of a fiber against descriptor children by position.
    /// - Existing fibers at matching positions are reconciled
    /// - New fibers are created for additional children
    /// - Excess fibers are removed
    pub fn reconcile_children(
        &mut self,
        parent_id: FiberId,
        children: &[AnyDescriptor],
    ) {
        // Collect existing children
        let existing_children: Vec<FiberId> = self.children(parent_id).collect();

        // Reconcile or create children
        for (i, child_desc) in children.iter().enumerate() {
            if i < existing_children.len() {
                // Existing fiber at this position - reconcile it
                let child_fiber_id = existing_children[i];
                self.reconcile(child_fiber_id, child_desc);
            } else {
                // No fiber at this position - create new one
                let new_fiber_id = self.create_fiber_for(child_desc);
                self.add_child(parent_id, new_fiber_id);
                if let Some(fiber) = self.fibers.get_mut(new_fiber_id) {
                    Self::update_cached_props(fiber, child_desc);
                }

                // Recursively create fibers for descendants
                self.reconcile_children(new_fiber_id, child_desc.children());

                // Mark parent as having subtree dirty
                self.propagate_has_dirty_descendant(new_fiber_id);
            }
        }

        // Remove excess fibers (children that no longer exist)
        for excess_fiber_id in existing_children.into_iter().skip(children.len()) {
            self.remove(excess_fiber_id);
        }
    }

    /// Unlink a child from its parent's linked list
    fn unlink_child(&mut self, parent_id: FiberId, child_id: FiberId) {
        self.invalidate_layout_cache_upwards(parent_id);
        let parent = match self.fibers.get_mut(parent_id) {
            Some(p) => p,
            None => return,
        };

        if parent.first_child == Some(child_id) {
            // Child is the first child - update first_child to next sibling
            let next = self.fibers.get(child_id).and_then(|f| f.next_sibling);
            if let Some(parent) = self.fibers.get_mut(parent_id) {
                parent.first_child = next;
            }
        } else {
            // Find the previous sibling and update its next_sibling
            let mut prev = parent.first_child;
            while let Some(prev_id) = prev {
                let next = self.fibers.get(prev_id).and_then(|f| f.next_sibling);
                if next == Some(child_id) {
                    // Found the previous sibling
                    let child_next = self.fibers.get(child_id).and_then(|f| f.next_sibling);
                    if let Some(prev_fiber) = self.fibers.get_mut(prev_id) {
                        prev_fiber.next_sibling = child_next;
                    }
                    break;
                }
                prev = next;
            }
        }

        // Clear the child's parent reference
        if let Some(child) = self.fibers.get_mut(child_id) {
            child.parent = None;
            child.next_sibling = None;
        }
    }

    /// Ensure a root fiber exists, creating one if necessary
    pub fn ensure_root(&mut self, descriptor: &AnyDescriptor) -> FiberId {
        if let Some(root_id) = self.root {
            root_id
        } else {
            let root_id = self.create_fiber_for(descriptor);
            if let Some(fiber) = self.fibers.get_mut(root_id) {
                Self::update_cached_props(fiber, descriptor);
            }
            self.root = Some(root_id);
            root_id
        }
    }

    fn update_cached_props(fiber: &mut Fiber, descriptor: &AnyDescriptor) {
        match descriptor {
            AnyDescriptor::Div(desc) => {
                fiber.cached_style = Some(desc.style.clone());
                fiber.cached_text = None;
            }
            AnyDescriptor::Text(desc) => {
                fiber.cached_style = None;
                fiber.cached_text = Some(desc.text.clone());
            }
            AnyDescriptor::View(_) | AnyDescriptor::Deferred(_) | AnyDescriptor::Empty => {
                fiber.cached_style = None;
                fiber.cached_text = None;
            }
        }
    }
}

/// Iterator over children of a fiber
pub struct FiberChildIter<'a> {
    tree: &'a FiberTree,
    current: Option<FiberId>,
}

impl<'a> Iterator for FiberChildIter<'a> {
    type Item = FiberId;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current?;
        self.current = self.tree.fibers.get(current).and_then(|f| f.next_sibling);
        Some(current)
    }
}

fn fiber_to_node_id(fiber_id: FiberId) -> NodeId {
    fiber_id.into()
}

impl TraversePartialTree for FiberTree {
    type ChildIter<'a> = std::iter::Map<FiberChildIter<'a>, fn(FiberId) -> NodeId>
    where
        Self: 'a;

    fn child_ids(&self, node_id: NodeId) -> Self::ChildIter<'_> {
        self.children(node_id.into()).map(fiber_to_node_id)
    }

    fn child_count(&self, node_id: NodeId) -> usize {
        self.children(node_id.into()).count()
    }

    fn get_child_id(&self, node_id: NodeId, index: usize) -> NodeId {
        self.children(node_id.into())
            .nth(index)
            .unwrap_or_else(|| panic!("missing child {index} for node {node_id:?}"))
            .into()
    }
}

impl TraverseTree for FiberTree {}

impl LayoutPartialTree for FiberTree {
    type CoreContainerStyle<'a>
        = &'a TaffyStyle
    where
        Self: 'a;

    type CustomIdent = String;

    fn get_core_container_style(&self, node_id: NodeId) -> Self::CoreContainerStyle<'_> {
        &self
            .get(node_id.into())
            .unwrap_or_else(|| panic!("missing fiber for node {node_id:?}"))
            .taffy_style
    }

    fn set_unrounded_layout(&mut self, node_id: NodeId, layout: &Layout) {
        if let Some(fiber) = self.get_mut(node_id.into()) {
            fiber.unrounded_layout = *layout;
        }
    }

    fn compute_child_layout(&mut self, node_id: NodeId, inputs: LayoutInput) -> LayoutOutput {
        if inputs.run_mode == RunMode::PerformHiddenLayout {
            return compute_hidden_layout(self, node_id);
        }

        compute_cached_layout(self, node_id, inputs, |tree, node_id, inputs| {
            let (display_mode, has_children) = {
                let fiber = tree
                    .get(node_id.into())
                    .unwrap_or_else(|| panic!("missing fiber for node {node_id:?}"));
                (fiber.taffy_style.display, tree.child_count(node_id) > 0)
            };

            match (display_mode, has_children) {
                (Display::None, _) => compute_hidden_layout(tree, node_id),
                (Display::Block, true) => compute_block_layout(tree, node_id, inputs),
                (Display::Flex, true) => compute_flexbox_layout(tree, node_id, inputs),
                (Display::Grid, true) => compute_grid_layout(tree, node_id, inputs),
                (_, false) => {
                    let (window_ptr, app_ptr, scale_factor) = {
                        let context = tree
                            .layout_context
                            .as_ref()
                            .expect("layout context is not set");
                        (context.window, context.app, context.scale_factor)
                    };
                    let (style_ptr, measure_ptr) = {
                        let fiber = tree
                            .get_mut(node_id.into())
                            .unwrap_or_else(|| panic!("missing fiber for node {node_id:?}"));
                        let style_ptr = &fiber.taffy_style as *const TaffyStyle;
                        let measure_ptr =
                            fiber.measure_func.as_mut().map(|measure| measure as *mut MeasureFunc);
                        (style_ptr, measure_ptr)
                    };

                    let style = unsafe { &*style_ptr };
                    let measure_fn = |known_dimensions: taffy::geometry::Size<Option<f32>>,
                                      available_space: taffy::geometry::Size<TaffyAvailableSpace>| {
                        let Some(measure_ptr) = measure_ptr else {
                            return taffy::geometry::Size {
                                width: 0.0,
                                height: 0.0,
                            };
                        };

                        let to_available_space = |space: TaffyAvailableSpace| match space {
                            TaffyAvailableSpace::Definite(value) => {
                                AvailableSpace::Definite(Pixels(value / scale_factor))
                            }
                            TaffyAvailableSpace::MinContent => AvailableSpace::MinContent,
                            TaffyAvailableSpace::MaxContent => AvailableSpace::MaxContent,
                        };

                        let known_dimensions = Size {
                            width: known_dimensions
                                .width
                                .map(|value| Pixels(value / scale_factor)),
                            height: known_dimensions
                                .height
                                .map(|value| Pixels(value / scale_factor)),
                        };
                        let available_space = Size {
                            width: to_available_space(available_space.width),
                            height: to_available_space(available_space.height),
                        };
                        let measured = unsafe {
                            (&mut *measure_ptr)(
                                known_dimensions,
                                available_space,
                                &mut *window_ptr,
                                &mut *app_ptr,
                            )
                        };
                        taffy::geometry::Size {
                            width: measured.width.0 * scale_factor,
                            height: measured.height.0 * scale_factor,
                        }
                    };

                    let resolve_calc_value = |val, basis| tree.resolve_calc_value(val, basis);
                    compute_leaf_layout(inputs, style, resolve_calc_value, measure_fn)
                }
            }
        })
    }
}

impl CacheTree for FiberTree {
    fn cache_get(
        &self,
        node_id: NodeId,
        known_dimensions: taffy::geometry::Size<Option<f32>>,
        available_space: taffy::geometry::Size<TaffyAvailableSpace>,
        run_mode: RunMode,
    ) -> Option<LayoutOutput> {
        self.get(node_id.into())?.taffy_cache.get(
            known_dimensions,
            available_space,
            run_mode,
        )
    }

    fn cache_store(
        &mut self,
        node_id: NodeId,
        known_dimensions: taffy::geometry::Size<Option<f32>>,
        available_space: taffy::geometry::Size<TaffyAvailableSpace>,
        run_mode: RunMode,
        output: LayoutOutput,
    ) {
        if let Some(fiber) = self.get_mut(node_id.into()) {
            fiber
                .taffy_cache
                .store(known_dimensions, available_space, run_mode, output);
        }
    }

    fn cache_clear(&mut self, node_id: NodeId) {
        if let Some(fiber) = self.get_mut(node_id.into()) {
            fiber.taffy_cache.clear();
        }
    }
}

impl LayoutBlockContainer for FiberTree {
    type BlockContainerStyle<'a>
        = &'a TaffyStyle
    where
        Self: 'a;
    type BlockItemStyle<'a>
        = &'a TaffyStyle
    where
        Self: 'a;

    fn get_block_container_style(&self, node_id: NodeId) -> Self::BlockContainerStyle<'_> {
        self.get_core_container_style(node_id)
    }

    fn get_block_child_style(&self, child_node_id: NodeId) -> Self::BlockItemStyle<'_> {
        self.get_core_container_style(child_node_id)
    }
}

impl LayoutFlexboxContainer for FiberTree {
    type FlexboxContainerStyle<'a>
        = &'a TaffyStyle
    where
        Self: 'a;
    type FlexboxItemStyle<'a>
        = &'a TaffyStyle
    where
        Self: 'a;

    fn get_flexbox_container_style(&self, node_id: NodeId) -> Self::FlexboxContainerStyle<'_> {
        self.get_core_container_style(node_id)
    }

    fn get_flexbox_child_style(&self, child_node_id: NodeId) -> Self::FlexboxItemStyle<'_> {
        self.get_core_container_style(child_node_id)
    }
}

impl LayoutGridContainer for FiberTree {
    type GridContainerStyle<'a>
        = &'a TaffyStyle
    where
        Self: 'a;
    type GridItemStyle<'a>
        = &'a TaffyStyle
    where
        Self: 'a;

    fn get_grid_container_style(&self, node_id: NodeId) -> Self::GridContainerStyle<'_> {
        self.get_core_container_style(node_id)
    }

    fn get_grid_child_style(&self, child_node_id: NodeId) -> Self::GridItemStyle<'_> {
        self.get_core_container_style(child_node_id)
    }
}

impl RoundTree for FiberTree {
    fn get_unrounded_layout(&self, node_id: NodeId) -> Layout {
        self.get(node_id.into())
            .unwrap_or_else(|| panic!("missing fiber for node {node_id:?}"))
            .unrounded_layout
    }

    fn set_final_layout(&mut self, node_id: NodeId, layout: &Layout) {
        if let Some(fiber) = self.get_mut(node_id.into()) {
            fiber.final_layout = *layout;
        }
    }
}
