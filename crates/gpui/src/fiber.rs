use crate::{
    AnyDescriptor, Bounds, EntityId, Hitbox, LayoutId, Pixels, SharedString, StyleRefinement,
    TaffyLayoutEngine,
};
use collections::FxHashMap;
use slotmap::{new_key_type, SlotMap};

new_key_type! {
    /// A unique identifier for a fiber in the fiber tree
    pub struct FiberId;

    /// A unique identifier for a cached paint list
    pub struct PaintListId;

    /// A unique identifier for cached prepaint state
    pub struct PrepaintStateId;
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

    /// The taffy layout node for this fiber
    pub layout_id: Option<LayoutId>,

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
            layout_id: None,
            bounds: None,
            prepaint_state: None,
            paint_list: None,
            parent: None,
            first_child: None,
            next_sibling: None,
            dirty: DirtyFlags::ALL,
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
    }

    /// Mark a fiber as dirty and propagate SUBTREE_DIRTY to ancestors
    pub fn mark_dirty(&mut self, fiber_id: FiberId, flags: DirtyFlags) {
        if let Some(fiber) = self.fibers.get_mut(fiber_id) {
            fiber.dirty |= flags;
        }

        self.propagate_has_dirty_descendant(fiber_id);
        if flags.intersects(DirtyFlags::NEEDS_LAYOUT) {
            self.propagate_needs_layout(fiber_id);
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

    /// Reconcile a descriptor against a fiber, returning whether it changed.
    /// If the descriptor's content hash differs from the fiber's stored hash,
    /// the fiber is marked as needing layout and its hash is updated.
    pub fn reconcile(
        &mut self,
        fiber_id: FiberId,
        descriptor: &AnyDescriptor,
        _taffy: &mut TaffyLayoutEngine,
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
        self.reconcile_children(fiber_id, children, _taffy);

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
        taffy: &mut TaffyLayoutEngine,
    ) {
        // Collect existing children
        let mut existing_children: Vec<FiberId> = self.children(parent_id).collect();

        // Reconcile or create children
        for (i, child_desc) in children.iter().enumerate() {
            if i < existing_children.len() {
                // Existing fiber at this position - reconcile it
                let child_fiber_id = existing_children[i];
                self.reconcile(child_fiber_id, child_desc, taffy);
            } else {
                // No fiber at this position - create new one
                let new_fiber_id = self.create_fiber_for(child_desc);
                self.add_child(parent_id, new_fiber_id);
                if let Some(fiber) = self.fibers.get_mut(new_fiber_id) {
                    Self::update_cached_props(fiber, child_desc);
                }

                // Recursively create fibers for descendants
                self.reconcile_children(new_fiber_id, child_desc.children(), taffy);

                // Mark parent as having subtree dirty
                self.propagate_has_dirty_descendant(new_fiber_id);
            }
        }

        // Remove excess fibers (children that no longer exist)
        for excess_fiber_id in existing_children.into_iter().skip(children.len()) {
            // Unlink from parent
            self.unlink_child(parent_id, excess_fiber_id);
            // Remove the fiber and its descendants (this also removes taffy nodes)
            self.remove_with_taffy(excess_fiber_id, taffy);
        }
    }

    /// Unlink a child from its parent's linked list
    fn unlink_child(&mut self, parent_id: FiberId, child_id: FiberId) {
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

    /// Remove a fiber and all its descendants, also removing their taffy nodes
    pub fn remove_with_taffy(&mut self, id: FiberId, taffy: &mut TaffyLayoutEngine) {
        let mut to_remove = vec![id];

        while let Some(fiber_id) = to_remove.pop() {
            if let Some(fiber) = self.fibers.remove(fiber_id) {
                // Remove the taffy node if it exists
                if let Some(layout_id) = fiber.layout_id {
                    taffy.remove(layout_id);
                }

                // Queue children for removal
                let mut child = fiber.first_child;
                while let Some(child_id) = child {
                    to_remove.push(child_id);
                    child = self.fibers.get(child_id).and_then(|f| f.next_sibling);
                }
            }
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
