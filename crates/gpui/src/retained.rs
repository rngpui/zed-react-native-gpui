//! Retained element tree for O(changed) frame work.
//!
//! This module provides persistent element storage that survives across frames.
//! Instead of rebuilding the entire element tree every frame via `render()`,
//! unchanged views reuse their stored element trees directly.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  render() - Called only when view is dirty                       │
//! │  Returns: Fresh AnyElement tree (ephemeral, for diffing only)    │
//! └──────────────────────────────────────────────────────────────────┘
//!                               │
//!                               ▼
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  RECONCILER                                                      │
//! │  - Walks ephemeral tree + retained tree in parallel              │
//! │  - Matches elements by (position, type, key)                     │
//! │  - Compares content_hash to detect changes                       │
//! │  - Updates retained tree IN-PLACE, sets dirty flags              │
//! │  - Discards ephemeral tree after reconciliation                  │
//! └──────────────────────────────────────────────────────────────────┘
//!                               │
//!                               ▼
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  RETAINED TREE (persistent in Window, survives frames)           │
//! │  - Elements have stable identity (RetainedElementId)             │
//! │  - Elements STAY IN PLACE, only properties update                │
//! │  - Dirty flags mark what needs reprocessing                      │
//! └──────────────────────────────────────────────────────────────────┘
//!                               │
//!                               ▼
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  LAYOUT / PREPAINT / PAINT                                       │
//! │  - Traverses retained tree, NOT ephemeral elements               │
//! │  - Skips subtrees without dirty flags                            │
//! │  - Returns cached results for clean nodes                        │
//! │  - O(changed) not O(total)                                       │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

use crate::{AnyElement, Bounds, ElementId, EntityId, LayoutId, Pixels};
use bitflags::bitflags;
use collections::FxHashMap;
use slotmap::{new_key_type, SlotMap};
use smallvec::SmallVec;
use std::any::TypeId;
use std::ops::Range;

// ============================================================================
// RETAINED ELEMENT TREE - Core data structures for incremental updates
// ============================================================================

new_key_type! {
    /// Stable ID for a retained element that survives across frames.
    /// Using slotmap ensures O(1) access and stable identity even when
    /// elements are removed from the middle of the tree.
    pub struct RetainedElementId;
}

bitflags! {
    /// Dirty flags for incremental updates.
    /// Each flag indicates a specific type of reprocessing needed.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DirtyFlags: u8 {
        /// Style or structure changed - needs layout recomputation
        const NEEDS_LAYOUT   = 0b0001;
        /// Position changed - needs prepaint (hit testing, etc.)
        const NEEDS_PREPAINT = 0b0010;
        /// Visual properties changed - needs paint
        const NEEDS_PAINT    = 0b0100;
        /// Some descendant has dirty flags - used to skip clean subtrees
        const SUBTREE_DIRTY  = 0b1000;
        /// All flags set - used when a view is first created or fully invalidated
        const ALL = Self::NEEDS_LAYOUT.bits() | Self::NEEDS_PREPAINT.bits()
                  | Self::NEEDS_PAINT.bits() | Self::SUBTREE_DIRTY.bits();
    }
}

/// Index into the prepaint state buffer for cached prepaint data.
pub type PrepaintIndex = usize;

/// Index into the paint buffer for cached paint commands.
pub type PaintIndex = usize;

/// A retained element node - persists across frames.
///
/// This is the core of the retained tree architecture. Each node represents
/// an element that was created during render() and survives across frames.
/// Instead of rebuilding elements every frame, we update these nodes in-place
/// when the ephemeral render() output changes.
pub struct RetainedElement {
    /// Stable identity for this element
    pub id: RetainedElementId,

    /// Type ID of the element (Div, Text, Image, etc.)
    /// Used for matching during reconciliation.
    pub element_type: TypeId,

    /// Optional element key for matching across frames.
    pub element_key: Option<ElementId>,

    /// Hash of the element's style properties ONLY (not children count).
    /// Used to detect changes during reconciliation.
    /// Unlike the old approach, this does NOT include children.len().
    pub content_hash: u64,

    /// Cached layout ID from Taffy. The Taffy node persists across frames,
    /// so we can reuse it without rebuilding.
    pub layout_id: Option<LayoutId>,

    /// Cached bounds from the last layout computation.
    pub bounds: Option<Bounds<Pixels>>,

    /// Parent element in the retained tree.
    pub parent: Option<RetainedElementId>,

    /// Children stored BY ID, not by value.
    /// This is the key difference from the old approach - children are
    /// references, not owned values, so we don't break tree structure.
    pub children: SmallVec<[RetainedElementId; 4]>,

    /// Dirty flags indicating what needs reprocessing.
    pub dirty: DirtyFlags,

    /// Cached prepaint range from the last prepaint.
    pub prepaint_range: Option<Range<PrepaintIndex>>,

    /// Cached paint range from the last paint.
    pub paint_range: Option<Range<PaintIndex>>,
}

impl RetainedElement {
    /// Create a new retained element with all dirty flags set.
    pub fn new(
        id: RetainedElementId,
        element_type: TypeId,
        element_key: Option<ElementId>,
        content_hash: u64,
    ) -> Self {
        Self {
            id,
            element_type,
            element_key,
            content_hash,
            layout_id: None,
            bounds: None,
            parent: None,
            children: SmallVec::new(),
            dirty: DirtyFlags::ALL,
            prepaint_range: None,
            paint_range: None,
        }
    }

    /// Check if this element needs layout recomputation.
    pub fn needs_layout(&self) -> bool {
        self.dirty.contains(DirtyFlags::NEEDS_LAYOUT)
    }

    /// Check if this element needs prepaint.
    pub fn needs_prepaint(&self) -> bool {
        self.dirty.contains(DirtyFlags::NEEDS_PREPAINT)
    }

    /// Check if this element needs paint.
    pub fn needs_paint(&self) -> bool {
        self.dirty.contains(DirtyFlags::NEEDS_PAINT)
    }

    /// Check if any descendant is dirty (for subtree skipping).
    pub fn has_dirty_subtree(&self) -> bool {
        self.dirty.contains(DirtyFlags::SUBTREE_DIRTY)
    }

    /// Clear layout dirty flag after layout is complete.
    pub fn clear_layout_dirty(&mut self) {
        self.dirty.remove(DirtyFlags::NEEDS_LAYOUT);
    }

    /// Clear prepaint dirty flag after prepaint is complete.
    pub fn clear_prepaint_dirty(&mut self) {
        self.dirty.remove(DirtyFlags::NEEDS_PREPAINT);
    }

    /// Clear paint dirty flag after paint is complete.
    pub fn clear_paint_dirty(&mut self) {
        self.dirty.remove(DirtyFlags::NEEDS_PAINT);
    }

    /// Clear the subtree dirty flag after processing children.
    pub fn clear_subtree_dirty(&mut self) {
        self.dirty.remove(DirtyFlags::SUBTREE_DIRTY);
    }

    /// Mark this element as needing full reprocessing.
    pub fn mark_fully_dirty(&mut self) {
        self.dirty = DirtyFlags::ALL;
    }
}

impl std::fmt::Debug for RetainedElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedElement")
            .field("id", &self.id)
            .field("element_type", &self.element_type)
            .field("element_key", &self.element_key)
            .field("content_hash", &self.content_hash)
            .field("layout_id", &self.layout_id)
            .field("bounds", &self.bounds)
            .field("children", &self.children.len())
            .field("dirty", &self.dirty)
            .finish()
    }
}

/// The retained tree - stores all retained elements for a window.
///
/// This is the persistent data structure that survives across frames.
/// Elements are stored in a SlotMap for O(1) access with stable IDs.
pub struct RetainedTree {
    /// All retained elements, indexed by stable ID.
    elements: SlotMap<RetainedElementId, RetainedElement>,

    /// Root element for each view, indexed by EntityId.
    view_roots: FxHashMap<EntityId, RetainedElementId>,
}

impl RetainedTree {
    /// Create a new empty retained tree.
    pub fn new() -> Self {
        Self {
            elements: SlotMap::with_key(),
            view_roots: FxHashMap::default(),
        }
    }

    /// Create a new retained element and return its ID.
    pub fn create_element(
        &mut self,
        element_type: TypeId,
        element_key: Option<ElementId>,
        content_hash: u64,
    ) -> RetainedElementId {
        let id = self.elements.insert_with_key(|id| {
            RetainedElement::new(id, element_type, element_key, content_hash)
        });
        id
    }

    /// Get a reference to an element by ID.
    pub fn get(&self, id: RetainedElementId) -> Option<&RetainedElement> {
        self.elements.get(id)
    }

    /// Get a mutable reference to an element by ID.
    pub fn get_mut(&mut self, id: RetainedElementId) -> Option<&mut RetainedElement> {
        self.elements.get_mut(id)
    }

    /// Ensure a retained root exists for a view and reconcile its identity/hash.
    pub fn reconcile_root(
        &mut self,
        entity_id: EntityId,
        element_type: TypeId,
        element_key: Option<ElementId>,
        content_hash: u64,
    ) -> RetainedElementId {
        if let Some(root_id) = self.view_roots.get(&entity_id).copied() {
            let matches = self
                .elements
                .get(root_id)
                .is_some_and(|element| element.element_type == element_type
                    && element.element_key == element_key);
            if matches {
                self.update_content_hash(root_id, content_hash);
                return root_id;
            }

            self.remove_subtree(root_id);
        }

        let root_id = self.create_element(element_type, element_key, content_hash);
        self.view_roots.insert(entity_id, root_id);
        root_id
    }

    /// Reconcile a child element against the retained tree, matching by key/type or index.
    pub fn reconcile_child(
        &mut self,
        parent_id: RetainedElementId,
        child_index: usize,
        element_type: TypeId,
        element_key: Option<ElementId>,
        content_hash: u64,
    ) -> RetainedElementId {
        let mut matched_id = None;
        let mut children_changed = false;
        let mut removed_child = None;
        let mut matched_pos = None;

        // First pass: find a match by copying children list to avoid borrow conflicts
        let children: SmallVec<[RetainedElementId; 8]> = self
            .elements
            .get(parent_id)
            .map(|p| p.children.iter().copied().collect())
            .unwrap_or_default();

        if let Some(ref key) = element_key {
            // Key-based matching: find child with same key and type
            for (pos, &child_id) in children.iter().enumerate() {
                let matches = self
                    .elements
                    .get(child_id)
                    .is_some_and(|child| {
                        child.element_key.as_ref() == Some(key) && child.element_type == element_type
                    });
                if matches {
                    matched_id = Some(child_id);
                    matched_pos = Some(pos);
                    if pos != child_index {
                        children_changed = true;
                    }
                    break;
                }
            }
        } else if child_index < children.len() {
            // Index-based matching: check if child at index has same type and no key
            let child_id = children[child_index];
            let matches = self
                .elements
                .get(child_id)
                .is_some_and(|child| child.element_key.is_none() && child.element_type == element_type);
            if matches {
                matched_id = Some(child_id);
            } else {
                removed_child = Some(child_id);
                children_changed = true;
            }
        }

        // Second pass: update parent's children list
        if let Some(pos) = matched_pos {
            // Key match found - reorder if needed
            if let Some(parent) = self.elements.get_mut(parent_id) {
                let child_id = parent.children.remove(pos);
                if child_index <= parent.children.len() {
                    parent.children.insert(child_index, child_id);
                } else {
                    parent.children.push(child_id);
                }
            }
        } else if removed_child.is_some() {
            // Index match failed - remove the mismatched child
            if let Some(parent) = self.elements.get_mut(parent_id) {
                if child_index < parent.children.len() {
                    parent.children.remove(child_index);
                }
            }
        }

        let child_id = if let Some(existing_id) = matched_id {
            existing_id
        } else {
            children_changed = true;
            let new_id = self.create_element(element_type, element_key.clone(), content_hash);
            if let Some(parent) = self.elements.get_mut(parent_id) {
                if child_index <= parent.children.len() {
                    parent.children.insert(child_index, new_id);
                } else {
                    parent.children.push(new_id);
                }
            }
            new_id
        };

        if let Some(child) = self.elements.get_mut(child_id) {
            child.parent = Some(parent_id);
            child.element_type = element_type;
            child.element_key = element_key;
        }
        self.update_content_hash(child_id, content_hash);

        if let Some(removed_child) = removed_child {
            self.remove_subtree(removed_child);
        }

        if children_changed {
            self.mark_dirty(parent_id, DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PREPAINT | DirtyFlags::NEEDS_PAINT);
        }

        child_id
    }

    /// Remove any trailing children that were not reconciled this frame.
    pub fn truncate_children(&mut self, parent_id: RetainedElementId, new_len: usize) {
        if let Some(parent) = self.elements.get_mut(parent_id) {
            if parent.children.len() > new_len {
                let removed: SmallVec<[RetainedElementId; 4]> =
                    parent.children.drain(new_len..).collect();
                drop(parent);
                for child_id in removed {
                    self.remove_subtree(child_id);
                }
                self.mark_dirty(parent_id, DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PREPAINT | DirtyFlags::NEEDS_PAINT);
            }
        }
    }

    fn update_content_hash(&mut self, id: RetainedElementId, new_hash: u64) {
        if let Some(element) = self.elements.get_mut(id) {
            if new_hash == 0 || element.content_hash != new_hash {
                element.content_hash = new_hash;
                element.dirty.insert(
                    DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PREPAINT | DirtyFlags::NEEDS_PAINT,
                );
                if let Some(parent_id) = element.parent {
                    self.propagate_dirty_to_ancestors(parent_id);
                }
            }
        }
    }

    /// Remove an element and all its descendants from the tree.
    pub fn remove_subtree(&mut self, id: RetainedElementId) {
        if let Some(element) = self.elements.remove(id) {
            // Recursively remove all children
            for child_id in element.children {
                self.remove_subtree(child_id);
            }
        }
    }

    /// Set the root element for a view.
    pub fn set_view_root(&mut self, entity_id: EntityId, root_id: RetainedElementId) {
        self.view_roots.insert(entity_id, root_id);
    }

    /// Get the root element for a view.
    pub fn get_view_root(&self, entity_id: EntityId) -> Option<RetainedElementId> {
        self.view_roots.get(&entity_id).copied()
    }

    /// Remove the root element for a view and its subtree.
    pub fn remove_view(&mut self, entity_id: EntityId) {
        if let Some(root_id) = self.view_roots.remove(&entity_id) {
            self.remove_subtree(root_id);
        }
    }

    /// Add a child to an element.
    pub fn add_child(&mut self, parent_id: RetainedElementId, child_id: RetainedElementId) {
        if let Some(child) = self.elements.get_mut(child_id) {
            child.parent = Some(parent_id);
        }
        if let Some(parent) = self.elements.get_mut(parent_id) {
            parent.children.push(child_id);
        }
    }

    /// Mark an element and its ancestors as having a dirty subtree.
    pub fn propagate_dirty_to_ancestors(&mut self, mut id: RetainedElementId) {
        while let Some(element) = self.elements.get_mut(id) {
            if element.dirty.contains(DirtyFlags::SUBTREE_DIRTY) {
                // Already marked, ancestors must already be marked too
                break;
            }
            element.dirty.insert(DirtyFlags::SUBTREE_DIRTY);
            if let Some(parent_id) = element.parent {
                id = parent_id;
            } else {
                break;
            }
        }
    }

    /// Mark an element as fully dirty and propagate to ancestors.
    pub fn mark_dirty(&mut self, id: RetainedElementId, flags: DirtyFlags) {
        if let Some(element) = self.elements.get_mut(id) {
            element.dirty.insert(flags);
            if let Some(parent_id) = element.parent {
                self.propagate_dirty_to_ancestors(parent_id);
            }
        }
    }

    /// Clear subtree-dirty if none of the children are dirty anymore.
    pub fn clear_subtree_dirty_if_clean(&mut self, id: RetainedElementId) {
        let has_dirty_child = self
            .elements
            .get(id)
            .map(|element| {
                element.children.iter().any(|child_id| {
                    self.elements
                        .get(*child_id)
                        .is_some_and(|child| !child.dirty.is_empty())
                })
            })
            .unwrap_or(false);

        if !has_dirty_child {
            if let Some(element) = self.elements.get_mut(id) {
                element.dirty.remove(DirtyFlags::SUBTREE_DIRTY);
            }
        }
    }

    /// Get the number of elements in the tree.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Reset all elements' dirty flags for a new frame.
    /// Called at the beginning of each frame before rendering dirty views.
    pub fn begin_frame(&mut self) {
        // Note: We don't clear dirty flags here - they're cleared after processing.
        // This method is a placeholder for any per-frame setup needed.
    }

    /// Iterate over all elements (for debugging/stats).
    pub fn iter(&self) -> impl Iterator<Item = (RetainedElementId, &RetainedElement)> {
        self.elements.iter()
    }
}

impl Default for RetainedTree {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RetainedTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedTree")
            .field("elements", &self.elements.len())
            .field("view_roots", &self.view_roots.len())
            .finish()
    }
}

// ============================================================================
// RETAINED VIEW - Root element storage per view
// ============================================================================

/// Retained state for a view's root element tree.
pub struct RetainedView {
    /// The retained root element tree for this view.
    pub root: Option<AnyElement>,
    /// The retained tree root ID for this view.
    pub root_id: Option<RetainedElementId>,
    /// Type ID of the view, for debugging.
    pub type_id: TypeId,
}

impl RetainedView {
    /// Create a new retained view for an entity of the given type.
    pub fn new<V: 'static>() -> Self {
        Self {
            root: None,
            root_id: None,
            type_id: TypeId::of::<V>(),
        }
    }
}

impl std::fmt::Debug for RetainedView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedView")
            .field("has_root", &self.root.is_some())
            .field("root_id", &self.root_id)
            .finish()
    }
}
