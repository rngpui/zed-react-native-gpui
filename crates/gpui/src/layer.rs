//! Layer Tree for Compositing
//!
//! This module provides a formal Layer Tree system that generalizes scroll container
//! handling. Any element can become a compositing layer (scroll, transform, opacity).
//!
//! ## Architecture
//!
//! ```text
//! LayerTree
//!   └── Root Layer (always exists, renders to Scene)
//!         ├── Layer (scroll container) → DisplayList → Tiles
//!         │     └── Layer (nested transform) → DisplayList
//!         └── Layer (opacity animation) → DisplayList
//! ```
//!
//! ## Key Benefits
//!
//! - Unified handling of scroll containers, transforms, and opacity
//! - Layer property changes (scroll offset, transform, opacity) = GPU-only composite
//! - No repaint needed for property-only changes

use crate::display_list::DisplayList;
use crate::property_trees::PropertyTrees;
use crate::scene::TransformationMatrix;
use crate::{Bounds, GlobalElementId, Pixels, Point, Size};
use collections::FxHashMap;

/// Unique identifier for a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerId(pub(crate) u32);

impl LayerId {
    /// The root layer ID (always 0).
    pub const ROOT: LayerId = LayerId(0);
}

/// Why this element became a compositing layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerReason {
    /// Root layer (always exists).
    Root,
    /// Scroll container with large content.
    ScrollContainer,
    /// Element has transform that may animate.
    Transform,
    /// Element has opacity that may animate.
    Opacity,
    /// Explicit will-change hint.
    WillChange,
}

/// A compositing layer with its own DisplayList.
///
/// Layers enable efficient property updates without full repaint:
/// - Scroll offset changes: just re-composite tiles at new positions
/// - Transform changes: apply new transform during composite
/// - Opacity changes: apply new opacity during composite
pub struct Layer {
    /// Unique identifier for this layer.
    pub id: LayerId,

    /// Element that owns this layer (None for root).
    pub element_id: Option<GlobalElementId>,

    /// Why this element became a layer.
    pub reason: LayerReason,

    /// Parent layer (None for root).
    pub parent: Option<LayerId>,

    /// Child layers (sorted by paint order).
    pub children: Vec<LayerId>,

    /// Layer-local transform (relative to parent).
    pub local_transform: TransformationMatrix,

    /// Layer opacity (0.0 - 1.0).
    pub opacity: f32,

    /// Scroll offset (for scroll containers).
    pub scroll_offset: Point<Pixels>,

    /// Content origin in window space (bounds.origin + scroll_offset).
    /// This is where content (0,0) appears on screen. Used to convert
    /// primitive coordinates from window space to content space.
    pub content_origin: Point<Pixels>,

    /// Clip bounds (in layer-local coordinates).
    pub clip: Option<Bounds<Pixels>>,

    /// Content rendered to this layer (None for root layer).
    pub(crate) display_list: Option<DisplayList>,

    /// Property trees for transforms and clips.
    /// Used by DisplayItems to reference transforms/clips by node ID.
    pub property_trees: PropertyTrees,

    /// Content bounds (union of all DisplayItems).
    pub content_bounds: Bounds<Pixels>,

    /// Content size for scroll containers.
    pub content_size: Size<Pixels>,

    /// Viewport size for scroll containers.
    pub viewport_size: Size<Pixels>,

    /// Content changed, rebuild DisplayList.
    pub needs_repaint: bool,

    /// DisplayList changed, re-rasterize tiles.
    pub needs_rasterize: bool,

    /// Transform/opacity/scroll changed, re-composite.
    pub needs_composite: bool,
}

impl Layer {
    /// Create a new layer.
    fn new(id: LayerId, element_id: Option<GlobalElementId>, reason: LayerReason) -> Self {
        // Non-root layers get a display list; root layer doesn't need one
        let display_list = element_id.clone().map(DisplayList::new);
        Self {
            id,
            element_id,
            reason,
            parent: None,
            children: Vec::new(),
            local_transform: TransformationMatrix::unit(),
            opacity: 1.0,
            scroll_offset: Point::default(),
            content_origin: Point::default(),
            clip: None,
            display_list,
            property_trees: PropertyTrees::new(),
            content_bounds: Bounds::default(),
            content_size: Size::default(),
            viewport_size: Size::default(),
            needs_repaint: true,
            needs_rasterize: false,
            needs_composite: false,
        }
    }

    /// Create the root layer.
    fn root() -> Self {
        Self::new(LayerId::ROOT, None, LayerReason::Root)
    }
}

/// Tree of all compositing layers for a window.
///
/// The LayerTree manages the hierarchy of layers and provides:
/// - Push/pop operations during paint traversal
/// - O(1) property updates (scroll, transform, opacity)
/// - Iteration in composite order (back to front)
/// - Layer persistence across frames (display lists are reused)
pub struct LayerTree {
    /// All layers indexed by ID.
    layers: FxHashMap<LayerId, Layer>,

    /// Lookup from element_id to layer_id for O(1) layer reuse.
    element_to_layer: FxHashMap<GlobalElementId, LayerId>,

    /// Root layer ID (always LayerId::ROOT).
    root: LayerId,

    /// Next layer ID to allocate.
    next_id: u32,

    /// Stack of currently active layers during paint.
    /// The bottom of the stack is always the root layer.
    layer_stack: Vec<LayerId>,
}

impl Default for LayerTree {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerTree {
    /// Create a new layer tree with only the root layer.
    pub fn new() -> Self {
        let mut layers = FxHashMap::default();
        layers.insert(LayerId::ROOT, Layer::root());

        Self {
            layers,
            element_to_layer: FxHashMap::default(),
            root: LayerId::ROOT,
            next_id: 1, // 0 is reserved for root
            layer_stack: vec![LayerId::ROOT],
        }
    }

    /// Reset the layer tree for a new frame.
    ///
    /// This preserves layers and their display lists across frames, but resets
    /// the tree structure (parent/children relationships) which gets rebuilt
    /// during the paint traversal. Property trees are also cleared since transforms
    /// and clips are rebuilt during paint.
    /// Call this at the start of each frame before painting.
    pub fn begin_frame(&mut self) {
        // Reset tree structure for all layers (will be rebuilt during paint)
        // but KEEP the layers and their display lists for reuse
        for layer in self.layers.values_mut() {
            layer.children.clear();
            layer.parent = if layer.id == LayerId::ROOT {
                None
            } else {
                // Will be set correctly during push_layer
                None
            };
            // Clear property trees - transforms/clips are rebuilt during paint
            layer.property_trees.clear();
        }

        // Reset layer stack to just root
        self.layer_stack.clear();
        self.layer_stack.push(LayerId::ROOT);

        // Note: We don't reset next_id because layer IDs are persistent
        // Note: We don't clear element_to_layer because layers persist
    }

    /// Get the currently active layer (top of stack).
    pub fn current_layer(&self) -> &Layer {
        let id = *self.layer_stack.last().expect("Layer stack should never be empty");
        self.layers.get(&id).expect("Current layer should exist")
    }

    /// Get mutable reference to the currently active layer.
    pub fn current_layer_mut(&mut self) -> &mut Layer {
        let id = *self.layer_stack.last().expect("Layer stack should never be empty");
        self.layers.get_mut(&id).expect("Current layer should exist")
    }

    /// Get a layer by ID.
    pub fn get(&self, id: LayerId) -> Option<&Layer> {
        self.layers.get(&id)
    }

    /// Get mutable reference to a layer by ID.
    pub fn get_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers.get_mut(&id)
    }

    /// Get the root layer.
    pub fn root_layer(&self) -> &Layer {
        self.layers.get(&LayerId::ROOT).expect("Root layer should exist")
    }

    /// Push a layer onto the stack.
    ///
    /// If a layer for this element_id already exists, reuses it (preserving its
    /// display list). Otherwise creates a new layer. The layer becomes a child
    /// of the current layer and is made active.
    /// Returns the layer's ID and whether this is a new layer (needs_repaint).
    pub fn push_layer(
        &mut self,
        element_id: GlobalElementId,
        reason: LayerReason,
    ) -> LayerId {
        let parent_id = *self.layer_stack.last().expect("Layer stack should never be empty");

        // Check if we already have a layer for this element
        let (id, is_new) = if let Some(&existing_id) = self.element_to_layer.get(&element_id) {
            // Reuse existing layer - its display list is preserved!
            (existing_id, false)
        } else {
            // Create new layer
            let id = LayerId(self.next_id);
            self.next_id += 1;

            let layer = Layer::new(id, Some(element_id.clone()), reason);
            self.layers.insert(id, layer);
            self.element_to_layer.insert(element_id, id);

            (id, true)
        };

        // Update parent relationship
        if let Some(layer) = self.layers.get_mut(&id) {
            layer.parent = Some(parent_id);
            // Only mark needs_repaint for NEW layers
            // Existing layers keep their display list and needs_repaint = false
            if is_new {
                layer.needs_repaint = true;
            }
        }

        // Add as child of parent
        if let Some(parent) = self.layers.get_mut(&parent_id) {
            parent.children.push(id);
        }

        // Push onto stack
        self.layer_stack.push(id);

        id
    }

    /// Pop the current layer from the stack.
    ///
    /// Returns to the parent layer. Panics if trying to pop the root layer.
    pub fn pop_layer(&mut self) {
        if self.layer_stack.len() <= 1 {
            panic!("Cannot pop the root layer");
        }
        self.layer_stack.pop();
    }

    /// Re-activate a layer by pushing it onto the stack.
    ///
    /// Used to restore a layer that was temporarily popped (e.g., during
    /// reuse of existing display list when primitives shouldn't be added).
    pub fn reactivate_layer(&mut self, layer_id: LayerId) {
        self.layer_stack.push(layer_id);
    }

    /// Check if we're inside any non-root layer.
    ///
    /// This replaces `is_inside_tiled_scroll_container()`.
    pub fn is_inside_layer(&self) -> bool {
        self.layer_stack.len() > 1
    }

    /// Get the current layer stack depth (1 = root only).
    pub fn layer_depth(&self) -> usize {
        self.layer_stack.len()
    }

    /// Update scroll offset without triggering repaint.
    ///
    /// Only marks the layer for re-composite.
    pub fn set_scroll_offset(&mut self, layer_id: LayerId, offset: Point<Pixels>) {
        if let Some(layer) = self.layers.get_mut(&layer_id) {
            layer.scroll_offset = offset;
            layer.needs_composite = true;
        }
    }

    /// Update transform without triggering repaint.
    ///
    /// Only marks the layer for re-composite.
    pub fn set_transform(&mut self, layer_id: LayerId, transform: TransformationMatrix) {
        if let Some(layer) = self.layers.get_mut(&layer_id) {
            layer.local_transform = transform;
            layer.needs_composite = true;
        }
    }

    /// Update opacity without triggering repaint.
    ///
    /// Only marks the layer for re-composite.
    pub fn set_opacity(&mut self, layer_id: LayerId, opacity: f32) {
        if let Some(layer) = self.layers.get_mut(&layer_id) {
            layer.opacity = opacity;
            layer.needs_composite = true;
        }
    }

    /// Iterate all layers in composite order (back to front).
    ///
    /// This performs a depth-first traversal of the layer tree.
    pub fn layers_for_composite(&self) -> impl Iterator<Item = &Layer> {
        LayerIterator {
            tree: self,
            stack: vec![self.root],
        }
    }

    /// Get all non-root layers.
    pub fn non_root_layers(&self) -> impl Iterator<Item = &Layer> {
        self.layers.values().filter(|l| l.id != LayerId::ROOT)
    }

    /// Get the number of layers (including root).
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the tree is empty (only root).
    pub fn is_empty(&self) -> bool {
        self.layers.len() <= 1
    }

    /// Find a layer by its element ID.
    pub fn find_by_element_id(&self, element_id: &GlobalElementId) -> Option<LayerId> {
        for (id, layer) in &self.layers {
            if layer.element_id.as_ref() == Some(element_id) {
                return Some(*id);
            }
        }
        None
    }
}

/// Iterator for traversing layers in composite order.
struct LayerIterator<'a> {
    tree: &'a LayerTree,
    stack: Vec<LayerId>,
}

impl<'a> Iterator for LayerIterator<'a> {
    type Item = &'a Layer;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(id) = self.stack.pop() {
            if let Some(layer) = self.tree.layers.get(&id) {
                // Push children in reverse order so they're processed front-to-back
                for &child_id in layer.children.iter().rev() {
                    self.stack.push(child_id);
                }
                return Some(layer);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_tree_creation() {
        let tree = LayerTree::new();
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.current_layer().id, LayerId::ROOT);
        assert!(!tree.is_inside_layer());
    }

    #[test]
    fn test_push_pop_layer() {
        let mut tree = LayerTree::new();

        let element_id = GlobalElementId::from(vec![1u32]);
        let layer_id = tree.push_layer(element_id, LayerReason::ScrollContainer);

        assert!(tree.is_inside_layer());
        assert_eq!(tree.current_layer().id, layer_id);
        assert_eq!(tree.layer_depth(), 2);

        tree.pop_layer();

        assert!(!tree.is_inside_layer());
        assert_eq!(tree.current_layer().id, LayerId::ROOT);
    }

    #[test]
    fn test_layers_for_composite() {
        let mut tree = LayerTree::new();

        let id1 = GlobalElementId::from(vec![1u32]);
        let id2 = GlobalElementId::from(vec![2u32]);

        tree.push_layer(id1, LayerReason::ScrollContainer);
        tree.pop_layer();

        tree.push_layer(id2, LayerReason::Transform);
        tree.pop_layer();

        let layer_ids: Vec<_> = tree.layers_for_composite().map(|l| l.id).collect();
        assert_eq!(layer_ids.len(), 3); // Root + 2 children
    }

    #[test]
    fn test_begin_frame_resets() {
        let mut tree = LayerTree::new();

        let id1 = GlobalElementId::from(vec![1u32]);
        tree.push_layer(id1, LayerReason::ScrollContainer);

        assert_eq!(tree.len(), 2);

        tree.begin_frame();

        // Layers are preserved for reuse across frames (display lists cached)
        assert_eq!(tree.len(), 2);
        // But layer stack is reset to root only
        assert!(!tree.is_inside_layer());
        // And tree structure (children) is cleared
        assert!(tree.root_layer().children.is_empty());
    }
}
