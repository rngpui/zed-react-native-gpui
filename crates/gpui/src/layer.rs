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
use crate::scene::{TileCoord, TileKey, TileSprite, TransformationMatrix};
use crate::window::{AnyMouseListener, Hitbox};
use crate::{
    point, size, Bounds, ContentMask, GlobalElementId, HitboxBehavior, HitboxId, Pixels, Point,
    ScaledPixels, Size,
};
use collections::FxHashMap;
use std::sync::Arc;

/// Tile size in scaled pixels (matches TileCache).
const TILE_SIZE: u32 = 512;

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

/// A hitbox stored in layer-local coordinates for retention across frames.
///
/// When a layer's content is unchanged (no repaint needed), we skip painting
/// children entirely and instead replay these retained hitboxes. The coordinates
/// are stored relative to the layer's content (0,0), then transformed to window
/// coordinates using the current scroll offset during replay.
#[derive(Clone, Debug)]
pub struct RetainedHitbox {
    /// The original hitbox ID (preserved so existing mouse listeners work).
    pub id: HitboxId,
    /// Bounds in layer-local (content) coordinates.
    /// To get window coordinates: origin + layer.content_origin
    pub bounds: Bounds<Pixels>,
    /// The content mask when the hitbox was created (in layer-local coordinates).
    pub content_mask: ContentMask<Pixels>,
    /// Hitbox behavior flags.
    pub behavior: HitboxBehavior,
}

impl RetainedHitbox {
    /// Create a retained hitbox from a window-space hitbox and layer content origin.
    pub fn from_hitbox(hitbox: &Hitbox, content_origin: Point<Pixels>) -> Self {
        Self {
            id: hitbox.id,
            bounds: Bounds {
                origin: Point {
                    x: hitbox.bounds.origin.x - content_origin.x,
                    y: hitbox.bounds.origin.y - content_origin.y,
                },
                size: hitbox.bounds.size,
            },
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: Point {
                        x: hitbox.content_mask.bounds.origin.x - content_origin.x,
                        y: hitbox.content_mask.bounds.origin.y - content_origin.y,
                    },
                    size: hitbox.content_mask.bounds.size,
                },
            },
            behavior: hitbox.behavior,
        }
    }

    /// Convert back to a window-space Hitbox using the current content origin.
    pub fn to_hitbox(&self, content_origin: Point<Pixels>) -> Hitbox {
        Hitbox {
            id: self.id,
            bounds: Bounds {
                origin: Point {
                    x: self.bounds.origin.x + content_origin.x,
                    y: self.bounds.origin.y + content_origin.y,
                },
                size: self.bounds.size,
            },
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: Point {
                        x: self.content_mask.bounds.origin.x + content_origin.x,
                        y: self.content_mask.bounds.origin.y + content_origin.y,
                    },
                    size: self.content_mask.bounds.size,
                },
            },
            behavior: self.behavior,
        }
    }
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
    /// Wrapped in Arc for cheap cloning to Scene. Use Arc::make_mut for mutation.
    pub(crate) display_list: Option<Arc<DisplayList>>,

    /// Previous frame's display list (for per-element cache lookup).
    /// At frame start, current display_list is swapped here, then cleared.
    /// Elements can then look up their previous items and copy if unchanged.
    pub(crate) previous_display_list: Option<Arc<DisplayList>>,

    /// Property trees for transforms and clips.
    /// Used by DisplayItems to reference transforms/clips by node ID.
    pub property_trees: PropertyTrees,

    /// Content bounds (union of all DisplayItems).
    pub content_bounds: Bounds<Pixels>,

    /// Content size for scroll containers.
    pub content_size: Size<Pixels>,

    /// Viewport size for scroll containers.
    pub viewport_size: Size<Pixels>,

    /// P2: Viewport origin in window space (bounds.origin from paint).
    /// Used for compositor-only tile sprite emission.
    pub viewport_origin: Point<Pixels>,

    /// P2: Ancestor clip mask (outer_clip_for_tiles from paint).
    /// Used for compositor-only tile sprite emission.
    pub outer_clip: ContentMask<Pixels>,

    /// P2: Scale factor for converting to ScaledPixels.
    /// Used for compositor-only tile sprite emission.
    pub scale_factor: f32,

    /// Content changed, rebuild DisplayList.
    pub needs_repaint: bool,

    /// DisplayList changed, re-rasterize tiles.
    pub needs_rasterize: bool,

    /// Transform/opacity/scroll changed, re-composite.
    pub needs_composite: bool,

    /// Retained hitboxes from previous paint (in layer-local coordinates).
    /// These are captured during initial paint and replayed during reuse
    /// without needing to run paint() on children again.
    pub retained_hitboxes: Vec<RetainedHitbox>,

    /// Retained mouse listeners from previous paint.
    /// These are moved from Frame during initial paint and moved back
    /// during reuse. Unlike hitboxes (which are cloneable data), listeners
    /// are moved because they're Box<dyn FnMut>.
    pub retained_mouse_listeners: Vec<Option<AnyMouseListener>>,

    /// Index range of hitboxes in the frame that belong to this layer.
    /// Used during capture to identify which hitboxes to retain.
    pub(crate) hitbox_range: Option<std::ops::Range<usize>>,

    /// Index range of mouse listeners in the frame that belong to this layer.
    /// Used during capture to identify which listeners to retain.
    pub(crate) listener_range: Option<std::ops::Range<usize>>,
}

impl Layer {
    /// Create a new layer.
    fn new(id: LayerId, element_id: Option<GlobalElementId>, reason: LayerReason) -> Self {
        // All layers with an element_id get a display list (wrapped in Arc)
        let display_list = element_id.clone().map(|id| Arc::new(DisplayList::new(id)));
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
            previous_display_list: None,
            property_trees: PropertyTrees::new(),
            content_bounds: Bounds::default(),
            content_size: Size::default(),
            viewport_size: Size::default(),
            viewport_origin: Point::default(),
            outer_clip: ContentMask::default(),
            scale_factor: 1.0,
            needs_repaint: true,
            needs_rasterize: false,
            needs_composite: false,
            retained_hitboxes: Vec::new(),
            retained_mouse_listeners: Vec::new(),
            hitbox_range: None,
            listener_range: None,
        }
    }

    /// Create the root layer.
    /// The root layer now gets a DisplayList so all primitives flow through DisplayList.
    fn root() -> Self {
        Self::new(LayerId::ROOT, Some(GlobalElementId::root()), LayerReason::Root)
    }

    /// Clear retained hitboxes and listeners (used when content changes).
    pub fn clear_retained(&mut self) {
        self.retained_hitboxes.clear();
        self.retained_mouse_listeners.clear();
        self.hitbox_range = None;
        self.listener_range = None;
    }

    /// Check if this layer has retained hitboxes available for reuse.
    pub fn has_retained_hitboxes(&self) -> bool {
        !self.retained_hitboxes.is_empty()
    }

    /// Prepare layer for a new frame (Phase 20e: Retained DisplayList).
    ///
    /// NOTE: This does NOT clear the display list. That only happens when
    /// `prepare_for_repaint()` is called, which indicates we're actually repainting.
    /// This allows layers that don't need repaint to keep their display list intact.
    pub fn begin_frame(&mut self) {
        // Clear previous_display_list - it's only needed during repaint for cache lookup
        self.previous_display_list = None;
    }

    /// Prepare layer for repainting (Phase 20e: Retained DisplayList).
    ///
    /// Call this when the layer needs to be repainted. Swaps display lists so
    /// previous_display_list has the old items for cache lookup, then clears
    /// current display_list for new items.
    ///
    /// P1 optimization: Reuses DisplayList allocations by swapping and clearing
    /// instead of allocating new Vec each frame.
    ///
    /// P1.1: Uses Arc for cheap cloning to Scene. Arc::make_mut provides
    /// mutable access without cloning when refcount is 1 (which is the case
    /// after Scene drops its reference at end of frame).
    pub fn prepare_for_repaint(&mut self) {
        // Swap current and previous display lists.
        // After swap:
        // - previous_display_list has this frame's items (for cache lookup)
        // - display_list has last frame's previous items (will be cleared)
        std::mem::swap(&mut self.display_list, &mut self.previous_display_list);

        // Clear and reuse the display list (or create new if none exists)
        if let Some(ref mut arc_display_list) = self.display_list {
            // Arc::make_mut gives mutable access. If refcount is 1, no clone.
            // Scene's Arc should have been dropped by now, so this should be cheap.
            Arc::make_mut(arc_display_list).clear();
        } else if let Some(ref element_id) = self.element_id {
            // First frame or after layer recreation - need to allocate
            self.display_list = Some(Arc::new(DisplayList::new(element_id.clone())));
        }
    }

    /// P2: Emit TileSprite primitives for visible tiles based on current scroll offset.
    ///
    /// This is used during compositor-only updates where the element tree is not traversed.
    /// Returns a Vec of TileSprite that should be inserted into the Scene.
    pub(crate) fn emit_tile_sprites(&self) -> Vec<TileSprite> {
        let Some(ref element_id) = self.element_id else {
            return Vec::new();
        };

        // Skip if this is the root layer or not a scroll container
        if self.reason != LayerReason::ScrollContainer {
            return Vec::new();
        }

        let scale_factor = self.scale_factor;
        let tile_size_px = TILE_SIZE as f32 / scale_factor;
        let scroll_offset = self.scroll_offset;
        let scroll_offset_scaled = point(
            ScaledPixels(scroll_offset.x.0 * scale_factor),
            ScaledPixels(scroll_offset.y.0 * scale_factor),
        );
        let viewport = Bounds {
            origin: self.viewport_origin,
            size: self.viewport_size,
        };
        let outer_clip = &self.outer_clip;

        // scroll_offset is negative (scroll down = negative y)
        let content_top_left_x = -scroll_offset.x.0;
        let content_top_left_y = -scroll_offset.y.0;

        let min_tile = TileCoord {
            x: (content_top_left_x / tile_size_px).floor() as i32,
            y: (content_top_left_y / tile_size_px).floor() as i32,
        };

        let max_tile = TileCoord {
            x: ((content_top_left_x + viewport.size.width.0) / tile_size_px).ceil() as i32 - 1,
            y: ((content_top_left_y + viewport.size.height.0) / tile_size_px).ceil() as i32 - 1,
        };

        let mut sprites = Vec::new();
        for tile_y in min_tile.y..=max_tile.y {
            for tile_x in min_tile.x..=max_tile.x {
                let coord = TileCoord { x: tile_x, y: tile_y };

                // Calculate content-space bounds for this tile
                let tile_content_origin = Point {
                    x: Pixels(coord.x as f32 * tile_size_px),
                    y: Pixels(coord.y as f32 * tile_size_px),
                };
                let tile_content_size = Size {
                    width: Pixels(tile_size_px),
                    height: Pixels(tile_size_px),
                };

                // Calculate where this tile should appear on screen
                let screen_origin = Point {
                    x: viewport.origin.x + tile_content_origin.x,
                    y: viewport.origin.y + tile_content_origin.y,
                };

                let tile_sprite = TileSprite {
                    order: 0, // Will be set by insert_primitive
                    _pad: 0,
                    bounds: Bounds {
                        origin: point(
                            ScaledPixels(screen_origin.x.0 * scale_factor) + scroll_offset_scaled.x,
                            ScaledPixels(screen_origin.y.0 * scale_factor) + scroll_offset_scaled.y,
                        ),
                        size: size(
                            ScaledPixels(tile_content_size.width.0 * scale_factor),
                            ScaledPixels(tile_content_size.height.0 * scale_factor),
                        ),
                    },
                    stable_bounds: Bounds {
                        origin: point(
                            ScaledPixels(screen_origin.x.0 * scale_factor),
                            ScaledPixels(screen_origin.y.0 * scale_factor),
                        ),
                        size: size(
                            ScaledPixels(tile_content_size.width.0 * scale_factor),
                            ScaledPixels(tile_content_size.height.0 * scale_factor),
                        ),
                    },
                    scroll_offset: scroll_offset_scaled,
                    content_mask: ContentMask {
                        bounds: Bounds {
                            origin: point(
                                ScaledPixels(outer_clip.bounds.origin.x.0 * scale_factor),
                                ScaledPixels(outer_clip.bounds.origin.y.0 * scale_factor),
                            ),
                            size: size(
                                ScaledPixels(outer_clip.bounds.size.width.0 * scale_factor),
                                ScaledPixels(outer_clip.bounds.size.height.0 * scale_factor),
                            ),
                        },
                    },
                    tile_key: TileKey {
                        container_id: element_id.clone(),
                        coord,
                    },
                };

                sprites.push(tile_sprite);
            }
        }

        sprites
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
    /// This preserves layers across frames, but resets the tree structure
    /// (parent/children relationships) which gets rebuilt during the paint traversal.
    /// Property trees are also cleared since transforms and clips are rebuilt during paint.
    ///
    /// Phase 20e: Also swaps each layer's display_list to previous_display_list,
    /// preserving element entries for cache lookup while building new items.
    ///
    /// Call this at the start of each frame before painting.
    pub fn begin_frame(&mut self) {
        // Reset tree structure for all layers (will be rebuilt during paint)
        // but KEEP the layers for reuse
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

            // Phase 0.3: Root layer uses prepare_for_repaint() to swap display lists,
            // enabling element-level caching (items baked at insert time are self-contained).
            // Other layers use begin_frame() which clears previous_display_list.
            if layer.id == LayerId::ROOT {
                layer.prepare_for_repaint();
            } else {
                layer.begin_frame();
            }
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

    /// Iterate over all layers (Phase 5: Instrumentation).
    pub fn layers(&self) -> impl Iterator<Item = &Layer> {
        self.layers.values()
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
    use crate::ElementId;
    use std::sync::Arc;

    fn test_element_id(n: u64) -> GlobalElementId {
        GlobalElementId(Arc::from([ElementId::Integer(n)]))
    }

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

        let element_id = test_element_id(1);
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

        let id1 = test_element_id(1);
        let id2 = test_element_id(2);

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

        let id1 = test_element_id(1);
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

    #[test]
    fn test_retained_hitbox_coordinate_transform() {
        use crate::px;

        // Test that coordinate transforms work correctly
        // We create a retained hitbox directly with layer-local coordinates
        // and verify to_hitbox transforms them correctly
        let retained = RetainedHitbox {
            id: HitboxId::default(), // Uses default ID
            bounds: Bounds {
                origin: Point {
                    x: px(50.0),  // layer-local
                    y: px(100.0), // layer-local
                },
                size: Size {
                    width: px(50.0),
                    height: px(30.0),
                },
            },
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: Point {
                        x: px(-50.0),  // relative to content origin
                        y: px(-100.0),
                    },
                    size: Size {
                        width: px(1000.0),
                        height: px(800.0),
                    },
                },
            },
            behavior: HitboxBehavior::Normal,
        };

        // Content origin at (50, 100)
        let content_origin = Point {
            x: px(50.0),
            y: px(100.0),
        };

        // Convert to window coordinates
        let restored = retained.to_hitbox(content_origin);

        // Window coordinates = layer-local + content_origin
        assert_eq!(restored.bounds.origin.x.0, 100.0); // 50 + 50
        assert_eq!(restored.bounds.origin.y.0, 200.0); // 100 + 100
        assert_eq!(restored.bounds.size.width.0, 50.0);
        assert_eq!(restored.bounds.size.height.0, 30.0);
    }

    #[test]
    fn test_retained_hitbox_with_scroll() {
        use crate::px;

        // Create a retained hitbox in layer-local (content) coordinates
        // This represents a hitbox at (100, 500) in content space
        let retained = RetainedHitbox {
            id: HitboxId::default(),
            bounds: Bounds {
                origin: Point {
                    x: px(100.0),
                    y: px(500.0),
                },
                size: Size {
                    width: px(100.0),
                    height: px(50.0),
                },
            },
            content_mask: ContentMask {
                bounds: Bounds::default(),
            },
            behavior: HitboxBehavior::Normal,
        };

        // Initial content origin (no scroll)
        let initial_content_origin = Point {
            x: px(0.0),
            y: px(0.0),
        };

        // With no scroll, window position = content position
        let restored = retained.to_hitbox(initial_content_origin);
        assert_eq!(restored.bounds.origin.x.0, 100.0);
        assert_eq!(restored.bounds.origin.y.0, 500.0);

        // Now scroll down by 200px (scroll_offset becomes (0, -200))
        // Content origin = bounds.origin + scroll_offset = (0, -200)
        let scrolled_content_origin = Point {
            x: px(0.0),
            y: px(-200.0),
        };

        // Restore hitbox with new scroll position
        let restored_scrolled = retained.to_hitbox(scrolled_content_origin);

        // The hitbox now appears at y = 500 + (-200) = 300 in window
        assert_eq!(restored_scrolled.bounds.origin.x.0, 100.0);
        assert_eq!(restored_scrolled.bounds.origin.y.0, 300.0); // Scrolled up by 200
    }

    #[test]
    fn test_layer_retained_hitboxes_storage() {
        use crate::px;

        let mut tree = LayerTree::new();

        let element_id = test_element_id(1);
        let layer_id = tree.push_layer(element_id, LayerReason::ScrollContainer);

        // Initially no retained hitboxes
        assert!(!tree.get(layer_id).unwrap().has_retained_hitboxes());

        // Manually add some retained hitboxes
        {
            let layer = tree.get_mut(layer_id).unwrap();
            layer.retained_hitboxes.push(RetainedHitbox {
                id: HitboxId::default(),
                bounds: Bounds {
                    origin: Point {
                        x: px(0.0),
                        y: px(0.0),
                    },
                    size: Size {
                        width: px(100.0),
                        height: px(50.0),
                    },
                },
                content_mask: ContentMask {
                    bounds: Bounds::default(),
                },
                behavior: HitboxBehavior::Normal,
            });
        }

        // Now has retained hitboxes
        assert!(tree.get(layer_id).unwrap().has_retained_hitboxes());

        // Clear retained
        tree.get_mut(layer_id).unwrap().clear_retained();

        // No longer has retained hitboxes
        assert!(!tree.get(layer_id).unwrap().has_retained_hitboxes());
    }
}
