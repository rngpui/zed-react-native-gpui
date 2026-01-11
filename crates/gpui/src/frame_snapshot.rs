//! Frame Snapshot for Compositor Decoupling
//!
//! This module provides the immutable snapshot structures that decouple main-thread
//! content production from compositor-thread presentation. The main thread produces
//! `FrameSnapshot` instances, which the compositor can then present at different
//! scroll offsets without requiring main-thread involvement.
//!
//! ## Architecture
//!
//! ```text
//! Main Thread                    Compositor
//! ───────────                    ──────────
//! draw() → FrameSnapshot    →   compositor_present(snapshot, state)
//!          (immutable)           ↓
//!                               Apply compositor state (scroll, transform, opacity)
//!                               Rasterize dirty tiles
//!                               Emit tile sprites
//!                               GPU render
//! ```

use crate::display_list::DisplayList;
use crate::layer::{LayerId, LayerReason, RetainedHitbox};
use crate::scene::TileKey;
use crate::{Bounds, ContentMask, GlobalElementId, Pixels, Point, Size};
use collections::FxHashMap;
use std::sync::Arc;

/// Immutable snapshot produced by main thread after draw().
///
/// The compositor owns this and can composite it at different scroll offsets
/// without requiring main-thread work. This is the "commit boundary" that
/// separates content production from presentation.
#[derive(Clone)]
pub struct FrameSnapshot {
    /// Layer tree with display lists (immutable after commit).
    pub layers: Arc<LayerTreeSnapshot>,

    /// Hit test data (immutable after commit).
    pub hitboxes: Arc<HitboxSnapshot>,

    /// Generation counter for staleness detection.
    /// Incremented each time main thread produces a new snapshot.
    pub generation: u64,
}

impl FrameSnapshot {
    /// Create a new empty frame snapshot.
    pub fn new() -> Self {
        Self {
            layers: Arc::new(LayerTreeSnapshot::new()),
            hitboxes: Arc::new(HitboxSnapshot::new()),
            generation: 0,
        }
    }

    /// Create a snapshot with the given generation.
    pub fn with_generation(generation: u64) -> Self {
        Self {
            layers: Arc::new(LayerTreeSnapshot::new()),
            hitboxes: Arc::new(HitboxSnapshot::new()),
            generation,
        }
    }
}

impl Default for FrameSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable snapshot of the layer tree structure.
///
/// Contains all layer data needed for compositing without main-thread access.
#[derive(Clone, Default)]
pub struct LayerTreeSnapshot {
    /// All layers indexed by ID.
    pub layers: FxHashMap<LayerId, LayerSnapshot>,

    /// Root layer ID.
    pub root: LayerId,
}

impl LayerTreeSnapshot {
    /// Create a new empty layer tree snapshot.
    pub fn new() -> Self {
        Self {
            layers: FxHashMap::default(),
            root: LayerId::ROOT,
        }
    }

    /// Get a layer by ID.
    pub fn get(&self, id: LayerId) -> Option<&LayerSnapshot> {
        self.layers.get(&id)
    }

    /// Iterate over all layers.
    pub fn iter(&self) -> impl Iterator<Item = (&LayerId, &LayerSnapshot)> {
        self.layers.iter()
    }

    /// Get the number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

/// Immutable snapshot of a single layer.
///
/// Contains all data needed to composite this layer without main-thread access.
#[derive(Clone)]
pub struct LayerSnapshot {
    /// Layer identifier.
    pub id: LayerId,

    /// Element that owns this layer (None for root).
    pub element_id: Option<GlobalElementId>,

    /// Why this element became a layer.
    pub reason: LayerReason,

    /// Display list (already Arc'd for cheap cloning).
    pub display_list: Option<Arc<DisplayList>>,

    /// Tile state for rasterization.
    pub tile_state: TileState,

    /// Bounds in window coordinates.
    pub bounds: Bounds<Pixels>,

    /// Content size for scroll containers.
    pub content_size: Size<Pixels>,

    /// Viewport size for scroll containers.
    pub viewport_size: Size<Pixels>,

    /// Viewport origin in window space.
    pub viewport_origin: Point<Pixels>,

    /// Outer clip mask for tile sprites.
    pub outer_clip: ContentMask<Pixels>,

    /// Scale factor for converting to ScaledPixels.
    pub scale_factor: f32,

    /// Retained hitboxes in layer-local coordinates.
    pub retained_hitboxes: Vec<RetainedHitbox>,

    /// Index range of hitboxes in the frame.
    pub hitbox_range: Option<std::ops::Range<usize>>,

    /// Child layer IDs (for tree traversal).
    pub children: Vec<LayerId>,
}

impl LayerSnapshot {
    /// Create a new layer snapshot from layer data.
    pub fn from_layer(
        id: LayerId,
        element_id: Option<GlobalElementId>,
        reason: LayerReason,
        display_list: Option<Arc<DisplayList>>,
        bounds: Bounds<Pixels>,
        content_size: Size<Pixels>,
        viewport_size: Size<Pixels>,
        viewport_origin: Point<Pixels>,
        outer_clip: ContentMask<Pixels>,
        scale_factor: f32,
        retained_hitboxes: Vec<RetainedHitbox>,
        hitbox_range: Option<std::ops::Range<usize>>,
        children: Vec<LayerId>,
    ) -> Self {
        Self {
            id,
            element_id,
            reason,
            display_list,
            tile_state: TileState::default(),
            bounds,
            content_size,
            viewport_size,
            viewport_origin,
            outer_clip,
            scale_factor,
            retained_hitboxes,
            hitbox_range,
            children,
        }
    }
}

/// State of tiles for a layer.
///
/// Tracks which tiles are valid and which need rasterization.
#[derive(Clone, Default)]
pub struct TileState {
    /// Set of tile keys that have valid cached textures.
    pub valid_tiles: Vec<TileKey>,

    /// Set of tile keys that need rasterization.
    pub dirty_tiles: Vec<TileKey>,
}

impl TileState {
    /// Create a new empty tile state.
    pub fn new() -> Self {
        Self {
            valid_tiles: Vec::new(),
            dirty_tiles: Vec::new(),
        }
    }

    /// Mark all tiles as dirty.
    pub fn invalidate_all(&mut self) {
        self.dirty_tiles.extend(self.valid_tiles.drain(..));
    }

    /// Mark a specific tile as dirty.
    pub fn invalidate_tile(&mut self, key: TileKey) {
        self.valid_tiles.retain(|k| k != &key);
        if !self.dirty_tiles.contains(&key) {
            self.dirty_tiles.push(key);
        }
    }

    /// Mark a tile as valid (after rasterization).
    pub fn mark_valid(&mut self, key: TileKey) {
        self.dirty_tiles.retain(|k| k != &key);
        if !self.valid_tiles.contains(&key) {
            self.valid_tiles.push(key);
        }
    }
}

/// Immutable snapshot of hitbox data.
///
/// Contains all hitboxes from the frame for hit testing during compositor updates.
#[derive(Clone, Default)]
pub struct HitboxSnapshot {
    /// All hitboxes from the frame, indexed by their position.
    pub hitboxes: Vec<HitboxData>,

    /// Mapping from layer ID to hitbox index ranges.
    pub layer_hitbox_ranges: FxHashMap<LayerId, std::ops::Range<usize>>,
}

impl HitboxSnapshot {
    /// Create a new empty hitbox snapshot.
    pub fn new() -> Self {
        Self {
            hitboxes: Vec::new(),
            layer_hitbox_ranges: FxHashMap::default(),
        }
    }

    /// Get hitboxes for a specific layer.
    pub fn hitboxes_for_layer(&self, layer_id: LayerId) -> &[HitboxData] {
        if let Some(range) = self.layer_hitbox_ranges.get(&layer_id) {
            &self.hitboxes[range.clone()]
        } else {
            &[]
        }
    }
}

/// Data for a single hitbox in the snapshot.
#[derive(Clone)]
pub struct HitboxData {
    /// Bounds in layer-local coordinates.
    pub bounds: Bounds<Pixels>,

    /// Content mask in layer-local coordinates.
    pub content_mask: ContentMask<Pixels>,

    /// The layer this hitbox belongs to.
    pub layer_id: LayerId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_snapshot_creation() {
        let snapshot = FrameSnapshot::new();
        assert_eq!(snapshot.generation, 0);
        assert!(snapshot.layers.is_empty());
    }

    #[test]
    fn test_frame_snapshot_with_generation() {
        let snapshot = FrameSnapshot::with_generation(42);
        assert_eq!(snapshot.generation, 42);
    }

    #[test]
    fn test_tile_state_invalidation() {
        let mut state = TileState::new();

        let key = TileKey {
            container_id: GlobalElementId::root(),
            coord: crate::scene::TileCoord { x: 0, y: 0 },
        };

        state.mark_valid(key.clone());
        assert!(state.valid_tiles.contains(&key));
        assert!(!state.dirty_tiles.contains(&key));

        state.invalidate_tile(key.clone());
        assert!(!state.valid_tiles.contains(&key));
        assert!(state.dirty_tiles.contains(&key));
    }
}
