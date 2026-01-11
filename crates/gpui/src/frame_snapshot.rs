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
use crate::scene::{TileCoord, TileKey, TileSprite};
use crate::{point, size, Bounds, ContentMask, GlobalElementId, Pixels, Point, ScaledPixels, Size};
use collections::FxHashMap;
use std::sync::Arc;

/// Tile size in scaled pixels (matches Layer's TILE_SIZE).
const TILE_SIZE: u32 = 512;

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

    /// Scroll offset at snapshot time (baseline for compositor updates).
    /// Uses CSS convention: negative when scrolled (e.g., -100 = viewing content at y=100).
    pub scroll_offset: Point<Pixels>,

    /// Content origin at snapshot time (viewport_origin + scroll_offset).
    /// This is where content (0,0) appears on screen.
    pub content_origin: Point<Pixels>,

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
        scroll_offset: Point<Pixels>,
        content_origin: Point<Pixels>,
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
            scroll_offset,
            content_origin,
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
    /// The original hitbox ID (preserved for mouse listener lookups).
    pub id: crate::HitboxId,

    /// Stable identity for this hitbox (per-element).
    pub key: crate::window::HitboxKey,

    /// Bounds in layer-local coordinates.
    pub bounds: Bounds<Pixels>,

    /// Content mask in layer-local coordinates.
    pub content_mask: ContentMask<Pixels>,

    /// Hitbox behavior flags.
    pub behavior: crate::HitboxBehavior,

    /// The layer this hitbox belongs to.
    pub layer_id: LayerId,
}

/// Emit tile sprites from a layer snapshot using the given scroll offset.
///
/// This is a pure function that works entirely from immutable snapshot data,
/// enabling compositor-thread presentation without main-thread access.
///
/// # Arguments
/// * `layer` - The layer snapshot containing viewport, content size, etc.
/// * `current_scroll_offset` - The current scroll offset (from compositor state)
///
/// # Returns
/// A vector of TileSprite that should be rendered for this layer.
pub fn emit_tile_sprites_from_snapshot(
    layer: &LayerSnapshot,
    current_scroll_offset: Point<Pixels>,
) -> Vec<TileSprite> {
    let Some(ref element_id) = layer.element_id else {
        return Vec::new();
    };

    // Only emit tiles for scroll containers
    if layer.reason != LayerReason::ScrollContainer {
        return Vec::new();
    }

    let scale_factor = layer.scale_factor;
    let tile_size_px = TILE_SIZE as f32 / scale_factor;

    // scroll_offset uses CSS convention: NEGATIVE when scrolled down (e.g., -1085 = viewing content at y=1085).
    // The shader computes: final_position = stable_bounds + scroll_offset_scaled
    // Tiles should shift UP (negative direction) when scrolled down, so pass scroll_offset as-is.
    let scroll_offset_scaled = point(
        ScaledPixels(current_scroll_offset.x.0 * scale_factor),
        ScaledPixels(current_scroll_offset.y.0 * scale_factor),
    );

    let viewport = Bounds {
        origin: layer.viewport_origin,
        size: layer.viewport_size,
    };
    let outer_clip = &layer.outer_clip;

    // scroll_offset is NEGATIVE when scrolled down (CSS convention).
    // If scroll_offset.y = -1085, we're viewing content starting at y = 1085.
    // Negate scroll_offset to get positive content position.
    let content_top_left_x = (-current_scroll_offset.x.0).max(0.0);
    let content_top_left_y = (-current_scroll_offset.y.0).max(0.0);

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
                order: 0, // Will be set by caller based on previous frame's ordering
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
