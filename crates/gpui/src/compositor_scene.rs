//! Compositor Scene for Unified Rendering
//!
//! This module provides a layer-based scene structure where each layer's contribution
//! is either tiled content OR direct primitives, never both. This eliminates the
//! content doubling issue during compositor-only frames.
//!
//! ## Key Insight
//!
//! A scroll container's content is ONLY in `Tiled` - never also in `Direct`.
//! This ensures no visual doubling during compositor-only frames.
//!
//! ## Architecture
//!
//! ```text
//! CompositorScene
//!   └── layer_outputs: Vec<LayerOutput>
//!         ├── LayerOutput::Tiled { layer_id, tile_sprites, transform }
//!         │     └── Scroll container content rendered via tiles
//!         └── LayerOutput::Direct { layer_id, primitives }
//!               └── Non-scrolling content (FPS counter, overlays)
//! ```

use crate::layer::LayerId;
use crate::scene::{Primitive, TileSprite, TransformationMatrix};
use collections::FxHashMap;

/// A compositor scene that separates tiled and direct content.
///
/// This structure ensures that each layer's content is rendered through
/// exactly one path: either via tile sprites (for scroll containers) or
/// via direct primitives (for non-scrolling content).
#[derive(Default)]
pub(crate) struct CompositorScene {
    /// Each layer's contribution (either direct primitives OR tile sprites, never both).
    pub layer_outputs: Vec<LayerOutput>,

    /// Mapping from layer ID to output index for quick lookup.
    layer_index: FxHashMap<LayerId, usize>,
}

/// How a layer's content should be rendered.
#[derive(Clone)]
pub(crate) enum LayerOutput {
    /// Layer with tiled content - render via tile sprites.
    ///
    /// Used for scroll containers where content is rasterized into tiles
    /// and composited at different scroll positions.
    Tiled {
        /// The layer this output belongs to.
        layer_id: LayerId,
        /// Tile sprites to render.
        tile_sprites: Vec<TileSprite>,
        /// Transform to apply during compositing.
        transform: TransformationMatrix,
        /// Draw order for this layer.
        order: u32,
    },

    /// Layer with direct primitives (non-scrolling content like FPS counter).
    ///
    /// Used for content that doesn't need tiling, like UI overlays,
    /// FPS counters, or small fixed elements.
    Direct {
        /// The layer this output belongs to.
        layer_id: LayerId,
        /// Primitives to render directly.
        primitives: Vec<Primitive>,
        /// Draw order for this layer.
        order: u32,
    },
}

impl LayerOutput {
    /// Get the layer ID for this output.
    pub fn layer_id(&self) -> LayerId {
        match self {
            LayerOutput::Tiled { layer_id, .. } => *layer_id,
            LayerOutput::Direct { layer_id, .. } => *layer_id,
        }
    }

    /// Get the draw order for this output.
    pub fn order(&self) -> u32 {
        match self {
            LayerOutput::Tiled { order, .. } => *order,
            LayerOutput::Direct { order, .. } => *order,
        }
    }

    /// Check if this is a tiled output.
    pub fn is_tiled(&self) -> bool {
        matches!(self, LayerOutput::Tiled { .. })
    }

    /// Check if this is a direct output.
    pub fn is_direct(&self) -> bool {
        matches!(self, LayerOutput::Direct { .. })
    }
}

impl CompositorScene {
    /// Create a new empty compositor scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the scene for a new frame.
    pub fn clear(&mut self) {
        self.layer_outputs.clear();
        self.layer_index.clear();
    }

    /// Add a tiled layer output.
    ///
    /// This should be used for scroll containers where content is rendered via tiles.
    pub fn add_tiled_output(
        &mut self,
        layer_id: LayerId,
        tile_sprites: Vec<TileSprite>,
        transform: TransformationMatrix,
        order: u32,
    ) {
        let index = self.layer_outputs.len();
        self.layer_outputs.push(LayerOutput::Tiled {
            layer_id,
            tile_sprites,
            transform,
            order,
        });
        self.layer_index.insert(layer_id, index);
    }

    /// Add a direct layer output.
    ///
    /// This should be used for non-scrolling content that doesn't need tiling.
    pub fn add_direct_output(&mut self, layer_id: LayerId, primitives: Vec<Primitive>, order: u32) {
        let index = self.layer_outputs.len();
        self.layer_outputs.push(LayerOutput::Direct {
            layer_id,
            primitives,
            order,
        });
        self.layer_index.insert(layer_id, index);
    }

    /// Get a layer output by layer ID.
    pub fn get_output(&self, layer_id: LayerId) -> Option<&LayerOutput> {
        self.layer_index
            .get(&layer_id)
            .and_then(|&idx| self.layer_outputs.get(idx))
    }

    /// Get a mutable layer output by layer ID.
    pub fn get_output_mut(&mut self, layer_id: LayerId) -> Option<&mut LayerOutput> {
        self.layer_index
            .get(&layer_id)
            .and_then(|&idx| self.layer_outputs.get_mut(idx))
    }

    /// Update tile sprites for a tiled layer.
    ///
    /// This is used during compositor-only updates where we just need to
    /// update the tile sprite positions without changing the content.
    pub fn update_tile_sprites(&mut self, layer_id: LayerId, new_sprites: Vec<TileSprite>) {
        if let Some(&idx) = self.layer_index.get(&layer_id) {
            if let Some(LayerOutput::Tiled { tile_sprites, .. }) = self.layer_outputs.get_mut(idx) {
                *tile_sprites = new_sprites;
            }
        }
    }

    /// Update transform for a layer.
    pub fn update_transform(&mut self, layer_id: LayerId, new_transform: TransformationMatrix) {
        if let Some(&idx) = self.layer_index.get(&layer_id) {
            if let Some(LayerOutput::Tiled { transform, .. }) = self.layer_outputs.get_mut(idx) {
                *transform = new_transform;
            }
        }
    }

    /// Get all layer outputs sorted by draw order.
    pub fn outputs_by_order(&self) -> Vec<&LayerOutput> {
        let mut outputs: Vec<_> = self.layer_outputs.iter().collect();
        outputs.sort_by_key(|o| o.order());
        outputs
    }

    /// Get all tile sprites from all tiled layers.
    pub fn all_tile_sprites(&self) -> Vec<&TileSprite> {
        let mut sprites = Vec::new();
        for output in &self.layer_outputs {
            if let LayerOutput::Tiled { tile_sprites, .. } = output {
                sprites.extend(tile_sprites.iter());
            }
        }
        sprites
    }

    /// Get all direct primitives from all direct layers.
    pub fn all_direct_primitives(&self) -> Vec<&Primitive> {
        let mut primitives = Vec::new();
        for output in &self.layer_outputs {
            if let LayerOutput::Direct {
                primitives: prims, ..
            } = output
            {
                primitives.extend(prims.iter());
            }
        }
        primitives
    }

    /// Check if a layer is rendered via tiles.
    pub fn is_layer_tiled(&self, layer_id: LayerId) -> bool {
        self.get_output(layer_id)
            .map(|o| o.is_tiled())
            .unwrap_or(false)
    }

    /// Get the number of layer outputs.
    pub fn len(&self) -> usize {
        self.layer_outputs.len()
    }

    /// Check if the scene is empty.
    pub fn is_empty(&self) -> bool {
        self.layer_outputs.is_empty()
    }

    /// Iterate over all layer outputs.
    pub fn iter(&self) -> impl Iterator<Item = &LayerOutput> {
        self.layer_outputs.iter()
    }

    /// Iterate over tiled outputs only.
    pub fn tiled_outputs(&self) -> impl Iterator<Item = &LayerOutput> {
        self.layer_outputs.iter().filter(|o| o.is_tiled())
    }

    /// Iterate over direct outputs only.
    pub fn direct_outputs(&self) -> impl Iterator<Item = &LayerOutput> {
        self.layer_outputs.iter().filter(|o| o.is_direct())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::TileKey;
    use crate::{point, size, Bounds, ContentMask, GlobalElementId, ScaledPixels};

    fn make_test_tile_sprite(order: u32) -> TileSprite {
        TileSprite {
            order,
            _pad: 0,
            bounds: Bounds {
                origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
                size: size(ScaledPixels(512.0), ScaledPixels(512.0)),
            },
            stable_bounds: Bounds {
                origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
                size: size(ScaledPixels(512.0), ScaledPixels(512.0)),
            },
            scroll_offset: point(ScaledPixels(0.0), ScaledPixels(0.0)),
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
                    size: size(ScaledPixels(1000.0), ScaledPixels(1000.0)),
                },
            },
            tile_key: TileKey {
                container_id: GlobalElementId::root(),
                coord: crate::scene::TileCoord { x: 0, y: 0 },
            },
        }
    }

    #[test]
    fn test_compositor_scene_creation() {
        let scene = CompositorScene::new();
        assert!(scene.is_empty());
        assert_eq!(scene.len(), 0);
    }

    #[test]
    fn test_add_tiled_output() {
        let mut scene = CompositorScene::new();
        let layer_id = LayerId(1);
        let tile = make_test_tile_sprite(0);

        scene.add_tiled_output(layer_id, vec![tile], TransformationMatrix::unit(), 0);

        assert_eq!(scene.len(), 1);
        assert!(scene.is_layer_tiled(layer_id));
        assert_eq!(scene.all_tile_sprites().len(), 1);
    }

    #[test]
    fn test_add_direct_output() {
        let mut scene = CompositorScene::new();
        let layer_id = LayerId(2);

        scene.add_direct_output(layer_id, vec![], 0);

        assert_eq!(scene.len(), 1);
        assert!(!scene.is_layer_tiled(layer_id));
    }

    #[test]
    fn test_update_tile_sprites() {
        let mut scene = CompositorScene::new();
        let layer_id = LayerId(1);
        let tile1 = make_test_tile_sprite(0);
        let tile2 = make_test_tile_sprite(1);

        scene.add_tiled_output(layer_id, vec![tile1], TransformationMatrix::unit(), 0);
        assert_eq!(scene.all_tile_sprites().len(), 1);

        scene.update_tile_sprites(layer_id, vec![tile2.clone(), tile2]);
        assert_eq!(scene.all_tile_sprites().len(), 2);
    }

    #[test]
    fn test_outputs_by_order() {
        let mut scene = CompositorScene::new();

        scene.add_tiled_output(LayerId(1), vec![], TransformationMatrix::unit(), 2);
        scene.add_direct_output(LayerId(2), vec![], 0);
        scene.add_tiled_output(LayerId(3), vec![], TransformationMatrix::unit(), 1);

        let ordered = scene.outputs_by_order();
        assert_eq!(ordered[0].order(), 0);
        assert_eq!(ordered[1].order(), 1);
        assert_eq!(ordered[2].order(), 2);
    }

    #[test]
    fn test_clear() {
        let mut scene = CompositorScene::new();
        scene.add_tiled_output(LayerId(1), vec![], TransformationMatrix::unit(), 0);
        scene.add_direct_output(LayerId(2), vec![], 1);

        scene.clear();

        assert!(scene.is_empty());
        assert!(scene.get_output(LayerId(1)).is_none());
    }
}
