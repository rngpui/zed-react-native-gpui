//! Bridge between TileManager and Platform-Specific TileCache
//!
//! This module provides the integration layer between the platform-agnostic
//! TileManager (handles priorities, scheduling, fallback tiers) and the
//! platform-specific TileCache (handles GPU texture allocation).
//!
//! ## Architecture
//!
//! ```text
//! TileManager (gpui core)           TileCache (platform/mac)
//!   ├── Priority calculation         ├── Metal texture allocation
//!   ├── Fallback tier logic          ├── Texture pooling
//!   ├── Memory budget (logical)      ├── GPU memory management
//!   └── Scheduling decisions         └── Render pass setup
//!              ↓                              ↓
//!          TileBridge (this module)
//!              ├── Maps TextureId ↔ metal::Texture
//!              ├── Coordinates state between systems
//!              └── Provides unified API for renderer
//! ```
//!
//! ## Usage
//!
//! The renderer uses TileBridge to:
//! 1. Get tiles sorted by priority for rasterization
//! 2. Look up textures with three-tier fallback
//! 3. Update tile state after rasterization completes

use crate::layer::LayerId;
use crate::scene::TileKey;
use crate::tile_manager::{TextureId, TileFallback, TileManager, TilePriority};
use crate::task_graph::{RasterTask, TaskGraph, TaskId};
use crate::raster_worker_pool::{RasterResult, RasterWorkerPool};
use crate::{Bounds, Pixels, Point, ScaledPixels, Size};
use crate::display_list::DisplayList;
use collections::FxHashMap;
use std::sync::Arc;

/// Mapping between platform-agnostic TextureId and platform-specific texture.
/// This trait allows the bridge to work with any texture type.
pub trait TextureRegistry {
    /// The platform-specific texture type.
    type Texture: Clone;

    /// Look up a texture by ID.
    fn get_texture(&self, id: TextureId) -> Option<&Self::Texture>;

    /// Register a new texture and get its ID.
    fn register_texture(&mut self, texture: Self::Texture) -> TextureId;

    /// Remove a texture by ID.
    fn remove_texture(&mut self, id: TextureId) -> Option<Self::Texture>;
}

/// A simple in-memory texture registry for testing.
#[derive(Default)]
pub struct SimpleTextureRegistry<T> {
    textures: FxHashMap<TextureId, T>,
}

impl<T: Clone> TextureRegistry for SimpleTextureRegistry<T> {
    type Texture = T;

    fn get_texture(&self, id: TextureId) -> Option<&T> {
        self.textures.get(&id)
    }

    fn register_texture(&mut self, texture: T) -> TextureId {
        let id = TextureId::new();
        self.textures.insert(id, texture);
        id
    }

    fn remove_texture(&mut self, id: TextureId) -> Option<T> {
        self.textures.remove(&id)
    }
}

impl<T: Clone> SimpleTextureRegistry<T> {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            textures: FxHashMap::default(),
        }
    }
}

/// Result of looking up a tile for rendering.
pub struct TileLookupResult<T> {
    /// The tile key.
    pub key: TileKey,
    /// The fallback tier being used.
    pub fallback: TileFallback,
    /// The texture if any is available (None for checkerboard).
    pub texture: Option<T>,
    /// Whether this is degraded quality.
    pub is_degraded: bool,
}

/// Describes a tile that needs rasterization.
pub struct TileRasterRequest {
    /// The tile key.
    pub key: TileKey,
    /// The layer ID.
    pub layer_id: LayerId,
    /// Priority for scheduling.
    pub priority: TilePriority,
    /// Bounds of the tile in scaled pixels.
    pub tile_bounds: Bounds<ScaledPixels>,
    /// Scale factor.
    pub scale_factor: f32,
}

/// Bridge coordinating TileManager with TaskGraph and RasterWorkerPool.
///
/// This is the main entry point for Phase 2 tile infrastructure.
/// The renderer uses this to:
/// 1. Update tile priorities based on viewport
/// 2. Get sorted list of tiles needing rasterization
/// 3. Dispatch work to the worker pool
/// 4. Collect results and update tile state
pub struct TileBridge {
    /// The tile manager handling priorities and fallback.
    pub tile_manager: TileManager,

    /// Task graph for dependency-aware scheduling.
    pub task_graph: TaskGraph,

    /// Worker pool for async rasterization.
    pub worker_pool: Option<RasterWorkerPool>,

    /// Current content generation for invalidation tracking.
    content_generation: u64,

    /// Tile size in device pixels.
    tile_size: u32,
}

impl TileBridge {
    /// Create a new tile bridge with default settings.
    pub fn new() -> Self {
        Self {
            tile_manager: TileManager::new(),
            task_graph: TaskGraph::new(),
            worker_pool: None,
            content_generation: 1,
            tile_size: 512,
        }
    }

    /// Create with a worker pool for async rasterization.
    pub fn with_worker_pool(worker_pool: RasterWorkerPool) -> Self {
        Self {
            tile_manager: TileManager::new(),
            task_graph: TaskGraph::new(),
            worker_pool: Some(worker_pool),
            content_generation: 1,
            tile_size: 512,
        }
    }

    /// Set the tile size in device pixels.
    pub fn set_tile_size(&mut self, size: u32) {
        self.tile_size = size;
    }

    /// Bump content generation (call when content changes).
    pub fn invalidate_content(&mut self) {
        self.content_generation += 1;
    }

    /// Get current content generation.
    pub fn content_generation(&self) -> u64 {
        self.content_generation
    }

    /// Update tile priorities for a layer based on viewport.
    pub fn update_priorities(
        &mut self,
        layer_id: LayerId,
        viewport_origin: Point<Pixels>,
        viewport_size: Size<Pixels>,
        scroll_offset: Point<Pixels>,
    ) {
        self.tile_manager.update_priorities(
            layer_id,
            viewport_origin,
            viewport_size,
            scroll_offset,
        );
    }

    /// Get tiles needing rasterization, sorted by priority.
    pub fn tiles_needing_rasterization(&self) -> Vec<TileRasterRequest> {
        self.tile_manager
            .tiles_needing_rasterization()
            .into_iter()
            .map(|tile| {
                let tile_size_px = self.tile_size as f32;
                let tile_bounds = Bounds {
                    origin: Point {
                        x: ScaledPixels(tile.key.coord.x as f32 * tile_size_px),
                        y: ScaledPixels(tile.key.coord.y as f32 * tile_size_px),
                    },
                    size: Size {
                        width: ScaledPixels(tile_size_px),
                        height: ScaledPixels(tile_size_px),
                    },
                };

                TileRasterRequest {
                    key: tile.key.clone(),
                    layer_id: tile.layer_id,
                    priority: tile.priority,
                    tile_bounds,
                    scale_factor: 1.0,
                }
            })
            .collect()
    }

    /// Start rasterization for a tile and add to task graph.
    ///
    /// Returns the task ID that can be used to track completion.
    pub fn start_tile_rasterization(
        &mut self,
        key: &TileKey,
        display_list: Arc<DisplayList>,
        tile_bounds: Bounds<ScaledPixels>,
        scale_factor: f32,
    ) -> Option<TaskId> {
        // Mark tile as pending in TileManager
        let _task_id_raw = self.tile_manager.start_rasterization(key)?;

        // Add to task graph
        let task_id = self.task_graph.add_task(RasterTask::TileRaster {
            tile_key: key.clone(),
            display_list,
            tile_bounds,
            scale_factor,
        });

        Some(task_id)
    }

    /// Dispatch ready tasks to worker pool.
    ///
    /// Returns number of tasks dispatched.
    pub fn dispatch_ready_tasks(&mut self) -> usize {
        let Some(pool) = &self.worker_pool else {
            return 0;
        };

        pool.dispatch_from_graph(&mut self.task_graph)
    }

    /// Collect completed results from worker pool.
    ///
    /// Returns raster results that should be uploaded to GPU.
    pub fn collect_completed(&mut self) -> Vec<RasterResult> {
        let Some(pool) = &self.worker_pool else {
            return Vec::new();
        };

        pool.collect_and_update_graph(&mut self.task_graph)
    }

    /// Mark a tile as ready after GPU upload.
    pub fn complete_tile(&mut self, key: &TileKey, texture_id: TextureId) {
        self.tile_manager.complete_rasterization(
            key,
            texture_id,
            self.content_generation,
        );
    }

    /// Look up a tile with three-tier fallback.
    pub fn lookup_tile<R: TextureRegistry>(
        &self,
        key: &TileKey,
        registry: &R,
    ) -> TileLookupResult<R::Texture> {
        let fallback = self.tile_manager.get_tile_fallback(key);
        let texture = fallback
            .texture_id()
            .and_then(|id| registry.get_texture(id))
            .cloned();
        let is_degraded = fallback.is_degraded();

        TileLookupResult {
            key: key.clone(),
            fallback,
            texture,
            is_degraded,
        }
    }

    /// Evict tiles to stay within memory budget.
    pub fn evict_if_needed(&mut self) -> Vec<TextureId> {
        self.tile_manager.evict_if_needed()
    }

    /// Invalidate all tiles for a layer (content changed).
    pub fn invalidate_layer(&mut self, layer_id: LayerId) {
        self.tile_manager.invalidate_layer(layer_id);
    }

    /// Clear all tiles for a layer (layer removed).
    pub fn clear_layer(&mut self, layer_id: LayerId) -> Vec<TextureId> {
        self.tile_manager.clear_layer(layer_id)
    }

    /// Check if worker pool is idle.
    pub fn is_idle(&self) -> bool {
        self.worker_pool
            .as_ref()
            .map(|p| p.is_idle())
            .unwrap_or(true)
    }

    /// Get the number of in-flight rasterization jobs.
    pub fn in_flight_count(&self) -> usize {
        self.worker_pool
            .as_ref()
            .map(|p| p.in_flight_count())
            .unwrap_or(0)
    }
}

impl Default for TileBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::TileCoord;
    use crate::GlobalElementId;

    #[test]
    fn test_simple_texture_registry() {
        let mut registry = SimpleTextureRegistry::<String>::new();

        let id1 = registry.register_texture("texture1".to_string());
        let id2 = registry.register_texture("texture2".to_string());

        assert_eq!(registry.get_texture(id1), Some(&"texture1".to_string()));
        assert_eq!(registry.get_texture(id2), Some(&"texture2".to_string()));

        let removed = registry.remove_texture(id1);
        assert_eq!(removed, Some("texture1".to_string()));
        assert_eq!(registry.get_texture(id1), None);
    }

    #[test]
    fn test_tile_bridge_lookup_fallback() {
        let mut bridge = TileBridge::new();
        let mut registry = SimpleTextureRegistry::<String>::new();

        let key = TileKey {
            container_id: GlobalElementId::root(),
            coord: TileCoord { x: 0, y: 0 },
        };

        // Initially should be checkerboard
        let result = bridge.lookup_tile(&key, &registry);
        assert!(result.fallback.is_checkerboard());
        assert!(result.texture.is_none());

        // Create tile and mark ready
        let layer_id = LayerId(1);
        bridge.tile_manager.get_or_create_tile(key.clone(), layer_id);
        let texture_id = registry.register_texture("high_res".to_string());
        bridge.complete_tile(&key, texture_id);

        // Now should be high-res
        let result = bridge.lookup_tile(&key, &registry);
        assert_eq!(result.fallback, TileFallback::HighRes(texture_id));
        assert_eq!(result.texture, Some("high_res".to_string()));
        assert!(!result.is_degraded);
    }
}
