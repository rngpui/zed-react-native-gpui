//! Tiled Texture Cache for Scroll Container Rendering
//!
//! This module provides infrastructure for caching scroll container content as fixed-size tiles,
//! enabling O(visible_tiles) scrolling by compositing only the tiles in view.
//!
//! Architecture:
//! - All tiles are the same size (512x512 device pixels by default)
//! - Tiles are pooled and reused across containers
//! - Per-container content generation tracking for invalidation
//! - LRU eviction when memory budget exceeded

use crate::scene::{TileCoord, TileKey};
use crate::{Bounds, GlobalElementId, Pixels, Point, Size};
use collections::FxHashMap;
use std::collections::VecDeque;

/// Size of each tile in device pixels. Power of 2 for GPU efficiency.
/// 512 is a good balance: small enough for fine-grained updates,
/// large enough to minimize draw call overhead.
pub const TILE_SIZE: u32 = 512;

/// Maximum tiles to keep in memory per scroll container.
/// Beyond this, LRU eviction kicks in.
pub const MAX_TILES_PER_CONTAINER: usize = 64;

/// Default memory budget in bytes (128 MB for tiles)
const DEFAULT_TILE_MEMORY_BUDGET: usize = 128 * 1024 * 1024;

/// State of a single cached tile.
pub struct CachedTile {
    /// The GPU texture holding rendered content.
    pub texture: metal::Texture,

    /// Generation when this tile was last rendered.
    /// If container's content_generation > this, tile is stale.
    pub rendered_generation: u64,

    /// Frame number when last used (for LRU eviction).
    pub last_used_frame: u64,
}

/// Tracks tile state for a single scroll container.
pub struct ScrollContainerTiles {
    /// Content generation counter. Incremented when content changes.
    pub content_generation: u64,

    /// Total content size (may be much larger than viewport).
    pub content_size: Size<Pixels>,

    /// Cached tiles indexed by coordinate.
    pub tiles: FxHashMap<TileCoord, CachedTile>,

    /// LRU order for eviction (oldest at front).
    lru_order: VecDeque<TileCoord>,
}

impl ScrollContainerTiles {
    fn new(content_size: Size<Pixels>) -> Self {
        Self {
            content_generation: 1,
            content_size,
            tiles: FxHashMap::default(),
            lru_order: VecDeque::new(),
        }
    }

    /// Update LRU order when a tile is accessed.
    fn touch_tile(&mut self, coord: TileCoord) {
        // Remove from current position if exists
        if let Some(pos) = self.lru_order.iter().position(|&c| c == coord) {
            self.lru_order.remove(pos);
        }
        // Add to back (most recently used)
        self.lru_order.push_back(coord);
    }

    /// Get the oldest tile coord for eviction.
    fn oldest_tile(&mut self) -> Option<TileCoord> {
        self.lru_order.pop_front()
    }
}

/// Statistics for tile cache performance.
#[derive(Default, Clone, Copy, Debug)]
pub struct TileCacheStats {
    /// Number of tile cache hits (texture reused without re-render)
    pub hits: u64,
    /// Number of tile renders required
    pub renders: u64,
    /// Number of tiles currently active
    pub active_tiles: u64,
    /// Number of tiles in the free pool
    pub pooled_tiles: u64,
    /// Total memory used (bytes)
    pub total_memory: u64,
}

/// Global tile cache managing all scroll containers.
pub struct TileCache {
    /// Per-container tile storage.
    containers: FxHashMap<GlobalElementId, ScrollContainerTiles>,

    /// Current frame number for LRU tracking.
    frame: u64,

    /// Metal device for texture allocation.
    device: metal::Device,

    /// Reusable texture pool (all tiles are same size).
    free_textures: Vec<metal::Texture>,

    /// Memory tracking.
    allocated_bytes: usize,
    max_bytes: usize,

    /// Statistics.
    stats: TileCacheStats,
}

impl TileCache {
    /// Create a new tile cache.
    pub fn new(device: metal::Device) -> Self {
        Self {
            containers: FxHashMap::default(),
            frame: 0,
            device,
            free_textures: Vec::new(),
            allocated_bytes: 0,
            max_bytes: DEFAULT_TILE_MEMORY_BUDGET,
            stats: TileCacheStats::default(),
        }
    }

    /// Called at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.stats.hits = 0;
        self.stats.renders = 0;
    }

    /// Called at the end of each frame. Evicts unused textures if over budget.
    pub fn end_frame(&mut self) {
        // Evict tiles not used this frame if over memory budget
        if self.allocated_bytes > self.max_bytes {
            self.evict_stale_tiles();
        }
        self.update_stats();
    }

    /// Register/update a scroll container's content info.
    /// Returns the container's current content_generation.
    pub fn register_scroll_container(
        &mut self,
        id: &GlobalElementId,
        content_size: Size<Pixels>,
        content_changed: bool,
    ) -> u64 {
        let container = self
            .containers
            .entry(id.clone())
            .or_insert_with(|| ScrollContainerTiles::new(content_size));

        // If content changed or size changed, bump generation to invalidate tiles
        if content_changed || container.content_size != content_size {
            container.content_generation += 1;
            container.content_size = content_size;
        }

        container.content_generation
    }

    /// Calculate which tiles are visible given scroll offset and viewport.
    /// Returns (min_tile, max_tile) inclusive range.
    pub fn visible_tile_range(
        scroll_offset: Point<Pixels>,
        viewport_size: Size<Pixels>,
        scale_factor: f32,
    ) -> (TileCoord, TileCoord) {
        // Tile size in logical pixels
        let tile_size_px = TILE_SIZE as f32 / scale_factor;

        // scroll_offset is negative (scroll down = negative y)
        // -scroll_offset is the content position at viewport top-left
        let content_top_left_x = -scroll_offset.x.0;
        let content_top_left_y = -scroll_offset.y.0;

        let min_tile = TileCoord {
            x: (content_top_left_x / tile_size_px).floor() as i32,
            y: (content_top_left_y / tile_size_px).floor() as i32,
        };

        let max_tile = TileCoord {
            x: ((content_top_left_x + viewport_size.width.0) / tile_size_px).ceil() as i32 - 1,
            y: ((content_top_left_y + viewport_size.height.0) / tile_size_px).ceil() as i32 - 1,
        };

        (min_tile, max_tile)
    }

    /// Calculate content-space bounds for a tile (in logical pixels).
    pub fn tile_content_bounds(coord: TileCoord, scale_factor: f32) -> Bounds<Pixels> {
        let tile_size_f32 = TILE_SIZE as f32 / scale_factor;
        Bounds {
            origin: Point {
                x: Pixels(coord.x as f32 * tile_size_f32),
                y: Pixels(coord.y as f32 * tile_size_f32),
            },
            size: Size {
                width: Pixels(tile_size_f32),
                height: Pixels(tile_size_f32),
            },
        }
    }

    /// Get or create a tile. Returns (needs_render, texture reference can be obtained via get_tile_texture).
    /// If needs_render is true, the caller should render content to the texture.
    ///
    /// Note: Due to borrow checker constraints, this returns just needs_render.
    /// Call get_tile_texture() afterwards to get the texture reference.
    pub fn acquire_tile(
        &mut self,
        container_id: &GlobalElementId,
        coord: TileCoord,
        content_generation: u64,
    ) -> bool {
        let current_frame = self.frame;

        // First pass: check if tile exists and update it
        let existing_result = {
            let container = self
                .containers
                .get_mut(container_id)
                .expect("Container must be registered before acquiring tiles");

            if let Some(tile) = container.tiles.get_mut(&coord) {
                tile.last_used_frame = current_frame;
                let needs_render = tile.rendered_generation < content_generation;
                if needs_render {
                    tile.rendered_generation = content_generation;
                }
                Some(needs_render)
            } else {
                None
            }
        };

        // Update LRU after releasing the tiles borrow
        if existing_result.is_some() {
            let container = self.containers.get_mut(container_id).unwrap();
            container.touch_tile(coord);
        }

        // Handle existing tile case
        if let Some(needs_render) = existing_result {
            if needs_render {
                self.stats.renders += 1;
            } else {
                self.stats.hits += 1;
            }
            return needs_render;
        }

        // Need to allocate - do this outside the container borrow
        let texture = self.allocate_tile_texture();
        self.stats.renders += 1;

        // Now insert into container
        {
            let container = self
                .containers
                .get_mut(container_id)
                .expect("Container should still exist");

            let tile = CachedTile {
                texture,
                rendered_generation: content_generation,
                last_used_frame: current_frame,
            };

            container.tiles.insert(coord, tile);
            container.touch_tile(coord);
        }

        // Evict if over per-container limit
        self.evict_if_needed(container_id);

        true // New tile always needs render
    }

    /// Look up a tile's texture by key (for rendering during compositing).
    pub fn get_tile_texture(&self, key: &TileKey) -> Option<&metal::Texture> {
        self.containers
            .get(&key.container_id)?
            .tiles
            .get(&key.coord)
            .map(|tile| &tile.texture)
    }

    /// Allocate a new tile texture (or reuse from pool).
    fn allocate_tile_texture(&mut self) -> metal::Texture {
        if let Some(texture) = self.free_textures.pop() {
            return texture;
        }

        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(TILE_SIZE as u64);
        descriptor.set_height(TILE_SIZE as u64);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor.set_usage(
            metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
        );
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);

        self.allocated_bytes += Self::tile_memory_size();
        self.device.new_texture(&descriptor)
    }

    /// Memory size of one tile in bytes.
    const fn tile_memory_size() -> usize {
        (TILE_SIZE as usize) * (TILE_SIZE as usize) * 4 // BGRA8 = 4 bytes/pixel
    }

    /// Evict old tiles if container exceeds per-container limit.
    fn evict_if_needed(&mut self, container_id: &GlobalElementId) {
        let container = self.containers.get_mut(container_id).unwrap();

        while container.tiles.len() > MAX_TILES_PER_CONTAINER {
            if let Some(oldest_coord) = container.oldest_tile() {
                if let Some(tile) = container.tiles.remove(&oldest_coord) {
                    self.free_textures.push(tile.texture);
                    // Memory stays same (moved from active to pool)
                }
            } else {
                break;
            }
        }
    }

    /// Evict tiles not used this frame (global memory pressure).
    fn evict_stale_tiles(&mut self) {
        let current_frame = self.frame;

        for container in self.containers.values_mut() {
            let tiles_to_remove: Vec<TileCoord> = container
                .tiles
                .iter()
                .filter(|(_, tile)| tile.last_used_frame < current_frame)
                .map(|(coord, _)| *coord)
                .collect();

            for coord in tiles_to_remove {
                if let Some(tile) = container.tiles.remove(&coord) {
                    self.free_textures.push(tile.texture);
                }
                // Also remove from LRU
                container.lru_order.retain(|&c| c != coord);
            }
        }

        // If still over budget, drop some pooled textures
        while self.allocated_bytes > self.max_bytes && !self.free_textures.is_empty() {
            self.free_textures.pop();
            self.allocated_bytes -= Self::tile_memory_size();
        }
    }

    /// Remove a scroll container and all its tiles.
    pub fn remove_container(&mut self, container_id: &GlobalElementId) {
        if let Some(container) = self.containers.remove(container_id) {
            for (_, tile) in container.tiles {
                self.free_textures.push(tile.texture);
            }
        }
    }

    /// Clear all cached tiles (e.g., on window resize).
    pub fn clear(&mut self) {
        for container in self.containers.values_mut() {
            for (_, tile) in container.tiles.drain() {
                self.free_textures.push(tile.texture);
            }
            container.lru_order.clear();
        }
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        let mut active_tiles = 0u64;
        for container in self.containers.values() {
            active_tiles += container.tiles.len() as u64;
        }

        self.stats.active_tiles = active_tiles;
        self.stats.pooled_tiles = self.free_textures.len() as u64;
        self.stats.total_memory = self.allocated_bytes as u64;
    }

    /// Get current statistics.
    pub fn stats(&self) -> TileCacheStats {
        self.stats
    }
}
