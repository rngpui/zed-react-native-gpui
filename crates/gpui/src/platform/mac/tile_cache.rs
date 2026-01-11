// Many methods in this module are infrastructure for future use.
// The tile cache architecture is incrementally being integrated.
#![allow(dead_code)]

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
use collections::{FxHashMap, FxHashSet};
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
    /// Slice index in the tile texture array.
    pub slice: u32,

    /// Generation when this tile was last rendered.
    /// If container's content_generation > this, tile is stale.
    pub rendered_generation: u64,

    /// Frame number when last used (for LRU eviction).
    pub last_used_frame: u64,
}

/// Best available draw state for a tile.
pub enum TileDrawState {
    /// Tile is up-to-date for the current content generation.
    Ready(u32),
    /// Tile has rendered content but is stale or re-rasterizing.
    Stale(u32),
    /// Tile has no valid content (use fallback).
    Missing,
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

    /// Tiles that need re-rasterization due to dirty regions.
    /// Set during invalidate_tiles_for_dirty_regions(), cleared after rasterization.
    invalidated_tiles: FxHashSet<TileCoord>,

    /// Last `(display_list.generation, display_list.dirty_generation)` token we processed.
    /// Used to avoid re-processing the same dirty regions on composite_only frames.
    last_processed_dirty_token: Option<(u64, u64)>,
}

impl ScrollContainerTiles {
    fn new(content_size: Size<Pixels>) -> Self {
        Self {
            content_generation: 1,
            content_size,
            tiles: FxHashMap::default(),
            lru_order: VecDeque::new(),
            invalidated_tiles: FxHashSet::default(),
            last_processed_dirty_token: None,
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

    /// Texture array storing all tile slices (slice 0 is reserved as fallback).
    tile_array_texture: metal::Texture,

    /// Reusable slices in the texture array.
    free_slices: Vec<u32>,

    /// Maximum number of tile slices (excluding fallback).
    max_slices: u32,

    /// Memory tracking.
    allocated_bytes: usize,
    max_bytes: usize,

    /// Statistics.
    stats: TileCacheStats,
}

impl TileCache {
    /// Create a new tile cache.
    pub fn new(device: metal::Device) -> Self {
        let max_bytes = DEFAULT_TILE_MEMORY_BUDGET;
        let tile_bytes = Self::tile_memory_size();
        let total_slices = (max_bytes / tile_bytes).max(2);
        let max_slices = (total_slices - 1) as u32;

        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(TILE_SIZE as u64);
        descriptor.set_height(TILE_SIZE as u64);
        descriptor.set_array_length(total_slices as u64);
        descriptor.set_texture_type(metal::MTLTextureType::D2Array);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor.set_usage(
            metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
        );
        descriptor.set_storage_mode(metal::MTLStorageMode::Managed);

        let tile_array_texture = device.new_texture(&descriptor);

        let mut free_slices = Vec::with_capacity(max_slices as usize);
        if max_slices > 0 {
            for slice in 1..=max_slices {
                free_slices.push(slice);
            }
        }

        let mut cache = Self {
            containers: FxHashMap::default(),
            frame: 0,
            device,
            tile_array_texture,
            free_slices,
            max_slices,
            allocated_bytes: tile_bytes * total_slices,
            max_bytes,
            stats: TileCacheStats::default(),
        };
        cache.fill_fallback_slice();
        cache
    }

    /// Called at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.stats.hits = 0;
        self.stats.renders = 0;
    }

    /// Called at the end of each frame. Evicts unused tiles if we need slices.
    pub fn end_frame(&mut self) {
        // Evict tiles not used this frame if we need more slices.
        if self.free_slices.is_empty() {
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

    /// Get or create a tile. Returns whether this tile needs to be rendered.
    /// If needs_render is true, the caller should render content to the tile slice.
    pub fn acquire_tile(
        &mut self,
        container_id: &GlobalElementId,
        coord: TileCoord,
        content_generation: u64,
    ) -> bool {
        let current_frame = self.frame;

        // Check if container is registered
        if !self.containers.contains_key(container_id) {
            panic!("Container {:?} must be registered before acquiring tiles", container_id);
        }

        // First pass: check if tile exists and update it
        let existing_result = {
            let container = self
                .containers
                .get_mut(container_id)
                .expect("Container must be registered before acquiring tiles");

            if let Some(tile) = container.tiles.get_mut(&coord) {
                tile.last_used_frame = current_frame;
                // Don't set rendered_generation here - wait until rasterization completes
                let needs_render =
                    tile.rendered_generation == 0 || tile.rendered_generation < content_generation;
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
        let slice = self.allocate_tile_slice();
        self.stats.renders += 1;

        // Now insert into container
        {
            let container = self
                .containers
                .get_mut(container_id)
                .expect("Container should still exist");

            let tile = CachedTile {
                slice,
                // Start at 0 - will be updated by mark_tile_rendered() after actual rasterization
                rendered_generation: 0,
                last_used_frame: current_frame,
            };

            container.tiles.insert(coord, tile);
            container.touch_tile(coord);
        }

        // Evict if over per-container limit
        self.evict_if_needed(container_id);

        true // New tile always needs render
    }

    /// Access the backing texture array for all tiles.
    pub fn tile_texture_array(&self) -> &metal::Texture {
        &self.tile_array_texture
    }

    /// Slice index reserved for the fallback tile.
    pub fn fallback_slice(&self) -> u32 {
        0
    }

    /// Look up a tile's slice by key (for rendering during compositing).
    pub fn tile_slice(&self, key: &TileKey) -> Option<u32> {
        self.containers
            .get(&key.container_id)?
            .tiles
            .get(&key.coord)
            .map(|tile| tile.slice)
    }

    /// Look up a tile's slice by key, but only if it has been rendered at least once.
    ///
    /// Newly-allocated tiles start with `rendered_generation == 0`. These slices are
    /// uninitialized GPU memory and must never be sampled for drawing.
    pub fn tile_slice_if_rendered(&self, key: &TileKey) -> Option<u32> {
        let container = self.containers.get(&key.container_id)?;
        let tile = container.tiles.get(&key.coord)?;
        if tile.rendered_generation == 0 {
            None
        } else {
            Some(tile.slice)
        }
    }

    /// Determine the best draw state for a tile at a given content generation.
    pub fn tile_draw_state(&self, key: &TileKey, content_generation: u64) -> TileDrawState {
        let Some(container) = self.containers.get(&key.container_id) else {
            return TileDrawState::Missing;
        };
        let Some(tile) = container.tiles.get(&key.coord) else {
            return TileDrawState::Missing;
        };
        if tile.rendered_generation == 0 {
            return TileDrawState::Missing;
        }
        if tile.rendered_generation < content_generation {
            return TileDrawState::Stale(tile.slice);
        }
        if container.invalidated_tiles.contains(&key.coord) {
            return TileDrawState::Stale(tile.slice);
        }
        TileDrawState::Ready(tile.slice)
    }

    /// Mark a tile as having been successfully rendered to the given generation.
    /// Call this after rasterization completes and the slice has been updated.
    pub fn mark_tile_rendered(
        &mut self,
        container_id: &GlobalElementId,
        coord: TileCoord,
        generation: u64,
    ) {
        if let Some(container) = self.containers.get_mut(container_id) {
            if let Some(tile) = container.tiles.get_mut(&coord) {
                tile.rendered_generation = generation;
            }
            // Once we've successfully rasterized the tile, it's no longer considered invalidated.
            container.invalidated_tiles.remove(&coord);
        }
    }

    /// Allocate a new tile slice (or reuse from the free list).
    fn allocate_tile_slice(&mut self) -> u32 {
        if let Some(slice) = self.free_slices.pop() {
            return slice;
        }

        self.evict_stale_tiles();
        if let Some(slice) = self.free_slices.pop() {
            return slice;
        }

        self.evict_oldest_tile();
        self.free_slices
            .pop()
            .expect("tile cache ran out of available slices")
    }

    /// Fill the reserved fallback slice with a simple checkerboard pattern.
    fn fill_fallback_slice(&self) {
        let tile_pixels = TILE_SIZE as usize;
        let bytes_per_row = tile_pixels * 4;
        let bytes_per_image = bytes_per_row * tile_pixels;
        let mut pixels = vec![0u8; bytes_per_image];
        let checker_size = 32;

        for y in 0..tile_pixels {
            for x in 0..tile_pixels {
                let is_dark = ((x / checker_size) + (y / checker_size)) % 2 == 0;
                let (r, g, b) = if is_dark { (0x40, 0x40, 0x40) } else { (0x80, 0x80, 0x80) };
                let idx = (y * tile_pixels + x) * 4;
                pixels[idx] = b;
                pixels[idx + 1] = g;
                pixels[idx + 2] = r;
                pixels[idx + 3] = 0xFF;
            }
        }

        let region =
            metal::MTLRegion::new_2d(0, 0, TILE_SIZE as u64, TILE_SIZE as u64);
        self.tile_array_texture.replace_region_in_slice(
            region,
            0,
            self.fallback_slice() as u64,
            pixels.as_ptr() as *const _,
            bytes_per_row as u64,
            bytes_per_image as u64,
        );
    }

    /// Memory size of one tile in bytes.
    const fn tile_memory_size() -> usize {
        (TILE_SIZE as usize) * (TILE_SIZE as usize) * 4 // BGRA8 = 4 bytes/pixel
    }

    /// Evict the oldest tile across all containers to free a slice.
    fn evict_oldest_tile(&mut self) {
        let mut oldest: Option<(GlobalElementId, TileCoord, u64)> = None;
        for (container_id, container) in self.containers.iter() {
            for (&coord, tile) in &container.tiles {
                if oldest
                    .as_ref()
                    .map_or(true, |(_, _, last_used)| tile.last_used_frame < *last_used)
                {
                    oldest = Some((container_id.clone(), coord, tile.last_used_frame));
                }
            }
        }

        let Some((container_id, coord, _)) = oldest else {
            return;
        };

        if let Some(container) = self.containers.get_mut(&container_id) {
            if let Some(tile) = container.tiles.remove(&coord) {
                self.free_slices.push(tile.slice);
            }
            container.invalidated_tiles.remove(&coord);
            container.lru_order.retain(|&c| c != coord);
        }
    }

    /// Evict old tiles if container exceeds per-container limit.
    fn evict_if_needed(&mut self, container_id: &GlobalElementId) {
        let container = self.containers.get_mut(container_id).unwrap();

        while container.tiles.len() > MAX_TILES_PER_CONTAINER {
            if let Some(oldest_coord) = container.oldest_tile() {
                if let Some(tile) = container.tiles.remove(&oldest_coord) {
                    self.free_slices.push(tile.slice);
                    // Memory stays same (moved from active to pool)
                }
                container.invalidated_tiles.remove(&oldest_coord);
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
                    self.free_slices.push(tile.slice);
                }
                container.invalidated_tiles.remove(&coord);
                // Also remove from LRU
                container.lru_order.retain(|&c| c != coord);
            }
        }

        // If still no free slices, evict the oldest remaining tile.
        if self.free_slices.is_empty() {
            self.evict_oldest_tile();
        }
    }

    /// Remove a scroll container and all its tiles.
    pub fn remove_container(&mut self, container_id: &GlobalElementId) {
        if let Some(container) = self.containers.remove(container_id) {
            for (_, tile) in container.tiles {
                self.free_slices.push(tile.slice);
            }
        }
    }

    /// Clear all cached tiles (e.g., on window resize).
    pub fn clear(&mut self) {
        for container in self.containers.values_mut() {
            container.tiles.clear();
            container.lru_order.clear();
            container.invalidated_tiles.clear();
            container.last_processed_dirty_token = None;
        }
        self.free_slices.clear();
        if self.max_slices > 0 {
            for slice in 1..=self.max_slices {
                self.free_slices.push(slice);
            }
        }
    }

    /// Update statistics.
    fn update_stats(&mut self) {
        let mut active_tiles = 0u64;
        for container in self.containers.values() {
            active_tiles += container.tiles.len() as u64;
        }

        self.stats.active_tiles = active_tiles;
        self.stats.pooled_tiles = self.free_slices.len() as u64;
        self.stats.total_memory = self.allocated_bytes as u64;
    }

    /// Get current statistics.
    pub fn stats(&self) -> TileCacheStats {
        self.stats
    }

    // ========================================================================
    // Dirty Region Invalidation (Phase 14: Incremental Invalidation)
    // ========================================================================

    /// Invalidate tiles that intersect any of the given dirty regions.
    ///
    /// This is more efficient than invalidating all tiles when only a small
    /// part of the content changed. Tiles outside the dirty regions remain
    /// cached and don't need re-rasterization.
    ///
    /// Call this with dirty regions from DisplayList before rendering.
    /// The `display_list_generation` is used to avoid re-processing the same
    /// dirty regions on composite_only frames.
    pub fn invalidate_tiles_for_dirty_regions(
        &mut self,
        container_id: &GlobalElementId,
        dirty_regions: &[Bounds<Pixels>],
        scale_factor: f32,
        display_list_generation: u64,
        dirty_generation: u64,
    ) {
        if dirty_regions.is_empty() {
            return;
        }

        let Some(container) = self.containers.get_mut(container_id) else {
            return;
        };

        // Skip if we've already processed this exact dirty-region set.
        let token = (display_list_generation, dirty_generation);
        if container.last_processed_dirty_token == Some(token) {
            return;
        }
        container.last_processed_dirty_token = Some(token);

        // For each existing tile, check if it intersects any dirty region
        for &coord in container.tiles.keys() {
            let tile_bounds = Self::tile_content_bounds(coord, scale_factor);

            for dirty_region in dirty_regions {
                if tile_bounds.intersects(dirty_region) {
                    container.invalidated_tiles.insert(coord);
                    break; // Don't need to check other regions
                }
            }
        }
    }

    /// Check if a specific tile is invalidated (needs re-rasterization).
    ///
    /// This considers both:
    /// 1. Generation mismatch (full invalidation)
    /// 2. Dirty region invalidation (partial invalidation)
    pub fn is_tile_invalidated(
        &self,
        container_id: &GlobalElementId,
        coord: TileCoord,
        content_generation: u64,
    ) -> bool {
        let Some(container) = self.containers.get(container_id) else {
            return true; // Unknown container = needs render
        };

        // Check dirty region invalidation
        if container.invalidated_tiles.contains(&coord) {
            return true;
        }

        // Check generation mismatch
        if let Some(tile) = container.tiles.get(&coord) {
            tile.rendered_generation == 0 || tile.rendered_generation < content_generation
        } else {
            true // No tile = needs render
        }
    }

    /// Clear the invalidated tiles set for a container.
    ///
    /// Call this after all dirty tiles have been re-rasterized.
    pub fn clear_invalidated_tiles(&mut self, container_id: &GlobalElementId) {
        if let Some(container) = self.containers.get_mut(container_id) {
            container.invalidated_tiles.clear();
        }
    }

    /// Mark a specific tile as invalidated (needs re-rasterization).
    ///
    /// Use this for fine-grained control when you know exactly which tile changed.
    pub fn invalidate_tile(&mut self, container_id: &GlobalElementId, coord: TileCoord) {
        if let Some(container) = self.containers.get_mut(container_id) {
            container.invalidated_tiles.insert(coord);
        }
    }

    /// Get the number of invalidated tiles for a container.
    pub fn invalidated_tile_count(&self, container_id: &GlobalElementId) -> usize {
        self.containers
            .get(container_id)
            .map(|c| c.invalidated_tiles.len())
            .unwrap_or(0)
    }
}
