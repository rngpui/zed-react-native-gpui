//! TileManager for Chromium-Style Tile Rasterization
//!
//! This module provides the central coordinator for all tile rasterization work,
//! implementing Chromium's TileManager patterns:
//!
//! - Priority calculation based on distance to viewport and scroll velocity
//! - Three-tier fallback: high-res → low-res → stale → checkerboard
//! - Memory budget management with priority-based eviction
//! - Task graph for dependency management (tiles can depend on image decodes)
//!
//! ## Architecture
//!
//! ```text
//! TileManager
//!   ├── tiles: HashMap<TileKey, ManagedTile>  -- All tiles across all layers
//!   ├── task_graph: TaskGraph                 -- Dependencies between tasks
//!   ├── raster_worker_pool: RasterWorkerPool  -- Dedicated worker threads
//!   └── memory_budget: MemoryBudget           -- Global memory tracking
//!
//! ManagedTile
//!   ├── priority: TilePriority     -- Distance + time-to-visible
//!   ├── state: TileState           -- Missing/Dirty/Pending/Ready
//!   ├── texture: Option<Texture>   -- High-res texture
//!   ├── low_res_texture: Option<Texture>  -- Fast fallback
//!   └── stale_texture: Option<Texture>    -- Previous frame's texture
//! ```

use crate::layer::LayerId;
use crate::scene::TileKey;
use crate::{Pixels, Point, Size};
use collections::FxHashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Resolution level for tiles (Chromium-style multi-resolution).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TileResolution {
    /// Full resolution (1:1 with content).
    High,
    /// Quarter resolution (1/4 scale) for fast fallback.
    Low,
}

impl Default for TileResolution {
    fn default() -> Self {
        TileResolution::High
    }
}

/// State of a managed tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TileState {
    /// No texture exists for this tile.
    Missing,
    /// Tile content is stale and needs re-rasterization.
    Dirty,
    /// Rasterization is in progress.
    Pending {
        /// The task ID tracking this rasterization.
        task_id: u64,
    },
    /// Tile is ready with a valid texture.
    Ready,
}

impl Default for TileState {
    fn default() -> Self {
        TileState::Missing
    }
}

/// Represents the best available content for rendering a tile.
/// Follows Chromium's three-tier fallback: high-res → low-res → stale → checkerboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TileFallback {
    /// High-resolution texture is available (preferred).
    HighRes(TextureId),
    /// Low-resolution texture is available (scaled up).
    LowRes(TextureId),
    /// Stale texture from previous frame (may be outdated).
    Stale(TextureId),
    /// No texture available, should render checkerboard.
    Checkerboard,
}

impl TileFallback {
    /// Get the texture ID if any texture is available.
    pub fn texture_id(&self) -> Option<TextureId> {
        match self {
            TileFallback::HighRes(id)
            | TileFallback::LowRes(id)
            | TileFallback::Stale(id) => Some(*id),
            TileFallback::Checkerboard => None,
        }
    }

    /// Check if this represents degraded quality (not high-res).
    pub fn is_degraded(&self) -> bool {
        !matches!(self, TileFallback::HighRes(_))
    }

    /// Check if this is checkerboard (worst case).
    pub fn is_checkerboard(&self) -> bool {
        matches!(self, TileFallback::Checkerboard)
    }
}

/// Chromium-style tile priority with time-to-visible calculation.
///
/// Priority determines rasterization order and eviction order.
/// Lower values = higher priority (will be rasterized first, evicted last).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TilePriority {
    /// Distance from tile to viewport edge in pixels.
    /// 0.0 = visible, positive = offscreen.
    pub distance_to_visible: f32,

    /// Estimated time until tile becomes visible based on scroll velocity.
    /// 0.0 = already visible or will be visible very soon.
    /// f32::MAX = tile is scrolling away or stationary offscreen.
    pub time_to_visible: f32,

    /// Resolution bucket for this tile.
    pub resolution: TileResolution,

    /// Whether this tile is required for pending tree activation.
    /// Required tiles block activation until ready.
    pub required_for_activation: bool,
}

impl TilePriority {
    /// Priority value for currently visible tiles.
    pub const VISIBLE: Self = TilePriority {
        distance_to_visible: 0.0,
        time_to_visible: 0.0,
        resolution: TileResolution::High,
        required_for_activation: true,
    };

    /// Calculate a single priority score for sorting.
    /// Lower score = higher priority.
    pub fn priority_score(&self) -> f32 {
        // Required tiles get highest priority
        let required_bonus = if self.required_for_activation {
            0.0
        } else {
            1000.0
        };

        // High-res tiles get priority over low-res
        let resolution_bonus = match self.resolution {
            TileResolution::High => 0.0,
            TileResolution::Low => 500.0,
        };

        // Combine distance and time-to-visible
        // Time-to-visible is more important for scroll prediction
        required_bonus + resolution_bonus + self.time_to_visible * 100.0 + self.distance_to_visible
    }
}

impl PartialEq for TilePriority {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score() == other.priority_score()
    }
}

impl Eq for TilePriority {}

impl PartialOrd for TilePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TilePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower score = higher priority
        self.priority_score()
            .partial_cmp(&other.priority_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// A managed tile with priority, state, and multi-resolution textures.
///
/// This is the core unit of the TileManager. Each tile tracks:
/// - Its current state (missing, dirty, pending, ready)
/// - Priority for rasterization scheduling
/// - Multiple texture versions for fallback
pub(crate) struct ManagedTile {
    /// Unique key identifying this tile.
    pub key: TileKey,

    /// Layer this tile belongs to.
    pub layer_id: LayerId,

    /// Current priority for scheduling.
    pub priority: TilePriority,

    /// Current state.
    pub state: TileState,

    /// High-resolution texture (None if not rasterized).
    /// This is a platform-agnostic handle that will be:
    /// - metal::Texture on macOS
    /// - wgpu::Texture after Phase 5
    pub texture_id: Option<TextureId>,

    /// Low-resolution texture for fast fallback (1/4 scale).
    pub low_res_texture_id: Option<TextureId>,

    /// Previous frame's texture for stale fallback.
    /// Kept when transitioning to Dirty/Pending state.
    pub stale_texture_id: Option<TextureId>,

    /// Content generation when this tile was last rasterized.
    /// Used to detect stale tiles when content changes.
    pub rendered_generation: u64,

    /// Monotonic access counter for LRU eviction.
    pub last_used: u64,
}

impl ManagedTile {
    /// Create a new managed tile.
    pub fn new(key: TileKey, layer_id: LayerId) -> Self {
        Self {
            key,
            layer_id,
            priority: TilePriority::default(),
            state: TileState::Missing,
            texture_id: None,
            low_res_texture_id: None,
            stale_texture_id: None,
            rendered_generation: 0,
            last_used: 0,
        }
    }

    /// Mark this tile as needing rasterization.
    /// Preserves current texture as stale for fallback.
    pub fn mark_dirty(&mut self) {
        // Save current texture as stale before transitioning
        if self.state == TileState::Ready {
            self.stale_texture_id = self.texture_id.take();
        }
        self.state = TileState::Dirty;
    }

    /// Mark this tile as pending rasterization.
    pub fn mark_pending(&mut self, task_id: u64) {
        // Save current texture as stale if transitioning from Ready
        if self.state == TileState::Ready {
            self.stale_texture_id = self.texture_id.take();
        }
        self.state = TileState::Pending { task_id };
    }

    /// Mark this tile as ready with a new texture.
    pub fn mark_ready(&mut self, texture_id: TextureId, generation: u64) {
        self.texture_id = Some(texture_id);
        self.stale_texture_id = None; // Clear stale, no longer needed
        self.rendered_generation = generation;
        self.state = TileState::Ready;
    }

    /// Get the best available texture for rendering.
    /// Implements three-tier fallback: high-res → low-res → stale.
    pub fn best_texture(&self) -> Option<TextureId> {
        self.texture_id
            .or(self.low_res_texture_id)
            .or(self.stale_texture_id)
    }

    /// Get the fallback tier for this tile.
    /// Returns which texture tier should be used for rendering, following
    /// Chromium's three-tier fallback: high-res → low-res → stale → checkerboard.
    pub fn fallback(&self) -> TileFallback {
        if let Some(id) = self.texture_id {
            TileFallback::HighRes(id)
        } else if let Some(id) = self.low_res_texture_id {
            TileFallback::LowRes(id)
        } else if let Some(id) = self.stale_texture_id {
            TileFallback::Stale(id)
        } else {
            TileFallback::Checkerboard
        }
    }

    /// Check if this tile has any texture available (not checkerboard).
    pub fn has_any_texture(&self) -> bool {
        self.texture_id.is_some()
            || self.low_res_texture_id.is_some()
            || self.stale_texture_id.is_some()
    }
}

/// Platform-agnostic texture identifier.
///
/// This is an opaque handle that will be resolved to actual textures
/// by the platform-specific renderer. This allows TileManager to be
/// platform-agnostic while still tracking texture ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TextureId(pub u64);

impl TextureId {
    /// Generate a new unique texture ID.
    pub fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
        TextureId(NEXT_ID.fetch_add(1, Ordering::Relaxed) as u64)
    }
}

impl Default for TextureId {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory budget for tile textures.
///
/// Tracks memory usage and provides eviction decisions.
#[derive(Debug)]
pub(crate) struct MemoryBudget {
    /// Total memory limit in bytes.
    pub total_bytes: usize,

    /// Currently allocated bytes.
    pub allocated_bytes: AtomicUsize,

    /// Soft limit - start evicting when exceeded.
    pub soft_limit_bytes: usize,

    /// Bytes per tile (computed from tile size and format).
    pub bytes_per_tile: usize,
}

impl MemoryBudget {
    /// Create a new memory budget.
    ///
    /// # Arguments
    /// * `total_bytes` - Maximum memory for tiles
    /// * `tile_size` - Size of each tile in pixels (e.g., 512)
    /// * `bytes_per_pixel` - Bytes per pixel (e.g., 4 for RGBA8)
    pub fn new(total_bytes: usize, tile_size: u32, bytes_per_pixel: usize) -> Self {
        let bytes_per_tile = (tile_size as usize) * (tile_size as usize) * bytes_per_pixel;
        Self {
            total_bytes,
            allocated_bytes: AtomicUsize::new(0),
            soft_limit_bytes: (total_bytes as f64 * 0.8) as usize, // 80% soft limit
            bytes_per_tile,
        }
    }

    /// Check if we're over the soft limit and should start evicting.
    pub fn should_evict(&self) -> bool {
        self.allocated_bytes.load(Ordering::Relaxed) > self.soft_limit_bytes
    }

    /// Check if we can allocate a new tile.
    pub fn can_allocate(&self) -> bool {
        self.allocated_bytes.load(Ordering::Relaxed) + self.bytes_per_tile <= self.total_bytes
    }

    /// Record a tile allocation.
    pub fn allocate(&self) {
        self.allocated_bytes
            .fetch_add(self.bytes_per_tile, Ordering::Relaxed);
    }

    /// Record a tile deallocation.
    pub fn deallocate(&self) {
        self.allocated_bytes
            .fetch_sub(self.bytes_per_tile, Ordering::Relaxed);
    }

    /// Get current memory usage.
    pub fn used_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Get percentage of budget used.
    pub fn usage_percent(&self) -> f32 {
        (self.used_bytes() as f32 / self.total_bytes as f32) * 100.0
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        // Default: 128MB budget, 512x512 tiles, 4 bytes per pixel (RGBA8)
        Self::new(128 * 1024 * 1024, 512, 4)
    }
}

/// Scroll velocity for time-to-visible priority calculation.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScrollVelocity {
    /// Pixels per second in X direction.
    pub x: f32,
    /// Pixels per second in Y direction.
    pub y: f32,
}

impl ScrollVelocity {
    /// Check if scrolling.
    pub fn is_scrolling(&self) -> bool {
        self.x.abs() > 1.0 || self.y.abs() > 1.0
    }

    /// Get scroll speed magnitude.
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

/// Central coordinator for tile rasterization.
///
/// Implements Chromium's TileManager patterns:
/// - Owns all tiles across all layers
/// - Calculates priorities based on viewport and scroll velocity
/// - Manages memory budget with priority-based eviction
/// - Coordinates with TaskGraph for dependency management
pub(crate) struct TileManager {
    /// All tiles indexed by key.
    tiles: FxHashMap<TileKey, ManagedTile>,

    /// Tiles grouped by layer for efficient iteration.
    tiles_by_layer: FxHashMap<LayerId, Vec<TileKey>>,

    /// Memory budget for tile textures.
    memory_budget: MemoryBudget,

    /// Next task ID for rasterization jobs.
    next_task_id: u64,

    /// Current scroll velocity for priority calculation.
    scroll_velocity: FxHashMap<LayerId, ScrollVelocity>,

    /// Monotonic counter for LRU tracking.
    usage_counter: u64,
}

impl TileManager {
    /// Create a new TileManager with default memory budget.
    pub fn new() -> Self {
        Self::with_memory_budget(MemoryBudget::default())
    }

    /// Create a TileManager with custom memory budget.
    pub fn with_memory_budget(memory_budget: MemoryBudget) -> Self {
        Self {
            tiles: FxHashMap::default(),
            tiles_by_layer: FxHashMap::default(),
            memory_budget,
            next_task_id: 1,
            scroll_velocity: FxHashMap::default(),
            usage_counter: 0,
        }
    }

    /// Get or create a managed tile.
    pub fn get_or_create_tile(&mut self, key: TileKey, layer_id: LayerId) -> &mut ManagedTile {
        if !self.tiles.contains_key(&key) {
            let tile = ManagedTile::new(key.clone(), layer_id);
            self.tiles.insert(key.clone(), tile);
            self.tiles_by_layer
                .entry(layer_id)
                .or_default()
                .push(key.clone());
        }
        self.tiles.get_mut(&key).expect("tile should exist")
    }

    /// Get a tile by key.
    pub fn get_tile(&self, key: &TileKey) -> Option<&ManagedTile> {
        self.tiles.get(key)
    }

    /// Get a mutable tile by key.
    pub fn get_tile_mut(&mut self, key: &TileKey) -> Option<&mut ManagedTile> {
        self.tiles.get_mut(key)
    }

    /// Check if a tile is ready (has valid high-res texture).
    pub fn is_tile_ready(&self, key: &TileKey) -> bool {
        self.tiles
            .get(key)
            .map(|t| t.state == TileState::Ready)
            .unwrap_or(false)
    }

    /// Get the best available texture for a tile (three-tier fallback).
    pub fn get_tile_texture(&self, key: &TileKey) -> Option<TextureId> {
        self.tiles.get(key).and_then(|t| t.best_texture())
    }

    /// Get the fallback tier for a tile.
    /// This is used by the renderer to determine which texture to use
    /// and whether to show degraded quality indicators.
    pub fn get_tile_fallback(&self, key: &TileKey) -> TileFallback {
        self.tiles
            .get(key)
            .map(|t| t.fallback())
            .unwrap_or(TileFallback::Checkerboard)
    }

    /// Record a tile access for LRU eviction tracking.
    pub fn touch_tile(&mut self, key: &TileKey) {
        if let Some(tile) = self.tiles.get_mut(key) {
            self.usage_counter = self.usage_counter.wrapping_add(1);
            tile.last_used = self.usage_counter;
        }
    }

    /// Update scroll velocity for priority calculation.
    pub fn set_scroll_velocity(&mut self, layer_id: LayerId, velocity: ScrollVelocity) {
        self.scroll_velocity.insert(layer_id, velocity);
    }

    /// Update priorities for all tiles in a layer based on viewport and scroll velocity.
    ///
    /// # Arguments
    /// * `layer_id` - The layer to update
    /// * `viewport` - Current visible viewport bounds
    /// * `content_size` - Total content size
    /// * `scroll_offset` - Current scroll offset
    pub fn update_priorities(
        &mut self,
        layer_id: LayerId,
        _viewport_origin: Point<Pixels>,
        viewport_size: Size<Pixels>,
        scroll_offset: Point<Pixels>,
        required_for_activation: bool,
    ) {
        let velocity = self.scroll_velocity.get(&layer_id).copied().unwrap_or_default();

        // Calculate visible content region
        let visible_left = -scroll_offset.x.0;
        let visible_top = -scroll_offset.y.0;
        let visible_right = visible_left + viewport_size.width.0;
        let visible_bottom = visible_top + viewport_size.height.0;

        if let Some(tile_keys) = self.tiles_by_layer.get(&layer_id) {
            for key in tile_keys.iter() {
                if let Some(tile) = self.tiles.get_mut(key) {
                    // Calculate tile bounds in content space
                    let tile_size = 512.0; // TODO: Make configurable
                    let tile_left = key.coord.x as f32 * tile_size;
                    let tile_top = key.coord.y as f32 * tile_size;
                    let tile_right = tile_left + tile_size;
                    let tile_bottom = tile_top + tile_size;

                    // Calculate distance to viewport
                    let distance = Self::distance_to_rect(
                        tile_left,
                        tile_top,
                        tile_right,
                        tile_bottom,
                        visible_left,
                        visible_top,
                        visible_right,
                        visible_bottom,
                    );

                    // Calculate time-to-visible based on scroll velocity
                    let time_to_visible = if distance <= 0.0 {
                        0.0 // Already visible
                    } else if velocity.is_scrolling() {
                        // Estimate time based on scroll direction and speed
                        let toward_velocity = Self::velocity_toward_tile(
                            velocity,
                            tile_left,
                            tile_top,
                            visible_left,
                            visible_top,
                        );
                        if toward_velocity > 0.0 {
                            distance / toward_velocity
                        } else {
                            f32::MAX // Scrolling away
                        }
                    } else {
                        f32::MAX // Not scrolling
                    };

                    tile.priority = TilePriority {
                        distance_to_visible: distance.max(0.0),
                        time_to_visible,
                        resolution: TileResolution::High,
                        required_for_activation: required_for_activation && distance <= 0.0,
                    };
                }
            }
        }
    }

    /// Calculate distance from a tile rect to the viewport rect.
    /// Returns 0 or negative if overlapping.
    fn distance_to_rect(
        tile_left: f32,
        tile_top: f32,
        tile_right: f32,
        tile_bottom: f32,
        vp_left: f32,
        vp_top: f32,
        vp_right: f32,
        vp_bottom: f32,
    ) -> f32 {
        let dx = if tile_right < vp_left {
            vp_left - tile_right
        } else if tile_left > vp_right {
            tile_left - vp_right
        } else {
            0.0
        };

        let dy = if tile_bottom < vp_top {
            vp_top - tile_bottom
        } else if tile_top > vp_bottom {
            tile_top - vp_bottom
        } else {
            0.0
        };

        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate velocity component toward a tile.
    fn velocity_toward_tile(
        velocity: ScrollVelocity,
        tile_left: f32,
        tile_top: f32,
        vp_left: f32,
        vp_top: f32,
    ) -> f32 {
        // Positive velocity = scrolling content up/left (viewport moving down/right in content)
        // So if tile is below viewport and velocity.y > 0, tile is getting closer
        let dx = tile_left - vp_left;
        let dy = tile_top - vp_top;

        // Dot product of direction to tile with scroll velocity
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 0.001 {
            return 0.0;
        }

        let dir_x = dx / dist;
        let dir_y = dy / dist;

        // Scroll velocity in content space: positive = content moving up/left
        // For viewport, this means viewport moving down/right in content space
        (dir_x * velocity.x + dir_y * velocity.y).max(0.0)
    }

    /// Get tiles that need rasterization, sorted by priority.
    pub fn tiles_needing_rasterization(&self) -> Vec<&ManagedTile> {
        let mut tiles: Vec<_> = self
            .tiles
            .values()
            .filter(|t| matches!(t.state, TileState::Dirty | TileState::Missing))
            .collect();

        // Sort by priority (lower score = higher priority)
        tiles.sort_by(|a, b| a.priority.cmp(&b.priority));
        tiles
    }

    /// Mark a tile as pending rasterization.
    pub fn start_rasterization(&mut self, key: &TileKey) -> Option<u64> {
        let task_id = self.next_task_id;
        self.next_task_id += 1;

        if let Some(tile) = self.tiles.get_mut(key) {
            tile.mark_pending(task_id);
            Some(task_id)
        } else {
            None
        }
    }

    /// Mark a tile as ready after rasterization completes.
    pub fn complete_rasterization(
        &mut self,
        key: &TileKey,
        texture_id: TextureId,
        generation: u64,
    ) {
        if let Some(tile) = self.tiles.get_mut(key) {
            tile.mark_ready(texture_id, generation);
            self.memory_budget.allocate();
        }
    }

    /// Invalidate all tiles for a layer (content changed).
    pub fn invalidate_layer(&mut self, layer_id: LayerId) {
        if let Some(tile_keys) = self.tiles_by_layer.get(&layer_id) {
            for key in tile_keys.iter() {
                if let Some(tile) = self.tiles.get_mut(key) {
                    tile.mark_dirty();
                }
            }
        }
    }

    /// Evict tiles to stay within memory budget.
    /// Evicts lowest priority tiles first.
    pub fn evict_if_needed(&mut self) -> Vec<TextureId> {
        if !self.memory_budget.should_evict() {
            return Vec::new();
        }

        // Collect tiles with their priorities (only Ready tiles can be evicted)
        let mut eviction_candidates: Vec<_> = self
            .tiles
            .iter()
            .filter(|(_, t)| t.state == TileState::Ready)
            .map(|(k, t)| (k.clone(), t.priority.priority_score(), t.last_used))
            .collect();

        // Sort by priority score descending, then least recently used.
        eviction_candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut evicted_textures = Vec::new();

        // Evict until under soft limit
        while self.memory_budget.should_evict() {
            if let Some((key, _, _)) = eviction_candidates.pop() {
                if let Some(tile) = self.tiles.get_mut(&key) {
                    if let Some(texture_id) = tile.texture_id.take() {
                        evicted_textures.push(texture_id);
                        self.memory_budget.deallocate();
                    }
                    tile.state = TileState::Dirty;
                }
            } else {
                break; // No more candidates
            }
        }

        evicted_textures
    }

    /// Clear all tiles for a layer (layer removed).
    pub fn clear_layer(&mut self, layer_id: LayerId) -> Vec<TextureId> {
        let mut freed_textures = Vec::new();

        if let Some(tile_keys) = self.tiles_by_layer.remove(&layer_id) {
            for key in tile_keys {
                if let Some(tile) = self.tiles.remove(&key) {
                    if let Some(id) = tile.texture_id {
                        freed_textures.push(id);
                        self.memory_budget.deallocate();
                    }
                    if let Some(id) = tile.low_res_texture_id {
                        freed_textures.push(id);
                        self.memory_budget.deallocate();
                    }
                    if let Some(id) = tile.stale_texture_id {
                        freed_textures.push(id);
                        self.memory_budget.deallocate();
                    }
                }
            }
        }

        freed_textures
    }

    /// Get memory budget reference.
    pub fn memory_budget(&self) -> &MemoryBudget {
        &self.memory_budget
    }

    /// Get statistics about the tile manager state.
    pub fn stats(&self) -> TileManagerStats {
        let mut ready_count = 0;
        let mut dirty_count = 0;
        let mut pending_count = 0;
        let mut missing_count = 0;

        for tile in self.tiles.values() {
            match tile.state {
                TileState::Ready => ready_count += 1,
                TileState::Dirty => dirty_count += 1,
                TileState::Pending { .. } => pending_count += 1,
                TileState::Missing => missing_count += 1,
            }
        }

        TileManagerStats {
            total_tiles: self.tiles.len(),
            ready_count,
            dirty_count,
            pending_count,
            missing_count,
            memory_used_bytes: self.memory_budget.used_bytes(),
            memory_budget_bytes: self.memory_budget.total_bytes,
        }
    }
}

impl Default for TileManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about TileManager state.
#[derive(Clone, Debug, Default)]
pub(crate) struct TileManagerStats {
    /// Total number of managed tiles.
    pub total_tiles: usize,
    /// Number of tiles ready to render.
    pub ready_count: usize,
    /// Number of tiles needing rasterization.
    pub dirty_count: usize,
    /// Number of tiles currently being rasterized.
    pub pending_count: usize,
    /// Number of tiles with no texture.
    pub missing_count: usize,
    /// Current memory usage in bytes.
    pub memory_used_bytes: usize,
    /// Memory budget limit in bytes.
    pub memory_budget_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::TileCoord;
    use crate::GlobalElementId;

    #[test]
    fn test_tile_priority_ordering() {
        let visible = TilePriority::VISIBLE;
        let nearby = TilePriority {
            distance_to_visible: 100.0,
            time_to_visible: 0.5,
            resolution: TileResolution::High,
            required_for_activation: false,
        };
        let far = TilePriority {
            distance_to_visible: 1000.0,
            time_to_visible: 5.0,
            resolution: TileResolution::High,
            required_for_activation: false,
        };

        assert!(visible < nearby); // Visible has higher priority (lower score)
        assert!(nearby < far);
    }

    #[test]
    fn test_managed_tile_fallback() {
        let key = TileKey {
            container_id: GlobalElementId::root(),
            coord: TileCoord { x: 0, y: 0 },
        };
        let mut tile = ManagedTile::new(key, LayerId(1));

        // Initially no texture - should show checkerboard
        assert!(tile.best_texture().is_none());
        assert!(tile.fallback().is_checkerboard());

        // Add high-res
        let high_res = TextureId::new();
        tile.mark_ready(high_res, 1);
        assert_eq!(tile.best_texture(), Some(high_res));
        assert_eq!(tile.fallback(), TileFallback::HighRes(high_res));
        assert!(!tile.fallback().is_degraded());

        // Mark dirty - high-res moves to stale
        tile.mark_dirty();
        assert_eq!(tile.best_texture(), Some(high_res)); // Stale fallback
        assert_eq!(tile.stale_texture_id, Some(high_res));
        assert_eq!(tile.fallback(), TileFallback::Stale(high_res));
        assert!(tile.fallback().is_degraded());

        // Add low-res texture - should prefer low-res over stale
        let low_res = TextureId::new();
        tile.low_res_texture_id = Some(low_res);
        assert_eq!(tile.fallback(), TileFallback::LowRes(low_res));
        assert!(tile.fallback().is_degraded());
    }

    #[test]
    fn test_memory_budget() {
        let budget = MemoryBudget::new(10 * 1024 * 1024, 512, 4); // 10MB

        assert!(!budget.should_evict());
        assert!(budget.can_allocate());

        // Simulate allocations
        for _ in 0..5 {
            budget.allocate();
        }

        // Check usage
        assert!(budget.used_bytes() > 0);
        assert!(budget.usage_percent() > 0.0);
    }
}
