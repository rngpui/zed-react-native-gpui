//! Threaded Tile Rasterization
//!
//! This module provides parallel rasterization of DisplayList tiles using rayon.
//! The CPU-bound work (converting DisplayItems to Scene primitives) is parallelized,
//! while GPU work (Metal rendering) remains on the main thread.
//!
//! ## Architecture
//!
//! ```text
//! Main Thread:    collect tiles → [submit to rayon] → [wait] → GPU render each tile
//!                                         ↓
//! Worker Threads:          rasterize_tile() in parallel
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! // Collect tiles that need rasterization
//! let jobs: Vec<TileRasterJob> = ...;
//!
//! // Parallel CPU rasterization
//! let results = TileRasterizer::rasterize_parallel(jobs, scale_factor);
//!
//! // Sequential GPU rendering (must be on main thread)
//! for result in results {
//!     renderer.rasterize_display_list_tile(&result.container_id, result.coord, &result.raster_result);
//! }
//! ```

use crate::display_list::{DisplayList, TileRasterResult};
use crate::property_trees::PropertyTrees;
use crate::scene::TileCoord;
use crate::{Bounds, GlobalElementId, Pixels, Point, Size};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

use super::tile_cache::TileCache;

/// A tile rasterization job submitted to the worker pool.
#[derive(Clone)]
pub struct TileRasterJob {
    /// The scroll container this tile belongs to.
    pub container_id: GlobalElementId,

    /// The tile coordinate within the container.
    pub coord: TileCoord,

    /// The display list to rasterize (Arc for cheap cloning across tiles).
    pub display_list: Arc<DisplayList>,

    /// Property trees for transform/clip resolution (cloned for thread safety).
    /// PropertyTrees needs to be owned because rasterize_tile takes &mut for caching.
    pub property_trees: PropertyTrees,

    /// Priority for scheduling (higher = more important).
    /// Visible tiles should have higher priority.
    pub priority: TilePriority,
}

/// Priority value for tile rasterization scheduling.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TilePriority(pub f32);

impl TilePriority {
    /// Highest priority: tile is currently visible in viewport.
    pub const VISIBLE: TilePriority = TilePriority(1000.0);

    /// High priority: tile is adjacent to visible area.
    pub const ADJACENT: TilePriority = TilePriority(500.0);

    /// Medium priority: tile is in scroll direction (predictive).
    pub const PREDICTIVE: TilePriority = TilePriority(250.0);

    /// Low priority: tile is outside viewport but cached.
    pub const BACKGROUND: TilePriority = TilePriority(100.0);

    /// Compute priority for a tile based on viewport and scroll velocity.
    ///
    /// Higher priority for:
    /// 1. Tiles currently visible in viewport
    /// 2. Tiles adjacent to visible area
    /// 3. Tiles in the scroll direction (predictive prefetch)
    pub fn compute(
        coord: TileCoord,
        viewport: Bounds<Pixels>,
        scroll_offset: Point<Pixels>,
        scroll_velocity: Point<Pixels>,
        scale_factor: f32,
    ) -> TilePriority {
        let tile_bounds = TileCache::tile_content_bounds(coord, scale_factor);

        // Convert viewport to content space (add scroll offset)
        let content_viewport = Bounds {
            origin: Point {
                x: viewport.origin.x - scroll_offset.x,
                y: viewport.origin.y - scroll_offset.y,
            },
            size: viewport.size,
        };

        // Check if tile is visible
        if tile_bounds.intersects(&content_viewport) {
            return TilePriority::VISIBLE;
        }

        // Check if tile is adjacent (within one tile size of viewport)
        let tile_size = super::tile_cache::TILE_SIZE as f32 / scale_factor;
        let expanded_viewport = Bounds {
            origin: Point {
                x: content_viewport.origin.x - Pixels(tile_size),
                y: content_viewport.origin.y - Pixels(tile_size),
            },
            size: Size {
                width: content_viewport.size.width + Pixels(tile_size * 2.0),
                height: content_viewport.size.height + Pixels(tile_size * 2.0),
            },
        };

        if tile_bounds.intersects(&expanded_viewport) {
            // Check if tile is in scroll direction (predictive)
            let tile_center = Point {
                x: tile_bounds.origin.x + tile_bounds.size.width * 0.5,
                y: tile_bounds.origin.y + tile_bounds.size.height * 0.5,
            };
            let viewport_center = Point {
                x: content_viewport.origin.x + content_viewport.size.width * 0.5,
                y: content_viewport.origin.y + content_viewport.size.height * 0.5,
            };

            // Direction from viewport to tile
            let to_tile = Point {
                x: tile_center.x - viewport_center.x,
                y: tile_center.y - viewport_center.y,
            };

            // Check if scroll velocity points toward this tile
            let dot = to_tile.x.0 * scroll_velocity.x.0 + to_tile.y.0 * scroll_velocity.y.0;
            if dot > 0.0 {
                return TilePriority::PREDICTIVE;
            }

            return TilePriority::ADJACENT;
        }

        TilePriority::BACKGROUND
    }
}

impl Eq for TilePriority {}

impl PartialOrd for TilePriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TilePriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority should come first
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal).reverse()
    }
}

/// Result of rasterizing a tile on a worker thread.
pub struct TileRasterJobResult {
    /// The scroll container this tile belongs to.
    pub container_id: GlobalElementId,

    /// The tile coordinate.
    pub coord: TileCoord,

    /// The rasterized primitives ready for GPU rendering.
    pub raster_result: TileRasterResult,
}

/// Statistics for tile rasterization performance.
#[derive(Default, Clone, Copy, Debug)]
pub struct TileRasterStats {
    /// Number of tiles rasterized this frame.
    pub tiles_rasterized: u32,

    /// Total time spent on CPU rasterization (microseconds).
    pub cpu_time_us: u64,

    /// Number of tiles skipped due to cache hit.
    pub cache_hits: u32,
}

/// Tile rasterizer providing parallel CPU rasterization.
pub struct TileRasterizer {
    /// Statistics for the current frame.
    stats: TileRasterStats,
}

impl TileRasterizer {
    /// Create a new tile rasterizer.
    pub fn new() -> Self {
        Self {
            stats: TileRasterStats::default(),
        }
    }

    /// Reset statistics for a new frame.
    pub fn begin_frame(&mut self) {
        self.stats = TileRasterStats::default();
    }

    /// Get current frame statistics.
    pub fn stats(&self) -> TileRasterStats {
        self.stats
    }

    /// Rasterize tiles in parallel using rayon.
    ///
    /// This performs the CPU-bound work (converting DisplayItems to Scene primitives)
    /// in parallel across worker threads. The results can then be rendered to GPU
    /// sequentially on the main thread.
    ///
    /// Jobs are sorted by priority before processing, so higher-priority tiles
    /// are more likely to complete first.
    pub fn rasterize_parallel(
        mut jobs: Vec<TileRasterJob>,
        scale_factor: f32,
    ) -> Vec<TileRasterJobResult> {
        if jobs.is_empty() {
            return Vec::new();
        }

        // Sort by priority (highest first) for better responsiveness
        jobs.sort_by(|a, b| a.priority.cmp(&b.priority));

        // Parallel rasterization
        jobs.into_par_iter()
            .map(|job| {
                let tile_bounds = TileCache::tile_content_bounds(job.coord, scale_factor);

                // Clone property_trees since rasterize_tile needs &mut
                let mut property_trees = job.property_trees;
                let raster_result =
                    job.display_list
                        .rasterize_tile(tile_bounds, scale_factor, &mut property_trees);

                TileRasterJobResult {
                    container_id: job.container_id,
                    coord: job.coord,
                    raster_result,
                }
            })
            .collect()
    }

    /// Rasterize tiles in parallel with statistics tracking.
    pub fn rasterize_parallel_with_stats(
        &mut self,
        jobs: Vec<TileRasterJob>,
        scale_factor: f32,
    ) -> Vec<TileRasterJobResult> {
        let start = std::time::Instant::now();
        let num_tiles = jobs.len() as u32;

        let results = Self::rasterize_parallel(jobs, scale_factor);

        self.stats.tiles_rasterized += num_tiles;
        self.stats.cpu_time_us += start.elapsed().as_micros() as u64;

        results
    }
}

impl Default for TileRasterizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_priority_ordering() {
        let visible = TilePriority::VISIBLE;
        let adjacent = TilePriority::ADJACENT;
        let predictive = TilePriority::PREDICTIVE;
        let background = TilePriority::BACKGROUND;

        // Higher priority should sort first (be "less than" in Ord)
        assert!(visible < adjacent);
        assert!(adjacent < predictive);
        assert!(predictive < background);
    }

    #[test]
    fn test_priority_compute_visible() {
        let coord = TileCoord { x: 0, y: 0 };
        let viewport = Bounds {
            origin: Point {
                x: Pixels(0.0),
                y: Pixels(0.0),
            },
            size: Size {
                width: Pixels(1000.0),
                height: Pixels(800.0),
            },
        };
        let scroll_offset = Point::default();
        let scroll_velocity = Point::default();
        let scale_factor = 1.0;

        let priority =
            TilePriority::compute(coord, viewport, scroll_offset, scroll_velocity, scale_factor);

        assert_eq!(priority, TilePriority::VISIBLE);
    }

    #[test]
    fn test_priority_compute_adjacent() {
        // Tile at (2, 0) is just outside a small viewport
        let coord = TileCoord { x: 2, y: 0 };
        let viewport = Bounds {
            origin: Point {
                x: Pixels(0.0),
                y: Pixels(0.0),
            },
            size: Size {
                width: Pixels(600.0),
                height: Pixels(400.0),
            },
        };
        let scroll_offset = Point::default();
        let scroll_velocity = Point::default();
        let scale_factor = 1.0;

        let priority =
            TilePriority::compute(coord, viewport, scroll_offset, scroll_velocity, scale_factor);

        // Should be ADJACENT (within one tile of viewport)
        assert_eq!(priority, TilePriority::ADJACENT);
    }
}
