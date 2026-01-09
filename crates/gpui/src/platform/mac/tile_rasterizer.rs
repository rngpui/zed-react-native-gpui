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
//! let results = rasterize_tiles_parallel(jobs, scale_factor);
//!
//! // Sequential GPU rendering (must be on main thread)
//! for result in results {
//!     renderer.rasterize_display_list_tile(&result.container_id, result.coord, &result.raster_result);
//! }
//! ```

use crate::display_list::{DisplayList, TileRasterResult};
use crate::property_trees::PropertyTrees;
use crate::scene::TileCoord;
use crate::GlobalElementId;
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

/// Rasterize tiles in parallel using rayon.
///
/// This performs the CPU-bound work (converting DisplayItems to Scene primitives)
/// in parallel across worker threads. The results can then be rendered to GPU
/// sequentially on the main thread.
///
/// Jobs are sorted by priority before processing, so higher-priority tiles
/// are more likely to complete first.
pub fn rasterize_tiles_parallel(
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

