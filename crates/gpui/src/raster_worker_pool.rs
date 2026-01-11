//! Dedicated Raster Worker Pool
//!
//! This module provides a dedicated worker pool for tile rasterization,
//! similar to Chromium's RasterTaskWorkerPool. Unlike the general rayon pool,
//! this provides:
//!
//! - Job tracking with in-flight count
//! - Non-blocking result collection
//! - Maximum concurrent jobs limit
//! - Integration with TaskGraph for dependency-aware scheduling
//!
//! ## Architecture
//!
//! ```text
//! Main Thread                     Worker Pool
//! ───────────                     ───────────
//! dispatch_ready() ──────────────→ work_rx (workers grab jobs)
//!                                        ↓
//! collect_completed() ←────────── result_tx (workers send results)
//! ```

use crate::display_list::{DisplayList, TileRasterResult};
use crate::scene::TileKey;
use crate::task_graph::{RasterTask, TaskGraph, TaskId};
use crate::{Bounds, Pixels, ScaledPixels};
use flume::{Receiver, Sender, TryRecvError};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Work item sent to worker threads.
pub struct RasterWork {
    /// Task ID for tracking completion.
    pub task_id: TaskId,

    /// The work to perform.
    pub work_type: RasterWorkType,
}

/// Types of raster work.
pub enum RasterWorkType {
    /// Rasterize a tile from a display list.
    TileRaster {
        /// The tile's key identifying its position.
        tile_key: TileKey,
        /// The display list to rasterize from.
        display_list: Arc<DisplayList>,
        /// The bounds of the tile in scaled pixels.
        tile_bounds: Bounds<ScaledPixels>,
        /// The display scale factor.
        scale_factor: f32,
    },

    /// Decode an image (placeholder for future implementation).
    ImageDecode {
        /// The ID of the image to decode.
        image_id: u64,
        /// The encoded image data.
        encoded_data: Arc<[u8]>,
    },
}

/// Result from a completed raster job.
pub struct RasterResult {
    /// Task ID for matching with TaskGraph.
    pub task_id: TaskId,

    /// The result type.
    pub result_type: RasterResultType,
}

/// Types of raster results.
pub enum RasterResultType {
    /// Tile rasterization completed.
    TileRaster {
        /// The tile's key.
        tile_key: TileKey,
        /// The rasterization result.
        raster_result: TileRasterResult,
    },

    /// Image decode completed (placeholder).
    ImageDecode {
        /// The decoded image ID.
        image_id: u64,
    },
}

/// A worker thread in the pool.
struct RasterWorker {
    /// Thread handle.
    handle: JoinHandle<()>,
}

impl RasterWorker {
    /// Spawn a new worker thread.
    fn spawn(
        id: usize,
        work_rx: Receiver<RasterWork>,
        result_tx: Sender<RasterResult>,
        shutdown: Arc<AtomicBool>,
    ) -> Self {
        let handle = thread::Builder::new()
            .name(format!("raster-worker-{}", id))
            .spawn(move || {
                Self::run(work_rx, result_tx, shutdown);
            })
            .expect("failed to spawn raster worker");

        Self { handle }
    }

    /// Worker thread main loop.
    fn run(
        work_rx: Receiver<RasterWork>,
        result_tx: Sender<RasterResult>,
        shutdown: Arc<AtomicBool>,
    ) {
        loop {
            // Check for shutdown
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Try to receive work with timeout
            match work_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(work) => {
                    let result = Self::process_work(work);
                    // Ignore send errors (main thread may have dropped receiver)
                    let _ = result_tx.send(result);
                }
                Err(flume::RecvTimeoutError::Timeout) => {
                    // No work, loop back to check shutdown
                    continue;
                }
                Err(flume::RecvTimeoutError::Disconnected) => {
                    // Channel closed, exit
                    break;
                }
            }
        }
    }

    /// Process a work item.
    fn process_work(work: RasterWork) -> RasterResult {
        let result_type = match work.work_type {
            RasterWorkType::TileRaster {
                tile_key,
                display_list,
                tile_bounds,
                scale_factor,
            } => {
                // Convert ScaledPixels bounds to Pixels for rasterization
                let pixel_bounds = Bounds {
                    origin: crate::Point {
                        x: Pixels(tile_bounds.origin.x.0 / scale_factor),
                        y: Pixels(tile_bounds.origin.y.0 / scale_factor),
                    },
                    size: crate::Size {
                        width: Pixels(tile_bounds.size.width.0 / scale_factor),
                        height: Pixels(tile_bounds.size.height.0 / scale_factor),
                    },
                };

                // Perform the rasterization (CPU-bound work)
                let raster_result = display_list.rasterize_tile(pixel_bounds, scale_factor);

                RasterResultType::TileRaster {
                    tile_key,
                    raster_result,
                }
            }
            RasterWorkType::ImageDecode { image_id, .. } => {
                // Placeholder for future image decode implementation
                RasterResultType::ImageDecode { image_id }
            }
        };

        RasterResult {
            task_id: work.task_id,
            result_type,
        }
    }
}

/// Configuration for the worker pool.
#[derive(Clone, Debug)]
pub struct RasterWorkerPoolConfig {
    /// Number of worker threads.
    pub num_workers: usize,

    /// Maximum concurrent jobs (queue depth).
    pub max_concurrent: usize,
}

impl Default for RasterWorkerPoolConfig {
    fn default() -> Self {
        // Use available parallelism, but cap at reasonable limit
        let num_workers = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .min(8);

        Self {
            num_workers,
            max_concurrent: num_workers * 2, // Allow some pipelining
        }
    }
}

/// Dedicated worker pool for rasterization.
///
/// Provides controlled parallelism for CPU-bound rasterization work
/// with job tracking and non-blocking result collection.
pub struct RasterWorkerPool {
    /// Worker threads.
    workers: Vec<RasterWorker>,

    /// Work sender.
    work_tx: Sender<RasterWork>,

    /// Result receiver.
    result_rx: Receiver<RasterResult>,

    /// Number of in-flight jobs.
    in_flight: AtomicUsize,

    /// Maximum concurrent jobs.
    max_concurrent: usize,

    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,
}

impl RasterWorkerPool {
    /// Create a new worker pool with default configuration.
    pub fn new() -> Self {
        Self::with_config(RasterWorkerPoolConfig::default())
    }

    /// Create a worker pool with custom configuration.
    pub fn with_config(config: RasterWorkerPoolConfig) -> Self {
        let (work_tx, work_rx) = flume::unbounded();
        let (result_tx, result_rx) = flume::unbounded();
        let shutdown = Arc::new(AtomicBool::new(false));

        let workers: Vec<_> = (0..config.num_workers)
            .map(|i| {
                RasterWorker::spawn(
                    i,
                    work_rx.clone(),
                    result_tx.clone(),
                    Arc::clone(&shutdown),
                )
            })
            .collect();

        Self {
            workers,
            work_tx,
            result_rx,
            in_flight: AtomicUsize::new(0),
            max_concurrent: config.max_concurrent,
            shutdown,
        }
    }

    /// Submit a work item to the pool.
    ///
    /// Returns true if submitted, false if pool is at capacity.
    pub fn submit(&self, work: RasterWork) -> bool {
        if self.in_flight.load(Ordering::Relaxed) >= self.max_concurrent {
            return false;
        }

        if self.work_tx.send(work).is_ok() {
            self.in_flight.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Dispatch ready tasks from a TaskGraph.
    ///
    /// Pops ready tasks from the graph and submits them to workers.
    /// Stops when pool is at capacity or no more ready tasks.
    pub fn dispatch_from_graph(&self, task_graph: &mut TaskGraph) -> usize {
        let mut dispatched = 0;

        while self.in_flight.load(Ordering::Relaxed) < self.max_concurrent {
            if let Some((task_id, task)) = task_graph.pop_ready() {
                let work = match task {
                    RasterTask::TileRaster {
                        tile_key,
                        display_list,
                        tile_bounds,
                        scale_factor,
                    } => RasterWork {
                        task_id,
                        work_type: RasterWorkType::TileRaster {
                            tile_key,
                            display_list,
                            tile_bounds,
                            scale_factor,
                        },
                    },
                    RasterTask::ImageDecode {
                        image_id,
                        encoded_data,
                    } => RasterWork {
                        task_id,
                        work_type: RasterWorkType::ImageDecode {
                            image_id: image_id.0,
                            encoded_data,
                        },
                    },
                };

                if self.submit(work) {
                    dispatched += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        dispatched
    }

    /// Collect completed results (non-blocking).
    ///
    /// Returns all results that are currently available.
    pub fn collect_completed(&self) -> Vec<RasterResult> {
        let mut results = Vec::new();

        loop {
            match self.result_rx.try_recv() {
                Ok(result) => {
                    self.in_flight.fetch_sub(1, Ordering::Relaxed);
                    results.push(result);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        results
    }

    /// Collect completed results and update TaskGraph.
    ///
    /// Marks completed tasks in the graph, which may make dependent tasks ready.
    pub fn collect_and_update_graph(&self, task_graph: &mut TaskGraph) -> Vec<RasterResult> {
        let results = self.collect_completed();

        for result in &results {
            task_graph.complete_task(result.task_id);
        }

        results
    }

    /// Get number of in-flight jobs.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Check if pool is at capacity.
    pub fn is_at_capacity(&self) -> bool {
        self.in_flight.load(Ordering::Relaxed) >= self.max_concurrent
    }

    /// Check if pool is idle (no in-flight jobs).
    pub fn is_idle(&self) -> bool {
        self.in_flight.load(Ordering::Relaxed) == 0
    }

    /// Get the number of worker threads.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Shutdown the worker pool.
    ///
    /// Signals workers to stop and waits for them to finish.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        // Drop the work sender to unblock workers waiting on recv
        // (workers will exit on Disconnected error)

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.handle.join();
        }
    }
}

impl Default for RasterWorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RasterWorkerPool {
    fn drop(&mut self) {
        if !self.workers.is_empty() {
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display_list::DisplayList;
    use crate::scene::TileCoord;
    use crate::GlobalElementId;

    #[test]
    fn test_worker_pool_creation() {
        let pool = RasterWorkerPool::new();
        assert!(pool.num_workers() > 0);
        assert!(pool.is_idle());
    }

    #[test]
    fn test_submit_and_collect() {
        let pool = RasterWorkerPool::new();

        // Use ImageDecode for testing since it's simpler and doesn't require
        // spatial index setup like TileRaster does
        let work = RasterWork {
            task_id: TaskId(1),
            work_type: RasterWorkType::ImageDecode {
                image_id: 42,
                encoded_data: Arc::new([]),
            },
        };

        assert!(pool.submit(work));
        assert_eq!(pool.in_flight_count(), 1);

        // Wait for completion
        std::thread::sleep(std::time::Duration::from_millis(100));

        let results = pool.collect_completed();
        assert_eq!(results.len(), 1);
        assert_eq!(pool.in_flight_count(), 0);

        // Verify the result
        if let RasterResultType::ImageDecode { image_id } = &results[0].result_type {
            assert_eq!(*image_id, 42);
        } else {
            panic!("Expected ImageDecode result");
        }
    }

    #[test]
    fn test_capacity_limit() {
        let config = RasterWorkerPoolConfig {
            num_workers: 2,
            max_concurrent: 2,
        };
        let pool = RasterWorkerPool::with_config(config);

        // Submit up to capacity
        for i in 0..2 {
            let work = RasterWork {
                task_id: TaskId(i as u64),
                work_type: RasterWorkType::ImageDecode {
                    image_id: i as u64,
                    encoded_data: Arc::new([]),
                },
            };
            assert!(pool.submit(work));
        }

        assert!(pool.is_at_capacity());

        // Next submit should fail
        let work = RasterWork {
            task_id: TaskId(99),
            work_type: RasterWorkType::ImageDecode {
                image_id: 99,
                encoded_data: Arc::new([]),
            },
        };
        assert!(!pool.submit(work));
    }
}
