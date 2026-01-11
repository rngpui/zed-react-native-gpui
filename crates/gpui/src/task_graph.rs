//! Task Graph for Dependency-Aware Scheduling
//!
//! This module provides a task graph system similar to Chromium's TaskGraphRunner.
//! It handles dependencies between tasks, such as tiles depending on image decodes.
//!
//! ## Key Concepts
//!
//! - **Task**: A unit of work (image decode, tile raster)
//! - **Dependency**: Task A depends on Task B means B must complete before A starts
//! - **Ready Queue**: Tasks with all dependencies satisfied, sorted by priority
//!
//! ## Example
//!
//! ```text
//! Image Decode Task (dependency)
//!        ↓
//! Tile Raster Task (depends on decode)
//! ```

use crate::display_list::DisplayList;
use crate::scene::TileKey;
use crate::{Bounds, ScaledPixels};
use collections::{FxHashMap, FxHashSet};
use std::collections::BinaryHeap;
use std::sync::Arc;

/// Unique identifier for a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

/// Unique identifier for an image to be decoded.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RasterImageId(pub u64);

/// A task in the dependency graph.
#[derive(Clone)]
pub enum RasterTask {
    /// Decode an image from encoded bytes.
    ImageDecode {
        /// The ID of the image to decode.
        image_id: RasterImageId,
        /// The encoded image data.
        encoded_data: Arc<[u8]>,
    },

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
}

impl RasterTask {
    /// Get the priority for this task.
    pub fn priority(&self) -> f32 {
        match self {
            // Image decodes have higher priority (lower score) since other tasks depend on them
            RasterTask::ImageDecode { .. } => 0.0,
            // Tile raster priority comes from tile manager
            RasterTask::TileRaster { .. } => 100.0,
        }
    }
}

/// A task with its dependencies and metadata.
struct TaskNode {
    /// The task to execute.
    task: RasterTask,

    /// Task priority (lower = higher priority).
    priority: f32,

    /// Number of dependencies not yet satisfied.
    unsatisfied_dependencies: usize,

    /// Tasks that depend on this task.
    dependents: Vec<TaskId>,
}

/// A prioritized task for the ready queue.
struct PrioritizedTask {
    id: TaskId,
    priority: f32,
}

impl PartialEq for PrioritizedTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedTask {}

impl PartialOrd for PrioritizedTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower priority score = higher priority in queue (reverse order)
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Task graph for dependency-aware scheduling.
///
/// Manages task dependencies and provides tasks in dependency-respecting order.
/// Tasks are only made available when all their dependencies are complete.
pub struct TaskGraph {
    /// All tasks indexed by ID.
    tasks: FxHashMap<TaskId, TaskNode>,

    /// Next task ID.
    next_id: u64,

    /// Ready queue (tasks with all dependencies satisfied).
    ready_queue: BinaryHeap<PrioritizedTask>,

    /// Completed task IDs.
    completed: FxHashSet<TaskId>,

    /// Image decode tasks indexed by image ID (for dependency lookup).
    image_decode_tasks: FxHashMap<RasterImageId, TaskId>,
}

impl TaskGraph {
    /// Create a new empty task graph.
    pub fn new() -> Self {
        Self {
            tasks: FxHashMap::default(),
            next_id: 1,
            ready_queue: BinaryHeap::new(),
            completed: FxHashSet::default(),
            image_decode_tasks: FxHashMap::default(),
        }
    }

    /// Clear all tasks and reset the graph.
    pub fn clear(&mut self) {
        self.tasks.clear();
        self.ready_queue.clear();
        self.completed.clear();
        self.image_decode_tasks.clear();
    }

    /// Add a task to the graph without dependencies.
    pub fn add_task(&mut self, task: RasterTask) -> TaskId {
        self.add_task_with_dependencies(task, &[])
    }

    /// Add a task with dependencies.
    ///
    /// The task won't be ready until all dependencies are complete.
    pub fn add_task_with_dependencies(&mut self, task: RasterTask, dependencies: &[TaskId]) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;

        let priority = task.priority();

        // Track image decode tasks for lookup
        if let RasterTask::ImageDecode { ref image_id, .. } = task {
            self.image_decode_tasks.insert(image_id.clone(), id);
        }

        // Count unsatisfied dependencies (exclude completed ones)
        let unsatisfied = dependencies
            .iter()
            .filter(|dep_id| !self.completed.contains(dep_id))
            .count();

        // Register this task as a dependent of its dependencies
        for dep_id in dependencies {
            if let Some(dep_node) = self.tasks.get_mut(dep_id) {
                dep_node.dependents.push(id);
            }
        }

        let node = TaskNode {
            task,
            priority,
            unsatisfied_dependencies: unsatisfied,
            dependents: Vec::new(),
        };

        self.tasks.insert(id, node);

        // If no unsatisfied dependencies, add to ready queue
        if unsatisfied == 0 {
            self.ready_queue.push(PrioritizedTask { id, priority });
        }

        id
    }

    /// Get or create an image decode task for an image.
    ///
    /// Returns the existing task ID if already queued, or creates a new one.
    pub fn ensure_image_decode_task(&mut self, image_id: RasterImageId, encoded_data: Arc<[u8]>) -> TaskId {
        if let Some(&task_id) = self.image_decode_tasks.get(&image_id) {
            return task_id;
        }

        self.add_task(RasterTask::ImageDecode {
            image_id,
            encoded_data,
        })
    }

    /// Add a tile raster task with image dependencies.
    pub fn add_tile_raster_task(
        &mut self,
        tile_key: TileKey,
        display_list: Arc<DisplayList>,
        tile_bounds: Bounds<ScaledPixels>,
        scale_factor: f32,
        image_dependencies: &[RasterImageId],
    ) -> TaskId {
        // Get task IDs for image dependencies
        let dep_ids: Vec<TaskId> = image_dependencies
            .iter()
            .filter_map(|img_id| self.image_decode_tasks.get(img_id).copied())
            .collect();

        self.add_task_with_dependencies(
            RasterTask::TileRaster {
                tile_key,
                display_list,
                tile_bounds,
                scale_factor,
            },
            &dep_ids,
        )
    }

    /// Pop the highest-priority ready task.
    pub fn pop_ready(&mut self) -> Option<(TaskId, RasterTask)> {
        while let Some(prioritized) = self.ready_queue.pop() {
            // Skip if already completed (shouldn't happen, but be safe)
            if self.completed.contains(&prioritized.id) {
                continue;
            }

            if let Some(node) = self.tasks.get(&prioritized.id) {
                return Some((prioritized.id, node.task.clone()));
            }
        }
        None
    }

    /// Peek at the highest-priority ready task without removing it.
    pub fn peek_ready(&self) -> Option<TaskId> {
        self.ready_queue.peek().map(|p| p.id)
    }

    /// Check if there are any ready tasks.
    pub fn has_ready_tasks(&self) -> bool {
        !self.ready_queue.is_empty()
    }

    /// Mark a task as complete and update dependents.
    pub fn complete_task(&mut self, task_id: TaskId) {
        if self.completed.contains(&task_id) {
            return;
        }

        self.completed.insert(task_id);

        // Get the dependents before removing the task
        let dependents = if let Some(node) = self.tasks.get(&task_id) {
            node.dependents.clone()
        } else {
            return;
        };

        // Update each dependent
        for dependent_id in dependents {
            if let Some(dep_node) = self.tasks.get_mut(&dependent_id) {
                if dep_node.unsatisfied_dependencies > 0 {
                    dep_node.unsatisfied_dependencies -= 1;

                    // If all dependencies satisfied, add to ready queue
                    if dep_node.unsatisfied_dependencies == 0 {
                        self.ready_queue.push(PrioritizedTask {
                            id: dependent_id,
                            priority: dep_node.priority,
                        });
                    }
                }
            }
        }
    }

    /// Get the number of pending tasks (not yet complete).
    pub fn pending_count(&self) -> usize {
        self.tasks.len() - self.completed.len()
    }

    /// Get the number of ready tasks.
    pub fn ready_count(&self) -> usize {
        self.ready_queue.len()
    }

    /// Check if all tasks are complete.
    pub fn is_complete(&self) -> bool {
        self.completed.len() == self.tasks.len()
    }

    /// Get statistics about the graph.
    pub fn stats(&self) -> TaskGraphStats {
        TaskGraphStats {
            total_tasks: self.tasks.len(),
            completed_tasks: self.completed.len(),
            ready_tasks: self.ready_queue.len(),
            pending_tasks: self.tasks.len() - self.completed.len() - self.ready_queue.len(),
        }
    }
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about TaskGraph state.
#[derive(Clone, Debug, Default)]
pub struct TaskGraphStats {
    /// Total number of tasks in the graph.
    pub total_tasks: usize,
    /// Number of completed tasks.
    pub completed_tasks: usize,
    /// Number of tasks ready to execute.
    pub ready_tasks: usize,
    /// Number of tasks waiting on dependencies.
    pub pending_tasks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::TileCoord;
    use crate::GlobalElementId;

    #[test]
    fn test_task_graph_no_dependencies() {
        let mut graph = TaskGraph::new();

        let id = graph.add_task(RasterTask::ImageDecode {
            image_id: RasterImageId(1),
            encoded_data: Arc::new([]),
        });

        // Should be immediately ready
        assert!(graph.has_ready_tasks());

        let (popped_id, _) = graph.pop_ready().unwrap();
        assert_eq!(popped_id, id);

        graph.complete_task(id);
        assert!(graph.is_complete());
    }

    #[test]
    fn test_task_graph_with_dependencies() {
        let mut graph = TaskGraph::new();

        // Add image decode task
        let decode_id = graph.add_task(RasterTask::ImageDecode {
            image_id: RasterImageId(1),
            encoded_data: Arc::new([]),
        });

        // Add tile raster that depends on decode
        let raster_id = graph.add_task_with_dependencies(
            RasterTask::TileRaster {
                tile_key: TileKey {
                    container_id: GlobalElementId::root(),
                    coord: TileCoord { x: 0, y: 0 },
                },
                display_list: Arc::new(DisplayList::new(GlobalElementId::root())),
                tile_bounds: Bounds::default(),
                scale_factor: 1.0,
            },
            &[decode_id],
        );

        // Only decode should be ready
        let (id, _) = graph.pop_ready().unwrap();
        assert_eq!(id, decode_id);
        assert!(!graph.has_ready_tasks()); // Raster not ready yet

        // Complete decode
        graph.complete_task(decode_id);

        // Now raster should be ready
        assert!(graph.has_ready_tasks());
        let (id, _) = graph.pop_ready().unwrap();
        assert_eq!(id, raster_id);

        graph.complete_task(raster_id);
        assert!(graph.is_complete());
    }

    #[test]
    fn test_task_graph_priority() {
        let mut graph = TaskGraph::new();

        // Image decode has priority 0 (higher)
        let decode_id = graph.add_task(RasterTask::ImageDecode {
            image_id: RasterImageId(1),
            encoded_data: Arc::new([]),
        });

        // Tile raster has priority 100 (lower)
        let raster_id = graph.add_task(RasterTask::TileRaster {
            tile_key: TileKey {
                container_id: GlobalElementId::root(),
                coord: TileCoord { x: 0, y: 0 },
            },
            display_list: Arc::new(DisplayList::new(GlobalElementId::root())),
            tile_bounds: Bounds::default(),
            scale_factor: 1.0,
        });

        // Decode should come first (higher priority)
        let (first_id, _) = graph.pop_ready().unwrap();
        assert_eq!(first_id, decode_id);

        graph.complete_task(decode_id);

        let (second_id, _) = graph.pop_ready().unwrap();
        assert_eq!(second_id, raster_id);
    }
}
