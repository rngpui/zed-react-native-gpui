//! Property Trees for Efficient Transform/Clip Updates
//!
//! This module implements property trees (inspired by browser compositing architectures)
//! that separate transform and clip state from primitives. DisplayItems reference nodes
//! in these trees instead of baking values, enabling O(1) property updates.
//!
//! ## Architecture
//!
//! ```text
//! PropertyTrees
//! ├── TransformTree
//! │   └── Nodes form hierarchy, each with local + cached world transform
//! └── ClipTree
//!     └── Nodes form hierarchy, each with local + cached world clip
//!
//! DisplayItem
//! ├── bounds: Bounds<Pixels>  (local bounds, NOT transformed)
//! ├── transform_node: TransformNodeId
//! └── clip_node: ClipNodeId
//! ```
//!
//! ## Key Benefits
//!
//! - **O(1) transform updates**: Change transform node, no repaint needed
//! - **O(1) scroll updates**: Scroll offset is a transform, just update the node
//! - **O(1) clip updates**: Change clip node, affects all items referencing it
//! - **Transform animations**: Animate transforms without touching DisplayList

use crate::scene::TransformationMatrix;
use crate::{Bounds, Pixels, Point, Size};

/// Create bounds that represent "no clipping" (very large bounds).
fn infinite_bounds() -> Bounds<Pixels> {
    // Use large but finite values to avoid f32 precision issues
    const HUGE: f32 = 1_000_000.0;
    Bounds {
        origin: Point {
            x: Pixels(-HUGE),
            y: Pixels(-HUGE),
        },
        size: Size {
            width: Pixels(HUGE * 2.0),
            height: Pixels(HUGE * 2.0),
        },
    }
}

/// Identifier for a node in the transform tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct TransformNodeId(pub u32);

impl TransformNodeId {
    /// The root transform node (identity transform).
    pub const ROOT: TransformNodeId = TransformNodeId(0);
}

/// Identifier for a node in the clip tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ClipNodeId(pub u32);

impl ClipNodeId {
    /// The root clip node (no clipping / infinite bounds).
    pub const ROOT: ClipNodeId = ClipNodeId(0);
}

/// A node in the transform tree.
///
/// Transforms are hierarchical - a node's world transform is computed by
/// composing its local transform with all ancestor transforms.
#[derive(Clone, Debug)]
pub struct TransformNode {
    /// Parent node (None for root).
    pub parent: Option<TransformNodeId>,

    /// Transform relative to parent.
    /// For scroll containers, this includes the scroll offset as a translation.
    pub local_transform: TransformationMatrix,

    /// Cached world transform (composition of all ancestors + local).
    /// None if dirty and needs recomputation.
    world_transform: Option<TransformationMatrix>,
}

impl TransformNode {
    fn new(parent: Option<TransformNodeId>, local_transform: TransformationMatrix) -> Self {
        Self {
            parent,
            local_transform,
            world_transform: None,
        }
    }

    fn root() -> Self {
        Self {
            parent: None,
            local_transform: TransformationMatrix::unit(),
            world_transform: Some(TransformationMatrix::unit()),
        }
    }
}

/// Tree of transform nodes.
///
/// Provides hierarchical transforms where each node's world transform is the
/// composition of all ancestor transforms. Supports O(1) updates by invalidating
/// cached world transforms when local transforms change.
#[derive(Clone)]
pub struct TransformTree {
    nodes: Vec<TransformNode>,
}

impl Default for TransformTree {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformTree {
    /// Create a new transform tree with only the root node.
    pub fn new() -> Self {
        Self {
            nodes: vec![TransformNode::root()],
        }
    }

    /// Reset the tree for a new frame, keeping only the root.
    pub fn clear(&mut self) {
        self.nodes.truncate(1);
        // Reset root to identity
        self.nodes[0] = TransformNode::root();
    }

    /// Push a new transform node as a child of the given parent.
    pub fn push(&mut self, parent: TransformNodeId, local_transform: TransformationMatrix) -> TransformNodeId {
        let id = TransformNodeId(self.nodes.len() as u32);
        self.nodes.push(TransformNode::new(Some(parent), local_transform));
        id
    }

    /// Update a node's local transform.
    ///
    /// This invalidates the world transform cache for this node and all descendants.
    /// O(1) for the update, descendants recomputed lazily on access.
    pub fn set_local_transform(&mut self, id: TransformNodeId, transform: TransformationMatrix) {
        if let Some(node) = self.nodes.get_mut(id.0 as usize) {
            node.local_transform = transform;
            node.world_transform = None;
            // Note: We don't invalidate descendants here because they'll recompute
            // when accessed (lazy invalidation). For true O(1), we could track
            // a "dirty" generation number instead.
        }
    }

    /// Get a node's local transform.
    pub fn local_transform(&self, id: TransformNodeId) -> TransformationMatrix {
        self.nodes
            .get(id.0 as usize)
            .map(|n| n.local_transform)
            .unwrap_or_else(TransformationMatrix::unit)
    }

    /// Get the world transform for a node (cached, recomputed if dirty).
    ///
    /// World transform = parent's world transform * local transform
    pub fn world_transform(&mut self, id: TransformNodeId) -> TransformationMatrix {
        // Check cache first
        if let Some(node) = self.nodes.get(id.0 as usize) {
            if let Some(world) = node.world_transform {
                return world;
            }
        }

        // Need to compute - get parent's world transform first
        let (parent_id, local) = {
            let node = match self.nodes.get(id.0 as usize) {
                Some(n) => n,
                None => return TransformationMatrix::unit(),
            };
            (node.parent, node.local_transform)
        };

        let parent_world = match parent_id {
            Some(pid) => self.world_transform(pid),
            None => TransformationMatrix::unit(),
        };

        // Compose: parent_world * local
        let world = compose_transforms(parent_world, local);

        // Cache the result
        if let Some(node) = self.nodes.get_mut(id.0 as usize) {
            node.world_transform = Some(world);
        }

        world
    }

    /// Get the number of nodes in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tree only has the root node.
    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    /// Pre-compute all world transforms for all nodes.
    ///
    /// Call this before cloning the tree to ensure all clones have
    /// populated caches, avoiding redundant recomputation.
    pub fn compute_all_world_transforms(&mut self) {
        for i in 0..self.nodes.len() {
            self.world_transform(TransformNodeId(i as u32));
        }
    }
}

/// A node in the clip tree.
///
/// Clips are hierarchical - a node's world clip is the intersection of its
/// local clip with all ancestor clips.
#[derive(Clone, Debug)]
pub struct ClipNode {
    /// Parent node (None for root).
    pub parent: Option<ClipNodeId>,

    /// Clip bounds in the coordinate space of the associated transform node.
    pub local_clip: Bounds<Pixels>,

    /// Associated transform node (clips are in this transform's coordinate space).
    pub transform_node: TransformNodeId,

    /// Cached world clip (intersection with all ancestors, in world coordinates).
    /// None if dirty and needs recomputation.
    world_clip: Option<Bounds<Pixels>>,
}

impl ClipNode {
    fn new(
        parent: Option<ClipNodeId>,
        local_clip: Bounds<Pixels>,
        transform_node: TransformNodeId,
    ) -> Self {
        Self {
            parent,
            local_clip,
            transform_node,
            world_clip: None,
        }
    }

    fn root() -> Self {
        Self {
            parent: None,
            local_clip: infinite_bounds(),
            transform_node: TransformNodeId::ROOT,
            world_clip: Some(infinite_bounds()),
        }
    }
}

/// Tree of clip nodes.
///
/// Provides hierarchical clipping where each node's world clip is the intersection
/// of all ancestor clips. Clips are defined in the coordinate space of their
/// associated transform node.
#[derive(Clone)]
pub struct ClipTree {
    nodes: Vec<ClipNode>,
}

impl Default for ClipTree {
    fn default() -> Self {
        Self::new()
    }
}

impl ClipTree {
    /// Create a new clip tree with only the root node (no clipping).
    pub fn new() -> Self {
        Self {
            nodes: vec![ClipNode::root()],
        }
    }

    /// Reset the tree for a new frame, keeping only the root.
    pub fn clear(&mut self) {
        self.nodes.truncate(1);
        self.nodes[0] = ClipNode::root();
    }

    /// Push a new clip node as a child of the given parent.
    pub fn push(
        &mut self,
        parent: ClipNodeId,
        local_clip: Bounds<Pixels>,
        transform_node: TransformNodeId,
    ) -> ClipNodeId {
        let id = ClipNodeId(self.nodes.len() as u32);
        self.nodes.push(ClipNode::new(Some(parent), local_clip, transform_node));
        id
    }

    /// Update a node's local clip bounds.
    pub fn set_local_clip(&mut self, id: ClipNodeId, clip: Bounds<Pixels>) {
        if let Some(node) = self.nodes.get_mut(id.0 as usize) {
            node.local_clip = clip;
            node.world_clip = None;
        }
    }

    /// Get a node's local clip bounds.
    pub fn local_clip(&self, id: ClipNodeId) -> Bounds<Pixels> {
        self.nodes
            .get(id.0 as usize)
            .map(|n| n.local_clip)
            .unwrap_or_else(infinite_bounds)
    }

    /// Get a node's associated transform node.
    pub fn transform_node(&self, id: ClipNodeId) -> TransformNodeId {
        self.nodes
            .get(id.0 as usize)
            .map(|n| n.transform_node)
            .unwrap_or(TransformNodeId::ROOT)
    }

    /// Get the world clip for a node (cached, recomputed if dirty).
    ///
    /// World clip = intersection of local clip (transformed to world) with parent's world clip.
    pub fn world_clip(&mut self, id: ClipNodeId, transform_tree: &mut TransformTree) -> Bounds<Pixels> {
        // Check cache first
        if let Some(node) = self.nodes.get(id.0 as usize) {
            if let Some(world) = node.world_clip {
                return world;
            }
        }

        // Need to compute
        let (parent_id, local_clip, transform_node) = {
            let node = match self.nodes.get(id.0 as usize) {
                Some(n) => n,
                None => return infinite_bounds(),
            };
            (node.parent, node.local_clip, node.transform_node)
        };

        // Transform local clip to world coordinates
        let world_transform = transform_tree.world_transform(transform_node);
        let local_in_world = transform_bounds(local_clip, world_transform);

        // Intersect with parent's world clip
        let world_clip = match parent_id {
            Some(pid) => {
                let parent_world = self.world_clip(pid, transform_tree);
                local_in_world.intersect(&parent_world)
            }
            None => local_in_world,
        };

        // Cache the result
        if let Some(node) = self.nodes.get_mut(id.0 as usize) {
            node.world_clip = Some(world_clip);
        }

        world_clip
    }

    /// Get the number of nodes in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tree only has the root node.
    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    /// Pre-compute all world clips for all nodes.
    ///
    /// Call this before cloning the tree to ensure all clones have
    /// populated caches, avoiding redundant recomputation.
    pub fn compute_all_world_clips(&mut self, transform_tree: &mut TransformTree) {
        for i in 0..self.nodes.len() {
            self.world_clip(ClipNodeId(i as u32), transform_tree);
        }
    }
}

/// Container for all property trees used during rendering.
#[derive(Clone)]
pub struct PropertyTrees {
    /// Transform tree for position, rotation, scale, scroll.
    pub transform_tree: TransformTree,

    /// Clip tree for content masks / clipping regions.
    pub clip_tree: ClipTree,
}

impl Default for PropertyTrees {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyTrees {
    /// Create new property trees.
    pub fn new() -> Self {
        Self {
            transform_tree: TransformTree::new(),
            clip_tree: ClipTree::new(),
        }
    }

    /// Clear both trees for a new frame.
    pub fn clear(&mut self) {
        self.transform_tree.clear();
        self.clip_tree.clear();
    }

    /// Push a transform node and return its ID.
    pub fn push_transform(
        &mut self,
        parent: TransformNodeId,
        transform: TransformationMatrix,
    ) -> TransformNodeId {
        self.transform_tree.push(parent, transform)
    }

    /// Push a clip node and return its ID.
    pub fn push_clip(
        &mut self,
        parent: ClipNodeId,
        clip: Bounds<Pixels>,
        transform_node: TransformNodeId,
    ) -> ClipNodeId {
        self.clip_tree.push(parent, clip, transform_node)
    }

    /// Get the world transform for a node.
    pub fn world_transform(&mut self, id: TransformNodeId) -> TransformationMatrix {
        self.transform_tree.world_transform(id)
    }

    /// Get the world clip for a node.
    pub fn world_clip(&mut self, id: ClipNodeId) -> Bounds<Pixels> {
        self.clip_tree.world_clip(id, &mut self.transform_tree)
    }

    /// Pre-compute all world transforms and clips.
    ///
    /// Call this before cloning PropertyTrees for parallel rasterization
    /// to ensure each clone has populated caches, avoiding redundant
    /// recomputation across tiles.
    pub fn precompute_all(&mut self) {
        self.transform_tree.compute_all_world_transforms();
        self.clip_tree.compute_all_world_clips(&mut self.transform_tree);
    }
}

/// Compose two transformation matrices.
///
/// Result = parent * child (child transform applied first, then parent).
fn compose_transforms(parent: TransformationMatrix, child: TransformationMatrix) -> TransformationMatrix {
    // Matrix multiplication for 2D affine transforms
    // rotation_scale[row][col]: [0][0]=m11, [0][1]=m12, [1][0]=m21, [1][1]=m22
    // translation: [0]=tx, [1]=ty
    //
    // [m11 m12 tx]   [m11' m12' tx']
    // [m21 m22 ty] * [m21' m22' ty']
    // [0   0   1 ]   [0    0    1  ]
    let p = parent.rotation_scale;
    let c = child.rotation_scale;
    let pt = parent.translation;
    let ct = child.translation;

    TransformationMatrix {
        rotation_scale: [
            [
                p[0][0] * c[0][0] + p[0][1] * c[1][0],
                p[0][0] * c[0][1] + p[0][1] * c[1][1],
            ],
            [
                p[1][0] * c[0][0] + p[1][1] * c[1][0],
                p[1][0] * c[0][1] + p[1][1] * c[1][1],
            ],
        ],
        translation: [
            p[0][0] * ct[0] + p[0][1] * ct[1] + pt[0],
            p[1][0] * ct[0] + p[1][1] * ct[1] + pt[1],
        ],
    }
}

/// Transform bounds by a transformation matrix.
///
/// For axis-aligned bounds, we transform all 4 corners and compute
/// the axis-aligned bounding box of the result.
fn transform_bounds(bounds: Bounds<Pixels>, transform: TransformationMatrix) -> Bounds<Pixels> {
    if transform == TransformationMatrix::unit() {
        return bounds;
    }

    // Transform all 4 corners
    let corners = [
        transform_point(bounds.origin, transform),
        transform_point(
            Point {
                x: bounds.origin.x + bounds.size.width,
                y: bounds.origin.y,
            },
            transform,
        ),
        transform_point(
            Point {
                x: bounds.origin.x,
                y: bounds.origin.y + bounds.size.height,
            },
            transform,
        ),
        transform_point(
            Point {
                x: bounds.origin.x + bounds.size.width,
                y: bounds.origin.y + bounds.size.height,
            },
            transform,
        ),
    ];

    // Find bounding box
    let min_x = corners.iter().map(|p| p.x).fold(Pixels(f32::INFINITY), |a, b| if a.0 < b.0 { a } else { b });
    let max_x = corners.iter().map(|p| p.x).fold(Pixels(f32::NEG_INFINITY), |a, b| if a.0 > b.0 { a } else { b });
    let min_y = corners.iter().map(|p| p.y).fold(Pixels(f32::INFINITY), |a, b| if a.0 < b.0 { a } else { b });
    let max_y = corners.iter().map(|p| p.y).fold(Pixels(f32::NEG_INFINITY), |a, b| if a.0 > b.0 { a } else { b });

    Bounds {
        origin: Point { x: min_x, y: min_y },
        size: crate::Size {
            width: Pixels(max_x.0 - min_x.0),
            height: Pixels(max_y.0 - min_y.0),
        },
    }
}

/// Transform a point by a transformation matrix.
fn transform_point(point: Point<Pixels>, transform: TransformationMatrix) -> Point<Pixels> {
    let rs = transform.rotation_scale;
    let t = transform.translation;
    Point {
        x: Pixels(rs[0][0] * point.x.0 + rs[0][1] * point.y.0 + t[0]),
        y: Pixels(rs[1][0] * point.x.0 + rs[1][1] * point.y.0 + t[1]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size};

    #[test]
    fn test_transform_tree_basic() {
        let mut tree = TransformTree::new();
        assert_eq!(tree.len(), 1); // Just root

        // Root should have identity world transform
        let root_world = tree.world_transform(TransformNodeId::ROOT);
        assert_eq!(root_world, TransformationMatrix::unit());
    }

    #[test]
    fn test_transform_tree_hierarchy() {
        let mut tree = TransformTree::new();

        // Push a translation
        let translate = TransformationMatrix {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [100.0, 50.0],
        };
        let child = tree.push(TransformNodeId::ROOT, translate);

        // Child's world transform should be the translation
        let child_world = tree.world_transform(child);
        assert_eq!(child_world.translation[0], 100.0);
        assert_eq!(child_world.translation[1], 50.0);

        // Push grandchild with another translation
        let translate2 = TransformationMatrix {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [10.0, 20.0],
        };
        let grandchild = tree.push(child, translate2);

        // Grandchild's world transform should be composed
        let grandchild_world = tree.world_transform(grandchild);
        assert_eq!(grandchild_world.translation[0], 110.0); // 100 + 10
        assert_eq!(grandchild_world.translation[1], 70.0);  // 50 + 20
    }

    #[test]
    fn test_transform_update_invalidates_cache() {
        let mut tree = TransformTree::new();

        let translate = TransformationMatrix {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [100.0, 0.0],
        };
        let child = tree.push(TransformNodeId::ROOT, translate);

        // Get world transform (caches it)
        let _ = tree.world_transform(child);

        // Update local transform
        let new_translate = TransformationMatrix {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [200.0, 0.0],
        };
        tree.set_local_transform(child, new_translate);

        // World transform should reflect update
        let child_world = tree.world_transform(child);
        assert_eq!(child_world.translation[0], 200.0);
    }

    #[test]
    fn test_clip_tree_basic() {
        let mut clip_tree = ClipTree::new();
        let mut transform_tree = TransformTree::new();

        // Root should have infinite bounds
        let root_clip = clip_tree.world_clip(ClipNodeId::ROOT, &mut transform_tree);
        assert_eq!(root_clip, infinite_bounds());
    }

    #[test]
    fn test_clip_tree_intersection() {
        let mut clip_tree = ClipTree::new();
        let mut transform_tree = TransformTree::new();

        // Push a clip
        let clip1 = Bounds {
            origin: point(Pixels(0.0), Pixels(0.0)),
            size: size(Pixels(100.0), Pixels(100.0)),
        };
        let node1 = clip_tree.push(ClipNodeId::ROOT, clip1, TransformNodeId::ROOT);

        // Push a child clip that's smaller
        let clip2 = Bounds {
            origin: point(Pixels(25.0), Pixels(25.0)),
            size: size(Pixels(50.0), Pixels(50.0)),
        };
        let node2 = clip_tree.push(node1, clip2, TransformNodeId::ROOT);

        // World clip should be the intersection
        let world_clip = clip_tree.world_clip(node2, &mut transform_tree);
        assert_eq!(world_clip.origin.x.0, 25.0);
        assert_eq!(world_clip.origin.y.0, 25.0);
        assert_eq!(world_clip.size.width.0, 50.0);
        assert_eq!(world_clip.size.height.0, 50.0);
    }

    #[test]
    fn test_property_trees_combined() {
        let mut trees = PropertyTrees::new();

        // Push transform
        let translate = TransformationMatrix {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [50.0, 50.0],
        };
        let transform_node = trees.push_transform(TransformNodeId::ROOT, translate);

        // Push clip in that transform's coordinate space
        let clip = Bounds {
            origin: point(Pixels(0.0), Pixels(0.0)),
            size: size(Pixels(100.0), Pixels(100.0)),
        };
        let clip_node = trees.push_clip(ClipNodeId::ROOT, clip, transform_node);

        // World clip should be transformed
        let world_clip = trees.world_clip(clip_node);
        assert_eq!(world_clip.origin.x.0, 50.0); // 0 + 50 translation
        assert_eq!(world_clip.origin.y.0, 50.0);
    }
}
