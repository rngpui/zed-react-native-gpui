// Many methods in this module are infrastructure for future use.
// The display list architecture is incrementally being integrated.
#![allow(dead_code)]

//! Display list for deferred rendering.
//!
//! This module implements a browser-like display list architecture that separates
//! the "paint" phase (recording drawing commands) from the "rasterize" phase
//! (rendering to tiles). This enables:
//!
//! - Arbitrarily large scrollable content (tiles are fixed size)
//! - O(changed) paint work (only rebuild display list when content changes)
//! - O(visible) rasterize work (only rasterize visible tiles)
//! - True scroll optimization (scroll changes tile positions, no re-rasterization)
//!
//! ## Architecture
//!
//! ```text
//! Paint Phase: Element → DisplayList
//!     - Records drawing commands, doesn't execute them
//!     - Each element paints once, display list persists across frames
//!
//! Rasterize Phase: DisplayList → Tiles
//!     - Only visible tiles are rasterized
//!     - Tiles cached until display list generation changes
//!     - Can re-rasterize any tile from the display list
//!
//! Composite Phase: Tiles → Screen
//!     - O(visible_tiles) GPU draw calls
//!     - Scroll = different tiles composited, no re-rasterization
//! ```

use crate::{
    AtlasTile, Background, Bounds, ContentMask, Corners, Edges, GlobalElementId, Hsla, Pixels,
    Point, ScaledPixels, scene::TransformationMatrix, scene::BorderStyle, point, size,
    scene::{Quad, Shadow, BackdropBlur, Underline, MonochromeSprite, SubpixelSprite, PolychromeSprite, DrawOrder},
};
use collections::FxHashMap;
use smallvec::SmallVec;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for a display list layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DisplayListId(pub u64);

impl DisplayListId {
    /// Generate the next unique display list ID.
    pub fn next() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// A recorded drawing command, independent of final rasterization.
///
/// DisplayItems work in logical Pixels coordinates. They are scaled to device
/// coordinates during the rasterize phase. This allows display lists to be
/// resolution-independent and reusable across different scale factors.
#[derive(Clone, Debug)]
pub(crate) enum DisplayItem {
    /// A filled rectangle with optional border and rounded corners.
    Quad(DisplayQuad),
    /// A drop shadow behind content.
    Shadow(DisplayShadow),
    /// A blur effect applied to content behind the element.
    BackdropBlur(DisplayBackdropBlur),
    /// An underline or strikethrough.
    Underline(DisplayUnderline),
    /// A monochrome glyph (text).
    MonochromeSprite(DisplayMonochromeSprite),
    /// A subpixel-rendered glyph (text with subpixel antialiasing).
    SubpixelSprite(DisplaySubpixelSprite),
    /// A full-color image or emoji.
    PolychromeSprite(DisplayPolychromeSprite),
    /// A vector path.
    Path(DisplayPath),
}

impl DisplayItem {
    /// Returns the bounds of this display item in logical pixels.
    pub fn bounds(&self) -> Option<Bounds<Pixels>> {
        match self {
            DisplayItem::Quad(q) => Some(q.bounds),
            DisplayItem::Shadow(s) => Some(s.bounds),
            DisplayItem::BackdropBlur(b) => Some(b.bounds),
            DisplayItem::Underline(u) => Some(u.bounds),
            DisplayItem::MonochromeSprite(s) => Some(s.bounds),
            DisplayItem::SubpixelSprite(s) => Some(s.bounds),
            DisplayItem::PolychromeSprite(s) => Some(s.bounds),
            DisplayItem::Path(p) => Some(p.bounds),
        }
    }

    /// Check if this item's bounds intersect with the given bounds.
    pub fn intersects(&self, bounds: &Bounds<Pixels>) -> bool {
        match self.bounds() {
            Some(item_bounds) => item_bounds.intersects(bounds),
            None => true, // Clip operations always apply
        }
    }

    /// Returns the baked world-space transform for this item.
    pub fn world_transform(&self) -> TransformationMatrix {
        match self {
            DisplayItem::Quad(q) => q.world_transform,
            DisplayItem::Shadow(s) => s.world_transform,
            DisplayItem::BackdropBlur(b) => b.world_transform,
            DisplayItem::Underline(u) => u.world_transform,
            DisplayItem::MonochromeSprite(s) => s.world_transform,
            DisplayItem::SubpixelSprite(s) => s.world_transform,
            DisplayItem::PolychromeSprite(s) => s.world_transform,
            DisplayItem::Path(p) => p.world_transform,
        }
    }

    /// Returns the baked world-space clip bounds for this item.
    pub fn world_clip(&self) -> Bounds<Pixels> {
        match self {
            DisplayItem::Quad(q) => q.world_clip,
            DisplayItem::Shadow(s) => s.world_clip,
            DisplayItem::BackdropBlur(b) => b.world_clip,
            DisplayItem::Underline(u) => u.world_clip,
            DisplayItem::MonochromeSprite(s) => s.world_clip,
            DisplayItem::SubpixelSprite(s) => s.world_clip,
            DisplayItem::PolychromeSprite(s) => s.world_clip,
            DisplayItem::Path(p) => p.world_clip,
        }
    }
}

/// A quad (filled rectangle) in the display list.
#[derive(Clone, Debug)]
pub(crate) struct DisplayQuad {
    /// Bounds in logical pixels (local coordinates, not transformed).
    pub bounds: Bounds<Pixels>,
    /// Baked world-space clip bounds (computed at insert time).
    pub world_clip: Bounds<Pixels>,
    /// Baked world-space transform (computed at insert time).
    pub world_transform: TransformationMatrix,
    /// Background fill (solid color or gradient).
    pub background: Background,
    /// Border color.
    pub border_color: Hsla,
    /// Border style (solid, dashed).
    pub border_style: BorderStyle,
    /// Corner radii for rounded rectangles.
    pub corner_radii: Corners<Pixels>,
    /// Border widths on each edge.
    pub border_widths: Edges<Pixels>,
}

/// A shadow in the display list.
#[derive(Clone, Debug)]
pub(crate) struct DisplayShadow {
    /// Bounds in logical pixels (local coordinates, not transformed).
    pub bounds: Bounds<Pixels>,
    /// Baked world-space clip bounds (computed at insert time).
    pub world_clip: Bounds<Pixels>,
    /// Baked world-space transform (computed at insert time).
    pub world_transform: TransformationMatrix,
    /// Shadow color.
    pub color: Hsla,
    /// Blur radius in logical pixels.
    pub blur_radius: Pixels,
    /// Corner radii.
    pub corner_radii: Corners<Pixels>,
}

/// A backdrop blur in the display list.
#[derive(Clone, Debug)]
pub(crate) struct DisplayBackdropBlur {
    /// Bounds in logical pixels (local coordinates, not transformed).
    pub bounds: Bounds<Pixels>,
    /// Baked world-space clip bounds (computed at insert time).
    pub world_clip: Bounds<Pixels>,
    /// Baked world-space transform (computed at insert time).
    pub world_transform: TransformationMatrix,
    /// Blur radius in logical pixels.
    pub blur_radius: Pixels,
    /// Corner radii.
    pub corner_radii: Corners<Pixels>,
    /// Tint color applied over the blur.
    pub tint: Hsla,
}

/// An underline or strikethrough in the display list.
#[derive(Clone, Debug)]
pub(crate) struct DisplayUnderline {
    /// Bounds in logical pixels (local coordinates, not transformed).
    pub bounds: Bounds<Pixels>,
    /// Baked world-space clip bounds (computed at insert time).
    pub world_clip: Bounds<Pixels>,
    /// Baked world-space transform (computed at insert time).
    pub world_transform: TransformationMatrix,
    /// Line color.
    pub color: Hsla,
    /// Line thickness in logical pixels.
    pub thickness: Pixels,
    /// Whether the line is wavy.
    pub wavy: bool,
}

/// A monochrome sprite (glyph) in the display list.
#[derive(Clone, Debug)]
pub(crate) struct DisplayMonochromeSprite {
    /// Bounds in logical pixels (local coordinates, not transformed).
    pub bounds: Bounds<Pixels>,
    /// Baked world-space clip bounds (computed at insert time).
    pub world_clip: Bounds<Pixels>,
    /// Baked world-space transform (computed at insert time).
    pub world_transform: TransformationMatrix,
    /// Glyph color.
    pub color: Hsla,
    /// Atlas tile containing the glyph.
    pub tile: AtlasTile,
}

/// A subpixel sprite (glyph with subpixel AA) in the display list.
#[derive(Clone, Debug)]
pub(crate) struct DisplaySubpixelSprite {
    /// Bounds in logical pixels (local coordinates, not transformed).
    pub bounds: Bounds<Pixels>,
    /// Baked world-space clip bounds (computed at insert time).
    pub world_clip: Bounds<Pixels>,
    /// Baked world-space transform (computed at insert time).
    pub world_transform: TransformationMatrix,
    /// Glyph color.
    pub color: Hsla,
    /// Atlas tile containing the glyph.
    pub tile: AtlasTile,
}

/// A polychrome sprite (color image/emoji) in the display list.
#[derive(Clone, Debug)]
pub(crate) struct DisplayPolychromeSprite {
    /// Bounds in logical pixels (local coordinates, not transformed).
    pub bounds: Bounds<Pixels>,
    /// Baked world-space clip bounds (computed at insert time).
    pub world_clip: Bounds<Pixels>,
    /// Baked world-space transform (computed at insert time).
    pub world_transform: TransformationMatrix,
    /// Whether to render in grayscale.
    pub grayscale: bool,
    /// Opacity (0.0 to 1.0).
    pub opacity: f32,
    /// Corner radii for clipping.
    pub corner_radii: Corners<Pixels>,
    /// Atlas tile containing the image.
    pub tile: AtlasTile,
}

/// A vector path in the display list.
#[derive(Clone, Debug)]
pub(crate) struct DisplayPath {
    /// Bounds in logical pixels (local coordinates, not transformed).
    pub bounds: Bounds<Pixels>,
    /// Baked world-space clip bounds (computed at insert time).
    pub world_clip: Bounds<Pixels>,
    /// Baked world-space transform (computed at insert time).
    pub world_transform: TransformationMatrix,
    /// Path fill color.
    pub color: Background,
    /// Path vertices (in local coordinates).
    pub vertices: Vec<PathVertex>,
}

/// A vertex in a path.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PathVertex {
    /// Position in logical pixels (local coordinates).
    pub xy_position: Point<Pixels>,
    /// Position along the path (0.0 to 1.0).
    pub st_position: Point<f32>,
}

// ============================================================================
// Phase 20: Unified Element Identity
// ============================================================================

/// The local component of an element's identity within its parent.
///
/// For most elements, this is their position among siblings (Index).
/// For elements in dynamic lists that may reorder, an explicit Key
/// provides stable identity across reorders.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum LocalElementId {
    /// Position-based identity (default) - index among siblings.
    /// Stable for static content, but identity shifts if siblings are added/removed.
    Index(u32),
    /// Key-based identity (explicit) - for dynamic lists.
    /// Stable across reorders. Use `.id()` or `.key()` to set.
    Key(crate::ElementId),
}

impl Default for LocalElementId {
    fn default() -> Self {
        Self::Index(0)
    }
}

/// A computed element identifier. Always present for every element.
///
/// This is the unified identity system for GPUI elements, inspired by:
/// - React's reconciliation (position + type + key)
/// - Chromium's display item identity (object pointer + type)
///
/// Every element has a `ComputedElementId`, computed automatically from:
/// - `path_hash`: Hash of the ancestor chain (parent's identity)
/// - `local_id`: Position among siblings, or explicit key if provided
/// - `element_type`: Rust TypeId of the element (for type-based diffing)
///
/// ## When to use explicit keys
///
/// Most elements don't need explicit keys - position-based identity works for:
/// - Static UI (headers, footers, sidebars)
/// - Lists that only append at the end
/// - UI that doesn't reorder
///
/// Use `.id()` or `.key()` when:
/// - List items can be reordered (drag-and-drop, sorting)
/// - List items can be inserted/removed in the middle
/// - You need state to follow a specific item across reorders
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ComputedElementId {
    /// Hash of the path from root to this element's parent.
    /// Combined with local_id to form full identity.
    path_hash: u64,
    /// Position among siblings (Index) or explicit key hash (Key).
    /// Stored as hash for uniform size and fast comparison.
    local_id_hash: u64,
    /// Type of element. Different types at same position = different identity.
    element_type: std::any::TypeId,
}

impl ComputedElementId {
    /// Create a new computed element ID.
    pub fn new(path_hash: u64, local_id: &LocalElementId, element_type: std::any::TypeId) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = collections::FxHasher::default();
        local_id.hash(&mut hasher);
        let local_id_hash = hasher.finish();

        Self {
            path_hash,
            local_id_hash,
            element_type,
        }
    }

    /// Create from a global element ID and type.
    pub fn from_global(global: &GlobalElementId, element_type: std::any::TypeId) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = collections::FxHasher::default();
        global.hash(&mut hasher);
        let path_hash = hasher.finish();

        Self {
            path_hash,
            local_id_hash: 0, // Global IDs encode the full path
            element_type,
        }
    }

    /// Compute the path hash for a child element.
    /// The child's path_hash is derived from parent's full identity.
    pub fn child_path_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = collections::FxHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Get a hash suitable for use as a map key.
    pub fn to_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = collections::FxHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Entry tracking an element's own display items (Phase 20: Per-Element Item Tracking).
///
/// Tracks only the element's OWN items - children are tracked separately with their
/// own `ElementEntry` records. This separation is critical for per-element caching:
/// when a child changes, we can invalidate just that child's items while preserving
/// the parent's cached items.
#[derive(Clone, Debug)]
pub(crate) struct ElementEntry {
    /// Range of items owned directly by this element (NOT including children).
    /// Items in this range are the element's own primitives (background, border, etc.).
    pub own_items: Range<usize>,
    /// Combined bounds of own items only (not including children).
    pub own_bounds: Bounds<Pixels>,
    /// Hash of element inputs (for change detection).
    /// If the same element at the same position has the same hash, we can reuse items.
    pub input_hash: u64,
    /// IDs of child elements painted during this element's paint.
    /// Used for hierarchical invalidation if needed.
    pub children: SmallVec<[ComputedElementId; 8]>,
}

impl ElementEntry {
    /// Check if this element has any own items.
    pub fn has_own_items(&self) -> bool {
        !self.own_items.is_empty()
    }

    /// Get the number of own items.
    pub fn own_item_count(&self) -> usize {
        self.own_items.len()
    }
}

/// Stack entry for tracking element painting state during paint phase.
///
/// When `begin_element_v2()` is called, an entry is pushed to track the element's
/// items as they're painted. `end_element_own_items()` marks where the element's
/// own items end, and `finalize_element()` pops the entry and records the result.
#[derive(Clone, Debug)]
struct ElementPaintStackEntry {
    /// The element's computed identity.
    id: ComputedElementId,
    /// Index in items vec where this element's own items start.
    start_index: usize,
    /// Index where own items end (set by end_element_own_items).
    /// None means own items haven't been marked yet.
    own_items_end: Option<usize>,
    /// Hash of element inputs for change detection.
    input_hash: u64,
    /// Children painted during this element's paint.
    children: SmallVec<[ComputedElementId; 8]>,
    /// If true, this element had a cache hit and its own items are being
    /// copied from the previous frame. New items for this element should
    /// be skipped (but children still paint normally).
    cache_hit: bool,
}

/// Grid-based spatial index for fast item lookup by bounds.
///
/// Divides the content space into fixed-size cells. Each cell stores
/// indices of items that intersect it. Query performance is O(cells_overlapped × items_per_cell)
/// instead of O(total_items).
///
/// Cell size matches tile size (512 device pixels / scale_factor) for optimal
/// query performance during tile rasterization.
#[derive(Clone, Debug)]
pub(crate) struct SpatialIndex {
    /// Grid cells, indexed by (row * cols + col).
    /// Each cell contains indices into the DisplayList's items vec.
    cells: Vec<Vec<usize>>,
    /// Number of columns in the grid.
    cols: usize,
    /// Number of rows in the grid.
    rows: usize,
    /// Size of each cell in logical pixels.
    cell_size: f32,
    /// Origin of the grid (top-left corner of content bounds).
    origin: Point<Pixels>,
    /// Whether the index needs rebuilding (items added/removed since last build).
    dirty: bool,
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self {
            cells: Vec::new(),
            cols: 0,
            rows: 0,
            cell_size: 256.0, // Default cell size in logical pixels
            origin: Point::default(),
            dirty: true,
        }
    }
}

impl SpatialIndex {
    /// Create a new spatial index with the given cell size.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            ..Default::default()
        }
    }

    /// Clear the spatial index.
    pub fn clear(&mut self) {
        self.cells.clear();
        self.cols = 0;
        self.rows = 0;
        self.dirty = true;
    }

    /// Build or rebuild the spatial index from the given items and content bounds.
    pub fn build(&mut self, items: &[DisplayItem], content_bounds: Bounds<Pixels>) {
        // Calculate grid dimensions
        let width = content_bounds.size.width.0.max(1.0);
        let height = content_bounds.size.height.0.max(1.0);

        self.cols = ((width / self.cell_size).ceil() as usize).max(1);
        self.rows = ((height / self.cell_size).ceil() as usize).max(1);
        self.origin = content_bounds.origin;

        // Resize and clear cells
        let total_cells = self.cols * self.rows;
        self.cells.clear();
        self.cells.resize_with(total_cells, Vec::new);

        // Insert each item into overlapping cells
        for (index, item) in items.iter().enumerate() {
            if let Some(bounds) = item.bounds() {
                self.insert_into_cells(index, bounds);
            }
        }

        self.dirty = false;
    }

    /// Insert an item index into all cells it overlaps.
    fn insert_into_cells(&mut self, item_index: usize, bounds: Bounds<Pixels>) {
        let (min_col, min_row, max_col, max_row) = self.cells_for_bounds(bounds);

        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let cell_index = row * self.cols + col;
                if cell_index < self.cells.len() {
                    self.cells[cell_index].push(item_index);
                }
            }
        }
    }

    /// Get the range of cells that a bounds overlaps.
    /// Returns (min_col, min_row, max_col, max_row).
    fn cells_for_bounds(&self, bounds: Bounds<Pixels>) -> (usize, usize, usize, usize) {
        let rel_x = (bounds.origin.x - self.origin.x).0.max(0.0);
        let rel_y = (bounds.origin.y - self.origin.y).0.max(0.0);

        let min_col = (rel_x / self.cell_size) as usize;
        let min_row = (rel_y / self.cell_size) as usize;

        let max_x = rel_x + bounds.size.width.0;
        let max_y = rel_y + bounds.size.height.0;

        let max_col = ((max_x / self.cell_size) as usize).min(self.cols.saturating_sub(1));
        let max_row = ((max_y / self.cell_size) as usize).min(self.rows.saturating_sub(1));

        (min_col, min_row, max_col, max_row)
    }

    /// Query items that potentially intersect the given bounds.
    ///
    /// Returns indices of items that may intersect. Note: may include items that don't
    /// actually intersect (false positives from cell-level granularity), so
    /// callers should do a final intersection check.
    pub fn query(&self, bounds: Bounds<Pixels>) -> Vec<usize> {
        let (min_col, min_row, max_col, max_row) = self.cells_for_bounds(bounds);

        // Collect unique indices from overlapping cells
        let mut seen = collections::FxHashSet::default();
        let mut result = Vec::new();

        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let cell_index = row * self.cols + col;
                if cell_index < self.cells.len() {
                    for &index in &self.cells[cell_index] {
                        if seen.insert(index) {
                            result.push(index);
                        }
                    }
                }
            }
        }

        result
    }

    /// Check if the index needs rebuilding.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the index as needing rebuild.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }
}

/// Iterator over display items that intersect a given bounds.
/// Uses spatial index for efficient lookup.
struct ItemsIntersectingIter<'a> {
    items: &'a [DisplayItem],
    indices: std::vec::IntoIter<usize>,
    query_bounds: Bounds<Pixels>,
}

impl<'a> Iterator for ItemsIntersectingIter<'a> {
    type Item = &'a DisplayItem;

    fn next(&mut self) -> Option<Self::Item> {
        // Iterate through spatial index results, doing final intersection check
        for index in self.indices.by_ref() {
            if let Some(item) = self.items.get(index) {
                if item.intersects(&self.query_bounds) {
                    return Some(item);
                }
            }
        }
        None
    }
}

/// A display list containing recorded drawing commands for a layer.
///
/// Display lists are created per scroll container and persist across frames
/// until their content changes. They can be rasterized to tiles on demand.
///
/// ## Incremental Invalidation
///
/// Instead of clearing and rebuilding the entire display list when content changes,
/// you can use `invalidate_region()` to mark specific areas as dirty. This enables:
/// - Only re-rasterizing tiles that intersect dirty regions
/// - Preserving cached tiles for unchanged content
/// - O(dirty_area) work instead of O(content_area)
///
/// ## Element Item Tracking
///
/// When painting with `begin_element()`/`end_element()`, items are tracked by
/// element ID. This enables partial updates via `update_element()` where only
/// the changed element's items are replaced.
#[derive(Clone, Debug)]
pub(crate) struct DisplayList {
    /// Unique identifier for this display list.
    pub id: DisplayListId,
    /// The element this display list belongs to.
    pub element_id: GlobalElementId,
    /// Generation counter, incremented when content changes.
    pub generation: u64,
    /// Total content bounds (may be larger than viewport).
    pub content_bounds: Bounds<Pixels>,
    /// Recorded display items.
    pub items: Vec<DisplayItem>,
    /// Spatial index for fast item lookup by bounds.
    /// Enables O(cells × items_per_cell) queries instead of O(total_items).
    spatial_index: SpatialIndex,
    /// Dirty regions that need re-rasterization.
    /// Empty means entire display list is valid (or was just cleared).
    dirty_regions: Vec<Bounds<Pixels>>,

    // ========================================================================
    // Phase 20: Per-Element Item Tracking
    // ========================================================================

    /// Per-element entry tracking (Phase 20).
    /// Maps computed element IDs to their entry records, which track:
    /// - Element's own items (not including children)
    /// - Input hash for change detection
    /// - Children IDs for hierarchical operations
    element_entries: FxHashMap<ComputedElementId, ElementEntry>,
    /// Stack for tracking element painting state.
    /// Pushed by begin_element_v2(), popped by finalize_element().
    element_paint_stack: Vec<ElementPaintStackEntry>,
}

impl DisplayList {
    /// Create a new empty display list.
    pub fn new(element_id: GlobalElementId) -> Self {
        Self {
            id: DisplayListId::next(),
            element_id,
            generation: 0,
            content_bounds: Bounds::default(),
            items: Vec::new(),
            spatial_index: SpatialIndex::default(),
            dirty_regions: Vec::new(),
            // Phase 20: Per-element tracking
            element_entries: FxHashMap::default(),
            element_paint_stack: Vec::new(),
        }
    }

    /// Clear the display list and increment the generation.
    ///
    /// This invalidates ALL tiles for this display list.
    /// For partial updates, use `invalidate_region()` instead.
    pub fn clear(&mut self) {
        self.items.clear();
        self.spatial_index.clear();
        self.content_bounds = Bounds::default();
        self.dirty_regions.clear();
        // Phase 20: Clear per-element tracking
        self.element_entries.clear();
        self.element_paint_stack.clear();
        self.generation += 1;
    }

    // ========================================================================
    // Phase 20: Per-Element Item Tracking
    // ========================================================================

    /// Begin painting an element with per-element item tracking.
    ///
    /// Tracks element's OWN items separately from children, enabling per-element caching.
    ///
    /// Flow:
    /// 1. `begin_element(id, input_hash)` - start tracking
    /// 2. Paint element's own content (background, border, etc.)
    /// 3. `end_element_own_items()` - mark end of own items
    /// 4. Paint children (each child uses its own begin/end)
    /// 5. `finalize_element()` - complete and record entry
    ///
    /// The `input_hash` is used for change detection: if the same element at
    /// the same position has the same hash on the next frame, its items can
    /// potentially be reused.
    pub fn begin_element(&mut self, id: ComputedElementId, input_hash: u64, cache_hit: bool) {
        // When a child begins, the parent's own items have ended.
        // Mark the parent's own_items_end if not already set.
        if let Some(parent) = self.element_paint_stack.last_mut() {
            if parent.own_items_end.is_none() {
                parent.own_items_end = Some(self.items.len());
            }
        }

        self.element_paint_stack.push(ElementPaintStackEntry {
            id,
            start_index: self.items.len(),
            own_items_end: None,
            input_hash,
            children: SmallVec::new(),
            cache_hit,
        });
    }

    /// Check if the current element has a cache hit (Phase 20).
    ///
    /// When true, primitives for this element's OWN content should be skipped
    /// (they'll be copied from the previous frame). Children still paint normally.
    pub fn current_element_has_cache_hit(&self) -> bool {
        self.element_paint_stack.last().is_some_and(|e| e.cache_hit)
    }

    /// Mark the end of an element's own items (Phase 20).
    ///
    /// Call this after painting the element's own content (background, border)
    /// but BEFORE painting children. This allows us to distinguish between
    /// the element's own items and children's items.
    ///
    /// If not called before `finalize_element()`, all items from start to
    /// finalize are considered "own items" (for leaf elements with no children).
    pub fn end_element_own_items(&mut self) {
        if let Some(entry) = self.element_paint_stack.last_mut() {
            entry.own_items_end = Some(self.items.len());
        }
    }

    /// Finalize element painting and record the entry (Phase 20).
    ///
    /// Call this after all children have been painted. Returns the bounds
    /// of the element's own items (not including children).
    ///
    /// This method:
    /// 1. Records the element's own item range and bounds
    /// 2. Registers this element as a child of the parent (if any)
    /// 3. Stores the entry for later lookup
    pub fn finalize_element(&mut self) -> Option<Bounds<Pixels>> {
        let entry = self.element_paint_stack.pop()?;

        // If end_element_own_items wasn't called, assume all items are own items
        // (this is the case for leaf elements with no children)
        let own_items_end = entry.own_items_end.unwrap_or(self.items.len());

        // Calculate bounds of own items only
        let own_bounds = self.calculate_bounds_for_range(entry.start_index, own_items_end);

        // Register this element as a child of the parent (if any)
        if let Some(parent) = self.element_paint_stack.last_mut() {
            parent.children.push(entry.id);
        }

        // Store the entry
        self.element_entries.insert(
            entry.id,
            ElementEntry {
                own_items: entry.start_index..own_items_end,
                own_bounds,
                input_hash: entry.input_hash,
                children: entry.children,
            },
        );

        Some(own_bounds)
    }

    /// Calculate bounds for a range of items.
    fn calculate_bounds_for_range(&self, start: usize, end: usize) -> Bounds<Pixels> {
        let mut bounds: Bounds<Pixels> = Bounds::default();
        for item in &self.items[start..end] {
            if let Some(item_bounds) = item.bounds() {
                if bounds.size.width.0 == 0.0 && bounds.size.height.0 == 0.0 {
                    bounds = item_bounds;
                } else {
                    bounds = bounds.union(&item_bounds);
                }
            }
        }
        bounds
    }

    /// Get the element entry for a computed ID (Phase 20).
    pub fn get_element_entry(&self, id: &ComputedElementId) -> Option<&ElementEntry> {
        self.element_entries.get(id)
    }

    /// Check if an element entry exists (Phase 20).
    pub fn has_element_entry(&self, id: &ComputedElementId) -> bool {
        self.element_entries.contains_key(id)
    }

    /// Get the number of tracked element entries (Phase 20).
    pub fn element_entry_count(&self) -> usize {
        self.element_entries.len()
    }

    /// Get all element entries (Phase 20).
    /// Used for cache transfer between frames.
    pub fn element_entries(&self) -> &FxHashMap<ComputedElementId, ElementEntry> {
        &self.element_entries
    }

    /// Copy items from a previous frame's entry into this display list (Phase 20).
    ///
    /// This is the core of per-element caching: when an element's input_hash
    /// matches the previous frame, we copy its items instead of repainting.
    ///
    /// Returns the bounds of the copied items, or None if source_items is empty.
    pub fn copy_items_from_range(
        &mut self,
        source_display_list: &DisplayList,
        source_range: Range<usize>,
    ) -> Option<Bounds<Pixels>> {
        if source_range.is_empty() {
            return None;
        }

        let start_index = self.items.len();

        // Copy items from source
        for item in &source_display_list.items[source_range.clone()] {
            // Clone the item and add to our list
            // Note: This extends content_bounds and marks spatial index dirty
            match item {
                DisplayItem::Quad(q) => self.push_quad(q.clone()),
                DisplayItem::Shadow(s) => self.push_shadow(s.clone()),
                DisplayItem::BackdropBlur(b) => self.push_backdrop_blur(b.clone()),
                DisplayItem::Underline(u) => self.push_underline(u.clone()),
                DisplayItem::MonochromeSprite(s) => self.push_monochrome_sprite(s.clone()),
                DisplayItem::SubpixelSprite(s) => self.push_subpixel_sprite(s.clone()),
                DisplayItem::PolychromeSprite(s) => self.push_polychrome_sprite(s.clone()),
                DisplayItem::Path(p) => self.push_path(p.clone()),
            }
        }

        let end_index = self.items.len();
        Some(self.calculate_bounds_for_range(start_index, end_index))
    }

    /// Check if we're currently inside an element paint operation (Phase 20).
    pub fn is_painting_element(&self) -> bool {
        !self.element_paint_stack.is_empty()
    }

    /// Get the current element being painted (Phase 20).
    pub fn current_element(&self) -> Option<ComputedElementId> {
        self.element_paint_stack.last().map(|e| e.id)
    }

    // ========================================================================
    // Dirty Region Tracking (Phase 14: Incremental Invalidation)
    // ========================================================================

    /// Mark a region as dirty (needing re-rasterization).
    ///
    /// Call this when content in a specific area changes. Only tiles that
    /// intersect dirty regions will be re-rasterized.
    ///
    /// Note: This does NOT increment the generation counter. Tiles check
    /// dirty_regions separately from generation.
    pub fn invalidate_region(&mut self, bounds: Bounds<Pixels>) {
        // Don't add zero-size regions
        if bounds.size.width.0 <= 0.0 || bounds.size.height.0 <= 0.0 {
            return;
        }

        // Check if this region is already covered by existing dirty regions
        // (optimization to avoid redundant regions)
        let dominated = self.dirty_regions.iter().any(|existing| {
            existing.origin.x <= bounds.origin.x
                && existing.origin.y <= bounds.origin.y
                && existing.origin.x.0 + existing.size.width.0
                    >= bounds.origin.x.0 + bounds.size.width.0
                && existing.origin.y.0 + existing.size.height.0
                    >= bounds.origin.y.0 + bounds.size.height.0
        });

        if !dominated {
            self.dirty_regions.push(bounds);
        }
    }

    /// Mark the entire display list as dirty.
    ///
    /// Use this when you can't determine exactly what changed.
    /// All tiles will be re-rasterized.
    pub fn invalidate_all(&mut self) {
        self.dirty_regions.clear();
        self.dirty_regions.push(self.content_bounds);
    }

    /// Clear all dirty regions (call after tiles have been re-rasterized).
    pub fn clear_dirty_regions(&mut self) {
        self.dirty_regions.clear();
    }

    /// Check if there are any dirty regions.
    pub fn has_dirty_regions(&self) -> bool {
        !self.dirty_regions.is_empty()
    }

    /// Get the dirty regions.
    pub fn dirty_regions(&self) -> &[Bounds<Pixels>] {
        &self.dirty_regions
    }

    /// Check if a given bounds intersects any dirty region.
    ///
    /// Used to determine if a tile needs re-rasterization.
    pub fn intersects_dirty_region(&self, bounds: &Bounds<Pixels>) -> bool {
        // If no dirty regions, nothing is dirty
        if self.dirty_regions.is_empty() {
            return false;
        }

        self.dirty_regions
            .iter()
            .any(|dirty| dirty.intersects(bounds))
    }

    /// Ensure the spatial index is built and up-to-date.
    ///
    /// Call this before cloning the display list for rasterization to ensure
    /// the clone has an efficient spatial index for `items_intersecting` queries.
    pub fn ensure_spatial_index_built(&mut self) {
        if self.spatial_index.is_dirty() {
            self.spatial_index.build(&self.items, self.content_bounds);
        }
    }

    /// Iterate over items that intersect the given bounds.
    ///
    /// Uses spatial index for O(cells × items_per_cell) performance.
    ///
    /// The spatial index must be built before calling this method.
    /// Call `ensure_spatial_index_built()` before rasterization.
    pub fn items_intersecting(&self, bounds: Bounds<Pixels>) -> impl Iterator<Item = &DisplayItem> + '_ {
        debug_assert!(
            !self.spatial_index.is_dirty(),
            "Spatial index not built before items_intersecting() - call ensure_spatial_index_built() first"
        );

        ItemsIntersectingIter {
            items: &self.items,
            indices: self.spatial_index.query(bounds).into_iter(),
            query_bounds: bounds,
        }
    }

    /// Iterate over items that intersect the given bounds, with their indices.
    ///
    /// Returns (index, &item) pairs for callers that need item indices.
    pub fn items_intersecting_with_index(&self, bounds: Bounds<Pixels>) -> impl Iterator<Item = (usize, &DisplayItem)> + '_ {
        debug_assert!(
            !self.spatial_index.is_dirty(),
            "Spatial index not built before items_intersecting_with_index() - call ensure_spatial_index_built() first"
        );

        self.spatial_index
            .query(bounds)
            .into_iter()
            .filter_map(move |index| {
                let item = self.items.get(index)?;
                if item.intersects(&bounds) {
                    Some((index, item))
                } else {
                    None
                }
            })
    }

    /// Extend content_bounds to include the given bounds.
    /// Also marks the spatial index as dirty since content has changed.
    pub fn extend_bounds(&mut self, bounds: Bounds<Pixels>) {
        if self.content_bounds.size.width.0 == 0.0 && self.content_bounds.size.height.0 == 0.0 {
            self.content_bounds = bounds;
        } else {
            self.content_bounds = self.content_bounds.union(&bounds);
        }
        // Mark spatial index dirty when content changes
        self.spatial_index.mark_dirty();
    }

    /// Push a quad display item.
    pub fn push_quad(&mut self, quad: DisplayQuad) {
        self.extend_bounds(quad.bounds);
        self.items.push(DisplayItem::Quad(quad));
    }

    /// Push a shadow display item.
    pub fn push_shadow(&mut self, shadow: DisplayShadow) {
        self.extend_bounds(shadow.bounds);
        self.items.push(DisplayItem::Shadow(shadow));
    }

    /// Push a monochrome sprite display item.
    pub fn push_monochrome_sprite(&mut self, sprite: DisplayMonochromeSprite) {
        self.extend_bounds(sprite.bounds);
        self.items.push(DisplayItem::MonochromeSprite(sprite));
    }

    /// Push a subpixel sprite display item.
    pub fn push_subpixel_sprite(&mut self, sprite: DisplaySubpixelSprite) {
        self.extend_bounds(sprite.bounds);
        self.items.push(DisplayItem::SubpixelSprite(sprite));
    }

    /// Push a polychrome sprite display item.
    pub fn push_polychrome_sprite(&mut self, sprite: DisplayPolychromeSprite) {
        self.extend_bounds(sprite.bounds);
        self.items.push(DisplayItem::PolychromeSprite(sprite));
    }

    /// Push an underline display item.
    pub fn push_underline(&mut self, underline: DisplayUnderline) {
        self.extend_bounds(underline.bounds);
        self.items.push(DisplayItem::Underline(underline));
    }

    /// Push a backdrop blur display item.
    pub fn push_backdrop_blur(&mut self, blur: DisplayBackdropBlur) {
        self.extend_bounds(blur.bounds);
        self.items.push(DisplayItem::BackdropBlur(blur));
    }

    /// Push a path display item.
    pub fn push_path(&mut self, path: DisplayPath) {
        self.extend_bounds(path.bounds);
        self.items.push(DisplayItem::Path(path));
    }

    /// Rasterize items intersecting the tile bounds to Scene primitives.
    ///
    /// Each item contains baked world-space clip and transform from insert time.
    /// Coordinates are offset so that the tile's origin is at (0, 0) in the output.
    /// The scale factor converts from logical Pixels to device ScaledPixels.
    ///
    /// Returns vectors of primitives ready for GPU rendering.
    pub fn rasterize_tile(
        &self,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
    ) -> TileRasterResult {
        let mut result = TileRasterResult::default();
        let mut draw_order: DrawOrder = 0;

        for (_index, item) in self.items_intersecting_with_index(tile_bounds) {
            draw_order += 1;

            match item {
                DisplayItem::Quad(q) => {
                    if let Some((quad, transform)) = self.rasterize_quad(q, tile_bounds, scale_factor, draw_order, q.world_transform, q.world_clip) {
                        result.quads.push(quad);
                        result.quad_transforms.push(transform);
                    }
                }
                DisplayItem::Shadow(s) => {
                    if let Some((shadow, transform)) = self.rasterize_shadow(s, tile_bounds, scale_factor, draw_order, s.world_transform, s.world_clip) {
                        result.shadows.push(shadow);
                        result.shadow_transforms.push(transform);
                    }
                }
                DisplayItem::BackdropBlur(b) => {
                    if let Some((blur, transform)) = self.rasterize_backdrop_blur(b, tile_bounds, scale_factor, draw_order, b.world_transform, b.world_clip) {
                        result.backdrop_blurs.push(blur);
                        result.backdrop_blur_transforms.push(transform);
                    }
                }
                DisplayItem::Underline(u) => {
                    if let Some((underline, transform)) = self.rasterize_underline(u, tile_bounds, scale_factor, draw_order, u.world_transform, u.world_clip) {
                        result.underlines.push(underline);
                        result.underline_transforms.push(transform);
                    }
                }
                DisplayItem::MonochromeSprite(s) => {
                    if let Some(sprite) = self.rasterize_monochrome_sprite(s, tile_bounds, scale_factor, draw_order, s.world_transform, s.world_clip) {
                        result.monochrome_sprites.push(sprite);
                    }
                }
                DisplayItem::SubpixelSprite(s) => {
                    if let Some(sprite) = self.rasterize_subpixel_sprite(s, tile_bounds, scale_factor, draw_order, s.world_transform, s.world_clip) {
                        result.subpixel_sprites.push(sprite);
                    }
                }
                DisplayItem::PolychromeSprite(s) => {
                    if let Some((sprite, transform)) = self.rasterize_polychrome_sprite(s, tile_bounds, scale_factor, draw_order, s.world_transform, s.world_clip) {
                        result.polychrome_sprites.push(sprite);
                        result.polychrome_sprite_transforms.push(transform);
                    }
                }
                DisplayItem::Path(_) => {
                    // TODO: Path rasterization requires more complex handling
                }
            }
        }

        result
    }

    /// Rasterize the entire display list to primitives.
    /// This is used for the root layer which doesn't use tiling.
    /// Items are converted to window-coordinate primitives for direct Scene insertion.
    pub fn rasterize_all(&self, scale_factor: f32) -> TileRasterResult {
        let mut result = TileRasterResult::default();
        let mut draw_order: DrawOrder = 0;

        // Use a very large tile bounds to include all items (origin is implicitly 0,0)
        let tile_bounds = Bounds {
            origin: Point::default(),
            size: crate::Size {
                width: Pixels(100000.0),
                height: Pixels(100000.0),
            },
        };

        for item in self.items.iter() {
            draw_order += 1;

            match item {
                DisplayItem::Quad(q) => {
                    if let Some((quad, transform)) = self.rasterize_quad(q, tile_bounds, scale_factor, draw_order, q.world_transform, q.world_clip) {
                        result.quads.push(quad);
                        result.quad_transforms.push(transform);
                    }
                }
                DisplayItem::Shadow(s) => {
                    if let Some((shadow, transform)) = self.rasterize_shadow(s, tile_bounds, scale_factor, draw_order, s.world_transform, s.world_clip) {
                        result.shadows.push(shadow);
                        result.shadow_transforms.push(transform);
                    }
                }
                DisplayItem::BackdropBlur(b) => {
                    if let Some((blur, transform)) = self.rasterize_backdrop_blur(b, tile_bounds, scale_factor, draw_order, b.world_transform, b.world_clip) {
                        result.backdrop_blurs.push(blur);
                        result.backdrop_blur_transforms.push(transform);
                    }
                }
                DisplayItem::Underline(u) => {
                    if let Some((underline, transform)) = self.rasterize_underline(u, tile_bounds, scale_factor, draw_order, u.world_transform, u.world_clip) {
                        result.underlines.push(underline);
                        result.underline_transforms.push(transform);
                    }
                }
                DisplayItem::MonochromeSprite(s) => {
                    if let Some(sprite) = self.rasterize_monochrome_sprite(s, tile_bounds, scale_factor, draw_order, s.world_transform, s.world_clip) {
                        result.monochrome_sprites.push(sprite);
                    }
                }
                DisplayItem::SubpixelSprite(s) => {
                    if let Some(sprite) = self.rasterize_subpixel_sprite(s, tile_bounds, scale_factor, draw_order, s.world_transform, s.world_clip) {
                        result.subpixel_sprites.push(sprite);
                    }
                }
                DisplayItem::PolychromeSprite(s) => {
                    if let Some((sprite, transform)) = self.rasterize_polychrome_sprite(s, tile_bounds, scale_factor, draw_order, s.world_transform, s.world_clip) {
                        result.polychrome_sprites.push(sprite);
                        result.polychrome_sprite_transforms.push(transform);
                    }
                }
                DisplayItem::Path(_) => {
                    // TODO: Path rasterization requires more complex handling
                }
            }
        }

        result
    }

    /// Offset bounds relative to tile origin and scale to device pixels.
    fn offset_and_scale_bounds(
        &self,
        bounds: Bounds<Pixels>,
        tile_origin: Point<Pixels>,
        scale_factor: f32,
    ) -> Bounds<ScaledPixels> {
        let offset_origin = point(
            bounds.origin.x - tile_origin.x,
            bounds.origin.y - tile_origin.y,
        );
        Bounds {
            origin: point(
                ScaledPixels(offset_origin.x.0 * scale_factor),
                ScaledPixels(offset_origin.y.0 * scale_factor),
            ),
            size: size(
                ScaledPixels(bounds.size.width.0 * scale_factor),
                ScaledPixels(bounds.size.height.0 * scale_factor),
            ),
        }
    }

    /// Offset and scale content mask.
    fn offset_and_scale_content_mask(
        &self,
        mask: &ContentMask<Pixels>,
        tile_origin: Point<Pixels>,
        scale_factor: f32,
    ) -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: self.offset_and_scale_bounds(mask.bounds, tile_origin, scale_factor),
        }
    }

    /// Scale corners to device pixels.
    fn scale_corners(&self, corners: &Corners<Pixels>, scale_factor: f32) -> Corners<ScaledPixels> {
        Corners {
            top_left: ScaledPixels(corners.top_left.0 * scale_factor),
            top_right: ScaledPixels(corners.top_right.0 * scale_factor),
            bottom_right: ScaledPixels(corners.bottom_right.0 * scale_factor),
            bottom_left: ScaledPixels(corners.bottom_left.0 * scale_factor),
        }
    }

    /// Scale edges to device pixels.
    fn scale_edges(&self, edges: &Edges<Pixels>, scale_factor: f32) -> Edges<ScaledPixels> {
        Edges {
            top: ScaledPixels(edges.top.0 * scale_factor),
            right: ScaledPixels(edges.right.0 * scale_factor),
            bottom: ScaledPixels(edges.bottom.0 * scale_factor),
            left: ScaledPixels(edges.left.0 * scale_factor),
        }
    }

    #[inline]
    fn rasterize_quad(
        &self,
        q: &DisplayQuad,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        world_transform: TransformationMatrix,
        world_clip: Bounds<Pixels>,
    ) -> Option<(Quad, TransformationMatrix)> {
        let content_mask = ContentMask { bounds: world_clip };

        Some((Quad {
            order,
            border_style: q.border_style,
            bounds: self.offset_and_scale_bounds(q.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&content_mask, tile_bounds.origin, scale_factor),
            background: q.background.clone(),
            border_color: q.border_color,
            corner_radii: self.scale_corners(&q.corner_radii, scale_factor),
            border_widths: self.scale_edges(&q.border_widths, scale_factor),
        }, world_transform))
    }

    #[inline]
    fn rasterize_shadow(
        &self,
        s: &DisplayShadow,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        world_transform: TransformationMatrix,
        world_clip: Bounds<Pixels>,
    ) -> Option<(Shadow, TransformationMatrix)> {
        let content_mask = ContentMask { bounds: world_clip };

        Some((Shadow {
            order,
            blur_radius: ScaledPixels(s.blur_radius.0 * scale_factor),
            bounds: self.offset_and_scale_bounds(s.bounds, tile_bounds.origin, scale_factor),
            corner_radii: self.scale_corners(&s.corner_radii, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&content_mask, tile_bounds.origin, scale_factor),
            color: s.color,
        }, world_transform))
    }

    #[inline]
    fn rasterize_backdrop_blur(
        &self,
        b: &DisplayBackdropBlur,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        world_transform: TransformationMatrix,
        world_clip: Bounds<Pixels>,
    ) -> Option<(BackdropBlur, TransformationMatrix)> {
        let content_mask = ContentMask { bounds: world_clip };

        Some((BackdropBlur {
            order,
            blur_radius: ScaledPixels(b.blur_radius.0 * scale_factor),
            bounds: self.offset_and_scale_bounds(b.bounds, tile_bounds.origin, scale_factor),
            corner_radii: self.scale_corners(&b.corner_radii, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&content_mask, tile_bounds.origin, scale_factor),
            tint: b.tint,
        }, world_transform))
    }

    #[inline]
    fn rasterize_underline(
        &self,
        u: &DisplayUnderline,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        world_transform: TransformationMatrix,
        world_clip: Bounds<Pixels>,
    ) -> Option<(Underline, TransformationMatrix)> {
        let content_mask = ContentMask { bounds: world_clip };

        Some((Underline {
            order,
            pad: 0,
            bounds: self.offset_and_scale_bounds(u.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&content_mask, tile_bounds.origin, scale_factor),
            color: u.color,
            thickness: ScaledPixels(u.thickness.0 * scale_factor),
            wavy: if u.wavy { 1 } else { 0 },
        }, world_transform))
    }

    #[inline]
    fn rasterize_monochrome_sprite(
        &self,
        s: &DisplayMonochromeSprite,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        world_transform: TransformationMatrix,
        world_clip: Bounds<Pixels>,
    ) -> Option<MonochromeSprite> {
        let content_mask = ContentMask { bounds: world_clip };

        Some(MonochromeSprite {
            order,
            pad: 0,
            bounds: self.offset_and_scale_bounds(s.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&content_mask, tile_bounds.origin, scale_factor),
            color: s.color,
            tile: s.tile.clone(),
            transformation: world_transform,
        })
    }

    #[inline]
    fn rasterize_subpixel_sprite(
        &self,
        s: &DisplaySubpixelSprite,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        world_transform: TransformationMatrix,
        world_clip: Bounds<Pixels>,
    ) -> Option<SubpixelSprite> {
        let content_mask = ContentMask { bounds: world_clip };

        Some(SubpixelSprite {
            order,
            pad: 0,
            bounds: self.offset_and_scale_bounds(s.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&content_mask, tile_bounds.origin, scale_factor),
            color: s.color,
            tile: s.tile.clone(),
            transformation: world_transform,
        })
    }

    #[inline]
    fn rasterize_polychrome_sprite(
        &self,
        s: &DisplayPolychromeSprite,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        world_transform: TransformationMatrix,
        world_clip: Bounds<Pixels>,
    ) -> Option<(PolychromeSprite, TransformationMatrix)> {
        let content_mask = ContentMask { bounds: world_clip };

        Some((PolychromeSprite {
            order,
            pad: 0,
            grayscale: s.grayscale,
            opacity: s.opacity,
            bounds: self.offset_and_scale_bounds(s.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&content_mask, tile_bounds.origin, scale_factor),
            corner_radii: self.scale_corners(&s.corner_radii, scale_factor),
            tile: s.tile.clone(),
        }, world_transform))
    }
}

/// Result of rasterizing a tile from a display list.
/// Contains all primitives ready for GPU rendering.
#[derive(Default)]
pub(crate) struct TileRasterResult {
    pub quads: Vec<Quad>,
    pub quad_transforms: Vec<TransformationMatrix>,
    pub shadows: Vec<Shadow>,
    pub shadow_transforms: Vec<TransformationMatrix>,
    pub backdrop_blurs: Vec<BackdropBlur>,
    pub backdrop_blur_transforms: Vec<TransformationMatrix>,
    pub underlines: Vec<Underline>,
    pub underline_transforms: Vec<TransformationMatrix>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub polychrome_sprite_transforms: Vec<TransformationMatrix>,
}

impl TileRasterResult {
    /// Check if the result is empty (no primitives).
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
            && self.shadows.is_empty()
            && self.backdrop_blurs.is_empty()
            && self.underlines.is_empty()
            && self.monochrome_sprites.is_empty()
            && self.subpixel_sprites.is_empty()
            && self.polychrome_sprites.is_empty()
    }
}

/// Manages display lists for scroll containers.
///
/// Each scroll container with an element ID can have a display list that
/// persists across frames. The display list is rebuilt when content changes
/// and used to rasterize visible tiles.
#[derive(Default)]
pub(crate) struct DisplayListManager {
    /// Display lists indexed by element ID.
    lists: FxHashMap<GlobalElementId, DisplayList>,
    /// Current frame number.
    frame: u64,
}

impl DisplayListManager {
    /// Create a new display list manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Called at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Called at the end of each frame.
    pub fn end_frame(&mut self) {
        // Could implement LRU eviction here for unused display lists
    }

    /// Get or create a display list for the given element.
    pub fn get_or_create(&mut self, element_id: &GlobalElementId) -> &mut DisplayList {
        self.lists
            .entry(element_id.clone())
            .or_insert_with(|| DisplayList::new(element_id.clone()))
    }

    /// Get a display list by element ID.
    pub fn get(&self, element_id: &GlobalElementId) -> Option<&DisplayList> {
        self.lists.get(element_id)
    }

    /// Get a mutable display list by element ID.
    pub fn get_mut(&mut self, element_id: &GlobalElementId) -> Option<&mut DisplayList> {
        self.lists.get_mut(element_id)
    }

    /// Check if a display list exists for the given element.
    pub fn contains(&self, element_id: &GlobalElementId) -> bool {
        self.lists.contains_key(element_id)
    }

    /// Insert a display list for the given element.
    pub fn insert(&mut self, element_id: GlobalElementId, display_list: DisplayList) {
        self.lists.insert(element_id, display_list);
    }

    /// Remove a display list.
    pub fn remove(&mut self, element_id: &GlobalElementId) -> Option<DisplayList> {
        self.lists.remove(element_id)
    }

    /// Clear all display lists.
    pub fn clear(&mut self) {
        self.lists.clear();
    }

    /// Get the number of display lists.
    pub fn len(&self) -> usize {
        self.lists.len()
    }

    /// Check if there are no display lists.
    pub fn is_empty(&self) -> bool {
        self.lists.is_empty()
    }
}

/// Convert a Scene Primitive to a DisplayItem.
///
/// This is a legacy conversion function used during the transition to direct-to-DisplayList
/// painting. New code should create DisplayItems directly.
///
/// `content_origin` is the window-space origin of the scroll container in ScaledPixels.
/// This is subtracted from all bounds to convert from window coordinates to content coordinates.
/// `inv_scale` is 1.0 / scale_factor to convert from ScaledPixels to Pixels.
/// `world_transform` and `world_clip` are the baked world-space values from property trees.
pub fn convert_primitive_to_display_item(
    primitive: &crate::scene::Primitive,
    inv_scale: f32,
    content_origin: Point<crate::ScaledPixels>,
    world_transform: TransformationMatrix,
    world_clip: Bounds<Pixels>,
) -> Option<DisplayItem> {
    use crate::scene::Primitive;

    match primitive {
        Primitive::Quad(quad, _transform) => {
            Some(DisplayItem::Quad(DisplayQuad {
                bounds: offset_and_scale_bounds_to_pixels(&quad.bounds, content_origin, inv_scale),
                world_clip,
                world_transform,
                background: quad.background.clone(),
                border_color: quad.border_color,
                border_style: quad.border_style,
                corner_radii: scale_corners_to_pixels(&quad.corner_radii, inv_scale),
                border_widths: scale_edges_to_pixels(&quad.border_widths, inv_scale),
            }))
        }
        Primitive::Shadow(shadow, _transform) => {
            Some(DisplayItem::Shadow(DisplayShadow {
                bounds: offset_and_scale_bounds_to_pixels(&shadow.bounds, content_origin, inv_scale),
                world_clip,
                world_transform,
                color: shadow.color,
                blur_radius: Pixels(shadow.blur_radius.0 * inv_scale),
                corner_radii: scale_corners_to_pixels(&shadow.corner_radii, inv_scale),
            }))
        }
        Primitive::BackdropBlur(blur, _transform) => {
            Some(DisplayItem::BackdropBlur(DisplayBackdropBlur {
                bounds: offset_and_scale_bounds_to_pixels(&blur.bounds, content_origin, inv_scale),
                world_clip,
                world_transform,
                blur_radius: Pixels(blur.blur_radius.0 * inv_scale),
                corner_radii: scale_corners_to_pixels(&blur.corner_radii, inv_scale),
                tint: blur.tint,
            }))
        }
        Primitive::Underline(underline, _transform) => {
            Some(DisplayItem::Underline(DisplayUnderline {
                bounds: offset_and_scale_bounds_to_pixels(&underline.bounds, content_origin, inv_scale),
                world_clip,
                world_transform,
                color: underline.color,
                thickness: Pixels(underline.thickness.0 * inv_scale),
                wavy: underline.wavy != 0,
            }))
        }
        Primitive::MonochromeSprite(sprite) => {
            Some(DisplayItem::MonochromeSprite(DisplayMonochromeSprite {
                bounds: offset_and_scale_bounds_to_pixels(&sprite.bounds, content_origin, inv_scale),
                world_clip,
                world_transform,
                color: sprite.color,
                tile: sprite.tile.clone(),
            }))
        }
        Primitive::SubpixelSprite(sprite) => {
            Some(DisplayItem::SubpixelSprite(DisplaySubpixelSprite {
                bounds: offset_and_scale_bounds_to_pixels(&sprite.bounds, content_origin, inv_scale),
                world_clip,
                world_transform,
                color: sprite.color,
                tile: sprite.tile.clone(),
            }))
        }
        Primitive::PolychromeSprite(sprite, _transform) => {
            Some(DisplayItem::PolychromeSprite(DisplayPolychromeSprite {
                bounds: offset_and_scale_bounds_to_pixels(&sprite.bounds, content_origin, inv_scale),
                world_clip,
                world_transform,
                grayscale: sprite.grayscale,
                opacity: sprite.opacity,
                corner_radii: scale_corners_to_pixels(&sprite.corner_radii, inv_scale),
                tile: sprite.tile.clone(),
            }))
        }
        Primitive::Path(path) => {
            let vertices = path
                .vertices
                .iter()
                .map(|v| PathVertex {
                    xy_position: point(
                        Pixels((v.xy_position.x.0 - content_origin.x.0) * inv_scale),
                        Pixels((v.xy_position.y.0 - content_origin.y.0) * inv_scale),
                    ),
                    st_position: v.st_position,
                })
                .collect();

            Some(DisplayItem::Path(DisplayPath {
                bounds: offset_and_scale_bounds_to_pixels(&path.bounds, content_origin, inv_scale),
                world_clip,
                world_transform,
                color: path.color.clone(),
                vertices,
            }))
        }
        // Surface, CachedTexture, and TileSprite are not converted to display list
        // as they are compositing primitives, not content primitives
        Primitive::Surface(_) | Primitive::CachedTexture(_) | Primitive::TileSprite(_) => None,
    }
}

fn scale_bounds_to_pixels(bounds: &Bounds<crate::ScaledPixels>, inv_scale: f32) -> Bounds<Pixels> {
    Bounds {
        origin: point(
            Pixels(bounds.origin.x.0 * inv_scale),
            Pixels(bounds.origin.y.0 * inv_scale),
        ),
        size: size(
            Pixels(bounds.size.width.0 * inv_scale),
            Pixels(bounds.size.height.0 * inv_scale),
        ),
    }
}

/// Offset bounds by subtracting content_origin (window to content coordinates) and scale to pixels.
fn offset_and_scale_bounds_to_pixels(
    bounds: &Bounds<crate::ScaledPixels>,
    content_origin: Point<crate::ScaledPixels>,
    inv_scale: f32,
) -> Bounds<Pixels> {
    Bounds {
        origin: point(
            Pixels((bounds.origin.x.0 - content_origin.x.0) * inv_scale),
            Pixels((bounds.origin.y.0 - content_origin.y.0) * inv_scale),
        ),
        size: size(
            Pixels(bounds.size.width.0 * inv_scale),
            Pixels(bounds.size.height.0 * inv_scale),
        ),
    }
}

fn scale_content_mask_to_pixels(
    mask: &ContentMask<crate::ScaledPixels>,
    inv_scale: f32,
) -> ContentMask<Pixels> {
    ContentMask {
        bounds: scale_bounds_to_pixels(&mask.bounds, inv_scale),
    }
}

/// Offset content mask bounds by subtracting content_origin and scale to pixels.
fn offset_and_scale_content_mask_to_pixels(
    mask: &ContentMask<crate::ScaledPixels>,
    content_origin: Point<crate::ScaledPixels>,
    inv_scale: f32,
) -> ContentMask<Pixels> {
    ContentMask {
        bounds: offset_and_scale_bounds_to_pixels(&mask.bounds, content_origin, inv_scale),
    }
}

fn scale_corners_to_pixels(
    corners: &Corners<crate::ScaledPixels>,
    inv_scale: f32,
) -> Corners<Pixels> {
    Corners {
        top_left: Pixels(corners.top_left.0 * inv_scale),
        top_right: Pixels(corners.top_right.0 * inv_scale),
        bottom_right: Pixels(corners.bottom_right.0 * inv_scale),
        bottom_left: Pixels(corners.bottom_left.0 * inv_scale),
    }
}

fn scale_edges_to_pixels(edges: &Edges<crate::ScaledPixels>, inv_scale: f32) -> Edges<Pixels> {
    Edges {
        top: Pixels(edges.top.0 * inv_scale),
        right: Pixels(edges.right.0 * inv_scale),
        bottom: Pixels(edges.bottom.0 * inv_scale),
        left: Pixels(edges.left.0 * inv_scale),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size, ElementId};
    use std::sync::Arc;

    fn test_element_id() -> GlobalElementId {
        GlobalElementId(Arc::from([ElementId::Integer(1)]))
    }

    #[test]
    fn test_display_list_creation() {
        let mut manager = DisplayListManager::new();
        let element_id = test_element_id();

        let list = manager.get_or_create(&element_id);
        assert_eq!(list.generation, 0);
        assert!(list.items.is_empty());
    }

    #[test]
    fn test_display_list_clear_increments_generation() {
        let mut list = DisplayList::new(test_element_id());
        assert_eq!(list.generation, 0);

        list.clear();
        assert_eq!(list.generation, 1);

        list.clear();
        assert_eq!(list.generation, 2);
    }

    #[test]
    fn test_content_bounds_expansion() {
        let mut list = DisplayList::new(test_element_id());

        list.push_quad(DisplayQuad {
            bounds: Bounds {
                origin: point(Pixels(10.0), Pixels(10.0)),
                size: size(Pixels(100.0), Pixels(100.0)),
            },
            world_clip: Bounds::default(),
            world_transform: TransformationMatrix::unit(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        });

        assert_eq!(list.content_bounds.origin.x, Pixels(10.0));
        assert_eq!(list.content_bounds.origin.y, Pixels(10.0));
        assert_eq!(list.content_bounds.size.width, Pixels(100.0));
        assert_eq!(list.content_bounds.size.height, Pixels(100.0));

        // Add another quad that extends bounds
        list.push_quad(DisplayQuad {
            bounds: Bounds {
                origin: point(Pixels(50.0), Pixels(50.0)),
                size: size(Pixels(200.0), Pixels(200.0)),
            },
            world_clip: Bounds::default(),
            world_transform: TransformationMatrix::unit(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        });

        // Bounds should now cover both quads
        assert_eq!(list.content_bounds.origin.x, Pixels(10.0));
        assert_eq!(list.content_bounds.origin.y, Pixels(10.0));
        assert_eq!(list.content_bounds.size.width, Pixels(240.0)); // 50 + 200 - 10
        assert_eq!(list.content_bounds.size.height, Pixels(240.0));
    }

    #[test]
    fn test_items_intersecting() {
        let mut list = DisplayList::new(test_element_id());

        // Add a quad at (0, 0) to (100, 100)
        list.push_quad(DisplayQuad {
            bounds: Bounds {
                origin: point(Pixels(0.0), Pixels(0.0)),
                size: size(Pixels(100.0), Pixels(100.0)),
            },
            world_clip: Bounds::default(),
            world_transform: TransformationMatrix::unit(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        });

        // Add a quad at (200, 200) to (300, 300)
        list.push_quad(DisplayQuad {
            bounds: Bounds {
                origin: point(Pixels(200.0), Pixels(200.0)),
                size: size(Pixels(100.0), Pixels(100.0)),
            },
            world_clip: Bounds::default(),
            world_transform: TransformationMatrix::unit(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        });

        // Build spatial index before querying
        list.ensure_spatial_index_built();

        // Query for items in (0, 0) to (50, 50) - should find first quad
        let search_bounds = Bounds {
            origin: point(Pixels(0.0), Pixels(0.0)),
            size: size(Pixels(50.0), Pixels(50.0)),
        };
        let count = list.items_intersecting(search_bounds).count();
        assert_eq!(count, 1);

        // Query for items in (150, 150) to (250, 250) - should find second quad
        let search_bounds = Bounds {
            origin: point(Pixels(150.0), Pixels(150.0)),
            size: size(Pixels(100.0), Pixels(100.0)),
        };
        let count = list.items_intersecting(search_bounds).count();
        assert_eq!(count, 1);

        // Query for items in (0, 0) to (300, 300) - should find both quads
        let search_bounds = Bounds {
            origin: point(Pixels(0.0), Pixels(0.0)),
            size: size(Pixels(300.0), Pixels(300.0)),
        };
        let count = list.items_intersecting(search_bounds).count();
        assert_eq!(count, 2);
    }

    // ========================================================================
    // Dirty Region Tests (Phase 14: Incremental Invalidation)
    // ========================================================================

    #[test]
    fn test_dirty_region_tracking() {
        let mut list = DisplayList::new(test_element_id());

        // Initially no dirty regions
        assert!(!list.has_dirty_regions());

        // Invalidate a region
        let dirty_bounds = Bounds {
            origin: point(Pixels(10.0), Pixels(10.0)),
            size: size(Pixels(50.0), Pixels(50.0)),
        };
        list.invalidate_region(dirty_bounds);

        assert!(list.has_dirty_regions());
        assert_eq!(list.dirty_regions().len(), 1);

        // Clear dirty regions
        list.clear_dirty_regions();
        assert!(!list.has_dirty_regions());
    }

    #[test]
    fn test_intersects_dirty_region() {
        let mut list = DisplayList::new(test_element_id());

        // Dirty region at (100, 100) to (200, 200)
        list.invalidate_region(Bounds {
            origin: point(Pixels(100.0), Pixels(100.0)),
            size: size(Pixels(100.0), Pixels(100.0)),
        });

        // Bounds that intersect dirty region
        let intersecting = Bounds {
            origin: point(Pixels(150.0), Pixels(150.0)),
            size: size(Pixels(100.0), Pixels(100.0)),
        };
        assert!(list.intersects_dirty_region(&intersecting));

        // Bounds that don't intersect dirty region
        let non_intersecting = Bounds {
            origin: point(Pixels(0.0), Pixels(0.0)),
            size: size(Pixels(50.0), Pixels(50.0)),
        };
        assert!(!list.intersects_dirty_region(&non_intersecting));
    }

    #[test]
    fn test_invalidate_all() {
        let mut list = DisplayList::new(test_element_id());

        // Add some content to set content_bounds
        list.push_quad(DisplayQuad {
            bounds: Bounds {
                origin: point(Pixels(0.0), Pixels(0.0)),
                size: size(Pixels(1000.0), Pixels(1000.0)),
            },
            world_clip: Bounds::default(),
            world_transform: TransformationMatrix::unit(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        });

        // Invalidate all
        list.invalidate_all();

        // Any bounds within content should intersect dirty
        let test_bounds = Bounds {
            origin: point(Pixels(500.0), Pixels(500.0)),
            size: size(Pixels(50.0), Pixels(50.0)),
        };
        assert!(list.intersects_dirty_region(&test_bounds));
    }

    #[test]
    fn test_dominated_dirty_region_optimization() {
        let mut list = DisplayList::new(test_element_id());

        // Add a large dirty region
        list.invalidate_region(Bounds {
            origin: point(Pixels(0.0), Pixels(0.0)),
            size: size(Pixels(500.0), Pixels(500.0)),
        });

        // Try to add a smaller region fully inside the larger one
        list.invalidate_region(Bounds {
            origin: point(Pixels(100.0), Pixels(100.0)),
            size: size(Pixels(100.0), Pixels(100.0)),
        });

        // Should still only have 1 dirty region (the smaller was dominated)
        assert_eq!(list.dirty_regions().len(), 1);
    }

    #[test]
    fn test_clear_also_clears_dirty_regions() {
        let mut list = DisplayList::new(test_element_id());

        list.invalidate_region(Bounds {
            origin: point(Pixels(0.0), Pixels(0.0)),
            size: size(Pixels(100.0), Pixels(100.0)),
        });
        assert!(list.has_dirty_regions());

        list.clear();
        assert!(!list.has_dirty_regions());
    }

    // ========================================================================
    // Spatial Index Tests
    // ========================================================================

    #[test]
    fn test_spatial_index_basic() {
        let mut list = DisplayList::new(test_element_id());

        // Add quads in a grid pattern
        for row in 0..4 {
            for col in 0..4 {
                list.push_quad(DisplayQuad {
                    bounds: Bounds {
                        origin: point(Pixels(col as f32 * 100.0), Pixels(row as f32 * 100.0)),
                        size: size(Pixels(80.0), Pixels(80.0)),
                    },
                    world_clip: Bounds::default(),
                    world_transform: TransformationMatrix::unit(),
                    background: Background::default(),
                    border_color: Hsla::default(),
                    border_style: BorderStyle::default(),
                    corner_radii: Corners::default(),
                    border_widths: Edges::default(),
                });
            }
        }

        assert_eq!(list.items.len(), 16);

        // Build spatial index
        list.ensure_spatial_index_built();
        assert!(!list.spatial_index.is_dirty());

        // Query top-left corner - should find 1 item
        let query = Bounds {
            origin: point(Pixels(0.0), Pixels(0.0)),
            size: size(Pixels(50.0), Pixels(50.0)),
        };
        let count = list.items_intersecting(query).count();
        assert_eq!(count, 1);

        // Query middle area - should find 4 items (2x2 grid center)
        let query = Bounds {
            origin: point(Pixels(90.0), Pixels(90.0)),
            size: size(Pixels(120.0), Pixels(120.0)),
        };
        let count = list.items_intersecting(query).count();
        assert_eq!(count, 4);

        // Query entire area - should find all 16 items
        let query = Bounds {
            origin: point(Pixels(0.0), Pixels(0.0)),
            size: size(Pixels(400.0), Pixels(400.0)),
        };
        let count = list.items_intersecting(query).count();
        assert_eq!(count, 16);
    }

    #[test]
    fn test_spatial_index_rebuild_on_clear() {
        let mut list = DisplayList::new(test_element_id());

        // Add items and build index
        list.push_quad(DisplayQuad {
            bounds: Bounds {
                origin: point(Pixels(0.0), Pixels(0.0)),
                size: size(Pixels(100.0), Pixels(100.0)),
            },
            world_clip: Bounds::default(),
            world_transform: TransformationMatrix::unit(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        });
        list.ensure_spatial_index_built();
        assert!(!list.spatial_index.is_dirty());

        // Clear should mark index dirty
        list.clear();
        assert!(list.spatial_index.is_dirty());
    }
}
