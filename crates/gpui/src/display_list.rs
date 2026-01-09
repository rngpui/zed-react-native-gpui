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
    property_trees::{TransformNodeId, ClipNodeId, PropertyTrees},
};
use collections::FxHashMap;
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

    /// Returns the transform node ID for this item.
    pub fn transform_node(&self) -> TransformNodeId {
        match self {
            DisplayItem::Quad(q) => q.transform_node,
            DisplayItem::Shadow(s) => s.transform_node,
            DisplayItem::BackdropBlur(b) => b.transform_node,
            DisplayItem::Underline(u) => u.transform_node,
            DisplayItem::MonochromeSprite(s) => s.transform_node,
            DisplayItem::SubpixelSprite(s) => s.transform_node,
            DisplayItem::PolychromeSprite(s) => s.transform_node,
            DisplayItem::Path(p) => p.transform_node,
        }
    }

    /// Returns the clip node ID for this item.
    pub fn clip_node(&self) -> ClipNodeId {
        match self {
            DisplayItem::Quad(q) => q.clip_node,
            DisplayItem::Shadow(s) => s.clip_node,
            DisplayItem::BackdropBlur(b) => b.clip_node,
            DisplayItem::Underline(u) => u.clip_node,
            DisplayItem::MonochromeSprite(s) => s.clip_node,
            DisplayItem::SubpixelSprite(s) => s.clip_node,
            DisplayItem::PolychromeSprite(s) => s.clip_node,
            DisplayItem::Path(p) => p.clip_node,
        }
    }
}

/// A quad (filled rectangle) in the display list.
#[derive(Clone, Debug)]
pub(crate) struct DisplayQuad {
    /// Bounds in logical pixels (local coordinates, not transformed).
    pub bounds: Bounds<Pixels>,
    /// Reference to transform tree node (enables O(1) transform updates).
    pub transform_node: TransformNodeId,
    /// Reference to clip tree node (enables O(1) clip updates).
    pub clip_node: ClipNodeId,
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
    /// Reference to transform tree node.
    pub transform_node: TransformNodeId,
    /// Reference to clip tree node.
    pub clip_node: ClipNodeId,
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
    /// Reference to transform tree node.
    pub transform_node: TransformNodeId,
    /// Reference to clip tree node.
    pub clip_node: ClipNodeId,
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
    /// Reference to transform tree node.
    pub transform_node: TransformNodeId,
    /// Reference to clip tree node.
    pub clip_node: ClipNodeId,
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
    /// Reference to transform tree node.
    pub transform_node: TransformNodeId,
    /// Reference to clip tree node.
    pub clip_node: ClipNodeId,
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
    /// Reference to transform tree node.
    pub transform_node: TransformNodeId,
    /// Reference to clip tree node.
    pub clip_node: ClipNodeId,
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
    /// Reference to transform tree node.
    pub transform_node: TransformNodeId,
    /// Reference to clip tree node.
    pub clip_node: ClipNodeId,
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
    /// Reference to transform tree node.
    pub transform_node: TransformNodeId,
    /// Reference to clip tree node.
    pub clip_node: ClipNodeId,
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

/// Tracks the range of display items owned by an element.
///
/// Used for incremental updates: when an element's content changes,
/// we can remove its items and insert new ones without rebuilding
/// the entire display list.
#[derive(Clone, Debug)]
pub(crate) struct ElementItemRange {
    /// Starting index in the items vec (inclusive).
    pub start: usize,
    /// Ending index in the items vec (exclusive).
    pub end: usize,
    /// Combined bounds of all items in this range.
    pub bounds: Bounds<Pixels>,
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
    /// Returns an iterator of item indices. Note: may include items that don't
    /// actually intersect (false positives from cell-level granularity), so
    /// callers should do a final intersection check.
    pub fn query(&self, bounds: Bounds<Pixels>) -> impl Iterator<Item = usize> + '_ {
        let (min_col, min_row, max_col, max_row) = self.cells_for_bounds(bounds);

        // Collect unique indices from overlapping cells
        let mut seen = collections::FxHashSet::default();

        (min_row..=max_row)
            .flat_map(move |row| (min_col..=max_col).map(move |col| row * self.cols + col))
            .filter(move |&cell_index| cell_index < self.cells.len())
            .flat_map(move |cell_index| self.cells[cell_index].iter().copied())
            .filter(move |&index| seen.insert(index))
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
    /// Index mapping element IDs to their item ranges.
    /// Enables O(1) lookup for partial updates.
    element_items: FxHashMap<GlobalElementId, ElementItemRange>,
    /// Stack of element IDs currently being painted.
    /// Used by begin_element()/end_element() to track nesting.
    element_stack: Vec<(GlobalElementId, usize)>,
    /// Pre-baked world transforms and clips for each item.
    /// Parallel array to `items` - index i contains the resolved transform/clip for items[i].
    /// Populated by `bake_transforms()` to avoid per-item property tree lookups during rasterization.
    baked_transforms: Vec<(TransformationMatrix, Bounds<Pixels>)>,
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
            element_items: FxHashMap::default(),
            element_stack: Vec::new(),
            baked_transforms: Vec::new(),
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
        self.element_items.clear();
        self.element_stack.clear();
        self.baked_transforms.clear();
        self.generation += 1;
    }

    // ========================================================================
    // Element Item Tracking (Phase 14: Incremental Invalidation)
    // ========================================================================

    /// Begin tracking items for an element.
    ///
    /// Call this before painting an element's items. Call `end_element()`
    /// when done. Items painted between begin/end are associated with
    /// the element ID for future partial updates.
    pub fn begin_element(&mut self, element_id: GlobalElementId) {
        let start_index = self.items.len();
        self.element_stack.push((element_id, start_index));
    }

    /// End tracking items for the current element.
    ///
    /// Records the item range for the element started with `begin_element()`.
    /// Returns the bounds of all items painted for this element.
    pub fn end_element(&mut self) -> Option<Bounds<Pixels>> {
        let (element_id, start_index) = self.element_stack.pop()?;
        let end_index = self.items.len();

        // Calculate bounds of items in this range
        let mut bounds: Bounds<Pixels> = Bounds::default();
        for item in &self.items[start_index..end_index] {
            if let Some(item_bounds) = item.bounds() {
                if bounds.size.width.0 == 0.0 && bounds.size.height.0 == 0.0 {
                    bounds = item_bounds;
                } else {
                    bounds = bounds.union(&item_bounds);
                }
            }
        }

        self.element_items.insert(
            element_id,
            ElementItemRange {
                start: start_index,
                end: end_index,
                bounds,
            },
        );

        Some(bounds)
    }

    /// Get the item range for an element.
    pub fn get_element_range(&self, element_id: &GlobalElementId) -> Option<&ElementItemRange> {
        self.element_items.get(element_id)
    }

    /// Get the bounds of items owned by an element.
    pub fn get_element_bounds(&self, element_id: &GlobalElementId) -> Option<Bounds<Pixels>> {
        self.element_items.get(element_id).map(|r| r.bounds)
    }

    /// Check if an element has items in this display list.
    pub fn has_element(&self, element_id: &GlobalElementId) -> bool {
        self.element_items.contains_key(element_id)
    }

    /// Update items for a specific element.
    ///
    /// This is the key method for incremental updates:
    /// 1. Removes the element's existing items
    /// 2. Inserts new items in their place
    /// 3. Marks both old and new regions as dirty
    ///
    /// Returns the combined dirty region (old bounds + new bounds).
    pub fn update_element(
        &mut self,
        element_id: &GlobalElementId,
        new_items: Vec<DisplayItem>,
    ) -> Option<Bounds<Pixels>> {
        // Get old range and bounds
        let old_range = self.element_items.remove(element_id)?;
        let old_bounds = old_range.bounds;

        // Calculate new bounds
        let mut new_bounds: Bounds<Pixels> = Bounds::default();
        for item in &new_items {
            if let Some(item_bounds) = item.bounds() {
                if new_bounds.size.width.0 == 0.0 && new_bounds.size.height.0 == 0.0 {
                    new_bounds = item_bounds;
                } else {
                    new_bounds = new_bounds.union(&item_bounds);
                }
            }
        }

        // Calculate size difference for index updates
        let old_count = old_range.end - old_range.start;
        let new_count = new_items.len();
        let count_diff = new_count as isize - old_count as isize;

        // Remove old items and insert new ones
        self.items.splice(old_range.start..old_range.end, new_items);

        // Update element range
        self.element_items.insert(
            element_id.clone(),
            ElementItemRange {
                start: old_range.start,
                end: old_range.start + new_count,
                bounds: new_bounds,
            },
        );

        // Update indices for elements that come after this one
        if count_diff != 0 {
            for range in self.element_items.values_mut() {
                if range.start > old_range.start {
                    range.start = (range.start as isize + count_diff) as usize;
                    range.end = (range.end as isize + count_diff) as usize;
                }
            }
        }

        // Mark both old and new bounds as dirty
        let dirty_bounds = old_bounds.union(&new_bounds);
        self.invalidate_region(dirty_bounds);

        Some(dirty_bounds)
    }

    /// Remove all items for an element.
    ///
    /// Marks the element's bounds as dirty.
    pub fn remove_element(&mut self, element_id: &GlobalElementId) -> Option<Bounds<Pixels>> {
        let range = self.element_items.remove(element_id)?;
        let bounds = range.bounds;

        // Remove items
        self.items.drain(range.start..range.end);

        // Update indices for elements that come after
        let count_diff = range.end - range.start;
        for r in self.element_items.values_mut() {
            if r.start > range.start {
                r.start -= count_diff;
                r.end -= count_diff;
            }
        }

        // Mark as dirty
        self.invalidate_region(bounds);

        Some(bounds)
    }

    /// Get the number of tracked elements.
    pub fn element_count(&self) -> usize {
        self.element_items.len()
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

    /// Pre-bake world transforms and clips for all items.
    ///
    /// Call this after property trees are finalized to avoid per-item lookups
    /// during rasterization. The baked values are stored in a parallel array
    /// and can be accessed via `get_baked_transform()`.
    pub fn bake_transforms(&mut self, property_trees: &mut PropertyTrees) {
        self.baked_transforms.clear();
        self.baked_transforms.reserve(self.items.len());

        for item in &self.items {
            let world_transform = property_trees.world_transform(item.transform_node());
            let world_clip = property_trees.world_clip(item.clip_node());
            self.baked_transforms.push((world_transform, world_clip));
        }
    }

    /// Get the pre-baked world transform and clip for an item by index.
    ///
    /// Returns None if transforms haven't been baked or index is out of bounds.
    #[inline]
    pub fn get_baked_transform(&self, index: usize) -> Option<&(TransformationMatrix, Bounds<Pixels>)> {
        self.baked_transforms.get(index)
    }

    /// Iterate over items that intersect the given bounds.
    ///
    /// Uses spatial index for O(cells × items_per_cell) performance.
    ///
    /// The spatial index must be built before calling this method.
    /// Call `ensure_spatial_index_built()` before rasterization.
    pub fn items_intersecting(&self, bounds: Bounds<Pixels>) -> impl Iterator<Item = &DisplayItem> + '_ {
        // Spatial index must be built before querying - if this panics, ensure_spatial_index_built()
        // wasn't called before the display list was used for rasterization.
        debug_assert!(
            !self.spatial_index.is_dirty(),
            "Spatial index not built before items_intersecting() - call ensure_spatial_index_built() first"
        );

        ItemsIntersectingIter {
            items: &self.items,
            indices: self.spatial_index.query(bounds).collect::<Vec<_>>().into_iter(),
            query_bounds: bounds,
        }
    }

    /// Iterate over items that intersect the given bounds, with their indices.
    ///
    /// Returns (index, &item) pairs so callers can also look up baked transforms
    /// via `get_baked_transform(index)`.
    pub fn items_intersecting_with_index(&self, bounds: Bounds<Pixels>) -> impl Iterator<Item = (usize, &DisplayItem)> + '_ {
        debug_assert!(
            !self.spatial_index.is_dirty(),
            "Spatial index not built before items_intersecting_with_index() - call ensure_spatial_index_built() first"
        );

        self.spatial_index
            .query(bounds)
            .collect::<Vec<_>>()
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
    /// Uses pre-baked transforms from `bake_transforms()` for efficient rasterization.
    /// Falls back to property tree lookups if transforms are not baked.
    /// Coordinates are offset so that the tile's origin is at (0, 0) in the output.
    /// The scale factor converts from logical Pixels to device ScaledPixels.
    ///
    /// Returns vectors of primitives ready for GPU rendering.
    pub fn rasterize_tile(
        &self,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        _property_trees: &mut PropertyTrees,
    ) -> TileRasterResult {
        debug_assert!(
            !self.baked_transforms.is_empty() || self.items.is_empty(),
            "bake_transforms() must be called before rasterize_tile()"
        );

        let mut result = TileRasterResult::default();
        let mut draw_order: DrawOrder = 0;

        for (index, item) in self.items_intersecting_with_index(tile_bounds) {
            draw_order += 1;

            // Get pre-baked world transform and clip
            let &(world_transform, world_clip) = &self.baked_transforms[index];

            match item {
                DisplayItem::Quad(q) => {
                    if let Some((quad, transform)) = self.rasterize_quad(q, tile_bounds, scale_factor, draw_order, world_transform, world_clip) {
                        result.quads.push(quad);
                        result.quad_transforms.push(transform);
                    }
                }
                DisplayItem::Shadow(s) => {
                    if let Some((shadow, transform)) = self.rasterize_shadow(s, tile_bounds, scale_factor, draw_order, world_transform, world_clip) {
                        result.shadows.push(shadow);
                        result.shadow_transforms.push(transform);
                    }
                }
                DisplayItem::BackdropBlur(b) => {
                    if let Some((blur, transform)) = self.rasterize_backdrop_blur(b, tile_bounds, scale_factor, draw_order, world_transform, world_clip) {
                        result.backdrop_blurs.push(blur);
                        result.backdrop_blur_transforms.push(transform);
                    }
                }
                DisplayItem::Underline(u) => {
                    if let Some((underline, transform)) = self.rasterize_underline(u, tile_bounds, scale_factor, draw_order, world_transform, world_clip) {
                        result.underlines.push(underline);
                        result.underline_transforms.push(transform);
                    }
                }
                DisplayItem::MonochromeSprite(s) => {
                    if let Some(sprite) = self.rasterize_monochrome_sprite(s, tile_bounds, scale_factor, draw_order, world_transform, world_clip) {
                        result.monochrome_sprites.push(sprite);
                    }
                }
                DisplayItem::SubpixelSprite(s) => {
                    if let Some(sprite) = self.rasterize_subpixel_sprite(s, tile_bounds, scale_factor, draw_order, world_transform, world_clip) {
                        result.subpixel_sprites.push(sprite);
                    }
                }
                DisplayItem::PolychromeSprite(s) => {
                    if let Some((sprite, transform)) = self.rasterize_polychrome_sprite(s, tile_bounds, scale_factor, draw_order, world_transform, world_clip) {
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
/// painting. New code should create DisplayItems directly with property tree node references.
///
/// `content_origin` is the window-space origin of the scroll container in ScaledPixels.
/// This is subtracted from all bounds to convert from window coordinates to content coordinates.
/// `inv_scale` is 1.0 / scale_factor to convert from ScaledPixels to Pixels.
/// `transform_node` and `clip_node` are the property tree node IDs to reference.
pub fn convert_primitive_to_display_item(
    primitive: &crate::scene::Primitive,
    inv_scale: f32,
    content_origin: Point<crate::ScaledPixels>,
    transform_node: TransformNodeId,
    clip_node: ClipNodeId,
) -> Option<DisplayItem> {
    use crate::scene::Primitive;

    match primitive {
        Primitive::Quad(quad, _transform) => {
            Some(DisplayItem::Quad(DisplayQuad {
                bounds: offset_and_scale_bounds_to_pixels(&quad.bounds, content_origin, inv_scale),
                transform_node,
                clip_node,
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
                transform_node,
                clip_node,
                color: shadow.color,
                blur_radius: Pixels(shadow.blur_radius.0 * inv_scale),
                corner_radii: scale_corners_to_pixels(&shadow.corner_radii, inv_scale),
            }))
        }
        Primitive::BackdropBlur(blur, _transform) => {
            Some(DisplayItem::BackdropBlur(DisplayBackdropBlur {
                bounds: offset_and_scale_bounds_to_pixels(&blur.bounds, content_origin, inv_scale),
                transform_node,
                clip_node,
                blur_radius: Pixels(blur.blur_radius.0 * inv_scale),
                corner_radii: scale_corners_to_pixels(&blur.corner_radii, inv_scale),
                tint: blur.tint,
            }))
        }
        Primitive::Underline(underline, _transform) => {
            Some(DisplayItem::Underline(DisplayUnderline {
                bounds: offset_and_scale_bounds_to_pixels(&underline.bounds, content_origin, inv_scale),
                transform_node,
                clip_node,
                color: underline.color,
                thickness: Pixels(underline.thickness.0 * inv_scale),
                wavy: underline.wavy != 0,
            }))
        }
        Primitive::MonochromeSprite(sprite) => {
            Some(DisplayItem::MonochromeSprite(DisplayMonochromeSprite {
                bounds: offset_and_scale_bounds_to_pixels(&sprite.bounds, content_origin, inv_scale),
                transform_node,
                clip_node,
                color: sprite.color,
                tile: sprite.tile.clone(),
            }))
        }
        Primitive::SubpixelSprite(sprite) => {
            Some(DisplayItem::SubpixelSprite(DisplaySubpixelSprite {
                bounds: offset_and_scale_bounds_to_pixels(&sprite.bounds, content_origin, inv_scale),
                transform_node,
                clip_node,
                color: sprite.color,
                tile: sprite.tile.clone(),
            }))
        }
        Primitive::PolychromeSprite(sprite, _transform) => {
            Some(DisplayItem::PolychromeSprite(DisplayPolychromeSprite {
                bounds: offset_and_scale_bounds_to_pixels(&sprite.bounds, content_origin, inv_scale),
                transform_node,
                clip_node,
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
                transform_node,
                clip_node,
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
            transform_node: TransformNodeId::ROOT,
            clip_node: ClipNodeId::ROOT,
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
            transform_node: TransformNodeId::ROOT,
            clip_node: ClipNodeId::ROOT,
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
            transform_node: TransformNodeId::ROOT,
            clip_node: ClipNodeId::ROOT,
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
            transform_node: TransformNodeId::ROOT,
            clip_node: ClipNodeId::ROOT,
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
            transform_node: TransformNodeId::ROOT,
            clip_node: ClipNodeId::ROOT,
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
    // Element Item Tracking Tests (Phase 14: Incremental Invalidation)
    // ========================================================================

    fn test_element_id_2() -> GlobalElementId {
        GlobalElementId(Arc::from([ElementId::Integer(2)]))
    }

    fn test_element_id_3() -> GlobalElementId {
        GlobalElementId(Arc::from([ElementId::Integer(3)]))
    }

    fn make_test_quad(x: f32, y: f32, w: f32, h: f32) -> DisplayQuad {
        DisplayQuad {
            bounds: Bounds {
                origin: point(Pixels(x), Pixels(y)),
                size: size(Pixels(w), Pixels(h)),
            },
            transform_node: TransformNodeId::ROOT,
            clip_node: ClipNodeId::ROOT,
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    #[test]
    fn test_element_tracking_begin_end() {
        let mut list = DisplayList::new(test_element_id());

        // Begin element 1
        list.begin_element(test_element_id());
        list.push_quad(make_test_quad(0.0, 0.0, 100.0, 100.0));
        list.push_quad(make_test_quad(50.0, 50.0, 100.0, 100.0));
        let bounds = list.end_element();

        // Should have recorded the element
        assert!(list.has_element(&test_element_id()));
        assert_eq!(list.element_count(), 1);

        // Bounds should cover both quads: (0,0) to (150,150)
        let bounds = bounds.unwrap();
        assert_eq!(bounds.origin.x, Pixels(0.0));
        assert_eq!(bounds.origin.y, Pixels(0.0));
        assert_eq!(bounds.size.width, Pixels(150.0));
        assert_eq!(bounds.size.height, Pixels(150.0));

        // Should have correct range
        let range = list.get_element_range(&test_element_id()).unwrap();
        assert_eq!(range.start, 0);
        assert_eq!(range.end, 2);
    }

    #[test]
    fn test_element_update() {
        let mut list = DisplayList::new(test_element_id());

        // Add element 1 with 2 items
        list.begin_element(test_element_id());
        list.push_quad(make_test_quad(0.0, 0.0, 100.0, 100.0));
        list.push_quad(make_test_quad(100.0, 0.0, 100.0, 100.0));
        list.end_element();

        // Add element 2 with 1 item
        list.begin_element(test_element_id_2());
        list.push_quad(make_test_quad(0.0, 200.0, 100.0, 100.0));
        list.end_element();

        assert_eq!(list.items.len(), 3);

        // Update element 1 with different items (1 item instead of 2)
        let new_items = vec![DisplayItem::Quad(make_test_quad(0.0, 0.0, 50.0, 50.0))];
        let dirty_bounds = list.update_element(&test_element_id(), new_items);

        // Should have dirty bounds
        assert!(dirty_bounds.is_some());

        // List should now have 2 items total
        assert_eq!(list.items.len(), 2);

        // Element 1 range should be updated
        let range = list.get_element_range(&test_element_id()).unwrap();
        assert_eq!(range.start, 0);
        assert_eq!(range.end, 1);

        // Element 2 range should be shifted
        let range2 = list.get_element_range(&test_element_id_2()).unwrap();
        assert_eq!(range2.start, 1);
        assert_eq!(range2.end, 2);

        // Dirty regions should be set
        assert!(list.has_dirty_regions());
    }

    #[test]
    fn test_element_remove() {
        let mut list = DisplayList::new(test_element_id());

        // Add 3 elements
        list.begin_element(test_element_id());
        list.push_quad(make_test_quad(0.0, 0.0, 100.0, 100.0));
        list.end_element();

        list.begin_element(test_element_id_2());
        list.push_quad(make_test_quad(100.0, 0.0, 100.0, 100.0));
        list.end_element();

        list.begin_element(test_element_id_3());
        list.push_quad(make_test_quad(200.0, 0.0, 100.0, 100.0));
        list.end_element();

        assert_eq!(list.items.len(), 3);
        assert_eq!(list.element_count(), 3);

        // Remove element 2
        let removed_bounds = list.remove_element(&test_element_id_2());
        assert!(removed_bounds.is_some());

        // Should have 2 items and 2 elements
        assert_eq!(list.items.len(), 2);
        assert_eq!(list.element_count(), 2);

        // Element 2 should be gone
        assert!(!list.has_element(&test_element_id_2()));

        // Element 3 range should be shifted
        let range3 = list.get_element_range(&test_element_id_3()).unwrap();
        assert_eq!(range3.start, 1);
        assert_eq!(range3.end, 2);

        // Dirty regions should be set
        assert!(list.has_dirty_regions());
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
                    transform_node: TransformNodeId::ROOT,
                    clip_node: ClipNodeId::ROOT,
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
        list.push_quad(make_test_quad(0.0, 0.0, 100.0, 100.0));
        list.ensure_spatial_index_built();
        assert!(!list.spatial_index.is_dirty());

        // Clear should mark index dirty
        list.clear();
        assert!(list.spatial_index.is_dirty());
    }
}
