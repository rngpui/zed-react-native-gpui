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

/// A display list containing recorded drawing commands for a layer.
///
/// Display lists are created per scroll container and persist across frames
/// until their content changes. They can be rasterized to tiles on demand.
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
        }
    }

    /// Clear the display list and increment the generation.
    pub fn clear(&mut self) {
        self.items.clear();
        self.content_bounds = Bounds::default();
        self.generation += 1;
    }

    /// Iterate over items that intersect the given bounds.
    /// Used during tile rasterization to only process relevant items.
    pub fn items_intersecting(&self, bounds: Bounds<Pixels>) -> impl Iterator<Item = &DisplayItem> {
        self.items.iter().filter(move |item| item.intersects(&bounds))
    }

    /// Extend content_bounds to include the given bounds.
    pub fn extend_bounds(&mut self, bounds: Bounds<Pixels>) {
        if self.content_bounds.size.width.0 == 0.0 && self.content_bounds.size.height.0 == 0.0 {
            self.content_bounds = bounds;
        } else {
            self.content_bounds = self.content_bounds.union(&bounds);
        }
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
    /// This converts DisplayItems to Scene primitives, looking up transforms and
    /// clips from the PropertyTrees at render time. Coordinates are offset so that
    /// the tile's origin is at (0, 0) in the output. The scale factor converts
    /// from logical Pixels to device ScaledPixels.
    ///
    /// Returns vectors of primitives ready for GPU rendering.
    pub fn rasterize_tile(
        &self,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        property_trees: &mut PropertyTrees,
    ) -> TileRasterResult {
        let mut result = TileRasterResult::default();
        let mut draw_order: DrawOrder = 0;

        for item in self.items_intersecting(tile_bounds) {
            draw_order += 1;
            match item {
                DisplayItem::Quad(q) => {
                    if let Some((quad, transform)) = self.rasterize_quad(q, tile_bounds, scale_factor, draw_order, property_trees) {
                        result.quads.push(quad);
                        result.quad_transforms.push(transform);
                    }
                }
                DisplayItem::Shadow(s) => {
                    if let Some((shadow, transform)) = self.rasterize_shadow(s, tile_bounds, scale_factor, draw_order, property_trees) {
                        result.shadows.push(shadow);
                        result.shadow_transforms.push(transform);
                    }
                }
                DisplayItem::BackdropBlur(b) => {
                    if let Some((blur, transform)) = self.rasterize_backdrop_blur(b, tile_bounds, scale_factor, draw_order, property_trees) {
                        result.backdrop_blurs.push(blur);
                        result.backdrop_blur_transforms.push(transform);
                    }
                }
                DisplayItem::Underline(u) => {
                    if let Some((underline, transform)) = self.rasterize_underline(u, tile_bounds, scale_factor, draw_order, property_trees) {
                        result.underlines.push(underline);
                        result.underline_transforms.push(transform);
                    }
                }
                DisplayItem::MonochromeSprite(s) => {
                    if let Some(sprite) = self.rasterize_monochrome_sprite(s, tile_bounds, scale_factor, draw_order, property_trees) {
                        result.monochrome_sprites.push(sprite);
                    }
                }
                DisplayItem::SubpixelSprite(s) => {
                    if let Some(sprite) = self.rasterize_subpixel_sprite(s, tile_bounds, scale_factor, draw_order, property_trees) {
                        result.subpixel_sprites.push(sprite);
                    }
                }
                DisplayItem::PolychromeSprite(s) => {
                    if let Some((sprite, transform)) = self.rasterize_polychrome_sprite(s, tile_bounds, scale_factor, draw_order, property_trees) {
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

    fn rasterize_quad(
        &self,
        q: &DisplayQuad,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        property_trees: &mut PropertyTrees,
    ) -> Option<(Quad, TransformationMatrix)> {
        // Look up world transform and clip from property trees
        let world_transform = property_trees.world_transform(q.transform_node);
        let world_clip = property_trees.world_clip(q.clip_node);
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

    fn rasterize_shadow(
        &self,
        s: &DisplayShadow,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        property_trees: &mut PropertyTrees,
    ) -> Option<(Shadow, TransformationMatrix)> {
        let world_transform = property_trees.world_transform(s.transform_node);
        let world_clip = property_trees.world_clip(s.clip_node);
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

    fn rasterize_backdrop_blur(
        &self,
        b: &DisplayBackdropBlur,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        property_trees: &mut PropertyTrees,
    ) -> Option<(BackdropBlur, TransformationMatrix)> {
        let world_transform = property_trees.world_transform(b.transform_node);
        let world_clip = property_trees.world_clip(b.clip_node);
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

    fn rasterize_underline(
        &self,
        u: &DisplayUnderline,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        property_trees: &mut PropertyTrees,
    ) -> Option<(Underline, TransformationMatrix)> {
        let world_transform = property_trees.world_transform(u.transform_node);
        let world_clip = property_trees.world_clip(u.clip_node);
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

    fn rasterize_monochrome_sprite(
        &self,
        s: &DisplayMonochromeSprite,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        property_trees: &mut PropertyTrees,
    ) -> Option<MonochromeSprite> {
        let world_transform = property_trees.world_transform(s.transform_node);
        let world_clip = property_trees.world_clip(s.clip_node);
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

    fn rasterize_subpixel_sprite(
        &self,
        s: &DisplaySubpixelSprite,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        property_trees: &mut PropertyTrees,
    ) -> Option<SubpixelSprite> {
        let world_transform = property_trees.world_transform(s.transform_node);
        let world_clip = property_trees.world_clip(s.clip_node);
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

    fn rasterize_polychrome_sprite(
        &self,
        s: &DisplayPolychromeSprite,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
        property_trees: &mut PropertyTrees,
    ) -> Option<(PolychromeSprite, TransformationMatrix)> {
        let world_transform = property_trees.world_transform(s.transform_node);
        let world_clip = property_trees.world_clip(s.clip_node);
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
    use crate::{point, size};

    fn test_element_id() -> GlobalElementId {
        GlobalElementId::from(vec![1u32.into()])
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
}
