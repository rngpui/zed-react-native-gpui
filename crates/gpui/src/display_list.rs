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
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for a display list layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DisplayListId(pub u64);

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
pub enum DisplayItem {
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
    /// Start of a clipping layer.
    PushClip(ContentMask<Pixels>),
    /// End of a clipping layer.
    PopClip,
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
            DisplayItem::PushClip(_) | DisplayItem::PopClip => None,
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
pub struct DisplayQuad {
    /// Bounds in logical pixels.
    pub bounds: Bounds<Pixels>,
    /// Content mask for clipping.
    pub content_mask: ContentMask<Pixels>,
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
    /// Transformation matrix.
    pub transform: TransformationMatrix,
}

/// A shadow in the display list.
#[derive(Clone, Debug)]
pub struct DisplayShadow {
    /// Bounds in logical pixels.
    pub bounds: Bounds<Pixels>,
    /// Content mask for clipping.
    pub content_mask: ContentMask<Pixels>,
    /// Shadow color.
    pub color: Hsla,
    /// Blur radius in logical pixels.
    pub blur_radius: Pixels,
    /// Corner radii.
    pub corner_radii: Corners<Pixels>,
    /// Transformation matrix.
    pub transform: TransformationMatrix,
}

/// A backdrop blur in the display list.
#[derive(Clone, Debug)]
pub struct DisplayBackdropBlur {
    /// Bounds in logical pixels.
    pub bounds: Bounds<Pixels>,
    /// Content mask for clipping.
    pub content_mask: ContentMask<Pixels>,
    /// Blur radius in logical pixels.
    pub blur_radius: Pixels,
    /// Corner radii.
    pub corner_radii: Corners<Pixels>,
    /// Tint color applied over the blur.
    pub tint: Hsla,
    /// Transformation matrix.
    pub transform: TransformationMatrix,
}

/// An underline or strikethrough in the display list.
#[derive(Clone, Debug)]
pub struct DisplayUnderline {
    /// Bounds in logical pixels.
    pub bounds: Bounds<Pixels>,
    /// Content mask for clipping.
    pub content_mask: ContentMask<Pixels>,
    /// Line color.
    pub color: Hsla,
    /// Line thickness in logical pixels.
    pub thickness: Pixels,
    /// Whether the line is wavy.
    pub wavy: bool,
    /// Transformation matrix.
    pub transform: TransformationMatrix,
}

/// A monochrome sprite (glyph) in the display list.
#[derive(Clone, Debug)]
pub struct DisplayMonochromeSprite {
    /// Bounds in logical pixels.
    pub bounds: Bounds<Pixels>,
    /// Content mask for clipping.
    pub content_mask: ContentMask<Pixels>,
    /// Glyph color.
    pub color: Hsla,
    /// Atlas tile containing the glyph.
    pub tile: AtlasTile,
    /// Transformation matrix.
    pub transform: TransformationMatrix,
}

/// A subpixel sprite (glyph with subpixel AA) in the display list.
#[derive(Clone, Debug)]
pub struct DisplaySubpixelSprite {
    /// Bounds in logical pixels.
    pub bounds: Bounds<Pixels>,
    /// Content mask for clipping.
    pub content_mask: ContentMask<Pixels>,
    /// Glyph color.
    pub color: Hsla,
    /// Atlas tile containing the glyph.
    pub tile: AtlasTile,
    /// Transformation matrix.
    pub transform: TransformationMatrix,
}

/// A polychrome sprite (color image/emoji) in the display list.
#[derive(Clone, Debug)]
pub struct DisplayPolychromeSprite {
    /// Bounds in logical pixels.
    pub bounds: Bounds<Pixels>,
    /// Content mask for clipping.
    pub content_mask: ContentMask<Pixels>,
    /// Whether to render in grayscale.
    pub grayscale: bool,
    /// Opacity (0.0 to 1.0).
    pub opacity: f32,
    /// Corner radii for clipping.
    pub corner_radii: Corners<Pixels>,
    /// Atlas tile containing the image.
    pub tile: AtlasTile,
    /// Transformation matrix.
    pub transform: TransformationMatrix,
}

/// A vector path in the display list.
#[derive(Clone, Debug)]
pub struct DisplayPath {
    /// Bounds in logical pixels.
    pub bounds: Bounds<Pixels>,
    /// Content mask for clipping.
    pub content_mask: ContentMask<Pixels>,
    /// Path fill color.
    pub color: Background,
    /// Path vertices.
    pub vertices: Vec<PathVertex>,
}

/// A vertex in a path.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathVertex {
    /// Position in logical pixels.
    pub xy_position: Point<Pixels>,
    /// Position along the path (0.0 to 1.0).
    pub st_position: Point<f32>,
    /// Content mask at this vertex.
    pub content_mask: ContentMask<Pixels>,
}

/// A display list containing recorded drawing commands for a layer.
///
/// Display lists are created per scroll container and persist across frames
/// until their content changes. They can be rasterized to tiles on demand.
#[derive(Clone, Debug)]
pub struct DisplayList {
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

    /// Rasterize items intersecting the tile bounds to Scene primitives.
    ///
    /// This converts DisplayItems to Scene primitives, offsetting coordinates
    /// so that the tile's origin is at (0, 0) in the output. The scale factor
    /// converts from logical Pixels to device ScaledPixels.
    ///
    /// Returns vectors of primitives ready for GPU rendering.
    pub fn rasterize_tile(
        &self,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
    ) -> TileRasterResult {
        let mut result = TileRasterResult::default();
        let mut draw_order: DrawOrder = 0;

        for item in self.items_intersecting(tile_bounds) {
            draw_order += 1;
            match item {
                DisplayItem::Quad(q) => {
                    if let Some(quad) = self.rasterize_quad(q, tile_bounds, scale_factor, draw_order) {
                        result.quads.push(quad);
                        result.quad_transforms.push(TransformationMatrix::unit());
                    }
                }
                DisplayItem::Shadow(s) => {
                    if let Some(shadow) = self.rasterize_shadow(s, tile_bounds, scale_factor, draw_order) {
                        result.shadows.push(shadow);
                        result.shadow_transforms.push(TransformationMatrix::unit());
                    }
                }
                DisplayItem::BackdropBlur(b) => {
                    if let Some(blur) = self.rasterize_backdrop_blur(b, tile_bounds, scale_factor, draw_order) {
                        result.backdrop_blurs.push(blur);
                        result.backdrop_blur_transforms.push(TransformationMatrix::unit());
                    }
                }
                DisplayItem::Underline(u) => {
                    if let Some(underline) = self.rasterize_underline(u, tile_bounds, scale_factor, draw_order) {
                        result.underlines.push(underline);
                        result.underline_transforms.push(TransformationMatrix::unit());
                    }
                }
                DisplayItem::MonochromeSprite(s) => {
                    if let Some(sprite) = self.rasterize_monochrome_sprite(s, tile_bounds, scale_factor, draw_order) {
                        result.monochrome_sprites.push(sprite);
                    }
                }
                DisplayItem::SubpixelSprite(s) => {
                    if let Some(sprite) = self.rasterize_subpixel_sprite(s, tile_bounds, scale_factor, draw_order) {
                        result.subpixel_sprites.push(sprite);
                    }
                }
                DisplayItem::PolychromeSprite(s) => {
                    if let Some(sprite) = self.rasterize_polychrome_sprite(s, tile_bounds, scale_factor, draw_order) {
                        result.polychrome_sprites.push(sprite);
                        result.polychrome_sprite_transforms.push(TransformationMatrix::unit());
                    }
                }
                DisplayItem::Path(_) => {
                    // TODO: Path rasterization requires more complex handling
                }
                DisplayItem::PushClip(_) | DisplayItem::PopClip => {
                    // Clips are handled by the content_mask on each primitive
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
    ) -> Option<Quad> {
        Some(Quad {
            order,
            border_style: q.border_style,
            bounds: self.offset_and_scale_bounds(q.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&q.content_mask, tile_bounds.origin, scale_factor),
            background: q.background.clone(),
            border_color: q.border_color,
            corner_radii: self.scale_corners(&q.corner_radii, scale_factor),
            border_widths: self.scale_edges(&q.border_widths, scale_factor),
        })
    }

    fn rasterize_shadow(
        &self,
        s: &DisplayShadow,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
    ) -> Option<Shadow> {
        Some(Shadow {
            order,
            blur_radius: ScaledPixels(s.blur_radius.0 * scale_factor),
            bounds: self.offset_and_scale_bounds(s.bounds, tile_bounds.origin, scale_factor),
            corner_radii: self.scale_corners(&s.corner_radii, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&s.content_mask, tile_bounds.origin, scale_factor),
            color: s.color,
        })
    }

    fn rasterize_backdrop_blur(
        &self,
        b: &DisplayBackdropBlur,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
    ) -> Option<BackdropBlur> {
        Some(BackdropBlur {
            order,
            blur_radius: ScaledPixels(b.blur_radius.0 * scale_factor),
            bounds: self.offset_and_scale_bounds(b.bounds, tile_bounds.origin, scale_factor),
            corner_radii: self.scale_corners(&b.corner_radii, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&b.content_mask, tile_bounds.origin, scale_factor),
            tint: b.tint,
        })
    }

    fn rasterize_underline(
        &self,
        u: &DisplayUnderline,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
    ) -> Option<Underline> {
        Some(Underline {
            order,
            pad: 0,
            bounds: self.offset_and_scale_bounds(u.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&u.content_mask, tile_bounds.origin, scale_factor),
            color: u.color,
            thickness: ScaledPixels(u.thickness.0 * scale_factor),
            wavy: if u.wavy { 1 } else { 0 },
        })
    }

    fn rasterize_monochrome_sprite(
        &self,
        s: &DisplayMonochromeSprite,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
    ) -> Option<MonochromeSprite> {
        Some(MonochromeSprite {
            order,
            pad: 0,
            bounds: self.offset_and_scale_bounds(s.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&s.content_mask, tile_bounds.origin, scale_factor),
            color: s.color,
            tile: s.tile.clone(),
            transformation: s.transform,
        })
    }

    fn rasterize_subpixel_sprite(
        &self,
        s: &DisplaySubpixelSprite,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
    ) -> Option<SubpixelSprite> {
        Some(SubpixelSprite {
            order,
            pad: 0,
            bounds: self.offset_and_scale_bounds(s.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&s.content_mask, tile_bounds.origin, scale_factor),
            color: s.color,
            tile: s.tile.clone(),
            transformation: s.transform,
        })
    }

    fn rasterize_polychrome_sprite(
        &self,
        s: &DisplayPolychromeSprite,
        tile_bounds: Bounds<Pixels>,
        scale_factor: f32,
        order: DrawOrder,
    ) -> Option<PolychromeSprite> {
        Some(PolychromeSprite {
            order,
            pad: 0,
            grayscale: s.grayscale,
            opacity: s.opacity,
            bounds: self.offset_and_scale_bounds(s.bounds, tile_bounds.origin, scale_factor),
            content_mask: self.offset_and_scale_content_mask(&s.content_mask, tile_bounds.origin, scale_factor),
            corner_radii: self.scale_corners(&s.corner_radii, scale_factor),
            tile: s.tile.clone(),
        })
    }
}

/// Result of rasterizing a tile from a display list.
/// Contains all primitives ready for GPU rendering.
#[derive(Default)]
pub struct TileRasterResult {
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
pub struct DisplayListManager {
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
/// `content_origin` is the window-space origin of the scroll container in ScaledPixels.
/// This is subtracted from all bounds to convert from window coordinates to content coordinates.
/// `inv_scale` is 1.0 / scale_factor to convert from ScaledPixels to Pixels.
pub fn convert_primitive_to_display_item(
    primitive: &crate::scene::Primitive,
    inv_scale: f32,
    content_origin: Point<crate::ScaledPixels>,
) -> Option<DisplayItem> {
    use crate::scene::Primitive;

    match primitive {
        Primitive::Quad(quad, transform) => {
            Some(DisplayItem::Quad(DisplayQuad {
                bounds: offset_and_scale_bounds_to_pixels(&quad.bounds, content_origin, inv_scale),
                content_mask: offset_and_scale_content_mask_to_pixels(&quad.content_mask, content_origin, inv_scale),
                background: quad.background.clone(),
                border_color: quad.border_color,
                border_style: quad.border_style,
                corner_radii: scale_corners_to_pixels(&quad.corner_radii, inv_scale),
                border_widths: scale_edges_to_pixels(&quad.border_widths, inv_scale),
                transform: *transform,
            }))
        }
        Primitive::Shadow(shadow, transform) => {
            Some(DisplayItem::Shadow(DisplayShadow {
                bounds: offset_and_scale_bounds_to_pixels(&shadow.bounds, content_origin, inv_scale),
                content_mask: offset_and_scale_content_mask_to_pixels(&shadow.content_mask, content_origin, inv_scale),
                color: shadow.color,
                blur_radius: Pixels(shadow.blur_radius.0 * inv_scale),
                corner_radii: scale_corners_to_pixels(&shadow.corner_radii, inv_scale),
                transform: *transform,
            }))
        }
        Primitive::BackdropBlur(blur, transform) => {
            Some(DisplayItem::BackdropBlur(DisplayBackdropBlur {
                bounds: offset_and_scale_bounds_to_pixels(&blur.bounds, content_origin, inv_scale),
                content_mask: offset_and_scale_content_mask_to_pixels(&blur.content_mask, content_origin, inv_scale),
                blur_radius: Pixels(blur.blur_radius.0 * inv_scale),
                corner_radii: scale_corners_to_pixels(&blur.corner_radii, inv_scale),
                tint: blur.tint,
                transform: *transform,
            }))
        }
        Primitive::Underline(underline, transform) => {
            Some(DisplayItem::Underline(DisplayUnderline {
                bounds: offset_and_scale_bounds_to_pixels(&underline.bounds, content_origin, inv_scale),
                content_mask: offset_and_scale_content_mask_to_pixels(&underline.content_mask, content_origin, inv_scale),
                color: underline.color,
                thickness: Pixels(underline.thickness.0 * inv_scale),
                wavy: underline.wavy != 0,
                transform: *transform,
            }))
        }
        Primitive::MonochromeSprite(sprite) => {
            Some(DisplayItem::MonochromeSprite(DisplayMonochromeSprite {
                bounds: offset_and_scale_bounds_to_pixels(&sprite.bounds, content_origin, inv_scale),
                content_mask: offset_and_scale_content_mask_to_pixels(&sprite.content_mask, content_origin, inv_scale),
                color: sprite.color,
                tile: sprite.tile.clone(),
                transform: sprite.transformation,
            }))
        }
        Primitive::SubpixelSprite(sprite) => {
            Some(DisplayItem::SubpixelSprite(DisplaySubpixelSprite {
                bounds: offset_and_scale_bounds_to_pixels(&sprite.bounds, content_origin, inv_scale),
                content_mask: offset_and_scale_content_mask_to_pixels(&sprite.content_mask, content_origin, inv_scale),
                color: sprite.color,
                tile: sprite.tile.clone(),
                transform: sprite.transformation,
            }))
        }
        Primitive::PolychromeSprite(sprite, transform) => {
            Some(DisplayItem::PolychromeSprite(DisplayPolychromeSprite {
                bounds: offset_and_scale_bounds_to_pixels(&sprite.bounds, content_origin, inv_scale),
                content_mask: offset_and_scale_content_mask_to_pixels(&sprite.content_mask, content_origin, inv_scale),
                grayscale: sprite.grayscale,
                opacity: sprite.opacity,
                corner_radii: scale_corners_to_pixels(&sprite.corner_radii, inv_scale),
                tile: sprite.tile.clone(),
                transform: *transform,
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
                    content_mask: offset_and_scale_content_mask_to_pixels(&v.content_mask, content_origin, inv_scale),
                })
                .collect();

            Some(DisplayItem::Path(DisplayPath {
                bounds: offset_and_scale_bounds_to_pixels(&path.bounds, content_origin, inv_scale),
                content_mask: offset_and_scale_content_mask_to_pixels(&path.content_mask, content_origin, inv_scale),
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
            content_mask: ContentMask::default(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            transform: TransformationMatrix::unit(),
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
            content_mask: ContentMask::default(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            transform: TransformationMatrix::unit(),
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
            content_mask: ContentMask::default(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            transform: TransformationMatrix::unit(),
        });

        // Add a quad at (200, 200) to (300, 300)
        list.push_quad(DisplayQuad {
            bounds: Bounds {
                origin: point(Pixels(200.0), Pixels(200.0)),
                size: size(Pixels(100.0), Pixels(100.0)),
            },
            content_mask: ContentMask::default(),
            background: Background::default(),
            border_color: Hsla::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
            transform: TransformationMatrix::unit(),
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
