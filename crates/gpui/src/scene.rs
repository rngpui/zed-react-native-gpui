// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, GlobalElementId,
    Hsla, Pixels, Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
    sync::atomic::{AtomicU64, Ordering},
};

#[allow(non_camel_case_types, unused)]
pub(crate) type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

pub(crate) type DrawOrder = u32;

/// Statistics about scene dirty state for a single frame.
/// Used to track which parts of the scene changed and could benefit from partial GPU updates.
#[derive(Default, Clone, Copy, Debug)]
pub struct SceneDirtyStats {
    /// Total number of primitives in the scene
    pub total_primitives: usize,
    /// Number of primitives that were freshly painted (not from cache)
    pub dirty_primitives: usize,
    /// Number of quads that were freshly painted
    pub dirty_quads: usize,
    /// Number of paths that were freshly painted
    pub dirty_paths: usize,
    /// Number of shadows that were freshly painted
    pub dirty_shadows: usize,
    /// Number of underlines that were freshly painted
    pub dirty_underlines: usize,
    /// Number of monochrome sprites that were freshly painted
    pub dirty_monochrome_sprites: usize,
    /// Number of subpixel sprites that were freshly painted
    pub dirty_subpixel_sprites: usize,
    /// Number of polychrome sprites that were freshly painted
    pub dirty_polychrome_sprites: usize,
    /// Number of surfaces that were freshly painted
    pub dirty_surfaces: usize,
    /// Number of backdrop blurs that were freshly painted
    pub dirty_backdrop_blurs: usize,
}

impl SceneDirtyStats {
    /// Returns true if no primitives were freshly painted (all from cache)
    pub fn is_fully_cached(&self) -> bool {
        self.dirty_primitives == 0
    }

    /// Returns the cache hit rate as a percentage (0.0 to 100.0)
    pub fn cache_hit_rate(&self) -> f32 {
        if self.total_primitives == 0 {
            100.0
        } else {
            let cached = self.total_primitives.saturating_sub(self.dirty_primitives);
            (cached as f32 / self.total_primitives as f32) * 100.0
        }
    }
}

/// Represents a captured subtree that should be rendered to a texture.
/// Used for render-to-texture caching of subtrees during scrolling.
#[derive(Clone, Debug)]
pub(crate) struct SubtreeCapture {
    /// The element ID of the subtree root
    pub id: GlobalElementId,
    /// Screen bounds of the subtree
    pub bounds: Bounds<ScaledPixels>,
    /// Content mask applied to this subtree
    pub content_mask: ContentMask<ScaledPixels>,
    /// Index of the first paint operation in this subtree
    pub primitive_start: usize,
    /// Index past the last paint operation in this subtree (exclusive)
    pub primitive_end: usize,
}

#[derive(Default)]
pub(crate) struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    generation: u64,
    dirty_ranges: Vec<Range<usize>>,
    dirty_stats: SceneDirtyStats,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub(crate) shadows: Vec<Shadow>,
    pub(crate) shadow_transforms: Vec<TransformationMatrix>,
    pub(crate) quads: Vec<Quad>,
    pub(crate) quad_transforms: Vec<TransformationMatrix>,
    pub(crate) backdrop_blurs: Vec<BackdropBlur>,
    pub(crate) backdrop_blur_transforms: Vec<TransformationMatrix>,
    pub(crate) paths: Vec<Path<ScaledPixels>>,
    pub(crate) underlines: Vec<Underline>,
    pub(crate) underline_transforms: Vec<TransformationMatrix>,
    pub(crate) monochrome_sprites: Vec<MonochromeSprite>,
    pub(crate) subpixel_sprites: Vec<SubpixelSprite>,
    pub(crate) polychrome_sprites: Vec<PolychromeSprite>,
    pub(crate) polychrome_sprite_transforms: Vec<TransformationMatrix>,
    pub(crate) surfaces: Vec<PaintSurface>,
    pub(crate) cached_textures: Vec<CachedTextureSprite>,
    /// Stack of subtrees currently being captured
    pending_subtree_captures: Vec<SubtreeCapture>,
    /// Completed subtree captures ready for render-to-texture
    pub(crate) completed_subtree_captures: Vec<SubtreeCapture>,
}

impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.generation = self.generation.wrapping_add(1);
        self.dirty_ranges.clear();
        self.dirty_stats = SceneDirtyStats::default();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.shadow_transforms.clear();
        self.quads.clear();
        self.quad_transforms.clear();
        self.backdrop_blurs.clear();
        self.backdrop_blur_transforms.clear();
        self.underlines.clear();
        self.underline_transforms.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.polychrome_sprite_transforms.clear();
        self.surfaces.clear();
        self.cached_textures.clear();
        self.pending_subtree_captures.clear();
        self.completed_subtree_captures.clear();
    }

    /// Begin capturing primitives for a subtree that may be cached to a texture.
    /// Call this before painting the subtree's content.
    pub fn begin_subtree_capture(
        &mut self,
        id: GlobalElementId,
        bounds: Bounds<ScaledPixels>,
        content_mask: ContentMask<ScaledPixels>,
    ) {
        self.pending_subtree_captures.push(SubtreeCapture {
            id,
            bounds,
            content_mask,
            primitive_start: self.paint_operations.len(),
            primitive_end: 0, // Will be set in end_subtree_capture
        });
    }

    /// End subtree capture and mark it for render-to-texture.
    /// Returns the capture if successful.
    pub fn end_subtree_capture(&mut self) -> Option<SubtreeCapture> {
        self.pending_subtree_captures.pop().map(|mut capture| {
            capture.primitive_end = self.paint_operations.len();
            // Only add to completed if it has actual content
            if capture.primitive_end > capture.primitive_start {
                self.completed_subtree_captures.push(capture.clone());
            }
            capture
        })
    }

    /// Returns the completed subtree captures for this frame.
    pub fn subtree_captures(&self) -> &[SubtreeCapture] {
        &self.completed_subtree_captures
    }

    /// Create a mini-scene from a subtree capture, with primitives translated
    /// to be relative to the capture bounds origin (for render-to-texture).
    pub fn create_capture_scene(&self, capture: &SubtreeCapture) -> Scene {
        let mut mini_scene = Scene::default();
        // Negate by subtracting from zero (ScaledPixels doesn't implement Neg)
        let offset = Point {
            x: ScaledPixels(0.0) - capture.bounds.origin.x,
            y: ScaledPixels(0.0) - capture.bounds.origin.y,
        };

        for op in &self.paint_operations[capture.primitive_start..capture.primitive_end] {
            match op {
                PaintOperation::Primitive(primitive) => {
                    let translated = primitive.translate(offset);
                    mini_scene.insert_primitive(translated);
                }
                PaintOperation::StartLayer(bounds) => {
                    let translated_bounds = Bounds {
                        origin: bounds.origin + offset,
                        size: bounds.size,
                    };
                    mini_scene.push_layer(translated_bounds);
                }
                PaintOperation::EndLayer => {
                    mini_scene.pop_layer();
                }
            }
        }

        mini_scene.finish();
        mini_scene
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        self.push_layer_internal(bounds, false);
    }

    pub fn push_layer_from_cache(&mut self, bounds: Bounds<ScaledPixels>) {
        self.push_layer_internal(bounds, true);
    }

    fn push_layer_internal(&mut self, bounds: Bounds<ScaledPixels>, from_cache: bool) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        if !from_cache {
            self.mark_dirty(self.paint_operations.len());
        }
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.pop_layer_internal(false);
    }

    pub fn pop_layer_from_cache(&mut self) {
        self.pop_layer_internal(true);
    }

    fn pop_layer_internal(&mut self, from_cache: bool) {
        self.layer_stack.pop();
        if !from_cache {
            self.mark_dirty(self.paint_operations.len());
        }
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        self.insert_primitive_internal(primitive, false);
    }

    pub fn insert_primitive_from_cache(&mut self, primitive: impl Into<Primitive>) {
        self.insert_primitive_internal(primitive, true);
    }

    /// Insert a primitive with an offset applied during insertion.
    /// This avoids the need to clone+translate the primitive before insertion,
    /// reducing the total clone count from 2 to 1 for cached primitives.
    ///
    /// Note: This method DOES add to paint_operations because SubtreeCache
    /// needs complete paint_operations to replay entire subtrees correctly.
    /// These primitives are from PrimitiveCache, so they count as cached (not dirty).
    pub fn insert_primitive_with_offset(
        &mut self,
        primitive: &Primitive,
        offset: Point<ScaledPixels>,
        content_mask: ContentMask<ScaledPixels>,
    ) {
        let transformed_bounds = self.offset_bounds(primitive, offset);
        let clipped_bounds = transformed_bounds.intersect(&content_mask.bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));

        // Track total primitives (but not dirty - these are from PrimitiveCache)
        self.dirty_stats.total_primitives += 1;

        // Clone with offset and content_mask applied in one pass
        // Also add to paint_operations for SubtreeCache replay support
        match primitive {
            Primitive::Shadow(shadow, transform) => {
                let mut shadow = shadow.clone();
                shadow.order = order;
                shadow.bounds = Self::apply_offset(shadow.bounds, offset);
                shadow.content_mask = content_mask;
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::Shadow(shadow.clone(), *transform)));
                self.shadows.push(shadow);
                self.shadow_transforms.push(*transform);
            }
            Primitive::Quad(quad, transform) => {
                let mut quad = quad.clone();
                quad.order = order;
                quad.bounds = Self::apply_offset(quad.bounds, offset);
                quad.content_mask = content_mask;
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::Quad(quad.clone(), *transform)));
                self.quads.push(quad);
                self.quad_transforms.push(*transform);
            }
            Primitive::BackdropBlur(blur, transform) => {
                let mut blur = blur.clone();
                blur.order = order;
                blur.bounds = Self::apply_offset(blur.bounds, offset);
                blur.content_mask = content_mask;
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::BackdropBlur(blur.clone(), *transform)));
                self.backdrop_blurs.push(blur);
                self.backdrop_blur_transforms.push(*transform);
            }
            Primitive::Path(path) => {
                let mut path = path.clone();
                path.order = order;
                path.id = PathId(self.paths.len());
                path.bounds = Self::apply_offset(path.bounds, offset);
                path.content_mask = content_mask.clone();
                for vertex in &mut path.vertices {
                    vertex.xy_position = Point {
                        x: vertex.xy_position.x + offset.x,
                        y: vertex.xy_position.y + offset.y,
                    };
                    vertex.content_mask = content_mask.clone();
                }
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::Path(path.clone())));
                self.paths.push(path);
            }
            Primitive::Underline(underline, transform) => {
                let mut underline = underline.clone();
                underline.order = order;
                underline.bounds = Self::apply_offset(underline.bounds, offset);
                underline.content_mask = content_mask;
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::Underline(underline.clone(), *transform)));
                self.underlines.push(underline);
                self.underline_transforms.push(*transform);
            }
            Primitive::MonochromeSprite(sprite) => {
                let mut sprite = sprite.clone();
                sprite.order = order;
                sprite.bounds = Self::apply_offset(sprite.bounds, offset);
                sprite.content_mask = content_mask;
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::MonochromeSprite(sprite.clone())));
                self.monochrome_sprites.push(sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                let mut sprite = sprite.clone();
                sprite.order = order;
                sprite.bounds = Self::apply_offset(sprite.bounds, offset);
                sprite.content_mask = content_mask;
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::SubpixelSprite(sprite.clone())));
                self.subpixel_sprites.push(sprite);
            }
            Primitive::PolychromeSprite(sprite, transform) => {
                let mut sprite = sprite.clone();
                sprite.order = order;
                sprite.bounds = Self::apply_offset(sprite.bounds, offset);
                sprite.content_mask = content_mask;
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::PolychromeSprite(sprite.clone(), *transform)));
                self.polychrome_sprites.push(sprite);
                self.polychrome_sprite_transforms.push(*transform);
            }
            Primitive::Surface(surface) => {
                let mut surface = surface.clone();
                surface.order = order;
                surface.bounds = Self::apply_offset(surface.bounds, offset);
                surface.content_mask = content_mask;
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::Surface(surface.clone())));
                self.surfaces.push(surface);
            }
            Primitive::CachedTexture(sprite) => {
                let mut sprite = sprite.clone();
                sprite.order = order;
                sprite.bounds = Self::apply_offset(sprite.bounds, offset);
                sprite.content_mask = content_mask;
                self.paint_operations
                    .push(PaintOperation::Primitive(Primitive::CachedTexture(sprite)));
                // Note: CachedTextureSprites are not collected into a typed array
                // because each one has its own texture and needs individual draw calls
            }
        }
    }

    fn apply_offset(mut bounds: Bounds<ScaledPixels>, offset: Point<ScaledPixels>) -> Bounds<ScaledPixels> {
        bounds.origin.x = bounds.origin.x + offset.x;
        bounds.origin.y = bounds.origin.y + offset.y;
        bounds
    }

    fn offset_bounds(&self, primitive: &Primitive, offset: Point<ScaledPixels>) -> Bounds<ScaledPixels> {
        let bounds = match primitive {
            Primitive::Shadow(shadow, _) => shadow.bounds,
            Primitive::Quad(quad, _) => quad.bounds,
            Primitive::BackdropBlur(blur, _) => blur.bounds,
            Primitive::Path(path) => path.bounds,
            Primitive::Underline(underline, _) => underline.bounds,
            Primitive::MonochromeSprite(sprite) => sprite.bounds,
            Primitive::SubpixelSprite(sprite) => sprite.bounds,
            Primitive::PolychromeSprite(sprite, _) => sprite.bounds,
            Primitive::Surface(surface) => surface.bounds,
            Primitive::CachedTexture(sprite) => sprite.bounds,
        };
        Self::apply_offset(bounds, offset)
    }

    fn insert_primitive_internal(&mut self, primitive: impl Into<Primitive>, from_cache: bool) {
        let mut primitive = primitive.into();
        let transformed_bounds = transformed_bounds(&primitive);
        let clipped_bounds = transformed_bounds.intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));

        // Track total and dirty primitives by type
        self.dirty_stats.total_primitives += 1;
        if !from_cache {
            self.dirty_stats.dirty_primitives += 1;
        }

        match &mut primitive {
            Primitive::Shadow(shadow, transform) => {
                shadow.order = order;
                if !from_cache {
                    self.dirty_stats.dirty_shadows += 1;
                }
                self.shadows.push(shadow.clone());
                self.shadow_transforms.push(*transform);
            }
            Primitive::Quad(quad, transform) => {
                quad.order = order;
                if !from_cache {
                    self.dirty_stats.dirty_quads += 1;
                }
                self.quads.push(quad.clone());
                self.quad_transforms.push(*transform);
            }
            Primitive::BackdropBlur(blur, transform) => {
                blur.order = order;
                if !from_cache {
                    self.dirty_stats.dirty_backdrop_blurs += 1;
                }
                self.backdrop_blurs.push(blur.clone());
                self.backdrop_blur_transforms.push(*transform);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                if !from_cache {
                    self.dirty_stats.dirty_paths += 1;
                }
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline, transform) => {
                underline.order = order;
                if !from_cache {
                    self.dirty_stats.dirty_underlines += 1;
                }
                self.underlines.push(underline.clone());
                self.underline_transforms.push(*transform);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                if !from_cache {
                    self.dirty_stats.dirty_monochrome_sprites += 1;
                }
                self.monochrome_sprites.push(sprite.clone());
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                if !from_cache {
                    self.dirty_stats.dirty_subpixel_sprites += 1;
                }
                self.subpixel_sprites.push(sprite.clone());
            }
            Primitive::PolychromeSprite(sprite, transform) => {
                sprite.order = order;
                if !from_cache {
                    self.dirty_stats.dirty_polychrome_sprites += 1;
                }
                self.polychrome_sprites.push(sprite.clone());
                self.polychrome_sprite_transforms.push(*transform);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                if !from_cache {
                    self.dirty_stats.dirty_surfaces += 1;
                }
                self.surfaces.push(surface.clone());
            }
            Primitive::CachedTexture(sprite) => {
                sprite.order = order;
                // Note: No dirty tracking yet for cached textures
                self.cached_textures.push(sprite.clone());
            }
        }

        /// Compute an axis-aligned bounding box for a transformed primitive.
        fn transformed_bounds(primitive: &Primitive) -> Bounds<ScaledPixels> {
            fn apply_transform(
                bounds: &Bounds<ScaledPixels>,
                transform: &TransformationMatrix,
            ) -> Bounds<ScaledPixels> {
                if transform.is_unit() {
                    return *bounds;
                }
                let [[a, b], [c, d]] = transform.rotation_scale;
                let tx = transform.translation[0];
                let ty = transform.translation[1];

                let x0 = bounds.origin.x.0;
                let x1 = x0 + bounds.size.width.0;
                let y0 = bounds.origin.y.0;
                let y1 = y0 + bounds.size.height.0;

                let ax0 = a * x0;
                let ax1 = a * x1;
                let by0 = b * y0;
                let by1 = b * y1;

                let cx0 = c * x0;
                let cx1 = c * x1;
                let dy0 = d * y0;
                let dy1 = d * y1;

                let min_x = tx + ax0.min(ax1) + by0.min(by1);
                let max_x = tx + ax0.max(ax1) + by0.max(by1);
                let min_y = ty + cx0.min(cx1) + dy0.min(dy1);
                let max_y = ty + cx0.max(cx1) + dy0.max(dy1);

                Bounds {
                    origin: point(ScaledPixels(min_x), ScaledPixels(min_y)),
                    size: Size {
                        width: ScaledPixels((max_x - min_x).max(0.0)),
                        height: ScaledPixels((max_y - min_y).max(0.0)),
                    },
                }
            }

            match primitive {
                Primitive::Shadow(shadow, transform) => apply_transform(&shadow.bounds, transform),
                Primitive::Quad(quad, transform) => apply_transform(&quad.bounds, transform),
                Primitive::BackdropBlur(blur, transform) => apply_transform(&blur.bounds, transform),
                Primitive::Underline(underline, transform) => {
                    apply_transform(&underline.bounds, transform)
                }
                Primitive::MonochromeSprite(sprite) => {
                    apply_transform(&sprite.bounds, &sprite.transformation)
                }
                Primitive::PolychromeSprite(sprite, transform) => {
                    apply_transform(&sprite.bounds, transform)
                }
                Primitive::SubpixelSprite(sprite) => {
                    apply_transform(&sprite.bounds, &sprite.transformation)
                }
                Primitive::Path(path) => path.bounds,
                Primitive::Surface(surface) => surface.bounds,
                Primitive::CachedTexture(sprite) => sprite.bounds,
            }
        }
        if !from_cache {
            self.mark_dirty(self.paint_operations.len());
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => {
                    self.insert_primitive_from_cache(primitive.clone())
                }
                PaintOperation::StartLayer(bounds) => self.push_layer_from_cache(*bounds),
                PaintOperation::EndLayer => self.pop_layer_from_cache(),
            }
        }
    }

    fn sort_with_aux_by_key<T, U, K: Ord>(
        items: &mut Vec<T>,
        aux: &mut Vec<U>,
        mut key_fn: impl FnMut(&T) -> K,
    ) {
        debug_assert_eq!(items.len(), aux.len());
        if items.len() <= 1 {
            return;
        }

        let mut perm: Vec<usize> = (0..items.len()).collect();
        perm.sort_by_key(|&index| key_fn(&items[index]));

        let mut desired_pos_by_current_pos = vec![0usize; perm.len()];
        for (desired_pos, current_pos) in perm.into_iter().enumerate() {
            desired_pos_by_current_pos[current_pos] = desired_pos;
        }

        for i in 0..items.len() {
            while desired_pos_by_current_pos[i] != i {
                let j = desired_pos_by_current_pos[i];
                items.swap(i, j);
                aux.swap(i, j);
                desired_pos_by_current_pos.swap(i, j);
            }
        }
    }

    pub fn finish(&mut self) {
        Self::sort_with_aux_by_key(&mut self.shadows, &mut self.shadow_transforms, |shadow| {
            shadow.order
        });
        Self::sort_with_aux_by_key(&mut self.quads, &mut self.quad_transforms, |quad| {
            quad.order
        });
        Self::sort_with_aux_by_key(
            &mut self.backdrop_blurs,
            &mut self.backdrop_blur_transforms,
            |blur| blur.order,
        );
        self.paths.sort_by_key(|path| path.order);
        Self::sort_with_aux_by_key(
            &mut self.underlines,
            &mut self.underline_transforms,
            |underline| underline.order,
        );
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        Self::sort_with_aux_by_key(
            &mut self.polychrome_sprites,
            &mut self.polychrome_sprite_transforms,
            |sprite| (sprite.order, sprite.tile.tile_id),
        );
        self.surfaces.sort_by_key(|surface| surface.order);
        self.cached_textures.sort_by_key(|texture| texture.order);
    }

    fn mark_dirty(&mut self, index: usize) {
        if let Some(last) = self.dirty_ranges.last_mut() {
            if last.end == index {
                last.end += 1;
                return;
            }
        }
        self.dirty_ranges.push(index..index + 1);
    }

    pub fn dirty_batch_ranges(&self) -> &[Range<usize>] {
        &self.dirty_ranges
    }

    /// Returns statistics about which primitives were freshly painted vs from cache.
    /// Useful for determining whether GPU buffer updates can be skipped or minimized.
    pub fn dirty_stats(&self) -> SceneDirtyStats {
        self.dirty_stats
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub(crate) fn batches(&self) -> impl Iterator<Item = PrimitiveBatch<'_>> {
        BatchIterator {
            shadows: &self.shadows,
            shadow_transforms: &self.shadow_transforms,
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads: &self.quads,
            quad_transforms: &self.quad_transforms,
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            backdrop_blurs: &self.backdrop_blurs,
            backdrop_blur_transforms: &self.backdrop_blur_transforms,
            backdrop_blurs_start: 0,
            backdrop_blurs_iter: self.backdrop_blurs.iter().peekable(),
            paths: &self.paths,
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines: &self.underlines,
            underline_transforms: &self.underline_transforms,
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites: &self.monochrome_sprites,
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites: &self.subpixel_sprites,
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites: &self.polychrome_sprites,
            polychrome_sprite_transforms: &self.polychrome_sprite_transforms,
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces: &self.surfaces,
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            cached_textures: &self.cached_textures,
            cached_textures_start: 0,
            cached_textures_iter: self.cached_textures.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    BackdropBlur,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
    CachedTexture,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
pub(crate) enum Primitive {
    Shadow(Shadow, TransformationMatrix),
    Quad(Quad, TransformationMatrix),
    BackdropBlur(BackdropBlur, TransformationMatrix),
    Path(Path<ScaledPixels>),
    Underline(Underline, TransformationMatrix),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite, TransformationMatrix),
    Surface(PaintSurface),
    CachedTexture(CachedTextureSprite),
}

impl Primitive {
    #[allow(dead_code)]
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow, _) => &shadow.bounds,
            Primitive::Quad(quad, _) => &quad.bounds,
            Primitive::BackdropBlur(blur, _) => &blur.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline, _) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite, _) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::CachedTexture(sprite) => &sprite.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow, _) => &shadow.content_mask,
            Primitive::Quad(quad, _) => &quad.content_mask,
            Primitive::BackdropBlur(blur, _) => &blur.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline, _) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite, _) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::CachedTexture(sprite) => &sprite.content_mask,
        }
    }

    /// Create a new primitive with bounds translated by the given offset.
    /// Used for render-to-texture where primitives need texture-relative coordinates.
    pub fn translate(&self, offset: Point<ScaledPixels>) -> Primitive {
        match self {
            Primitive::Shadow(shadow, transform) => {
                let mut s = shadow.clone();
                s.bounds.origin = s.bounds.origin + offset;
                s.content_mask.bounds.origin = s.content_mask.bounds.origin + offset;
                Primitive::Shadow(s, *transform)
            }
            Primitive::Quad(quad, transform) => {
                let mut q = quad.clone();
                q.bounds.origin = q.bounds.origin + offset;
                q.content_mask.bounds.origin = q.content_mask.bounds.origin + offset;
                Primitive::Quad(q, *transform)
            }
            Primitive::BackdropBlur(blur, transform) => {
                let mut b = blur.clone();
                b.bounds.origin = b.bounds.origin + offset;
                b.content_mask.bounds.origin = b.content_mask.bounds.origin + offset;
                Primitive::BackdropBlur(b, *transform)
            }
            Primitive::Path(path) => {
                let mut p = path.clone();
                p.bounds.origin = p.bounds.origin + offset;
                p.content_mask.bounds.origin = p.content_mask.bounds.origin + offset;
                // Also translate vertices
                for vertex in &mut p.vertices {
                    vertex.xy_position = point(
                        vertex.xy_position.x + offset.x,
                        vertex.xy_position.y + offset.y,
                    );
                }
                Primitive::Path(p)
            }
            Primitive::Underline(underline, transform) => {
                let mut u = underline.clone();
                u.bounds.origin = u.bounds.origin + offset;
                u.content_mask.bounds.origin = u.content_mask.bounds.origin + offset;
                Primitive::Underline(u, *transform)
            }
            Primitive::MonochromeSprite(sprite) => {
                let mut s = sprite.clone();
                s.bounds.origin = s.bounds.origin + offset;
                s.content_mask.bounds.origin = s.content_mask.bounds.origin + offset;
                Primitive::MonochromeSprite(s)
            }
            Primitive::SubpixelSprite(sprite) => {
                let mut s = sprite.clone();
                s.bounds.origin = s.bounds.origin + offset;
                s.content_mask.bounds.origin = s.content_mask.bounds.origin + offset;
                Primitive::SubpixelSprite(s)
            }
            Primitive::PolychromeSprite(sprite, transform) => {
                let mut s = sprite.clone();
                s.bounds.origin = s.bounds.origin + offset;
                s.content_mask.bounds.origin = s.content_mask.bounds.origin + offset;
                Primitive::PolychromeSprite(s, *transform)
            }
            Primitive::Surface(surface) => {
                let mut s = surface.clone();
                s.bounds.origin = s.bounds.origin + offset;
                s.content_mask.bounds.origin = s.content_mask.bounds.origin + offset;
                Primitive::Surface(s)
            }
            Primitive::CachedTexture(sprite) => {
                let mut s = sprite.clone();
                s.bounds.origin = s.bounds.origin + offset;
                s.content_mask.bounds.origin = s.content_mask.bounds.origin + offset;
                Primitive::CachedTexture(s)
            }
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows: &'a [Shadow],
    shadow_transforms: &'a [TransformationMatrix],
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads: &'a [Quad],
    quad_transforms: &'a [TransformationMatrix],
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    backdrop_blurs: &'a [BackdropBlur],
    backdrop_blur_transforms: &'a [TransformationMatrix],
    backdrop_blurs_start: usize,
    backdrop_blurs_iter: Peekable<slice::Iter<'a, BackdropBlur>>,
    paths: &'a [Path<ScaledPixels>],
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines: &'a [Underline],
    underline_transforms: &'a [TransformationMatrix],
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites: &'a [MonochromeSprite],
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites: &'a [SubpixelSprite],
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites: &'a [PolychromeSprite],
    polychrome_sprite_transforms: &'a [TransformationMatrix],
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces: &'a [PaintSurface],
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    cached_textures: &'a [CachedTextureSprite],
    cached_textures_start: usize,
    cached_textures_iter: Peekable<slice::Iter<'a, CachedTextureSprite>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (
                self.backdrop_blurs_iter.peek().map(|b| b.order),
                PrimitiveKind::BackdropBlur,
            ),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.cached_textures_iter.peek().map(|s| s.order),
                PrimitiveKind::CachedTexture,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(
                    &self.shadows[shadows_start..shadows_end],
                    &self.shadow_transforms[shadows_start..shadows_end],
                ))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(
                    &self.quads[quads_start..quads_end],
                    &self.quad_transforms[quads_start..quads_end],
                ))
            }
            PrimitiveKind::BackdropBlur => {
                let blurs_start = self.backdrop_blurs_start;
                let mut blurs_end = blurs_start + 1;
                self.backdrop_blurs_iter.next();
                while self
                    .backdrop_blurs_iter
                    .next_if(|blur| (blur.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    blurs_end += 1;
                }
                self.backdrop_blurs_start = blurs_end;
                Some(PrimitiveBatch::BackdropBlurs(
                    &self.backdrop_blurs[blurs_start..blurs_end],
                    &self.backdrop_blur_transforms[blurs_start..blurs_end],
                ))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(&self.paths[paths_start..paths_end]))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(
                    &self.underlines[underlines_start..underlines_end],
                    &self.underline_transforms[underlines_start..underlines_end],
                ))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites: &self.monochrome_sprites[sprites_start..sprites_end],
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    sprites: &self.subpixel_sprites[sprites_start..sprites_end],
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = self.polychrome_sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites: &self.polychrome_sprites[sprites_start..sprites_end],
                    transforms: &self.polychrome_sprite_transforms[sprites_start..sprites_end],
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(
                    &self.surfaces[surfaces_start..surfaces_end],
                ))
            }
            PrimitiveKind::CachedTexture => {
                let textures_start = self.cached_textures_start;
                let mut textures_end = textures_start + 1;
                self.cached_textures_iter.next();
                while self
                    .cached_textures_iter
                    .next_if(|texture| (texture.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    textures_end += 1;
                }
                self.cached_textures_start = textures_end;
                Some(PrimitiveBatch::CachedTextures(
                    &self.cached_textures[textures_start..textures_end],
                ))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveBatch<'a> {
    Shadows(&'a [Shadow], &'a [TransformationMatrix]),
    Quads(&'a [Quad], &'a [TransformationMatrix]),
    BackdropBlurs(&'a [BackdropBlur], &'a [TransformationMatrix]),
    Paths(&'a [Path<ScaledPixels>]),
    Underlines(&'a [Underline], &'a [TransformationMatrix]),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [MonochromeSprite],
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [SubpixelSprite],
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        sprites: &'a [PolychromeSprite],
        transforms: &'a [TransformationMatrix],
    },
    Surfaces(&'a [PaintSurface]),
    CachedTextures(&'a [CachedTextureSprite]),
}

#[derive(Default, Debug, Clone)]
#[repr(C)]
pub(crate) struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<(Quad, TransformationMatrix)> for Primitive {
    fn from((quad, transform): (Quad, TransformationMatrix)) -> Self {
        Primitive::Quad(quad, transform)
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct BackdropBlur {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub tint: Hsla,
}

impl From<(BackdropBlur, TransformationMatrix)> for Primitive {
    fn from((blur, transform): (BackdropBlur, TransformationMatrix)) -> Self {
        Primitive::BackdropBlur(blur, transform)
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: u32,
}

impl From<(Underline, TransformationMatrix)> for Primitive {
    fn from((underline, transform): (Underline, TransformationMatrix)) -> Self {
        Primitive::Underline(underline, transform)
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
}

impl From<(Shadow, TransformationMatrix)> for Primitive {
    fn from((shadow, transform): (Shadow, TransformationMatrix)) -> Self {
        Primitive::Shadow(shadow, transform)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
///
/// Matrices are stored in row-major order and applied as `M * position + t` in logical
/// (window) space. Callers should scale the translation exactly once when converting to
/// device space.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point in logical pixels.
    pub fn translate(mut self, point: Point<Pixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Returns true when the matrix has no effect (identity rotation/scale and zero translation).
    pub fn is_unit(&self) -> bool {
        *self == Self::unit()
    }

    /// Returns true when only translation is present (rotation/scale is the identity).
    pub fn is_translation_only(&self) -> bool {
        self.rotation_scale == [[1.0, 0.0], [0.0, 1.0]]
    }

    /// Apply the inverse transform to the given point. Returns `None` if the matrix
    /// is not invertible.
    pub fn apply_inverse(&self, point: Point<Pixels>) -> Option<Point<Pixels>> {
        // Inverse of a 2x2 matrix
        let det = self.rotation_scale[0][0] * self.rotation_scale[1][1]
            - self.rotation_scale[0][1] * self.rotation_scale[1][0];
        if det == 0.0 {
            return None;
        }

        let inv = [
            [
                self.rotation_scale[1][1] / det,
                -self.rotation_scale[0][1] / det,
            ],
            [
                -self.rotation_scale[1][0] / det,
                self.rotation_scale[0][0] / det,
            ],
        ];

        let translated = [
            point.x.0 - self.translation[0],
            point.y.0 - self.translation[1],
        ];

        let local = [
            inv[0][0] * translated[0] + inv[0][1] * translated[1],
            inv[1][0] * translated[0] + inv[1][1] * translated[1],
        ];

        Some(Point::new(local[0].into(), local[1].into()))
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub grayscale: bool,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

impl From<(PolychromeSprite, TransformationMatrix)> for Primitive {
    fn from((sprite, transform): (PolychromeSprite, TransformationMatrix)) -> Self {
        Primitive::PolychromeSprite(sprite, transform)
    }
}

/// Unique identifier for a cached texture used in render-to-texture caching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CachedTextureId(pub u64);

impl CachedTextureId {
    /// Generate the next unique texture ID.
    pub fn next() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// A sprite that composites a cached GPU texture to the screen.
/// Used for render-to-texture caching where subtrees are rendered once
/// and then composited at potentially different offsets.
#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct CachedTextureSprite {
    /// Draw order for z-ordering with other primitives
    pub order: DrawOrder,
    /// Padding for alignment
    pub _pad: u32,
    /// Screen bounds where the texture should be drawn
    pub bounds: Bounds<ScaledPixels>,
    /// Content mask for clipping
    pub content_mask: ContentMask<ScaledPixels>,
    /// UV bounds within the texture (0.0 to 1.0 range)
    /// Used when the texture is larger than the content (due to size bucketing)
    pub uv_bounds: Bounds<f32>,
    /// ID of the cached texture to sample from
    pub texture_id: CachedTextureId,
}

impl From<CachedTextureSprite> for Primitive {
    fn from(sprite: CachedTextureSprite) -> Self {
        Primitive::CachedTexture(sprite)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(target_os = "macos")]
    pub image_buffer: core_video::pixel_buffer::CVPixelBuffer,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PathId(pub(crate) usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub(crate) id: PathId,
    pub(crate) order: DrawOrder,
    pub(crate) bounds: Bounds<P>,
    pub(crate) content_mask: ContentMask<P>,
    pub(crate) vertices: Vec<PathVertex<P>>,
    pub(crate) color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    pub(crate) fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub(crate) struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub(crate) xy_position: Point<P>,
    pub(crate) st_position: Point<f32>,
    pub(crate) content_mask: ContentMask<P>,
}

impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}
