use crate::{Bounds, ContentMask, GlobalElementId, Path, Point, Primitive, Scene, ScaledPixels};
use collections::FxHashMap;
use smallvec::SmallVec;
use std::collections::VecDeque;

/// Statistics about primitive cache performance for the current frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrimitiveCacheStats {
    /// Number of cache hits (element paint was replayed from cache)
    pub hits: u64,
    /// Number of cache misses (element had to be repainted)
    pub misses: u64,
    /// Number of new entries inserted into the cache
    pub inserts: u64,
    /// Number of entries evicted due to LRU policy
    pub evictions: u64,
}

pub(crate) struct PrimitiveCache {
    entries: FxHashMap<GlobalElementId, CacheEntry>,
    access_order: VecDeque<(GlobalElementId, u64)>,
    max_entries: usize,
    generation: u64,
    stats: PrimitiveCacheStats,
}

#[derive(Clone)]
pub(crate) struct CacheReplay {
    pub(crate) cached_bounds: Bounds<ScaledPixels>,
    pub(crate) operations: SmallVec<[CachedPaintOperation; 8]>,
}

struct CacheEntry {
    content_hash: u64,
    last_access: u64,
    cached_bounds: Bounds<ScaledPixels>,
    operations: SmallVec<[CachedPaintOperation; 8]>,
    scale_factor: f32,
}

#[derive(Clone)]
pub(crate) enum CachedPaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

impl CachedPaintOperation {
    pub(crate) fn from_primitive(
        primitive: Primitive,
        origin: Point<ScaledPixels>,
    ) -> CachedPaintOperation {
        CachedPaintOperation::Primitive(translate_primitive(primitive, neg_point(origin)))
    }

    pub(crate) fn from_layer_start(
        bounds: Bounds<ScaledPixels>,
        origin: Point<ScaledPixels>,
    ) -> CachedPaintOperation {
        CachedPaintOperation::StartLayer(offset_bounds(bounds, neg_point(origin)))
    }

    pub(crate) fn replay(&self, delta: Point<ScaledPixels>, scene: &mut Scene) {
        match self {
            CachedPaintOperation::Primitive(primitive) => {
                scene.insert_primitive_from_cache(translate_primitive(primitive.clone(), delta));
            }
            CachedPaintOperation::StartLayer(bounds) => {
                scene.push_layer_from_cache(offset_bounds(*bounds, delta));
            }
            CachedPaintOperation::EndLayer => {
                scene.pop_layer_from_cache();
            }
        }
    }

    pub(crate) fn replay_with_dirty(&self, delta: Point<ScaledPixels>, scene: &mut Scene) {
        match self {
            CachedPaintOperation::Primitive(primitive) => {
                scene.insert_primitive(translate_primitive(primitive.clone(), delta));
            }
            CachedPaintOperation::StartLayer(bounds) => {
                scene.push_layer(offset_bounds(*bounds, delta));
            }
            CachedPaintOperation::EndLayer => {
                scene.pop_layer();
            }
        }
    }
}

impl PrimitiveCache {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            entries: FxHashMap::default(),
            access_order: VecDeque::new(),
            max_entries,
            generation: 0,
            stats: PrimitiveCacheStats::default(),
        }
    }

    pub(crate) fn begin_frame(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    pub(crate) fn lookup(
        &mut self,
        id: &GlobalElementId,
        content_hash: u64,
        current_bounds: Bounds<ScaledPixels>,
        scale_factor: f32,
    ) -> Option<CacheReplay> {
        let Some(entry) = self.entries.get_mut(id) else {
            self.stats.misses += 1;
            return None;
        };
        if entry.content_hash != content_hash
            || entry.scale_factor != scale_factor
            || entry.cached_bounds.size != current_bounds.size
        {
            self.stats.misses += 1;
            return None;
        }

        self.stats.hits += 1;
        entry.last_access = self.generation;
        self.access_order
            .push_back((id.clone(), self.generation));
        Some(CacheReplay {
            cached_bounds: entry.cached_bounds,
            operations: entry.operations.clone(),
        })
    }

    pub(crate) fn insert(
        &mut self,
        id: GlobalElementId,
        content_hash: u64,
        bounds: Bounds<ScaledPixels>,
        scale_factor: f32,
        operations: SmallVec<[CachedPaintOperation; 8]>,
    ) {
        let entry = CacheEntry {
            content_hash,
            last_access: self.generation,
            cached_bounds: bounds,
            operations,
            scale_factor,
        };
        self.entries.insert(id.clone(), entry);
        self.access_order
            .push_back((id, self.generation));
        self.stats.inserts += 1;
        self.evict_if_needed();
    }

    pub(crate) fn take_stats(&mut self) -> PrimitiveCacheStats {
        std::mem::take(&mut self.stats)
    }

    fn evict_if_needed(&mut self) {
        while self.entries.len() > self.max_entries {
            let Some((id, generation)) = self.access_order.pop_front() else {
                break;
            };
            let remove = self
                .entries
                .get(&id)
                .is_some_and(|entry| entry.last_access == generation);
            if remove {
                self.entries.remove(&id);
                self.stats.evictions += 1;
            }
        }
    }
}

fn neg_point(point: Point<ScaledPixels>) -> Point<ScaledPixels> {
    Point {
        x: ScaledPixels(0.0) - point.x,
        y: ScaledPixels(0.0) - point.y,
    }
}

fn offset_bounds(
    mut bounds: Bounds<ScaledPixels>,
    delta: Point<ScaledPixels>,
) -> Bounds<ScaledPixels> {
    bounds.origin = Point {
        x: bounds.origin.x + delta.x,
        y: bounds.origin.y + delta.y,
    };
    bounds
}

fn offset_mask(
    mut mask: ContentMask<ScaledPixels>,
    delta: Point<ScaledPixels>,
) -> ContentMask<ScaledPixels> {
    mask.bounds = offset_bounds(mask.bounds, delta);
    mask
}

fn translate_primitive(mut primitive: Primitive, delta: Point<ScaledPixels>) -> Primitive {
    match &mut primitive {
        Primitive::Shadow(shadow, _) => {
            shadow.bounds = offset_bounds(shadow.bounds, delta);
            shadow.content_mask = offset_mask(shadow.content_mask.clone(), delta);
        }
        Primitive::Quad(quad, _) => {
            quad.bounds = offset_bounds(quad.bounds, delta);
            quad.content_mask = offset_mask(quad.content_mask.clone(), delta);
        }
        Primitive::BackdropBlur(blur, _) => {
            blur.bounds = offset_bounds(blur.bounds, delta);
            blur.content_mask = offset_mask(blur.content_mask.clone(), delta);
        }
        Primitive::Path(path) => {
            translate_path(path, delta);
        }
        Primitive::Underline(underline, _) => {
            underline.bounds = offset_bounds(underline.bounds, delta);
            underline.content_mask = offset_mask(underline.content_mask.clone(), delta);
        }
        Primitive::MonochromeSprite(sprite) => {
            sprite.bounds = offset_bounds(sprite.bounds, delta);
            sprite.content_mask = offset_mask(sprite.content_mask.clone(), delta);
        }
        Primitive::SubpixelSprite(sprite) => {
            sprite.bounds = offset_bounds(sprite.bounds, delta);
            sprite.content_mask = offset_mask(sprite.content_mask.clone(), delta);
        }
        Primitive::PolychromeSprite(sprite, _) => {
            sprite.bounds = offset_bounds(sprite.bounds, delta);
            sprite.content_mask = offset_mask(sprite.content_mask.clone(), delta);
        }
        Primitive::Surface(surface) => {
            surface.bounds = offset_bounds(surface.bounds, delta);
            surface.content_mask = offset_mask(surface.content_mask.clone(), delta);
        }
    }
    primitive
}

fn translate_path(path: &mut Path<ScaledPixels>, delta: Point<ScaledPixels>) {
    path.bounds = offset_bounds(path.bounds, delta);
    path.content_mask = offset_mask(path.content_mask.clone(), delta);

    for vertex in &mut path.vertices {
        vertex.xy_position = Point {
            x: vertex.xy_position.x + delta.x,
            y: vertex.xy_position.y + delta.y,
        };
        vertex.content_mask = offset_mask(vertex.content_mask.clone(), delta);
    }
}
