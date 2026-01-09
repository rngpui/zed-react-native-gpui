use crate::{Bounds, ContentMask, GlobalElementId, Path, Point, Primitive, Scene, ScaledPixels};
use collections::FxHashMap;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::sync::Arc;

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
    /// Number of cache hits where position also matched (truly unchanged)
    pub position_unchanged: u64,
}

impl PrimitiveCacheStats {
    /// Returns true if the frame was completely stable (100% cache hits, no changes).
    /// This means the scene is identical to the previous frame.
    pub fn is_scene_stable(&self) -> bool {
        // A frame is stable if we had cache activity and all lookups were hits
        // (no misses, no new inserts). Position changes are OK because the
        // cached primitives get translated to the correct position during replay.
        self.hits > 0 && self.misses == 0 && self.inserts == 0
    }
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
    pub(crate) operations: Arc<SmallVec<[CachedPaintOperation; 8]>>,
}

struct CacheEntry {
    content_hash: u64,
    last_access: u64,
    cached_bounds: Bounds<ScaledPixels>,
    operations: Arc<SmallVec<[CachedPaintOperation; 8]>>,
    scale_factor: f32,
}

#[derive(Clone)]
pub(crate) enum CachedPaintOperation {
    Primitive(Arc<Primitive>),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

impl CachedPaintOperation {
    pub(crate) fn from_primitive(
        primitive: Primitive,
        origin: Point<ScaledPixels>,
    ) -> CachedPaintOperation {
        CachedPaintOperation::Primitive(Arc::new(translate_primitive(primitive, neg_point(origin))))
    }

    pub(crate) fn from_layer_start(
        bounds: Bounds<ScaledPixels>,
        origin: Point<ScaledPixels>,
    ) -> CachedPaintOperation {
        CachedPaintOperation::StartLayer(offset_bounds(bounds, neg_point(origin)))
    }

    pub(crate) fn replay(
        &self,
        origin: Point<ScaledPixels>,
        current_content_mask: ContentMask<ScaledPixels>,
        scene: &mut Scene,
    ) {
        match self {
            CachedPaintOperation::Primitive(primitive) => {
                // Use Scene's offset-based insertion to avoid clone+translate overhead.
                // This reduces clone count from 2 to 1 per cached primitive.
                // The Scene applies the offset during its internal clone.
                scene.insert_primitive_with_offset(primitive, origin, current_content_mask);
            }
            CachedPaintOperation::StartLayer(bounds) => {
                scene.push_layer_from_cache(offset_bounds(*bounds, origin));
            }
            CachedPaintOperation::EndLayer => {
                scene.pop_layer_from_cache();
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
        // Track if position is also unchanged (truly stable element)
        if entry.cached_bounds.origin == current_bounds.origin {
            self.stats.position_unchanged += 1;
        }
        entry.last_access = self.generation;
        self.access_order
            .push_back((id.clone(), self.generation));
        Some(CacheReplay {
            // Arc::clone is cheap - just increments reference count
            operations: Arc::clone(&entry.operations),
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
            operations: Arc::new(operations),
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

    /// Returns the current stats without resetting them.
    pub(crate) fn peek_stats(&self) -> PrimitiveCacheStats {
        self.stats
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

fn translate_primitive(mut primitive: Primitive, delta: Point<ScaledPixels>) -> Primitive {
    // Only translate primitive bounds, NOT content_mask.
    // content_mask represents viewport/layer clipping in absolute coordinates
    // and should not be made element-relative.
    match &mut primitive {
        Primitive::Shadow(shadow, _) => {
            shadow.bounds = offset_bounds(shadow.bounds, delta);
        }
        Primitive::Quad(quad, _) => {
            quad.bounds = offset_bounds(quad.bounds, delta);
        }
        Primitive::BackdropBlur(blur, _) => {
            blur.bounds = offset_bounds(blur.bounds, delta);
        }
        Primitive::Path(path) => {
            translate_path(path, delta);
        }
        Primitive::Underline(underline, _) => {
            underline.bounds = offset_bounds(underline.bounds, delta);
        }
        Primitive::MonochromeSprite(sprite) => {
            sprite.bounds = offset_bounds(sprite.bounds, delta);
        }
        Primitive::SubpixelSprite(sprite) => {
            sprite.bounds = offset_bounds(sprite.bounds, delta);
        }
        Primitive::PolychromeSprite(sprite, _) => {
            sprite.bounds = offset_bounds(sprite.bounds, delta);
        }
        Primitive::Surface(surface) => {
            surface.bounds = offset_bounds(surface.bounds, delta);
        }
        Primitive::CachedTexture(sprite) => {
            sprite.bounds = offset_bounds(sprite.bounds, delta);
        }
        Primitive::TileSprite(sprite) => {
            sprite.bounds = offset_bounds(sprite.bounds, delta);
        }
    }
    primitive
}

fn translate_path(path: &mut Path<ScaledPixels>, delta: Point<ScaledPixels>) {
    path.bounds = offset_bounds(path.bounds, delta);

    for vertex in &mut path.vertices {
        vertex.xy_position = Point {
            x: vertex.xy_position.x + delta.x,
            y: vertex.xy_position.y + delta.y,
        };
    }
}

