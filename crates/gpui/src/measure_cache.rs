use crate::{AvailableSpace, Pixels, Size};
use collections::FxHashMap;

/// Statistics for measure cache performance.
#[derive(Default, Clone, Copy, Debug)]
pub struct MeasureCacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of inserts
    pub inserts: u64,
}

/// Cache for measured layout results.
///
/// This caches the SIZE returned by measure functions, keyed by:
/// - content_hash: Identifies the content being measured (e.g., text + style)
/// - known_dimensions: The dimensions already known (width or height constraints)
/// - available_space: The space available for layout
///
/// This enables skipping expensive measure computations when:
/// 1. The same content is measured with the same constraints
/// 2. The same content appears multiple times in the UI
/// 3. Content is re-measured across frames (when combined with retained state)
pub struct MeasureCache {
    // Key: (content_hash, known_dims_hash, available_space_hash)
    // Value: computed size
    entries: FxHashMap<MeasureCacheKey, Size<Pixels>>,
    max_entries: usize,
    stats: MeasureCacheStats,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MeasureCacheKey {
    content_hash: u64,
    // Encode known dimensions as Option<i32> for width and height
    known_width: Option<i32>,
    known_height: Option<i32>,
    // Encode available space
    available_width: AvailableSpaceKey,
    available_height: AvailableSpaceKey,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum AvailableSpaceKey {
    Definite(i32), // Pixels rounded to i32
    MinContent,
    MaxContent,
}

impl From<AvailableSpace> for AvailableSpaceKey {
    fn from(space: AvailableSpace) -> Self {
        match space {
            AvailableSpace::Definite(px) => AvailableSpaceKey::Definite(px.0 as i32),
            AvailableSpace::MinContent => AvailableSpaceKey::MinContent,
            AvailableSpace::MaxContent => AvailableSpaceKey::MaxContent,
        }
    }
}

impl MeasureCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: FxHashMap::default(),
            max_entries,
            stats: MeasureCacheStats::default(),
        }
    }

    /// Look up a cached measure result.
    pub fn get(
        &mut self,
        content_hash: u64,
        known_dimensions: Size<Option<Pixels>>,
        available_space: Size<AvailableSpace>,
    ) -> Option<Size<Pixels>> {
        let key = MeasureCacheKey {
            content_hash,
            known_width: known_dimensions.width.map(|px| px.0 as i32),
            known_height: known_dimensions.height.map(|px| px.0 as i32),
            available_width: available_space.width.into(),
            available_height: available_space.height.into(),
        };

        if let Some(&size) = self.entries.get(&key) {
            self.stats.hits += 1;
            Some(size)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a measure result into the cache.
    pub fn insert(
        &mut self,
        content_hash: u64,
        known_dimensions: Size<Option<Pixels>>,
        available_space: Size<AvailableSpace>,
        size: Size<Pixels>,
    ) {
        let key = MeasureCacheKey {
            content_hash,
            known_width: known_dimensions.width.map(|px| px.0 as i32),
            known_height: known_dimensions.height.map(|px| px.0 as i32),
            available_width: available_space.width.into(),
            available_height: available_space.height.into(),
        };

        self.entries.insert(key, size);
        self.stats.inserts += 1;

        // Simple eviction - clear half when too large
        if self.entries.len() > self.max_entries {
            let target = self.max_entries / 2;
            let mut count = 0;
            self.entries.retain(|_, _| {
                count += 1;
                count <= target
            });
        }
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the stats and resets the counters.
    pub fn take_stats(&mut self) -> MeasureCacheStats {
        std::mem::take(&mut self.stats)
    }
}
