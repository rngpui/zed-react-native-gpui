use crate::{Pixels, Size, WrappedLine};
use collections::FxHashMap;
use smallvec::SmallVec;

/// Cached result of text shaping.
/// This allows reusing shaped text across frames without re-shaping.
#[derive(Clone)]
pub struct ShapedText {
    pub lines: SmallVec<[WrappedLine; 1]>,
    pub len: usize,
    pub line_height: Pixels,
    pub size: Size<Pixels>,
}

/// Statistics for text shape cache performance.
#[derive(Default, Clone, Copy, Debug)]
pub struct TextShapeCacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of inserts
    pub inserts: u64,
}

/// Cache for shaped text, keyed by content hash and wrap width.
///
/// This enables:
/// 1. Skipping text shaping when content hasn't changed
/// 2. Restoring TextLayoutInner when Taffy layout caching skips measure
pub struct TextShapeCache {
    // Key: (content_hash, wrap_width rounded to pixels)
    entries: FxHashMap<(u64, Option<i32>), ShapedText>,
    max_entries: usize,
    stats: TextShapeCacheStats,
}

impl TextShapeCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: FxHashMap::default(),
            max_entries,
            stats: TextShapeCacheStats::default(),
        }
    }

    pub fn get(&mut self, content_hash: u64, wrap_width: Option<Pixels>) -> Option<&ShapedText> {
        let key = (content_hash, wrap_width.map(|w| w.0 as i32));
        if self.entries.contains_key(&key) {
            self.stats.hits += 1;
            self.entries.get(&key)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, content_hash: u64, wrap_width: Option<Pixels>, shaped: ShapedText) {
        let key = (content_hash, wrap_width.map(|w| w.0 as i32));
        self.entries.insert(key, shaped);
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

    /// Returns the current stats without resetting them.
    pub fn peek_stats(&self) -> TextShapeCacheStats {
        self.stats
    }

    /// Returns the stats and resets the counters.
    pub fn take_stats(&mut self) -> TextShapeCacheStats {
        std::mem::take(&mut self.stats)
    }
}
