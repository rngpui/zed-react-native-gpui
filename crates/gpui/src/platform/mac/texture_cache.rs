//! GPU Texture Cache for Render-to-Texture Caching
//!
//! This module provides infrastructure for caching rendered subtrees as GPU textures,
//! enabling O(1) scrolling by compositing cached textures instead of re-rendering.
//!
//! Architecture:
//! - Textures are pooled by size bucket (256, 512, 1024, 2048, 4096)
//! - Each cached element gets a texture allocated from the pool
//! - LRU eviction when memory budget exceeded
//! - Textures can be reused when element content signature matches

use crate::scene::CachedTextureId;
use crate::{Bounds, DevicePixels, GlobalElementId, Size};
use collections::{FxHashMap, VecDeque};

/// Size buckets for texture pooling.
/// Textures are allocated in power-of-2 sizes for efficient reuse.
const SIZE_BUCKETS: &[u32] = &[256, 512, 1024, 2048, 4096];

/// Default memory budget in bytes (256 MB)
const DEFAULT_MEMORY_BUDGET: usize = 256 * 1024 * 1024;

/// Minimum size for texture caching (64x64 pixels)
pub const MIN_TEXTURE_SIZE: u32 = 64;

/// Maximum size for texture caching (4096x4096 pixels)
pub const MAX_TEXTURE_SIZE: u32 = 4096;

/// Statistics for texture cache performance.
#[derive(Default, Clone, Copy, Debug)]
pub struct TextureCacheStats {
    /// Number of cache hits (texture reused)
    pub hits: u64,
    /// Number of cache misses (new texture allocated)
    pub misses: u64,
    /// Number of textures currently in use
    pub active_textures: u64,
    /// Number of textures in the free pool
    pub pooled_textures: u64,
    /// Total memory used by active textures (bytes)
    pub active_memory: u64,
    /// Total memory in the free pool (bytes)
    pub pooled_memory: u64,
    /// Number of textures evicted this frame
    pub evictions: u64,
}

/// A pooled texture that can be reused.
pub struct PooledTexture {
    /// The Metal texture
    pub texture: metal::Texture,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Memory size in bytes
    pub memory_bytes: usize,
}

impl PooledTexture {
    fn new(device: &metal::Device, width: u32, height: u32) -> Self {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(width as u64);
        descriptor.set_height(height as u64);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor.set_usage(
            metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
        );
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);

        let texture = device.new_texture(&descriptor);
        let memory_bytes = (width as usize) * (height as usize) * 4; // BGRA8 = 4 bytes/pixel

        Self {
            texture,
            width,
            height,
            memory_bytes,
        }
    }
}

/// Entry for a cached element texture.
pub struct CachedTextureEntry {
    /// Unique ID for this cached texture
    pub id: CachedTextureId,
    /// The pooled texture
    pub texture: PooledTexture,
    /// Element signature for cache validation
    pub signature: u64,
    /// Bounds of the rendered content within the texture
    pub content_bounds: Bounds<DevicePixels>,
    /// Generation when this texture was last used
    pub last_used_generation: u64,
}

/// Key for the texture free pool (size bucket)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SizeBucket {
    width: u32,
    height: u32,
}

impl SizeBucket {
    /// Round up to the nearest size bucket
    fn from_size(size: Size<DevicePixels>) -> Self {
        let width = round_up_to_bucket(size.width.0 as u32);
        let height = round_up_to_bucket(size.height.0 as u32);
        Self { width, height }
    }
}

/// Round up to the nearest power-of-2 size bucket
fn round_up_to_bucket(size: u32) -> u32 {
    for &bucket in SIZE_BUCKETS {
        if size <= bucket {
            return bucket;
        }
    }
    // If larger than all buckets, round up to next power of 2
    size.next_power_of_two().min(MAX_TEXTURE_SIZE)
}

/// Manages GPU textures for render-to-texture caching.
pub struct TextureCacheManager {
    /// Textures currently in use, keyed by element ID
    active_textures: FxHashMap<GlobalElementId, CachedTextureEntry>,

    /// Reverse lookup from texture ID to element ID (for render-time lookups)
    texture_id_to_element: FxHashMap<CachedTextureId, GlobalElementId>,

    /// Free textures available for reuse, keyed by size bucket
    free_pool: FxHashMap<SizeBucket, Vec<PooledTexture>>,

    /// LRU queue for eviction (element_id, generation)
    lru_queue: VecDeque<(GlobalElementId, u64)>,

    /// Current frame generation
    generation: u64,

    /// Memory budget in bytes
    memory_budget: usize,

    /// Current memory usage (active + pooled)
    current_memory: usize,

    /// Statistics
    stats: TextureCacheStats,
}

impl TextureCacheManager {
    pub fn new() -> Self {
        Self {
            active_textures: FxHashMap::default(),
            texture_id_to_element: FxHashMap::default(),
            free_pool: FxHashMap::default(),
            lru_queue: VecDeque::new(),
            generation: 0,
            memory_budget: DEFAULT_MEMORY_BUDGET,
            current_memory: 0,
            stats: TextureCacheStats::default(),
        }
    }

    /// Set the memory budget for the texture cache.
    pub fn set_memory_budget(&mut self, bytes: usize) {
        self.memory_budget = bytes;
    }

    /// Called at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.generation += 1;
        self.stats.evictions = 0;
    }

    /// Called at the end of each frame. Evicts unused textures if over budget.
    pub fn end_frame(&mut self, device: &metal::Device) {
        self.evict_until_budget(device);
        self.update_stats();
    }

    /// Look up a cached texture for an element.
    /// Returns Some if the element has a cached texture with matching signature.
    pub fn lookup(
        &mut self,
        element_id: &GlobalElementId,
        signature: u64,
    ) -> Option<&CachedTextureEntry> {
        if let Some(entry) = self.active_textures.get_mut(element_id) {
            if entry.signature == signature {
                // Update LRU tracking
                entry.last_used_generation = self.generation;
                self.stats.hits += 1;
                return Some(entry);
            }
        }
        self.stats.misses += 1;
        None
    }

    /// Look up a cached texture by element ID only (for offset-only cache hits).
    /// Does not validate signature - caller must handle signature mismatch.
    pub fn lookup_by_id(&mut self, element_id: &GlobalElementId) -> Option<&CachedTextureEntry> {
        if let Some(entry) = self.active_textures.get_mut(element_id) {
            entry.last_used_generation = self.generation;
            Some(entry)
        } else {
            None
        }
    }

    /// Look up a cached texture by its texture ID (for render-time lookups from sprites).
    /// Returns the texture entry if found.
    pub fn lookup_by_texture_id(&mut self, texture_id: CachedTextureId) -> Option<&CachedTextureEntry> {
        if let Some(element_id) = self.texture_id_to_element.get(&texture_id).cloned() {
            if let Some(entry) = self.active_textures.get_mut(&element_id) {
                entry.last_used_generation = self.generation;
                return Some(entry);
            }
        }
        None
    }

    /// Acquire a texture for an element. Returns existing or allocates new.
    pub fn acquire(
        &mut self,
        device: &metal::Device,
        element_id: GlobalElementId,
        size: Size<DevicePixels>,
        signature: u64,
        content_bounds: Bounds<DevicePixels>,
    ) -> &CachedTextureEntry {
        // Check if we can reuse existing texture
        if let Some(entry) = self.active_textures.get(&element_id) {
            let bucket = SizeBucket::from_size(size);
            if entry.texture.width == bucket.width && entry.texture.height == bucket.height {
                // Same size, can reuse - update entry
                let entry = self.active_textures.get_mut(&element_id).unwrap();
                entry.signature = signature;
                entry.content_bounds = content_bounds;
                entry.last_used_generation = self.generation;
                return self.active_textures.get(&element_id).unwrap();
            } else {
                // Different size, release old texture
                self.release_internal(&element_id);
            }
        }

        // Try to get a texture from the pool
        let bucket = SizeBucket::from_size(size);
        let texture = if let Some(pool) = self.free_pool.get_mut(&bucket) {
            if let Some(tex) = pool.pop() {
                self.current_memory -= tex.memory_bytes;
                tex
            } else {
                self.allocate_texture(device, bucket)
            }
        } else {
            self.allocate_texture(device, bucket)
        };

        let id = CachedTextureId::next();
        let memory_bytes = texture.memory_bytes;

        let entry = CachedTextureEntry {
            id,
            texture,
            signature,
            content_bounds,
            last_used_generation: self.generation,
        };

        self.active_textures.insert(element_id.clone(), entry);
        self.texture_id_to_element.insert(id, element_id.clone());
        self.lru_queue.push_back((element_id.clone(), self.generation));
        self.current_memory += memory_bytes;

        self.active_textures.get(&element_id).unwrap()
    }

    /// Release a texture back to the pool for an element.
    pub fn release(&mut self, element_id: &GlobalElementId) {
        self.release_internal(element_id);
    }

    fn release_internal(&mut self, element_id: &GlobalElementId) {
        if let Some(entry) = self.active_textures.remove(element_id) {
            // Remove from reverse lookup map
            self.texture_id_to_element.remove(&entry.id);

            let bucket = SizeBucket {
                width: entry.texture.width,
                height: entry.texture.height,
            };
            let memory_bytes = entry.texture.memory_bytes;

            // Return texture to pool
            self.free_pool
                .entry(bucket)
                .or_insert_with(Vec::new)
                .push(entry.texture);

            // Memory stays the same (moved from active to pooled)
            // current_memory is total, so no change needed
        }
    }

    fn allocate_texture(&mut self, device: &metal::Device, bucket: SizeBucket) -> PooledTexture {
        PooledTexture::new(device, bucket.width, bucket.height)
    }

    /// Evict textures until under memory budget.
    fn evict_until_budget(&mut self, _device: &metal::Device) {
        // First, clean up the LRU queue (remove entries for elements no longer active)
        self.lru_queue
            .retain(|(id, _)| self.active_textures.contains_key(id));

        // If under budget, nothing to do
        if self.current_memory <= self.memory_budget {
            return;
        }

        // Evict from free pool first
        let mut pools_to_remove = Vec::new();
        for (bucket, pool) in &mut self.free_pool {
            while !pool.is_empty() && self.current_memory > self.memory_budget {
                if let Some(tex) = pool.pop() {
                    self.current_memory -= tex.memory_bytes;
                    self.stats.evictions += 1;
                    // Texture is dropped here, releasing GPU memory
                }
            }
            if pool.is_empty() {
                pools_to_remove.push(*bucket);
            }
        }
        for bucket in pools_to_remove {
            self.free_pool.remove(&bucket);
        }

        // If still over budget, evict LRU active textures
        while self.current_memory > self.memory_budget {
            if let Some((element_id, queued_gen)) = self.lru_queue.pop_front() {
                // Only evict if it's an old entry (not used this frame)
                if let Some(entry) = self.active_textures.get(&element_id) {
                    if entry.last_used_generation < self.generation && queued_gen < self.generation {
                        let memory_bytes = entry.texture.memory_bytes;
                        let texture_id = entry.id;
                        self.active_textures.remove(&element_id);
                        self.texture_id_to_element.remove(&texture_id);
                        self.current_memory -= memory_bytes;
                        self.stats.evictions += 1;
                    }
                }
            } else {
                break; // No more entries to evict
            }
        }
    }

    fn update_stats(&mut self) {
        let mut active_memory = 0u64;
        for entry in self.active_textures.values() {
            active_memory += entry.texture.memory_bytes as u64;
        }

        let mut pooled_memory = 0u64;
        let mut pooled_count = 0u64;
        for pool in self.free_pool.values() {
            for tex in pool {
                pooled_memory += tex.memory_bytes as u64;
                pooled_count += 1;
            }
        }

        self.stats.active_textures = self.active_textures.len() as u64;
        self.stats.pooled_textures = pooled_count;
        self.stats.active_memory = active_memory;
        self.stats.pooled_memory = pooled_memory;
    }

    /// Returns the current stats without resetting them.
    pub fn peek_stats(&self) -> TextureCacheStats {
        self.stats
    }

    /// Returns the stats and resets the hit/miss counters.
    pub fn take_stats(&mut self) -> TextureCacheStats {
        let stats = self.stats;
        self.stats.hits = 0;
        self.stats.misses = 0;
        self.stats.evictions = 0;
        stats
    }

    /// Check if a size is suitable for texture caching.
    pub fn is_cacheable_size(size: Size<DevicePixels>) -> bool {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;
        width >= MIN_TEXTURE_SIZE
            && height >= MIN_TEXTURE_SIZE
            && width <= MAX_TEXTURE_SIZE
            && height <= MAX_TEXTURE_SIZE
    }
}

impl Default for TextureCacheManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_up_to_bucket() {
        assert_eq!(round_up_to_bucket(64), 256);
        assert_eq!(round_up_to_bucket(256), 256);
        assert_eq!(round_up_to_bucket(257), 512);
        assert_eq!(round_up_to_bucket(512), 512);
        assert_eq!(round_up_to_bucket(1000), 1024);
        assert_eq!(round_up_to_bucket(4096), 4096);
        assert_eq!(round_up_to_bucket(5000), 4096); // Capped at MAX_TEXTURE_SIZE
    }

    #[test]
    fn test_size_bucket() {
        let size = crate::size(crate::DevicePixels(300), crate::DevicePixels(400));
        let bucket = SizeBucket::from_size(size);
        assert_eq!(bucket.width, 512);
        assert_eq!(bucket.height, 512);
    }
}
