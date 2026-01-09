use super::metal_atlas::MetalAtlas;
use super::tile_cache::TileCache;
use super::tile_rasterizer::{rasterize_tiles_parallel, TilePriority, TileRasterJob};
use crate::{
    AtlasTextureId, BackdropBlur, Background, Bounds, ContentMask, DevicePixels, MonochromeSprite,
    PaintSurface, Path, Point, PolychromeSprite, PrimitiveBatch, Quad, ScaledPixels, Scene, Shadow,
    Size, Surface, TileSprite, TransformationMatrix, Underline, point, size,
};
use anyhow::Result;
use block::ConcreteBlock;
use cocoa::{
    base::{NO, YES},
    foundation::{NSSize, NSUInteger},
    quartzcore::AutoresizingMask,
};
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;

use core_foundation::base::TCFType;
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLOrigin, MTLPixelFormat, MTLResourceOptions, MTLSize, NSRange,
    RenderPassColorAttachmentDescriptorRef,
};
use objc::{self, msg_send, sel, sel_impl};
use parking_lot::Mutex;

use collections::FxHashMap;
use std::{cell::Cell, ffi::c_void, mem, ptr, slice, sync::Arc};

// Exported to metal
pub(crate) type PointF = crate::Point<f32>;

/// Tracks which typed arrays changed from the previous frame.
/// Used for partial GPU uploads - unchanged arrays can skip memcpy.
#[derive(Default, Clone, Copy)]
struct ArrayDirtyFlags {
    quads: bool,
    shadows: bool,
    underlines: bool,
    paths: bool,
    backdrop_blurs: bool,
    monochrome_sprites: bool,
    polychrome_sprites: bool,
    surfaces: bool,
}

impl ArrayDirtyFlags {
    fn all_dirty() -> Self {
        Self {
            quads: true,
            shadows: true,
            underlines: true,
            paths: true,
            backdrop_blurs: true,
            monochrome_sprites: true,
            polychrome_sprites: true,
            surfaces: true,
        }
    }
}

/// Stores previous frame's typed arrays for comparison.
/// Enables skipping GPU uploads for unchanged arrays.
#[derive(Default)]
struct PreviousFrameArrays {
    quads: Vec<Quad>,
    quad_transforms: Vec<TransformationMatrix>,
    shadows: Vec<Shadow>,
    shadow_transforms: Vec<TransformationMatrix>,
    underlines: Vec<Underline>,
    underline_transforms: Vec<TransformationMatrix>,
    paths_len: usize, // Paths contain Vecs, so we just compare length
    backdrop_blurs: Vec<BackdropBlur>,
    backdrop_blur_transforms: Vec<TransformationMatrix>,
    monochrome_sprites: Vec<MonochromeSprite>,
    polychrome_sprites: Vec<PolychromeSprite>,
    polychrome_sprite_transforms: Vec<TransformationMatrix>,
    surfaces_len: usize, // Surfaces contain CVPixelBuffer, not Clone, so we compare length
}

impl PreviousFrameArrays {
    /// Compare typed arrays with current scene and return dirty flags.
    /// Uses byte-wise comparison for repr(C) structs.
    fn compare_with_scene(&self, scene: &Scene) -> ArrayDirtyFlags {
        ArrayDirtyFlags {
            quads: !Self::slices_equal(&self.quads, &scene.quads)
                || !Self::slices_equal(&self.quad_transforms, &scene.quad_transforms),
            shadows: !Self::slices_equal(&self.shadows, &scene.shadows)
                || !Self::slices_equal(&self.shadow_transforms, &scene.shadow_transforms),
            underlines: !Self::slices_equal(&self.underlines, &scene.underlines)
                || !Self::slices_equal(&self.underline_transforms, &scene.underline_transforms),
            paths: self.paths_len != scene.paths.len(), // Conservative: any length change = dirty
            backdrop_blurs: !Self::slices_equal(&self.backdrop_blurs, &scene.backdrop_blurs)
                || !Self::slices_equal(
                    &self.backdrop_blur_transforms,
                    &scene.backdrop_blur_transforms,
                ),
            monochrome_sprites: !Self::slices_equal(
                &self.monochrome_sprites,
                &scene.monochrome_sprites,
            ),
            polychrome_sprites: !Self::slices_equal(
                &self.polychrome_sprites,
                &scene.polychrome_sprites,
            ) || !Self::slices_equal(
                &self.polychrome_sprite_transforms,
                &scene.polychrome_sprite_transforms,
            ),
            surfaces: self.surfaces_len != scene.surfaces.len(), // Conservative: CVPixelBuffer not comparable
        }
    }

    /// Byte-wise comparison of two slices of repr(C) structs.
    /// Returns true if slices are identical.
    fn slices_equal<T>(a: &[T], b: &[T]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        if a.is_empty() {
            return true;
        }
        let bytes_a = unsafe {
            slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * mem::size_of::<T>())
        };
        let bytes_b = unsafe {
            slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * mem::size_of::<T>())
        };
        bytes_a == bytes_b
    }

    /// Clone arrays from the scene for next frame comparison.
    fn update_from_scene(&mut self, scene: &Scene) {
        self.quads.clear();
        self.quads.extend_from_slice(&scene.quads);
        self.quad_transforms.clear();
        self.quad_transforms.extend_from_slice(&scene.quad_transforms);

        self.shadows.clear();
        self.shadows.extend_from_slice(&scene.shadows);
        self.shadow_transforms.clear();
        self.shadow_transforms.extend_from_slice(&scene.shadow_transforms);

        self.underlines.clear();
        self.underlines.extend_from_slice(&scene.underlines);
        self.underline_transforms.clear();
        self.underline_transforms
            .extend_from_slice(&scene.underline_transforms);

        self.paths_len = scene.paths.len();

        self.backdrop_blurs.clear();
        self.backdrop_blurs.extend_from_slice(&scene.backdrop_blurs);
        self.backdrop_blur_transforms.clear();
        self.backdrop_blur_transforms
            .extend_from_slice(&scene.backdrop_blur_transforms);

        self.monochrome_sprites.clear();
        self.monochrome_sprites
            .extend_from_slice(&scene.monochrome_sprites);

        self.polychrome_sprites.clear();
        self.polychrome_sprites
            .extend_from_slice(&scene.polychrome_sprites);
        self.polychrome_sprite_transforms.clear();
        self.polychrome_sprite_transforms
            .extend_from_slice(&scene.polychrome_sprite_transforms);

        self.surfaces_len = scene.surfaces.len();
    }
}

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: crate::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent)
}

pub(crate) struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(&mut self, device: &metal::Device) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            device.new_buffer(
                self.buffer_size as u64,
                MTLResourceOptions::StorageModeManaged,
            )
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub(crate) struct MetalRenderer {
    pending_dirty_ranges: Vec<std::ops::Range<usize>>,
    incremental_draw: bool,
    last_scene_len: usize,
    #[allow(clippy::arc_with_non_send_sync)]
    cached_instance_buffer: Arc<Mutex<Option<InstanceBuffer>>>,
    /// Previous frame's typed arrays for per-array dirty tracking.
    previous_frame_arrays: PreviousFrameArrays,
    /// Per-array dirty flags computed by comparing with previous frame.
    array_dirty_flags: ArrayDirtyFlags,
    device: metal::Device,
    layer: metal::MetalLayer,
    presents_with_transaction: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    backdrop_blurs_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    cached_textures_pipeline_state: metal::RenderPipelineState,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    backdrop_texture: Option<metal::Texture>,
    path_sample_count: u32,
    /// Texture cache for render-to-texture caching of subtrees.
    texture_cache: super::texture_cache::TextureCacheManager,
    /// Tile cache for scroll container tiled rendering.
    tile_cache: TileCache,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

impl MetalRenderer {
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        let device = if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            metal::Device::system_default().unwrap_or_else(|| {
                log::error!("unable to access a compatible graphics device");
                std::process::exit(1);
            })
        };

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        // Allow texture reading for visual tests (captures screenshots without ScreenCaptureKit)
        #[cfg(any(test, feature = "test-support"))]
        layer.set_framebuffer_only(false);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                    | AutoresizingMask::HEIGHT_SIZABLE
            ];
        }
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let backdrop_blurs_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "backdrop_blurs",
            "backdrop_blur_vertex",
            "backdrop_blur_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        // Cached textures use premultiplied alpha blending because the RTT texture
        // already contains premultiplied colors from rendering with standard alpha blending
        // onto a transparent background.
        let cached_textures_pipeline_state = build_premultiplied_pipeline_state(
            &device,
            &library,
            "cached_textures",
            "cached_texture_vertex",
            "cached_texture_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone()));
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        Self {
            pending_dirty_ranges: Vec::new(),
            incremental_draw: false,
            last_scene_len: 0,
            cached_instance_buffer: Arc::new(Mutex::new(None)),
            previous_frame_arrays: PreviousFrameArrays::default(),
            array_dirty_flags: ArrayDirtyFlags::all_dirty(),
            device: device.clone(),
            layer,
            presents_with_transaction: false,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            backdrop_blurs_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            cached_textures_pipeline_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            backdrop_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            texture_cache: super::texture_cache::TextureCacheManager::new(),
            tile_cache: TileCache::new(device),
        }
    }

    pub fn layer(&self) -> &metal::MetalLayerRef {
        &self.layer
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer.as_ptr()
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    /// Get the current texture cache statistics without resetting counters.
    #[allow(dead_code)]
    pub fn texture_cache_stats(&self) -> super::texture_cache::TextureCacheStats {
        self.texture_cache.peek_stats()
    }

    /// Get the texture cache statistics and reset hit/miss counters.
    #[allow(dead_code)]
    pub fn take_texture_cache_stats(&mut self) -> super::texture_cache::TextureCacheStats {
        self.texture_cache.take_stats()
    }

    /// Access the tile cache for scroll container tiled rendering.
    pub fn tile_cache(&mut self) -> &mut TileCache {
        &mut self.tile_cache
    }

    /// Query if an element has a cached texture for RTT compositing.
    /// Returns texture info including ID and UV bounds if the element has a valid cached texture.
    pub fn get_cached_texture_info(
        &mut self,
        element_id: &crate::GlobalElementId,
    ) -> Option<crate::scene::CachedTextureInfo> {
        self.texture_cache.lookup_by_id(element_id).map(|entry| {
            // Compute UV bounds from content size vs texture size
            // Textures are allocated in size buckets, so content may not fill the whole texture
            let uv_width = entry.content_bounds.size.width.0 as f32 / entry.texture.width as f32;
            let uv_height = entry.content_bounds.size.height.0 as f32 / entry.texture.height as f32;
            crate::scene::CachedTextureInfo {
                id: entry.id,
                uv_bounds: Bounds {
                    origin: point(0.0, 0.0),
                    size: size(uv_width, uv_height),
                },
            }
        })
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        self.layer
            .set_presents_with_transaction(presents_with_transaction);
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let size = NSSize {
            width: size.width.0 as f64,
            height: size.height.0 as f64,
        };
        unsafe {
            let _: () = msg_send![
                self.layer(),
                setDrawableSize: size
            ];
        }
        let device_pixels_size = Size {
            width: DevicePixels(size.width as i32),
            height: DevicePixels(size.height as i32),
        };
        self.update_path_intermediate_textures(device_pixels_size);
        self.update_backdrop_texture(device_pixels_size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            let mut msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    fn update_backdrop_texture(&mut self, size: Size<DevicePixels>) {
        // Avoid zero-sized texture creation (can SIGABRT).
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.backdrop_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::RenderTarget);
        self.backdrop_texture = Some(self.device.new_texture(&texture_descriptor));
    }

    pub fn update_transparency(&self, transparent: bool) {
        self.layer.set_opaque(!transparent);
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        // Begin texture cache frame for RTT caching
        self.texture_cache.begin_frame();
        // Begin tile cache frame for scroll container tiled rendering
        self.tile_cache.begin_frame();

        // Pre-pass: render captured subtrees to textures for RTT caching
        self.render_subtrees_to_textures(scene);

        // Pre-pass: rasterize display lists to tile textures for scroll containers
        self.rasterize_tiles_from_display_lists(scene);

        let layer = self.layer.clone();
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );

        let needs_backdrop_texture = match &self.backdrop_texture {
            Some(t) => {
                t.width() != viewport_size.width.0 as u64
                    || t.height() != viewport_size.height.0 as u64
            }
            None => true,
        };
        if needs_backdrop_texture {
            self.update_backdrop_texture(viewport_size);
        }

        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            self.pending_dirty_ranges.clear();
            self.incremental_draw = false;
            self.texture_cache.end_frame(&self.device);
            self.tile_cache.end_frame();
            return;
        };

        let mut has_dirty = if self.incremental_draw {
            !self.pending_dirty_ranges.is_empty()
        } else {
            true
        };
        if !has_dirty && scene.len() != self.last_scene_len {
            has_dirty = true;
        }
        let mut reuse_buffer = None;
        if !has_dirty {
            reuse_buffer = self.cached_instance_buffer.lock().take();
            if reuse_buffer.is_none() {
                has_dirty = true;
            }
        }

        loop {
            let mut instance_buffer = if let Some(buffer) = reuse_buffer.take() {
                buffer
            } else {
                self.instance_buffer_pool.lock().acquire(&self.device)
            };

            let command_buffer = self.draw_primitives(
                scene,
                &mut instance_buffer,
                drawable,
                viewport_size,
                has_dirty,
            );

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let cached_instance_buffer = self.cached_instance_buffer.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            let mut cached = cached_instance_buffer.lock();
                            if let Some(old_buffer) = cached.replace(instance_buffer) {
                                instance_buffer_pool.lock().release(old_buffer);
                            }
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    if self.presents_with_transaction {
                        command_buffer.commit();
                        command_buffer.wait_until_scheduled();
                        drawable.present();
                    } else {
                        command_buffer.present_drawable(drawable);
                        command_buffer.commit();
                    }
                    self.pending_dirty_ranges.clear();
                    self.incremental_draw = false;
                    self.last_scene_len = scene.len();
                    // End texture cache frame - evict unused textures
                    self.texture_cache.end_frame(&self.device);
                    self.tile_cache.end_frame();
                    return;
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        log::error!("instance buffer size grew too large: {}", buffer_size);
                        self.pending_dirty_ranges.clear();
                        self.incremental_draw = false;
                        break;
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                    reuse_buffer = None;
                    has_dirty = true;
                }
            }
        }
        // End texture cache frame if we broke out of the loop
        self.texture_cache.end_frame(&self.device);
        self.tile_cache.end_frame();
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self.layer.clone();
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        loop {
            let mut instance_buffer = self.instance_buffer_pool.lock().acquire(&self.device);

            let command_buffer =
                self.draw_primitives(scene, &mut instance_buffer, drawable, viewport_size, true);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    // Commit and wait for completion without presenting
                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    // Read pixels from the texture
                    let texture = drawable.texture();
                    let width = texture.width() as u32;
                    let height = texture.height() as u32;
                    let bytes_per_row = width as usize * 4;
                    let buffer_size = height as usize * bytes_per_row;

                    let mut pixels = vec![0u8; buffer_size];

                    let region = metal::MTLRegion {
                        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                        size: metal::MTLSize {
                            width: width as u64,
                            height: height as u64,
                            depth: 1,
                        },
                    };

                    texture.get_bytes(
                        pixels.as_mut_ptr() as *mut std::ffi::c_void,
                        bytes_per_row as u64,
                        region,
                        0,
                    );

                    // Convert BGRA to RGBA (swap B and R channels)
                    for chunk in pixels.chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }

                    return RgbaImage::from_raw(width, height, pixels).ok_or_else(|| {
                        anyhow::anyhow!("Failed to create RgbaImage from pixel data")
                    });
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        anyhow::bail!("instance buffer size grew too large: {}", buffer_size);
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
    }

    fn copy_drawable_to_backdrop(
        &self,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> bool {
        let Some(backdrop_texture) = &self.backdrop_texture else {
            return false;
        };

        if viewport_size.width.0 <= 0 || viewport_size.height.0 <= 0 {
            return false;
        }

        let blit = command_buffer.new_blit_command_encoder();
        let origin = MTLOrigin { x: 0, y: 0, z: 0 };
        let size = MTLSize {
            width: viewport_size.width.0 as u64,
            height: viewport_size.height.0 as u64,
            depth: 1,
        };

        blit.copy_from_texture(
            drawable.texture(),
            0,
            0,
            origin,
            size,
            backdrop_texture,
            0,
            0,
            origin,
        );
        blit.end_encoding();
        true
    }

    /// Incremental draw; reuses instance buffers when nothing changed.
    ///
    /// This method compares typed arrays with the previous frame to determine
    /// if buffer reuse is possible. Because batches are interleaved in the
    /// instance buffer, we can only skip writes when ALL arrays are unchanged.
    /// If any array differs, the buffer layout changes and we must rewrite all.
    pub fn draw_incremental(
        &mut self,
        scene: &Scene,
        dirty_ranges: &[std::ops::Range<usize>],
    ) {
        // Compare current scene's typed arrays with previous frame.
        // We need ALL arrays to be unchanged to safely reuse the buffer.
        self.array_dirty_flags = self.previous_frame_arrays.compare_with_scene(scene);

        // Check if ALL arrays are unchanged (can reuse entire buffer).
        // Due to interleaved buffer layout, we can't do partial updates.
        let all_arrays_clean = !self.array_dirty_flags.quads
            && !self.array_dirty_flags.shadows
            && !self.array_dirty_flags.underlines
            && !self.array_dirty_flags.paths
            && !self.array_dirty_flags.backdrop_blurs
            && !self.array_dirty_flags.monochrome_sprites
            && !self.array_dirty_flags.polychrome_sprites
            && !self.array_dirty_flags.surfaces;

        self.pending_dirty_ranges.clear();
        // If all arrays are byte-identical to previous frame, we can skip all uploads
        if !all_arrays_clean {
            self.pending_dirty_ranges.extend_from_slice(dirty_ranges);
        }
        self.incremental_draw = true;
        self.draw(scene);

        // Update previous frame arrays for next frame comparison.
        self.previous_frame_arrays.update_from_scene(scene);
    }

    fn draw_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
        write_instances: bool,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.layer.is_opaque() { 1. } else { 0. };
        let mut instance_offset = 0;

        let mut command_encoder = new_command_encoder(
            command_buffer,
            drawable,
            viewport_size,
            |color_attachment| {
                color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., alpha));
            },
        );

        for batch in scene.batches() {
            let ok = match batch {
                PrimitiveBatch::Shadows(shadows, transforms) => self.draw_shadows(
                    shadows,
                    transforms,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                    write_instances,
                ),
                PrimitiveBatch::Quads(quads, transforms) => self.draw_quads(
                    quads,
                    transforms,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                    write_instances,
                ),
                PrimitiveBatch::BackdropBlurs(blurs, transforms) => {
                    command_encoder.end_encoding();

                    let did_copy =
                        self.copy_drawable_to_backdrop(drawable, viewport_size, command_buffer);

                    command_encoder = new_command_encoder(
                        command_buffer,
                        drawable,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.set_load_action(metal::MTLLoadAction::Load);
                        },
                    );

                    if did_copy {
                        self.draw_backdrop_blurs(
                            blurs,
                            transforms,
                            instance_buffer,
                            &mut instance_offset,
                            viewport_size,
                            command_encoder,
                            write_instances,
                        )
                    } else {
                        false
                    }
                }
                PrimitiveBatch::Paths(paths) => {
                    command_encoder.end_encoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_buffer,
                        write_instances,
                    );

                    command_encoder = new_command_encoder(
                        command_buffer,
                        drawable,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.set_load_action(metal::MTLLoadAction::Load);
                        },
                    );

                    if did_draw {
                        self.draw_paths_from_intermediate(
                            paths,
                            instance_buffer,
                            &mut instance_offset,
                            viewport_size,
                            command_encoder,
                            write_instances,
                        )
                    } else {
                        false
                    }
                }
                PrimitiveBatch::Underlines(underlines, transforms) => self.draw_underlines(
                    underlines,
                    transforms,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                    write_instances,
                ),
                PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites,
                } => self.draw_monochrome_sprites(
                    texture_id,
                    sprites,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                    write_instances,
                ),
                PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites,
                    transforms,
                } => self.draw_polychrome_sprites(
                    texture_id,
                    sprites,
                    transforms,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                    write_instances,
                ),
                PrimitiveBatch::Surfaces(surfaces) => self.draw_surfaces(
                    surfaces,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                    write_instances,
                ),
                PrimitiveBatch::CachedTextures(sprites) => self.draw_cached_textures(
                    sprites,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                    write_instances,
                ),
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
                PrimitiveBatch::TileSprites(sprites) => self.draw_tile_sprites(
                    sprites,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                    write_instances,
                ),
            };
            if !ok {
                command_encoder.end_encoding();
                anyhow::bail!(
                    "scene too large: {} paths, {} shadows, {} quads, {} blurs, {} underlines, {} mono, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.backdrop_blurs.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                );
            }
        }

        command_encoder.end_encoding();

        if write_instances {
            instance_buffer.metal_buffer.did_modify_range(NSRange {
                location: 0,
                length: instance_offset as NSUInteger,
            });
        }
        Ok(command_buffer.to_owned())
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
        write_instances: bool,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }
        let Some(intermediate_texture) = &self.path_intermediate_texture else {
            return false;
        };

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);

        align_offset(instance_offset);
        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertices_bytes_len = mem::size_of_val(vertices.as_slice());
        let next_offset = *instance_offset + vertices_bytes_len;
        if next_offset > instance_buffer.size {
            command_encoder.end_encoding();
            return false;
        }
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        if write_instances {
            let buffer_contents = unsafe {
                (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset)
            };
            unsafe {
                ptr::copy_nonoverlapping(
                    vertices.as_ptr() as *const u8,
                    buffer_contents,
                    vertices_bytes_len,
                );
            }
        }
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );
        *instance_offset = next_offset;

        command_encoder.end_encoding();
        true
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        shadow_transforms: &[TransformationMatrix],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        if shadows.is_empty() {
            return true;
        }
        debug_assert_eq!(shadows.len(), shadow_transforms.len());
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        let shadows_offset = *instance_offset;

        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let shadow_bytes_len = mem::size_of_val(shadows);
        let mut transforms_offset = shadows_offset + shadow_bytes_len;
        align_offset(&mut transforms_offset);
        let transform_bytes_len = mem::size_of_val(shadow_transforms);
        let next_offset = transforms_offset + transform_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            shadows_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            shadows_offset as u64,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );

        if write_instances {
            let shadow_contents = unsafe {
                (instance_buffer.metal_buffer.contents() as *mut u8).add(shadows_offset)
            };
            let transform_contents = unsafe {
                (instance_buffer.metal_buffer.contents() as *mut u8).add(transforms_offset)
            };

            unsafe {
                ptr::copy_nonoverlapping(
                    shadows.as_ptr() as *const u8,
                    shadow_contents,
                    shadow_bytes_len,
                );
                ptr::copy_nonoverlapping(
                    shadow_transforms.as_ptr() as *const u8,
                    transform_contents,
                    transform_bytes_len,
                );
            }
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        quad_transforms: &[TransformationMatrix],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        if quads.is_empty() {
            return true;
        }
        debug_assert_eq!(quads.len(), quad_transforms.len());
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        let quads_offset = *instance_offset;

        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let quad_bytes_len = mem::size_of_val(quads);
        let mut transforms_offset = quads_offset + quad_bytes_len;
        align_offset(&mut transforms_offset);
        let transform_bytes_len = mem::size_of_val(quad_transforms);
        let next_offset = transforms_offset + transform_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            quads_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            quads_offset as u64,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );

        if write_instances {
            unsafe {
                let quad_contents =
                    (instance_buffer.metal_buffer.contents() as *mut u8).add(quads_offset);
                ptr::copy_nonoverlapping(
                    quads.as_ptr() as *const u8,
                    quad_contents,
                    quad_bytes_len,
                );

                let transform_contents =
                    (instance_buffer.metal_buffer.contents() as *mut u8).add(transforms_offset);
                ptr::copy_nonoverlapping(
                    quad_transforms.as_ptr() as *const u8,
                    transform_contents,
                    transform_bytes_len,
                );
            }
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_backdrop_blurs(
        &self,
        blurs: &[BackdropBlur],
        blur_transforms: &[TransformationMatrix],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        if blurs.is_empty() {
            return true;
        }
        debug_assert_eq!(blurs.len(), blur_transforms.len());
        align_offset(instance_offset);

        let Some(backdrop_texture) = &self.backdrop_texture else {
            return false;
        };

        command_encoder.set_render_pipeline_state(&self.backdrop_blurs_pipeline_state);
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        let blurs_offset = *instance_offset;

        command_encoder.set_vertex_bytes(
            BackdropBlurInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_bytes(
            BackdropBlurInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            BackdropBlurInputIndex::BackdropTexture as u64,
            Some(backdrop_texture),
        );

        let blur_bytes_len = mem::size_of_val(blurs);
        let mut transforms_offset = blurs_offset + blur_bytes_len;
        align_offset(&mut transforms_offset);
        let transform_bytes_len = mem::size_of_val(blur_transforms);
        let next_offset = transforms_offset + transform_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instance_buffer.metal_buffer),
            blurs_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            BackdropBlurInputIndex::BackdropBlurs as u64,
            Some(&instance_buffer.metal_buffer),
            blurs_offset as u64,
        );
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            BackdropBlurInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );

        if write_instances {
            unsafe {
                let blur_contents =
                    (instance_buffer.metal_buffer.contents() as *mut u8).add(blurs_offset);
                ptr::copy_nonoverlapping(
                    blurs.as_ptr() as *const u8,
                    blur_contents,
                    blur_bytes_len,
                );

                let transform_contents =
                    (instance_buffer.metal_buffer.contents() as *mut u8).add(transforms_offset);
                ptr::copy_nonoverlapping(
                    blur_transforms.as_ptr() as *const u8,
                    transform_contents,
                    transform_bytes_len,
                );
            }
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            blurs.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        let Some(first_path) = paths.first() else {
            return true;
        };

        let Some(ref intermediate_texture) = self.path_intermediate_texture else {
            return false;
        };

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        align_offset(instance_offset);
        let sprite_bytes_len = mem::size_of_val(sprites.as_slice());
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        if write_instances {
            let buffer_contents = unsafe {
                (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset)
            };
            unsafe {
                ptr::copy_nonoverlapping(
                    sprites.as_ptr() as *const u8,
                    buffer_contents,
                    sprite_bytes_len,
                );
            }
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;

        true
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        underline_transforms: &[TransformationMatrix],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        if underlines.is_empty() {
            return true;
        }
        debug_assert_eq!(underlines.len(), underline_transforms.len());
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        let underlines_offset = *instance_offset;

        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let underline_bytes_len = mem::size_of_val(underlines);
        let mut transforms_offset = underlines_offset + underline_bytes_len;
        align_offset(&mut transforms_offset);
        let transform_bytes_len = mem::size_of_val(underline_transforms);
        let next_offset = transforms_offset + transform_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            underlines_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            underlines_offset as u64,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );

        if write_instances {
            unsafe {
                let underline_contents =
                    (instance_buffer.metal_buffer.contents() as *mut u8).add(underlines_offset);
                ptr::copy_nonoverlapping(
                    underlines.as_ptr() as *const u8,
                    underline_contents,
                    underline_bytes_len,
                );

                let transform_contents =
                    (instance_buffer.metal_buffer.contents() as *mut u8).add(transforms_offset);
                ptr::copy_nonoverlapping(
                    underline_transforms.as_ptr() as *const u8,
                    transform_contents,
                    transform_bytes_len,
                );
            }
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[MonochromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        if write_instances {
            unsafe {
                ptr::copy_nonoverlapping(
                    sprites.as_ptr() as *const u8,
                    buffer_contents,
                    sprite_bytes_len,
                );
            }
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[PolychromeSprite],
        sprite_transforms: &[TransformationMatrix],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        debug_assert_eq!(sprites.len(), sprite_transforms.len());
        align_offset(instance_offset);

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        let sprite_bytes_len = mem::size_of_val(sprites);
        let sprites_offset = *instance_offset;
        let mut transforms_offset = sprites_offset + sprite_bytes_len;
        align_offset(&mut transforms_offset);
        let transform_bytes_len = mem::size_of_val(sprite_transforms);
        let next_offset = transforms_offset + transform_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Transforms as u64,
            Some(&instance_buffer.metal_buffer),
            transforms_offset as u64,
        );

        if write_instances {
            unsafe {
                let sprite_contents =
                    (instance_buffer.metal_buffer.contents() as *mut u8).add(sprites_offset);
                ptr::copy_nonoverlapping(
                    sprites.as_ptr() as *const u8,
                    sprite_contents,
                    sprite_bytes_len,
                );

                let transform_contents =
                    (instance_buffer.metal_buffer.contents() as *mut u8).add(transforms_offset);
                ptr::copy_nonoverlapping(
                    sprite_transforms.as_ptr() as *const u8,
                    transform_contents,
                    transform_bytes_len,
                );
            }
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for surface in surfaces {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );

            assert_eq!(
                surface.image_buffer.get_pixel_format(),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::R8Unorm,
                    surface.image_buffer.get_width_of_plane(0),
                    surface.image_buffer.get_height_of_plane(0),
                    0,
                )
                .unwrap();
            let cb_cr_texture = self
                .core_video_texture_cache
                .create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    MTLPixelFormat::RG8Unorm,
                    surface.image_buffer.get_width_of_plane(1),
                    surface.image_buffer.get_height_of_plane(1),
                    1,
                )
                .unwrap();

            align_offset(instance_offset);
            let next_offset = *instance_offset + mem::size_of::<Surface>();
            if next_offset > instance_buffer.size {
                return false;
            }

            command_encoder.set_vertex_buffer(
                SurfaceInputIndex::Surfaces as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );
            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            // let y_texture = y_texture.get_texture().unwrap().
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });

            if write_instances {
                unsafe {
                    let buffer_contents = (instance_buffer.metal_buffer.contents() as *mut u8)
                        .add(*instance_offset)
                        as *mut SurfaceBounds;
                    ptr::write(
                        buffer_contents,
                        SurfaceBounds {
                            bounds: surface.bounds,
                            content_mask: surface.content_mask.clone(),
                        },
                    );
                }
            }

            command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
            *instance_offset = next_offset;
        }
        true
    }

    /// Draws a cached texture sprite to the screen.
    /// Used for render-to-texture caching where subtrees are rendered once
    /// and then composited at potentially different offsets.
    #[allow(dead_code)]
    fn draw_cached_texture(
        &mut self,
        bounds: Bounds<ScaledPixels>,
        content_mask: ContentMask<ScaledPixels>,
        uv_bounds: Bounds<f32>,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        instance_buffer: &InstanceBuffer,
        instance_offset: &mut usize,
        write_instances: bool,
    ) -> bool {
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of::<CachedTextureSpriteGpu>();
        let next_offset = *instance_offset + sprite_bytes_len;

        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.set_render_pipeline_state(&self.cached_textures_pipeline_state);
        command_encoder.set_vertex_buffer(
            CachedTextureInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            CachedTextureInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            CachedTextureInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_texture(
            CachedTextureInputIndex::Texture as u64,
            Some(texture),
        );

        if write_instances {
            unsafe {
                let buffer_contents = (instance_buffer.metal_buffer.contents() as *mut u8)
                    .add(*instance_offset) as *mut CachedTextureSpriteGpu;
                ptr::write(
                    buffer_contents,
                    CachedTextureSpriteGpu {
                        bounds,
                        content_mask,
                        uv_bounds,
                    },
                );
            }
        }

        command_encoder.draw_primitives_instanced(metal::MTLPrimitiveType::Triangle, 0, 6, 1);
        *instance_offset = next_offset;
        true
    }

    /// Pre-pass: render captured subtrees to offscreen textures.
    /// This enables O(1) scrolling by compositing cached textures instead of re-rendering.
    fn render_subtrees_to_textures(&mut self, scene: &Scene) {
        let captures = scene.subtree_captures();
        if captures.is_empty() {
            return;
        }

        // Create command buffer for RTT pre-pass
        let command_buffer = self.command_queue.new_command_buffer();

        for capture in captures {
            // Convert ScaledPixels to DevicePixels (both are physical pixel coordinates)
            let texture_size: Size<DevicePixels> = size(
                DevicePixels(capture.bounds.size.width.0.ceil() as i32),
                DevicePixels(capture.bounds.size.height.0.ceil() as i32),
            );
            let content_bounds: Bounds<DevicePixels> = Bounds {
                origin: point(
                    DevicePixels(capture.bounds.origin.x.0.round() as i32),
                    DevicePixels(capture.bounds.origin.y.0.round() as i32),
                ),
                size: texture_size,
            };

            // Skip if texture is too small or too large
            if texture_size.width.0 < 64
                || texture_size.height.0 < 64
                || texture_size.width.0 > 4096
                || texture_size.height.0 > 4096
            {
                continue;
            }

            // Acquire texture from the pool
            // Use a simple signature based on the element ID hash for now
            let signature = {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                capture.id.hash(&mut hasher);
                hasher.finish()
            };

            let acquire_result = self.texture_cache.acquire(
                &self.device,
                capture.id.clone(),
                texture_size,
                signature,
                content_bounds,
            );

            // Skip rendering if texture already exists and is up to date
            if !acquire_result.needs_render {
                log::trace!(
                    "RTT: reusing existing texture for element, skipping render"
                );
                continue;
            }

            let texture = acquire_result.entry.texture.texture.clone();

            // Create render pass descriptor targeting the texture
            let render_pass_descriptor = metal::RenderPassDescriptor::new();
            let color_attachment = render_pass_descriptor
                .color_attachments()
                .object_at(0)
                .unwrap();
            color_attachment.set_texture(Some(&texture));
            color_attachment.set_load_action(metal::MTLLoadAction::Clear);
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
            // Clear to transparent
            color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

            // Create command encoder for this texture
            let command_encoder =
                command_buffer.new_render_command_encoder(&render_pass_descriptor);
            command_encoder.set_viewport(metal::MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: texture_size.width.0 as f64,
                height: texture_size.height.0 as f64,
                znear: 0.0,
                zfar: 1.0,
            });

            // Create mini-scene with primitives translated to texture coordinates
            let mini_scene = scene.create_capture_scene(capture);

            // Render all primitive types to the texture
            // Order matters for correct blending - render back to front

            // 1. Shadows (typically behind other content)
            if !mini_scene.shadows.is_empty() {
                Self::render_shadows_to_encoder_static(
                    &self.device,
                    &self.shadows_pipeline_state,
                    &self.unit_vertices,
                    &mini_scene.shadows,
                    &mini_scene.shadow_transforms,
                    texture_size,
                    command_encoder,
                );
            }

            // 2. Quads (backgrounds, borders)
            if !mini_scene.quads.is_empty() {
                Self::render_quads_to_encoder_static(
                    &self.device,
                    &self.quads_pipeline_state,
                    &self.unit_vertices,
                    &mini_scene.quads,
                    &mini_scene.quad_transforms,
                    texture_size,
                    command_encoder,
                );
            }

            // 3. Underlines
            if !mini_scene.underlines.is_empty() {
                Self::render_underlines_to_encoder_static(
                    &self.device,
                    &self.underlines_pipeline_state,
                    &self.unit_vertices,
                    &mini_scene.underlines,
                    &mini_scene.underline_transforms,
                    texture_size,
                    command_encoder,
                );
            }

            // 4. Monochrome sprites (text)
            if !mini_scene.monochrome_sprites.is_empty() {
                Self::render_monochrome_sprites_to_encoder_static(
                    &self.device,
                    &self.monochrome_sprites_pipeline_state,
                    &self.unit_vertices,
                    &self.sprite_atlas,
                    &mini_scene.monochrome_sprites,
                    texture_size,
                    command_encoder,
                );
            }

            // 5. Polychrome sprites (colored text, icons)
            if !mini_scene.polychrome_sprites.is_empty() {
                Self::render_polychrome_sprites_to_encoder_static(
                    &self.device,
                    &self.polychrome_sprites_pipeline_state,
                    &self.unit_vertices,
                    &self.sprite_atlas,
                    &mini_scene.polychrome_sprites,
                    &mini_scene.polychrome_sprite_transforms,
                    texture_size,
                    command_encoder,
                );
            }

            // Note: Paths and surfaces are more complex (require intermediate textures)
            // and are less common in scrollable content. Skipping for now.
            // TODO: Add path rendering if needed for RTT

            command_encoder.end_encoding();
        }

        // Commit the RTT command buffer without waiting.
        // Phase 0.2: Metal guarantees ordering on the same queue - the main pass
        // command buffer will naturally wait for this prepass to complete.
        command_buffer.commit();
    }

    /// Pre-pass: rasterize display lists to tile textures for scroll containers.
    ///
    /// This iterates over all TileSprites in the scene, checks if their tiles need
    /// rasterization, and if so, renders the corresponding display list to the tile texture.
    ///
    /// **Threaded Rasterization (Phase 15)**: The CPU-bound work (converting DisplayItems
    /// to Scene primitives via `rasterize_tile()`) is performed in parallel using rayon,
    /// while the GPU work (Metal rendering) remains sequential on the main thread.
    fn rasterize_tiles_from_display_lists(&mut self, scene: &Scene) {
        let tile_sprites = &scene.tile_sprites;
        if tile_sprites.is_empty() {
            return;
        }

        // Get scale factor from layer contents scale
        // macOS retina displays typically use 2.0, non-retina use 1.0
        let scale_factor = self.layer.contents_scale() as f32;

        // Collect tiles that need rendering as TileRasterJobs
        let mut jobs: Vec<TileRasterJob> = Vec::new();

        for sprite in tile_sprites {
            let container_id = &sprite.tile_key.container_id;
            let coord = sprite.tile_key.coord;

            // Look up the display list for this container
            let Some((display_list, _)) = scene.get_display_list(container_id) else {
                continue;
            };

            // Get the content generation from the display list
            let content_generation = display_list.generation;

            // Try to acquire the tile - this creates/finds the tile texture and tells us if it needs rendering
            let needs_render = self.tile_cache.acquire_tile(
                container_id,
                coord,
                content_generation,
            );

            if needs_render {
                jobs.push(TileRasterJob {
                    container_id: container_id.clone(),
                    coord,
                    display_list: Arc::clone(display_list),
                    priority: TilePriority::VISIBLE,
                });
            }
        }

        if jobs.is_empty() {
            return;
        }

        // Parallel CPU rasterization using rayon
        // This is the performance-critical change: multiple tiles rasterize concurrently
        let results = rasterize_tiles_parallel(jobs, scale_factor);

        // Phase 0.2: Sequential GPU encoding, but no synchronous waits.
        // Each tile gets its own command buffer for better GPU parallelism.
        // Metal guarantees ordering on the same queue - main pass will wait for these.
        for result in results {
            self.rasterize_display_list_tile(
                &result.container_id,
                result.coord,
                &result.raster_result,
            );
        }
    }

    /// Render quads directly to an encoder (for RTT pre-pass).
    /// This is a static method to avoid borrow conflicts with command_buffer.
    fn render_quads_to_encoder_static(
        device: &metal::Device,
        pipeline_state: &metal::RenderPipelineState,
        unit_vertices: &metal::Buffer,
        quads: &[crate::scene::Quad],
        transforms: &[crate::TransformationMatrix],
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if quads.is_empty() {
            return;
        }

        // Allocate a temporary instance buffer for quads
        let quad_size = std::mem::size_of::<crate::scene::Quad>();
        let quad_buffer_size = quads.len() * quad_size;

        let quad_buffer = device.new_buffer(
            quad_buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Copy quad data to the buffer
        unsafe {
            let ptr = quad_buffer.contents() as *mut crate::scene::Quad;
            std::ptr::copy_nonoverlapping(quads.as_ptr(), ptr, quads.len());
        }

        // Allocate transforms buffer
        let transform_size = std::mem::size_of::<crate::TransformationMatrix>();
        let transform_buffer_size = transforms.len() * transform_size;

        let transform_buffer = device.new_buffer(
            transform_buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Copy transform data to the buffer
        unsafe {
            let ptr = transform_buffer.contents() as *mut crate::TransformationMatrix;
            std::ptr::copy_nonoverlapping(transforms.as_ptr(), ptr, transforms.len());
        }

        // Set up pipeline state
        command_encoder.set_render_pipeline_state(pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&quad_buffer),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Transforms as u64,
            Some(&transform_buffer),
            0,
        );
        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            std::mem::size_of::<Size<DevicePixels>>() as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        // Set fragment buffers (needed for quad_fragment shader)
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&quad_buffer),
            0,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Transforms as u64,
            Some(&transform_buffer),
            0,
        );

        // Draw instanced
        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6, // 2 triangles = 6 vertices for a quad
            quads.len() as u64,
        );
    }

    /// Render shadows directly to an encoder (for RTT pre-pass).
    fn render_shadows_to_encoder_static(
        device: &metal::Device,
        pipeline_state: &metal::RenderPipelineState,
        unit_vertices: &metal::Buffer,
        shadows: &[crate::scene::Shadow],
        transforms: &[crate::TransformationMatrix],
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if shadows.is_empty() {
            return;
        }

        // Allocate buffer for shadows + transforms
        let shadow_bytes = std::mem::size_of_val(shadows);
        let transform_bytes = std::mem::size_of_val(transforms);
        let buffer_size = shadow_bytes + 16 + transform_bytes; // 16 for alignment

        let instance_buffer = device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Copy data to buffer
        let shadow_offset = 0usize;
        let mut transform_offset = shadow_bytes;
        // Align to 16 bytes
        transform_offset = (transform_offset + 15) & !15;

        unsafe {
            let base = instance_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(
                shadows.as_ptr() as *const u8,
                base.add(shadow_offset),
                shadow_bytes,
            );
            std::ptr::copy_nonoverlapping(
                transforms.as_ptr() as *const u8,
                base.add(transform_offset),
                transform_bytes,
            );
        }

        // Set up pipeline state
        command_encoder.set_render_pipeline_state(pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer),
            shadow_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer),
            shadow_offset as u64,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Transforms as u64,
            Some(&instance_buffer),
            transform_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Transforms as u64,
            Some(&instance_buffer),
            transform_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            std::mem::size_of::<Size<DevicePixels>>() as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
        );
    }

    /// Render underlines directly to an encoder (for RTT pre-pass).
    fn render_underlines_to_encoder_static(
        device: &metal::Device,
        pipeline_state: &metal::RenderPipelineState,
        unit_vertices: &metal::Buffer,
        underlines: &[crate::scene::Underline],
        transforms: &[crate::TransformationMatrix],
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if underlines.is_empty() {
            return;
        }

        // Allocate buffer for underlines + transforms
        let underline_bytes = std::mem::size_of_val(underlines);
        let transform_bytes = std::mem::size_of_val(transforms);
        let buffer_size = underline_bytes + 16 + transform_bytes;

        let instance_buffer = device.new_buffer(
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let underline_offset = 0usize;
        let mut transform_offset = underline_bytes;
        transform_offset = (transform_offset + 15) & !15;

        unsafe {
            let base = instance_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(
                underlines.as_ptr() as *const u8,
                base.add(underline_offset),
                underline_bytes,
            );
            std::ptr::copy_nonoverlapping(
                transforms.as_ptr() as *const u8,
                base.add(transform_offset),
                transform_bytes,
            );
        }

        command_encoder.set_render_pipeline_state(pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer),
            underline_offset as u64,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Transforms as u64,
            Some(&instance_buffer),
            transform_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            std::mem::size_of::<Size<DevicePixels>>() as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
        );
    }

    /// Render monochrome sprites (text) directly to an encoder (for RTT pre-pass).
    fn render_monochrome_sprites_to_encoder_static(
        device: &metal::Device,
        pipeline_state: &metal::RenderPipelineState,
        unit_vertices: &metal::Buffer,
        sprite_atlas: &Arc<MetalAtlas>,
        sprites: &[crate::scene::MonochromeSprite],
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        // Group sprites by texture ID
        let mut sprites_by_texture: FxHashMap<AtlasTextureId, Vec<&crate::scene::MonochromeSprite>> =
            FxHashMap::default();
        for sprite in sprites {
            sprites_by_texture
                .entry(sprite.tile.texture_id)
                .or_default()
                .push(sprite);
        }

        for (texture_id, batch) in sprites_by_texture {
            let sprite_bytes = batch.len() * std::mem::size_of::<crate::scene::MonochromeSprite>();

            let instance_buffer = device.new_buffer(
                sprite_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            // Copy sprites to buffer
            unsafe {
                let base = instance_buffer.contents() as *mut crate::scene::MonochromeSprite;
                for (i, sprite) in batch.iter().enumerate() {
                    std::ptr::copy_nonoverlapping(*sprite, base.add(i), 1);
                }
            }

            let texture = sprite_atlas.metal_texture(texture_id);
            let texture_size = size(
                DevicePixels(texture.width() as i32),
                DevicePixels(texture.height() as i32),
            );

            command_encoder.set_render_pipeline_state(pipeline_state);
            command_encoder.set_vertex_buffer(
                SpriteInputIndex::Vertices as u64,
                Some(unit_vertices),
                0,
            );
            command_encoder.set_vertex_buffer(
                SpriteInputIndex::Sprites as u64,
                Some(&instance_buffer),
                0,
            );
            command_encoder.set_vertex_bytes(
                SpriteInputIndex::ViewportSize as u64,
                std::mem::size_of::<Size<DevicePixels>>() as u64,
                &viewport_size as *const Size<DevicePixels> as *const _,
            );
            command_encoder.set_vertex_bytes(
                SpriteInputIndex::AtlasTextureSize as u64,
                std::mem::size_of::<Size<DevicePixels>>() as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            command_encoder.set_fragment_buffer(
                SpriteInputIndex::Sprites as u64,
                Some(&instance_buffer),
                0,
            );
            command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

            command_encoder.draw_primitives_instanced(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                batch.len() as u64,
            );
        }
    }

    /// Render polychrome sprites directly to an encoder (for RTT pre-pass).
    fn render_polychrome_sprites_to_encoder_static(
        device: &metal::Device,
        pipeline_state: &metal::RenderPipelineState,
        unit_vertices: &metal::Buffer,
        sprite_atlas: &Arc<MetalAtlas>,
        sprites: &[crate::scene::PolychromeSprite],
        transforms: &[crate::TransformationMatrix],
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        // Group sprites by texture ID (keeping track of transforms)
        let mut sprites_by_texture: FxHashMap<
            AtlasTextureId,
            Vec<(&crate::scene::PolychromeSprite, &crate::TransformationMatrix)>,
        > = FxHashMap::default();
        for (sprite, transform) in sprites.iter().zip(transforms.iter()) {
            sprites_by_texture
                .entry(sprite.tile.texture_id)
                .or_default()
                .push((sprite, transform));
        }

        for (texture_id, batch) in sprites_by_texture {
            let sprite_bytes =
                batch.len() * std::mem::size_of::<crate::scene::PolychromeSprite>();
            let transform_bytes = batch.len() * std::mem::size_of::<crate::TransformationMatrix>();
            let buffer_size = sprite_bytes + 16 + transform_bytes;

            let instance_buffer = device.new_buffer(
                buffer_size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let sprite_offset = 0usize;
            let mut transform_offset = sprite_bytes;
            transform_offset = (transform_offset + 15) & !15;

            // Copy sprites and transforms to buffer
            unsafe {
                let base = instance_buffer.contents() as *mut u8;
                let sprite_ptr = base as *mut crate::scene::PolychromeSprite;
                let transform_ptr = base.add(transform_offset) as *mut crate::TransformationMatrix;
                for (i, (sprite, transform)) in batch.iter().enumerate() {
                    std::ptr::copy_nonoverlapping(*sprite, sprite_ptr.add(i), 1);
                    std::ptr::copy_nonoverlapping(*transform, transform_ptr.add(i), 1);
                }
            }

            let texture = sprite_atlas.metal_texture(texture_id);
            let texture_size = size(
                DevicePixels(texture.width() as i32),
                DevicePixels(texture.height() as i32),
            );

            command_encoder.set_render_pipeline_state(pipeline_state);
            command_encoder.set_vertex_buffer(
                SpriteInputIndex::Vertices as u64,
                Some(unit_vertices),
                0,
            );
            command_encoder.set_vertex_buffer(
                SpriteInputIndex::Sprites as u64,
                Some(&instance_buffer),
                sprite_offset as u64,
            );
            command_encoder.set_vertex_buffer(
                SpriteInputIndex::Transforms as u64,
                Some(&instance_buffer),
                transform_offset as u64,
            );
            command_encoder.set_vertex_bytes(
                SpriteInputIndex::ViewportSize as u64,
                std::mem::size_of::<Size<DevicePixels>>() as u64,
                &viewport_size as *const Size<DevicePixels> as *const _,
            );
            command_encoder.set_vertex_bytes(
                SpriteInputIndex::AtlasTextureSize as u64,
                std::mem::size_of::<Size<DevicePixels>>() as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            command_encoder.set_fragment_buffer(
                SpriteInputIndex::Sprites as u64,
                Some(&instance_buffer),
                sprite_offset as u64,
            );
            command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

            command_encoder.draw_primitives_instanced(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                batch.len() as u64,
            );
        }
    }

    /// Draws a batch of cached texture sprites to the screen.
    fn draw_cached_textures(
        &mut self,
        sprites: &[crate::scene::CachedTextureSprite],
        instance_buffer: &InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        for sprite in sprites {
            // Look up the texture from the cache and clone the texture reference
            // (metal::Texture implements Clone as an Arc-like reference)
            let texture = self
                .texture_cache
                .lookup_by_texture_id(sprite.texture_id)
                .map(|entry| entry.texture.texture.clone());

            if let Some(texture) = texture {
                let ok = self.draw_cached_texture(
                    sprite.bounds,
                    sprite.content_mask.clone(),
                    sprite.uv_bounds,
                    &texture,
                    viewport_size,
                    command_encoder,
                    instance_buffer,
                    instance_offset,
                    write_instances,
                );
                if !ok {
                    return false;
                }
            } else {
                log::warn!(
                    "draw_cached_textures: texture {:?} not found (evicted?), skipping sprite",
                    sprite.texture_id
                );
            }
        }
        true
    }

    /// Rasterize a DisplayList to a tile texture.
    ///
    /// This is the key method for the display list architecture. It takes a
    /// Encodes tile rendering commands to an existing command buffer.
    /// Phase 0.2: This is the internal method used for batched tile rendering
    /// to eliminate per-tile GPU stalls.
    fn encode_tile_to_command_buffer(
        &self,
        container_id: &crate::GlobalElementId,
        coord: crate::scene::TileCoord,
        raster_result: &crate::display_list::TileRasterResult,
        command_buffer: &metal::CommandBufferRef,
    ) {
        use crate::scene::TileKey;

        // Get the tile texture
        let tile_key = TileKey {
            container_id: container_id.clone(),
            coord,
        };

        let texture = match self.tile_cache.get_tile_texture(&tile_key) {
            Some(t) => t.clone(),
            None => {
                log::warn!(
                    "encode_tile_to_command_buffer: tile {:?} not found in cache",
                    tile_key
                );
                return;
            }
        };

        // Skip if nothing to render
        if raster_result.is_empty() {
            return;
        }

        let tile_size: Size<DevicePixels> = size(
            DevicePixels(super::tile_cache::TILE_SIZE as i32),
            DevicePixels(super::tile_cache::TILE_SIZE as i32),
        );

        // Create render pass descriptor targeting the tile texture
        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_texture(Some(&texture));
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_store_action(metal::MTLStoreAction::Store);
        // Clear to transparent
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        // Create command encoder for this tile
        let command_encoder = command_buffer.new_render_command_encoder(&render_pass_descriptor);
        command_encoder.set_viewport(metal::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: super::tile_cache::TILE_SIZE as f64,
            height: super::tile_cache::TILE_SIZE as f64,
            znear: 0.0,
            zfar: 1.0,
        });

        // Render all primitive types to the tile
        // Order matters for correct blending - render back to front

        // 1. Shadows (typically behind other content)
        if !raster_result.shadows.is_empty() {
            Self::render_shadows_to_encoder_static(
                &self.device,
                &self.shadows_pipeline_state,
                &self.unit_vertices,
                &raster_result.shadows,
                &raster_result.shadow_transforms,
                tile_size,
                command_encoder,
            );
        }

        // 2. Quads (backgrounds, borders)
        if !raster_result.quads.is_empty() {
            Self::render_quads_to_encoder_static(
                &self.device,
                &self.quads_pipeline_state,
                &self.unit_vertices,
                &raster_result.quads,
                &raster_result.quad_transforms,
                tile_size,
                command_encoder,
            );
        }

        // 3. Underlines
        if !raster_result.underlines.is_empty() {
            Self::render_underlines_to_encoder_static(
                &self.device,
                &self.underlines_pipeline_state,
                &self.unit_vertices,
                &raster_result.underlines,
                &raster_result.underline_transforms,
                tile_size,
                command_encoder,
            );
        }

        // 4. Monochrome sprites (text)
        if !raster_result.monochrome_sprites.is_empty() {
            Self::render_monochrome_sprites_to_encoder_static(
                &self.device,
                &self.monochrome_sprites_pipeline_state,
                &self.unit_vertices,
                &self.sprite_atlas,
                &raster_result.monochrome_sprites,
                tile_size,
                command_encoder,
            );
        }

        // 5. Polychrome sprites (colored text, icons)
        if !raster_result.polychrome_sprites.is_empty() {
            Self::render_polychrome_sprites_to_encoder_static(
                &self.device,
                &self.polychrome_sprites_pipeline_state,
                &self.unit_vertices,
                &self.sprite_atlas,
                &raster_result.polychrome_sprites,
                &raster_result.polychrome_sprite_transforms,
                tile_size,
                command_encoder,
            );
        }

        // Note: Paths and backdrop blurs are more complex and less common in scrollable content.
        // TODO: Add path and backdrop blur rendering if needed for display list tiles

        command_encoder.end_encoding();
    }

    /// Renders a TileRasterResult to the tile texture acquired from the tile cache.
    ///
    /// The caller is responsible for:
    /// 1. Calling `tile_cache.acquire_tile()` to get a tile texture
    /// 2. Calling this method only if `acquire_tile` returned `true` (needs_render)
    /// 3. Ensuring the TileRasterResult contains primitives offset to tile-local coordinates
    ///
    /// Phase 0.2: Commits without waiting - Metal queue ordering handles correctness.
    pub fn rasterize_display_list_tile(
        &self,
        container_id: &crate::GlobalElementId,
        coord: crate::scene::TileCoord,
        raster_result: &crate::display_list::TileRasterResult,
    ) {
        // Create command buffer and encode tile rendering.
        // Phase 0.2: No synchronous wait - Metal queue ordering handles correctness.
        let command_buffer = self.command_queue.new_command_buffer();
        self.encode_tile_to_command_buffer(container_id, coord, raster_result, command_buffer);
        command_buffer.commit();
    }

    /// Access to the tile cache for registering scroll containers and acquiring tiles.
    #[allow(dead_code)]
    pub fn tile_cache_mut(&mut self) -> &mut TileCache {
        &mut self.tile_cache
    }

    /// Draws a batch of tile sprites to the screen.
    /// This composites tiles from scroll container tile caches.
    fn draw_tile_sprites(
        &mut self,
        sprites: &[TileSprite],
        instance_buffer: &InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
        write_instances: bool,
    ) -> bool {
        for sprite in sprites {
            // Look up the tile texture from the tile cache and clone the reference
            // (metal::Texture implements Clone as an Arc-like reference)
            let texture = self
                .tile_cache
                .get_tile_texture(&sprite.tile_key)
                .cloned();

            if let Some(texture) = texture {
                // Tiles use full UV bounds (0,0 to 1,1) since they are exact size
                let uv_bounds = Bounds {
                    origin: point(0.0, 0.0),
                    size: size(1.0, 1.0),
                };

                let ok = self.draw_cached_texture(
                    sprite.bounds,
                    sprite.content_mask.clone(),
                    uv_bounds,
                    &texture,
                    viewport_size,
                    command_encoder,
                    instance_buffer,
                    instance_offset,
                    write_instances,
                );
                if !ok {
                    return false;
                }
            } else {
                log::warn!(
                    "draw_tile_sprites: tile {:?} not found (evicted?), skipping sprite",
                    sprite.tile_key
                );
            }
        }
        true
    }
}

fn new_command_encoder<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    drawable: &'a metal::MetalDrawableRef,
    viewport_size: Size<DevicePixels>,
    configure_color_attachment: impl Fn(&RenderPassColorAttachmentDescriptorRef),
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(drawable.texture()));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    configure_color_attachment(color_attachment);

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// Build a pipeline state for rendering content that already has premultiplied alpha.
/// This is used for cached texture compositing where the texture was rendered with
/// standard alpha blending onto a transparent background, resulting in premultiplied content.
fn build_premultiplied_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    // Use One for source RGB since texture content is already premultiplied
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

// Align to multiples of 256 make Metal happy.
fn align_offset(offset: &mut usize) {
    *offset = (*offset).div_ceil(256) * 256;
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
    Transforms = 3,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
    Transforms = 3,
}

#[repr(C)]
enum BackdropBlurInputIndex {
    Vertices = 0,
    BackdropBlurs = 1,
    ViewportSize = 2,
    BackdropTexture = 3,
    Transforms = 4,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
    Transforms = 3,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
    Transforms = 5,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[repr(C)]
enum CachedTextureInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    Texture = 3,
}

/// GPU-side struct for rendering cached textures.
/// This is a simplified version of CachedTextureSprite for the shader.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct CachedTextureSpriteGpu {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub uv_bounds: Bounds<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}
