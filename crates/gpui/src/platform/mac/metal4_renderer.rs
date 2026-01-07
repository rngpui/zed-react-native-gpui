//! Metal 4 renderer implementation for GPUI.
//!
//! This renderer leverages Metal 4 APIs including:
//! - MTL4CommandAllocator for explicit command memory management
//! - MTL4ArgumentTable for efficient resource binding
//! - MTL4RenderCommandEncoder with argument table support
//! - MTLResidencySet for GPU memory residency management
//! - MTLSharedEvent for CPU-GPU synchronization

use super::metal_atlas::MetalAtlas;
use super::metal_renderer::InstanceBufferPool;
use crate::{
    Background, Bounds, ContentMask, DevicePixels, PaintSurface, Path, Point, PrimitiveBatch,
    ScaledPixels, Scene, Size, size,
};
#[cfg(any(test, feature = "test-support"))]
use anyhow::Result;
use cocoa::foundation::NSSize;
use core_foundation::base::TCFType;
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;
use metal::{CAMetalLayer, MTLPixelFormat};
use objc::{msg_send, sel, sel_impl};
use objc2::ffi::NSUInteger;
use parking_lot::Mutex;
use std::cell::Cell;
use std::sync::Arc;
use std::{mem, ptr};

// objc2-metal MTL4 imports
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4CommandQueueDescriptor, MTL4ComputeCommandEncoder,
    MTL4RenderCommandEncoder, MTL4RenderPassDescriptor, MTL4VisibilityOptions, MTLAllocation,
    MTLBuffer, MTLDevice, MTLGPUFamily, MTLLoadAction, MTLOrigin, MTLPrimitiveType,
    MTLRenderPipelineState, MTLRenderStages, MTLResidencySet, MTLResidencySetDescriptor,
    MTLSharedEvent, MTLSize, MTLStages, MTLStoreAction, MTLTexture, MTLViewport,
};

/// Number of frames in flight for triple buffering.
/// This matches Apple's Metal 4 sample code pattern.
const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Buffer binding indices matching shaders_mtl4.metal
#[repr(u64)]
#[derive(Clone, Copy)]
enum BufferBindingIndex {
    UnitVertices = 0,
    ViewportSize = 1,
    Primitives = 2,
    Transforms = 3,
    AtlasSize = 4,
    TextureSize = 5,
    PathVertices = 6,
}

/// Texture binding indices matching shaders_mtl4.metal
#[repr(u64)]
#[derive(Clone, Copy)]
enum TextureBindingIndex {
    Atlas = 0,
    Backdrop = 1,
    Intermediate = 2,
    SurfaceY = 3,
    SurfaceCbCr = 4,
}

/// Barrier type for inter-pass synchronization in Metal 4.
/// Different render passes require different barrier configurations.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BarrierType {
    /// No barrier needed
    None,
    /// Barrier after blit operation (for backdrop blur)
    AfterBlit,
    /// Barrier after render pass (for path rasterization)
    AfterRender,
}

/// MSAA sample count for path rasterization (4x MSAA is universally supported)
const PATH_SAMPLE_COUNT: u32 = 4;

/// Vertex data for path rasterization shader
#[repr(C)]
#[derive(Clone, Debug)]
struct PathRasterizationVertex {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
}

/// Sprite data for compositing paths from intermediate texture
#[repr(C)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

/// Surface bounds for video rendering
#[repr(C)]
#[derive(Clone, Debug)]
struct SurfaceBounds {
    bounds: Bounds<ScaledPixels>,
    content_mask: ContentMask<ScaledPixels>,
}

/// Align offset to 256-byte boundary (Metal buffer offset alignment requirement)
fn align_offset(offset: &mut usize) {
    const ALIGNMENT: usize = 256;
    *offset = (*offset + ALIGNMENT - 1) & !(ALIGNMENT - 1);
}

/// Metal 4 renderer that uses the new Metal 4 APIs for improved performance.
///
/// Key differences from classic Metal:
/// - Uses MTL4CommandAllocator for explicit command buffer memory management
/// - Uses MTL4ArgumentTable for binding resources instead of per-encoder bindings
/// - Uses MTLResidencySet for explicit GPU memory residency management
/// - Uses MTLSharedEvent for CPU-GPU synchronization with triple buffering
/// - Supports parallel render encoding with suspend/resume
#[allow(dead_code)]
pub struct Metal4Renderer {
    // Core Metal objects
    device: metal::Device,
    layer: metal::MetalLayer,
    presents_with_transaction: bool,

    // MTL4 command submission infrastructure
    command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,

    // Triple buffering: one allocator per frame in flight
    // This prevents CPU-GPU resource contention
    command_allocators: [Retained<ProtocolObject<dyn MTL4CommandAllocator>>; MAX_FRAMES_IN_FLIGHT],

    // CPU-GPU synchronization
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    frame_number: Cell<u64>,

    // Argument table for resource binding
    // All buffers and textures are bound here instead of per-encoder calls
    argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,

    // Residency management - ensures GPU resources are resident during execution
    residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,

    // Pipeline states for each primitive type
    quad_pipeline: metal::RenderPipelineState,
    shadow_pipeline: metal::RenderPipelineState,
    underline_pipeline: metal::RenderPipelineState,
    backdrop_blur_pipeline: metal::RenderPipelineState,
    mono_sprite_pipeline: metal::RenderPipelineState,
    poly_sprite_pipeline: metal::RenderPipelineState,
    path_rasterization_pipeline: metal::RenderPipelineState,
    path_sprite_pipeline: metal::RenderPipelineState,
    surface_pipeline: metal::RenderPipelineState,

    // Shared infrastructure
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,

    // CoreVideo texture cache for video surface rendering
    core_video_texture_cache: CVMetalTextureCache,

    // Unit quad vertices buffer (6 float2 vertices for instanced quad rendering)
    unit_vertices: metal::Buffer,
    viewport_size_buffer: metal::Buffer,
    atlas_size_buffer: metal::Buffer,
    texture_size_buffer: metal::Buffer,

    // Intermediate textures for multi-pass rendering
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    backdrop_texture: Option<metal::Texture>,

    // In-flight instance buffers - keep them alive until GPU is done
    // This prevents multi-window flickering when buffers are reused too early
    in_flight_instance_buffers: [Option<super::metal_renderer::InstanceBuffer>; MAX_FRAMES_IN_FLIGHT],

    // Track the last committed instance buffer to avoid redundant commit() calls
    // commit() is expensive - only call when we have a new buffer
    last_committed_buffer_addr: Cell<u64>,
}

/// Error type for Metal 4 renderer initialization failures.
#[derive(Debug)]
pub enum Metal4RendererError {
    /// Metal 4 is not supported on this device.
    NotSupported,
    /// Failed to create MTL4 command queue.
    CommandQueueCreationFailed,
    /// Failed to create MTL4 command allocator.
    CommandAllocatorCreationFailed,
    /// Failed to create command buffer.
    CommandBufferCreationFailed,
    /// Failed to create shared event.
    SharedEventCreationFailed,
    /// Failed to create argument tables.
    ArgumentTableCreationFailed,
    /// Failed to create residency set.
    ResidencySetCreationFailed,
    /// Failed to create pipeline state.
    PipelineCreationFailed(String),
    /// Failed to load shader library.
    ShaderLibraryFailed(String),
    /// Other initialization error.
    Other(String),
}

impl std::fmt::Display for Metal4RendererError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSupported => write!(f, "Metal 4 is not supported on this device"),
            Self::CommandQueueCreationFailed => write!(f, "Failed to create MTL4 command queue"),
            Self::CommandAllocatorCreationFailed => {
                write!(f, "Failed to create MTL4 command allocator")
            }
            Self::CommandBufferCreationFailed => write!(f, "Failed to create command buffer"),
            Self::SharedEventCreationFailed => write!(f, "Failed to create shared event"),
            Self::ArgumentTableCreationFailed => write!(f, "Failed to create argument table"),
            Self::ResidencySetCreationFailed => write!(f, "Failed to create residency set"),
            Self::PipelineCreationFailed(msg) => write!(f, "Failed to create pipeline: {}", msg),
            Self::ShaderLibraryFailed(msg) => write!(f, "Failed to load shader library: {}", msg),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for Metal4RendererError {}

impl Metal4Renderer {
    /// Attempts to create a new Metal 4 renderer.
    ///
    /// Returns an error if Metal 4 is not supported or initialization fails.
    /// The caller should fall back to classic Metal renderer on error.
    pub fn try_new(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
    ) -> Result<Self, Metal4RendererError> {
        // 1. Get system default device
        let device =
            metal::Device::system_default().ok_or(Metal4RendererError::NotSupported)?;

        // 2. Check Metal 4 support via objc2
        // MTLGPUFamily::Metal4 = 5002
        let objc2_device = Self::get_objc2_device(&device)?;
        if !objc2_device.supportsFamily(MTLGPUFamily::Metal4) {
            return Err(Metal4RendererError::NotSupported);
        }

        // 3. Create MTL4 command queue
        let queue_desc = MTL4CommandQueueDescriptor::new();
        let command_queue = objc2_device
            .newMTL4CommandQueueWithDescriptor_error(&queue_desc)
            .map_err(|_| Metal4RendererError::CommandQueueCreationFailed)?;

        // 4. Create command allocators for triple buffering
        let allocator0 = objc2_device
            .newCommandAllocator()
            .ok_or(Metal4RendererError::CommandAllocatorCreationFailed)?;
        let allocator1 = objc2_device
            .newCommandAllocator()
            .ok_or(Metal4RendererError::CommandAllocatorCreationFailed)?;
        let allocator2 = objc2_device
            .newCommandAllocator()
            .ok_or(Metal4RendererError::CommandAllocatorCreationFailed)?;
        let command_allocators: [Retained<ProtocolObject<dyn MTL4CommandAllocator>>;
            MAX_FRAMES_IN_FLIGHT] = [allocator0, allocator1, allocator2];

        // 5. Create reusable command buffer
        let command_buffer = objc2_device
            .newCommandBuffer()
            .ok_or(Metal4RendererError::CommandBufferCreationFailed)?;

        // 6. Create shared event for CPU-GPU sync
        let shared_event = objc2_device
            .newSharedEvent()
            .ok_or(Metal4RendererError::SharedEventCreationFailed)?;

        // 7. Create argument table descriptor
        let arg_table_desc = MTL4ArgumentTableDescriptor::new();
        arg_table_desc.setMaxBufferBindCount(8); // buffers 0-7
        arg_table_desc.setMaxTextureBindCount(8); // textures 0-7
        arg_table_desc.setMaxSamplerStateBindCount(2); // samplers 0-1
        arg_table_desc.setInitializeBindings(true);

        let argument_table = objc2_device
            .newArgumentTableWithDescriptor_error(&arg_table_desc)
            .map_err(|_| Metal4RendererError::ArgumentTableCreationFailed)?;

        // 8. Create residency set
        let residency_desc = MTLResidencySetDescriptor::new();
        let residency_set = objc2_device
            .newResidencySetWithDescriptor_error(&residency_desc)
            .map_err(|_| Metal4RendererError::ResidencySetCreationFailed)?;

        // 9. Create Metal layer
        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        layer.set_opaque(!transparent);

        // 10. Load Metal 4 shader library and create pipelines
        let library_data = include_bytes!(concat!(env!("OUT_DIR"), "/shaders_mtl4.metallib"));
        let library = device
            .new_library_with_data(library_data)
            .map_err(|e| Metal4RendererError::ShaderLibraryFailed(e.to_string()))?;

        // 11. Create pipeline states
        let quad_pipeline = Self::create_pipeline(
            &device,
            &library,
            "quad_vertex_mtl4",
            "quad_fragment_mtl4",
        )?;
        let shadow_pipeline = Self::create_pipeline(
            &device,
            &library,
            "shadow_vertex_mtl4",
            "shadow_fragment_mtl4",
        )?;
        let underline_pipeline = Self::create_pipeline(
            &device,
            &library,
            "underline_vertex_mtl4",
            "underline_fragment_mtl4",
        )?;
        let backdrop_blur_pipeline = Self::create_pipeline(
            &device,
            &library,
            "backdrop_blur_vertex_mtl4",
            "backdrop_blur_fragment_mtl4",
        )?;
        let mono_sprite_pipeline = Self::create_pipeline(
            &device,
            &library,
            "monochrome_sprite_vertex_mtl4",
            "monochrome_sprite_fragment_mtl4",
        )?;
        let poly_sprite_pipeline = Self::create_pipeline(
            &device,
            &library,
            "polychrome_sprite_vertex_mtl4",
            "polychrome_sprite_fragment_mtl4",
        )?;
        let path_rasterization_pipeline = Self::create_path_rasterization_pipeline(
            &device,
            &library,
            "path_rasterization_vertex_mtl4",
            "path_rasterization_fragment_mtl4",
            PATH_SAMPLE_COUNT,
        )?;
        let path_sprite_pipeline = Self::create_pipeline(
            &device,
            &library,
            "path_sprite_vertex_mtl4",
            "path_sprite_fragment_mtl4",
        )?;
        let surface_pipeline = Self::create_pipeline(
            &device,
            &library,
            "surface_vertex_mtl4",
            "surface_fragment_mtl4",
        )?;

        // 12. Create unit vertices buffer (6 vertices for a unit quad)
        let unit_vertices: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const _,
            std::mem::size_of_val(&unit_vertices) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // 13. Create viewport size buffer
        let viewport_size_buffer = device.new_buffer(
            std::mem::size_of::<[i32; 2]>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // 13b. Create atlas size buffer (for sprite rendering)
        let atlas_size_buffer = device.new_buffer(
            std::mem::size_of::<[i32; 2]>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // 13c. Create texture size buffer (for surface rendering)
        let texture_size_buffer = device.new_buffer(
            std::mem::size_of::<[i32; 2]>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // 14. Create sprite atlas
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone()));

        // 14b. Create CoreVideo texture cache for video surface rendering
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        // 15. Add long-lived resources to residency set
        // This is CRITICAL - without adding buffers to the residency set,
        // the GPU cannot access them even when using argument tables.
        // See Apple Metal 4 sample: Metal4Renderer.m lines 246-252
        residency_set.addAllocation(unit_vertices.as_allocation());
        residency_set.addAllocation(viewport_size_buffer.as_allocation());
        residency_set.addAllocation(atlas_size_buffer.as_allocation());
        residency_set.addAllocation(texture_size_buffer.as_allocation());

        // 16. Commit residency set and add to queue
        residency_set.commit();
        command_queue.addResidencySet(&residency_set);

        // 17. Add the CAMetalLayer's residency set to the command queue
        // This is needed for the drawable textures to be resident
        // See Apple Metal 4 sample: Metal4Renderer.m line 243
        if let Some(layer_residency_set) = get_layer_residency_set(&layer) {
            command_queue.addResidencySet(&layer_residency_set);
        }

        Ok(Self {
            device,
            layer,
            presents_with_transaction: false,
            command_queue,
            command_buffer,
            command_allocators,
            shared_event,
            frame_number: Cell::new(0),
            argument_table,
            residency_set,
            quad_pipeline,
            shadow_pipeline,
            underline_pipeline,
            backdrop_blur_pipeline,
            mono_sprite_pipeline,
            poly_sprite_pipeline,
            path_rasterization_pipeline,
            path_sprite_pipeline,
            surface_pipeline,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            unit_vertices,
            viewport_size_buffer,
            atlas_size_buffer,
            texture_size_buffer,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            backdrop_texture: None,
            in_flight_instance_buffers: [None, None, None],
            last_committed_buffer_addr: Cell::new(0),
        })
    }

    /// Gets the objc2 device wrapper from a metal-rs device.
    fn get_objc2_device(
        device: &metal::Device,
    ) -> Result<&ProtocolObject<dyn MTLDevice>, Metal4RendererError> {
        // SAFETY: metal::Device and objc2_metal::MTLDevice are both wrappers
        // around the same underlying Objective-C protocol object
        unsafe {
            let ptr = device.as_ptr() as *const ProtocolObject<dyn MTLDevice>;
            ptr.as_ref().ok_or(Metal4RendererError::NotSupported)
        }
    }

    /// Creates a render pipeline state for a given vertex/fragment shader pair.
    fn create_pipeline(
        device: &metal::Device,
        library: &metal::Library,
        vertex_fn: &str,
        fragment_fn: &str,
    ) -> Result<metal::RenderPipelineState, Metal4RendererError> {
        let vertex_function = library.get_function(vertex_fn, None).map_err(|e| {
            Metal4RendererError::PipelineCreationFailed(format!(
                "Vertex function '{}': {}",
                vertex_fn, e
            ))
        })?;

        let fragment_function = library.get_function(fragment_fn, None).map_err(|e| {
            Metal4RendererError::PipelineCreationFailed(format!(
                "Fragment function '{}': {}",
                fragment_fn, e
            ))
        })?;

        let descriptor = metal::RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&vertex_function));
        descriptor.set_fragment_function(Some(&fragment_function));

        // Configure color attachment with blend settings matching classic renderer
        let attachment = descriptor.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
        attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
        // SourceAlpha for source RGB (non-premultiplied), One for source Alpha
        attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

        device.new_render_pipeline_state(&descriptor).map_err(|e| {
            Metal4RendererError::PipelineCreationFailed(format!(
                "Pipeline {}/{}: {}",
                vertex_fn, fragment_fn, e
            ))
        })
    }

    /// Creates a render pipeline state for path rasterization with MSAA support.
    /// Path rasterization uses different blend factors than regular primitives.
    fn create_path_rasterization_pipeline(
        device: &metal::Device,
        library: &metal::Library,
        vertex_fn: &str,
        fragment_fn: &str,
        sample_count: u32,
    ) -> Result<metal::RenderPipelineState, Metal4RendererError> {
        let vertex_function = library.get_function(vertex_fn, None).map_err(|e| {
            Metal4RendererError::PipelineCreationFailed(format!(
                "Vertex function '{}': {}",
                vertex_fn, e
            ))
        })?;

        let fragment_function = library.get_function(fragment_fn, None).map_err(|e| {
            Metal4RendererError::PipelineCreationFailed(format!(
                "Fragment function '{}': {}",
                fragment_fn, e
            ))
        })?;

        let descriptor = metal::RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&vertex_function));
        descriptor.set_fragment_function(Some(&fragment_function));

        // Set MSAA sample count for path rasterization (4x MSAA)
        if sample_count > 1 {
            descriptor.set_raster_sample_count(sample_count as u64);
        }

        // Configure color attachment with blend settings for path rasterization
        // Path rasterization uses One/OneMinusSourceAlpha blend factors
        // This allows proper alpha compositing for overlapping path geometry
        let attachment = descriptor.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
        attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
        // Use One for source (premultiplied alpha output from shader)
        attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
        attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

        device.new_render_pipeline_state(&descriptor).map_err(|e| {
            Metal4RendererError::PipelineCreationFailed(format!(
                "Path rasterization pipeline {}/{}: {}",
                vertex_fn, fragment_fn, e
            ))
        })
    }

    /// Returns a reference to the Metal layer.
    pub fn layer(&self) -> &metal::MetalLayerRef {
        &self.layer
    }

    /// Returns a raw pointer to the Metal layer.
    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer.as_ptr()
    }

    /// Returns a reference to the sprite atlas.
    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    /// Sets whether to present with transaction.
    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
    }

    /// Updates the drawable size.
    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let ns_size = NSSize {
            width: size.width.0 as f64,
            height: size.height.0 as f64,
        };
        unsafe {
            let _: () = msg_send![self.layer(), setDrawableSize: ns_size];
        }

        // Update viewport size buffer
        let viewport_data: [i32; 2] = [size.width.0, size.height.0];
        unsafe {
            let ptr = self.viewport_size_buffer.contents() as *mut [i32; 2];
            *ptr = viewport_data;
        }

        // Recreate intermediate textures at new size
        self.recreate_intermediate_textures(size);
    }

    /// Recreates intermediate textures when viewport size changes.
    fn recreate_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0 as u64;
        let height = size.height.0 as u64;

        if width == 0 || height == 0 {
            return;
        }

        // Path intermediate texture (for path rasterization - resolve target)
        let path_desc = metal::TextureDescriptor::new();
        path_desc.set_width(width);
        path_desc.set_height(height);
        path_desc.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        path_desc.set_usage(
            metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
        );
        path_desc.set_storage_mode(metal::MTLStorageMode::Private);
        self.path_intermediate_texture = Some(self.device.new_texture(&path_desc));

        // Path intermediate MSAA texture (for antialiased path rasterization)
        let msaa_desc = metal::TextureDescriptor::new();
        msaa_desc.set_width(width);
        msaa_desc.set_height(height);
        msaa_desc.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        msaa_desc.set_usage(metal::MTLTextureUsage::RenderTarget);
        msaa_desc.set_storage_mode(metal::MTLStorageMode::Private);
        msaa_desc.set_sample_count(PATH_SAMPLE_COUNT as u64);
        msaa_desc.set_texture_type(metal::MTLTextureType::D2Multisample);
        self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_desc));

        // Backdrop texture (for blur effects)
        let backdrop_desc = metal::TextureDescriptor::new();
        backdrop_desc.set_width(width);
        backdrop_desc.set_height(height);
        backdrop_desc.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        backdrop_desc.set_usage(
            metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
        );
        backdrop_desc.set_storage_mode(metal::MTLStorageMode::Private);
        self.backdrop_texture = Some(self.device.new_texture(&backdrop_desc));
    }

    /// Updates transparency setting.
    pub fn update_transparency(&self, transparent: bool) {
        self.layer.set_opaque(!transparent);
    }

    /// Cleanup resources.
    pub fn destroy(&self) {
        // Residency set will be cleaned up when dropped
        // Command queue and allocators will be cleaned up when dropped
    }

    /// Main draw method - renders a scene using Metal 4 APIs.
    ///
    /// This implements the Metal 4 command submission flow:
    /// 1. Wait for GPU to finish with this frame's resources (triple buffering)
    /// 2. Reset the frame's command allocator
    /// 3. Begin command buffer with allocator
    /// 4. Create render encoder and bind argument table
    /// 5. Encode draw calls for each primitive batch
    /// 6. End encoding and command buffer
    /// 7. Commit to queue and present
    /// 8. Signal completion for CPU synchronization
    ///
    /// If the instance buffer is too small, this method will automatically
    /// retry with a larger buffer (up to 256 MB).
    pub fn draw(&mut self, scene: &Scene) {
        // 1. Get drawable
        let Some(drawable) = self.layer.next_drawable() else {
            log::warn!("Metal 4: No drawable available");
            return;
        };

        // Get viewport size from layer
        let layer_size = self.layer.drawable_size();
        let viewport_size: Size<DevicePixels> = Size {
            width: DevicePixels(layer_size.width.ceil() as i32),
            height: DevicePixels(layer_size.height.ceil() as i32),
        };

        // 2. Wait for GPU to finish with this frame's resources (triple buffering)
        let frame_num = self.frame_number.get();
        let frame_index = (frame_num % MAX_FRAMES_IN_FLIGHT as u64) as usize;
        if frame_num >= MAX_FRAMES_IN_FLIGHT as u64 {
            let wait_value = frame_num - MAX_FRAMES_IN_FLIGHT as u64;
            // Wait up to 1 second for GPU to catch up
            let _success = self
                .shared_event
                .waitUntilSignaledValue_timeoutMS(wait_value, 1000);
        }

        // Release the old instance buffer from this frame slot (GPU is now done with it)
        if let Some(old_buffer) = self.in_flight_instance_buffers[frame_index].take() {
            self.instance_buffer_pool.lock().release(old_buffer);
        }

        // Retry loop - if buffer is too small, we'll retry with a larger one
        loop {
            let result = self.draw_scene_inner(scene, &drawable, viewport_size, frame_index);

            match result {
                Ok(instance_buffer) => {
                    // Success - store buffer and return
                    self.in_flight_instance_buffers[frame_index] = Some(instance_buffer);
                    return;
                }
                Err(instance_buffer) => {
                    // Buffer overflow - release the too-small buffer and try with a larger one
                    let mut pool = self.instance_buffer_pool.lock();
                    pool.release(instance_buffer);

                    let buffer_size = pool.buffer_size();
                    if buffer_size >= 256 * 1024 * 1024 {
                        log::error!(
                            "Metal 4: instance buffer size grew too large: {}",
                            buffer_size
                        );
                        return;
                    }

                    pool.reset(buffer_size * 2);
                    log::info!(
                        "Metal 4: increased instance buffer size to {}",
                        pool.buffer_size()
                    );
                    // Continue loop to retry with larger buffer
                }
            }
        }
    }

    /// Inner draw method that returns the instance buffer on success,
    /// or returns the buffer as an error if it was too small.
    fn draw_scene_inner(
        &self,
        scene: &Scene,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
        frame_index: usize,
    ) -> Result<super::metal_renderer::InstanceBuffer, super::metal_renderer::InstanceBuffer> {

        // 3. Reset allocator for this frame
        let allocator = &self.command_allocators[frame_index];
        allocator.reset();

        // 4. Begin command buffer with allocator
        self.command_buffer
            .beginCommandBufferWithAllocator(allocator);

        // Mark residency set for this command buffer
        self.command_buffer.useResidencySet(&self.residency_set);

        // 5. Acquire instance buffer for this frame's data
        let mut instance_buffer = self.instance_buffer_pool.lock().acquire(&self.device);
        let mut instance_offset: usize = 0;

        // 6. Add resources to residency set and commit only if instance buffer changed
        // This avoids expensive commit() calls when using the same buffer
        let buffer_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
        let needs_commit = buffer_addr != self.last_committed_buffer_addr.get();

        if needs_commit {
            self.residency_set
                .addAllocation(instance_buffer.metal_buffer().as_allocation());
            self.last_committed_buffer_addr.set(buffer_addr);
        }

        // Add intermediate textures (reused every frame, addAllocation is idempotent)
        if let Some(tex) = &self.path_intermediate_texture {
            self.residency_set.addAllocation(tex.as_allocation());
        }
        if let Some(tex) = &self.path_intermediate_msaa_texture {
            self.residency_set.addAllocation(tex.as_allocation());
        }
        if let Some(tex) = &self.backdrop_texture {
            self.residency_set.addAllocation(tex.as_allocation());
        }

        // Pre-scan batches to add atlas textures
        let batches: Vec<_> = scene.batches().collect();
        for batch in &batches {
            match batch {
                PrimitiveBatch::MonochromeSprites { texture_id, .. } |
                PrimitiveBatch::PolychromeSprites { texture_id, .. } => {
                    let atlas_texture = self.sprite_atlas.metal_texture(*texture_id);
                    self.residency_set.addAllocation(atlas_texture.as_allocation());
                }
                _ => {}
            }
        }

        // Only commit if we added a new buffer (commit is expensive)
        if needs_commit {
            self.residency_set.commit();
        }

        // Update viewport size buffer
        let viewport_data: [i32; 2] = [viewport_size.width.0, viewport_size.height.0];
        unsafe {
            let ptr = self.viewport_size_buffer.contents() as *mut [i32; 2];
            *ptr = viewport_data;
        }

        // 6. Create MTL4 render pass descriptor and encoder
        let render_pass_desc = MTL4RenderPassDescriptor::new();
        unsafe {
            let color_attachments = render_pass_desc.colorAttachments();
            let color_attachment = color_attachments.objectAtIndexedSubscript(0);
            color_attachment.setTexture(Some(AsObjc2Texture::as_objc2(drawable.texture())));
            color_attachment.setLoadAction(MTLLoadAction::Clear);
            color_attachment.setStoreAction(MTLStoreAction::Store);

            // Set clear color (black with alpha based on layer opacity)
            let alpha = if self.layer.is_opaque() { 1.0 } else { 0.0 };
            color_attachment.setClearColor(objc2_metal::MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha,
            });
        }

        let Some(encoder) = self.command_buffer.renderCommandEncoderWithDescriptor(&render_pass_desc) else {
            log::error!("Metal 4: Failed to create render encoder");
            self.command_buffer.endCommandBuffer();
            // Return Ok because this isn't a buffer overflow issue
            return Ok(instance_buffer);
        };

        // Set viewport (required for rendering to work correctly)
        encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: viewport_size.width.0 as f64,
            height: viewport_size.height.0 as f64,
            znear: 0.0,
            zfar: 1.0,
        });

        // Set argument table ONCE for the entire render pass (Apple sample pattern)
        // Individual draw calls only update bindings, not the table itself
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Vertex | MTLRenderStages::Fragment,
        );

        // Bind common resources to argument table (unit vertices and viewport size)
        // These are shared across all draw calls
        unsafe {
            // Unit vertices at index 0
            self.argument_table.setAddress_atIndex(
                self.unit_vertices.as_objc2().gpuAddress(),
                BufferBindingIndex::UnitVertices as NSUInteger,
            );

            // Viewport size at index 1
            self.argument_table.setAddress_atIndex(
                self.viewport_size_buffer.as_objc2().gpuAddress(),
                BufferBindingIndex::ViewportSize as NSUInteger,
            );
        }

        // 7. Render each batch in the scene
        // We use Option<encoder> because backdrop blur and paths require breaking the render pass
        let mut current_encoder: Option<Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>>> = Some(encoder);

        // Note: batches were already collected during the residency pre-scan above
        for batch in batches {
            // If we don't have an encoder, something went wrong (shouldn't happen)
            let Some(ref encoder) = current_encoder else {
                log::warn!("Metal 4: No encoder available for batch");
                break;
            };

            let success = match batch {
                PrimitiveBatch::Quads(quads, transforms) => {
                    self.draw_quads_mtl4(
                        quads,
                        transforms,
                        &mut instance_buffer,
                        &mut instance_offset,
                        encoder,
                    )
                }
                PrimitiveBatch::Shadows(shadows, transforms) => {
                    self.draw_shadows_mtl4(
                        shadows,
                        transforms,
                        &mut instance_buffer,
                        &mut instance_offset,
                        encoder,
                    )
                }
                PrimitiveBatch::Underlines(underlines, transforms) => {
                    self.draw_underlines_mtl4(
                        underlines,
                        transforms,
                        &mut instance_buffer,
                        &mut instance_offset,
                        encoder,
                    )
                }
                PrimitiveBatch::MonochromeSprites { texture_id, sprites } => {
                    self.draw_monochrome_sprites_mtl4(
                        texture_id,
                        sprites,
                        &mut instance_buffer,
                        &mut instance_offset,
                        encoder,
                    )
                }
                PrimitiveBatch::PolychromeSprites { texture_id, sprites, transforms } => {
                    self.draw_polychrome_sprites_mtl4(
                        texture_id,
                        sprites,
                        transforms,
                        &mut instance_buffer,
                        &mut instance_offset,
                        encoder,
                    )
                }
                PrimitiveBatch::BackdropBlurs(blurs, transforms) => {
                    // Backdrop blur requires:
                    // 1. End current render encoder
                    // 2. Copy drawable to backdrop texture (via compute encoder)
                    // 3. Create new render encoder with LoadAction::Load
                    // 4. Draw blur quads

                    // End current encoder
                    if let Some(enc) = current_encoder.take() {
                        enc.endEncoding();
                    }

                    // Copy drawable to backdrop texture
                    let did_copy = self.copy_drawable_to_backdrop_mtl4(&drawable, viewport_size);

                    // Create new encoder with LoadAction::Load and blit barrier
                    current_encoder = self.create_render_encoder_mtl4(
                        &drawable,
                        viewport_size,
                        MTLLoadAction::Load,
                        BarrierType::AfterBlit,
                    );

                    if did_copy {
                        if let Some(ref enc) = current_encoder {
                            self.draw_backdrop_blurs_mtl4(
                                blurs,
                                transforms,
                                &mut instance_buffer,
                                &mut instance_offset,
                                enc,
                            )
                        } else {
                            false
                        }
                    } else {
                        // Skip blur if copy failed, but continue with other batches
                        true
                    }
                }
                PrimitiveBatch::Paths(paths) => {
                    // Paths require multi-pass rendering:
                    // 1. End current encoder
                    // 2. Render paths to intermediate texture (with MSAA)
                    // 3. Create new encoder with LoadAction::Load
                    // 4. Draw path sprites from intermediate

                    // End current encoder
                    if let Some(enc) = current_encoder.take() {
                        enc.endEncoding();
                    }

                    // Render paths to intermediate texture
                    let did_draw = self.draw_paths_to_intermediate_mtl4(
                        paths,
                        &mut instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                    );

                    // Create new encoder with LoadAction::Load (preserve existing content)
                    // Barrier after render ensures path rasterization completes before reading
                    current_encoder = self.create_render_encoder_mtl4(
                        &drawable,
                        viewport_size,
                        MTLLoadAction::Load,
                        BarrierType::AfterRender,
                    );

                    if did_draw {
                        if let Some(ref enc) = current_encoder {
                            self.draw_paths_from_intermediate_mtl4(
                                paths,
                                &mut instance_buffer,
                                &mut instance_offset,
                                enc,
                            )
                        } else {
                            false
                        }
                    } else {
                        // Intermediate draw failed (likely buffer overflow) - return false to trigger retry
                        false
                    }
                }
                PrimitiveBatch::Surfaces(surfaces) => {
                    // Surfaces render video frames using YCbCr textures
                    if let Some(ref enc) = current_encoder {
                        self.draw_surfaces_mtl4(
                            surfaces,
                            &mut instance_buffer,
                            &mut instance_offset,
                            viewport_size,
                            enc,
                        )
                    } else {
                        false
                    }
                }
                PrimitiveBatch::SubpixelSprites { .. } => {
                    // Not used on macOS
                    true
                }
            };

            if !success {
                log::warn!("Metal 4: Failed to render batch, instance buffer may be full - retrying with larger buffer");

                // End encoding (if we still have an encoder)
                if let Some(encoder) = current_encoder {
                    encoder.endEncoding();
                }

                // End command buffer without committing - this closes the command buffer
                // so the allocator can be reset for the retry
                self.command_buffer.endCommandBuffer();

                // Return error to trigger retry with larger buffer
                return Err(instance_buffer);
            }
        }

        // 8. End encoding (if we still have an encoder)
        if let Some(encoder) = current_encoder {
            encoder.endEncoding();
        }

        // 9. End command buffer
        self.command_buffer.endCommandBuffer();

        // 10. Submit to GPU
        self.command_queue.waitForDrawable(AsObjc2Drawable::as_objc2(drawable));

        // Commit the command buffer
        unsafe {
            use std::ptr::NonNull;
            // Get the raw pointer to the command buffer object (not the Retained wrapper)
            let cmd_buf_raw = Retained::as_ptr(&self.command_buffer) as *mut _;
            // Create a NonNull from the raw pointer
            let mut cmd_buf_nn = NonNull::new_unchecked(cmd_buf_raw);
            // commit_count expects NonNull<NonNull<...>>, i.e. pointer to array of pointers
            self.command_queue
                .commit_count(NonNull::new_unchecked(&mut cmd_buf_nn), 1);
        }

        self.command_queue.signalDrawable(AsObjc2Drawable::as_objc2(drawable));

        // 11. Present
        drawable.present();

        // 12. Signal completion for CPU synchronization
        let new_frame_number = self.frame_number.get() + 1;
        self.frame_number.set(new_frame_number);
        self.command_queue
            .signalEvent_value(self.shared_event.as_event_ref(), new_frame_number);

        // Return success with the instance buffer
        Ok(instance_buffer)
    }

    // ========================================================================
    // Metal 4 Draw Methods for Each Primitive Type
    // ========================================================================

    fn draw_quads_mtl4(
        &self,
        quads: &[crate::Quad],
        transforms: &[crate::TransformationMatrix],
        instance_buffer: &mut super::metal_renderer::InstanceBuffer,
        instance_offset: &mut usize,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) -> bool {
        if quads.is_empty() {
            return true;
        }

        align_offset(instance_offset);
        let quads_offset = *instance_offset;

        // Calculate sizes
        let quads_bytes = mem::size_of_val(quads);
        let mut transforms_offset = quads_offset + quads_bytes;
        align_offset(&mut transforms_offset);
        let transforms_bytes = mem::size_of_val(transforms);
        let next_offset = transforms_offset + transforms_bytes;

        if next_offset > instance_buffer.size() {
            return false;
        }

        // Copy data to instance buffer
        unsafe {
            let buffer_ptr = instance_buffer.metal_buffer().contents() as *mut u8;
            ptr::copy_nonoverlapping(
                quads.as_ptr() as *const u8,
                buffer_ptr.add(quads_offset),
                quads_bytes,
            );
            ptr::copy_nonoverlapping(
                transforms.as_ptr() as *const u8,
                buffer_ptr.add(transforms_offset),
                transforms_bytes,
            );
        }

        // Set pipeline (argument table is already bound for the render pass)
        encoder.setRenderPipelineState(self.quad_pipeline.as_objc2());

        // Bind batch-specific resources to argument table
        let buffer_gpu_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
        unsafe {
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + quads_offset as u64,
                BufferBindingIndex::Primitives as NSUInteger,
            );
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + transforms_offset as u64,
                BufferBindingIndex::Transforms as NSUInteger,
            );
        }

        // Draw
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                quads.len() as NSUInteger,
            );
        }

        *instance_offset = next_offset;
        true
    }

    fn draw_shadows_mtl4(
        &self,
        shadows: &[crate::Shadow],
        transforms: &[crate::TransformationMatrix],
        instance_buffer: &mut super::metal_renderer::InstanceBuffer,
        instance_offset: &mut usize,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) -> bool {
        if shadows.is_empty() {
            return true;
        }

        align_offset(instance_offset);
        let shadows_offset = *instance_offset;

        let shadows_bytes = mem::size_of_val(shadows);
        let mut transforms_offset = shadows_offset + shadows_bytes;
        align_offset(&mut transforms_offset);
        let transforms_bytes = mem::size_of_val(transforms);
        let next_offset = transforms_offset + transforms_bytes;

        if next_offset > instance_buffer.size() {
            return false;
        }

        unsafe {
            let buffer_ptr = instance_buffer.metal_buffer().contents() as *mut u8;
            ptr::copy_nonoverlapping(
                shadows.as_ptr() as *const u8,
                buffer_ptr.add(shadows_offset),
                shadows_bytes,
            );
            ptr::copy_nonoverlapping(
                transforms.as_ptr() as *const u8,
                buffer_ptr.add(transforms_offset),
                transforms_bytes,
            );
        }

        // Set pipeline (argument table is already bound for the render pass)
        encoder.setRenderPipelineState(self.shadow_pipeline.as_objc2());

        // Bind resources
        let buffer_gpu_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
        unsafe {
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + shadows_offset as u64,
                BufferBindingIndex::Primitives as NSUInteger,
            );
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + transforms_offset as u64,
                BufferBindingIndex::Transforms as NSUInteger,
            );
        }

        // Draw
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                shadows.len() as NSUInteger,
            );
        }

        *instance_offset = next_offset;
        true
    }

    fn draw_underlines_mtl4(
        &self,
        underlines: &[crate::Underline],
        transforms: &[crate::TransformationMatrix],
        instance_buffer: &mut super::metal_renderer::InstanceBuffer,
        instance_offset: &mut usize,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) -> bool {
        if underlines.is_empty() {
            return true;
        }

        align_offset(instance_offset);
        let underlines_offset = *instance_offset;

        let underlines_bytes = mem::size_of_val(underlines);
        let mut transforms_offset = underlines_offset + underlines_bytes;
        align_offset(&mut transforms_offset);
        let transforms_bytes = mem::size_of_val(transforms);
        let next_offset = transforms_offset + transforms_bytes;

        if next_offset > instance_buffer.size() {
            return false;
        }

        unsafe {
            let buffer_ptr = instance_buffer.metal_buffer().contents() as *mut u8;
            ptr::copy_nonoverlapping(
                underlines.as_ptr() as *const u8,
                buffer_ptr.add(underlines_offset),
                underlines_bytes,
            );
            ptr::copy_nonoverlapping(
                transforms.as_ptr() as *const u8,
                buffer_ptr.add(transforms_offset),
                transforms_bytes,
            );
        }

        // Set pipeline (argument table is already bound for the render pass)
        encoder.setRenderPipelineState(self.underline_pipeline.as_objc2());

        // Bind resources
        let buffer_gpu_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
        unsafe {
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + underlines_offset as u64,
                BufferBindingIndex::Primitives as NSUInteger,
            );
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + transforms_offset as u64,
                BufferBindingIndex::Transforms as NSUInteger,
            );
        }

        // Draw
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                underlines.len() as NSUInteger,
            );
        }

        *instance_offset = next_offset;
        true
    }

    fn draw_monochrome_sprites_mtl4(
        &self,
        texture_id: crate::AtlasTextureId,
        sprites: &[crate::MonochromeSprite],
        instance_buffer: &mut super::metal_renderer::InstanceBuffer,
        instance_offset: &mut usize,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }

        align_offset(instance_offset);
        let sprites_offset = *instance_offset;
        let sprites_bytes = mem::size_of_val(sprites);
        let next_offset = sprites_offset + sprites_bytes;

        if next_offset > instance_buffer.size() {
            return false;
        }

        unsafe {
            let buffer_ptr = instance_buffer.metal_buffer().contents() as *mut u8;
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_ptr.add(sprites_offset),
                sprites_bytes,
            );
        }

        // Get atlas texture and its size
        // Note: Texture was already added to residency set during batch pre-scan
        let atlas_texture: metal::Texture = self.sprite_atlas.metal_texture(texture_id);
        let atlas_size: [i32; 2] = [
            atlas_texture.width() as i32,
            atlas_texture.height() as i32,
        ];

        // Update atlas size buffer
        unsafe {
            let ptr = self.atlas_size_buffer.contents() as *mut [i32; 2];
            *ptr = atlas_size;
        }

        // Set pipeline (argument table is already bound for the render pass)
        encoder.setRenderPipelineState(self.mono_sprite_pipeline.as_objc2());

        // Bind sprite data and atlas size
        let buffer_gpu_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
        unsafe {
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + sprites_offset as u64,
                BufferBindingIndex::Primitives as NSUInteger,
            );

            // Bind atlas size buffer
            self.argument_table.setAddress_atIndex(
                self.atlas_size_buffer.as_objc2().gpuAddress(),
                BufferBindingIndex::AtlasSize as NSUInteger,
            );

            // Bind atlas texture
            self.argument_table.setTexture_atIndex(
                atlas_texture.as_objc2().gpuResourceID(),
                TextureBindingIndex::Atlas as NSUInteger,
            );
        }

        // Draw
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len() as NSUInteger,
            );
        }

        *instance_offset = next_offset;
        true
    }

    fn draw_polychrome_sprites_mtl4(
        &self,
        texture_id: crate::AtlasTextureId,
        sprites: &[crate::PolychromeSprite],
        transforms: &[crate::TransformationMatrix],
        instance_buffer: &mut super::metal_renderer::InstanceBuffer,
        instance_offset: &mut usize,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }

        align_offset(instance_offset);
        let sprites_offset = *instance_offset;

        let sprites_bytes = mem::size_of_val(sprites);
        let mut transforms_offset = sprites_offset + sprites_bytes;
        align_offset(&mut transforms_offset);
        let transforms_bytes = mem::size_of_val(transforms);
        let next_offset = transforms_offset + transforms_bytes;

        if next_offset > instance_buffer.size() {
            return false;
        }

        unsafe {
            let buffer_ptr = instance_buffer.metal_buffer().contents() as *mut u8;
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_ptr.add(sprites_offset),
                sprites_bytes,
            );
            ptr::copy_nonoverlapping(
                transforms.as_ptr() as *const u8,
                buffer_ptr.add(transforms_offset),
                transforms_bytes,
            );
        }

        // Get atlas texture and its size
        // Note: Texture was already added to residency set during batch pre-scan
        let atlas_texture: metal::Texture = self.sprite_atlas.metal_texture(texture_id);
        let atlas_size: [i32; 2] = [
            atlas_texture.width() as i32,
            atlas_texture.height() as i32,
        ];

        // Update atlas size buffer
        unsafe {
            let ptr = self.atlas_size_buffer.contents() as *mut [i32; 2];
            *ptr = atlas_size;
        }

        // Set pipeline (argument table is already bound for the render pass)
        encoder.setRenderPipelineState(self.poly_sprite_pipeline.as_objc2());

        // Bind resources
        let buffer_gpu_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
        unsafe {
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + sprites_offset as u64,
                BufferBindingIndex::Primitives as NSUInteger,
            );
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + transforms_offset as u64,
                BufferBindingIndex::Transforms as NSUInteger,
            );
            // Bind atlas size buffer
            self.argument_table.setAddress_atIndex(
                self.atlas_size_buffer.as_objc2().gpuAddress(),
                BufferBindingIndex::AtlasSize as NSUInteger,
            );
            self.argument_table.setTexture_atIndex(
                atlas_texture.as_objc2().gpuResourceID(),
                TextureBindingIndex::Atlas as NSUInteger,
            );
        }

        // Draw
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len() as NSUInteger,
            );
        }

        *instance_offset = next_offset;
        true
    }

    /// Renders a scene to an image (for testing).
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, _scene: &Scene) -> Result<RgbaImage> {
        // TODO: Implement for testing
        anyhow::bail!("Metal 4 render_to_image not implemented yet")
    }

    // ========================================================================
    // Multi-Pass Rendering Helpers (Backdrop Blur, Paths)
    // ========================================================================

    /// Creates a new MTL4 render encoder with specified load action.
    /// Used when we need to break the render pass (e.g., for backdrop blur, paths).
    fn create_render_encoder_mtl4(
        &self,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
        load_action: MTLLoadAction,
        barrier_type: BarrierType,
    ) -> Option<Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>>> {
        let render_pass_desc = MTL4RenderPassDescriptor::new();
        unsafe {
            let color_attachments = render_pass_desc.colorAttachments();
            let color_attachment = color_attachments.objectAtIndexedSubscript(0);
            color_attachment.setTexture(Some(AsObjc2Texture::as_objc2(drawable.texture())));
            color_attachment.setLoadAction(load_action);
            color_attachment.setStoreAction(MTLStoreAction::Store);

            // Only set clear color if we're clearing
            if load_action == MTLLoadAction::Clear {
                let alpha = if self.layer.is_opaque() { 1.0 } else { 0.0 };
                color_attachment.setClearColor(objc2_metal::MTLClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha,
                });
            }
        }

        let encoder = self.command_buffer.renderCommandEncoderWithDescriptor(&render_pass_desc)?;

        // Set viewport
        encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: viewport_size.width.0 as f64,
            height: viewport_size.height.0 as f64,
            znear: 0.0,
            zfar: 1.0,
        });

        // Set argument table for this encoder
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Vertex | MTLRenderStages::Fragment,
        );

        // Add consumer barrier based on barrier type
        // This ensures the previous pass completes before we start rendering
        match barrier_type {
            BarrierType::None => {}
            BarrierType::AfterBlit => {
                // Wait for blit operation to complete before fragment stage reads
                encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
                    MTLStages::Blit,
                    MTLStages::Fragment,
                    MTL4VisibilityOptions::Device,
                );
            }
            BarrierType::AfterRender => {
                // Wait for previous render pass (path rasterization) to complete
                // before fragment stage reads from the intermediate texture
                encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
                    MTLStages::Fragment,
                    MTLStages::Fragment,
                    MTL4VisibilityOptions::Device,
                );
            }
        }

        // Bind common resources
        unsafe {
            self.argument_table.setAddress_atIndex(
                self.unit_vertices.as_objc2().gpuAddress(),
                BufferBindingIndex::UnitVertices as NSUInteger,
            );
            self.argument_table.setAddress_atIndex(
                self.viewport_size_buffer.as_objc2().gpuAddress(),
                BufferBindingIndex::ViewportSize as NSUInteger,
            );
        }

        Some(encoder)
    }

    /// Copies the drawable texture to the backdrop texture using a compute encoder.
    /// Metal 4 uses compute encoder for blit operations.
    fn copy_drawable_to_backdrop_mtl4(
        &self,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
    ) -> bool {
        let Some(backdrop_texture) = &self.backdrop_texture else {
            return false;
        };

        if viewport_size.width.0 <= 0 || viewport_size.height.0 <= 0 {
            return false;
        }

        // Create compute encoder for blit operation
        let Some(compute_encoder) = self.command_buffer.computeCommandEncoder() else {
            return false;
        };

        // Copy drawable to backdrop texture
        let origin = MTLOrigin { x: 0, y: 0, z: 0 };
        let size = MTLSize {
            width: viewport_size.width.0 as NSUInteger,
            height: viewport_size.height.0 as NSUInteger,
            depth: 1,
        };

        unsafe {
            compute_encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                AsObjc2Texture::as_objc2(drawable.texture()),
                0,
                0,
                origin,
                size,
                backdrop_texture.as_objc2(),
                0,
                0,
                origin,
            );
        }

        // Add producer barrier: ensure blit completes before subsequent render passes read
        compute_encoder.barrierAfterStages_beforeQueueStages_visibilityOptions(
            MTLStages::Blit,
            MTLStages::Fragment,
            MTL4VisibilityOptions::Device,
        );

        compute_encoder.endEncoding();
        true
    }

    /// Draws backdrop blur quads.
    fn draw_backdrop_blurs_mtl4(
        &self,
        blurs: &[crate::BackdropBlur],
        transforms: &[crate::TransformationMatrix],
        instance_buffer: &mut super::metal_renderer::InstanceBuffer,
        instance_offset: &mut usize,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) -> bool {
        if blurs.is_empty() {
            return true;
        }

        let Some(backdrop_texture) = &self.backdrop_texture else {
            return false;
        };

        align_offset(instance_offset);
        let blurs_offset = *instance_offset;

        let blurs_bytes = mem::size_of_val(blurs);
        let mut transforms_offset = blurs_offset + blurs_bytes;
        align_offset(&mut transforms_offset);
        let transforms_bytes = mem::size_of_val(transforms);
        let next_offset = transforms_offset + transforms_bytes;

        if next_offset > instance_buffer.size() {
            return false;
        }

        // Copy blur and transform data to instance buffer
        unsafe {
            let buffer_ptr = instance_buffer.metal_buffer().contents() as *mut u8;
            ptr::copy_nonoverlapping(
                blurs.as_ptr() as *const u8,
                buffer_ptr.add(blurs_offset),
                blurs_bytes,
            );
            ptr::copy_nonoverlapping(
                transforms.as_ptr() as *const u8,
                buffer_ptr.add(transforms_offset),
                transforms_bytes,
            );
        }

        // Note: Backdrop texture was already added to residency set during batch pre-scan

        // Set backdrop blur pipeline
        encoder.setRenderPipelineState(self.backdrop_blur_pipeline.as_objc2());

        // Bind resources
        let buffer_gpu_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
        unsafe {
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + blurs_offset as u64,
                BufferBindingIndex::Primitives as NSUInteger,
            );
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + transforms_offset as u64,
                BufferBindingIndex::Transforms as NSUInteger,
            );
            // Bind backdrop texture
            self.argument_table.setTexture_atIndex(
                backdrop_texture.as_objc2().gpuResourceID(),
                TextureBindingIndex::Backdrop as NSUInteger,
            );
        }

        // Draw
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                blurs.len() as NSUInteger,
            );
        }

        *instance_offset = next_offset;
        true
    }

    /// Renders paths to the intermediate texture using MSAA.
    /// This is the first pass of path rendering - rasterizing the path geometry.
    fn draw_paths_to_intermediate_mtl4(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut super::metal_renderer::InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }

        let Some(intermediate_texture) = &self.path_intermediate_texture else {
            return false;
        };

        // Build vertex data from paths
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

        if vertices.is_empty() {
            return true;
        }

        let vertices_bytes = mem::size_of_val(vertices.as_slice());
        let next_offset = *instance_offset + vertices_bytes;

        if next_offset > instance_buffer.size() {
            return false;
        }

        // Copy vertex data to instance buffer
        unsafe {
            let buffer_ptr = instance_buffer.metal_buffer().contents() as *mut u8;
            ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                buffer_ptr.add(*instance_offset),
                vertices_bytes,
            );
        }

        // Create render pass descriptor for intermediate texture
        let render_pass_desc = MTL4RenderPassDescriptor::new();
        unsafe {
            let color_attachments = render_pass_desc.colorAttachments();
            let color_attachment = color_attachments.objectAtIndexedSubscript(0);
            color_attachment.setLoadAction(MTLLoadAction::Clear);
            color_attachment.setClearColor(objc2_metal::MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });

            // Use MSAA if available
            if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
                color_attachment.setTexture(Some(msaa_texture.as_objc2()));
                color_attachment.setResolveTexture(Some(intermediate_texture.as_objc2()));
                color_attachment.setStoreAction(MTLStoreAction::MultisampleResolve);
            } else {
                color_attachment.setTexture(Some(intermediate_texture.as_objc2()));
                color_attachment.setStoreAction(MTLStoreAction::Store);
            }
        }

        let Some(encoder) = self.command_buffer.renderCommandEncoderWithDescriptor(&render_pass_desc) else {
            return false;
        };

        // Set viewport for intermediate texture
        encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: viewport_size.width.0 as f64,
            height: viewport_size.height.0 as f64,
            znear: 0.0,
            zfar: 1.0,
        });

        // Set argument table
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Vertex | MTLRenderStages::Fragment,
        );

        // Note: Intermediate textures were already added to residency set during batch pre-scan

        // Set path rasterization pipeline
        encoder.setRenderPipelineState(self.path_rasterization_pipeline.as_objc2());

        // Bind vertex data
        // Note: The shader expects BufferBindingIndexAtlasSize for the intermediate texture size,
        // which is the same as viewport size. The naming is historical from the classic renderer.
        let buffer_gpu_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
        unsafe {
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + *instance_offset as u64,
                BufferBindingIndex::PathVertices as NSUInteger,
            );
            // Path rasterization uses AtlasSize (not ViewportSize) for the intermediate texture dimensions
            self.argument_table.setAddress_atIndex(
                self.viewport_size_buffer.as_objc2().gpuAddress(),
                BufferBindingIndex::AtlasSize as NSUInteger,
            );
        }

        // Draw all path vertices
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                vertices.len() as NSUInteger,
            );
        }

        // Add producer barrier: ensure path rasterization writes are visible to subsequent
        // encoders that will read from the intermediate texture. This is critical in Metal 4
        // because resources are untracked and require explicit synchronization.
        encoder.barrierAfterStages_beforeQueueStages_visibilityOptions(
            MTLStages::Fragment, // Path rasterization writes via fragment shader
            MTLStages::Fragment, // Path sprite reads in subsequent encoder
            MTL4VisibilityOptions::Device,
        );

        encoder.endEncoding();

        *instance_offset = next_offset;
        true
    }

    /// Draws paths from the intermediate texture to the drawable.
    /// This is the second pass - compositing the rasterized paths.
    fn draw_paths_from_intermediate_mtl4(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut super::metal_renderer::InstanceBuffer,
        instance_offset: &mut usize,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) -> bool {
        let Some(first_path) = paths.first() else {
            return true;
        };

        let Some(intermediate_texture) = &self.path_intermediate_texture else {
            return false;
        };

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites: Vec<PathSprite>;
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
        let sprites_bytes = mem::size_of_val(sprites.as_slice());
        let next_offset = *instance_offset + sprites_bytes;

        if next_offset > instance_buffer.size() {
            return false;
        }

        // Copy sprite data to instance buffer
        unsafe {
            let buffer_ptr = instance_buffer.metal_buffer().contents() as *mut u8;
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_ptr.add(*instance_offset),
                sprites_bytes,
            );
        }

        // Set path sprite pipeline
        encoder.setRenderPipelineState(self.path_sprite_pipeline.as_objc2());

        // Bind resources
        let buffer_gpu_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
        unsafe {
            self.argument_table.setAddress_atIndex(
                buffer_gpu_addr + *instance_offset as u64,
                BufferBindingIndex::Primitives as NSUInteger,
            );
            // Bind intermediate texture
            self.argument_table.setTexture_atIndex(
                intermediate_texture.as_objc2().gpuResourceID(),
                TextureBindingIndex::Intermediate as NSUInteger,
            );
        }

        // Draw
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len() as NSUInteger,
            );
        }

        *instance_offset = next_offset;
        true
    }

    /// Draws video surfaces using YCbCr textures.
    fn draw_surfaces_mtl4(
        &self,
        surfaces: &[PaintSurface],
        instance_buffer: &mut super::metal_renderer::InstanceBuffer,
        instance_offset: &mut usize,
        _viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) -> bool {
        // Set surface pipeline
        encoder.setRenderPipelineState(self.surface_pipeline.as_objc2());

        for surface in surfaces {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );

            // Verify expected pixel format
            assert_eq!(
                surface.image_buffer.get_pixel_format(),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            // Create Y texture (luminance)
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

            // Create CbCr texture (chrominance)
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
            let next_offset = *instance_offset + mem::size_of::<SurfaceBounds>();

            if next_offset > instance_buffer.size() {
                return false;
            }

            // Write surface bounds to instance buffer
            unsafe {
                let buffer_ptr = instance_buffer.metal_buffer().contents() as *mut u8;
                let surface_bounds_ptr = buffer_ptr.add(*instance_offset) as *mut SurfaceBounds;
                ptr::write(
                    surface_bounds_ptr,
                    SurfaceBounds {
                        bounds: surface.bounds,
                        content_mask: surface.content_mask.clone(),
                    },
                );
            }

            // Update texture size buffer
            let texture_size_data: [i32; 2] = [texture_size.width.0, texture_size.height.0];
            unsafe {
                let ptr = self.texture_size_buffer.contents() as *mut [i32; 2];
                *ptr = texture_size_data;
            }

            // Bind resources
            let buffer_gpu_addr = instance_buffer.metal_buffer().as_objc2().gpuAddress();
            unsafe {
                self.argument_table.setAddress_atIndex(
                    buffer_gpu_addr + *instance_offset as u64,
                    BufferBindingIndex::Primitives as NSUInteger,
                );
                self.argument_table.setAddress_atIndex(
                    self.texture_size_buffer.as_objc2().gpuAddress(),
                    BufferBindingIndex::TextureSize as NSUInteger,
                );

                // Bind Y and CbCr textures
                let y_metal_texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                let cbcr_metal_texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());

                if !y_metal_texture.is_null() && !cbcr_metal_texture.is_null() {
                    let y_tex_ref = metal::TextureRef::from_ptr(y_metal_texture as *mut _);
                    let cbcr_tex_ref = metal::TextureRef::from_ptr(cbcr_metal_texture as *mut _);

                    self.argument_table.setTexture_atIndex(
                        y_tex_ref.as_objc2().gpuResourceID(),
                        TextureBindingIndex::SurfaceY as NSUInteger,
                    );
                    self.argument_table.setTexture_atIndex(
                        cbcr_tex_ref.as_objc2().gpuResourceID(),
                        TextureBindingIndex::SurfaceCbCr as NSUInteger,
                    );
                }
            }

            // Draw this surface (each surface is a single quad)
            unsafe {
                encoder.drawPrimitives_vertexStart_vertexCount(
                    MTLPrimitiveType::Triangle,
                    0,
                    6,
                );
            }

            *instance_offset = next_offset;
        }

        true
    }
}

// Additional trait implementations to help with objc2 interop
trait AsObjc2Drawable {
    fn as_objc2(&self) -> &ProtocolObject<dyn objc2_metal::MTLDrawable>;
}

impl AsObjc2Drawable for metal::MetalDrawableRef {
    fn as_objc2(&self) -> &ProtocolObject<dyn objc2_metal::MTLDrawable> {
        // SAFETY: metal::MetalDrawableRef wraps the same Objective-C object
        unsafe {
            let ptr = self.as_ptr() as *const ProtocolObject<dyn objc2_metal::MTLDrawable>;
            &*ptr
        }
    }
}

trait AsObjc2Event {
    fn as_event_ref(&self) -> &ProtocolObject<dyn objc2_metal::MTLEvent>;
}

impl AsObjc2Event for Retained<ProtocolObject<dyn MTLSharedEvent>> {
    fn as_event_ref(&self) -> &ProtocolObject<dyn objc2_metal::MTLEvent> {
        // SAFETY: MTLSharedEvent conforms to MTLEvent
        unsafe {
            let ptr: *const ProtocolObject<dyn MTLSharedEvent> = &**self;
            &*(ptr as *const ProtocolObject<dyn objc2_metal::MTLEvent>)
        }
    }
}

/// Helper trait to convert metal-rs Buffer to objc2-metal MTLBuffer reference
trait AsObjc2Buffer {
    fn as_objc2(&self) -> &ProtocolObject<dyn MTLBuffer>;
}

impl AsObjc2Buffer for metal::Buffer {
    fn as_objc2(&self) -> &ProtocolObject<dyn MTLBuffer> {
        // SAFETY: metal::Buffer wraps the same Objective-C object as objc2_metal::MTLBuffer
        unsafe {
            let ptr = self.as_ptr() as *const ProtocolObject<dyn MTLBuffer>;
            &*ptr
        }
    }
}

/// Helper trait to convert metal-rs Texture to objc2-metal MTLTexture reference
trait AsObjc2Texture {
    fn as_objc2(&self) -> &ProtocolObject<dyn MTLTexture>;
}

impl AsObjc2Texture for metal::Texture {
    fn as_objc2(&self) -> &ProtocolObject<dyn MTLTexture> {
        // SAFETY: metal::Texture wraps the same Objective-C object as objc2_metal::MTLTexture
        unsafe {
            let ptr = self.as_ptr() as *const ProtocolObject<dyn MTLTexture>;
            &*ptr
        }
    }
}

impl AsObjc2Texture for metal::TextureRef {
    fn as_objc2(&self) -> &ProtocolObject<dyn MTLTexture> {
        // SAFETY: metal::TextureRef wraps the same Objective-C object as objc2_metal::MTLTexture
        unsafe {
            let ptr = self.as_ptr() as *const ProtocolObject<dyn MTLTexture>;
            &*ptr
        }
    }
}

/// Helper trait to convert metal-rs RenderPipelineState to objc2-metal MTLRenderPipelineState reference
trait AsObjc2RenderPipelineState {
    fn as_objc2(&self) -> &ProtocolObject<dyn MTLRenderPipelineState>;
}

impl AsObjc2RenderPipelineState for metal::RenderPipelineState {
    fn as_objc2(&self) -> &ProtocolObject<dyn MTLRenderPipelineState> {
        // SAFETY: metal::RenderPipelineState wraps the same Objective-C object
        unsafe {
            let ptr = self.as_ptr() as *const ProtocolObject<dyn MTLRenderPipelineState>;
            &*ptr
        }
    }
}

/// Helper trait to convert metal-rs Buffer to objc2-metal MTLAllocation reference
/// (needed for residency set management)
trait AsObjc2Allocation {
    fn as_allocation(&self) -> &ProtocolObject<dyn MTLAllocation>;
}

impl AsObjc2Allocation for metal::Buffer {
    fn as_allocation(&self) -> &ProtocolObject<dyn MTLAllocation> {
        // SAFETY: MTLBuffer conforms to MTLResource which conforms to MTLAllocation
        unsafe {
            let ptr = self.as_ptr() as *const ProtocolObject<dyn MTLAllocation>;
            &*ptr
        }
    }
}

impl AsObjc2Allocation for metal::Texture {
    fn as_allocation(&self) -> &ProtocolObject<dyn MTLAllocation> {
        // SAFETY: MTLTexture conforms to MTLResource which conforms to MTLAllocation
        unsafe {
            let ptr = self.as_ptr() as *const ProtocolObject<dyn MTLAllocation>;
            &*ptr
        }
    }
}

/// Helper function to get the residency set from a CAMetalLayer
/// This is needed because the metal-rs crate doesn't expose this API directly.
fn get_layer_residency_set(layer: &metal::MetalLayerRef) -> Option<Retained<ProtocolObject<dyn MTLResidencySet>>> {
    // SAFETY: CAMetalLayer has a residencySet property that returns an MTLResidencySet
    // We use objc messaging to access it since metal-rs doesn't expose this API
    unsafe {
        // Get the objc object ID type to cast to for msg_send
        let layer_ptr = layer.as_ptr() as *mut objc::runtime::Object;
        let residency_set: *mut objc::runtime::Object = msg_send![layer_ptr, residencySet];
        if residency_set.is_null() {
            return None;
        }
        // Retain the object since we're getting a property value
        let _: () = msg_send![residency_set, retain];
        Some(Retained::from_raw(residency_set as *mut ProtocolObject<dyn MTLResidencySet>).unwrap())
    }
}
