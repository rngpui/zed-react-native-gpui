//! Runtime selection between Metal 4 and classic Metal renderers.
//!
//! This module provides a unified interface that automatically selects
//! the appropriate Metal renderer based on hardware capabilities.

use super::metal4_renderer::Metal4Renderer;
use super::metal_atlas::MetalAtlas;
use super::metal_renderer::{InstanceBufferPool, MetalRenderer};
use crate::{DevicePixels, Scene, Size};
#[cfg(any(test, feature = "test-support"))]
use anyhow::Result;
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;
use metal::CAMetalLayer;
use objc2_metal::{MTLDevice, MTLGPUFamily};
use parking_lot::Mutex;
use std::{ffi::c_void, sync::Arc};

// Re-export types that window.rs expects
pub type Context = Arc<Mutex<InstanceBufferPool>>;

/// Enum dispatch for Metal renderer backends.
///
/// Uses enum dispatch rather than trait objects to avoid vtable overhead
/// in this performance-critical code path.
pub enum MetalRendererBackend {
    Classic(MetalRenderer),
    Metal4(Metal4Renderer),
}

impl MetalRendererBackend {
    /// Creates a new renderer, automatically selecting Metal 4 if available.
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        if should_use_metal4() {
            match Metal4Renderer::try_new(instance_buffer_pool.clone(), transparent) {
                Ok(renderer) => {
                    log::info!("Using Metal 4 renderer");
                    return Self::Metal4(renderer);
                }
                Err(e) => {
                    log::warn!("Metal 4 initialization failed, falling back to classic: {e}");
                }
            }
        }

        log::info!("Using classic Metal renderer");
        Self::Classic(MetalRenderer::new(instance_buffer_pool, transparent))
    }

    pub fn layer(&self) -> &metal::MetalLayerRef {
        match self {
            Self::Classic(r) => r.layer(),
            Self::Metal4(r) => r.layer(),
        }
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        match self {
            Self::Classic(r) => r.layer_ptr(),
            Self::Metal4(r) => r.layer_ptr(),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        match self {
            Self::Classic(r) => r.sprite_atlas(),
            Self::Metal4(r) => r.sprite_atlas(),
        }
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        match self {
            Self::Classic(r) => r.set_presents_with_transaction(presents_with_transaction),
            Self::Metal4(r) => r.set_presents_with_transaction(presents_with_transaction),
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        match self {
            Self::Classic(r) => r.update_drawable_size(size),
            Self::Metal4(r) => r.update_drawable_size(size),
        }
    }

    pub fn update_transparency(&self, transparent: bool) {
        match self {
            Self::Classic(r) => r.update_transparency(transparent),
            Self::Metal4(r) => r.update_transparency(transparent),
        }
    }

    pub fn destroy(&self) {
        match self {
            Self::Classic(r) => r.destroy(),
            Self::Metal4(r) => r.destroy(),
        }
    }

    pub fn draw(&mut self, scene: &Scene) {
        match self {
            Self::Classic(r) => r.draw(scene),
            Self::Metal4(r) => r.draw(scene),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        match self {
            Self::Classic(r) => r.render_to_image(scene),
            Self::Metal4(r) => r.render_to_image(scene),
        }
    }
}

pub type Renderer = MetalRendererBackend;

/// Creates a new renderer with runtime backend selection.
///
/// # Safety
/// The native_window and native_view pointers must be valid.
pub unsafe fn new_renderer(
    context: Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: crate::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRendererBackend::new(context, transparent)
}

/// Checks if Metal 4 should be used based on hardware and environment.
fn should_use_metal4() -> bool {
    // Check for environment variable override to force classic Metal
    if std::env::var("GPUI_FORCE_CLASSIC_METAL").is_ok() {
        log::info!("GPUI_FORCE_CLASSIC_METAL set, using classic Metal renderer");
        return false;
    }

    // Check for environment variable to force Metal 4 (skip hardware check)
    if std::env::var("GPUI_FORCE_METAL4").is_ok() {
        log::info!("GPUI_FORCE_METAL4 set, attempting Metal 4 renderer");
        return true;
    }

    // Check for Metal 4 hardware support using objc2-metal
    let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
        return false;
    };

    // Use objc2-metal's supportsFamily method with MTLGPUFamily::Metal4
    device.supportsFamily(MTLGPUFamily::Metal4)
}
