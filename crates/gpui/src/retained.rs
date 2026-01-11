//! Retained element tree for O(changed) frame work.
//!
//! This module provides persistent element storage that survives across frames.
//! Instead of rebuilding the entire element tree every frame via `render()`,
//! unchanged views reuse their stored element trees directly.
//!
//! ## Architecture
//!
//! The retained element system works at the VIEW level (not individual elements):
//!
//! 1. **First render**: `render()` is called, element tree is built and stored
//! 2. **Subsequent frames (unchanged)**: Stored element tree is reused without calling `render()`
//! 3. **After `cx.notify()`**: View is marked dirty, `render()` is called, new tree replaces old
//!
//! This achieves O(changed) work per frame because unchanged views skip `render()` entirely.

use crate::{AnyElement, AvailableSpace, Bounds, LayoutId, Pixels, Size};
use std::any::TypeId;

/// Retained state for a view, including its cached element tree.
///
/// This is stored persistently in `Window.retained_views` and survives across frames.
/// When a view is clean (not in `dirty_views`), we reuse the stored element tree
/// instead of calling `render()`.
pub struct RetainedView {
    /// The cached element tree from the last render.
    /// This is the actual element that was returned by `render()`.
    pub element: Option<AnyElement>,

    /// The layout ID for the root of this view's element tree.
    pub layout_id: Option<LayoutId>,

    /// The bounds computed during the last layout.
    pub bounds: Option<Bounds<Pixels>>,

    /// The size available during the last layout.
    pub available_size: Option<Size<AvailableSpace>>,

    /// Type ID of the view, for debugging.
    pub type_id: TypeId,

    /// Whether prepaint has been done for the current frame.
    pub prepaint_done: bool,

    /// Whether paint has been done for the current frame.
    pub paint_done: bool,
}

impl RetainedView {
    /// Create a new retained view for an entity of the given type.
    pub fn new<V: 'static>() -> Self {
        Self {
            element: None,
            layout_id: None,
            bounds: None,
            available_size: None,
            type_id: TypeId::of::<V>(),
            prepaint_done: false,
            paint_done: false,
        }
    }

    /// Store a new element tree from render().
    pub fn store_element(&mut self, element: AnyElement) {
        self.element = Some(element);
        self.prepaint_done = false;
        self.paint_done = false;
    }

    /// Mark the view as having completed prepaint for this frame.
    pub fn mark_prepaint_done(&mut self) {
        self.prepaint_done = true;
    }

    /// Mark the view as having completed paint for this frame.
    pub fn mark_paint_done(&mut self) {
        self.paint_done = true;
    }

    /// Reset frame-specific state at the start of a new frame.
    pub fn begin_frame(&mut self) {
        self.prepaint_done = false;
        self.paint_done = false;
    }

    /// Check if this view has a cached element that can be reused.
    pub fn has_cached_element(&self) -> bool {
        self.element.is_some()
    }

    /// Take the element for layout, leaving the slot empty.
    /// The element must be returned via `return_element` after layout.
    pub fn take_element(&mut self) -> Option<AnyElement> {
        self.element.take()
    }

    /// Return the element after layout.
    pub fn return_element(&mut self, element: AnyElement) {
        self.element = Some(element);
    }
}

impl std::fmt::Debug for RetainedView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedView")
            .field("has_element", &self.element.is_some())
            .field("layout_id", &self.layout_id)
            .field("bounds", &self.bounds)
            .field("prepaint_done", &self.prepaint_done)
            .field("paint_done", &self.paint_done)
            .finish()
    }
}
