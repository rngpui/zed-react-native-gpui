//! Compositor State for Mutable Properties
//!
//! This module provides mutable state owned by the compositor that can be updated
//! without main-thread involvement. Scroll offsets, transforms, and opacity can be
//! changed freely here, and the compositor will apply them during presentation.
//!
//! ## Key Insight
//!
//! In Chromium's cc architecture, certain properties are "compositor-owned":
//! - Scroll offset
//! - Transform (for animations)
//! - Opacity (for animations)
//!
//! Changing these properties NEVER triggers main-thread invalidation. The compositor
//! simply applies the new values during the next composite frame.

use crate::layer::LayerId;
use crate::scene::TransformationMatrix;
use crate::{Pixels, Point};
use collections::FxHashMap;

/// Mutable state owned by compositor, not main thread.
///
/// The compositor can update these properties freely without triggering
/// main-thread work. Changes here only require a composite frame, not a full draw.
#[derive(Clone, Default)]
pub struct CompositorState {
    /// Per-layer scroll offsets (compositor can change freely).
    scroll_offsets: FxHashMap<LayerId, Point<Pixels>>,

    /// Per-layer transforms (for animations).
    transforms: FxHashMap<LayerId, TransformationMatrix>,

    /// Per-layer opacity (for animations).
    opacities: FxHashMap<LayerId, f32>,

    /// Current mouse position (for hit testing with compositor-owned scroll).
    mouse_position: Point<Pixels>,

    /// Flag indicating compositor state has pending changes.
    has_pending_changes: bool,

    /// Generation counter for tracking updates.
    generation: u64,
}

impl CompositorState {
    /// Create a new empty compositor state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update scroll offset for a layer - NO main thread involvement.
    ///
    /// This is the key API for compositor-driven scrolling. Changing scroll offset
    /// here does NOT invalidate the view or trigger a redraw.
    pub fn set_scroll_offset(&mut self, layer: LayerId, offset: Point<Pixels>) {
        let changed = self.scroll_offsets.get(&layer) != Some(&offset);
        if changed {
            self.scroll_offsets.insert(layer, offset);
            self.has_pending_changes = true;
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Get scroll offset for a layer.
    pub fn scroll_offset(&self, layer: LayerId) -> Point<Pixels> {
        self.scroll_offsets.get(&layer).copied().unwrap_or_default()
    }

    /// Check if a scroll offset exists for a layer.
    pub fn has_scroll_offset(&self, layer: LayerId) -> bool {
        self.scroll_offsets.contains_key(&layer)
    }

    /// Update transform for a layer - NO main thread involvement.
    pub fn set_transform(&mut self, layer: LayerId, transform: TransformationMatrix) {
        let changed = self.transforms.get(&layer) != Some(&transform);
        if changed {
            self.transforms.insert(layer, transform);
            self.has_pending_changes = true;
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Get transform for a layer.
    pub fn transform(&self, layer: LayerId) -> TransformationMatrix {
        self.transforms
            .get(&layer)
            .copied()
            .unwrap_or_else(TransformationMatrix::unit)
    }

    /// Update opacity for a layer - NO main thread involvement.
    pub fn set_opacity(&mut self, layer: LayerId, opacity: f32) {
        let clamped = opacity.clamp(0.0, 1.0);
        let changed = self.opacities.get(&layer) != Some(&clamped);
        if changed {
            self.opacities.insert(layer, clamped);
            self.has_pending_changes = true;
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Get opacity for a layer.
    pub fn opacity(&self, layer: LayerId) -> f32 {
        self.opacities.get(&layer).copied().unwrap_or(1.0)
    }

    /// Update mouse position for hit testing.
    pub fn set_mouse_position(&mut self, position: Point<Pixels>) {
        self.mouse_position = position;
    }

    /// Get current mouse position.
    pub fn mouse_position(&self) -> Point<Pixels> {
        self.mouse_position
    }

    /// Check if there are pending changes that need compositing.
    pub fn has_pending_changes(&self) -> bool {
        self.has_pending_changes
    }

    /// Clear the pending changes flag after compositing.
    pub fn clear_pending_changes(&mut self) {
        self.has_pending_changes = false;
    }

    /// Get the current generation counter.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Remove state for a layer (when layer is removed).
    pub fn remove_layer(&mut self, layer: LayerId) {
        self.scroll_offsets.remove(&layer);
        self.transforms.remove(&layer);
        self.opacities.remove(&layer);
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.scroll_offsets.clear();
        self.transforms.clear();
        self.opacities.clear();
        self.has_pending_changes = false;
    }

    /// Get all layers with scroll offsets.
    pub fn scroll_layers(&self) -> impl Iterator<Item = (&LayerId, &Point<Pixels>)> {
        self.scroll_offsets.iter()
    }

    /// Get all layers with transforms.
    pub fn transform_layers(&self) -> impl Iterator<Item = (&LayerId, &TransformationMatrix)> {
        self.transforms.iter()
    }

    /// Get all layers with opacity.
    pub fn opacity_layers(&self) -> impl Iterator<Item = (&LayerId, &f32)> {
        self.opacities.iter()
    }

    /// Compute effective content origin for a layer given its viewport origin.
    ///
    /// content_origin = viewport_origin + scroll_offset
    ///
    /// This is where content (0,0) appears on screen.
    pub fn content_origin(&self, layer: LayerId, viewport_origin: Point<Pixels>) -> Point<Pixels> {
        let scroll = self.scroll_offset(layer);
        Point {
            x: viewport_origin.x + scroll.x,
            y: viewport_origin.y + scroll.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::px;

    #[test]
    fn test_compositor_state_creation() {
        let state = CompositorState::new();
        assert!(!state.has_pending_changes());
        assert_eq!(state.generation(), 0);
    }

    #[test]
    fn test_scroll_offset() {
        let mut state = CompositorState::new();
        let layer = LayerId(1);

        assert_eq!(state.scroll_offset(layer), Point::default());

        state.set_scroll_offset(layer, Point { x: px(10.0), y: px(20.0) });
        assert!(state.has_pending_changes());
        assert_eq!(state.scroll_offset(layer), Point { x: px(10.0), y: px(20.0) });
        assert_eq!(state.generation(), 1);
    }

    #[test]
    fn test_opacity() {
        let mut state = CompositorState::new();
        let layer = LayerId(1);

        assert_eq!(state.opacity(layer), 1.0);

        state.set_opacity(layer, 0.5);
        assert_eq!(state.opacity(layer), 0.5);

        // Test clamping
        state.set_opacity(layer, 1.5);
        assert_eq!(state.opacity(layer), 1.0);

        state.set_opacity(layer, -0.5);
        assert_eq!(state.opacity(layer), 0.0);
    }

    #[test]
    fn test_transform() {
        let mut state = CompositorState::new();
        let layer = LayerId(1);

        assert!(state.transform(layer).is_unit());

        let transform = TransformationMatrix::unit().translate(Point { x: px(100.0), y: px(200.0) });
        state.set_transform(layer, transform);
        assert_eq!(state.transform(layer), transform);
    }

    #[test]
    fn test_content_origin() {
        let mut state = CompositorState::new();
        let layer = LayerId(1);
        let viewport_origin = Point { x: px(50.0), y: px(100.0) };

        // No scroll
        assert_eq!(
            state.content_origin(layer, viewport_origin),
            Point { x: px(50.0), y: px(100.0) }
        );

        // With scroll (negative scroll = content moved up/left)
        state.set_scroll_offset(layer, Point { x: px(-20.0), y: px(-30.0) });
        assert_eq!(
            state.content_origin(layer, viewport_origin),
            Point { x: px(30.0), y: px(70.0) }
        );
    }

    #[test]
    fn test_remove_layer() {
        let mut state = CompositorState::new();
        let layer = LayerId(1);

        state.set_scroll_offset(layer, Point { x: px(10.0), y: px(20.0) });
        state.set_opacity(layer, 0.5);

        state.remove_layer(layer);

        assert_eq!(state.scroll_offset(layer), Point::default());
        assert_eq!(state.opacity(layer), 1.0);
    }

    #[test]
    fn test_clear_pending_changes() {
        let mut state = CompositorState::new();
        let layer = LayerId(1);

        state.set_scroll_offset(layer, Point { x: px(10.0), y: px(20.0) });
        assert!(state.has_pending_changes());

        state.clear_pending_changes();
        assert!(!state.has_pending_changes());
    }
}
