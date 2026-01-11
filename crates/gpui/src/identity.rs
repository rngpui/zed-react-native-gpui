//! Identity types for element reconciliation.
//!
//! This module contains types used for identifying elements across frames
//! during fiber reconciliation. These are deliberately separate from the
//! fiber implementation to avoid coupling element definitions to fiber internals.

use crate::{ElementId, EntityId};

/// Key for identifying elements across frames during reconciliation.
///
/// During reconciliation, the fiber tree uses keys to match elements from
/// the previous frame with elements from the current frame. Matching elements
/// can reuse cached layout and paint data, enabling incremental rendering.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VKey {
    /// Keyed by explicit element ID. Elements with the same ElementId are
    /// considered the same element and will be matched during reconciliation.
    Element(ElementId),
    /// Keyed by position in parent's child list. Used when no explicit key
    /// is provided - elements are matched by their index in the children array.
    Positional(u32),
    /// Keyed by view entity ID. Used for view roots to ensure stable identity.
    View(EntityId),
    /// No key (anonymous). The element will still participate in reconciliation
    /// but may not be matched as precisely.
    None,
}
