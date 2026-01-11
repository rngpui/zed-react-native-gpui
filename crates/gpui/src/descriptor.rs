use crate::{ElementId, EntityId, SharedString, StyleRefinement, TextStyleRefinement};
use collections::FxHasher;
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// A lightweight descriptor for an element.
/// Descriptors are created each frame by `render()` and are cheap to create.
/// They contain only the data needed for layout and change detection.
pub enum AnyDescriptor {
    /// A div container element
    Div(Box<DivDescriptor>),
    /// A text element
    Text(TextDescriptor),
    /// A deferred element - defers drawing its child
    Deferred(Box<DeferredDescriptor>),
    /// A view element - delegates rendering to an entity
    View(ViewDescriptor),
    /// An empty placeholder element
    Empty,
}

impl Clone for AnyDescriptor {
    fn clone(&self) -> Self {
        match self {
            AnyDescriptor::Div(desc) => AnyDescriptor::Div(desc.clone()),
            AnyDescriptor::Text(desc) => AnyDescriptor::Text(desc.clone()),
            AnyDescriptor::Deferred(desc) => AnyDescriptor::Deferred(desc.clone()),
            AnyDescriptor::View(desc) => AnyDescriptor::View(desc.clone()),
            AnyDescriptor::Empty => AnyDescriptor::Empty,
        }
    }
}

impl AnyDescriptor {
    /// Get the children of this descriptor, if any
    pub fn children(&self) -> &[AnyDescriptor] {
        match self {
            AnyDescriptor::Div(desc) => &desc.children,
            AnyDescriptor::Text(_) => &[],
            AnyDescriptor::Deferred(desc) => &desc.children,
            AnyDescriptor::View(_) => &[],
            AnyDescriptor::Empty => &[],
        }
    }

    /// Compute a content hash for this descriptor for change detection.
    /// This hash is based on the actual content of the descriptor, allowing
    /// reconciliation to detect when a descriptor has truly changed.
    pub fn content_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        std::mem::discriminant(self).hash(&mut hasher);
        match self {
            AnyDescriptor::Div(desc) => desc.content_hash_into(&mut hasher),
            AnyDescriptor::Text(desc) => desc.content_hash_into(&mut hasher),
            AnyDescriptor::Deferred(desc) => desc.content_hash_into(&mut hasher),
            AnyDescriptor::View(desc) => desc.content_hash_into(&mut hasher),
            AnyDescriptor::Empty => {}
        }
        hasher.finish()
    }

    /// Legacy method - use content_hash() instead
    #[deprecated(note = "Use content_hash() instead")]
    pub fn descriptor_hash(&self) -> u64 {
        self.content_hash()
    }

    /// Get the number of children
    pub fn child_count(&self) -> usize {
        match self {
            AnyDescriptor::Div(desc) => desc.children.len(),
            AnyDescriptor::Deferred(desc) => desc.children.len(),
            _ => 0,
        }
    }
}

/// Descriptor for a div container element
#[derive(Clone)]
pub struct DivDescriptor {
    /// The base style of this div
    pub style: StyleRefinement,
    /// Optional element ID for stateful interactivity
    pub element_id: Option<ElementId>,
    /// Children of this div
    pub children: SmallVec<[AnyDescriptor; 2]>,
}

impl DivDescriptor {
    /// Create a new DivDescriptor
    pub fn new(style: StyleRefinement, element_id: Option<ElementId>) -> Self {
        Self {
            style,
            element_id,
            children: SmallVec::new(),
        }
    }

    /// Add a child to this descriptor
    pub fn with_child(mut self, child: AnyDescriptor) -> Self {
        self.children.push(child);
        self
    }

    /// Add multiple children to this descriptor
    pub fn with_children(mut self, children: impl IntoIterator<Item = AnyDescriptor>) -> Self {
        self.children.extend(children);
        self
    }

    /// Compute a content hash for change detection.
    /// Hashes the element_id, style, and child count.
    pub fn content_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.content_hash_into(&mut hasher);
        hasher.finish()
    }

    fn content_hash_into<H: Hasher>(&self, state: &mut H) {
        self.element_id.hash(state);
        self.children.len().hash(state);
        self.style.layout_hash().hash(state);
    }
}

/// Descriptor for a text element
#[derive(Clone)]
pub struct TextDescriptor {
    /// The text content
    pub text: SharedString,
    /// Text styling
    pub style: TextStyleRefinement,
}

/// Descriptor for a deferred element
#[derive(Clone)]
pub struct DeferredDescriptor {
    /// Child of the deferred element
    pub children: SmallVec<[AnyDescriptor; 1]>,
    /// Priority for deferred drawing
    pub priority: usize,
}

impl DeferredDescriptor {
    /// Create a new DeferredDescriptor
    pub fn new(child: AnyDescriptor, priority: usize) -> Self {
        Self {
            children: SmallVec::from_iter([child]),
            priority,
        }
    }

    /// Compute a content hash for change detection.
    pub fn content_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.content_hash_into(&mut hasher);
        hasher.finish()
    }

    fn content_hash_into<H: Hasher>(&self, state: &mut H) {
        self.priority.hash(state);
        self.children.len().hash(state);
    }
}

impl TextDescriptor {
    /// Create a new TextDescriptor
    pub fn new(text: impl Into<SharedString>) -> Self {
        Self {
            text: text.into(),
            style: TextStyleRefinement::default(),
        }
    }

    /// Set the text style
    pub fn with_style(mut self, style: TextStyleRefinement) -> Self {
        self.style = style;
        self
    }

    /// Compute a content hash for change detection.
    /// Hashes the text content and style.
    pub fn content_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.content_hash_into(&mut hasher);
        hasher.finish()
    }

    fn content_hash_into<H: Hasher>(&self, state: &mut H) {
        self.text.hash(state);
        self.style.layout_hash().hash(state);
    }
}

/// Descriptor for a view element
#[derive(Clone)]
pub struct ViewDescriptor {
    /// The entity ID of the view
    pub entity_id: EntityId,
}

impl ViewDescriptor {
    /// Create a new ViewDescriptor
    pub fn new(entity_id: EntityId) -> Self {
        Self { entity_id }
    }

    /// Compute a content hash for change detection.
    /// Views are identified by their entity_id - the view's content
    /// is managed separately through the entity's dirty tracking.
    pub fn content_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.content_hash_into(&mut hasher);
        hasher.finish()
    }

    fn content_hash_into<H: Hasher>(&self, state: &mut H) {
        self.entity_id.hash(state);
    }
}

/// Trait for types that can be converted into a descriptor
pub trait IntoDescriptor {
    /// Convert this type into an AnyDescriptor
    fn into_descriptor(self) -> AnyDescriptor;
}

impl IntoDescriptor for AnyDescriptor {
    fn into_descriptor(self) -> AnyDescriptor {
        self
    }
}

impl IntoDescriptor for DivDescriptor {
    fn into_descriptor(self) -> AnyDescriptor {
        AnyDescriptor::Div(Box::new(self))
    }
}

impl IntoDescriptor for TextDescriptor {
    fn into_descriptor(self) -> AnyDescriptor {
        AnyDescriptor::Text(self)
    }
}

impl IntoDescriptor for ViewDescriptor {
    fn into_descriptor(self) -> AnyDescriptor {
        AnyDescriptor::View(self)
    }
}

impl IntoDescriptor for DeferredDescriptor {
    fn into_descriptor(self) -> AnyDescriptor {
        AnyDescriptor::Deferred(Box::new(self))
    }
}

impl IntoDescriptor for () {
    fn into_descriptor(self) -> AnyDescriptor {
        AnyDescriptor::Empty
    }
}

impl IntoDescriptor for SharedString {
    fn into_descriptor(self) -> AnyDescriptor {
        AnyDescriptor::Text(TextDescriptor::new(self))
    }
}

impl IntoDescriptor for &'static str {
    fn into_descriptor(self) -> AnyDescriptor {
        AnyDescriptor::Text(TextDescriptor::new(self))
    }
}

impl IntoDescriptor for String {
    fn into_descriptor(self) -> AnyDescriptor {
        AnyDescriptor::Text(TextDescriptor::new(self))
    }
}

impl<T: IntoDescriptor> IntoDescriptor for Option<T> {
    fn into_descriptor(self) -> AnyDescriptor {
        match self {
            Some(inner) => inner.into_descriptor(),
            None => AnyDescriptor::Empty,
        }
    }
}
