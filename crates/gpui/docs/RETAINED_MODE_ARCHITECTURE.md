# Retained-Mode GPUI: Design & Architecture

## Executive Summary

This document proposes a fundamental architectural change to GPUI: transitioning from an **immediate-mode** rendering model (rebuild everything every frame) to a **retained-mode** model (persist the element tree, only update what changed).

**The Problem**: GPUI currently does O(n) work per frame regardless of what changed, resulting in ~30fps in debug mode for a static 375-element grid on an M2 Ultra.

**The Solution**: Retain the element tree across frames, track changes through Rust's ownership model, and only re-render/re-layout/re-paint elements that actually changed.

**The Goal**: O(changed) work per frame, enabling 120fps in debug mode for typical UI workloads.

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Design Goals](#2-design-goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Core Data Structures](#4-core-data-structures)
5. [The Frame Cycle](#5-the-frame-cycle)
6. [Automatic Reactivity](#6-automatic-reactivity)
7. [Reconciliation](#7-reconciliation)
8. [Incremental Layout](#8-incremental-layout)
9. [Incremental Paint](#9-incremental-paint)
10. [Integration with Existing Systems](#10-integration-with-existing-systems)
11. [Developer Experience](#11-developer-experience)
12. [Migration Strategy](#12-migration-strategy)
13. [Performance Expectations](#13-performance-expectations)
14. [Open Questions](#14-open-questions)

---

## 1. Problem Analysis

### 1.1 Current Architecture

GPUI currently uses an immediate-mode-inspired architecture where the element tree is rebuilt from scratch every frame:

```
Every frame (regardless of what changed):
┌─────────────────────────────────────────────────────────────┐
│  View::render()     → Rebuild entire element tree           │
│  request_layout()   → Touch every Taffy node                │
│  compute_layout()   → Walk entire layout tree               │
│  prepaint()         → Visit every element                   │
│  paint()            → Visit every element                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The Performance Impact

Using `cache_bench` as a concrete example:
- **Grid**: 375 cells (15×25), completely static
- **Stats overlay**: ~30 elements, only text values change
- **Total**: ~405 elements

**Current behavior**:
```
Every frame:
├── render() rebuilds 375 grid divs (unchanged!)
├── render() rebuilds 30 stat elements (only values change)
├── request_layout() × 405 elements
├── compute_layout() walks all 405 nodes
├── prepaint() × 405 elements
├── paint() × 405 elements
└── Result: ~30fps in debug mode on M2 Ultra
```

**Expected behavior**:
```
Frame where only stats change:
├── Check grid dirty → NO, skip 375 elements
├── Check stats dirty → YES, update ~10 text values
├── Layout only stats subtree
├── Paint only changed text
└── Result: 120fps in debug mode
```

### 1.3 Comparison with React Native

React Native achieves 120fps in debug mode (even with Hermes interpreter) because:

1. **Retained native views**: Views persist across frames
2. **Reconciliation**: Only changed components re-render
3. **Minimal work per frame**: O(changed) not O(total)

GPUI, despite being compiled Rust with GPU acceleration, is slower because it does O(n) work unconditionally.

### 1.4 Root Cause

The architectural issue is not "debug mode is slow" but rather:

> **GPUI does O(n) work every frame when O(changed) would suffice.**

Debug mode amplifies this by adding a constant multiplier to all operations. But the fundamental problem is the linear work, not the constant factor.

---

## 2. Design Goals

### 2.1 Primary Goals

1. **O(changed) frame complexity**: Work done should be proportional to what changed, not total elements
2. **120fps in debug mode**: For typical UI workloads with few changes per frame
3. **Zero-annotation reactivity**: Developers should not need to manually specify dependencies
4. **Backward compatible**: Existing GPUI code should work without modification (possibly with degraded performance initially)

### 2.2 Secondary Goals

1. **Minimal memory overhead**: Retained tree should not significantly increase memory usage
2. **Predictable performance**: No surprise O(n²) behaviors or GC pauses
3. **Debuggability**: Clear tools to understand why re-renders happen
4. **Incremental adoption**: Can be enabled per-view or globally

### 2.3 Non-Goals

1. **Immediate-mode purity**: We're explicitly moving away from immediate mode
2. **Perfect automatic optimization**: Some edge cases may need hints
3. **Breaking API changes**: Core Element/View traits should remain similar

---

## 3. Architecture Overview

### 3.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RETAINED ELEMENT TREE                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  RetainedNode                                                │   │
│  │  ├── element: Box<dyn Element>                              │   │
│  │  ├── children: Vec<RetainedNodeId>                          │   │
│  │  ├── cached_state: RetainedState (layout, paint cache)      │   │
│  │  └── flags: DirtyFlags (NEEDS_RENDER | NEEDS_LAYOUT | ...)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Frame Cycle                                                 │   │
│  │  1. Check dirty flags (O(1) for clean subtrees)             │   │
│  │  2. Re-render dirty views only                               │   │
│  │  3. Reconcile changed subtrees                               │   │
│  │  4. Layout dirty subtrees only                               │   │
│  │  5. Paint dirty elements only                                │   │
│  │  6. Composite (unchanged from current)                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Concepts

| Concept | Description |
|---------|-------------|
| **Retained Tree** | Element tree persists across frames |
| **Dirty Flags** | Track what needs re-render/layout/paint |
| **Automatic Tracking** | Rust's ownership model enables dependency detection |
| **Reconciliation** | Diff new elements against retained tree |
| **Incremental Layout** | Skip layout for clean subtrees |
| **Incremental Paint** | Skip paint for clean elements |

### 3.3 Comparison with Current Architecture

| Aspect | Current (Immediate) | Proposed (Retained) |
|--------|---------------------|---------------------|
| Element tree lifetime | One frame | Persists across frames |
| Work per frame | O(total elements) | O(changed elements) |
| Render calls | Every view, every frame | Only dirty views |
| Layout calls | Every element, every frame | Only dirty subtrees |
| Paint calls | Every element, every frame | Only dirty elements |
| Memory | Lower (temporary) | Higher (persistent) |
| Complexity | Simpler | More complex |

---

## 4. Core Data Structures

### 4.1 RetainedNodeId

```rust
/// Unique identifier for a retained element node.
/// Stable across frames for the lifetime of the element.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RetainedNodeId(u64);

impl RetainedNodeId {
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}
```

### 4.2 DirtyFlags

```rust
bitflags! {
    /// Flags indicating what work is needed for a retained node.
    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    pub struct DirtyFlags: u8 {
        /// View inputs changed, need to call render()
        const NEEDS_RENDER = 0b0000_0001;

        /// Layout inputs changed, need to recompute layout
        const NEEDS_LAYOUT = 0b0000_0010;

        /// Visual appearance changed, need to repaint
        const NEEDS_PAINT = 0b0000_0100;

        /// Some descendant has dirty flags set.
        /// Enables O(1) skip of clean subtrees.
        const SUBTREE_DIRTY = 0b0000_1000;

        /// Element was just created, needs full initialization
        const NEWLY_CREATED = 0b0001_0000;

        /// All flags set (for initialization)
        const ALL = 0b0001_1111;
    }
}
```

### 4.3 RetainedState

```rust
/// Cached state for a retained element node.
#[derive(Debug)]
pub struct RetainedState {
    // === Layout Cache ===
    /// Taffy layout node ID
    pub layout_id: Option<LayoutId>,
    /// Computed bounds from last layout
    pub bounds: Bounds<Pixels>,
    /// Constraints used for last layout (for cache validity)
    pub layout_constraints: Option<Size<AvailableSpace>>,

    // === Paint Cache ===
    /// Range of items in the DisplayList belonging to this element
    pub display_items: Range<usize>,
    /// Content mask at paint time
    pub content_mask: ContentMask<Pixels>,

    // === Render Cache (for Views) ===
    /// Hash of inputs that produced the current children
    pub render_hash: u64,
    /// Cached element output from render() (before reconciliation)
    pub rendered_element: Option<AnyElement>,

    // === Dependency Tracking ===
    /// Entity handles read during last render
    pub entity_dependencies: SmallVec<[EntityId; 4]>,
}

impl Default for RetainedState {
    fn default() -> Self {
        Self {
            layout_id: None,
            bounds: Bounds::default(),
            layout_constraints: None,
            display_items: 0..0,
            content_mask: ContentMask::default(),
            render_hash: 0,
            rendered_element: None,
            entity_dependencies: SmallVec::new(),
        }
    }
}
```

### 4.4 RetainedNode

```rust
/// A node in the retained element tree.
pub struct RetainedNode {
    /// Unique identifier for this node
    pub id: RetainedNodeId,

    /// Key for reconciliation (user-provided or auto-generated)
    pub key: ElementKey,

    /// Type ID for same-type checking during reconciliation
    pub type_id: TypeId,

    /// The actual element (boxed for heterogeneous storage)
    pub element: Option<AnyElement>,

    /// For view nodes: the view's entity handle
    pub view_entity: Option<AnyEntityHandle>,

    // === Tree Structure ===
    pub parent: Option<RetainedNodeId>,
    pub children: SmallVec<[RetainedNodeId; 4]>,
    /// Index in parent's children array (for efficient removal)
    pub index_in_parent: usize,

    // === State ===
    pub state: RetainedState,
    pub flags: DirtyFlags,
}
```

### 4.5 ElementKey

```rust
/// Key for identifying elements during reconciliation.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub enum ElementKey {
    /// Auto-generated from position in parent
    Index(usize),
    /// User-provided stable key
    Named(SharedString),
    /// Derived from data (e.g., database ID)
    Data(u64),
}

impl Default for ElementKey {
    fn default() -> Self {
        ElementKey::Index(0)
    }
}
```

### 4.6 RetainedTree

```rust
/// The retained element tree for a window.
pub struct RetainedTree {
    /// All nodes in the tree, indexed by ID
    nodes: SlotMap<RetainedNodeId, RetainedNode>,

    /// Root node of the tree
    root: Option<RetainedNodeId>,

    /// Fast lookup: element path → node ID
    path_to_node: FxHashMap<ElementPath, RetainedNodeId>,

    /// Fast lookup: entity ID → nodes that depend on it
    entity_to_dependents: FxHashMap<EntityId, SmallVec<[RetainedNodeId; 4]>>,

    /// Nodes that need processing this frame
    dirty_roots: Vec<RetainedNodeId>,

    /// Statistics for debugging/profiling
    stats: RetainedTreeStats,
}

#[derive(Default, Debug)]
pub struct RetainedTreeStats {
    pub total_nodes: usize,
    pub nodes_rendered: usize,
    pub nodes_laid_out: usize,
    pub nodes_painted: usize,
    pub nodes_skipped: usize,
}
```

---

## 5. The Frame Cycle

### 5.1 Overview

```rust
impl Window {
    /// Main draw entry point - replaces current immediate-mode draw
    pub fn draw(&mut self, cx: &mut App) {
        // Reset per-frame stats
        self.retained_tree.stats = RetainedTreeStats::default();

        // Phase 1: Process dirty renders
        // Only visits views marked NEEDS_RENDER
        self.process_dirty_renders(cx);

        // Phase 2: Incremental layout
        // Skips entire subtrees without NEEDS_LAYOUT or SUBTREE_DIRTY
        self.layout_dirty_subtrees(cx);

        // Phase 3: Incremental paint
        // Skips elements without NEEDS_PAINT, reuses cached display items
        self.paint_dirty_elements(cx);

        // Phase 4: Composite (unchanged from current)
        // Uses existing compositor/tile system
        self.composite(cx);

        // Clear dirty flags for next frame
        self.retained_tree.clear_processed_flags();
    }
}
```

### 5.2 Phase 1: Process Dirty Renders

```rust
impl Window {
    fn process_dirty_renders(&mut self, cx: &mut App) {
        // Process dirty roots in order (parents before children)
        // This ensures parent render happens before child reconciliation
        let dirty_roots = std::mem::take(&mut self.retained_tree.dirty_roots);

        for node_id in dirty_roots {
            let node = match self.retained_tree.nodes.get(node_id) {
                Some(n) => n,
                None => continue, // Node was removed
            };

            if !node.flags.contains(DirtyFlags::NEEDS_RENDER) {
                continue; // Already processed or cleaned
            }

            if node.view_entity.is_some() {
                self.render_view_node(node_id, cx);
            }
        }
    }

    fn render_view_node(&mut self, node_id: RetainedNodeId, cx: &mut App) {
        let node = &self.retained_tree.nodes[node_id];
        let view_entity = node.view_entity.clone().unwrap();

        // Start dependency tracking
        let tracker = cx.start_dependency_tracking();

        // Compute render hash from view state
        let new_render_hash = view_entity.render_hash(cx);

        // Check if render is actually needed
        if new_render_hash == node.state.render_hash
            && !node.flags.contains(DirtyFlags::NEWLY_CREATED)
        {
            // Render hash unchanged, skip render entirely!
            self.retained_tree.nodes[node_id].flags.remove(DirtyFlags::NEEDS_RENDER);
            self.retained_tree.stats.nodes_skipped += 1;
            return;
        }

        // Actually call render()
        let new_element = view_entity.render(self, cx);

        // Record dependencies
        let dependencies = tracker.finish();

        // Reconcile new element with existing children
        self.reconcile_node_children(node_id, vec![new_element], cx);

        // Update state
        let node = &mut self.retained_tree.nodes[node_id];
        node.state.render_hash = new_render_hash;
        node.state.entity_dependencies = dependencies;
        node.flags.remove(DirtyFlags::NEEDS_RENDER);

        self.retained_tree.stats.nodes_rendered += 1;
    }
}
```

### 5.3 Phase 2: Incremental Layout

```rust
impl Window {
    fn layout_dirty_subtrees(&mut self, cx: &mut App) {
        if let Some(root_id) = self.retained_tree.root {
            let available_space = self.viewport_size().into();
            self.layout_node_if_needed(root_id, available_space, cx);
        }
    }

    fn layout_node_if_needed(
        &mut self,
        node_id: RetainedNodeId,
        available_space: Size<AvailableSpace>,
        cx: &mut App,
    ) {
        let node = &self.retained_tree.nodes[node_id];

        // Fast path: entire subtree is clean
        if !node.flags.intersects(DirtyFlags::NEEDS_LAYOUT | DirtyFlags::SUBTREE_DIRTY) {
            self.retained_tree.stats.nodes_skipped += 1;
            return; // Skip entire subtree!
        }

        // This node or a descendant needs layout
        if node.flags.contains(DirtyFlags::NEEDS_LAYOUT) {
            self.compute_node_layout(node_id, available_space, cx);
            self.retained_tree.stats.nodes_laid_out += 1;
        }

        // Recurse into children that might need layout
        let children = node.children.clone();
        for child_id in children {
            // Child available space comes from this node's layout
            let child_space = self.child_available_space(node_id, child_id);
            self.layout_node_if_needed(child_id, child_space, cx);
        }

        // Clear layout flags
        let node = &mut self.retained_tree.nodes[node_id];
        node.flags.remove(DirtyFlags::NEEDS_LAYOUT | DirtyFlags::SUBTREE_DIRTY);
    }

    fn compute_node_layout(
        &mut self,
        node_id: RetainedNodeId,
        available_space: Size<AvailableSpace>,
        cx: &mut App,
    ) {
        let node = &mut self.retained_tree.nodes[node_id];

        // Check if we can reuse cached layout
        if node.state.layout_constraints == Some(available_space) {
            // Constraints unchanged, layout is still valid
            return;
        }

        // Actually compute layout using Taffy
        // ... (existing layout logic, but for single node)

        node.state.layout_constraints = Some(available_space);
    }
}
```

### 5.4 Phase 3: Incremental Paint

```rust
impl Window {
    fn paint_dirty_elements(&mut self, cx: &mut App) {
        // Prepare display list for incremental updates
        self.prepare_display_list_for_incremental_paint();

        if let Some(root_id) = self.retained_tree.root {
            self.paint_node_if_needed(root_id, cx);
        }
    }

    fn paint_node_if_needed(&mut self, node_id: RetainedNodeId, cx: &mut App) {
        let node = &self.retained_tree.nodes[node_id];

        // Fast path: this element and all descendants are clean
        if !node.flags.intersects(DirtyFlags::NEEDS_PAINT | DirtyFlags::SUBTREE_DIRTY) {
            // Copy cached display items to output
            self.display_list.copy_items_from_cache(
                &self.cached_display_list,
                node.state.display_items.clone(),
            );
            self.retained_tree.stats.nodes_skipped += 1;
            return;
        }

        if node.flags.contains(DirtyFlags::NEEDS_PAINT) {
            // Actually paint this element
            let items_start = self.display_list.len();

            // Paint element's own content (background, borders, etc.)
            self.paint_element_content(node_id, cx);

            // Paint children in order
            let children = node.children.clone();
            for child_id in children {
                self.paint_node_if_needed(child_id, cx);
            }

            // Record painted range for future cache
            let items_end = self.display_list.len();
            self.retained_tree.nodes[node_id].state.display_items = items_start..items_end;

            self.retained_tree.stats.nodes_painted += 1;
        } else {
            // This node is clean but has dirty descendants
            // Paint our cached content, then recurse for children
            self.paint_element_content_from_cache(node_id);

            let children = node.children.clone();
            for child_id in children {
                self.paint_node_if_needed(child_id, cx);
            }
        }

        // Clear paint flags
        let node = &mut self.retained_tree.nodes[node_id];
        node.flags.remove(DirtyFlags::NEEDS_PAINT);
    }
}
```

---

## 6. Automatic Reactivity

### 6.1 Why Rust Makes This Easier

Unlike JavaScript, Rust's ownership model provides structural guarantees that enable automatic dependency tracking:

| JavaScript Challenge | Rust Advantage |
|---------------------|----------------|
| Objects mutable anywhere | Mutation requires `&mut` |
| Unknown variable types | Full type info at compile time |
| Closures capture anything | Explicit capture semantics |
| Runtime type uncertainty | Proc macros see all types |
| Need runtime tracking (MobX) | Compile-time analysis possible |

### 6.2 Dependency Sources

A view's render output can depend on exactly three things:

```rust
fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
    // 1. self.* - view's own state (tracked via derive macro)
    // 2. cx.read_entity() - other entities (tracked at runtime)
    // 3. window.* - window state (known set, tracked by framework)
}
```

This closed set of dependency sources enables comprehensive automatic tracking.

### 6.3 View State Tracking (Compile-Time)

```rust
/// Derive macro automatically generates render_hash from struct fields
#[derive(View)]
pub struct MyView {
    label: String,           // Included in render_hash
    count: u32,              // Included in render_hash
    items: Vec<Item>,        // Included in render_hash

    #[no_render]             // Excluded from render_hash
    internal_cache: Cache,
}

// Generated by #[derive(View)]:
impl MyView {
    fn render_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.label.hash(&mut hasher);
        self.count.hash(&mut hasher);
        self.items.hash(&mut hasher);
        // internal_cache excluded due to #[no_render]
        hasher.finish()
    }
}
```

### 6.4 Entity Dependency Tracking (Runtime)

```rust
impl<'a> Context<'a> {
    /// Read an entity's state, recording a dependency
    pub fn read_entity<T: 'static>(&self, handle: &Entity<T>) -> &T {
        // Record that current view depends on this entity
        if let Some(tracker) = self.dependency_tracker.as_ref() {
            tracker.record_read(handle.entity_id());
        }

        // Return the entity's state
        handle.read(self.app)
    }
}

// When an entity is updated:
impl<T: 'static> Entity<T> {
    pub fn update(&self, cx: &mut App, f: impl FnOnce(&mut T, &mut ModelContext<T>)) {
        // Perform the update
        f(&mut self.state, &mut model_cx);

        // Mark all dependent views as needing re-render
        if let Some(dependents) = cx.entity_dependents.get(&self.entity_id()) {
            for &node_id in dependents {
                cx.window.retained_tree.mark_needs_render(node_id);
            }
        }
    }
}
```

### 6.5 Dirty Flag Propagation

```rust
impl RetainedTree {
    /// Mark a node as needing re-render
    pub fn mark_needs_render(&mut self, node_id: RetainedNodeId) {
        let node = &mut self.nodes[node_id];

        if node.flags.contains(DirtyFlags::NEEDS_RENDER) {
            return; // Already dirty
        }

        node.flags |= DirtyFlags::NEEDS_RENDER
                    | DirtyFlags::NEEDS_LAYOUT
                    | DirtyFlags::NEEDS_PAINT;

        // Propagate SUBTREE_DIRTY up to root
        self.propagate_subtree_dirty(node.parent);

        // Add to processing queue
        self.dirty_roots.push(node_id);
    }

    /// Mark a node as needing re-layout (but not re-render)
    pub fn mark_needs_layout(&mut self, node_id: RetainedNodeId) {
        let node = &mut self.nodes[node_id];

        if node.flags.contains(DirtyFlags::NEEDS_LAYOUT) {
            return;
        }

        node.flags |= DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT;
        self.propagate_subtree_dirty(node.parent);
    }

    /// Mark a node as needing repaint only
    pub fn mark_needs_paint(&mut self, node_id: RetainedNodeId) {
        let node = &mut self.nodes[node_id];

        if node.flags.contains(DirtyFlags::NEEDS_PAINT) {
            return;
        }

        node.flags |= DirtyFlags::NEEDS_PAINT;
        self.propagate_subtree_dirty(node.parent);
    }

    fn propagate_subtree_dirty(&mut self, node_id: Option<RetainedNodeId>) {
        let mut current = node_id;
        while let Some(id) = current {
            let node = &mut self.nodes[id];
            if node.flags.contains(DirtyFlags::SUBTREE_DIRTY) {
                break; // Already marked, ancestors must be too
            }
            node.flags |= DirtyFlags::SUBTREE_DIRTY;
            current = node.parent;
        }
    }
}
```

---

## 7. Reconciliation

### 7.1 Overview

When a view re-renders, we must diff the new element output against the existing retained tree to determine minimal updates.

```rust
impl Window {
    fn reconcile_node_children(
        &mut self,
        parent_id: RetainedNodeId,
        new_children: Vec<AnyElement>,
        cx: &mut App,
    ) {
        let parent = &self.retained_tree.nodes[parent_id];
        let old_children = parent.children.clone();

        // Build map of old children by key
        let mut old_by_key: FxHashMap<ElementKey, RetainedNodeId> = old_children
            .iter()
            .map(|&id| {
                let node = &self.retained_tree.nodes[id];
                (node.key.clone(), id)
            })
            .collect();

        let mut new_child_ids = SmallVec::new();

        for (index, new_element) in new_children.into_iter().enumerate() {
            let key = new_element.key().unwrap_or(ElementKey::Index(index));

            if let Some(existing_id) = old_by_key.remove(&key) {
                // Found matching child by key - reconcile in place
                self.reconcile_existing_node(existing_id, new_element, cx);
                new_child_ids.push(existing_id);
            } else {
                // No match - create new retained node
                let new_id = self.create_retained_node(new_element, Some(parent_id), index);
                new_child_ids.push(new_id);
            }
        }

        // Remove old children that weren't matched
        for (_, orphan_id) in old_by_key {
            self.remove_retained_subtree(orphan_id);
        }

        // Update parent's children list
        let parent = &mut self.retained_tree.nodes[parent_id];
        parent.children = new_child_ids;
    }
}
```

### 7.2 Same-Type Reconciliation

```rust
impl Window {
    fn reconcile_existing_node(
        &mut self,
        node_id: RetainedNodeId,
        new_element: AnyElement,
        cx: &mut App,
    ) {
        let node = &self.retained_tree.nodes[node_id];

        // Check if types match
        if node.type_id != new_element.type_id() {
            // Type changed - replace entirely
            self.replace_retained_node(node_id, new_element, cx);
            return;
        }

        // Same type - check what changed
        let diff = self.diff_element_props(node_id, &new_element);

        match diff {
            ElementDiff::Unchanged => {
                // Nothing changed, keep everything as-is
            }
            ElementDiff::PropsChanged { affects_layout } => {
                if affects_layout {
                    self.retained_tree.mark_needs_layout(node_id);
                } else {
                    self.retained_tree.mark_needs_paint(node_id);
                }
                // Update stored element
                self.retained_tree.nodes[node_id].element = Some(new_element);
            }
            ElementDiff::ChildrenChanged { new_children } => {
                // Recursively reconcile children
                self.reconcile_node_children(node_id, new_children, cx);
            }
        }
    }
}

enum ElementDiff {
    Unchanged,
    PropsChanged { affects_layout: bool },
    ChildrenChanged { new_children: Vec<AnyElement> },
}
```

### 7.3 Keying Strategy

```rust
// Recommended: Use stable data IDs
div().children(items.iter().map(|item| {
    div()
        .key(ElementKey::Data(item.id))  // Stable across reorders
        .child(item.name.clone())
}))

// Fallback: Position-based keys (default)
div().children(items.iter().map(|item| {
    div().child(item.name.clone())  // Key = Index(0), Index(1), ...
}))

// Anti-pattern: Random keys (defeats reconciliation)
div().children(items.iter().map(|item| {
    div()
        .key(ElementKey::Data(rand::random()))  // DON'T DO THIS
        .child(item.name.clone())
}))
```

---

## 8. Incremental Layout

### 8.1 Layout Caching Strategy

```rust
struct LayoutCache {
    /// Last constraints used for layout
    constraints: Size<AvailableSpace>,
    /// Computed size from last layout
    size: Size<Pixels>,
    /// Whether layout is still valid
    valid: bool,
}

impl RetainedNode {
    fn needs_relayout(&self, new_constraints: Size<AvailableSpace>) -> bool {
        // Needs layout if:
        // 1. Explicitly marked dirty
        if self.flags.contains(DirtyFlags::NEEDS_LAYOUT) {
            return true;
        }

        // 2. Constraints changed
        if self.state.layout_constraints != Some(new_constraints) {
            return true;
        }

        false
    }
}
```

### 8.2 Integration with Taffy

```rust
impl Window {
    fn compute_node_layout(
        &mut self,
        node_id: RetainedNodeId,
        constraints: Size<AvailableSpace>,
        cx: &mut App,
    ) {
        let node = &self.retained_tree.nodes[node_id];

        // Get or create Taffy node
        let layout_id = node.state.layout_id.unwrap_or_else(|| {
            self.layout_engine.create_node()
        });

        // Update Taffy node style if needed
        if let Some(ref element) = node.element {
            let style = element.style();
            self.layout_engine.set_style(layout_id, style);
        }

        // Set children in Taffy
        let child_layout_ids: Vec<_> = node.children
            .iter()
            .filter_map(|&child_id| {
                self.retained_tree.nodes[child_id].state.layout_id
            })
            .collect();
        self.layout_engine.set_children(layout_id, &child_layout_ids);

        // Compute layout
        self.layout_engine.compute_layout(layout_id, constraints);

        // Store results
        let node = &mut self.retained_tree.nodes[node_id];
        node.state.layout_id = Some(layout_id);
        node.state.bounds = self.layout_engine.get_bounds(layout_id);
        node.state.layout_constraints = Some(constraints);
    }
}
```

---

## 9. Incremental Paint

### 9.1 Display List Management

```rust
pub struct IncrementalDisplayList {
    /// Current frame's display items
    items: Vec<DisplayItem>,

    /// Previous frame's items (for cache lookup)
    previous_items: Vec<DisplayItem>,

    /// Map from node ID to item range in previous_items
    node_to_cached_range: FxHashMap<RetainedNodeId, Range<usize>>,
}

impl IncrementalDisplayList {
    /// Copy cached items for a clean node
    pub fn copy_cached_items(&mut self, node_id: RetainedNodeId) {
        if let Some(range) = self.node_to_cached_range.get(&node_id) {
            let items = self.previous_items[range.clone()].to_vec();
            self.items.extend(items);
        }
    }

    /// Begin painting a node, returns start index
    pub fn begin_node(&mut self, node_id: RetainedNodeId) -> usize {
        self.items.len()
    }

    /// End painting a node, records range for future caching
    pub fn end_node(&mut self, node_id: RetainedNodeId, start: usize) {
        let end = self.items.len();
        self.node_to_cached_range.insert(node_id, start..end);
    }

    /// Swap current to previous for next frame
    pub fn end_frame(&mut self) {
        std::mem::swap(&mut self.items, &mut self.previous_items);
        self.items.clear();
    }
}
```

### 9.2 Paint Ordering

The retained tree maintains paint order through child ordering. Reconciliation preserves order where possible and handles reordering via keys.

```rust
impl Window {
    fn paint_node(&mut self, node_id: RetainedNodeId, cx: &mut App) {
        let start = self.display_list.begin_node(node_id);

        let node = &self.retained_tree.nodes[node_id];

        // 1. Paint node's own background/borders
        if let Some(ref element) = node.element {
            element.paint_background(self, cx);
        }

        // 2. Paint children in order (back to front)
        for &child_id in &node.children {
            self.paint_node_if_needed(child_id, cx);
        }

        // 3. Paint node's foreground (text, etc.)
        if let Some(ref element) = node.element {
            element.paint_foreground(self, cx);
        }

        self.display_list.end_node(node_id, start);
    }
}
```

---

## 10. Integration with Existing Systems

### 10.1 Compositor Thread Integration

The retained tree and compositor thread are complementary:

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│      MAIN THREAD            │    │    COMPOSITOR THREAD        │
│                             │    │                             │
│  Retained tree updates      │───▶│  Receives tile updates      │
│  Only dirty nodes painted   │    │  Composites at 120fps       │
│  DisplayList → Tiles        │    │  Handles scroll offset      │
│                             │    │                             │
└─────────────────────────────┘    └─────────────────────────────┘
```

Retained mode reduces main thread work, compositor thread decouples scroll from main thread. Together they enable:
- 120fps scroll always (compositor)
- 120fps content updates for sparse changes (retained mode)

### 10.2 Entity System

```rust
// Current entity update:
entity.update(cx, |state, cx| {
    state.value = new_value;
    cx.notify();  // Manual notification
});

// With retained mode:
entity.update(cx, |state, cx| {
    state.value = new_value;
    // Automatic: framework marks dependent views dirty
});
```

### 10.3 GlobalElementId Mapping

```rust
impl RetainedTree {
    /// Map from GlobalElementId (existing) to RetainedNodeId (new)
    global_id_to_node: FxHashMap<GlobalElementId, RetainedNodeId>,

    /// Get retained node for a global element ID
    pub fn node_for_global_id(&self, id: &GlobalElementId) -> Option<RetainedNodeId> {
        self.global_id_to_node.get(id).copied()
    }
}
```

---

## 11. Developer Experience

### 11.1 Zero-Annotation Usage (Goal)

```rust
#[derive(View)]
struct MyView {
    label: String,
    count: u32,
}

impl Render for MyView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Just write normal code - framework handles memoization
        div()
            .child(self.label.clone())
            .child(format!("Count: {}", self.count))
    }
}

// Updates automatically trigger minimal re-render:
entity.update(cx, |view, _| {
    view.count += 1;
    // Only this view re-renders, and only the count text repaints
});
```

### 11.2 Explicit Keying (When Needed)

```rust
// For dynamic lists, provide stable keys:
div().children(items.iter().map(|item| {
    div()
        .key(item.id)  // Stable identity for reconciliation
        .child(item.name.clone())
}))
```

### 11.3 Debugging Tools

```rust
// In debug builds, track why re-renders happen:
impl Window {
    pub fn debug_render_reason(&self, node_id: RetainedNodeId) -> RenderReason {
        // Returns: StateChanged { field: "count" }
        //      or: EntityChanged { entity: "user_data" }
        //      or: ParentRerendered
        //      or: NewlyCreated
    }
}

// Stats available for profiling:
let stats = window.retained_tree_stats();
println!("Rendered: {}/{} nodes", stats.nodes_rendered, stats.total_nodes);
println!("Skipped: {} nodes", stats.nodes_skipped);
```

### 11.4 Escape Hatches

```rust
// Force re-render (escape hatch)
cx.force_render();

// Mark specific element as always-dirty (for animations)
div()
    .always_repaint()  // Opts out of paint caching
    .child(animated_content)

// Exclude field from render hash
#[derive(View)]
struct MyView {
    label: String,
    #[no_render]
    animation_state: AnimationState,  // Changes don't trigger re-render
}
```

---

## 12. Migration Strategy

### 12.1 Phase 1: Infrastructure (Non-Breaking)

**Goal**: Add retained tree alongside existing immediate mode.

```rust
// Window gains retained tree, but doesn't use it yet
pub struct Window {
    // ... existing fields ...
    retained_tree: Option<RetainedTree>,  // New, optional
}
```

**Tasks**:
- [ ] Implement `RetainedTree`, `RetainedNode`, `DirtyFlags`
- [ ] Add `RetainedNodeId` generation
- [ ] Implement dirty flag propagation
- [ ] Add tree traversal utilities
- [ ] Unit tests for tree operations

### 12.2 Phase 2: Parallel Execution (Validation)

**Goal**: Run both paths, validate they produce identical output.

```rust
impl Window {
    fn draw(&mut self, cx: &mut App) {
        // Run existing immediate mode
        let immediate_result = self.draw_immediate(cx);

        // Run new retained mode
        let retained_result = self.draw_retained(cx);

        // Validate (debug builds only)
        #[cfg(debug_assertions)]
        self.validate_results(&immediate_result, &retained_result);

        // Use immediate mode result for now
        immediate_result
    }
}
```

**Tasks**:
- [ ] Implement retained render path
- [ ] Implement reconciliation
- [ ] Add result comparison logic
- [ ] Fix discrepancies until outputs match

### 12.3 Phase 3: View Memoization (Opt-In)

**Goal**: Views can opt-in to memoization via derive macro.

```rust
// Opt-in to retained mode benefits
#[derive(View)]  // New derive macro
struct MyView { ... }

// Without derive, falls back to always-render (existing behavior)
struct LegacyView { ... }
```

**Tasks**:
- [ ] Implement `#[derive(View)]` proc macro
- [ ] Generate `render_hash` from struct fields
- [ ] Add `#[no_render]` attribute support
- [ ] Documentation and examples

### 12.4 Phase 4: Incremental Layout (Automatic)

**Goal**: Layout automatically skips clean subtrees.

**Tasks**:
- [ ] Integrate retained tree with Taffy
- [ ] Implement layout constraint caching
- [ ] Skip layout for unchanged constraints
- [ ] Validate layout correctness

### 12.5 Phase 5: Incremental Paint (Automatic)

**Goal**: Paint automatically skips clean elements.

**Tasks**:
- [ ] Implement `IncrementalDisplayList`
- [ ] Cache display item ranges per node
- [ ] Copy cached items for clean nodes
- [ ] Validate paint correctness

### 12.6 Phase 6: Entity Integration (Automatic)

**Goal**: Entity updates automatically mark dependent views dirty.

**Tasks**:
- [ ] Implement dependency tracking in `Context`
- [ ] Record entity reads during render
- [ ] Mark dependents dirty on entity update
- [ ] Remove manual `cx.notify()` requirement

### 12.7 Phase 7: Default Enable (Breaking)

**Goal**: Retained mode becomes the default.

**Tasks**:
- [ ] Enable retained mode by default
- [ ] Remove immediate mode code paths
- [ ] Performance benchmarking
- [ ] Documentation updates

---

## 13. Performance Expectations

### 13.1 Theoretical Complexity

| Scenario | Current | Retained |
|----------|---------|----------|
| Nothing changed | O(n) | O(1) |
| One element changed | O(n) | O(1) |
| k elements changed | O(n) | O(k) |
| All elements changed | O(n) | O(n) |
| Scroll (with compositor) | O(visible) | O(new rows) |

### 13.2 cache_bench Predictions

Current (375 grid cells + 30 stats elements):
- Every frame: O(405) work
- Debug mode: ~30fps

With retained mode:
- Frame where only stats change: O(30) work
- Speedup: ~13x
- Debug mode prediction: ~100+ fps

With stats in separate view:
- Frame where nothing changes: O(1) work
- Debug mode prediction: 120fps easily

### 13.3 Memory Overhead

Per retained node:
- `RetainedNode` struct: ~200 bytes
- `RetainedState`: ~100 bytes
- Tree pointers: ~32 bytes
- **Total**: ~332 bytes per element

For 1000 elements: ~332 KB additional memory (acceptable)

### 13.4 Benchmarks to Create

```rust
#[bench]
fn bench_static_grid_current(b: &mut Bencher) {
    // Measure current immediate mode with static grid
}

#[bench]
fn bench_static_grid_retained(b: &mut Bencher) {
    // Measure retained mode with static grid (should be ~O(1))
}

#[bench]
fn bench_single_update_current(b: &mut Bencher) {
    // Measure current mode with one element changing
}

#[bench]
fn bench_single_update_retained(b: &mut Bencher) {
    // Measure retained mode with one element changing
}
```

---

## 14. Open Questions

### 14.1 Context Propagation

**Question**: How do context changes (theme, locale) invalidate dependent subtrees?

**Potential approach**: Track context reads like entity reads, invalidate when context changes.

### 14.2 Layout Dependencies

**Question**: How to handle layout dependencies where sibling size affects my size?

**Potential approach**: Taffy handles this internally; we just need to re-run layout when any child changes.

### 14.3 Animation Integration

**Question**: How do animations work with retained mode?

**Potential approach**: Animated elements mark themselves `always_repaint`, or animations update element state which triggers normal dirty propagation.

### 14.4 Event Handler Identity

**Question**: Event handler closures change identity every render. Does this cause issues?

**Potential approach**: Event handlers are stored by element, not compared for equality. Reconciliation by key handles handler updates.

### 14.5 Memory Management

**Question**: When are retained nodes garbage collected?

**Potential approach**: Nodes removed from tree during reconciliation are immediately dropped. No separate GC needed.

### 14.6 Hot Reloading

**Question**: How does hot reloading interact with retained tree?

**Potential approach**: Hot reload clears retained tree, forces full re-render. Acceptable for development.

---

## Appendix A: Comparison with Other Frameworks

| Framework | Architecture | Memoization | Change Detection |
|-----------|--------------|-------------|------------------|
| React | Virtual DOM, reconciliation | Manual (useMemo) or Compiler | Props comparison |
| React Native | Retained native views | Manual or Compiler | Props + bridge messages |
| Flutter | Retained widget tree | Automatic (const widgets) | Widget identity |
| SwiftUI | Retained view tree | Automatic (value types) | Equatable conformance |
| Solid.js | Fine-grained reactivity | Automatic (signals) | Signal subscriptions |
| **GPUI (current)** | Immediate mode | Paint-level only | None (always rebuild) |
| **GPUI (proposed)** | Retained mode | Automatic (ownership) | Render hash + entity deps |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Immediate Mode** | UI rebuilt from scratch every frame |
| **Retained Mode** | UI tree persists across frames |
| **Reconciliation** | Diffing new elements against existing tree |
| **Dirty Flags** | Bits indicating what work is needed |
| **Render Hash** | Hash of inputs affecting render output |
| **Entity Dependency** | View depends on entity state |
| **SUBTREE_DIRTY** | Flag enabling O(1) skip of clean subtrees |

---

## Appendix C: References

1. React Fiber Architecture: https://github.com/acdlite/react-fiber-architecture
2. React Compiler: https://react.dev/learn/react-compiler
3. Flutter Rendering: https://docs.flutter.dev/resources/architectural-overview#rendering
4. SwiftUI View Updates: https://developer.apple.com/documentation/swiftui/view
5. Solid.js Reactivity: https://www.solidjs.com/guides/reactivity
