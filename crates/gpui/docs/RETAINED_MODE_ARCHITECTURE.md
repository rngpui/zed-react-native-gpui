# Fiber Architecture for GPUI

## Executive Summary

This document describes the **fiber architecture** for GPUI - a fundamental redesign that achieves **O(changed) work per frame** by separating lightweight descriptors from persistent fibers.

**The Problem**: GPUI currently does O(n) work per frame regardless of what changed, resulting in ~30fps in debug mode for a static 375-element grid.

**The Solution**: Split the element system into two layers:
1. **Descriptors** - Lightweight data objects created fresh each frame (cheap)
2. **Fibers** - Persistent nodes that cache layout/paint results (expensive work cached)

**The Goal**: 120fps in debug mode. O(1) frames when nothing changes.

**Key Insight**: Creating descriptors is cheap. Layout and paint are expensive. Cache the expensive work in fibers, let descriptors be ephemeral.

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Structures](#3-data-structures)
4. [Frame Lifecycle](#4-frame-lifecycle)
5. [Reconciliation](#5-reconciliation)
6. [Dirty Flag Propagation](#6-dirty-flag-propagation)
7. [Layout Integration](#7-layout-integration)
8. [Paint Integration](#8-paint-integration)
9. [Implementation Plan](#9-implementation-plan)
10. [API Changes](#10-api-changes)
11. [Performance Analysis](#11-performance-analysis)

---

## 1. Problem Analysis

### 1.1 Current Architecture

GPUI is immediate-mode: `render()` creates fresh element structs every frame.

```
Frame N (something changes):
  render() → allocate 375 Divs → request_layout → prepaint → paint
                ↑
            O(n) work even if only 1 element changed

Frame N+1 (nothing changed):
  render() → allocate 375 Divs → request_layout → prepaint → paint
                ↑
            STILL O(n) work - no caching helps here
```

### 1.2 Why Previous Approaches Failed

**Approach 1: Retain Element Instances**
- Elements are arena-allocated, cleared every frame
- Retaining them requires redesigning allocation and lifetimes
- Doesn't solve the fundamental problem of O(n) traversal

**Approach 2: Retained Artifacts (layout nodes, paint entries)**
- Caches expensive results but still traverses entire tree
- `render()` still creates fresh elements before any caching kicks in
- O(n) allocation happens before reconciliation can help

**Approach 3: Retained Tree + Reconciliation**
- Reconciles ephemeral elements against persistent tree
- Problem: `render()` already created all elements by the time reconciliation runs
- We've done O(n) allocation before any diffing happens

### 1.3 The Core Insight

> **Don't create elements. Create descriptors. Only do expensive work for dirty fibers.**

Descriptors are just data - struct allocation. The expensive work is:
1. Layout computation (Taffy calls)
2. Prepaint (bounds, hitboxes)
3. Paint (GPU commands)

If we cache expensive work in persistent fibers, and descriptors are cheap, we achieve O(changed) work even though descriptor creation is O(n).

### 1.4 Comparison with React

React solved this exact problem:

| Concept | React | GPUI Fiber |
|---------|-------|------------|
| Lightweight output | JSX elements | Descriptors |
| Persistent tree | Fiber tree | Fiber tree |
| Diffing | Reconciliation | Reconciliation |
| Cached work | DOM nodes, layout | Layout, paint |

React creates JSX elements every render (cheap). The reconciler diffs them against fibers. Only changed fibers update the DOM (expensive).

---

## 2. Architecture Overview

### 2.1 Two-Layer Design

```
┌─────────────────────────────────────────────────────────────────┐
│  DESCRIPTORS (created fresh each frame)                         │
│                                                                 │
│  render() → DivDescriptor { style, children: [TextDescriptor] } │
│                                                                 │
│  Properties:                                                    │
│  - Just data (style config, text content, children)             │
│  - No layout, no paint, no GPU resources                        │
│  - Cheap to create: ~100-200 bytes per element                  │
│  - O(n) creation is acceptable                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ reconcile
┌─────────────────────────────────────────────────────────────────┐
│  FIBER TREE (persistent across frames)                          │
│                                                                 │
│  Fiber {                                                        │
│      descriptor_hash,    // For change detection                │
│      layout_id,          // Cached Taffy result                 │
│      bounds,             // Cached bounds                       │
│      paint_range,        // Cached paint commands               │
│      children: [FiberId],                                       │
│      dirty: DirtyFlags,                                         │
│  }                                                              │
│                                                                 │
│  Properties:                                                    │
│  - Persists across frames                                       │
│  - Caches expensive computation results                         │
│  - Skip flags enable O(1) subtree skipping                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ if dirty
┌─────────────────────────────────────────────────────────────────┐
│  LAYOUT / PAINT (only for dirty fibers)                         │
│                                                                 │
│  - Check fiber.dirty flags                                      │
│  - If clean && no SUBTREE_DIRTY: return cached results          │
│  - If dirty: compute layout/paint, update fiber cache           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Principles

| Principle | Description |
|-----------|-------------|
| **Descriptors are cheap** | Just data, no expensive work |
| **Fibers are persistent** | Survive across frames, cache results |
| **Reconciliation diffs** | Compare descriptors to fibers, mark dirty |
| **Dirty drives work** | Only dirty fibers do layout/paint |
| **Subtree skip** | Clean subtrees skip entirely via SUBTREE_DIRTY |

### 2.3 No Special "View" Concept

In this architecture, there's no distinction between "views" and "elements". Everything is:
- A descriptor (ephemeral data)
- Backed by a fiber (persistent cache)

`Entity<V>` becomes just another fiber node that happens to have state. The rendering model is uniform.

---

## 3. Data Structures

### 3.1 Descriptors

Descriptors are lightweight data objects. They describe UI structure without doing expensive work.

```rust
/// Lightweight descriptor - just configuration, no layout/paint
pub enum AnyDescriptor {
    Div(DivDescriptor),
    Text(TextDescriptor),
    Image(ImageDescriptor),
    Svg(SvgDescriptor),
    Canvas(CanvasDescriptor),
    // ... other element types
}

pub struct DivDescriptor {
    /// Style configuration
    pub style: StyleRefinement,

    /// Pre-computed style hash for fast comparison
    pub style_hash: u64,

    /// Optional key for stable identity in lists
    pub key: Option<ElementId>,

    /// Child descriptors
    pub children: SmallVec<[AnyDescriptor; 4]>,

    /// Event handlers (stored separately, not hashed)
    pub interactivity: Interactivity,
}

pub struct TextDescriptor {
    /// The text content
    pub text: SharedString,

    /// Text styling
    pub style: TextStyle,

    /// Pre-computed content hash
    pub content_hash: u64,
}

impl DivDescriptor {
    /// Compute hash for change detection (excludes children - they're compared recursively)
    pub fn hash(&self) -> u64 {
        self.style_hash
    }
}

impl TextDescriptor {
    /// Compute hash for change detection
    pub fn hash(&self) -> u64 {
        self.content_hash
    }
}
```

### 3.2 Fibers

Fibers are persistent nodes that cache expensive computation results.

```rust
/// Opaque fiber identifier
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FiberId(u32);

/// A fiber node - persistent across frames
pub struct Fiber {
    // === Identity ===

    /// Unique identifier
    pub id: FiberId,

    /// Type of element this fiber represents
    pub element_type: TypeId,

    /// Optional key for list reconciliation
    pub key: Option<ElementId>,

    // === Change Detection ===

    /// Hash of the descriptor that created/updated this fiber
    pub descriptor_hash: u64,

    // === Cached Layout ===

    /// Taffy layout node (persists across frames)
    pub layout_id: Option<LayoutId>,

    /// Cached layout result
    pub bounds: Option<Bounds<Pixels>>,

    /// Cached size from layout
    pub size: Option<Size<Pixels>>,

    // === Cached Paint ===

    /// Range of paint commands in the scene
    pub paint_range: Option<Range<PaintIndex>>,

    /// Cached prepaint state
    pub prepaint_state: Option<Box<dyn Any + Send>>,

    // === Tree Structure ===

    /// Parent fiber (None for root)
    pub parent: Option<FiberId>,

    /// Child fibers
    pub children: SmallVec<[FiberId; 4]>,

    // === Dirty Tracking ===

    /// What work needs to be done
    pub dirty: DirtyFlags,
}

bitflags::bitflags! {
    /// Flags indicating what work is needed for this fiber
    #[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
    pub struct DirtyFlags: u8 {
        /// Layout inputs changed, need to recompute
        const NEEDS_LAYOUT  = 0b0001;

        /// Visual appearance changed, need to repaint
        const NEEDS_PAINT   = 0b0010;

        /// Some descendant has dirty flags (enables subtree skip)
        const SUBTREE_DIRTY = 0b0100;

        /// Fiber is newly created this frame
        const NEW           = 0b1000;
    }
}

impl Fiber {
    /// Check if this fiber needs any work
    pub fn needs_work(&self) -> bool {
        self.dirty.intersects(DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT)
    }

    /// Check if this fiber or any descendant needs work
    pub fn subtree_needs_work(&self) -> bool {
        self.dirty.intersects(
            DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT | DirtyFlags::SUBTREE_DIRTY
        )
    }
}
```

### 3.3 Fiber Tree

The fiber tree is stored in `Window` and persists across frames.

```rust
/// The persistent fiber tree
pub struct FiberTree {
    /// All fibers, indexed by FiberId
    fibers: SlotMap<FiberId, Fiber>,

    /// Root fiber for each view
    view_roots: HashMap<EntityId, FiberId>,

    /// Free list for fiber reuse
    free_list: Vec<FiberId>,

    /// Statistics for debugging
    stats: FiberTreeStats,
}

#[derive(Default, Debug)]
pub struct FiberTreeStats {
    pub fibers_created: usize,
    pub fibers_reused: usize,
    pub fibers_removed: usize,
    pub reconcile_comparisons: usize,
    pub subtrees_skipped: usize,
}

impl FiberTree {
    /// Get a fiber by ID
    pub fn get(&self, id: FiberId) -> Option<&Fiber> {
        self.fibers.get(id)
    }

    /// Get a mutable fiber by ID
    pub fn get_mut(&mut self, id: FiberId) -> Option<&mut Fiber> {
        self.fibers.get_mut(id)
    }

    /// Create a new fiber
    pub fn create(&mut self, element_type: TypeId, key: Option<ElementId>) -> FiberId {
        self.stats.fibers_created += 1;

        // Try to reuse from free list
        if let Some(id) = self.free_list.pop() {
            let fiber = self.fibers.get_mut(id).unwrap();
            fiber.element_type = element_type;
            fiber.key = key;
            fiber.dirty = DirtyFlags::NEW | DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT;
            fiber.children.clear();
            self.stats.fibers_reused += 1;
            return id;
        }

        // Allocate new
        self.fibers.insert(Fiber {
            id: FiberId(0), // Will be set by SlotMap
            element_type,
            key,
            descriptor_hash: 0,
            layout_id: None,
            bounds: None,
            size: None,
            paint_range: None,
            prepaint_state: None,
            parent: None,
            children: SmallVec::new(),
            dirty: DirtyFlags::NEW | DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT,
        })
    }

    /// Remove a fiber and its descendants
    pub fn remove(&mut self, id: FiberId) {
        if let Some(fiber) = self.fibers.get(id) {
            // Recursively remove children
            let children: SmallVec<[FiberId; 4]> = fiber.children.clone();
            for child_id in children {
                self.remove(child_id);
            }

            // Add to free list for reuse
            self.free_list.push(id);
            self.stats.fibers_removed += 1;
        }
    }

    /// Mark a fiber dirty and propagate SUBTREE_DIRTY to ancestors
    pub fn mark_dirty(&mut self, id: FiberId, flags: DirtyFlags) {
        if let Some(fiber) = self.fibers.get_mut(id) {
            fiber.dirty.insert(flags);

            // Propagate SUBTREE_DIRTY up to root
            let mut current = fiber.parent;
            while let Some(parent_id) = current {
                if let Some(parent) = self.fibers.get_mut(parent_id) {
                    if parent.dirty.contains(DirtyFlags::SUBTREE_DIRTY) {
                        break; // Already propagated
                    }
                    parent.dirty.insert(DirtyFlags::SUBTREE_DIRTY);
                    current = parent.parent;
                } else {
                    break;
                }
            }
        }
    }

    /// Clear dirty flags after frame
    pub fn clear_dirty_flags(&mut self) {
        for (_, fiber) in &mut self.fibers {
            fiber.dirty = DirtyFlags::empty();
        }
    }
}
```

---

## 4. Frame Lifecycle

### 4.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  FRAME CYCLE                                                     │
│                                                                 │
│  1. render()      → Build descriptor tree (O(n) but cheap)      │
│  2. reconcile()   → Diff descriptors vs fibers, mark dirty      │
│  3. layout()      → Compute layout for dirty fibers only        │
│  4. prepaint()    → Compute bounds for dirty fibers only        │
│  5. paint()       → Generate commands for dirty fibers only     │
│  6. composite()   → Send to GPU (unchanged)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Frame 1: Initial Render

```
1. render() builds descriptor tree:
   DivDescriptor {
       style: {bg: 0x1e1e2e},
       children: [
           TextDescriptor { text: "FPS: 60" },
           DivDescriptor {
               key: "grid",
               children: [
                   DivDescriptor { children: [TextDescriptor { text: "0" }] },
                   DivDescriptor { children: [TextDescriptor { text: "1" }] },
                   ... (375 total)
               ]
           }
       ]
   }

2. reconcile() creates fiber tree (all new):
   - Create Fiber for each descriptor
   - All fibers marked NEEDS_LAYOUT | NEEDS_PAINT | NEW
   - Build parent/child relationships

3. layout() computes all (all dirty):
   - Request Taffy layout for each fiber
   - Cache layout_id and bounds in fiber

4. paint() generates all (all dirty):
   - Generate paint commands for each fiber
   - Cache paint_range in fiber

5. composite() sends to GPU
```

### 4.3 Frame 2: FPS Changes, Grid Unchanged

```
1. render() builds new descriptor tree:
   DivDescriptor {
       style: {bg: 0x1e1e2e},        // SAME
       children: [
           TextDescriptor { text: "FPS: 61" },  // CHANGED
           DivDescriptor {
               key: "grid",
               children: [ ... same 375 ... ]   // SAME
           }
       ]
   }

2. reconcile() diffs against fiber tree:
   - Root div: hash matches → no dirty flags
   - FPS text: hash DIFFERENT → mark NEEDS_LAYOUT | NEEDS_PAINT
   - Grid div: hash matches → no dirty flags
   - Grid children: all hashes match → no dirty flags
   - Propagate SUBTREE_DIRTY from FPS text to root

   Result:
   - FPS text fiber: NEEDS_LAYOUT | NEEDS_PAINT
   - Root fiber: SUBTREE_DIRTY
   - Grid fiber: (clean)
   - 375 grid children: (clean)

3. layout() with skip logic:
   - Root: has SUBTREE_DIRTY → check children
     - FPS text: NEEDS_LAYOUT → recompute
     - Grid: clean, no SUBTREE_DIRTY → SKIP ENTIRE SUBTREE

   Work done: 2 fibers. Skipped: 376 fibers.

4. paint() with skip logic:
   - Root: has SUBTREE_DIRTY → check children
     - FPS text: NEEDS_PAINT → repaint
     - Grid: clean → replay cached paint_range

   Work done: 1 repaint. Replayed: 375 cached.

5. composite() sends to GPU
```

### 4.4 Frame 3: Nothing Changed

```
1. render() builds same descriptor tree

2. reconcile() finds all hashes match:
   - All fibers: dirty = (empty)
   - No SUBTREE_DIRTY anywhere

3. layout():
   - Root: no NEEDS_LAYOUT, no SUBTREE_DIRTY
   - Return cached layout_id immediately

   Work done: O(1)

4. paint():
   - Root: no NEEDS_PAINT, no SUBTREE_DIRTY
   - Replay entire cached paint_range

   Work done: O(1)

5. composite() sends to GPU
```

---

## 5. Reconciliation

### 5.1 Overview

Reconciliation compares descriptors against the existing fiber tree and:
1. Updates fibers that changed
2. Creates fibers for new descriptors
3. Removes fibers for deleted descriptors
4. Sets dirty flags appropriately

### 5.2 Algorithm

```rust
impl FiberTree {
    /// Reconcile a descriptor tree against the fiber tree
    pub fn reconcile(
        &mut self,
        fiber_id: FiberId,
        descriptor: &AnyDescriptor,
    ) {
        let fiber = self.fibers.get_mut(fiber_id).unwrap();
        let new_hash = descriptor.hash();

        // Check if this fiber changed
        if fiber.descriptor_hash != new_hash {
            fiber.descriptor_hash = new_hash;
            fiber.dirty.insert(DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT);
        }

        // Reconcile children
        self.reconcile_children(fiber_id, descriptor.children());

        self.stats.reconcile_comparisons += 1;
    }

    /// Reconcile child descriptors against fiber's children
    fn reconcile_children(
        &mut self,
        parent_id: FiberId,
        descriptors: &[AnyDescriptor],
    ) {
        let parent = self.fibers.get(parent_id).unwrap();
        let old_children = parent.children.clone();

        // Build map of keyed children for O(1) lookup
        let mut old_by_key: HashMap<ElementId, FiberId> = old_children
            .iter()
            .filter_map(|&id| {
                self.fibers.get(id)
                    .and_then(|f| f.key.map(|k| (k, id)))
            })
            .collect();

        // Track which old children are reused
        let mut reused: HashSet<FiberId> = HashSet::new();
        let mut new_children: SmallVec<[FiberId; 4]> = SmallVec::new();

        for (i, desc) in descriptors.iter().enumerate() {
            let child_id = if let Some(key) = desc.key() {
                // Keyed: look up by key
                if let Some(existing_id) = old_by_key.remove(&key) {
                    reused.insert(existing_id);
                    self.reconcile(existing_id, desc);
                    existing_id
                } else {
                    // New keyed child
                    self.create_from_descriptor(desc, Some(parent_id))
                }
            } else {
                // Unkeyed: positional matching
                if i < old_children.len() {
                    let existing_id = old_children[i];
                    let existing = self.fibers.get(existing_id).unwrap();

                    // Same type? Update in place
                    if existing.element_type == desc.type_id() && existing.key.is_none() {
                        reused.insert(existing_id);
                        self.reconcile(existing_id, desc);
                        existing_id
                    } else {
                        // Type mismatch: create new
                        self.create_from_descriptor(desc, Some(parent_id))
                    }
                } else {
                    // New child
                    self.create_from_descriptor(desc, Some(parent_id))
                }
            };

            new_children.push(child_id);
        }

        // Remove orphaned children (not reused)
        for old_id in old_children {
            if !reused.contains(&old_id) {
                self.remove(old_id);
            }
        }

        // Update parent's children
        self.fibers.get_mut(parent_id).unwrap().children = new_children;

        // Propagate SUBTREE_DIRTY if any child is dirty
        let any_child_dirty = new_children.iter().any(|&id| {
            self.fibers.get(id).map_or(false, |f| f.subtree_needs_work())
        });

        if any_child_dirty {
            self.fibers.get_mut(parent_id).unwrap()
                .dirty.insert(DirtyFlags::SUBTREE_DIRTY);
        }
    }

    /// Create a new fiber from a descriptor
    fn create_from_descriptor(
        &mut self,
        descriptor: &AnyDescriptor,
        parent: Option<FiberId>,
    ) -> FiberId {
        let fiber_id = self.create(descriptor.type_id(), descriptor.key());

        let fiber = self.fibers.get_mut(fiber_id).unwrap();
        fiber.descriptor_hash = descriptor.hash();
        fiber.parent = parent;

        // Recursively create children
        for child_desc in descriptor.children() {
            let child_id = self.create_from_descriptor(child_desc, Some(fiber_id));
            self.fibers.get_mut(fiber_id).unwrap().children.push(child_id);
        }

        fiber_id
    }
}
```

### 5.3 Keyed Reconciliation

Keys are essential for list stability. Without keys, inserting at the beginning causes all fibers to be recreated:

```rust
// Without keys: insert at index 0 → all fibers shift, all recreated
[A, B, C] + insert X at 0 → [X, A, B, C]
// Positional matching: X!=A, A!=B, B!=C, C!=? → recreate all

// With keys: insert at index 0 → only X is new
[A:1, B:2, C:3] + insert X:0 at 0 → [X:0, A:1, B:2, C:3]
// Key matching: A:1→A:1, B:2→B:2, C:3→C:3, X:0 is new → create only X
```

Usage:

```rust
div().children(
    items.iter().map(|item| {
        div()
            .key(item.id)  // Stable identity
            .child(text(&item.name))
    })
)
```

---

## 6. Dirty Flag Propagation

### 6.1 Bottom-Up: SUBTREE_DIRTY

When a fiber is marked dirty, propagate `SUBTREE_DIRTY` to all ancestors:

```rust
fn mark_dirty(&mut self, id: FiberId, flags: DirtyFlags) {
    self.fibers.get_mut(id).unwrap().dirty.insert(flags);

    // Propagate up
    let mut current = self.fibers.get(id).unwrap().parent;
    while let Some(parent_id) = current {
        let parent = self.fibers.get_mut(parent_id).unwrap();
        if parent.dirty.contains(DirtyFlags::SUBTREE_DIRTY) {
            break; // Already propagated
        }
        parent.dirty.insert(DirtyFlags::SUBTREE_DIRTY);
        current = parent.parent;
    }
}
```

### 6.2 Top-Down: Skip Logic

During traversal, skip entire subtrees that are clean:

```rust
fn traverse_layout(&mut self, fiber_id: FiberId) -> LayoutId {
    let fiber = self.fibers.get(fiber_id).unwrap();

    // SKIP: clean and no dirty descendants
    if !fiber.needs_work() && !fiber.dirty.contains(DirtyFlags::SUBTREE_DIRTY) {
        self.stats.subtrees_skipped += 1;
        return fiber.layout_id.unwrap(); // Return cached
    }

    // Process children (recursive, may skip clean subtrees)
    let child_layouts: Vec<LayoutId> = fiber.children.iter()
        .map(|&child_id| self.traverse_layout(child_id))
        .collect();

    // Compute layout if this fiber needs it
    if fiber.dirty.contains(DirtyFlags::NEEDS_LAYOUT) {
        let layout_id = self.taffy.compute_layout(fiber, &child_layouts);
        self.fibers.get_mut(fiber_id).unwrap().layout_id = Some(layout_id);
        layout_id
    } else {
        fiber.layout_id.unwrap()
    }
}
```

### 6.3 Clearing Flags

After each frame, clear dirty flags:

```rust
fn end_frame(&mut self) {
    self.fiber_tree.clear_dirty_flags();
}
```

---

## 7. Layout Integration

### 7.1 Taffy Node Persistence

Each fiber gets a persistent Taffy `NodeId`:

```rust
impl Fiber {
    fn ensure_layout_node(&mut self, taffy: &mut TaffyLayoutEngine) {
        if self.layout_id.is_none() {
            self.layout_id = Some(taffy.create_node());
        }
    }
}
```

### 7.2 Layout Computation

```rust
fn layout_fiber(
    tree: &mut FiberTree,
    taffy: &mut TaffyLayoutEngine,
    fiber_id: FiberId,
) -> LayoutId {
    let fiber = tree.get(fiber_id).unwrap();

    // Skip if clean
    if !fiber.dirty.contains(DirtyFlags::NEEDS_LAYOUT)
        && !fiber.dirty.contains(DirtyFlags::SUBTREE_DIRTY)
    {
        return fiber.layout_id.unwrap();
    }

    // Layout children first
    let children = fiber.children.clone();
    let child_layouts: Vec<LayoutId> = children.iter()
        .map(|&id| layout_fiber(tree, taffy, id))
        .collect();

    let fiber = tree.get_mut(fiber_id).unwrap();

    if fiber.dirty.contains(DirtyFlags::NEEDS_LAYOUT) {
        // Update Taffy node with current style and children
        let layout_id = fiber.layout_id.unwrap();
        taffy.set_style(layout_id, &fiber.style);
        taffy.set_children(layout_id, &child_layouts);
        taffy.compute_layout(layout_id);

        // Cache results
        fiber.bounds = Some(taffy.get_bounds(layout_id));
        fiber.dirty.remove(DirtyFlags::NEEDS_LAYOUT);
    }

    fiber.dirty.remove(DirtyFlags::SUBTREE_DIRTY);
    fiber.layout_id.unwrap()
}
```

---

## 8. Paint Integration

### 8.1 Paint with Caching

```rust
fn paint_fiber(
    tree: &mut FiberTree,
    scene: &mut Scene,
    fiber_id: FiberId,
) {
    let fiber = tree.get(fiber_id).unwrap();

    // Skip if clean - replay cached commands
    if !fiber.dirty.contains(DirtyFlags::NEEDS_PAINT)
        && !fiber.dirty.contains(DirtyFlags::SUBTREE_DIRTY)
    {
        if let Some(range) = &fiber.paint_range {
            scene.replay_range(range.clone());
        }
        return;
    }

    let start = scene.len();

    // Paint this fiber
    paint_fiber_content(fiber, scene);

    // Paint children
    let children = fiber.children.clone();
    for child_id in children {
        paint_fiber(tree, scene, child_id);
    }

    let end = scene.len();

    // Cache paint range
    let fiber = tree.get_mut(fiber_id).unwrap();
    fiber.paint_range = Some(start..end);
    fiber.dirty.remove(DirtyFlags::NEEDS_PAINT | DirtyFlags::SUBTREE_DIRTY);
}
```

### 8.2 Scene Replay

For clean subtrees, replay cached paint commands:

```rust
impl Scene {
    /// Replay previously recorded commands
    fn replay_range(&mut self, range: Range<PaintIndex>) {
        // Copy commands from cache to current scene
        // Commands already have correct bounds from when they were recorded
        for i in range {
            self.commands.push(self.cached_commands[i].clone());
        }
    }
}
```

---

## 9. Implementation Plan

### Phase 1: Descriptor Types

**Files**: `src/descriptor.rs` (new)

1. Define `AnyDescriptor` enum
2. Define `DivDescriptor`, `TextDescriptor`, etc.
3. Implement `hash()` for change detection
4. Implement `IntoDescriptor` trait

### Phase 2: Fiber Tree

**Files**: `src/fiber.rs` (new), `src/window.rs`

1. Define `Fiber` struct
2. Define `FiberTree` with SlotMap storage
3. Implement dirty flag propagation
4. Add `FiberTree` to `Window`

### Phase 3: Reconciliation

**Files**: `src/fiber.rs`

1. Implement `reconcile()` algorithm
2. Implement keyed child matching
3. Implement fiber creation/removal
4. Wire reconciliation into frame cycle

### Phase 4: Builder API

**Files**: `src/elements/div.rs`, `src/styled.rs`

1. Change `div()` to return descriptor builder
2. Implement `IntoDescriptor` for builder
3. Update `Styled` trait for descriptors
4. Ensure backward compatibility

### Phase 5: Layout Integration

**Files**: `src/taffy.rs`, `src/window.rs`

1. Wire fiber tree into layout phase
2. Implement layout skip logic
3. Cache layout results in fibers
4. Verify Taffy node reuse

### Phase 6: Paint Integration

**Files**: `src/window.rs`, `src/scene.rs`

1. Wire fiber tree into paint phase
2. Implement paint skip logic
3. Implement scene replay for cached commands
4. Cache paint ranges in fibers

### Phase 7: Testing & Validation

1. Update `cache_bench` example
2. Add fiber tree statistics display
3. Verify O(changed) behavior
4. Performance benchmarks

---

## 10. API Changes

### 10.1 Render Trait

```rust
// Before
pub trait Render {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement;
}

// After
pub trait Render {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoDescriptor;
}
```

### 10.2 Element Builders

```rust
// Before: div() returns Div (element)
pub fn div() -> Div { ... }

// After: div() returns DivBuilder (descriptor builder)
pub fn div() -> DivBuilder { ... }

impl IntoDescriptor for DivBuilder {
    fn into_descriptor(self) -> AnyDescriptor {
        AnyDescriptor::Div(DivDescriptor {
            style: self.style,
            style_hash: hash(&self.style),
            key: self.key,
            children: self.children,
            interactivity: self.interactivity,
        })
    }
}
```

### 10.3 User Code (Minimal Changes)

```rust
// Before
fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
    div()
        .bg(rgb(0x1e1e2e))
        .child(text("Hello"))
}

// After (identical syntax, different return type)
fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoDescriptor {
    div()
        .bg(rgb(0x1e1e2e))
        .child(text("Hello"))
}
```

The user-facing API is nearly identical. The change is in what the framework does with the result.

---

## 11. Performance Analysis

### 11.1 Complexity Comparison

| Scenario | Current GPUI | Fiber Architecture |
|----------|--------------|-------------------|
| Nothing changed | O(n) | O(n) descriptors + O(1) layout/paint |
| One element changed | O(n) | O(n) descriptors + O(path) layout/paint |
| k elements changed | O(n) | O(n) descriptors + O(k) layout/paint |
| Full re-render | O(n) | O(n) |

### 11.2 Why O(n) Descriptors is OK

Descriptor creation is just struct allocation:
- ~100-200 bytes per descriptor
- 750 descriptors ≈ 150KB per frame
- At 120fps ≈ 18MB/s allocation

This is fast because:
1. No layout computation
2. No paint generation
3. No GPU interaction
4. Just memory writes

The expensive work (layout, paint) is O(changed), which is the goal.

### 11.3 Memory Overhead

Per fiber:
- Identity: ~32 bytes
- Cached layout: ~48 bytes
- Cached paint: ~24 bytes
- Tree structure: ~32 bytes
- Dirty flags: ~8 bytes
- **Total**: ~144 bytes per fiber

For 1000 elements: ~144KB persistent memory (acceptable)

### 11.4 Expected Results

**cache_bench with 375 grid cells:**

| Metric | Current | Fiber Architecture |
|--------|---------|-------------------|
| Debug FPS (static) | ~30 | ~120 |
| Layout calls/frame (static) | 375 | 0 |
| Paint calls/frame (static) | 375 | 0 (replay) |
| Layout calls/frame (FPS change) | 375 | ~3 |
| Paint calls/frame (FPS change) | 375 | ~3 |

---

## Appendix A: Comparison with React Fiber

| Concept | React | GPUI Fiber |
|---------|-------|------------|
| Render output | JSX elements | Descriptors |
| Persistent tree | Fiber tree | Fiber tree |
| Change detection | Props comparison | Hash comparison |
| Reconciliation | `reconcileChildren` | `reconcile_children` |
| Work scheduling | Lanes, priorities | Dirty flags |
| Commit phase | DOM updates | Layout + Paint |

GPUI's fiber architecture is simpler than React's because:
1. No concurrent rendering (single-threaded)
2. No suspense/transitions
3. No server components
4. Simpler dirty flag model

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Descriptor** | Lightweight data object describing UI structure |
| **Fiber** | Persistent node caching layout/paint results |
| **Reconciliation** | Diffing descriptors against fibers |
| **SUBTREE_DIRTY** | Flag indicating a descendant needs work |
| **Skip logic** | Returning cached results for clean subtrees |

---

## Appendix C: References

1. React Fiber Architecture: https://github.com/acdlite/react-fiber-architecture
2. React Reconciliation: https://react.dev/learn/preserving-and-resetting-state
3. Chromium Compositor: https://chromium.googlesource.com/chromium/src/+/master/docs/how_cc_works.md
4. Flutter Rendering: https://docs.flutter.dev/resources/architectural-overview#rendering
