# Retained-Mode GPUI: Design & Architecture

## Executive Summary

This document proposes evolving GPUI from an **immediate-mode** rendering model (rebuild everything every frame) to a **retained-artifact** model (persist paint/layout artifacts, only rebuild what changed).

**The Problem**: GPUI currently does O(n) work per frame regardless of what changed, resulting in ~30fps in debug mode for a static 375-element grid on an M2 Ultra.

**The Solution**: Retain *artifacts* (layout nodes, display list entries, tiles, textures) across frames, track changes through entity invalidation, and only re-render/re-layout/re-paint elements that actually changed.

**The Goal**: O(changed) work per frame, enabling 120fps in debug mode for typical UI workloads.

**Key Insight**: GPUI already has significant retained-mode infrastructure. The path forward is to unify and extend these existing systems, not introduce a parallel retained element tree.

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Design Goals](#2-design-goals)
3. [Existing Infrastructure](#3-existing-infrastructure)
4. [Architecture Overview](#4-architecture-overview)
5. [Core Mechanisms](#5-core-mechanisms)
6. [Incremental Layout](#6-incremental-layout)
7. [Incremental Paint](#7-incremental-paint)
8. [Tile & Raster Cache Unification](#8-tile--raster-cache-unification)
9. [Implementation Plan](#9-implementation-plan)
10. [Performance Expectations](#10-performance-expectations)
11. [Open Questions](#11-open-questions)

---

## 1. Problem Analysis

### 1.1 Current Architecture

GPUI uses an immediate-mode-inspired architecture where the element tree is rebuilt from scratch every frame:

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
- **Grid**: 375 cells (15x25), completely static
- **Stats overlay**: ~30 elements, only text values change
- **Total**: ~405 elements

**Current behavior**:
```
Every frame:
├── render() rebuilds 375 grid divs (unchanged!)
├── render() rebuilds 30 stat elements (only values change)
├── request_layout() x 405 elements
├── compute_layout() walks all 405 nodes
├── prepaint() x 405 elements
├── paint() x 405 elements
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

### 1.5 Why Not Retain Element Instances?

The original proposal suggested retaining `AnyElement` instances across frames. This is **not feasible** with current GPUI:

- `AnyElement` is `ArenaBox<dyn ElementObject>` (`src/element.rs:663`)
- Elements are allocated in `ELEMENT_ARENA` (`src/window.rs:227`)
- The arena is cleared every frame (`src/arena.rs:179`)
- Elements and callbacks are intentionally dropped each frame (`src/element.rs:13`)

Retaining element instances would require redesigning allocation, lifetimes, and update semantics — a massive rewrite with unclear benefits over the artifact-based approach.

---

## 2. Design Goals

### 2.1 Primary Goals

1. **O(changed) frame complexity**: Work done should be proportional to what changed, not total elements
2. **120fps in debug mode**: For typical UI workloads with few changes per frame
3. **Build on existing infrastructure**: Extend what GPUI already has, don't introduce parallel systems
4. **Backward compatible**: Existing GPUI code should work without modification

### 2.2 Secondary Goals

1. **Minimal memory overhead**: Retained artifacts should not significantly increase memory usage
2. **Predictable performance**: No surprise O(n²) behaviors
3. **Debuggability**: Clear tools to understand why re-renders happen
4. **Incremental adoption**: Benefits increase as more infrastructure is connected

### 2.3 Non-Goals

1. **Retaining element instances**: Elements remain ephemeral (arena-allocated, per-frame)
2. **Generic reconciliation/diff**: No framework-level element diffing
3. **Render hash from struct fields**: Too expensive, unclear benefits over entity notification

---

## 3. Existing Infrastructure

GPUI already contains significant retained-mode building blocks. The strategy is to unify and extend these, not replace them.

### 3.1 Runtime Dependency Tracking & Invalidation

**Already implemented:**
- `App::detect_accessed_entities` (`src/app.rs:854`)
- `App::record_entities_accessed` (`src/app.rs:868`)
- `App::notify` (`src/app.rs:2165`)
- `Window::mark_view_dirty` (`src/window.rs:1502`)

Views already track which entities they read during render. When an entity is updated via `notify()`, dependent views are marked dirty.

### 3.2 View Subtree Reuse (AnyView::cached)

**Already implemented:**
- `AnyView::cached` reuses prepaint/paint ranges (`src/view.rs:99`, `src/view.rs:202`)
- Cache lookup/application (`src/window.rs:2650`, `src/window.rs:2711`)

This is extremely close to "retained mode" — it caches *artifacts* (paint ranges), not element instances. Currently opt-in via `AnyView::cached(style)`.

### 3.3 Layout Node Reuse

**Already implemented:**
- `TaffyLayoutEngine.element_to_node: FxHashMap<GlobalElementId, NodeId>` (`src/taffy.rs:49`)

Layout nodes persist across frames for elements with explicit IDs. This avoids recreating Taffy nodes.

### 3.4 Universal Element Identity & Per-Element Paint Caching

**Already implemented:**
- `ComputedElementId` (`src/display_list.rs:325`)
- `ElementEntry` with cached paint state (`src/display_list.rs:383`)
- Phase 20 per-element caching in Window (`src/window.rs:3408`)

Every element gets a stable identity. Paint artifacts can be cached and reused per-element.

### 3.5 SubtreeCache / Render-to-Texture

**Already implemented:**
- Subtree cache lookup/insert (`src/window.rs:2759`, `src/window.rs:2854`)
- RTT capture for scroll container optimization

### 3.6 DisplayList & Tile System

**Already implemented:**
- `DisplayList` with spatial index (`src/display_list.rs`)
- `DisplayList::invalidate_region` for dirty rect tracking (`src/display_list.rs:917`)
- Tile cache with generation-based invalidation (`src/platform/mac/tile_cache.rs`)

---

## 4. Architecture Overview

### 4.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RETAINED ARTIFACT MODEL                          │
│                                                                      │
│  Elements: Ephemeral (arena-allocated, rebuilt when dirty)          │
│  Artifacts: Retained (layout nodes, paint entries, tiles, textures) │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Invalidation Flow                                            │   │
│  │                                                               │   │
│  │  Entity.notify() ──► mark_view_dirty() ──► NEEDS_RENDER      │   │
│  │                                                               │   │
│  │  View dirty ──► rebuild element subtree ──► NEEDS_LAYOUT     │   │
│  │                                                               │   │
│  │  Layout changed ──► NEEDS_PAINT ──► invalidate_region()      │   │
│  │                                                               │   │
│  │  Dirty region ──► tile invalidation ──► re-rasterize tiles   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Frame Cycle                                                  │   │
│  │                                                               │   │
│  │  1. Check view dirty flags (O(dirty views))                  │   │
│  │  2. Rebuild dirty view subtrees only                         │   │
│  │  3. Layout with cached nodes (O(dirty + ancestors))          │   │
│  │  4. Paint with artifact reuse (O(dirty elements))            │   │
│  │  5. Rasterize dirty tiles only (O(dirty tiles))              │   │
│  │  6. Composite (unchanged)                                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Key Principles

| Principle | Description |
|-----------|-------------|
| **Artifacts, not instances** | Retain layout nodes, paint entries, tiles — not element objects |
| **Entity-driven invalidation** | `notify()` is the primary signal, not hashing state |
| **Unified identity** | `ComputedElementId` is the key for all caches |
| **Dirty rect propagation** | Changes generate regions that invalidate tiles |
| **Subtree skip** | Clean subtrees skip render/layout/paint entirely |

### 4.3 What Changes vs. What's Retained

| Component | Lifetime | Cache Key |
|-----------|----------|-----------|
| `AnyElement` | One frame (arena) | N/A |
| View render output | One frame | N/A |
| Layout `NodeId` | Persistent | `ComputedElementId` |
| `ElementEntry` (paint cache) | Persistent | `ComputedElementId` |
| `DisplayList` items | Per-layer, swapped | Element ranges |
| Tile textures | LRU cached | `(LayerId, TileCoord, generation)` |
| RTT textures | LRU cached | `(ElementId, bounds, clip)` |

---

## 5. Core Mechanisms

### 5.1 Dirty Flags

```rust
bitflags! {
    /// Flags indicating what work is needed for a view/element.
    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    pub struct DirtyFlags: u8 {
        /// View's entity was notified, need to call render()
        const NEEDS_RENDER = 0b0000_0001;

        /// Layout inputs changed, need to recompute layout
        const NEEDS_LAYOUT = 0b0000_0010;

        /// Visual appearance changed, need to repaint
        const NEEDS_PAINT  = 0b0000_0100;

        /// Some descendant has dirty flags set (enables subtree skip)
        const SUBTREE_DIRTY = 0b0000_1000;
    }
}
```

### 5.2 Invalidation via Entity Notification

The existing `notify()` system is the primary invalidation mechanism:

```rust
// When an entity updates:
entity.update(cx, |state, cx| {
    state.value = new_value;
    cx.notify();  // Marks all dependent views as NEEDS_RENDER
});

// Framework automatically:
// 1. Finds views that read this entity (via detect_accessed_entities)
// 2. Marks them NEEDS_RENDER
// 3. Propagates SUBTREE_DIRTY to ancestors
```

**No render hash needed.** The entity notification system already provides O(1) invalidation. Hashing entire view structs would reintroduce O(n) work and require `Hash` bounds on all fields.

### 5.3 Subtree Dirty Propagation

```rust
impl Window {
    fn mark_view_dirty(&mut self, view_id: EntityId) {
        // Mark this view
        self.dirty_views.insert(view_id, DirtyFlags::NEEDS_RENDER);

        // Propagate SUBTREE_DIRTY up to root
        // This enables O(1) skip of clean subtrees
        let mut current = self.view_parent(view_id);
        while let Some(parent_id) = current {
            let flags = self.dirty_views.entry(parent_id).or_default();
            if flags.contains(DirtyFlags::SUBTREE_DIRTY) {
                break;  // Already propagated
            }
            *flags |= DirtyFlags::SUBTREE_DIRTY;
            current = self.view_parent(parent_id);
        }
    }
}
```

### 5.4 Frame Cycle with Dirty Checking

```rust
impl Window {
    pub fn draw(&mut self, cx: &mut App) {
        // Phase 1: Render dirty views only
        // Clean views skip render() entirely
        self.render_dirty_views(cx);

        // Phase 2: Layout with node reuse
        // Clean subtrees reuse cached layout
        self.layout_incrementally(cx);

        // Phase 3: Paint with artifact reuse
        // Clean elements copy cached display items
        self.paint_incrementally(cx);

        // Phase 4: Rasterize dirty tiles only
        // Clean tiles are not re-rendered
        self.rasterize_dirty_tiles(cx);

        // Phase 5: Composite (unchanged)
        self.composite(cx);

        // Clear per-frame dirty flags
        self.clear_dirty_flags();
    }
}
```

---

## 6. Incremental Layout

### 6.1 Extend Layout Node Reuse

Currently, layout nodes are reused only for elements with explicit `GlobalElementId`. Extend this to all elements using `ComputedElementId`:

```rust
// Current (src/taffy.rs:49):
element_to_node: FxHashMap<GlobalElementId, NodeId>

// Extended:
element_to_node: FxHashMap<ComputedElementId, NodeId>
```

This gives every element stable layout node identity, not just `.id()` elements.

### 6.2 Layout Cache Validity

```rust
struct LayoutCache {
    node_id: NodeId,
    constraints: Size<AvailableSpace>,
    result: Size<Pixels>,
}

impl Window {
    fn layout_element(&mut self, element_id: ComputedElementId, constraints: Size<AvailableSpace>) -> Size<Pixels> {
        // Check if cached layout is still valid
        if let Some(cache) = self.layout_cache.get(&element_id) {
            if cache.constraints == constraints && !self.is_layout_dirty(element_id) {
                return cache.result;  // O(1) cache hit
            }
        }

        // Actually compute layout
        let result = self.compute_layout(element_id, constraints);

        // Cache for next frame
        self.layout_cache.insert(element_id, LayoutCache {
            node_id: self.get_or_create_node(element_id),
            constraints,
            result,
        });

        result
    }
}
```

### 6.3 Let Taffy Handle Incremental Layout

Don't hand-roll incremental layout walking. Taffy already supports this:

```rust
// Taffy's compute_layout is already incremental if:
// 1. Node identity is stable (same NodeId across frames)
// 2. Style hasn't changed
// 3. Children haven't changed

// Our job is just to ensure stable identity via ComputedElementId
```

---

## 7. Incremental Paint

### 7.1 Extend AnyView::cached to be Default

Currently `AnyView::cached(style)` is opt-in. Make it the default by storing the view's last root style in `AnyViewState`:

```rust
// Current (src/view.rs):
pub struct AnyViewState {
    root_style: Option<Style>,  // Only set if cached() was called
    // ...
}

// Extended:
pub struct AnyViewState {
    root_style: Style,           // Always stored
    cached_prepaint: Option<PrepaintState>,
    cached_paint_range: Option<Range<usize>>,
    // ...
}
```

This allows caching to be automatic without requiring `AnyView::cached(style)`.

### 7.2 Offset-Tolerant Reuse

Currently cached views require exact origin match. Allow reuse when size matches but origin differs:

```rust
impl Window {
    fn can_reuse_cached_paint(&self, view_id: EntityId, new_bounds: Bounds<Pixels>) -> bool {
        let cache = self.view_paint_cache.get(&view_id)?;

        // Size must match exactly
        if cache.bounds.size != new_bounds.size {
            return false;
        }

        // Origin can differ - we'll apply offset when copying items
        true
    }

    fn copy_cached_paint_with_offset(&mut self, view_id: EntityId, new_origin: Point<Pixels>) {
        let cache = self.view_paint_cache.get(&view_id).unwrap();
        let offset = new_origin - cache.bounds.origin;

        // Copy items with offset applied
        for item in &cache.items {
            self.display_list.push_item_with_offset(item.clone(), offset);
        }
    }
}
```

### 7.3 Subtree Paint Skip

Track full subtree item ranges. If a subtree is unchanged and no descendant is invalidated, skip calling child `paint()` entirely:

```rust
impl Window {
    fn paint_view(&mut self, view_id: EntityId, cx: &mut Context) {
        let flags = self.dirty_flags(view_id);

        // Fast path: entire subtree is clean
        if !flags.intersects(DirtyFlags::NEEDS_PAINT | DirtyFlags::SUBTREE_DIRTY) {
            // Copy cached paint artifacts, skip paint() call
            self.copy_cached_subtree_paint(view_id);
            return;
        }

        if flags.contains(DirtyFlags::NEEDS_PAINT) {
            // Actually paint this view
            self.paint_view_fresh(view_id, cx);
        } else {
            // This view is clean but has dirty descendants
            // Paint our cached content, recurse for children
            self.copy_cached_view_paint(view_id);
            self.paint_dirty_children(view_id, cx);
        }
    }
}
```

### 7.4 Dirty Region Generation

When paint changes, generate dirty regions from bounds deltas:

```rust
impl Window {
    fn record_paint_change(&mut self, element_id: ComputedElementId, old_bounds: Bounds<Pixels>, new_bounds: Bounds<Pixels>) {
        // Invalidate old location
        if old_bounds != Bounds::default() {
            self.display_list.invalidate_region(old_bounds);
        }

        // Invalidate new location
        self.display_list.invalidate_region(new_bounds);
    }
}
```

---

## 8. Tile & Raster Cache Unification

### 8.1 Unified Raster Cache Abstraction

RTT (render-to-texture) and tiling are both "cached rasterization keyed by element identity + context":

| Cache Type | Key | Value |
|------------|-----|-------|
| RTT | `(ElementId, bounds, clip)` | Single texture |
| Tile | `(LayerId, TileCoord, generation)` | Texture grid |

Unify under a single abstraction:

```rust
pub struct RasterCache {
    /// Single-texture cache for small elements (RTT)
    element_textures: FxHashMap<RasterKey, CachedTexture>,

    /// Tiled texture cache for large scroll containers
    tile_textures: FxHashMap<TileKey, CachedTexture>,
}

#[derive(Hash, Eq, PartialEq)]
pub struct RasterKey {
    element_id: ComputedElementId,
    bounds: Bounds<DevicePixels>,
    clip_hash: u64,
    generation: u64,
}
```

### 8.2 Dirty Region → Tile Invalidation

Wire dirty regions from DisplayList into tile cache:

```rust
impl MetalRenderer {
    fn rasterize_display_list_tiles(&mut self, scene: &Scene, scale_factor: f32) {
        for (container_id, display_list) in &scene.display_lists {
            // Get dirty regions from display list
            let dirty_regions = display_list.dirty_regions();

            // Invalidate affected tiles
            self.tile_cache.invalidate_tiles_for_regions(
                container_id,
                &dirty_regions,
                scale_factor,
            );

            // Clear dirty regions after processing
            display_list.clear_dirty_regions();

            // Only rasterize invalidated tiles
            // ... existing tile rasterization loop
        }
    }
}
```

### 8.3 Avoid Scene-Index-Based Caches

SubtreeCache currently stores Scene indices (`primitive_start`, `primitive_end`). These are brittle once root rendering is DisplayList-driven.

**Migration**: Key caches by `ComputedElementId` + context, not Scene indices. The Phase 20 `ElementEntry` system already does this correctly.

---

## 9. Implementation Plan

### Phase 1: Make View Caching Default (High Impact, Low Risk)

**Goal**: `AnyView::cached` behavior without explicit opt-in.

**Tasks**:
- [ ] Store view's last root style in `AnyViewState` automatically
- [ ] Check cache validity on every view paint
- [ ] Reuse cached prepaint/paint ranges when valid
- [ ] Add offset-tolerant reuse (same size, different origin)

**Files**: `src/view.rs`, `src/window.rs`

**Validation**: `cache_bench` shows reduced paint calls for static grid

### Phase 2: Extend Layout Identity (Medium Impact, Low Risk)

**Goal**: All elements get stable layout node identity, not just `.id()` elements.

**Tasks**:
- [ ] Change `TaffyLayoutEngine.element_to_node` key from `GlobalElementId` to `ComputedElementId`
- [ ] Ensure `ComputedElementId` is computed before layout
- [ ] Verify layout cache hits for elements without explicit IDs

**Files**: `src/taffy.rs`, `src/window.rs`

**Validation**: Layout node count stays stable across frames for static UI

### Phase 3: Subtree Paint Skip (High Impact, Medium Risk)

**Goal**: Clean subtrees skip `paint()` call entirely, copy cached artifacts.

**Tasks**:
- [ ] Track subtree paint ranges in `ElementEntry`
- [ ] Implement `SUBTREE_DIRTY` flag propagation
- [ ] Copy cached subtree items when clean
- [ ] Handle offset application for moved subtrees

**Files**: `src/window.rs`, `src/display_list.rs`

**Validation**: `cache_bench` shows near-zero paint work for static grid

### Phase 4: Dirty Region → Tile Invalidation (Medium Impact, Medium Risk)

**Goal**: Only re-rasterize tiles that intersect dirty regions.

**Tasks**:
- [ ] Wire `DisplayList::dirty_regions()` into tile rasterization
- [ ] Implement `TileCache::invalidate_tiles_for_regions()`
- [ ] Clear dirty regions after tile invalidation
- [ ] Track per-tile invalidation in stats

**Files**: `src/display_list.rs`, `src/platform/mac/metal_renderer.rs`, `src/platform/mac/tile_cache.rs`

**Validation**: Changing one cell in grid re-rasterizes only affected tiles

### Phase 5: Unify Raster Caches (Low Impact, High Value Long-Term)

**Goal**: Single abstraction for RTT and tile caching.

**Tasks**:
- [ ] Define `RasterCache` abstraction
- [ ] Migrate SubtreeCache to use `ComputedElementId` keys
- [ ] Migrate tile cache to unified interface
- [ ] Remove Scene-index-based cache keys

**Files**: `src/window.rs`, `src/scene.rs`, `src/platform/mac/tile_cache.rs`

**Validation**: RTT and tiling share code paths, stats unified

### Phase 6: Instrumentation & Validation

**Goal**: Prove the system is truly O(changed).

**Tasks**:
- [ ] Add counters: views rendered/skipped, layout nodes reused, elements cache-hit
- [ ] Add counters: tiles re-rasterized, bytes uploaded to GPU
- [ ] Create benchmark suite: static UI, single-element change, bulk change
- [ ] Document expected performance characteristics

**Files**: `src/window.rs`, `examples/cache_bench.rs`

**Validation**: Counters show O(changed) behavior in all benchmarks

---

## 10. Performance Expectations

### 10.1 Theoretical Complexity

| Scenario | Current | Retained Artifacts |
|----------|---------|-------------------|
| Nothing changed | O(n) | O(1) |
| One element changed | O(n) | O(1) |
| k elements changed | O(n) | O(k) |
| All elements changed | O(n) | O(n) |
| Scroll (with compositor) | O(visible) | O(new tiles) |

### 10.2 cache_bench Predictions

**Current** (375 grid cells + 30 stats elements):
- Every frame: O(405) work
- Debug mode: ~30fps

**Phase 1** (view caching default):
- Frame where nothing changes: O(30) stats elements + O(1) grid check
- Speedup: ~10x
- Debug mode prediction: ~100fps

**Phase 3** (subtree paint skip):
- Frame where nothing changes: O(1) dirty check
- Debug mode prediction: 120fps

### 10.3 Memory Overhead

Per cached view:
- `PrepaintState`: ~100 bytes
- Paint range: ~16 bytes
- Layout cache: ~64 bytes
- **Total**: ~180 bytes per view

For 100 views: ~18 KB additional memory (negligible)

---

## 11. Open Questions

### 11.1 Animation Integration

**Question**: How do animations work with retained artifacts?

**Approach**: Animated elements mark themselves dirty each frame via `cx.notify()`. The animation system already does this. No special handling needed — animations are just frequent invalidations.

### 11.2 Context/Theme Changes

**Question**: How do theme changes invalidate dependent subtrees?

**Approach**: Theme is an entity. Views that read theme during render are automatically tracked as dependents. Theme change → `notify()` → all theme-dependent views marked dirty.

### 11.3 Event Handler Identity

**Question**: Event handler closures change identity every render. Does this cause issues?

**Approach**: Event handlers aren't compared for equality. They're simply replaced when the element is rebuilt. Since elements are ephemeral anyway, this isn't a problem.

### 11.4 Layout Dependencies

**Question**: How to handle layout dependencies where sibling size affects my size?

**Approach**: Taffy handles this internally. When any child changes, parent layout is recomputed, which may affect siblings. The `NEEDS_LAYOUT` flag propagates appropriately.

### 11.5 Memory Management

**Question**: When are cached artifacts garbage collected?

**Approach**:
- Layout nodes: Removed when element no longer appears in tree
- Paint cache: LRU eviction based on element activity
- Tiles: Existing LRU eviction in tile cache

---

## Appendix A: Relationship to Compositor Thread

The retained artifact model and compositor thread are complementary:

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│      MAIN THREAD            │    │    COMPOSITOR THREAD        │
│                             │    │                             │
│  Retained artifacts reduce  │───►│  Receives tile updates      │
│  work to O(changed)         │    │  Composites at 120fps       │
│  DisplayList → Tiles        │    │  Handles scroll offset      │
│                             │    │                             │
└─────────────────────────────┘    └─────────────────────────────┘
```

- **Compositor thread**: Scroll at 120fps regardless of main thread work
- **Retained artifacts**: Main thread does O(changed) work

Together: 120fps everything.

---

## Appendix B: What We're NOT Doing

| Approach | Why Not |
|----------|---------|
| Retain `AnyElement` instances | Arena allocation clears every frame; would require massive rewrite |
| Render hash from struct fields | O(n) hashing, requires `Hash` bounds, unclear benefit over `notify()` |
| Generic element reconciliation | No framework-level diffing; elements are ephemeral |
| Separate retained tree | Duplicates existing infrastructure; artifacts are the right abstraction |

---

## Appendix C: References

1. Chromium Compositor: https://chromium.googlesource.com/chromium/src/+/master/docs/how_cc_works.md
2. React Fiber Architecture: https://github.com/acdlite/react-fiber-architecture
3. Flutter Rendering: https://docs.flutter.dev/resources/architectural-overview#rendering
4. GPUI Phase 20 (per-element caching): `src/display_list.rs`, `src/window.rs`
