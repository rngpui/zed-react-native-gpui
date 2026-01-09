//! A benchmark that demonstrates the primitive cache and tiled rendering effectiveness.
//!
//! Renders a grid of styled elements that remain static across frames.
//! After the first frame, all elements should hit the cache, showing
//! significant reduction in paint work.
//!
//! ## Grid Size Configurations
//!
//! **Small Grid (Default - RTT Caching):**
//! - GRID_COLS = 15, GRID_ROWS = 25 (375 cells, ~1875px height)
//! - Fits in 4096 texture at 2x scale
//! - Uses RTT (Render-to-Texture) caching for the grid-container
//!
//! **Large Grid (Tiled Rendering - when enabled):**
//! - GRID_COLS = 50, GRID_ROWS = 100 (5000 cells, ~7500px height)
//! - Exceeds texture limits, requires tiled rendering
//! - Content is rendered to fixed-size tiles for O(visible_tiles) compositing
//!
//! Displays FPS and cache hit/miss statistics in real-time.

use std::time::Instant;

use gpui::{
    Application, Bounds, Context, LayoutCacheStats, MeasureCacheStats, PrimitiveCacheStats, Render,
    SceneDirtyStats, TextShapeCacheStats, TitlebarOptions, Window, WindowBounds, WindowOptions,
    div, prelude::*, px, rgb, size,
};

const WINDOW_WIDTH: f32 = 1200.0;
const WINDOW_HEIGHT: f32 = 900.0;

// Grid configuration - adjust to test different element counts
//
// SMALL GRID (RTT mode - default):
// Uses RTT caching, must stay under 2048px per dimension to fit in 4096 texture at 2x scale
const GRID_COLS: usize = 15;
const GRID_ROWS: usize = 25; // ~1875px height, fits in 4096 texture at 2x

// LARGE GRID (Tiled mode):
// NOTE: Tiled rendering needs architectural changes to work with GPUI's prepaint/paint system.
// The current implementation can't paint children multiple times per frame.
// const GRID_COLS: usize = 50;
// const GRID_ROWS: usize = 100; // ~7500px height, would require tiled rendering

const CELL_SIZE: f32 = 70.0; // Must be >= 64px for RTT caching
const CELL_GAP: f32 = 5.0;

struct CacheBench {
    frame_times: Vec<f32>,
    last_frame: Instant,
    cache_stats: PrimitiveCacheStats,
    text_cache_stats: TextShapeCacheStats,
    measure_cache_stats: MeasureCacheStats,
    layout_cache_stats: LayoutCacheStats,
    scene_dirty_stats: SceneDirtyStats,
    frame_count: u64,
}

impl CacheBench {
    fn new(_window: &mut Window, _cx: &mut Context<Self>) -> Self {
        Self {
            frame_times: Vec::with_capacity(120),
            last_frame: Instant::now(),
            cache_stats: PrimitiveCacheStats::default(),
            text_cache_stats: TextShapeCacheStats::default(),
            measure_cache_stats: MeasureCacheStats::default(),
            layout_cache_stats: LayoutCacheStats::default(),
            scene_dirty_stats: SceneDirtyStats::default(),
            frame_count: 0,
        }
    }

    fn update_stats(&mut self, window: &mut Window) -> (f32, f32) {
        // Update frame timing
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame).as_secs_f32() * 1000.0;
        self.last_frame = now;

        self.frame_times.push(frame_time);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }

        let avg_frame_time: f32 =
            self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
        let fps = 1000.0 / avg_frame_time;

        // Get cache stats from the previous frame
        self.cache_stats = window.take_primitive_cache_stats();
        self.text_cache_stats = window.take_text_shape_cache_stats();
        self.measure_cache_stats = window.take_measure_cache_stats();
        self.layout_cache_stats = window.take_layout_cache_stats();
        self.scene_dirty_stats = window.scene_dirty_stats();
        self.frame_count += 1;

        (fps, avg_frame_time)
    }
}

impl Render for CacheBench {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        // Request continuous redraw
        window.request_animation_frame();

        let (fps, frame_time) = self.update_stats(window);
        let stats = self.cache_stats;
        let text_stats = self.text_cache_stats;
        let measure_stats = self.measure_cache_stats;
        let layout_stats = self.layout_cache_stats;
        let scene_stats = self.scene_dirty_stats;
        let frame_count = self.frame_count;

        // Calculate hit rates
        let total = stats.hits + stats.misses;
        let hit_rate = if total > 0 {
            (stats.hits as f32 / total as f32) * 100.0
        } else {
            0.0
        };

        let text_total = text_stats.hits + text_stats.misses;
        let text_hit_rate = if text_total > 0 {
            (text_stats.hits as f32 / text_total as f32) * 100.0
        } else {
            0.0
        };

        let measure_total = measure_stats.hits + measure_stats.misses;
        let measure_hit_rate = if measure_total > 0 {
            (measure_stats.hits as f32 / measure_total as f32) * 100.0
        } else {
            0.0
        };

        let layout_total = layout_stats.hits + layout_stats.misses;
        let layout_hit_rate = if layout_total > 0 {
            (layout_stats.hits as f32 / layout_total as f32) * 100.0
        } else {
            0.0
        };

        div()
            .size_full()
            .bg(rgb(0x1e1e2e))
            .flex()
            .flex_col()
            .child(
                // Stats overlay
                div()
                    .p_4()
                    .bg(rgb(0x313244))
                    .m_4()
                    .rounded_lg()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_color(rgb(0xcdd6f4))
                            .text_xl()
                            .child("Cache Benchmark"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_4()
                            .child(stat_box("FPS", format!("{:.1}", fps), rgb(0xa6e3a1)))
                            .child(stat_box("Frame", format!("{:.2}ms", frame_time), rgb(0x89b4fa)))
                            .child(stat_box("Frame #", format!("{}", frame_count), rgb(0xf5c2e7))),
                    )
                    .child(
                        div()
                            .text_color(rgb(0x9399b2))
                            .text_sm()
                            .mt_2()
                            .child("Primitive Cache"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_4()
                            .child(stat_box("Hits", format!("{}", stats.hits), rgb(0xa6e3a1)))
                            .child(stat_box("Misses", format!("{}", stats.misses), rgb(0xf38ba8)))
                            .child(stat_box("Hit Rate", format!("{:.1}%", hit_rate), rgb(0xf9e2af)))
                            .child(stat_box("Inserts", format!("{}", stats.inserts), rgb(0x89dceb))),
                    )
                    .child(
                        div()
                            .text_color(rgb(0x9399b2))
                            .text_sm()
                            .mt_2()
                            .child("Text Shape Cache"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_4()
                            .child(stat_box("Hits", format!("{}", text_stats.hits), rgb(0xa6e3a1)))
                            .child(stat_box("Misses", format!("{}", text_stats.misses), rgb(0xf38ba8)))
                            .child(stat_box("Hit Rate", format!("{:.1}%", text_hit_rate), rgb(0xf9e2af)))
                            .child(stat_box("Inserts", format!("{}", text_stats.inserts), rgb(0x89dceb))),
                    )
                    .child(
                        div()
                            .text_color(rgb(0x9399b2))
                            .text_sm()
                            .mt_2()
                            .child("Measure Cache"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_4()
                            .child(stat_box("Hits", format!("{}", measure_stats.hits), rgb(0xa6e3a1)))
                            .child(stat_box("Misses", format!("{}", measure_stats.misses), rgb(0xf38ba8)))
                            .child(stat_box("Hit Rate", format!("{:.1}%", measure_hit_rate), rgb(0xf9e2af)))
                            .child(stat_box("Inserts", format!("{}", measure_stats.inserts), rgb(0x89dceb))),
                    )
                    .child(
                        div()
                            .text_color(rgb(0x9399b2))
                            .text_sm()
                            .mt_2()
                            .child("Layout Cache"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_4()
                            .child(stat_box("Hits", format!("{}", layout_stats.hits), rgb(0xa6e3a1)))
                            .child(stat_box("Misses", format!("{}", layout_stats.misses), rgb(0xf38ba8)))
                            .child(stat_box("Hit Rate", format!("{:.1}%", layout_hit_rate), rgb(0xf9e2af)))
                            .child(stat_box("Pruned", format!("{}", layout_stats.pruned), rgb(0x89dceb))),
                    )
                    .child(
                        div()
                            .text_color(rgb(0x9399b2))
                            .text_sm()
                            .mt_2()
                            .child("Scene Dirty Stats"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_4()
                            .child(stat_box("Total", format!("{}", scene_stats.total_primitives), rgb(0xcdd6f4)))
                            .child(stat_box("Dirty", format!("{}", scene_stats.dirty_primitives), rgb(0xf38ba8)))
                            .child(stat_box("Hit Rate", format!("{:.1}%", scene_stats.cache_hit_rate()), rgb(0xf9e2af)))
                            .child(stat_box(
                                "Cached",
                                if scene_stats.is_fully_cached() { "Yes" } else { "No" }.to_string(),
                                if scene_stats.is_fully_cached() { rgb(0xa6e3a1) } else { rgb(0xf38ba8) },
                            )),
                    ),
            )
            .child(
                // Scrollable grid of styled elements - these should all cache after frame 1
                //
                // For SMALL grids: RTT caching is used for grid-container
                // For LARGE grids: Tiled rendering is used (when enabled) for the scroll-container
                //
                // Note: scroll-container has ID for scroll functionality. When tiled rendering
                // is enabled and content exceeds viewport by >256px, it will use 512px tiles
                // for O(visible_tiles) compositing instead of RTT caching.
                div()
                    .id("scroll-container")
                    .flex_1()
                    .p_4()
                    .overflow_y_scroll()
                    .child(
                        div()
                            .id("grid-container") // ID enables RTT caching for the grid
                            .flex()
                            .flex_wrap()
                            .gap(px(CELL_GAP))
                            .children((0..GRID_ROWS * GRID_COLS).map(|i| {
                                let row = i / GRID_COLS;
                                let col = i % GRID_COLS;

                                // Create varied styles to exercise different code paths
                                let hue = ((row * GRID_COLS + col) as f32 / (GRID_ROWS * GRID_COLS) as f32) * 360.0;
                                let bg_color = hsl_to_rgb(hue, 0.6, 0.3);
                                let border_color = hsl_to_rgb(hue, 0.8, 0.5);

                                // Note: Cells intentionally have no ID to avoid nested RTT captures.
                                // The grid-container handles RTT for the entire grid.
                                div()
                                    .w(px(CELL_SIZE))
                                    .h(px(CELL_SIZE))
                                    .bg(bg_color)
                                    .border_2()
                                    .border_color(border_color)
                                    .rounded_md()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(
                                        div()
                                            .text_color(rgb(0xffffff))
                                            .text_xs()
                                            .child(format!("{}", i)),
                                    )
                            })),
                    ),
            )
    }
}

fn stat_box(label: &str, value: String, color: gpui::Rgba) -> impl IntoElement {
    div()
        .bg(rgb(0x45475a))
        .rounded_md()
        .px_3()
        .py_2()
        .flex()
        .flex_col()
        .items_center()
        .child(
            div()
                .text_color(rgb(0x9399b2))
                .text_xs()
                .child(label.to_string()),
        )
        .child(
            div()
                .text_color(color)
                .text_lg()
                .font_weight(gpui::FontWeight::BOLD)
                .child(value),
        )
}

// Simple HSL to RGB conversion
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> gpui::Rgba {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    gpui::rgba((((r + m) * 255.0) as u32) << 24 | (((g + m) * 255.0) as u32) << 16 | (((b + m) * 255.0) as u32) << 8 | 0xFF)
}

fn main() {
    env_logger::init();
    Application::new().run(|cx| {
        cx.open_window(
            WindowOptions {
                titlebar: Some(TitlebarOptions {
                    title: Some("Primitive Cache Benchmark".into()),
                    ..Default::default()
                }),
                focus: true,
                window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                    None,
                    size(px(WINDOW_WIDTH), px(WINDOW_HEIGHT)),
                    cx,
                ))),
                ..Default::default()
            },
            |window, cx| cx.new(|cx| CacheBench::new(window, cx)),
        )
        .unwrap();
        cx.activate(true);
    });
}
