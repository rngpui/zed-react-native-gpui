use std::collections::VecDeque;
use std::env;
use std::time::Instant;

use gpui::{
    App, Application, Bounds, Context, ElementId, Entity, Window, WindowBounds, WindowOptions,
    deferred, div, prelude::*, px, rgb, size,
};

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

const GRID_SIZE: usize = 50;
const FRAME_HISTORY: usize = 60;

struct FpsCounter {
    times: VecDeque<Instant>,
    fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            times: VecDeque::with_capacity(FRAME_HISTORY + 1),
            fps: 0.0,
        }
    }

    fn record(&mut self) {
        let now = Instant::now();
        self.times.push_back(now);

        if self.times.len() > FRAME_HISTORY {
            self.times.pop_front();
        }

        if self.times.len() >= 2 {
            if let Some(oldest) = self.times.front() {
                let elapsed = now.duration_since(*oldest).as_secs_f64();
                self.fps = (self.times.len() - 1) as f64 / elapsed;
            }
        }
    }
}

struct GridBench {
    render_fps: FpsCounter,
    frame_fps: FpsCounter,
    row_count: usize,
    col_count: usize,
    enable_hover: bool,
    enable_click: bool,
    step_size: usize,
}

impl GridBench {
    fn new() -> Self {
        Self {
            render_fps: FpsCounter::new(),
            frame_fps: FpsCounter::new(),
            row_count: env_usize("GRID_BENCH_ROWS", GRID_SIZE),
            col_count: env_usize("GRID_BENCH_COLS", GRID_SIZE),
            enable_hover: env_bool("GRID_BENCH_HOVER", true),
            enable_click: env_bool("GRID_BENCH_CLICK", true),
            step_size: env_usize("GRID_BENCH_STEP", 1),
        }
    }

    fn add_row(&mut self) {
        self.row_count += self.step_size;
    }

    fn remove_row(&mut self) {
        self.row_count = self.row_count.saturating_sub(self.step_size).max(1);
    }

    fn add_col(&mut self) {
        self.col_count += self.step_size;
    }

    fn remove_col(&mut self) {
        self.col_count = self.col_count.saturating_sub(self.step_size).max(1);
    }

    fn schedule_frame_callback(this: Entity<Self>, window: &mut Window) {
        let this_weak = this.downgrade();
        window.on_next_frame(move |window, cx| {
            if let Some(this) = this_weak.upgrade() {
                this.update(cx, |bench, _cx| {
                    bench.frame_fps.record();
                });
                Self::schedule_frame_callback(this, window);
            }
        });
    }
}

impl Render for GridBench {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        window.request_animation_frame();
        self.render_fps.record();

        let row_count = self.row_count;
        let col_count = self.col_count;
        let total_cells = row_count * col_count;
        let enable_hover = self.enable_hover;
        let enable_click = self.enable_click;

        div()
            .size_full()
            .bg(rgb(0x1e1e1e))
            .child(deferred(
                div()
                    .absolute()
                    .top_2()
                    .left_2()
                    .px_3()
                    .py_2()
                    .bg(gpui::black().opacity(0.7))
                    .rounded_md()
                    .text_sm()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_color(rgb(0x00ff00))
                                    .child(format!("Render: {:.1} FPS", self.render_fps.fps)),
                            )
                            .child(
                                div()
                                    .text_color(rgb(0xffff00))
                                    .child(format!("Frame:  {:.1} FPS", self.frame_fps.fps)),
                            )
                            .child(
                                div()
                                    .text_color(rgb(0xaaaaaa))
                                    .child(format!("Grid: {}x{} ({} cells)", row_count, col_count, total_cells)),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_1()
                                    .child(div().text_color(rgb(0x888888)).child("Rows"))
                                    .child(
                                        div()
                                            .flex()
                                            .gap_1()
                                            .child(self.control_button("-", cx.listener(|this, _, _, _| this.remove_row())))
                                            .child(self.control_button("+", cx.listener(|this, _, _, _| this.add_row()))),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_1()
                                    .child(div().text_color(rgb(0x888888)).child("Cols"))
                                    .child(
                                        div()
                                            .flex()
                                            .gap_1()
                                            .child(self.control_button("-", cx.listener(|this, _, _, _| this.remove_col())))
                                            .child(self.control_button("+", cx.listener(|this, _, _, _| this.add_col()))),
                                    ),
                            ),
                    ),
            ))
            .child(
                div()
                    .size_full()
                    .id("scroll")
                    .overflow_scroll()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .p_4()
                            .gap_1()
                            .children((0..row_count).map(|row| {
                                div()
                                    .flex()
                                    .gap_1()
                                    .children((0..col_count).map(|col| {
                                        let cell_num = row * col_count + col;
                                        let hue = (cell_num as f32
                                            / (row_count * col_count).max(1) as f32
                                            * 360.0)
                                            as u32;
                                        let color = hsv_to_rgb(hue, 70, 60);
                                        let hover_color = hsv_to_rgb(hue, 80, 80);
                                        div()
                                            .id(ElementId::NamedInteger("cell".into(), cell_num as u64))
                                            .size(px(32.0))
                                            .rounded_sm()
                                            .bg(color)
                                            .when(enable_hover, |this| {
                                                this.hover(|style| style.bg(hover_color).border_1().border_color(gpui::white()))
                                            })
                                            .flex()
                                            .items_center()
                                            .justify_center()
                                            .text_xs()
                                            .text_color(gpui::white())
                                            .child(format!("{}", cell_num))
                                            .when(enable_click, |this| {
                                                this.on_click(move |_event, _window, _cx| {
                                                    log::info!("Clicked cell {}", cell_num);
                                                })
                                            })
                                    }))
                            })),
                    ),
            )
    }
}

impl GridBench {
    fn control_button(
        &self,
        label: &'static str,
        on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
    ) -> impl IntoElement {
        div()
            .id(label)
            .px_2()
            .py_1()
            .bg(rgb(0x444444))
            .hover(|style| style.bg(rgb(0x555555)))
            .active(|style| style.bg(rgb(0x333333)))
            .rounded_sm()
            .cursor_pointer()
            .text_color(gpui::white())
            .child(label)
            .on_click(on_click)
    }
}

fn hsv_to_rgb(h: u32, s: u32, v: u32) -> gpui::Hsla {
    gpui::hsla(h as f32 / 360.0, s as f32 / 100.0, v as f32 / 100.0, 1.0)
}

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(800.), px(600.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |window, cx| {
                let entity = cx.new(|_| GridBench::new());
                GridBench::schedule_frame_callback(entity.clone(), window);
                entity
            },
        )
        .unwrap();
        cx.activate(true);
    });
}
