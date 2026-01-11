use std::collections::VecDeque;
use std::time::Instant;

use gpui::{
    App, Application, Bounds, Context, Entity, Window, WindowBounds, WindowOptions, deferred, div,
    prelude::*, px, rgb, size,
};

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
}

impl GridBench {
    fn new() -> Self {
        Self {
            render_fps: FpsCounter::new(),
            frame_fps: FpsCounter::new(),
        }
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
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        window.request_animation_frame();
        self.render_fps.record();

        div()
            .size_full()
            .bg(rgb(0x1e1e1e))
            .child(deferred(
                div()
                    .absolute()
                    .top_2()
                    .left_2()
                    .px_3()
                    .py_1()
                    .bg(gpui::black().opacity(0.7))
                    .rounded_md()
                    .text_sm()
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
                            .children((0..GRID_SIZE).map(|row| {
                                div()
                                    .flex()
                                    .gap_1()
                                    .children((0..GRID_SIZE).map(|col| {
                                        let hue = ((row * GRID_SIZE + col) as f32
                                            / (GRID_SIZE * GRID_SIZE) as f32
                                            * 360.0)
                                            as u32;
                                        let color = hsv_to_rgb(hue, 70, 60);
                                        div().size(px(24.0)).rounded_sm().bg(color)
                                    }))
                            })),
                    ),
            )
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
