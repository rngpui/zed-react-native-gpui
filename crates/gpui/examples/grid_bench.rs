use std::collections::VecDeque;
use std::time::Instant;

use gpui::{
    App, Application, Bounds, Context, Window, WindowBounds, WindowOptions, deferred, div,
    prelude::*, px, rgb, size,
};

const GRID_SIZE: usize = 50;
const FRAME_HISTORY: usize = 60;

struct GridBench {
    frame_times: VecDeque<Instant>,
    fps: f64,
}

impl GridBench {
    fn new() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(FRAME_HISTORY + 1),
            fps: 0.0,
        }
    }

    fn record_frame(&mut self) {
        let now = Instant::now();
        self.frame_times.push_back(now);

        if self.frame_times.len() > FRAME_HISTORY {
            self.frame_times.pop_front();
        }

        if self.frame_times.len() >= 2 {
            let oldest = self.frame_times.front().unwrap();
            let elapsed = now.duration_since(*oldest).as_secs_f64();
            self.fps = (self.frame_times.len() - 1) as f64 / elapsed;
        }
    }
}

impl Render for GridBench {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        window.request_animation_frame();
        self.record_frame();

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
                    .text_color(rgb(0x00ff00))
                    .text_sm()
                    .child(format!("{:.1} FPS", self.fps)),
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
            |_, cx| cx.new(|_| GridBench::new()),
        )
        .unwrap();
        cx.activate(true);
    });
}
