use craby::{prelude::*, throw};

use crate::ffi::bridging::*;
use crate::generated::*;

pub struct ZedThreads {
    ctx: Context,
}

#[craby_module]
impl ZedThreadsSpec for ZedThreads {
    fn add_listener(&mut self, event_name: &str) -> Void {
        unimplemented!();
    }

    fn delete_thread(&mut self, thread_id: &str) -> Boolean {
        unimplemented!();
    }

    fn get_selected_thread_id(&mut self, session_id: &str) -> Nullable<String> {
        unimplemented!();
    }

    fn list_threads(&mut self) -> String {
        unimplemented!();
    }

    fn load_thread(&mut self, thread_id: &str) -> String {
        unimplemented!();
    }

    fn minimize_history_pane(&mut self) -> Void {
        unimplemented!();
    }

    fn remove_listeners(&mut self, count: Number) -> Void {
        unimplemented!();
    }

    fn save_thread(&mut self, thread_json: &str) -> Boolean {
        unimplemented!();
    }

    fn search_threads(&mut self, query: &str) -> String {
        unimplemented!();
    }

    fn set_history_pane_visible(&mut self, visible: Boolean) -> Void {
        unimplemented!();
    }

    fn set_selected_thread_id(&mut self, session_id: &str, thread_id: Nullable<String>) -> Void {
        unimplemented!();
    }
}
