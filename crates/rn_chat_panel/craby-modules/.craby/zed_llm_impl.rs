use rngpui_craby::{prelude::*, throw};

use crate::ffi::bridging::*;
use crate::generated::*;

pub struct ZedLlm {
    ctx: Context,
}

#[craby_module]
impl ZedLlmSpec for ZedLlm {
    fn add_listener(&mut self, event_name: &str) -> Void {
        unimplemented!();
    }

    fn cancel_request(&mut self, request_id: Number) -> Void {
        unimplemented!();
    }

    fn cancel_tool_call(&mut self, tool_call_id: &str) -> Boolean {
        unimplemented!();
    }

    fn connect_agent(&mut self, agent_type: &str) -> Boolean {
        unimplemented!();
    }

    fn disconnect_agent(&mut self, agent_type: &str) -> Void {
        unimplemented!();
    }

    fn ensure_agent_thread(&mut self, agent_type: &str) -> Boolean {
        unimplemented!();
    }

    fn get_agent_models(&mut self, agent_type: &str) -> String {
        unimplemented!();
    }

    fn get_agent_modes(&mut self, agent_type: &str) -> String {
        unimplemented!();
    }

    fn get_agent_servers(&mut self) -> String {
        unimplemented!();
    }

    fn get_default_model_id(&mut self) -> String {
        unimplemented!();
    }

    fn get_models(&mut self) -> String {
        unimplemented!();
    }

    fn get_providers(&mut self) -> String {
        unimplemented!();
    }

    fn is_available(&mut self) -> Boolean {
        unimplemented!();
    }

    fn list_acp_threads(&mut self) -> String {
        unimplemented!();
    }

    fn minimize_chat_pane(&mut self) -> Void {
        unimplemented!();
    }

    fn open_acp_thread(&mut self, thread_id: &str) -> Boolean {
        unimplemented!();
    }

    fn remove_listeners(&mut self, count: Number) -> Void {
        unimplemented!();
    }

    fn respond_to_tool_authorization(&mut self, tool_call_id: &str, permission_option_id: &str) -> Boolean {
        unimplemented!();
    }

    fn search_acp_threads(&mut self, query: &str) -> String {
        unimplemented!();
    }

    fn send_message(&mut self, message: &str) -> Number {
        unimplemented!();
    }

    fn send_message_to_agent(&mut self, agent_type: &str, message: &str) -> Number {
        unimplemented!();
    }

    fn send_message_to_model(&mut self, model_id: &str, message: &str) -> Number {
        unimplemented!();
    }

    fn set_agent_mode(&mut self, agent_type: &str, mode_id: &str) -> Boolean {
        unimplemented!();
    }

    fn set_agent_model(&mut self, agent_type: &str, model_id: &str) -> Boolean {
        unimplemented!();
    }

    fn set_chat_pane_visible(&mut self, visible: Boolean) -> Void {
        unimplemented!();
    }

    fn start_agent_thread(&mut self, agent_type: &str) -> Boolean {
        unimplemented!();
    }
}
