use craby::prelude::*;

use crate::generated::*;
use rn_chat_panel_types::{send, send_with_reply_blocking, AgentServerType, ZedHostCommand};

pub struct ZedLlm {
    ctx: Context,
}

#[craby_module]
impl ZedLlmSpec for ZedLlm {
    fn add_listener(&mut self, event_name: &str) -> Void {
        match event_name {
            "toolCallEvent" => {
                send(ZedHostCommand::RegisterToolCallListener {
                    module_id: self.ctx.id,
                })
                .ok();
            }
            "threadSnapshot" => {
                send(ZedHostCommand::RegisterThreadSnapshotListener {
                    module_id: self.ctx.id,
                })
                .ok();
            }
            "acpThreadsChanged" => {
                send(ZedHostCommand::RegisterAcpThreadsListener {
                    module_id: self.ctx.id,
                })
                .ok();
            }
            "agentConnected" | "agentDisconnected" => {
                send(ZedHostCommand::RegisterAgentListener {
                    module_id: self.ctx.id,
                })
                .ok();
            }
            _ => {
                // LLM events (llmChunk, llmDone, llmError) are emitted based on request_id correlation
            }
        }
    }

    fn cancel_request(&mut self, request_id: Number) -> Void {
        send(ZedHostCommand::CancelLlmRequest {
            request_id: request_id as u64,
        })
        .ok();
    }

    fn is_available(&mut self) -> Boolean {
        send_with_reply_blocking(|reply| ZedHostCommand::IsLlmAvailable { reply }).unwrap_or(false)
    }

    fn remove_listeners(&mut self, _count: Number) -> Void {
        // Unregister all listeners for this module
        send(ZedHostCommand::UnregisterToolCallListener {
            module_id: self.ctx.id,
        })
        .ok();
        send(ZedHostCommand::UnregisterThreadSnapshotListener {
            module_id: self.ctx.id,
        })
        .ok();
        send(ZedHostCommand::UnregisterAcpThreadsListener {
            module_id: self.ctx.id,
        })
        .ok();
        send(ZedHostCommand::UnregisterAgentListener {
            module_id: self.ctx.id,
        })
        .ok();
    }

    fn send_message(&mut self, message: &str) -> Number {
        let module_id = self.ctx.id;
        let request_id = send_with_reply_blocking(|reply| ZedHostCommand::SendLlmMessage {
            module_id,
            message: message.to_string(),
            reply,
        })
        .unwrap_or(0);

        request_id as f64
    }

    // Extended language model methods

    fn get_providers(&mut self) -> String {
        let providers =
            send_with_reply_blocking(|reply| ZedHostCommand::GetProviders { reply })
                .unwrap_or_default();
        serde_json::to_string(&providers).unwrap_or_else(|_| "[]".to_string())
    }

    fn get_models(&mut self) -> String {
        let models =
            send_with_reply_blocking(|reply| ZedHostCommand::GetModels { reply })
                .unwrap_or_default();
        serde_json::to_string(&models).unwrap_or_else(|_| "[]".to_string())
    }

    fn get_default_model_id(&mut self) -> String {
        send_with_reply_blocking(|reply| ZedHostCommand::GetDefaultModelId { reply })
            .unwrap_or(None)
            .unwrap_or_default()
    }

    fn send_message_to_model(&mut self, model_id: &str, message: &str) -> Number {
        let module_id = self.ctx.id;
        let request_id = send_with_reply_blocking(|reply| ZedHostCommand::SendLlmMessageToModel {
            module_id,
            model_id: model_id.to_string(),
            message: message.to_string(),
            reply,
        })
        .unwrap_or(0);

        request_id as f64
    }

    // Agent server methods

    fn get_agent_servers(&mut self) -> String {
        let servers =
            send_with_reply_blocking(|reply| ZedHostCommand::GetAgentServers { reply })
                .unwrap_or_default();
        serde_json::to_string(&servers).unwrap_or_else(|_| "[]".to_string())
    }

    fn connect_agent(&mut self, agent_type: &str) -> Boolean {
        let Some(agent_type) = AgentServerType::from_str(agent_type) else {
            return false;
        };
        send_with_reply_blocking(|reply| ZedHostCommand::ConnectAgent { agent_type, reply })
            .unwrap_or(false)
    }

    fn disconnect_agent(&mut self, agent_type: &str) -> Void {
        if let Some(agent_type) = AgentServerType::from_str(agent_type) {
            send(ZedHostCommand::DisconnectAgent { agent_type }).ok();
        }
    }

    fn send_message_to_agent(&mut self, agent_type: &str, message: &str) -> Number {
        let Some(agent_type) = AgentServerType::from_str(agent_type) else {
            return 0.0;
        };
        let module_id = self.ctx.id;
        let request_id = send_with_reply_blocking(|reply| ZedHostCommand::SendAgentMessage {
            module_id,
            agent_type,
            message: message.to_string(),
            reply,
        })
        .unwrap_or(0);

        request_id as f64
    }

    fn ensure_agent_thread(&mut self, agent_type: &str) -> Boolean {
        let Some(agent_type) = AgentServerType::from_str(agent_type) else {
            return false;
        };
        send_with_reply_blocking(|reply| ZedHostCommand::EnsureAgentThread { agent_type, reply })
            .unwrap_or(false)
    }

    fn start_agent_thread(&mut self, agent_type: &str) -> Boolean {
        let Some(agent_type) = AgentServerType::from_str(agent_type) else {
            return false;
        };
        send_with_reply_blocking(|reply| ZedHostCommand::StartAgentThread { agent_type, reply })
            .unwrap_or(false)
    }

    fn get_agent_models(&mut self, agent_type: &str) -> String {
        let Some(agent_type) = AgentServerType::from_str(agent_type) else {
            return "[]".to_string();
        };
        let models =
            send_with_reply_blocking(|reply| ZedHostCommand::GetAgentModels { agent_type, reply })
                .unwrap_or_default();
        serde_json::to_string(&models).unwrap_or_else(|_| "[]".to_string())
    }

    fn set_agent_model(&mut self, agent_type: &str, model_id: &str) -> Boolean {
        let Some(agent_type) = AgentServerType::from_str(agent_type) else {
            return false;
        };
        send_with_reply_blocking(|reply| ZedHostCommand::SetAgentModel {
            agent_type,
            model_id: model_id.to_string(),
            reply,
        })
        .unwrap_or(false)
    }

    fn get_agent_modes(&mut self, agent_type: &str) -> String {
        let Some(agent_type) = AgentServerType::from_str(agent_type) else {
            return "[]".to_string();
        };
        let modes =
            send_with_reply_blocking(|reply| ZedHostCommand::GetAgentModes { agent_type, reply })
                .unwrap_or_default();
        serde_json::to_string(&modes).unwrap_or_else(|_| "[]".to_string())
    }

    fn set_agent_mode(&mut self, agent_type: &str, mode_id: &str) -> Boolean {
        let Some(agent_type) = AgentServerType::from_str(agent_type) else {
            return false;
        };
        send_with_reply_blocking(|reply| ZedHostCommand::SetAgentMode {
            agent_type,
            mode_id: mode_id.to_string(),
            reply,
        })
        .unwrap_or(false)
    }

    fn list_acp_threads(&mut self) -> String {
        let threads =
            send_with_reply_blocking(|reply| ZedHostCommand::ListAcpThreads { reply })
                .unwrap_or_default();
        serde_json::to_string(&threads).unwrap_or_else(|_| "[]".to_string())
    }

    fn search_acp_threads(&mut self, query: &str) -> String {
        let threads = send_with_reply_blocking(|reply| ZedHostCommand::SearchAcpThreads {
            query: query.to_string(),
            reply,
        })
        .unwrap_or_default();
        serde_json::to_string(&threads).unwrap_or_else(|_| "[]".to_string())
    }

    fn open_acp_thread(&mut self, thread_id: &str) -> Boolean {
        send_with_reply_blocking(|reply| ZedHostCommand::OpenAcpThread {
            thread_id: thread_id.to_string(),
            reply,
        })
        .unwrap_or(false)
    }

    fn set_chat_pane_visible(&mut self, visible: Boolean) -> Void {
        send(ZedHostCommand::SetChatPaneVisible { visible }).ok();
    }

    fn minimize_chat_pane(&mut self) -> Void {
        send(ZedHostCommand::MinimizeChatPane).ok();
    }

    // Tool call methods

    fn respond_to_tool_authorization(
        &mut self,
        tool_call_id: &str,
        permission_option_id: &str,
    ) -> Boolean {
        send_with_reply_blocking(|reply| ZedHostCommand::RespondToToolAuthorization {
            tool_call_id: tool_call_id.to_string(),
            permission_option_id: permission_option_id.to_string(),
            reply,
        })
        .unwrap_or(false)
    }

    fn cancel_tool_call(&mut self, tool_call_id: &str) -> Boolean {
        send_with_reply_blocking(|reply| ZedHostCommand::CancelToolCall {
            tool_call_id: tool_call_id.to_string(),
            reply,
        })
        .unwrap_or(false)
    }
}
