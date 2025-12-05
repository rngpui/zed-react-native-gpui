use acp_thread::{AcpThread, AgentThreadEntry, AssistantMessageChunk, ContentBlock, ThreadStatus, ToolCallStatus};
use agent_client_protocol as acp;
use gpui::App;
use rn_chat_panel_types::{
    AssistantChunkSnapshot, EntrySnapshot, PermissionOptionSnapshot, RnTokenUsage,
    ThreadSnapshot, ThreadStatusSnapshot,
};

pub fn thread_snapshot_from_acp_thread(thread: &AcpThread, cx: &App) -> ThreadSnapshot {
    ThreadSnapshot {
        session_id: thread.session_id().0.to_string(),
        title: thread.title().to_string(),
        status: match thread.status() {
            ThreadStatus::Idle => ThreadStatusSnapshot::Idle,
            ThreadStatus::Generating => ThreadStatusSnapshot::Generating,
        },
        entries: thread
            .entries()
            .iter()
            .map(|entry| entry_to_snapshot(entry, cx))
            .collect(),
        token_usage: thread.token_usage().map(|tu| RnTokenUsage {
            input_tokens: 0,
            output_tokens: tu.used_tokens,
        }),
    }
}

fn entry_to_snapshot(entry: &AgentThreadEntry, cx: &App) -> EntrySnapshot {
    match entry {
        AgentThreadEntry::UserMessage(msg) => EntrySnapshot::UserMessage {
            id: None,
            content: msg.content.to_markdown(cx).to_string(),
        },
        AgentThreadEntry::AssistantMessage(msg) => EntrySnapshot::AssistantMessage {
            chunks: msg.chunks.iter().map(|c| chunk_to_snapshot(c, cx)).collect(),
        },
        AgentThreadEntry::ToolCall(call) => {
            let input_json = call
                .raw_input
                .as_ref()
                .and_then(|value| serde_json::to_string(value).ok());
            let result_text = call.raw_output.as_ref().map(raw_json_to_display_string);
            let error_message = match call.status {
                ToolCallStatus::Failed => call
                    .raw_output
                    .as_ref()
                    .and_then(raw_json_error_message)
                    .or_else(|| call.raw_output.as_ref().map(raw_json_to_display_string)),
                _ => None,
            };

            let diff_count = call.diffs().count() as u32;
            let terminal_count = call.terminals().count() as u32;

            let authorization_options = match &call.status {
                ToolCallStatus::WaitingForConfirmation { options, .. } => Some(
                    options
                        .iter()
                        .map(|option| PermissionOptionSnapshot {
                            id: option.option_id.to_string(),
                            label: option.name.clone(),
                            description: None,
                            is_default: matches!(option.kind, acp::PermissionOptionKind::AllowOnce),
                        })
                        .collect(),
                ),
                _ => None,
            };

            EntrySnapshot::ToolCall {
                id: call.id.to_string(),
                title: call.label.read(cx).source().to_string(),
                kind: format!("{:?}", call.kind),
                status: match call.status {
                    ToolCallStatus::Pending => "Pending",
                    ToolCallStatus::WaitingForConfirmation { .. } => "WaitingForConfirmation",
                    ToolCallStatus::InProgress => "InProgress",
                    ToolCallStatus::Completed => "Completed",
                    ToolCallStatus::Failed => "Failed",
                    ToolCallStatus::Rejected => "Rejected",
                    ToolCallStatus::Canceled => "Canceled",
                }
                .to_string(),
                input_json,
                result_text,
                error_message,
                diff_count,
                terminal_count,
                authorization_options,
            }
        }
    }
}

fn chunk_to_snapshot(chunk: &AssistantMessageChunk, cx: &App) -> AssistantChunkSnapshot {
    match chunk {
        AssistantMessageChunk::Message { block } => AssistantChunkSnapshot::Text {
            content: content_block_to_string(block, cx),
        },
        AssistantMessageChunk::Thought { block } => AssistantChunkSnapshot::Thinking {
            content: content_block_to_string(block, cx),
        },
    }
}

fn content_block_to_string(block: &ContentBlock, cx: &App) -> String {
    block.to_markdown(cx).to_string()
}

fn raw_json_to_display_string(value: &serde_json::Value) -> String {
    if let Some(s) = value.as_str() {
        return s.to_string();
    }
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

fn raw_json_error_message(value: &serde_json::Value) -> Option<String> {
    let serde_json::Value::Object(obj) = value else {
        return value.as_str().map(|s| s.to_string());
    };
    for key in ["error", "message", "detail", "reason"] {
        if let Some(serde_json::Value::String(s)) = obj.get(key) {
            return Some(s.clone());
        }
    }
    None
}
