use craby::prelude::*;

use crate::ffi::bridging::*;
use crate::generated::*;
use rn_chat_panel_types::{send, send_with_reply_blocking, ZedHostCommand};

fn option_to_nullable_string(opt: Option<String>) -> NullableString {
    match opt {
        Some(val) => NullableString { null: false, val },
        None => NullableString {
            null: true,
            val: String::new(),
        },
    }
}

pub struct ZedWorkspace {
    ctx: Context,
}

#[craby_module]
impl ZedWorkspaceSpec for ZedWorkspace {
    fn add_listener(&mut self, event_name: &str) -> Void {
        if event_name == "workspaceChanged" {
            send(ZedHostCommand::RegisterWorkspaceListener {
                module_id: self.ctx.id,
            })
            .ok();
        }
    }

    fn get_workspace_info(&mut self) -> WorkspaceInfo {
        let result = send_with_reply_blocking(|reply| ZedHostCommand::GetWorkspaceInfo { reply });

        match result {
            Ok(info) => WorkspaceInfo {
                project_name: option_to_nullable_string(info.project_name),
                root_path: option_to_nullable_string(info.root_path),
                current_file_path: option_to_nullable_string(info.current_file_path),
            },
            Err(_) => WorkspaceInfo::default(),
        }
    }

    fn remove_listeners(&mut self, _count: Number) -> Void {
        send(ZedHostCommand::UnregisterWorkspaceListener {
            module_id: self.ctx.id,
        })
        .ok();
    }
}
