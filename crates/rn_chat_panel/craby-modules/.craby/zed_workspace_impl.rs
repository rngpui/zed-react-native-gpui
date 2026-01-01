use rngpui_craby::{prelude::*, throw};

use crate::ffi::bridging::*;
use crate::generated::*;

pub struct ZedWorkspace {
    ctx: Context,
}

#[craby_module]
impl ZedWorkspaceSpec for ZedWorkspace {
    fn add_listener(&mut self, event_name: &str) -> Void {
        unimplemented!();
    }

    fn get_workspace_info(&mut self) -> WorkspaceInfo {
        unimplemented!();
    }

    fn remove_listeners(&mut self, count: Number) -> Void {
        unimplemented!();
    }
}
