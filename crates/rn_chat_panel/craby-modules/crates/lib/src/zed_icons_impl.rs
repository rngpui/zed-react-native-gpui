use craby::prelude::*;

use crate::generated::*;
use rn_chat_panel_types::{send_with_reply_blocking, ZedHostCommand};

pub struct ZedIcons {
    ctx: Context,
}

#[craby_module]
impl ZedIconsSpec for ZedIcons {
    fn get_icon_svg(&mut self, name: &str) -> String {
        send_with_reply_blocking(|reply| ZedHostCommand::GetIconSvg {
            name: name.to_string(),
            reply,
        })
        .unwrap_or_default()
    }

    fn list_icons(&mut self) -> String {
        let icons = send_with_reply_blocking(|reply| ZedHostCommand::ListIcons { reply })
            .unwrap_or_default();
        serde_json::to_string(&icons).unwrap_or_else(|_| "[]".to_string())
    }
}
