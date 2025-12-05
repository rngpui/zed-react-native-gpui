use craby::prelude::*;

use crate::generated::*;
use rn_chat_panel_types::{send_with_reply_blocking, ZedHostCommand};

pub struct ZedMarkdown {
    ctx: Context,
}

#[craby_module]
impl ZedMarkdownSpec for ZedMarkdown {
    fn parse(&mut self, markdown: &str) -> String {
        let result = send_with_reply_blocking(|reply| ZedHostCommand::ParseMarkdown {
            markdown: markdown.to_string(),
            reply,
        });

        match result {
            Ok(parsed) => serde_json::to_string(&parsed).unwrap_or_else(|_| {
                r#"{"elements":[]}"#.to_string()
            }),
            Err(_) => r#"{"elements":[]}"#.to_string(),
        }
    }
}
