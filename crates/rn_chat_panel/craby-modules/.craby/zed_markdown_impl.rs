use craby::{prelude::*, throw};

use crate::ffi::bridging::*;
use crate::generated::*;

pub struct ZedMarkdown {
    ctx: Context,
}

#[craby_module]
impl ZedMarkdownSpec for ZedMarkdown {
    fn parse(&mut self, markdown: &str) -> String {
        unimplemented!();
    }
}
