use rngpui_craby::{prelude::*, throw};

use crate::ffi::bridging::*;
use crate::generated::*;

pub struct ZedIcons {
    ctx: Context,
}

#[craby_module]
impl ZedIconsSpec for ZedIcons {
    fn get_icon_svg(&mut self, name: &str) -> String {
        unimplemented!();
    }

    fn list_icons(&mut self) -> String {
        unimplemented!();
    }
}
