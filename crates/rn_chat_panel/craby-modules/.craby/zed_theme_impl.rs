use craby::{prelude::*, throw};

use crate::ffi::bridging::*;
use crate::generated::*;

pub struct ZedTheme {
    ctx: Context,
}

#[craby_module]
impl ZedThemeSpec for ZedTheme {
    fn add_listener(&mut self, event_name: &str) -> Void {
        unimplemented!();
    }

    fn get_color(&mut self, name: &str) -> Nullable<String> {
        unimplemented!();
    }

    fn get_theme(&mut self) -> ThemeData {
        unimplemented!();
    }

    fn remove_listeners(&mut self, count: Number) -> Void {
        unimplemented!();
    }
}
