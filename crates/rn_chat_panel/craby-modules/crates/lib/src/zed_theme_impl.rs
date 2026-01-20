use rngpui_craby::prelude::*;

use crate::ffi::bridging::*;
use crate::generated::*;
use rn_chat_panel_types::{send, send_with_reply_blocking, ZedHostCommand};

pub struct ZedTheme {
    ctx: Context,
}

#[craby_module]
impl ZedThemeSpec for ZedTheme {
    fn add_listener(&mut self, event_name: &str) -> Void {
        if event_name == "themeChanged" {
            // Register this module instance for theme change notifications
            send(ZedHostCommand::RegisterThemeListener {
                module_id: self.ctx.id,
            })
            .ok();
        }
    }

    fn get_color(&mut self, name: &str) -> Nullable<String> {
        let result = send_with_reply_blocking(|reply| {
            ZedHostCommand::GetColor {
                name: name.to_string(),
                reply,
            }
        });

        match result {
            Ok(Some(color)) => Nullable::some(color),
            _ => Nullable::none(),
        }
    }

    fn get_theme(&mut self) -> ThemeData {
        let result = send_with_reply_blocking(|reply| ZedHostCommand::GetTheme { reply });

        match result {
            Ok(data) => ThemeData {
                name: data.name,
                appearance: data.appearance,
                colors: ThemeColors {
                    background: data.colors.background,
                    surface_background: data.colors.surface_background,
                    elevated_surface_background: data.colors.elevated_surface_background,
                    panel_background: data.colors.panel_background,
                    element_background: data.colors.element_background,
                    element_hover: data.colors.element_hover,
                    element_active: data.colors.element_active,
                    element_selected: data.colors.element_selected,
                    element_disabled: data.colors.element_disabled,
                    ghost_element_background: data.colors.ghost_element_background,
                    ghost_element_hover: data.colors.ghost_element_hover,
                    ghost_element_active: data.colors.ghost_element_active,
                    ghost_element_selected: data.colors.ghost_element_selected,
                    text: data.colors.text,
                    text_muted: data.colors.text_muted,
                    text_placeholder: data.colors.text_placeholder,
                    text_disabled: data.colors.text_disabled,
                    text_accent: data.colors.text_accent,
                    icon: data.colors.icon,
                    icon_muted: data.colors.icon_muted,
                    icon_disabled: data.colors.icon_disabled,
                    icon_accent: data.colors.icon_accent,
                    border: data.colors.border,
                    border_variant: data.colors.border_variant,
                    border_focused: data.colors.border_focused,
                    border_selected: data.colors.border_selected,
                    border_disabled: data.colors.border_disabled,
                    tab_bar_background: data.colors.tab_bar_background,
                    tab_active_background: data.colors.tab_active_background,
                    tab_inactive_background: data.colors.tab_inactive_background,
                    status_bar_background: data.colors.status_bar_background,
                    title_bar_background: data.colors.title_bar_background,
                    toolbar_background: data.colors.toolbar_background,
                    scrollbar_thumb_background: data.colors.scrollbar_thumb_background,
                },
                fonts: FontSettings {
                    ui_font_family: data.fonts.ui_font_family,
                    ui_font_size: data.fonts.ui_font_size as f64,
                    buffer_font_family: data.fonts.buffer_font_family,
                    buffer_font_size: data.fonts.buffer_font_size as f64,
                },
            },
            Err(_) => ThemeData::default(),
        }
    }

    fn remove_listeners(&mut self, _count: Number) -> Void {
        // Unregister this module instance from theme change notifications
        send(ZedHostCommand::UnregisterThemeListener {
            module_id: self.ctx.id,
        })
        .ok();
    }
}
