import type { ZedTheme, ZedThemeColors } from './types';

/**
 * Transform native theme data from NativeZedTheme turbomodule
 * into the JavaScript-friendly ZedTheme format.
 */
export function transformNativeTheme(nativeData: {
  name: string;
  appearance: string;
  colors: Record<string, string>;
  fonts: {
    uiFontFamily: string;
    uiFontSize: number;
    bufferFontFamily: string;
    bufferFontSize: number;
  };
}): ZedTheme {
  const baseUiFontSize = nativeData.fonts.uiFontSize;
  const result: ZedTheme = {
    name: nativeData.name,
    appearance: nativeData.appearance === 'light' ? 'light' : 'dark',
    colors: transformColors(nativeData.colors),
    fonts: {
      uiFontFamily: nativeData.fonts.uiFontFamily,
      uiFontSize: nativeData.fonts.uiFontSize,
      bufferFontFamily: nativeData.fonts.bufferFontFamily,
      bufferFontSize: nativeData.fonts.bufferFontSize,
      // Pre-computed sizes matching native Zed's TextSize scales
      // Native uses rems: Large=1.0, Default=0.8125, Small=0.8125, XSmall=0.6875
      ui: {
        lg: baseUiFontSize * 0.9,     // ~14.4px at 16px base
        md: baseUiFontSize * 0.8125,  // ~13px at 16px base (matches text_ui)
        sm: baseUiFontSize * 0.75,    // ~12px at 16px base (matches text_sm)
        xs: baseUiFontSize * 0.6875,  // ~11px at 16px base (matches text_xs)
      },
    },
  };
  return result;
}

function transformColors(colors: Record<string, string>): ZedThemeColors {
  return {
    background: colors.background ?? '#000000',
    surfaceBackground: colors.surfaceBackground ?? '#1a1a1a',
    elevatedSurfaceBackground: colors.elevatedSurfaceBackground ?? '#2a2a2a',
    panelBackground: colors.panelBackground ?? '#1a1a1a',
    elementBackground: colors.elementBackground ?? '#2a2a2a',
    elementHover: colors.elementHover ?? '#3a3a3a',
    elementActive: colors.elementActive ?? '#4a4a4a',
    elementSelected: colors.elementSelected ?? '#3a3a3a',
    elementDisabled: colors.elementDisabled ?? '#2a2a2a',
    ghostElementBackground: colors.ghostElementBackground ?? 'transparent',
    ghostElementHover: colors.ghostElementHover ?? 'rgba(255,255,255,0.1)',
    ghostElementActive: colors.ghostElementActive ?? 'rgba(255,255,255,0.15)',
    ghostElementSelected: colors.ghostElementSelected ?? 'rgba(255,255,255,0.1)',
    text: colors.text ?? '#ffffff',
    textMuted: colors.textMuted ?? '#888888',
    textPlaceholder: colors.textPlaceholder ?? '#666666',
    textDisabled: colors.textDisabled ?? '#444444',
    textAccent: colors.textAccent ?? '#6b8afd',
    icon: colors.icon ?? '#ffffff',
    iconMuted: colors.iconMuted ?? '#888888',
    iconDisabled: colors.iconDisabled ?? '#444444',
    iconAccent: colors.iconAccent ?? '#6b8afd',
    border: colors.border ?? '#3a3a3a',
    borderVariant: colors.borderVariant ?? '#2a2a2a',
    borderFocused: colors.borderFocused ?? '#6b8afd',
    borderSelected: colors.borderSelected ?? '#6b8afd',
    borderDisabled: colors.borderDisabled ?? '#2a2a2a',
    tabBarBackground: colors.tabBarBackground ?? '#1a1a1a',
    tabActiveBackground: colors.tabActiveBackground ?? '#2a2a2a',
    tabInactiveBackground: colors.tabInactiveBackground ?? 'transparent',
    statusBarBackground: colors.statusBarBackground ?? '#1a1a1a',
    titleBarBackground: colors.titleBarBackground ?? '#1a1a1a',
    toolbarBackground: colors.toolbarBackground ?? '#1a1a1a',
    scrollbarThumbBackground: colors.scrollbarThumbBackground ?? '#3a3a3a',
  };
}

