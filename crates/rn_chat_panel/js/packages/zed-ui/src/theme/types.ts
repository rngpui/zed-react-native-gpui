export interface ZedThemeColors {
  // Backgrounds
  background: string;
  surfaceBackground: string;
  elevatedSurfaceBackground: string;
  panelBackground: string;

  // Element states
  elementBackground: string;
  elementHover: string;
  elementActive: string;
  elementSelected: string;
  elementDisabled: string;

  // Ghost element states
  ghostElementBackground: string;
  ghostElementHover: string;
  ghostElementActive: string;
  ghostElementSelected: string;

  // Text
  text: string;
  textMuted: string;
  textPlaceholder: string;
  textDisabled: string;
  textAccent: string;

  // Icons
  icon: string;
  iconMuted: string;
  iconDisabled: string;
  iconAccent: string;

  // Borders
  border: string;
  borderVariant: string;
  borderFocused: string;
  borderSelected: string;
  borderDisabled: string;

  // UI elements
  tabBarBackground: string;
  tabActiveBackground: string;
  tabInactiveBackground: string;
  statusBarBackground: string;
  titleBarBackground: string;
  toolbarBackground: string;
  scrollbarThumbBackground: string;
}

export interface ZedFontSettings {
  uiFontFamily: string;
  uiFontSize: number;
  bufferFontFamily: string;
  bufferFontSize: number;
  // Pre-computed UI font sizes matching native Zed's LabelSize enum
  // These are derived from uiFontSize using native's rem scales
  ui: {
    lg: number;   // 1.0x    - LabelSize::Large (16px from 16px base)
    md: number;   // 0.825x  - LabelSize::Default (~14px)
    sm: number;   // 0.75x   - LabelSize::Small (12px)
    xs: number;   // 0.625x  - LabelSize::XSmall (10px)
  };
}

export interface ZedTheme {
  name: string;
  appearance: 'light' | 'dark';
  colors: ZedThemeColors;
  fonts: ZedFontSettings;
}
