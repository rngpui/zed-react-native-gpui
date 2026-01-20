import type { NativeModule, Signal } from '@rngpui/craby-modules';
import { NativeModuleRegistry } from '@rngpui/craby-modules';

export interface ThemeColors {
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

export interface FontSettings {
  uiFontFamily: string;
  uiFontSize: number;
  bufferFontFamily: string;
  bufferFontSize: number;
}

export interface ThemeData {
  name: string;
  appearance: string; // "light" or "dark"
  colors: ThemeColors;
  fonts: FontSettings;
}

export interface ThemeChangedEvent {
  theme: ThemeData;
}

export interface Spec extends NativeModule {
  getTheme(): ThemeData;
  getColor(name: string): string | null;
  themeChanged: Signal<ThemeChangedEvent>;
  // Required by NativeEventEmitter - no-ops since events go through __rctDeviceEventEmitter
  addListener(eventName: string): void;
  removeListeners(count: number): void;
}

export default NativeModuleRegistry.getEnforcing<Spec>('ZedTheme');
