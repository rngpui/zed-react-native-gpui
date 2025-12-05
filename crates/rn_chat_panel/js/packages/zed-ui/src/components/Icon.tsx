import React, { useEffect, useState } from 'react';
import { View, Text, type ViewStyle } from 'react-native';
import { SvgXml } from '@rngpui/svg';
import { useZedTheme } from '../theme';
import NativeZedIcons from '../../../../../craby-modules/src/NativeZedIcons';

// Mapping from our simplified icon names to Zed's IconName enum (snake_case)
const ICON_NAME_MAP: Record<string, string> = {
  // Common icons
  'send': 'send',
  'close': 'close',
  'check': 'check',
  'chevron-right': 'chevron_right',
  'chevron-down': 'chevron_down',
  'chevron-up': 'chevron_up',
  'chevron-left': 'chevron_left',
  'plus': 'plus',
  'minus': 'dash',
  'search': 'magnifying_glass',
  'settings': 'settings',
  'copy': 'copy',
  'edit': 'pencil',
  'trash': 'trash',
  'refresh': 'rotate_cw',
  'spinner': 'load_circle',
  'user': 'person',
  'folder': 'folder',
  'file': 'file',
  'warning': 'warning',
  'error': 'x_circle',
  'info': 'info',
  'success': 'check',
  'history': 'countdown_timer',
  'at-sign': 'at_sign',
  'cpu': 'database_zap',
  'message': 'chat',
  'message-circle': 'thread',
  'minimize': 'minimize',
  'maximize': 'maximize',

  // Tool icons (agent parity)
  'tool-search': 'tool_search',
  'tool-pencil': 'tool_pencil',
  'tool-terminal': 'tool_terminal',
  'tool-web': 'tool_web',
  'tool-think': 'tool_think',
  'tool-hammer': 'tool_hammer',
  'tool-delete-file': 'tool_delete_file',

  // Agent/UI icons
  'crosshair': 'crosshair',
  'thread': 'thread',
  'shield': 'shield',
  'sparkle': 'sparkle',
  'sparkles': 'sparkle',
  'ai-claude': 'ai_claude',
  'ai-gemini': 'ai_gemini',
  'ai-openai': 'ai_open_ai',

  // Legacy camelCase (deprecated)
  'chevronRight': 'chevron_right',
  'chevronDown': 'chevron_down',
  'chevronUp': 'chevron_up',
  'chevronLeft': 'chevron_left',
};

export type IconName = keyof typeof ICON_NAME_MAP;
export type IconColor = 'default' | 'muted' | 'accent' | 'error' | 'success' | 'warning' | 'disabled';

export interface IconProps {
  name: IconName;
  size?: number;
  color?: IconColor;
  style?: ViewStyle;
}

// Cache for loaded SVG content
const svgCache = new Map<string, string>();

export function Icon({ name, size = 16, color = 'default', style }: IconProps) {
  const { colors } = useZedTheme();
  const [svgContent, setSvgContent] = useState<string | null>(() => svgCache.get(name) ?? null);

  const iconColor = {
    default: colors.icon,
    muted: colors.iconMuted,
    accent: colors.iconAccent,
    error: '#f85149',
    success: '#3fb950',
    warning: '#d29922',
    disabled: colors.iconDisabled,
  }[color];

  useEffect(() => {
    if (svgCache.has(name)) {
      setSvgContent(svgCache.get(name)!);
      return;
    }

    const zedIconName = ICON_NAME_MAP[name];
    if (!zedIconName) return;

    try {
      const svg = NativeZedIcons.getIconSvg(zedIconName);
      if (svg) {
        svgCache.set(name, svg);
        setSvgContent(svg);
      }
    } catch (e) {
      console.error(`[Icon] Failed to get icon ${zedIconName}:`, e);
    }
  }, [name]);

  // Render with SVG if available
  if (svgContent) {
    // Replace stroke/fill colors with the theme color
    const coloredSvg = svgContent
      .replace(/stroke="[^"]*"/g, `stroke="${iconColor}"`)
      .replace(/fill="(?!none)[^"]*"/g, `fill="${iconColor}"`);

    return (
      <View style={[{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }, style]}>
        <SvgXml xml={coloredSvg} width={size} height={size} />
      </View>
    );
  }

  return (
    <View style={[{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }, style]}>
      <Text style={{ fontSize: size, lineHeight: size * 1.1, color: iconColor, textAlign: 'center' }}>
        ?
      </Text>
    </View>
  );
}
