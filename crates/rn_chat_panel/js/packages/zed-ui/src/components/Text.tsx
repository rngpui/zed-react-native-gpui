import React from 'react';
import { Text as RNText, type TextProps as RNTextProps } from 'react-native';
import { useZedTheme } from '../theme';

// Variants map to native Zed's LabelSize:
// - body/muted: Default (14px, 0.825x)
// - small: Small (12px, 0.75x)
// - label: XSmall (10px, 0.625x)
// - title: Large (16px, 1.0x)
export type TextVariant = 'body' | 'muted' | 'small' | 'label' | 'title' | 'code';
export type TextColor = 'default' | 'muted' | 'accent' | 'error' | 'disabled';

export interface TextProps extends Omit<RNTextProps, 'style'> {
  variant?: TextVariant;
  color?: TextColor;
  style?: RNTextProps['style'];
}

export function Text({ variant = 'body', color = 'default', style, children, ...props }: TextProps) {
  const { colors, fonts } = useZedTheme();

  const textColor = {
    default: colors.text,
    muted: colors.textMuted,
    accent: colors.textAccent,
    error: '#f85149',
    disabled: colors.textDisabled,
  }[color];

  const variantStyle = {
    body: { fontSize: fonts.ui.md, fontFamily: fonts.uiFontFamily, lineHeight: fonts.ui.md * 1.5 },
    muted: { fontSize: fonts.ui.md, fontFamily: fonts.uiFontFamily, lineHeight: fonts.ui.md * 1.5 },
    small: { fontSize: fonts.ui.sm, fontFamily: fonts.uiFontFamily, lineHeight: fonts.ui.sm * 1.5 },
    label: { fontSize: fonts.ui.xs, fontFamily: fonts.uiFontFamily, fontWeight: '500' as const, textTransform: 'uppercase' as const, letterSpacing: 0.5 },
    title: { fontSize: fonts.ui.lg, fontFamily: fonts.uiFontFamily, fontWeight: '600' as const },
    code: { fontSize: fonts.bufferFontSize, fontFamily: fonts.bufferFontFamily, lineHeight: fonts.bufferFontSize * 1.4 },
  }[variant];

  return <RNText style={[{ color: textColor }, variantStyle, style]} {...props}>{children}</RNText>;
}
