import React, { useState } from 'react';
import { Pressable, type PressableProps, type ViewStyle } from 'react-native';
import { Text } from './Text';
import { useZedTheme } from '../theme';

export type ButtonStyle = 'filled' | 'subtle' | 'outlined' | 'ghost' | 'transparent';
export type ButtonSize = 'compact' | 'default' | 'medium' | 'large';

export interface ButtonProps extends Omit<PressableProps, 'style' | 'children'> {
  title: string;
  buttonStyle?: ButtonStyle;
  size?: ButtonSize;
  disabled?: boolean;
  style?: ViewStyle;
}

export function Button({
  title,
  buttonStyle = 'filled',
  size = 'default',
  disabled = false,
  style,
  onPressIn,
  onPressOut,
  ...props
}: ButtonProps) {
  const { colors, fonts } = useZedTheme();
  const [pressed, setPressed] = useState(false);
  const [hovered, setHovered] = useState(false);

  const sizes: Record<ButtonSize, ViewStyle & { fontSize: number }> = {
    compact: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4, fontSize: fonts.ui.xs },
    default: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 6, fontSize: fonts.ui.sm },
    medium: { paddingHorizontal: 16, paddingVertical: 8, borderRadius: 6, fontSize: fonts.ui.sm },
    large: { paddingHorizontal: 20, paddingVertical: 10, borderRadius: 8, fontSize: fonts.ui.md },
  };

  const getBg = () => {
    const isGhostLike = buttonStyle === 'outlined' || buttonStyle === 'transparent' || buttonStyle === 'ghost';
    if (disabled) return isGhostLike ? 'transparent' : colors.elementDisabled;
    if (isGhostLike) return pressed ? colors.ghostElementActive : hovered ? colors.ghostElementHover : 'transparent';
    if (buttonStyle === 'subtle') return pressed ? colors.elementActive : hovered ? colors.elementHover : colors.elementBackground;
    return colors.textAccent;
  };

  const getBorder = () => {
    if (disabled) return colors.borderDisabled;
    if (buttonStyle === 'outlined') return pressed || hovered ? colors.borderFocused : colors.border;
    return 'transparent';
  };

  const getTextColor = () => {
    if (disabled) return colors.textDisabled;
    if (buttonStyle === 'filled') return colors.background;
    // Ghost/transparent buttons don't change text color on hover - only background changes
    return colors.text;
  };

  const { fontSize, ...sizeStyle } = sizes[size];

  return (
    <Pressable
      disabled={disabled}
      onPressIn={(e) => { setPressed(true); onPressIn?.(e); }}
      onPressOut={(e) => { setPressed(false); onPressOut?.(e); }}
      onHoverIn={() => setHovered(true)}
      onHoverOut={() => setHovered(false)}
      style={[sizeStyle, { backgroundColor: getBg(), borderWidth: buttonStyle === 'outlined' ? 1 : 0, borderColor: getBorder(), alignItems: 'center', justifyContent: 'center' }, style]}
      {...props}
    >
      <Text style={{ fontSize, color: getTextColor(), fontWeight: '500' }}>{title}</Text>
    </Pressable>
  );
}
