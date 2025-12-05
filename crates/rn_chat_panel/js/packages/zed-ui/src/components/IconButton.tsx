import React, { useState } from 'react';
import { Pressable, type PressableProps, type ViewStyle } from 'react-native';
import { Icon, type IconName, type IconColor } from './Icon';
import { useZedTheme } from '../theme';

export type IconButtonStyle = 'filled' | 'subtle' | 'ghost' | 'transparent';
export type IconButtonSize = 'compact' | 'default' | 'medium' | 'large';

export interface IconButtonProps extends Omit<PressableProps, 'style' | 'children'> {
  icon: IconName;
  buttonStyle?: IconButtonStyle;
  size?: IconButtonSize;
  disabled?: boolean;
  style?: ViewStyle;
}

const SIZE_CONFIG: Record<IconButtonSize, { containerSize: number; iconSize: number; borderRadius: number }> = {
  compact: { containerSize: 20, iconSize: 12, borderRadius: 4 },
  default: { containerSize: 24, iconSize: 14, borderRadius: 4 },
  medium: { containerSize: 28, iconSize: 16, borderRadius: 6 },
  large: { containerSize: 32, iconSize: 18, borderRadius: 6 },
};

export function IconButton({
  icon,
  buttonStyle = 'ghost',
  size = 'default',
  disabled = false,
  style,
  onPressIn,
  onPressOut,
  ...props
}: IconButtonProps) {
  const { colors } = useZedTheme();
  const [pressed, setPressed] = useState(false);
  const [hovered, setHovered] = useState(false);

  const { containerSize, iconSize, borderRadius } = SIZE_CONFIG[size];

  const getBg = (): string => {
    const isGhost = buttonStyle === 'ghost' || buttonStyle === 'transparent';
    if (disabled) return isGhost ? 'transparent' : colors.elementDisabled;
    if (isGhost) return pressed ? colors.ghostElementActive : hovered ? colors.ghostElementHover : 'transparent';
    if (buttonStyle === 'subtle') return pressed ? colors.elementActive : hovered ? colors.elementHover : colors.elementBackground;
    // filled
    return pressed ? colors.textAccent : hovered ? colors.textAccent : colors.textAccent;
  };

  const getIconColor = (): IconColor => {
    if (disabled) return 'disabled';
    if (buttonStyle === 'filled') return 'default';
    // Ghost buttons don't change icon color on hover - only background changes
    return 'default';
  };

  return (
    <Pressable
      disabled={disabled}
      onPressIn={(e) => { setPressed(true); onPressIn?.(e); }}
      onPressOut={(e) => { setPressed(false); onPressOut?.(e); }}
      onHoverIn={() => setHovered(true)}
      onHoverOut={() => setHovered(false)}
      style={[
        {
          width: containerSize,
          height: containerSize,
          borderRadius,
          backgroundColor: getBg(),
          alignItems: 'center',
          justifyContent: 'center',
        },
        style,
      ]}
      {...props}
    >
      <Icon name={icon} size={iconSize} color={getIconColor()} />
    </Pressable>
  );
}
