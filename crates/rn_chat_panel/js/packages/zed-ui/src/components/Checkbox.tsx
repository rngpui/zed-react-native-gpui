import React, { useState } from 'react';
import { Pressable, View, type PressableProps, type ViewStyle } from 'react-native';
import { Text } from './Text';
import { Icon } from './Icon';
import { useZedTheme } from '../theme';

export interface CheckboxProps extends Omit<PressableProps, 'style' | 'children'> {
  checked: boolean;
  onCheckedChange?: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
  size?: 'default' | 'small';
  style?: ViewStyle;
}

export function Checkbox({
  checked,
  onCheckedChange,
  label,
  disabled = false,
  size = 'default',
  style,
  ...props
}: CheckboxProps) {
  const { colors } = useZedTheme();
  const [hovered, setHovered] = useState(false);

  const boxSize = size === 'small' ? 14 : 18;
  const iconSize = size === 'small' ? 10 : 12;

  const getBorderColor = () => {
    if (disabled) return colors.borderDisabled;
    if (checked) return colors.textAccent;
    if (hovered) return colors.borderFocused;
    return colors.border;
  };

  const getBgColor = () => {
    if (disabled && checked) return colors.elementDisabled;
    if (checked) return colors.textAccent;
    return 'transparent';
  };

  return (
    <Pressable
      disabled={disabled}
      onPress={() => onCheckedChange?.(!checked)}
      onHoverIn={() => setHovered(true)}
      onHoverOut={() => setHovered(false)}
      style={[{ flexDirection: 'row', alignItems: 'center', gap: 8 }, style]}
      {...props}
    >
      <View
        style={{
          width: boxSize,
          height: boxSize,
          borderRadius: 4,
          borderWidth: 1,
          borderColor: getBorderColor(),
          backgroundColor: getBgColor(),
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {checked && (
          <Icon
            name="check"
            size={iconSize}
            color={disabled ? 'disabled' : 'default'}
            style={{ opacity: checked ? 1 : 0 }}
          />
        )}
      </View>
      {label && (
        <Text color={disabled ? 'disabled' : 'default'}>{label}</Text>
      )}
    </Pressable>
  );
}
