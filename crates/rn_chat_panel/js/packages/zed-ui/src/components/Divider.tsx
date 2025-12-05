import React from 'react';
import { View, type ViewStyle } from 'react-native';
import { useZedTheme } from '../theme';

export interface DividerProps {
  orientation?: 'horizontal' | 'vertical';
  variant?: 'default' | 'subtle';
  spacing?: number;
  style?: ViewStyle;
}

export function Divider({
  orientation = 'horizontal',
  variant = 'default',
  spacing = 0,
  style,
}: DividerProps) {
  const { colors } = useZedTheme();

  const borderColor = variant === 'subtle' ? colors.borderVariant : colors.border;

  if (orientation === 'vertical') {
    return (
      <View
        style={[
          {
            width: 1,
            backgroundColor: borderColor,
            marginHorizontal: spacing,
            alignSelf: 'stretch',
          },
          style,
        ]}
      />
    );
  }

  return (
    <View
      style={[
        {
          height: 1,
          backgroundColor: borderColor,
          marginVertical: spacing,
          alignSelf: 'stretch',
        },
        style,
      ]}
    />
  );
}
