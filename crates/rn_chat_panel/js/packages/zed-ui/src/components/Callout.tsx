import React from 'react';
import { View, StyleSheet, type ViewStyle } from 'react-native';
import { useZedTheme } from '../theme';
import { radii, spacing } from '../tokens';

type CalloutSize = 'compact' | 'default';

export interface CalloutProps {
  children: React.ReactNode;
  size?: CalloutSize;
  style?: ViewStyle;
}

const sizeStyles: Record<CalloutSize, ViewStyle> = {
  compact: {
    paddingHorizontal: spacing.px2,
    paddingVertical: spacing.px1,
  },
  default: {
    paddingHorizontal: spacing.px2,
    paddingVertical: spacing.px1p5,
  },
};

export function Callout({ children, size = 'default', style }: CalloutProps) {
  const { colors } = useZedTheme();

  return (
    <View
      style={[
        styles.container,
        sizeStyles[size],
        {
          backgroundColor: colors.background,
          borderColor: colors.border,
        },
        style,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderWidth: 1,
    borderRadius: radii.md,
  },
});
