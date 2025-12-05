import React from 'react';
import { Pressable, View, StyleSheet, type PressableProps, type ViewStyle } from 'react-native';
import { useZedTheme } from '../theme';
import { radii, spacing } from '../tokens';

export interface ListRowProps extends Omit<PressableProps, 'style' | 'children'> {
  children: React.ReactNode;
  selected?: boolean;
  accentColor?: string;
  style?: ViewStyle;
  contentStyle?: ViewStyle;
}

export function ListRow({
  children,
  selected = false,
  accentColor,
  style,
  contentStyle,
  ...props
}: ListRowProps) {
  const { colors } = useZedTheme();

  return (
    <Pressable
      {...props}
      style={({ pressed, hovered }) => [
        styles.row,
        {
          backgroundColor: selected
            ? colors.ghostElementSelected
            : pressed || hovered
              ? colors.ghostElementHover
              : 'transparent',
        },
        style,
      ]}
    >
      {accentColor ? (
        <View style={[styles.accent, { backgroundColor: accentColor }]} />
      ) : null}
      <View style={[styles.content, contentStyle]}>{children}</View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    position: 'relative',
    paddingHorizontal: spacing.px2,
    paddingVertical: spacing.px1,
    borderRadius: radii.sm,
  },
  content: {
    gap: spacing.px1,
  },
  accent: {
    position: 'absolute',
    left: spacing.px1,
    top: spacing.px1,
    bottom: spacing.px1,
    width: 3,
    borderRadius: 1.5,
  },
});
