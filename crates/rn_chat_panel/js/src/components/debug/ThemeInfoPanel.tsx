import React from 'react';
import { View, StyleSheet } from 'react-native';
import { useZedTheme, useZedColors, Text } from '@zed/ui';

export function ThemeInfoPanel() {
  const theme = useZedTheme();
  const colors = useZedColors();

  return (
    <View style={[styles.container, { backgroundColor: colors.surfaceBackground }]}>
      <Text variant="label" color="muted">Theme Info</Text>
      <View style={styles.content}>
        <View style={styles.row}>
          <Text color="muted" style={styles.label}>Name:</Text>
          <Text>{theme.name}</Text>
        </View>
        <View style={styles.row}>
          <Text color="muted" style={styles.label}>Appearance:</Text>
          <Text>{theme.appearance}</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 12,
    borderRadius: 8,
    gap: 8,
  },
  content: {
    gap: 4,
  },
  row: {
    flexDirection: 'row',
    gap: 8,
  },
  label: {
    width: 88,
  },
});
