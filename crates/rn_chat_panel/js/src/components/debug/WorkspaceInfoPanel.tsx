import React from 'react';
import { View, StyleSheet } from 'react-native';
import { useZedColors, Text } from '@zed/ui';
import { useWorkspaceInfo } from '../../hooks/useWorkspaceInfo';

export function WorkspaceInfoPanel() {
  const colors = useZedColors();
  const workspace = useWorkspaceInfo();

  if (!workspace) {
    return (
      <View style={[styles.container, { backgroundColor: colors.surfaceBackground }]}>
        <Text color="muted">Workspace info not available</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, { backgroundColor: colors.surfaceBackground }]}>
      <Text variant="label" color="muted">Workspace Info</Text>
      <View style={styles.content}>
        <View style={styles.row}>
          <Text color="muted" style={styles.label}>Project:</Text>
          <Text>{workspace.projectName ?? '(none)'}</Text>
        </View>
        <View style={styles.row}>
          <Text color="muted" style={styles.label}>File:</Text>
          <Text style={styles.value} numberOfLines={1}>
            {workspace.currentFilePath?.split('/').pop() ?? '(none)'}
          </Text>
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
    width: 80,
  },
  value: {
    flex: 1,
  },
});
