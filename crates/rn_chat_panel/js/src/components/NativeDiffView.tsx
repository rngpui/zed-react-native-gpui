/**
 * NativeDiffView - Wrapper for the native diff view component.
 *
 * This component renders a native Editor displaying file diffs.
 * The diff content is looked up on the native side using the toolCallId.
 */

import React from 'react';
import { StyleSheet, type StyleProp, type ViewStyle } from 'react-native';
import NativeDiffViewComponent from '../specs/NativeDiffViewNativeComponent';

export interface NativeDiffViewProps {
  /** The tool call ID to look up the diff content. */
  toolCallId: string;
  /** Index of the diff within the tool call's diffs array. Defaults to 0. */
  diffIndex?: number;
  /** Optional style overrides. */
  style?: StyleProp<ViewStyle>;
}

export function NativeDiffView({
  toolCallId,
  diffIndex = 0,
  style,
}: NativeDiffViewProps) {
  return (
    <NativeDiffViewComponent
      toolCallId={toolCallId}
      diffIndex={diffIndex}
      style={[styles.container, style]}
    />
  );
}

const styles = StyleSheet.create({
  container: {
    minHeight: 100,
  },
});

export default NativeDiffView;
