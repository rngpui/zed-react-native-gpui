/**
 * NativeTerminalView - Wrapper for the native terminal view component.
 *
 * This component renders a native TerminalView displaying command output.
 * The terminal entity is looked up on the native side using the toolCallId.
 */

import React from 'react';
import { StyleSheet, type StyleProp, type ViewStyle } from 'react-native';
import NativeTerminalViewComponent from '../specs/NativeTerminalViewNativeComponent';

export interface NativeTerminalViewProps {
  /** The tool call ID to look up the terminal content. */
  toolCallId: string;
  /** Index of the terminal within the tool call's terminals array. Defaults to 0. */
  terminalIndex?: number;
  /** Optional style overrides. */
  style?: StyleProp<ViewStyle>;
}

export function NativeTerminalView({
  toolCallId,
  terminalIndex = 0,
  style,
}: NativeTerminalViewProps) {
  return (
    <NativeTerminalViewComponent
      toolCallId={toolCallId}
      terminalIndex={terminalIndex}
      style={[styles.container, style]}
    />
  );
}

const styles = StyleSheet.create({
  container: {
    minHeight: 80,
  },
});

export default NativeTerminalView;
