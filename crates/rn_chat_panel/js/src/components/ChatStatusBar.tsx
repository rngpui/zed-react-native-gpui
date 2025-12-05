import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Callout, Spinner, Text, useZedTheme } from '@zed/ui';

import { useChat } from '../contexts';
import { spacing } from '../styles/tokens';

export function ChatStatusBar() {
  const { colors, fonts } = useZedTheme();
  const { isLoading, threadSnapshot } = useChat();
  const tokenUsage = threadSnapshot?.token_usage;
  const showUsage =
    tokenUsage &&
    (tokenUsage.input_tokens > 0 || tokenUsage.output_tokens > 0);

  if (!isLoading && !showUsage) return null;

  return (
    <View style={styles.wrapper}>
      {isLoading && (
        <Callout size="compact" style={styles.rowCallout}>
          <Spinner size="small" color="muted" />
          <Text style={{ fontSize: fonts.ui.sm, color: colors.textMuted }}>
            Generating…
          </Text>
        </Callout>
      )}

      {showUsage && tokenUsage ? (
        <Callout style={styles.columnCallout}>
          <Text style={{ fontSize: fonts.ui.sm, fontWeight: '500' }}>
            Token usage
          </Text>
          <Text style={{ fontSize: fonts.ui.sm, color: colors.textMuted }}>
            Input {tokenUsage.input_tokens} • Output {tokenUsage.output_tokens}
          </Text>
        </Callout>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    gap: spacing.px1,
  },
  rowCallout: {
    marginTop: spacing.px1,
    marginHorizontal: spacing.px2,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px1p5,
  },
  columnCallout: {
    marginHorizontal: spacing.px2,
    gap: spacing.px1,
  },
});
