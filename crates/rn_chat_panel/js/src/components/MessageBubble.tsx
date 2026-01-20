import React from 'react';
import { View, StyleSheet, Pressable } from 'react-native';
import { Text, Icon, MarkdownRenderer } from '@zed/ui';
import { useTheme } from '../contexts';
import type { Message } from '../types';
import { radii, sizing, spacing } from '../styles/tokens';

export interface MessageBubbleProps {
  message: Message;
  isLast?: boolean;
}

export function MessageBubble({ message, isLast = false }: MessageBubbleProps) {
  const { colors, fonts } = useTheme();
  const isUser = message.role === 'user';
  const [openThinking, setOpenThinking] = React.useState<Record<number, boolean>>({});

  const assistantSegments = React.useMemo(() => {
    const rawChunks =
      message.chunks && message.chunks.length > 0
        ? message.chunks
        : [{ type: 'text', content: message.content } as const];
    const combined: Array<{ type: 'text' | 'thinking'; content: string }> = [];

    rawChunks.forEach((chunk) => {
      if (!chunk.content) return;
      if (chunk.type === 'text') {
        const last = combined[combined.length - 1];
        if (last && last.type === 'text') {
          last.content += chunk.content;
        } else {
          combined.push({ type: 'text', content: chunk.content });
        }
      } else {
        combined.push({ type: 'thinking', content: chunk.content });
      }
    });

    return combined.length > 0
      ? combined
      : [{ type: 'text', content: message.content }];
  }, [message.chunks, message.content]);
  const lastTextIndex = assistantSegments.reduce(
    (last, segment, index) => (segment.type === 'text' ? index : last),
    -1,
  );

  // User messages: white card with shadow and monospace text (like native)
  // Native uses: shadow_md(), py_3 (12px), px_2 (8px), rounded_md, border_1
  if (isUser) {
    return (
      <View style={styles.userWrapper}>
        <View
          style={[
            styles.userCard,
            {
              backgroundColor: colors.background,
              borderColor: colors.border,
              // Shadow for depth - matches native shadow_md() (iOS)
              shadowColor: '#000',
              shadowOffset: { width: 0, height: 2 },
              shadowOpacity: 0.15,
              shadowRadius: 6,
              // Shadow (Android)
              elevation: 4,
            },
          ]}
        >
          <Text
            style={{
              fontFamily: fonts.bufferFontFamily,
              fontSize: fonts.bufferFontSize,
              lineHeight: fonts.bufferFontSize * 1.5,
            }}
          >
            {message.content}
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={[styles.assistantMessage, isLast && styles.assistantMessageLast]}>
      {assistantSegments.map((segment, index) => {
        if (segment.type === 'thinking') {
          const isOpen = openThinking[index] ?? false;
          return (
            <View key={`thinking-${index}`} style={styles.thinkingBlock}>
              <Pressable
                onPress={() =>
                  setOpenThinking((prev) => ({ ...prev, [index]: !isOpen }))
                }
                style={styles.thinkingHeader}
              >
                <View style={styles.thinkingTitle}>
                  <Icon name="tool-think" size={sizing.iconSm} color="muted" />
                  <Text style={{ color: colors.textMuted, fontSize: fonts.ui.sm }}>
                    Thinking
                  </Text>
                </View>
                <Icon
                  name={isOpen ? 'chevron-up' : 'chevron-down'}
                  size={sizing.iconXs}
                  color="muted"
                />
              </Pressable>
              {isOpen && (
                <View style={[styles.thinkingContent, { borderLeftColor: colors.border }]}>
                  <MarkdownRenderer content={segment.content} />
                </View>
              )}
            </View>
          );
        }

        if (message.streaming && index === lastTextIndex) {
          return (
            <Text key={`text-${index}`}>
              {segment.content}
              <Text color="accent">|</Text>
            </Text>
          );
        }

        return <MarkdownRenderer key={`text-${index}`} content={segment.content} />;
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  userWrapper: {
    paddingHorizontal: spacing.px2,
    paddingTop: spacing.px2,
    paddingBottom: spacing.px3,
  },
  userCard: {
    // Native: rounded_md, border_1, py_3 (12px), px_2 (8px)
    borderRadius: radii.md,
    borderWidth: 1,
    paddingVertical: spacing.px3,
    paddingHorizontal: spacing.px2,
  },
  assistantMessage: {
    // Native: px_5 (20px), py_1p5 (6px)
    paddingHorizontal: spacing.px5,
    paddingVertical: spacing.px1p5,
    gap: spacing.px1p5,
  },
  assistantMessageLast: {
    // Native: pb_4 (16px) when last message
    paddingBottom: spacing.px4,
  },
  thinkingBlock: {
    gap: spacing.px1,
  },
  thinkingHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  thinkingTitle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px1,
  },
  thinkingContent: {
    marginLeft: spacing.px1p5,
    paddingLeft: spacing.px3p5,
    borderLeftWidth: 1,
    opacity: 0.85,
  },
});
