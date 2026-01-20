import React, { useEffect, useState } from 'react';
import { View, Pressable, ScrollView, StyleSheet } from 'react-native';
import { Text, Icon, type IconName, type IconColor } from '@zed/ui';
import { useTheme } from '../contexts';
import type { ToolCallInfo, ToolKind, ToolAuthorizationRequest } from '../types';
import { NativeDiffView } from './NativeDiffView';
import { NativeTerminalView } from './NativeTerminalView';
import { radii, sizing, spacing } from '../styles/tokens';

export interface ToolCallCardProps {
  toolCall: ToolCallInfo;
  authorizationRequest?: ToolAuthorizationRequest;
  onAuthorize?: (toolCallId: string, permissionOptionId: string) => void;
}

const TOOL_KIND_ICONS: Record<ToolKind, IconName> = {
  read: 'tool-search',
  write: 'tool-pencil',
  execute: 'tool-terminal',
  network: 'tool-web',
  other: 'tool-hammer',
};

interface PermissionButtonStyle {
  icon: IconName;
  iconColor: IconColor;
}

function getPermissionButtonStyle(optionId: string): PermissionButtonStyle {
  const id = optionId.toLowerCase();
  if (id.includes('reject') || id.includes('deny') || id.includes('cancel')) {
    return { icon: 'close', iconColor: 'error' };
  }
  return { icon: 'check', iconColor: 'success' };
}

export function ToolCallCard({
  toolCall,
  authorizationRequest,
  onAuthorize,
}: ToolCallCardProps) {
  const { colors, fonts } = useTheme();
  const shouldAutoExpand =
    toolCall.status === 'waiting_for_confirmation' ||
    toolCall.status === 'in_progress' ||
    toolCall.status === 'pending';
  const [expanded, setExpanded] = useState(shouldAutoExpand);
  const errorColor = '#f85149';

  useEffect(() => {
    if (shouldAutoExpand) {
      setExpanded(true);
    }
  }, [shouldAutoExpand]);

  const kindIcon = TOOL_KIND_ICONS[toolCall.kind] ?? 'info';
  const isWaitingForAuth = toolCall.status === 'waiting_for_confirmation' && authorizationRequest;
  const hasContent =
    toolCall.inputJson ||
    toolCall.resultText ||
    toolCall.errorMessage ||
    toolCall.diffCount > 0 ||
    toolCall.terminalCount > 0;
  const failedOrCanceled =
    toolCall.status === 'failed' ||
    toolCall.status === 'rejected' ||
    toolCall.status === 'canceled';
  const isTerminal = toolCall.kind === 'execute' || toolCall.terminalCount > 0;
  const isEdit = toolCall.kind === 'write' || toolCall.diffCount > 0;
  const needsConfirmation = toolCall.status === 'waiting_for_confirmation';
  const useCardLayout = needsConfirmation || isTerminal || isEdit;
  const headerBg = useCardLayout ? colors.elementBackground + '08' : 'transparent';
  const titleColor = useCardLayout ? colors.text : colors.textMuted;
  const sectionStyle = useCardLayout ? styles.cardSection : styles.inlineSection;
  const contentStyle = useCardLayout
    ? [styles.cardContent, { borderTopColor: colors.border }]
    : [styles.inlineContent, { borderLeftColor: colors.border }];
  const showDiffLoading =
    isEdit &&
    toolCall.diffCount === 0 &&
    (toolCall.status === 'pending' || toolCall.status === 'in_progress');
  const hasDisplayContent = hasContent || showDiffLoading;

  return (
    <View
      style={[
        useCardLayout ? styles.container : styles.inlineContainer,
        useCardLayout
          ? {
              backgroundColor: colors.background,
              borderColor: colors.border,
              borderStyle: failedOrCanceled ? 'dashed' : 'solid',
            }
          : null,
      ]}
    >
      {/* Card header with blended background */}
      <Pressable
        onPress={hasDisplayContent ? () => setExpanded(!expanded) : undefined}
        style={[
          useCardLayout ? styles.header : styles.inlineHeader,
          { backgroundColor: headerBg },
        ]}
      >
        <Icon name={kindIcon} size={sizing.iconMd} color="muted" />

        <Text
          numberOfLines={1}
          style={[styles.title, { color: titleColor, fontSize: fonts.ui.sm }]}
        >
          {toolCall.title}
        </Text>

        <View style={styles.headerRight}>
          {failedOrCanceled ? (
            <Icon name="close" size={sizing.iconSm} color="error" />
          ) : null}
          {hasDisplayContent ? (
            <Icon
              name={expanded ? 'chevron-up' : 'chevron-down'}
              size={sizing.iconSm}
              color="muted"
            />
          ) : null}
        </View>
      </Pressable>

      {expanded && (
        <View style={contentStyle}>
          {useCardLayout && isTerminal ? (
            <View
              style={[
                styles.terminalHeader,
                sectionStyle,
                { borderBottomColor: colors.border, backgroundColor: colors.elementBackground + '08' },
              ]}
            >
              <Text
                style={{
                  fontFamily: fonts.bufferFontFamily,
                  fontSize: fonts.ui.xs,
                  color: colors.textMuted,
                }}
              >
                Run Command
              </Text>
              <Text variant="code">{toolCall.title}</Text>
            </View>
          ) : null}

          {/* Native terminal views for execute tool calls */}
          {toolCall.terminalCount > 0 &&
            Array.from({ length: toolCall.terminalCount }, (_, i) => (
              <View
                key={`terminal-${i}`}
                style={[styles.nativeViewSection, sectionStyle, { borderBottomColor: colors.border }]}
              >
                <NativeTerminalView
                  toolCallId={toolCall.id}
                  terminalIndex={i}
                  style={styles.nativeView}
                />
              </View>
            ))}

          {/* Native diff views for write tool calls */}
          {toolCall.diffCount > 0 &&
            Array.from({ length: toolCall.diffCount }, (_, i) => (
              <View
                key={`diff-${i}`}
                style={[styles.nativeViewSection, sectionStyle, { borderBottomColor: colors.border }]}
              >
                <NativeDiffView
                  toolCallId={toolCall.id}
                  diffIndex={i}
                  style={styles.nativeView}
                />
              </View>
            ))}

          {showDiffLoading ? (
            <View
              style={[
                styles.diffLoading,
                sectionStyle,
                { borderBottomColor: colors.border, backgroundColor: colors.background },
              ]}
            >
              <View style={[styles.diffBar, { backgroundColor: colors.elementActive }]} />
              <View style={[styles.diffBar, styles.diffBarShort, { backgroundColor: colors.elementActive }]} />
              <View style={[styles.diffBar, styles.diffBarMedium, { backgroundColor: colors.elementActive }]} />
              <View style={[styles.diffBar, styles.diffBarShort, { backgroundColor: colors.elementActive }]} />
            </View>
          ) : null}

          {toolCall.inputJson && (
            <View style={[styles.section, sectionStyle, { borderBottomColor: colors.border }]}>
              <Text
                style={{
                  fontFamily: fonts.bufferFontFamily,
                  fontSize: fonts.ui.xs,
                  color: colors.textMuted,
                }}
              >
                Input
              </Text>
              <ScrollView
                horizontal
                style={[styles.codeBlock, { backgroundColor: colors.background }]}
              >
                <Text variant="code">
                  {formatJson(toolCall.inputJson)}
                </Text>
              </ScrollView>
            </View>
          )}

          {toolCall.resultText && (
            <View style={[styles.section, sectionStyle, { borderBottomColor: colors.border }]}>
              <Text
                style={{
                  fontFamily: fonts.bufferFontFamily,
                  fontSize: fonts.ui.xs,
                  color: colors.textMuted,
                }}
              >
                Result
              </Text>
              <ScrollView style={[styles.resultBlock, { backgroundColor: colors.background }]}>
                <Text variant="code">
                  {toolCall.resultText}
                </Text>
              </ScrollView>
            </View>
          )}

          {toolCall.errorMessage && (
            <View style={[styles.section, sectionStyle, { borderBottomColor: colors.border }]}>
              <Text
                style={{
                  fontFamily: fonts.bufferFontFamily,
                  fontSize: fonts.ui.xs,
                  color: errorColor,
                }}
              >
                Error
              </Text>
              <Text variant="small" style={{ color: errorColor }}>{toolCall.errorMessage}</Text>
            </View>
          )}
        </View>
      )}

      {useCardLayout && isWaitingForAuth && authorizationRequest && (
        <View
          style={[
            styles.authContainer,
            {
              borderTopColor: colors.border,
              backgroundColor: colors.background,
            },
          ]}
        >
          {authorizationRequest.options.map((option) => {
            const { icon, iconColor } = getPermissionButtonStyle(option.id);
            const isReject = iconColor === 'error';
            return (
              <Pressable
                key={option.id}
                onPress={() => onAuthorize?.(toolCall.id, option.id)}
                style={({ pressed }) => [
                  styles.permissionButton,
                  {
                    backgroundColor: pressed
                      ? colors.elementActive
                      : option.isDefault
                        ? colors.elementBackground
                        : 'transparent',
                  },
                ]}
              >
                <Icon name={icon} size={sizing.iconSm} color={iconColor} />
                <Text
                  style={[
                    styles.permissionButtonText,
                    {
                      color: isReject ? '#f85149' : colors.text,
                      fontSize: fonts.ui.sm,
                    },
                  ]}
                >
                  {option.label}
                </Text>
              </Pressable>
            );
          })}
        </View>
      )}
    </View>
  );
}

function formatJson(json: string): string {
  try {
    return JSON.stringify(JSON.parse(json), null, 2);
  } catch {
    return json;
  }
}

const styles = StyleSheet.create({
  container: {
    borderRadius: radii.md,
    borderWidth: 1,
    overflow: 'hidden',
    marginLeft: spacing.px5,
    marginRight: spacing.px5,
    marginVertical: spacing.px1p5,
  },
  inlineContainer: {
    marginLeft: spacing.px5,
    marginRight: spacing.px5,
    marginVertical: spacing.px1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.px0p5,
    gap: spacing.px1,
    borderTopLeftRadius: radii.md,
    borderTopRightRadius: radii.md,
  },
  inlineHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.px1,
    gap: spacing.px1,
  },
  title: {
    flex: 1,
    minWidth: 0,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px1,
  },
  cardContent: {
    borderTopWidth: 1,
  },
  inlineContent: {
    marginLeft: spacing.px1,
    paddingLeft: spacing.px3p5,
    paddingTop: spacing.px1,
    borderLeftWidth: 1,
  },
  nativeViewSection: {
    borderBottomWidth: 0,
  },
  nativeView: {
    minHeight: 100,
    maxHeight: 300,
  },
  section: {
    paddingHorizontal: spacing.px3p5,
    paddingTop: spacing.px2,
    paddingBottom: spacing.px2,
  },
  cardSection: {
    borderBottomWidth: 1,
  },
  inlineSection: {
    borderBottomWidth: 0,
  },
  codeBlock: {
    borderRadius: radii.sm,
    padding: spacing.px2,
    marginTop: spacing.px1,
  },
  resultBlock: {
    borderRadius: radii.sm,
    padding: spacing.px2,
    marginTop: spacing.px1,
    maxHeight: 150,
  },
  terminalHeader: {
    paddingHorizontal: spacing.px3p5,
    paddingTop: spacing.px1p5,
    paddingBottom: spacing.px1p5,
    gap: spacing.px1,
  },
  diffLoading: {
    paddingHorizontal: spacing.px3p5,
    paddingTop: spacing.px2,
    paddingBottom: spacing.px2,
    gap: spacing.px1,
  },
  diffBar: {
    height: 6,
    borderRadius: 999,
  },
  diffBarShort: {
    width: '40%',
  },
  diffBarMedium: {
    width: '60%',
  },
  authContainer: {
    flexDirection: 'row',
    gap: spacing.px1p5,
    padding: spacing.px2,
    borderTopWidth: 1,
  },
  permissionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px1,
    paddingHorizontal: spacing.px2,
    paddingVertical: spacing.px1,
    borderRadius: radii.sm,
  },
  permissionButtonText: {
    fontWeight: '500',
  },
});
