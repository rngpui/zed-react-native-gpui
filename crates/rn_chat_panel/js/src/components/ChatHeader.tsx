import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, Icon, IconButton, useZedTheme } from '@zed/ui';

import { useChat } from '../contexts';
import NativeZedLLM from '../../../craby-modules/src/NativeZedLLM';
import { layout, sizing, spacing } from '../styles/tokens';

function getAgentIcon(agentType: string): string | null {
  switch (agentType) {
    case 'claude':
      return 'ai-claude';
    case 'gemini':
      return 'ai-gemini';
    case 'codex':
      return 'ai-openai';
    case 'custom':
      return 'sparkle';
    default:
      return null;
  }
}

export function ChatHeader({ paneSide }: { paneSide?: 'left' | 'right' }) {
  const { colors, fonts } = useZedTheme();
  const { currentThreadTitle, selectedAgent, isLoading } = useChat();
  const isRight = paneSide === 'right';

  const handleMinimize = () => {
    NativeZedLLM.minimizeChatPane();
  };

  const handleClose = () => {
    NativeZedLLM.setChatPaneVisible(false);
  };

  const agentIcon = selectedAgent ? getAgentIcon(selectedAgent.type) : null;

  return (
    <View style={[styles.header, { borderBottomColor: colors.border }]}>
      <View style={[styles.headerRow, isRight && styles.headerRowRight]}>
        {/* Left: Agent icon or toggle button */}
        {agentIcon ? (
          <View
            style={[
              styles.agentIconContainer,
              isLoading && styles.agentIconLoading,
            ]}
          >
            <Icon name={agentIcon} size={sizing.iconSm} color="muted" />
          </View>
        ) : (
          <IconButton
            icon="thread"
            size="compact"
            buttonStyle="ghost"
            onPress={handleMinimize}
            accessibilityLabel="Minimize Chat Pane"
          />
        )}

        {/* Center: Thread title */}
        <View style={[styles.titleContainer, isRight && styles.titleContainerRight]}>
          <Text
            style={{ fontSize: fonts.ui.sm, fontWeight: '500' }}
            numberOfLines={1}
            ellipsizeMode="tail"
          >
            {currentThreadTitle}
          </Text>
        </View>

        {/* Right: Close button */}
        <IconButton
          icon="close"
          size="compact"
          buttonStyle="ghost"
          onPress={handleClose}
          accessibilityLabel="Close Agent Pane"
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    paddingHorizontal: spacing.px1p5,
    borderBottomWidth: 1,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px1p5,
    height: layout.tabHeight,
  },
  headerRowRight: {
    flexDirection: 'row-reverse',
  },
  agentIconContainer: {
    width: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  agentIconLoading: {
    opacity: 0.5,
  },
  titleContainer: {
    flex: 1,
    minWidth: 0,
    alignItems: 'flex-start',
  },
  titleContainerRight: {
    alignItems: 'flex-end',
  },
});
