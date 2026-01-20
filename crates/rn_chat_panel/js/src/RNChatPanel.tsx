import React, { useRef, useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { ZedThemeProvider } from '@zed/ui';

import { ThemeProvider, ChatProvider, DropdownProvider, useColors, useChat } from './contexts';
import { ChatHeader } from './components/ChatHeader';
import { ChatMessageList, type ChatMessageListRef } from './components/ChatMessageList';
import { ChatStatusBar } from './components/ChatStatusBar';
import { ChatInputArea } from './components/ChatInputArea';

import { useAvailableBackends } from './hooks/useAvailableBackends';
import { useToolCallEvents } from './hooks/useToolCallEvents';

type RNChatPanelProps = {
  sessionId?: string;
  paneSide?: 'left' | 'right';
};

function ChatPanel({
  sessionId,
  paneSide,
}: {
  sessionId: string;
  paneSide?: 'left' | 'right';
}) {
  const colors = useColors();
  const { selectedAgent, setSelectedAgent } = useChat();
  const messageListRef = useRef<ChatMessageListRef>(null);

  const { agents } = useAvailableBackends();
  const {
    toolCalls,
    pendingAuthorizations,
    respondToAuthorization,
    clearCompletedToolCalls,
  } = useToolCallEvents();

  // Default to Zed Agent (custom) when available.
  useEffect(() => {
    if (selectedAgent) return;
    const zedAgent = agents.find((a) => a.type === 'custom') ?? agents[0];
    if (zedAgent) setSelectedAgent(zedAgent);
  }, [agents, selectedAgent, setSelectedAgent]);

  return (
    <View style={[styles.container, { backgroundColor: colors.panelBackground }]}>
      <ChatHeader paneSide={paneSide} />

      <ChatMessageList
        ref={messageListRef}
        toolCalls={toolCalls}
        pendingAuthorizations={pendingAuthorizations}
        onAuthorize={respondToAuthorization}
        onClearCompletedToolCalls={clearCompletedToolCalls}
      />

      <ChatStatusBar />

      <ChatInputArea agents={agents} />
    </View>
  );
}

function ChatPanelWithContext({
  sessionId,
  paneSide,
}: {
  sessionId: string;
  paneSide?: 'left' | 'right';
}) {
  return (
    <ThemeProvider>
      <ChatProvider sessionId={sessionId}>
        <DropdownProvider>
          <ChatPanel sessionId={sessionId} paneSide={paneSide} />
        </DropdownProvider>
      </ChatProvider>
    </ThemeProvider>
  );
}

export function RNChatPanel(props: RNChatPanelProps) {
  const sessionId = props.sessionId ?? 'default';
  return (
    <ZedThemeProvider>
      <ChatPanelWithContext sessionId={sessionId} paneSide={props.paneSide} />
    </ZedThemeProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'column',
  },
  mainArea: {
    flex: 1,
  },
  infoPanels: {
    // Match message list horizontal padding
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 8,
  },
});
