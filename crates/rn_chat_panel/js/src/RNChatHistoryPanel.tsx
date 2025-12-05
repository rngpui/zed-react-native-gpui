import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, Pressable, ScrollView, Animated } from 'react-native';
import { ZedThemeProvider, Text, Input, Icon, useZedTheme } from '@zed/ui';
import { DropdownProvider } from './contexts';
import NativeZedLLM from '../../craby-modules/src/NativeZedLLM';

import { useThreadSnapshot } from './hooks/useThreadSnapshot';
import { useAvailableBackends } from './hooks/useAvailableBackends';
import type { AcpThreadMetadata, AgentServerInfo } from './types';
import { layout, radii, sizing, spacing } from './styles/tokens';

type RNChatHistoryPanelProps = {
  sessionId?: string;
};

const AGENT_ICON_MAP: Record<string, React.ComponentProps<typeof Icon>['name']> = {
  codex: 'ai-openai',
  claude: 'ai-claude',
  gemini: 'ai-gemini',
  custom: 'sparkle',
};

function BackendOptionButton({
  agent,
  onPress,
}: {
  agent: AgentServerInfo;
  onPress: () => void;
}) {
  const { colors, fonts } = useZedTheme();
  const iconName = AGENT_ICON_MAP[agent.type] ?? 'sparkle';
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const [isHovered, setIsHovered] = useState(false);

  const handleHoverIn = () => {
    setIsHovered(true);
    Animated.timing(scaleAnim, {
      toValue: 1.02,
      duration: 100,
      useNativeDriver: true,
    }).start();
  };

  const handleHoverOut = () => {
    setIsHovered(false);
    Animated.timing(scaleAnim, {
      toValue: 1,
      duration: 100,
      useNativeDriver: true,
    }).start();
  };

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.98,
      speed: 50,
      bounciness: 4,
      useNativeDriver: true,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: isHovered ? 1.02 : 1,
      speed: 20,
      bounciness: 8,
      useNativeDriver: true,
    }).start();
  };

  return (
    <Pressable
      onPress={onPress}
      onHoverIn={handleHoverIn}
      onHoverOut={handleHoverOut}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
    >
      <Animated.View
        style={[
          styles.backendOptionButton,
          {
            cursor: 'pointer',
            borderColor: colors.border,
            backgroundColor: isHovered ? colors.elementHover : colors.elementBackground,
            transform: [{ scale: scaleAnim }],
          },
        ]}
      >
        <Icon name={iconName} size={sizing.iconMd} color="default" />
        <View style={styles.backendOptionContent}>
          <Text style={{ fontSize: fonts.ui.sm, fontWeight: '500' }}>{agent.name}</Text>
          <Text
            variant="small"
            color={agent.isConnected ? 'accent' : 'muted'}
            style={{ fontSize: fonts.ui.xs }}
          >
            {agent.isConnected ? 'Connected' : 'Not connected'}
          </Text>
        </View>
        <Icon name="chevron-right" size={sizing.iconSm} color="muted" />
      </Animated.View>
    </Pressable>
  );
}

function NewThreadButton({
  isOpen,
  onPress,
}: {
  isOpen: boolean;
  onPress: () => void;
}) {
  const { colors, fonts } = useZedTheme();
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const chevronAnim = useRef(new Animated.Value(0)).current;
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    Animated.timing(chevronAnim, {
      toValue: isOpen ? 1 : 0,
      duration: 150,
      useNativeDriver: true,
    }).start();
  }, [isOpen, chevronAnim]);

  const handleHoverIn = () => {
    setIsHovered(true);
    Animated.timing(scaleAnim, {
      toValue: 1.02,
      duration: 100,
      useNativeDriver: true,
    }).start();
  };

  const handleHoverOut = () => {
    setIsHovered(false);
    Animated.timing(scaleAnim, {
      toValue: 1,
      duration: 100,
      useNativeDriver: true,
    }).start();
  };

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.98,
      speed: 50,
      bounciness: 4,
      useNativeDriver: true,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: isHovered ? 1.02 : 1,
      speed: 20,
      bounciness: 8,
      useNativeDriver: true,
    }).start();
  };

  const chevronRotate = chevronAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '180deg'],
  });

  return (
    <Pressable
      onPress={onPress}
      onHoverIn={handleHoverIn}
      onHoverOut={handleHoverOut}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
    >
      <Animated.View
        style={[
          styles.newButton,
          {
            cursor: 'pointer',
            borderColor: colors.border,
            backgroundColor: isHovered ? colors.elementHover : colors.elementBackground,
            transform: [{ scale: scaleAnim }],
          },
        ]}
      >
        <Icon name="plus" size={sizing.iconSm} color="default" />
        <Text style={{ fontSize: fonts.ui.sm }}>New</Text>
        <Animated.View style={{ transform: [{ rotate: chevronRotate }] }}>
          <Icon name="chevron-down" size={sizing.iconXs} color="muted" />
        </Animated.View>
      </Animated.View>
    </Pressable>
  );
}

function ThreadHistoryItem({
  thread,
  isActive,
  onPress,
}: {
  thread: AcpThreadMetadata;
  isActive: boolean;
  onPress: () => void;
}) {
  const { colors } = useZedTheme();
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const [isHovered, setIsHovered] = useState(false);

  const title = thread.title?.trim().length > 0 ? thread.title : 'Untitled thread';
  const updatedLabel = thread.updatedAt
    ? new Date(thread.updatedAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    : '';

  const handleHoverIn = () => {
    setIsHovered(true);
    Animated.timing(scaleAnim, {
      toValue: 1.02,
      duration: 100,
      useNativeDriver: true,
    }).start();
  };

  const handleHoverOut = () => {
    setIsHovered(false);
    Animated.timing(scaleAnim, {
      toValue: 1,
      duration: 100,
      useNativeDriver: true,
    }).start();
  };

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.98,
      speed: 50,
      bounciness: 4,
      useNativeDriver: true,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: isHovered ? 1.02 : 1,
      speed: 20,
      bounciness: 8,
      useNativeDriver: true,
    }).start();
  };

  const baseColor = isActive ? colors.ghostElementSelected : 'transparent';
  const hoverColor = isActive ? colors.ghostElementSelected : colors.elementHover;

  return (
    <Pressable
      onPress={onPress}
      onHoverIn={handleHoverIn}
      onHoverOut={handleHoverOut}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
    >
      <Animated.View
        style={[
          styles.threadItem,
          {
            cursor: 'pointer',
            backgroundColor: isHovered ? hoverColor : baseColor,
            transform: [{ scale: scaleAnim }],
          },
        ]}
      >
        <Text numberOfLines={1} style={{ flex: 1, minWidth: 0 }}>
          {title}
        </Text>
        {updatedLabel ? (
          <Text variant="small" color="muted" numberOfLines={1}>
            {updatedLabel}
          </Text>
        ) : null}
      </Animated.View>
    </Pressable>
  );
}

function HistoryPanelInner({
  sessionId: _sessionId,
}: {
  sessionId: string;
}) {
  const { colors, fonts } = useZedTheme();
  const threadSnapshot = useThreadSnapshot();
  const activeThreadId = threadSnapshot?.session_id ?? null;
  const [threads, setThreads] = React.useState<AcpThreadMetadata[]>([]);
  const [searchQuery, setSearchQuery] = React.useState('');
  const [confirmingDelete, setConfirmingDelete] = React.useState(false);
  const [showBackendOptions, setShowBackendOptions] = React.useState(false);
  const { agents } = useAvailableBackends();

  const heightAnim = useRef(new Animated.Value(0)).current;
  const optionHeight = 58;
  const containerPadding = spacing.px2 * 2;
  const optionGap = spacing.px1p5;
  const expandedHeight = agents.length * optionHeight + (agents.length - 1) * optionGap + containerPadding + spacing.px1;

  useEffect(() => {
    Animated.timing(heightAnim, {
      toValue: showBackendOptions ? expandedHeight : 0,
      duration: 150,
      useNativeDriver: false,
    }).start();
  }, [showBackendOptions, heightAnim, expandedHeight]);

  const toggleBackendOptions = React.useCallback(() => {
    setShowBackendOptions((prev) => !prev);
  }, []);

  const handleNewThread = React.useCallback((agentType: string) => {
    setShowBackendOptions(false);
    setTimeout(() => {
      const connected = NativeZedLLM.connectAgent(agentType);
      if (!connected) {
        console.warn('Failed to connect agent:', agentType);
        return;
      }
      NativeZedLLM.startAgentThread(agentType);
      NativeZedLLM.setChatPaneVisible(true);
    }, 0);
  }, []);

  const parseThreads = React.useCallback((json: string): AcpThreadMetadata[] => {
    try {
      const parsed = JSON.parse(json);
      if (!Array.isArray(parsed)) return [];
      return parsed.map((item) => ({
        id: String(item.id ?? ''),
        title: String(item.title ?? ''),
        updatedAt: String(item.updated_at ?? item.updatedAt ?? ''),
      }));
    } catch {
      return [];
    }
  }, []);

  const loadThreads = React.useCallback(() => {
    const json = NativeZedLLM.listAcpThreads();
    setThreads(parseThreads(json));
  }, [parseThreads]);

  React.useEffect(() => {
    loadThreads();
    NativeZedLLM.addListener('acpThreadsChanged');
    const unsub = NativeZedLLM.acpThreadsChanged((event) => {
      setThreads(parseThreads(event.threads));
    });
    return () => {
      unsub();
      NativeZedLLM.removeListeners(1);
    };
  }, [loadThreads, parseThreads]);

  const handleSelectThread = (threadId: string) => {
    const ok = NativeZedLLM.openAcpThread(threadId);
    if (!ok) {
      console.warn('Failed to open ACP thread.');
    }
  };

  const normalizedQuery = searchQuery.trim().toLowerCase();
  const visibleThreads = normalizedQuery
    ? threads.filter((thread) =>
        (thread.title ?? '').toLowerCase().includes(normalizedQuery)
      )
    : threads;
  const hasNoHistory = threads.length === 0;
  const hasNoMatches = !hasNoHistory && visibleThreads.length === 0;

  const borderWidthInterpolated = heightAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 1],
    extrapolate: 'clamp',
  });

  return (
    <View style={[styles.container, { backgroundColor: colors.panelBackground }]}>
      {/* Header with new thread button */}
      <View style={[styles.header, { borderBottomColor: colors.border }]}>
        <Text style={{ fontSize: fonts.ui.sm, fontWeight: '500' }}>History</Text>
        <NewThreadButton isOpen={showBackendOptions} onPress={toggleBackendOptions} />
      </View>

      {/* Backend Options */}
      <Animated.View
        style={[
          styles.backendOptionsContainer,
          {
            borderBottomColor: colors.border,
            height: heightAnim,
            borderBottomWidth: borderWidthInterpolated,
          },
        ]}
      >
        <View style={styles.backendOptions}>
          {agents.map((agent) => (
            <BackendOptionButton
              key={agent.type}
              agent={agent}
              onPress={() => handleNewThread(agent.type)}
            />
          ))}
        </View>
      </Animated.View>

      {/* Search */}
      <View style={[styles.searchHeader, { borderBottomColor: colors.border }]}>
        <Icon name="search" size={sizing.iconSm} color="muted" />
        <Input
          value={searchQuery}
          onChangeText={setSearchQuery}
          placeholder="Search threads..."
          style={{
            flex: 1,
            minHeight: layout.tabHeight - spacing.px1,
            borderWidth: 0,
            backgroundColor: 'transparent',
            paddingHorizontal: 0,
            paddingVertical: 0,
            fontSize: fonts.ui.sm,
          }}
        />
      </View>

      <View style={styles.listContainer}>
        <ScrollView contentContainerStyle={styles.list}>
          {hasNoHistory ? (
            <View style={styles.placeholder}>
              <Text
                color="muted"
                style={{ fontSize: fonts.ui.sm, textAlign: 'center' }}
              >
                You don't have any past threads yet.
              </Text>
            </View>
          ) : hasNoMatches ? (
            <View style={styles.placeholder}>
              <Text style={{ fontSize: fonts.ui.sm, textAlign: 'center' }}>
                No threads match your search.
              </Text>
            </View>
          ) : (
            visibleThreads.map((thread) => (
              <ThreadHistoryItem
                key={thread.id}
                thread={thread}
                isActive={activeThreadId === thread.id}
                onPress={() => handleSelectThread(thread.id)}
              />
            ))
          )}
        </ScrollView>
      </View>

      {!hasNoHistory && (
        <View style={[styles.footer, { borderTopColor: colors.borderVariant }]}>
          {!confirmingDelete ? (
            <Pressable
              onPress={() => setConfirmingDelete(true)}
              style={({ pressed, hovered }) => [
                styles.footerButton,
                {
                  borderColor: colors.border,
                  backgroundColor: pressed || hovered ? colors.ghostElementHover : 'transparent',
                },
              ]}
            >
              <Text color="muted" style={{ fontSize: fonts.ui.sm }}>
                Delete All History
              </Text>
            </Pressable>
          ) : (
            <View style={styles.deleteConfirm}>
              <View style={styles.deleteText}>
                <Text style={{ fontSize: fonts.ui.sm }}>Delete all threads?</Text>
                <Text color="muted" style={{ fontSize: fonts.ui.sm }}>
                  You won't be able to recover them later.
                </Text>
              </View>
              <View style={styles.deleteActions}>
                <Pressable
                  onPress={() => setConfirmingDelete(false)}
                  style={({ pressed, hovered }) => [
                    styles.footerButton,
                    {
                      borderColor: colors.border,
                      backgroundColor: pressed || hovered ? colors.ghostElementHover : 'transparent',
                    },
                  ]}
                >
                  <Text style={{ fontSize: fonts.ui.sm }}>Cancel</Text>
                </Pressable>
                <Pressable
                  onPress={() => setConfirmingDelete(false)}
                  style={({ pressed, hovered }) => [
                    styles.footerButton,
                    {
                      borderColor: colors.border,
                      backgroundColor: pressed || hovered ? '#f85149' : '#dc2626',
                    },
                  ]}
                >
                  <Text style={{ fontSize: fonts.ui.sm, color: colors.background }}>
                    Delete
                  </Text>
                </Pressable>
              </View>
            </View>
          )}
        </View>
      )}
    </View>
  );
}

export function RNChatHistoryPanel(props: RNChatHistoryPanelProps) {
  const resolvedSessionId = props.sessionId ?? 'default';
  return (
    <ZedThemeProvider>
      <DropdownProvider>
        <HistoryPanelInner key={resolvedSessionId} sessionId={resolvedSessionId} />
      </DropdownProvider>
    </ZedThemeProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    overflow: 'visible',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.px2,
    height: layout.tabHeight,
    borderBottomWidth: 1,
    overflow: 'visible',
    zIndex: 100,
  },
  searchHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.px2,
    paddingVertical: spacing.px1,
    gap: spacing.px2,
    borderBottomWidth: 1,
    minHeight: layout.tabHeight,
    zIndex: 1,
  },
  listContainer: {
    flex: 1,
    zIndex: 1,
  },
  list: {
    flexGrow: 1,
    padding: spacing.px1,
    paddingRight: spacing.px4,
  },
  placeholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.px4,
  },
  threadItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.px2,
    paddingHorizontal: spacing.px2,
    paddingVertical: spacing.px1,
    borderRadius: radii.sm,
  },
  footer: {
    padding: spacing.px2,
    borderTopWidth: 1,
  },
  footerButton: {
    paddingVertical: spacing.px1p5,
    paddingHorizontal: spacing.px2,
    borderRadius: radii.sm,
    borderWidth: 1,
    alignItems: 'center',
  },
  deleteConfirm: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.px2,
    flexWrap: 'wrap',
  },
  deleteText: {
    flex: 1,
    minWidth: 180,
    gap: spacing.px1,
  },
  deleteActions: {
    flexDirection: 'row',
    gap: spacing.px1p5,
  },
  newButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px1,
    paddingVertical: spacing.px1,
    paddingHorizontal: spacing.px2,
    borderRadius: radii.sm,
    borderWidth: 1,
  },
  backendOptionsContainer: {
    overflow: 'hidden',
  },
  backendOptions: {
    padding: spacing.px2,
    gap: spacing.px1p5,
  },
  backendOptionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px2,
    paddingVertical: spacing.px2,
    paddingHorizontal: spacing.px3,
    borderRadius: radii.md,
    borderWidth: 1,
  },
  backendOptionContent: {
    flex: 1,
    gap: spacing.px0p5,
  },
});
