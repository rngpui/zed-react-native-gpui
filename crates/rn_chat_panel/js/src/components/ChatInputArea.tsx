import React from 'react';
import {
  View,
  StyleSheet,
  TextInput,
  Pressable,
  type NativeSyntheticEvent,
  type TextInputKeyPressEventData,
} from 'react-native';
import { IconButton, Icon, useZedTheme } from '@zed/ui';

import { useChat } from '../contexts';
import NativeZedLLM from '../../../craby-modules/src/NativeZedLLM';
import { DropdownSelect } from '@zed/ui';
import type { Agent, AgentAccessModeId, AgentModeInfo } from '../types';
import { radii, sizing, spacing } from '../styles/tokens';

interface ChatInputAreaProps {
  agents: Agent[];
}

export function ChatInputArea({ agents }: ChatInputAreaProps) {
  const { colors, fonts } = useZedTheme();
  const {
    inputValue,
    setInputValue,
    isLoading,
    selectedAgent,
    setSelectedAgent,
    agentAccessMode,
    setAgentAccessMode,
    isFollowing,
    setIsFollowing,
    sendMessage,
  } = useChat();

  const [agentModes, setAgentModes] = React.useState<AgentModeInfo[]>([]);

  const refreshAgentConfig = React.useCallback(
    (agentType: string) => {
      try {
        const modesJson = NativeZedLLM.getAgentModes(agentType);
        const parsedModes = JSON.parse(modesJson) as AgentModeInfo[];
        setAgentModes(parsedModes);

        const nativeSelectedMode = parsedModes.find((m) => m.selected)?.id as AgentAccessModeId | undefined;
        if (nativeSelectedMode) {
          setAgentAccessMode(nativeSelectedMode);
        }
      } catch {
        setAgentModes([]);
      }
    },
    [setAgentAccessMode],
  );

  React.useEffect(() => {
    if (!selectedAgent) return;
    refreshAgentConfig(selectedAgent.type);
  }, [selectedAgent?.type, refreshAgentConfig]);

  const handleKeyPress = (e: NativeSyntheticEvent<TextInputKeyPressEventData>) => {
    if (e.nativeEvent.key === 'Enter' && !e.nativeEvent.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const agentItems = React.useMemo(
    () =>
      agents.map((agent) => ({
        id: agent.type,
        title: agent.name,
        subtitle: agent.isConnected ? 'Connected' : 'Not connected',
      })),
    [agents],
  );

  const modeItems = React.useMemo(() => {
    const fromNative = agentModes.map((mode) => ({
      id: mode.id,
      title: mode.name,
      subtitle: mode.description ?? undefined,
    }));
    if (fromNative.length > 0) return fromNative;
    return [
      { id: 'read_only', title: 'Read Only', subtitle: 'No tools or edits' },
      { id: 'agent', title: 'Agent', subtitle: 'Ask before tools or edits' },
      { id: 'agent_full_access', title: 'Agent (Full Access)', subtitle: 'Auto-allow tools and edits' },
    ];
  }, [agentModes]);

  return (
    <View
      style={[
        styles.container,
        {
          borderTopColor: colors.border,
          backgroundColor: colors.background,
        },
      ]}
    >
      {/* Input area */}
      <View style={styles.inputRow}>
        <View style={styles.editorContainer}>
          <TextInput
            value={inputValue}
            onChangeText={setInputValue}
            placeholder="Message the agent — @ to include context"
            placeholderTextColor={colors.textPlaceholder}
            onKeyPress={handleKeyPress}
            editable={!isLoading}
            multiline
            style={[
              styles.input,
              {
                color: colors.text,
                fontFamily: fonts.bufferFontFamily,
                fontSize: fonts.bufferFontSize,
                lineHeight: fonts.bufferFontSize * 1.5,
              },
            ]}
          />
          <View style={styles.editorControls}>
            <IconButton
              icon="maximize"
              size="compact"
              buttonStyle="ghost"
              accessibilityLabel="Expand Message Editor"
              onPress={() => {}}
            />
          </View>
        </View>
      </View>

      {/* Bottom toolbar */}
      <View style={styles.toolbar}>
        <View style={styles.toolbarLeft}>
          <IconButton
            icon="at-sign"
            size="compact"
            buttonStyle="ghost"
            onPress={() => {}}
            accessibilityLabel="Add Context"
            accessibilityHint="Or type @ to include context"
          />
          <Pressable
            onPress={() => setIsFollowing(!isFollowing)}
            accessibilityLabel={isFollowing ? 'Stop Following Agent' : 'Follow Agent'}
            style={[
              styles.followToggle,
              {
                backgroundColor: isFollowing ? `${colors.textAccent}20` : 'transparent',
              },
            ]}
          >
            <Icon
              name="crosshair"
              size={sizing.iconSm}
              color={isFollowing ? 'accent' : 'muted'}
            />
          </Pressable>
        </View>

        <View style={styles.toolbarRight}>
          <DropdownSelect
            dropdownId="agent-select"
            icon="cpu"
            placeholder="Agent"
            selectedId={selectedAgent?.type ?? null}
            items={agentItems}
            onSelect={(id) => {
              const agent = agents.find((a) => a.type === id);
              if (!agent) return;

              const connected = NativeZedLLM.connectAgent(agent.type);
              if (!connected) {
                console.warn('Failed to connect the agent.');
                return;
              }

              const next = { ...agent, isConnected: true };
              setSelectedAgent(next);
              NativeZedLLM.ensureAgentThread(next.type);

              // Apply current mode selection (best-effort) and refresh UI state.
              NativeZedLLM.setAgentMode(next.type, agentAccessMode);
              refreshAgentConfig(next.type);
            }}
            compact
            placement="up"
          />

          <DropdownSelect
            dropdownId="agent-mode"
            icon="shield"
            placeholder="Mode"
            selectedId={agentAccessMode}
            items={modeItems}
            onSelect={(id) => {
              if (!selectedAgent) return;
              const ok = NativeZedLLM.setAgentMode(selectedAgent.type, id);
              if (ok) {
                setAgentAccessMode(id as AgentAccessModeId);
                refreshAgentConfig(selectedAgent.type);
              }
            }}
            compact
            placement="up"
          />

          {/* Send button */}
          <IconButton
            icon="send"
            size="compact"
            buttonStyle={inputValue.trim() && selectedAgent ? 'filled' : 'ghost'}
            disabled={isLoading || !inputValue.trim() || !selectedAgent}
            onPress={sendMessage}
          />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    // Native: border_t_1, p_2 (8px), gap_2 (8px)
    borderTopWidth: 1,
    padding: spacing.px2,
    gap: spacing.px2,
  },
  inputRow: {
    flexDirection: 'row',
    paddingTop: spacing.px1,
    paddingHorizontal: spacing.px2,
    minHeight: spacing.px5 * 2,
  },
  editorContainer: {
    flex: 1,
    position: 'relative',
  },
  input: {
    flex: 1,
    paddingVertical: 0,
    paddingRight: spacing.px2p5,
    minHeight: spacing.px6,
  },
  editorControls: {
    position: 'absolute',
    top: 0,
    right: 0,
    opacity: 0.5,
  },
  toolbar: {
    // Native: flex_wrap, justify_between
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
  },
  toolbarLeft: {
    // Native: gap_0p5 (2px)
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px0p5,
  },
  toolbarRight: {
    // Native: gap_1 (4px)
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px1,
  },
  followToggle: {
    width: spacing.px5,
    height: spacing.px5,
    borderRadius: radii.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
