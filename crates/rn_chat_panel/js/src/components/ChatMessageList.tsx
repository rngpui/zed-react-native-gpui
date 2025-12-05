import React, { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import { FlatList, View, StyleSheet } from 'react-native';

import { MessageBubble } from './MessageBubble';
import { ToolCallCard } from './ToolCallCard';
import { useChat } from '../contexts';
import { toolCallFromEntry } from '../utils/threadSnapshot';
import { spacing } from '../styles/tokens';

import type { Message, ToolAuthorizationRequest, ToolCallInfo } from '../types';

export interface ChatMessageListRef {
  scrollToBottom: () => void;
}

interface ChatMessageListProps {
  toolCalls: ToolCallInfo[];
  pendingAuthorizations: ToolAuthorizationRequest[];
  onAuthorize: (toolCallId: string, permissionOptionId: string) => void;
  onClearCompletedToolCalls: () => void;
}

type ChatListItem =
  | { type: 'message'; message: Message; isLastMessage: boolean }
  | { type: 'tool_call'; toolCall: ToolCallInfo };

export const ChatMessageList = forwardRef<ChatMessageListRef, ChatMessageListProps>(
  function ChatMessageList(
    {
      toolCalls,
      pendingAuthorizations,
      onAuthorize,
      onClearCompletedToolCalls: _onClearCompletedToolCalls,
    },
    ref
  ) {
    const { messages, threadSnapshot } = useChat();
    const listRef = useRef<FlatList<ChatListItem>>(null);

    const scrollToBottom = () => {
      listRef.current?.scrollToOffset({ offset: 0, animated: true });
    };

    useImperativeHandle(ref, () => ({
      scrollToBottom,
    }));

    const items: ChatListItem[] = useMemo(() => {
      if (threadSnapshot) {
        const lastMessageId = messages.at(-1)?.id;
        const toolCallById = new Map(toolCalls.map((toolCall) => [toolCall.id, toolCall]));
        let messageIndex = 0;
        const list: ChatListItem[] = [];

        for (const entry of threadSnapshot.entries) {
          if (entry.type === 'tool_call') {
            const toolCall = toolCallById.get(entry.id) ?? toolCallFromEntry(entry);
            list.push({ type: 'tool_call', toolCall });
            continue;
          }

          const message = messages[messageIndex];
          if (message) {
            list.push({
              type: 'message',
              message,
              isLastMessage: message.id === lastMessageId,
            });
          }
          messageIndex += 1;
        }

        // Reverse for inverted FlatList: newest at index 0 appears at bottom
        return list.reverse();
      }

      const lastMessageId = messages.at(-1)?.id;

      const list: ChatListItem[] = [];
      for (const message of messages) {
        list.push({
          type: 'message',
          message,
          isLastMessage: message.id === lastMessageId,
        });
      }

      for (const toolCall of toolCalls) {
        list.push({ type: 'tool_call', toolCall });
      }

      // Reverse for inverted FlatList: newest at index 0 appears at bottom
      return list.reverse();
    }, [threadSnapshot, messages, toolCalls]);

    return (
      <FlatList
        ref={listRef}
        style={styles.list}
        contentContainerStyle={styles.listContent}
        data={items}
        inverted
        onContentSizeChange={scrollToBottom}
        keyExtractor={(item) =>
          item.type === 'message' ? item.message.id : `tool-${item.toolCall.id}`
        }
        ListEmptyComponent={<View style={styles.emptyState} />}
        renderItem={({ item }) => {
          if (item.type === 'message') {
            return (
              <MessageBubble
                message={item.message}
                isLast={item.isLastMessage && toolCalls.length === 0}
              />
            );
          }

          const authorizationRequest = pendingAuthorizations.find(
            (auth) => auth.toolCallId === item.toolCall.id
          );

          return (
            <ToolCallCard
              toolCall={item.toolCall}
              authorizationRequest={authorizationRequest}
              onAuthorize={onAuthorize}
            />
          );
        }}
      />
    );
  }
);

const styles = StyleSheet.create({
  list: {
    flex: 1,
  },
  listContent: {
    // Native: per-item horizontal padding with vertical breathing room.
    paddingHorizontal: 0,
    paddingTop: spacing.px2,
    paddingBottom: spacing.px3,
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.px6,
  },
});
